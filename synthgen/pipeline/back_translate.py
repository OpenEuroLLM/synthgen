"""Back-translate generated prompts to English; emit a review TSV.

Uses an OpenRouter model different from the generators (independent reader).
"""
from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path

import httpx

from synthgen.backends import OpenRouterBackend
from synthgen.config import LANGUAGES_PHASE3, SynthConfig
from synthgen.io import iter_jsonl, load_done_ids
from synthgen.log import get_logger

log = get_logger("synthgen.bt")

BT_TEMPLATE = """Translate the following text into clear, natural English. Preserve meaning, register, and any formatting markers (bullets, numbers, JSON). Do not add commentary. Output the translation only.

---
{text}
---"""


def _stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    import math
    import random
    from collections import defaultdict
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        by_lang[r.get("lang", "?")].append(r)
    per_lang = math.ceil(n / max(1, len(by_lang)))
    rng = random.Random(seed)
    picked: list[dict] = []
    for rs in by_lang.values():
        rng.shuffle(rs)
        picked.extend(rs[:per_lang])
    return picked[:n]


async def _run_for_file(backend: OpenRouterBackend, gen_path: Path,
                        cfg: SynthConfig, *,
                        sample_n: int | None, seed: int) -> tuple[Path, list[dict]]:
    bt_path = cfg.paths.outputs / gen_path.name.replace("gen_", "bt_")
    rows = list(iter_jsonl(gen_path))
    if sample_n:
        rows = _stratified_sample(rows, sample_n, seed)
        log.info("[bt:%s] sampled %d rows", gen_path.name, len(rows))
    done = load_done_ids(bt_path)
    todo = [r for r in rows if r["id"] not in done]
    log.info("[bt:%s] %d to translate, %d cached", gen_path.name, len(todo), len(done))

    sampling = {"temperature": 0.0, "max_tokens": 1024}
    sem = asyncio.Semaphore(cfg.concurrency)
    async with httpx.AsyncClient() as client:
        with open(bt_path, "a", encoding="utf-8") as fout:
            async def worker(row: dict):
                if "error" in row or not row.get("generated_prompt"):
                    fout.write(json.dumps(
                        {"id": row["id"], "back_translation": None, "skip": True},
                        ensure_ascii=False) + "\n")
                    return
                async with sem:
                    res = await backend.chat(
                        client,
                        messages=[{"role": "user",
                                   "content": BT_TEMPLATE.format(text=row["generated_prompt"])}],
                        sampling=sampling,
                    )
                    rec = {"id": row["id"], "back_translation": res.get("content")}
                    if res.get("error"):
                        rec["error"] = res["error"]
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
            t0 = time.time()
            await asyncio.gather(*(worker(r) for r in todo))
            log.info("[bt:%s] done in %.1fs", gen_path.name, time.time() - t0)
    return bt_path, rows


def _write_review_tsv(gen_rows: list[dict], bt_path: Path, review_path: Path) -> None:
    bt_by_id: dict[str, str | None] = {}
    for r in iter_jsonl(bt_path):
        bt_by_id[r["id"]] = r.get("back_translation")
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "lang", "topic", "persona", "constraint_name",
                    "constraint_text", "generated_prompt", "back_translation",
                    "human_verdict"])
        for r in gen_rows:
            w.writerow([
                r.get("id"),
                LANGUAGES_PHASE3.get(r.get("lang", ""), r.get("lang")),
                r.get("topic"), r.get("persona"),
                r.get("constraint_name"), r.get("constraint_text"),
                (r.get("generated_prompt") or "").replace("\t", " ").replace("\n", " \\n "),
                (bt_by_id.get(r.get("id", "")) or "").replace("\t", " ").replace("\n", " \\n "),
                "",
            ])
    log.info("wrote review to %s", review_path)


def _pick_files(cfg: SynthConfig, gen: list[Path] | None) -> list[Path]:
    if gen:
        return list(gen)
    all_files = sorted(cfg.paths.outputs.glob("gen_*.jsonl"))
    filtered = [f for f in all_files if f.name.endswith(".filtered.jsonl")]
    if not filtered:
        return [f for f in all_files
                if not f.name.endswith((".filtered.jsonl", ".leaked.jsonl"))]
    stems = {f.name.replace(".filtered.jsonl", "") for f in filtered}
    rest = [f for f in all_files
            if not f.name.endswith((".filtered.jsonl", ".leaked.jsonl"))
            and f.name.replace(".jsonl", "") not in stems]
    return filtered + rest


def run(cfg: SynthConfig, *, gen: list[Path] | None = None,
        sample: int | None = None, seed: int = 0,
        model: str | None = None) -> None:
    cfg.paths.ensure()
    backend = OpenRouterBackend(model=model or cfg.back_translator, cfg=cfg)
    for gen_path in _pick_files(cfg, gen):
        bt_path, gen_rows = asyncio.run(
            _run_for_file(backend, gen_path, cfg, sample_n=sample, seed=seed))
        review_path = cfg.paths.review / (gen_path.stem.replace("gen_", "review_") + ".tsv")
        _write_review_tsv(gen_rows, bt_path, review_path)
