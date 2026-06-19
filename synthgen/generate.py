"""Backend-agnostic generator with checkpoint/resume.

Two modes:
    prompt    meta_prompt -> generated_prompt
              Input  : prompts/prompts.jsonl  (from `synthgen build-prompts`)
              Output : outputs/gen_<model_slug>.jsonl
              Feeds  : filter / verify / back-translate

    response  generated_prompt -> assistant response
              Input  : outputs/gen_<model_slug>.filtered.jsonl
              Output : outputs/sft_<model_slug>.jsonl
              One SFT pair per row.

Two backends (selected at the CLI):
    openrouter  hosted API, requires OPENROUTER_API_KEY
    vllm        local OpenAI-compatible servers, addresses discovered from a
                directory of `*.endpoint` files (round-robin pool)

Sharding (`split=True`) round-robins the input rows across multiple backends
for diversity when running multiple models — useful when calling several
OpenRouter models on a single shared prompt set.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Iterable

import httpx

from synthgen.backends import Backend
from synthgen.config import SynthConfig
from synthgen.io import iter_jsonl, load_done_ids, model_slug
from synthgen.log import get_logger

log = get_logger("synthgen.generate")


# ----- mode helpers --------------------------------------------------------

def _input_path_for(mode: str, cfg: SynthConfig, model: str,
                    explicit: Path | None) -> Path:
    if explicit:
        return explicit
    if mode == "prompt":
        return cfg.paths.prompts / "prompts.jsonl"
    s = model_slug(model)
    p = cfg.paths.outputs / f"gen_{s}.filtered.jsonl"
    return p if p.exists() else cfg.paths.outputs / f"gen_{s}.jsonl"


def _output_path_for(mode: str, cfg: SynthConfig, model: str,
                     explicit: Path | None) -> Path:
    if explicit:
        return explicit
    s = model_slug(model)
    if mode == "prompt":
        return cfg.paths.outputs / f"gen_{s}.jsonl"
    return cfg.paths.outputs / f"sft_{s}.jsonl"


def _load_rows(mode: str, path: Path) -> list[dict]:
    if mode == "prompt":
        return list(iter_jsonl(path))
    rows: list[dict] = []
    for r in iter_jsonl(path):
        if r.get("error") or not r.get("generated_prompt"):
            continue
        rows.append(r)
    return rows


def _messages_for(mode: str, row: dict) -> list[dict]:
    key = "meta_prompt" if mode == "prompt" else "generated_prompt"
    return [{"role": "user", "content": row[key]}]


def _record_for(mode: str, row: dict, model: str,
                content: str | None, usage: dict | None, error: str | None) -> dict:
    if error:
        return {"id": row["id"], "lang": row["lang"], "model": model, "error": error}
    if mode == "prompt":
        return {
            "id": row["id"], "lang": row["lang"], "topic": row["topic"],
            "constraint_name": row.get("constraint_name"),
            "constraint_text": row.get("constraint_text"),
            "persona": row.get("persona"),
            "model": model,
            "generated_prompt": content,
            "usage": usage or {},
        }
    return {
        "id": row["id"], "lang": row["lang"], "topic": row.get("topic"),
        "constraint_name": row.get("constraint_name"),
        "constraint_text": row.get("constraint_text"),
        "persona": row.get("persona"),
        "model": model,
        "prompt": row["generated_prompt"],
        "response": content,
        "usage": usage or {},
    }


# ----- core runner ---------------------------------------------------------

async def _run_backend(backend: Backend, mode: str, rows: list[dict],
                       out_path: Path, cfg: SynthConfig,
                       sampling: dict | None = None) -> None:
    done = load_done_ids(out_path)
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        log.info("[%s/%s] nothing to do (%d already done)",
                 backend.name, backend.model, len(done))
        return
    log.info("[%s/%s] %d rows to run (skipping %d) -> %s",
             backend.name, backend.model, len(todo), len(done), out_path.name)

    sampling = sampling or cfg.sampling
    sem = asyncio.Semaphore(cfg.concurrency)
    progress = {"done": 0, "errs": 0}
    total = len(todo)
    t0 = time.time()

    async with httpx.AsyncClient() as client:
        with open(out_path, "a", encoding="utf-8") as fout:
            async def worker(row: dict):
                async with sem:
                    res = await backend.chat(
                        client,
                        messages=_messages_for(mode, row),
                        sampling=sampling,
                    )
                    rec = _record_for(mode, row, backend.model,
                                      res.get("content"), res.get("usage"),
                                      res.get("error"))
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    progress["done"] += 1
                    if "error" in rec:
                        progress["errs"] += 1
                    if progress["done"] % 1000 == 0 or progress["done"] == total:
                        rate = progress["done"] / max(1e-6, time.time() - t0)
                        eta = (total - progress["done"]) / max(1e-6, rate)
                        log.info("  [%s] %d/%d errs=%d %.1f req/s eta=%.1fmin",
                                 backend.model, progress["done"], total,
                                 progress["errs"], rate, eta / 60)
            await asyncio.gather(*(worker(r) for r in todo))
            log.info("[%s/%s] done in %.1fs, errors=%d/%d",
                     backend.name, backend.model,
                     time.time() - t0, progress["errs"], total)


def _max_tokens_default(mode: str) -> int:
    return 1024 if mode == "prompt" else 1536


# ----- public API ----------------------------------------------------------

def run(
    cfg: SynthConfig,
    *,
    backends: Iterable[Backend],
    mode: str = "prompt",
    in_path: Path | None = None,
    split: bool | None = None,
    max_tokens: int | None = None,
) -> list[Path]:
    """Run generation across `backends`.

    - mode="prompt"    : meta_prompt → generated_prompt
    - mode="response"  : generated_prompt → response (SFT pair)
    - split=True       : partition rows across backends (round-robin) for diversity
                         default True iff multiple backends are passed
    """
    if mode not in ("prompt", "response"):
        raise ValueError(f"mode must be 'prompt' or 'response', got {mode!r}")
    cfg.paths.ensure()
    backends = list(backends)
    if not backends:
        raise ValueError("at least one backend required")

    sampling = {**cfg.sampling}
    sampling["max_tokens"] = max_tokens or _max_tokens_default(mode)

    do_split = split if split is not None else (len(backends) > 1)
    outputs: list[Path] = []

    for be in backends:
        ip = _input_path_for(mode, cfg, be.model, in_path)
        if not ip.exists():
            raise SystemExit(f"input file not found: {ip}")

    if do_split and len(backends) > 1:
        # Load rows from each backend's input path; if all backends share an input
        # (the common case) we shard once and dispatch.
        # We require all backends to read the same input for split mode.
        shared_in = _input_path_for(mode, cfg, backends[0].model, in_path)
        for be in backends[1:]:
            if _input_path_for(mode, cfg, be.model, in_path) != shared_in:
                raise SystemExit("split mode requires all backends to share an input file")
        rows = _load_rows(mode, shared_in)
        log.info("loaded %d rows from %s", len(rows), shared_in)
        shards: dict[str, list[dict]] = {be.model: [] for be in backends}
        for i, r in enumerate(rows):
            shards[backends[i % len(backends)].model].append(r)
        for be in backends:
            log.info("  [split] %s: %d rows", be.model, len(shards[be.model]))
        for be in backends:
            out = _output_path_for(mode, cfg, be.model, None)
            asyncio.run(_run_backend(be, mode, shards[be.model], out, cfg, sampling))
            outputs.append(out)
    else:
        for be in backends:
            ip = _input_path_for(mode, cfg, be.model, in_path)
            out = _output_path_for(mode, cfg, be.model, None)
            rows = _load_rows(mode, ip)
            log.info("loaded %d rows from %s", len(rows), ip)
            asyncio.run(_run_backend(be, mode, rows, out, cfg, sampling))
            outputs.append(out)

    return outputs
