"""Persona-rate sweep: does PERSONA_P=0.5 (current default) actually earn its
keep over a much lower suggestion rate?

`synthgen.localized.prompts.build_rows()` already accepts `persona_p` as a
parameter (production code, not forked here) — this script just calls it at
several values, holding domain/intent/salt sampling identical across arms
(same seed stream) so only the persona rate varies.

Run inside the vLLM container from the repo root, e.g.:
  python studies/localized_axes_ablation/02_persona_rate/persona_sweep.py \
      --gen-endpoints endpoints/gen --judge-endpoints endpoints/judge \
      --n-per-lang 30 --langs es fr de
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STUDY_ROOT))

from synthgen.backends import VLLMBackend  # noqa: E402
from synthgen.config import LANGUAGES_PHASE3, LANG_COUNTRY  # noqa: E402
from synthgen.io import extract_json, write_jsonl  # noqa: E402
from synthgen.localized.prompts import build_rows, judge_quality_prompt  # noqa: E402
from _common import diversity_metrics as dm  # noqa: E402
from _common import report as rpt  # noqa: E402

ARMS = [0.5, 0.15, 0.10, 0.0]  # 0.5 = current default (baseline arm)

GEN_SAMPLING = {"temperature": 0.9, "top_p": 0.95, "max_tokens": 1536}
JUDGE_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512,
                  "chat_template_kwargs": {"enable_thinking": False}}


async def _chat(backend, client, prompt, sampling):
    res = await backend.chat(client, messages=[{"role": "user", "content": prompt}],
                             sampling=sampling)
    return res.get("content"), res.get("error")


async def process_row(row: dict, gen, judge, client) -> dict:
    rec = {**row}
    lang_name = LANGUAGES_PHASE3[row["lang"]]
    country = LANG_COUNTRY[row["lang"]]
    txt, err = await _chat(gen, client, row["meta_prompt"], GEN_SAMPLING)
    ex = extract_json(txt) or {}
    rec["instruction"] = ex.get("instruction")
    rec["response"] = ex.get("response")
    rec["role_used"] = ex.get("role_used")
    rec["gen_err"] = err
    rec.pop("meta_prompt", None)

    if rec["instruction"] and rec["response"]:
        jp = judge_quality_prompt(lang_name=lang_name, country=country,
                                  instruction=rec["instruction"], response=rec["response"])
        txt, _ = await _chat(judge, client, jp, JUDGE_SAMPLING)
        j = extract_json(txt) or {}
        sc = j.get("score")
        if isinstance(sc, str) and sc.strip().lstrip("-").isdigit():
            sc = int(sc)
        rec["score"] = sc if isinstance(sc, (int, float)) else None
        rec["reason"] = j.get("reason")
    return rec


async def run_arm(persona_p: float, langs: list[str], n_per_lang: int, seed: int,
                  gen, judge, client, sem) -> list[dict]:
    rows = []
    for code in langs:
        rows.extend(build_rows(n_per_lang, code, seed=seed, persona_p=persona_p))

    async def worker(row):
        async with sem:
            return await process_row(row, gen, judge, client)

    return await asyncio.gather(*(worker(r) for r in rows))


async def main_async(args):
    langs = args.langs or list(LANGUAGES_PHASE3)
    gen = VLLMBackend(model=args.gen_model, endpoints_dir=args.gen_endpoints)
    judge = VLLMBackend(model=args.judge_model, endpoints_dir=args.judge_endpoints)
    sem = asyncio.Semaphore(args.concurrency)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    arm_stats: dict[str, dict] = {}
    async with httpx.AsyncClient() as client:
        await gen.ensure_ready()
        await judge.ensure_ready()

        for persona_p in ARMS:
            label = f"persona_p={persona_p}"
            print(f"\n[persona_sweep] running arm: {label}")
            records = await run_arm(persona_p, langs, args.n_per_lang, args.seed,
                                    gen, judge, client, sem)
            arm_dir = out / f"persona_p_{persona_p}"
            arm_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(arm_dir / "records.jsonl", records)

            stats = rpt.print_arm_report(label, records)
            arm_stats[label] = stats

            ok_records = [r for r in records if r.get("instruction") and r.get("response")]
            diversity = dm.diversity_report(ok_records, include_embedding=not args.no_embedding)
            rpt.print_diversity_summary(label, diversity)
            (arm_dir / "summary.json").write_text(json.dumps(
                {"persona_p": persona_p, "score_stats": stats, "diversity": diversity},
                indent=2, ensure_ascii=False))

    rpt.compare_arms(arm_stats)
    (out / "arm_comparison.json").write_text(json.dumps(arm_stats, indent=2))
    print(f"\nwrote per-arm outputs under {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-endpoints", required=True)
    ap.add_argument("--judge-endpoints", required=True)
    ap.add_argument("--gen-model", default="gen")
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--n-per-lang", type=int, default=30)
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0,
                    help="same seed across arms -> domain/intent/salt draws "
                         "stay identical, isolating persona_p as the only "
                         "difference")
    ap.add_argument("--no-embedding", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
