"""Reference numbers every other ablation in this study diffs against.

Runs the CURRENT, unmodified `synthgen.localized` pipeline (default
PERSONA_P=0.5, default GENERAL_P=0.2, current 30-domain taxonomy, intent
always injected) at pilot scale, judges it, and measures near-duplicate rate.
Nothing here is a variant — this establishes "how bad is mode collapse
today," which 01/02/06 all need before their own deltas mean anything.

Run inside the vLLM container from the repo root, e.g.:
  python studies/localized_axes_ablation/04_diversity_baseline/measure_baseline.py \
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
from synthgen.config import LANGUAGES_PHASE3  # noqa: E402
from synthgen.io import extract_json, write_jsonl  # noqa: E402
from synthgen.localized.prompts import build_rows, judge_quality_prompt  # noqa: E402
from _common import diversity_metrics as dm  # noqa: E402
from _common import report as rpt  # noqa: E402

GEN_SAMPLING = {"temperature": 0.9, "top_p": 0.95, "max_tokens": 1536}
# Judge is a reasoning model (Qwen3.6) — disable thinking so JSON comes back
# directly instead of getting truncated behind a <think> block. See CLAUDE.md.
JUDGE_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512,
                  "chat_template_kwargs": {"enable_thinking": False}}


async def _chat(backend, client, prompt, sampling):
    res = await backend.chat(client, messages=[{"role": "user", "content": prompt}],
                             sampling=sampling)
    return res.get("content"), res.get("error")


async def process_row(row: dict, lang_name: str, country: str, gen, judge, client) -> dict:
    rec = {**row}
    txt, err = await _chat(gen, client, row["meta_prompt"], GEN_SAMPLING)
    ex = extract_json(txt) or {}
    rec["instruction"] = ex.get("instruction")
    rec["response"] = ex.get("response")
    rec["role_used"] = ex.get("role_used")
    rec["intent_used"] = ex.get("intent_used")
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


async def main_async(args):
    langs = args.langs or list(LANGUAGES_PHASE3)
    rows = []
    for code in langs:
        rows.extend(build_rows(args.n_per_lang, code, seed=args.seed))

    gen = VLLMBackend(model=args.gen_model, endpoints_dir=args.gen_endpoints)
    judge = VLLMBackend(model=args.judge_model, endpoints_dir=args.judge_endpoints)
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient() as client:
        await gen.ensure_ready()
        await judge.ensure_ready()

        async def worker(row):
            lang_name = LANGUAGES_PHASE3[row["lang"]]
            from synthgen.config import LANG_COUNTRY
            country = LANG_COUNTRY[row["lang"]]
            async with sem:
                return await process_row(row, lang_name, country, gen, judge, client)

        records = await asyncio.gather(*(worker(r) for r in rows))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "records.jsonl", records)

    scores = [r.get("score") for r in records]
    stats = rpt.score_stats([s for s in scores if s is not None])
    print(f"\n=== BASELINE (unmodified pipeline) === n={stats['n']}  mean={stats['mean']}")
    print(rpt.hist([s for s in scores if s is not None]))

    ok_records = [r for r in records if r.get("instruction") and r.get("response")]
    diversity = dm.diversity_report(ok_records, include_embedding=not args.no_embedding)
    rpt.print_diversity_summary("baseline", diversity)

    (out / "summary.json").write_text(json.dumps(
        {"score_stats": stats, "diversity": diversity}, indent=2, ensure_ascii=False))
    print(f"\nwrote {out / 'records.jsonl'} and {out / 'summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-endpoints", required=True)
    ap.add_argument("--judge-endpoints", required=True)
    ap.add_argument("--gen-model", default="gen")
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--n-per-lang", type=int, default=30)
    ap.add_argument("--langs", nargs="*", default=None,
                    help="defaults to all of synthgen.config.LANGUAGES_PHASE3")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-embedding", action="store_true",
                    help="skip the sentence-transformers embedding metric "
                         "(e.g. no internet yet to download the model)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
