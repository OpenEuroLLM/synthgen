"""Judge-model sensitivity: does a second judge model agree with Qwen/Qwen3.6-27B
on the same instruction/response pairs, or is a score delta seen elsewhere in
this study a property of one judge model rather than the axis being tested?

No new generation — reuses an already-generated records.jsonl (e.g. from
04_diversity_baseline or any other ablation's output) and re-judges each row
with a second judge server, using the SAME judge_quality_prompt production
code so the only thing that changes is the model. Cheap: judge-only calls,
no gen server needed at all (run.slurm here serves only one model, on 2 GPUs).

Run inside the vLLM container from the repo root, e.g.:
  python studies/localized_axes_ablation/07_judge_model_sensitivity/rejudge.py \
      --judge-endpoints endpoints/judge2 --judge-model judge2 \
      --records ../04_diversity_baseline/outputs/records.jsonl
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
from synthgen.io import extract_json, iter_jsonl, write_jsonl  # noqa: E402
from synthgen.localized.prompts import judge_quality_prompt  # noqa: E402
from _common import report as rpt  # noqa: E402

# Same sampling contract as every other script in this study (see CLAUDE.md's
# judge-sampling note) -- applies regardless of which judge model is served,
# since we don't know in advance whether the second judge is also a reasoning
# model; disabling thinking is a no-op for non-reasoning models.
JUDGE_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512,
                  "chat_template_kwargs": {"enable_thinking": False}}


async def _chat(backend, client, prompt, sampling):
    res = await backend.chat(client, messages=[{"role": "user", "content": prompt}],
                             sampling=sampling)
    return res.get("content"), res.get("error")


async def rejudge_one(rec: dict, judge, client) -> dict:
    lang = rec.get("lang")
    lang_name = LANGUAGES_PHASE3.get(lang, lang)
    country = LANG_COUNTRY.get(lang, lang)
    jp = judge_quality_prompt(lang_name=lang_name, country=country,
                              instruction=rec.get("instruction", ""),
                              response=rec.get("response", ""))
    txt, err = await _chat(judge, client, jp, JUDGE_SAMPLING)
    j = extract_json(txt) or {}
    sc = j.get("score")
    if isinstance(sc, str) and sc.strip().lstrip("-").isdigit():
        sc = int(sc)
    new_score = sc if isinstance(sc, (int, float)) else None
    return {
        "id": rec.get("id"), "lang": lang, "domain": rec.get("domain"),
        "original_score": rec.get("score"), "second_judge_score": new_score,
        "second_judge_reason": j.get("reason"), "rejudge_err": err,
    }


async def main_async(args):
    records = [r for r in iter_jsonl(args.records)
              if r.get("instruction") and r.get("response")
              and isinstance(r.get("score"), (int, float))]
    if args.n:
        records = records[:args.n]

    judge = VLLMBackend(model=args.judge_model, endpoints_dir=args.judge_endpoints)
    sem = asyncio.Semaphore(args.concurrency)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as client:
        await judge.ensure_ready()

        async def worker(rec):
            async with sem:
                return await rejudge_one(rec, judge, client)

        results = await asyncio.gather(*(worker(r) for r in records))

    write_jsonl(out / "rejudge.jsonl", results)

    scored = [r for r in results if r["second_judge_score"] is not None]
    orig_stats = rpt.score_stats([r["original_score"] for r in scored])
    new_stats = rpt.score_stats([r["second_judge_score"] for r in scored])
    rpt.compare_arms({"original_judge": orig_stats, "second_judge": new_stats})

    deltas = [r["second_judge_score"] - r["original_score"] for r in scored]
    mean_abs_delta = (sum(abs(d) for d in deltas) / len(deltas)) if deltas else None
    agree_within_1 = (sum(1 for d in deltas if abs(d) <= 1) / len(deltas)) if deltas else None

    print(f"\nmean |delta| between judges: {mean_abs_delta}")
    print(f"agreement within 1 point:    {agree_within_1:.1%}" if agree_within_1 is not None else "")

    summary = {
        "n": len(scored), "original_judge_stats": orig_stats, "second_judge_stats": new_stats,
        "mean_abs_delta": mean_abs_delta, "agreement_within_1_point": agree_within_1,
        "records_source": args.records,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out / 'rejudge.jsonl'} and {out / 'summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-endpoints", required=True)
    ap.add_argument("--judge-model", default="judge2",
                    help="served-model-name of the SECOND judge (keep distinct "
                         "from 'judge' so both can run if ever served together)")
    ap.add_argument("--records", required=True,
                    help="an existing records.jsonl with instruction/response/score, "
                         "e.g. 04_diversity_baseline/outputs/records.jsonl")
    ap.add_argument("--n", type=int, default=None, help="cap rows re-judged (default: all)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
