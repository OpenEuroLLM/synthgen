"""Pool split (GENERAL_P) sweep — how much of the raw sampled ratio survives
quality filtering?

`synthgen.localized.prompts.build_rows()` already accepts `general_p` as a
parameter (production code, not forked here). This script samples at several
split ratios, judges everything, then runs the actual production
`synthgen.localized.quality_filter.run()` on each arm's output — so the
"post-filter ratio" number comes from the real filtering code, not a
reimplementation of it.

The whole point of this ablation: a 60/40 RAW split can land somewhere else
entirely once Pool B (local-culture) rows survive the judge threshold at a
different rate than Pool A (general) rows. Both ratios are reported,
explicitly, per arm — never assume they match.

Run inside the vLLM container from the repo root, e.g.:
  python studies/localized_axes_ablation/06_domain_split_ratio/split_sweep.py \
      --gen-endpoints endpoints/gen --judge-endpoints endpoints/judge \
      --n-per-lang 40 --langs es fr de --threshold 6.0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
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
from synthgen.localized import quality_filter  # noqa: E402
from _common import diversity_metrics as dm  # noqa: E402
from _common import report as rpt  # noqa: E402

# 0.2 = current default; 0.6 = the composition under discussion.
ARMS = [0.2, 0.4, 0.6, 0.8]

GEN_SAMPLING = {"temperature": 0.9, "top_p": 0.95, "max_tokens": 1536}
JUDGE_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512,
                  "chat_template_kwargs": {"enable_thinking": False}}


async def _chat(backend, client, prompt, sampling):
    res = await backend.chat(client, messages=[{"role": "user", "content": prompt}],
                             sampling=sampling)
    return res.get("content"), res.get("error")


async def process_row(row: dict, gen_model_id: str, gen, judge, client) -> tuple[dict, dict]:
    """Returns (gen_row, judged_row) shaped exactly like generate.py's
    mode="localized"/"judge" outputs, so quality_filter.run() can consume
    them unmodified."""
    lang_name = LANGUAGES_PHASE3[row["lang"]]
    country = LANG_COUNTRY[row["lang"]]
    txt, err = await _chat(gen, client, row["meta_prompt"], GEN_SAMPLING)
    ex = extract_json(txt) or {}
    instruction, response = ex.get("instruction"), ex.get("response")

    gen_row = {"id": row["id"], "lang": row["lang"], "model": gen_model_id,
              "instruction": instruction, "response": response,
              "domain": row["domain"], "is_local": row["is_local"],
              "intent": row["intent"], "role": row["role"]}
    if err or not instruction or not response:
        gen_row["error"] = err or "empty_generation"

    judged_row = {"id": row["id"], "lang": row["lang"]}
    if instruction and response:
        jp = judge_quality_prompt(lang_name=lang_name, country=country,
                                  instruction=instruction, response=response)
        jtxt, _ = await _chat(judge, client, jp, JUDGE_SAMPLING)
        j = extract_json(jtxt) or {}
        sc = j.get("score")
        if isinstance(sc, str) and sc.strip().lstrip("-").isdigit():
            sc = int(sc)
        if isinstance(sc, (int, float)):
            judged_row["score"] = sc
            judged_row["reason"] = j.get("reason")
        else:
            judged_row["error"] = "unparseable_score"
    else:
        judged_row["error"] = "no_instruction_response"
    return gen_row, judged_row


async def run_arm(general_p: float, langs: list[str], n_per_lang: int, seed: int,
                  gen_model_id: str, gen, judge, client, sem) -> tuple[list[dict], list[dict]]:
    rows = []
    for code in langs:
        rows.extend(build_rows(n_per_lang, code, seed=seed, general_p=general_p))

    async def worker(row):
        async with sem:
            return await process_row(row, gen_model_id, gen, judge, client)

    pairs = await asyncio.gather(*(worker(r) for r in rows))
    gen_rows = [p[0] for p in pairs]
    judged_rows = [p[1] for p in pairs]
    return gen_rows, judged_rows


def pool_ratio(rows: list[dict]) -> dict:
    c = Counter("general" if not r.get("is_local") else "local" for r in rows)
    total = sum(c.values()) or 1
    return {"local": c["local"], "general": c["general"],
            "local_frac": round(c["local"] / total, 3),
            "general_frac": round(c["general"] / total, 3)}


async def main_async(args):
    langs = args.langs or list(LANGUAGES_PHASE3)
    gen = VLLMBackend(model=args.gen_model, endpoints_dir=args.gen_endpoints)
    judge = VLLMBackend(model=args.judge_model, endpoints_dir=args.judge_endpoints)
    sem = asyncio.Semaphore(args.concurrency)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thresholds = {"global": args.threshold}

    arm_summaries: dict[str, dict] = {}
    async with httpx.AsyncClient() as client:
        await gen.ensure_ready()
        await judge.ensure_ready()

        for general_p in ARMS:
            label = f"general_p={general_p}"
            print(f"\n[split_sweep] running arm: {label}")
            gen_rows, judged_rows = await run_arm(
                general_p, langs, args.n_per_lang, args.seed,
                args.gen_id, gen, judge, client, sem)

            arm_dir = out / f"general_p_{general_p}"
            arm_dir.mkdir(parents=True, exist_ok=True)
            gen_path, judged_path = arm_dir / "gen.jsonl", arm_dir / "judged.jsonl"
            write_jsonl(gen_path, gen_rows)
            write_jsonl(judged_path, judged_rows)

            raw_ratio = pool_ratio(gen_rows)

            filtered_path = arm_dir / "filtered.jsonl"
            filt_summary = quality_filter.run(
                gen=gen_path, judged=judged_path, thresholds=thresholds,
                output=filtered_path, report=arm_dir / "filter_report.json")
            survivors = list(_iter_jsonl(filtered_path))
            post_filter_ratio = pool_ratio(survivors)

            scores = [j.get("score") for j in judged_rows if isinstance(j.get("score"), (int, float))]
            score_stats = rpt.score_stats(scores)
            print(f"    raw ratio:          {raw_ratio}")
            print(f"    post-filter ratio:  {post_filter_ratio}  "
                  f"(kept {filt_summary['kept']}/{len(gen_rows)})")

            ok_texts = [{"instruction": g["instruction"], "lang": g["lang"], "domain": g["domain"]}
                       for g in gen_rows if g.get("instruction") and g.get("response")]
            diversity = dm.diversity_report(ok_texts, include_embedding=not args.no_embedding)
            rpt.print_diversity_summary(label, diversity)

            arm_summaries[label] = {
                "general_p": general_p, "raw_ratio": raw_ratio,
                "post_filter_ratio": post_filter_ratio,
                "filter_summary": filt_summary, "score_stats": score_stats,
            }
            (arm_dir / "summary.json").write_text(json.dumps(
                {**arm_summaries[label], "diversity": diversity}, indent=2, ensure_ascii=False))

    print("\n=== RAW vs POST-FILTER RATIO BY ARM ===")
    for label, s in arm_summaries.items():
        print(f"    {label:16s} raw_general={s['raw_ratio']['general_frac']:.2f}  "
              f"post_filter_general={s['post_filter_ratio']['general_frac']:.2f}")
    (out / "arm_comparison.json").write_text(json.dumps(arm_summaries, indent=2))
    print(f"\nwrote per-arm outputs under {out}")


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-endpoints", required=True)
    ap.add_argument("--judge-endpoints", required=True)
    ap.add_argument("--gen-model", default="gen")
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--gen-id", default="google/gemma-4-31b-it")
    ap.add_argument("--n-per-lang", type=int, default=40)
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=6.0,
                    help="global judge-score threshold — a human-reviewed "
                         "number, NOT decide-thresholds output at this scale "
                         "(see CLAUDE.md's percentile-at-small-n caveat)")
    ap.add_argument("--no-embedding", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
