"""Reclassifies the already-generated instructions from the completed 12 run
(outputs/records.jsonl) with the SPLIT 4-category task-type prompt (generation
/ edit_rewrite / extraction / classification), instead of the original 3-way
lump (generation / edit_rewrite / extraction_classification). No regeneration
-- reads existing instructions, re-runs only the cheap classifier call.

Motivation: the first pass showed extraction_classification flat at 3.3% in
both arms while edit_rewrite nearly doubled -- suspiciously flat for a real
signal, and the leading suspect is that lumping two different operations
(pulling structured data out of text vs. judging something against labels)
diluted a real effect happening in only one of them.

  python reclassify_task_type.py --judge-endpoints $EP/judge --judge-model judge
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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthgen.backends import VLLMBackend  # noqa: E402
from task_type_difficulty_ablation import TASK_TYPE_PROMPT, CLASSIFY_SAMPLING  # noqa: E402

VALID = {"generation", "edit_rewrite", "extraction", "classification"}


async def classify_one(judge, client, instruction: str, sem) -> str:
    async with sem:
        res = await judge.chat(client, messages=[{"role": "user",
                               "content": TASK_TYPE_PROMPT.format(instruction=instruction)}],
                               sampling=CLASSIFY_SAMPLING)
        tt = (res.get("content") or "").strip()
        return tt if tt in VALID else "unparseable"


async def main_async(args):
    records_path = Path(args.records)
    pairs = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]

    judge = VLLMBackend(model=args.judge_model, endpoints_dir=args.judge_endpoints)
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient() as client:
        await judge.ensure_ready()

        async def tag(arm_result: dict):
            instr = arm_result.get("instruction")
            if not instr:
                return "unparseable"
            return await classify_one(judge, client, instr, sem)

        full_types = await asyncio.gather(*(tag(p["full"]) for p in pairs))
        stripped_types = await asyncio.gather(*(tag(p["stripped"]) for p in pairs))

    for p, ft, st in zip(pairs, full_types, stripped_types):
        p["full"]["task_type_v2"] = ft
        p["stripped"]["task_type_v2"] = st

    def dist(types):
        c = Counter(types)
        total = sum(c.values()) or 1
        return {k: round(v / total, 4) for k, v in c.items()}

    full_dist = dist(full_types)
    stripped_dist = dist(stripped_types)
    print(f"full (4-way):     {full_dist}")
    print(f"stripped (4-way): {stripped_dist}")

    out = Path(args.out)
    out.write_text(json.dumps({
        "full_task_type_v2": full_dist, "stripped_task_type_v2": stripped_dist,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")

    # also patch records.jsonl in place so the reclassified rows stay with the
    # original pairs, not just summarized
    with open(records_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"updated {records_path} with task_type_v2 fields")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-endpoints", required=True)
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--records", default=str(Path(__file__).parent / "outputs" / "records.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs" / "task_type_v2.json"))
    ap.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
