"""Task-type + difficulty layers ablation: do the two NEW optional prompt
clauses (added alongside the already-validated general/local angle pair --
see synthgen/localized/prompts.py's docstring) actually do anything, or are
they ignored the way the old suggested-`intent` was (03_intent_necessity:
followed only ~32% of the time)?

Paired design, same shape as 03/09: each row generated twice with identical
(lang, domain, role, salt) -- once with the FULL production prompt (all four
optional layers: general angle, local angle, task type, difficulty), once
with a STRIPPED fork that removes ONLY the task-type and difficulty clauses,
keeping general/local angle (already validated separately) and everything
else identical. Three things compared:

  1. Judge score, paired -- does removing the two clauses cost anything?
  2. Task-type shift -- a follow-up classifier call tags each instruction as
     generation / edit_rewrite / extraction / classification (split into
     four, not three -- a first pass lumped extraction+classification
     together and it showed flat 3.3% in both arms while edit_rewrite nearly
     doubled, suspiciously flat for a real signal; splitting checks whether
     one sub-type is actually moving and the lumped average was hiding it).
     If the "full" arm doesn't show a higher edit/extraction/classification
     share than "stripped", the task-type clause isn't doing real work.
  3. Difficulty shift -- a follow-up classifier call rates each instruction
     1-5 for expertise/complexity required; compare distributions. If "full"
     doesn't skew higher than "stripped", the difficulty clause isn't
     doing real work either.

Run inside the vLLM container from the repo root, e.g.:
  python studies/localized_axes_ablation/12_task_type_difficulty/task_type_difficulty_ablation.py \
      --gen-endpoints endpoints/gen --judge-endpoints endpoints/judge \
      --n-per-lang 30 --langs es fr de pl uk
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
from synthgen.localized.prompts import (  # noqa: E402
    build_rows, judge_quality_prompt, _fluency_clause, _PERSONA_ON, _PERSONA_OFF,
)
from _common import report as rpt  # noqa: E402

GEN_SAMPLING = {"temperature": 0.9, "top_p": 0.95, "max_tokens": 1536}
JUDGE_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512,
                  "chat_template_kwargs": {"enable_thinking": False}}
CLASSIFY_SAMPLING = {"temperature": 0.0, "max_tokens": 20,
                     "chat_template_kwargs": {"enable_thinking": False}}

# Same as GENERATION_PROMPT, with the task-type and difficulty clauses
# removed -- general/local angle, persona, intent-free-choice, everything
# else kept byte-identical so the ONLY difference from production is the
# absence of those two clauses.
STRIPPED_PROMPT = """\
You are a native speaker and expert writer of {lang_name} as used in {country}.
Create ONE high-quality instruction-tuning example: a realistic user
instruction in {lang_name} and an excellent assistant response.

Domain: {domain}

Silently decide a realistic user intent that fits this domain and the
overall theme of the request -- do not default to the single most obvious
intent for this domain.
{persona_block}

If a more specific, non-obvious general-capability angle fits naturally
with this domain, narrow to it -- otherwise keep the task as a direct
{domain} request.

If a genuine {country}-specific angle fits naturally with this domain (a
real local fact, institution, custom, price, place, or convention -- not a
cliche), weave it in -- otherwise keep the request general. Do not force
this if it would feel unnatural.

Write the INSTRUCTION so that ALL hold:
  - {fluency_clause}
  - Realistic & self-contained: something a real person in {country} would type
    (they may include their own short draft/notes if the task needs it); NOT a
    textbook/quiz question about a supplied passage.
  - Substantive & non-trivial: a real task worth doing, shaped naturally by
    whichever angle(s) above actually fit -- never forced, never listed out.
  - Any constraint must be NATURAL and motivated (a real user would impose it);
    no pointless lexical/format tricks.

Write the RESPONSE so that ALL hold:
  - WRONG-ANSWER GUARD: everything stated must be factually correct. If you are
    not certain of a fact, leave it out — never fabricate.
  - BAD-EXPLANATION GUARD: explanations must be clear, correct, and appropriately
    detailed. No hand-waving, no filler, no padding.
  - Name real, specific entities from {country} where relevant. No clichés/stereotypes.

META-EVALUATION (do this silently before you output): re-read your example and
confirm — the role/intent/domain fit together naturally; language is consistent
throughout; the instruction is natural and non-trivial; the response is factually
correct and well-explained. Fix any issue.

Report which role and intent you actually used (role "none" if you dropped it).

Output JSON only, no commentary, no code fences:
{{"instruction":"...","response":"...","role_used":"...","intent_used":"..."}}"""


def stripped_prompt(*, lang_name: str, country: str, domain: str,
                    role: str | None = None) -> str:
    persona = _PERSONA_ON.format(role=role) if role else _PERSONA_OFF
    return STRIPPED_PROMPT.format(lang_name=lang_name, country=country, domain=domain,
                                  persona_block=persona, fluency_clause=_fluency_clause(lang_name))


# --- follow-up classifiers: did the clause actually change anything? --------

TASK_TYPE_PROMPT = """Classify the following instruction into EXACTLY ONE category.
Reply with only the category name, nothing else.

Categories:
- generation: asks to create/write/draft new content
- edit_rewrite: asks to edit, rewrite, fix, shorten, or otherwise transform existing text
- extraction: asks to pull out or list specific structured information already
  present in a given text (dates, names, facts) -- the answer is IN the source
- classification: asks to sort, categorize, or judge something against a set
  of labels/categories (positive/negative, which department, which type) --
  the answer is a JUDGMENT about the source, not a piece of it

INSTRUCTION:
\"\"\"{instruction}\"\"\"

Category:"""

DIFFICULTY_PROMPT = """Rate how much domain expertise or careful multi-step reasoning
is genuinely required to answer the following instruction well, 1 (trivial/generic)
to 5 (requires real expert-level knowledge or careful problem-solving).
Reply with only the digit, nothing else.

INSTRUCTION:
\"\"\"{instruction}\"\"\"

Rating:"""


async def _chat(backend, client, prompt, sampling):
    res = await backend.chat(client, messages=[{"role": "user", "content": prompt}],
                             sampling=sampling)
    return res.get("content"), res.get("error")


async def gen_and_judge(prompt: str, lang_name: str, country: str, gen, judge, client) -> dict:
    txt, err = await _chat(gen, client, prompt, GEN_SAMPLING)
    ex = extract_json(txt) or {}
    instruction, response = ex.get("instruction"), ex.get("response")
    result = {"instruction": instruction, "response": response,
             "role_used": ex.get("role_used"), "intent_used": ex.get("intent_used"),
             "gen_err": err}
    if instruction and response:
        jp = judge_quality_prompt(lang_name=lang_name, country=country,
                                  instruction=instruction, response=response)
        jtxt, _ = await _chat(judge, client, jp, JUDGE_SAMPLING)
        j = extract_json(jtxt) or {}
        sc = j.get("score")
        if isinstance(sc, str) and sc.strip().lstrip("-").isdigit():
            sc = int(sc)
        result["score"] = sc if isinstance(sc, (int, float)) else None
        result["reason"] = j.get("reason")

        tt_txt, _ = await _chat(judge, client, TASK_TYPE_PROMPT.format(instruction=instruction),
                                CLASSIFY_SAMPLING)
        tt = (tt_txt or "").strip()
        result["task_type"] = tt if tt in {"generation", "edit_rewrite",
                                           "extraction", "classification"} else "unparseable"

        diff_txt, _ = await _chat(judge, client, DIFFICULTY_PROMPT.format(instruction=instruction),
                                  CLASSIFY_SAMPLING)
        diff = (diff_txt or "").strip()
        result["difficulty"] = int(diff) if diff.isdigit() and diff in "12345" else None
    return result


async def process_pair(row: dict, gen, judge, client) -> dict:
    lang_name = LANGUAGES_PHASE3[row["lang"]]
    country = LANG_COUNTRY[row["lang"]]

    full = await gen_and_judge(row["meta_prompt"], lang_name, country, gen, judge, client)
    stripped = stripped_prompt(lang_name=lang_name, country=country,
                               domain=row["domain"], role=row["role"])
    stripped_result = await gen_and_judge(stripped, lang_name, country, gen, judge, client)

    return {
        "id": row["id"], "lang": row["lang"], "domain": row["domain"],
        "full": full, "stripped": stripped_result,
    }


async def main_async(args):
    langs = args.langs or list(LANGUAGES_PHASE3)
    rows = []
    for code in langs:
        rows.extend(build_rows(args.n_per_lang, code, seed=args.seed))

    gen = VLLMBackend(model=args.gen_model, endpoints_dir=args.gen_endpoints)
    judge = VLLMBackend(model=args.judge_model, endpoints_dir=args.judge_endpoints)
    sem = asyncio.Semaphore(args.concurrency)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as client:
        await gen.ensure_ready()
        await judge.ensure_ready()

        async def worker(row):
            async with sem:
                return await process_pair(row, gen, judge, client)

        pairs = await asyncio.gather(*(worker(r) for r in rows))

    write_jsonl(out / "records.jsonl", pairs)

    full_scores = [p["full"].get("score") for p in pairs]
    stripped_scores = [p["stripped"].get("score") for p in pairs]
    full_stats = rpt.print_arm_report("full_4layer", pairs and
                                      [{"score": s} for s in full_scores])
    stripped_stats = rpt.print_arm_report("stripped_2layer", pairs and
                                          [{"score": s} for s in stripped_scores])
    rpt.compare_arms({"full_4layer": full_stats, "stripped_2layer": stripped_stats})

    def dist(key, arm):
        from collections import Counter
        c = Counter(p[arm].get(key) for p in pairs)
        total = sum(c.values()) or 1
        return {k: round(v / total, 4) for k, v in c.items()}

    task_type_full = dist("task_type", "full")
    task_type_stripped = dist("task_type", "stripped")
    print(f"\ntask_type distribution -- full: {task_type_full}")
    print(f"task_type distribution -- stripped: {task_type_stripped}")

    diff_full = [p["full"].get("difficulty") for p in pairs if p["full"].get("difficulty")]
    diff_stripped = [p["stripped"].get("difficulty") for p in pairs if p["stripped"].get("difficulty")]
    mean_diff_full = sum(diff_full) / len(diff_full) if diff_full else None
    mean_diff_stripped = sum(diff_stripped) / len(diff_stripped) if diff_stripped else None
    print(f"mean difficulty rating -- full: {mean_diff_full}, stripped: {mean_diff_stripped}")

    (out / "summary.json").write_text(json.dumps({
        "full_4layer_stats": full_stats, "stripped_2layer_stats": stripped_stats,
        "task_type_distribution": {"full": task_type_full, "stripped": task_type_stripped},
        "mean_difficulty": {"full": mean_diff_full, "stripped": mean_diff_stripped},
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {out / 'records.jsonl'} and {out / 'summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-endpoints", required=True)
    ap.add_argument("--judge-endpoints", required=True)
    ap.add_argument("--gen-model", default="gen")
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--n-per-lang", type=int, default=30)
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
