"""Salt necessity ablation: does the `salt` "diversity seed" text nudge do
anything measurable, or would raising generation temperature alone achieve
the same diversity? Nobody has ablated salt with salt fully removed --
this is item 1 in the study's "logged for later" list.

Paired design, same shape as 03_intent_necessity/intent_ablation.py: each
row generated twice with identical (lang, domain, role, intent) -- once with
the production `Diversity seed: {salt}` clause, once with a stripped fork
that removes it entirely (not just zeroed) -- and both compared on judge
score AND on _common/diversity_metrics (this is the one ablation in the
study where the diversity metric IS the primary outcome, not a secondary
check, since salt's whole job is fighting near-duplication).

Run inside the vLLM container from the repo root, e.g.:
  python studies/localized_axes_ablation/09_salt_necessity/salt_ablation.py \
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
from synthgen.localized.prompts import (  # noqa: E402
    build_rows, judge_quality_prompt, _fluency_clause, _PERSONA_ON, _PERSONA_OFF,
    _SCOPE_LOCAL, _SCOPE_GENERAL,
)
from _common import diversity_metrics as dm  # noqa: E402
from _common import report as rpt  # noqa: E402

GEN_SAMPLING = {"temperature": 0.9, "top_p": 0.95, "max_tokens": 1536}
JUDGE_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512,
                  "chat_template_kwargs": {"enable_thinking": False}}

# Same template as GENERATION_PROMPT, with the "Diversity seed: {salt}" line
# and its parenthetical instruction removed entirely -- everything else
# (fluency, persona, scope, intent, guards, meta-evaluation) kept identical
# so the ONLY difference from production generation is the absence of salt.
NO_SALT_PROMPT = """\
You are a native speaker and expert writer of {lang_name} as used in {country}.
Create ONE high-quality instruction-tuning example: a realistic user
instruction in {lang_name} and an excellent assistant response.

Domain: {domain}

Silently choose a SPECIFIC sub-aspect of "{domain}" in {country}. Then decide a
realistic user intent (suggested: {intent}) — use it only if it fits this domain,
otherwise pick an intent that does.
{persona_block}

Write the INSTRUCTION so that ALL hold:
  - {fluency_clause}
  - Realistic & self-contained: something a real person in {country} would type
    (they may include their own short draft/notes if the task needs it); NOT a
    textbook/quiz question about a supplied passage.
  - {scope_clause}
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


def no_salt_prompt(*, lang_name: str, country: str, domain: str, intent: str,
                   role: str | None = None, localized: bool = True) -> str:
    persona = _PERSONA_ON.format(role=role) if role else _PERSONA_OFF
    scope = (_SCOPE_LOCAL if localized else _SCOPE_GENERAL).format(
        country=country, lang_name=lang_name)
    return NO_SALT_PROMPT.format(
        lang_name=lang_name, country=country, domain=domain, intent=intent,
        persona_block=persona, fluency_clause=_fluency_clause(lang_name),
        scope_clause=scope)


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
    return result


async def process_pair(row: dict, gen, judge, client) -> dict:
    lang_name = LANGUAGES_PHASE3[row["lang"]]
    country = LANG_COUNTRY[row["lang"]]

    with_salt = await gen_and_judge(row["meta_prompt"], lang_name, country, gen, judge, client)
    stripped_prompt = no_salt_prompt(
        lang_name=lang_name, country=country, domain=row["domain"], intent=row["intent"],
        role=row["role"], localized=row["is_local"])
    without_salt = await gen_and_judge(stripped_prompt, lang_name, country, gen, judge, client)

    return {
        "id": row["id"], "lang": row["lang"], "domain": row["domain"],
        "is_local": row["is_local"], "salt": row["salt"],
        "with_salt": with_salt, "without_salt": without_salt,
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

    with_scores = [p["with_salt"].get("score") for p in pairs]
    without_scores = [p["without_salt"].get("score") for p in pairs]
    with_stats = rpt.print_arm_report("salt_present", pairs and
                                      [{"score": s} for s in with_scores])
    without_stats = rpt.print_arm_report("salt_removed", pairs and
                                         [{"score": s} for s in without_scores])
    rpt.compare_arms({"salt_present": with_stats, "salt_removed": without_stats})

    # THE primary outcome for this ablation: does removing salt increase
    # near-duplication? Diversity, not just score, is what salt exists to fight.
    with_texts = [{"instruction": p["with_salt"]["instruction"], "lang": p["lang"],
                  "domain": p["domain"]} for p in pairs if p["with_salt"].get("instruction")]
    without_texts = [{"instruction": p["without_salt"]["instruction"], "lang": p["lang"],
                      "domain": p["domain"]} for p in pairs if p["without_salt"].get("instruction")]
    div_with = dm.diversity_report(with_texts, include_embedding=not args.no_embedding)
    div_without = dm.diversity_report(without_texts, include_embedding=not args.no_embedding)
    rpt.print_diversity_summary("salt_present", div_with)
    rpt.print_diversity_summary("salt_removed", div_without)

    (out / "summary.json").write_text(json.dumps({
        "salt_present_stats": with_stats, "salt_removed_stats": without_stats,
        "diversity_salt_present": div_with, "diversity_salt_removed": div_without,
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
    ap.add_argument("--no-embedding", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
