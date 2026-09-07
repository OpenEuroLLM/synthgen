"""Intent necessity ablation: does the always-injected `intent` axis do real
conditioning work, or would the model land on a sensible intent anyway?

`synthgen.localized.prompts.generation_prompt()` has no flag to omit intent —
it's baked into the template — so this script defines a local
`no_intent_prompt()` mirroring `GENERATION_PROMPT` with the intent clause and
`intent_used` field removed. This is the one script in the study that forks a
prompt template rather than reusing production code unmodified, because the
ablation is specifically about that template clause; everything else
(taxonomy, row sampling, judge prompt) is imported straight from
`synthgen.localized`.

Paired design: each row gets both variants — intent-injected (current
behavior) and intent-stripped — on identical (lang, domain, role, salt), so
the comparison is within-row, not just within-arm-aggregate.

Run inside the vLLM container from the repo root, e.g.:
  python studies/localized_axes_ablation/03_intent_necessity/intent_ablation.py \
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

# Same template as GENERATION_PROMPT in synthgen/localized/prompts.py, with
# the intent clause and intent_used output field removed — everything else
# (fluency, persona, scope, guards, meta-evaluation) kept identical so the
# ONLY difference from production generation is the presence of intent.
NO_INTENT_PROMPT = """\
You are a native speaker and expert writer of {lang_name} as used in {country}.
Create ONE high-quality instruction-tuning example: a realistic user
instruction in {lang_name} and an excellent assistant response.

Domain: {domain}
Diversity seed: {salt}   (use it to pick a NON-OBVIOUS angle; do not default to
the single most famous example of this domain.)

Silently choose a SPECIFIC sub-aspect of "{domain}" in {country}, and a
realistic user intent that fits it naturally.
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
confirm — the role/domain fit together naturally; language is consistent
throughout; the instruction is natural and non-trivial; the response is factually
correct and well-explained. Fix any issue.

Report which role you actually used (role "none" if you dropped it).

Output JSON only, no commentary, no code fences:
{{"instruction":"...","response":"...","role_used":"..."}}"""


def no_intent_prompt(*, lang_name: str, country: str, domain: str, salt: int,
                     role: str | None = None, localized: bool = True) -> str:
    persona = _PERSONA_ON.format(role=role) if role else _PERSONA_OFF
    scope = (_SCOPE_LOCAL if localized else _SCOPE_GENERAL).format(
        country=country, lang_name=lang_name)
    return NO_INTENT_PROMPT.format(
        lang_name=lang_name, country=country, domain=domain, salt=salt,
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

    with_intent = await gen_and_judge(row["meta_prompt"], lang_name, country, gen, judge, client)
    stripped_prompt = no_intent_prompt(
        lang_name=lang_name, country=country, domain=row["domain"], salt=row["salt"],
        role=row["role"], localized=row["is_local"])
    without_intent = await gen_and_judge(stripped_prompt, lang_name, country, gen, judge, client)

    # Cheap sanity signal: does the model land on the *sampled* intent even
    # when not told to? A "yes" would suggest intent is redundant with what
    # domain+role already imply.
    matches_sampled_intent = (
        with_intent.get("intent_used") is not None
        and row["intent"].lower() in (with_intent.get("intent_used") or "").lower()
    )

    return {
        "id": row["id"], "lang": row["lang"], "domain": row["domain"],
        "is_local": row["is_local"], "sampled_intent": row["intent"],
        "with_intent": with_intent, "without_intent": without_intent,
        "self_reported_intent_matches_sampled": matches_sampled_intent,
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

    with_scores = [p["with_intent"].get("score") for p in pairs]
    without_scores = [p["without_intent"].get("score") for p in pairs]
    with_stats = rpt.print_arm_report("intent_injected", pairs and
                                      [{"score": s} for s in with_scores])
    without_stats = rpt.print_arm_report("intent_stripped", pairs and
                                         [{"score": s} for s in without_scores])
    rpt.compare_arms({"intent_injected": with_stats, "intent_stripped": without_stats})

    echo_rate = (sum(1 for p in pairs if p["self_reported_intent_matches_sampled"])
                / len(pairs)) if pairs else 0.0
    print(f"\nself-reported intent matches sampled intent (with intent given): "
          f"{echo_rate:.1%}")

    with_texts = [{"instruction": p["with_intent"]["instruction"], "lang": p["lang"],
                  "domain": p["domain"]} for p in pairs if p["with_intent"].get("instruction")]
    without_texts = [{"instruction": p["without_intent"]["instruction"], "lang": p["lang"],
                      "domain": p["domain"]} for p in pairs if p["without_intent"].get("instruction")]
    div_with = dm.diversity_report(with_texts, include_embedding=not args.no_embedding)
    div_without = dm.diversity_report(without_texts, include_embedding=not args.no_embedding)
    rpt.print_diversity_summary("intent_injected", div_with)
    rpt.print_diversity_summary("intent_stripped", div_without)

    (out / "summary.json").write_text(json.dumps({
        "with_intent_stats": with_stats, "without_intent_stats": without_stats,
        "self_reported_intent_echo_rate": echo_rate,
        "diversity_with_intent": div_with, "diversity_without_intent": div_without,
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
