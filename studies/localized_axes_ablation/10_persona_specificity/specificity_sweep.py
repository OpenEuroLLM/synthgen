"""Persona specificity: does a concrete, situated persona ("a 34-year-old
teacher in Kraków, planning a weekend trip") outperform the current generic
label ("parent") at the SAME suggestion rate, or is specificity orthogonal to
rate (per 02_persona_rate's finding on rate itself)?

`synthgen.localized.prompts.generation_prompt()`'s `_PERSONA_ON` clause takes
a plain `{role}` string -- this script calls it with either the production
generic ROLES list, or a parallel list of concrete situated personas (one per
generic role, same country-appropriate flavor added at call time via
`{role}, {situated_detail}`), holding persona_p, domain/intent/salt sampling
identical across arms (same seed stream) so specificity is the only variable.

Run inside the vLLM container from the repo root, e.g.:
  python studies/localized_axes_ablation/10_persona_specificity/specificity_sweep.py \
      --gen-endpoints endpoints/gen --judge-endpoints endpoints/judge \
      --n-per-lang 30 --langs es fr de
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
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
from synthgen.localized import taxonomy as T  # noqa: E402
from synthgen.localized.prompts import generation_prompt, judge_quality_prompt  # noqa: E402
from _common import diversity_metrics as dm  # noqa: E402
from _common import report as rpt  # noqa: E402

GEN_SAMPLING = {"temperature": 0.9, "top_p": 0.95, "max_tokens": 1536}
JUDGE_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512,
                  "chat_template_kwargs": {"enable_thinking": False}}

PERSONA_P = 0.5  # current production default -- held fixed; specificity is the only axis

# One concrete, situated variant per generic T.ROLES entry -- same archetype,
# added age/occupation/situation detail. Deliberately generic-nationality
# (the model still localizes to {country} itself); this tests SPECIFICITY of
# the persona description, not a new axis of cultural grounding.
CONCRETE_ROLES: dict[str, str] = {
    "parent": "a 34-year-old parent of two young kids, juggling work and school pickups",
    "university student": "a 20-year-old university student in their second year, on a tight budget",
    "retiree": "a 68-year-old retiree who just stopped working after a long career",
    "small-business owner": "a 45-year-old who runs a small independent shop and does the books themself",
    "recent immigrant": "someone who moved here 8 months ago and is still learning the local systems",
    "office worker": "a 29-year-old office worker with a long commute and little free time",
    "teenager": "a 16-year-old in secondary school, texting a friend for advice",
    "tourist": "a tourist on a 5-day trip, trying to make the most of a short visit",
}

ARMS = ["generic", "concrete"]


def _persona_arm(rng: random.Random, arm: str) -> str | None:
    if rng.random() >= PERSONA_P:
        return None
    generic = rng.choice(T.ROLES)
    return generic if arm == "generic" else CONCRETE_ROLES[generic]


def build_rows_specificity(n: int, lang_code: str, arm: str, *, seed: int = 0,
                          general_p: float = 0.2) -> list[dict]:
    lang_name = LANGUAGES_PHASE3[lang_code]
    country = LANG_COUNTRY[lang_code]
    # Same seed STRING as production build_rows (no arm-specific suffix) so
    # domain/intent/salt/persona-coinflip draws stay identical across arms --
    # only which persona STRING gets used (generic vs. concrete) differs.
    # (See 01_domain_grounding's fixed bug: an arm-specific seed suffix here
    # would silently re-sample different rows per arm instead of comparing
    # the same rows under different treatment.)
    rng = random.Random(f"{seed}:{lang_code}:0")

    rows = []
    for i in range(n):
        is_local = rng.random() >= general_p
        domain = rng.choice(T.LOCAL_DOMAINS if is_local else T.GENERAL_DOMAINS)
        intents = T.LOCAL_INTENTS if is_local else T.GENERAL_INTENTS
        intent = rng.choice(intents)
        role = _persona_arm(rng, arm)
        salt = rng.randint(1000, 9999)
        rows.append({
            "id": f"{lang_code}-{arm}-{i:06d}", "lang": lang_code,
            "domain": domain, "is_local": is_local, "intent": intent,
            "role": role, "salt": salt,
            "meta_prompt": generation_prompt(
                lang_name=lang_name, country=country, domain=domain,
                intent=intent, salt=salt, role=role, localized=is_local),
        })
    return rows


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
    rec["instruction"], rec["response"] = ex.get("instruction"), ex.get("response")
    rec["role_used"] = ex.get("role_used")
    rec["gen_err"] = err
    rec.pop("meta_prompt", None)
    if rec["instruction"] and rec["response"]:
        jp = judge_quality_prompt(lang_name=lang_name, country=country,
                                  instruction=rec["instruction"], response=rec["response"])
        jtxt, _ = await _chat(judge, client, jp, JUDGE_SAMPLING)
        j = extract_json(jtxt) or {}
        sc = j.get("score")
        if isinstance(sc, str) and sc.strip().lstrip("-").isdigit():
            sc = int(sc)
        rec["score"] = sc if isinstance(sc, (int, float)) else None
        rec["reason"] = j.get("reason")
    return rec


async def run_arm(arm: str, langs: list[str], n_per_lang: int, seed: int,
                  gen, judge, client, sem) -> list[dict]:
    rows = []
    for code in langs:
        rows.extend(build_rows_specificity(n_per_lang, code, arm, seed=seed))

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

        for arm in ARMS:
            print(f"\n[specificity_sweep] running arm: {arm}")
            records = await run_arm(arm, langs, args.n_per_lang, args.seed,
                                    gen, judge, client, sem)
            arm_dir = out / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(arm_dir / "records.jsonl", records)

            stats = rpt.print_arm_report(arm, records)
            arm_stats[arm] = stats

            ok_records = [r for r in records if r.get("instruction") and r.get("response")]
            diversity = dm.diversity_report(ok_records, include_embedding=not args.no_embedding)
            rpt.print_diversity_summary(arm, diversity)
            (arm_dir / "summary.json").write_text(json.dumps(
                {"arm": arm, "score_stats": stats, "diversity": diversity},
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-embedding", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
