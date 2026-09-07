"""Is `LANG_COUNTRY`'s single-country-per-language mapping itself a
mode-collapse source? Every Portuguese row currently anchors to Portugal
only (never Brazil); every French row to France only (never Belgium/Canada/
Switzerland).

`synthgen.localized.prompts.generation_prompt()` already takes `country` as a
plain parameter — this script calls it directly with a sampled alternate
country instead of going through `build_rows()` (which hardcodes
`LANG_COUNTRY[lang_code]` internally), so no production code is forked, just
called with a different argument.

Standalone — does not depend on 01's output.

  python country_diversify.py \
      --gen-endpoints endpoints/gen --judge-endpoints endpoints/judge \
      --n-per-lang 30 --langs fr pt de es
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

# Plausible alternate countries per language, for languages spoken across
# multiple countries. Languages not listed here fall back to LANG_COUNTRY's
# single default (nothing to diversify — e.g. el, uk, ro, cs, pl, it are
# effectively single-country for this taxonomy's purposes).
COUNTRY_ALTERNATES: dict[str, list[str]] = {
    "es": ["Spain", "Mexico", "Argentina", "Colombia"],
    "fr": ["France", "Belgium", "Canada", "Switzerland"],
    "de": ["Germany", "Austria", "Switzerland"],
    "pt": ["Portugal", "Brazil"],
    "nl": ["the Netherlands", "Belgium"],
}

GEN_SAMPLING = {"temperature": 0.9, "top_p": 0.95, "max_tokens": 1536}
JUDGE_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512,
                  "chat_template_kwargs": {"enable_thinking": False}}


def build_rows_diversified(n: int, lang_code: str, *, seed: int = 0,
                           persona_p: float = 0.5, general_p: float = 0.2) -> list[dict]:
    lang_name = LANGUAGES_PHASE3[lang_code]
    alternates = COUNTRY_ALTERNATES.get(lang_code, [LANG_COUNTRY[lang_code]])
    rng = random.Random(f"{seed}:{lang_code}:diversify")

    rows = []
    for i in range(n):
        is_local = rng.random() >= general_p
        domain = rng.choice(T.LOCAL_DOMAINS if is_local else T.GENERAL_DOMAINS)
        intents = T.LOCAL_INTENTS if is_local else T.GENERAL_INTENTS
        intent = rng.choice(intents)
        role = rng.choice(T.ROLES) if rng.random() < persona_p else None
        salt = rng.randint(1000, 9999)
        country = rng.choice(alternates)
        rows.append({
            "id": f"{lang_code}-div-{i:06d}", "lang": lang_code, "country": country,
            "domain": domain, "is_local": is_local, "intent": intent, "role": role,
            "salt": salt,
            "meta_prompt": generation_prompt(
                lang_name=lang_name, country=country, domain=domain, intent=intent,
                salt=salt, role=role, localized=is_local),
        })
    return rows


async def _chat(backend, client, prompt, sampling):
    res = await backend.chat(client, messages=[{"role": "user", "content": prompt}],
                             sampling=sampling)
    return res.get("content"), res.get("error")


async def process_row(row: dict, gen, judge, client) -> dict:
    rec = {**row}
    lang_name = LANGUAGES_PHASE3[row["lang"]]
    txt, err = await _chat(gen, client, row["meta_prompt"], GEN_SAMPLING)
    ex = extract_json(txt) or {}
    rec["instruction"], rec["response"] = ex.get("instruction"), ex.get("response")
    rec["gen_err"] = err
    rec.pop("meta_prompt", None)
    if rec["instruction"] and rec["response"]:
        jp = judge_quality_prompt(lang_name=lang_name, country=row["country"],
                                  instruction=rec["instruction"], response=rec["response"])
        jtxt, _ = await _chat(judge, client, jp, JUDGE_SAMPLING)
        j = extract_json(jtxt) or {}
        sc = j.get("score")
        if isinstance(sc, str) and sc.strip().lstrip("-").isdigit():
            sc = int(sc)
        rec["score"] = sc if isinstance(sc, (int, float)) else None
        rec["reason"] = j.get("reason")
    return rec


async def main_async(args):
    langs = args.langs or list(COUNTRY_ALTERNATES)
    skipped = [c for c in langs if c not in COUNTRY_ALTERNATES]
    if skipped:
        print(f"[warn] no alternate countries known for {skipped} — nothing "
             f"to diversify for these, skipping")
    langs = [c for c in langs if c in COUNTRY_ALTERNATES]

    gen = VLLMBackend(model=args.gen_model, endpoints_dir=args.gen_endpoints)
    judge = VLLMBackend(model=args.judge_model, endpoints_dir=args.judge_endpoints)
    sem = asyncio.Semaphore(args.concurrency)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for code in langs:
        rows.extend(build_rows_diversified(args.n_per_lang, code, seed=args.seed))

    async with httpx.AsyncClient() as client:
        await gen.ensure_ready()
        await judge.ensure_ready()

        async def worker(row):
            async with sem:
                return await process_row(row, gen, judge, client)

        records = await asyncio.gather(*(worker(r) for r in rows))

    write_jsonl(out / "records.jsonl", records)
    stats = rpt.print_arm_report("country_diversified", records)

    ok = [r for r in records if r.get("instruction") and r.get("response")]
    diversity = dm.diversity_report(ok, group_keys=("lang", "country"),
                                    include_embedding=not args.no_embedding)
    rpt.print_diversity_summary("country_diversified", diversity)

    print("\nCompare this arm's summary.json against the single-country baseline "
         "(04_diversity_baseline, which always uses LANG_COUNTRY's single default) "
         "— same n_per_lang/seed for a fair comparison.")
    (out / "summary.json").write_text(json.dumps(
        {"score_stats": stats, "diversity": diversity,
         "country_alternates_used": COUNTRY_ALTERNATES},
        indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-endpoints", required=True)
    ap.add_argument("--judge-endpoints", required=True)
    ap.add_argument("--gen-model", default="gen")
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--n-per-lang", type=int, default=30)
    ap.add_argument("--langs", nargs="*", default=None,
                    help="defaults to all languages with known alternates: "
                         f"{list(COUNTRY_ALTERNATES)}")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-embedding", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
