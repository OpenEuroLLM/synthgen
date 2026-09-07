"""Grounding-mode axis (none/light/deep) layered onto generation, orthogonal
to domain choice — see ../PLAN.md ("Grounding depth has a ceiling set by
topic type"). Like `03_intent_necessity/intent_ablation.py`, this is one of
the few scripts in the study that forks a prompt template rather than
importing it unmodified, because the ablation is specifically about that
template's grounding clause.

  none   -> current _SCOPE_GENERAL/_SCOPE_LOCAL behavior, no real-data block
  light  -> adds a surface-texture instruction (currency/units/dates/names/
            register), no real-data block
  deep   -> current _SCOPE_LOCAL behavior (country fact as the task's
            substance) PLUS a real grounding example from
            domain_exemplars/<lang>.jsonl (ground_domains.py's output)

`max_grounding_mode()` from taxonomy_broadened.py caps which mode a given
domain may use — callers should clamp before calling generation_prompt_grounded,
not rely on this module to silently downgrade an invalid combination.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthgen.localized.prompts import (  # noqa: E402
    _fluency_clause, _PERSONA_ON, _PERSONA_OFF, _SCOPE_LOCAL, _SCOPE_GENERAL,
)
import taxonomy_broadened as TB  # noqa: E402

GROUNDED_GENERATION_PROMPT = """\
You are a native speaker and expert writer of {lang_name} as used in {country}.
Create ONE high-quality instruction-tuning example: a realistic user
instruction in {lang_name} and an excellent assistant response.

Domain: {domain}
Diversity seed: {salt}   (use it to pick a NON-OBVIOUS angle; do not default to
the single most famous example of this domain.)

Silently choose a SPECIFIC sub-aspect of "{domain}" in {country}. Then decide a
realistic user intent (suggested: {intent}) — use it only if it fits this domain,
otherwise pick an intent that does.
{persona_block}
{grounding_block}
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

_GROUNDING_LIGHT = (
    "Locale texture: use {country}-appropriate currency, units, date/number "
    "formatting, plausible local names, and a formality register natural for "
    "{lang_name} — as color, not as the point of the task.\n")

_GROUNDING_DEEP_NO_EXAMPLE = ""  # deep uses _SCOPE_LOCAL alone when no exemplar is available

_GROUNDING_DEEP_WITH_EXAMPLE = (
    "Real-world grounding (for topical/stylistic anchoring ONLY — do not copy "
    "it, do not reuse its exact wording or specifics; it exists to show what a "
    "genuine {country} request in this space looks like):\n"
    "  \"{exemplar}\"\n")


def load_domain_distribution(lang: str, dist_dir: Path) -> dict[str, float] | None:
    """Pool-A-only empirical weights from ground_domains.py's output,
    `{domain: weight}` over TB.POOL_A. Falls back to the cross-language
    `_aggregate.json` if this language's file is missing (thin per-language
    coverage), then to None (caller uses uniform sampling) if neither exists.
    Returns None rather than raising -- the ablation should still run (with a
    log-visible degradation) if ground_domains.py hasn't been run yet."""
    path = dist_dir / f"{lang}.json"
    if not path.exists():
        path = dist_dir / "_aggregate.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("smoothed_distribution")


def load_exemplars(lang: str, exemplar_dir: Path) -> dict[str, list[str]]:
    """domain -> list of real prompt texts, from ground_domains.py's output."""
    path = exemplar_dir / f"{lang}.jsonl"
    by_domain: dict[str, list[str]] = {}
    if not path.exists():
        return by_domain
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_domain.setdefault(row["domain"], []).append(row["text"])
    return by_domain


def grounding_block(mode: str, *, domain: str, country: str, lang_name: str,
                    exemplar: str | None) -> str:
    if mode == "none":
        return ""
    if mode == "light":
        return _GROUNDING_LIGHT.format(country=country, lang_name=lang_name)
    if mode == "deep":
        if exemplar:
            return _GROUNDING_DEEP_WITH_EXAMPLE.format(country=country, exemplar=exemplar[:300])
        return _GROUNDING_DEEP_NO_EXAMPLE
    raise ValueError(f"unknown grounding mode: {mode!r}")


def generation_prompt_grounded(*, lang_name: str, country: str, domain: str, intent: str,
                               salt: int, role: str | None, localized: bool,
                               grounding_mode: str, exemplar: str | None = None) -> str:
    persona = _PERSONA_ON.format(role=role) if role else _PERSONA_OFF
    # "deep" grounding on a local domain keeps the _SCOPE_LOCAL requirement;
    # "light"/"none" always use _SCOPE_GENERAL's substantive-but-not-fact-
    # dependent requirement, regardless of which pool the domain is from —
    # the point of this axis is decoupling "must hinge on a country fact"
    # from "which domain bucket was sampled".
    scope = (_SCOPE_LOCAL if (grounding_mode == "deep" and localized) else _SCOPE_GENERAL).format(
        country=country, lang_name=lang_name)
    gblock = grounding_block(grounding_mode, domain=domain, country=country,
                             lang_name=lang_name, exemplar=exemplar)
    return GROUNDED_GENERATION_PROMPT.format(
        lang_name=lang_name, country=country, domain=domain, intent=intent, salt=salt,
        persona_block=persona, grounding_block=gblock,
        fluency_clause=_fluency_clause(lang_name), scope_clause=scope)


def build_grounded_rows(n: int, lang_code: str, *, lang_name: str, country: str,
                        grounding_mode: str, exemplar_dir: Path, seed: int = 0,
                        start_index: int = 0, persona_p: float = 0.5,
                        general_p: float = 0.2,
                        dist_dir: Path | None = None) -> list[dict]:
    """Same sampling shape as `synthgen.localized.prompts.build_rows`, but
    domain is drawn from the broadened `taxonomy_broadened` pools, clamped to
    the requested grounding_mode's ceiling per domain, and the prompt is built
    via `generation_prompt_grounded` instead of production's `generation_prompt`.

    Pool A domain sampling is WEIGHTED by ground_domains.py's real-data
    distribution when `dist_dir` is given and a distribution file exists for
    `lang_code` (falls back to uniform otherwise, with a log warning — see
    `load_domain_distribution`). Pool B always stays uniform/unweighted (the
    protected local-culture allocation -- see ../PLAN.md's two-pool design).
    """
    # NOTE: seed deliberately excludes grounding_mode -- every arm must draw
    # the identical (domain, is_pool_a, intent, role, salt) sequence per row
    # index so grounding_mode is the ONLY thing that differs between arms
    # (matches synthgen.localized.prompts.build_rows' seeding pattern, and
    # 02/persona_sweep.py / 06/split_sweep.py's use of one shared `seed`
    # across their own arms). Including grounding_mode here was a bug: it
    # gave each arm an independent random stream, so "none"/"light"/"deep"
    # were silently comparing DIFFERENT sampled rows, not the same rows
    # under different grounding treatment.
    rng = random.Random(f"{seed}:{lang_code}:{start_index}")
    exemplars = load_exemplars(lang_code, exemplar_dir) if grounding_mode == "deep" else {}

    pool_a_dist = load_domain_distribution(lang_code, dist_dir) if dist_dir else None
    if dist_dir and pool_a_dist is None:
        import logging
        logging.getLogger("axes_ablation.prompts_grounded").warning(
            "no domain_distribution found for lang=%s in %s -- Pool A falling "
            "back to UNIFORM sampling (run ground_domains.py first for "
            "real-data-weighted sampling)", lang_code, dist_dir)
    pool_a_weights = ([pool_a_dist.get(d, 0.0) for d in TB.POOL_A]
                     if pool_a_dist else None)

    rows = []
    for i in range(n):
        idx = start_index + i
        is_pool_a = rng.random() < general_p  # note: inverted vs. build_rows' is_local
        if is_pool_a:
            domain = (rng.choices(TB.POOL_A, weights=pool_a_weights, k=1)[0]
                      if pool_a_weights else rng.choice(TB.POOL_A))
        else:
            domain = rng.choice(TB.POOL_B)
        ceiling = TB.max_grounding_mode(domain)
        mode = grounding_mode
        if ceiling == "light" and mode == "deep":
            mode = "light"  # clamp, don't silently error — record what actually ran
        intents_local = ["ask (factual question)", "ask for a recommendation",
                         "ask for help planning something", "ask how to do something",
                         "ask to compare options", "ask for an explanation"]
        intents_general = ["ask to draft or write something", "ask to improve or edit a draft",
                           "ask to summarize or rewrite text", "ask for creative help",
                           "ask how to do something", "ask for an explanation"]
        intent = rng.choice(intents_general if is_pool_a else intents_local)
        role = rng.choice(["parent", "university student", "retiree", "small-business owner",
                          "recent immigrant", "office worker", "teenager", "tourist"]) \
            if rng.random() < persona_p else None
        salt = rng.randint(1000, 9999)
        exemplar = None
        if mode == "deep":
            pool = exemplars.get(domain, [])
            if pool:
                exemplar = rng.choice(pool)

        rows.append({
            "id": f"{lang_code}-{grounding_mode}-{idx:06d}", "lang": lang_code,
            "domain": domain, "is_pool_a": is_pool_a, "intent": intent, "role": role,
            "salt": salt, "requested_grounding_mode": grounding_mode,
            "effective_grounding_mode": mode, "exemplar_used": exemplar,
            "meta_prompt": generation_prompt_grounded(
                lang_name=lang_name, country=country, domain=domain, intent=intent,
                salt=salt, role=role, localized=not is_pool_a,
                grounding_mode=mode, exemplar=exemplar),
        })
    return rows


# --- CLI driver: sweep grounding modes, generate, judge, report --------------
#
# Kept in this file rather than a separate script since it's a thin driver
# over build_grounded_rows() above — see ../PLAN.md section 1 for the full
# design (validation loop, Pool A/B comparison).

def _cli():
    import argparse
    import asyncio

    import httpx

    from synthgen.backends import VLLMBackend
    from synthgen.config import LANGUAGES_PHASE3, LANG_COUNTRY
    from synthgen.io import extract_json, write_jsonl
    from synthgen.localized.prompts import judge_quality_prompt
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _common import diversity_metrics as dm
    from _common import report as rpt

    GEN_SAMPLING = {"temperature": 0.9, "top_p": 0.95, "max_tokens": 1536}
    JUDGE_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512,
                      "chat_template_kwargs": {"enable_thinking": False}}
    MODES = ("none", "light", "deep")

    async def _chat(backend, client, prompt, sampling):
        res = await backend.chat(client, messages=[{"role": "user", "content": prompt}],
                                 sampling=sampling)
        return res.get("content"), res.get("error")

    async def process_row(row, lang_name, country, gen, judge, client):
        rec = {**row}
        txt, err = await _chat(gen, client, row["meta_prompt"], GEN_SAMPLING)
        ex = extract_json(txt) or {}
        rec["instruction"], rec["response"] = ex.get("instruction"), ex.get("response")
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

    async def main_async(args):
        exemplar_dir = Path(args.exemplar_dir)
        dist_dir = Path(args.dist_dir)
        langs = args.langs or list(LANGUAGES_PHASE3)
        gen = VLLMBackend(model=args.gen_model, endpoints_dir=args.gen_endpoints)
        judge = VLLMBackend(model=args.judge_model, endpoints_dir=args.judge_endpoints)
        sem = asyncio.Semaphore(args.concurrency)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)

        arm_stats = {}
        async with httpx.AsyncClient() as client:
            await gen.ensure_ready()
            await judge.ensure_ready()

            for mode in MODES:
                label = f"grounding_mode={mode}"
                print(f"\n[prompts_grounded] running arm: {label}")
                rows = []
                for code in langs:
                    lang_name, country = LANGUAGES_PHASE3[code], LANG_COUNTRY[code]
                    rows.extend(build_grounded_rows(
                        args.n_per_lang, code, lang_name=lang_name, country=country,
                        grounding_mode=mode, exemplar_dir=exemplar_dir, seed=args.seed,
                        dist_dir=dist_dir))

                async def worker(row, code=None):
                    lang_name, country = LANGUAGES_PHASE3[row["lang"]], LANG_COUNTRY[row["lang"]]
                    async with sem:
                        return await process_row(row, lang_name, country, gen, judge, client)

                records = await asyncio.gather(*(worker(r) for r in rows))
                arm_dir = out / f"mode_{mode}"
                arm_dir.mkdir(parents=True, exist_ok=True)
                write_jsonl(arm_dir / "records.jsonl", records)

                stats = rpt.print_arm_report(label, records)
                arm_stats[label] = stats

                ok = [r for r in records if r.get("instruction") and r.get("response")]
                diversity = dm.diversity_report(ok, include_embedding=not args.no_embedding)
                rpt.print_diversity_summary(label, diversity)

                pool_a_n = sum(1 for r in ok if r.get("is_pool_a"))
                clamped_n = sum(1 for r in ok
                               if r.get("requested_grounding_mode") != r.get("effective_grounding_mode"))
                (arm_dir / "summary.json").write_text(json.dumps({
                    "requested_mode": mode, "score_stats": stats, "diversity": diversity,
                    "pool_a_rows": pool_a_n, "pool_b_rows": len(ok) - pool_a_n,
                    "clamped_to_light": clamped_n,
                }, indent=2, ensure_ascii=False))

        rpt.compare_arms(arm_stats)
        (out / "arm_comparison.json").write_text(json.dumps(arm_stats, indent=2))
        print(f"\nwrote per-arm outputs under {out}")
        print("\n[reminder] this only measures grounding effect on JUDGE SCORE + "
             "diversity. See 05_country_grounding/check_country_accuracy.py for "
             "whether 'deep' grounding actually lands on the right COUNTRY.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-endpoints", required=True)
    ap.add_argument("--judge-endpoints", required=True)
    ap.add_argument("--gen-model", default="gen")
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--n-per-lang", type=int, default=30)
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--exemplar-dir", default=str(Path(__file__).parent / "domain_exemplars"),
                    help="ground_domains.py's output — must exist before running "
                         "the 'deep' arm with real exemplars")
    ap.add_argument("--dist-dir", default=str(Path(__file__).parent / "domain_distribution"),
                    help="ground_domains.py's Pool-A domain_distribution output — "
                         "used to weight Pool A domain sampling by real-data "
                         "frequency; falls back to uniform sampling (with a "
                         "warning) if missing")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-embedding", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    _cli()
