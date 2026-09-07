"""Prompts for the judge-validation study.

We generate known-GOOD and known-BAD instruction-tuning examples, then have an
LLM judge assign a single holistic quality score (0-10) to each. Comparing the
good vs bad score distributions tells us whether the judge is trustworthy.

The GOOD-path prompt and the judge prompt were promoted verbatim into
`synthgen.localized.prompts` once this study validated the judge — reuse them
from there instead of drifting a second copy. Only the BAD/MEDIUM generators
(deliberately-defective examples, not needed in production) live here.

  good_example_prompt(...)      -> {instruction, response}   (native, localized;
                                    re-exported from synthgen.localized.prompts)
  bad_example_prompt(bad_type)  -> {instruction, response}   (a deliberately low-
                                    quality example of a named failure mode)
  medium_example_prompt(...)    -> {instruction, response}   (one modest, non-
                                    fatal weakness)
  judge_quality_prompt(...)     -> {score, reason}           (holistic 0-10, no
                                    rubric/dimensions; re-exported)
"""
from __future__ import annotations

from synthgen.localized.prompts import (  # noqa: F401
    _PERSONA_OFF,
    _PERSONA_ON,
    generation_prompt as good_example_prompt,
    judge_quality_prompt,
)

# --- BAD generation ---------------------------------------------------------
# Each reproduces a real failure mode seen in the v1 (hard-constraint + persona)
# pipeline. The instruction to the model is explicit so the badness is reliable.

_BAD_HEADER = """\
You are producing ONE deliberately LOW-QUALITY instruction-tuning example in
{lang_name} ({country}), of a specific defective kind, so a quality filter can be
tested. Make the defect real, but keep it superficially plausible (as a careless
data pipeline would produce). Do NOT add any note that it is bad.

Diversity seed: {salt}   (use it to pick a DIFFERENT, non-obvious instance — do
NOT reuse any example wording given below, and do NOT default to a country's capital.)

Defect to inject: {defect}

Output JSON only, no commentary, no code fences:
{{"instruction":"...","response":"..."}}"""

BAD_DEFECTS: dict[str, str] = {
    "mixed_language": (
        "The instruction is written mostly in {lang_name}, but the format/"
        "constraint part (and ideally a phrase or two) is left in ENGLISH — "
        "careless code-switching. The response is in {lang_name}."),
    "unnatural_constraint": (
        "A plain, ordinary question carrying a POINTLESS, unmotivated constraint no "
        "real user would impose. VARY both the question AND the trick across the many "
        "possibilities (avoid a chosen letter, ALL CAPS, no spaces, answer only in "
        "questions, capitalize every word, an arbitrary exact word count) — do not "
        "reuse the same one. The response awkwardly obeys the pointless rule."),
    "trivial": (
        "A TRIVIAL trivia question with an obvious, universally-known one-word answer "
        "and near-zero training value. VARY the fact widely (a well-known small "
        "number, a common date, a primary color, a unit, a simple one-word "
        "definition, a very famous name) — do NOT use a country's capital. The "
        "response is a bare one-liner."),
    "wrong_answer": (
        "A reasonable-looking question, but the response contains a confident "
        "FACTUAL ERROR (a wrong date, name, number, or fact) stated as truth."),
    "bad_explanation": (
        "A reasonable question, but the response is a BAD EXPLANATION: vague, "
        "hand-wavy, circular, or padded with filler, explaining nothing."),
}

# Which defects apply per language (mixed_language needs a non-English base).
_NON_EN_BAD = ["mixed_language", "unnatural_constraint", "trivial", "wrong_answer",
               "bad_explanation"]
_EN_BAD = ["unnatural_constraint", "trivial", "wrong_answer", "bad_explanation"]
BAD_TYPES_BY_LANG: dict[str, list[str]] = {"en": _EN_BAD}


def bad_types_for(lang_code: str) -> list[str]:
    """Defect pool for a language. Every non-English language gets the full set
    (including mixed_language, which needs a non-English base); English drops it."""
    return BAD_TYPES_BY_LANG.get(lang_code, _NON_EN_BAD)


def bad_example_prompt(*, lang_name, country, bad_type, salt=0) -> str:
    defect = BAD_DEFECTS[bad_type].format(lang_name=lang_name, country=country)
    return _BAD_HEADER.format(lang_name=lang_name, country=country, defect=defect,
                              salt=salt)


# --- MEDIUM generation ------------------------------------------------------
# Decent, usable examples with exactly ONE modest, non-fatal weakness. These
# should land in the middle of the scale (~4-7) if the judge grades properly
# rather than only flooring/ceiling.

_MEDIUM_HEADER = """\
You are producing ONE MEDIUM-QUALITY instruction-tuning example in {lang_name}
({country}) — decent and usable, but NOT excellent, because of exactly ONE
modest, non-fatal weakness (below). Everything else should be fine: natural
{lang_name}, on-topic, coherent. Do NOT make it clearly bad — the weakness must
be subtle, the kind of thing that separates a 6 from a 9.

Localized domain: {domain}. Suggested intent: {intent}.
{persona_block}
{mixing_hint}

Modest weakness to include: {weakness}

Output JSON only, no commentary, no code fences:
{{"instruction":"...","response":"..."}}"""

MEDIUM_DEFECTS: dict[str, str] = {
    "slightly_generic": (
        "The instruction is good, but the RESPONSE, though correct, stays a bit "
        "generic — it misses the specific local detail (named dishes, places, "
        "schemes, brands) that would make it excellent."),
    "somewhat_shallow": (
        "The instruction is good and the response is correct, but the explanation "
        "is a bit SHALLOW — it answers, yet stays surface-level and could go "
        "deeper."),
    "minor_imprecision": (
        "Mostly good, but the response has ONE small imprecision or slightly "
        "vague/outdated detail — not a blatant factual error."),
    "slightly_unnatural": (
        "Correct and understandable, but the phrasing is a little stiff or "
        "textbook-ish in places — not fully natural native {lang_name}."),
    "borderline_basic": (
        "The question is fine and answerable but a bit basic / low-effort (not "
        "deeply knowledge-demanding), with an adequate but unremarkable response."),
}


def _medium_mixing_hint(lang_name: str) -> str:
    return (f"Follow the real usage norms of {lang_name} (natural English "
            f"loanwords are fine where idiomatic).")


def medium_example_prompt(*, lang_name, country, domain, intent, salt,
                          role=None, medium_type) -> str:
    persona = _PERSONA_ON.format(role=role) if role else _PERSONA_OFF
    weakness = MEDIUM_DEFECTS[medium_type].format(lang_name=lang_name)
    return _MEDIUM_HEADER.format(lang_name=lang_name, country=country,
                                 domain=domain, intent=intent,
                                 persona_block=persona, weakness=weakness,
                                 mixing_hint=_medium_mixing_hint(lang_name))
