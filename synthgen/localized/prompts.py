"""Localized single-call generation + judge prompts.

Promoted from studies/localized_bootstrap/prompts.py once the judge-validation
pilot confirmed the judge separates good from bad. Production only ever wants
the GOOD path: one model call renders both the instruction and the response as
one JSON object, which the judge then scores.

  build_rows(...)           -> per-row dicts (id, lang, domain, intent, role,
                                salt, meta_prompt) ready for generate.py's
                                mode="localized"
  judge_quality_prompt(...) -> {score, reason}, holistic 0-10, no rubric
"""
from __future__ import annotations

import random

from synthgen.config import LANG_COUNTRY, LANGUAGES_PHASE3
from synthgen.localized import taxonomy as T

# Fraction of examples that even get a persona *suggested* (the model may still
# drop it). The rest are deliberately persona-free direct requests.
PERSONA_P = 0.5

# Fraction of examples drawn from the GENERAL (professional/creative) domains;
# the rest come from the LOCAL domains that carry the localization signal.
GENERAL_P = 0.2


# --- language-agnostic fluency / mixing clauses ----------------------------
# No per-language flags: the MODEL applies each language's real norms (it
# scales to any language we add later), and the judge is told to be LENIENT on
# script/loanword/transliteration matters so unfamiliar languages aren't
# wrongly zeroed.

def _fluency_clause(lang_name: str) -> str:
    return (f"Native fluency: write the way an educated native speaker of "
            f"{lang_name} actually writes, following that language's real usage "
            f"norms. If native {lang_name} speakers routinely use English "
            f"loanwords or technical terms, that is fine; if they do not, keep it "
            f"fully in {lang_name}. Avoid translationese, and never switch whole "
            f"phrases or the instruction itself into English.")


def _judge_lang_note(lang_name: str) -> str:
    return (f"LANGUAGE NOTE for {lang_name}: individual English loanwords or "
            f"technical terms are FINE where {lang_name} speakers use them — do NOT "
            f"penalize those as code-switching or spelling errors. Phonetic or "
            f"approximate transliteration is acceptable but rate it somewhat LOWER "
            f"than clean native spelling (a minor reduction, not 0-1). When unsure "
            f"whether a single word or spelling is acceptable in {lang_name}, assume "
            f"it IS. "
            f"BUT a whole clause or sentence of the INSTRUCTION written in English "
            f"(or any language other than {lang_name}) is a code-switching DEFECT — "
            f"score it 0-3 EVEN IF that clause requests something perfectly "
            f"reasonable (e.g. a {lang_name} question with 'Please provide the answer "
            f"in bullet points and keep it under 100 words' tacked on in English). A "
            f"sensible request in the WRONG language is still wrong: the whole "
            f"instruction must be in {lang_name}. Text in the wrong script is likewise "
            f"a defect.")


# --- GOOD generation --------------------------------------------------------

GENERATION_PROMPT = """\
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

_PERSONA_ON = ("Suggested persona: a {role} — use it ONLY if such a person would "
               "plausibly ask about this domain; otherwise adapt to a role who "
               "would, or drop the persona. If used, invent a specific realistic "
               "situation and goal and let it shape the instruction.")
_PERSONA_OFF = "No persona: write a direct, standalone request."

# Locality requirement branches on the domain kind: local domains must demand
# country-specific knowledge; general (professional/creative) ones need only be
# substantive and written in natural in-language register.
_SCOPE_LOCAL = ("Genuinely local & non-trivial: answering well requires knowledge "
                "specific to {country}; avoid trivia with an obvious one-word answer.")
_SCOPE_GENERAL = ("Substantive & non-trivial: a real task worth doing (e.g. draft, "
                  "write, edit, or summarize something). It need NOT hinge on "
                  "{country}-specific facts, but keep it in natural {lang_name} and "
                  "let local touches appear where they fit; avoid trivial one-line asks.")


def generation_prompt(*, lang_name: str, country: str, domain: str, intent: str,
                      salt: int, role: str | None = None,
                      localized: bool = True) -> str:
    persona = _PERSONA_ON.format(role=role) if role else _PERSONA_OFF
    scope = (_SCOPE_LOCAL if localized else _SCOPE_GENERAL).format(
        country=country, lang_name=lang_name)
    return GENERATION_PROMPT.format(
        lang_name=lang_name, country=country, domain=domain, intent=intent,
        salt=salt, persona_block=persona, fluency_clause=_fluency_clause(lang_name),
        scope_clause=scope)


# --- JUDGE (holistic, no rubric) -------------------------------------------

JUDGE_QUALITY_PROMPT = """\
You are a strict quality filter for multilingual instruction-tuning data. You are
fluent in {lang_name} and knowledgeable about {country}. Only the highest-quality
examples should be kept; low-quality data harms training.

Give ONE holistic quality score, 0 to 10, for the example as training data. In
that single score weigh BOTH (a) whether the TASK is worth training on and (b) how
well the RESPONSE executes it — and let the WEAKER of the two decide the score. A
flawless, fluent, correct response to a worthless task is still low-quality data,
because the model learns nothing useful from it.

DISQUALIFYING TASKS — score 0-2 no matter how correct, fluent, or obedient the
response is, if the INSTRUCTION is any of these:
  - Trivial trivia: an obvious, universally-known fact with a one-word answer
    (the capital of a major country, 2+2). A correct "Paris" teaches nothing.
  - A pointless / unmotivated constraint: a lexical or formatting trick no real
    user would impose (answer without a given letter, reply in ALL CAPS, use no
    commas). Flawless obedience to a silly rule is still worthless — the
    constraint itself IS the defect. (Constraints a real user WOULD ask for — "in
    3 bullet points", "in simple language for a beginner", "keep it under 100
    words" — are natural and NOT disqualifying.)
  - Not a genuine request: a textbook/quiz item, not something a real person types.

Otherwise the task is worthwhile — score by the SEVERITY of the single WORST
problem in the RESPONSE (do NOT count flaws):
  9-10  Excellent — fluent, natural, correct, genuinely useful. Tiny nits are ok.
  7-8   Good — solid and usable; only MINOR issues (a little generic, a small
        stylistic slip, one typo, heavy but natural loanwords).
  5-6   Fair — usable but clearly imperfect: a shallow explanation or one small
        factual imprecision.
  3-4   Weak — a real problem that limits usefulness: a partly-off or thin answer.
  1-2   Poor — a clearly wrong or incoherent part of the answer.
  0     Unusable — wrong language or script entirely, or a wholly wrong/empty answer.

For a WORTHWHILE task, a single small blemish — a lone typo, one garbled word,
natural loanwords, or phonetic transliteration — should cost only a point or two;
never floor an otherwise-excellent example for one such blemish. This leniency
covers RESPONSE execution only — it never lifts a disqualifying task above 2.

{lang_note}

Example ({lang_name}, {country}):
Instruction:
{instruction}

Response:
{response}

Output JSON only, no commentary, no code fences:
{{"score": <integer 0-10>, "reason": "<one sentence>"}}"""


def judge_quality_prompt(*, lang_name: str, country: str, instruction: str,
                         response: str) -> str:
    return JUDGE_QUALITY_PROMPT.format(
        lang_name=lang_name, country=country,
        instruction=instruction, response=response,
        lang_note=_judge_lang_note(lang_name))


# --- row building ------------------------------------------------------------

def build_rows(n: int, lang_code: str, *, start_index: int = 0, seed: int = 0,
              persona_p: float = PERSONA_P, general_p: float = GENERAL_P) -> list[dict]:
    """`n` new localized-generation rows for `lang_code`, ids continuing from
    `start_index` — so a later top-up round can append more without colliding
    with rows already generated/judged/kept. Deterministic given the same
    (lang_code, start_index, seed), distinct across successive start_index
    batches (each round advances start_index)."""
    lang_name = LANGUAGES_PHASE3[lang_code]
    country = LANG_COUNTRY[lang_code]
    rng = random.Random(f"{seed}:{lang_code}:{start_index}")

    rows = []
    for i in range(n):
        idx = start_index + i
        is_local = rng.random() >= general_p
        domain = rng.choice(T.LOCAL_DOMAINS if is_local else T.GENERAL_DOMAINS)
        intents = T.LOCAL_INTENTS if is_local else T.GENERAL_INTENTS
        intent = rng.choice(intents)
        role = rng.choice(T.ROLES) if rng.random() < persona_p else None
        salt = rng.randint(1000, 9999)
        rows.append({
            "id": f"{lang_code}-{idx:06d}", "lang": lang_code,
            "domain": domain, "is_local": is_local, "intent": intent,
            "role": role, "salt": salt,
            "meta_prompt": generation_prompt(
                lang_name=lang_name, country=country, domain=domain,
                intent=intent, salt=salt, role=role, localized=is_local),
        })
    return rows


def build(*, n_per_lang: int, seed: int = 0,
         lang_codes: list[str] | None = None) -> list[dict]:
    """All rows for a fresh run (every language starting at index 0)."""
    codes = lang_codes or list(LANGUAGES_PHASE3)
    rows: list[dict] = []
    for code in codes:
        rows.extend(build_rows(n_per_lang, code, start_index=0, seed=seed))
    return rows
