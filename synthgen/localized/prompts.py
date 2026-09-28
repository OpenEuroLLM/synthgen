"""Localized single-call generation + judge prompts.

Promoted from studies/localized_bootstrap/prompts.py once the judge-validation
pilot confirmed the judge separates good from bad. Production only ever wants
the GOOD path: one model call renders both the instruction and the response as
one JSON object, which the judge then scores.

  build_rows(...)           -> per-row dicts (id, lang, domain, role, salt,
                                meta_prompt) ready for generate.py's
                                mode="localized"
  judge_quality_prompt(...) -> {score, reason}, holistic 0-10, no rubric

**Redesigned per studies/localized_axes_ablation/'s 11 full-scale ablations**
(see taxonomy.py's docstring for the domain-side half of this): `intent` is no
longer sampled from a suggested list -- `03_intent_necessity` found the model
ignores the suggestion ~68% of the time anyway, with no score benefit either
way, so the prompt now just asks the model to decide one itself. Persona rate
dropped `PERSONA_P` 0.5 -> 0.1 (`02_persona_rate`: no quality benefit from the
higher rate). `salt` (the old "diversity seed" nudge) is gone entirely
(`09_salt_necessity`: no measurable diversity benefit, and it had no
programmatic backstop anyway -- see synthgen/pipeline/near_dup.py for the real
one); the RNG draw is kept below purely for stream stability.
"""
from __future__ import annotations

import random

from synthgen.config import LANG_COUNTRY, LANGUAGES_PHASE3
from synthgen.localized import taxonomy as T

# Fraction of examples that even get a persona *suggested* (the model may still
# drop it). The rest are deliberately persona-free direct requests.
PERSONA_P = 0.1


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

If this domain would more naturally take the shape of editing/rewriting
existing text, or extracting/classifying information from it, rather than
open-ended generation, shape the instruction that way -- otherwise write it
as a direct generation request.

If a more specific, higher-difficulty version of this task fits naturally
(one requiring deeper domain expertise or more careful problem-solving to
answer well), prefer it over a generic/easy version -- otherwise keep the
task at a normal difficulty level. Do not force unnecessary complexity.

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

_PERSONA_ON = ("Suggested persona: a {role} — use it ONLY if such a person would "
               "plausibly ask about this domain; otherwise adapt to a role who "
               "would, or drop the persona. If used, invent a specific realistic "
               "situation and goal and let it shape the instruction.")
_PERSONA_OFF = "No persona: write a direct, standalone request."


def generation_prompt(*, lang_name: str, country: str, domain: str,
                      role: str | None = None) -> str:
    persona = _PERSONA_ON.format(role=role) if role else _PERSONA_OFF
    return GENERATION_PROMPT.format(
        lang_name=lang_name, country=country, domain=domain,
        persona_block=persona, fluency_clause=_fluency_clause(lang_name))


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
    NOTE: an edit/rewrite request ("fix this paragraph", "make this shorter")
    or an extraction/classification request ("pull out the dates in this
    text", "which category does this belong to") is a GENUINE real-world
    task, not automatically a quiz item — judge it as you would any other
    task, by whether a real person would plausibly ask it. Likewise, a task
    that requires real expertise or careful multi-step reasoning is not
    "textbook" just because it's hard — contrived academic phrasing
    ("Prove that...", "Given the following axioms...") is the actual
    defect to watch for, not difficulty itself.

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

def _weighted_domains(weights: dict[str, float]) -> tuple[list[str], list[float]]:
    """DOMAINS + matching weight list, uniform fallback for any domain missing
    from `weights` (e.g. before ground_domains.py has been (re)run) so a
    stale/incomplete weights dict degrades gracefully instead of KeyError-ing
    or silently never sampling a domain."""
    if not weights:
        return T.DOMAINS, [1.0] * len(T.DOMAINS)
    default = min(weights.values()) if weights else 1.0
    return T.DOMAINS, [weights.get(d, default) for d in T.DOMAINS]


def build_rows(n: int, lang_code: str, *, start_index: int = 0, seed: int = 0,
              persona_p: float = PERSONA_P) -> list[dict]:
    """`n` new localized-generation rows for `lang_code`, ids continuing from
    `start_index` — so a later top-up round can append more without colliding
    with rows already generated/judged/kept. Deterministic given the same
    (lang_code, start_index, seed), distinct across successive start_index
    batches (each round advances start_index).

    Domain is drawn from `T.DOMAINS`, weighted by `T.DOMAIN_WEIGHTS` (real
    WildChat-1M frequency -- see taxonomy.py's docstring); intent is no
    longer sampled at all (the model decides one itself, per
    GENERATION_PROMPT -- see this module's docstring)."""
    lang_name = LANGUAGES_PHASE3[lang_code]
    country = LANG_COUNTRY[lang_code]
    rng = random.Random(f"{seed}:{lang_code}:{start_index}")
    domains, domain_weights = _weighted_domains(T.DOMAIN_WEIGHTS)

    rows = []
    for i in range(n):
        idx = start_index + i
        domain = rng.choices(domains, weights=domain_weights, k=1)[0]
        role = rng.choice(T.ROLES) if rng.random() < persona_p else None
        # Still drawn (not just dropped) so the RNG stream stays aligned with
        # every prior generation batch -- removing this call would shift the
        # domain/role draws for every row after it. No longer passed into the
        # prompt: the study's 09_salt_necessity ablation found no measurable
        # diversity benefit from surfacing it as a text nudge (see
        # studies/localized_axes_ablation/REPORT.md), and it had no
        # programmatic backstop anyway -- see synthgen/pipeline/near_dup.py
        # for the real one. Kept on the row for provenance/debugging.
        salt = rng.randint(1000, 9999)
        rows.append({
            "id": f"{lang_code}-{idx:06d}", "lang": lang_code,
            "domain": domain, "role": role, "salt": salt,
            "meta_prompt": generation_prompt(
                lang_name=lang_name, country=country, domain=domain, role=role),
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
