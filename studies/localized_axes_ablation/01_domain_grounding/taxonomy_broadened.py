"""Broadened domain taxonomy for this study: two pools instead of one axis.

See ../PLAN.md ("Design decision this study is built around") for the full
rationale. Summary:

  POOL_A (general/dataset-common) — sampled with weights from a real
  dataset's empirical distribution (ground_domains.py). Includes the
  existing `synthgen.localized.taxonomy.GENERAL_DOMAINS` plus the topic
  categories `synthgen/pipeline/topics.py`'s TOPIC_NAMES represents
  (coding, math, summarization, translation, data_analysis, generic qa,
  roleplay, brainstorming) that were never ported to the localized route.
  `trivia` and `other` from TOPIC_NAMES are deliberately excluded, same
  reasoning as the original taxonomy's exclusion of encyclopedic trivia.

  POOL_B (local-culture) — unchanged from `synthgen.localized.taxonomy
  .LOCAL_DOMAINS`. Never weighted by real-dataset frequency (see
  ground_domains.py) — real chat logs underrepresent this shape of request
  by construction, not because it's low-value.

Grounding-depth ceiling: which domains may take the `deep` grounding tier
(country-specific fact as the task's substance) vs. being capped at `light`
(surface texture only — currency, units, dates, names, register). Pool B
domains and the "has a real-world referent" subset of Pool A can go deep;
abstract/self-contained topics are capped at light because forcing "this
coding task's point is a cultural fact" is usually awkward.

**Historical note**: production `synthgen/localized/taxonomy.py` has since
merged POOL_A_EXISTING + POOL_A_DATASET_COMMON into a single `DOMAINS` list
(this study's own recommendation, acted on), and dropped `LOCAL_DOMAINS` from
active sampling entirely (locality is now an orthogonal, model-judged prompt
layer, not a domain pool -- see that module's docstring). This file no longer
imports those production names (they don't exist there anymore) and instead
keeps its own frozen copies below, exactly as they were when this study ran --
so `ground_domains.py` (which still imports `TB.POOL_A`) keeps working for
future re-classification without depending on production's current shape.
"""
from __future__ import annotations

# --- Pool A: general / dataset-common -----------------------------------------

# Existing general-capability domains (professional + creative) as of this
# study's run -- frozen copy, NOT imported from production taxonomy.py
# anymore (see module docstring's historical note).
POOL_A_EXISTING: list[str] = [
    "workplace email & communication",
    "reports, summaries & documentation",
    "job applications & CVs",
    "meetings, planning & project coordination",
    "customer & client communication",
    "short stories & fiction",
    "poetry & song lyrics",
    "personal essays & reflections",
    "humor, jokes & satire",
    "social media & blog posts",
]

# New: ported from synthgen/pipeline/topics.py's TOPIC_NAMES, which already
# exists for the OTHER (topic-based) pipeline but was never wired into the
# localized route. `trivia`/`other` excluded (see module docstring).
POOL_A_DATASET_COMMON: list[str] = [
    "coding",
    "math (word problems & explanations)",
    "summarization",
    "translation",
    "data analysis (tables, lists, datasets)",
    "general question answering",
    "roleplay & character play",
    "brainstorming & idea generation",
]

POOL_A: list[str] = POOL_A_EXISTING + POOL_A_DATASET_COMMON

# --- Pool B: local-culture, unchanged -----------------------------------------

# Frozen copy of the local-culture domains as of this study's run -- not
# imported from production (see module docstring's historical note); kept
# only as a comment there now, not an active list.
POOL_B: list[str] = [
    "food & drink",
    "public holidays & festivals",
    "bureaucracy & paperwork",
    "transport & commuting",
    "school & education system",
    "healthcare & pharmacies",
    "money, prices & tipping",
    "housing & renting",
    "weather & seasonal life",
    "idioms, slang & humor",
    "local geography & regions",
    "history taught in school",
    "religion & customs",
    "sports & local teams",
    "music, TV & pop culture",
    "shopping & local brands",
    "etiquette & social norms",
    "work culture & leave",
    "legal & administrative rules",
    "recipes & home cooking",
]

ALL_DOMAINS: list[str] = POOL_A + POOL_B
POOL_A_SET: set[str] = set(POOL_A)


def is_pool_a(domain: str) -> bool:
    return domain in POOL_A_SET


# --- grounding-depth ceiling ---------------------------------------------------

GroundingMode = str  # "none" | "light" | "deep"

# Pool A domains allowed to reach "deep" (they have a real-world cultural
# referent to source from). Everything else in Pool A is capped at "light".
POOL_A_DEEP_ELIGIBLE: set[str] = {
    "general question answering",
    "roleplay & character play",
    "brainstorming & idea generation",
}


def max_grounding_mode(domain: str) -> GroundingMode:
    """The deepest grounding tier this domain may use. All of Pool B, plus the
    Pool A domains with a real cultural referent (`POOL_A_DEEP_ELIGIBLE`), may
    go "deep"; every other Pool A domain (coding, math, summarization,
    translation, data_analysis, and the existing professional/creative
    domains) is capped at "light" — see module docstring."""
    if domain in POOL_A_SET and domain not in POOL_A_DEEP_ELIGIBLE:
        return "light"
    return "deep"


GROUNDING_MODES: tuple[str, ...] = ("none", "light", "deep")
