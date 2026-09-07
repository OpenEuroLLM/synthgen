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
"""
from __future__ import annotations

from synthgen.localized import taxonomy as T

# --- Pool A: general / dataset-common -----------------------------------------

# Existing general-capability domains (professional + creative), unchanged.
POOL_A_EXISTING: list[str] = list(T.GENERAL_DOMAINS)

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

POOL_B: list[str] = list(T.LOCAL_DOMAINS)

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
