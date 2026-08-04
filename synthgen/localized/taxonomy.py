"""Conditioning axes for localized instruction generation.

Promoted from studies/localized_bootstrap/taxonomy.py once the judge-validation
pilot confirmed the judge separates good from bad on this taxonomy. Everything
downstream of the sampled *domain* is either conditioned on it or instantiated
by the generator model, so nothing is stapled on independently.
"""
from __future__ import annotations

# L1 localized domains — answering well REQUIRES country-specific knowledge, so
# these carry the anti-translationese localization signal.
LOCAL_DOMAINS: list[str] = [
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

# General-capability domains — variety beyond local knowledge. Deliberately just
# professional work and creative writing: NO math/code (awkward and low-value
# across languages) and NO abstract/encyclopedic knowledge. These need NOT hinge
# on country facts, but are still written in natural in-language register.
GENERAL_DOMAINS: list[str] = [
    # professional work
    "workplace email & communication",
    "reports, summaries & documentation",
    "job applications & CVs",
    "meetings, planning & project coordination",
    "customer & client communication",
    # creative writing
    "short stories & fiction",
    "poetry & song lyrics",
    "personal essays & reflections",
    "humor, jokes & satire",
    "social media & blog posts",
]

DOMAINS: list[str] = LOCAL_DOMAINS + GENERAL_DOMAINS
GENERAL_DOMAIN_SET: set[str] = set(GENERAL_DOMAINS)


def is_local(domain: str) -> bool:
    """True if the domain demands country-specific knowledge (vs. a general
    professional/creative task that just happens to be written in-language)."""
    return domain not in GENERAL_DOMAIN_SET


# Persona role archetypes — sampled to control the marginal mix; the model
# instantiates a concrete situation/goal from the grounding (veto-down allowed).
ROLES: list[str] = [
    "parent",
    "university student",
    "retiree",
    "small-business owner",
    "recent immigrant",
    "office worker",
    "teenager",
    "tourist",
]

# Realistic user intents. LOCAL_INTENTS fit knowledge/advice requests about a
# place; GENERAL_INTENTS fit professional/creative tasks (drafting, editing).
# The row builder picks the pool matching the sampled domain.
LOCAL_INTENTS: list[str] = [
    "ask (factual question)",
    "ask for a recommendation",
    "ask for help planning something",
    "ask how to do something",
    "ask to compare options",
    "ask for an explanation",
]

GENERAL_INTENTS: list[str] = [
    "ask to draft or write something",
    "ask to improve or edit a draft",
    "ask to summarize or rewrite text",
    "ask for creative help",
    "ask how to do something",
    "ask for an explanation",
]
