"""Conditioning axes for localized instruction generation.

Promoted from studies/localized_bootstrap/taxonomy.py once the judge-validation
pilot confirmed the judge separates good from bad on this taxonomy. Everything
downstream of the sampled *domain* is either conditioned on it or instantiated
by the generator model, so nothing is stapled on independently.

**Domain/locality redesign** (studies/localized_axes_ablation/, 11 full-scale
ablations): domain used to be a forced binary -- sampling from one of two
disjoint pools (general vs. local) determined which scope clause the prompt
got. That's gone. `DOMAINS` is now a single, real-data-weighted taxonomy
(`DOMAIN_WEIGHTS`, from classifying real WildChat-1M user turns -- see
studies/localized_axes_ablation/01_domain_grounding/ground_domains.py), and
locality/general-angle/task-type/difficulty are four independent, model-judged
optional layers in prompts.py's GENERATION_PROMPT -- the model decides
per-example whether each one fits, nothing is forced. See PLAN.md/REPORT.md
in that study directory for the full rationale and the null/negative results
that ruled out the alternatives (exemplar-anchored grounding measurably hurt
country-specificity; the old is_local binary meant "coding"-type domains could
never carry a cultural angle even when one would fit).
"""
from __future__ import annotations

# The taxonomy: former GENERAL_DOMAINS (professional + creative, unchanged)
# plus former Pool-A-only additions ported from synthgen/pipeline/topics.py's
# TOPIC_NAMES (general-capability categories that existed for the OTHER,
# topic-based route but were never wired into this one). `trivia`/`other` from
# TOPIC_NAMES excluded, same reasoning as this taxonomy's original exclusion
# of encyclopedic trivia.
DOMAINS: list[str] = [
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
    # general-capability (ex-topics.py)
    "coding",
    "math (word problems & explanations)",
    "summarization",
    "translation",
    "data analysis (tables, lists, datasets)",
    "general question answering",
    "roleplay & character play",
    "brainstorming & idea generation",
]

# Real-data sampling weights for DOMAINS, from classifying real WildChat-1M
# user turns (studies/localized_axes_ablation/01_domain_grounding/
# ground_domains.py, n=400/lang classified across es/fr/de/pl/uk -- n=1905
# total after filtering -- aggregated and floor-smoothed at 0.02 so no domain
# is ever unreachable, see that module's `_smoothed_distribution` docstring).
# Re-run ground_domains.py and update this dict if the taxonomy changes.
DOMAIN_WEIGHTS: dict[str, float] = {
    "general question answering": 0.3217,
    "brainstorming & idea generation": 0.1331,
    "coding": 0.1163,
    "reports, summaries & documentation": 0.0525,
    "roleplay & character play": 0.0520,
    "math (word problems & explanations)": 0.0445,
    "translation": 0.0366,
    "short stories & fiction": 0.0361,
    "data analysis (tables, lists, datasets)": 0.0277,
    "social media & blog posts": 0.0267,
    "summarization": 0.0208,
    "workplace email & communication": 0.0189,
    "job applications & CVs": 0.0189,
    "meetings, planning & project coordination": 0.0189,
    "customer & client communication": 0.0189,
    "poetry & song lyrics": 0.0189,
    "personal essays & reflections": 0.0189,
    "humor, jokes & satire": 0.0189,
}

# Local/cultural themes a genuine "{country}-specific angle" (see prompts.py's
# GENERATION_PROMPT) can draw on -- NOT sampled from directly (locality is an
# orthogonal, model-judged layer now, not a domain category), kept here only
# as a reference for what "genuine local angle" is meant to cover.
#   food & drink, public holidays & festivals, bureaucracy & paperwork,
#   transport & commuting, school & education system, healthcare &
#   pharmacies, money/prices & tipping, housing & renting, weather &
#   seasonal life, idioms/slang & humor, local geography & regions, history
#   taught in school, religion & customs, sports & local teams, music/TV &
#   pop culture, shopping & local brands, etiquette & social norms, work
#   culture & leave, legal & administrative rules, recipes & home cooking

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
