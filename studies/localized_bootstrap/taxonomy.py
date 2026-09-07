"""Conditioning axes for the localized-instruction bootstrap study.

The domain/role/intent axes were promoted verbatim into
`synthgen.localized.taxonomy` once this study validated the judge — reuse them
from there instead of drifting a second copy. Only the study-specific probe
language list lives here.
"""
from __future__ import annotations

from synthgen.localized.taxonomy import (  # noqa: F401
    DOMAINS,
    GENERAL_DOMAIN_SET,
    GENERAL_DOMAINS,
    GENERAL_INTENTS,
    LOCAL_DOMAINS,
    LOCAL_INTENTS,
    ROLES,
    is_local,
)

# Languages to probe: one high-resource, one mid, one low-resource, mixed scripts.
# (code, English name, country/variety used for localization)
LANGUAGES: list[tuple[str, str, str]] = [
    ("de", "German",   "Germany"),
    ("el", "Greek",    "Greece"),
    ("et", "Estonian", "Estonia"),
    ("mt", "Maltese",  "Malta"),
    ("hi", "Hindi",    "India"),
]

# Backward-compatible default (knowledge/advice style).
INTENTS: list[str] = LOCAL_INTENTS
