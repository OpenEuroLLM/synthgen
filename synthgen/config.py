"""Typed configuration for synthgen.

Constants (LANGUAGES, PERSONAS, CONSTRAINTS, ...) live here; runtime overrides
go through SynthConfig (instantiated by the CLI). Path layout is in Paths.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ----- Language sets -------------------------------------------------------

LANGUAGES_PILOT: dict[str, str] = {
    "el": "Greek",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "cs": "Czech",
    "pl": "Polish",
}

LANGUAGES_PHASE3: dict[str, str] = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "nl": "Dutch",
    "cs": "Czech",
    "ro": "Romanian",
    "el": "Greek",
    "uk": "Ukrainian",
}

LANGUAGES_DOLCI_TRAINED = ["es", "fr", "de", "it", "pt", "pl", "nl", "cs"]
LANGUAGES_HELDOUT = ["ro", "el", "uk"]

LANG_SCRIPT: dict[str, str] = {
    "el": "non_latin", "uk": "non_latin",
    "ro": "latin", "cs": "latin", "pl": "latin",
    "es": "latin", "fr": "latin", "de": "latin",
    "it": "latin", "pt": "latin", "nl": "latin",
}


# ----- Generation axes -----------------------------------------------------

TOPICS: list[tuple[str, float]] = [
    ("creative_writing", 0.15),
    ("qa",               0.12),
    ("translation",      0.10),
    ("math",             0.08),
    ("summarization",    0.08),
    ("trivia",           0.07),
    ("brainstorming",    0.07),
    ("roleplay",         0.06),
    ("coding",           0.05),
    ("data_analysis",    0.05),
    ("other",            0.17),
]

PERSONAS: list[str] = [
    "a university student preparing for an exam",
    "a working professional on a tight deadline",
    "a curious teenager exploring a new hobby",
    "a researcher writing for peer audience",
    "a casual user chatting after work",
    "a parent helping their child with homework",
    "a small-business owner solving a practical problem",
    "a non-native speaker still learning the language",
]

# (name, instruction template, script_safe_for_non_latin)
CONSTRAINTS: list[tuple[str, str, bool]] = [
    ("format_bullets",       "Respond using exactly {n} bullet points.", True),
    ("format_numbered",      "Respond as a numbered list with {n} items.", True),
    ("format_json",          "Respond as a single JSON object with keys: {keys}.", True),
    ("length_min_words",     "Your answer must be at least {n} words long.", True),
    ("length_max_words",     "Your answer must be at most {n} words long.", True),
    ("length_sentences",     "Your answer must be exactly {n} sentences.", True),
    ("structural_sections",  "Structure your answer under these headings: {sections}.", True),
    ("structural_paragraphs","Your answer must contain exactly {n} paragraphs separated by a blank line.", True),
    ("keyword_include",      "Your answer must include the word \"{kw}\" verbatim.", True),
    ("keyword_exclude",      "Your answer must not contain the word \"{kw}\".", True),
    ("end_with",             "End your answer with the exact phrase: \"{phrase}\".", True),
    ("start_with",           "Begin your answer with the exact phrase: \"{phrase}\".", True),
    ("casing_all_caps",      "Your entire answer must be in UPPERCASE.", False),
    ("casing_title",         "Every word in your answer must start with a capital letter.", False),
    ("letter_freq",          "The letter \"{ch}\" must appear at least {n} times in your answer.", False),
]


# ----- Runtime config ------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    """Resolved on-disk layout. All other paths derive from `root`."""
    root: Path
    prompts: Path
    outputs: Path
    review: Path
    logs: Path

    @classmethod
    def from_root(cls, root: str | os.PathLike[str] | None = None) -> "Paths":
        r = Path(root) if root else Path(os.environ.get("SYNTHGEN_ROOT", Path.cwd()))
        r = r.resolve()
        return cls(
            root=r,
            prompts=r / "prompts",
            outputs=r / "outputs",
            review=r / "review",
            logs=r / "logs",
        )

    def ensure(self) -> None:
        for p in (self.prompts, self.outputs, self.review, self.logs):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class SynthConfig:
    # Phase selection.
    phase: int = 3
    n_per_lang_pilot: int = 50
    n_per_lang_phase3: int = 200_000

    # Generator pools (one or many; production phase 3 splits across them).
    generators_pilot: tuple[str, ...] = (
        "google/gemma-4-26b-a4b-it",
        "qwen/qwen-2.5-72b-instruct",
        "anthropic/claude-sonnet-4.6",
    )
    generators_phase3: tuple[str, ...] = (
        "google/gemma-4-26b-a4b-it",
        "openai/gpt-oss-120b",
    )
    back_translator: str = "openai/gpt-4.1-nano"

    # Prompt axis mix probabilities (phase-3 production).
    persona_p: float = 0.2
    unconstrained_p: float = 0.3
    topic_source: str = "static"  # "static" | "empirical"

    # Sampling / request.
    temperature: float = 0.9
    top_p: float = 0.95
    max_tokens: int = 1024
    concurrency: int = 16
    max_retries: int = 5
    retry_base_delay_s: float = 2.0

    # Paths (resolved separately so they can be swapped in tests).
    paths: Paths = field(default_factory=Paths.from_root)

    # ----- convenience accessors -----
    @property
    def languages(self) -> dict[str, str]:
        return LANGUAGES_PHASE3 if self.phase == 3 else LANGUAGES_PILOT

    @property
    def n_per_lang(self) -> int:
        return self.n_per_lang_phase3 if self.phase == 3 else self.n_per_lang_pilot

    @property
    def generators(self) -> tuple[str, ...]:
        return self.generators_phase3 if self.phase == 3 else self.generators_pilot

    @property
    def sampling(self) -> dict:
        return {"temperature": self.temperature, "top_p": self.top_p,
                "max_tokens": self.max_tokens}
