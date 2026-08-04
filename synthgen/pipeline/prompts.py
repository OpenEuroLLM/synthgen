"""Build meta-prompts. Axes: language, topic, [persona], [constraint].

Phase 1 (eyeball pilot): persona always on, constraint always on.
Phase 3 (production): persona on with prob cfg.persona_p, constraint on with prob
1 - cfg.unconstrained_p.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from synthgen.config import (
    CONSTRAINTS,
    LANG_SCRIPT,
    PERSONAS,
    TOPICS,
    SynthConfig,
)
from synthgen.io import write_jsonl
from synthgen.log import get_logger

log = get_logger("synthgen.prompts")

TOPIC_SEED = {
    "creative_writing": "write a short piece of creative fiction or poetry",
    "qa":               "ask a factual question that has a clear answer",
    "translation":      "ask the assistant to translate a short passage",
    "math":             "pose a word problem requiring arithmetic or basic algebra",
    "summarization":    "ask the assistant to summarize a short paragraph (include the paragraph in the prompt)",
    "trivia":           "ask a trivia question about culture, history, or science",
    "brainstorming":    "ask the assistant to brainstorm ideas for a real-life situation",
    "roleplay":         "set up a roleplay scenario the assistant should act out",
    "coding":           "ask for a small code snippet solving a concrete task",
    "data_analysis":    "give a small table or list and ask for a specific analysis",
    "other":            "ask any natural everyday request you might type to an assistant",
}

HEADER = """You are generating one realistic instruction-following training example.

Target language: {lang_name} ({lang_code}). Write the user prompt entirely in {lang_name}, using the native script. Do not include any English except for fixed technical tokens (e.g. JSON keys) when a constraint demands it.

Topic: {topic_hint}
"""
PERSONA_LINE = "The prompt should sound like {persona} would naturally write it.\n"
CONSTRAINT_LINE = ("The prompt MUST embed this verifiable constraint, translated naturally "
                   "into {lang_name}:\n  {constraint_text}\n")
FOOTER = "\nOutput ONLY the user prompt itself. No preamble, no explanation, no quotes around it. Just the prompt that a real user would type."


def load_topics(cfg: SynthConfig) -> list[tuple[str, float]]:
    if cfg.topic_source == "empirical":
        path = cfg.paths.root / "topic_distribution.json"
        if path.exists():
            dist = json.loads(path.read_text())["distribution"]
            return [(t, w) for t, w in dist.items() if w > 0]
    return list(TOPICS)


def _weighted_choice(rng: random.Random, pairs: list[tuple[str, float]]) -> str:
    r = rng.random()
    acc = 0.0
    for item, w in pairs:
        acc += w
        if r <= acc:
            return item
    return pairs[-1][0]


def _fill_constraint(rng: random.Random, name: str, template: str) -> str:
    n_small = rng.randint(3, 7)
    n_words_min = rng.choice([50, 80, 120, 150])
    n_words_max = rng.choice([40, 60, 80, 100])
    if name == "format_json":
        keys = ", ".join(rng.sample(
            ["title", "summary", "items", "answer", "steps", "reason", "tags"], 3))
        return template.format(keys=keys)
    if name == "structural_sections":
        sections = ", ".join(rng.sample(
            ["Introduction", "Details", "Examples", "Conclusion", "Caveats"], 3))
        return template.format(sections=sections)
    if name in ("format_bullets", "format_numbered", "length_sentences", "structural_paragraphs"):
        return template.format(n=n_small)
    if name == "length_min_words":
        return template.format(n=n_words_min)
    if name == "length_max_words":
        return template.format(n=n_words_max)
    if name in ("keyword_include", "keyword_exclude"):
        kw = rng.choice(["history", "future", "river", "music", "science",
                         "family", "freedom", "memory"])
        return template.format(kw=kw)
    if name in ("end_with", "start_with"):
        phrase = rng.choice(["That is my answer.", "In conclusion.",
                             "Here we go:", "Final thought."])
        return template.format(phrase=phrase)
    if name == "letter_freq":
        return template.format(ch=rng.choice(["e", "a", "s", "r"]),
                               n=rng.randint(5, 12))
    return template  # casing_*


def build_one(rng: random.Random, lang_code: str, lang_name: str,
              persona_p: float, unconstrained_p: float,
              topics: list[tuple[str, float]]) -> dict:
    topic = _weighted_choice(rng, topics)
    script = LANG_SCRIPT[lang_code]

    persona = rng.choice(PERSONAS) if rng.random() < persona_p else None

    use_constraint = rng.random() >= unconstrained_p
    cname = ctext = None
    if use_constraint:
        pool = [c for c in CONSTRAINTS if c[2] or script == "latin"]
        cname, ctmpl, _ = rng.choice(pool)
        ctext = _fill_constraint(rng, cname, ctmpl)

    parts = [HEADER.format(lang_name=lang_name, lang_code=lang_code,
                           topic_hint=TOPIC_SEED[topic])]
    if persona:
        parts.append(PERSONA_LINE.format(persona=persona))
    if ctext:
        parts.append(CONSTRAINT_LINE.format(lang_name=lang_name, constraint_text=ctext))
    parts.append(FOOTER)

    return {
        "lang": lang_code, "topic": topic, "persona": persona,
        "constraint_name": cname, "constraint_text": ctext,
        "meta_prompt": "".join(parts),
    }


def build(cfg: SynthConfig, *, seed: int = 0, n_override: int | None = None) -> Path:
    """Build the prompts JSONL for the configured phase. Returns the output path."""
    cfg.paths.ensure()
    out = cfg.paths.prompts / "prompts.jsonl"
    if cfg.phase == 1:
        persona_p, unconstrained_p = 1.0, 0.0
    else:
        persona_p, unconstrained_p = cfg.persona_p, cfg.unconstrained_p
    n = n_override or cfg.n_per_lang
    topics = load_topics(cfg)
    rng = random.Random(seed)

    rows: list[dict] = []
    for lang_code, lang_name in cfg.languages.items():
        for i in range(n):
            row = build_one(rng, lang_code, lang_name, persona_p, unconstrained_p, topics)
            row["id"] = f"{lang_code}-{i:05d}"
            rows.append(row)

    write_jsonl(out, rows)
    n_persona = sum(1 for r in rows if r["persona"])
    n_constr = sum(1 for r in rows if r["constraint_name"])
    log.info("wrote %d prompts to %s", len(rows), out)
    log.info("  with_persona=%d (%.0f%%)  with_constraint=%d (%.0f%%)",
             n_persona, 100 * n_persona / len(rows),
             n_constr, 100 * n_constr / len(rows))
    return out
