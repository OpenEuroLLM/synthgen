"""Prompt building: axis-mix probabilities, script safety, seed determinism."""
from __future__ import annotations

import random

from synthgen.config import CONSTRAINTS, Paths, SynthConfig
from synthgen.io import iter_jsonl
from synthgen.prompts import build, build_one

ONE_TOPIC = [("qa", 1.0)]
_SCRIPT_SAFE = {c[0] for c in CONSTRAINTS if c[2]}


def test_persona_p_zero_never_adds_persona():
    rng = random.Random(0)
    for _ in range(50):
        row = build_one(rng, "fr", "French", persona_p=0.0,
                        unconstrained_p=0.0, topics=ONE_TOPIC)
        assert row["persona"] is None


def test_persona_p_one_always_adds_persona():
    rng = random.Random(0)
    for _ in range(50):
        row = build_one(rng, "fr", "French", persona_p=1.0,
                        unconstrained_p=0.0, topics=ONE_TOPIC)
        assert row["persona"] is not None


def test_unconstrained_p_one_never_adds_constraint():
    rng = random.Random(0)
    for _ in range(50):
        row = build_one(rng, "fr", "French", persona_p=0.0,
                        unconstrained_p=1.0, topics=ONE_TOPIC)
        assert row["constraint_name"] is None
        assert row["constraint_text"] is None


def test_non_latin_script_only_gets_script_safe_constraints():
    # Greek is non-latin: casing_* / letter_freq (script_safe=False) must be excluded.
    rng = random.Random(0)
    for _ in range(200):
        row = build_one(rng, "el", "Greek", persona_p=0.0,
                        unconstrained_p=0.0, topics=ONE_TOPIC)
        assert row["constraint_name"] in _SCRIPT_SAFE


def test_constraint_text_is_filled_no_placeholders():
    rng = random.Random(1)
    for _ in range(200):
        row = build_one(rng, "ro", "Romanian", persona_p=0.0,
                        unconstrained_p=0.0, topics=ONE_TOPIC)
        if row["constraint_text"]:
            assert "{" not in row["constraint_text"]


def _cfg(tmp_path, **kw):
    return SynthConfig(paths=Paths.from_root(tmp_path), **kw)


def test_build_is_deterministic_for_a_seed(tmp_path):
    a = build(_cfg(tmp_path / "a"), seed=7, n_override=3)
    b = build(_cfg(tmp_path / "b"), seed=7, n_override=3)
    assert a.read_text() == b.read_text()


def test_build_emits_n_per_language_with_ids(tmp_path):
    cfg = _cfg(tmp_path, phase=3)
    out = build(cfg, seed=0, n_override=2)
    rows = list(iter_jsonl(out))
    assert len(rows) == 2 * len(cfg.languages)
    assert {r["id"] for r in rows} >= {"fr-00000", "fr-00001"}
    assert all(r["meta_prompt"] for r in rows)


def test_phase1_forces_persona_and_constraint(tmp_path):
    # phase 1 overrides to persona_p=1.0, unconstrained_p=0.0 inside build().
    out = build(_cfg(tmp_path, phase=1), seed=0, n_override=4)
    rows = list(iter_jsonl(out))
    assert all(r["persona"] for r in rows)
    assert all(r["constraint_name"] for r in rows)
