"""Embedding-based near-dup filtering: mask logic, per-lang bucketing, and
`dedupe.py`'s optional embedding pass. All tests inject a fake embedder
(deterministic hand-built vectors) rather than loading a real
sentence-transformers model -- keeps this fast and offline.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from synthgen.pipeline import near_dup, dedupe
from synthgen.io import write_jsonl, iter_jsonl


class _FakeEmbedder:
    """Maps each text to a hand-picked vector via `vectors`, in call order."""

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return np.array([self.vectors[t] for t in texts], dtype=float)


def _patch_embedder(monkeypatch, vectors):
    monkeypatch.setattr(near_dup, "_embedder", None)
    monkeypatch.setattr(near_dup, "_load_embedder", lambda model_name=None: _FakeEmbedder(vectors))


def test_mask_drops_near_identical_vectors(monkeypatch):
    # "a" and "b" are near-identical (cosine ~1.0); "c" is orthogonal.
    vectors = {
        "a": [1.0, 0.0],
        "b": [0.99, 0.01],
        "c": [0.0, 1.0],
    }
    _patch_embedder(monkeypatch, vectors)
    mask = near_dup.embedding_dedupe_mask(["a", "b", "c"], threshold=0.9)
    assert mask == [True, False, True]  # b drops, first occurrence (a) wins


def test_mask_keeps_all_when_dissimilar(monkeypatch):
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    _patch_embedder(monkeypatch, vectors)
    assert near_dup.embedding_dedupe_mask(["a", "b"], threshold=0.9) == [True, True]


def test_mask_single_text_always_kept(monkeypatch):
    assert near_dup.embedding_dedupe_mask(["only one"]) == [True]


def test_embedding_dedupe_buckets_by_lang_independently(monkeypatch):
    # Same near-dup pair of vectors, once per language -- both languages
    # should drop their own duplicate, not interact with each other.
    vectors = {
        "fr1": [1.0, 0.0], "fr2": [0.99, 0.01],
        "de1": [1.0, 0.0], "de2": [0.99, 0.01],
    }
    _patch_embedder(monkeypatch, vectors)
    rows = [
        {"lang": "fr", "prompt": "fr1"},
        {"lang": "fr", "prompt": "fr2"},
        {"lang": "de", "prompt": "de1"},
        {"lang": "de", "prompt": "de2"},
    ]
    kept, report = near_dup.embedding_dedupe(rows, threshold=0.9)
    assert len(kept) == 2
    assert {r["prompt"] for r in kept} == {"fr1", "de1"}
    assert report["dropped_by_lang"] == {"fr": 1, "de": 1}
    assert report["total_dropped"] == 2


def test_embedder_unavailable_propagates(monkeypatch):
    def _raise(model_name=None):
        raise near_dup.EmbedderUnavailable("no sentence-transformers")

    monkeypatch.setattr(near_dup, "_embedder", None)
    monkeypatch.setattr(near_dup, "_load_embedder", _raise)
    with pytest.raises(near_dup.EmbedderUnavailable):
        near_dup.embedding_dedupe([{"lang": "fr", "prompt": "a"},
                                    {"lang": "fr", "prompt": "b"}])


def test_dedupe_run_with_embedding_pass_drops_paraphrase(tmp_path, monkeypatch):
    vectors = {
        "Bonjour le monde": [1.0, 0.0],
        "Salut la terre": [0.99, 0.01],   # paraphrase of the above, not exact-dup
        "Autre chose": [0.0, 1.0],
    }
    _patch_embedder(monkeypatch, vectors)

    rows = [
        {"lang": "fr", "prompt": "Bonjour le monde", "response": "reponse un"},
        {"lang": "fr", "prompt": "Salut la terre", "response": "reponse deux"},
        {"lang": "fr", "prompt": "Autre chose", "response": "reponse trois"},
    ]
    src = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    write_jsonl(src, rows)

    summary = dedupe.run(input=src, output=out, embedding_dedupe=True,
                          embedding_threshold=0.9)

    # exact-match pass keeps all 3 (no identical prompts); embedding pass
    # then drops the paraphrase.
    assert summary["dropped_duplicate"] == 0
    assert summary["dropped_near_dup_embedding"] == 1
    assert summary["kept"] == 2
    kept = list(iter_jsonl(out))
    assert {r["prompt"] for r in kept} == {"Bonjour le monde", "Autre chose"}


def test_dedupe_run_embedding_unavailable_is_skipped_not_fatal(tmp_path, monkeypatch):
    def _raise(model_name=None):
        raise near_dup.EmbedderUnavailable("no sentence-transformers")

    monkeypatch.setattr(near_dup, "_embedder", None)
    monkeypatch.setattr(near_dup, "_load_embedder", _raise)

    rows = [{"lang": "fr", "prompt": "a", "response": "reponse un"},
            {"lang": "fr", "prompt": "b", "response": "reponse deux"}]
    src = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    write_jsonl(src, rows)

    summary = dedupe.run(input=src, output=out, embedding_dedupe=True)

    assert summary["kept"] == 2
    assert "embedding_dedupe_skipped" in summary
    assert "dropped_near_dup_embedding" not in summary
