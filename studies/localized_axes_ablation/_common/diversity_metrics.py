"""Near-duplicate / mode-collapse metrics, shared by every ablation in this study.

Two independent signals, deliberately kept separate rather than merged into one
score — they catch different failure modes and a reader needs to know which one
fired:

  lexical_near_dup_rate(...)    word n-gram Jaccard overlap. Cheap, no model
                                 download, catches near-identical PHRASING.
  embedding_near_dup_rate(...)  cosine similarity of sentence embeddings
                                 (sentence-transformers/all-MiniLM-L6-v2,
                                 downloaded once and cached, then loaded fully
                                 offline). Catches PARAPHRASED / semantic
                                 collapse that lexical checks miss.

Both operate on a flat list of strings (usually `instruction` text) and return
the same shape: {"n": ..., "near_dup_pairs": ..., "near_dup_rate": ...,
"pairwise_scores": [...]} so `report.py` can histogram either one identically.

`diversity_report(rows, group_keys=...)` is the entry point every ablation
script should call: it buckets rows by e.g. (lang, domain) and runs both
metrics per bucket, so results are never accidentally pooled across languages
or domains (which would hide a per-language collapse behind an aggregate).
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Callable

# --- lexical -----------------------------------------------------------------


def _word_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = text.lower().split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lexical_near_dup_rate(texts: list[str], *, n: int = 3,
                          threshold: float = 0.5) -> dict:
    """Pairwise word-{n}-gram Jaccard similarity over `texts`.

    `threshold`: pairs at or above this Jaccard score count as near-duplicate.
    0.5 is a reasonable starting point for n=3 (roughly: half the trigrams
    shared) — tune per corpus rather than trusting it blindly.
    """
    grams = [_word_ngrams(t, n=n) for t in texts]
    scores: list[float] = []
    dup_pairs: list[tuple[int, int, float]] = []
    for (i, gi), (j, gj) in combinations(enumerate(grams), 2):
        s = _jaccard(gi, gj)
        scores.append(s)
        if s >= threshold:
            dup_pairs.append((i, j, s))
    return {
        "n": len(texts),
        "metric": f"lexical_jaccard_n{n}",
        "threshold": threshold,
        "near_dup_pairs": len(dup_pairs),
        "near_dup_rate": (len(dup_pairs) / len(scores)) if scores else 0.0,
        "pairwise_scores": scores,
        "examples": dup_pairs[:10],  # (i, j, score) — first few, for spot-checking
    }


def self_bleu_proxy(texts: list[str], *, n: int = 3) -> float:
    """Cheap self-BLEU-style proxy: mean pairwise n-gram Jaccard over the whole
    set. Not real BLEU (no brevity penalty, no precision/recall split, no nltk
    dependency) — use `lexical_near_dup_rate` for the actual near-dup rate;
    this is only a single scalar for quick eyeballing across arms/conditions.
    """
    r = lexical_near_dup_rate(texts, n=n, threshold=1.1)  # threshold >1: never trips
    scores = r["pairwise_scores"]
    return sum(scores) / len(scores) if scores else 0.0


# --- embedding -----------------------------------------------------------------

_embedder = None  # lazy singleton, loaded once per process

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# NOT all-MiniLM-L6-v2: that model is English-only-tuned. Verified empirically
# on this study's own data -- it flagged a spurious ~33% "near-dup rate" on
# Ukrainian (uk) text that vanished (0%) under this multilingual model. Using
# an English-only embedder on non-English (especially non-Latin-script) text
# produces inflated, spurious similarity, not a real diversity signal -- this
# study is multilingual by design, so the embedder must be too.


class EmbedderUnavailable(RuntimeError):
    """Raised when sentence-transformers isn't installed / the model can't be
    loaded — caught by diversity_report() so a run degrades to lexical-only
    metrics instead of crashing (this is a `RuntimeError`, not `SystemExit`,
    specifically so callers can catch it without also swallowing Ctrl-C)."""


def _load_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbedderUnavailable(
                "sentence-transformers not installed (needed for embedding "
                "diversity metrics; pip install it once with internet, then "
                "it's cached and loads fully offline) -- falling back to "
                "lexical-only diversity metrics"
            ) from e
        try:
            _embedder = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbedderUnavailable(
                f"could not load {model_name!r} ({e}) -- falling back to "
                "lexical-only diversity metrics"
            ) from e
    return _embedder


def embedding_near_dup_rate(texts: list[str], *, threshold: float = 0.85,
                            model_name: str = DEFAULT_EMBED_MODEL
                            ) -> dict:
    """Pairwise cosine similarity of sentence embeddings.

    `threshold`: 0.85 is a conservative starting point for this model —
    tighten/loosen after eyeballing a few flagged pairs by hand; don't trust
    it as calibrated out of the box.
    """
    import numpy as np

    if len(texts) < 2:
        return {"n": len(texts), "metric": "embedding_cosine", "threshold": threshold,
                "near_dup_pairs": 0, "near_dup_rate": 0.0, "pairwise_scores": [],
                "examples": []}

    model = _load_embedder(model_name)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    sims = emb @ emb.T  # cosine similarity, since embeddings are normalized

    scores: list[float] = []
    dup_pairs: list[tuple[int, int, float]] = []
    for i, j in combinations(range(len(texts)), 2):
        s = float(sims[i, j])
        scores.append(s)
        if s >= threshold:
            dup_pairs.append((i, j, s))
    return {
        "n": len(texts),
        "metric": "embedding_cosine",
        "model": model_name,
        "threshold": threshold,
        "near_dup_pairs": len(dup_pairs),
        "near_dup_rate": (len(dup_pairs) / len(scores)) if scores else 0.0,
        "pairwise_scores": scores,
        "examples": dup_pairs[:10],
    }


# --- grouped entry point -------------------------------------------------------

def diversity_report(rows: list[dict], *, text_key: str = "instruction",
                     group_keys: tuple[str, ...] = ("lang",),
                     lexical_threshold: float = 0.5,
                     embedding_threshold: float = 0.85,
                     embedding_model: str = DEFAULT_EMBED_MODEL,
                     include_embedding: bool = True,
                     max_per_group: int | None = 200) -> dict:
    """Buckets `rows` by `group_keys` (default: `lang` only) and runs both
    metrics per bucket — deliberately never pooled across groups, since an
    aggregate near-dup rate can hide a collapse in one language behind
    healthy diversity in the rest.

    Default is `("lang",)`, NOT `("lang", "domain")`: domain is sampled
    near-uniformly across ~20-38 labels, so at pilot scale (n_per_lang~30-40)
    a per-domain bucket has under 1 expected row — nowhere near enough for a
    pairwise near-dup rate to mean anything. Getting a trustworthy per-domain
    number needs ~20x more rows per language (20+ examples per domain
    bucket). Pass `group_keys=("lang", "domain")` explicitly only for a
    domain-restricted follow-up run (a handful of domains, oversampled), not
    the first-pass pilot.

    `max_per_group`: pairwise metrics are O(n^2); cap group size (random
    subsample) rather than let one oversized bucket blow up runtime.
    """
    import random

    # Once sentence-transformers proves unavailable/broken, stop retrying it
    # for every remaining group/pooled call in this process -- log once, note
    # it once in the output, and fall back to lexical-only for the rest.
    embedding_disabled_reason: str | None = None

    def _try_embedding(texts: list[str]) -> dict | None:
        nonlocal embedding_disabled_reason
        if embedding_disabled_reason is not None:
            return None
        try:
            return embedding_near_dup_rate(texts, threshold=embedding_threshold,
                                            model_name=embedding_model)
        except EmbedderUnavailable as e:
            embedding_disabled_reason = str(e)
            return None

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k) for k in group_keys)
        buckets[key].append(r)

    out: dict = {"group_keys": list(group_keys), "groups": {}}
    for key, group_rows in buckets.items():
        texts = [r[text_key] for r in group_rows if r.get(text_key)]
        if max_per_group and len(texts) > max_per_group:
            texts = random.Random(0).sample(texts, max_per_group)
        entry: dict = {"n": len(texts)}
        if len(texts) >= 2:
            entry["lexical"] = lexical_near_dup_rate(texts, threshold=lexical_threshold)
            if include_embedding:
                emb = _try_embedding(texts)
                if emb is not None:
                    entry["embedding"] = emb
        label = ":".join(str(k) for k in key)
        out["groups"][label] = entry

    # overall (pooled) numbers too, clearly labeled as pooled so they're never
    # mistaken for a per-group figure
    all_texts = [r[text_key] for r in rows if r.get(text_key)]
    if max_per_group and len(all_texts) > max_per_group:
        all_texts = random.Random(0).sample(all_texts, max_per_group)
    if len(all_texts) >= 2:
        out["pooled_lexical"] = lexical_near_dup_rate(all_texts, threshold=lexical_threshold)
        if include_embedding:
            emb = _try_embedding(all_texts)
            if emb is not None:
                out["pooled_embedding"] = emb

    if embedding_disabled_reason is not None:
        out["embedding_unavailable"] = embedding_disabled_reason
    return out
