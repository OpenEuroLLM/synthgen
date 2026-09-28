"""Embedding-based near-duplicate filtering — a real dedup *filter*, not just
a diagnostic.

`dedupe.py`'s existing pass only catches exact-string duplicates (after
whitespace/case normalization). It does not catch paraphrased near-duplicates
— a real mode-collapse risk for `synthgen.localized`, which conditions
generation on a `salt` text nudge that has no programmatic backstop: nothing
currently checks whether the model actually produced something different, it
just asks it to.

This module is `sentence-transformers`-optional (imported lazily, same
degrade-gracefully convention as everywhere else in this codebase that touches
it): if it isn't installed, `embedding_dedupe()` raises `EmbedderUnavailable`
and callers decide whether to skip the stage or fail.

Deliberately not imported from `studies/localized_axes_ablation/_common/
diversity_metrics.py` — that module lives on the `studies` branch, which is
never merged into `main` (see repo `CLAUDE.md`), and this is production
pipeline code. The default model and threshold below are kept in sync with it
by hand; if you tune one, check the other.
"""
from __future__ import annotations

from collections import defaultdict

from synthgen.log import get_logger

log = get_logger("synthgen.near_dup")

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Multilingual, not the more common all-MiniLM-L6-v2 (English-only-tuned): an
# English-only embedder produces inflated, spurious similarity on non-English
# (especially non-Latin-script) text -- language mismatch, not real semantic
# overlap. Verified empirically on this project's own multilingual data: the
# English-only model flagged a ~33% false near-dup rate on Ukrainian text that
# vanished to 0% under this model. synthgen.localized is multilingual by
# design, so the embedder must be too. See studies/localized_axes_ablation/
# _common/diversity_metrics.py for the full writeup.
DEFAULT_THRESHOLD = 0.85

_embedder = None


class EmbedderUnavailable(RuntimeError):
    """sentence-transformers isn't installed, or the model failed to load."""


def _load_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbedderUnavailable(
                "sentence-transformers not installed -- embedding-based "
                "near-dup filtering unavailable (pip install it once with "
                "internet; it caches and loads fully offline afterward)"
            ) from e
        try:
            _embedder = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbedderUnavailable(f"could not load {model_name!r} ({e})") from e
    return _embedder


def embedding_dedupe_mask(texts: list[str], *, threshold: float = DEFAULT_THRESHOLD,
                           model_name: str = DEFAULT_EMBED_MODEL) -> list[bool]:
    """Keep-mask over `texts`, in order: a text is dropped (False) if its
    cosine similarity to any *already-kept* earlier text in this same list is
    >= `threshold`. Greedy, order-preserving (first occurrence wins) -- same
    convention as `dedupe.py`'s exact-match pass.

    O(n^2) similarity comparisons, vectorized via numpy/BLAS rather than a
    Python double loop -- fine into the tens of thousands of rows per
    language bucket; for much larger batches, swap in an approximate
    nearest-neighbor index (e.g. faiss) instead of this brute-force scan.
    """
    import numpy as np

    n = len(texts)
    if n < 2:
        return [True] * n

    model = _load_embedder(model_name)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    keep_mask = [True] * n
    kept_idx: list[int] = []
    for i in range(n):
        if kept_idx:
            sims = emb[kept_idx] @ emb[i]
            if float(sims.max()) >= threshold:
                keep_mask[i] = False
                continue
        kept_idx.append(i)
    return keep_mask


def embedding_dedupe(rows: list[dict], *, text_key: str = "prompt",
                      lang_key: str = "lang", threshold: float = DEFAULT_THRESHOLD,
                      model_name: str = DEFAULT_EMBED_MODEL,
                      ) -> tuple[list[dict], dict]:
    """Buckets `rows` by `lang_key` -- never pooled across languages, since a
    near-dup rate in one language says nothing about another -- and drops
    embedding near-duplicates within each bucket. Returns
    `(kept_rows, report)`; raises `EmbedderUnavailable` if the embedder can't
    load (caller decides whether to skip this stage or fail the run).
    """
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(rows):
        buckets[r.get(lang_key, "")].append(idx)

    drop: set[int] = set()
    by_lang_dropped: dict[str, int] = {}
    for lang, idxs in buckets.items():
        texts = [rows[i].get(text_key, "") or "" for i in idxs]
        mask = embedding_dedupe_mask(texts, threshold=threshold, model_name=model_name)
        n_dropped = sum(1 for m in mask if not m)
        if n_dropped:
            log.info("embedding near-dup: lang=%s dropped=%d/%d", lang, n_dropped, len(idxs))
        by_lang_dropped[lang] = n_dropped
        for i, keep in zip(idxs, mask):
            if not keep:
                drop.add(i)

    kept_rows = [r for i, r in enumerate(rows) if i not in drop]
    report = {
        "threshold": threshold,
        "model": model_name,
        "total_dropped": len(drop),
        "dropped_by_lang": by_lang_dropped,
    }
    return kept_rows, report
