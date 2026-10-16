"""Corrected dedup sizing benchmark (debug partition, not login node).

The first attempt (production/35lang_run/_dedup_sizing_check.py) used only
10 vocabulary words per fake text -- that made ~99.5% of generated texts
collide as "near-duplicates", which collapses embedding_dedupe_mask's
`kept_idx` list to a tiny near-constant size regardless of n, making the
observed cost look linear in n instead of its real O(n^2) worst case. Real
generated instructions are far more lexically diverse than that, so the
realistic near-dup drop rate is low (single-digit to low tens of percent,
not 99.5%) -- meaning `kept_idx` grows close to n for most of the scan, the
actual worst case this benchmark needs to measure.

This version isolates exactly that worst case directly: random unit vectors
in the same embedding dimensionality (384, paraphrase-multilingual-MiniLM-
L12-v2's output size) very rarely exceed the 0.85 cosine-similarity
threshold by chance, so kept_idx grows ~linearly with n (near-zero drop
rate) -- the real worst-case cost driver, isolated from the embedding-encode
step (which is a separate, cheap, linear-time cost already observed to be
fast). Runs the SAME keep/drop loop structure as
synthgen.pipeline.near_dup.embedding_dedupe_mask, just fed vectors directly
instead of running them through SentenceTransformer.encode().
"""
from __future__ import annotations

import time

import numpy as np


def keep_drop_loop(emb: np.ndarray, threshold: float = 0.85) -> int:
    n = emb.shape[0]
    kept_idx: list[int] = []
    dropped = 0
    for i in range(n):
        if kept_idx:
            sims = emb[kept_idx] @ emb[i]
            if float(sims.max()) >= threshold:
                dropped += 1
                continue
        kept_idx.append(i)
    return dropped


def make_random_unit_vectors(n: int, d: int = 384, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(n, d).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


if __name__ == "__main__":
    budget_s = 12 * 60  # stay well inside the 25-min job time limit, leave margin
    start = time.time()
    sizes = [2000, 4000, 8000, 16000, 32000]
    results = []
    for n in sizes:
        elapsed = time.time() - start
        if elapsed > budget_s:
            print(f"[budget] stopping before n={n}, {elapsed:.0f}s elapsed", flush=True)
            break
        emb = make_random_unit_vectors(n)
        t0 = time.time()
        dropped = keep_drop_loop(emb)
        t1 = time.time()
        dt = t1 - t0
        results.append((n, dt))
        print(f"[dedup_timing_v2] n={n} time={dt:.2f}s dropped={dropped} (drop_rate={dropped/n:.4f})", flush=True)

    # fit against n^2 using the two largest samples for a stable slope
    if len(results) >= 2:
        (n1, t1_), (n2, t2_) = results[-2], results[-1]
        # c2 solved from the ratio of the two largest points (more robust to
        # small-n overhead than using the single largest point alone)
        import math
        c2 = (t2_ - t1_) / (n2 ** 2 - n1 ** 2) if n2 != n1 else t2_ / (n2 ** 2)
        print(f"\n[fit_v2] c2={c2:.3e} (t ~= c2 * n^2), fit from n={n1}->{n2}", flush=True)

        for label, n_proj in [("avg lang (142733 kept)", 142733),
                               ("largest lang (209024 kept)", 209024)]:
            est_s = c2 * (n_proj ** 2)
            print(f"[projection_v2] {label}: estimated dedup time "
                  f"~{est_s:.1f}s ({est_s/60:.1f} min) for that bucket alone")

        # sum across the real 35-language kept-count distribution (using the
        # same per-lang counts and 0.8237 judge pass rate found by the first
        # sizing job)
        lang_counts = {
            "lv": 253761, "eu": 237910, "is": 180952, "sq": 176323, "cs": 174250,
            "et": 174211, "lt": 172147, "sv": 171527, "sk": 170638, "sl": 170420,
            "mt": 170274, "de": 170028, "fi": 170025, "pt": 169554, "hu": 169536,
            "da": 169152, "ga": 168956, "el": 168339, "uk": 168207, "ro": 168078,
            "no": 167988, "fr": 167751, "pl": 167747, "it": 167697, "ka": 166821,
            "es": 165888, "hr": 165686, "tr": 165516, "ca": 165399, "bg": 165330,
            "nl": 165268, "bs": 165228, "gl": 164960, "mk": 164758, "sr": 164552,
        }
        pass_rate = 0.8237
        total_est = sum(c2 * ((v * pass_rate) ** 2) for v in lang_counts.values())
        print(f"[projection_v2] sum across all 35 language buckets (sequential): "
              f"~{total_est:.1f}s (~{total_est/60:.1f} min, ~{total_est/3600:.2f} h)")
    else:
        print("[fit_v2] not enough data points collected within budget to fit")
