"""Threshold sweep for the embedding near-dup pass (within ONE language only).

For each language: sample up to N prompts from loc_full.kept.jsonl, embed once
(same model as production), then apply near_dup's greedy keep/drop rule at
several cosine thresholds and at several sample sizes (to see how the drop rate
grows with n, for projecting to the full ~157k/lang). Also prints example
(prompt, most-similar earlier prompt) pairs per similarity band so the
threshold can be chosen by eye. Writes nothing.

usage: python _dedupe_threshold_sweep.py <kept.jsonl> <model_dir> <n_per_lang>
                                         [langs,comma,sep|all] [thresholds,comma,sep]
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer

from synthgen.pipeline import near_dup

THRESHOLDS = [0.85, 0.90, 0.93, 0.95, 0.97]
BANDS = [(0.85, 0.90), (0.90, 0.93), (0.93, 0.95), (0.95, 0.97), (0.97, 1.01)]
EXAMPLE_LANGS = ["de", "es", "fr", "pl", "hr", "fi", "mt", "ga", "eu", "is"]
MODEL_LANGS = set("ar bg ca cs da de el en es et fa fi fr gl gu he hi hr hu hy id it ja ka ko ku "
                  "lt lv mk mn mr ms my nb nl pl pt ro ru sk sl sq sr sv th tr uk ur vi".split()) | {"no"}


def main() -> None:
    global THRESHOLDS, BANDS, EXAMPLE_LANGS
    path, model_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    only = None
    if len(sys.argv) > 4 and sys.argv[4] != "all":
        only = set(sys.argv[4].split(","))
        EXAMPLE_LANGS = sorted(only)
    if len(sys.argv) > 5:
        THRESHOLDS = [float(x) for x in sys.argv[5].split(",")]
        BANDS = list(zip(THRESHOLDS, THRESHOLDS[1:] + [1.01]))
    t0 = time.time()
    by_lang: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if only is not None and r.get("lang") not in only:
                continue
            lst = by_lang[r.get("lang", "")]
            if len(lst) < n:
                lst.append(r.get("prompt", "") or "")
    print(f"sampled {len(by_lang)} languages, up to {n} each, in {time.time() - t0:.0f}s", flush=True)

    sizes = sorted({max(n // 3, 1), max(2 * n // 3, 1), n})
    model = SentenceTransformer(model_dir, device="cuda")
    table: dict[str, dict] = {}
    examples: dict[str, dict] = {}
    rng = random.Random(0)
    for lang in sorted(by_lang):
        texts = by_lang[lang]
        emb = model.encode(texts, batch_size=256, normalize_embeddings=True,
                           convert_to_tensor=True, show_progress_bar=False).float()
        res = {}
        for s in sizes:
            if s > len(texts):
                continue
            for t in THRESHOLDS:
                m = near_dup.greedy_keep_mask_blocked(emb[:s], t)
                res[(s, t)] = m.count(False) / s
        table[lang] = res

        if lang in EXAMPLE_LANGS:
            # most similar EARLIER prompt for every row (block-wise, lower triangle)
            nn_val = torch.full((len(texts),), -1.0, device=emb.device)
            nn_idx = torch.zeros(len(texts), dtype=torch.long, device=emb.device)
            B = 4096
            for a in range(0, len(texts), B):
                s_ = emb[a:a + B] @ emb[:a + B].T
                rows = torch.arange(a, min(a + B, len(texts)), device=emb.device)
                s_[torch.arange(len(rows)), rows] = -1          # self
                s_ = torch.where(torch.arange(a + B, device=emb.device)[: s_.shape[1]][None, :] < rows[:, None], s_,
                                 torch.full_like(s_, -1))        # only earlier rows
                v, j = s_.max(dim=1)
                nn_val[a:a + B], nn_idx[a:a + B] = v, j
            ex = {}
            nv, ni = nn_val.cpu().tolist(), nn_idx.cpu().tolist()
            for lo, hi in BANDS:
                cand = [i for i, v in enumerate(nv) if lo <= v < hi]
                if cand:
                    i = rng.choice(cand)
                    ex[(lo, hi)] = (nv[i], texts[i], texts[ni[i]], len(cand))
            examples[lang] = ex
        print(f"  {lang}: done ({time.time() - t0:.0f}s)", flush=True)

    hdr = " ".join(f"{t:>6.2f}" for t in THRESHOLDS)
    for s in sizes:
        print(f"\n==== drop % at n={s} per language, by threshold (within-language only) ====")
        print(f"{'lang':5} {'model':5} {hdr}")
        for lang in sorted(table, key=lambda l: -table[l].get((s, THRESHOLDS[0]), 0)):
            if (s, THRESHOLDS[0]) not in table[lang]:
                continue
            vals = " ".join(f"{100 * table[lang][(s, t)]:6.1f}" for t in THRESHOLDS)
            print(f"{lang:5} {'yes' if lang in MODEL_LANGS else 'NO':5} {vals}")
        cov = [l for l in table if l in MODEL_LANGS and (s, THRESHOLDS[0]) in table[l]]
        unc = [l for l in table if l not in MODEL_LANGS and (s, THRESHOLDS[0]) in table[l]]
        for name, group in (("MEAN covered", cov), ("MEAN not-cov", unc)):
            if group:
                vals = " ".join(f"{100 * sum(table[l][(s, t)] for l in group) / len(group):6.1f}" for t in THRESHOLDS)
                print(f"{name:11} {vals}")

    print("\n==== example: prompt vs its MOST SIMILAR EARLIER prompt, per similarity band ====")
    for lang in EXAMPLE_LANGS:
        if lang not in examples:
            continue
        for (lo, hi), (v, a, b, cnt) in examples[lang].items():
            print(f"[{lang}] band {lo:.2f}-{min(hi, 1):.2f} ({cnt} rows in band)  cos={v:.3f}")
            print(f"    A: {a[:220]!r}")
            print(f"    B: {b[:220]!r}")


if __name__ == "__main__":
    main()
