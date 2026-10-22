"""Dry run of the embedding near-dup pass: per-language drop rate on a sample.

Same rule as production (near_dup: within ONE language only, cosine >= 0.85 to
an already-kept earlier prompt -> drop, first occurrence wins). Writes nothing;
prints a per-language table plus a few dropped/kept example pairs so the drops
can be eyeballed. Run inside the vLLM SIF on 1 GPU.

usage: python _dedupe_dryrun.py <kept.jsonl> <model_dir> <n_per_lang>
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer

from synthgen.pipeline import near_dup

# Languages listed on the paraphrase-multilingual-MiniLM-L12-v2 model card
# ("no" counted via "nb").
MODEL_LANGS = set("ar bg ca cs da de el en es et fa fi fr gl gu he hi hr hu hy id it ja ka ko ku "
                  "lt lv mk mn mr ms my nb nl pl pt ro ru sk sl sq sr sv th tr uk ur vi".split()) | {"no"}


def main() -> None:
    path, model_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    thr = near_dup.DEFAULT_THRESHOLD
    t0 = time.time()
    by_lang: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lst = by_lang[r.get("lang", "")]
            if len(lst) < n:
                lst.append(r.get("prompt", "") or "")
    print(f"sampled {len(by_lang)} languages, up to {n} prompts each, in {time.time() - t0:.0f}s", flush=True)

    model = SentenceTransformer(model_dir, device="cuda")
    rows_out = []
    examples: dict[str, list[tuple[float, str, str]]] = {}
    for lang in sorted(by_lang):
        texts = by_lang[lang]
        emb = model.encode(texts, batch_size=256, normalize_embeddings=True,
                           convert_to_tensor=True, show_progress_bar=False).float()
        mask = near_dup.greedy_keep_mask_blocked(emb, thr)
        dropped = mask.count(False)
        kept_idx = torch.as_tensor([i for i, m in enumerate(mask) if m], device=emb.device)
        # mean nearest-neighbour cosine among kept rows (sample of 2000) -> how "bunched" the space is
        sub = kept_idx[:2000]
        s = emb[sub] @ emb[sub].T
        s.fill_diagonal_(-1)
        mean_nn = float(s.amax(dim=1).mean())
        ex = []
        for i in [i for i, m in enumerate(mask) if not m][:2]:
            sims = emb[kept_idx[kept_idx < i]] @ emb[i]
            j = int(kept_idx[kept_idx < i][int(sims.argmax())])
            ex.append((float(sims.max()), texts[i], texts[j]))
        examples[lang] = ex
        rows_out.append((lang, len(texts), dropped, mean_nn))
        print(f"  {lang}: n={len(texts)} dropped={dropped} ({100 * dropped / len(texts):.1f}%)", flush=True)

    print("\n==== per-language drop rate (within-language only, cosine >= %.2f) ====" % thr)
    print(f"{'lang':5} {'in_model':8} {'n':>7} {'dropped':>8} {'drop%':>6} {'mean_nn_cos_kept':>17}")
    for lang, nn_, d, mnn in sorted(rows_out, key=lambda x: -x[2] / max(x[1], 1)):
        print(f"{lang:5} {'yes' if lang in MODEL_LANGS else 'NO':8} {nn_:7d} {d:8d} {100 * d / nn_:6.1f} {mnn:17.3f}")
    tot_n = sum(r[1] for r in rows_out); tot_d = sum(r[2] for r in rows_out)
    print(f"TOTAL {'':8} {tot_n:7d} {tot_d:8d} {100 * tot_d / tot_n:6.1f}")

    print("\n==== example dropped prompt vs the earlier kept prompt it matched ====")
    for lang in sorted(examples):
        for sim, a, b in examples[lang]:
            print(f"[{lang}] cos={sim:.3f}\n   DROPPED: {a[:160]!r}\n   KEPT:    {b[:160]!r}")


if __name__ == "__main__":
    main()
