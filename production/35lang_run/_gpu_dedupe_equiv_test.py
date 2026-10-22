"""Equivalence check: near_dup GPU path vs the CPU reference, on a real
language sample from loc_full.kept.jsonl. Run inside the vLLM SIF on 1 GPU.

usage: python _gpu_dedupe_equiv_test.py <kept.jsonl> <model_dir> <lang> <n>
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from synthgen.pipeline import near_dup


def ref_mask(emb: np.ndarray, thr: float) -> list[bool]:
    """Verbatim keep/drop loop of near_dup.embedding_dedupe_mask."""
    keep = [True] * len(emb)
    kept: list[int] = []
    for i in range(len(emb)):
        if kept and float((emb[kept] @ emb[i]).max()) >= thr:
            keep[i] = False
            continue
        kept.append(i)
    return keep


def main() -> None:
    path, model, lang, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    thr = near_dup.DEFAULT_THRESHOLD
    texts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("lang") == lang:
                texts.append(r.get("prompt", "") or "")
                if len(texts) >= n:
                    break
    print(f"sample lang={lang} n={len(texts)} threshold={thr}", flush=True)

    t = time.time()
    e_cpu = SentenceTransformer(model, device="cpu").encode(
        texts, normalize_embeddings=True, show_progress_bar=False)
    tc = time.time() - t
    t = time.time()
    e_gpu = SentenceTransformer(model, device="cuda").encode(
        texts, batch_size=256, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    torch.cuda.synchronize()
    tg = time.time() - t
    diff = float(np.abs(e_cpu - e_gpu.float().cpu().numpy()).max())
    print(f"encode: cpu={tc:.1f}s gpu={tg:.1f}s speedup={tc / tg:.0f}x max|emb diff|={diff:.2e}", flush=True)

    t = time.time(); m_ref = ref_mask(e_cpu, thr); tr = time.time() - t
    e_cpu_t = torch.from_numpy(e_cpu).cuda()
    t = time.time(); m_same = near_dup.greedy_keep_mask_blocked(e_cpu_t, thr); tb = time.time() - t
    m_b7 = near_dup.greedy_keep_mask_blocked(e_cpu_t, thr, block=7)
    m_e2e = near_dup.greedy_keep_mask_blocked(e_gpu.float(), thr)

    mism = lambda a, b: sum(x != y for x, y in zip(a, b))
    print(f"scan: cpu reference loop={tr:.1f}s  gpu blocked={tb:.2f}s", flush=True)
    print(f"dropped: reference={m_ref.count(False)} gpu_same_emb={m_same.count(False)} "
          f"gpu_end_to_end={m_e2e.count(False)}")
    print(f"MISMATCH algorithm, same embeddings, block=1024: {mism(m_ref, m_same)}")
    print(f"MISMATCH algorithm, same embeddings, block=7:    {mism(m_ref, m_b7)}")
    print(f"MISMATCH end-to-end (gpu encode vs cpu encode):  {mism(m_ref, m_e2e)}")

    rows = [{"lang": lang, "prompt": x} for x in texts]
    kept, rep = near_dup.embedding_dedupe_gpu(rows, model_name=model)
    print(f"embedding_dedupe_gpu: kept={len(kept)} report={rep}")
    print("RESULT:", "PASS" if mism(m_ref, m_same) == 0 and mism(m_ref, m_b7) == 0 else "FAIL")


if __name__ == "__main__":
    main()
