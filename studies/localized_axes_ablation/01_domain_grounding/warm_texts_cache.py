#!/usr/bin/env python3
"""Standalone login-node-only helper: stream+sample raw real-data texts and
populate ground_domains.py's --texts-cache, WITHOUT doing any classification
(so it needs neither OPENROUTER_API_KEY nor a running vLLM server) — just HF
Hub access, which only the login node has.

Run this once from the login node, then ground_domains.py --texts-cache <dir>
--classifier-backend vllm can classify from that cache inside a SLURM job
with no internet.

Usage:
    python3 warm_texts_cache.py --n-per-lang 400 --langs es fr de pl uk \
        --cache-dir studies/localized_axes_ablation/01_domain_grounding/texts_cache
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).parent))
from ground_domains import _sample_texts_by_lang, LANGUAGES_PHASE3  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="allenai/WildChat-1M")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-per-lang", type=int, default=400)
    ap.add_argument("--langs", nargs="*", default=list(LANGUAGES_PHASE3))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = _sample_texts_by_lang(args.dataset, args.split, args.n_per_lang,
                                args.langs, args.seed, cache_dir=cache_dir)
    for code, texts in out.items():
        print(f"{code}: {len(texts)} texts cached")


if __name__ == "__main__":
    main()
