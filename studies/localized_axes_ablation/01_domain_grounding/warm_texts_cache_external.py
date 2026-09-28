#!/usr/bin/env python3
"""Login-node-only sampler for datasets whose schema `ground_domains.py`'s
`_sample_texts_by_lang` doesn't natively parse (OpenAssistant/oasst2's flat
message-tree rows, CohereForAI/aya_dataset's inputs/targets rows) -- writes
the SAME per-language JSON-list cache format `_sample_texts_by_lang` does
(`{lang}.json` = flat list of raw text strings), so `classify_external.py`
can classify from it exactly like a native-schema dataset's cache.

Needs internet (HF Hub) -- run from the login node, same as
warm_texts_cache.py.

Usage:
    python3 warm_texts_cache_external.py --dataset oasst2 --n-per-lang 400 \
        --cache-dir dataset_compare/oasst2/texts_cache
    python3 warm_texts_cache_external.py --dataset aya --n-per-lang 400 \
        --cache-dir dataset_compare/aya/texts_cache
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("warm_texts_cache_external")

LANGS = ["es", "fr", "de", "pl", "uk"]

# aya_dataset's `language` field uses full English names, same convention as
# WildChat/lmsys-chat-1m's `language` field.
AYA_LANG_NAMES = {"es": "Spanish", "fr": "French", "de": "German",
                  "pl": "Polish", "uk": "Ukrainian"}


def sample_oasst2(n_per_lang: int, seed: int) -> dict[str, list[str]]:
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
    buffers: dict[str, list[str]] = {l: [] for l in LANGS}
    for ex in ds:
        # root-level real user requests only (prompter role, no parent) --
        # a reply-to-assistant "prompter" turn is a follow-up, not a fresh
        # instruction, and would skew topic classification toward whatever
        # the prior turn was about.
        if ex.get("role") != "prompter" or ex.get("parent_id") not in (None, "None"):
            continue
        lang = ex.get("lang")
        if lang in buffers and len(buffers[lang]) < n_per_lang * 3:
            text = ex.get("text")
            if text and len(text) > 20:
                buffers[lang].append(text)
    rng = random.Random(seed)
    out = {}
    for l in LANGS:
        texts = buffers[l]
        rng.shuffle(texts)
        out[l] = texts[:n_per_lang]
        if len(out[l]) < n_per_lang:
            log.warning("oasst2: only found %d/%d for lang=%s (dataset is thin/absent here)",
                       len(out[l]), n_per_lang, l)
    return out


def sample_aya(n_per_lang: int, seed: int) -> dict[str, list[str]]:
    from datasets import load_dataset
    ds = load_dataset("CohereForAI/aya_dataset", split="train", streaming=True)
    wanted = {v: k for k, v in AYA_LANG_NAMES.items()}
    buffers: dict[str, list[str]] = {l: [] for l in LANGS}
    for ex in ds:
        code = wanted.get(ex.get("language"))
        if code and len(buffers[code]) < n_per_lang * 3:
            text = ex.get("inputs")
            if text and len(text) > 20:
                buffers[code].append(text)
    rng = random.Random(seed)
    out = {}
    for l in LANGS:
        texts = buffers[l]
        rng.shuffle(texts)
        out[l] = texts[:n_per_lang]
        if len(out[l]) < n_per_lang:
            log.warning("aya_dataset: only found %d/%d for lang=%s (dataset is thin here)",
                       len(out[l]), n_per_lang, l)
    return out


SAMPLERS = {"oasst2": sample_oasst2, "aya": sample_aya}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SAMPLERS))
    ap.add_argument("--n-per-lang", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = SAMPLERS[args.dataset](args.n_per_lang, args.seed)
    for code, texts in out.items():
        (cache_dir / f"{code}.json").write_text(json.dumps(texts, ensure_ascii=False))
        print(f"{code}: {len(texts)} texts cached")


if __name__ == "__main__":
    main()
