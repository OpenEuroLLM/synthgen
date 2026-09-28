"""Classifies pre-cached texts (from warm_texts_cache_external.py, for
datasets whose schema ground_domains.py can't stream natively) into the
same Pool A taxonomy, using the same classifier machinery as
ground_domains.py -- reused via import, not reimplemented, so results are
directly comparable to WildChat's/lmsys's domain_distribution/*.json.

Runs multiple datasets in ONE process against the same vLLM server, to
avoid paying separate SLURM queue/startup costs per dataset.

  python3 classify_external.py --vllm-endpoints $EP/gen --vllm-model gen \
      --job oasst2:dataset_compare/oasst2 --job aya:dataset_compare/aya
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ground_domains import _classify_all, _smoothed_distribution  # noqa: E402
from synthgen.backends import VLLMBackend  # noqa: E402

LANGS = ["es", "fr", "de", "pl", "uk"]


def run_one(name: str, base_dir: Path, backend, concurrency: int, floor: float) -> None:
    texts_cache = base_dir / "texts_cache"
    exemplar_dir = base_dir / "domain_exemplars"
    dist_dir = base_dir / "domain_distribution"
    exemplar_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    agg_counts: Counter[str] = Counter()
    for code in LANGS:
        cache_file = texts_cache / f"{code}.json"
        if not cache_file.exists():
            print(f"[{name}] skip {code}: no cache file")
            continue
        texts = json.loads(cache_file.read_text())
        if not texts:
            print(f"[{name}] skip {code}: 0 texts cached")
            continue
        print(f"[{name}] classifying {len(texts)} prompts for lang={code}...")
        cats = asyncio.run(_classify_all(backend, texts, concurrency, is_vllm=True))

        by_domain: dict[str, list[str]] = defaultdict(list)
        for text, cat in zip(texts, cats):
            if cat == "none_of_these":
                continue
            by_domain[cat].append(text)

        exemplar_rows = [{"lang": code, "domain": d, "text": t}
                         for d, texts_for_d in by_domain.items() for t in texts_for_d]
        with (exemplar_dir / f"{code}.jsonl").open("w", encoding="utf-8") as f:
            for r in exemplar_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        counts = Counter({d: len(v) for d, v in by_domain.items()})
        agg_counts.update(counts)
        dist = _smoothed_distribution(counts, floor=floor)
        (dist_dir / f"{code}.json").write_text(json.dumps({
            "lang": code, "source": name, "n_classified": len(texts),
            "raw_counts": dict(counts), "smoothed_distribution": dist,
        }, indent=2, ensure_ascii=False))
        print(f"[{name}] wrote distribution for lang={code}")

    agg_dist = _smoothed_distribution(agg_counts, floor=floor)
    (dist_dir / "_aggregate.json").write_text(json.dumps({
        "source": name, "langs": LANGS,
        "raw_counts": dict(agg_counts), "smoothed_distribution": agg_dist,
    }, indent=2, ensure_ascii=False))
    print(f"[{name}] wrote aggregate distribution")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", required=True,
                    help="name:base_dir -- base_dir must contain texts_cache/, "
                         "output written to base_dir/domain_distribution and "
                         "base_dir/domain_exemplars. Repeatable.")
    ap.add_argument("--vllm-endpoints", required=True)
    ap.add_argument("--vllm-model", default="gen")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--floor", type=float, default=0.02)
    args = ap.parse_args()

    backend = VLLMBackend(model=args.vllm_model, endpoints_dir=args.vllm_endpoints)
    for job in args.job:
        name, base_dir = job.split(":", 1)
        run_one(name, Path(base_dir), backend, args.concurrency, args.floor)


if __name__ == "__main__":
    main()
