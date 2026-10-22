"""Standalone embedding near-dup dedupe driver.

Calls `synthgen.pipeline.dedupe.run(embedding_dedupe=True, ...)` directly, so it
only needs numpy + torch + sentence-transformers (the dedup-venv) plus
PYTHONPATH=$REPO -- it deliberately bypasses `synthgen.cli`, whose top-level
imports (`synthgen.config`, backends) pull in heavier deps the lean venv does
not carry.

Why this exists: the production run (run_production_mn5.slurm) ran dedupe with
embedding_dedupe OFF (exact-string only, ~1.6% removed). This re-runs the
*embedding* near-dup pass on loc_full.kept.jsonl to produce the semantically
deduped dataset that was actually required. See near_dup.py for the algorithm.

Hard-fails if the embedder is unavailable: dedupe.run() otherwise falls back to
exact-match-only *silently* (logs a warning, writes the file anyway), which is
exactly the trap that produced the first, wrong result.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

from synthgen.pipeline import dedupe


def _load_overrides(path: str | None) -> dict | None:
    """Read + validate the per-language override file; fail before any work."""
    if not path:
        return None
    ov = json.loads(Path(path).read_text())
    for lang, cfg in ov.items():
        if not isinstance(cfg.get("threshold"), (int, float)):
            raise SystemExit(f"FATAL: --lang-overrides: {lang} has no numeric threshold ({cfg})")
        if "model" in cfg and not Path(cfg["model"], "modules.json").is_file():
            raise SystemExit(f"FATAL: --lang-overrides: {lang} model folder missing: {cfg['model']}")
    return ov


def main() -> None:
    ap = argparse.ArgumentParser(description="Embedding near-dup dedupe (forces embedding_dedupe=True)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--embedding-threshold", type=float, default=None)
    ap.add_argument("--embedding-model", default=None)
    ap.add_argument("--embedding-workers", type=int, default=None,
                    help="CPU: process-pool size; GPU: number of GPUs to use")
    ap.add_argument("--embedding-device", default=None,
                    help="'cuda' -> near_dup.embedding_dedupe_gpu (same rule, on GPUs); default CPU")
    ap.add_argument("--lang-overrides", default=None,
                    help='JSON file: {"mt": {"model": "/path/labse", "threshold": 0.9}, ...} (GPU only)')
    a = ap.parse_args()

    # Spawn (not fork) the embedding worker pool: dedupe.run() holds all ~5.6M
    # rows in the parent, and forked workers gradually copy that address space
    # (refcount writes defeat copy-on-write) -- 20 of them OOM-killed job
    # 47780739 at 125 GB. Spawned workers only receive their own language's texts.
    multiprocessing.set_start_method("spawn", force=True)

    summary = dedupe.run(
        input=Path(a.input),
        output=Path(a.output),
        report=Path(a.report) if a.report else None,
        embedding_dedupe=True,
        embedding_threshold=a.embedding_threshold,
        embedding_model=a.embedding_model,
        embedding_workers=a.embedding_workers,
        embedding_device=a.embedding_device,
        embedding_lang_overrides=_load_overrides(a.lang_overrides),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    if summary.get("embedding_dedupe_skipped"):
        raise SystemExit(
            "FATAL: embedding dedupe was SKIPPED (embedder unavailable) -- "
            "output is exact-match only, NOT what was requested: "
            + str(summary["embedding_dedupe_skipped"])
        )
    if "dropped_near_dup_embedding" not in summary:
        raise SystemExit("FATAL: embedding pass did not report any result; aborting.")


if __name__ == "__main__":
    main()
