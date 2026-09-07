"""One-off re-scoring pass: compute the embedding-based near-dup metric that
never ran during the original full-scale batch (sentence-transformers wasn't
installed in the vLLM container, so every run silently fell back to
lexical-only diversity — see `embedding_unavailable` in every summary.json).

No regeneration, no GPU: reads each ablation's existing `records.jsonl` and
re-runs `diversity_metrics.diversity_report` with embeddings now available
(via a plain login-node venv with sentence-transformers installed, model
cached from `_common/.diversity_venv`). Writes one `embedding_diversity.json`
per arm, and prints a compact summary table.

Run: `.diversity_venv/bin/python3 rescore_embeddings.py`
"""
from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from diversity_metrics import diversity_report  # noqa: E402

ROOT = Path(__file__).parent.parent
LATIN = {"es", "fr", "de", "pl"}
NON_LATIN = {"uk"}


def load(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def report_for(rows: list[dict], text_key: str = "instruction",
                group_keys: tuple = ("lang",)) -> dict:
    rows = [r for r in rows if r.get(text_key)]  # drop gen_err/null-instruction rows
    # diversity_metrics.DEFAULT_EMBED_MODEL is now the multilingual model
    # (paraphrase-multilingual-MiniLM-L12-v2) -- see that module for why.
    return diversity_report(rows, text_key=text_key, group_keys=group_keys,
                             include_embedding=True, embedding_threshold=0.85)


def pooled_rates(rep: dict) -> tuple[float, float, int]:
    lex = rep.get("pooled_lexical", {})
    emb = rep.get("pooled_embedding", {})
    return lex.get("near_dup_rate", 0.0), emb.get("near_dup_rate", float("nan")), lex.get("n", 0)


def run():
    results: dict[str, dict] = {}

    jobs: list[tuple[str, Path, str | None]] = [
        ("04 baseline", ROOT / "04_diversity_baseline/outputs/records.jsonl", None),
        ("01 grounding: none", ROOT / "01_domain_grounding/outputs/mode_none/records.jsonl", None),
        ("01 grounding: light", ROOT / "01_domain_grounding/outputs/mode_light/records.jsonl", None),
        ("01 grounding: deep", ROOT / "01_domain_grounding/outputs/mode_deep/records.jsonl", None),
        ("02 persona: 0.5", ROOT / "02_persona_rate/outputs/persona_p_0.5/records.jsonl", None),
        ("02 persona: 0.15", ROOT / "02_persona_rate/outputs/persona_p_0.15/records.jsonl", None),
        ("02 persona: 0.10", ROOT / "02_persona_rate/outputs/persona_p_0.1/records.jsonl", None),
        ("02 persona: 0.0", ROOT / "02_persona_rate/outputs/persona_p_0.0/records.jsonl", None),
        ("05 diversify (multi-country)", ROOT / "05_country_grounding/outputs/diversify/records.jsonl", None),
        ("10 specificity: generic", ROOT / "10_persona_specificity/outputs/generic/records.jsonl", None),
        ("10 specificity: concrete", ROOT / "10_persona_specificity/outputs/concrete/records.jsonl", None),
    ]

    for label, path, _ in jobs:
        if not path.exists():
            print(f"[skip] {label}: {path} not found")
            continue
        rows = load(path)
        rep = report_for(rows)
        lex, emb, n = pooled_rates(rep)
        results[label] = rep
        print(f"{label:35s} n={n:4d}  lexical={lex:.4f}  embedding={emb:.4f}")

    # 03 intent necessity: paired arms nested inside each row
    p = ROOT / "03_intent_necessity/outputs/records.jsonl"
    if p.exists():
        rows = load(p)
        with_rows = [{"lang": r["lang"], "instruction": r["with_intent"]["instruction"]} for r in rows]
        without_rows = [{"lang": r["lang"], "instruction": r["without_intent"]["instruction"]} for r in rows]
        for label, sub in [("03 intent: injected", with_rows), ("03 intent: stripped", without_rows)]:
            rep = report_for(sub)
            lex, emb, n = pooled_rates(rep)
            results[label] = rep
            print(f"{label:35s} n={n:4d}  lexical={lex:.4f}  embedding={emb:.4f}")

    # 09 salt necessity: same paired-nested shape
    p = ROOT / "09_salt_necessity/outputs/records.jsonl"
    if p.exists():
        rows = load(p)
        with_rows = [{"lang": r["lang"], "instruction": r["with_salt"]["instruction"]} for r in rows]
        without_rows = [{"lang": r["lang"], "instruction": r["without_salt"]["instruction"]} for r in rows]
        for label, sub in [("09 salt: present", with_rows), ("09 salt: removed", without_rows)]:
            rep = report_for(sub)
            lex, emb, n = pooled_rates(rep)
            results[label] = rep
            print(f"{label:35s} n={n:4d}  lexical={lex:.4f}  embedding={emb:.4f}")

    # 08 script effects: reuse 04's records, regroup by script instead of lang
    p = ROOT / "04_diversity_baseline/outputs/records.jsonl"
    if p.exists():
        rows = load(p)
        latin_rows = [r for r in rows if r["lang"] in LATIN]
        nonlatin_rows = [r for r in rows if r["lang"] in NON_LATIN]
        for label, sub in [("08 script: latin", latin_rows), ("08 script: non_latin", nonlatin_rows)]:
            rep = report_for(sub, group_keys=("lang",))
            lex, emb, n = pooled_rates(rep)
            results[label] = rep
            print(f"{label:35s} n={n:4d}  lexical={lex:.4f}  embedding={emb:.4f}")

    out_path = ROOT / "_common/embedding_rescore_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
