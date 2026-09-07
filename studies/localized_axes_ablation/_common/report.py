"""Score-histogram / report helpers shared by every ablation.

Deliberately a generalized sibling of
`studies/localized_bootstrap/run_judge_validation.py`'s `_hist`/`print_report`
(reused pattern, not a rewrite) — that script's version is hardcoded to the
good/medium/bad labels of the judge-validation pilot; this version takes
arbitrary arm labels (e.g. grounding mode, PERSONA_P value, pool-split ratio)
since every ablation here compares N arms, not 3 fixed ones.
"""
from __future__ import annotations

from collections import Counter


def hist(scores: list[int | float]) -> str:
    """ASCII histogram, 0-10 score band (same shape as run_judge_validation.py's
    `_hist`, so reports from different ablations read the same way)."""
    c = Counter(int(round(s)) for s in scores if s is not None)
    return "\n".join(
        f"    {s:2d} | {'#' * c.get(s, 0)}{'' if c.get(s) else '.'} ({c.get(s, 0)})"
        for s in range(10, -1, -1)
    )


def score_stats(scores: list[int | float]) -> dict:
    xs = [s for s in scores if isinstance(s, (int, float))]
    if not xs:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 2), "min": min(xs), "max": max(xs)}


def print_arm_report(arm_label: str, records: list[dict], *, score_key: str = "score") -> dict:
    """Per-arm score summary + histogram, printed and returned as a dict for
    `summary.json`. Call once per arm (e.g. once per PERSONA_P value, once per
    grounding mode) — do not pool arms together, that's the whole point of an
    ablation."""
    scores = [r.get(score_key) for r in records if isinstance(r.get(score_key), (int, float))]
    stats = score_stats(scores)
    print(f"\n=== {arm_label} === n={stats['n']}  mean={stats['mean']}")
    print(hist(scores))
    return stats


def print_diversity_summary(arm_label: str, diversity: dict) -> None:
    """Prints the per-group near-dup rates from
    `_common.diversity_metrics.diversity_report()` output for one arm."""
    print(f"\n--- diversity: {arm_label} ---")
    for group, entry in diversity.get("groups", {}).items():
        lex = entry.get("lexical", {})
        emb = entry.get("embedding", {})
        print(f"    {group:40s} n={entry['n']:4d}  "
              f"lexical_near_dup={lex.get('near_dup_rate', 0):.3f}  "
              f"embedding_near_dup={emb.get('near_dup_rate', 0):.3f}")
    pl = diversity.get("pooled_lexical", {})
    pe = diversity.get("pooled_embedding", {})
    if pl or pe:
        print(f"    {'[POOLED — all groups]':40s}      "
              f"lexical_near_dup={pl.get('near_dup_rate', 0):.3f}  "
              f"embedding_near_dup={pe.get('near_dup_rate', 0):.3f}")
    if diversity.get("embedding_unavailable"):
        print(f"    [embedding metric skipped: {diversity['embedding_unavailable']}]")


def compare_arms(arm_stats: dict[str, dict]) -> None:
    """Side-by-side mean/n table across arms — the thing every ablation
    README asks for as the headline comparison. `arm_stats` is
    {arm_label: score_stats_dict}."""
    print("\n=== ARM COMPARISON ===")
    width = max(len(k) for k in arm_stats) + 2
    for label, st in arm_stats.items():
        print(f"    {label:<{width}s} n={st['n']:4d}  mean={st['mean']}")
