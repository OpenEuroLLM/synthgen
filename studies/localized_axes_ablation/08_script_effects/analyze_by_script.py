"""Script effects: do judge score and diversity behave differently for
non-Latin-script languages (el, uk per synthgen.config.LANG_SCRIPT) than
Latin-script ones? The production judge already applies a different leniency
policy for loanwords/transliteration by language, so the axes this study
tests (grounding, persona rate, intent, pool split) might interact with
script too, not just language.

Pure re-analysis, NO new generation: this study's five languages (es fr de
pl uk) already span both script groups (uk=non_latin; the rest=latin), so
04_diversity_baseline's full-scale run already has both groups in it. This
script just regroups an existing records.jsonl by script instead of by
language and reports score + diversity per group.

Run locally (no GPU/vLLM needed) once 04 (or any other ablation) has
produced a full-scale records.jsonl:
  python studies/localized_axes_ablation/08_script_effects/analyze_by_script.py \
      --records ../04_diversity_baseline/outputs/records.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STUDY_ROOT))

from synthgen.config import LANG_SCRIPT  # noqa: E402
from synthgen.io import iter_jsonl  # noqa: E402
from _common import diversity_metrics as dm  # noqa: E402
from _common import report as rpt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True,
                    help="an existing records.jsonl with lang/instruction/score, "
                         "e.g. 04_diversity_baseline/outputs/records.jsonl")
    ap.add_argument("--no-embedding", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    args = ap.parse_args()

    rows = [r for r in iter_jsonl(args.records) if r.get("instruction") and r.get("response")]
    for r in rows:
        r["script"] = LANG_SCRIPT.get(r.get("lang"), "unknown")

    by_script: dict[str, list[dict]] = {}
    for r in rows:
        by_script.setdefault(r["script"], []).append(r)

    stats_by_script = {}
    for script, group in by_script.items():
        scores = [r.get("score") for r in group if isinstance(r.get("score"), (int, float))]
        stats_by_script[script] = rpt.print_arm_report(
            f"script={script} (langs: {sorted({r['lang'] for r in group})})", group)

    rpt.compare_arms(stats_by_script)

    diversity = dm.diversity_report(rows, group_keys=("script",),
                                    include_embedding=not args.no_embedding)
    rpt.print_diversity_summary("by_script", diversity)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "records_source": args.records,
        "n_by_script": {s: len(g) for s, g in by_script.items()},
        "score_stats_by_script": stats_by_script,
        "diversity": diversity,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
