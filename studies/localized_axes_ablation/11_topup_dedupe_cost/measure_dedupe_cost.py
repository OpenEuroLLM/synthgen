"""Topup embedding-dedupe cost ablation: what does turning on
`embedding_dedupe=True` in `synthgen.localized.topup.run()` actually cost in
extra generate/judge calls, compared to running the exact same target with it
off?

Context: `09_salt_necessity` found no measurable diversity benefit from the
`salt` prompt nudge, which was then removed from `GENERATION_PROMPT` entirely
(see synthgen/localized/prompts.py). The real backstop against paraphrased
near-duplicates is now `synthgen/pipeline/near_dup.py`'s embedding filter,
wired into `topup.run()` as an opt-in (`embedding_dedupe=True`): a dropped
near-duplicate shrinks that round's survivor count, which the topup loop's
own deficit calculation turns into a real replacement generation. That makes
the diversity guarantee structural rather than a text hint -- but every drop
costs one extra generate+judge call pair, so the decision to enable it in
production needs a real number, not a guess. This ablation produces that
number, at pilot scale, before anyone flips it on for a full run.

Design: run `topup.run()` to the SAME `--target` twice, same seed, same
gen/judge backends and endpoints -- once with `embedding_dedupe=False`
(arm "off", matches current production behavior) and once with
`embedding_dedupe=True` (arm "on") -- into two isolated `SynthConfig` roots
so they don't share any on-disk state. Compares:

  - total_raw_generated   (raw successful generate() calls, both arms)
  - rounds                (how many topup rounds it took to hit target)
  - embedding_near_dup_dropped_total (arm "on" only -- the actual collapse
                                       rate this run found, at this scale)
  - cost delta: (on.total_raw_generated - off.total_raw_generated) /
                off.total_raw_generated -- the number that answers "does
                turning this on for real production actually cost anything."

Run inside the vLLM container from the repo root, e.g.:
  python studies/localized_axes_ablation/11_topup_dedupe_cost/measure_dedupe_cost.py \
      --gen-endpoints endpoints/gen --judge-endpoints endpoints/judge \
      --target 30 --langs es fr de pl uk
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from synthgen.backends import VLLMBackend  # noqa: E402
from synthgen.config import LANGUAGES_PHASE3, Paths, SynthConfig  # noqa: E402
from synthgen.localized import topup  # noqa: E402

# Pilot threshold, not decide-thresholds output: REPORT.md's score
# distributions across this whole study cluster tightly at 9-10 (e.g. 04's
# baseline: 123/148 at 10, only 4 at 5) -- a permissive global cut keeps this
# ablation about topup-loop mechanics, not re-litigating threshold placement,
# which decide-thresholds is documented (CLAUDE.md) as unreliable at small n
# anyway.
DEFAULT_THRESHOLDS = {"global": 7}


def run_arm(*, name: str, out_root: Path, gen_backend, judge_backend,
           target: int, langs: list[str], seed: int, thresholds: dict,
           embedding_dedupe: bool, max_rounds: int) -> dict:
    if out_root.exists():
        shutil.rmtree(out_root)
    cfg = SynthConfig()
    cfg.paths = Paths.from_root(out_root)

    result = topup.run(
        cfg, gen_backends=[gen_backend], judge_backend=judge_backend,
        target_per_lang=target, thresholds=thresholds, lang_codes=langs,
        seed=seed, max_rounds=max_rounds,
        embedding_dedupe=embedding_dedupe,
    )
    result["arm"] = name
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-endpoints", required=True)
    ap.add_argument("--judge-endpoints", required=True)
    ap.add_argument("--gen-model", default="gen")
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--target", type=int, default=30,
                    help="survivors per language for BOTH arms (pilot scale, "
                         "not production -- see README for why)")
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    args = ap.parse_args()

    langs = args.langs or list(LANGUAGES_PHASE3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gen = VLLMBackend(model=args.gen_model, endpoints_dir=args.gen_endpoints)
    judge = VLLMBackend(model=args.judge_model, endpoints_dir=args.judge_endpoints)

    results = {}
    for name, embedding_dedupe in (("off", False), ("on", True)):
        print(f"\n=== arm: embedding_dedupe={embedding_dedupe} ===")
        r = run_arm(name=name, out_root=out / f"run_{name}", gen_backend=gen,
                    judge_backend=judge, target=args.target, langs=langs,
                    seed=args.seed, thresholds=DEFAULT_THRESHOLDS,
                    embedding_dedupe=embedding_dedupe, max_rounds=args.max_rounds)
        results[name] = r
        print(json.dumps(r, indent=2, ensure_ascii=False))

    off_raw = results["off"]["total_raw_generated"]
    on_raw = results["on"]["total_raw_generated"]
    dropped = results["on"]["embedding_near_dup_dropped_total"]
    cost_delta_pct = ((on_raw - off_raw) / off_raw * 100) if off_raw else float("nan")

    summary = {
        "target_per_lang": args.target,
        "langs": langs,
        "off": results["off"],
        "on": results["on"],
        "embedding_near_dup_dropped_total": dropped,
        "extra_raw_generations": on_raw - off_raw,
        "cost_delta_pct": cost_delta_pct,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n--- cost delta ---")
    print(f"off: total_raw_generated={off_raw}  rounds={results['off']['rounds']}")
    print(f"on:  total_raw_generated={on_raw}  rounds={results['on']['rounds']}  "
         f"embedding_near_dup_dropped={dropped}")
    print(f"extra raw generations from enabling embedding_dedupe: "
         f"{on_raw - off_raw} ({cost_delta_pct:.2f}%)")
    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
