"""One-off sizing check for the production run's dedupe stage -- run on a
debug-partition compute node, NOT the login node. Does three things:
  1. per-language row counts in loc_gen.jsonl (to project kept-count/lang)
  2. judge score>=7 pass rate over judged_judge.jsonl so far
  3. a small-n timing benchmark of the real embedding_dedupe_mask() to fit
     its O(n^2) growth and extrapolate to the real per-language kept count
Outputs everything to stdout, captured by the SLURM job log.
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter

sys.path.insert(0, "/scratch/project_462001516/abhasjha/synthgen")
from synthgen.pipeline.near_dup import embedding_dedupe_mask  # noqa: E402

GEN_PATH = "/scratch/project_465002530/abhasjha/oellm_production_35lang/outputs/loc_gen.jsonl"
JUDGE_PATH = "/scratch/project_465002530/abhasjha/oellm_production_35lang/outputs/judged_judge.jsonl"


def lang_counts():
    c = Counter()
    n = 0
    with open(GEN_PATH) as f:
        for line in f:
            n += 1
            r = json.loads(line)
            rid = r.get("id", "")
            lang = rid.split("-")[0] if rid else "?"
            c[lang] += 1
    print(f"[lang_counts] total rows={n}")
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print(f"[lang_counts] {k} {v}")
    return c


def score_pass_rate():
    n = parsed = ge7 = 0
    with open(JUDGE_PATH) as f:
        for line in f:
            n += 1
            r = json.loads(line)
            s = r.get("score")
            if s is None:
                continue
            parsed += 1
            if s >= 7:
                ge7 += 1
    rate = ge7 / parsed if parsed else 0.0
    print(f"[score] total={n} parsed={parsed} score>=7={ge7} rate={rate:.4f}")
    return rate


def dedup_timing():
    random.seed(0)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
             "theta", "iota", "kappa"]

    def rand_text(n_words=12):
        return " ".join(random.choice(words) for _ in range(n_words))

    sizes = [2000, 5000, 10000, 20000, 40000]
    results = []
    for n in sizes:
        texts = [rand_text() for _ in range(n)]
        t0 = time.time()
        mask = embedding_dedupe_mask(texts)
        t1 = time.time()
        dropped = sum(1 for m in mask if not m)
        dt = t1 - t0
        results.append((n, dt))
        print(f"[dedup_timing] n={n} time={dt:.2f}s dropped={dropped}")
    return results


if __name__ == "__main__":
    c = lang_counts()
    rate = score_pass_rate()
    results = dedup_timing()

    # fit c2 in t = c2 * n^2 (ignore the O(n) embedding-encode term, small
    # relative to the O(n^2) similarity-scan term at these sizes) using the
    # largest sample for the best-conditioned estimate
    n_last, t_last = results[-1]
    c2 = t_last / (n_last ** 2)
    print(f"\n[fit] c2={c2:.3e} (t ~= c2 * n^2)")

    total_gen = sum(c.values())
    n_langs = len(c)
    avg_per_lang = total_gen / n_langs if n_langs else 0
    projected_kept_avg = avg_per_lang * rate
    max_per_lang = max(c.values()) if c else 0
    projected_kept_max = max_per_lang * rate

    for label, n_proj in [("avg lang", projected_kept_avg), ("largest lang", projected_kept_max)]:
        est_s = c2 * (n_proj ** 2)
        print(f"[projection] {label}: projected kept-rows~{n_proj:.0f} "
              f"-> estimated dedup time ~{est_s:.1f}s ({est_s/60:.1f} min) for that bucket alone")
    total_est = sum(c2 * ((v * rate) ** 2) for v in c.values())
    print(f"[projection] sum across all {n_langs} language buckets (sequential): "
          f"~{total_est:.1f}s (~{total_est/60:.1f} min, ~{total_est/3600:.2f} h)")
