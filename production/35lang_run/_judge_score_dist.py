"""Score distribution over the current judged_judge.jsonl checkpoint.
Debug-partition job, not login node (full JSON parse of ~4M lines)."""
from __future__ import annotations

import json
from collections import Counter

PATH = "/scratch/project_465002530/abhasjha/oellm_production_35lang/outputs/judged_judge.jsonl"

total = parsed = 0
dist = Counter()
by_lang_scores: dict[str, list[int]] = {}

with open(PATH) as f:
    for line in f:
        total += 1
        r = json.loads(line)
        s = r.get("score")
        if s is None:
            continue
        parsed += 1
        dist[s] += 1
        rid = r.get("id", "")
        lang = rid.split("-")[0] if rid else "?"
        by_lang_scores.setdefault(lang, [0, 0])  # [count>=7, count total]
        by_lang_scores[lang][1] += 1
        if s >= 7:
            by_lang_scores[lang][0] += 1

print(f"[overall] total_rows={total} parsed_scores={parsed} unparseable={total - parsed}", flush=True)
print(f"[overall] average_score={sum(k * v for k, v in dist.items()) / parsed:.3f}", flush=True)
print(f"[overall] pass_rate_score_ge_7={sum(v for k, v in dist.items() if k >= 7) / parsed:.4f}", flush=True)
print("", flush=True)
print("[distribution] score : count : pct", flush=True)
for k in sorted(dist):
    print(f"[distribution] {k:>2} : {dist[k]:>9} : {dist[k]/parsed*100:5.2f}%", flush=True)
print("", flush=True)
TARGET = 100_000
print("[by_lang] lang : pass_rate(score>=7) : n_judged : passing_so_far : pct_of_100k_target", flush=True)
for lang, (ge7, n) in sorted(by_lang_scores.items(), key=lambda x: -x[1][1]):
    print(f"[by_lang] {lang:>3} : {ge7/n:6.3f} : {n:>7} : {ge7:>7} : {ge7/TARGET*100:6.2f}%", flush=True)
