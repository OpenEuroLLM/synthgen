"""Verify embedding_dedupe's new max_workers path: (1) parallel result is
identical to the serial result (not just "doesn't crash"), (2) it's actually
faster at a meaningful scale. Debug-partition job, not login node."""
from __future__ import annotations

import random
import sys
import time

sys.path.insert(0, "/scratch/project_462001516/abhasjha/synthgen")
from synthgen.pipeline.near_dup import embedding_dedupe  # noqa: E402

random.seed(0)
LANGS = ["es", "fr", "de", "it", "pt", "pl", "nl", "cs"]
WORDS = [f"w{i}" for i in range(4000)]  # large vocab -> realistic low near-dup rate


def rand_text(n_words=14):
    return " ".join(random.choice(WORDS) for _ in range(n_words))


N_PER_LANG = 6000
rows = []
rid = 0
for lang in LANGS:
    for _ in range(N_PER_LANG):
        rows.append({"id": f"{lang}-{rid}", "lang": lang, "prompt": rand_text()})
        rid += 1

print(f"[setup] {len(rows)} rows across {len(LANGS)} languages, {N_PER_LANG}/lang", flush=True)

t0 = time.time()
kept_serial, report_serial = embedding_dedupe(rows, max_workers=1)
t1 = time.time()
print(f"[serial] time={t1-t0:.1f}s kept={len(kept_serial)} dropped={report_serial['total_dropped']}", flush=True)

t0 = time.time()
kept_parallel, report_parallel = embedding_dedupe(rows, max_workers=8)
t1 = time.time()
print(f"[parallel] time={t1-t0:.1f}s kept={len(kept_parallel)} dropped={report_parallel['total_dropped']}", flush=True)

ids_serial = sorted(r["id"] for r in kept_serial)
ids_parallel = sorted(r["id"] for r in kept_parallel)
match = ids_serial == ids_parallel
print(f"[check] serial and parallel kept the SAME rows: {match}", flush=True)
if not match:
    only_serial = set(ids_serial) - set(ids_parallel)
    only_parallel = set(ids_parallel) - set(ids_serial)
    print(f"[check] only_in_serial={len(only_serial)} only_in_parallel={len(only_parallel)}", flush=True)

print(f"[check] dropped_by_lang match: {report_serial['dropped_by_lang'] == report_parallel['dropped_by_lang']}", flush=True)
