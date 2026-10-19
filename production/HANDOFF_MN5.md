# Handoff: continuing the 35-language production run on MN5

You already have `synthgen` set up on MN5. This doc is the specific
hand-off for *this in-flight production run* — current state, what to
transfer, and what to change in your MN5 launch script so it resumes
cleanly instead of restarting from scratch. For the general
LUMI(AMD/ROCm)→NVIDIA porting concerns (container image, `CUDA_VISIBLE_
DEVICES`, GPU topology, `--cpus-per-task`, etc.), see `MIGRATE_TO_NVIDIA.md`
in this same directory — that doc still applies in full; this one is just
the "what's already done, don't redo it" layer on top.

## Current state (as of 2026-10-16)

- **Gen phase: complete.** `loc_gen.jsonl` — 19,108,653,679 bytes,
  6,064,877 rows across all 35 languages. Nothing to regenerate.
- **Judge phase: in progress, resumable.** `judged_judge.jsonl` —
  1,603,932,015 bytes, **last checkpointed at 3,964,000 / 5,559,456 rows
  (71.3%)** before the most recent LUMI timeout. A judge-only resubmit
  (job 22895003) is currently queued on LUMI to keep pushing this forward
  in parallel with your MN5 setup — so if you start from a *fresh* copy of
  this data, check its judge row count first; it may already be further
  along than 71.3% by the time you pull it.
- **Quality-filter + exact-dedupe: not yet run** (they run automatically
  as the last step of each `topup` round, which hasn't completed a full
  round yet since judge wasn't done). No `kept_*.jsonl` or
  `loc_full.dedup.jsonl` exist yet.
- **Embedding near-dup dedupe: not yet run, and was just fixed.** The
  original serial implementation (`embedding_dedupe_mask`'s O(n^2) scan,
  run one language at a time) was benchmarked at **~15-54h total** across
  all 35 languages at this scale — bigger than the judge phase itself.
  Fixed this session: `synthgen/pipeline/near_dup.py`'s `embedding_dedupe()`
  now takes `max_workers` and runs language buckets concurrently via a
  process pool (buckets are fully independent, so this is a real, verified
  — not just "didn't crash," actually checked serial vs. parallel produce
  identical kept-row sets — speedup). Wired through `--embedding-workers`
  on both `synthgen dedupe` and `synthgen topup`. **This is already on
  `origin/main`** (commit `55cd05f`) — a normal `git pull` on MN5 picks it
  up, nothing manual needed.

## What to transfer from LUMI to MN5

Nothing in git needs manual transfer — `git clone`/`git pull` on MN5 gets
the current `main` (code) and `studies` (ablations, not required for this
run) branches directly from `origin`.

**Data that DOES need manual transfer** (not in git, lives on LUMI scratch):

| Path on LUMI | Size | Needed? |
|---|---|---|
| `/scratch/project_465002530/abhasjha/oellm_production_35lang/outputs/loc_gen.jsonl` | 19.1 GB | Yes — this is the completed gen checkpoint, skip re-generating 5.6M rows |
| `/scratch/project_465002530/abhasjha/oellm_production_35lang/outputs/judged_judge.jsonl` | 1.6 GB (growing) | Yes — partial judge checkpoint, resume from here instead of row 0 |
| `production/35lang_run/thresholds.json` | tiny | Already in git (`{"global": 7}`) — no transfer needed, just confirm it pulled |

Since LUMI and MN5 are different HPC centers with no direct node-to-node
path, route the transfer through wherever you normally move data between
them (your own machine via `scp`/`rsync`, or Globus if both centers expose
an endpoint — check with MN5/LUMI support if you haven't moved data between
them before). Re-check file sizes/line counts after transfer before
resuming — a truncated copy would silently look like "less progress than
real" to `topup`'s resume logic, not error out.

Once on MN5, place them at whatever `$DATA_ROOT` your MN5 launch script
uses, keeping the same `prompts/`, `outputs/`, `review/`, `logs/` layout
`synthgen.config.Paths` expects (`outputs/loc_gen.jsonl`,
`outputs/judged_<judge_model_slug>.jsonl` specifically — the judge output
filename encodes the model slug, so if MN5 serves the judge model under a
*different* `--served-model-name` than LUMI's `judge`, rename the file to
match what `generate.output_path_for("judge", cfg, judge_backend.model)`
will look for, or just keep using `judge` as the served name to avoid the
rename entirely).

## MN5 launch script changes (on top of MIGRATE_TO_NVIDIA.md)

1. **Everything in `MIGRATE_TO_NVIDIA.md`** — container image, `CUDA_
   VISIBLE_DEVICES`, GPU-per-node topology, re-derive `--cpus-per-task`,
   re-check `--enforce-eager`, re-validate the `*256`-per-replica
   concurrency figure, re-stage the HF model cache. All of that is
   MN5-specific work your own setup presumably already covers if you've
   run other jobs there — just don't skip re-validating the concurrency
   number, it's LUMI-GPU-specific, not portable.

2. **Drop `--embedding-dedupe` from the first resumed run**, same as the
   LUMI judge-resume job (22895003) — let judge finish and get real
   quality-filter survivor counts first, uncomplicated by the dedup stage.
   Add `--embedding-dedupe --embedding-workers N` back in as a **separate,
   focused run once judge is done** (see next point) rather than bolting
   it onto the same GPU allocation.

3. **Run the embedding-dedupe stage on MN5's CPU partition, not the GPU
   one.** `embedding_dedupe_mask`'s bottleneck (the O(n^2) similarity scan)
   is plain NumPy/BLAS — no GPU involved today, confirmed by benchmarking
   it on a GPU-less debug node on LUMI. Don't burn GPU-node allocation on
   it. On MN5: submit a CPU-only job (whatever your general-purpose CPU
   partition is called there — LUMI's equivalent was `small`, 256
   CPUs/node, 72h limit) running:
   ```
   python3 -m synthgen.cli dedupe \
     --input outputs/loc_full.kept.jsonl --output outputs/loc_full.dedup.jsonl \
     --embedding-dedupe --embedding-workers <cpus-per-node, or len(langs)=35, whichever is smaller> \
     --report outputs/dedupe_report.json
   ```
   or, if driving it through `topup` directly instead of as a standalone
   `dedupe` call, add `--embedding-dedupe --embedding-workers N` to the
   `topup` invocation once judge/quality-filter are done, and run that
   round on a CPU allocation (gen/judge backends won't be hit again this
   round if `topup`'s deficit check finds every language already at
   target post-quality-filter — only true if judge-pass-rate alone already
   clears 100k/language, which is plausible given the ~82% pass rate seen
   on LUMI's partial judge run, worth checking before assuming you need
   another gen/judge round at all).

4. **Pick `--embedding-workers` based on real memory, not just core
   count.** Each worker process loads its own copy of the embedder
   (~470MB, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-
   v2`) — `35 workers * ~470MB ≈ 16GB`, trivial on a real compute node, but
   don't set `--embedding-workers` higher than `min(num_languages,
   cpu_count)` since more workers than language buckets just idles extra
   processes.

## Sizing numbers to carry over (already measured, don't re-derive)

From the LUMI debug-node benchmarks (`production/35lang_run/
_dedup_sizing_check.py`, `_dedup_sizing_check2.py`, both in git):
- Judge pass rate (score ≥ 7, the current `thresholds.json` cutoff) on
  ~2.9M judged rows so far: **82.4%**.
- Per-language gen row counts range **~165k (es) to ~254k (lv)** (35
  languages, not perfectly even — `lv`/`eu`/`is`/`sq` overgenerated more
  than the rest due to `topup`'s adaptive overgen-factor logic).
- Worst-case (near-zero real duplicate rate) embedding-dedupe cost scales
  as `t ≈ 2.68e-07 * n^2` seconds per language bucket of size `n` — e.g.
  ~91 min for an average-size bucket (~143k kept rows), ~195 min for the
  largest (`lv`, ~209k kept rows), serially. With `--embedding-workers`
  parallelizing across the 35 independent buckets, wall-clock drops to
  roughly the single slowest bucket (~195 min) instead of the sum
  (~15-54h) — confirm this holds at MN5's actual per-core throughput
  (likely different from LUMI's CPUs) before trusting the number verbatim.
