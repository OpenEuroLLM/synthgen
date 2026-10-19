# Handoff: continuing the 35-language production run on MN5

Supersedes the previous version of this doc (dated 2026-10-16, written
before the topup fix below and before the LUMI job was cancelled). Read
this one; the old one is gone from git history but you may still have a
stale local copy — discard it.

## Current state (as of 2026-10-19, end of the LUMI phase of this run)

- **LUMI job 22895003 has been cancelled** (by explicit request — the rest
  of this run continues on MN5 from here, not on LUMI). It was mid-gen on
  a new round when killed; nothing was lost (`generate.run()` checkpoints
  every successful row as it goes, and the in-flight round's new prompts
  are harmless to throw away since they were sized by the now-fixed buggy
  logic anyway — see below).
- **A real bug in `topup.py`'s round loop was found and fixed this
  session**, already on `origin/main` (`bb879f8`,
  "Fix topup's blind, pooled generation-provisioning decisions") and
  **already pulled onto your MN5 checkout** (I ran `git pull origin main`
  there directly — you're at `bb879f8`, up from `55cd05f`). Summary (full
  detail in `production/35lang_run/FINDINGS.md`, which is also new this
  session and now in git):
  1. The round-start decision to write new generation prompts for a
     language was blind to judge data — it only checked whether
     `loc_full.dedup.jsonl` existed (it never has, since no round had ever
     reached dedupe) and whether that language's prompt backlog was
     empty. It had no way to tell an already-satisfied language (e.g. `de`
     at ~150% of target) from a desperately short one (`ga`/Irish, stuck
     at 8.3% judge pass-rate) — both just read as "survivors=0". Fixed by
     adding a lazy interim judge-aware estimate (real `quality_filter.run()`
     + `dedupe.run()` against current on-disk data, written to throwaway
     `interim_*` paths so the official round output is untouched) that
     gates the skip/generate decision per language.
  2. The overgen factor (`n_new = deficit × factor`) was pooled across all
     35 languages into one number, so a language needing ~12× overgen
     (`ga`) got the same ~1.3-1.6× as the 90%+-pass-rate majority — badly
     under-provisioned. Fixed: factor is now computed **per-language** from
     that same interim estimate, clamped to `[1.1, max_overgen_factor]`
     (new `--max-overgen-factor` CLI flag, default 5.0 — raise it for `ga`
     specifically if round 1 on MN5 still leaves it short).
  3. Additive fix: judge's dedicated nodes used to sit idle for an entire
     round's gen phase even when there was already a multi-million-row
     generated-but-unjudged backlog on disk (exactly our situation). Each
     round now drains that backlog first (an extra `generate.run(mode=
     "judge")` pass at the top of the round, before deciding on new
     prompts) — still strictly sequential with gen (not true concurrency),
     just reordered so judge gets first claim on existing work.

  All three are covered by `tests/test_topup_integration.py` (a new third
  test added alongside the two pre-existing ones; full suite — 83 tests —
  green). **No change needed to `run_production_mn5.slurm`** — it already
  invokes `python3 -m synthgen.cli topup` with no hardcoded factor logic,
  so it picks up the fix automatically now that the repo is pulled.
  Consider passing `--max-overgen-factor 12` or so on this next run given
  `ga`'s real measured pass rate (see table below) — the default 5.0 cap
  will still under-provision it somewhat, just far less badly than the old
  pooled behavior.

- **Gen phase: effectively complete, but inflated by the bug above.**
  `loc_gen.jsonl` — 23,879,567,672 bytes, **7,575,024 rows** (up from the
  16-10 handoff's 6,064,877 — the extra ~1.5M rows are the wasted
  blind-overgen batches bug #1 above describes; harmless, just extra raw
  material, quality-filter/dedupe will sort it out). Every language already
  has well over 100% of target in raw generations (`ga` included, at 169%
  raw — its shortfall is 100% a pass-rate problem, not a quantity one, so
  don't be tempted to just throw more raw gen at it; see FINDINGS.md).
- **Judge phase: in progress, resumable.** `judged_judge.jsonl` —
  1,603,932,015 bytes, **4,000,378 / ~7.6M rows judged** (the denominator
  grew because of the extra gen rows above, so the percentage looks lower
  than the 16-10 handoff's 71.3% even though strictly more judging has
  happened in absolute terms). Score distribution, per-language pass
  rates, and the full "% of 100k target already met" table are in
  `production/35lang_run/FINDINGS.md` — use that instead of re-deriving it;
  it was computed from a real debug-node job (`_judge_score_dist.py`), not
  guesswork.
- **Quality-filter + exact-dedupe + embedding-dedupe: still never run even
  once** — unchanged from the 16-10 handoff, this is why `loc_full.
  dedup.jsonl` still doesn't exist anywhere. This is expected and fine;
  the new interim-estimate logic in the fix above is specifically designed
  to make good decisions despite this.

## Data transfer

Already in progress as of this handoff — I'm rsyncing the current
`outputs/loc_gen.jsonl` (23.9 GB) and `outputs/judged_judge.jsonl` (1.6 GB)
from LUMI directly to `mn5-transfer:/gpfs/scratch/ehpc532/eit234648/
oellm_production_35lang/outputs/`, overwriting the smaller/staler copies
already there (6,064,877-row gen / 3,964,000-row judge, from the 16-10
hand-off). **Check the transfer actually finished and re-verify row counts
before launching the next MN5 job** — don't trust this doc's byte counts
blindly, confirm live:

```bash
wc -l /gpfs/scratch/ehpc532/eit234648/oellm_production_35lang/outputs/*.jsonl
# expect: loc_gen.jsonl 7575024 lines, judged_judge.jsonl 4000378 lines
ls -la /gpfs/scratch/ehpc532/eit234648/oellm_production_35lang/outputs/
# expect: loc_gen.jsonl 23879567672 bytes, judged_judge.jsonl 1603932015 bytes
```

`production/35lang_run/thresholds.json` (`{"global": 7}`) was already
correct in git on both sides — no transfer needed there.

## Everything else from the previous handoff still applies

- All of `MIGRATE_TO_NVIDIA.md`'s general AMD→NVIDIA porting concerns.
- Drop (keep dropped) `--embedding-dedupe` from the main topup run; run it
  as a **separate CPU-only job** once judge is actually done, using
  `--embedding-workers N` (the process-pool parallelization from
  `55cd05f`, already on MN5) — don't burn GPU-node allocation on it, it's
  pure NumPy/BLAS with no GPU involvement, verified on a LUMI debug node.
- Re-validate the `256`-concurrent-requests-per-replica figure on H100 —
  it's LUMI/MI250-specific, not portable, and `run_production_mn5.slurm`
  already has a comment flagging this.

## What's committed and pushed vs. what's local-only

Pushed to `origin/main` this session (all now on your MN5 checkout):
`bb879f8` (the topup fix), plus everything from `55cd05f` and earlier
(embedding-dedupe parallelization, judge-only resume support). Your MN5
local files (`MN5.md`, `activate_env.sh`, `run.sh`, `serve_vllm.slurm`,
`run_production_mn5.slurm`, `smoke_mn5.slurm`, `endpoints/`) are untracked
and were left untouched by the pull — still exactly as you had them.
