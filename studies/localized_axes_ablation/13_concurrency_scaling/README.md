# 13 — concurrency scaling smoke test

Does throughput actually scale with replica count when `concurrency` is
raised proportionally, or does the endpoint pool's plain round-robin dilute
a fixed concurrency budget across replicas the way the code trace suggested?

## Why this exists

While estimating the compute cost of a 36-language, 100k-example-per-language
production run, tracing `synthgen/backends/vllm.py`'s `EndpointPool` (plain
round-robin over `*.endpoint` files) and `synthgen/pipeline/generate.py`'s
`asyncio.Semaphore(cfg.concurrency)` (one shared value per phase) showed that
a single `concurrency` setting gets spread evenly across however many
replicas are registered in a pool — meaning more hardware only helps if
`concurrency` is raised proportionally. This was traced through the code,
not measured against a real multi-node pool. This ablation is that
measurement, and also exercises the new `gen_concurrency`/`judge_concurrency`
overrides added to `topup.run()` this session (needed because gen and judge
pools realistically have very different replica counts — gen needs ~12x more
raw compute per this study's own cost analysis).

## Design

4 real LUMI-G nodes, `srun`-launched, each running the *exact* proven
single-node serve pattern from every other ablation in this study (2 GPUs
gen + 2 GPUs judge, TP=2 each) — deliberately the simple symmetric layout,
not the more GPU-efficient asymmetric one, to keep this smoke test low-risk.
Gives one shared pool of 4 gen replicas + one shared pool of 4 judge
replicas, all endpoint files written to a common Lustre directory so a
single `VLLMBackend` per role sees and round-robins across all of them.

`bench_pool_scaling.py` then sweeps concurrency (at 8x, 32x, 128x, 256x the
detected replica count) through the **real** `VLLMBackend`/`EndpointPool`
code path — not raw HTTP like `studies/localized_bootstrap/bench_throughput.py`
— and reports achieved aggregate tok/s against the expected ceiling
(`n_replicas × single-replica peak`, where single-replica peak is 968 tok/s
gen / 795 tok/s judge, from that earlier benchmark).

```
sbatch run_multinode.slurm
```

## What a pass/fail result means

- **Pass**: achieved tok/s at concurrency ≈ `n_replicas × 256` approaches
  `n_replicas × single-replica peak` (allowing for real overhead — some
  falloff from the ideal is expected, e.g. shared filesystem/network
  contention across nodes that a single-node benchmark can't surface).
  Confirms the pooling + concurrency-scaling story holds at real multi-node
  scale, and that the `gen_concurrency`/`judge_concurrency` overrides in
  `topup.py` are the right lever for a production run.
- **Fail** (throughput plateaus well below `n_replicas ×` the single-replica
  peak even at high concurrency): would mean either the round-robin isn't
  actually spreading load evenly, or some other bottleneck (shared
  filesystem endpoint-file polling, network, orchestrating-process overhead)
  caps aggregate throughput below what raw compute should allow — worth
  knowing before committing a production run's node count to the naive
  linear-scaling assumption.

## Output

`outputs/summary.json` — replica counts detected, peak tok/s for gen and
judge, and each as a multiple of the single-replica peak (the scaling factor
this test exists to measure).
