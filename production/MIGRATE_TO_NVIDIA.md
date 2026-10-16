# Migrating this workflow off LUMI (AMD/ROCm) to an NVIDIA cluster

This repo's SLURM/container plumbing (`production/35lang_run/run_production.slurm`,
`studies/localized_axes_ablation/13_concurrency_scaling/run_multinode.slurm`) was
built and tuned specifically for LUMI: ROCm GPUs (MI250, exposed as 8 GCDs/node),
Slingshot networking, and a NUMA/core-binding quirk unique to that scheduler. None
of `synthgen`'s Python code (`synthgen/`) is AMD-specific — the backend protocol
just talks HTTP to a vLLM OpenAI-compatible server, so it's portable as-is. What
needs to change is entirely in the two SLURM launch scripts and their container
image. This doc is a checklist of exactly what to touch, in order.

## 0. Things that do NOT need to change

- `synthgen/` package code, `tests/`, CLI (`synthgen topup ...` invocation) — all
  backend-agnostic, talks to vLLM over HTTP regardless of GPU vendor.
- The `httpx.Limits` connection-pool fix in `synthgen/pipeline/generate.py` —
  general httpx behavior, not LUMI-specific.
- The endpoint-pool file pattern (`*.endpoint` files with `host:port`, polled
  every ~30s) — pure filesystem/HTTP, no ROCm dependency.
- The heredoc-based multi-node launch pattern itself (two child scripts written
  to disk, config passed via exported env vars, `srun --ntasks-per-node=1` per
  node) — this structure carries over unchanged; only its *contents* change.
- `thresholds.json`, `synthgen/localized/taxonomy.py` (`DOMAINS`/`DOMAIN_WEIGHTS`),
  the 35-language config in `synthgen/config.py` — all data/config, not infra.

## 1. Container image

- LUMI uses `/scratch/project_465002530/containers/vllm-openai-rocm.sif` — a
  ROCm-built vLLM image (`SIF` env var in both `.slurm` scripts).
- On NVIDIA, use vLLM's official CUDA image instead:
  `docker://vllm/vllm-openai:latest` (or a pinned version matching whatever
  vLLM version the ROCm image used — check `vllm --version` inside the old
  container if you need to match exactly, since prompt/output behavior can
  shift slightly across vLLM versions).
- Build/pull the `.sif` on the new cluster:
  ```bash
  singularity pull vllm-openai-cuda.sif docker://vllm/vllm-openai:latest
  # or, if the cluster uses apptainer:
  apptainer pull vllm-openai-cuda.sif docker://vllm/vllm-openai:latest
  ```
- Update `export SIF=...` in both SLURM scripts to point at the new `.sif`.
- Check whether the new cluster's container runtime binary is `singularity` or
  `apptainer` (many newer clusters ship only `apptainer`, sometimes symlinked
  as `singularity` — confirm with `which singularity apptainer` on a login
  node before assuming the existing `singularity exec ...` invocations work
  unmodified).

## 2. GPU visibility env var

- ROCm: `ROCR_VISIBLE_DEVICES=0,1` (used in both scripts' `serve()` function
  to pin each vLLM replica to a GPU pair).
- CUDA: `CUDA_VISIBLE_DEVICES=0,1` — **this is the one line that will
  silently do the wrong thing if missed** (ROCm ignores an unset
  `CUDA_VISIBLE_DEVICES` and vice versa, so a stale copy-paste won't error,
  it'll just schedule all replicas onto GPU 0).
- In `production/35lang_run/_in_container_launch.sh` (and the equivalent in
  `run_multinode.slurm`'s heredoc), change:
  ```bash
  ROCR_VISIBLE_DEVICES=$gpus \
  vllm serve "$MODEL" ...
  ```
  to:
  ```bash
  CUDA_VISIBLE_DEVICES=$gpus \
  vllm serve "$MODEL" ...
  ```

## 3. GPUs-per-node — re-derive, don't assume

- LUMI's MI250 nodes expose **8 GCDs/node** (`Gres=gpu:mi250:8`), which is why
  the production script uses `REPLICAS_PER_NODE=4` (4x TP=2, GPU pairs
  `0,1`/`2,3`/`4,5`/`6,7`).
- NVIDIA nodes vary by cluster: commonly 4x A100/H100 or 8x A100/H100 per
  node. **Check the new cluster's actual node spec first**:
  ```bash
  sinfo -o "%N %G %c %m" -p <partition>   # or scontrol show node <name>
  ```
- Recompute the replica/TP layout from whatever GPU count you actually get.
  E.g. 4 GPUs/node → either `REPLICAS_PER_NODE=2` (TP=2 each, pairs `0,1`/
  `2,3`) or `REPLICAS_PER_NODE=4` (TP=1 each) depending on whether a single
  GPU has enough VRAM for the model at your target `--max-model-len`. MI250's
  128GB/GCD is unusually large — A100-40GB/H100-80GB may force TP=2 where
  LUMI could run TP=1, or vice versa. Don't port `REPLICAS_PER_NODE=4` and
  the `gpu_pairs` array verbatim; check VRAM headroom for the actual models
  (`google/gemma-4-31b-it`, `Qwen/Qwen3.6-27B`) against the new GPU's memory
  before picking a layout.

## 4. `--cpus-per-task` — re-bisect on the new cluster

- The `production/35lang_run/run_production.slurm` header currently has
  `--cpus-per-task=56`, arrived at by bisection after LUMI rejected `112`
  with "Requested node configuration is not available" (a LUMI-specific
  core/GPU NUMA-binding limit when combined with `--gpus-per-node=8
  --ntasks-per-node=1`).
- This number does **not** transfer to a different cluster — CPU-per-node
  counts and NUMA topology differ. Re-derive it the same way it was found
  here: check `sinfo -o "%c"` for the partition's real core count per node,
  then `sbatch --test-only` bisect down from `cores_per_node / ntasks_per_node`
  if the naive value gets rejected.

## 5. SLURM partition/account/time-limit names

Every `#SBATCH` line naming a LUMI-specific value needs a same-shape
replacement from the new cluster's docs/`sinfo`:

| LUMI (current) | Replace with |
|---|---|
| `--partition=standard-g` (prod) / `dev-g` (smoke test) | new cluster's GPU partition name(s) — check `sinfo` |
| `--account=project_462001516` | new cluster's allocation/account ID |
| `--time=48:00:00` | check the new partition's max walltime — `standard-g` allows 48h; a shorter-limit partition may force splitting the run or using `--dependency` chaining |

## 6. Paths — every `/scratch/project_462001516/...` reference

Both SLURM scripts and `run_production.slurm`'s exported env vars hardcode
LUMI's Lustre paths:
```bash
export REPO=/scratch/project_462001516/abhasjha/synthgen
export SIF=/scratch/project_465002530/containers/vllm-openai-rocm.sif
export HF_HOME=/scratch/project_462001516/cache/huggingface/abhasjha
export DATA_ROOT=/scratch/project_462001516/abhasjha/oellm_production_35lang
```
and the `singularity exec --bind /scratch/project_462001516:... --bind
/scratch/project_465002530:...` bind-mount flags in both `_per_node_launch.sh`
and the inline `singularity exec` call at the bottom of `run_production.slurm`.

- Update all four `export`s to the new cluster's filesystem layout.
- Update (or drop, if the new cluster's scratch is auto-bind-mounted, which
  many are) the `--bind` flags to match.
- Re-download the HF model cache on the new machine — it will **not** exist
  there. Either:
  - `huggingface_hub.snapshot_download()` from a login node into the new
    `HF_HOME` (same approach used earlier this session to rebuild LUMI's
    cache after it was found missing — `HF_HUB_DISABLE_XET=1` avoided
    chunked-storage inode bloat there; check whether the new cluster has a
    similar inode-quota constraint before assuming that flag is needed), or
  - if the new cluster has faster/unrestricted internet egress from compute
    nodes, let vLLM download on first launch instead (drop
    `HF_HUB_OFFLINE=1`) — but this adds startup latency per replica and risks
    a rate-limit/flake under 16-way parallel first-launch downloads, so the
    pre-staged-cache approach is safer for a real multi-node run.

## 7. vLLM launch flags worth re-checking (not just porting blindly)

`serve()` in both scripts currently passes:
```bash
--tensor-parallel-size $(echo $gpus | tr "," "\n" | wc -l) \
--max-model-len 16384 --gpu-memory-utilization 0.90 --enforce-eager
```
- `--enforce-eager` was likely needed/safe on ROCm (CUDA-graph capture has
  historically been flakier on ROCm builds of vLLM). On NVIDIA, CUDA graphs
  are well-supported and typically give a real throughput win — **try
  dropping `--enforce-eager` on the new cluster** and re-run the
  concurrency-scaling-style smoke test (`studies/localized_axes_ablation/
  13_concurrency_scaling/`) to confirm it's stable and faster before using it
  in a production run.
- `--gpu-memory-utilization 0.90` and `--max-model-len 16384` are model/VRAM
  decisions, not vendor-specific — safe to port as-is, but re-verify no
  OOM once the new GPU's actual free VRAM and the new `REPLICAS_PER_NODE`/TP
  layout (step 3) are settled.

## 8. Concurrency sizing — re-validate, don't assume ~256/replica ports over

The `~256 concurrent requests/replica` figure baked into
`run_production.slurm` (`GEN_CONCURRENCY=$((n_gen * 256))`) was empirically
found via a real 4-node scaling test on LUMI's specific GPUs/model/vLLM
version. It is a throughput-saturation point, not a portable constant —
different GPU (A100/H100 vs MI250), different achievable batch size, likely
a different vLLM version bundled in the new CUDA image. Re-run an equivalent
scaling smoke test on the new cluster (adapt
`studies/localized_axes_ablation/13_concurrency_scaling/bench_pool_scaling.py`
— that script itself has zero ROCm dependency, only its SLURM wrapper does)
before trusting `*256` for a real production run there.

## 9. Suggested order of operations on the new machine

1. Confirm container runtime (`singularity` vs `apptainer`) and pull a CUDA
   vLLM image (step 1).
2. Check real node GPU/CPU topology via `sinfo`/`scontrol show node` (steps
   3-5) and fill in the partition/account/time table.
3. Port `_per_node_launch.sh`/`_in_container_launch.sh`: swap
   `ROCR_VISIBLE_DEVICES`→`CUDA_VISIBLE_DEVICES`, update `gpu_pairs`/
   `REPLICAS_PER_NODE` for the real GPU count, update all paths/binds.
4. Re-stage the HF model cache (step 6).
5. Run a **small** multi-node smoke test first (2-4 nodes, short walltime —
   copy `13_concurrency_scaling/`'s structure) to shake out the
   `--cpus-per-task`/GPU-layout/container issues cheaply, same way this was
   done on LUMI before the full 16-node submission. Do not go straight to a
   16-node multi-day job on unvalidated infra.
6. Re-run the concurrency-scaling bench (step 8) on that smoke test's
   endpoints to pick a real per-replica concurrency figure.
7. Only then adapt `run_production.slurm` itself with the validated numbers
   and submit the real run.

## Open question for you to resolve before starting

This doc assumes you'll port the *existing* 16-node / 12-gen+4-judge /
`gemma-4-31b-it`+`Qwen3.6-27B` production plan as-is, just onto new hardware.
If the new machine has a meaningfully different GPU count, node count budget,
or queue-time profile than LUMI, the gen/judge node-split math (`12+4`,
chosen by minimizing `gen_work/g + judge_work/j` s.t. `g+j=N`) should be
re-derived for the new `N` and per-node throughput rather than reused
verbatim — the optimal split is a function of the two node types' relative
throughput, not a fixed ratio.
