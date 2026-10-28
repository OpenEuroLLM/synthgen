# Running synthgen on MareNostrum 5 (BSC)

Setup for the `ehpc532` project. Everything lives under
`/gpfs/scratch/ehpc532/eit234648/`.

| Thing | Path |
|-------|------|
| Repo + client venv | `synthgen/` , `synthgen/venv` |
| Env activation helper | `synthgen/activate_env.sh` |
| Proxy-safe client wrapper | `synthgen/run.sh` |
| vLLM server launcher (SLURM) | `synthgen/serve_vllm.slurm` |
| vLLM container | `vllm/vllm-v0.22.1.sif` (vLLM 0.22.1 / transformers 5.10.2) |
| Shared models | `/gpfs/scratch/ehpc532/models/google/gemma-4-31B-it` |

---

## 1. Client environment

The synthgen CLI is a lightweight HTTP client (httpx + langdetect, + datasets
for the `topics` extra). It runs on the login node or inside a job.

```bash
cd /gpfs/scratch/ehpc532/eit234648/synthgen
source activate_env.sh        # loads hdf5 + python/3.12.1, activates venv,
                              # clears the BSC-injected PYTHONPATH
synthgen --help
```

To (re)install after pulling changes:

```bash
source activate_env.sh
pip install -e ".[topics,dev]"   # proxy IS needed here (login node only)
```

> The BSC `python/3.12.1` module injects a shared `PYTHONPATH`; `activate_env.sh`
> unsets it so the venv is self-contained. (On the very first install this had
> made pip skip `httpx` because it saw the system copy.)

---

## 2. Start a vLLM server (GPU job)

gemma-4-31B is ~59 GB, so it needs **2 GPUs** (`TP=2`) -- it does not fit on one
64 GB H100.

```bash
cd /gpfs/scratch/ehpc532/eit234648/synthgen
sbatch --gres=gpu:2 --cpus-per-task=40 --export=ALL,TP=2,PORT=8000 serve_vllm.slurm
# quick test instead of production? add:  --qos=acc_debug --time=00:40:00
```

Overridable env vars (via `--export=ALL,VAR=...`):
`MODEL` (default gemma-4-31B-it), `PORT` (8000), `TP` (1), `MAX_LEN` (8192),
`SIF`, `ENDPOINTS_DIR`.

The job writes `synthgen/endpoints/<node>_<port>.endpoint` once the server
answers `/health`. synthgen discovers servers by polling that directory every
30 s, so you can start several server jobs and they are picked up automatically
(remove the file / let the job end to evict one). Server log:
`synthgen/logs/vllm-<jobid>.log`.

Watch for readiness:

```bash
ls synthgen/endpoints/            # endpoint file appears when ready
tail -f synthgen/logs/vllm-<jobid>.log
```

---

## 3. Run generation

**Always launch client commands through `run.sh`**, not bare `synthgen` -- it
unsets the login-node proxy (which otherwise hijacks calls to the internal vLLM
endpoints) and sets `HF_HUB_OFFLINE=1`.

```bash
cd /gpfs/scratch/ehpc532/eit234648/synthgen
export SYNTHGEN_ROOT=/gpfs/scratch/ehpc532/<your-run-dir>

./run.sh build-prompts --phase 3                     # or --n N for a small set
./run.sh generate --backend vllm --mode prompt \
    --models gemma-4-31B-it --endpoints-dir endpoints --min-endpoints 1
./run.sh filter
./run.sh verify
./run.sh generate --backend vllm --mode response \
    --models gemma-4-31B-it --endpoints-dir endpoints
```

Generation is checkpointed: re-running the same command resumes and retries only
errored/missing rows. Add `--max-attempts N --retry-delay S` to self-heal a run.

---

## 4. Gotchas (already handled in the scripts here)

These bit us during setup; the fixes are baked into `serve_vllm.slurm` /
`run.sh`, but know them in case you write your own launcher:

1. **transformers too old.** `vllm-v0.19.0.sif` (transformers 4.57.6) cannot
   parse `model_type: gemma4`. Use `vllm-v0.22.1.sif` (transformers 5.10.2) or
   newer.
2. **`CC=icc` leaks into the container.** MN5 Intel modules set `CC=icc`, which
   Singularity passes through; `torch.compile` then fails with
   `FileNotFoundError: icc`. `serve_vllm.slurm` exports
   `SINGULARITYENV_CC=gcc SINGULARITYENV_CXX=g++` (image ships gcc 11).
3. **Proxy hijacks internal calls.** `HTTP_PROXY=localhost:8888` (for pip/git)
   makes httpx route vLLM requests through the proxy -> `All connection attempts
   failed`. `run.sh` unsets the proxy vars for client runs.

---

## 5. SLURM quick reference

- Account `ehpc532`; GPU partition `acc` (4x H100-64GB/node, 20 CPUs per GPU).
- QOS: `acc_debug` (2 h, 1 job, fast turnaround -- use for tests), `acc_ehpc`
  (production), `acc_interactive`.
- `squeue -u $USER` / `scancel <jobid>` / log in `synthgen/logs/`.
