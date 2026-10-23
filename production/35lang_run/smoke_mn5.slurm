#!/usr/bin/env bash
#SBATCH --job-name=oellm-smoke-mn5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc532
#SBATCH --qos=acc_debug
#SBATCH --output=production/35lang_run/logs/smoke_%j.out

# ============================================================================
# Single-node acc_debug smoke test for the MN5 port of the 35-lang topup run.
# Brings up 1 gen replica (gemma-4-31B-it, GPUs 0,1) + 1 judge replica
# (Qwen3.6-27B, GPUs 2,3) -- the exact per-node layout the production script
# uses -- then runs a tiny 2-language topup into a THROWAWAY data root so it
# never touches the 20G production outputs.
#
# Validates: singularity exec --nv, CUDA_VISIBLE_DEVICES pinning, both models
# load in vllm-v0.22.1.sif (gemma4 + qwen3_5 arch), and gen->judge->filter->
# dedupe runs end-to-end on MN5.
#
# Submit from repo root:  sbatch production/35lang_run/smoke_mn5.slurm
# ============================================================================
set -uo pipefail

# Drop the login-shell download proxy so it can't hijack the localhost curl
# health-checks or synthgen's calls to the vLLM endpoints (MN5.md gotcha #3).
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

REPO=/gpfs/scratch/ehpc532/eit234648/synthgen
SIF=/gpfs/scratch/ehpc532/eit234648/vllm/vllm-v0.22.1.sif
HF_HOME=/gpfs/scratch/ehpc532/eit234648/.hf_cache
RUN="$REPO/production/35lang_run"
EP="$RUN/endpoints_smoke"
DATA_ROOT="$RUN/_smoke_data"          # throwaway; NOT the production outputs
GEN_MODEL=/gpfs/scratch/ehpc532/models/google/gemma-4-31B-it
JUDGE_MODEL=/gpfs/scratch/ehpc532/eit234648/models/Qwen/Qwen3.6-27B

module load singularity/4.1.5
export SINGULARITYENV_CC=gcc SINGULARITYENV_CXX=g++
export SINGULARITYENV_VLLM_DISABLE_COMPILE_CACHE=1

rm -rf "$EP" "$DATA_ROOT"
mkdir -p "$EP/gen" "$EP/judge" "$RUN/logs" "$DATA_ROOT"

# --- launch both servers inside one container (gen on 0,1 ; judge on 2,3) ----
singularity exec --nv \
  --bind /gpfs/scratch/ehpc532 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
  "$SIF" bash -c '
set -uo pipefail
CUDA_VISIBLE_DEVICES=0,1 vllm serve "'"$GEN_MODEL"'" --served-model-name gen \
  --host 0.0.0.0 --port 8001 --tensor-parallel-size 2 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --trust-remote-code \
  > "'"$RUN"'/logs/smoke_vllm_gen.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 vllm serve "'"$JUDGE_MODEL"'" --served-model-name judge \
  --host 0.0.0.0 --port 8002 --tensor-parallel-size 2 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --trust-remote-code \
  > "'"$RUN"'/logs/smoke_vllm_judge.log" 2>&1 &
wait
' &
SERVE_PID=$!

HOST=$(hostname)
# --- wait for both to answer, then publish endpoint files --------------------
ready_gen=0; ready_judge=0
for t in $(seq 1 90); do
  if [ "$ready_gen" -eq 0 ] && curl -sf "http://localhost:8001/v1/models" >/dev/null 2>&1; then
    echo "${HOST}:8001" > "$EP/gen/gen_0.endpoint"; ready_gen=1; echo "[smoke] gen ready after ${t}0s"
  fi
  if [ "$ready_judge" -eq 0 ] && curl -sf "http://localhost:8002/v1/models" >/dev/null 2>&1; then
    echo "${HOST}:8002" > "$EP/judge/judge_0.endpoint"; ready_judge=1; echo "[smoke] judge ready after ${t}0s"
  fi
  [ "$ready_gen" -eq 1 ] && [ "$ready_judge" -eq 1 ] && break
  sleep 10
done

if [ "$ready_gen" -eq 0 ] || [ "$ready_judge" -eq 0 ]; then
  echo "[smoke] FATAL: servers did not come up (gen=$ready_gen judge=$ready_judge) -- see logs/smoke_vllm_*.log"
  kill $SERVE_PID 2>/dev/null || true
  exit 1
fi

# --- tiny end-to-end topup (2 langs, target 50, 2 rounds) --------------------
singularity exec --nv \
  --bind /gpfs/scratch/ehpc532 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
  "$SIF" bash -c "
set -uo pipefail
cd '$REPO'
python3 -m synthgen.cli topup \
  --root '$DATA_ROOT' \
  --target 50 \
  --thresholds '$RUN/thresholds.json' \
  --langs es fr \
  --gen-backend vllm --gen-models gen --gen-endpoints-dir '$EP/gen' --gen-min-endpoints 1 \
  --judge-backend vllm --judge-model judge --judge-endpoints-dir '$EP/judge' --judge-min-endpoints 1 \
  --gen-concurrency 64 --judge-concurrency 64 --concurrency 64 \
  --max-rounds 2 --max-attempts 3 --retry-delay 10
"
RC=$?

echo "[smoke] topup exit code: $RC"
echo "[smoke] survivors:"; ls -la "$DATA_ROOT/outputs/" 2>/dev/null
kill $SERVE_PID 2>/dev/null || true
exit $RC
