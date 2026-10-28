#!/bin/bash
#SBATCH --job-name=synthgen-vllm
#SBATCH --account=ehpc532
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/ehpc532/eit234648/synthgen/logs/vllm-%j.log

set -euo pipefail

# ---- config (override via env) --------------------------------------------
ROOT=/gpfs/scratch/ehpc532/eit234648
SIF=${SIF:-$ROOT/vllm/vllm-v0.22.1.sif}
MODEL=${MODEL:-/gpfs/scratch/ehpc532/models/google/gemma-4-31B-it}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-8192}
TP=${TP:-1}
ENDPOINTS_DIR=${ENDPOINTS_DIR:-$ROOT/synthgen/endpoints}
# ---------------------------------------------------------------------------

module load singularity/4.1.5

# MN5 Intel modules leak CC=icc into the container; torch.compile (Inductor)
# needs a real compiler. The image ships gcc 11 -- point the build tools at it.
export SINGULARITYENV_CC=gcc SINGULARITYENV_CXX=g++
export SINGULARITYENV_VLLM_DISABLE_COMPILE_CACHE=1
mkdir -p "$ENDPOINTS_DIR" "$ROOT/synthgen/logs"

HOST=$(hostname)
EP_FILE="$ENDPOINTS_DIR/${HOST}_${PORT}.endpoint"

# Publish endpoint for synthgen once the server is up; remove on exit.
cleanup() { rm -f "$EP_FILE"; }
trap cleanup EXIT

# Offline HF (weights are local); write endpoint after server answers.
(
  until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do sleep 5; done
  echo "${HOST}:${PORT}" > "$EP_FILE"
  echo "[serve_vllm] endpoint published: $EP_FILE -> ${HOST}:${PORT}"
) &

exec singularity exec --nv \
  -B /gpfs/scratch/ehpc532 \
  "$SIF" \
  python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port "${PORT}" \
    --model "${MODEL}" \
    --served-model-name "$(basename "${MODEL}")" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --no-enable-log-requests
