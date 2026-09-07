#!/usr/bin/env bash
#SBATCH --job-name=loc-bench
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --partition=dev-g
#SBATCH --account=project_462001516
#SBATCH --output=studies/localized_bootstrap/logs/bench-%j.out

# Saturation throughput benchmark: serve gemma-4-31b-it and Qwen3.6-27B, then
# sweep concurrency against each to find peak sustained decode tok/s per replica.
set -uo pipefail

REPO=/scratch/project_462001516/abhasjha/synthgen
SIF=/scratch/project_465002530/containers/vllm-openai-rocm.sif
export HF_HOME=/scratch/project_462001516/cache/huggingface/abhasjha
STUDY="$REPO/studies/localized_bootstrap"
EP="$STUDY/endpoints_bench"

rm -rf "$EP"; mkdir -p "$EP/gen" "$EP/judge" "$STUDY/logs"
HOST=$(hostname)

singularity exec \
  --bind /scratch/project_462001516:/scratch/project_462001516 \
  --bind /scratch/project_465002530:/scratch/project_465002530 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
  "$SIF" bash -c '
set -uo pipefail
cd "'"$REPO"'"
serve () {  # name port gpus model
  ROCR_VISIBLE_DEVICES=$3 vllm serve "$4" --served-model-name "$1" \
    --host 0.0.0.0 --port "$2" --tensor-parallel-size $(echo $3 | tr "," "\n" | wc -l) \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --enforce-eager \
    > "'"$STUDY"'/logs/bench_vllm_$1.log" 2>&1 &
}
wait_ready () {  # name port
  for t in $(seq 1 72); do
    if curl -sf "http://localhost:${2}/v1/models" >/dev/null 2>&1; then
      echo "'"$HOST"':${2}" > "'"$EP"'/${1}/${1}.endpoint"; echo "[ready] ${1}"; return 0
    fi; sleep 10
  done
  echo "[FATAL] ${1} failed to start"; return 1
}
# Stagger init so the two TP=2 groups do not race on shared memory.
serve gen 8001 0,1 google/gemma-4-31b-it
wait_ready gen 8001 || { kill $(jobs -p) 2>/dev/null; exit 1; }
serve judge 8002 2,3 Qwen/Qwen3.6-27B
wait_ready judge 8002 || { kill $(jobs -p) 2>/dev/null; exit 1; }

echo "===== GENERATOR (gemma-4-31b-it) ====="
python3 studies/localized_bootstrap/bench_throughput.py \
  --endpoints studies/localized_bootstrap/endpoints_bench/gen --model gen --max-tokens 512
echo "===== JUDGE (Qwen3.6-27B) ====="
python3 studies/localized_bootstrap/bench_throughput.py \
  --endpoints studies/localized_bootstrap/endpoints_bench/judge --model judge --max-tokens 512
kill $(jobs -p) 2>/dev/null || true
'
