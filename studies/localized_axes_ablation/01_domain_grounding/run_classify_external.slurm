#!/usr/bin/env bash
#SBATCH --job-name=axes-01-external-classify
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001516
#SBATCH --output=studies/localized_axes_ablation/01_domain_grounding/logs/%j.out

# Classifies oasst2 + aya_dataset's already-cached texts (warm_texts_cache_
# external.py, login-node step) against the same Pool A taxonomy, one vLLM
# server, both datasets in one allocation.
# Submit from repo root: sbatch studies/localized_axes_ablation/01_domain_grounding/run_classify_external.slurm
set -uo pipefail

REPO=/scratch/project_462001516/abhasjha/synthgen
SIF=/scratch/project_465002530/containers/vllm-openai-rocm.sif
export HF_HOME=/scratch/project_462001516/cache/huggingface/abhasjha
STUDY="$REPO/studies/localized_axes_ablation/01_domain_grounding"
EP="$STUDY/endpoints_external"

rm -rf "$EP"; mkdir -p "$EP/gen" "$STUDY/logs"
HOST=$(hostname)

singularity exec \
  --bind /scratch/project_462001516:/scratch/project_462001516 \
  --bind /scratch/project_465002530:/scratch/project_465002530 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 \
  --env PYTHONPATH="$REPO" \
  "$SIF" bash -c '
set -uo pipefail
cd "'"$REPO"'"

serve () {
  local name=$1 port=$2 gpus=$3 model=$4
  ROCR_VISIBLE_DEVICES=$gpus \
  vllm serve "$model" --served-model-name "$name" \
    --host 0.0.0.0 --port "$port" \
    --tensor-parallel-size $(echo $gpus | tr "," "\n" | wc -l) \
    --max-model-len 4096 --gpu-memory-utilization 0.90 --enforce-eager \
    > "'"$STUDY"'/logs/vllm_${name}.log" 2>&1 &
}

wait_ready () {
  local name=$1 port=$2
  for t in $(seq 1 72); do
    if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
      echo "'"$HOST"':${port}" > "'"$EP"'/${name}/${name}.endpoint"
      echo "[ready] ${name} on ${port} after ${t}0s"; return 0
    fi
    sleep 10
  done
  echo "[FATAL] ${name} failed to start (see logs/vllm_${name}.log)"; return 1
}

serve gen 8001 0,1 google/gemma-4-31b-it
wait_ready gen 8001 || { kill $(jobs -p) 2>/dev/null; exit 1; }

echo "[01-external] classifying oasst2 + aya_dataset cached texts"
python3 "'"$STUDY"'/classify_external.py" \
  --job "oasst2:'"$STUDY"'/dataset_compare/oasst2" \
  --job "aya:'"$STUDY"'/dataset_compare/aya" \
  --vllm-endpoints "'"$EP"'/gen" --vllm-model gen
RC=$?

echo "[01-external] exit code: $RC"
kill $(jobs -p) 2>/dev/null || true
exit $RC
'
