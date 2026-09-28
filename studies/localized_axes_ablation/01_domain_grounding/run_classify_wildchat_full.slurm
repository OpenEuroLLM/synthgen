#!/usr/bin/env bash
#SBATCH --job-name=axes-01-wildchat-full
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001516
#SBATCH --output=studies/localized_axes_ablation/01_domain_grounding/logs/%j.out

# Re-classifies WildChat-1M's ALREADY-CACHED texts (texts_cache/, from the
# original full-scale 01 run -- 1200 candidates/lang cached, only 30/lang
# were ever actually classified) at the full n=400/lang, for production
# promotion (see the plan: domain weights need real signal, not smoke-scale).
# No internet needed -- texts are already on disk.
# Submit from repo root: sbatch studies/localized_axes_ablation/01_domain_grounding/run_classify_wildchat_full.slurm
set -uo pipefail

REPO=/scratch/project_462001516/abhasjha/synthgen
SIF=/scratch/project_465002530/containers/vllm-openai-rocm.sif
export HF_HOME=/scratch/project_462001516/cache/huggingface/abhasjha
STUDY="$REPO/studies/localized_axes_ablation/01_domain_grounding"
EP="$STUDY/endpoints_wildchat_full"

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

echo "[01-wildchat-full] classifying cached WildChat-1M texts at n=400/lang"
python3 "'"$STUDY"'/ground_domains.py" \
  --dataset allenai/WildChat-1M --n-per-lang 400 --langs es fr de pl uk \
  --texts-cache "'"$STUDY"'/texts_cache" \
  --classifier-backend vllm --vllm-endpoints "'"$EP"'/gen" --classifier-model gen \
  --out "'"$STUDY"'"
RC=$?

echo "[01-wildchat-full] exit code: $RC"
kill $(jobs -p) 2>/dev/null || true
exit $RC
'
