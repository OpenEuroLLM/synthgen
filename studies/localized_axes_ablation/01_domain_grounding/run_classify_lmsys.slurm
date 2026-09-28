#!/usr/bin/env bash
#SBATCH --job-name=axes-01-lmsys-classify
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001516
#SBATCH --output=studies/localized_axes_ablation/01_domain_grounding/logs/%j.out

# Classifies the lmsys-chat-1m texts already cached by warm_texts_cache.py
# (login-node step, needs internet) via vLLM (no internet needed here) --
# a topic-distribution comparison against the existing WildChat-1M numbers
# in domain_distribution/*.json, using the SAME classifier machinery
# (ground_domains.py), just pointed at a different --dataset/--out.
# Submit from repo root: sbatch studies/localized_axes_ablation/01_domain_grounding/run_classify_lmsys.slurm
set -uo pipefail

REPO=/scratch/project_462001516/abhasjha/synthgen
SIF=/scratch/project_465002530/containers/vllm-openai-rocm.sif
export HF_HOME=/scratch/project_462001516/cache/huggingface/abhasjha
STUDY="$REPO/studies/localized_axes_ablation/01_domain_grounding"
EP="$STUDY/endpoints_lmsys"
OUT="$STUDY/dataset_compare/lmsys"

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

serve () {  # name port gpus model
  local name=$1 port=$2 gpus=$3 model=$4
  ROCR_VISIBLE_DEVICES=$gpus \
  vllm serve "$model" --served-model-name "$name" \
    --host 0.0.0.0 --port "$port" \
    --tensor-parallel-size $(echo $gpus | tr "," "\n" | wc -l) \
    --max-model-len 4096 --gpu-memory-utilization 0.90 --enforce-eager \
    > "'"$STUDY"'/logs/vllm_${name}.log" 2>&1 &
}

wait_ready () {  # name port
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

# Classification only needs short outputs (one category word) -- a smaller
# max-model-len than the gen/judge servers elsewhere in this study is fine
# and starts faster.
serve gen 8001 0,1 google/gemma-4-31b-it
wait_ready gen 8001 || { kill $(jobs -p) 2>/dev/null; exit 1; }

echo "[01-lmsys] classifying cached lmsys-chat-1m texts against Pool A taxonomy"
python3 "'"$STUDY"'/ground_domains.py" \
  --dataset lmsys/lmsys-chat-1m --n-per-lang 400 --langs es fr de pl uk \
  --texts-cache "'"$OUT"'/texts_cache" \
  --classifier-backend vllm --vllm-endpoints "'"$EP"'/gen" --classifier-model gen \
  --out "'"$OUT"'"
RC=$?

echo "[01-lmsys] exit code: $RC"
kill $(jobs -p) 2>/dev/null || true
exit $RC
'
