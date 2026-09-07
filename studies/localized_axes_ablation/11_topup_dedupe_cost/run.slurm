#!/usr/bin/env bash
#SBATCH --job-name=axes-11-dedupe-cost
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001516
#SBATCH --output=studies/localized_axes_ablation/11_topup_dedupe_cost/logs/%j.out

# embedding_dedupe=False vs True topup runs, same target, to measure the real
# extra-generation cost of turning it on. Same serve-then-run pattern as
# studies/localized_bootstrap/smoke_pipeline.slurm.
# Submit from repo root: sbatch studies/localized_axes_ablation/11_topup_dedupe_cost/run.slurm
set -uo pipefail

REPO=/scratch/project_462001516/abhasjha/synthgen
SIF=/scratch/project_465002530/containers/vllm-openai-rocm.sif
export HF_HOME=/scratch/project_462001516/cache/huggingface/abhasjha
STUDY="$REPO/studies/localized_axes_ablation/11_topup_dedupe_cost"
EP="$STUDY/endpoints"

GEN_MODEL=google/gemma-4-31b-it
JUDGE_MODEL=Qwen/Qwen3.6-27B

rm -rf "$EP"; mkdir -p "$EP/gen" "$EP/judge" "$STUDY/logs"
HOST=$(hostname)

singularity exec \
  --bind /scratch/project_462001516:/scratch/project_462001516 \
  --bind /scratch/project_465002530:/scratch/project_465002530 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 \
  --env PYTHONPATH="$REPO" \
  "$SIF" bash -c '
set -uo pipefail
cd "'"$REPO"'"

# This is the one script in the study that ALSO needs sentence-transformers
# inside the container (the "on" arm calls near_dup.embedding_dedupe for
# real, not just as a post-hoc diagnostic) -- install once if missing, same
# offline-after-first-pull convention as _common/diversity_metrics.py.
python3 -c "import sentence_transformers" 2>/dev/null || pip install --quiet sentence-transformers

serve () {  # name port gpus model
  local name=$1 port=$2 gpus=$3 model=$4
  ROCR_VISIBLE_DEVICES=$gpus \
  vllm serve "$model" --served-model-name "$name" \
    --host 0.0.0.0 --port "$port" \
    --tensor-parallel-size $(echo $gpus | tr "," "\n" | wc -l) \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --enforce-eager \
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

serve gen 8001 0,1 "'"$GEN_MODEL"'"
wait_ready gen 8001 || { kill $(jobs -p) 2>/dev/null; exit 1; }
serve judge 8002 2,3 "'"$JUDGE_MODEL"'"
wait_ready judge 8002 || { kill $(jobs -p) 2>/dev/null; exit 1; }

echo "[11] running topup embedding-dedupe cost ablation"
python3 "'"$STUDY"'/measure_dedupe_cost.py" \
  --gen-endpoints "'"$EP"'/gen" --judge-endpoints "'"$EP"'/judge" \
  --target 30 --langs es fr de pl uk \
  --out "'"$STUDY"'/outputs"
RC=$?

echo "[11] exit code: $RC"
kill $(jobs -p) 2>/dev/null || true
exit $RC
'
