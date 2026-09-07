#!/usr/bin/env bash
#SBATCH --job-name=axes-07-judge2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001516
#SBATCH --output=studies/localized_axes_ablation/07_judge_model_sensitivity/logs/%j.out

# Judge-only -- no gen server needed, so this uses HALF the GPUs of every
# other subfolder's run.slurm. Depends on 04_diversity_baseline's FULL-SCALE
# output (04 must have completed already: outputs/records.jsonl).
# Serves google/gemma-4-31b-it as the second judge -- a different model
# family from Qwen/Qwen3.6-27B (the study's primary judge), already cached
# from being served as "gen" elsewhere, so no extra download needed.
# Submit from repo root: sbatch studies/localized_axes_ablation/07_judge_model_sensitivity/run.slurm
set -uo pipefail

REPO=/scratch/project_462001516/abhasjha/synthgen
SIF=/scratch/project_465002530/containers/vllm-openai-rocm.sif
export HF_HOME=/scratch/project_462001516/cache/huggingface/abhasjha
STUDY="$REPO/studies/localized_axes_ablation/07_judge_model_sensitivity"
RECORDS="$REPO/studies/localized_axes_ablation/04_diversity_baseline/outputs/records.jsonl"
EP="$STUDY/endpoints"

JUDGE2_MODEL=google/gemma-4-31b-it

rm -rf "$EP"; mkdir -p "$EP/judge2" "$STUDY/logs"
HOST=$(hostname)

if [ ! -f "$RECORDS" ]; then
  echo "[07] FATAL: $RECORDS not found -- run 04_diversity_baseline's full-scale job first"
  exit 1
fi

singularity exec \
  --bind /scratch/project_462001516:/scratch/project_462001516 \
  --bind /scratch/project_465002530:/scratch/project_465002530 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 \
  --env PYTHONPATH="$REPO" \
  "$SIF" bash -c '
set -uo pipefail
cd "'"$REPO"'"

ROCR_VISIBLE_DEVICES=0,1 \
vllm serve "'"$JUDGE2_MODEL"'" --served-model-name judge2 \
  --host 0.0.0.0 --port 8003 \
  --tensor-parallel-size 2 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --enforce-eager \
  > "'"$STUDY"'/logs/vllm_judge2.log" 2>&1 &

for t in $(seq 1 72); do
  if curl -sf "http://localhost:8003/v1/models" >/dev/null 2>&1; then
    echo "'"$HOST"':8003" > "'"$EP"'/judge2/judge2.endpoint"
    echo "[ready] judge2 on 8003 after ${t}0s"; break
  fi
  sleep 10
done

echo "[07] running judge-model sensitivity re-judge"
python3 "'"$STUDY"'/rejudge.py" \
  --judge-endpoints "'"$EP"'/judge2" --judge-model judge2 \
  --records "'"$RECORDS"'" \
  --out "'"$STUDY"'/outputs"
RC=$?

echo "[07] exit code: $RC"
kill $(jobs -p) 2>/dev/null || true
exit $RC
'
