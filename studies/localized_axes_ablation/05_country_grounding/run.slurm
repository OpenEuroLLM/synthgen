#!/usr/bin/env bash
#SBATCH --job-name=axes-05-country
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --partition=dev-g
#SBATCH --account=project_462001516
#SBATCH --output=studies/localized_axes_ablation/05_country_grounding/logs/%j.out

# Runs BOTH scripts: check_country_accuracy.py (needs 01's output — must run
# 01_domain_grounding/run.slurm first) and country_diversify.py (standalone).
# Same serve-then-run pattern as studies/localized_bootstrap/smoke_pipeline.slurm.
# Submit from repo root: sbatch studies/localized_axes_ablation/05_country_grounding/run.slurm
set -uo pipefail

REPO=/scratch/project_462001516/abhasjha/synthgen
SIF=/scratch/project_465002530/containers/vllm-openai-rocm.sif
export HF_HOME=/scratch/project_462001516/cache/huggingface/abhasjha
STUDY="$REPO/studies/localized_axes_ablation/05_country_grounding"
GROUNDING_RECORDS="$REPO/studies/localized_axes_ablation/01_domain_grounding/outputs/mode_deep/records.jsonl"
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

if [ -f "'"$GROUNDING_RECORDS"'" ]; then
  echo "[05a] running check_country_accuracy.py"
  python3 "'"$STUDY"'/check_country_accuracy.py" \
    --judge-endpoints "'"$EP"'/judge" --records "'"$GROUNDING_RECORDS"'" \
    --out "'"$STUDY"'/outputs/accuracy"
else
  echo "[05a] SKIPPED — 01_domain_grounding has not been run yet " \
       "(expected '"$GROUNDING_RECORDS"')"
fi

echo "[05b] running country_diversify.py"
python3 "'"$STUDY"'/country_diversify.py" \
  --gen-endpoints "'"$EP"'/gen" --judge-endpoints "'"$EP"'/judge" \
  --n-per-lang 30 --langs fr pt de es \
  --out "'"$STUDY"'/outputs/diversify"
RC=$?

echo "[05] exit code: $RC"
kill $(jobs -p) 2>/dev/null || true
exit $RC
'
