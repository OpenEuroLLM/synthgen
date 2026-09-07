#!/usr/bin/env bash
#SBATCH --job-name=synthgen-smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:20:00
#SBATCH --partition=dev-g
#SBATCH --account=project_462001516
#SBATCH --output=studies/localized_bootstrap/logs/smoke-%j.out

# One-off smoke test of the refactored synthgen.pipeline / synthgen.localized
# CLI (post test-import-fix) against real vLLM servers, before committing.
# Not part of the production study — throwaway run, throwaway output dir.
# Submit from the synthgen repo root: sbatch studies/localized_bootstrap/smoke_pipeline.slurm
set -uo pipefail

REPO=/scratch/project_462001516/abhasjha/synthgen
SIF=/scratch/project_465002530/containers/vllm-openai-rocm.sif
export HF_HOME=/scratch/project_462001516/cache/huggingface/abhasjha
RUN_ROOT=/scratch/project_462001516/abhasjha/synthgen_smoke_run
STUDY="$REPO/studies/localized_bootstrap"
EP="$RUN_ROOT/endpoints"

GEN_MODEL=google/gemma-4-31b-it
JUDGE_MODEL=Qwen/Qwen3.6-27B

rm -rf "$EP"; mkdir -p "$EP/gen" "$EP/judge" "$STUDY/logs" "$RUN_ROOT"
HOST=$(hostname)

singularity exec \
  --bind /scratch/project_462001516:/scratch/project_462001516 \
  --bind /scratch/project_465002530:/scratch/project_465002530 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 \
  --env PYTHONPATH="$REPO" --env SYNTHGEN_ROOT="$RUN_ROOT" \
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
    > "'"$STUDY"'/logs/smoke_vllm_${name}.log" 2>&1 &
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
  echo "[FATAL] ${name} failed to start (see logs/smoke_vllm_${name}.log)"; return 1
}

serve gen 8001 0,1 "'"$GEN_MODEL"'"
wait_ready gen 8001 || { kill $(jobs -p) 2>/dev/null; exit 1; }
serve judge 8002 2,3 "'"$JUDGE_MODEL"'"
wait_ready judge 8002 || { kill $(jobs -p) 2>/dev/null; exit 1; }

echo "[smoke] running: build-localized-prompts + generate(localized) + generate(judge) + quality-filter + dedupe via synthgen.cli topup"
python3 -m synthgen topup \
  --root "'"$RUN_ROOT"'" \
  --target 5 \
  --langs de fr \
  --thresholds "'"$RUN_ROOT"'/thresholds_smoke.json" \
  --gen-backend vllm --gen-models gen --gen-endpoints-dir "'"$EP"'/gen" \
  --judge-backend vllm --judge-model judge --judge-endpoints-dir "'"$EP"'/judge" \
  --max-rounds 2 --max-attempts 2 --retry-delay 10
RC=$?

echo "[smoke] topup exit code: $RC"
echo "[smoke] survivors file:"
cat "'"$RUN_ROOT"'/outputs/loc_full.dedup.jsonl" 2>&1 | wc -l

echo "[done] smoke test finished; shutting down servers"
kill $(jobs -p) 2>/dev/null || true
exit $RC
'
