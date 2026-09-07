#!/usr/bin/env bash
#SBATCH --job-name=axes-01-grounding-full
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001516
#SBATCH --output=studies/localized_axes_ablation/01_domain_grounding/logs/full_%j.out

# Grounding-mode sweep {none, light, deep}. REQUIRES ground_domains.py to have
# already been run on a login node (needs internet) — this job only reads its
# static domain_exemplars/ + domain_distribution/ output, never fetches
# anything itself. Same serve-then-run pattern as
# studies/localized_bootstrap/smoke_pipeline.slurm.
# Submit from repo root: sbatch studies/localized_axes_ablation/01_domain_grounding/run.slurm
set -uo pipefail

REPO=/scratch/project_462001516/abhasjha/synthgen
SIF=/scratch/project_465002530/containers/vllm-openai-rocm.sif
export HF_HOME=/scratch/project_462001516/cache/huggingface/abhasjha
STUDY="$REPO/studies/localized_axes_ablation/01_domain_grounding"
EP="$STUDY/endpoints"

# ground_domains.py's HF-Hub streaming step needs internet, which SLURM
# compute nodes don't have. Fix: warm_texts_cache.py runs once from the
# login node (see 01_domain_grounding/README.md) and writes raw sampled
# texts to texts_cache/. This job then classifies from that cache via the
# vLLM gen server started below (--classifier-backend vllm), so no
# OPENROUTER_API_KEY or network access is needed inside the job.

GEN_MODEL=google/gemma-4-31b-it
JUDGE_MODEL=Qwen/Qwen3.6-27B
N_PER_LANG=30
LANGS="es fr de pl uk"

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

echo "[01] classifying real-data domains from texts_cache (vLLM backend)"
python3 "'"$STUDY"'/ground_domains.py" \
  --classifier-backend vllm --vllm-endpoints "'"$EP"'/gen" \
  --texts-cache "'"$STUDY"'/texts_cache" \
  --n-per-lang '"$N_PER_LANG"' --langs '"$LANGS"' \
  --out "'"$STUDY"'"
RC=$?
if [ $RC -ne 0 ]; then
  echo "[01] ground_domains.py failed (rc=$RC) -- is texts_cache/ populated? "
  echo "     run warm_texts_cache.py from the login node first, see README.md"
  kill $(jobs -p) 2>/dev/null; exit $RC
fi

echo "[01] running grounding-mode sweep"
python3 "'"$STUDY"'/prompts_grounded.py" \
  --gen-endpoints "'"$EP"'/gen" --judge-endpoints "'"$EP"'/judge" \
  --n-per-lang '"$N_PER_LANG"' --langs '"$LANGS"' \
  --exemplar-dir "'"$STUDY"'/domain_exemplars" \
  --out "'"$STUDY"'/outputs"
RC=$?

echo "[01] exit code: $RC"
kill $(jobs -p) 2>/dev/null || true
exit $RC
'
