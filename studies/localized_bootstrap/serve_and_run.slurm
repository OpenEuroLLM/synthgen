#!/usr/bin/env bash
#SBATCH --job-name=loc-bootstrap
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --partition=dev-g
#SBATCH --account=project_462001516
#SBATCH --output=studies/localized_bootstrap/logs/slurm-%j.out

# Judge-validation study: serve gemma-4-31b-it (generator) and Qwen3.6-27B
# (judge) on 4 GPUs of one LUMI-G node, then score good vs bad examples.
# Submit from the synthgen repo root:  sbatch studies/localized_bootstrap/serve_and_run.slurm
set -uo pipefail

REPO=/scratch/project_462001516/abhasjha/synthgen
SIF=/scratch/project_465002530/containers/vllm-openai-rocm.sif
export HF_HOME=/scratch/project_462001516/cache/huggingface/abhasjha
STUDY="$REPO/studies/localized_bootstrap"
EP="$STUDY/endpoints"
N_GOOD=${N_GOOD:-10}
N_MEDIUM=${N_MEDIUM:-10}
N_BAD=${N_BAD:-10}
# 11 languages x 30 examples = 330, vs. the 4-language/120-example runs this
# defaulted for. Raise concurrency so the run still fits the dev-g 30-min cap.
CONCURRENCY=${CONCURRENCY:-16}
# Stall watchdog: abort the job if generation makes no progress for this long.
# Pointless on dev-g (the 30-min wall fires first), so disable it there with
# WATCHDOG=0. Default on for long production runs; STALL_SECS tunes the timeout.
WATCHDOG=${WATCHDOG:-1}
STALL_SECS=${STALL_SECS:-1800}

GEN_MODEL=google/gemma-4-31b-it
JUDGE_MODEL=Qwen/Qwen3.6-27B

rm -rf "$EP"; mkdir -p "$EP/gen" "$EP/judge" "$STUDY/logs" "$STUDY/outputs"
HOST=$(hostname)

singularity exec \
  --bind /scratch/project_462001516:/scratch/project_462001516 \
  --bind /scratch/project_465002530:/scratch/project_465002530 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 \
  --env PYTHONPATH="$REPO" --env STALL_SECS="$STALL_SECS" --env WATCHDOG="$WATCHDOG" \
  "$SIF" bash -c '
set -uo pipefail
cd "'"$REPO"'"

serve () {  # name port "gpus" model
  local name=$1 port=$2 gpus=$3 model=$4
  ROCR_VISIBLE_DEVICES=$gpus \
  vllm serve "$model" --served-model-name "$name" \
    --host 0.0.0.0 --port "$port" \
    --tensor-parallel-size $(echo $gpus | tr "," "\n" | wc -l) \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --enforce-eager \
    > "'"$STUDY"'/logs/vllm_${name}.log" 2>&1 &
}

# Publish the endpoint once the server answers; fail fast if it never does
# (avoids the 20-min hang when TP workers crash on init).
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

# Stagger: bring up the generator fully BEFORE the judge, so the two TP=2
# multiproc groups do not race on shared memory during init.
serve gen 8001 0,1 "'"$GEN_MODEL"'"
wait_ready gen 8001 || { kill $(jobs -p) 2>/dev/null; exit 1; }
serve judge 8002 2,3 "'"$JUDGE_MODEL"'"
wait_ready judge 8002 || { kill $(jobs -p) 2>/dev/null; exit 1; }

# Stall watchdog: the driver rewrites outputs/.heartbeat as each example finishes.
# If its mtime freezes for STALL_SECS, generation has hung (dead server, stuck
# loop) — kill the driver + servers so the job ends instead of idling to wall.
# Per-request timeouts (backend: 10 min x5 retries) catch single hangs; this
# catches the whole-run stall the retries cannot.
HB="'"$STUDY"'/outputs/.heartbeat"
rm -f "$HB"; mkdir -p "'"$STUDY"'/outputs"
WD=""
if [ "${WATCHDOG:-1}" = "1" ]; then
  (
    while sleep 60; do
      [ -f "$HB" ] || continue                     # not armed until driver starts
      age=$(( $(date +%s) - $(stat -c %Y "$HB") ))
      if [ "$age" -gt "${STALL_SECS:-1800}" ]; then
        echo "[WATCHDOG] no generation progress for ${age}s (> ${STALL_SECS:-1800}s) -- aborting job"
        pkill -f run_judge_validation.py 2>/dev/null
        pkill -f "vllm serve" 2>/dev/null
        break
      fi
    done
  ) &
  WD=$!
else
  echo "[watchdog] disabled (WATCHDOG=0)"
fi

python3 studies/localized_bootstrap/run_judge_validation.py \
  --gen-endpoints   studies/localized_bootstrap/endpoints/gen \
  --judge-endpoints studies/localized_bootstrap/endpoints/judge \
  --gen-model gen --judge-model judge \
  --n-good '"$N_GOOD"' --n-medium '"$N_MEDIUM"' --n-bad '"$N_BAD"' \
  --concurrency '"$CONCURRENCY"' \
  --out studies/localized_bootstrap/outputs &
PY=$!
wait $PY; RC=$?
[ -n "$WD" ] && kill $WD 2>/dev/null

echo "[done] study finished (driver rc=$RC); shutting down servers"
kill $(jobs -p) 2>/dev/null || true
'
