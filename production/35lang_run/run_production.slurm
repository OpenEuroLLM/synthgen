#!/usr/bin/env bash
#SBATCH --job-name=oellm-production-35lang
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=112
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --partition=standard-g
#SBATCH --account=project_462001516
#SBATCH --output=production/35lang_run/logs/%j.out

# Production run: 35 languages (OpenEuroLLM target list minus English --
# github.com/OpenEuroLLM/training-data-catalogue/blob/main/languages), 100k
# survivors/language, via `synthgen topup`.
#
# 16 real nodes, role-split (not the symmetric 2+2 pattern used for
# smoke-scale testing): 12 nodes dedicated entirely to gen (4x TP=2 replicas
# each = 48 gen replicas), 4 nodes dedicated entirely to judge (4x TP=2
# replicas each = 16 judge replicas) -- the near-optimal split for
# synthgen.localized.topup's sequential gen-then-judge round structure,
# given gen needs far more raw compute per example than judge.
#
# Uses the same proven heredoc-based multi-node pattern as
# studies/localized_axes_ablation/13_concurrency_scaling/run_multinode.slurm
# (avoids fragile triple-nested quoting) and the httpx.Limits connection-pool
# fix already in synthgen/pipeline/generate.py -- validated on that same
# smoke test to scale ~5.5-6.9x across 4 replicas, well past the old
# (undiagnosed) ~100-connection ceiling.
#
# Submit from repo root: sbatch production/35lang_run/run_production.slurm
set -uo pipefail

export REPO=/scratch/project_462001516/abhasjha/synthgen
export SIF=/scratch/project_465002530/containers/vllm-openai-rocm.sif
export HF_HOME=/scratch/project_462001516/cache/huggingface/abhasjha
export RUN="$REPO/production/35lang_run"
export EP="$RUN/endpoints"          # shared across all 16 nodes via Lustre
export DATA_ROOT=/scratch/project_462001516/abhasjha/oellm_production_35lang
export GEN_MODEL=google/gemma-4-31b-it
export JUDGE_MODEL=Qwen/Qwen3.6-27B
export NUM_GEN_NODES=12             # SLURM_PROCID < this -> gen, else judge
export REPLICAS_PER_NODE=4          # 4x TP=2 = all 8 GPUs/node

rm -rf "$EP"; mkdir -p "$EP/gen" "$EP/judge" "$RUN/logs" "$DATA_ROOT"

cat > "$RUN/_per_node_launch.sh" <<'NODE_SCRIPT'
#!/usr/bin/env bash
set -uo pipefail
NODE_ID=$SLURM_PROCID
HOST=$(hostname)
if [ "$NODE_ID" -lt "$NUM_GEN_NODES" ]; then ROLE=gen; MODEL="$GEN_MODEL"
else ROLE=judge; MODEL="$JUDGE_MODEL"
fi

singularity exec \
  --bind /scratch/project_462001516:/scratch/project_462001516 \
  --bind /scratch/project_465002530:/scratch/project_465002530 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
  --env RUN="$RUN" --env EP="$EP" --env ROLE="$ROLE" --env MODEL="$MODEL" \
  --env NODE_ID="$NODE_ID" --env HOST="$HOST" \
  --env REPLICAS_PER_NODE="$REPLICAS_PER_NODE" \
  "$SIF" bash "$RUN/_in_container_launch.sh"
NODE_SCRIPT

cat > "$RUN/_in_container_launch.sh" <<'CONTAINER_SCRIPT'
#!/usr/bin/env bash
set -uo pipefail
cd "$REPO"

serve() {
  local port=$1 gpus=$2
  ROCR_VISIBLE_DEVICES=$gpus \
  vllm serve "$MODEL" --served-model-name "$ROLE" \
    --host 0.0.0.0 --port "$port" \
    --tensor-parallel-size $(echo $gpus | tr "," "\n" | wc -l) \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --enforce-eager \
    > "$RUN/logs/vllm_${ROLE}_node${NODE_ID}_port${port}.log" 2>&1 &
}

wait_ready() {
  local port=$1 replica=$2
  for t in $(seq 1 90); do
    if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
      echo "${HOST}:${port}" > "$EP/$ROLE/${ROLE}_node${NODE_ID}_${replica}.endpoint"
      echo "[ready] $ROLE replica=$replica node=$NODE_ID port=$port after ${t}0s"
      return 0
    fi
    sleep 10
  done
  echo "[WARN] $ROLE replica=$replica node=$NODE_ID port=$port never came up (see log)"
  return 1
}

# 4 replicas of TP=2, GPU pairs 0-1 2-3 4-5 6-7, ports 8001-8004
gpu_pairs=("0,1" "2,3" "4,5" "6,7")
for i in $(seq 0 $((REPLICAS_PER_NODE - 1))); do
  serve $((8001 + i)) "${gpu_pairs[$i]}"
done
for i in $(seq 0 $((REPLICAS_PER_NODE - 1))); do
  wait_ready $((8001 + i)) "$i" || true   # one flaky replica shouldn't kill the node
done

wait
CONTAINER_SCRIPT

chmod +x "$RUN/_per_node_launch.sh" "$RUN/_in_container_launch.sh"

srun --ntasks=16 --ntasks-per-node=1 bash "$RUN/_per_node_launch.sh" &
SRUN_PID=$!

# Lenient readiness check: expect 48 gen + 16 judge, proceed once at least
# ~85% of each role is up rather than requiring literal 100% -- a single
# flaky node (observed once already in the 4-node smoke test) shouldn't
# block a multi-day job. synthgen's own --gen-min-endpoints/--judge-min-
# endpoints below is the real, enforced floor; this is just a fast-fail
# check before spending time launching the CLI.
EXPECT_GEN=48; EXPECT_JUDGE=16
MIN_GEN=40; MIN_JUDGE=13
echo "[prod] waiting for endpoints (want >= $MIN_GEN/$EXPECT_GEN gen, >= $MIN_JUDGE/$EXPECT_JUDGE judge)..."
n_gen=0; n_judge=0
for t in $(seq 1 90); do
  n_gen=$(ls "$EP/gen"/*.endpoint 2>/dev/null | wc -l)
  n_judge=$(ls "$EP/judge"/*.endpoint 2>/dev/null | wc -l)
  echo "  [${t}0s elapsed] gen=$n_gen/$EXPECT_GEN judge=$n_judge/$EXPECT_JUDGE"
  if [ "$n_gen" -ge "$MIN_GEN" ] && [ "$n_judge" -ge "$MIN_JUDGE" ]; then
    break
  fi
  sleep 10
done

if [ "$n_gen" -lt "$MIN_GEN" ] || [ "$n_judge" -lt "$MIN_JUDGE" ]; then
  echo "[prod] FATAL: not enough replicas came up (gen=$n_gen/$MIN_GEN judge=$n_judge/$MIN_JUDGE required) -- see logs/vllm_*.log"
  kill $SRUN_PID 2>/dev/null || true
  exit 1
fi

# Concurrency ~256 per replica (the level the smoke test showed actually
# saturates a replica with the httpx.Limits fix in place), scaled to
# however many replicas actually came up (not the ideal-case expected count).
GEN_CONCURRENCY=$((n_gen * 256))
JUDGE_CONCURRENCY=$((n_judge * 256))
echo "[prod] gen_concurrency=$GEN_CONCURRENCY judge_concurrency=$JUDGE_CONCURRENCY"

LANGS="es fr de it pt pl nl cs ro el uk bg da et fi ga hr hu lv lt mt sk sl sv ca eu gl bs ka mk sq sr tr is no"

singularity exec \
  --bind /scratch/project_462001516:/scratch/project_462001516 \
  --bind /scratch/project_465002530:/scratch/project_465002530 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
  "$SIF" bash -c "
set -uo pipefail
pip install --quiet sentence-transformers 2>/dev/null || true
cd '$REPO'
python3 -m synthgen.cli topup \
  --root '$DATA_ROOT' \
  --target 100000 \
  --thresholds '$RUN/thresholds.json' \
  --langs $LANGS \
  --gen-backend vllm --gen-models gen --gen-endpoints-dir '$EP/gen' --gen-min-endpoints $MIN_GEN \
  --judge-backend vllm --judge-model judge --judge-endpoints-dir '$EP/judge' --judge-min-endpoints $MIN_JUDGE \
  --embedding-dedupe \
  --gen-concurrency $GEN_CONCURRENCY --judge-concurrency $JUDGE_CONCURRENCY --concurrency $JUDGE_CONCURRENCY \
  --max-rounds 30 --max-attempts 3 --retry-delay 15
"
RC=$?

echo "[prod] exit code: $RC"
kill $SRUN_PID 2>/dev/null || true
exit $RC
