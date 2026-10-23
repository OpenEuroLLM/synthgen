#!/usr/bin/env bash
#SBATCH --job-name=oellm-prod-35lang-mn5
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --time=2-00:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc532
#SBATCH --qos=acc_ehpc
#SBATCH --output=production/35lang_run/logs/mn5_%j.out

# ============================================================================
# MN5 (NVIDIA/CUDA) port of run_production.slurm (originally LUMI/ROCm).
# Continues the 35-language `synthgen topup` run from the state staged under
# DATA_ROOT (loc_gen.jsonl + judged_judge.jsonl) -- id-checkpointed, so only
# unjudged/missing rows get (re)generated or (re)judged.
#
# SIZE THE RUN AT SUBMIT TIME WITH -N (do NOT hardcode --nodes here):
#   sbatch -N 32 production/35lang_run/run_production_mn5.slurm
# NUM_GEN_NODES defaults to 3/4 of the nodes (the LUMI 12:4 = 3:1 gen:judge
# split, which carries over because MN5 runs the SAME 2 replicas/node for both
# roles -- the optimal split is set by the two models' relative throughput, not
# the hardware). Override with: --export=ALL,NUM_GEN_NODES=<n>
#
# MN5 vs LUMI differences baked in below (see production/MIGRATE_TO_NVIDIA.md):
#   - singularity exec --nv (CUDA), CUDA_VISIBLE_DEVICES (not ROCR_...)
#   - acc nodes are 4x H100-64GB -> REPLICAS_PER_NODE=2 (TP=2, pairs 0,1 / 2,3),
#     half LUMI's 8-GCD density
#   - CC=icc leak fix (SINGULARITYENV_CC=gcc) + VLLM_DISABLE_COMPILE_CACHE=1
#   - paths/SIF/partition/account/qos for ehpc532
#   - NO runtime `pip install` (compute nodes have no internet); embedding-dedupe
#     is off (exact-string dedupe only), same as the LUMI production call.
# ============================================================================
set -uo pipefail

# MN5 login shell sets HTTP_PROXY=localhost:8888 for the download tunnel;
# sbatch --export=ALL + singularity's default env pass-through would leak it
# into the job and hijack the localhost curl health-checks and synthgen's
# httpx calls to the vLLM endpoints (see MN5.md gotcha #3). Compute nodes have
# no internet and we run fully offline (HF_HUB_OFFLINE=1), so drop them.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

export REPO=/gpfs/scratch/ehpc532/eit234648/synthgen
export SIF=/gpfs/scratch/ehpc532/eit234648/vllm/vllm-v0.22.1.sif
export HF_HOME=/gpfs/scratch/ehpc532/eit234648/.hf_cache
export RUN="$REPO/production/35lang_run"
export EP="$RUN/endpoints"          # shared across all nodes via GPFS
export DATA_ROOT=/gpfs/scratch/ehpc532/eit234648/oellm_production_35lang
export GEN_MODEL=/gpfs/scratch/ehpc532/models/google/gemma-4-31B-it
export JUDGE_MODEL=/gpfs/scratch/ehpc532/eit234648/models/Qwen/Qwen3.6-27B
export REPLICAS_PER_NODE=2           # 2x TP=2 = all 4 GPUs/node

NNODES=${SLURM_NNODES:-1}
export NUM_GEN_NODES=${NUM_GEN_NODES:-$(( (NNODES * 3 + 3) / 4 ))}   # ~3/4, ceil
echo "[prod] nodes=$NNODES gen_nodes=$NUM_GEN_NODES judge_nodes=$(( NNODES - NUM_GEN_NODES )) replicas/node=$REPLICAS_PER_NODE"

module load singularity/4.1.5

rm -rf "$EP"; mkdir -p "$EP/gen" "$EP/judge" "$RUN/logs" "$DATA_ROOT"

cat > "$RUN/_per_node_launch_mn5.sh" <<'NODE_SCRIPT'
#!/usr/bin/env bash
set -uo pipefail
NODE_ID=$SLURM_PROCID
HOST=$(hostname)
if [ "$NODE_ID" -lt "$NUM_GEN_NODES" ]; then ROLE=gen; MODEL="$GEN_MODEL"
else ROLE=judge; MODEL="$JUDGE_MODEL"
fi

# MN5 Intel modules leak CC=icc into the container; torch.compile needs a real
# compiler. Image ships gcc 11. Disable the shared compile cache so 64 replicas
# starting at once don't race on it.
export SINGULARITYENV_CC=gcc SINGULARITYENV_CXX=g++
export SINGULARITYENV_VLLM_DISABLE_COMPILE_CACHE=1

singularity exec --nv \
  --bind /gpfs/scratch/ehpc532 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
  --env RUN="$RUN" --env EP="$EP" --env ROLE="$ROLE" --env MODEL="$MODEL" \
  --env NODE_ID="$NODE_ID" --env HOST="$HOST" \
  --env REPLICAS_PER_NODE="$REPLICAS_PER_NODE" \
  "$SIF" bash "$RUN/_in_container_launch_mn5.sh"
NODE_SCRIPT

cat > "$RUN/_in_container_launch_mn5.sh" <<'CONTAINER_SCRIPT'
#!/usr/bin/env bash
set -uo pipefail
cd "$REPO"

serve() {
  local port=$1 gpus=$2
  CUDA_VISIBLE_DEVICES=$gpus \
  vllm serve "$MODEL" --served-model-name "$ROLE" \
    --host 0.0.0.0 --port "$port" \
    --tensor-parallel-size $(echo $gpus | tr "," "\n" | wc -l) \
    --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --trust-remote-code \
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

# 2 replicas of TP=2 on a 4-GPU node: GPU pairs 0,1 and 2,3; ports 8001-8002.
gpu_pairs=("0,1" "2,3")
for i in $(seq 0 $((REPLICAS_PER_NODE - 1))); do
  serve $((8001 + i)) "${gpu_pairs[$i]}"
done
for i in $(seq 0 $((REPLICAS_PER_NODE - 1))); do
  wait_ready $((8001 + i)) "$i" || true
done

wait
CONTAINER_SCRIPT

chmod +x "$RUN/_per_node_launch_mn5.sh" "$RUN/_in_container_launch_mn5.sh"

srun --ntasks="$NNODES" --ntasks-per-node=1 bash "$RUN/_per_node_launch_mn5.sh" &
SRUN_PID=$!

# Lenient readiness floor (one flaky node shouldn't block a multi-day job);
# synthgen's own --gen-min-endpoints/--judge-min-endpoints is the enforced floor.
EXPECT_GEN=$(( NUM_GEN_NODES * REPLICAS_PER_NODE ))
EXPECT_JUDGE=$(( (NNODES - NUM_GEN_NODES) * REPLICAS_PER_NODE ))
MIN_GEN=$(( EXPECT_GEN * 85 / 100 )); [ "$MIN_GEN" -lt 1 ] && MIN_GEN=1
MIN_JUDGE=$(( EXPECT_JUDGE * 85 / 100 )); [ "$MIN_JUDGE" -lt 1 ] && MIN_JUDGE=1
echo "[prod] waiting for endpoints (want >= $MIN_GEN/$EXPECT_GEN gen, >= $MIN_JUDGE/$EXPECT_JUDGE judge)..."
n_gen=0; n_judge=0
for t in $(seq 1 90); do
  n_gen=$(ls "$EP/gen"/*.endpoint 2>/dev/null | wc -l)
  n_judge=$(ls "$EP/judge"/*.endpoint 2>/dev/null | wc -l)
  echo "  [${t}0s elapsed] gen=$n_gen/$EXPECT_GEN judge=$n_judge/$EXPECT_JUDGE"
  if [ "$n_gen" -ge "$MIN_GEN" ] && [ "$n_judge" -ge "$MIN_JUDGE" ]; then break; fi
  sleep 10
done

if [ "$n_gen" -lt "$MIN_GEN" ] || [ "$n_judge" -lt "$MIN_JUDGE" ]; then
  echo "[prod] FATAL: not enough replicas came up (gen=$n_gen/$MIN_GEN judge=$n_judge/$MIN_JUDGE required) -- see logs/vllm_*.log"
  kill $SRUN_PID 2>/dev/null || true
  exit 1
fi

# ~256 concurrent requests/replica was LUMI's saturation point (MI250). It is
# NOT portable -- re-validate on H100 with the concurrency-scaling bench before
# trusting it for a long run. Scaled to replicas that actually came up.
GEN_CONCURRENCY=$((n_gen * 256))
JUDGE_CONCURRENCY=$((n_judge * 256))
echo "[prod] gen_concurrency=$GEN_CONCURRENCY judge_concurrency=$JUDGE_CONCURRENCY"

LANGS="es fr de it pt pl nl cs ro el uk bg da et fi ga hr hu lv lt mt sk sl sv ca eu gl bs ka mk sq sr tr is no"

export SINGULARITYENV_CC=gcc SINGULARITYENV_CXX=g++
singularity exec --nv \
  --bind /gpfs/scratch/ehpc532 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
  "$SIF" bash -c "
set -uo pipefail
cd '$REPO'
python3 -m synthgen.cli topup \
  --root '$DATA_ROOT' \
  --target 100000 \
  --thresholds '$RUN/thresholds.json' \
  --langs $LANGS \
  --gen-backend vllm --gen-models gen --gen-endpoints-dir '$EP/gen' --gen-min-endpoints $MIN_GEN \
  --judge-backend vllm --judge-model judge --judge-endpoints-dir '$EP/judge' --judge-min-endpoints $MIN_JUDGE \
  --gen-concurrency $GEN_CONCURRENCY --judge-concurrency $JUDGE_CONCURRENCY --concurrency $JUDGE_CONCURRENCY \
  --max-rounds 1 --max-attempts 3 --retry-delay 15 \
  --max-overgen-factor 1.1
"
# GOAL OF THIS RUN: complete ONE full pass of the whole pipeline on the staged
# data -- judge the ~3.57M-row unjudged backlog, then run quality_filter +
# dedupe to produce the first-ever loc_full.dedup.jsonl -- NOT to chase the
# 100k/lang target. New generation is a hindrance here (a high overgen factor
# already over-produced ~300k wasted examples), so:
#   --max-rounds 1        : exactly one loop (judge backlog -> filter -> dedupe
#                           -> minimal gen -> judge new -> build final output).
#   --max-overgen-factor 1.1 : pin every deficit lang to the clamp FLOOR
#                           (factor = min(cap, max(1.1, 1/rate)); cap 1.1 => 1.1
#                           for all), i.e. the least new generation this code
#                           path allows -- it cannot reach zero via this flag.
RC=$?

echo "[prod] exit code: $RC"
kill $SRUN_PID 2>/dev/null || true
exit $RC
