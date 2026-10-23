#!/usr/bin/env bash
#SBATCH --job-name=oellm-judge-pass-mn5
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --time=1-00:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc532
#SBATCH --qos=acc_ehpc
#SBATCH --output=production/35lang_run/logs/judge_%j.out

# ============================================================================
# JUDGE-ONLY pass (option B): complete ONE full pipeline loop on the already-
# staged data WITHOUT generating a single new row. Does exactly:
#   1. generate --mode judge   -- drain the ~3.57M-row unjudged backlog
#                                 (resumable: skips the ~4.0M already judged)
#   2. quality-filter          -- keep rows scoring >= thresholds.json
#   3. dedupe                  -- exact-string dedupe -> loc_full.dedup.jsonl
#                                 (the first complete pipeline output ever)
#
# This deliberately does NOT run topup: topup's round loop would, after the
# judge drain, generate 1.1*deficit (~384k rows, heavily skewed to the low-
# pass-rate langs ga/eu/mt/et/lv) and judge those too -- churn that can't
# close those gaps in one round anyway. New generation is a separate later
# round. So this is a pure judge->filter->dedupe chain, all nodes = judge.
#
# SIZE AT SUBMIT TIME:  sbatch -N 8 production/35lang_run/run_judge_pass_mn5.slurm
#   8 nodes x 1 replica/node (TP=4) = 8 judge replicas. (Was 2x TP=2 = 16; the
#   two-PG-per-node layout hung NCCL collectives, see REPLICAS_PER_NODE note.)
#   Backlog ~2.9M rows; id-checkpointed so a resubmit resumes if it runs over.
#
# MN5 port specifics are identical to run_production_mn5.slurm (see that file).
# ============================================================================
set -uo pipefail

# MN5 login shell leaks HTTP_PROXY=localhost:8888; sbatch --export=ALL +
# singularity pass-through would hijack the localhost curl health-checks and
# synthgen's httpx calls to the vLLM endpoints. Compute nodes have no internet
# and we run fully offline (HF_HUB_OFFLINE=1), so drop them.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

export REPO=/gpfs/scratch/ehpc532/eit234648/synthgen
export SIF=/gpfs/scratch/ehpc532/eit234648/vllm/vllm-v0.22.1.sif
export HF_HOME=/gpfs/scratch/ehpc532/eit234648/.hf_cache
export RUN="$REPO/production/35lang_run"
export EP="$RUN/endpoints"                 # shared across all nodes via GPFS
export DATA_ROOT=/gpfs/scratch/ehpc532/eit234648/oellm_production_35lang
export JUDGE_MODEL=/gpfs/scratch/ehpc532/eit234648/models/Qwen/Qwen3.6-27B
export REPLICAS_PER_NODE=1                  # 1x TP=4 = all 4 GPUs/node
# NOTE (2026-10-20): switched from 2x TP=2 to 1x TP=4. The 2-replicas-per-node
# layout ran TWO independent NCCL process groups on each node, which hung an
# _ALLGATHER_BASE collective under load (job 47619579: 7/16 replicas died one
# by one, "RPC call to sample_tokens timed out" -> NCCL watchdog 600s timeout).
# One TP=4 PG per node removes that intra-node contention. Also added
# --disable-custom-all-reduce to the serve() command as extra insurance.

NNODES=${SLURM_NNODES:-1}
echo "[judge] nodes=$NNODES replicas/node=$REPLICAS_PER_NODE (all judge)"

module load singularity/4.1.5

rm -rf "$EP"; mkdir -p "$EP/judge" "$RUN/logs" "$DATA_ROOT"

# -------- per-node: launch 2 judge replicas (TP=2) on the 4 GPUs -------------
cat > "$RUN/_per_node_judge_mn5.sh" <<'NODE_SCRIPT'
#!/usr/bin/env bash
set -uo pipefail
NODE_ID=$SLURM_PROCID
HOST=$(hostname)
ROLE=judge
MODEL="$JUDGE_MODEL"

export SINGULARITYENV_CC=gcc SINGULARITYENV_CXX=g++
export SINGULARITYENV_VLLM_DISABLE_COMPILE_CACHE=1

singularity exec --nv \
  --bind /gpfs/scratch/ehpc532 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
  --env RUN="$RUN" --env EP="$EP" --env ROLE="$ROLE" --env MODEL="$MODEL" \
  --env NODE_ID="$NODE_ID" --env HOST="$HOST" \
  --env REPLICAS_PER_NODE="$REPLICAS_PER_NODE" \
  "$SIF" bash "$RUN/_in_container_judge_mn5.sh"
NODE_SCRIPT

cat > "$RUN/_in_container_judge_mn5.sh" <<'CONTAINER_SCRIPT'
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
    --disable-custom-all-reduce \
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

# 1 replica per node using all 4 GPUs (TP=4). gpu_pairs has one entry now;
# TP size is derived from the comma-count in serve(), so this gives TP=4.
gpu_pairs=("0,1,2,3")
for i in $(seq 0 $((REPLICAS_PER_NODE - 1))); do
  serve $((8001 + i)) "${gpu_pairs[$i]}"
done
for i in $(seq 0 $((REPLICAS_PER_NODE - 1))); do
  wait_ready $((8001 + i)) "$i" || true
done

wait
CONTAINER_SCRIPT

chmod +x "$RUN/_per_node_judge_mn5.sh" "$RUN/_in_container_judge_mn5.sh"

srun --ntasks="$NNODES" --ntasks-per-node=1 bash "$RUN/_per_node_judge_mn5.sh" &
SRUN_PID=$!

EXPECT_JUDGE=$(( NNODES * REPLICAS_PER_NODE ))
MIN_JUDGE=$(( EXPECT_JUDGE * 85 / 100 )); [ "$MIN_JUDGE" -lt 1 ] && MIN_JUDGE=1
echo "[judge] waiting for endpoints (want >= $MIN_JUDGE/$EXPECT_JUDGE judge)..."
n_judge=0
for t in $(seq 1 90); do
  n_judge=$(ls "$EP/judge"/*.endpoint 2>/dev/null | wc -l)
  echo "  [${t}0s elapsed] judge=$n_judge/$EXPECT_JUDGE"
  if [ "$n_judge" -ge "$MIN_JUDGE" ]; then break; fi
  sleep 10
done

if [ "$n_judge" -lt "$MIN_JUDGE" ]; then
  echo "[judge] FATAL: not enough judge replicas came up (judge=$n_judge/$MIN_JUDGE) -- see logs/vllm_*.log"
  kill $SRUN_PID 2>/dev/null || true
  exit 1
fi

# Concurrency: 32/replica. With TP=4 (1 replica/node) an 8-node run has 8
# replicas -> 8*32 = 256 total. That is well under the 448-512 total that the
# old 2x-TP=2 layout was running when its NCCL collectives hung (job 47619579),
# and each replica now spans 4 GPUs so 32 concurrent/replica is comfortable.
# (Earlier total-concurrency reference points, same orchestrator/client path:
#  128 total clean @28.5 req/s; 256 "degraded" and 512 hung -- but those were on
#  the 2-PG-per-node layout, so the degradation was likely the NCCL contention
#  we just removed, not pure client concurrency.) Still WATCH the first ~15 min:
# the previous hang first appeared ~7 min / ~row 6000 in.
JUDGE_CONCURRENCY=$((n_judge * 32))
echo "[judge] judge replicas up=$n_judge  concurrency=$JUDGE_CONCURRENCY (32/replica, TP=4)"

LANGS="es fr de it pt pl nl cs ro el uk bg da et fi ga hr hu lv lt mt sk sl sv ca eu gl bs ka mk sq sr tr is no"

export SINGULARITYENV_CC=gcc SINGULARITYENV_CXX=g++
singularity exec --nv \
  --bind /gpfs/scratch/ehpc532 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
  "$SIF" bash -c "
set -uo pipefail
cd '$REPO'

echo '[judge] === step 1/3: drain unjudged backlog (resumable) ==='
python3 -m synthgen.cli generate \
  --root '$DATA_ROOT' \
  --backend vllm --mode judge --models judge \
  --in '$DATA_ROOT/outputs/loc_gen.jsonl' \
  --endpoints-dir '$EP/judge' --min-endpoints $MIN_JUDGE \
  --concurrency $JUDGE_CONCURRENCY \
  --max-attempts 3 --retry-delay 15
RC1=\$?
echo \"[judge] step 1 exit=\$RC1\"
[ \$RC1 -ne 0 ] && exit \$RC1

echo '[judge] === step 2/3: quality-filter (score >= thresholds) ==='
python3 -m synthgen.cli quality-filter \
  --gen '$DATA_ROOT/outputs/loc_gen.jsonl' \
  --judged '$DATA_ROOT/outputs/judged_judge.jsonl' \
  --thresholds '$RUN/thresholds.json' \
  --output '$DATA_ROOT/outputs/loc_full.kept.jsonl' \
  --report '$RUN/logs/quality_filter_${SLURM_JOB_ID}.json'
RC2=\$?
echo \"[judge] step 2 exit=\$RC2\"
[ \$RC2 -ne 0 ] && exit \$RC2

echo '[judge] === step 3/3: exact-string dedupe -> loc_full.dedup.jsonl ==='
python3 -m synthgen.cli dedupe \
  --input '$DATA_ROOT/outputs/loc_full.kept.jsonl' \
  --output '$DATA_ROOT/outputs/loc_full.dedup.jsonl' \
  --report '$RUN/logs/dedupe_${SLURM_JOB_ID}.json'
RC3=\$?
echo \"[judge] step 3 exit=\$RC3\"
exit \$RC3
"
RC=$?

echo "[judge] pipeline exit code: $RC"
echo "[judge] outputs:"; ls -la "$DATA_ROOT/outputs/" 2>/dev/null
kill $SRUN_PID 2>/dev/null || true
exit $RC
