#!/usr/bin/env bash
#SBATCH --job-name=oellm-judge-sup-mn5
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --time=1-00:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc532
#SBATCH --qos=acc_ehpc
#SBATCH --output=/gpfs/scratch/ehpc532/eit234648/synthgen/production/35lang_run/logs/judge_%j.out

# ============================================================================
# JUDGE-ONLY pass WITH PER-NODE SUPERVISOR (self-revive).  Same 3-step chain as
# run_judge_pass_mn5.slurm (judge-drain -> quality-filter -> dedupe), but each
# node runs a supervisor loop around its vLLM server: if the server crashes or
# hangs (the NCCL `sample_tokens`/all-gather timeout that killed replicas one by
# one in jobs 47619579 and 47627757), the supervisor frees the GPUs, relaunches
# vLLM on the SAME host:port, and re-writes the endpoint file. synthgen's pool
# re-polls every ~30s (synthgen/CLAUDE.md), so the revived replica rejoins
# automatically. A replica death becomes a ~4-min hiccup instead of permanent
# capacity loss -- because TP=4 + --disable-custom-all-reduce only SLOWED the
# cascade (deaths every ~21 min vs ~6 min) rather than stopping it, so we make
# the run survive deaths instead of trying to perfectly prevent them.
#
# Single-variable change vs run_judge_pass_mn5.slurm: ONLY the supervisor loop
# is added. TP=4 / --disable-custom-all-reduce / 32-per-replica concurrency are
# unchanged, to keep this test clean and preserve throughput (fit the 24h wall).
#
#   sbatch -N 8 synthgen/production/35lang_run/run_judge_pass_mn5_sup.slurm
# ============================================================================
set -uo pipefail

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

export REPO=/gpfs/scratch/ehpc532/eit234648/synthgen
export SIF=/gpfs/scratch/ehpc532/eit234648/vllm/vllm-v0.22.1.sif
export HF_HOME=/gpfs/scratch/ehpc532/eit234648/.hf_cache
export RUN="$REPO/production/35lang_run"
# ISOLATED from any concurrently-running run_judge_pass_mn5.slurm job: its own
# endpoints dir, its own DATA_ROOT (parallel outputs), and distinct vllm log
# names (see LOGF in the container script). This lets us run this supervised
# job alongside the limping non-supervised one without either corrupting the
# other's endpoint pool or judged_judge.jsonl. loc_gen.jsonl is symlinked
# (shared read-only input); judged_judge.jsonl is a snapshot copy so this run
# resumes from the other run's progress as of submit time.
export EP="$RUN/endpoints_sup"             # SEPARATE pool dir
export DATA_ROOT=/gpfs/scratch/ehpc532/eit234648/oellm_production_35lang_sup
export JUDGE_MODEL=/gpfs/scratch/ehpc532/eit234648/models/Qwen/Qwen3.6-27B
export REPLICAS_PER_NODE=1                  # 1x TP=4 = all 4 GPUs/node

NNODES=${SLURM_NNODES:-1}
echo "[judge-sup] nodes=$NNODES replicas/node=$REPLICAS_PER_NODE (all judge, supervised)"

module load singularity/4.1.5

rm -rf "$EP"; mkdir -p "$EP/judge" "$RUN/logs" "$DATA_ROOT/outputs"

# ---- seed this isolated DATA_ROOT from the MAIN run's LATEST progress --------
# Runs at job START (i.e. AFTER the ~2h queue wait), so it captures the freshest
# judged_judge.jsonl the concurrent/limping main run has produced by then -- not
# a stale snapshot taken at submit time. loc_gen.jsonl is symlinked (23.8GB,
# read-only shared input, never copied). The judged copy is seeded ONLY IF
# ABSENT, so a resubmit of THIS job keeps its own (more complete) progress
# instead of being overwritten by main's.
export MAIN_ROOT=/gpfs/scratch/ehpc532/eit234648/oellm_production_35lang
ln -sfn "$MAIN_ROOT/outputs/loc_gen.jsonl" "$DATA_ROOT/outputs/loc_gen.jsonl"
if [ ! -f "$DATA_ROOT/outputs/judged_judge.jsonl" ]; then
  echo "[judge-sup] seeding judged_judge.jsonl from MAIN_ROOT ($(wc -l < "$MAIN_ROOT/outputs/judged_judge.jsonl" 2>/dev/null) lines) at $(date +%T)"
  cp -f "$MAIN_ROOT/outputs/judged_judge.jsonl" "$DATA_ROOT/outputs/judged_judge.jsonl"
else
  echo "[judge-sup] judged_judge.jsonl already present in DATA_ROOT ($(wc -l < "$DATA_ROOT/outputs/judged_judge.jsonl") lines) -- keeping it (resume)"
fi

# -------- per-node launcher (distinct filename so it can't clobber a running
#          run_judge_pass_mn5.slurm job's helper scripts) ---------------------
cat > "$RUN/_per_node_judge_sup_mn5.sh" <<'NODE_SCRIPT'
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
  "$SIF" bash "$RUN/_in_container_judge_sup_mn5.sh"
NODE_SCRIPT

cat > "$RUN/_in_container_judge_sup_mn5.sh" <<'CONTAINER_SCRIPT'
#!/usr/bin/env bash
set -uo pipefail
cd "$REPO"

PORT=8001
GPUS="0,1,2,3"
EPFILE="$EP/$ROLE/${ROLE}_node${NODE_ID}_0.endpoint"
LOGF="$RUN/logs/vllm_sup_${ROLE}_node${NODE_ID}_port${PORT}.log"

serve() {
  # fresh log each (re)launch (truncate) so death-marker grep can't see a stale one
  CUDA_VISIBLE_DEVICES=$GPUS \
  vllm serve "$MODEL" --served-model-name "$ROLE" \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size 4 \
    --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --disable-custom-all-reduce \
    > "$LOGF" 2>&1 &
  SERVE_PID=$!
}

wait_ready() {
  for t in $(seq 1 90); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "${HOST}:${PORT}" > "$EPFILE"
      echo "[ready] node=$NODE_ID port=$PORT after ${t}0s"
      return 0
    fi
    sleep 10
  done
  echo "[WARN] node=$NODE_ID port=$PORT never came up (see log)"
  return 1
}

# Free the GPUs completely before a relaunch: kill the vllm CLI, then any
# process still holding the visible GPUs, then wait for VRAM to drain. Only this
# replica uses this node's GPUs (1 replica/node), so this is safe per-node.
free_gpus() {
  pkill -9 -f "vllm serve" 2>/dev/null || true
  sleep 5
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true
  done
  for t in $(seq 1 24); do   # up to ~2 min for VRAM to release
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
    [ "${used:-9999}" -lt 2000 ] && break
    sleep 5
  done
}

revive() {
  local why=$1
  RESTARTS=$((RESTARTS+1))
  echo "[supervisor] node=$NODE_ID REVIVE #$RESTARTS ($why) at $(date +%T)"
  rm -f "$EPFILE"                       # evict stale endpoint (pool drops it in ~30s)
  free_gpus
  serve
  wait_ready || true
}

# ---- initial launch ----
RESTARTS=0
serve
wait_ready || true

# ---- supervise forever (the main sbatch kills this task when judging ends) ----
HFAIL=0
while true; do
  sleep 20
  # (a) process exited?
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    revive "process exited"; HFAIL=0; continue
  fi
  # (b) engine death marker in the (truncated-per-launch) log?
  if grep -qE "EngineDeadError|RPC call to sample_tokens timed out|collective operation timeout" "$LOGF" 2>/dev/null; then
    revive "engine-dead marker in log"; HFAIL=0; continue
  fi
  # (c) health endpoint unreachable for 3 consecutive checks?
  if curl -sf -m 10 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    HFAIL=0
  else
    HFAIL=$((HFAIL+1))
    echo "[supervisor] node=$NODE_ID health-check miss $HFAIL/3 at $(date +%T)"
    [ "$HFAIL" -ge 3 ] && { revive "health-check failed x3"; HFAIL=0; }
  fi
done
CONTAINER_SCRIPT

chmod +x "$RUN/_per_node_judge_sup_mn5.sh" "$RUN/_in_container_judge_sup_mn5.sh"

srun --ntasks="$NNODES" --ntasks-per-node=1 bash "$RUN/_per_node_judge_sup_mn5.sh" &
SRUN_PID=$!

EXPECT_JUDGE=$(( NNODES * REPLICAS_PER_NODE ))
MIN_JUDGE=$(( EXPECT_JUDGE * 85 / 100 )); [ "$MIN_JUDGE" -lt 1 ] && MIN_JUDGE=1
echo "[judge-sup] waiting for endpoints (want >= $MIN_JUDGE/$EXPECT_JUDGE judge)..."
n_judge=0
for t in $(seq 1 90); do
  n_judge=$(ls "$EP/judge"/*.endpoint 2>/dev/null | wc -l)
  echo "  [${t}0s elapsed] judge=$n_judge/$EXPECT_JUDGE"
  if [ "$n_judge" -ge "$MIN_JUDGE" ]; then break; fi
  sleep 10
done

if [ "$n_judge" -lt "$MIN_JUDGE" ]; then
  echo "[judge-sup] FATAL: not enough judge replicas came up (judge=$n_judge/$MIN_JUDGE) -- see logs/vllm_*.log"
  kill $SRUN_PID 2>/dev/null || true
  exit 1
fi

# 32/replica (unchanged). Supervisor keeps the live count near EXPECT_JUDGE, so
# this is per the healthy-replica assumption; min-endpoints lets the client ride
# through the ~4-min revive windows (pool re-polls, dead endpoint evicted).
JUDGE_CONCURRENCY=$((n_judge * 32))
echo "[judge-sup] judge replicas up=$n_judge  concurrency=$JUDGE_CONCURRENCY (32/replica, TP=4, supervised)"

export SINGULARITYENV_CC=gcc SINGULARITYENV_CXX=g++
singularity exec --nv \
  --bind /gpfs/scratch/ehpc532 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
  "$SIF" bash -c "
set -uo pipefail
cd '$REPO'

echo '[judge-sup] === step 1/3: drain unjudged backlog (resumable) ==='
python3 -m synthgen.cli generate \
  --root '$DATA_ROOT' \
  --backend vllm --mode judge --models judge \
  --in '$DATA_ROOT/outputs/loc_gen.jsonl' \
  --endpoints-dir '$EP/judge' --min-endpoints $MIN_JUDGE \
  --concurrency $JUDGE_CONCURRENCY \
  --max-attempts 3 --retry-delay 15
RC1=\$?
echo \"[judge-sup] step 1 exit=\$RC1\"
[ \$RC1 -ne 0 ] && exit \$RC1

echo '[judge-sup] === step 2/3: quality-filter (score >= thresholds) ==='
python3 -m synthgen.cli quality-filter \
  --gen '$DATA_ROOT/outputs/loc_gen.jsonl' \
  --judged '$DATA_ROOT/outputs/judged_judge.jsonl' \
  --thresholds '$RUN/thresholds.json' \
  --output '$DATA_ROOT/outputs/loc_full.kept.jsonl' \
  --report '$RUN/logs/quality_filter_${SLURM_JOB_ID}.json'
RC2=\$?
echo \"[judge-sup] step 2 exit=\$RC2\"
[ \$RC2 -ne 0 ] && exit \$RC2

echo '[judge-sup] === step 3/3: exact-string dedupe -> loc_full.dedup.jsonl ==='
python3 -m synthgen.cli dedupe \
  --input '$DATA_ROOT/outputs/loc_full.kept.jsonl' \
  --output '$DATA_ROOT/outputs/loc_full.dedup.jsonl' \
  --report '$RUN/logs/dedupe_${SLURM_JOB_ID}.json'
RC3=\$?
echo \"[judge-sup] step 3 exit=\$RC3\"
exit \$RC3
"
RC=$?

echo "[judge-sup] pipeline exit code: $RC"
echo "[judge-sup] outputs:"; ls -la "$DATA_ROOT/outputs/" 2>/dev/null
kill $SRUN_PID 2>/dev/null || true
exit $RC
