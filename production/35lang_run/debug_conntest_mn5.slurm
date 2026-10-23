#!/usr/bin/env bash
#SBATCH --job-name=oellm-judge-conntest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc532
#SBATCH --qos=acc_debug
#SBATCH --output=production/35lang_run/logs/conntest_%j.out

# ============================================================================
# Debug repro for the judge-pass connection failures (job 47549930: ~96% of
# judge requests failed with "All connection attempts failed", servers healthy).
#
# Isolates whether the failure scales with CLIENT CONCURRENCY on MN5's
# multi-node topology -- something the 1-node (all-localhost) smoke never
# exercised. 2 nodes x 2 judge replicas = 4 endpoints (2 local + 2 remote to
# the orchestrator). All work goes to THROWAWAY data roots; the production
# outputs are never touched.
#
# Step A: curl-probe each endpoint from the orchestrator (no load).
# Step B: judge the SAME small sample at concurrency 64 / 256 / 1024
#         (= 16 / 64 / 256 per replica), each into its own root, and report
#         the error rate per concurrency.
#
#   sbatch production/35lang_run/debug_conntest_mn5.slurm    (ONLY on user's say-so)
# ============================================================================
set -uo pipefail

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

export REPO=/gpfs/scratch/ehpc532/eit234648/synthgen
export SIF=/gpfs/scratch/ehpc532/eit234648/vllm/vllm-v0.22.1.sif
export HF_HOME=/gpfs/scratch/ehpc532/eit234648/.hf_cache
export RUN="$REPO/production/35lang_run"
export EP="$RUN/endpoints_conntest"
export JUDGE_MODEL=/gpfs/scratch/ehpc532/eit234648/models/Qwen/Qwen3.6-27B
export REPLICAS_PER_NODE=2
export SAMPLE_SRC=/gpfs/scratch/ehpc532/eit234648/oellm_production_35lang/outputs/loc_gen.jsonl
export SAMPLE_N=5000
export DBG_BASE="$RUN/_conntest_data"       # throwaway

NNODES=${SLURM_NNODES:-2}
echo "[conntest] nodes=$NNODES replicas/node=$REPLICAS_PER_NODE"

module load singularity/4.1.5
rm -rf "$EP" "$DBG_BASE"; mkdir -p "$EP/judge" "$RUN/logs" "$DBG_BASE"

# -------- per-node judge servers (reuse the prod launch pattern) -------------
cat > "$RUN/_per_node_conntest.sh" <<'NODE_SCRIPT'
#!/usr/bin/env bash
set -uo pipefail
NODE_ID=$SLURM_PROCID
HOST=$(hostname)
ROLE=judge; MODEL="$JUDGE_MODEL"
export SINGULARITYENV_CC=gcc SINGULARITYENV_CXX=g++
export SINGULARITYENV_VLLM_DISABLE_COMPILE_CACHE=1
singularity exec --nv --bind /gpfs/scratch/ehpc532 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
  --env RUN="$RUN" --env EP="$EP" --env ROLE="$ROLE" --env MODEL="$MODEL" \
  --env NODE_ID="$NODE_ID" --env HOST="$HOST" --env REPLICAS_PER_NODE="$REPLICAS_PER_NODE" \
  "$SIF" bash "$RUN/_in_container_conntest.sh"
NODE_SCRIPT

cat > "$RUN/_in_container_conntest.sh" <<'CONTAINER_SCRIPT'
#!/usr/bin/env bash
set -uo pipefail
cd "$REPO"
serve() {
  local port=$1 gpus=$2
  CUDA_VISIBLE_DEVICES=$gpus vllm serve "$MODEL" --served-model-name "$ROLE" \
    --host 0.0.0.0 --port "$port" --tensor-parallel-size $(echo $gpus | tr "," "\n" | wc -l) \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --trust-remote-code \
    > "$RUN/logs/conntest_vllm_${ROLE}_node${NODE_ID}_port${port}.log" 2>&1 &
}
wait_ready() {
  local port=$1 replica=$2
  for t in $(seq 1 90); do
    if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
      echo "${HOST}:${port}" > "$EP/$ROLE/${ROLE}_node${NODE_ID}_${replica}.endpoint"
      echo "[ready] node=$NODE_ID port=$port after ${t}0s"; return 0
    fi; sleep 10
  done; echo "[WARN] node=$NODE_ID port=$port never came up"; return 1
}
gpu_pairs=("0,1" "2,3")
for i in $(seq 0 $((REPLICAS_PER_NODE - 1))); do serve $((8001 + i)) "${gpu_pairs[$i]}"; done
for i in $(seq 0 $((REPLICAS_PER_NODE - 1))); do wait_ready $((8001 + i)) "$i" || true; done
wait
CONTAINER_SCRIPT

chmod +x "$RUN/_per_node_conntest.sh" "$RUN/_in_container_conntest.sh"
srun --ntasks="$NNODES" --ntasks-per-node=1 bash "$RUN/_per_node_conntest.sh" &
SRUN_PID=$!

EXPECT=$(( NNODES * REPLICAS_PER_NODE ))
echo "[conntest] waiting for $EXPECT endpoints..."
for t in $(seq 1 90); do
  n=$(ls "$EP/judge"/*.endpoint 2>/dev/null | wc -l)
  echo "  [${t}0s] endpoints=$n/$EXPECT"
  [ "$n" -ge "$EXPECT" ] && break; sleep 10
done
n=$(ls "$EP/judge"/*.endpoint 2>/dev/null | wc -l)
if [ "$n" -lt "$EXPECT" ]; then echo "[conntest] FATAL: only $n/$EXPECT endpoints"; kill $SRUN_PID 2>/dev/null; exit 1; fi

echo "[conntest] ===== STEP A: no-load curl probe of every endpoint ====="
for f in "$EP"/judge/*.endpoint; do
  ADDR=$(cat "$f")
  if curl -sf -m 10 "http://${ADDR}/v1/models" >/dev/null 2>&1; then echo "  OK   $ADDR"; else echo "  FAIL $ADDR"; fi
done

echo "[conntest] ===== STEP B: judge the same $SAMPLE_N-row sample at varying concurrency ====="
export SINGULARITYENV_CC=gcc SINGULARITYENV_CXX=g++
for C in 64 256 1024; do
  ROOT="$DBG_BASE/c${C}"
  mkdir -p "$ROOT/outputs"
  head -n "$SAMPLE_N" "$SAMPLE_SRC" > "$ROOT/outputs/loc_gen.jsonl"
  echo "[conntest] --- concurrency=$C -> $ROOT ---"
  singularity exec --nv --bind /gpfs/scratch/ehpc532 \
    --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
    "$SIF" bash -c "
cd '$REPO'
python3 -m synthgen.cli generate --root '$ROOT' \
  --backend vllm --mode judge --models judge --in '$ROOT/outputs/loc_gen.jsonl' \
  --endpoints-dir '$EP/judge' --min-endpoints $EXPECT \
  --concurrency $C --max-attempts 1
"
  JF="$ROOT/outputs/judged_judge.jsonl"
  if [ -f "$JF" ]; then
    python3 - "$JF" "$C" <<'PY'
import sys,json
f,c=sys.argv[1],sys.argv[2]; ok=err=0; msgs={}
for l in open(f):
    try: r=json.loads(l)
    except: continue
    if r.get("error"): err+=1; m=str(r["error"])[:40]; msgs[m]=msgs.get(m,0)+1
    elif isinstance(r.get("score"),(int,float)): ok+=1
tot=ok+err
print(f"[RESULT] concurrency={c}  ok={ok}  err={err}  err_rate={100*err/max(1,tot):.1f}%  top={sorted(msgs.items(),key=lambda x:-x[1])[:2]}")
PY
  else
    echo "[RESULT] concurrency=$C  no judged output produced"
  fi
done

echo "[conntest] done; cleaning servers"
kill $SRUN_PID 2>/dev/null || true
exit 0
