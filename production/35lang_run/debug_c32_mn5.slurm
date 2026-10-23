#!/usr/bin/env bash
#SBATCH --job-name=oellm-judge-c32
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80
#SBATCH --time=02:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc532
#SBATCH --qos=acc_debug
#SBATCH --output=production/35lang_run/logs/c32_%j.out

# ============================================================================
# Single-point concurrency test: 32 requests/replica (= 128 total on 2 nodes x
# 2 replicas). Sits between the clean 16/replica and the collapsing 64/replica
# seen in debug_conntest. Uses a LARGER 20k-row sample so slow-onset
# degradation (64/replica only turned into errors after ~2k rows / ~10 min)
# would actually surface instead of a short, unconvincing clean window.
# Throwaway data root; never touches production outputs.
#
#   sbatch production/35lang_run/debug_c32_mn5.slurm    (ONLY on user's say-so)
# ============================================================================
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

export REPO=/gpfs/scratch/ehpc532/eit234648/synthgen
export SIF=/gpfs/scratch/ehpc532/eit234648/vllm/vllm-v0.22.1.sif
export HF_HOME=/gpfs/scratch/ehpc532/eit234648/.hf_cache
export RUN="$REPO/production/35lang_run"
export EP="$RUN/endpoints_c32"
export JUDGE_MODEL=/gpfs/scratch/ehpc532/eit234648/models/Qwen/Qwen3.6-27B
export REPLICAS_PER_NODE=2
export SAMPLE_SRC=/gpfs/scratch/ehpc532/eit234648/oellm_production_35lang/outputs/loc_gen.jsonl
export SAMPLE_N=20000
export PER_REPLICA=32
export DBG_ROOT="$RUN/_c32_data"

NNODES=${SLURM_NNODES:-2}
module load singularity/4.1.5
rm -rf "$EP" "$DBG_ROOT"; mkdir -p "$EP/judge" "$RUN/logs" "$DBG_ROOT/outputs"

cat > "$RUN/_per_node_c32.sh" <<'NODE_SCRIPT'
#!/usr/bin/env bash
set -uo pipefail
NODE_ID=$SLURM_PROCID; HOST=$(hostname); ROLE=judge; MODEL="$JUDGE_MODEL"
export SINGULARITYENV_CC=gcc SINGULARITYENV_CXX=g++
export SINGULARITYENV_VLLM_DISABLE_COMPILE_CACHE=1
singularity exec --nv --bind /gpfs/scratch/ehpc532 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
  --env RUN="$RUN" --env EP="$EP" --env ROLE="$ROLE" --env MODEL="$MODEL" \
  --env NODE_ID="$NODE_ID" --env HOST="$HOST" --env REPLICAS_PER_NODE="$REPLICAS_PER_NODE" \
  "$SIF" bash "$RUN/_in_container_c32.sh"
NODE_SCRIPT

cat > "$RUN/_in_container_c32.sh" <<'CONTAINER_SCRIPT'
#!/usr/bin/env bash
set -uo pipefail
cd "$REPO"
serve() {
  local port=$1 gpus=$2
  CUDA_VISIBLE_DEVICES=$gpus vllm serve "$MODEL" --served-model-name "$ROLE" \
    --host 0.0.0.0 --port "$port" --tensor-parallel-size $(echo $gpus | tr "," "\n" | wc -l) \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --trust-remote-code \
    > "$RUN/logs/c32_vllm_node${NODE_ID}_port${port}.log" 2>&1 &
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

chmod +x "$RUN/_per_node_c32.sh" "$RUN/_in_container_c32.sh"
srun --ntasks="$NNODES" --ntasks-per-node=1 bash "$RUN/_per_node_c32.sh" &
SRUN_PID=$!

EXPECT=$(( NNODES * REPLICAS_PER_NODE ))
for t in $(seq 1 90); do
  n=$(ls "$EP/judge"/*.endpoint 2>/dev/null | wc -l)
  echo "  [${t}0s] endpoints=$n/$EXPECT"; [ "$n" -ge "$EXPECT" ] && break; sleep 10
done
n=$(ls "$EP/judge"/*.endpoint 2>/dev/null | wc -l)
if [ "$n" -lt "$EXPECT" ]; then echo "[c32] FATAL: only $n/$EXPECT endpoints"; kill $SRUN_PID 2>/dev/null; exit 1; fi

TOTAL=$(( EXPECT * PER_REPLICA ))
head -n "$SAMPLE_N" "$SAMPLE_SRC" > "$DBG_ROOT/outputs/loc_gen.jsonl"
echo "[c32] ===== judging $SAMPLE_N rows at ${PER_REPLICA}/replica = $TOTAL total concurrency ====="
export SINGULARITYENV_CC=gcc SINGULARITYENV_CXX=g++
singularity exec --nv --bind /gpfs/scratch/ehpc532 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
  "$SIF" bash -c "
cd '$REPO'
python3 -m synthgen.cli generate --root '$DBG_ROOT' \
  --backend vllm --mode judge --models judge --in '$DBG_ROOT/outputs/loc_gen.jsonl' \
  --endpoints-dir '$EP/judge' --min-endpoints $EXPECT \
  --concurrency $TOTAL --max-attempts 1
"
JF="$DBG_ROOT/outputs/judged_judge.jsonl"
python3 - "$JF" "$PER_REPLICA" "$TOTAL" <<'PY'
import sys,json
f,pr,tot=sys.argv[1],sys.argv[2],sys.argv[3]; ok=err=0; msgs={}
try:
    for l in open(f):
        try: r=json.loads(l)
        except: continue
        if r.get("error"): err+=1; m=str(r["error"])[:40]; msgs[m]=msgs.get(m,0)+1
        elif isinstance(r.get("score"),(int,float)): ok+=1
except FileNotFoundError:
    print(f"[RESULT] per_replica={pr} total={tot}  no judged output produced"); sys.exit()
n=ok+err
print(f"[RESULT] per_replica={pr} total={tot}  ok={ok}  err={err}  err_rate={100*err/max(1,n):.1f}%  top={sorted(msgs.items(),key=lambda x:-x[1])[:3]}")
PY
kill $SRUN_PID 2>/dev/null || true
exit 0
