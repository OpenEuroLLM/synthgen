#!/usr/bin/env bash
set -uo pipefail
cd "$REPO"

serve() {
  local name=$1 port=$2 gpus=$3 model=$4
  ROCR_VISIBLE_DEVICES=$gpus \
  vllm serve "$model" --served-model-name "$name" \
    --host 0.0.0.0 --port "$port" \
    --tensor-parallel-size $(echo $gpus | tr "," "\n" | wc -l) \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --enforce-eager \
    > "$STUDY/logs/vllm_${name}_node${NODE_ID}.log" 2>&1 &
}

wait_ready() {
  local name=$1 port=$2 outfile=$3
  for t in $(seq 1 90); do
    if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
      echo "${HOST}:${port}" > "$outfile"
      echo "[ready] $name port=$port node=$NODE_ID after ${t}0s"
      return 0
    fi
    sleep 10
  done
  echo "[FATAL] $name failed to start on node=$NODE_ID (see logs)"
  return 1
}

serve gen 8001 0,1 "$GEN_MODEL"
serve judge 8002 2,3 "$JUDGE_MODEL"
wait_ready gen 8001 "$EP/gen/gen_node${NODE_ID}.endpoint"
wait_ready judge 8002 "$EP/judge/judge_node${NODE_ID}.endpoint"

# keep both servers alive until the whole SLURM job is torn down
wait
