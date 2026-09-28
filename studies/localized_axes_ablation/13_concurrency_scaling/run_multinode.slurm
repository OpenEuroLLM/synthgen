#!/usr/bin/env bash
#SBATCH --job-name=axes-13-concurrency-scaling
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --partition=dev-g
#SBATCH --account=project_462001516
#SBATCH --output=studies/localized_axes_ablation/13_concurrency_scaling/logs/%j.out

# Concurrency-scaling smoke test: the EXACT proven single-node serve/
# wait_ready pattern (2 GPUs gen + 2 GPUs judge, same as every other ablation
# in this study), replicated across 4 real nodes via srun -- deliberately the
# simple symmetric layout (not the more GPU-efficient asymmetric one) to keep
# this smoke test low-risk. Gives 4 gen replicas + 4 judge replicas in one
# shared pool, then runs bench_pool_scaling.py through the REAL
# VLLMBackend/EndpointPool code path to check whether throughput actually
# scales ~4x when concurrency is raised proportionally.
#
# Uses exported environment variables (not string interpolation) to pass
# config into the per-node srun'd script and the container -- much less
# fragile than nesting quoted nested bash -c strings three deep.
# Submit from repo root: sbatch studies/localized_axes_ablation/13_concurrency_scaling/run_multinode.slurm
set -uo pipefail

export REPO=/scratch/project_462001516/abhasjha/synthgen
export SIF=/scratch/project_465002530/containers/vllm-openai-rocm.sif
export HF_HOME=/scratch/project_462001516/cache/huggingface/abhasjha
export STUDY="$REPO/studies/localized_axes_ablation/13_concurrency_scaling"
export EP="$STUDY/endpoints"   # on Lustre -- shared/visible from every node
export GEN_MODEL=google/gemma-4-31b-it
export JUDGE_MODEL=Qwen/Qwen3.6-27B

rm -rf "$EP"; mkdir -p "$EP/gen" "$EP/judge" "$STUDY/logs"

cat > "$STUDY/_per_node_launch.sh" <<'NODE_SCRIPT'
#!/usr/bin/env bash
set -uo pipefail
NODE_ID=$SLURM_PROCID
HOST=$(hostname)

singularity exec \
  --bind /scratch/project_462001516:/scratch/project_462001516 \
  --bind /scratch/project_465002530:/scratch/project_465002530 \
  --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env PYTHONPATH="$REPO" \
  --env STUDY="$STUDY" --env EP="$EP" \
  --env GEN_MODEL="$GEN_MODEL" --env JUDGE_MODEL="$JUDGE_MODEL" \
  --env NODE_ID="$NODE_ID" --env HOST="$HOST" \
  "$SIF" bash "$STUDY/_in_container_launch.sh"
NODE_SCRIPT

cat > "$STUDY/_in_container_launch.sh" <<'CONTAINER_SCRIPT'
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
CONTAINER_SCRIPT

chmod +x "$STUDY/_per_node_launch.sh" "$STUDY/_in_container_launch.sh"

srun --ntasks=4 --ntasks-per-node=1 bash "$STUDY/_per_node_launch.sh" &
SRUN_PID=$!

echo "[13] waiting for all 8 replica endpoints (4 gen + 4 judge)..."
n_gen=0; n_judge=0
for t in $(seq 1 90); do
  n_gen=$(ls "$EP/gen"/*.endpoint 2>/dev/null | wc -l)
  n_judge=$(ls "$EP/judge"/*.endpoint 2>/dev/null | wc -l)
  echo "  [${t}0s elapsed] gen=$n_gen/4 judge=$n_judge/4"
  if [ "$n_gen" -ge 4 ] && [ "$n_judge" -ge 4 ]; then
    break
  fi
  sleep 10
done

if [ "$n_gen" -lt 4 ] || [ "$n_judge" -lt 4 ]; then
  echo "[13] FATAL: not all replicas came up (gen=$n_gen/4 judge=$n_judge/4) -- see logs/vllm_*_node*.log"
  kill $SRUN_PID 2>/dev/null || true
  exit 1
fi

echo "[13] all replicas ready -- running concurrency-scaling bench"
# Runs on the login-node side (outside the singularity container), where the
# default python3 is 3.6.15 -- too old for this repo's `from __future__
# import annotations` + `X | None` union syntax (same issue hit earlier this
# study with ground_domains.py). Use the existing login-node venv instead.
VENV_PY="$REPO/studies/localized_axes_ablation/_common/.diversity_venv/bin/python3"
"$VENV_PY" "$STUDY/bench_pool_scaling.py" \
  --gen-endpoints "$EP/gen" --judge-endpoints "$EP/judge" \
  --gen-model gen --judge-model judge \
  --out "$STUDY/outputs"
RC=$?

echo "[13] exit code: $RC"
kill $SRUN_PID 2>/dev/null || true
exit $RC
