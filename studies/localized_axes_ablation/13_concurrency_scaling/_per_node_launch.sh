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
