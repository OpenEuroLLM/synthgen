#!/bin/bash
#SBATCH --job-name=dedupe-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc532
#SBATCH --qos=acc_debug
#SBATCH --output=production/35lang_run/logs/dedupe_sweep_%j.out
#
# Threshold sweep of the embedding near-dup pass (within-language only), 30k
# prompts/lang, thresholds 0.85-0.97. Writes nothing. See _dedupe_threshold_sweep.py.

set -uo pipefail
module load singularity/4.1.5

SIF=/gpfs/scratch/ehpc532/eit234648/vllm/vllm-v0.22.1.sif
OVL=/gpfs/scratch/ehpc532/eit234648/st_overlay
REPO=/gpfs/scratch/ehpc532/eit234648/synthgen
IN=/gpfs/scratch/ehpc532/eit234648/oellm_production_35lang_sup/outputs/loc_full.kept.jsonl
MODEL=/gpfs/scratch/ehpc532/eit234648/models/paraphrase-ml-minilm

singularity exec --nv --bind /gpfs/scratch/ehpc532 \
  --env PYTHONPATH=$OVL:$REPO --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  "$SIF" python3 "$REPO/production/35lang_run/_dedupe_threshold_sweep.py" "$IN" "$MODEL" 30000
