#!/bin/bash
#SBATCH --job-name=gpu-dedupe-equiv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:40:00
#SBATCH --partition=acc
#SBATCH --account=ehpc532
#SBATCH --qos=acc_debug
#SBATCH --output=production/35lang_run/logs/gpu_dedupe_equiv_%j.out
#
# Small equivalence test for near_dup.embedding_dedupe_gpu vs the CPU reference
# (see _gpu_dedupe_equiv_test.py). 20k prompts of one language.

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
  "$SIF" python3 "$REPO/production/35lang_run/_gpu_dedupe_equiv_test.py" "$IN" "$MODEL" mt 20000
