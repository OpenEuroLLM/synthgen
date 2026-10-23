#!/bin/bash
#SBATCH --job-name=hf-release-build
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc532
#SBATCH --qos=acc_debug
#SBATCH --output=production/35lang_run/logs/hf_release_%j.out
#
# Convert loc_full.dedup_embed.jsonl into a Hugging Face-ready folder
# (per-language parquet + dataset card). CPU only. See build_hf_release.py.

set -uo pipefail
module load singularity/4.1.5

SIF=/gpfs/scratch/ehpc532/eit234648/vllm/vllm-v0.22.1.sif
REPO=/gpfs/scratch/ehpc532/eit234648/synthgen
OUTS=/gpfs/scratch/ehpc532/eit234648/oellm_production_35lang_sup/outputs
DEST=/gpfs/scratch/ehpc532/eit234648/hf_release/synthgen-35lang-if

singularity exec --bind /gpfs/scratch/ehpc532 --env PYTHONPATH=/gpfs/scratch/ehpc532/eit234648/hf_overlay "$SIF" python3 \
  "$REPO/production/35lang_run/build_hf_release.py" \
  "$OUTS/loc_full.dedup_embed.jsonl" "$OUTS/loc_full.dedup_embed.report.json" "$DEST"
