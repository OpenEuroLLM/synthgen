#!/bin/bash
#SBATCH --job-name=dedupe-labse-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc532
#SBATCH --qos=acc_debug
#SBATCH --output=production/35lang_run/logs/dedupe_labse_sweep_%j.out
#
# Threshold sweep with LaBSE for the languages MiniLM does not cover
# (mt/ga/eu/is), plus de/hr as calibration anchors: compare which LaBSE
# threshold removes the same kind of near-copies MiniLM removes at 0.97.
# (The sweep's "model" yes/NO column refers to MiniLM coverage, not LaBSE.)
# Then a small end-to-end check of near_dup's per-language overrides.

set -uo pipefail
module load singularity/4.1.5

SIF=/gpfs/scratch/ehpc532/eit234648/vllm/vllm-v0.22.1.sif
OVL=/gpfs/scratch/ehpc532/eit234648/st_overlay
REPO=/gpfs/scratch/ehpc532/eit234648/synthgen
IN=/gpfs/scratch/ehpc532/eit234648/oellm_production_35lang_sup/outputs/loc_full.kept.jsonl
LABSE=/gpfs/scratch/ehpc532/eit234648/models/labse
MINILM=/gpfs/scratch/ehpc532/eit234648/models/paraphrase-ml-minilm
RUN="singularity exec --nv --bind /gpfs/scratch/ehpc532 --env PYTHONPATH=$OVL:$REPO --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env TOKENIZERS_PARALLELISM=false $SIF python3"

[ -f "$LABSE/modules.json" ] || { echo "FATAL: LaBSE not found at $LABSE" >&2; exit 2; }

$RUN "$REPO/production/35lang_run/_dedupe_threshold_sweep.py" "$IN" "$LABSE" 30000 \
  mt,ga,eu,is,de,hr 0.80,0.85,0.88,0.90,0.92,0.94,0.96,0.98

echo "==== per-language override end-to-end check ===="
$RUN "$REPO/production/35lang_run/_override_check.py" "$IN" "$MINILM" "$LABSE"
