#!/bin/bash
#SBATCH --job-name=oellm-embed-dedupe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=80     # full acc node: 4 GPUs + all its memory (MN5 forbids --mem)
#SBATCH --time=04:00:00
#SBATCH --partition=acc
#SBATCH --account=ehpc532
#SBATCH --qos=acc_ehpc
#SBATCH --output=production/35lang_run/logs/embed_dedupe_%j.out
#
# Re-run dedupe on loc_full.kept.jsonl with the EMBEDDING near-dup pass ON
# (the production run did exact-string only, ~1.6% removed).
#
# Runs inside the vLLM SIF (which already has torch/numpy/transformers) with a
# small overlay dir on PYTHONPATH that adds only sentence-transformers + scipy +
# scikit-learn (built on the login node -- see st_overlay). GPU path
# (near_dup.embedding_dedupe_gpu): one worker per GPU encodes whole language
# buckets and runs the same greedy keep/drop rule as blocked matmuls.
# Thresholds (chosen from the 2026-10-22 sweeps, _dedupe_threshold_sweep.py):
# MiniLM cosine >= 0.97 for 31 languages; mt/ga/eu/is (not covered by MiniLM)
# use LaBSE at 0.96 -- calibrated on de/hr to remove the same share as MiniLM
# 0.97 -- via embed_dedupe_lang_overrides.json. The CPU path (35 single-thread workers) encoded at ~18 texts/s per
# worker and grew ~1GB/worker/5min, heading for OOM on a 503GB node (job
# 47793070). Equivalence vs the CPU reference: _gpu_dedupe_equiv_test.py.
#
# Model (paraphrase-multilingual-MiniLM-L12-v2) is pre-staged as a self-
# contained folder at $MODEL (transferred from LUMI); loaded by local path,
# fully offline.

set -uo pipefail
module load singularity/4.1.5

SIF=/gpfs/scratch/ehpc532/eit234648/vllm/vllm-v0.22.1.sif
OVL=/gpfs/scratch/ehpc532/eit234648/st_overlay
REPO=/gpfs/scratch/ehpc532/eit234648/synthgen
OUT=/gpfs/scratch/ehpc532/eit234648/oellm_production_35lang_sup/outputs
MODEL=/gpfs/scratch/ehpc532/eit234648/models/paraphrase-ml-minilm

COMMON="--nv --bind /gpfs/scratch/ehpc532 \
  --env PYTHONPATH=$OVL:$REPO \
  --env HF_HOME=/gpfs/scratch/ehpc532/eit234648/.hf_cache \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false --env OMP_NUM_THREADS=1"

# --- preflight: model present + embedder loads, else fail fast (seconds, not hours) ---
if [ ! -f "$MODEL/config.json" ]; then
  echo "[embed-dedupe] FATAL: model folder missing at $MODEL" >&2; exit 2
fi
echo "[embed-dedupe] preflight $(date)"
singularity exec $COMMON "$SIF" python3 -c \
  "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('$MODEL'); print('[embed-dedupe] embedder loads OK, dim=', m.get_sentence_embedding_dimension())" \
  || { echo "[embed-dedupe] FATAL: embedder failed to load" >&2; exit 3; }

# --- the actual dedupe ---
echo "[embed-dedupe] start $(date) on $(hostname)"
singularity exec $COMMON "$SIF" python3 "$REPO/production/35lang_run/run_embed_dedupe.py" \
  --input  "$OUT/loc_full.kept.jsonl" \
  --output "$OUT/loc_full.dedup_embed.jsonl" \
  --report "$OUT/loc_full.dedup_embed.report.json" \
  --embedding-model "$MODEL" \
  --embedding-device cuda \
  --embedding-threshold 0.97 \
  --lang-overrides "$REPO/production/35lang_run/embed_dedupe_lang_overrides.json" \
  --embedding-workers 4
RC=$?
echo "[embed-dedupe] exit $RC $(date)"
exit $RC
