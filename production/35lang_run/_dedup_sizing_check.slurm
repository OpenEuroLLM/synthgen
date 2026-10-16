#!/usr/bin/env bash
#SBATCH --job-name=dedup-sizing-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --partition=debug
#SBATCH --account=project_462001516
#SBATCH --output=production/35lang_run/logs/dedup_sizing_%j.out

# One-off CPU-only sizing check (no GPU needed): per-language row counts,
# judge score>=7 pass rate, and a small-n timing benchmark of the real
# embedding_dedupe_mask() extrapolated to the real per-language kept-count --
# all deliberately run on a debug compute node, not the login node.
set -uo pipefail
cd /scratch/project_462001516/abhasjha/synthgen
VENV_PY=studies/localized_axes_ablation/_common/.diversity_venv/bin/python3
"$VENV_PY" production/35lang_run/_dedup_sizing_check.py
