#!/usr/bin/env bash
#SBATCH --job-name=dedup-sizing-check2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --partition=debug
#SBATCH --account=project_462001516
#SBATCH --output=production/35lang_run/logs/dedup_sizing2_%j.out

# Corrected worst-case dedup cost benchmark (random unit vectors, near-zero
# collision rate -> kept_idx grows ~linearly with n, isolating the real
# O(n^2) cost the first sizing check's degenerate small-vocab text
# generator accidentally avoided). CPU-only, debug partition, not login node.
set -uo pipefail
cd /scratch/project_462001516/abhasjha/synthgen
VENV_PY=studies/localized_axes_ablation/_common/.diversity_venv/bin/python3
"$VENV_PY" -u production/35lang_run/_dedup_sizing_check2.py
