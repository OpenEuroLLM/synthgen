#!/usr/bin/env bash
#SBATCH --job-name=verify-parallel-dedup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --partition=debug
#SBATCH --account=project_462001516
#SBATCH --output=production/35lang_run/logs/verify_parallel_dedup_%j.out

set -uo pipefail
cd /scratch/project_462001516/abhasjha/synthgen
VENV_PY=studies/localized_axes_ablation/_common/.diversity_venv/bin/python3
"$VENV_PY" -u production/35lang_run/_verify_parallel_dedup.py
