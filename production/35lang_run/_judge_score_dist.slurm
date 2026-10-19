#!/usr/bin/env bash
#SBATCH --job-name=judge-score-dist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --partition=debug
#SBATCH --account=project_462001516
#SBATCH --output=production/35lang_run/logs/judge_score_dist_%j.out

set -uo pipefail
cd /scratch/project_462001516/abhasjha/synthgen
VENV_PY=studies/localized_axes_ablation/_common/.diversity_venv/bin/python3
"$VENV_PY" -u production/35lang_run/_judge_score_dist.py
