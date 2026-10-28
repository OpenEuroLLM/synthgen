#!/bin/bash
# Activate the synthgen client environment on MN5.
#   source activate_env.sh
module load hdf5 python/3.12.1 >/dev/null 2>&1
source /gpfs/scratch/ehpc532/eit234648/synthgen/venv/bin/activate
# BSC python module injects a shared PYTHONPATH that leaks system packages
# into the venv; clear it so the venv is fully self-contained.
unset PYTHONPATH
export PIP_CACHE_DIR=/gpfs/scratch/ehpc532/eit234648/.pip-cache
