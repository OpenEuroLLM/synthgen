#!/bin/bash
# Run synthgen client commands (generate/filter/verify/...) on MN5.
# Bypasses the login-node internet proxy (localhost:8888), which otherwise
# hijacks calls to internal vLLM endpoints and makes every request fail with
# "All connection attempts failed".
#   ./run.sh generate --backend vllm --mode prompt --models gemma-4-31B-it --endpoints-dir endpoints
source /gpfs/scratch/ehpc532/eit234648/synthgen/activate_env.sh
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
exec synthgen "$@"
