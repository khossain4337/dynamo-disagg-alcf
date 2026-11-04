#!/bin/bash
# Restores clean vLLM files, then applies the decode-PP diff. Never run while a job uses the env.
set -euxo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFF=$HERE/muse_spark/nixl_decode_pp_9e3db48873.diff
SOFT_ROOT=/vast/draco/tara/projects/Tara_Deployment/software
P=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=$P HTTPS_PROXY=$P http_proxy=$P https_proxy=$P
export UV_CACHE_DIR=$SOFT_ROOT/envs/conda_envs/.uv
set +u
source $SOFT_ROOT/miniforge3/bin/activate
conda activate $SOFT_ROOT/envs/conda_envs/vllm_0.32.1rc1_disaggpp_nixl_1.5.2_python_3.12.12
set -u

uv pip install --reinstall --no-deps "vllm==0.32.1rc1.dev562+g9e3db4887" \
    --extra-index-url https://wheels.vllm.ai/9e3db48873b87a3c232f16991675d33abb49b72b
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
patch -p1 --dry-run -d "$SP" < "$DIFF"
patch -p1 -d "$SP" < "$DIFF"
[ "$(grep -c disaggpp "$SP"/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py)" = 2 ]
patch -p1 -R --dry-run -d "$SP" < "$DIFF"
