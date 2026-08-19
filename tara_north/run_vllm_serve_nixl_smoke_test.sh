#!/bin/bash -x
set -uo pipefail

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
#
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1

source /vast/draco/tara/projects/Tara_Deployment/software/miniforge3/bin/activate
conda activate /vast/draco/tara/projects/Tara_Deployment/software/envs/conda_envs/vllm_0.27.1_nixl_1.4.0_python_3.12.12

export HF_TOKEN=$(cat ~/.hf_token)
export PYTHONNOUSERSITE=1

export HF_HOME="/vast/draco/tara/projects/Tara_Deployment/software/model-weights"
export HF_DATASETS_CACHE="/vast/draco/tara/projects/Tara_Deployment/software/model-weights"
export HF_MODULES_CACHE="/vast/draco/tara/projects/Tara_Deployment/software/model-weights"
export RAY_TMPDIR="/tmp"
export TMPDIR="/tmp"

mkdir -p ${HF_HOME}

export VLLM_LOGGING_LEVEL="DEBUG"

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
PORT=${PORT:-8100}
LOG_FILE=vllm_nixl_smoke_$(date +%Y%m%d_%H%M%S).log

export UCX_TLS=cuda_copy,cuda_ipc,sm,tcp,self
export UCX_MODULE_DIR=$(python3 -c "import site,glob; print(glob.glob(site.getsitepackages()[0]+'/nixl_cu13.libs/ucx')[0])")

echo "Using UCX_MODULE_DIR: ${UCX_MODULE_DIR}"
echo "Logging to: ${LOG_FILE}"

vllm serve ${MODEL} \
    --port ${PORT} \
    --enforce-eager \
    --gpu-memory-utilization 0.3 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
    > ${LOG_FILE} 2>&1 &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

tail -f ${LOG_FILE} &
TAIL_PID=$!

echo "=== Waiting for server to become healthy ==="
READY=0
for i in $(seq 1 60); do
    echo "Health check attempt ${i}/60..."
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/health 2>/dev/null | grep -q "200"; then
        echo "Server healthy after $((i * 5))s"
        READY=1
        break
    fi
    sleep 5
done

if [ ${READY} -eq 0 ]; then
    echo "--- Server never became healthy ---"
    kill -TERM ${SERVER_PID} 2>/dev/null
    kill ${TAIL_PID} 2>/dev/null
    exit 1
fi

echo "=== Sending a real completion request ==="
curl -s http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL}\", \"prompt\": \"The capital of France is\", \"max_tokens\": 10}"
echo

echo "=== Request done, sending SIGTERM (graceful shutdown) ==="
kill -TERM ${SERVER_PID}

wait ${SERVER_PID}
EXIT_CODE=$?

kill ${TAIL_PID} 2>/dev/null

echo "=== Server exited with code ${EXIT_CODE} ==="
if [ ${EXIT_CODE} -eq 139 ]; then
    echo "--- SEGFAULT on shutdown (signal 11) ---"
elif [ ${EXIT_CODE} -eq 0 ]; then
    echo "--- CLEAN EXIT ---"
else
    echo "--- Exited with non-zero, non-segfault code ${EXIT_CODE} ---"
fi

echo "=== Scanning log for crash markers ==="
grep -i "segmentation fault\|core dumped\|Fatal Python error\|Traceback" ${LOG_FILE} || echo "No crash markers found in log"

echo "Full log at: ${LOG_FILE}"
