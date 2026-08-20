#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   run_glm51_tp_dp.sh \
#       [--dp-size 1|2] \
#       [--layerwise-prefill on|off] \
#       [--load-8-layers true|false] \
#       [--max-model-len N|-1]

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

rm -rf /dev/shm/*

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/workspace/models/GLM-5.1-w4a8}"
WORK_DIR="${WORK_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
LMCACHE_CONFIG="${LMCACHE_CONFIG:-/workspace/lmy/lmcache_config.yaml}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9960}"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
DP_SIZE="${DP_SIZE:-2}"
LAYERWISE_MODE="${LAYERWISE_MODE:-on}"
LOAD_8_LAYERS="${LOAD_8_LAYERS:-false}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

usage() {
    echo "usage: $0 [--dp-size 1|2] [--layerwise-prefill on|off] [--load-8-layers true|false] [--max-model-len N|-1]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dp-size)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            DP_SIZE="$2"
            shift 2
            ;;
        --layerwise-prefill)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            LAYERWISE_MODE="$2"
            shift 2
            ;;
        --load-8-layers)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            LOAD_8_LAYERS="$2"
            shift 2
            ;;
        --max-model-len)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

case "${DP_SIZE}" in
    1)
        TP_SIZE=8
        ;;
    2)
        TP_SIZE=4
        ;;
    *)
        echo "DP_SIZE must be 1 or 2, got: ${DP_SIZE}" >&2
        exit 2
        ;;
esac

case "${LOAD_8_LAYERS}" in
    true)
        MODEL_VARIANT="8layers"
        HF_OVERRIDES_ARGS=(
            --hf-overrides
            '{"num_hidden_layers":8,"num_nextn_predict_layers":0,"vllm_skip_extra_layer_weights":true}'
        )
        ;;
    false)
        MODEL_VARIANT="full"
        HF_OVERRIDES_ARGS=()
        ;;
    *)
        echo "LOAD_8_LAYERS must be true or false, got: ${LOAD_8_LAYERS}" >&2
        exit 2
        ;;
esac

case "${LAYERWISE_MODE}" in
    on)
        LAYERWISE_PREFILL_P_NODE=true
        DSA_SHARED_POOL=1
        LMCACHE_USE_LAYERWISE_VALUE=true
        LMCACHE_ASYNC_DECODE_SAVE_VALUE="${LMCACHE_ASYNC_DECODE_SAVE:-false}"
        ;;
    off)
        LAYERWISE_PREFILL_P_NODE=false
        # Preserve the original bundle-pool + layerwise LMCache connector.
        # Only disable the P-node two-bank KV-cache reuse protocol.
        DSA_SHARED_POOL=1
        LMCACHE_USE_LAYERWISE_VALUE=true
        LMCACHE_ASYNC_DECODE_SAVE_VALUE=false
        ;;
    *)
        echo "layerwise mode must be on or off, got: ${LAYERWISE_MODE}" >&2
        exit 2
        ;;
esac

if [[ -z "${MAX_MODEL_LEN}" ]]; then
    if [[ "${MODEL_VARIANT}" == "full" && "${LAYERWISE_MODE}" == "off" ]]; then
        # A full 78-layer resident KV cache cannot hold 140K tokens on the
        # current 64-GiB setup. Let vLLM fit the largest context that actually
        # fits instead of failing startup.
        MAX_MODEL_LEN=-1
    else
        MAX_MODEL_LEN=140000
    fi
fi
if [[ ! "${MAX_MODEL_LEN}" =~ ^(-1|[1-9][0-9]*)$ ]]; then
    echo "MAX_MODEL_LEN must be -1 or a positive integer, got: ${MAX_MODEL_LEN}" >&2
    exit 2
fi

DP_LOCAL_SIZE="${DP_SIZE}"
TOPOLOGY="tp${TP_SIZE}_dp${DP_SIZE}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm51-prefill}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/${TOPOLOGY}_${LAYERWISE_MODE}_${MODEL_VARIANT}_logs}"
PROFILE_DIR="${PROFILE_DIR:-${WORK_DIR}/vllm_profile/${TOPOLOGY}_${LAYERWISE_MODE}_${MODEL_VARIANT}}"

if [[ ! -f "${LMCACHE_CONFIG}" ]]; then
    echo "LMCache config does not exist: ${LMCACHE_CONFIG}" >&2
    echo "Set LMCACHE_CONFIG to the LMCache YAML file." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${PROFILE_DIR}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/server_${RUN_ID}.log"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=200

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ASCEND_BUFFER_POOL="${ASCEND_BUFFER_POOL:-4:8}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-0}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export VLLM_ASCEND_DSA_UNBUNDLE="${VLLM_ASCEND_DSA_UNBUNDLE:-1}"
export VLLM_ASCEND_DSA_TWO_GROUPS="${VLLM_ASCEND_DSA_TWO_GROUPS:-1}"
export VLLM_ASCEND_DSA_SHARED_POOL="${DSA_SHARED_POOL}"
export VLLM_ASCEND_DSA_SHRINK_LATENT="${VLLM_ASCEND_DSA_SHRINK_LATENT:-2}"
export VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE=0
export VLLM_ASCEND_DSA_DISABLE_TARGET_SLOT_MAPPING=0
export VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE="${LAYERWISE_PREFILL_P_NODE}"
export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0
export LMCACHE_USE_LAYERWISE="${LMCACHE_USE_LAYERWISE_VALUE}"
export LMCACHE_ASYNC_DECODE_SAVE="${LMCACHE_ASYNC_DECODE_SAVE_VALUE}"
export LMCACHE_SAVE_UNFULL_CHUNK=true
export LMCACHE_ASCEND_SPARSE_TRANSFER_TOPK=2048
export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export ASCEND_AGGREGATE_ENABLE="${ASCEND_AGGREGATE_ENABLE:-1}"
export ASCEND_TRANSPORT_PRINT="${ASCEND_TRANSPORT_PRINT:-1}"
export ACL_OP_INIT_MODE="${ACL_OP_INIT_MODE:-1}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"

PROFILER_CONFIG="$(printf \
    '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_memory":false,"ignore_frontend":true,"torch_profiler_with_stack":false}' \
    "${PROFILE_DIR}")"

echo "MODEL_PATH=${MODEL_PATH}"
echo "LMCACHE_CONFIG=${LMCACHE_CONFIG}"
echo "LAYERWISE_MODE=${LAYERWISE_MODE}"
echo "LOAD_8_LAYERS=${LOAD_8_LAYERS}"
echo "MODEL_VARIANT=${MODEL_VARIANT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "LAYERWISE_PREFILL_P_NODE=${VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE}"
echo "LMCACHE_USE_LAYERWISE=${LMCACHE_USE_LAYERWISE}"
echo "LMCACHE_ASYNC_DECODE_SAVE=${LMCACHE_ASYNC_DECODE_SAVE}"
echo "DSA_SHARED_POOL=${VLLM_ASCEND_DSA_SHARED_POOL}"
echo "TP=${TP_SIZE} DP=${DP_SIZE} DP_LOCAL=${DP_LOCAL_SIZE} DP_BACKEND=mp"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S}"
echo "LOG_FILE=${LOG_FILE}"
echo "PROFILE_DIR=${PROFILE_DIR}"

cd "${WORK_DIR}"

# In reduced mode, num_nextn_predict_layers=0 avoids constructing an MTP draft
# layer and vllm_skip_extra_layer_weights ignores checkpoint layers 8+.
# max-num-seqs is per DP replica. Each local replica accepts one long-prefill
# request without reserving layerwise shared-pool capacity for unused
# sequences.
vllm serve "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --pipeline-parallel-size 1 \
    --data-parallel-size "${DP_SIZE}" \
    --data-parallel-size-local "${DP_LOCAL_SIZE}" \
    --data-parallel-backend mp \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.93 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs 1 \
    --max-num-batched-tokens "${CHUNK_SIZE}" \
    --enable-chunked-prefill \
    --seed 1024 \
    --trust-remote-code \
    --quantization ascend \
    --compilation-config \
        "{\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[${CHUNK_SIZE}]}" \
    "${HF_OVERRIDES_ARGS[@]}" \
    --additional-config \
        '{
            "recompute_scheduler_enable": false,
            "multistream_overlap_shared_expert": false,
            "fuse_muls_add": true,
            "fuse_qknorm_rope": false,
            "enable_npugraph_ex": true,
            "layer_sharding": ["q_b_proj"]
        }' \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --profiler-config "${PROFILER_CONFIG}" \
    --no-enable-prefix-caching \
    --kv-transfer-config \
        '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_producer","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}' \
    2>&1 | tee "${LOG_FILE}"
