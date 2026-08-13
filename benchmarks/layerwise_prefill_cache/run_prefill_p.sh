#!/usr/bin/env bash
set -euo pipefail

rm -rf /dev/shm/*

MODE="${1:?usage: $0 off|on}"
case "${MODE}" in
  off)
    LAYERWISE=false
    DEFAULT_MAX_MODEL_LEN=66000
    ALLOW_LONG_MAX_MODEL_LEN=0
    ;;
  on)
    LAYERWISE=true
    DEFAULT_MAX_MODEL_LEN=264000
    ALLOW_LONG_MAX_MODEL_LEN=1
    ;;
  *)
    echo "mode must be off or on" >&2
    exit 2
    ;;
esac

MODEL="${MODEL:-/workspace/models/GLM-5.1-w4a8}"
LMCACHE_CONFIG="${LMCACHE_CONFIG:-/workspace/lmy/lmcache_config.yaml}"
CHUNK_SIZE="${CHUNK_SIZE:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${DEFAULT_MAX_MODEL_LEN}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
PREFILL_PROFILE_EDGE_CHUNKS="${PREFILL_PROFILE_EDGE_CHUNKS:-0}"
PORT="${PORT:-9960}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/workspace/lmy/layerwise-prefill-bench/${MODE}-${RUN_ID}"
LOG="${OUTPUT_DIR}/server.log"
PROFILE_DIR="${PROFILE_DIR:-${OUTPUT_DIR}/profile}"

mkdir -p "${OUTPUT_DIR}" "${PROFILE_DIR}"
printf 'MODE=%s\nLAYERWISE=%s\nMAX_MODEL_LEN=%s\nPREFILL_PROFILE_EDGE_CHUNKS=%s\nLOG=%s\nPROFILE_DIR=%s\n' \
  "${MODE}" "${LAYERWISE}" "${MAX_MODEL_LEN}" \
  "${PREFILL_PROFILE_EDGE_CHUNKS}" "${LOG}" "${PROFILE_DIR}"

VLLM_USE_V1=1 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN="${ALLOW_LONG_MAX_MODEL_LEN}" \
VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE="${LAYERWISE}" \
VLLM_ASCEND_PREFILL_PROFILE_EDGE_CHUNKS="${PREFILL_PROFILE_EDGE_CHUNKS}" \
VLLM_ASCEND_DSA_UNBUNDLE=1 \
VLLM_ASCEND_DSA_TWO_GROUPS=1 \
VLLM_ASCEND_DSA_SHARED_POOL=1 \
VLLM_ASCEND_DSA_SHRINK_LATENT=2 \
VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE=0 \
VLLM_ASCEND_DSA_DISABLE_TARGET_SLOT_MAPPING=0 \
LMCACHE_USE_LAYERWISE=true \
LMCACHE_SAVE_UNFULL_CHUNK=true \
LMCACHE_ASCEND_SPARSE_TRANSFER_TOPK=2048 \
LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG}" \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HCCL_DETERMINISTIC=strict \
PYTHONHASHSEED=0 \
VLLM_LOG_STATS_INTERVAL=1 \
vllm serve "${MODEL}" \
  --served-model-name glm51-prefill \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 1 \
  --max-num-batched-tokens "${CHUNK_SIZE}" \
  --enable-chunked-prefill \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --no-enable-prefix-caching \
  --quantization ascend \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --profiler-config \
    "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\",\"torch_profiler_with_stack\":false,\"torch_profiler_with_memory\":false,\"ignore_frontend\":true}" \
  --kv-transfer-config \
    '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_producer","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}' \
  2>&1 | tee "${LOG}"
