#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: $0 off|on}"
case "${MODE}" in
  off) LAYERWISE=false ;;
  on) LAYERWISE=true ;;
  *)
    echo "mode must be off or on" >&2
    exit 2
    ;;
esac

MODEL="${MODEL:-/workspace/models/GLM-5.1-w4a8}"
LMCACHE_CONFIG="${LMCACHE_CONFIG:-/workspace/lmy/lmcache_config.yaml}"
CHUNK_SIZE="${CHUNK_SIZE:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-70000}"
PORT="${PORT:-9960}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/workspace/lmy/layerwise-prefill-bench/${MODE}-${RUN_ID}"
LOG="${OUTPUT_DIR}/server.log"

mkdir -p "${OUTPUT_DIR}"
printf 'MODE=%s\nLAYERWISE=%s\nLOG=%s\n' "${MODE}" "${LAYERWISE}" "${LOG}"

VLLM_USE_V1=1 \
VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE="${LAYERWISE}" \
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
  --gpu-memory-utilization 0.94 \
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
  --kv-transfer-config \
    '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_producer","kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}' \
  2>&1 | tee "${LOG}"
