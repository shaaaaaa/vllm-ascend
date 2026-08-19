#!/usr/bin/env bash
set -Eeuo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

rm -rf /dev/shm/*

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/workspace/models/GLM-5.1-w4a8}"
WORK_DIR="${WORK_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
LMCACHE_CONFIG="${LMCACHE_CONFIG:-/workspace/lmy/lmcache_config.yaml}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9960}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm51-tp4-dp2-8layers}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/tp4_dp2_8layers_logs}"
PROFILE_DIR="${PROFILE_DIR:-${WORK_DIR}/vllm_profile/tp4_dp2_8layers}"

if [[ ! -f "${LMCACHE_CONFIG}" ]]; then
    echo "LMCache config does not exist: ${LMCACHE_CONFIG}" >&2
    echo "Set LMCACHE_CONFIG to the P-node LMCache YAML file." >&2
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
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export VLLM_ASCEND_DSA_UNBUNDLE="${VLLM_ASCEND_DSA_UNBUNDLE:-1}"
export VLLM_ASCEND_DSA_TWO_GROUPS="${VLLM_ASCEND_DSA_TWO_GROUPS:-1}"
export VLLM_ASCEND_DSA_SHARED_POOL="${VLLM_ASCEND_DSA_SHARED_POOL:-1}"
export VLLM_ASCEND_DSA_SHRINK_LATENT="${VLLM_ASCEND_DSA_SHRINK_LATENT:-2}"
export VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE=0
export VLLM_ASCEND_DSA_DISABLE_TARGET_SLOT_MAPPING=0
export VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=true
export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0
export LMCACHE_USE_LAYERWISE=true
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
echo "LAYERWISE_PREFILL_P_NODE=${VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE}"
echo "TP=4 DP=2 DP_LOCAL=2 DP_BACKEND=mp"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "LOG_FILE=${LOG_FILE}"
echo "PROFILE_DIR=${PROFILE_DIR}"

cd "${WORK_DIR}"

# num_nextn_predict_layers=0 means that no MTP draft layer is constructed, so
# this reduced-model smoke launcher intentionally does not enable speculative
# decoding. vllm_skip_extra_layer_weights lets the loader ignore layers 8+ in
# the original checkpoint instead of treating them as unexpected weights.
# max-num-seqs is per DP replica. One sequence on each of the two local
# replicas gives two concurrent long-prefill requests without reserving
# layerwise shared-pool capacity for unused sequences.
vllm serve "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --data-parallel-size 2 \
    --data-parallel-size-local 2 \
    --data-parallel-backend mp \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.93 \
    --max-model-len 140000 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --seed 1024 \
    --trust-remote-code \
    --quantization ascend \
    --enforce-eager \
    --hf-overrides \
        '{"num_hidden_layers":8,"num_nextn_predict_layers":0,"vllm_skip_extra_layer_weights":true}' \
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
