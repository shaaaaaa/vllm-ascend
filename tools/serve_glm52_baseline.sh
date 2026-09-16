#!/usr/bin/env bash
# Full GLM-5.2 weights, single host / TP8 / DP1, MTP enabled.
# Ordinary prefill+decode baseline; NOT the P-node two-bank offload path.
# Requires enough host RAM and at least 50 GiB free in /dev/shm for LMCache.
set -euo pipefail

# Dedicated host/container only: clears shared memory used by other services too.
rm -rf /dev/shm/*

model_path="${1:-/workspace/models/GLM-5.2-w4a8c8-0723}"

# Isolate this server from an inherited PD/Mooncake/debug configuration.
# This only changes the script's environment, not the caller's shell.
for name in ${!LMCACHE_@} ${!VLLM_@} ${!MOONCAKE_@}; do
    unset "$name"
done
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONHASHSEED=0
export HCCL_DETERMINISTIC=strict
export HCCL_BUFFSIZE=200
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=false
export VLLM_ASCEND_DSA_SPARSE_DECODE_D_NODE=false
export VLLM_ASCEND_DSA_UNBUNDLE=1
export VLLM_ASCEND_DSA_TWO_GROUPS=1
export VLLM_ASCEND_DSA_SHARED_POOL=1
export VLLM_ASCEND_DSA_SHRINK_LATENT=2
export VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_SFA_STAGED_GRAPH=0
export VLLM_ASCEND_SFA_FULL_GRAPH=0

# Local-only cache: no external YAML, Mooncake master or remote prefiller.
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=true
export LMCACHE_MAX_LOCAL_CPU_SIZE=50
export LMCACHE_USE_LAYERWISE=true
export LMCACHE_ENABLE_SPARSE_ATTENTION=true
export LMCACHE_DSA_TWO_GROUPS=true
export LMCACHE_STORE_ASYNC=false
export LMCACHE_SAVE_DECODE_CACHE=false
export LMCACHE_SAVE_UNFULL_CHUNK=true
export LMCACHE_SAVE_FULL_CHUNK_IN_DECODE=false
export LMCACHE_ENABLE_SHARED_CPU_CACHE=true
export LMCACHE_SHARED_CPU_CACHE_STRICT=true
export LMCACHE_EXTRA_CONFIG='{"save_only_first_rank": true}'

# FLASHCOMM1 sequence parallelism needs a TP8-aligned graph capture size.
vllm serve "$model_path" \
    --trust-remote-code \
    --load-format safetensors \
    --quantization ascend \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --distributed-executor-backend mp \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.93 \
    --max-model-len 140000 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --seed 1024 \
    --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}' \
    --compilation-config '{"cudagraph_capture_sizes": [8]}' \
    --additional-config '{"recompute_scheduler_enable": false, "multistream_overlap_shared_expert": false, "fuse_muls_add": true, "fuse_qknorm_rope": false, "enable_npugraph_ex": true, "layer_sharding": ["q_b_proj"]}' \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --kv-transfer-config '{"kv_connector": "LMCacheAscendConnectorV1Dynamic", "kv_role": "kv_both", "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}' \
    --host 0.0.0.0 \
    --port 8000 \
    2>&1 | tee log.log
