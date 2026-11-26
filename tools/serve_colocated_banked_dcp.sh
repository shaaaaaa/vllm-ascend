#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run each role in its own shell on one Ascend host with a live Mooncake master.
set -euo pipefail
role=${1:?usage: serve_colocated_banked_dcp.sh prefill|decode|proxy}
tools_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
p_port=${PREFILL_PORT:-7910}
d_port=${DECODE_PORT:-7920}
proxy_port=${PROXY_PORT:-8080}
case "$role" in
    proxy)
        exec python3 "$tools_dir/../examples/disaggregated_prefill_v1/load_balance_proxy_server_enhanced.py" \
            --host 127.0.0.1 --port "$proxy_port" --prefiller-hosts 127.0.0.1 --prefiller-ports "$p_port" \
            --decoder-hosts 127.0.0.1 --decoder-ports "$d_port" --enable-remote-lmcache-store \
            --max-model-len "${MAX_MODEL_LEN:-84096}" --disable-tokenizer-analysis --disable-metrics-polling ;;
    prefill|decode) ;;
    *) echo "Expected prefill, decode or proxy" >&2; exit 2 ;;
esac

IFS=, read -ra p_devices <<< "${PREFILL_DEVICES:-0,1,2,3}"
IFS=, read -ra d_devices <<< "${DECODE_DEVICES:-4,5,6,7}"
tp=${#p_devices[@]}
if [[ ! "$tp" =~ ^(2|4|8)$ || ${#d_devices[@]} -ne $tp ]]; then
    echo "Use two equal-sized TP groups of 2, 4 or 8 devices" >&2; exit 2
fi
declare -A used
for npu in "${p_devices[@]}" "${d_devices[@]}"; do
    if [[ ! "$npu" =~ ^(0|[1-9][0-9]*)$ || -n "${used[$npu]+x}" ]]; then
        echo "Device lists must contain distinct physical device IDs across both roles" >&2; exit 2
    fi
    used[$npu]=1
done

export PYTHONPATH="$tools_dir${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_SERVER_DEV_MODE=1 VLLM_USE_V1=1 VLLM_ASCEND_BALANCE_SCHEDULING=0 PYTHONHASHSEED=0
export VLLM_ASCEND_SFA_INDEPENDENT_DP=0 VLLM_ASCEND_SFA_SINGLE_AGREEMENT=0
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-1800}
export VLLM_ASCEND_DSA_UNBUNDLE=1 VLLM_ASCEND_DSA_TWO_GROUPS=1 VLLM_ASCEND_DSA_SHARED_POOL=1
export VLLM_ASCEND_DSA_SHRINK_LATENT=2 VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE=0
export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0 VLLM_ASCEND_DSA_DISABLE_TARGET_SLOT_MAPPING=0
export LMCACHE_ASCEND_SPARSE_TRANSFER_TOPK=2048 TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV} HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-256}

if [[ "$role" == prefill ]]; then
    export ASCEND_RT_VISIBLE_DEVICES=${PREFILL_DEVICES:-0,1,2,3}
    export LMCACHE_CONFIG_FILE=${PREFILL_CONFIG:?set PREFILL_CONFIG to the rendered sender YAML}
    export LMCACHE_INTERNAL_API_SERVER_PORT_START=6999
    export VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=true VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS=4
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1 VLLM_ASCEND_DSA_DCP_INDEXER_REQUEST_SHARDING=0
    export VLLM_ASCEND_SFA_STAGED_GRAPH=0 VLLM_ASCEND_SFA_FULL_GRAPH=0
    port=$p_port; dcp=${PREFILL_DCP:-$tp}
    [[ "$dcp" == 1 || "$dcp" == "$tp" ]] || { echo "PREFILL_DCP must be 1 or $tp" >&2; exit 2; }
    extra=(--enforce-eager --enable-chunked-prefill --max-num-batched-tokens 16384
        --additional-config '{"enable_dsa_cp":true,"layer_sharding":["q_b_proj"],"fuse_qknorm_rope":false,"fuse_muls_add":true,"enable_sparse_li_c8":true,"enable_sparse_sfa_c8":false,"ascend_compilation_config":{"enable_npugraph_ex":false}}')
else
    export ASCEND_RT_VISIBLE_DEVICES=${DECODE_DEVICES:-4,5,6,7}
    export LMCACHE_CONFIG_FILE=${DECODE_CONFIG:?set DECODE_CONFIG to the rendered process-mode receiver YAML}
    export LMCACHE_INTERNAL_API_SERVER_PORT_START=7099 LMCACHE_DECODER_REQUIRE_KV_PREFIX=1
    export VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=false VLLM_ASCEND_ENABLE_FLASHCOMM1=0
    export VLLM_ASCEND_DSA_DCP_INDEXER_REQUEST_SHARDING=1 VLLM_ASCEND_DSA_RESIDENT_CACHE=1
    export VLLM_ASCEND_SFA_SHARED_RESIDENT_PLAN=1 VLLM_ASCEND_DSA_RESIDENT_QUERY_MEMBERSHIP=1
    export VLLM_ASCEND_SFA_SHARED_RETRIEVAL_OVERLAP=1 VLLM_ASCEND_SFA_ASYNC_MTP_PREP=1
    export VLLM_ASCEND_SFA_STAGED_GRAPH=1 VLLM_ASCEND_SFA_FULL_GRAPH=1 VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES=1
    port=$d_port; dcp=$tp
    extra=(--max-num-batched-tokens 4096 --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
        --additional-config '{"enable_flashcomm1":false,"enable_dsa_cp":false,"recompute_scheduler_enable":true,"fuse_qknorm_rope":false,"fuse_muls_add":true,"enable_sparse_li_c8":true,"enable_sparse_sfa_c8":false,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}}')
fi

[[ -r "$LMCACHE_CONFIG_FILE" ]] || { echo "Cannot read $LMCACHE_CONFIG_FILE" >&2; exit 2; }
# The production YAML sizes a whole host for one role. Reuse its Mooncake NIC,
# placement and page format, sizing each role's two CPU arenas for this test.
export LMCACHE_MAX_LOCAL_CPU_SIZE=${CPU_GIB_PER_ROLE:-32}
export LMCACHE_SHARED_CPU_CACHE_SIZE_GB=$LMCACHE_MAX_LOCAL_CPU_SIZE
extra_config=$(python3 - "$LMCACHE_CONFIG_FILE" "$role" "${STORE_GIB_PER_ROLE:-32}" <<'PY'
import json, os, sys, yaml
path, role, store_gib = sys.argv[1:]
with open(path) as config_file:
    cfg = yaml.safe_load(config_file)
if cfg.get('pd_role') != {'prefill': 'sender', 'decode': 'receiver'}[role]:
    raise SystemExit(f'{path} has the wrong pd_role for {role}')
extra = dict(cfg.get('extra_config') or {})
extra.update(global_segment_size=int(store_gib) * 1024**3,
             shared_cpu_cache_size_gb=float(os.environ['LMCACHE_SHARED_CPU_CACHE_SIZE_GB']))
print(json.dumps(extra))
PY
)
export LMCACHE_EXTRA_CONFIG="$extra_config"
exec vllm serve "${MODEL:?set MODEL to the checkpoint directory}" --host 127.0.0.1 --port "$port" \
    --served-model-name "${SERVED_MODEL_NAME:-glm}" --trust-remote-code --quantization ascend \
    --tensor-parallel-size "$tp" --decode-context-parallel-size "$dcp" --data-parallel-size 1 \
    --block-size 128 --cp-kv-cache-interleave-size 128 --enable-expert-parallel --no-enable-prefix-caching \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.93}" --max-model-len "${MAX_MODEL_LEN:-84096}" \
    --max-num-seqs 1 --seed 1024 --speculative-config '{"num_speculative_tokens":1,"method":"deepseek_mtp"}' \
    --worker-extension-cls check_banked_dcp_handoff.BankedDcpWitness \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheAscendConnectorV1Dynamic\",\"kv_role\":\"kv_both\",\"kv_connector_module_path\":\"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1\",\"engine_id\":\"local-$role-$port\"}" \
    "${extra[@]}"
