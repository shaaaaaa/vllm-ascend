# ondemand-selected prefill

ondemand-selected reduces main-KV history transfers for DSA requests with long
cached histories and short new queries. The indexer still loads its full history;
the ondemand-selected path builds a union of TopK rows on the NPU and gathers only those
main-KV rows from registered CPU memory into the existing P-node bank layout.
Other requests keep the native layerwise load path, including mixed batches.

## Enablement

The feature is disabled by default. Enable it before engine initialization:

```bash
export VLLM_ASCEND_ENABLE_SELECTED_KV_PREFILL=true
```

The variable accepts `true` or `false`, case-insensitively, and contains no
sensitive data. An explicit `selected_kv_config.enabled` boolean in
`additional_config` overrides the environment default:

```json
{"selected_kv_config": {"enabled": true}}
```

The engine reserves workspace before sizing the KV cache. Changing this setting
requires restarting the engine. Enabling it does not guarantee that every request
qualifies for selected loading; the planner classifies each step and retains
native loading where needed.

## Supported deployment

- Ascend A2 or A3 builds with at least 32 vector cores available to the process.
- Eager v1 model runner on a layerwise-prefill P node with at least two banks.
  Existing DSA unbundling, two KV groups and shared-pool configuration are required.
- Exactly one producer-capable LMCache connector, directly or inside a
  MultiConnector, using shared CPU KV sources and compatible source hooks.
- Native two-plane main KV: BF16 or FP16 with 512 non-RoPE and 64 RoPE elements
  per row, 128-token pages and indexer TopK width 2048.
- At most 16,384 scheduled tokens and 256 requests per step. Selected histories
  must be between 16,384 and 524,288 tokens; request selection also depends on
  query/history geometry and source-cache availability.
- Optional `mtp` / `deepseek_mtp` speculation with exactly one speculative token.
  Target and draft passes share a deferred source lifetime.
- Pipeline, prefill-context and decode-context parallel sizes must all be one.
  Encoder-decoder models, sleep mode, DBO and the v2 model runner are unsupported.
- DSA latent offload must be disabled; it bypasses the attention path that
  performs selected loads, so enabling both is rejected at startup.
- DSA adapter cache (`VLLM_ASCEND_DSA_USE_ADAPTER_CACHE`) must also be
  disabled. A one-token cached-prefill tail can use the adapter decode path,
  bypassing the selected load while still advancing selected prefetch.
  Enabling both is rejected at startup.

This is an opt-in prefill path. It does not enable decode full graph or main-KV
INT8 storage, and weight quantization is independent of the KV dtype.

## Build and compatibility

Build the Ascend extensions from this source checkout with the matching Torch,
Torch-NPU and CANN toolchain. The build produces the selected union and in-place
MLA gather libraries from `csrc/sparse_prefill_native`.

The implementation originated from vLLM-Ascend `5258cccf` and is integrated with
the current main runner. The original companion snapshots were vLLM `a386a401`,
LMCache `c310c328`, and LMCache-Ascend `7c9a62d8`; these identify the source
contract, not an assertion that every newer combination has passed NPU validation.
The connector checks required methods and address-binding hooks at startup and
fails if the installed LMCache integration cannot provide them. Callable hooks
alone do not establish binary, layout or asynchronous-lifetime compatibility.

## Validation and limitations

The accompanying CPU tests cover planner decisions, source ownership, metadata
alignment, event-generation filtering and host execution of kernel logic through
AscendC stubs. They do not emulate real NPU DCache or asynchronous device timing.

Before production use, validate the exact four-package combination on NPU hardware:
compare selected OFF/ON using identical prompts, model weights and scheduler
settings; check output/logprob finiteness and numerical agreement; exercise mixed
batches, partial prefix hits, target/draft passes and repeated buffer reuse across
history lengths. Measure device forward and end-to-end timings separately.

Earlier W4A8 selected runs on the original baseline produced NaNs for a
64K-history, 128-query, batch-eight case. This port includes subsequent scalar
DCache, long-union and buffer-reuse fixes, but those changes alone do not prove
the original NaN is resolved. A successful build or matching first token is not
full numerical acceptance. Keep the feature disabled for deployments that have
not passed their required numerical and lifetime validation.
