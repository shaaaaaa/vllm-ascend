# Single-forward SFA graph (experimental)

This opt-in captures the **target model forward**, including per-layer sparse
KV loads, in one ACL graph. Each target forward replays one graph. It does not
put prefill, sampling, MTP draft forward, or the entire generation loop into
that graph. Native NPU and TP numerical/performance validation is required
before production use.

## Enable

Use the matching `feat/decode-full-graph` branches of vLLM, vLLM-Ascend,
LMCache, and LMCache-Ascend. vLLM itself has no source changes. Keep the
existing GLM-5.1-w4a8 TP8, max-num-seqs=1, MTP-one-token launch configuration,
then add:

```bash
export VLLM_ASCEND_SFA_STAGED_GRAPH=1
export VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES=1
export VLLM_ASCEND_SFA_FULL_GRAPH=1
```

`VLLM_ASCEND_SFA_FULL_GRAPH` is non-sensitive, accepts `0` (default, original
staged path) or `1`, and requires a worker restart. Keep the existing PIECEWISE
**compilation configuration**: FX partitions are still used to compile the
model, but their ACL wrappers are bypassed during root capture. Runtime does
not replay those partitions: it replays one outer ACL graph covering all of
them and all target-layer transfers. Do not switch `cudagraph_mode` to FULL.

The initial scope is DP=1, max-num-seqs=1, unbundled two-group SFA,
SHRINK_LATENT=2, shared local CPU cache, and Q1 or fixed MTP Q2. TP is supported
by the design and must be tested on the serving hardware. Keep your existing
`LMCACHE_ENABLE_SHARED_CPU_CACHE=true`, strict shared-cache configuration,
256-token chunks/window, and LMCacheAscendConnectorV1Dynamic. Unsupported
decode routes fail explicitly; they do not silently use per-layer splits.

Both Q1 and Q2 graphs are captured at startup when MTP is configured. A shape
change chooses the appropriate whole graph; a 256-token frontier change does
not cause recapture. Graph-memory profiling uses disposable graphs and source
tables, cleared before the temporary KV cache is released.

## Execution contract

1. Before forward, LMCache resolves the request, acquires source ownership,
   prepares CPU chunk pointer tables, and materializes the indexer cache if
   needed. Initial bootstrap uses empty **latent** selections: it does not
   guess or precompute top-k and does not load all latent history into HBM.
2. The target graph computes each layer's live top-k, plans resident misses,
   copies the selected CPU KV, and performs sparse attention, all in compute
   stream order. No target-layer generator is advanced during replay.
3. A single model-boundary stream fence protects source leases. Existing save
   callbacks and MTP draft retrieval remain outside the target graph. Prepared
   draft generators start at the first draft layer, not layer zero.

The transfer uses two single-plane kernels (K and PE) inside the same graph.
This keeps the existing native kernel ABI while allowing partial-tail PE
offsets, source pointers and token limits to change via fixed device buffers.
This is not a claim that two kernels are optimal: benchmark the added kernel
cost against the removed host dispatch overhead.

Q1 temporarily uses the ordinary sparse planner in an MTP-Q2 configuration;
the existing resident registry invalidates incompatible scratch state before
Q2 resumes. Zero-frontier steps disable copies on-device. Request changes
replace pointer-table contents without retaining the previous request lease.

Per-layer target host tensor probes cannot run during a replay. The startup
log explicitly reports their omission. Connector/window boundary diagnostics
and the existing `MTP_DW_DIAG` / `MTP_DW_DEEP_DIAG` settings remain available.
The separate `VLLM_ASCEND_MTP_DRAFT_DEBUG=1` mode is rejected for this path.

## Server validation

First rebuild/install the matching repos using your normal environment. The
native kernel ABI is unchanged; use the compiled extensions for these bases.
Before starting the large model, run from LMCache-Ascend:

```bash
pytest -q tests/v1/test_sparse_graph_transfer_npu.py
```

This captures real top-k and native copies once, then changes top-k, source
pointers, 256/512-token frontiers, a partial tail, and the request. It checks K
and PE values and untouched destination slots. A failure here is a blocker
for serving with `SFA_FULL_GRAPH=1`.

Start the server and send a request, for example:

```bash
curl http://127.0.0.1:9000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.1","prompt":"Explain matrix multiplication.","max_tokens":128,"temperature":0,"seed":0}'
```

Check for both startup and actual replay messages:

```text
[SFA full graph] target_layers=... keys=2 graphs_per_forward=1 layer_transfer_splits=0
[SFA full graph] first target replay: ... layer_transfer_splits=0
```

A short request verifies dispatch, not offload correctness. Also use a prompt
longer than the 4096-token MTP scratch prefix (within max-model-len=10000),
generate at least 512 tokens, and send a second distinct request. Compare
deterministic output/token IDs and TPOT against a separate restart with
`VLLM_ASCEND_SFA_FULL_GRAPH=0`, keeping every other option unchanged. Inspect
the first decode, committed 256-token boundaries, and request replacement.

In a CPU+NPU profiler trace, each `sfa_full_graph::target_replay` scope must
contain exactly one target ACL model execute call. Target-layer top-k, copies
and attention must be in that graph; there must be no target-layer
`sfa_cross_layer::lmcache_retrieve` Python scopes between replay calls. Draft
and save callbacks outside the target-forward scope are expected. Startup
logs and CPU unit tests alone do not prove graph capture or output parity.
