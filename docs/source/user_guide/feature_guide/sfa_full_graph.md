# Single-forward SFA graph (experimental)

This opt-in captures the **target model forward**, including per-layer sparse
KV loads, in one ACL graph. Each target forward replays one graph. It does not
put prefill, sampling, MTP draft forward, or the entire generation loop into
that graph. Native NPU and TP numerical/performance validation is required
before production use.

## Enable

Use the matching `feat/decode-full-graph` branches of vLLM, vLLM-Ascend,
LMCache, and LMCache-Ascend. Keep MTP at one speculative token. For the
TP4/DP4 P/D deployment with `max-num-seqs=16`, use on D workers:

```bash
export VLLM_ASCEND_SFA_STAGED_GRAPH=1
export VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES=4,8,12,16
export VLLM_ASCEND_SFA_FULL_GRAPH=1
```

`VLLM_ASCEND_SFA_FULL_GRAPH` is non-sensitive, accepts `0` (default, original
staged path) or `1`, and requires a worker restart. Keep the existing PIECEWISE
**compilation configuration**: FX partitions are still used to compile the
model, but their ACL wrappers are bypassed during root capture. Runtime does
not replay those partitions: it replays one outer ACL graph covering all of
them and all target-layer transfers. Do not switch `cudagraph_mode` to FULL.

The graph supports internal DP/TP/EP, unbundled two-group SFA,
SHRINK_LATENT=2, and batches mixing Q1 and Q2 requests. Capture sizes are
request capacities: with MTP=1, the above captures 8/16/24/32-token graphs.
Smaller batches pad to the next request bucket. A singleton deployment can
still use `max-num-seqs=1` and capture size `1`. LoRA, PP, CP and the external
launcher remain unsupported. The connector must provide pinned CPU sources
(local or shared); remote Mooncake history must be materialized before replay.
The existing LMCache configuration file is retained, not replaced.

P workers using `--enforce-eager` ignore the full-graph flag, so exporting it
globally in a P/D launch template is safe. D workers must enable staged SFA and
use PIECEWISE compilation. `recompute_scheduler_enable=true` is supported:
target DP coordination is not skipped even when the ordinary MC2 optimization
would skip it. Idle replicas replay a masked graph at the agreed capacity,
without changing live request block ownership or writing dummy KV.

If any EP-participating DP rank has prefill/recompute work, the whole DP group
uses the coordinated eager path for that iteration; it is not a decode-only
group iteration. Once all ranks are decoding or idle, target forwards use one
root replay. MTP draft and prefill remain outside the target graph. Thus this
does not claim one replay on a decoding rank while a peer is doing prefill.

Each request bucket captures one bounded-query topology at startup. The same
graph handles Q1, Q2 and mixed batches: device-only packing groups live top-k
by request for the sparse planner, then restores packed token order. An
additional attention-only sequence absorbs padding without extending the last
real request's causal query length. A 256-token frontier change does not cause
recapture. Graph-memory profiling uses disposable graphs and source
tables, cleared before the temporary KV cache is released.

## Execution contract

1. Before forward, LMCache resolves the request, acquires source ownership,
   prepares per-request CPU chunk pointer tables, and materializes the indexer cache if
   needed. Initial bootstrap uses empty **latent** selections: it does not
   guess or precompute top-k and does not load all latent history into HBM.
2. The target graph computes each layer's live top-k, plans resident misses,
   copies the selected CPU KV, and performs sparse attention, all in compute
   stream order. No target-layer generator is advanced during replay.
3. A single model-boundary stream fence protects source leases. Existing save
   callbacks and MTP draft retrieval remain outside the target graph. Prepared
   draft generators start at the first draft layer, not layer zero.

The transfer uses two batched single-plane kernels (K and PE) inside the same graph.
This keeps the existing native kernel ABI while allowing partial-tail PE
offsets, source pointers and token limits to change via fixed device buffers.
Each request has a separate virtual chunk-address range; no-history and padded
lanes are masked on device. Source, metadata and input-signature errors are agreed
across TP/DP before any worker enters captured collectives.
This is not a claim that two kernels are optimal: benchmark the added kernel
cost against the removed host dispatch overhead.

Q1 and Q2 share request-major planner lanes and resident state in the bounded
topology. Zero-frontier steps disable copies on-device. Request changes
replace pointer-table contents without retaining the previous request lease.

Per-layer target host tensor probes cannot run during a replay. The startup
log explicitly reports their omission. Connector/window boundary diagnostics
and the existing `MTP_DW_DIAG` / `MTP_DW_DEEP_DIAG` settings remain available.
The separate `VLLM_ASCEND_MTP_DRAFT_DEBUG=1` mode is rejected for this path.

## Server validation

Install the matching repos using your normal environment. This batched extension
changes Python code only; an existing editable installation of these branches
does not need another native rebuild. Keep the compiled extensions for these bases.
Before starting the large model, run from LMCache-Ascend:

```bash
python -m pytest -q --confcutdir=tests/v1 tests/v1/test_sparse_graph_transfer_npu.py
```

This captures real top-k and native copies once for 1/4/16 request lanes, then
changes top-k, source pointers, 256/512-token frontiers, a partial tail, and the request. It checks K
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
[SFA full graph] target_layers=... keys=4 graphs_per_forward=1 layer_transfer_splits=0
[SFA full graph] first target replay: ... layer_transfer_splits=0
```

A short request verifies dispatch, not offload correctness. Also use a prompt
longer than the 4096-token MTP scratch prefix (within the configured model limit),
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
