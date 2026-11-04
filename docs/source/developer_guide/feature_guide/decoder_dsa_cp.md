# DSA-CP for the compact-cache decoder

## Scope and baseline

Design recorded before implementation, 2026-11-04. Base is
`feat/decoder-flashcomm1-only` at `667c0c2e03c2c59f6f3176ac445db9b86e70b741`,
whose parent is `0d7c199fdf2a38bbffdaa96ea1f5a763ab52d17c`. Changes belong
to vLLM-Ascend. No branch switch, commit, deployment change or cache-service
interface change is required by this task.

Workspace: `C:/Users/q00856117/Desktop/full-graph-c8-production-integration-20260929`.
The unchanged companion repositories are vLLM at
`db438c3c82bd9d7d9972e765690c8447b5eb77da`, LMCache-NPU at
`e309ab29e6d33736b8798aeecc58df1c01f1577b`, and LMCache-Ascend at
`be18f761111b8dbdf580b82434111558b13589c8`, all on `perf/sfa-dp-sync`.

The supplied decoder uses TP4/DP4/EP, PP=PCP=DCP=1, MTP=1,
PIECEWISE compilation, staged/full SFA graphs, asynchronous MTP preparation,
independent DP, resident/shared-resident plans and retrieval overlap. Both KV
groups, compact cold loading, checkpointing and GC-off ownership must remain
compatible. Target capture request sizes are 2,4,6,8,10,12. Keep npugraph_ex,
static kernels and shared-expert multistream overlap at their supplied false
settings; retain the optional existing npugraph_ex lowering contract. This work
does not resolve the deferred Torch-NPU shared-expert overlap problem.

Decoder limits are 12 sequences, 4096 batched tokens and 1048576 context tokens.
The supplied KV policy uses BF16 latent/RoPE and C8 indexer keys with scales;
storage is the existing layerwise asynchronous process service plus Mooncake.

The deployment model path is GLM-5.3-w4a8c8; its actual checkpoint configuration
and installed native binaries are unavailable locally. Do not substitute the
different upstream GLM-5.3-Flash pooled/NoPE/KDA architecture based on its name.

## Configuration and roles

Explicit `additional_config.enable_dsa_cp=true`, with FlashComm1 enabled,
opts a KV consumer into decoder CP unless the existing layerwise P-node marker
is set. Both supplied roles use kv_both, so is_kv_producer alone cannot select
prefiller layer sharding. Omitted switches and the marked prefiller retain their
legacy behavior. No new knob or public connector API is introduced.

The decoder always keeps the original TP o_proj and its original quantization
parameters. Reuse the existing activation all-to-all before that projection in
target, draft and native recovery. Recovery may have longer queries, but the
same padded token/full-head to global-token/TP-head exchange is valid. Do not
allocate full o_proj weights or mutate parameter bindings on this new path.
Legacy mixed/prefill parameter switching remains outside the change.

## Query and cache layout

Let T be the existing graph/model token count, P=ceil(T/TP)*TP, and L=P/TP.
Rank r computes query rows [rL,(r+1)L), with all attention heads. Communication
padding owns no KV slot and does not change scheduler token frontiers.

Retain global request rows in block tables, resident state, union/remap scratch,
source tables and connector payloads. Gather each indexer producer's raw local
top-k into global token order before the existing publication and union/remap
path. Shared consumers reuse this global publication and shared resident plan;
they must not repeat the gather. Slice the remapped global top-k only at sparse
attention consumption. This is an intentional refinement of the preliminary
CP-local-planner proposal: it reuses the existing native, graph and persistence
contracts and avoids changing request ownership or cache-service behavior.

The extra top-k gather is a real cost, once per indexer producer, and needs
measurement. It preserves global selection/union semantics and resident reuse;
it does not promise a TP-fold reduction of sparse-load traffic or attention FLOPs.

Current latent+RoPE KV and indexer keys/scales are gathered before global cache
publication, as on the native CP path. All ranks publish the complete current
rows before attention, including ranks with no real local queries. Use ordinary
model-stream collectives; introduce none in source preparation or storage work.
Reuse padding-write protection for the graph path. Preserve native save events,
Group-1 load waits, selective-load ordering and graph source leases.

## Metadata, graphs and MTP

Keep the decoder metadata's cos/sin, slot mappings, remap boundaries and request
indices in global row order. CP context carries local slot views and stable local
query/key lengths. Derive local query ends by clipping global ends to the rank's
slice. Key lengths subtract the unowned causal query suffix; zero-KV graph padding
stays zero. Never clamp invalid real-request metadata into apparent validity.
Vectorize this calculation on device, retaining the extra bounded attention
padding sequence and covering TP padding beyond T.

The first two graph bridge tensors use L query rows and all heads; top-k and
request transfer tensors keep global capacities. Fake implementations must match
these shapes without changing the operator ABI. Eligibility checks validate local
query shapes and global cache metadata independently; they must not force a graph
route or drop a configured bucket. Include CP buffers in capture signatures.

Prepare local lengths once per metadata build. Async MTP updates the same lengths
after acceptance-dependent device metadata, before replay; retain the existing
same-stream ordering and snapshot eligibility. Idle clearing must also clear CP
key lengths/local slots. New requests, reordered rows, cold resumes and changed
buckets continue through ordinary preparation. Preserve the existing rollback
and failure agreements.

MTP receives replicated target hidden states and positions. The new decoder CP
path must use the existing local-slice helper, not SUM reduce-scatter. Padding
ranks receive initialized rows; full source buffers remain unchanged. Retain
the legacy P CP route outside this decoder opt-in.

## Upstream provenance

- Adapt [11762](https://github.com/vllm-project/vllm-ascend/pull/11762)
  (`71e0921d5f38c1fbe4e80c822576578df460a1f3`) for vectorized CP lengths and
  [13779](https://github.com/vllm-project/vllm-ascend/pull/13779)
  (`059df1f41933888e1f0f65ba1fd2f0e728a544f5`) for zero-KV graph padding.
  The current loop reproduces key length -1 for a padded candidate layout.
- [11068](https://github.com/vllm-project/vllm-ascend/pull/11068) establishes
  the original TP weight as the persistent source of truth. Our decoder reuses
  its existing activation exchange and never enters full-weight switching;
  therefore it needs neither cloned shards nor the full-weight W4A8 switching
  machinery from [10523](https://github.com/vllm-project/vllm-ascend/pull/10523).
  This narrows the initial backport list based on the actual decoder route.
- The local indexer query extent already follows the intent of
  [10827](https://github.com/vllm-project/vllm-ascend/pull/10827). Global top-k
  publication here is deliberate, downstream of local indexer computation.
- MTP>1 changes in [10878](https://github.com/vllm-project/vllm-ascend/pull/10878),
  the backend split in [13921](https://github.com/vllm-project/vllm-ascend/pull/13921)
  and the different model architecture in
  [17987](https://github.com/vllm-project/vllm-ascend/pull/17987) are not wholesale
  backports for this MTP=1 custom decoder.

## Validation and review

CPU tests must execute production helpers/methods: explicit/omitted role switches,
CP on/off and P unchanged, TP1/TP4 and every rank, Q1/Q2/mixed queries, non-divisible
T, all-padding ranks, causal suffixes, zero-KV padding, global publication and
local attention selection, replicated MTP values, graph fake shapes and stable
metadata after async advancement/idle clearing. Cover shared producer/consumer
publication and BF16/C8 key/scale gathers, ordinary native recovery and unchanged
source/transfer shapes. Extend the installed-model NPU parity harness with CP
selection, preserving existing default modes.

Review the final diff separately for collective order, producer waits, source
ownership, metadata lifetime, padding writes and disabled-mode cost. Add fixes
only for demonstrated defects. Report changed operation counts separately from
measured CPU/NPU performance. No NPU is available on this development host;
CPU stand-ins are not server qualification.

Before deployment, run TP4/DP4 tests on the actual stack for all capture sizes,
idle and unequal DP loads, Q1/Q2, admission/finish, preemption/restore, native
recovery, both groups and GC off. Compare FC1-only against CP using identical
sampling, trace and cache warmth: output parity/finish reasons, MTP acceptance,
TTFT/TPOT tails, throughput, peak graph/HBM memory and preemptions. Retain existing
memory profiling/admission limits: replicated q_b/kv_b projections can consume
more HBM even though decoder o_proj has no extra full-weight buffer.

## Implementation and separate review

Implementation retains the global planner design above. Review identified and
corrected the following concrete integration issues:

- CP metadata must include the bounded graph's extra zero-KV sequence. The old
  loop could compute a negative key length for that sequence and then slice it
  off. Vectorized clipping preserves the sequence and leaves its key length zero;
  invalid nonzero real lengths are not clamped into valid-looking metadata.
- The existing replicated MTP inputs cannot be SUM reduce-scattered on decoder
  CP. TP4 tests verify positions/hidden values, initialized padding and unchanged
  source buffers using the existing local-slice implementation.
- Sparse-attention TND kernels visit declared query intervals. Additional TP
  rows beyond the global token extent can remain unwritten; the decoder clears
  that suffix before value/output projection. A poison-output regression executes
  the real Python attention method for partial and entirely padded ranks.
- Query bridge capacity is local, whereas raw top-k/transfer capacity is global.
  Applying the former to diagnostic top-k reshaping failed. Diagnostic snapshots
  now retain separate query counts and CP metadata; staged fingerprints identify
  local query owners explicitly.
- The initial metadata adaptation unnecessarily copied an already aligned P slot
  mapping. Review restored the original shared storage relationship, including
  one shared padded allocation for legacy P CP views.

The new decoder path avoids full o_proj weight switching in both decode and
mixed/native recovery. Threaded four-rank CPU stand-ins execute production
pre-compute/native preparation and output exchange, covering BF16/C8 indexers,
shared consumers and paired/strided latent views. These are not HCCL tests.
Compiler tests exercise the real fake signatures and shared-buffer lowering via
CPU AOT, including both functionalization versions and the npugraph_ex backend
adapter with a CPU compiler stand-in. They do not compile NPU graphs.

The optional parity driver accepts `--dsa-cp` (which implies FlashComm1). Its
test-only worker gathers CP probe rows into global order before comparison;
these additional observability collectives are absent from serving and make
the driver unsuitable as a performance benchmark. For example:

```bash
python tools/sfa_full_graph_parity.py --model /path/to/model \
    --devices 0,1,2,3 --dsa-cp --compare-output
```

This reduced-model/dummy-weight parity harness supplements the full-server
qualification described above; it cannot replace pretrained-output, EP/DP,
admission/restore, memory-pressure or throughput measurements.

Enable only on the decoder by merging these fields into its existing
`--additional-config` JSON:

```json
{
  "enable_flashcomm1": true,
  "enable_dsa_cp": true,
  "multistream_overlap_shared_expert": false
}
```

Retain the other supplied options and the prefiller's existing environment.
Restart all decoder workers together after installing the matching Python
checkout. No native operator or ABI was changed, so this patch itself does not
require a native rebuild. Actual installed binary identity and CANN 8.5.1 NPU
qualification remain deployment checks, not inferred from this checkout.

Operation-count evidence: CP length preparation has no per-request device
readbacks. Producer top-k is gathered once and shared consumers reuse it; at
24 query rows and top-k 2048, the gathered int32 tensor is 192 KiB per producer
per rank. This is additional communication, not a claim of zero overhead or
measured throughput improvement.

## Validation record

Final available CPU run: **1902 passed, 19 skipped** on Windows/PyTorch 2.10 CPU.
It includes the compilation suite and parity-driver/output/shutdown tests, with
the new CP cases, real cross-repository source/transfer wrappers, threaded TP
stand-ins, async ordering, graph buffer reuse and CPU AOT tests. The skips are
not passes. Separate focused compiler run: 66 passed; parity-driver/probe run:
175 passed (overlapping subsets, not additional totals).

The command excluded `test_acl_graph.py` and `test_npugraph_ex_utils_check.py`,
which require the installed vLLM/NPU environment, and `test_sfa_parity_gloo.py`.
The latter's two real subprocess cases failed in the exploratory Windows run
with `makeDeviceForHostname(): unsupported gloo device`; they remain unverified
here. NPU native compilation, actual capture/replay, HCCL/MC2 distributed parity,
installed binary identity, pretrained model accuracy and production performance
were not run. No skipped/unavailable check is counted as hardware qualification.

Affected stale CPU fixtures were updated to supply the new CP state and the
existing idle/context and TensorMemoryObj dependencies. No production behavior
was changed to suppress those fixture failures. CP compiler cases isolate their
Dynamo caches as separate worker configurations instead of raising runtime
recompilation limits. The separate final review rechecked global source ownership,
native fallback, collective placement, C8 sidecars, metadata reuse, diagnostics
and feature-disabled behavior after implementation.

Python syntax and whitespace checks pass. Ruff findings remaining in changed
files are seven pre-existing import-order/line-length findings, verified against
the baseline; unrelated formatting is retained. No commit, push, branch switch
or deployment mutation was performed.

## Follow-up independent audit, 2026-11-04

The follow-up audit found one introduced regression in the CP-disabled path:
assigning the fresh exec_kv outputs to current_kv retained unused tensors through
query/indexer planning. Deleting that tuple immediately after its CP-only use
restores the baseline lifetime. GC-disabled weak-reference regressions for both
producer and consumer failed before the correction and pass afterward. CP's
required operand aliases remain retained; no wait or collective changed.

Final selected CPU validation: **1904 passed, 19 skipped**, with the same native
environment/Gloo exclusions described above. Three independent reviewers checked
the final correction. A separate 1800-case CPU interval oracle covered TP1/2/4/8,
query splits and sequence-length boundaries through 1M. Neither test set measures
NPU memory or performance.

One inherited concern remains unverified: idle metadata can keep nonnegative
synthetic row owners while clearing scatter slots to -1. The CPU path reproduces
that operand, but the deployed CANN negative-index behavior and any device fault
have not been established. No speculative clamp or owner mutation was made.
Actual installed binary identity, NPU capture/replay, distributed ordering and
production performance still require hardware qualification.
