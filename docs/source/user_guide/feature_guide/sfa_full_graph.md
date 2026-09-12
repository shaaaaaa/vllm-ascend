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
Synthetic warmup/capture sequence lengths are bounded by `max_model_len` and
both KV groups' logical block-table capacities. The 6144-token workspace
heuristic is not a minimum context length; live request lengths are unchanged.

## Execution contract

1. Before forward, LMCache resolves the request, acquires source ownership,
   prepares per-request CPU chunk pointer tables, and materializes the indexer cache if
   needed. Initial bootstrap uses empty **latent** selections: it does not
   guess or precompute top-k and does not load all latent history into HBM.
2. The target graph computes each layer's live top-k, plans resident misses,
   copies the selected CPU KV, and performs sparse attention, all in compute
   stream order. No target-layer generator is advanced during replay.
3. A completion event protects independently retained source allocations;
   replay returns without a CPU stream synchronization. Existing save
   callbacks and MTP draft retrieval remain outside the target graph. Prepared
   draft generators start at the first draft layer, not layer zero.

The transfer uses two batched single-plane kernels (K and PE) inside the same graph.
This keeps the existing native kernel ABI while allowing partial-tail PE
offsets, source pointers and token limits to change via fixed device buffers.
Each request has a separate virtual chunk-address range; no-history and padded
lanes are masked on device. Source, metadata and input-signature errors use
error-only worker fail-stop for live, single-node TP-only `mp` execution;
startup capture and other topologies agree across TP/DP before entering
captured collectives. See the failure-handling details below.
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

## Eight-layer performance and MindStudio profile

Use the independent benchmark, **not the numerical-parity driver**, to measure
the optimization. From vllm-ascend on the existing eight-NPU host:

```bash
set -o pipefail
python tools/sfa_graph_benchmark.py --profile 2>&1 | tee log.log
```

No HTTP server or client is needed. The four matching repositories and their
existing native extensions must already be installed. This Python-only tool
does not require a new native build. Defaults are the local
`/workspace/models/GLM-5.1-w4a8` configuration, **TP8/DP1, eight target layers,
MTP1, dummy weights, 30000 input tokens and 512 output tokens**. It preserves the
parity fixture's MTP quantization remapping and deterministic integer weight
initialization, but does **not** install parity hooks, tensor snapshots,
checkpoint save/restore, per-layer fences or replacement sampling. EP,
FlashComm and sequence parallelism remain disabled, as in the parity fixture.

The driver starts and fully closes two engines sequentially:

- `staged`: the original staged retrieve-split path, `SFA_FULL_GRAPH=0`.
- `full`: the single-target-forward path, `SFA_FULL_GRAPH=1`.

Both retain `SFA_STAGED_GRAPH=1` and PIECEWISE compilation; the baseline is
**not enforce-eager**. By default each engine performs exactly **one request**,
with **zero extra warmup requests** (`--warmups 0 --repeats 1`). Prompts/seeds
match across modes. EOS is ignored to fix the output length. Startup model
initialization and mandatory graph capture still run. To explicitly restore
the longer benchmark, pass `--warmups 1 --repeats 5`.
Prefill executes normally in each engine, but is excluded from decode timing.
This is a performance experiment, **not** the one-prefill numerical comparison.
The 30000-token length is the **input context**, not 30000 generated tokens.
Prefill remains chunked at 512 tokens to bound temporary memory. Use
`--prompt-tokens 4351` to repeat the earlier short-context workload; do not
attribute differences between 4351-token and 30000-token runs to this patch.

TPOT is measured at the offline engine's output boundary: elapsed time from
the first committed-token emission to the last, divided by the number of
additional committed tokens. This handles MTP multi-token emissions and includes
scheduling, IPC, MTP, sampling and cache work, not just target graph execution.
Startup, warmup, prefill and the separate profiler request are not timed as
decode. Worker state/synchronization RPCs run only outside timed requests.
Without `--diagnose`, measured requests have no per-layer or per-step benchmark
instrumentation. With `--diagnose` and the single-request defaults, that sole
request also supplies the stage statistics; its TPOT is labelled as including
diagnostic overhead. With `--diagnose`, supported captured timing markers also
remain in graphs during explicit multi-request measurements; diagnostic runs
are conservatively labelled instrumented even if these markers are unavailable.
Ordinary serving is never instrumented by this tool.

Full-graph source binding now compares ordered request IDs and immutable
`PreparedSparseSource` snapshot identities **before** enumerating transfers.
Unchanged sources skip every layer's `bind_batch`: no source-table clearing,
chunk-count uploads, PE-pointer arithmetic or pointer copies. Newly published
windows, source replacement/restore, changed request lanes and empty batches
rebind; source references are retained to prevent identity reuse. Graph keys
sharing request capacity share a last-binding cache, reset with startup capture.
This does not freeze top-k, suppress actual graph-internal KV loads, or bypass
per-step attention metadata/address validation. Those metadata checks still
visit layers; this optimization removes the **source rebinding** loop and its
device work, not every graph-external Python loop.

Full-graph validation now removes duplication **within a forward**, without
memoizing mutable metadata across steps:

- Input address/layout validation runs once, before the existing error
  agreement/fail-stop boundary. `prepare_run()` hands `run()` a single-use
  validated call containing the actual model kwargs. Another context, graph
  key, owner, reset generation or entry cannot reuse it; overriding its inputs
  or replaying it twice is rejected. The caller must not mutate input layouts
  between these adjacent preparation and launch phases.
- The signature walk inspects each identical Tensor object once per call.
  Distinct views are checked independently, even if their pointers match.
- Shared attention-metadata eligibility is checked once per implementation
  type, metadata object, graph key and resident-state policy. A fresh memo is created every forward;
  layers with different metadata or stricter resident requirements get their
  own check. Layer-local KV layout/dtype/configuration checks remain in place.

Replay no longer calls `current_stream().synchronize()`. A changed source batch
acquires independent `TensorMemoryObj` references once (including rank 0's real
shared-slab allocations and passive-rank views). Each replay records a completion
event. On replacement or request finish, the old batch enters a retirement queue;
nonblocking event queries release completed batches. Unchanged batches neither
rescan owners/layers nor wait. Tensor-only and proxy/no-op-refcount sources are
rejected before replay because Python references cannot prevent allocator reuse.

Pointer uploads and replay must remain on the same runner stream. Existing KV
stores enqueue `store_stream.wait_stream(current_stream)`, so removing the CPU
fence does not permit stores to overtake graph writes. Sampling/readbacks and
store publication can still wait for their actual data dependencies. Reset and
shutdown drain work before releasing leases or closing the allocator. A failed
submission retains owners until teardown; a non-completing retirement backlog
is bounded at 64 batches and fails closed, not by freeing in-flight memory.

For **single-node, TP-only, DP1/PP1 `mp` worker processes**, live target decode
no longer performs the pre-replay TP CPU error all-reduce. Source and metadata
validation still run locally. A preparation failure logs the original traceback
and raises `SystemExit`, bypassing the worker RPC loop's catch-and-continue
handler. The existing independent process-sentinel monitor then shuts down the
owned peer workers, even if their computation threads are blocked. Only on
failure, a five-second daemon timer is armed to exit the failed process if its
cleanup stalls. Healthy forwards start no timers and perform no health polling
or error-agreement communication. This is fail-stop requiring engine restart,
not retry or recovery of the failed forward.

Startup capture, multi-node/multi-DP jobs and other/custom executors retain
the existing synchronous error agreement. This change does not remove model TP collectives or replace
HCCL/native hang detection when no Python preparation error has occurred.
CPU tests in `test_sfa_fail_stop.py` execute the sibling vLLM RPC loop and
sentinel-monitor methods with two/eight real child processes, including a
nonzero-rank failure, blocked cleanup and unrelated-process isolation. They
verify process supervision, **not native HCCL cancellation on an NPU**.
`full.json` records `source_binding_updates_per_rank` beside
`root_replays_per_rank`, sampled only before/after each request. The timing line
also shows rank 0's counts. Long steady requests should have many more replays
than source updates; frequent updates require investigating source publication,
not assuming the cache is effective. These counters cover the whole request,
not just the interval between first and last token emissions.

`[SFA_BENCH]` prints mean/median/standard deviation of request TPOT and:

```text
reduction = (staged_mean_TPOT - full_mean_TPOT) / staged_mean_TPOT * 100%
speedup   = staged_mean_TPOT / full_mean_TPOT
```

For one request, `count=1` and `std=null` (`n/a` in the log): there is no
between-request variability estimate. Host diagnostic statistics still
cover all measured decode forwards within that request; captured graph events
describe only its last decode, as labelled in the log. With no request
warmup, first-request lazy setup may affect the result; this is a quick
comparison, not evidence of multi-request stability.

A negative reduction is a slowdown. `tokens_equal=false` is reported explicitly:
different outputs can change MTP acceptance or MoE routing and confound timing.
Even matching dummy outputs do not prove numerical correctness. These results
only describe this truncated dummy workload, not full-depth production speed.
For a performance-only repeat in reverse order, use:

```bash
python tools/sfa_graph_benchmark.py --order full,staged 2>&1 | tee log.log
```

For a rigorous performance study, use repeated runs in both orders to distinguish gains from run-to-run noise,
thermal state or cache effects; there is no hard-coded performance pass threshold.

### Compact timing diagnostics without a profiler

For a regression on the 5000-token workload, keep that same length:

```bash
set -o pipefail
python tools/sfa_graph_benchmark.py --prompt-tokens 5000 --diagnose 2>&1 | tee log.log
```

`--diagnose` and `--profile` are mutually exclusive. This mode never starts
the profiler, exports a trace, or calls the trace analyser. With the defaults,
each engine runs only **one 512-token generation**, and temporary timing wrappers
collect statistics during that same request. There is no hidden warmup or
second diagnostic generation. They are restored afterwards. Both the log and
`comparison.json` flag that TPOT includes diagnostic overhead; it is not a clean
performance measurement. If the runtime passes the captured-event capability
check, timing records are added inside existing opaque SFA and TP operations at
startup capture; replay has no new per-layer Python
callbacks or graph splits. KV values and sampling computations are unchanged.
With explicit `--warmups 1 --repeats 5`, host diagnostics run once after the
measured requests, but supported captured markers remain in all requests. Omit
`--diagnose` for clean performance measurements.

The worker uses scheduler `num_computed_tokens` and `num_output_tokens` to
exclude prefill, including a final one-token prefill chunk. It does **not**
guess decode from query length or arm timing with a mid-request RPC. Start/stop
RPCs and their synchronization happen outside the diagnostic request. There
are no new per-layer or per-step synchronization calls. Coarse current-stream
events cover target-forward, root-replay, logits and MTP boundaries. Fine host-side
events for sampling substages and staged retrieval are limited to the first
four decode forwards and then every 32 forwards. Host wall/CPU statistics still
cover every decode forward. All event timestamps and elapsed times are read
after the request finishes. Event storage is bounded;
`event_drops` warns if an unusually long diagnostic request exceeds the cap.

A tiny startup check, before loading weights, captures and replays timing
events twice to verify that this torch_npu/CANN version refreshes their
timestamps. A known optional timestamp limitation (including
`Event.recorded_time()` reporting `event recorder null`, code `507000`), a missing
timestamp API or stale timestamps disables graph-internal markers on **all TP
ranks**. `query()==True` alone does not establish that a readable recorder exists.
The log and JSON explicitly report `graph_phases UNAVAILABLE`; no per-layer
numbers or success claim are invented. Detailed host sampling and graph-external
NPU intervals continue without a profiler or an eager-model fallback. This
reduced mode cannot identify individual graph-internal kernels or transfer
durations. Actual capture/replay/synchronize failures and unrelated runtime
errors still stop startup; they are not swallowed as capability limitations.
The check and TP agreement happen at startup only and do not run an extra model
request or add a per-forward collective.

This is an event-readback limitation, not evidence of broken KV or model output.
An external synchronization event is not a substitute timing event: the
[CANN external-record API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/API/runtimeapi/aclcppdevg_03_2282.html)
requires a synchronization-only event flag for that path.

Both modes print compact `[SFA_TIMING]` lines directly to `log.log`. To share
only the summary, without any large trace:

```bash
grep -aF '[SFA_TIMING]' log.log
```

Each rank reports actual decode forwards, root replays, source binding updates,
query-size and sampled-token-count histograms. The tool rejects missing ranks,
inconsistent forward counts or a full-mode forward without a root replay.
`committed/forward` and the histograms help identify changed MTP work even when
the final token IDs match. Sampled worker tokens can exceed final committed
tokens at the generation length boundary; this is not an exact acceptance-rate
counter. Source update/root counters span the diagnostic request; phase timings
and histograms exclude prefill.

Per-stage lines include `wall`, `self`, `cpu`, `call_max`, and optional `stream`:

- `wall`: inclusive host elapsed time, normalized by target forwards.
- `self`: host elapsed time excluding measured child scopes. Use this to avoid
  counting source preparation, metadata and replay time twice.
- `cpu`: exclusive CPU time of the worker's execution thread, not process CPU
  usage. A large wall/self time with low CPU can be waiting, driver activity or
  descheduling; it does not identify a particular kernel.
- Numbers such as `1.200(1.800)` are the rank mean and the largest rank mean in
  ms/forward. `call_max` is the slowest individual host call in milliseconds.
- `stream`: mean NPU current-stream elapsed time per recorded interval, including dependencies and host
  submission gaps. It is **not pure kernel time**, and overlapping intervals
  must not be summed. `stream_samples` gives the actual event sample count;
  sampled totals are never divided by all decode forwards. With nonzero
  `event_drops`, it is only partial coverage.

When captured timestamps are supported, `graph_last.L0` through `graph_last.L7`
split the **last target decode only**:

- `pre`: projections, KV/indexer update, indexer and sparse-index preparation.
  Nested `indexer` and `select` identify top-k and its mapping/deduplication work.
- `pre_to_post`: the gap between pre-compute and post-compute, including bridge
  copies, retrieval and dependencies/host submission gaps.
- `transfer`: full graph's captured sparse KV transfer. Staged retrieval is
  outside the graph, so this field is `unobserved`; use `pre_to_post` and the
  sampled `retrieve.L*` stream intervals for that mode, not a zero estimate.
- `post`: sparse attention and output projection, including nested `attention`.
- `after_post`: end of this attention's post-compute to the next attention's
  pre-compute, exposing FFN/MoE, residual/norm, TP and submission gaps outside
  SFA. The last layer extends to target-forward completion. This is not an
  isolated FFN kernel time.
- `graph_last.TP.*`: captured model collective spans, including dependencies,
  not the removed CPU error-check collective. Fused/custom paths bypassing the
  observed group methods are `unobserved`, never claimed to have zero cost.

These graph events are overwritten on replay, so they are **not request-wide
averages**. Only timestamps enclosed by the last live target-forward boundary
are accepted; other graph keys, capture-only and MTP timestamps are excluded.
`complete_ranks=8/8` means all required layer phases and gaps were observed exactly
once on every rank. `INCOMPLETE` lists missing or duplicate phases; do not treat
such output as complete localization. Nested phases and TP spans must not be
added together. No per-layer completion waits are introduced to obtain them.

`sampling.bonus_index`, `bonus_sampler`, `target_index_cast`,
`logits_processors`, `sampling_constraints`, `rejection_kernel` and optional
`logprobs` split rejection sampling using its existing scoped recorder. Nested
`sampling.sampler.*` rows identify sampler processors, sampling and logprob
gathering. They call the original operations once, without reading tensor
values or forcing data readiness. `target.compute_logits` isolates logits work;
`target.before_replay` uses existing events to show the current-stream interval
between target entry and root submission, including preparation gaps.

Useful comparisons are `source.prepare`, `source.bind`, `metadata.L0` through
`metadata.L7`, `signature.validate`, `target.forward`, `root.replay_submit`,
`mtp.propose`, `sampling`, and `bookkeeping`. Staged retrieval has one
`retrieve.L*` row per target layer, with KV waits distinguished from MTP waits.
Full mode should not execute these Python retrieval callbacks in target
replay. `signature.validate` must now run exactly once per full target forward;
the diagnostic rejects repeated or missing calls. `metadata.shared_check`
counts the actual shared eligibility checks: normally one per forward when
all eight layers share metadata and resident-state policy, not eight.
`metadata.L*` still measures each layer's preparation, including its necessary
layer-local KV checks. Its inclusive time includes `metadata.shared_check`
when that layer is the first consumer; do not add those nested times twice.

`root.run` exclusive time now covers completion-event recording and bookkeeping,
not a post-replay CPU completion fence. The single input validation runs before
`root.run`, under `target.forward`. Older results included a blocking fence here.
Some wait time may move to real downstream sampling/readback dependencies;
removing this fence alone does not prove a TPOT gain. Compare target/root
stream spans and MTP/readback waits. `engine_minus_worker` is a
signed residual including scheduling, IPC, idle gaps and interval-boundary
skew, not a precise scheduler measurement or a sum across TP ranks.

Small `staged-timing.json` and `full-timing.json` files retain each rank's
per-call mean/standard deviation/max alongside the existing benchmark JSON
under the printed `profile/sfa-.../` result directory. No large profile is
created. In single-request diagnostic mode, `comparison.json` contains the same
request's TPOT with `instrumented_measurements=true`; comparisons mixing an
instrumented and uninstrumented mode are rejected. In explicit multi-request
mode, the additional host diagnostic request is excluded from the TPOT samples,
but supported captured markers still contribute overhead and diagnostic runs
remain labelled instrumented. Local CPU tests cover gating, nesting, asynchronous replay call
order, stale timestamp rejection, phase coverage, sampled-event normalization,
wrapper restoration, the server's `query()==True`/`event recorder null` failure,
all-rank capability agreement, propagation of unrelated device errors,
sampling-forward equivalence against the sibling vLLM implementation and driver
orchestration. Real captured-event support and NPU
timings still require this host run.

With `--profile`, each engine makes one additional request **after its timed
requests**, starts the profiler after eight committed output tokens (past
prefill), and finishes after 32 additional output tokens. The control RPC can
take effect a few steps later; this is not an exact step-count capture. Profiled
request timings are deliberately discarded. Both engines exit before offline
trace parsing, which uses at most two analysis processes. The generated path is
printed as `profile/sfa-.../`; new runs never overwrite earlier traces:

```text
comparison.json           TPOT comparison, configuration and output-match flag
staged.json / full.json    samples, token IDs and timing-instrumentation flag
staged/ / full/            rank-specific MindStudio profile directories
*-trace-check.json         per-rank CPU-scope audit and trace paths
```

Open the `full/` and `staged/` profiles in MindStudio Insight. In **each rank's
steady-state decode**, locate `sfa_full_graph::target_replay`:

1. There should be one target ACL model-execute submission per root scope.
2. Follow its device execution and check that all eight target layers' top-k,
   KV transfer and attention are covered, without target-layer Python retrieval
   gaps. Many kernels inside one graph are expected.
3. Do not count prefill, MTP draft, sampling or cache saves outside the target
   scope as splits of that target graph.

The tool checks every rank's trace. `ONE_EXECUTE_PER_ROOT` means the recorded
**CPU scopes** each contain one outermost known ACL execute API; nested API
aliases are not double-counted. This alone does not prove device-side coverage
or KV correctness. `SPLIT_DETECTED` fails the trace check. `UNVERIFIED` is **not
a pass**: for example, a CANN version may place runtime events under remapped
pid/tid lanes, or omit the known API names. Keep the traces and inspect their
device/runtime correlations manually; the checker never equates unrelated
threads just because their timestamps overlap. Missing/duplicate rank traces
also fail. Existing performance JSON is retained if subsequent profiling fails.

If the CPU trace check fails after export, **do not rerun the model**. Recheck
the latest profiled run using its existing `trace_view.json` files:

```bash
python tools/sfa_graph_trace.py --latest profile 2>&1 | tee -a log.log
```

To select an older run, replace `--latest profile` with
`--run-dir profile/sfa-...`. The default is eight ranks; use `--ranks N` for a
different TP size. This command does not import vLLM/torch_npu, capture, or
re-export anything. It only regenerates `*-trace-check.json`; original trace
and performance files remain unchanged. The parser accepts numeric strings
and JSON numbers for timestamps/durations, preserves decimal boundary
precision, and reports malformed timing as `UNVERIFIED`, never silently
drops it to claim success.

## Eight-layer numerical parity (one host, eight NPUs)

To check data rather than generated text, run from **vllm-ascend**:

```bash
set -o pipefail
python -m pytest -q -s --confcutdir=tests/e2e/multicard tests/e2e/multicard/test_sfa_full_graph_parity.py 2>&1 | tee log.log
```

This requires the existing native extensions and the local model configuration
at `/workspace/models/GLM-5.1-w4a8`. It does not need an HTTP server or a client.
All four repositories (`vllm`, `vllm-ascend`, `LMCache`, `LMCache-Ascend`) must use
the matching `feat/decode-full-graph` code; updating only vllm-ascend is not
sufficient. In particular, LMCache-Ascend needs `SparseGraphTransfer`'s
`request_capacity` argument and `bind_batch`, and both connector layers need
the ordered `request_ids`/`frontiers` preparation API.

Before either engine starts, an isolated preflight process checks the actual
imported Python interfaces and required LMCache-Ascend native exports. It
prints their import paths, so an old installed package cannot be mistaken for
an updated checkout. Failure stops before loading weights or launching TP
workers. The preflight neither executes transfer kernels nor proves numerical
parity. These checks and the request-capacity Python update introduce no new
native compilation requirement; missing native exports still require a rebuild.

The default is designed for one host with eight 64-GB NPUs (devices 0 through 7),
not a single 64-GB card. Reducing the layer count alone does not guarantee that
the model fits on one card. The test starts two fresh engines sequentially, but
**computes real prefill only once**. Both use **TP8/DP1**, eight target layers,
MTP1 and dummy weights. TP shards the weights across the eight cards; DP is not
used to replicate the whole model.
The test reads the original depth from `config.json` and remaps the original
MTP layer's entire ModelSlim quantization namespace to follow the eight target
layers, including head, attention, experts and FA/indexer metadata. This is an
in-memory test fixture adjustment; checkpoint files and normal serving are
unchanged. Missing original MTP quantization is an error, not a FLOAT fallback.
The reference runs **both staged/full graph disabled with enforce_eager**. It
computes the original chunked prefill, exports immutable target/MTP checkpoints,
then runs the original decode. The graph engine imports those checkpoints
**without calling the prefill models**, then runs the production root-graph
decode. No compiled-prefill-versus-eager comparison is involved. Normal serving
is not instrumented.

Each checkpoint contains model inputs/outputs, latent K/PE, indexer KV, live
sorted-resident state and Torch CPU/NPU RNG state. Cache blocks are mapped into
the second process's allocations without replacing graph-captured tensors.
The recorded original wait/save callback order rebuilds local LMCache sources,
partial tails, generators and request bookkeeping through the real connector
APIs. Scheduler and MTP input preparation still run normally; target and initial
MTP prefill computation do not run again. Imported bytes and input/position/
sequence metadata are checked exactly. Missing state, changed block aliases,
unsupported callbacks or a duplicate prefill computation fail; there is no
fallback that silently recomputes prefill. Subsequent eager and graph decode
trajectories use independent mutable caches.

The graph log distinguishes `prefill=IMPORTED` (with
`target_prefill_model_calls=0`) from `phase=decode ... PASS replay_per_rank=1`.
The final coverage gate requires all 4351 prompt tokens to have been computed
once in the reference and imported in the graph engine, with **zero graph-engine
target/MTP prefill model calls**. Startup dummy warmup/capture remains enabled
and is not real request prefill.
For another model directory/device set, the equivalent driver accepts
`python tools/sfa_full_graph_parity.py --model /path/to/model --devices 0,1,2,3,4,5,6,7`.
Explicit lists of 1/2/4/8 distinct devices are supported; fewer cards require
enough memory for the larger weight shard. This test disables EP, FlashComm,
sequence parallelism and context parallelism to keep target-row layouts aligned.

The test deliberately uses an isolated local CPU LMCache instead of inheriting
a server's shared/remote cache. Every TP rank stores its own KV and participates
in lookup (`save_only_first_rank=false`); the CPU cache cap is 2 GB per rank.
It exercises the model's real TP communication, but does not test Mooncake, DP,
EP, request recovery, or generated-language quality. Temporary prefill checkpoints
and decode references are managed and removed by the driver;
only console output needs to be kept. No native rebuild is introduced.

### Comparing actual generated output

To finish both generations without stopping at intermediate tolerance
differences, run from **vllm-ascend** on the same eight-NPU host:

```bash
set -o pipefail
python tools/sfa_full_graph_parity.py --compare-output 2>&1 | tee log.log
```

This still uses eight target layers, dummy weights, TP8/DP1/MTP1 and **one
computed prefill** imported by the graph engine. Both modes then generate
16 tokens greedily (`temperature=0`). Unlike the layer-parity test, there is
**no target vocabulary restriction and no replacement of MTP's proposed
tokens**. EOS is ignored for this fixed-length diagnostic. Actual token IDs and
decoded text (including special tokens) are printed under `[SFA_OUTPUT]`.

In this mode, intermediate tensors are checked for freshness, invalid KV
addresses and NaN/Inf. `[SFA_STATS]` reports descriptive absolute statistics
for each layer/stage and the target's final hidden state, without a floating-point
tolerance gate. Three distributions are shown: `diff_abs = abs(graph - eager)`,
`eager_abs = abs(eager)`, and `graph_abs = abs(graph)`. Each includes mean,
standard deviation, population variance, maximum and nonzero element count.
There is no percentage, denominator, epsilon or small-error pass/fail threshold.
Variance is in squared units; standard deviation is in the tensor's units.
All-zero observations are included and explicitly show `nonzero=0`.

Statistics are element-weighted across all aligned decode steps and TP ranks,
not averages of per-step/rank means or variances. `n` is the number of compared
elements, including TP-replicated observations. Maximum error is retained across
all ranks/steps. KV statistics exclude padding and entries where logical top-k
selections differ; `excluded` counts these elements. Top-k and validity tensors
are also reported, so exclusions do not hide selection differences. Statistics
of integer token indices are not floating-point activation error metrics.

The eager engine writes the existing CPU snapshots; graph collects statistics
after each complete forward, with no new per-layer synchronization or graph
split. It does not save a second set of large graph KV snapshots. Input token,
position, sequence length and speculative-row alignment is checked first.
After the first divergence, later steps are not pooled even if their latest
tokens happen to match again. Per-rank `COMPLETE`/`PARTIAL` lines disclose the
coverage and first alignment failure. Missing snapshots/probes and zero compared
steps cannot be reported as successful statistics. Temporary files are removed
by the driver as before.

Every graph decode must still execute one root replay on every rank, and every
layer must exercise historical KV loads. Both generations run to completion
before the driver compares their complete output token sequences and text.
Natural MTP acceptance can give different forward counts; the driver does not
force their decode histories or step counts back into alignment. The original
layer-parity mode and its tolerances are unchanged. The default statistics reuse
the original layer input/output, Q, top-k and KV probes. To also observe input
RMSNorm, attention output and post-attention RMSNorm stages, explicitly add
`--trace-residual`:

```bash
python tools/sfa_full_graph_parity.py --compare-output --trace-residual 2>&1 | tee log.log
```

These extra probes can change compiler fusion and their run is diagnostic, not
an acceptance result for the original uninstrumented path. This mode still
does not stop on small finite differences or recompute prefill.

`OUTPUT MATCH` means this case's generated tokens and text match, **not** that
all intermediate tensors match or that the implementation is generally
correct. `OUTPUT DIFFERENT` reports the first differing token (one-based) and
returns a failing exit status after printing both completed outputs. Different
outputs establish an observable divergence, not its cause: near-tied logits
and implementation errors still require further diagnosis. Identical decoded
text alone cannot hide differing token IDs. Runtime/coverage failures remain
failures, never an output match. No native rebuild is needed for this mode.

### Diagnosing startup HCCL bind failures

From **vllm-ascend**, run the following on the eight-NPU host (no rebuild for
these Python-only diagnostic additions):

```bash
set -o pipefail
python tools/sfa_startup_diagnostic.py 2>&1 | tee log.log
```

This uses the parity test's TP8/DP1/MTP1 configuration, runs its dependency
preflight in a separate process, then starts the real vLLM multiprocessing
executor. By default it creates the original worker/model-runner buffers and
runs the failing W4A8 MoE scheme constructor, including its MC2 communicator
lookup. **It does not load model weights, allocate KV caches, run prefill, or
capture/replay a graph.** It is a reduction of the startup path, not a claim
that the complete model constructor has been reproduced.

`[SFA_STARTUP]` records PID, parent PID, rank, assigned/current device,
visibility-derived device ID, optional device UUID, communication group
members, call stacks and begin/end/error events. It also records the worker's
already-loaded Torch/NPU versions and HCCL library paths. Native communicator-name
calls are observed without making extra calls. Multiple name lookups can hit
a native cache; their count alone does **not** prove duplicate socket binds.
The script enables INFO native logs and appends current-run PID-matched host
bind records under `[SFA_NATIVE_BIND]`, including on failure. Missing native
records are reported explicitly, not interpreted as an unoccupied port.
It does not change HCCL port settings, reset devices or kill existing jobs.
On timeout/interrupt it stops only its own newly created process session.

If the reduced startup passes, add `--load-model` to trace the original
eight-layer dummy target/draft model load. This still stops before profiling
and inference. Add `--skip-preflight` only to isolate the dependency-import
process from startup. `--model`, `--devices`, and `--timeout` are optional;
the defaults are the same GLM model path, devices 0 through 7 and 300 seconds
per child. All output can use `log.log`; scratch trace files are managed and
removed by the script.

For a controlled merge comparison, keep these diagnostic tools fixed and use
`--repo-root /path/to/old-checkouts`, whose children must be named `vllm`,
`vllm-ascend`, `LMCache`, and `LMCache-Ascend`. Run the same absolute diagnostic
script with the current and old roots. The root is checked before launching
workers and again after startup; resolved package paths and Git revisions are
logged. This option does not fetch, switch branches, or rebuild extensions:
each selected checkout must have compatible, already-built native extensions.
It does not claim binary-level A/B isolation when external libraries differ.

`STARTUP PASS` requires every rank's startup report, complete unique device
mapping and observed communicator-name calls. It is **not** an inference or
full-graph correctness result. CPU tests in
`tests/ut/tools/test_sfa_startup_diagnostic.py` cover transparent tracing,
error propagation, mapping checks, native log filtering and executor scope.

### What must match

- Complete target and draft weight fingerprints **for each matching TP rank**.
  Reference files are isolated by rank; different ranks' weight shards are not
  compared to each other. Integer dummy weights are
  deterministically initialized as part of this test fixture, since the
  upstream dummy loader only initializes floating-point tensors.
  Internal-format weights are hashed from complete, unpadded storage bytes
  without slicing or converting their device layout (including packed INT4
  represented as INT32/NZ). Ordinary tensor views are copied to CPU before
  hashing. Only one tensor's host copy and an 8-MiB hash chunk are retained;
  weights are never changed by fingerprinting. Unsupported padded/internal
  views fail explicitly with the tensor name instead of being skipped.
- Actual token IDs, positions, sequence lengths and query boundaries before
  every target forward. Target sampling and submitted MTP proposals are fixed
  to the same token. On decode, MTP computation and its KV callbacks still execute, but
  this does **not** test natural draft choices or rejection behavior.
- The one prefill's checkpoint inputs and imported KV/state bytes exactly.
- On decode, every target layer's hidden/residual inputs and outputs,
  followed by the final target hidden states.
- On decode: Q, logical top-k, history boundary, and the complete K/PE vectors
  at every valid top-k entry **as attention consumes them**. Physical slots and
  planner misses are recorded for diagnosis, not compared as numerical results:
  independently allocated caches can legitimately have different slot numbers
  and the Q1/bounded-Q2 planners can have different residency decisions.

The 4351-token prompt exceeds the MTP scratch prefix and is adjacent to a
256-token window boundary. Sixteen forced output tokens exercise repeated Q2
decode. Each of the eight layers must actually plan historical KV loads, and
every live target decode must execute exactly **one root replay per TP rank**.
All eight ranks must pass; success on rank 0 alone is insufficient. Missing
probes, stale snapshots, missing eager steps, wrong input tokens, invalid KV
addresses and NaN/Inf (even on both sides) fail the test. A hardware/model skip
is not a pass.

All snapshot buffers are allocated before capture. Decoder hooks trace device
copies into the graph; SFA probes add device operations inside existing opaque
SFA operators during capture. No new splitting operator or per-layer host fence
is added. Device counters must advance exactly once for each layer observation
on each live replay. Reading/comparison happens **after the whole forward**.
CPU/Gloo agreement at startup and before/after target forwards propagates a
diagnostic or reference-file failure on any rank to all peers before the next
model collective. There are no per-layer diagnostic collectives. This does not
recover from a failed/hung NPU/HCCL kernel; vLLM's worker supervisor handles that.

The first mismatch reports its rank, step, layer, tensor, coordinate, values, maximum
absolute/relative error and mismatch count. Integer top-k is compared exactly;
floating-point defaults are `atol=1e-7, rtol=1e-2`. A top-k difference near tied
scores is a divergence to investigate, not by itself proof of a graph bug;
do not automatically loosen tolerances until a test passes. Instrumentation
adds memory traffic and can change race timing, so this is not a performance
benchmark or a proof that the uninstrumented path is race-free.

The CPU regression tests for the observer/comparator live in
`tests/ut/compilation/test_sfa_parity.py`. They inject incorrect KV, mapping,
token and layer data, and verify compiled hooks and stale-probe detection.
`tests/ut/compilation/test_sfa_transfer_contract.py` executes the actual SFA
layer preparation method with LMCache's source dataclasses, LMCache-Ascend's
transfer class and its native Python wrappers from sibling checkouts. It
covers startup allocation, eight layer indices, request lanes, changing chunk
tails, empty sources and fixed pointer-table addresses on CPU. Native calls
and prevalidated attention metadata are fixtures; this is not a kernel test.
It skips if the sibling repositories are absent. Driver preflight/error-order
tests are in `tests/ut/tools/test_sfa_full_graph_parity.py`.
Each parity engine explicitly releases its cache connector and Ascend process
groups after generation, shuts down the engine, and waits for its identified
workers to exit before the next engine starts. A release/exit failure aborts
the comparison; the driver does not kill unrelated processes or change HCCL
ports. CPU process-lifecycle regressions are in
`tests/ut/tools/test_sfa_parity_shutdown.py`, and mocked worker release-order
tests are in `tests/ut/compilation/test_sfa_parity_worker_shutdown.py`. These do not
verify native HCCL port release on NPU hardware.
`tests/ut/compilation/test_sfa_prefill_checkpoint.py` checks exact import bytes,
physical block remapping, live-versus-dummy resident-state bindings, callback
order, immutable snapshots, zero prefill recomputation and untouched decode
dispatch. It executes the actual worker checkpoint path on CPU fixtures; this
is not a real NPU kernel test.
Real CPU/Gloo two- and eight-process failure-agreement tests are in
`tests/ut/compilation/test_sfa_parity_gloo.py`. They do not replace the real
eight-NPU/model test above. The small single-card probe replay test remains in
`tests/e2e/singlecard/test_sfa_parity_probe_npu.py`; it does not load the model.

### Diagnosing a residual mismatch

For a decode residual mismatch, run the separate
fine-probe diagnostic from **vllm-ascend**, still on TP8/DP1:

```bash
set -o pipefail
python -m pytest -q -s --confcutdir=tests/e2e/multicard tests/e2e/multicard/test_sfa_residual_trace.py 2>&1 | tee log.log
```

The equivalent driver option is `--trace-residual`. Both fresh engines use the
same fine probes and weights. Prefill still computes once and is imported by
the graph engine. The eager decode reference disables staged/full graph and
compilation; graph decode retains its compilation/fusion configuration.
Decode tolerances are unchanged.

In addition to the original layer/attention observations, every target layer
records input RMSNorm outputs, attention output, both actual operands entering
post-attention RMSNorm, and its normalized/residual outputs. The operands are
copied before any in-place update, including after intervening FP16 scaling.
There are no per-layer host reads, synchronization or diagnostic collectives;
copies/counters are captured, and analysis runs after the complete forward.
All extra buffers are included in the worker's memory budget. At BF16/6144
hidden size and 512 rows, these add about 336 MiB per rank. Checkpoints and
decode reference files are removed by the driver.

On failure, `[SFA_TRACE]` reports the actual phase, row count, root replay count,
earliest **elementwise difference** (even within tolerance), and stage comparisons
in the first failing layer. For a residual-output failure it also prints both
operands, their FP64 sum, the sum rounded to the output dtype, and the observed
residual at the failing coordinate. This distinguishes changed operands from
a differing addition result, but does not identify a particular fused kernel
or prove that a small difference is harmless. Earlier stages passing tolerance
does not imply bitwise-identical inputs. An import/checkpoint failure is a setup
failure, not a numerical comparison of two prefills or decode replay evidence.

`max_rel_nonzero` excludes exactly-zero reference values; their differences are
reported as `zero_ref_max_abs`. The pass/fail formula remains
`abs(actual-reference) <= atol + rtol*abs(reference)` at every element. NaN/Inf,
missing probes, wrong tokens/top-k/KV and incorrect root replay counts still fail.

Fine probes add observable intermediate values and **can change compiler
fusion**. A successful diagnostic prints `DIAGNOSTIC PASS`, not an acceptance
result. If the original mismatch disappears only with these probes, investigate
fusion/timing effects; the original `test_sfa_full_graph_parity.py` must still
pass separately. CPU regressions test hook ordering, in-place alias isolation,
fullgraph traceability, failure localization and the unchanged tolerance gate.
