# Decoder FlashComm1 without DSA context parallelism

## Scope and baseline

Implementation design, 2026-11-03. Base: `perf/sfa-dp-sync` at
`0d7c199fdf2a38bbffdaa96ea1f5a763ab52d17c`; implementation branch:
`feat/decoder-flashcomm1-only`. Production changes belong to vLLM-Ascend.
Core vLLM and both LMCache repositories retain their existing interfaces.

The target deployment is TP4/DP4 with EP, PP=PCP=DCP=1, one MTP token,
PIECEWISE compilation, staged and full SFA graphs, asynchronous MTP preparation,
two-group compact cold loading, resident/shared-resident planning and retrieval
overlap. Request capture sizes are `2,4,6,8,10,12`; independent DP is enabled.
The supplied script disables npugraph_ex, static kernels and shared-expert
multistream overlap. Preserve these settings. Optional npugraph_ex must retain
the existing mutation-lowering/root-capture path; hardware qualification is
separate. This change does not resolve the deferred Torch-NPU overlap issue.

Relevant upstream implementations are
[PR 9878](https://github.com/vllm-project/vllm-ascend/pull/9878) and
[shared-expert placement commit 23cfbd5b](https://github.com/vllm-project/vllm-ascend/commit/23cfbd5bae8028dc31d8ddd66673f985cf66e5f5),
checked against `v0.26.0rc2`. Port their FC1-only attention boundary and shared
placement/layout behavior, not the unrelated MoERunner, quantization or PCP
refactors. Reuse existing collective custom operators and graph ownership.

## Configuration and compatibility

- `additional_config.enable_flashcomm1` explicitly overrides the existing FC1
  environment switches. Omission retains their precedence and defaults.
- `additional_config.enable_dsa_cp=false` selects FC1-only for indexer models.
  Omission preserves this branch's historical FC1-to-DSA-CP coupling, including
  existing prefiller launch scripts. Explicit CP requires FC1.
- Omitted `enable_shared_expert_dp` retains existing placement. Explicit false
  permits TP-sharded shared weights under FC1; explicit true retains the
  established shared-DP enabling behavior. Placement is independent of overlap.
- No new environment variable, kernel ABI, storage layout or per-step DP vote.
  Resolve immutable switches at initialization. Include effective layout flags
  in the independent-DP startup contract; retain existing failure supervision.

Decoder opt-in, merged into the existing additional config:

```json
{
  "enable_flashcomm1": true,
  "enable_dsa_cp": false,
  "multistream_overlap_shared_expert": false
}
```

Shared-expert replication remains the compatible default. To compare the
memory-saving TP-sharded alternative, also set `enable_shared_expert_dp=false`.
Restart all decoder workers when changing layout/placement. Keep prefiller
configuration unchanged. The complete attention/graph implementation must be
installed before using these switches; a config-only backport is insufficient.

## Token-layout and execution invariants

Let T be the existing local-DP model/graph token extent and P=ceil(T/TP)*TP.
FC1 carries P/TP consecutive token rows between layers. Attention and compact
KV metadata continue to describe T rows, the original requests and both groups.
Internal communication padding owns no KV slot and changes no computed-token
frontier. DSA-CP remains disabled; attention head/weight partitioning is retained.

1. At native SFA entry, gather/unpad once before projections/indexing. At staged
   entry, gather/unpad once after choosing the native fallback, before validating
   global attention rows and executing staged pre-compute. Profiling without
   metadata retains its existing zero-output behavior.
2. The gather executes in the model's ordinary stream/captured custom operator.
   No collective is added to host preparation, source binding or asynchronous
   storage work. Existing producer waits, retrieval joins, loads, attention,
   saves and source-retirement events retain their order.
3. Pre-compute and all bridge/planner tensors retain global T-row capacities.
   Post-compute uses graph-key rows and the existing o_proj reduce-scatter to
   write TP-local output. The existing target-tail gather remains inside root
   capture; draft gathering remains in its existing execution path. The draft
   tail must select gathering from runtime FC1 layout, not shared-expert weight
   placement, before indexing full-row draft outputs.
4. Preserve logical capture keys, including keys not divisible by TP. For
   FC1-only staged configuration, avoid rounding the model/graph extent before
   dispatch; pad only communication operands. Fake gather/reduce shapes must
   match the actual unpadding/ceiling division. Do not enlarge request capacity,
   force graph eligibility, or silently drop configured buckets.
5. Native fallback, eager MTP, Q1/Q2, bounded/uniform graphs and idle peers use
   the same layout rules. Independent DP still varies T across DP instances;
   TP peers agree and EP dispatch/combine retain their common sequence/capacity.
6. Diagnostic snapshots of TP-local outputs must retain local row identity and
   the corresponding CPU request-row slice. Disabled diagnostics add no gather,
   synchronization or readback. Do not reshape local columns into global rows.

## Shared and routed experts

With replicated shared weights, retain the current local-token shared MLP.
With FC1 and TP-sharded shared weights, use:

`local tokens -> TP gather/unpad -> shared MLP -> TP pad/reduce-scatter`.

Reuse small input/finalization helpers in the serial/shared-stream and legacy
gate-overlap entries. Gathering follows the existing producer wait; shared
finalization follows the established stream join. Preserve the router's original
local token tensor. Existing non-FC1 TP reduction behavior is unchanged.

Routed MC2/All2All slicing and output gathering depend on replacement of the
usual TP reduction (the runtime token layout), not shared-weight placement.
Apply the corresponding upstream AllGather finalization correction. Keep the
optional MC2 mask optimization: a None mask must never be split or constructed.
No new communicator, executor, host round trip or device readback is introduced.

## Memory and performance

Use existing startup weight/activation/graph profiling to budget the changed
placement. Do not spend unused KV blocks as if they were unallocated HBM.
Retain all admission, source leases, graph memory and scratch capacity checks.
FC1 can reduce repeated layout work and communication; TP-sharded shared weights
save storage but add collectives. Neither variant is assumed faster. The baseline
already partitions MC2 routed tokens; no new TP-fold FLOP reduction is claimed.

## Implementation and validation

Audit correction to the inherited MTP input conversion: the input buffers are
replicated target hidden states and positions, not partial sums. Existing MTP
SUM reduce-scatter multiplies their values by TP (reproduced with TP4: position
10 becomes 40). FC1-only MTP must instead locally slice these buffers. Reuse
`split_inputs_tp_to_sp` with an optional fresh zero-filled output, leaving full
input buffers untouched and padding-only ranks initialized. Its TP selection
must use `rank_in_group`, not global `rank` (DP1 starts at global rank 4 in TP4).
Preserve the existing explicit-output API and legacy CP MTP route. This is a
demonstrated inherited mismatch encountered while enabling the new decoder path;
upstream still has the SUM conversion, so copying it is not sufficient.

Changes are limited to configuration/utils, SFA boundaries/diagnostics, existing
SP padding/fake operators, shared/routed MoE paths and startup contract fields.
No compiler schema change or storage-manager change is planned. If traced code
contradicts this design, record the concrete mismatch here before deviating.

CPU regressions must execute production helpers/methods, with stand-ins clearly
identified: configuration defaults/precedence, disabled and legacy CP paths,
exactly-once gather, global bridge/local output, padding, shared-TP numerical
reconstruction, routed slicing independent of placement, optional masks,
startup contract and target-tail gathering. Extend existing compiler tests where
needed to cover local input/global bridge/local output under mutation lowering.
Run affected existing routing, agreement, async-MTP, graph/lifetime tests and
lint; separately review the final diff for ordering and ordinary-path changes.

Extend the existing installed-model parity harness with an opt-in FC1-only mode
and the distributed MC2 probe with TP gather/reduce boundaries. The model harness
is a reduced TP-only cold-load/MTP test; the MC2 probe is a collective/graph test.
Neither alone proves full-server correctness. On the actual TP4/DP4 deployment,
compare FC1 off/on and shared placement with identical model, prompts, cache
warmth, graph sizes, sampling, MTP and diagnostics. Exercise every bucket, idle
and unequal peers, mixed Q1/Q2, admissions/finishes, native recovery, cold load,
preemption/resumption and GC disabled. Check output/finish parity, MTP acceptance,
TTFT/TPOT tails, throughput, launch gaps, graph/HBM use and preemption together.

This development host has CPU PyTorch only. NPU compilation, capture/replay,
collectives, memory and performance remain unverified until those runs succeed.

## Implementation review, 2026-11-03

The implementation follows the design above, including the recorded MTP
correction. Separate review found and corrected the projection fake's missing
padding (T=6, TP4: one inferred row versus two actual rows), in addition to the
replicated MTP-input SUM issue. Regression tests execute the real Python methods
with CPU collective stand-ins; dynamic-shape tests use CPU AOT compilation.
Existing source leases, cache mappings, scheduler frontiers and remote-transfer
protocols are unchanged. Process-service and native-kernel rewrites are not
applicable to this change.

Final affected-suite run: **1015 passed, 8 skipped**, on Windows/PyTorch 2.10 CPU.
Skipped Linux/NPU checks are not passes. An earlier broader run had 1717 passes,
12 failures, 27 setup errors and 31 skips. The same failures/errors reproduce
against the baseline: eight incomplete dummy-run fixtures, two incomplete mixed-C8
builder fixtures, 27 cross-repository transfer-fixture errors and two unsupported
Windows Gloo subprocess cases. No production change was made to hide them.

Changed/new Python code passes Ruff apart from five pre-existing import-order /
line-length findings in the large attention/model-runner files, verified against
HEAD. Existing unrelated formatting is retained. Whitespace checks pass with
`core.whitespace=cr-at-eol` for inherited CRLF files. The full `format.sh ci`
pipeline cannot run here because pre-commit is not installed.

There is no measured NPU speedup. FC1-off adds small Python helper-call overhead
but no additional device collective/copy. FC1-only MTP replaces two incorrect TP
SUM collectives with local copies; this operation-count reduction does not prove
an end-to-end improvement. Shared-TP placement trades memory for communication.

Run the installed-model probe on otherwise available NPUs:

```bash
python tools/sfa_full_graph_parity.py --devices 0,1,2,3 --flashcomm1
```

This remains a reduced, single-request TP-only cold-load/target/MTP test. For
the independent-DP collective probe, use the existing two-host torchrun command
in [the independent-DP guide](sfa_independent_dp.md) with:

```bash
pytest --noconftest -q tests/e2e/multicard/test_sfa_independent_mc2.py -k flashcomm1
```

The new variant includes TP padding/gather/reduce-scatter around MC2, unequal
local graph extents and zeroed idle-like inputs. It is not a full-server test.
Complete the server matrix above before treating this as production-qualified.
No native rebuild is needed for these Python changes; worker restart and graph
recapture are required. The Torch-NPU/CANN upgrade remains deferred.

## Follow-up audit, 2026-11-03

One additional integration defect was reproduced: TP4 with one request and
MTP=1 has valid staged token capacity 2, but the platform's earlier generic SP
filter removes initial sizes `[1,2]` and asserts before staged sizing runs.
Explicit FC1-only staged SFA now bypasses only that generic divisibility filter;
the existing staged request/token/stream limits still apply. Omitted CP, enabled
CP, disabled FC1 and ordinary non-staged execution retain their prior behavior.
This completes the logical-capacity invariant in step 4 of the design.

The new platform regressions had five failures before the correction and all
ten pass afterward. Final affected-suite validation: **1046 passed, 8 skipped**.
Separate review found no further demonstrated defect in the configured input-ID
path. The earlier baseline fixture/environment failures remain separately
classified; hardware qualification is still outstanding. Prompt-embedding input
was not enabled in the supplied script and remains an unverified combination.
