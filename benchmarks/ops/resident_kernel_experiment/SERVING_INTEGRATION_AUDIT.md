# Exact resident kernels: full-graph integration audit

Audited integration/decode-full-graph-production at 5e42634f on 2026-09-23,
against its pre-merge full-graph parent 084e264a. This audit follows the kernel
selection through serving; it does not claim to requalify every unchanged
LMCache, Mooncake, scheduler or transfer implementation.

## Outcome

No concrete runtime correctness defect was established in the merge. No
speculative runtime changes were made. Four additional regression cases execute
the actual AscendSFAImpl constructor section to check selection before ordinary
allocation, shared allocation deferral, and the resident-disabled path.

## Entry and selection

- AscendSFAImpl selects kernels only when dsa_resident_cache is enabled, before
  shared_resident_candidate decides whether allocation is deferred.
- The model runner still allocates the shared producer's state in
  _bind_shared_resident_plans; consumers retain the existing explicit read/write
  tensors. The merge does not allocate per-consumer replacement state.
- Python caches the startup configuration. C++ uses an atomic immutable table;
  repeated identical configuration is accepted and conflicting configuration
  fails rather than changing a live capture. An unconfigured native caller
  selects the default exact table on its first launch.
- Union, fused plan, no-remap plan and diagnostic update all select matching
  launchers. Finalize and remap algorithms are unchanged. The exact translation
  unit enables only VECTOR_INTERSECTION and VECTOR_STATE, not the approximate
  cache algorithms or other experimental variants.

## Data and ordering

- No Torch tensor schemas or transfer arguments changed. Both selections retain
  selected tokens, miss counts, target slots, shard state and generation contracts.
- sfa_forward_pre[_shared] -> sfa_lmcache_retrieve -> sfa_forward_post remains
  unchanged, as do explicit shared reads/writes and the staged event handoff.
- Host callbacks capture a static-lifetime launcher function pointer and retain
  the original stream. No cross-core spin barrier or collective was introduced.
- Existing generation mismatch, padding, empty selection, invalid-index and
  scalar-capacity fallbacks remain. Vector intersection is bounded by row width;
  vector state merge requires its supported capacity and old+current <= 2048.
- Payload publication fences and terminal MTE3 completion remain present. Native
  device ordering still requires qualification; CPU models cannot establish it.

## Memory, sources and recovery

- No new persistent HBM tensor or LocalCPU cache allocation is introduced by the
  selector. Exact state update adds 48 KiB of per-core UB scratch for supported
  geometry. Baseline does not allocate that scratch. Both implementations add
  compiled code size, not another resident KV cache.
- Group-0/Group-1 placement, CPU/Mooncake source choice, source pinning/lifetime,
  remote-fill admission/reservation, asynchronous stores, prefix proof and
  preemption/resumption policy are unchanged by this merge. Their existing
  recovery and graph tests were exercised; this is not a blanket certification
  of every distributed storage combination.
- The selector adds no thread, blocking wait, GIL management change, Python
  reference cycle, per-request metadata construction or GC-dependent cleanup.

## Performance limits

Captured replay contains selected device kernels, with no environment lookup or
host selector in replay. Eager enqueue does add an atomic table read, null check
and indirect launcher call. Zero eager overhead has NOT been demonstrated.
The kernel algorithms retain the retrieval workload; no new hit-copy pass is
introduced. Earlier exact_combined NPU measurements support a speedup for the
measured cases, but do not prove full-serving TPOT improvement on this merged
branch. Shared planning also reduces how many layers execute the planner, so
per-layer savings cannot simply be multiplied by the full layer count.

## Validation

- Kernel/selector suite: 278 passed, 735 NPU tests skipped.
- Added constructor regression cases: 4 passed (selector file: 9 total).
- Standalone recovery/full-graph composition: 142 passed.
- Compilation/graph suite: 1048 passed, 10 skipped, 2 failed at Gloo process-group
  initialization with makeDeviceForHostname(): unsupported gloo device.
- Two other compilation test files could not collect without installed vLLM
  dependencies (test_acl_graph.py and test_npugraph_ex_utils_check.py).
- Targeted Ruff and whitespace checks passed.

1472 distinct passing cases in these runs; no claim that all tests passed.

## Remaining NPU qualification

Build the normal serving extension, not just the experimental libraries. In
separate worker processes run the same full-graph workload with
VLLM_ASCEND_DSA_RESIDENT_EXACT_KERNELS=0 and =1. Keep shared planning, capture
sizes and all LMCache settings fixed. Verify the selected device kernel names,
output correctness, repeated replay, request-slot reuse/generation changes,
preemption/resumption and fallback. Compare full-step/TPOT and transfer counts,
not only standalone kernel time. Test shared planning both enabled and disabled
when deploying both modes. No serving speedup or deadlock-free distributed run
is claimed until these checks complete on NPU.
