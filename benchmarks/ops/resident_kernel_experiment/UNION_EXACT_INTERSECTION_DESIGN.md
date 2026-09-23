# Union-only optimization with unchanged retrieval work

## Finding

Implemented as the isolated `vector_intersection` variant. Host equivalence tests
and native qualification cases accompany it; CANN compilation and performance
qualification require the deployment NPU. Build/benchmark commands are in README.md.

The best-scoped next experiment is the scalar resident intersection inside
`DSAResidentShardedUnionKernel::ProcessBlock` in
`csrc/kernels/resident_sorted_cache.cpp` (currently around lines 608–642).
Keep its filtering, SortAll, scalar deduplication/mapping, generation checks,
finalize, update/remap and retrieval unchanged. This is an exact replacement of
one internal operation, not the approximate bounded-position redesign.

The existing `vector_union` variant changes deduplication/mapping, but leaves this
intersection scalar. It also rebuilds mapping with vector lower_bound over every
original top-k position in every shard. That adds work even for entries owned by
other shards; its performance is not evidence about an intersection-only variant.

## Exact replacement

Both current unique tokens and resident tokens are sorted by token identity.

1. For each current token, lower_bound in the old token array. An exact match
   supplies the associated old slot. Otherwise the prior slot is -1.
2. Stable mask compaction produces missing tokens and their current-union
   positions. Since current tokens are sorted, this matches the scalar miss order.
3. For each old token, lower_bound in the current token array. Stable compaction
   of nonmembers' old slots produces the exact scalar eviction-candidate order.
4. Leave count publication and all payload-before-count synchronization in place.

An empty old array produces all current entries as misses. An empty current array
produces all old slots as eviction candidates. Generation mismatch continues to
mean oldCount=0 and preserves the existing state-count reset. No unbounded index
or sentinel read may be introduced for empty arrays or padded vector lanes.

Finalize consumes eviction candidates shard-major and in their emitted order.
Set equality is therefore insufficient: the valid ordered arrays must match.
Equal arrays give the same selected eviction slots, physical target slots,
resident state and remapped attention indices. By induction this preserves later
cache retention and misses, not just the current step's payload values.

## Bounded implementation and UB lifetime plan

Start with a vector fast path only when both `rank` and `oldCount` fit `rowWidth_`
(2048). Retain the original scalar merge for larger/skewed shards. Ordinary
Q=2/S=8 geometry has approximately 384 current unique tokens and 512 old tokens
per shard, but the guard must use actual counts rather than assume balance.

This bound lets existing 2048-element row buffers hold lower_bound work vectors.
After deduplication those row buffers are dead, while sortedTokens, mapping,
oldTokens, oldSlots and priorSlots remain live.

Candidate scratch plan, to verify when implementing:

- Reuse input/work/clamped/index buffers for four 2048-element search vectors.
- Preserve the first `shardCapacity_` int32 entries of sortSrc for miss tokens;
  its second half can hold gathered candidates.
- Preserve the first two `shardCapacity_` int16 regions of sortTmp for miss
  positions and evictable slots; its second half can hold widened old slots.
- Existing row masks cover up to 2048 lanes. Pad vector counts within that bound,
  and explicitly mask inactive lanes before stable compaction.
- Compact int32 positions/slots into dead row scratch then narrow to int16, if
  the target GatherMask overload does not accept int16. Do not assume Scatter
  support or allow unaligned append offsets.

This permits a design without extra global buffers or peak UB allocation. Exact
alias ranges, vector alignment and target-supported overloads remain implementation
checks. MTE2-to-vector and scalar-to-vector dependencies must be established before
reading loaded old state or scalar-produced deduplication results.

## Unchanged work downstream

- Same sorted unique union and original-position mapping.
- Same prior slots, ordered unique misses, miss positions, ordered eviction slots.
- Same generation handling, counts, finalize slot assignments and resident updates.
- Same number/size/order of retrieval calls and CPU source rows.
- No hit copies, new payload banks, descriptor adapters or extra kernel launches.

The new experiment should dispatch its union followed by the baseline finalize and
update kernels. Its matched retrieval case must be the existing Original adapter,
changing only union dispatch. Do not run it through the redesign Replacement class.

## Validation and performance limits

A CPU mathematical model of two lower_bound searches plus stable compaction was
compared with the existing scalar-merge semantics in 240 deterministic cases.
The valid ordered prior-slot, miss-token, miss-position and eviction arrays agreed,
including empty arrays, subsets, randomized slots and sizes up to 4096. This proves
the transformation in that model, not the correctness or speed of a CANN kernel.

Native qualification must compare every valid union output before finalize, then
the complete state/miss/target/remap result afterward. Include Q=1/2, duplicates
across rows, different split boundaries, padding/inactive rows, invalid state rows,
generation resets, zero boundary, all-hit/all-miss, skew fallback and graph replay
with changing state and request counts. Follow with the same retrieval pipeline
and verify its bytes and call counts are identical over consecutive steps.

The current reported union time is 95.57 us within a 366.47 us complete path.
Its scalar intersection's share is not yet measured. Existing `union_sort`,
`union_dedup` and full `union` probes can bound where time is spent, but their
observable writebacks/code generation differ: subtracting them is not an exact
phase timer. Ultimately judge the full union and complete path, in paired order.

Vector binary search does more comparisons than a scalar linear merge; the intended
gain is parallel execution and fewer scalar GetValue/SetValue operations. A speedup
is plausible, not guaranteed. Even eliminating the entire union would save at most
about 26% of this measured complete path; union-only work cannot yield a 2x speedup.
