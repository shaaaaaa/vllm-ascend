---
orphan: true
---

# Latent-only prefill DCP with four rotating KV banks

Design for `feat/prefill-dcp-layerwise-cache` and the TP8/DP2 disaggregated launcher. The implementation applies to the four-bank LMCache **sender**. Each TP rank reloads its own Group 0 stripes from the shared CPU pages. A temporary attention view gives the existing prefill kernel the visible history for its local query rows. TP rank 0 saves canonical pages from that same view. The decoder can consume those pages with its existing layout, including a request-sharded Group 1.

## Running the implementation

Use matching checkouts of `feat/prefill-dcp-layerwise-cache` in all four repositories. On the **prefiller**, select `--decode-context-parallel-size 8 --block-size 128 --cp-kv-cache-interleave-size 128` for TP8. Keep four banks, eager execution, FlashComm1, chunked prefill, `VLLM_ASCEND_DSA_DCP_INDEXER_REQUEST_SHARDING=0`, and the sender LMCache YAML. The receiver consumes the existing canonical pages. DCP1 on P selects the original path.

The runner reserves one dense workspace and two local send planes during memory profiling. It warms the fused pack and fresh-row scatter, builds one map per batch and reuses it for the target and MTP. Contiguous shard ranges feed HCCL directly. Fragmented shards use one pack. The fresh-row kernel reads the existing TP packet once and writes the complete view and the owning physical shard. H2D and rank-0 D2H use the existing LMCache native extension. The additions are Python and Triton.

Run the HCCL/DMA test from the vllm-ascend checkout on idle NPUs in the serving container.

```bash
python3 -m torch.distributed.run --rdzv-backend=static --rdzv-endpoint=127.0.0.1:29571 \
  --nnodes=1 --node-rank=0 --nproc-per-node=8 --module pytest --noconftest -o addopts= -q \
  tests/e2e/multicard/test_sfa_dcp.py::test_banked_prefill_dcp_roundtrip
```

For a full-model comparison, run each command from vllm-ascend with the same model, devices and available CPU memory. Both arms use four banks, 16,384-token compute chunks, mixed C8 indexing and one MTP token. The local profile records token output and request time with the profiler attached. For production TTFT, repeat the same workload against the P/D service with Mooncake and profiling disabled.

```bash
python3 tools/layerwise_prefill_profile.py --model /workspace/models/GLM-5.3-w4a8c8 \
  --case 80k_on --prefill-dcp 1 --run-dir profile-p-dcp1
python3 tools/layerwise_prefill_profile.py --model /workspace/models/GLM-5.3-w4a8c8 \
  --case 80k_on --prefill-dcp 8 --run-dir profile-p-dcp8
```

## Implementation audit

The audit compared `feat/prefill-dcp-layerwise-cache` with the following `feat/decoder-dcp-indexer-sharding` baseline in each repository.

| Checkout | Base | Most recent first-parent merge and its parents | Scope |
| --- | --- | --- | --- |
| vllm | `28e80e0b1` | `52620ee77` (`d7ed086af`, `2d32a43a9`) | Four-bank admission and memory sizing. The cold-allocation test fixture now includes its inherited allocation method and logging setting. |
| vllm-ascend | `f8bef4fc1` | `df0ef0852` (`72955081d`, `c3d5a2192`) | Sender layout, native attention inputs, eager MTP, stream dependencies and memory profiling. |
| LMCache-NPU | `8d32a083` | `87b1f099` (`9503c43b`, `1138ae44`) | Scheduler request metadata, worker source coverage and the canonical save binding. |
| LMCache-Ascend | `81c135b` | `7f0a52b` (`7311f35`, `7b30de6`) | Per-rank H2D descriptors, rank-0 D2H addresses and completion fences. |

The provided launcher runs one TP8 group per host, with two P and two D instances across four hosts. The P template selects DCP1, four banks, a 1,048,576-token context limit, eight requests, 16,384 fresh tokens per step, 0.93 HBM utilization, eager execution and one MTP token. Its YAML selects a 250 GiB shared CPU cache by default, async stores, mixed C8 indexer support through the launcher, disabled cyclic GC and Mooncake. Changing P to DCP8 selects the path above. D uses its own DCP8 configuration, request-owned indexer, graph settings, process-mode loading and checkpoint writeback.

| Reproduced trigger | Correction and regression |
| --- | --- |
| A continuation requested 1,025 historical tokens. Both source preflight and final validation previously accepted 1,024 because the metadata also contained 1,025 IDs. | Use the first fresh position in the worker batch. The same one-token shortfall is allowed for an actual full hit that recomputes position 1,024. Both KV groups are covered. Out-of-range load extents fail before DMA priming. |
| A short allocation inherited a positive block ID from an older batch row, or only one rank encountered a missing final column. | Validate the active row length and all referenced physical columns on each peer before exchange. Guard, stale-ID and out-of-range cases exercise the actual row-reuse methods. |
| The second plane failed to launch, a work handle failed to wait, or the ready event failed to record after submission. | Retain the input storage and issued work, preserve earlier D2H fences, and prevent the next DMA prime or gather from reusing the workspace. The successful path retains device-side ordering. |
| The inherited MLA builder reserved a second, unused exchange workspace. | The banked SFA builder uses the runner's profiled workspace. At TP8 with 576 BF16 lanes the removed allocation is 169,869,312 bytes per builder. |

The final host suites passed 184 vLLM tests, 286 vLLM-Ascend tests, 409 LMCache worker and scheduler tests, 278 RemoteFill tests and 374 LMCache-Ascend tests. The Ascend multicard module was skipped on the CPU host. A separate run repeated 44 ownership and failure tests with cyclic GC disabled. An independent descriptor check enumerated every transferred token in 165 ragged or gapped allocations across DCP2, DCP4 and DCP8.

The host tests exercise allocator and descriptor arithmetic, CPU execution of the Triton address expressions, and production connector methods with simulated device streams. A six-layer round trip runs the D2H and H2D generators on the same canonical pages with a passive rank and a padded final page. RemoteFill protocol, lease and transport tests cover the existing handoff. The native DMA binding releases the GIL while submitting ACL copies. This audit makes no measurement of GIL contention or live link bandwidth.

The supplied configuration selects `LMCacheAscendConnectorV1Dynamic`. Its derived adapter sets persistent placement, delegates to the base worker for source validation, and uses `VLLMPagedMemLayerwiseNPUConnector` for actual bank DMA. Proxy routing, decoder process ownership, checkpoint format and graph capture are outside the sender change. The serving container's loaded modules and binaries, NPU kernel compilation, HCCL ordering on hardware, peak HBM, live ADXL transfers, output parity and end-to-end latency require hardware qualification with the commands above.

## Code used as the design baseline

| Path | Behavior relevant to this change |
| --- | --- |
| Public vLLM-Ascend v0.26.0rc2 | Prefill and mixed SFA batches compact the **referenced KV block table**, gather those blocks and run causal attention with global token IDs. The selection is by scheduled block table, before Lightning Indexer produces Top-K. Decode uses a different algorithm. The public implementation provides an attention reference. [Metadata][up-meta] · [KV exchange][up-gather] · [Prefill dispatch][up-attn] |
| Production P | TP8, one TP group per P host, eager chunked prefill, four banks, one MTP token, FlashComm1 and sharded `q_b_proj`. P already splits scheduled query rows across TP. Its existing collectives distribute the **fresh** latent/RoPE rows on every layer. C8 indexer producers also distribute keys and scales. [Query partition][p-query] · [Fresh KV exchange and scatter][p-fresh] |
| Chunk continuation | A `prefill_child` request that continues locally gets a load specification for the entire computed prefix. Earlier layers' banks have been reused. The native DMA planner currently addresses each logical G0 block on every rank. A fresh first chunk with zero cache hits has no history to reload. [Load specification][p-cont] · [DMA planner][dma] |
| Four-bank P/D handoff | The launcher selects `LMCacheAscendConnectorV1Dynamic`. The P-node flag forces the layerwise storer even with `mooncake_direct_npu_prefill_store: true`. In per-chunk RemoteFill, the writer leases completed CPU layer pages, registers their owners and supplies CPU pointers and D2H fences to transport. In `persistent_direct_hbm` mode, Group 0 can fill the decoder's registered CPU pages and Group 1 is supplied through persistence. [Sender selection][direct-mode] · [Fill groups][fill-groups] · [Source pointers and leases][fill-src] · [Native transport][transport] |

## Shard layout and admission

Use the production physical block size and interleave of 128 tokens, with `W = TP = DCP = 8`, PCP=1 and PP=1. For token position `p` **within its request**, physical block `b = p // 128` belongs to rank `b % W`. Its local column is `b // W` and its intra-block offset is `p % 128`. The column indexes the real Group 0 table of the correct bank. Keep the P origin at zero. Group 1 retains its full-history key and scale layout, ordinary block size and its own layer-to-bank assignment. A rank's contiguous piece of the scheduled query batch is a different coordinate from its interleaved KV positions. [Stripe oracle][stripe] · [TP query partition][p-query]

1. In vLLM-Ascend, recognize the banked sender in startup, the model runner and SFA backend selection. The existing platform check rejects P DCP outright. Simply relaxing it selects the fork's generic CP backend and its different cache interface. Use the bank-aware `AscendSFAMetadataBuilder` and `AscendSFAImpl` with an explicit P gather view. Publish KV spec widths `(W, 1)` and pass them to **all four** input block tables. Only the first bank receives `dsa_dcp_sizes` today. Keep decoder compact scratch, live-origin offsets, request-owner workspaces and decode output exchange behind the decoder policy. Preserve the working P query split, including a single-token tail or MTP forward labelled as decode. [Startup guard][platform] · [Backend selection][select] · [Runner spec][runner-spec] · [Decoder scratch initialization][runner-scratch] · [Four bank tables][bank-tables]
2. In vLLM, use `PrefillLayerBundlePool` and the existing per-group managers. For visible length `S`, G0 needs `ceil(S / (128 W))` local block columns per bank and G1 needs `ceil(S / 128)`. Round each count to its group's blocks per child, multiply by the bank count, then apply the existing parent-slot and scale-sidecar rounding. Some capacity calculations already multiply by the G0 width. The `prefill_child` maximum-memory calculation still passes the physical block size. Make allocation, maximum length, concurrency reporting and startup profiling use the same per-group spans exactly once. Charge the attention workspace separately. CPU admission and Mooncake reservations still account for complete canonical pages. [Parent and child allocator][pool] · [Per-group manager geometry][coord] · [Concurrency estimate][concur] · [P memory estimate][budget]
3. In LMCache, split sender and receiver initialization at the existing KV-spec validation. Today declaring a G0 DCP width on P raises `Invalid decoder DCP cache geometry` because it requires `pd_role=receiver`. That branch also installs a decoder scratch origin. Give P its own geometry with zero origin. Preserve logical save/load ranges and `block_ids_by_bank[bank][group]`. Carry this geometry through `ReqMeta` before any dense slot mapping or live-source construction. Building `ceil(S/128)` slots from a sharded physical block list can fail or address another child. Use the P sender and connector capabilities for routing. The launcher uses `kv_both` for both endpoints. Preserve G1 metadata, lookup keys, saved frontiers and the proxy's handoff fields. [Current receiver validation][lm-geometry] · [Request metadata][req-meta] · [Per-bank block IDs][meta-banks] · [Handoff intent][p-cont]

## One layer on the proposed fast path

**Reload historical rows once across the TP group.** For each existing LMCache page and each rank, intersect its valid global interval with that rank's G0 stripes. Use `DcpWriteLayout.runs` as the reference for a vectorized block planner. Emit two-plane DMA descriptors through the existing bank stream. For a page starting at token `a`, physical capacity `C` tokens, and layer base `P`, a BF16 run starting at `p` reads latent at `P + (p-a)*1024` and RoPE at `P + C*1024 + (p-a)*128`. Take `C` from the actual layer allocation. A partial 1,024-token page may expose fewer valid rows. Keep page handles and prior-producer fences until the last read. Coalesce adjacent descriptors only when both source and destination are contiguous and within the owned allocation. Reuse the branch's incremental map and pointer binding across layers. Invalidate a changed tail, bank allocation or CPU page generation before reuse. [Stripe runs][stripe] · [Host plane binding][dma-bind] · [Bank H2D and per-request completion][bank-load]

**Exchange history before its first attention consumer.** Start as soon as that layer's active local stripes are ready and the exchange workspace is free. Use a deterministic order within the local DCP group and device events or stream-ordered work handles. This can overlap local Q projection, Lightning Indexer on producer layers, or other independent work. Do not count time overlapped with those operations twice. Shared-index consumers have a shorter overlap window. The initial implementation uses two contiguous BF16 plane gathers into native attention storage. A contiguous compact shard can be passed directly. For fragmented bundle IDs, fuse the two local block selections into one local pack reading only the requested shard. Native DMA directly into the compact send region is also valid if that region is the **allocated shard backing** with the proper bank lifetime. Avoid a second H2D or an unaccounted duplicate bank. The gathered destinations are already paged latent and RoPE tensors. The same block table addresses both. There is no full-history receive transpose, unpack or per-token ID remap. [Upstream compact gather reference][up-gather] · [Existing SFA KV and table overrides][override]

**Reuse today's fresh-row exchange.** Let `exec_kv` produce the fresh normalized latent and RoPE for the local query partition, then keep the current TP exchange, including the existing C8 or BF16 indexer producer behavior. Feed the returned rows to their dense attention destinations on every participant and to their local G0 owner where physical shard storage is required. The optimized scatter can read each fresh row once, split its planes and conditionally write the owner destination. Retain the token and padding validity rules. The fused normalization op itself accepts cache destinations, so supply valid per-row destinations or proven scratch for nonowners while it produces those outputs. A rank's fresh keys are needed by peers even when it owns none of the corresponding physical slots. The history exchange excludes these fresh rows. When the batch has no history, bypass the history pack and both history collectives. [Fused KV producer][fresh-proj] · [Existing fresh-row collective and plane order][p-fresh]

**Read the result with ordinary prefill attention.** Keep causal lengths, logical Top-K IDs, local query order and the existing output projection. A producer's C8 key/scale and shared-consumer Top-K buffers continue to use the P path. A TP rank with zero real query rows still contributes its history stripes. Rank 0 also needs the full save range of requests whose queries ran on peers. Within one DP instance all ranks derive the same shapes and collective sequence from the scheduled batch. A DP peer with a different batch uses its own DCP group. There is no new inter-DP agreement or per-layer host synchronization in this design. Prefill compute is already query-partitioned, so the first-order gain to expect is PCIe relief. [Query partition][p-query] · [SFA overrides][override]

### Concrete layout for chunk tails and mixed batches

The model runner builds these block maps once per scheduled batch after row reorder. Let `H_i` be the prefix whose KV is valid before this forward and `E_i` the visible end including fresh rows for request `i`. Resolve logical coverage from the load/compute metadata and verify all referenced spans are available. For block size `B=128` the layout is

```text
h_i = ceil(H_i / B)                    historical logical blocks
m_i = ceil(h_i / W)                    stripe columns for this request
o_i = sum(m_j for j < i)               within-rank gather offset
M   = sum(m_i)                         equal HCCL send slots on each rank
n_i = ceil(E_i / B) - h_i              blocks containing only new rows
u_i = sum(n_j for j < i)               offset among new-only blocks

history block b < h_i:  slot = 1 + (b % W)*M + o_i + b//W
new-only block b:       slot = 1 + W*M + u_i + b - h_i
```

Slot zero is reserved. Values are physical **block** IDs in the temporary view. Both planes use the original intra-block offset. Construct deterministic padding in equal-size sends, independent of local query count. The work scales with active blocks, including up to `W-1` padding blocks per request in the history exchange. A rank reads from the real local bank IDs, which can be fragmented or differ between banks. This construction eliminates a gathered row permutation on the consumer. A partially filled history block is shared by old and new positions. Its full-block receive must finish **before** fresh writes to that block. New-only destinations are disjoint. SFA and rank-0 save must see both operations complete. Deduplicate any physically shared source pages in a reference implementation only when allocation identity, bank and generation match. The supplied launcher disables vLLM prefix caching.

## Single-writer offload, RemoteFill and bank reuse

For this sender, TP rank 0 can use the complete attention view as its G0 D2H source. Keep passive workers' layer callbacks and avoid additional transfers for publication. Rank 0 resolves **the storer's actual save range**, including an earlier prefix within a partial LMCache chunk when the save offset rounds down. Bind the scratch-plane addresses and their view block table to the primed P storer. It snapshots transfer inputs before model execution. Changing only the list of tensors given to the attention layer's save callback would leave DMA descriptors pointing at the old bank addresses. Precompute the binding using the scheduled view and workspace slot before priming, or defer just the G0 source binding until that view is selected. Keep CPU destination reservation and G1 preparation on the existing path. [Save range][save-range] · [Primed storer][primer] · [Passive ranks][passive] · [D2H submit][bank-save]

| Resource | Last consumer that must finish before reuse |
| --- | --- |
| G0 bank or compact send buffer | The last KV write/use and the HCCL send read. If a pack materializes independent input, the physical source bank can retire after its last local read. Retain the packed buffer through HCCL completion. |
| Dense attention workspace | All SFA reads and the writer's D2H for that layer. A deferred save at the start of the next layer can otherwise observe its successor's data. Preserve per-request completion, including multiple requests sharing a bank. |
| Shared CPU page | Fence rank 0's prior D2H before publishing a readable handle. Keep source references for each reader's H2D. Keep the writer's RemoteFill lease until native completion is known, even after its HBM bank is reused. |
| P/D visibility | Preserve window reservation, arming, transfer completion and terminal handoff. Require both persistent group frontiers through the promised end, including reused pages. On a native completion of unknown status, retain the existing quarantine and fatal recovery. |

Keep G0 and G1 **group-local layer ordinals** for the four FIFO streams per group. At entry to the next layer, submit the pending save first, obtain its recorded D2H event, then enqueue the next write to a reused dense workspace behind that event. This ordering avoids waiting on a save that has yet to be submitted. The existing connector advances its H2D cursor by the bank count. Preserve that lookahead, the explicit final-target and MTP drains, cancellation and generation checks. The decoder's cooperative post-forward checkpoint capture is useful for validating stripe addresses. It is too late to read a prefiller layer that has already rotated. [Transfer-window submission][window] · [Four-bank lookahead][cursor] · [Bank and source events][bank-events] · [Final layer and MTP][tails]

Ranks can safely **read** different offsets of the same published CPU chunk under their existing shared-cache handles. Only rank 0 writes each new canonical chunk and publishes its native-transfer lease. A multiwriter implementation would instead require one shared reservation, disjoint per-layer plane ranges and evidence of completion from every contributor before publishing. The full attention view on rank 0 makes that coordination and a second publication gather unnecessary. [CPU-page handoff][fill-src] · [Terminal publication][finish]

**ADXL boundary.** The Mooncake transport and registration stay in place. With the supplied connector and per-chunk route, the native transfer's source is the leased, completed P **CPU** page. The P HBM bank was read by the preceding D2H. This is why the bank may rotate while the P/D transfer is outstanding. For a deployment that requires direct P HBM → D CPU, a separate experiment can assemble the same canonical page in a bounded, registered HBM slot on rank 0 as its layers finish. Arm the existing page transfer only after source coverage and producer fences are complete. Retain the slot until terminal native completion and charge its D2D and HBM cost to that variant. A pointer into an already reused bank cannot serve as that source. [Direct-store bypass][direct-mode] · [Fence and lease release][lease-end] · [Transport registration and submission][transport]

## Costs, rollout and decision

For BF16 G0 with 512 latent and 64 RoPE values, each historical token is 1,152 bytes. On the final chunk of a 60,000-token prompt with 16,384-token chunks, the previous prefix is 49,152 tokens. The counts below are **per G0 layer**.

| Operation | Four-bank DCP1 | Proposed TP8/DCP8 |
| --- | ---: | ---: |
| Historical H2D per rank | 54 MiB | 6.75 MiB |
| Historical H2D across the host | 432 MiB | 54 MiB |
| Historical payload received from the other seven ranks, per rank | 0 | 47.25 MiB |
| Historical H2D across the host over all four chunks | 864 MiB | 108 MiB |

The CPU reload total falls by 87.5% for G0 in this example. The added HCCL traffic is local to the P host. The figures are logical payloads. Link traffic, padding for other shapes, AIV contention and bank waits affect wall time. For a TTFT win, the exposed H2D wait saved must exceed the added exposed gather, packing, scatter and workspace-reuse waits. A baseline already hiding all reloads can save PCIe bandwidth without improving single-request latency. G1 H2D, rank-0 D2H, fresh-row traffic and P/D page bytes retain their existing roles. A cold first chunk offers zero historical H2D savings.

Start with one reusable dense compute view and a bounded local send workspace. For a given step the dense planes need `(1 + W*M + sum(n_i)) * B * 1152` bytes per rank. An aligned 1,048,576-token history alone is 1.125 GiB per rank. Eight long histories can require much more. `max_num_batched_tokens=16384` limits fresh queries and gives no such bound on history. Reserve before sizing the shared child pool and use the minimum usable capacity across P ranks. Derive an upper bound from admitted cache columns and scheduler limits. If the chosen workspace covers a smaller batch, constrain the **sum of referenced blocks** before dispatch and carry whole requests to the next step. An individually supported maximum-length request must fit. Size the RAM admission and Mooncake quotas for canonical pages. Reuse existing workspaces for fresh outputs when possible and account for temporary pack and HCCL storage. If one dense buffer makes the next gather wait for writer D2H, measure a second bounded buffer with both slots included in the HBM budget. Four rotating banks do not require four full dense views.

1. **Establish parity.** Build the sender geometry and DMA path with an upstream-style referenced-block gather as the development reference. Compare it with current DCP1 on identical prompts. Prove per-layer G0 bytes, C8 keys/scales, Top-K and logits, both persistent groups, the decoder's restore and the first generated token. Then compare the history-only gather, ranks with no owned bytes, ragged batches, block/chunk boundaries, resumed chunked prefill, external prefix hits, 1-token/MTP tails and canceled or reused allocations. Exercise delayed bank DMA, a slow writer and a transport failure before and after arming. A poisoned bank immediately after allowed reuse must leave transferred page bytes intact.
2. **Measure the complete path.** Benchmark production C8 indexer and shared-index layers, MTP, four banks and the real P/D connector. Cover cold one-chunk, 60K four-chunk, warm-prefix and long ragged requests at batches 1/2/4/8. Record actual H2D descriptor bytes per group, slowest-rank exposed HCCL and bank waits, local pack/scatter and D2H time, and peak HBM. Time P handoff completion, D's first generated token, client TTFT p50/p95 and sustained P throughput separately. Compare HCCL under the same AIV setting and concurrent ADXL traffic. The existing local profile and correctness harnesses can instrument the bank pipeline. Qualify native transfer with the Mooncake check and the deployed P/D service. [Profile harness][profile] · [Correctness harness][correctness] · [Mooncake check][mc-check]
3. **Select the exchange from measurements.** Compare two plane gathers writing native layout with one packed gather plus any *actual* conversion required by native SFA. Include implicit contiguous copies and the potential loss of four-bank overlap. Keep a fused local compaction or fresh-row scatter only when slowest-rank stage time and end-to-end results justify it. For short requests, verify the cold path bypasses historical work. Enable P DCP in production only after long-prompt TTFT or throughput improves at the target mix and the cold case and handoff remain within measured baseline variation.
4. **Enable on P after qualification.** In the prefiller launcher change to `--decode-context-parallel-size 8 --block-size 128 --cp-kv-cache-interleave-size 128`, and update its printed DCP value. Keep P indexer request sharding at `0`, four banks, eager chunked prefill, MTP=1 and the supplied P/D LMCache configuration. Use the existing P marker and numerical DCP size as the selector. Rollback switches P to DCP1 on restart. The decoder uses the same canonical page contract with either sender.

**Verification performed for this design.** A CPU model using the branch's actual `DcpWriteLayout.runs` checked 84 batch/width/bank combinations and 288 canonical save pages for W=2/4/8, fragmented physical IDs, reordered requests, cold and partial histories, rounded save starts and partial or padded host allocations. The reconstructed data and untouched padding matched byte-for-byte. The existing DMA/DCP-capture suites passed 111 tests and the RemoteFill protocol, transport and transaction-lease suites passed 56. These checks validate addresses and existing contracts. The implementation adds CPU regressions for the stripe planner, packet kernels, bank fences and canonical writer. Native timings and end-to-end P/D behavior still require the hardware runs above.

[up-meta]: https://github.com/vllm-project/vllm-ascend/blob/155f68974c802633c19650ece430d41ee6802fa6/vllm_ascend/attention/context_parallel/sfa_cp.py#L278-L290
[up-gather]: https://github.com/vllm-project/vllm-ascend/blob/155f68974c802633c19650ece430d41ee6802fa6/vllm_ascend/attention/context_parallel/sfa_cp.py#L428-L491
[up-attn]: https://github.com/vllm-project/vllm-ascend/blob/155f68974c802633c19650ece430d41ee6802fa6/vllm_ascend/attention/context_parallel/sfa_cp.py#L697-L728
[platform]: https://github.com/shaaaaaa/vllm-ascend/blob/f8bef4fc10b1388083dc83b758aa0bdfd5d9fa8d/vllm_ascend/platform.py#L283-L307
[select]: https://github.com/shaaaaaa/vllm-ascend/blob/f8bef4fc10b1388083dc83b758aa0bdfd5d9fa8d/vllm_ascend/attention/sfa_v1.py#L934-L951
[runner-spec]: https://github.com/shaaaaaa/vllm-ascend/blob/f8bef4fc10b1388083dc83b758aa0bdfd5d9fa8d/vllm_ascend/worker/model_runner_v1.py#L7138-L7184
[runner-scratch]: https://github.com/shaaaaaa/vllm-ascend/blob/f8bef4fc10b1388083dc83b758aa0bdfd5d9fa8d/vllm_ascend/worker/model_runner_v1.py#L6879-L6914
[bank-tables]: https://github.com/shaaaaaa/vllm-ascend/blob/f8bef4fc10b1388083dc83b758aa0bdfd5d9fa8d/vllm_ascend/worker/npu_input_batch.py#L95-L129
[p-query]: https://github.com/shaaaaaa/vllm-ascend/blob/f8bef4fc10b1388083dc83b758aa0bdfd5d9fa8d/vllm_ascend/attention/sfa_v1.py#L1753-L1799
[fresh-proj]: https://github.com/shaaaaaa/vllm-ascend/blob/f8bef4fc10b1388083dc83b758aa0bdfd5d9fa8d/vllm_ascend/attention/sfa_v1.py#L2524-L2553
[p-fresh]: https://github.com/shaaaaaa/vllm-ascend/blob/f8bef4fc10b1388083dc83b758aa0bdfd5d9fa8d/vllm_ascend/attention/sfa_v1.py#L5006-L5100
[override]: https://github.com/shaaaaaa/vllm-ascend/blob/f8bef4fc10b1388083dc83b758aa0bdfd5d9fa8d/vllm_ascend/attention/sfa_v1.py#L2880-L2910
[tails]: https://github.com/shaaaaaa/vllm-ascend/blob/f8bef4fc10b1388083dc83b758aa0bdfd5d9fa8d/vllm_ascend/attention/sfa_v1.py#L5523-L5577
[pool]: https://github.com/shaaaaaa/vllm/blob/28e80e0b1a2fd5b307b974c745430127b7bb4019/vllm/v1/core/dsa_shared_pool.py#L555
[coord]: https://github.com/shaaaaaa/vllm/blob/28e80e0b1a2fd5b307b974c745430127b7bb4019/vllm/v1/core/kv_cache_coordinator.py#L213-L232
[concur]: https://github.com/shaaaaaa/vllm/blob/28e80e0b1a2fd5b307b974c745430127b7bb4019/vllm/v1/core/kv_cache_utils.py#L1015-L1035
[budget]: https://github.com/shaaaaaa/vllm/blob/28e80e0b1a2fd5b307b974c745430127b7bb4019/vllm/v1/core/kv_cache_utils.py#L2265-L2288
[stripe]: https://github.com/qiuziyan1998/LMCache/blob/8d32a083c83628cb2b84f59c4358e550ccad1744/lmcache/integration/vllm/dcp.py#L72-L98
[lm-geometry]: https://github.com/qiuziyan1998/LMCache/blob/8d32a083c83628cb2b84f59c4358e550ccad1744/lmcache/integration/vllm/vllm_v1_adapter.py#L2225-L2258
[req-meta]: https://github.com/qiuziyan1998/LMCache/blob/8d32a083c83628cb2b84f59c4358e550ccad1744/lmcache/integration/vllm/vllm_v1_adapter.py#L1905-L1947
[meta-banks]: https://github.com/qiuziyan1998/LMCache/blob/8d32a083c83628cb2b84f59c4358e550ccad1744/lmcache/integration/vllm/vllm_v1_adapter.py#L3019-L3031
[p-cont]: https://github.com/qiuziyan1998/LMCache/blob/8d32a083c83628cb2b84f59c4358e550ccad1744/lmcache/integration/vllm/vllm_v1_adapter.py#L12289-L12325
[save-range]: https://github.com/qiuziyan1998/LMCache/blob/8d32a083c83628cb2b84f59c4358e550ccad1744/lmcache/integration/vllm/vllm_v1_adapter.py#L10157-L10208
[primer]: https://github.com/qiuziyan1998/LMCache/blob/8d32a083c83628cb2b84f59c4358e550ccad1744/lmcache/integration/vllm/vllm_v1_adapter.py#L10315-L10391
[window]: https://github.com/shaaaaaa/vllm-ascend/blob/f8bef4fc10b1388083dc83b758aa0bdfd5d9fa8d/vllm_ascend/attention/sfa_v1.py#L4781-L4808
[cursor]: https://github.com/qiuziyan1998/LMCache/blob/8d32a083c83628cb2b84f59c4358e550ccad1744/lmcache/integration/vllm/vllm_v1_adapter.py#L9209-L9227
[dma]: https://github.com/qiuziyan1998/LMCache-Ascend/blob/81c135bf512f0e46ab65fa0b3e549a5bb8e9eff0/lmcache_ascend/v1/npu_connector/layerwise_dma.py#L147-L210
[dma-bind]: https://github.com/qiuziyan1998/LMCache-Ascend/blob/81c135bf512f0e46ab65fa0b3e549a5bb8e9eff0/lmcache_ascend/v1/npu_connector/layerwise_dma.py#L558-L594
[bank-load]: https://github.com/qiuziyan1998/LMCache-Ascend/blob/81c135bf512f0e46ab65fa0b3e549a5bb8e9eff0/lmcache_ascend/v1/npu_connector/npu_connectors.py#L6897-L6927
[bank-save]: https://github.com/qiuziyan1998/LMCache-Ascend/blob/81c135bf512f0e46ab65fa0b3e549a5bb8e9eff0/lmcache_ascend/v1/npu_connector/npu_connectors.py#L8044-L8065
[bank-events]: https://github.com/qiuziyan1998/LMCache-Ascend/blob/81c135bf512f0e46ab65fa0b3e549a5bb8e9eff0/lmcache_ascend/v1/npu_connector/npu_connectors.py#L2068-L2125
[direct-mode]: https://github.com/qiuziyan1998/LMCache-Ascend/blob/81c135bf512f0e46ab65fa0b3e549a5bb8e9eff0/lmcache_ascend/integration/vllm/vllm_v1_adapter.py#L519-L531
[fill-groups]: https://github.com/qiuziyan1998/LMCache-Ascend/blob/81c135bf512f0e46ab65fa0b3e549a5bb8e9eff0/lmcache_ascend/v1/cache_engine.py#L525-L536
[fill-src]: https://github.com/qiuziyan1998/LMCache-Ascend/blob/81c135bf512f0e46ab65fa0b3e549a5bb8e9eff0/lmcache_ascend/v1/cache_engine.py#L1568-L1627
[lease-end]: https://github.com/qiuziyan1998/LMCache-Ascend/blob/81c135bf512f0e46ab65fa0b3e549a5bb8e9eff0/lmcache_ascend/v1/cache_engine.py#L1748-L1761
[finish]: https://github.com/qiuziyan1998/LMCache-Ascend/blob/81c135bf512f0e46ab65fa0b3e549a5bb8e9eff0/lmcache_ascend/v1/cache_engine.py#L1763-L1818
[passive]: https://github.com/qiuziyan1998/LMCache-Ascend/blob/81c135bf512f0e46ab65fa0b3e549a5bb8e9eff0/lmcache_ascend/v1/cache_engine.py#L6109-L6127
[transport]: https://github.com/qiuziyan1998/LMCache/blob/8d32a083c83628cb2b84f59c4358e550ccad1744/lmcache/v1/remote_fill/mooncake_transport.py#L206-L264
[profile]: ../../../../tools/layerwise_prefill_profile.py
[correctness]: ../../../../tools/layerwise_prefill_correctness.py
[mc-check]: ../../../../tools/layerwise_prefill_mooncake_check.py
