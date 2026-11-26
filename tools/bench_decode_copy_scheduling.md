# Decode copy scheduling experiment

Use the matching vllm-ascend and LMCache-Ascend branches. The D worker runs a full target graph with shared resident planning and shared retrieval overlap. The experiment changes the copy used by an asynchronous prefetch edge. Each piece reads the live miss counts, source pointers and selected token indices on the device.

## Verify on a D worker

### Ascend 910B3 with CANN 9.1 and HDK 25.5.2

Use one idle NPU in the D container. The following command both runs the short scheduling test and captures a MindStudio profile. It needs the installed CANN math operators. It runs independently of the P/D service.

```bash
msprof --output="$PWD/priority-profile-$(date +%Y%m%d-%H%M%S)" \
  python tools/check_decode_creation_priority.py --device 0
```

The probe explicitly binds two persistent streams as heads of one ACL model. Each stream executes 48 identical FP16 Abs operations. It tries priorities A/B of 0/0, 0/7 and 7/0, then repeats with the creation, binding and enqueue order reversed. For each graph it warms up once and checks the replay outputs and two device-side completion timestamps on five measured iterations. The JSON field `b_minus_a_us` is positive when A finishes first. The six model names and stream IDs in `[CREATE_PRIORITY]` identify the lanes in MindStudio.

`creation_priority_effect_observed` means the higher-priority lane finished first on at least four of five replays in both directions and both creation orders, beyond the equal-priority lead and jitter. `no_clear_creation_priority_effect` reports a completed measurement with its six sets of timings. `failed` identifies a setup, replay or data-check error.

To inspect progress inside each replay, use its existing msprof export and the two stream IDs printed in the first case. This example uses a capture that reused streams 76 and 75 in creation order.

```bash
python tools/summarize_creation_priority.py priority-profile-20261124-061758 --streams 76 75
```

The summary measures device Abs tasks directly. It validates 48 kernels on each lane for each of 36 replays, identifies cases chronologically even when model IDs repeat, and omits the warm-up in each case. `ready_start_counts` counts which lane starts the next kernel when both still have work and both preceding kernels have ended, summed over the five measured replays. It skips initial branch activation and counts equal timestamps as ties. Positive `b_minus_a_half_us` and `b_minus_a_last_us` indicate that A finished its 24th and 48th kernels earlier. `a_lead_at_half_kernels` gives the difference in completed kernels when either lane reaches 24. `interval_overlap_pct` is the fraction of the shorter lane's cumulative device task time that intersects tasks from the other lane. These four progress measurements are reported as medians across the five replays.

On this 910B3, changing an existing queue's priority returned device status 4. The released 910B TS dispatchers in HDK 25.5.2, 26.0.RC1, 26.1.1 and 26.2.RC1 return 4 for the same update/query commands (14/15). The 25.5.2 queue creation path programs the requested three-bit priority. In stock CANN 9.1, automatic ACL graph capture creates internal streams at priority 0, and the graph queue pool also allocates at 0. Keep `VLLM_ASCEND_SFA_PREFETCH_LOW_PRIORITY=0` for the current 910B3 worker. A positive explicit-model result motivates propagating priority to the queues used by serving capture and then measuring TPOT. [Capture stream creation](https://github.com/hicann/runtime/blob/v9.1.0/src/runtime/feature/aclgraph/context_aclgraph.cc#L170) and [graph queue pool](https://github.com/hicann/runtime/blob/v9.1.0/src/runtime/core/src/device/device_sq_cq_pool.cc#L76).

### Verify dynamic priority on a deployment using updates after capture

Run the dynamic priority probe inside the CANN container. One initial replay binds the physical submission queues on CANN graphs that allocate them lazily. The probe then sets the copy queue's priority and checks three more replays.

```bash
python tools/check_decode_graph_priority.py --device 0
```

Look for `graph_priority_verified`. The JSON records the initial replay in `binding_replay_verified`, the copy queue's update in `set_side.rc`, device readbacks for both graph queues in `after_set` and `after_replay`, and the three subsequent data-checked replays in `verified_replays`.

If the probe reports a failure, it also runs controls on ordinary streams. Compare `set_side` with `ordinary.set_side` for updates, and the graph readbacks with `ordinary.after_set` for queries. `ordinary.create_low.creation_field` reports the priority recorded at stream creation. `ordinary.create_low.device_read` queries the scheduler. The probe includes the SoC name and the installed driver and firmware versions when their version files are visible in the container.

The current D worker applies priority at capture end. A deployment that binds hardware queues on first replay needs a post-binding configuration in the D worker before running case C. Run case C after verifying dynamic priority on the actual capture path of that deployment.

To validate split-copy correctness, build LMCache-Ascend and run its registered-host-memory transfer test from that checkout with the same Python environment as the D worker.

```bash
python -m pytest -q --confcutdir=tests/v1 tests/v1/test_sparse_graph_transfer_npu.py -k prefetch_pieces
```

## Compare serving configurations

### Compare the CANN 8.5.1 main-path gate

Use the existing D workers with full target graphs, shared resident planning
and shared retrieval overlap. Set `VLLM_ASCEND_SFA_PREFETCH_MAIN_GATE=1`
and restart D to recapture the target graphs. This Python scheduling change
reuses the existing LMCache-Ascend admission extension. On Ascend910B3 with
CANN 8.5.1, each D copy kernel defaults to a maximum of 20 AIVs.
`VLLM_ASCEND_SFA_COPY_MAX_AIV_CORES=0` selects the automatic grid.
The startup line `SFA main-path gate captured` includes the copy cap, SoC,
CANN version and a `windows` count for each foreground kernel admitted in that graph.

For the GLM-5.3 BF16 decode path the FIFO can run during V-up, the FP32 router,
K-up BMM, and the ND × NZ W8A8 QKV-A, Q-B, O and shared/dense MLP projections.
The mixed W8A8 decoder kernel uses up to 20 AIVs. DynamicQuant is admitted when
the actual padded row count fits the remaining AIV budget. SwiGlu uses its
column-tile bound, RMSNorm uses its row bound and the residual `muls_add`
uses its actual Triton launch grid. Layout conversion runs before K-up.
Small intra-host TP4 AllReduce and post-Combine AllGather may share with copy.
The ordinary O/shared-down projection and its adjacent eligible AllReduce use one
grant. An admitted piece finishes before execution enters sparse attention,
routed Dispatch, expert GMM or Combine. [CANN W8A8 tiling](https://github.com/hicann/ops-nn/blob/9f8a66e795ef1842c2118cf5bbadfe3624bdd1ef/matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_basic_tiling.cpp) · [CANN SwiGlu tiling](https://github.com/hicann/ops-nn/blob/9f8a66e795ef1842c2118cf5bbadfe3624bdd1ef/activation/swi_glu/op_host/swi_glu_tiling.cpp) · [HCCL small AllReduce](https://github.com/hicann/hcomm/blob/2981ffa77fd5a7bfe7e64055eb3ec4fe762c0b9f/src/algorithm/impl/coll_executor/coll_all_reduce/coll_all_reduce_mesh_aiv_smallcount_executor.cc)

For the shared-indexer comparison, keep the main gate and piece count fixed
and switch `VLLM_ASCEND_SFA_PREFETCH_MULTI_LAYER` from 0 to 1 on D.
The layer CSV's `indexer_layer_for_next` identifies the layer that queued
each consumer. The trace summary pairs side-stream completion records with
individual consumer waits, including when several layers finish in one window.
Each consumer waits on its own completion event. When its replicated QKV fits
the remaining budget, the same grant spans its wait and QKV. With larger
padded batches the drain closes before DynamicQuant and the GEMM can reopen it.

This short standalone test replays the same graph with different readiness
inputs. One replay forces all eight pieces through the router window, one
requires at least four through V-up, and one defers all eight to the tail.

```bash
timeout 90s python -m pytest -q -s --confcutdir=tests/e2e/singlecard -o addopts= \
  tests/e2e/singlecard/test_sfa_retrieval_overlap.py -k dynamic_copy_windows_same_graph
```

Run from the selected D instance's container with its serving Python.

```bash
python tools/bench_decode_copy_scheduling.py \
  --base-url "$PROXY_URL" --target-d-url "$D_URL" --model "$SERVED_MODEL" \
  --concurrency 16 --prefix-tokens 8192 --output-tokens 128 \
  --rounds 1 --capture --profile-dir auto --output "results/main-gate-$(date +%s)"
```

Read `pieces_per_window_histogram` for V-up, router and tail. For example,
a router entry of `"3": 100` means 100 layer/replay samples ran three
copy kernels in that window. A piece with zero live rows also counts as a
kernel launch. `vup_close_to_acquire` and `router_close_to_acquire`
measure the time from closing the admission word to reacquiring the token.
The histogram labels extra graph grants as `additional`. For a breakdown of
device overlap by operator and per-replay copy stall, use
`python tools/analyze_decode_moe_copy.py trace_view.json --output summary.json`.
The main-stream overlap report covers compute kernels. Inspect the HCCL stream
in MindStudio to see communication overlap. Compare real TPOT with profiling
disabled. Individual kernels also share memory bandwidth.

Use `VLLM_ASCEND_SFA_PREFETCH_MAIN_GATE=0` for the ordinary overlapped
schedule. To compare timing against an existing run of the same workload,
add `--reference results/ungated/metrics.json`.

### Compare native stream priority

Set the variables in the D worker launch environment. Restart the participating D workers between configurations so each uses the corresponding captured graph.

| Run | `VLLM_ASCEND_SFA_PREFETCH_PIECES` | `VLLM_ASCEND_SFA_PREFETCH_LOW_PRIORITY` |
| --- | ---: | ---: |
| A | 1 | 0 |
| B | 16 | 0 |
| C | 16 | 1 |

Use the existing P/D proxy and select one D instance. Run the following in that D container. Set `PROXY_URL`, `D_URL` and `SERVED_MODEL` to its deployment values. Each invocation defaults to 16 concurrent requests, 8192 input tokens and 128 output tokens.

```bash
common=(--base-url "$PROXY_URL" --target-d-url "$D_URL" --model "$SERVED_MODEL" --capture --profile-dir auto)

# Once each respective worker configuration is running
python tools/bench_decode_copy_scheduling.py "${common[@]}" --output results/A
python tools/bench_decode_copy_scheduling.py "${common[@]}" --output results/B --reference results/A/metrics.json
python tools/bench_decode_copy_scheduling.py "${common[@]}" --output results/C --reference results/A/metrics.json
```

The client warms the service, times three batches with the profiler disabled and then profiles one extra batch. The proxy starts the D profiler after all prefills have finished. `--profile-dir auto` uses the directory reported by the selected D instance. When its directory is mounted at a different location, pass that local absolute path explicitly.

Each run saves `metrics.json` with TPOT, TTFT, end-to-end request latency, batch makespan and generated-token hashes. `profile/traces.json` lists the new MindStudio trace files. The per-rank JSON summaries and layer CSVs are in the same client output directory. Compare A and B for the cost of splitting, B and C for the effect of priority, and A and C for overall performance. For C, the D startup log includes the graph-owned copy stream ID and the priority read back on that stream.

To analyze an existing GLM trace independently, select the target graph with `--model-id` and `--stream` when needed.

```bash
python tools/analyze_decode_layer_gaps.py trace_view.json --csv layers.csv --summary summary.json
```
