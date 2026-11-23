# Decode copy scheduling experiment

Use the matching vllm-ascend and LMCache-Ascend branches. The D worker runs a full target graph with shared resident planning and shared retrieval overlap. The experiment changes the copy used by an asynchronous prefetch edge. Each piece reads the live miss counts, source pointers and selected token indices on the device.

## Verify on a D worker

Run the priority probe inside the CANN container. One initial replay binds the physical submission queues on CANN graphs that allocate them lazily. The probe then sets the copy queue's priority and checks three more replays.

```bash
python tools/check_decode_graph_priority.py --device 0
```

Look for `graph_priority_verified`. The JSON records the initial replay in `binding_replay_verified`, the copy queue's update in `set_side.rc`, device readbacks for both graph queues in `after_set` and `after_replay`, and the three subsequent data-checked replays in `verified_replays`.

If the probe reports a failure, it also runs controls on ordinary streams. Compare `set_side` with `ordinary.set_side` for updates, and the graph readbacks with `ordinary.after_set` for queries. `ordinary.create_low.creation_field` reports the priority recorded at stream creation. `ordinary.create_low.device_read` queries the scheduler. The probe includes the SoC name and the installed driver and firmware versions when their version files are visible in the container.

When both ordinary and graph priority controls return a driver error, check the host firmware version and kernel task-scheduler messages before benchmarking. The current D worker applies priority at capture end. A deployment that binds hardware queues on first replay needs a post-binding configuration in the D worker before running case C.

To validate split-copy correctness, build LMCache-Ascend and run its registered-host-memory transfer test from that checkout with the same Python environment as the D worker.

```bash
python -m pytest -q --confcutdir=tests/v1 tests/v1/test_sparse_graph_transfer_npu.py -k prefetch_pieces
```

## Compare three serving configurations

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
