# Decode copy scheduling experiment

Use the matching vllm-ascend and LMCache-Ascend branches. The D worker runs a full target graph with shared resident planning and shared retrieval overlap. The experiment changes the copy used by an asynchronous prefetch edge. Each piece reads the live miss counts, source pointers and selected token indices on the device.

## Verify on a D worker

Run the priority probe inside the CANN container. It verifies the settings on the two queues owned by a captured graph and checks the output of three replays.

```bash
python tools/check_decode_graph_priority.py --device 0
```

Look for `graph_priority_verified` and distinct values in `after_replay`. Build LMCache-Ascend once, then run its registered-host-memory transfer test from that checkout with the same Python environment as the D worker.

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
