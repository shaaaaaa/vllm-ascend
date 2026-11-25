# Replay directly from raw snapshot files

`prefill_kv_replay.py` and `decode_kv_replay.py` read the existing raw snapshot
files using already-running P/D services. The native cache injector, standalone
store hook, device setup and process-environment discovery have been removed.

Update **LMCache and vllm-ascend** on `dev-lmy`. Set the following in the launch
environment of **both P and D**, then restart the services once:

```bash
export LMCACHE_FILE_REPLAY_ENABLED=true
```

The flag defaults to false. Only explicitly tagged replay reads are redirected;
ordinary requests and all writes retain the normal backend. Keep `p_use` and
other diagnostic synchronization off when reproducing the original fault.

## Commands

Run from the client/P machine against the already-running deployment:

```bash
python3 tools/prefill_kv_replay.py \
  --run-dir ./multi-turn-concurrent --session 11 --round 1 \
  --base-url http://7.150.2.72:8080 \
  --p-url http://7.150.2.72:7910 \
  --snapshot-hosts root@7.150.3.60 root@7.150.6.177 \
  --container vllm-ascend-v0.18.0rc1-lmcache-ascend-v0.4.3-fsi-sqh \
  --workspace /workspace/sqh/vllm-ascend \
  --remote-python /usr/local/python3.11.14/bin/python3 \
  --password 'YOUR_SSH_PASSWORD' --skip-host-key-check \
  2>&1 | tee prefill-replay.log
```

To test D alone using the saved erroneous request's received KV:

```bash
python3 tools/decode_kv_replay.py \
  --request-id 896251f4-42cb-4e56-9cfd-ceb283a7c9c2 \
  --d-url http://7.150.3.60:7920 \
  --model-name GLM-5.3-falcon \
  --snapshot-host root@7.150.3.60 \
  --container vllm-ascend-v0.18.0rc1-lmcache-ascend-v0.4.3-fsi-sqh \
  --workspace /workspace/sqh/vllm-ascend \
  --remote-python /usr/local/python3.11.14/bin/python3 \
  --password 'YOUR_SSH_PASSWORD' --skip-host-key-check \
  2>&1 | tee decode-replay.log
```

Update the tools on the client and snapshot host together. `--workspace` is the
path inside the container; the default source is `WORKSPACE/decode-kv-snapshots`.
Legacy snapshots without `replay.json` remain supported using the original
`multi-turn-concurrent` directory. Source files are never overwritten.

For P replay, pass **all possible D snapshot hosts** via `--snapshot-hosts`.
The tool resolves the previous request ID from the selected run/session/round,
then probes each host in parallel using only manifests, token metadata and file
sizes. It selects the host with the complete matching snapshot before starting
the file server or submitting inference. `--snapshot-host` remains available
when the source host is already known. Do not assume a new run used the same D
host as an earlier run, even when its session and round numbers are identical.

`snapshot_discovery.json` records each host's result: `missing_directory`,
`request_not_found`, `invalid_snapshot`, `probe_failed`, or `ready`. A missing
request is distinct from failed SSH or a truncated/incomplete capture. If no
host has a complete capture, replay stops without sending a completion request;
it does not substitute another request or regenerate the prefix. If multiple
hosts have complete copies, select one explicitly with `--snapshot-host`.
`--prepare-only` also performs this metadata check, without reading KV payloads.

## File transport

The script starts a temporary read-only HTTP service on the snapshot host.
`--local-hostname` overrides the advertised IP, which otherwise comes from the
SSH host. `--file-port` selects a fixed port; default zero chooses a free port.
P and D must be able to reach this address, including through container
networking. No device or HCCL environment is required for the file server. For local files,
omit the SSH/container flags and set a reachable `--local-hostname`.

The service validates model/layout metadata and every captured TP copy, then
serves only the selected historical token ranges. Responses stream layer-major
raw bytes in bounded 4 MiB pieces; BF16/C8 indexer planes retain their original
representation. The live LMCache token database generates keys using its
existing hash root and installed Ascend hash implementation.

P replay reads only `[0, original_cached_tokens)` from files and recomputes the
original tail. For cached=499712 and prompt=500095 that is exactly 383 tokens.
Both P and D can read the historical file prefix. **The new tail computed by P
still uses normal P-to-D transfer**, including persistent fallback. This avoids
asking D to find historical Mooncake entries that were never injected.

D-only replay reads the complete saved receive, including the final partial
block. D retains its normal final-token recomputation, kernels, graphs and MTP.
An unrecorded legacy bootstrap is not invented; ordinary cold decode is used
and this limitation is retained in the report.

## Verification and cost

Destination buffers acknowledge each page only after the copy completes. For
direct NPU destinations the copy stream is synchronized before acknowledgement.
Truncated/short reads fail. `file_reads.json` records acknowledged pages for
both groups, separately for P and D. Each reader record includes the host,
peer IP, PID, rank, role and completed page count. P replay requires P to have
read the full selected prefix; D reads cannot satisfy that check. D-only replay
requires D to have read it. `file_source.json` identifies the source and
`file_source.log` records setup progress. `file_plan.json` records the input plan.

The output directories remain `prefill-kv-replays/...` and
`decode-kv-replays/...`, with `output.txt`, `response.json` and `report.json`.
Cache-count mismatches fail; omitted API usage remains `replay_unverified`
(exit 2), with output retained. Copy completion does not prove model arithmetic
correctness. A single-request replay does not recreate the original mixed batch.

File reads have a bounded two-hour backend budget; stalled HTTP reads fail
after 120 seconds. Ordinary requests keep their original backend deadlines.
Service RPC and proxy deadlines still apply. `--workers` controls offline TP
validation. `--prepare-only` skips file serving and inference.

No second full KV copy is staged into Mooncake, and raw KV does not travel to
the client. P/D still read the **full historical KV bytes**: ~47 GB over a
100 Mb/s link can take about an hour per copy. This removes injection failures,
not the bandwidth cost.

## Four-machine routing

The snapshot stays on the host selected by `--snapshot-host`. For example,
`root@7.150.3.60` advertises `http://7.150.3.60:<port>/<run-id>`. The replay request
carries this source in `kv_transfer_params["lmcache.tag.file_replay"]`.
The normal and enhanced PD proxies preserve it when constructing both P and D
requests. If the proxy chooses P=2-72 or P=3-11, either one reads the same source;
D=3-60 and D=6-177 can do so too. No local copy is required on the chosen worker.
`--p-url` checks tokenization/model identity; it does not pin proxy routing.

Update the proxy and all possible P/D destinations, enable the file reader on
all of them, and make the advertised source port reachable from their containers.
The script validates the selected startup logs and reports actual acknowledged
readers, but does not probe every possible proxy destination before dispatch.
A file read/connect error fails the replay; incomplete acknowledgements fail its
verification. Missing usage fields still give an unverified result, not success.
Use `--file-port` when a fixed allowed port is needed.

Large KV bytes travel from the snapshot host to the selected worker, not through
the replay client. This is still a full data transfer: the source and destination
need sufficient bandwidth. `--source-timeout` controls source setup/validation
(default 7200 seconds); `--timeout` controls the inference request.
