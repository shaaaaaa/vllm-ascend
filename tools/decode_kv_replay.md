# Replay through an already-running D

`decode_kv_replay.py` injects a completed raw D snapshot into the current
Mooncake deployment and submits the recorded request directly to an existing
D API. It does not load model weights, start or stop vLLM, modify serving
environment variables, or call P/the PD proxy. Keep the D service running.

From the client machine, for the session 11 round 1 snapshot:

```bash
python3 tools/decode_kv_replay.py \
  --request-id 896251f4-42cb-4e56-9cfd-ceb283a7c9c2 \
  --d-url http://7.150.3.60:7920 \
  --snapshot-host root@7.150.3.60 \
  --container vllm-ascend-v0.18.0rc1-lmcache-ascend-v0.4.3-fsi-sqh \
  --workspace /workspace/sqh/vllm-ascend \
  --remote-python /usr/local/python3.11.14/bin/python3 \
  --seed-device 4 \
  --password-env PD_SSH_PASSWORD \
  --skip-host-key-check \
  2>&1 | tee decode-replay.log
```

`--password 'PASSWORD'` is also supported through native OpenSSH askpass; no
sshpass or interactive prompt is used. Update the tools on both the client
and snapshot host. `--workspace` is the path **inside** the container when
`--container` is present. By default, the script reads that workspace's
`decode-kv-snapshots` and current `tp4_dp4_logs/D_dp*_port7920.log` (the port
comes from `--d-url`). `--snapshot-root`
and `--d-logs` override those paths **on the snapshot host**, not the client.
The D logs must belong to the service selected by `--d-url` and contain its
startup cache layout/hash-root records. The selected D must use the same
model, TP size, and input-tail policy as the snapshot.

When running inside the D container itself, omit `--snapshot-host`,
`--container`, and the SSH options. Local metadata and raw files are used.
The script discovers the model alias through `/v1/models`; `--model-name`
selects an alias if more than one points to the saved model. The output limit
and sampling parameters are kept from the capture. `--output-tokens 128`
explicitly changes the output limit.

## Data and execution

The capture must include the complete TP ranks, full raw KV/indexer planes,
and `tokens.json`. When present, `replay.json` preserves the **actual D input
token IDs**, sampling parameters, and any `bootstrap_final_hidden` payload.
Tokens appended by the original proxy remain in that input; the script
neither tokenizes text nor synthesizes a new first token.

Older captures without `replay.json` are supported using the original client
run, defaulting to `./multi-turn-concurrent` on the machine running the command.
Use `--run-dir PATH` if that directory was moved. The script locates the unique
`response.json` matching `--request-id` and reads its sibling `request.json`
for sampling parameters. It checks the request against `summary.json` when
available. The model identity comes from the raw snapshot's manifest, and
the actual input IDs come from its `tokens.json`, not a new tokenization.

Legacy reconstruction requires a full client prompt receive: the saved
response's prompt-token count must equal the captured token count, and the
current D must use `skip_last_n_tokens=0`. It never guesses a missing
proxy-appended token. Optional final-hidden bootstrap was not recorded in
these captures: replay explicitly uses normal D cold decode, and records
this limitation in `source.json`/`report.json`. Unspecified API sampling
defaults come from the current D service. The original snapshot is not edited
and does not need to be captured again.

Only compressed metadata travels to the client. A temporary seed process
reads raw files on the snapshot host, checks that the TP copies agree, packs
the native layer-major Mooncake layout, and publishes both KV groups. This
includes the final **partial cache block**, with the native valid-token key
tag. P replay still seeds only the completed historical blocks.

The seed process uses an Ascend transport context, two pinned CPU staging
buffers and its own CPU Mooncake segment; it is not another model instance.
Its bytes travel over the native Mooncake transport. The segment stays alive
until the D response finishes, then the script releases its owner. The
default `--seed-device 0` is for transport initialization, not model loading.
Choose a device outside the target D instance: in the TP4 example above,
D port 7920 uses physical devices 0–3, so seed uses device 4. Independent
AscendDirect processes must not attempt a same-device ADXL connection. The
tool checks the selected device against the target workers' device visibility
before loading the native runtime or scanning the large archive.

Before injection, metadata collection finds the current worker PIDs in the
selected D logs and reads their `/proc/<pid>/environ` inside the serving
container. It copies an allowlist of HCCL/Ascend transport settings into the
seed plan, including `HCCL_INTRA_ROCE_ENABLE`, `ASCEND_BUFFER_POOL`,
`HCCL_IF_IP` and `HCCL_SOCKET_IFNAME`. Seed applies these **before importing
torch_npu or initializing Mooncake**. Settings absent on D are also unset
on seed. Worker rank, device visibility, model/debug flags and credentials
are not copied. This fixes seed silently using shell defaults while the
server uses RDMA and a Host-memory transfer buffer pool.

The first seed status line, `transport_config`, reports the source worker
PIDs, the seed device and the effective transport settings. An unavailable
worker environment or conflicting D configurations stops before injection.
Update the client and snapshot-host tools together; custom log names require
`--d-logs` pointing to the target D instance's current log.

A fresh `lmcache.tag.decode_replay` namespace distinguishes each replay's
LMCache keys. A fresh vLLM `cache_salt` prevents an existing NPU prefix-cache
entry from bypassing those injected bytes. Neither changes the token IDs or
the stored tensors. Existing snapshots and ordinary request cache entries
are not overwritten or cleared. D executes its current configured kernels,
graphs, MTP and scheduling. Its normal final-token recomputation is retained.

## Output and validation

The client creates `decode-kv-replays/<source UUID>-<unique suffix>/`:

- `request.json`, `source.json`: exact submitted input and saved metadata.
- `seed_plan.json`, `seed.log`, `seed_ready.json`: injection and readback checks.
- `output.txt`, `events.jsonl`, `response.json`: D output and raw SSE events.
- `report.json`: source/new request IDs, observed token counts and status.

`replay_complete` requires the reported prompt and cached-token counts to
match the saved receive boundary, capped at prompt length minus one.
A numeric mismatch fails. Missing API usage fields produce
`replay_unverified` (exit 2), never a successful cache-reuse verdict. The
output is retained in both cases; no inference retry is issued automatically.
`--prepare-only` checks metadata/API and writes the plan without injecting KV
or generating a response. `--timeout` and `--seed-timeout` default to 7200
seconds and do not change serving process timeout configuration.

The replay tests how **these saved tensors** behave in the running D. It does
not reproduce the original concurrent batch or prove deterministic output.
CPU tests cover partial-page layout/keys, metadata, HTTP request construction,
seed lifetime and validation failures. Native transport/model execution
requires the deployed Ascend environment.

## Regression audit against P replay failures

D replay calls the same `prefill_kv_replay` process/environment helpers and
`prefill_kv_seed` injector used by P replay. The regression checks cover:

| Prior failure | D replay behavior and check |
| --- | --- |
| Legacy snapshot has no `replay.json` | Recover API settings from the matching client response; retain the snapshot's exact token IDs. Both metadata-child and HTTP tests cover this path. |
| BF16/C8 indexer width interpreted in the wrong units | Use the shared per-layer wire validator and planar packer, including partial D receive blocks. |
| `torch_npu` or `ascend_transport.so` cannot load | Use the shared shell bootstrap and loader path before starting Python. Runtime/import checks precede the large raw-file scan. |
| Ordinary CPU staging memory rejected by Ascend transport | Use pinned allocations, register both buffers, and retain their owners until native transport cleanup. |
| Seed does not inherit the server launch script's communication settings | Read an allowlist from live D worker environments and apply it before native imports; preserve seed's separate device selection. |
| Same-process seed readback hides a same-device ADXL connection conflict | Reject a seed device within the target D's visible device set before injection; the example uses device 4 for D devices 0–3. |
| Store segment name confused with the transfer endpoint | Select replicas by segment name and validate placement against the separate engine endpoint. |
| Wrong HTTP port discovered after copying the archive | Check the supplied D API before metadata collection and injection; no hard-coded proxy port. |
| Native readback passes but serving lookup uses different keys | Compare injection keys with the actual Ascend hash and LMCache serializer for both token-based retrieval and hash/offset lookup RPCs. Tests include the request-config extraction and a 500095-token mixed-C8 receive with a 383-token tail. |
| Missing cache-usage fields treated as a cache miss | Preserve output and report `replay_unverified`; an actual numeric mismatch still fails. |
| Python 3.9 evaluates unsupported union annotations | Client modules postpone annotation evaluation; grammar and import checks cover the client entry point. The native remote runtime remains the supplied Python. |

The tests also check that an injection failure prevents the D request and that
the seed remains alive until the SSE response completes. They do not substitute
for running the installed Ascend/Mooncake binaries on the serving machines.
