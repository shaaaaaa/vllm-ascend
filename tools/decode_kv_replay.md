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
  --password-env PD_SSH_PASSWORD \
  --skip-host-key-check \
  2>&1 | tee decode-replay.log
```

`--password 'PASSWORD'` is also supported through native OpenSSH askpass; no
sshpass or interactive prompt is used. Update the tools on both the client
and snapshot host. `--workspace` is the path **inside** the container when
`--container` is present. By default, the script reads that workspace's
`decode-kv-snapshots` and current `tp4_dp4_logs/D_dp*.log`. `--snapshot-root`
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
`tokens.json`, and `replay.json`. The latter preserves the **actual D input
token IDs**, sampling parameters, and any `bootstrap_final_hidden` payload
for MTP. Tokens appended by the original proxy remain in that input; the
script neither tokenizes text nor synthesizes a new first token.

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
