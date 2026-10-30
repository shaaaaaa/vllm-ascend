# Replay a saved conversation turn through an existing PD deployment

`prefill_kv_replay.py` uses an **already running P/D deployment**. Start the
deployment normally, but do not send inference requests before this test. The
tool starts a temporary Mooncake client, not a model, API server, or master.
It does not modify serving code or enable `p_use`, tensor dumps, or eager mode.

The input is the original `multi-turn-concurrent` directory **and** the raw
round-0 `decode-kv-snapshots` files. Client logs alone do not contain KV.
Legacy snapshots without `replay.json` are supported: their `manifest.json`
already records the source model, layer inventory, and raw file layouts.
The standalone D launch/sampling metadata in `replay.json` is optional here;
when present, its model identity must also agree with the current P server.

## What runs

1. Reconstruct the exact original prompt from the session prefix, initial
   article, preceding answers, and saved `added_input.txt`. Preserve whitespace,
   the model alias, temperature, and output limit. Only the request ID changes.
2. Read the **new deployment's P startup logs** for the actual cache layout,
   hash algorithm, hash root, and Mooncake endpoint. Query its model identity and
   tokenizer. Reject incompatible token counts, layouts, or saved model paths.
3. On the snapshot host, validate all captured TP copies. The current MLA
   deployment shares the TP0 key namespace, so differing copies are rejected
   rather than silently discarding another rank's data.
4. Pack the original raw bytes into the deployed all-layer Mooncake page format,
   including mixed BF16/C8 indexer planes. Inject only full cache blocks up to
   the original `cached_tokens` frontier. Use native SDK put/get and compare
   every uploaded page against its readback before submitting the request.
5. Send the reconstructed full prompt through the **existing PD proxy**. P
   retrieves history using normal LMCache/Mooncake lookup, recomputes the tail,
   and transfers the new result to D using the deployed path.
6. Print the response and require the reported prompt and cached-token counts
   to match the original request. A missing cache count or cache miss is a
   failed replay, even if the server returned an answer.

For mixed indexers, the descriptor's `group1_raw_token_dim` is already a byte
capacity. Per-layer validation uses the precision policy and `head_dim`: a
128-element BF16 key is 256 bytes; C8 uses 128 key bytes plus a 2-byte FP16 scale.
An incompatible snapshot reports both the actual and expected plane layouts.

For session 0 in the reported case, round 0 has 499994 prompt tokens and round 1
has 500084. The script seeds `[0, 499712)` and P recomputes
`[499712, 500084)`: **372 tokens**, comprising the 282-token old tail and 90 new
tokens. Injecting all 499994 tokens would change the reproduction.

## Command

Run from the updated `vllm-ascend` repository on the client/P machine. The same
revision of the tools must also be available on the snapshot machine. Use its
Python environment containing the deployed vLLM, LMCache, torch_npu, and
Mooncake SDK.

Example when that environment and the snapshots are accessible directly over
SSH:

```bash
python3 tools/prefill_kv_replay.py \
  --run-dir ./multi-turn-concurrent \
  --session 0 --round 1 \
  --base-url http://7.150.2.72:8000 \
  --p-url http://7.150.2.72:7910 \
  --p-logs tp4_dp4_logs/P_dp*.log \
  --snapshot-host root@7.150.3.60 \
  --workspace /workspace/fsi_lab/sqh/vllm-ascend \
  --password 'YOUR_SSH_PASSWORD' \
  --skip-host-key-check \
  2>&1 | tee prefill-replay.log
```

`--base-url` is the PD proxy; `--p-url` is a direct P API endpoint for `/tokenize`
and `/v1/models`. Tokenizer requests do not run the model. P startup logs default
to this checkout's `tp4_dp4_logs/P_dp*.log`. They must belong to the current
deployment; do not mix old logs with different hash roots or layouts.

If the snapshot machine runs the Python environment **inside Docker**, add
`--container CONTAINER_NAME`, use the **container's** repository path for
`--workspace`, and specify its Python if necessary. For the previously used
container these arguments are:

```bash
--container vllm-ascend-v0.18.0rc1-lmcache-ascend-v0.4.3-fsi-sqh \
--workspace /workspace/sqh/vllm-ascend \
--remote-python /usr/local/python3.11.14/bin/python3
```

Use the container that is currently running; the tool does not start or stop
containers. The snapshot directory defaults to `WORKSPACE/decode-kv-snapshots`.
`--snapshot-root PATH` selects a different existing capture directory. Omit
`--snapshot-host` for local snapshots; in that mode the current Python environment
runs the temporary native client.

The master is inferred from P's existing connector startup log. If the endpoint
was not logged, pass `--master HOST:PORT` for the **same master the live P/D uses**.
No separate configuration file is needed. Passwords use native OpenSSH askpass;
`sshpass` is not required. `--password-env NAME` is also supported.

## Results and resource use

The tool automatically creates `vllm-ascend/prefill-kv-replays/s0-r1-.../`:

- `request.json`: full reconstructed request submitted to the proxy.
- `input_tail_tokens.json`: exact recomputed input token IDs and positions.
- `original.json`: original IDs, measurements, and answer.
- `seed_plan.json`, `seed.log`, `seed_ready.json`: injection identity and progress.
- `output.txt`, `events.jsonl`, `report.json`: replay answer, streaming response,
  actual cache counts, and completion status.

The source KV stays on its original machine. SSH carries only the small plan
and progress log. The native injector does read the raw files and creates a
Mooncake copy of the historical prefix: approximately one full KV copy plus
allocator headroom in CPU memory, not another model. Default segment capacity
is 120% of the raw prefix plus 2 GiB; override with `--store-gb` if needed.
`--workers` defaults to 8 for offline TP-copy validation. Progress is printed
during validation and page injection. The injector initializes an Ascend
transport context on `--seed-device` (default 0), without loading weights.

The injector keeps its replicas alive until the request finishes and then
closes only its own segment. It rejects existing seed keys rather than
overwriting another run. It neither clears caches nor deletes existing P/D
replicas. A subsequent replay should again use a fresh deployment/cache state.

`--prepare-only` checks local request reconstruction and live tokenization and
writes the plan, without injecting KV or generating an answer. Seed and request
timeouts default to 7200 seconds each. `--output-tokens` can override the saved
limit; omit it to preserve the original request.

`replay_complete` means the intended prompt and cache boundary were exercised,
not that output correctness was proven. A single-request replay preserves the
deployed serving path but does not recreate the original multi-request batch,
arrival times, or bank timing. Normal output here does not rule out a
concurrency-dependent defect. P's original compute configuration and debug
switches remain whatever the already running deployment uses.
