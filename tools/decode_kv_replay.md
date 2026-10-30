# Decode from a saved D receive

First collect with `LMCACHE_DECODE_KV_SNAPSHOT=1` on D. The updated snapshot
writer saves `replay.json` alongside the full raw KV/indexer files, automatically
under the loaded vllm-ascend root's `decode-kv-snapshots/<request UUID>/`.
No snapshot directory or config file needs to be specified when capturing.

After the request completes, stop the old D service to release its devices.
On the D machine holding that request's complete TP capture:

```bash
python3 tools/decode_kv_replay.py --request-id RESPONSE_UUID --output-tokens 128 2>&1 | tee decode-replay.log
```

The script starts one D model instance. P and a Mooncake service are unnecessary.
It uses the saved model path and TP size; devices default to `0,...,TP-1`.
Use `--devices 4,5,6,7` to choose different devices, or `--model /new/path` if the
**same** weights/tokenizer were moved. The captured TP size must stay unchanged.
Both groups' registered layers, including MTP, must be present.

`--local PATH [PATH ...]` accepts roots previously collected by
`decode_kv_compare.py`, including archives from several machines. The collector
now includes `replay.json`. Older captures without that file cannot reconstruct
the complete request; the script rejects them rather than guessing the tail.

Output is automatically created in `decode-kv-replays/<UUID>-<unique suffix>/`:

- `plan.json`: original D prompt token IDs, sampling and replay settings.
- `server.log`: the model startup, cache loading, and output log.
- `output.txt` and `output.json`: generated text, token IDs and cache hit count.

The replay restores the entire **received prefix**. Original D-side tail
recomputation still happens; the full 500k prompt is not supposed to be
recomputed. The script requires the observed cached-token count to match the
archived prefix (capped at prompt length minus one); otherwise it reports
`cache_mismatch`. The saved prompt includes any first P-generated token that
the proxy appended. That token remains input, not newly generated D output.

The only transport substitution is a read-only file implementation of the
Mooncake byte API. It builds normal LMCache keys using the current engine's
token database, maps them to saved logical token ranges, and supplies raw planar
layer data to the ordinary allocator/copy paths. It does not restore stale
physical slot numbers, pointers, streams, or events. Component sizes and layer
order are checked. Latent data shared across TP must be byte-identical in the
capture; disagreement is reported instead of silently choosing one rank.

This is a **single-request D replay**, not recreation of the original concurrent
schedule. It retains the recorded eager choice and MTP setting; `--eager`
explicitly disables graphs. Graph capture shapes/DP collectives can differ
from a busy multi-replica service. A successful replay therefore does not rule
out an original P race or a D race triggered only by batching. Nondeterministic
sampling or kernels can also change the text. The request's temperature/seed
are preserved; this tool does not force determinism.

CPU cache size defaults to 1.3 times one rank's full snapshot bytes plus 4 GiB,
rounded up. `--cpu-cache-gb` overrides it. GPU utilization uses the captured value;
`--gpu-memory-utilization` overrides it. Startup/step and file-transfer timeouts
default to 7200 seconds (`--timeout`). No `/dev/shm` contents are removed.
`--prepare-only` validates the archive and writes a plan without loading weights.

CPU tests cover file/byte layout, all-layer and per-layer reads, partial chunks,
TP validation, metadata and lazy SDK installation. Real model replay requires
the Ascend environment and is not covered by those CPU tests.
