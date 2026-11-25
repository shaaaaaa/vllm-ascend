# Batch prefill audit

Pass the complete client tee log to check **every** recorded request, including
normal answers as controls. Collection happens once per host; the offline index
is built once and reused across requests. This does not send inference requests
or change any server setting.

Reuse an existing collection, even after remote logs have been removed:

```bash
python3 tools/prefill_audit.py \
  --client-log multi-turn-concurrent.log \
  --local audit-session4 \
  --output audit-all
```

Or collect the current files from all four hosts once:

```bash
python3 tools/prefill_audit.py \
  --client-log multi-turn-concurrent.log \
  --hosts root@7.150.2.72 root@7.150.3.11 root@7.150.3.60 root@7.150.6.177 \
  --workspace /workspace/fsi_lab/sqh/vllm-ascend \
  --password 'YOUR_SSH_PASSWORD' --skip-host-key-check \
  --output audit-all
```

Use `--audit-dir` for a different remote audit directory, or `--container` if the
directory is only visible inside the container. Multiple explicit IDs are also
supported: `--request-id ID1 ID2 ID3`. They are added to all requests discovered
in the client log, not used to filter them.

Outputs:

- `report.json`: all request results and coverage gaps.
- `requests.csv`: one row per request, with session, round, symptoms and audit result.
- `requests/*.json`: detailed transfer comparison for each request.
- `client_outputs.jsonl`: original answers and reproducible text symptoms.

`repetitive_latin_output`, `repeated_question`, `repeated_answer_prefix` and
`copies_previous_output` flag answers for inspection. They do not prove a KV
error. `generated_user_turn` separately identifies invented dialogue: a raw
completions request without stop strings can continue the conversation itself.
Missing response IDs and truncated output remain in the report rather than
being silently discarded. Later rounds retain a link to a suspect previous
answer because that answer becomes part of the next prompt.

`samples_consistent` only means the observed byte samples match. Missing device
readbacks remain `incomplete`. New offline tools cannot recover samples that an
older server version did not record. The tools do not change generation options
or add any NPU work to the server.
