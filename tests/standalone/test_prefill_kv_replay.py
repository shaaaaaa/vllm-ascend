# SPDX-License-Identifier: Apache-2.0
"""CPU validation of exact turn reconstruction and native seed wire bytes."""

import builtins
import ctypes
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
import prefill_kv_replay as R  # noqa: E402
import prefill_kv_seed as S  # noqa: E402


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_run(root):
    prefix, article, answer, added = "unique\r\n", "原文\n", "回答\r\n", "\n用户：问题\n助手：\n"
    history = prefix + article
    session = root / "sessions" / "000"
    session.mkdir(parents=True)
    (root / "initial_prompt.txt").write_bytes(article.encode())
    (session / "session_prefix.txt").write_bytes(prefix.encode())
    rows = []
    for turn in range(2):
        folder = session / "turns" / f"{turn:03}"
        folder.mkdir(parents=True)
        if turn:
            history += answer + added
        (folder / "added_input.txt").write_bytes((added if turn else "").encode())
        (folder / "output.txt").write_bytes(("坏答案" if turn else answer).encode())
        payload = dict(
            model="alias",
            request_id=f"request-{turn}",
            temperature=0,
            max_tokens=128,
            stream=True,
            stream_options={"include_usage": True},
            session_id=0,
            round=turn,
            prompt_chars=len(history),
            prompt_source="pieces",
        )
        save(folder / "request.json", payload)
        rows.append(
            dict(
                session_id=0,
                round=turn,
                response_id=f"response-{turn}",
                request_id=f"request-{turn}",
                prompt_tokens=11 + turn * 3,
                cached_tokens=8 if turn else None,
            )
        )
    save(
        root / "summary.json",
        dict(
            sessions=[dict(session_id=0, directory="sessions/000")],
            warmups=rows[:1],
            rounds=rows[1:],
        ),
    )
    return history


def schema():
    return dict(
        version=3,
        use_mla=True,
        use_layerwise=True,
        dsa_two_groups=True,
        page_first_multi_buffer=True,
        save_chunk_meta=False,
        chunk_size=4,
        dtype="torch.bfloat16",
        group0_raw_token_dim=2,
        group1_raw_token_dim=2,
        runtime_kv_groups=[[2, ["l0", "l1"]], [1, ["i0"]]],
    )


def contract():
    value = schema()
    signature = hashlib.blake2b(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(), digest_size=8
    ).hexdigest()
    return dict(schema=value, signature=signature, algorithm="sha256_cbor", hash_root=repr(b"root"))


def log_file(path, hash_root=b"root", pid=1):
    c = contract()
    path.write_text(
        f"(Worker_DP0_TP1_EP1 pid={pid}) Initialized NONE_HASH={hash_root!r} from vLLM (>= PR#20511)\n"
        f"(Worker_DP0_TP1_EP1 pid={pid}) Using hash algorithm=sha256_cbor; Mooncake payload layout: "
        f"signature={c['signature']} role=worker rank=1 schema={c['schema']!r} (token_database.py)\n"
        "Connected to remote storage at mooncakestore://10.0.0.1:50051/\n",
        encoding="utf-8",
    )


def snapshot(root):
    tokens = list(range(11))
    for rank in range(2):
        folder = root / "response-0" / f"rank{rank}"
        folder.mkdir(parents=True)
        save(folder / "tokens.json", tokens)
        files = []
        for group, count in enumerate((2, 1)):
            for layer in range(count):
                name = f"g{group}-l{layer}-c0.bin"
                data = bytes((i + group * 30 + layer * 20) % 256 for i in range(44))
                (folder / name).write_bytes(data)
                files.append(
                    dict(
                        group=group,
                        layer=layer,
                        component=0,
                        row_bytes=4,
                        dtype="torch.bfloat16",
                        start=0,
                        end=11,
                        bytes=44,
                        path=name,
                    )
                )
        save(
            folder / "manifest.json",
            dict(
                format="decode-kv-raw-v1",
                request_id="response-0",
                model="/model",
                status="complete",
                tp_rank=rank,
                tp_size=2,
                byteorder=sys.byteorder,
                tokens=11,
                groups=[2, 1],
                layer_names=[["l0", "l1"], ["i0"]],
                files=files,
            ),
        )
        save(folder / "replay.json", dict(runtime=dict(engine_options=dict(model="/model"))))
    return dict(
        snapshot_root=str(root),
        previous="response-0",
        cached_tokens=8,
        prefix_digest=S.token_digest(tokens[:8]),
        previous_prompt_tokens=11,
        previous_prompt_digest=S.token_digest(tokens),
        model_path="/model",
        contract=contract(),
        tp_size=2,
        workers=2,
        seed_device=0,
        release_file=str(root / "lease.release"),
        hold_seconds=2,
        local_hostname="127.0.0.1",
        master="127.0.0.1:50051",
        store_bytes=1024,
    )


def test_reconstruct_exact_payload_and_no_new_session_prefix(tmp_path):
    history = make_run(tmp_path)
    result = R.restore_turn(tmp_path, 0, 1)
    assert result["payload"]["prompt"] == history
    assert "\r\n" in history
    assert result["previous_text"] == "unique\r\n原文\n"
    assert result["payload"]["temperature"] == 0
    assert result["payload"]["max_tokens"] == 128
    assert result["payload"]["request_id"].startswith("replay-")
    assert "prompt_chars" not in result["payload"]
    assert result["cached_tokens"] == 8


def test_changed_client_files_fail_before_seeding(tmp_path):
    make_run(tmp_path)
    (tmp_path / "initial_prompt.txt").write_text("changed")
    with pytest.raises(ValueError, match="length differs"):
        R.restore_turn(tmp_path, 0, 1)


def test_parse_live_abi_and_conflicting_hash_roots(tmp_path):
    log_file(tmp_path / "a.log")
    c = S.live_contract([tmp_path / "a.log"])
    assert c["hash_root"] == repr(b"root")
    assert c["master"] == "10.0.0.1:50051"
    assert c["tp_size"] == 2
    log_file(tmp_path / "b.log", hash_root=b"other", pid=2)
    with pytest.raises(ValueError, match="one common"):
        S.live_contract([tmp_path / "a.log", tmp_path / "b.log"])


def test_tail_excluded_and_group_plane_order(tmp_path):
    plan = snapshot(tmp_path)
    tokens, rows = S.validate_snapshot(plan)
    assert tokens == list(range(8))  # Never seed the saved 8..11 tail; it must be recomputed.
    target = bytearray(32)
    assert S.fill_page(rows, 0, 4, 8, memoryview(target)) == 32
    assert target == bytes(range(16, 32)) + bytes(range(36, 52))


def test_legacy_manifest_without_replay_metadata(tmp_path):
    plan = snapshot(tmp_path)
    for path in tmp_path.rglob("replay.json"):
        path.unlink()
    tokens, rows = S.validate_snapshot(plan)
    assert tokens == list(range(8))
    assert len(rows) == 3


@pytest.mark.parametrize("runtime", [None, {"error": "old metadata capture failed"}])
def test_manifest_does_not_require_d_launch_metadata(tmp_path, runtime):
    plan = snapshot(tmp_path)
    for path in tmp_path.rglob("replay.json"):
        save(path, dict(runtime=runtime))
    assert S.validate_snapshot(plan)[0] == list(range(8))


@pytest.mark.parametrize("failure", ["missing_model", "manifest_conflict", "replay_conflict", "layout", "tokens"])
def test_legacy_compatibility_keeps_model_and_payload_checks(tmp_path, failure):
    plan = snapshot(tmp_path)
    for path in tmp_path.rglob("replay.json"):
        path.unlink()
    # Check TP1 as well: its identity must not be discarded when injecting TP0.
    directory = tmp_path / "response-0/rank1"
    path = directory / "manifest.json"
    meta = json.loads(path.read_text())
    if failure == "missing_model":
        meta.pop("model")
    elif failure == "manifest_conflict":
        meta["model"] = "/different-model"
        save(directory / "replay.json", dict(runtime=dict(engine_options=dict(model="/model"))))
    elif failure == "replay_conflict":
        save(directory / "replay.json", dict(runtime=dict(engine_options=dict(model="/different-model"))))
    elif failure == "layout":
        meta["files"][0]["dtype"] = "torch.float16"
    else:
        plan["prefix_digest"] = "wrong"
    save(path, meta)
    with pytest.raises(ValueError):
        S.validate_snapshot(plan)


def test_replay_model_remains_a_supported_identity_source(tmp_path):
    plan = snapshot(tmp_path)
    for path in tmp_path.rglob("manifest.json"):
        meta = json.loads(path.read_text())
        meta.pop("model")
        save(path, meta)
    assert S.validate_snapshot(plan)[0] == list(range(8))


def test_child_failure_message_reaches_caller(tmp_path):
    log = tmp_path / "seed.log"
    log.write_text(
        S.PREFIX + '{"status": "validating"}\n' + R.PREFIX + " FAILED: source model differs\n", encoding="utf-8"
    )
    process = NS(poll=lambda: 1, returncode=1)
    with pytest.raises(RuntimeError, match="Seed failed: source model differs"):
        R.wait_seed(process, log, 1)


@pytest.mark.parametrize("failure", ["tokens", "tp_bytes", "layout", "model", "partial"])
def test_incompatible_snapshots_rejected(tmp_path, failure):
    plan = snapshot(tmp_path)
    if failure == "tokens":
        plan["prefix_digest"] = "wrong"
    elif failure == "tp_bytes":
        path = tmp_path / "response-0/rank1/g1-l0-c0.bin"
        data = bytearray(path.read_bytes())
        data[1] ^= 1
        path.write_bytes(data)
    elif failure == "layout":
        plan["contract"]["schema"]["runtime_kv_groups"][0][1][0] = "wrong"
    elif failure == "model":
        plan["model_path"] = "/wrong"
    else:
        plan["cached_tokens"] = 7
    with pytest.raises(ValueError):
        S.validate_snapshot(plan)


def test_actual_hash_input_and_lmcache_key_serializer(monkeypatch):
    calls = []

    def hash_fn(value):
        calls.append(value)
        return b"hashed" + bytes([len(calls)])

    class Key:
        def __init__(self, model, world, worker, value, dtype, tags, kv_group):
            self.identity = model, world, worker, value, dtype, tags, kv_group

    monkeypatch.setitem(sys.modules, "vllm.utils.hashing", NS(get_hash_fn_by_name=lambda _: hash_fn))
    monkeypatch.setitem(sys.modules, "lmcache.utils", NS(CacheEngineKey=Key))
    monkeypatch.setitem(
        sys.modules, "lmcache.v1.token_database", NS(DSA_INDEX_CACHE_SCHEMA="v2", DSA_INDEX_CACHE_SCHEMA_TAG="idx")
    )
    monkeypatch.setitem(
        sys.modules,
        "lmcache.v1.mooncake_layout",
        NS(MOONCAKE_PAYLOAD_LAYOUT_TAG="layout", mooncake_page_key=lambda k, n: (k.identity, n)),
    )
    c = contract()
    keys = list(S.page_keys(dict(contract=c, model_path="/model"), list(range(8))))
    assert calls == [(b"root", (0, 1, 2, 3), ()), (b"hashed\x01", (4, 5, 6, 7), ())]
    assert [(g, a, b) for _, g, a, b in keys] == [(0, 0, 4), (1, 0, 4), (0, 4, 8), (1, 4, 8)]
    assert keys[1][0][0][1:3] == (1, 0)
    assert keys[1][0][0][5] == {"layout": c["signature"], "idx": "v2"}


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "readback",
        "register_first",
        "register_second",
        "tp_bytes",
        "unregister",
        "wrong_endpoint",
        "transport_init",
    ],
)
def test_native_seed_readback_before_ready(tmp_path, monkeypatch, capsys, failure):
    import torch

    plan = snapshot(tmp_path)
    events = []
    owners = weakref.WeakValueDictionary()
    real_empty = torch.empty

    def empty(size, *, dtype, device, pin_memory=False):
        tensor = real_empty(size, dtype=dtype, device=device)
        # This CPU-only stand-in enforces the native allocator contract: an
        # ordinary torch.empty CPU pointer is not recognized as ACL HOST memory.
        if pin_memory:
            owners[tensor.data_ptr()] = tensor
        return tensor

    monkeypatch.setattr(torch, "empty", empty)
    validate_tp_copies = S.validate_tp_copies

    def scan(*args):
        events.append("scan")
        return validate_tp_copies(*args)

    monkeypatch.setattr(S, "validate_tp_copies", scan)
    if failure == "tp_bytes":
        path = tmp_path / "response-0/rank1/g1-l0-c0.bin"
        data = bytearray(path.read_bytes())
        data[1] ^= 1
        path.write_bytes(data)

    native_engine = object()

    class Transport:
        def initialize(self, host, metadata, protocol, device):
            assert (host, metadata, protocol, device) == ("127.0.0.1", "P2PHANDSHAKE", "ascend", "")
            return -9 if failure == "transport_init" else 0

        def get_rpc_port(self):
            return 16935

        def get_engine(self):
            return native_engine

    class Store:
        def __init__(self):
            self.data = {}
            self.closed = False
            self.registered = []
            self.unregistered = []
            self.attempts = 0

        def setup(self, *args):
            assert args[-1] is native_engine
            return 0

        def batch_is_exist(self, keys):
            return [0] * len(keys)

        def register_buffer(self, ptr, size):
            self.attempts += 1
            events.append("register")
            if ptr not in owners:
                return -1  # Reproduces location:* is not supported.
            assert owners[ptr].numel() == size
            if failure == "register_first" or (failure == "register_second" and self.attempts == 2):
                return -7
            self.registered.append(ptr)
            return 0

        def unregister_buffer(self, ptr):
            events.append("unregister")
            assert ptr in self.registered
            assert ptr in owners
            self.unregistered.append(ptr)
            return -8 if failure == "unregister" else 0

        def get_hostname(self):
            return "127.0.0.1:12962"

        def get_replica_desc(self, key):
            # Same host with a different port is NOT sufficient evidence of
            # ownership. A different process on this node must still fail.
            endpoint = "127.0.0.1:16936" if failure == "wrong_endpoint" else "127.0.0.1:16935"
            descriptor = NS(buffer_descriptor=NS(transport_endpoint=endpoint, size=len(self.data[key])))
            return [NS(is_memory_replica=lambda: True, get_memory_descriptor=lambda: descriptor)]

        def batch_put_from_multi_buffers(self, keys, ptrs, sizes, config):
            events.append("put")
            assert config.preferred_segment == self.get_hostname()
            assert ptrs[0][0] in self.registered
            self.data[keys[0]] = ctypes.string_at(ptrs[0][0], sizes[0][0])
            return [0]

        def batch_get_into_multi_buffers(self, keys, ptrs, sizes):
            data = self.data[keys[0]]
            assert ptrs[0][0] in self.registered
            if failure == "readback":
                data = bytes([data[0] ^ 1]) + data[1:]
            ctypes.memmove(ptrs[0][0], data, len(data))
            if len(self.data) == 4:
                Path(plan["release_file"]).touch()
            return [len(data)]

        def close(self):
            events.append("close")
            assert all(ptr in owners for ptr in self.registered)
            self.closed = True

    store = Store()
    monkeypatch.setitem(sys.modules, "mooncake.engine", NS(TransferEngine=Transport))
    monkeypatch.setitem(sys.modules, "mooncake.store", NS(MooncakeDistributedStore=lambda: store, ReplicateConfig=NS))
    monkeypatch.setitem(sys.modules, "torch_npu", NS(npu=NS(set_device=lambda _: None, init=lambda: None)))
    monkeypatch.setattr(S, "page_keys", lambda p, t: [(f"{g}:{i}", g, i, i + 4) for i in (0, 4) for g in (0, 1)])
    monkeypatch.setattr(S.signal, "signal", lambda *_: None)
    errors = {
        "readback": "readback differs",
        "register_first": "pinned seed CPU staging memory: bytes=32 status=-7",
        "register_second": "pinned seed CPU staging memory: bytes=32 status=-7",
        "tp_bytes": "Saved TP copies differ",
        "unregister": "unregister seed CPU staging memory: status=-8",
        "wrong_endpoint": "Unexpected Mooncake placement",
        "transport_init": "TransferEngine initialization failed: -9",
    }
    if failure:
        with pytest.raises((RuntimeError, ValueError), match=errors[failure]):
            S.seed(plan)
    else:
        S.seed(plan)
    output = capsys.readouterr().out
    assert ('"status": "ready"' in output) is (failure in (None, "unregister"))
    assert store.closed
    assert events[-1] == "close"
    if failure != "unregister":
        assert store.unregistered == store.registered
    if failure in ("register_first", "register_second", "transport_init"):
        assert "scan" not in events
        assert '"status": "staging_ready"' not in output
    else:
        assert events[:3] == ["register", "register", "scan"]
        assert '"status": "staging_ready"' in output
    if failure in ("register_first", "register_second", "tp_bytes", "transport_init"):
        assert not store.data
    else:
        assert len(store.data["0:0"]) == 32


def test_prepare_uses_live_tokenizer_and_does_not_run_inference(tmp_path, monkeypatch):
    make_run(tmp_path / "run")
    log_file(tmp_path / "p.log")
    calls = []

    def http(url, path, body=None):
        calls.append(path)
        if path == "/v1/models":
            return {"data": [{"id": "alias", "root": "/model"}]}
        return {"tokens": list(range(14 if "问题" in body["prompt"] else 11))}

    monkeypatch.setattr(R, "http_json", http)
    args = R.parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--base-url",
            "http://proxy:8000",
            "--p-url",
            "http://P:7910",
            "--p-logs",
            str(tmp_path / "p.log"),
        ]
    )
    (tmp_path / "out").mkdir()
    restored, plan = R.prepare(args, tmp_path / "out")
    assert calls == ["/v1/models", "/tokenize", "/tokenize"]
    assert plan["cached_tokens"] == 8
    tail = json.loads((tmp_path / "out/input_tail_tokens.json").read_text())
    assert tail["token_ids"] == list(range(8, 14))
    assert "kv_transfer_params" not in restored["payload"]


def test_release_cannot_touch_serving_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "REPO", tmp_path)
    with pytest.raises(ValueError):
        R.release_child(str(tmp_path / "tp4_dp4_logs" / "P.log"))
    path = tmp_path / "prefill-kv-replays/leases" / ("a" * 32 + ".release")
    R.release_child(str(path))
    assert path.exists()


def indexer_snapshot(tmp_path, policy):
    plan = snapshot(tmp_path)
    value = plan["contract"]["schema"]
    # Production resolve_mooncake_dsa_raw_token_dims returns max packet BYTES
    # for C8, including mixed precision. It does not return head_dim here.
    value["group1_raw_token_dim"] = 130 if all(policy) else 256
    names = [f"i{i}" for i in range(len(policy))]
    value["runtime_kv_groups"][1] = [len(policy), names]
    value["indexer_execution"] = dict(
        layout="dsa-index-c8-planar-v1" if all(policy) else "dsa-index-mixed-planar-v1",
        quantization="hadamard128-dynamic-int8-fp16-scale-v1",
        head_dim=128,
        key_dtype="int8" if all(policy) else "per-layer-int8-or-model-dtype",
        scale_dtype="float16",
        storage_dtype="uint8",
    )
    if not all(policy):
        value["indexer_execution"]["c8_layers"] = list(policy)
    components = []
    for layer, c8 in enumerate(policy):
        if c8:
            components.extend([(layer, 0, 128, "torch.int8", 21), (layer, 1, 2, "torch.float16", 77)])
        else:
            components.append((layer, 0, 256, "torch.bfloat16", 109))
    for rank in range(2):
        folder = tmp_path / f"response-0/rank{rank}"
        (folder / "replay.json").unlink()  # The reported failure used legacy snapshots.
        meta = json.loads((folder / "manifest.json").read_text())
        meta["groups"][1] = len(policy)
        meta["layer_names"][1] = names
        meta["files"] = [r for r in meta["files"] if r["group"] == 0]
        for layer, component, width, dtype, byte in components:
            name = f"g1-l{layer}-c{component}.bin"
            (folder / name).write_bytes(bytes([byte]) * 11 * width)
            meta["files"].append(
                dict(
                    group=1,
                    layer=layer,
                    component=component,
                    row_bytes=width,
                    dtype=dtype,
                    start=0,
                    end=11,
                    bytes=11 * width,
                    path=name,
                )
            )
        save(folder / "manifest.json", meta)
    return plan, components


@pytest.mark.parametrize(
    "policy",
    [(False, True), (True, False), (True, True), (False, False), (False,) * 3 + (True,) * 16 + (False,) * 3],
)
def test_mixed_indexer_wire_order(tmp_path, policy):
    plan, components = indexer_snapshot(tmp_path, policy)
    _, rows = S.validate_snapshot(plan)
    target = bytearray(4 * sum(width for _, _, width, _, _ in components))
    assert S.fill_page(rows, 1, 4, 8, memoryview(target)) == len(target)
    assert target == b"".join(bytes([byte]) * (4 * width) for _, _, width, _, byte in components)


@pytest.mark.parametrize(
    "layer,component,dtype,width",
    [
        (0, 0, "torch.bfloat16", 512),
        (1, 0, "torch.int8", 256),
        (1, 1, "torch.float32", 4),
    ],
)
def test_mixed_indexer_still_rejects_wrong_planes(tmp_path, layer, component, dtype, width):
    plan, _ = indexer_snapshot(tmp_path, (False, True))
    folder = tmp_path / "response-0/rank0"
    path = folder / "manifest.json"
    meta = json.loads(path.read_text())
    for row in meta["files"]:
        if (row["group"], row["layer"], row["component"]) == (1, layer, component):
            row.update(dtype=dtype, row_bytes=width, bytes=11 * width)
            (folder / row["path"]).write_bytes(bytes(11 * width))
    save(path, meta)
    with pytest.raises(ValueError, match="actual=.*expected="):
        S.validate_snapshot(plan)


def test_real_case_frontier_recomputes_372_not_90(tmp_path, monkeypatch):
    make_run(tmp_path / "run")
    summary_path = tmp_path / "run/summary.json"
    summary = json.loads(summary_path.read_text())
    summary["warmups"][0]["prompt_tokens"] = 499994
    summary["rounds"][0].update(prompt_tokens=500084, cached_tokens=499712)
    save(summary_path, summary)
    log_file(tmp_path / "p.log")

    def http(url, path, body=None):
        if path == "/v1/models":
            return {"data": [{"id": "alias", "root": "/model"}]}
        return {"tokens": list(range(500084 if "问题" in body["prompt"] else 499994))}

    monkeypatch.setattr(R, "http_json", http)
    args = R.parser().parse_args(["--run-dir", str(tmp_path / "run"), "--p-logs", str(tmp_path / "p.log")])
    out = tmp_path / "out"
    out.mkdir()
    _, plan = R.prepare(args, out)
    tail = json.loads((out / "input_tail_tokens.json").read_text())
    assert plan["cached_tokens"] == 499712
    assert tail["token_ids"] == list(range(499712, 500084))
    assert len(tail["token_ids"]) == 372


@pytest.mark.parametrize("cached", [8, 0, None])
def test_existing_pd_orchestration_and_cache_miss_verdict(tmp_path, monkeypatch, cached):
    make_run(tmp_path / "run")
    log_file(tmp_path / "p.log")
    monkeypatch.setattr(R, "REPO", tmp_path)
    active = []
    calls = []

    def http(url, path, body=None):
        calls.append((url, path))
        if path == "/v1/models":
            return {"data": [{"id": "alias", "root": "/model"}]}
        return {"tokens": list(range(14 if "问题" in body["prompt"] else 11))}

    @contextmanager
    def seeded(args, plan, output):
        active.append("seed")
        yield NS(poll=lambda: None)
        active.clear()

    class Client:
        def __init__(self, url, timeout):
            assert url == "http://proxy:8000"

        def complete(self, payload, output):
            assert active == ["seed"]
            assert payload["prompt"].startswith("unique\r\n原文")
            return "回放回答", dict(
                cached_tokens=cached,
                prompt_tokens=14,
                uncached_prompt_tokens=6,
                ttft_s=0.1,
                response_id="replay-response",
            )

    monkeypatch.setattr(R, "http_json", http)
    monkeypatch.setattr(R, "seeded", seeded)
    monkeypatch.setattr(R, "Client", Client)
    argv = [
        "--run-dir",
        str(tmp_path / "run"),
        "--p-logs",
        str(tmp_path / "p.log"),
        "--base-url",
        "http://proxy:8000",
        "--p-url",
        "http://P:7910",
    ]
    if cached == 8:
        assert R.main(argv) == 0
        assert not active
    else:
        with pytest.raises(RuntimeError, match="Cache mismatch"):
            R.main(argv)
    (path,) = (tmp_path / "prefill-kv-replays").glob("*/report.json")
    report = json.loads(path.read_text())
    assert report["status"] == ("replay_complete" if cached == 8 else "failed")
    assert report["accuracy_verdict"] == "not_assessed"
    assert calls == [("http://P:7910", p) for p in ("/v1/models", "/tokenize", "/tokenize")]


def test_remote_child_argv_preserves_linux_path_and_quoting(tmp_path):
    args = R.parser().parse_args(
        [
            "--snapshot-host",
            "root@host",
            "--workspace",
            "/work space/repo",
            "--container",
            "serving",
            "--remote-python",
            "/usr/bin/python3",
        ]
    )
    command = R.child_argv(args, "--release-child", "/work space/repo/leases/test.release")
    assert command[:2] == ["ssh", "-T"]
    assert shlex.split(command[-1]) == [
        "docker",
        "exec",
        "--",
        "serving",
        "bash",
        "-lc",
        R.REMOTE_ENV_BOOTSTRAP,
        "prefill-kv-replay",
        "/usr/bin/python3",
        "/work space/repo/tools/prefill_kv_replay.py",
        "--release-child",
        "/work space/repo/leases/test.release",
    ]


@pytest.mark.parametrize("inherited_libraries", ["", "/opt/test-cann/lib64:/opt/test-mooncake/lib"])
def test_remote_bootstrap_loads_environment_and_preserves_arguments(inherited_libraries):
    shell = shutil.which("bash")
    if os.name == "nt":
        git = shutil.which("git")
        candidate = Path(git).resolve().parent.parent / "bin/bash.exe" if git else None
        if candidate and candidate.is_file():
            shell = str(candidate)
    if shell is None:
        pytest.skip("Bash unavailable")
    # Stand in for the sourced file without changing HOME or reading real user
    # settings. Test the actual bootstrap in a child Bash, not a shell mock.
    bootstrap = (
        "source() { export PREFILL_REPLAY_TEST_ENV=loaded; export LD_LIBRARY_PATH="
        + shlex.quote(inherited_libraries)
        + "; }\n"
        + R.REMOTE_ENV_BOOTSTRAP
    )
    argument = "space ' quote ; $(printf wrong) `printf wrong`"
    code = (
        "import os,sys,json; print(json.dumps([os.getenv('PREFILL_REPLAY_TEST_ENV'),"
        "sys.argv[1:],os.getenv('LD_LIBRARY_PATH')]))"
    )
    result = subprocess.run(
        [
            shell,
            "--noprofile",
            "--norc",
            "-c",
            bootstrap,
            "prefill-kv-replay",
            sys.executable.replace("\\", "/"),
            "-c",
            code,
            argument,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    expected_libraries = "/usr/local/lib" + (":" + inherited_libraries if inherited_libraries else "")
    if os.name == "nt":
        # Git Bash translates this loader path for a native Windows Python.
        # Linux children receive the original colon-separated POSIX paths.
        expected_libraries = ";".join(
            str(Path(shell).resolve().parent.parent / p.lstrip("/")) for p in expected_libraries.split(":")
        )
    assert json.loads(result.stdout) == ["loaded", [argument], expected_libraries]


@pytest.mark.parametrize("inherited", [None, "", "/opt/test-cann/lib64"])
def test_local_child_library_environment_preserves_existing_paths(monkeypatch, inherited):
    if inherited is None:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH", inherited)
    env = R.child_environment(NS(snapshot_host=None, _passwords={}))
    assert env["LD_LIBRARY_PATH"] == "/usr/local/lib" + (":" + inherited if inherited else "")
    assert os.environ.get("LD_LIBRARY_PATH") == inherited


@pytest.mark.parametrize("failure", ["backend_import", "npu_init", "cache_import"])
def test_runtime_failure_precedes_snapshot_scan(monkeypatch, capsys, failure):
    real_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if failure == "backend_import" and name == "torch":
            raise RuntimeError("Failed to load backend torch_npu") from ImportError("test missing native library")
        return real_import(name, *args, **kwargs)

    def init():
        if failure == "npu_init":
            raise RuntimeError("test NPU init failure")

    def keys(plan, tokens):
        assert not tokens
        raise ImportError("test cache dependency unavailable")

    monkeypatch.setitem(sys.modules, "torch_npu", NS(npu=NS(set_device=lambda _: None, init=init)))
    monkeypatch.setitem(sys.modules, "mooncake.engine", NS(TransferEngine=NS))
    monkeypatch.setitem(sys.modules, "mooncake.store", NS(MooncakeDistributedStore=NS, ReplicateConfig=NS))
    monkeypatch.setattr(S, "page_keys", keys)
    monkeypatch.setattr(S, "snapshot_metadata", lambda _: pytest.fail("Read snapshots before runtime preflight"))
    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises((RuntimeError, ImportError)):
        S.seed(dict(seed_device=0))
    output = capsys.readouterr().out
    assert '"status": "runtime_check"' in output
    assert '"status": "validating"' not in output


def test_failure_message_preserves_extension_loader_cause():
    try:
        raise RuntimeError("Failed to load backend torch_npu") from ImportError(
            "test_native.so: cannot open shared object"
        )
    except RuntimeError as error:
        message = R.format_failure(error)
    assert "RuntimeError: Failed to load backend torch_npu" in message
    assert "ImportError: test_native.so: cannot open shared object" in message
