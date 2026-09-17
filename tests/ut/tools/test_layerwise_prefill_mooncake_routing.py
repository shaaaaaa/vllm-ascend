# SPDX-License-Identifier: Apache-2.0
"""Real CPU byte buffers, fake native boundary: routing/copy/placement regressions."""

import ctypes
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def routing(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools"))
    return importlib.import_module("layerwise_prefill_mooncake_routing")


class MemoryStore:
    def __init__(self):
        self.data = {}
        self.calls = []
        self.registered = set()
        self.endpoint = "segment0"
        self.bad_placement = False
        self.short_read = False
        self.corrupt_copy = False
        self.put_failure = False

    def setup(self, *args, **kwargs):
        self.calls.append(("setup", args))
        return 0

    def get_hostname(self):
        return self.endpoint

    def close(self):
        pass

    def remove(self, key):
        self.data.pop(key, None)
        return 0

    def register_buffer(self, pointer, size):
        self.registered.add(pointer)
        return 0

    def unregister_buffer(self, pointer):
        self.registered.remove(pointer)
        return 0

    def get_replica_desc(self, key):
        data, endpoint = self.data[key]
        return [
            NS(
                is_memory_replica=lambda: True,
                get_memory_descriptor=lambda: NS(buffer_descriptor=NS(transport_endpoint=endpoint, size=len(data))),
            )
        ]

    def batch_put_from_multi_buffers(self, keys, pointers, sizes, config):
        self.calls.append(("put", list(keys), config.preferred_segment))
        if self.put_failure:
            return [-1] * len(keys)
        for key, ptrs, counts in zip(keys, pointers, sizes, strict=True):
            payload = b"".join(ctypes.string_at(ptr, count) for ptr, count in zip(ptrs, counts, strict=True))
            if self.corrupt_copy:
                payload = bytes([payload[0] ^ 1]) + payload[1:]
            self.data[key] = payload, "wrong-holder" if self.bad_placement else config.preferred_segment
        return [0] * len(keys)

    def batch_put_from(self, keys, pointers, sizes, config):
        return self.batch_put_from_multi_buffers(keys, [[p] for p in pointers], [[s] for s in sizes], config)

    def batch_get_into_multi_buffers(self, keys, pointers, sizes):
        self.calls.append(("get", list(keys)))
        result = []
        for key, ptrs, counts in zip(keys, pointers, sizes, strict=True):
            data, _ = self.data[key]
            offset = 0
            for ptr, count in zip(ptrs, counts, strict=True):
                ctypes.memmove(ptr, data[offset : offset + count], count)
                offset += count
            result.append(offset - 1 if self.short_read else offset)
        return result

    def batch_get_into(self, keys, pointers, sizes):
        return self.batch_get_into_multi_buffers(keys, [[p] for p in pointers], [[s] for s in sizes])

    def batch_is_exist(self, keys):
        self.calls.append(("exists", list(keys)))
        return [int(key in self.data) for key in keys]

    def batch_get_buffer(self, keys):
        self.calls.append(("get_buffer", list(keys)))
        return [self.data[key][0] for key in keys]

    def is_exist(self, key):
        return int(key in self.data)


@pytest.fixture
def store():
    return MemoryStore()


def make_route(routing, store, root, stage, device):
    (root / "prefill").mkdir(exist_ok=True)
    (root / stage).mkdir(exist_ok=True)
    for holder in (0, 1):
        (root / f"holder{holder}").mkdir(exist_ok=True)
        (root / f"holder{holder}/ready.json").write_text(json.dumps({"segment": f"segment{holder}"}))
    route = {"root": str(root), "namespace": "run", "stage": stage}
    result = routing.RoutedStore(store, NS, route, lambda: device)
    assert result.setup("host", "P2PHANDSHAKE", 0, 0, "ascend", "", "master") == 0
    return result


@pytest.mark.parametrize("device", range(8))
def test_every_decode_rank_reads_only_other_device_copy(routing, store, tmp_path, device):
    routed = make_route(routing, store, tmp_path, "decode", device)
    holder = 1 if device == 0 else 0
    assert holder != device
    key = f"run/holder{holder}/key"
    store.data[key] = b"abcd", f"segment{holder}"
    buffer = torch.zeros(4, dtype=torch.uint8)
    assert routed.batch_is_exist(["key"]) == [1]
    assert routed.is_exist("key") == 1
    assert routed.batch_get_buffer(["key"]) == [b"abcd"]
    assert routed.batch_get_into(["key"], [buffer.data_ptr()], [4]) == [4]
    assert bytes(buffer.tolist()) == b"abcd"
    assert routed.batch_get_into_multi_buffers(["key"], [[buffer.data_ptr()]], [[4]]) == [4]
    assert all(call[1] == [key] for call in store.calls if call[0] != "setup")


def test_prefill_native_byte_buffer_write_is_unchanged_and_manifested(routing, store, tmp_path):
    routed = make_route(routing, store, tmp_path, "prefill", 0)
    buffer = torch.tensor([7, 0, 255, 6], dtype=torch.uint8)
    # Caller cannot override the holder route through preferred_segment.
    assert routed.batch_put_from(["tail"], [buffer.data_ptr()], [4], NS(preferred_segment="segment0")) == [0]
    assert routed.batch_put_from_multi_buffers(["page"], [[buffer.data_ptr(), buffer.data_ptr() + 2]], [[2, 2]]) == [0]
    assert store.data == {f"run/holder1/{key}": (bytes(buffer.tolist()), "segment1") for key in ("tail", "page")}
    assert routing.read_manifest(tmp_path / "prefill/objects.jsonl") == {"tail": 4, "page": 4}


@pytest.mark.parametrize("failure", ["bad_placement", "put_failure"])
def test_failed_or_misplaced_put_is_not_added_to_copy_manifest(routing, store, tmp_path, failure):
    routed = make_route(routing, store, tmp_path, "prefill", 0)
    setattr(store, failure, True)
    buffer = torch.zeros(4, dtype=torch.uint8)
    with pytest.raises(RuntimeError):
        routed.batch_put_from(["bad"], [buffer.data_ptr()], [4])
    assert not (tmp_path / "prefill/objects.jsonl").exists()


@pytest.mark.parametrize("device", range(8))
def test_all_prefill_ranks_initialize_and_query_metadata(routing, store, tmp_path, device):
    routed = make_route(routing, store, tmp_path, "prefill", device)
    assert routed.batch_is_exist(["missing"]) == [0]
    assert routed.is_exist("missing") == 0
    if device:
        with pytest.raises(RuntimeError, match="Only P rank0"):
            routed.batch_put_from(["key"], [123], [4])


def test_same_device_prefill_data_read_is_rejected_before_native_transfer(routing, store, tmp_path):
    routed = make_route(routing, store, tmp_path, "prefill", 1)
    with pytest.raises(RuntimeError, match="same-device"):
        routed.batch_get_buffer(["key"])
    assert [call[0] for call in store.calls] == ["setup"]


def test_close_records_actual_decode_route(routing, store, tmp_path):
    routed = make_route(routing, store, tmp_path, "decode", 3)
    store.data["run/holder0/key"] = b"abcd", "segment0"
    routed.batch_get_buffer(["key"])
    routed.close()
    report = json.loads(next((tmp_path / "decode").glob("route-*.json")).read_text())
    assert report == {"device": 3, "holder": 0, "segment": "segment0", "reads": 1, "written_objects": 0}


def test_decode_cannot_modify_either_copy(routing, store, tmp_path):
    routed = make_route(routing, store, tmp_path, "decode", 0)
    with pytest.raises(RuntimeError, match="Only P rank0"):
        routed.batch_put_from(["key"], [123], [4])
    with pytest.raises(AttributeError, match="does not implement"):
        routed.unreviewed_key_api


@pytest.mark.parametrize("bad", [None, "short_read", "corrupt_copy", "bad_placement", "put_failure"])
def test_holder_copy_verifies_real_bytes_and_cleans_buffers(routing, store, bad):
    store.data["run/holder1/index"] = b"\x00\xff\x01\x03", "segment1"
    store.data["run/holder1/latent-tail"] = b"\x45\x67", "segment1"
    if bad:
        setattr(store, bad, True)
    entries = {"index": 4, "latent-tail": 2}
    allocate = lambda size: torch.empty(size, dtype=torch.uint8)
    if bad:
        with pytest.raises(RuntimeError):
            routing.copy_objects(store, NS, entries, "run", "segment1", "segment0", allocate)
    else:
        result = routing.copy_objects(store, NS, entries, "run", "segment1", "segment0", allocate)
        assert result == {"objects": 2, "bytes": 6, "manifest_sha256": routing.manifest_digest(entries)}
        for key in entries:
            assert store.data[f"run/holder0/{key}"] == (store.data[f"run/holder1/{key}"][0], "segment0")
    assert not store.registered


@pytest.mark.parametrize("bad", [None, "short_read", "corrupt_copy", "bad_placement"])
def test_holder_preflight_checks_native_contract_before_model(routing, store, bad):
    tool = importlib.import_module("layerwise_prefill_mooncake_check")
    if bad:
        setattr(store, bad, True)
        with pytest.raises(RuntimeError):
            tool.holder_preflight(store, NS, "run", torch)
    else:
        tool.holder_preflight(store, NS, "run", torch)
    assert not store.data
    assert not store.registered


@pytest.mark.parametrize("entries", [[], [{"key": "x", "size": 0}], [{"key": "x", "size": 1}, {"key": "x", "size": 2}]])
def test_manifest_rejects_empty_invalid_or_inconsistent_objects(routing, tmp_path, entries):
    path = tmp_path / "objects.jsonl"
    path.write_text("\n".join(map(json.dumps, entries)), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        routing.read_manifest(path)


def test_placement_accepts_native_cpp_spelling(routing):
    descriptor = NS(transport_endpoint_="segment", size_=4)
    replica = NS(is_memory_replica=lambda: True, get_memory_descriptor=lambda: NS(buffer_descriptor=descriptor))
    routing.require_placement(NS(get_replica_desc=lambda key: [replica]), "key", "segment", 4)


def test_no_routing_installed_without_explicit_test_config(routing, monkeypatch):
    monkeypatch.setenv("LMCACHE_EXTRA_CONFIG", "{}")
    monkeypatch.setitem(sys.modules, "mooncake.store", None)
    routing.install_routing()  # Must not even import the unavailable native module.


def test_worker_extension_installs_before_connector_creates_native_store(routing, monkeypatch, tmp_path):
    native = MemoryStore
    fake = NS(MooncakeDistributedStore=native, ReplicateConfig=NS)
    monkeypatch.setitem(sys.modules, "mooncake", NS(store=fake))
    monkeypatch.setitem(sys.modules, "mooncake.store", fake)
    monkeypatch.setattr(torch, "npu", NS(current_device=lambda: 0), raising=False)
    monkeypatch.setenv(
        "LMCACHE_EXTRA_CONFIG",
        json.dumps({"prefill_check_routing": {"root": str(tmp_path), "namespace": "run", "stage": "decode"}}),
    )
    # vLLM imports the extension before it constructs Worker/LMCache. Use the
    # actual module import, not a mocked post-start RPC.
    monkeypatch.delitem(sys.modules, "layerwise_prefill_mooncake_worker", raising=False)
    importlib.import_module("layerwise_prefill_mooncake_worker")
    assert isinstance(fake.MooncakeDistributedStore(), routing.RoutedStore)
    installed = fake.MooncakeDistributedStore
    routing.install_routing()
    assert fake.MooncakeDistributedStore is installed
    monkeypatch.delitem(sys.modules, "layerwise_prefill_mooncake_worker")
