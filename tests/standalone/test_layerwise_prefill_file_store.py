# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the explicit file SDK, independent of NPU/vLLM imports."""

import gc
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def sdk():
    path = Path(__file__).resolve().parents[2] / "tools" / "layerwise_prefill_file_store.py"
    spec = importlib.util.spec_from_file_location("file_sdk_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def key(group, suffix="abc"):
    return f"__lmcache_page_v1__@3@glm@8@0@{suffix}@bfloat16@{group}"


def seed(sdk, root):
    writer = sdk.FileStore(root, "prefill")
    tensors = [torch.arange(10, dtype=torch.uint8), torch.arange(7, dtype=torch.uint8) + 20]
    writer.memory.bind(tensors)
    assert writer.batch_put_from([key(0), key(1)], [x.data_ptr() for x in tensors], [10, 7]) == [0, 0]
    assert sdk.seal_store(root)["passed"]
    return writer, tensors


@pytest.mark.parametrize("tokens", [5, 6, 128])
def test_actual_multi_buffer_roundtrip_and_short_destination_tail(sdk, tmp_path, tokens):
    writer = sdk.FileStore(tmp_path, "prefill")
    # Different physical plane sizes, including a final prompt chunk of 5/6 tokens.
    planes = [torch.arange(tokens * width, dtype=torch.int32) for width in (8, 3, 2)]
    writer.memory.bind(planes)
    pointers = [tensor.data_ptr() for tensor in planes]
    sizes = [tensor.numel() * tensor.element_size() for tensor in planes]
    assert writer.batch_put_from_multi_buffers([key(0)], [pointers[:2]], [sizes[:2]]) == [0]
    assert writer.put_from(key(1), pointers[2], sizes[2]) == 0
    for group, expected in ((0, planes[:2]), (1, planes[2:])):
        header, payload = sdk._read_object(writer.path(key(group)), payload=True)
        assert set(header) == {"schema", "key", "bytes", "buffer_sizes"}
        assert payload == b"".join(tensor.numpy().tobytes() for tensor in expected)
    writer.close()
    assert sdk.seal_store(tmp_path) == {"passed": True, "errors": [], "keys": 2, "bytes": sum(sizes), "groups": [0, 1]}
    reader = sdk.FileStore(tmp_path, "decode")
    assert reader.setup() == 0
    destinations = [torch.full((size + (5 if i == 1 else 0),), 255, dtype=torch.uint8) for i, size in enumerate(sizes)]
    reader.memory.bind(destinations)
    assert reader.batch_get_into_multi_buffers(
        [key(0)], [[x.data_ptr() for x in destinations[:2]]], [[len(x) for x in destinations[:2]]]
    ) == [sum(sizes[:2])]
    assert reader.batch_get_into([key(1)], [destinations[2].data_ptr()], [sizes[2]]) == [sizes[2]]
    for source, destination, size in zip(planes, destinations, sizes, strict=True):
        assert bytes(destination[:size].tolist()) == source.numpy().tobytes()
    assert destinations[1][-5:].tolist() == [255] * 5
    assert sdk.validate_store(tmp_path) == {"passed": True, "errors": [], "reads": 2, "read_keys": 2, "groups": [0, 1]}


def test_memory_requires_strong_known_storage_and_full_bounds(sdk):
    memory = sdk.TensorMemory()
    source = torch.arange(16, dtype=torch.uint8)
    pointer = source.data_ptr()
    with pytest.raises(ValueError, match="untracked"):
        memory.read(pointer, 1)
    memory.bind([source[4:8]])  # Keep the complete storage, even through a tensor view.
    del source
    gc.collect()
    assert memory.read(pointer + 3, 5) == bytes(range(3, 8))
    with pytest.raises(ValueError, match="untracked"):
        memory.read(pointer + 15, 2)
    with pytest.raises(ValueError, match="positive integers"):
        memory.read(pointer, -1)


def test_sdk_registration_does_not_authorize_unknown_pointer(sdk, tmp_path):
    writer = sdk.FileStore(tmp_path, "prefill")
    source = torch.ones(3, dtype=torch.uint8)
    assert writer.register_buffer(source.data_ptr(), 3) == 0
    with pytest.raises(ValueError, match="untracked"):
        writer.put_from(key(0), source.data_ptr(), 3)
    assert not list((tmp_path / "store").glob("*.bin"))


def test_decode_requires_seal_and_sealed_writes_fail_closed(sdk, tmp_path):
    reader = sdk.FileStore(tmp_path, "decode")
    with pytest.raises(FileNotFoundError):
        reader.setup()
    writer, sources = seed(sdk, tmp_path)
    original = writer.path(key(0)).read_bytes()
    for store in (reader, writer):
        store.memory.bind(sources)
        with pytest.raises(RuntimeError, match="Only unsealed P"):
            store.put_from(key(0), sources[0].data_ptr(), 10)
    assert writer.path(key(0)).read_bytes() == original
    assert not sdk.validate_store(tmp_path)["passed"]


def test_missing_key_and_small_destination_never_write(sdk, tmp_path):
    seed(sdk, tmp_path)
    reader = sdk.FileStore(tmp_path, "decode")
    destination = torch.full((8,), 99, dtype=torch.uint8)
    reader.memory.bind([destination])
    assert reader.batch_get_into([key(0, "missing")], [destination.data_ptr()], [8]) == [-1]
    assert reader.batch_get_into([key(0)], [destination.data_ptr()], [8]) == [-1]
    assert destination.tolist() == [99] * 8
    assert not sdk.validate_store(tmp_path)["passed"]


def test_multi_destination_unknown_last_pointer_prevents_any_copy(sdk, tmp_path):
    seed(sdk, tmp_path)
    reader = sdk.FileStore(tmp_path, "decode")
    destination = torch.full((6,), 99, dtype=torch.uint8)
    unknown = torch.zeros(4, dtype=torch.uint8)
    reader.memory.bind([destination])
    with pytest.raises(ValueError, match="untracked"):
        reader.batch_get_into_multi_buffers([key(0)], [[destination.data_ptr(), unknown.data_ptr()]], [[6, 4]])
    assert destination.tolist() == [99] * 6


@pytest.mark.parametrize("mutation", ["truncate", "wrong_key", "extra_file", "malformed_seal"])
def test_corruption_or_changed_inventory_fails_validation(sdk, tmp_path, mutation):
    writer, _ = seed(sdk, tmp_path)
    path = writer.path(key(0))
    if mutation == "truncate":
        path.write_bytes(path.read_bytes()[:-1])
    elif mutation == "wrong_key":
        header, payload = sdk._read_object(path, payload=True)
        header["key"] = key(0, "different")
        path.write_bytes(json.dumps(header).encode() + b"\n" + payload)
    elif mutation == "extra_file":
        writer.path(key(0, "extra")).write_bytes(path.read_bytes())
    else:
        (tmp_path / "store-sealed.json").write_text("[]", encoding="utf-8")
    result = sdk.validate_store(tmp_path)
    assert not result["passed"] and result["errors"]


@pytest.mark.parametrize("failure", ["only_one_group", "unfinished", "no_write_log"])
def test_seal_requires_successful_p_manifest_and_both_groups(sdk, tmp_path, failure):
    writer = sdk.FileStore(tmp_path, "prefill")
    source = torch.ones(3, dtype=torch.uint8)
    writer.memory.bind([source])
    writer.put_from(key(0), source.data_ptr(), 3)
    if failure != "only_one_group":
        writer.put_from(key(1), source.data_ptr(), 3)
    if failure == "unfinished":
        (tmp_path / "store" / "incomplete.pending").touch()
    elif failure == "no_write_log":
        writer.log.unlink()
    result = sdk.seal_store(tmp_path)
    assert not result["passed"] and result["errors"]
    assert not (tmp_path / "store-sealed.json").exists()


def test_existence_probes_do_not_count_as_payload_reads(sdk, tmp_path):
    seed(sdk, tmp_path)
    reader = sdk.FileStore(tmp_path, "decode")
    assert reader.batch_is_exist([key(0), key(1)]) == [1, 1]
    result = sdk.validate_store(tmp_path)
    assert not result["passed"] and result["reads"] == 0


def test_reading_only_one_group_fails_validation(sdk, tmp_path):
    seed(sdk, tmp_path)
    reader = sdk.FileStore(tmp_path, "decode")
    destination = torch.zeros(10, dtype=torch.uint8)
    reader.memory.bind([destination])
    assert reader.batch_get_into([key(0)], [destination.data_ptr()], [10]) == [10]
    result = sdk.validate_store(tmp_path)
    assert not result["passed"] and result["groups"] == [0]


def test_payload_is_never_hashed_and_same_size_changes_are_actually_read(sdk, tmp_path, monkeypatch):
    hashed_inputs = []
    real_hash = sdk.hashlib.sha256

    def key_hash(value):
        hashed_inputs.append(value)
        return real_hash(value)

    monkeypatch.setattr(sdk.hashlib, "sha256", key_hash)
    writer, _ = seed(sdk, tmp_path)
    path = writer.path(key(0))
    original = path.read_bytes()
    # Deliberately change one payload byte, preserving metadata and length.
    # Transport audit must not introduce an exact tensor digest comparison.
    path.write_bytes(original[:-1] + b"\xfe")
    reader = sdk.FileStore(tmp_path, "decode")
    destinations = [torch.zeros(10, dtype=torch.uint8), torch.zeros(7, dtype=torch.uint8)]
    reader.memory.bind(destinations)
    assert reader.batch_get_into([key(0), key(1)], [x.data_ptr() for x in destinations], [10, 7]) == [10, 7]
    assert destinations[0].tolist() == list(range(9)) + [254]
    assert sdk.validate_store(tmp_path)["passed"]
    assert hashed_inputs and set(hashed_inputs) == {key(0).encode(), key(1).encode()}


def test_failed_atomic_replace_preserves_previous_object_and_cleans_temporary(sdk, tmp_path, monkeypatch):
    path = tmp_path / "archive.bin"
    path.write_bytes(b"previous")

    def fail_replace(self, target):
        raise OSError("injected atomic replacement failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="atomic replacement"):
        sdk._atomic_write(path, b"replacement")
    assert path.read_bytes() == b"previous"
    assert not list(tmp_path.glob("*.pending"))


@pytest.fixture
def isolated_modules(monkeypatch):
    # Scope SDK monkeypatches to one CPU test, leaving the process otherwise unchanged.
    monkeypatch.setattr(sys, "modules", sys.modules.copy())
    for name in list(sys.modules):
        if name == "mooncake" or name.startswith("mooncake."):
            del sys.modules[name]
    monkeypatch.delenv("LMCACHE_EXTRA_CONFIG", raising=False)


def test_sdk_install_is_explicit_isolated_and_has_no_network_apis(sdk, tmp_path, monkeypatch, isolated_modules):
    sdk.install()
    assert "mooncake" not in sys.modules
    monkeypatch.setenv(
        "LMCACHE_EXTRA_CONFIG", json.dumps({"prefill_check_file_sdk": {"root": str(tmp_path), "stage": "prefill"}})
    )
    sdk.install()
    store_module = sys.modules["mooncake.store"]
    sdk.install()
    assert sys.modules["mooncake.store"] is store_module
    engine = sys.modules["mooncake.engine"].TransferEngine()
    assert engine.initialize("host", "P2PHANDSHAKE", "ascend", "") == 0
    assert engine.register_memory(123, 45) == 0
    assert not hasattr(engine, "batch_transfer_sync_read")
    assert not hasattr(engine, "batch_transfer_sync_write")
    assert not hasattr(store_module, "__file__")
    monkeypatch.setenv(
        "LMCACHE_EXTRA_CONFIG", json.dumps({"prefill_check_file_sdk": {"root": str(tmp_path), "stage": "decode"}})
    )
    with pytest.raises(RuntimeError, match="BEFORE Mooncake"):
        sdk.install()


def test_sdk_rejects_preloaded_real_sdk(sdk, tmp_path, monkeypatch, isolated_modules):
    sys.modules["mooncake"] = types.ModuleType("mooncake")
    monkeypatch.setenv(
        "LMCACHE_EXTRA_CONFIG", json.dumps({"prefill_check_file_sdk": {"root": str(tmp_path), "stage": "prefill"}})
    )
    with pytest.raises(RuntimeError, match="BEFORE Mooncake"):
        sdk.install()


def test_factory_preserves_connector_registration_and_tracks_owners(sdk, tmp_path, monkeypatch, isolated_modules):
    class Connector:
        def _register_cpu_buffer(self):
            self.cpu_calls += 1
            return 17

        def _register_external_owners(self, owners):
            self.external_owners = tuple(owners)
            return 18

    module = types.ModuleType("lmcache.v1.storage_backend.connector.mooncakestore_connector")
    module.MooncakestoreConnector = Connector
    sys.modules[module.__name__] = module
    monkeypatch.setenv(
        "LMCACHE_EXTRA_CONFIG", json.dumps({"prefill_check_file_sdk": {"root": str(tmp_path), "stage": "prefill"}})
    )
    sdk.install()
    connector = Connector()
    connector.cpu_calls = 0
    connector.store = sys.modules["mooncake.store"].MooncakeDistributedStore()
    source, external = torch.arange(8, dtype=torch.uint8), torch.arange(5, dtype=torch.uint8)
    connector.local_cpu_backend = types.SimpleNamespace(
        memory_allocator=types.SimpleNamespace(pin_allocator=types.SimpleNamespace(buffer=source))
    )
    sdk.observe_buffers(Connector)  # Wrapping is idempotent, never calls real methods twice.
    assert connector._register_cpu_buffer() == 17
    assert connector.cpu_calls == 1
    assert connector._register_external_owners(iter([external])) == 18
    assert connector.external_owners == (external,)
    assert connector.store.memory.read(source.data_ptr(), 8) == bytes(range(8))
    assert connector.store.memory.read(external.data_ptr(), 5) == bytes(range(5))
