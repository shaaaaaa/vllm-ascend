# SPDX-License-Identifier: Apache-2.0
"""File-backed Mooncake SDK for the isolated P/D correctness subprocesses.

The real LMCache connector still plans pages, fences producers and invokes the
SDK. Only that SDK is replaced. Payload bytes are stored verbatim, never hashed.
The SHA-256 used below encodes *keys as filenames*, not tensor contents.
"""

import functools
import hashlib
import importlib.machinery
import json
import os
import sys
import tempfile
import threading
import types
import uuid
from pathlib import Path

SCHEMA = 1
MAX_HEADER_BYTES = 65536
PUT_METHODS = frozenset({"batch_put_from", "batch_put_from_multi_buffers"})
GET_METHODS = frozenset({"batch_get_into", "batch_get_into_multi_buffers"})


def _filename(key):
    if not isinstance(key, str) or not key:
        raise ValueError("File SDK keys must be nonempty strings")
    return hashlib.sha256(key.encode("utf-8")).hexdigest() + ".bin"


def _group(key):
    parts = key.split("@")
    if parts[0] == "__lmcache_page_v1__":
        parts = parts[2:]
    if len(parts) < 6 or parts[5] not in ("0", "1"):
        raise ValueError(f"File SDK requires an explicit DSA group in key: {key}")
    return int(parts[5])


def _atomic_write(path, contents):
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pending", delete=False) as stream:
            pending = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def _read_object(path, *, payload=False):
    if path.is_symlink():
        raise ValueError(f"File SDK rejects symlink object: {path}")
    with path.open("rb") as stream:
        line = stream.readline(MAX_HEADER_BYTES + 1)
        if len(line) > MAX_HEADER_BYTES or not line.endswith(b"\n"):
            raise ValueError(f"Invalid file SDK header: {path}")
        header = json.loads(line)
        if not isinstance(header, dict) or header.get("schema") != SCHEMA:
            raise ValueError(f"Invalid file SDK schema: {path}")
        key, size, sizes = header.get("key"), header.get("bytes"), header.get("buffer_sizes")
        if (
            type(size) is not int
            or size <= 0
            or not isinstance(sizes, list)
            or not sizes
            or any(type(value) is not int or value <= 0 for value in sizes)
            or sum(sizes) != size
            or path.name != _filename(key)
            or os.fstat(stream.fileno()).st_size - stream.tell() != size
        ):
            raise ValueError(f"File SDK object key/size mismatch: {path}")
        return header, stream.read() if payload else None


def _seal_entries(root):
    marker = root / "store-sealed.json"
    if marker.is_symlink():
        raise ValueError("File SDK seal cannot be a symlink")
    seal = json.loads(marker.read_text(encoding="utf-8"))
    if (
        not isinstance(seal, dict)
        or seal.get("schema") != SCHEMA
        or not isinstance(seal.get("entries"), list)
        or not seal["entries"]
    ):
        raise ValueError("Invalid or empty file SDK seal")
    entries = {}
    for entry in seal["entries"]:
        if not isinstance(entry, dict):
            raise ValueError("Invalid file SDK seal entry")
        key = entry["key"]
        if key in entries or entry["file"] != _filename(key):
            raise ValueError("File SDK seal contains duplicate/mismatched keys")
        entries[key] = entry
    return entries


def _logs(root, stage):
    for path in sorted((root / stage).glob("store-io-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"Invalid file SDK log: {path}")
                yield item


def seal_store(root):
    """Seal successful P writes after the launcher has joined ALL P processes.

    Returns ``{passed, errors, keys, bytes, groups}``; errors never create a seal.
    The caller must ensure P has exited: this is not a live-writer locking API.
    No payload digest is created or checked.
    """
    root = Path(root).resolve()
    errors, entries = [], []
    try:
        if (root / "store-sealed.json").exists():
            raise ValueError("File SDK store is already sealed")
        if list((root / "store").glob("*.pending")):
            raise ValueError("File SDK store contains an unfinished atomic write")
        writes = {}
        for item in _logs(root, "prefill"):
            if item.get("method") in PUT_METHODS:
                if item.get("status") != "ok":
                    raise ValueError("P reported an unsuccessful file SDK write")
                writes[item["key"]] = item["bytes"]
        for path in sorted((root / "store").glob("*.bin")):
            header, _ = _read_object(path)
            key = header["key"]
            if writes.get(key) != header["bytes"]:
                raise ValueError(f"Object lacks a matching successful P write: {key}")
            entries.append({**header, "file": path.name, "kv_group": _group(key)})
        if not entries or set(writes) != {entry["key"] for entry in entries}:
            raise ValueError("P write manifest and file store do not match")
        groups = sorted({entry["kv_group"] for entry in entries})
        if groups != [0, 1]:
            raise ValueError("P file store must contain both DSA groups")
        seal = {"schema": SCHEMA, "entries": entries, "payload_hashes": False}
        _atomic_write(root / "store-sealed.json", json.dumps(seal, ensure_ascii=False).encode("utf-8"))
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(str(error))
    return {
        "passed": not errors,
        "errors": errors,
        "keys": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "groups": sorted({entry["kv_group"] for entry in entries}),
    }


def validate_store(root):
    """Validate sealed sizes/keys and actual D reads, including both DSA groups.

    Returns a summary with ``passed`` and ``errors``; it does not raise for bad
    archives. Same-size payload differences belong to tensor comparison, not
    this transport audit. Existence probes alone never count as a D read.
    """
    root = Path(root).resolve()
    errors, reads, read_keys, groups = [], 0, set(), set()
    try:
        entries = _seal_entries(root)
        files = {path.name for path in (root / "store").glob("*.bin")}
        if files != {entry["file"] for entry in entries.values()}:
            raise ValueError("Sealed file inventory changed")
        writes = {}
        for item in _logs(root, "prefill"):
            if item.get("method") in PUT_METHODS:
                if item.get("status") != "ok":
                    raise ValueError("P reported an unsuccessful file SDK write")
                writes[item["key"]] = item["bytes"]
        if set(writes) != set(entries):
            raise ValueError("Sealed keys differ from the P manifest")
        for key, entry in entries.items():
            header, _ = _read_object(root / "store" / entry["file"])
            if (
                header != {field: value for field, value in entry.items() if field not in ("file", "kv_group")}
                or writes[key] != entry["bytes"]
                or entry["kv_group"] != _group(key)
            ):
                raise ValueError(f"Sealed metadata or P write changed: {key}")
        for item in _logs(root, "decode"):
            if item.get("method") in PUT_METHODS:
                raise ValueError("D attempted to write its read-only file store")
            if item.get("method") not in GET_METHODS:
                continue
            key = item.get("key")
            if item.get("status") != "ok" or key not in entries or item.get("bytes") != entries[key]["bytes"]:
                raise ValueError(f"D read failed or differs from the P manifest: {key}")
            reads += 1
            read_keys.add(key)
            groups.add(_group(key))
        if not reads or groups != {0, 1}:
            raise ValueError("D must actually read file payloads for both DSA groups")
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(str(error))
    return {
        "passed": not errors,
        "errors": errors,
        "reads": reads,
        "read_keys": len(read_keys),
        "groups": sorted(groups),
    }


class TensorMemory:
    """Retain real torch storages and resolve only addresses inside those owners."""

    def __init__(self):
        self.storages = {}
        self.lock = threading.RLock()

    def bind(self, owners):
        with self.lock:
            for owner in owners:
                storage = owner.untyped_storage()
                self.storages[(str(storage.device), storage.data_ptr())] = storage

    def view(self, pointer, size):
        import torch

        if type(pointer) is not int or type(size) is not int or pointer <= 0 or size <= 0:
            raise ValueError("File SDK buffer addresses and sizes must be positive integers")
        with self.lock:
            matches = [
                (base, storage)
                for (_, base), storage in self.storages.items()
                if base <= pointer and pointer + size <= base + storage.nbytes()
            ]
            if len(matches) == 1:
                base, storage = matches[0]
                return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(
                    storage, pointer - base, (size,), (1,)
                )
        raise ValueError(f"File SDK received an untracked/ambiguous buffer: pointer={pointer:#x}, size={size}")

    def read(self, pointer, size):
        return self.view(pointer, size).cpu().numpy().tobytes()

    def write(self, pointer, data):
        import torch

        # Test-only synchronous copy matches the SDK's completed-transfer return.
        self.view(pointer, len(data)).copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8), non_blocking=False)


def observe_buffers(connector_cls):
    """Observe owners at registration while executing the real connector methods."""
    if getattr(connector_cls, "_file_check_observed", False):
        return
    register_cpu = connector_cls._register_cpu_buffer
    register_external = connector_cls._register_external_owners

    @functools.wraps(register_cpu)
    def cpu(self):
        allocator = self.local_cpu_backend.memory_allocator
        slab = getattr(getattr(allocator, "pin_allocator", None), "buffer", None)
        if slab is not None:
            self.store.memory.bind((slab,))
        return register_cpu(self)

    @functools.wraps(register_external)
    def external(self, owners):
        owners = tuple(owners)
        self.store.memory.bind(owners)
        return register_external(self, owners)

    connector_cls._register_cpu_buffer = cpu
    connector_cls._register_external_owners = external
    connector_cls._file_check_observed = True


class FileStore:
    """Mooncake byte API with atomic P writes and sealed, read-only D access."""

    def __init__(self, root, stage, memory=None):
        self.root, self.stage = Path(root), stage
        if not self.root.is_absolute() or stage not in ("prefill", "decode"):
            raise ValueError("File SDK requires an absolute run root and a P/D stage")
        self.archive = self.root / "store"
        if stage == "prefill":
            self.archive.mkdir(parents=True, exist_ok=True)
        (self.root / stage).mkdir(parents=True, exist_ok=True)
        self.memory = memory if memory is not None else TensorMemory()
        self.identity = f"{os.getpid()}-{uuid.uuid4().hex}"
        self.log = self.root / stage / f"store-io-{self.identity}.jsonl"
        self.lock = threading.RLock()
        self.sealed = None

    def _read_guard(self):
        if self.stage == "decode" and self.sealed is None:
            self.sealed = _seal_entries(self.root)

    def setup(self, *args, **kwargs):
        self._read_guard()
        self.record("setup", "", 0)
        return 0

    def get_hostname(self):
        return f"file-store-{self.identity}"

    def register_buffer(self, pointer, size):
        # Ownership is observed separately; registration cannot authorize an
        # unknown address (which may point to device memory).
        return 0

    def unregister_buffer(self, pointer):
        return 0

    def close(self):
        self.record("close", "", 0)
        with self.memory.lock:
            self.memory.storages.clear()

    def path(self, key):
        return self.archive / _filename(key)

    def record(self, method, key, size, **details):
        with self.lock, self.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"method": method, "key": key, "bytes": size, **details}) + "\n")

    def is_exist(self, key):
        self._read_guard()
        result = int((self.stage != "decode" or key in self.sealed) and self.path(key).is_file())
        self.record("is_exist", key, 0, status=result)
        return result

    def batch_is_exist(self, keys):
        return [self.is_exist(key) for key in keys]

    @staticmethod
    def _buffers(keys, pointers, sizes):
        if not (len(keys) == len(pointers) == len(sizes)):
            raise ValueError("File SDK keys/pointers/sizes have different lengths")
        for key, ptrs, counts in zip(keys, pointers, sizes, strict=True):
            _filename(key)
            if not ptrs or len(ptrs) != len(counts) or any(type(size) is not int or size <= 0 for size in counts):
                raise ValueError("File SDK requires matching pointers and positive buffer sizes")

    def _put(self, method, keys, pointers, sizes):
        self._buffers(keys, pointers, sizes)
        if self.stage != "prefill" or (self.root / "store-sealed.json").exists():
            for key in keys:
                self.record(method, key, 0, status="write_forbidden")
            raise RuntimeError("Only unsealed P may write the file store")
        statuses = []
        for key, ptrs, counts in zip(keys, pointers, sizes, strict=True):
            try:
                payload = b"".join(self.memory.read(ptr, count) for ptr, count in zip(ptrs, counts, strict=True))
                header = {"schema": SCHEMA, "key": key, "bytes": len(payload), "buffer_sizes": list(counts)}
                _atomic_write(self.path(key), json.dumps(header).encode("utf-8") + b"\n" + payload)
            except BaseException:
                self.record(method, key, 0, status="error")
                raise
            self.record(method, key, len(payload), buffers=len(ptrs), status="ok")
            statuses.append(0)
        return statuses

    def batch_put_from(self, keys, pointers, sizes, config=None):
        return self._put("batch_put_from", keys, [[ptr] for ptr in pointers], [[size] for size in sizes])

    def batch_put_from_multi_buffers(self, keys, pointers, sizes, config=None):
        return self._put("batch_put_from_multi_buffers", keys, pointers, sizes)

    def put_from(self, key, pointer, size, config=None):
        return self.batch_put_from([key], [pointer], [size], config)[0]

    def _get(self, method, keys, pointers, sizes):
        self._read_guard()
        self._buffers(keys, pointers, sizes)
        statuses = []
        for key, ptrs, capacities in zip(keys, pointers, sizes, strict=True):
            try:
                if self.stage == "decode" and key not in self.sealed:
                    raise FileNotFoundError(key)
                header, payload = _read_object(self.path(key), payload=True)
                if header["key"] != key:
                    raise ValueError(f"File SDK key mismatch: {key}")
                if self.stage == "decode" and header != {
                    field: value for field, value in self.sealed[key].items() if field not in ("file", "kv_group")
                }:
                    raise ValueError(f"Sealed file SDK metadata changed: {key}")
                if len(payload) > sum(capacities):
                    self.record(method, key, 0, status="destination_too_small")
                    statuses.append(-1)
                    continue
                # Validate every supplied destination before the first write.
                for ptr, capacity in zip(ptrs, capacities, strict=True):
                    self.memory.view(ptr, capacity)
                offset = 0
                for pointer, capacity in zip(ptrs, capacities, strict=True):
                    data = payload[offset : offset + capacity]
                    if data:
                        self.memory.write(pointer, data)
                    offset += len(data)
            except FileNotFoundError:
                self.record(method, key, 0, status="missing")
                statuses.append(-1)
                continue
            except BaseException:
                self.record(method, key, 0, status="error")
                raise
            self.record(method, key, len(payload), buffers=len(ptrs), status="ok")
            statuses.append(len(payload))
        return statuses

    def batch_get_into(self, keys, pointers, sizes):
        return self._get("batch_get_into", keys, [[ptr] for ptr in pointers], [[size] for size in sizes])

    def batch_get_into_multi_buffers(self, keys, pointers, sizes):
        return self._get("batch_get_into_multi_buffers", keys, pointers, sizes)


class RegistrationOnlyEngine:
    """SDK registration setup without endpoints; live P2P APIs are absent."""

    def initialize(self, *args):
        return 0

    def get_rpc_port(self):
        return 1  # Unused session identifier: never bound or connected.

    def get_engine(self):
        return self

    def register_memory(self, pointer, size):
        return 0

    def unregister_memory(self, pointer):
        return 0


class ReplicateConfig:
    def __init__(self):
        self.replica_num = 1
        self.preferred_segment = ""


def install():
    """Install only for explicit file-check config, before any Mooncake import."""
    config = json.loads(os.environ.get("LMCACHE_EXTRA_CONFIG", "{}")).get("prefill_check_file_sdk")
    if config is None:
        return
    if config["stage"] not in ("prefill", "decode") or not Path(config["root"]).is_absolute():
        raise ValueError("Invalid file SDK test configuration")
    identity = (str(Path(config["root"]).resolve()), config["stage"])
    existing = sys.modules.get("mooncake.store")
    if getattr(existing, "file_check_config", None) == identity:
        return
    if any(name == "mooncake" or name.startswith("mooncake.") for name in sys.modules):
        raise RuntimeError("File SDK must be installed BEFORE Mooncake is imported, with one immutable config")

    def factory():
        from lmcache.v1.storage_backend.connector.mooncakestore_connector import MooncakestoreConnector

        observe_buffers(MooncakestoreConnector)
        return FileStore(identity[0], identity[1])

    package = types.ModuleType("mooncake")
    package.__path__ = []
    store = types.ModuleType("mooncake.store")
    store.file_check = True
    store.file_check_config = identity
    store.MooncakeDistributedStore = factory
    store.ReplicateConfig = ReplicateConfig
    engine = types.ModuleType("mooncake.engine")
    engine.TransferEngine = RegistrationOnlyEngine
    package.store, package.engine = store, engine
    for module in (package, store, engine):
        module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, loader=None, is_package=module is package)
        sys.modules[module.__name__] = module
