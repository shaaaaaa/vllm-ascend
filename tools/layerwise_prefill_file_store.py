# SPDX-License-Identifier: Apache-2.0
"""Test-only replacement of the Mooncake SDK, NOT of its LMCache connector.

Loaded by the file-check launcher's private sitecustomize in every subprocess.
No import of Mooncake native libraries, transport, master or holder is needed.
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


class TensorMemory:
    """Resolve SDK raw addresses using their real tensor owners; never dereference
    an unknown address as CPU memory (it might be an NPU address).
    """

    def __init__(self):
        self.storages = {}
        self.lock = threading.RLock()

    def bind(self, owners):
        with self.lock:
            for owner in owners:
                storage = owner.untyped_storage()
                self.storages[storage.data_ptr()] = storage

    def view(self, pointer, size):
        import torch

        with self.lock:
            for base, storage in self.storages.items():
                if base <= pointer and pointer + size <= base + storage.nbytes() and size > 0:
                    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(
                        storage, pointer - base, (size,), (1,)
                    )
        raise ValueError(f"File SDK received an untracked buffer: pointer={pointer:#x}, size={size}")

    def read(self, pointer, size):
        return self.view(pointer, size).cpu().numpy().tobytes()

    def write(self, pointer, data):
        import torch

        # Blocking copy is intentional: the native API returns only after the
        # transfer finishes. NPU destination contents must be ready on return.
        self.view(pointer, len(data)).copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8), non_blocking=False)


def observe_buffers(connector_cls):
    """Observe ownership at registration; execute every original method unchanged."""
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
        self.store.memory.bind(owners)
        return register_external(self, owners)

    connector_cls._register_cpu_buffer = cpu
    connector_cls._register_external_owners = external
    connector_cls._file_check_observed = True


class FileStore:
    """One atomic file per ORIGINAL Mooncake key; multi-buffers stay in order."""

    def __init__(self, root, stage, memory=None):
        self.root, self.stage = Path(root), stage
        self.archive = self.root / "store"
        self.archive.mkdir(exist_ok=True)
        self.memory = memory if memory is not None else TensorMemory()
        self.identity = f"{os.getpid()}-{uuid.uuid4().hex}"
        self.log = self.root / stage / f"store-io-{self.identity}.jsonl"
        self.lock = threading.RLock()

    def setup(self, *args, **kwargs):
        if self.stage == "decode" and not (self.root / "store-sealed.json").is_file():
            raise RuntimeError("D cannot start before P has fenced and sealed its file store")
        self.record("setup", "", 0)
        return 0

    def get_hostname(self):
        return f"file-store-{self.identity}"

    def register_buffer(self, pointer, size):
        # Native registration only; ownership is observed above, not inferred.
        return 0

    def unregister_buffer(self, pointer):
        return 0

    def close(self):
        self.record("close", "", 0)
        with self.memory.lock:
            self.memory.storages.clear()

    def path(self, key):
        return self.archive / (hashlib.sha256(key.encode()).hexdigest() + ".bin")

    def record(self, method, key, size, **details):
        with self.lock, self.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"method": method, "key": key, "bytes": size, **details}) + "\n")

    def is_exist(self, key):
        result = int(self.path(key).is_file())
        self.record("is_exist", key, 0, status=result)
        return result

    def batch_is_exist(self, keys):
        return [self.is_exist(key) for key in keys]

    def _put(self, method, keys, pointers, sizes):
        if self.stage != "prefill" or (self.root / "store-sealed.json").exists():
            raise RuntimeError("Only unsealed P may write the file store")
        statuses = []
        for key, ptrs, counts in zip(keys, pointers, sizes, strict=True):
            payload = b"".join(self.memory.read(ptr, count) for ptr, count in zip(ptrs, counts, strict=True))
            digest = hashlib.sha256(payload).hexdigest()
            header = {"key": key, "bytes": len(payload), "sha256": digest}
            with tempfile.NamedTemporaryFile(dir=self.archive, suffix=".pending", delete=False) as stream:
                pending = Path(stream.name)
                try:
                    stream.write(json.dumps(header).encode() + b"\n")
                    stream.write(payload)
                except BaseException:
                    stream.close()
                    pending.unlink(missing_ok=True)
                    raise
            try:
                pending.replace(self.path(key))
            finally:
                pending.unlink(missing_ok=True)
            self.record(method, key, len(payload), sha256=digest, buffers=len(ptrs))
            statuses.append(0)
        return statuses

    def batch_put_from(self, keys, pointers, sizes, config=None):
        return self._put("batch_put_from", keys, [[ptr] for ptr in pointers], [[size] for size in sizes])

    def batch_put_from_multi_buffers(self, keys, pointers, sizes, config=None):
        return self._put("batch_put_from_multi_buffers", keys, pointers, sizes)

    def put_from(self, key, pointer, size, config=None):
        return self.batch_put_from([key], [pointer], [size], config)[0]

    def _get(self, method, keys, pointers, sizes):
        statuses = []
        for key, ptrs, capacities in zip(keys, pointers, sizes, strict=True):
            if not ptrs or len(ptrs) != len(capacities) or any(size <= 0 for size in capacities):
                raise ValueError("File SDK requires matching destination pointers and positive sizes")
            try:
                with self.path(key).open("rb") as stream:
                    header = json.loads(stream.readline())
                    payload = stream.read()
            except FileNotFoundError:
                self.record(method, key, 0, status="missing")
                statuses.append(-1)
                continue
            digest = hashlib.sha256(payload).hexdigest()
            if header != {"key": key, "bytes": len(payload), "sha256": digest}:
                raise RuntimeError(f"File store corrupt or key mismatch: {key}")
            if len(payload) > sum(capacities):
                self.record(method, key, 0, status="destination_too_small")
                statuses.append(-1)
                continue
            offset = 0
            for pointer, capacity in zip(ptrs, capacities, strict=True):
                data = payload[offset : offset + capacity]
                if data:
                    self.memory.write(pointer, data)
                offset += len(data)
            self.record(method, key, len(payload), sha256=digest, buffers=len(ptrs), status="ok")
            statuses.append(len(payload))  # Actual size, including short/tail chunks.
        return statuses

    def batch_get_into(self, keys, pointers, sizes):
        return self._get("batch_get_into", keys, [[ptr] for ptr in pointers], [[size] for size in sizes])

    def batch_get_into_multi_buffers(self, keys, pointers, sizes):
        return self._get("batch_get_into_multi_buffers", keys, pointers, sizes)


class RegistrationOnlyEngine:
    """Replace native registration setup, not any inference/collective code.

    No network endpoints are created. Live P2P transfer APIs are deliberately
    absent: attempting concurrent RemoteFill must fail rather than fake success.
    """

    def initialize(self, *args):
        return 0

    def get_rpc_port(self):
        return 1  # An unused session identifier, never bound or connected.

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
    """Explicit test configuration only; safe no-op on normal serving imports."""
    config = json.loads(os.environ.get("LMCACHE_EXTRA_CONFIG", "{}")).get("prefill_check_file_sdk")
    if config is None:
        return
    if config["stage"] not in ("prefill", "decode") or not Path(config["root"]).is_absolute():
        raise ValueError("Invalid file SDK test configuration")
    if getattr(sys.modules.get("mooncake.store"), "file_check", False):
        return
    if "mooncake.store" in sys.modules or "mooncake.engine" in sys.modules:
        raise RuntimeError("File SDK must be installed BEFORE Mooncake is imported")

    def factory():
        from lmcache.v1.storage_backend.connector.mooncakestore_connector import MooncakestoreConnector

        observe_buffers(MooncakestoreConnector)
        return FileStore(config["root"], config["stage"])

    package = types.ModuleType("mooncake")
    package.__path__ = []
    store = types.ModuleType("mooncake.store")
    store.file_check = True
    store.MooncakeDistributedStore = factory
    store.ReplicateConfig = ReplicateConfig
    engine = types.ModuleType("mooncake.engine")
    engine.TransferEngine = RegistrationOnlyEngine
    package.store, package.engine = store, engine
    for module in (package, store, engine):
        module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, loader=None, is_package=module is package)
        sys.modules[module.__name__] = module
