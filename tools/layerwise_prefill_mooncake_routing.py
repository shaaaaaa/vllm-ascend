# SPDX-License-Identifier: Apache-2.0
"""Test-only two-holder routing. Importing this module does not install hooks."""

import hashlib
import json
import os
import threading
from pathlib import Path


def routed_key(namespace, holder, key):
    return f"{namespace}/holder{holder}/{key}"


def read_holder(stage, device):
    if stage == "prefill" or device == 0:
        return 1
    return 0


def require_placement(store, key, segment, size):
    """Check real placement, since preferred_segment is only a preference."""
    replicas = store.get_replica_desc(key)
    if len(replicas) != 1 or not replicas[0].is_memory_replica():
        raise RuntimeError(f"Expected exactly one memory replica for {key}")
    buffer = replicas[0].get_memory_descriptor().buffer_descriptor
    # Mooncake bindings have used both public and C++-field spellings.
    endpoint = getattr(buffer, "transport_endpoint", None)
    if endpoint is None:
        endpoint = getattr(buffer, "transport_endpoint_", None)
    if endpoint is None:
        endpoint = getattr(buffer, "segment_name", None)
    actual_size = getattr(buffer, "size", None)
    if actual_size is None:
        actual_size = getattr(buffer, "size_", None)
    if endpoint != segment or actual_size != size:
        raise RuntimeError(
            f"Unexpected Mooncake placement for {key}: segment={endpoint}, bytes={actual_size}; "
            f"expected segment={segment}, bytes={size}. Refusing an ambiguous/same-device route."
        )


def read_manifest(path):
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        key, size = entry["key"], entry["size"]
        if not isinstance(key, str) or not key or not isinstance(size, int) or size <= 0:
            raise ValueError("Invalid KV copy manifest")
        if key in entries and entries[key] != size:
            raise ValueError(f"KV size changed for {key}")
        entries[key] = size
    if not entries:
        raise RuntimeError("P did not persist any KV; refusing to start D")
    return entries


def manifest_digest(entries):
    return hashlib.sha256(json.dumps(sorted(entries.items()), separators=(",", ":")).encode()).hexdigest()


class RoutedStore:
    """Keep native buffer/transport calls intact; route only object names/placement."""

    def __init__(self, store, replica_config_cls, route, current_device):
        self.store = store
        self.replica_config_cls = replica_config_cls
        self.route = route
        self.current_device = current_device
        self.lock = threading.Lock()
        self.records = 0
        self.reads = 0

    def setup(self, *args, **kwargs):
        self.device = int(self.current_device())
        self.holder = read_holder(self.route["stage"], self.device)
        if args[4] != "ascend" or args[2] != 0:
            raise ValueError("Model clients must use Ascend and must not own storage segments")
        self.root = Path(self.route["root"])
        self.segment = json.loads((self.root / f"holder{self.holder}/ready.json").read_text())["segment"]
        self.prefix = f"{self.route['namespace']}/holder{self.holder}/"
        status = self.store.setup(*args, **kwargs)
        print(
            f"[PREFILL_MOONCAKE] route: stage={self.route['stage']}, device={self.device}, "
            f"holder_device={self.holder}, segment={self.segment}",
            flush=True,
        )
        return status

    def close(self):
        self.store.close()
        if hasattr(self, "root"):
            report = {
                "device": self.device,
                "holder": self.holder,
                "segment": self.segment,
                "reads": self.reads,
                "written_objects": self.records,
            }
            (self.root / self.route["stage"] / f"route-{os.getpid()}-{id(self)}.json").write_text(
                json.dumps(report),
                encoding="utf-8",
            )

    def __getattr__(self, name):
        # Do not silently leave new key-based APIs un-routed.
        if name in {"get_hostname", "register_buffer", "unregister_buffer"}:
            return getattr(self.store, name)
        raise AttributeError(f"Test routing does not implement Mooncake API {name}")

    def keys(self, keys):
        return [self.prefix + key for key in keys]

    def read(self, method, keys, *args, **kwargs):
        # All TP ranks initialize clients and may query metadata, including P1.
        # P1 does not transfer KV (only P0 writes). Reject self-device DATA
        # transfers, not client initialization or master-only existence lookup.
        if method.startswith("batch_get") and self.device == self.holder:
            raise RuntimeError("Refusing same-device Mooncake holder data transfer")
        result = getattr(self.store, method)(self.keys(keys), *args, **kwargs)
        if method.startswith("batch_get"):
            self.reads += len(keys)
        return result

    def batch_is_exist(self, keys):
        return self.read("batch_is_exist", keys)

    def is_exist(self, key):
        return self.store.is_exist(self.prefix + key)

    def batch_get_buffer(self, keys):
        return self.read("batch_get_buffer", keys)

    def batch_get_into(self, keys, *args, **kwargs):
        return self.read("batch_get_into", keys, *args, **kwargs)

    def batch_get_into_multi_buffers(self, keys, *args, **kwargs):
        return self.read("batch_get_into_multi_buffers", keys, *args, **kwargs)

    def put(self, method, keys, ptrs, sizes):
        if self.route["stage"] != "prefill" or self.device != 0:
            raise RuntimeError("Only P rank0 may write test KV")
        config = self.replica_config_cls()
        config.replica_num = 1
        config.preferred_segment = self.segment
        mapped = self.keys(keys)
        result = getattr(self.store, method)(mapped, ptrs, sizes, config)
        if result is None or len(result) != len(keys) or any(status != 0 for status in result):
            raise RuntimeError(f"Mooncake test KV put failed: {result}")
        counts = [sum(size) if isinstance(size, (tuple, list)) else size for size in sizes]
        for key, count in zip(mapped, counts, strict=True):
            require_placement(self.store, key, self.segment, count)
        with self.lock:
            with (self.root / "prefill/objects.jsonl").open("a", encoding="utf-8") as output:
                for key, size in zip(keys, counts, strict=True):
                    output.write(json.dumps({"key": key, "size": size}) + "\n")
                    self.records += 1
        return result

    def batch_put_from(self, keys, ptrs, sizes, config=None):
        return self.put("batch_put_from", keys, ptrs, sizes)

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes, config=None):
        return self.put("batch_put_from_multi_buffers", keys, ptrs, sizes)


def install_routing():
    """Called only by the explicitly selected test worker extension module."""
    extra = json.loads(os.environ.get("LMCACHE_EXTRA_CONFIG", "{}"))
    route = extra.get("prefill_check_routing")
    if not route:
        return
    import mooncake.store as mooncake
    import torch

    native = mooncake.MooncakeDistributedStore
    if getattr(native, "prefill_check_routed", False):
        return

    def factory():
        return RoutedStore(native(), mooncake.ReplicateConfig, route, torch.npu.current_device)

    factory.prefill_check_routed = True
    mooncake.MooncakeDistributedStore = factory


def copy_objects(store, config_cls, entries, namespace, source_segment, dest_segment, allocate):
    """Copy via Ascend, bounded to two largest-object CPU buffers, then verify bytes."""
    import torch

    size = max(entries.values())
    source, check = allocate(size), allocate(size)
    registered = []
    try:
        for tensor in (source, check):
            status = store.register_buffer(tensor.data_ptr(), size)
            if status not in (None, 0):
                raise RuntimeError(f"Copy buffer registration failed: {status}")
            registered.append(tensor)
        config = config_cls()
        config.replica_num = 1
        config.preferred_segment = dest_segment
        for index, (key, count) in enumerate(entries.items(), 1):
            source_key = routed_key(namespace, 1, key)
            dest_key = routed_key(namespace, 0, key)
            require_placement(store, source_key, source_segment, count)
            result = store.batch_get_into_multi_buffers([source_key], [[source.data_ptr()]], [[count]])
            if result != [count]:
                raise RuntimeError(f"Copy read failed or was short for {key}: {result}")
            result = store.batch_put_from_multi_buffers([dest_key], [[source.data_ptr()]], [[count]], config)
            if result != [0]:
                raise RuntimeError(f"Copy put failed for {key}: {result}")
            require_placement(store, dest_key, dest_segment, count)
            result = store.batch_get_into_multi_buffers([dest_key], [[check.data_ptr()]], [[count]])
            if result != [count] or not torch.equal(source[:count], check[:count]):
                raise RuntimeError(f"Copy verification failed for {key}")
            print(f"[PREFILL_MOONCAKE] copied and verified {index}/{len(entries)} objects", flush=True)
    finally:
        for tensor in reversed(registered):
            store.unregister_buffer(tensor.data_ptr())
    return {"objects": len(entries), "bytes": sum(entries.values()), "manifest_sha256": manifest_digest(entries)}
