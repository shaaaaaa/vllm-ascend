# SPDX-License-Identifier: Apache-2.0
"""Read-only Mooncake byte API backed by a completed D receive snapshot.

Only the SDK is substituted. LMCache lookup, allocation, CPU sharing, H2D,
compact mapping, and model execution remain in the normal D implementation.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import sys
import types
from pathlib import Path

from layerwise_prefill_file_store import FileStore, RegistrationOnlyEngine, ReplicateConfig, observe_buffers


class SnapshotStore(FileStore):
    """Serve bounded file slices in the ordinary layer-major, planar wire layout."""

    def __init__(self, plan_path: str):
        self.plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        super().__init__(self.plan["output"], "decode")
        self.entries: dict[str, tuple[int, int, int, tuple[int, ...]]] = {}
        self.rows: dict[tuple[int, int], list[dict]] = {}

    def setup(self, *args, **kwargs):
        return 0

    def bind(self, config, metadata) -> None:
        """Bind keys to the current engine's ABI, never to a previous process's pointers."""
        from lmcache.v1.mooncake_layout import mooncake_page_key
        from lmcache.v1.token_database import ChunkedTokenDatabase

        rank = int(metadata.worker_id)
        worker = self.plan["workers"].get(str(rank))
        if worker is None:
            raise ValueError(f"No captured TP rank {rank}")
        if list(metadata.runtime_kv_group_layer_counts) != worker["groups"]:
            raise ValueError("Replay model KV layer counts differ from capture")
        if [list(x) for x in metadata.runtime_kv_group_layer_names] != worker["layer_names"]:
            raise ValueError("Replay model KV layer order differs from capture")
        for row in worker["files"]:
            self.rows.setdefault((row["group"], row["layer"]), []).append(row)
        for rows in self.rows.values():
            rows.sort(key=lambda row: row["component"])
        database = ChunkedTokenDatabase(config, metadata)
        tokens = self.plan["received_tokens"]
        for group, count in enumerate(worker["groups"]):
            layers = tuple(range(count))
            for start, end, key in database.process_tokens(tokens, kv_group=group):
                # MLA's first-rank key alias does not select the captured rank;
                # this SDK instance is already bound to its actual TP worker.
                for candidate in (key, key.with_new_worker_id(0)):
                    self.entries[mooncake_page_key(candidate, count)] = (group, start, end, layers)
                    for layer, layer_key in enumerate(candidate.split_layers(count)):
                        self.entries[layer_key.to_string()] = (group, start, end, (layer,))
        self.record("snapshot_bound", "", 0, rank=rank, tokens=len(tokens))

    def is_exist(self, key):
        return int(key in self.entries)

    def _get(self, method, keys, pointers, sizes):
        self._buffers(keys, pointers, sizes)
        statuses = []
        for key, ptrs, capacities in zip(keys, pointers, sizes, strict=True):
            entry = self.entries.get(key)
            if entry is None:
                statuses.append(-1)
                continue
            group, start, end, layers = entry
            rows = [row for layer in layers for row in self.rows[group, layer]]
            expected = (end - start) * sum(row["row_bytes"] for row in rows)
            if sum(capacities) != expected:
                raise ValueError(f"Replay wire layout mismatch: {key}: expected {expected}, got {sum(capacities)}")
            for pointer, capacity in zip(ptrs, capacities, strict=True):
                self.memory.view(pointer, capacity)
            destination, offset = 0, 0
            for row in rows:
                remaining = (end - start) * row["row_bytes"]
                with Path(row["absolute_path"]).open("rb") as stream:
                    stream.seek(start * row["row_bytes"])
                    while remaining:
                        size = min(remaining, capacities[destination] - offset, 4 * 1024 * 1024)
                        data = stream.read(size)
                        if len(data) != size:
                            raise ValueError("Snapshot payload was truncated after validation")
                        self.memory.write(ptrs[destination] + offset, data)
                        remaining -= size
                        offset += size
                        if offset == capacities[destination]:
                            destination += 1
                            offset = 0
            self.record(method, key, expected, status="ok", group=group, start=start, end=end)
            statuses.append(expected)
        return statuses


def bind_connector(connector_cls) -> None:
    """Wrap after the connector module loads, preserving Ascend's import order."""
    observe_buffers(connector_cls)
    original = connector_cls.__init__
    signature = inspect.signature(original)

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs).arguments
        backend = bound.get("local_cpu_backend")
        self.store.bind(
            bound.get("lmcache_config") or backend.config,
            bound.get("lmcache_metadata") or backend.metadata,
        )

    connector_cls.__init__ = initialize


class ConnectorHook(importlib.abc.MetaPathFinder):
    """One-module lazy hook; do not import LMCache before Ascend patches it."""

    target = "lmcache.v1.storage_backend.connector.mooncakestore_connector"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(fullname)
        execute = spec.loader.exec_module

        def execute_and_bind(module):
            execute(module)
            bind_connector(module.MooncakestoreConnector)
            sys.meta_path.remove(self)

        spec.loader.exec_module = execute_and_bind
        return spec


def install() -> None:
    """Install only inside subprocesses launched by decode_kv_replay.py."""
    config = json.loads(os.environ.get("LMCACHE_EXTRA_CONFIG", "{}")).get("decode_snapshot_replay")
    if config is None:
        return
    plan_path = config["plan"]
    existing = sys.modules.get("mooncake.store")
    if getattr(existing, "snapshot_plan", None) == plan_path:
        return
    if any(name == "mooncake" or name.startswith("mooncake.") for name in sys.modules):
        raise RuntimeError("Snapshot SDK must be installed before Mooncake import")

    def factory():
        return SnapshotStore(plan_path)

    package = types.ModuleType("mooncake")
    package.__path__ = []
    store = types.ModuleType("mooncake.store")
    store.snapshot_plan = plan_path
    store.MooncakeDistributedStore = factory
    store.ReplicateConfig = ReplicateConfig
    engine = types.ModuleType("mooncake.engine")
    engine.TransferEngine = RegistrationOnlyEngine
    package.store, package.engine = store, engine
    for module in (package, store, engine):
        module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, loader=None, is_package=module is package)
        sys.modules[module.__name__] = module

    module = sys.modules.get(ConnectorHook.target)
    if module is not None:
        bind_connector(module.MooncakestoreConnector)
    else:
        sys.meta_path.insert(0, ConnectorHook())
