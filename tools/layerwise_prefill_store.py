# SPDX-License-Identifier: Apache-2.0
"""Test-only durable LMCache transport for sequential P/D validation.

This deliberately does not emulate Mooncake. It preserves logical MemoryObj
metadata (including flat-object valid_tokens), never persists allocator pointers,
and uses the ordinary RemoteBackend get/put path. Only trusted, locally generated
archives from this test are accepted.
"""

import hashlib
import json
import os
from pathlib import Path

import torch
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector


def logical_nbytes(shapes, dtypes):
    """Size of packed KV planes, excluding allocator alignment padding."""
    size = sum(torch.Size(shape).numel() * dtype.itemsize for shape, dtype in zip(shapes, dtypes, strict=True))
    if size <= 0:
        raise RuntimeError("Archive must contain non-empty KV data")
    return size


class ValidationFileConnector(RemoteConnector):
    def __init__(self, loop, local_cpu_backend, config):
        # The scheduler uses a metadata-less CPU stub, so do not initialize
        # RemoteConnector's model-shape-dependent allocation defaults here.
        self.local_cpu_backend = local_cpu_backend
        self.root = Path(config.get_extra_config_value("validation_archive", ""))
        self.stage_dir = Path(config.get_extra_config_value("validation_stage_dir", ""))
        self.read_only = bool(config.get_extra_config_value("validation_read_only", False))
        if not self.root.is_absolute() or not self.stage_dir.is_absolute():
            raise ValueError("Validation archive paths must be absolute")
        self.root.mkdir(parents=True, exist_ok=True)
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key):
        return self.root / (hashlib.sha256(key.to_string().encode()).hexdigest() + ".pt")

    async def exists(self, key):
        return self.exists_sync(key)

    def exists_sync(self, key):
        return self.path_for(key).is_file()

    def requires_put_completion(self):
        return True

    async def put(self, key, memory_obj):
        # InstrumentedRemoteConnector.put releases the serializer's owned
        # reference in its finally block. Releasing it here too is a double free.
        if self.read_only:
            raise RuntimeError("D must not modify the P archive")
        # The copy finishes before publishing the file.
        positions = memory_obj.meta.cached_positions
        if positions is not None:
            positions = torch.as_tensor(positions, dtype=torch.int64).detach().cpu().clone()
        shapes, dtypes = memory_obj.get_shapes(), memory_obj.get_dtypes()
        size = logical_nbytes(shapes, dtypes)
        raw = torch.frombuffer(bytearray(memory_obj.byte_array), dtype=torch.uint8)
        if raw.numel() < size:
            raise RuntimeError(f"Archive source is truncated: logical_bytes={size}, raw_bytes={raw.numel()}")
        payload = {
            "key": key.to_string(),
            "kv_group": key.kv_group,
            "layer_id": getattr(key, "layer_id", None),
            "worker_id": key.worker_id,
            "shapes": [list(s) for s in shapes],
            "dtypes": [str(d) for d in dtypes],
            "fmt": memory_obj.get_memory_format().value,
            "valid_tokens": int(memory_obj.meta.valid_tokens) if memory_obj.meta.valid_tokens is not None else None,
            "cached_positions": positions,
            # batched_allocate exposes an aligned raw view; allocate exposes
            # only logical bytes. Padding is not KV and must not be serialized.
            "logical_bytes": size,
            "raw": raw[:size].clone(),
        }
        payload["sha256"] = hashlib.sha256(payload["raw"].numpy().tobytes()).hexdigest()
        path = self.path_for(key)
        temp = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(payload, temp)
        temp.replace(path)

    async def get(self, key):
        path = self.path_for(key)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["key"] != key.to_string():
            raise RuntimeError("Archive key mismatch")
        raw = payload["raw"]
        if raw.dtype != torch.uint8 or raw.ndim != 1:
            raise RuntimeError("Archive raw data must be a flat byte tensor")
        digest = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
        if digest != payload["sha256"]:
            raise RuntimeError(f"Archive checksum mismatch: {path}")
        shapes = [torch.Size(s) for s in payload["shapes"]]
        dtypes = [getattr(torch, d.removeprefix("torch.")) for d in payload["dtypes"]]
        size = logical_nbytes(shapes, dtypes)
        if raw.numel() < size:
            raise RuntimeError(f"Archive data is truncated: logical_bytes={size}, raw_bytes={raw.numel()}")
        if "logical_bytes" in payload and (payload["logical_bytes"] != size or raw.numel() != size):
            raise RuntimeError("Archive logical size metadata mismatch")
        # Legacy archives included the batched allocator's trailing padding.
        # Verify the ENTIRE saved checksum above, but restore only KV bytes.
        # Keep that original digest in the read log for sealed-archive checks.
        obj = self.local_cpu_backend.allocate(
            shapes,
            dtypes,
            MemoryFormat(payload["fmt"]),
            busy_loop=False,
        )
        if obj is None:
            raise RuntimeError("Insufficient CPU cache capacity for archive reload")
        try:
            # Flat objects also need this metadata on the cleanup/error path.
            obj.meta.valid_tokens = payload["valid_tokens"]
            obj.meta.cached_positions = payload["cached_positions"]
            dst = obj.raw_data.view(torch.uint8).reshape(-1)
            if obj.get_shapes() != shapes or obj.get_dtypes() != dtypes or dst.numel() < size:
                raise RuntimeError(f"Archive allocation size mismatch: logical_bytes={size}, raw_bytes={dst.numel()}")
            dst[:size].copy_(raw[:size])
            with (self.stage_dir / f"archive_reads_{os.getpid()}.jsonl").open("a", encoding="utf-8") as out:
                out.write(
                    json.dumps(
                        {
                            "key": payload["key"],
                            "sha256": digest,
                            "kv_group": payload["kv_group"],
                            "bytes": raw.numel(),
                            "logical_bytes": size,
                        }
                    )
                    + "\n"
                )
            return obj
        except BaseException:
            obj.ref_count_down()
            raise

    async def list(self):
        return [p.name for p in self.root.glob("*.pt")]

    async def close(self):
        pass
