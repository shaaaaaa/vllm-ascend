# SPDX-License-Identifier: Apache-2.0
"""Fixed-address native experiment wrapper. No production operator replacement."""
import ctypes
import hashlib
import json
from pathlib import Path

import torch
from candidates import Plan

HERE = Path(__file__).resolve().parent
MODES = {'reload': 0, 'fixed_position': 1, 'bounded_position': 2, 'hash_snapshot': 3, 'direct_directory': 4}
_loaded = None
_handles = []


def source_digest():
    h = hashlib.sha256()
    paths = [HERE / 'native.py', HERE / 'build.py'] + sorted((HERE / 'native').glob('*'))
    for path in paths:
        if path.is_file():
            h.update(path.relative_to(HERE).as_posix().encode())
            h.update(b'\0')
            h.update(path.read_bytes())
    return h.hexdigest()


def load_library(build_dir):
    global _loaded
    import torch_npu
    build_dir = Path(build_dir).resolve()
    info = json.loads((build_dir / 'build-info.json').read_text())
    if str(info['torch']) != str(torch.__version__) or str(info['torch_npu']) != str(torch_npu.__version__):
        raise RuntimeError('Torch/torch-npu versions differ from the native build; rebuild')
    if info['source_sha256'] != source_digest():
        raise RuntimeError('native sources changed; rebuild before testing or timing')
    if _loaded is not None:
        if _loaded != build_dir:
            raise RuntimeError('load different builds in separate processes')
        return info
    for name in ('libresident_redesign_kernels.so', 'libresident_redesign_ops.so'):
        options = [build_dir / name, build_dir / 'lib' / name]
        path = next((p for p in options if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f'{name} not found under {build_dir}')
        if name.endswith('_ops.so'):
            torch.ops.load_library(str(path))
        else:
            _handles.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
    _loaded = build_dir
    return info


class NativeCase:
    """Retain every input/output owner until the current stream completes.

    Payload is normalized token-major HBM. Fused mode tests lookup+copy, not
    PCIe/LMCache. run() never publishes a new snapshot. refresh() keeps addresses
    stable for graph replay and must be ordered before that replay on its stream.
    """
    def __init__(self, query, snapshot, dense, variant, *, radius=2, buckets=8192):
        query.validate()
        snapshot.validate(query)
        if variant not in MODES:
            raise ValueError(f'no native specialization for {variant}; use the functional candidate')
        if dense.shape != (query.tokens.shape[0], query.universe, snapshot.kv.shape[-1]):
            raise ValueError('dense source shape mismatch')
        if dense.dtype != snapshot.kv.dtype or dense.device != query.tokens.device:
            raise ValueError('dense source dtype/device mismatch')
        b, q, k = query.tokens.shape
        size = buckets if variant == 'hash_snapshot' else ((query.universe + 255) // 256 * 256 if variant == 'direct_directory' else 256)
        self.variant, self.mode, self.radius, self.universe = variant, MODES[variant], radius, query.universe
        self.meta = torch.zeros((b, q, 16), dtype=torch.int32, device=query.tokens.device)
        self.epochs = torch.zeros((b, 8), dtype=torch.int64, device=query.tokens.device)
        self.tensors = [query.tokens.clone().contiguous(), query.versions.clone().contiguous(),
                        snapshot.tokens.clone().contiguous(), snapshot.versions.clone().contiguous(),
                        snapshot.ready.int().contiguous(), self.meta, self.epochs,
                        torch.empty((b, size), dtype=torch.int32, device=query.tokens.device),
                        torch.empty((b, q*k), dtype=torch.int32, device=query.tokens.device),
                        snapshot.kv.clone().contiguous(), dense.clone().contiguous(), torch.empty_like(snapshot.kv).contiguous()]
        self.refresh(query, snapshot, dense)

    def refresh(self, query, snapshot, dense):
        query.validate()
        snapshot.validate(query)
        if query.tokens.shape != self.tensors[0].shape or query.universe != self.universe:
            raise ValueError('refresh cannot change graph geometry')
        if (snapshot.kv.shape != self.tensors[9].shape or dense.shape != self.tensors[10].shape
                or snapshot.kv.dtype != self.tensors[9].dtype or dense.dtype != self.tensors[10].dtype):
            raise ValueError('refresh cannot change payload shape or dtype')
        values = (query.tokens, query.versions, snapshot.tokens, snapshot.versions, snapshot.ready)
        for destination, source in zip(self.tensors[:5], values, strict=True):
            destination.copy_(source)
        self.meta[..., 0].copy_(query.active)
        self.meta[..., 1].copy_(query.boundary)
        self.meta[..., 2].copy_(query.lengths)
        self.epochs[:, 0].copy_(query.epochs)
        self.epochs[:, 1].copy_(snapshot.epochs)
        self.tensors[9].copy_(snapshot.kv)
        self.tensors[10].copy_(dense)

    def run(self, fused=False):
        torch.ops.resident_redesign.run_(self.tensors, self.universe, self.mode, self.radius, fused)

    @property
    def plan(self):
        return Plan(self.tensors[8], self.variant)

    @property
    def payload(self):
        return self.tensors[11]
