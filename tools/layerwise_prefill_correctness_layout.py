# SPDX-License-Identifier: Apache-2.0
"""Explicit local-page layout selection for the correctness tool only.

Importing this module has no effect. The correctness launcher and its worker
extension call ``install_local_merged_layout`` before creating LMCache engines.
Only two layout predicates are replaced; allocation, page keys, source owners,
DMA, and LocalCPU publication continue through their production implementations.
This is neither a production configuration option nor a Mooncake SDK substitute.
"""

import importlib
import sys
from typing import Any


def _require_local_merged_config(config: Any) -> None:
    """Reject configurations that could create a nonlocal storage backend."""
    extra = getattr(config, "extra_config", None) or {}
    required = {
        "remote_url is None": getattr(config, "remote_url", None) is None,
        "remote fill disabled": not getattr(config, "enable_remote_lmcache_store", False),
        "PD backend disabled": not getattr(config, "enable_pd", False),
        "P2P backend disabled": not getattr(config, "enable_p2p", False),
        "disk backend disabled": not getattr(config, "local_disk", None),
        "GDS backend disabled": not getattr(config, "gds_path", None),
        "Maru backend disabled": not getattr(config, "maru_path", None),
        "local CPU enabled": bool(getattr(config, "local_cpu", False)),
        "positive CPU cache capacity": float(getattr(config, "max_local_cpu_size", 0)) > 0,
        "layerwise enabled": bool(getattr(config, "use_layerwise", False)),
        "shared CPU enabled": bool(
            extra.get("enable_shared_cpu_cache", getattr(config, "enable_shared_cpu_cache", False))
        ),
        "save_only_first_rank": bool(extra.get("save_only_first_rank", False)),
        "page-first layout": bool(extra.get("mooncake_page_first_multi_buffer", False)),
        "merged layer objects": bool(extra.get("mooncake_layer_merged_page_objects", False)),
        "native direct store disabled": not extra.get("mooncake_direct_npu_prefill_store", False),
        "storage plugins disabled": not getattr(config, "storage_plugins", None),
        "NIXL storage disabled": not extra.get("enable_nixl_storage", False),
    }
    failed = [name for name, valid in required.items() if not valid]
    if failed:
        raise RuntimeError("Local merged correctness layout requires " + ", ".join(failed))


def _local_merged_layout_enabled(config: Any) -> bool:
    _require_local_merged_config(config)
    return True


def install_local_merged_layout() -> dict[str, Any]:
    """Enable the two local-page layout predicates in this test process.

    Call before importing vLLM/creating engines, and from the correctness worker
    extension in spawned processes. Already imported LMCache/Ascend function
    aliases are also replaced. Repeated calls are harmless; no environment
    variable, backend implementation, device operation, or SDK is modified.
    """
    layout = importlib.import_module("lmcache.v1.mooncake_layout")
    names = ("mooncake_page_layout_enabled", "mooncake_layer_pages_enabled")
    originals = tuple(getattr(layout, name) for name in names)
    patched = []
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not (
            module_name == "lmcache"
            or module_name.startswith("lmcache.")
            or module_name == "lmcache_ascend"
            or module_name.startswith("lmcache_ascend.")
        ):
            continue
        namespace = vars(module)
        for name in names:
            value = namespace.get(name)
            if any(value is original for original in originals):
                setattr(module, name, _local_merged_layout_enabled)
                patched.append(f"{module_name}.{name}")
    return {
        "layout": "merged",
        "selection": "correctness-tool-only predicate override",
        "patched_aliases": sorted(patched),
        "production_transfer_unchanged": True,
        "native_mooncake": False,
    }


def validate_local_merged_engine(engine: Any) -> dict[str, Any]:
    """Validate a worker engine and return observable layout/backend facts.

    Rank 0 must have only the real LocalCPUBackend with page support. Passive
    ranks may have no manager, but must have an attached shared allocator.
    Page counts describe objects observed now, not a promise that a future
    request will exercise them; callers must check runtime coverage separately.
    Raises RuntimeError if the tool layout or backend isolation is not active.
    """
    config = engine.config
    _require_local_merged_config(config)
    layout = importlib.import_module("lmcache.v1.mooncake_layout")
    if any(
        getattr(layout, name) is not _local_merged_layout_enabled
        for name in ("mooncake_page_layout_enabled", "mooncake_layer_pages_enabled")
    ):
        raise RuntimeError("Local merged correctness layout was not installed")
    if not getattr(engine, "enable_shared_cpu_cache", False):
        raise RuntimeError("Correctness engine did not enable shared CPU cache")
    metadata = engine.metadata
    rank = getattr(metadata, "worker_id", None)
    first_rank = getattr(metadata, "first_rank", None)
    if rank is None or first_rank is None:
        raise RuntimeError("Correctness engine has no explicit worker rank")
    passive = rank != first_rank
    if not getattr(getattr(engine, "token_database", None), "mooncake_payload_layout", None):
        raise RuntimeError("Token database was initialized without the merged page key layout")
    manager = getattr(engine, "storage_manager", None)
    merged_objects = legacy_objects = 0
    if passive and manager is None:
        if getattr(engine, "shared_cpu_cache_passive_allocator", None) is None:
            raise RuntimeError("Passive correctness rank has no attached shared allocator")
        backend_names = []
    else:
        backends = getattr(manager, "storage_backends", {})
        if set(backends) != {"LocalCPUBackend"}:
            raise RuntimeError(f"Correctness engine requires LocalCPUBackend only, got {sorted(backends)}")
        backend_type = importlib.import_module("lmcache.v1.storage_backend.local_cpu_backend").LocalCPUBackend
        local = backends["LocalCPUBackend"]
        if not isinstance(local, backend_type):
            raise RuntimeError("Correctness engine LocalCPUBackend was replaced")
        if not manager.supports_batched_put_layer_pages(getattr(engine, "store_location", None)):
            raise RuntimeError("Correctness LocalCPUBackend cannot publish merged pages")
        page_type = importlib.import_module("lmcache.v1.memory_management").LayerPageMemoryObj
        with local.cpu_lock:
            for obj in local.hot_cache.values():
                if isinstance(obj, page_type):
                    merged_objects += 1
                else:
                    legacy_objects += 1
        if legacy_objects:
            raise RuntimeError(f"Correctness merged cache contains {legacy_objects} legacy objects")
        backend_names = sorted(backends)
    return {
        "layout": "merged",
        "selection": "correctness-tool-only predicate override",
        "worker_id": rank,
        "passive": passive,
        "backend_names": backend_names,
        "remote_url": None,
        "native_mooncake": False,
        "page_key_layout": engine.token_database.mooncake_payload_layout,
        "merged_objects": merged_objects,
        "legacy_objects": legacy_objects,
        "merged_objects_observed": merged_objects > 0,
        "production_transfer_unchanged": True,
    }
