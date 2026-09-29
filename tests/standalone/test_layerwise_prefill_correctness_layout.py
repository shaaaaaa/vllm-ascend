# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the opt-in correctness-only physical layout selection."""

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def tool():
    path = Path(__file__).resolve().parents[2] / "tools/layerwise_prefill_correctness_layout.py"
    spec = importlib.util.spec_from_file_location("correctness_layout_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(**changes):
    values = dict(
        remote_url=None,
        enable_remote_lmcache_store=False,
        use_layerwise=True,
        local_cpu=True,
        max_local_cpu_size=24,
        enable_shared_cpu_cache=True,
        extra_config={
            "save_only_first_rank": True,
            "mooncake_page_first_multi_buffer": True,
            "mooncake_layer_merged_page_objects": True,
        },
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture
def modules(monkeypatch):
    def page(config):
        return bool(config.extra_config.get("mooncake_page_first_multi_buffer"))

    def merged(config):
        return bool(config.remote_url and config.remote_url.startswith("mooncakestore://"))

    names = (
        "lmcache.v1.mooncake_layout",
        "lmcache.v1.cache_engine",
        "lmcache.v1.token_database",
        "lmcache_ascend.v1.cache_engine",
    )
    result = {}
    for name in names:
        module = ModuleType(name)
        module.mooncake_page_layout_enabled = page
        module.mooncake_layer_pages_enabled = merged
        result[name] = module
        monkeypatch.setitem(sys.modules, name, module)

    class LocalCPUBackend:
        def __init__(self):
            self.cpu_lock = threading.Lock()
            self.hot_cache = {}

    class LayerPageMemoryObj:
        pass

    backend = ModuleType("lmcache.v1.storage_backend.local_cpu_backend")
    backend.LocalCPUBackend = LocalCPUBackend
    memory = ModuleType("lmcache.v1.memory_management")
    memory.LayerPageMemoryObj = LayerPageMemoryObj
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(sys.modules, memory.__name__, memory)
    result.update(backend=backend, memory=memory, original_page=page, original_merged=merged)
    return result


def engine(modules, *, rank=0):
    local = modules["backend"].LocalCPUBackend()
    return SimpleNamespace(
        config=config(),
        enable_shared_cpu_cache=True,
        metadata=SimpleNamespace(worker_id=rank, first_rank=0),
        token_database=SimpleNamespace(mooncake_payload_layout="real-layout-signature"),
        store_location=None,
        storage_manager=SimpleNamespace(
            storage_backends={"LocalCPUBackend": local},
            supports_batched_put_layer_pages=lambda location: True,
        ),
    )


def test_import_does_not_install_or_load_runtime(tool, modules):
    layout = modules["lmcache.v1.mooncake_layout"]
    assert layout.mooncake_layer_pages_enabled is modules["original_merged"]
    assert not layout.mooncake_layer_pages_enabled(config())


def test_install_patches_existing_aliases_and_future_imports_idempotently(tool, modules):
    old_page = modules["original_page"]
    facts = tool.install_local_merged_layout()
    assert facts["production_transfer_unchanged"]
    assert facts["layout"] == "merged"
    assert len(facts["patched_aliases"]) == 8
    for name in ("lmcache.v1.cache_engine", "lmcache_ascend.v1.cache_engine", "lmcache.v1.token_database"):
        assert modules[name].mooncake_page_layout_enabled(config())
        assert modules[name].mooncake_layer_pages_enabled(config())
        assert modules[name].mooncake_page_layout_enabled is not old_page
    before = modules["lmcache.v1.mooncake_layout"].mooncake_layer_pages_enabled
    tool.install_local_merged_layout()
    assert modules["lmcache.v1.mooncake_layout"].mooncake_layer_pages_enabled is before


@pytest.mark.parametrize(
    "changes",
    [
        {"remote_url": "mooncakestore://unused:1234"},
        {"remote_url": ""},
        {"enable_remote_lmcache_store": True},
        {"enable_pd": True},
        {"enable_p2p": True},
        {"local_disk": "/cache"},
        {"max_local_cpu_size": 0},
        {"local_cpu": False},
        {"use_layerwise": False},
        {"enable_shared_cpu_cache": False},
        {"storage_plugins": ["other"]},
        {"extra_config": {"save_only_first_rank": True}},
    ],
)
def test_layout_predicate_fails_closed_before_nonlocal_backend_creation(tool, modules, changes):
    tool.install_local_merged_layout()
    with pytest.raises(RuntimeError, match="correctness layout requires"):
        modules["lmcache.v1.mooncake_layout"].mooncake_layer_pages_enabled(config(**changes))


def test_rank_zero_requires_real_local_backend_and_reports_observed_pages(tool, modules):
    tool.install_local_merged_layout()
    instance = engine(modules)
    initial = tool.validate_local_merged_engine(instance)
    assert not initial["merged_objects_observed"]
    local = instance.storage_manager.storage_backends["LocalCPUBackend"]
    local.hot_cache["chunk"] = modules["memory"].LayerPageMemoryObj()
    facts = tool.validate_local_merged_engine(instance)
    assert facts["merged_objects"] == 1
    assert facts["backend_names"] == ["LocalCPUBackend"]
    assert facts["remote_url"] is None and not facts["native_mooncake"]


@pytest.mark.parametrize("failure", ["remote", "fake_local", "legacy", "capability", "keys", "uninstalled"])
def test_engine_validation_rejects_unproven_layout(tool, modules, failure):
    tool.install_local_merged_layout()
    instance = engine(modules)
    backends = instance.storage_manager.storage_backends
    if failure == "remote":
        backends["RemoteBackend"] = object()
    elif failure == "fake_local":
        backends["LocalCPUBackend"] = SimpleNamespace()
    elif failure == "legacy":
        backends["LocalCPUBackend"].hot_cache["chunk"] = object()
    elif failure == "capability":
        instance.storage_manager.supports_batched_put_layer_pages = lambda location: False
    elif failure == "keys":
        instance.token_database.mooncake_payload_layout = None
    else:
        modules["lmcache.v1.mooncake_layout"].mooncake_page_layout_enabled = modules["original_page"]
    with pytest.raises(RuntimeError):
        tool.validate_local_merged_engine(instance)


def test_passive_rank_requires_attachment_without_claiming_page_observation(tool, modules):
    tool.install_local_merged_layout()
    instance = engine(modules, rank=1)
    instance.storage_manager = None
    with pytest.raises(RuntimeError, match="attached shared allocator"):
        tool.validate_local_merged_engine(instance)
    instance.shared_cpu_cache_passive_allocator = object()
    facts = tool.validate_local_merged_engine(instance)
    assert facts["passive"] and not facts["backend_names"]
    assert not facts["merged_objects_observed"]
