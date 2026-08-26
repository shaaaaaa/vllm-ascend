import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

fake_mooncake = types.ModuleType("mooncake")
fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
fake_mooncake.engine = fake_engine
sys.modules.setdefault("mooncake", fake_mooncake)
sys.modules.setdefault("mooncake.engine", fake_engine)

fake_torch_npu = types.ModuleType("torch_npu")
fake_torch_npu.atb = SimpleNamespace(npu_paged_cache_load=MagicMock())
sys.modules.setdefault("torch_npu", fake_torch_npu)

from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (  # noqa: E402
    MultiKVConnectorMetadata,
    MultiKVConnectorWorkerMetadata,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks  # noqa: E402

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import (  # noqa: E402
    AscendMultiConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (  # noqa: E402
    MooncakeConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_index_connector import (  # noqa: E402
    MooncakeDSAIndexConnector,
)


class _Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.block_hash = None
        self.is_null = False


def _blocks() -> KVCacheBlocks:
    return KVCacheBlocks(((_Block(10), _Block(11)), (_Block(20), _Block(21))))


def _remote_index_params(remote_block_ids=None):
    return {
        "do_remote_prefill": True,
        "remote_block_ids": [20, 21] if remote_block_ids is None else remote_block_ids,
        "remote_engine_id": "engine-0",
        "remote_host": "127.0.0.1",
        "remote_port": 29000,
        "remote_request_id": "remote-req-1",
    }


def test_lmcache_ascend_connector_advertises_dsa_index_support():
    pytest.importorskip("lmcache_ascend")

    from vllm_ascend.distributed.kv_transfer.kv_pool.lmcache_ascend_connector import (
        LMCacheConnectorV1,
    )

    assert getattr(LMCacheConnectorV1, "supports_dsa_index_lmcache", False) is True


def test_ascend_multi_delegates_dsa_index_lmcache_capability():
    multi = object.__new__(AscendMultiConnector)
    assert multi.supports_dsa_index_lmcache is False

    children = [
        SimpleNamespace(supports_dsa_index_lmcache=False),
        SimpleNamespace(supports_dsa_index_lmcache=True),
    ]
    multi._supports_dsa_index_lmcache = any(
        multi._supports_dsa_index_cache(child) for child in children
    )

    assert multi.supports_dsa_index_lmcache is True

    children = [
        SimpleNamespace(supports_dsa_index_lmcache=False),
        SimpleNamespace(),
    ]
    multi._supports_dsa_index_lmcache = any(
        multi._supports_dsa_index_cache(child) for child in children
    )

    assert multi.supports_dsa_index_lmcache is False


def test_ascend_multi_routes_same_step_worker_metadata():
    update = MagicMock()
    multi = object.__new__(AscendMultiConnector)
    multi._connectors = [
        SimpleNamespace(update_connector_worker_metadata=update),
        SimpleNamespace(),
    ]
    metadata = object()

    multi.update_connector_worker_metadata(
        MultiKVConnectorWorkerMetadata(metadata=(metadata, None)),
        {"active"},
    )

    update.assert_called_once_with(metadata, {"active"})


def test_ascend_multi_forwards_live_source_event_to_lmcache_engine():
    forward_context = object()
    capture = MagicMock(return_value=True)
    multi = object.__new__(AscendMultiConnector)
    multi._connectors = [
        SimpleNamespace(),
        SimpleNamespace(
            _lmcache_engine=SimpleNamespace(
                capture_live_source_event_handoff=capture
            )
        ),
    ]

    assert multi.capture_live_source_event_handoff(forward_context)
    capture.assert_called_once_with(forward_context)


def test_ascend_multi_init_supports_legacy_child_connector_signature(monkeypatch):
    calls = []

    class LegacyConnector:
        def __init__(self, vllm_config, role):
            calls.append(("legacy", vllm_config, role, None))

    class NewConnector:
        def __init__(self, vllm_config, role, kv_cache_config=None):
            calls.append(("new", vllm_config, role, kv_cache_config))

    top_config = SimpleNamespace(kv_transfer_config=object())
    legacy_config = SimpleNamespace(kv_transfer_config="legacy-ktc")
    new_config = SimpleNamespace(kv_transfer_config="new-ktc")
    role = object()
    kv_cache_config = object()

    monkeypatch.setattr(
        AscendMultiConnector,
        "_get_connector_classes_and_configs",
        classmethod(
            lambda cls, config: [
                (LegacyConnector, legacy_config),
                (NewConnector, new_config),
            ]
        ),
    )

    multi = AscendMultiConnector(top_config, role, kv_cache_config)

    assert [type(connector) for connector in multi._connectors] == [
        LegacyConnector,
        NewConnector,
    ]
    assert multi._ktc_kv_transfer_config == ["legacy-ktc", "new-ktc"]
    assert multi._requests_to_connector == {}
    assert multi._extra_async_saves == {}
    assert multi._index_load_async_req_ids == set()
    assert calls == [
        ("legacy", legacy_config, role, None),
        ("new", new_config, role, kv_cache_config),
    ]


def test_ascend_multi_forwards_remote_fill_child_contract(monkeypatch):
    placement = {"enabled": True, "destination_engine_epoch": 7}
    metrics = {"transactions_started": 3}

    class LMCacheChild:
        def __init__(self, _config, _role):
            self.fatal = False

        def get_remote_fill_placement_info(self):
            return placement

        def remote_fill_requires_paired_restart(self):
            return self.fatal

        def get_remote_fill_metrics(self):
            return metrics

    class MooncakeChild:
        def __init__(self, _config, _role):
            pass

    top_config = SimpleNamespace(kv_transfer_config=object())
    child_configs = [
        SimpleNamespace(kv_transfer_config="lmcache"),
        SimpleNamespace(kv_transfer_config="mooncake"),
    ]
    monkeypatch.setattr(
        AscendMultiConnector,
        "_get_connector_classes_and_configs",
        classmethod(
            lambda cls, config: list(
                zip((LMCacheChild, MooncakeChild), child_configs, strict=True)
            )
        ),
    )

    multi = AscendMultiConnector(top_config, object())

    assert multi.get_remote_fill_placement_info() == placement
    assert multi.get_remote_fill_metrics() == metrics
    assert multi.remote_fill_requires_paired_restart() is False
    multi._connectors[0].fatal = True
    assert multi.remote_fill_requires_paired_restart() is True


def test_ascend_multi_forwards_real_lmcache_ascend_wrapper(monkeypatch):
    pytest.importorskip("lmcache_ascend")
    from lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1 import (
        LMCacheAscendConnectorV1Dynamic,
    )

    placement = {"enabled": True, "destination_engine_epoch": 17}
    engine = SimpleNamespace(
        use_layerwise=True,
        get_remote_fill_placement_info=lambda: placement,
        get_remote_fill_metrics=lambda: {"active_transactions": 1},
        remote_fill_requires_paired_restart=lambda: True,
    )

    def init_lmcache(self, _config, _role):
        self._lmcache_engine = SimpleNamespace(lmcache_engine=engine)

    class MooncakeChild:
        def __init__(self, _config, _role):
            pass

    monkeypatch.setattr(
        LMCacheAscendConnectorV1Dynamic,
        "__init__",
        init_lmcache,
    )
    monkeypatch.setattr(
        AscendMultiConnector,
        "_get_connector_classes_and_configs",
        classmethod(
            lambda cls, config: [
                (
                    LMCacheAscendConnectorV1Dynamic,
                    SimpleNamespace(kv_transfer_config="lmcache"),
                ),
                (
                    MooncakeChild,
                    SimpleNamespace(kv_transfer_config="mooncake"),
                ),
            ]
        ),
    )

    multi = AscendMultiConnector(
        SimpleNamespace(kv_transfer_config=object()), object()
    )

    assert multi.get_remote_fill_placement_info() == placement
    assert multi.get_remote_fill_metrics() == {"active_transactions": 1}
    assert multi.remote_fill_requires_paired_restart() is True


def test_ascend_multi_rejects_conflicting_remote_fill_owners():
    multi = object.__new__(AscendMultiConnector)
    multi._remote_fill_placement_providers = (
        lambda: {"destination_engine_epoch": 1},
        lambda: {"destination_engine_epoch": 2},
    )
    multi._remote_fill_metrics_providers = (lambda: {"started": 1},) * 2
    multi._remote_fill_restart_providers = (lambda: False,)

    with pytest.raises(RuntimeError, match="conflicting remote-fill placement"):
        multi.get_remote_fill_placement_info()
    assert multi.get_remote_fill_metrics() == {"started": 1}


def test_ascend_multi_fails_closed_when_restart_probe_raises():
    multi = object.__new__(AscendMultiConnector)

    def broken_probe():
        raise RuntimeError("child unavailable")

    multi._remote_fill_restart_providers = (broken_probe,)

    assert multi.remote_fill_requires_paired_restart() is True


def test_live_latent_requires_capable_provider_and_hybrid_consumer():
    class Provider:
        supports_dsa_live_latent_split_source = True

        def __init__(self):
            self.decisions = []

        def configure_live_latent_source(self, enabled):
            self.decisions.append(enabled)

    provider = Provider()
    multi = object.__new__(AscendMultiConnector)
    multi._connectors = [provider]
    multi._configure_live_latent_split()
    assert provider.decisions == [False]

    consumer = object.__new__(MooncakeDSAIndexConnector)
    consumer._latent_live_enabled = False
    consumer.connector_scheduler = None
    consumer.connector_worker = SimpleNamespace(kv_role="kv_consumer")
    multi._connectors = [provider, consumer]
    multi._configure_live_latent_split()

    assert provider.decisions == [False, True]
    assert consumer._latent_live_enabled is True


def test_live_latent_accepts_capability_based_transport_wrapper():
    class Provider:
        supports_dsa_live_latent_split_source = True

        def __init__(self):
            self.decisions = []

        def configure_live_latent_source(self, enabled):
            self.decisions.append(enabled)

    class WrappedTransport:
        supports_dsa_live_latent_transport = True

        def __init__(self):
            self.decisions = []

        def configure_live_latent_transport(
            self, source_enabled, destination_enabled
        ):
            self.decisions.append((source_enabled, destination_enabled))

    provider = Provider()
    transport = WrappedTransport()
    multi = object.__new__(AscendMultiConnector)
    multi._connectors = [provider, transport]

    multi._configure_live_latent_split()

    assert provider.decisions == [True]
    assert transport.decisions == [(True, False)]


def test_live_latent_old_or_unconfigurable_provider_fails_closed():
    class OldProvider:
        supports_dsa_live_latent_split_source = True

    consumer = object.__new__(MooncakeDSAIndexConnector)
    consumer._latent_live_enabled = True
    consumer.connector_scheduler = None
    consumer.connector_worker = None
    multi = object.__new__(AscendMultiConnector)
    multi._connectors = [OldProvider(), consumer]

    multi._configure_live_latent_split()

    assert consumer._latent_live_enabled is False


def test_dsa_index_connector_supports_hma_and_selects_index_group():
    connector = object.__new__(MooncakeDSAIndexConnector)
    connector.index_group_id = 1
    connector.connector_scheduler = MagicMock()
    connector.connector_scheduler.request_finished.return_value = (
        True,
        {"remote_block_ids": [20, 21]},
    )

    request = SimpleNamespace(request_id="req-1")
    async_save, params = connector.request_finished_all_groups(
        request,
        ([10, 11], [20, 21]),
    )

    assert isinstance(connector, SupportsHMA)
    assert async_save is True
    assert params == {"remote_block_ids": [20, 21]}
    connector.connector_scheduler.request_finished.assert_called_once_with(
        request,
        [20, 21],
    )


def test_dsa_index_update_state_uses_index_group_and_preserves_prefill_flag():
    connector = object.__new__(MooncakeDSAIndexConnector)
    connector.index_group_id = 1
    connector.connector_scheduler = MagicMock()

    def _mutate_prefill_flag(request, _blocks, _tokens):
        request.kv_transfer_params["do_remote_prefill"] = False

    connector.connector_scheduler.update_state_after_alloc.side_effect = (
        _mutate_prefill_flag
    )
    request = SimpleNamespace(
        request_id="req-1",
        kv_transfer_params={"do_remote_prefill": True},
    )

    connector.update_state_after_alloc(request, _blocks(), 32)

    assert request.kv_transfer_params["do_remote_prefill"] is True
    passed_blocks = (
        connector.connector_scheduler.update_state_after_alloc.call_args.args[1]
    )
    assert passed_blocks.get_unhashed_block_ids() == [20, 21]


def test_dsa_index_update_state_skips_without_external_tokens():
    connector = object.__new__(MooncakeDSAIndexConnector)
    connector.index_group_id = 1
    connector.connector_scheduler = MagicMock()
    request = SimpleNamespace(
        request_id="req-1",
        kv_transfer_params={"do_remote_prefill": True},
    )

    connector.update_state_after_alloc(request, _blocks(), 0)

    connector.connector_scheduler.update_state_after_alloc.assert_not_called()


def test_dsa_index_registers_only_indexer_layers():
    connector = object.__new__(MooncakeDSAIndexConnector)
    connector.index_group_id = 1
    connector.connector_worker = MagicMock()
    latent = (object(), object())
    indexer = (object(),)

    connector.register_kv_caches(
        {
            "model.layers.0.self_attn": latent,
            "model.layers.0.self_attn.indexer": indexer,
        }
    )

    connector.connector_worker.register_kv_caches.assert_called_once_with(
        {"model.layers.0.self_attn.indexer": indexer}
    )


def test_dsa_index_live_latent_registers_full_source_only_on_tp0():
    connector = object.__new__(MooncakeDSAIndexConnector)
    connector.index_group_id = 1
    connector._latent_live_enabled = True
    connector._latent_live_source_enabled = True
    connector._latent_live_destination_enabled = False
    connector._dsa_role = KVConnectorRole.WORKER
    connector.connector_scheduler = None
    connector.connector_worker = MagicMock(kv_role="kv_producer", tp_rank=0)
    latent = (object(), object())
    indexer = (object(),)
    caches = {"layer": latent, "layer.indexer": indexer}

    connector.register_kv_caches(caches)

    connector.connector_worker.register_kv_caches.assert_called_once_with(
        caches, ordinary_kv_caches={"layer.indexer": indexer}
    )


def test_dsa_index_live_latent_registers_full_source_for_kv_both_tp0():
    connector = object.__new__(MooncakeDSAIndexConnector)
    connector.index_group_id = 1
    connector._latent_live_enabled = True
    connector._latent_live_source_enabled = True
    connector._latent_live_destination_enabled = True
    connector._dsa_role = KVConnectorRole.WORKER
    connector.connector_scheduler = None
    connector.connector_worker = MagicMock(kv_role="kv_both", tp_rank=0)
    latent = (object(), object())
    indexer = (object(),)
    caches = {"layer": latent, "layer.indexer": indexer}

    assert connector._live_split_source_groups() == (0, 1)
    connector.register_kv_caches(caches)

    connector.connector_worker.register_kv_caches.assert_called_once_with(
        caches, ordinary_kv_caches={"layer.indexer": indexer}
    )


def test_dsa_index_live_latent_negotiation_supports_kv_both_worker():
    connector = object.__new__(MooncakeDSAIndexConnector)
    connector.index_group_id = 1
    connector._latent_live_enabled = False
    connector.connector_scheduler = MagicMock()
    connector.connector_worker = MagicMock(kv_role="kv_both", tp_rank=0)

    connector.configure_live_latent_source(True)

    assert connector._latent_live_enabled is True
    assert connector.connector_scheduler.live_split_source_groups == (0, 1)
    assert connector.connector_worker.live_latent_enabled is True


def test_ascend_multi_enables_provider_for_kv_both_transport():
    class Provider:
        supports_dsa_live_latent_split_source = True

        def __init__(self):
            self.decisions = []

        def configure_live_latent_source(self, enabled):
            self.decisions.append(enabled)

    provider = Provider()
    consumer = object.__new__(MooncakeDSAIndexConnector)
    consumer._latent_live_enabled = False
    consumer.connector_scheduler = None
    consumer.connector_worker = SimpleNamespace(kv_role="kv_both")
    multi = object.__new__(AscendMultiConnector)
    multi._connectors = [provider, consumer]

    multi._configure_live_latent_split()

    assert provider.decisions == [True]
    assert consumer._latent_live_enabled is True


def test_ascend_multi_enables_generic_mooncake_hybrid_transport():
    class Provider:
        supports_dsa_live_latent_split_source = True

        def __init__(self):
            self.decisions = []

        def configure_live_latent_source(self, enabled):
            self.decisions.append(enabled)

    provider = Provider()
    transport = object.__new__(MooncakeConnector)
    transport._latent_live_enabled = False
    transport._latent_live_source_enabled = False
    transport._latent_live_destination_enabled = False
    transport.connector_scheduler = None
    transport.connector_worker = SimpleNamespace(
        kv_role="kv_producer", tp_rank=0
    )
    multi = object.__new__(AscendMultiConnector)
    multi._connectors = [provider, transport]

    multi._configure_live_latent_split()

    assert provider.decisions == [True]
    assert transport._latent_live_enabled is True
    assert transport._latent_live_source_enabled is True
    assert transport._latent_live_destination_enabled is False
    assert transport.connector_worker.live_latent_source_enabled is True


def test_generic_mooncake_live_latent_decoder_groups_are_rank_aware():
    connector = object.__new__(MooncakeConnector)
    connector._latent_live_enabled = False
    connector._latent_live_source_enabled = False
    connector._latent_live_destination_enabled = False
    connector._dsa_role = KVConnectorRole.WORKER
    connector.connector_scheduler = None
    connector.connector_worker = SimpleNamespace(
        kv_role="kv_consumer", tp_rank=0
    )

    connector.configure_live_latent_transport(False, True)

    assert connector._live_split_source_groups() == (0, 1)
    connector.connector_worker.tp_rank = 1
    assert connector._live_split_source_groups() == (1,)


def test_ascend_multi_separates_source_and_destination_transport():
    class Provider:
        supports_dsa_live_latent_split_source = False
        supports_dsa_live_latent_split_destination = True

        def __init__(self):
            self.decisions = []

        def configure_live_latent_source(self, enabled):
            self.decisions.append(enabled)

    provider = Provider()
    transport = object.__new__(MooncakeDSAIndexConnector)
    transport._latent_live_enabled = False
    transport._latent_live_source_enabled = False
    transport._latent_live_destination_enabled = False
    transport.connector_scheduler = None
    transport.connector_worker = SimpleNamespace(kv_role="kv_both")
    multi = object.__new__(AscendMultiConnector)
    multi._connectors = [provider, transport]

    multi._configure_live_latent_split()

    assert provider.decisions == [True]
    assert transport._latent_live_source_enabled is False
    assert transport._latent_live_destination_enabled is True


def test_dsa_index_live_latent_decoder_groups_are_rank_aware():
    connector = object.__new__(MooncakeDSAIndexConnector)
    connector.index_group_id = 1
    connector._latent_live_enabled = True
    connector._latent_live_source_enabled = False
    connector._latent_live_destination_enabled = True
    connector._dsa_role = KVConnectorRole.WORKER
    connector.connector_scheduler = None
    connector.connector_worker = SimpleNamespace(kv_role="kv_consumer", tp_rank=0)
    assert connector._live_split_source_groups() == (0, 1)

    connector.connector_worker.tp_rank = 1
    assert connector._live_split_source_groups() == (1,)


def test_ascend_multi_registers_latent_and_indexer_separately():
    multi = object.__new__(AscendMultiConnector)
    latent_connector = MagicMock()
    index_connector = object.__new__(MooncakeDSAIndexConnector)
    index_connector.register_kv_caches = MagicMock()
    multi._connectors = [latent_connector, index_connector]

    latent = (object(), object())
    indexer = (object(),)
    kv_caches = {
        "model.layers.0.self_attn": latent,
        "model.layers.0.self_attn.indexer": indexer,
    }

    multi.register_kv_caches(kv_caches)

    latent_connector.register_kv_caches.assert_called_once_with(
        {"model.layers.0.self_attn": latent}
    )
    index_connector.register_kv_caches.assert_called_once_with(kv_caches)


def test_ascend_multi_registers_all_groups_with_hma_child():
    class HMAConnector(SupportsHMA):
        def __init__(self):
            self.register_kv_caches = MagicMock()

        def request_finished_all_groups(self, request, block_ids):
            raise NotImplementedError

    multi = object.__new__(AscendMultiConnector)
    hma_connector = HMAConnector()
    index_connector = object.__new__(MooncakeDSAIndexConnector)
    index_connector.register_kv_caches = MagicMock()
    multi._connectors = [hma_connector, index_connector]

    latent = (object(), object())
    indexer = (object(),)
    kv_caches = {
        "model.layers.0.self_attn": latent,
        "model.layers.0.self_attn.indexer": indexer,
    }

    multi.register_kv_caches(kv_caches)

    hma_connector.register_kv_caches.assert_called_once_with(kv_caches)
    index_connector.register_kv_caches.assert_called_once_with(kv_caches)


def test_ascend_multi_keeps_generic_mooncake_on_latent_group():
    multi = object.__new__(AscendMultiConnector)
    mooncake = object.__new__(MooncakeConnector)
    mooncake.register_kv_caches = MagicMock()
    multi._connectors = [mooncake]
    kv_caches = {
        "model.layers.0.self_attn": (object(), object()),
        "model.layers.0.self_attn.indexer": (object(),),
    }

    multi.register_kv_caches(kv_caches)

    mooncake.register_kv_caches.assert_called_once_with(
        {"model.layers.0.self_attn": kv_caches["model.layers.0.self_attn"]}
    )


def test_ascend_multi_forwards_staged_sfa_capabilities():
    multi = object.__new__(AscendMultiConnector)
    multi._connectors = [
        SimpleNamespace(
            supports_staged_sfa_sparse_load=False,
            uses_layerwise_model_callbacks=False,
        ),
        SimpleNamespace(
            supports_staged_sfa_sparse_load=True,
            uses_layerwise_model_callbacks=True,
            wait_for_layer_load=lambda _layer_name: None,
            _get_connector_metadata=lambda: object(),
        ),
    ]

    assert multi.uses_layerwise_model_callbacks
    assert multi.supports_staged_sfa_sparse_load


def test_ascend_multi_rejects_split_staged_sfa_capabilities():
    multi = object.__new__(AscendMultiConnector)
    multi._connectors = [
        SimpleNamespace(
            supports_staged_sfa_sparse_load=True,
            uses_layerwise_model_callbacks=False,
            wait_for_layer_load=lambda _layer_name: None,
            _get_connector_metadata=lambda: object(),
        ),
        SimpleNamespace(
            supports_staged_sfa_sparse_load=False,
            uses_layerwise_model_callbacks=True,
            wait_for_layer_load=lambda _layer_name: None,
        ),
    ]

    assert multi.uses_layerwise_model_callbacks
    assert not multi.supports_staged_sfa_sparse_load


def test_ascend_multi_unwraps_scheduler_metadata_from_capable_child():
    target_metadata = object()
    multi = object.__new__(AscendMultiConnector)
    multi._connectors = [
        SimpleNamespace(_get_connector_metadata=lambda: object()),
        SimpleNamespace(
            supports_staged_sfa_sparse_load=True,
            uses_layerwise_model_callbacks=True,
            wait_for_layer_load=lambda _layer_name: None,
            _get_connector_metadata=lambda: object(),
        ),
    ]
    metadata = MultiKVConnectorMetadata(
        metadata=(object(), target_metadata)
    )

    assert (
        multi._unwrap_staged_sfa_connector_metadata(metadata)
        is target_metadata
    )


def test_ascend_multi_wait_for_layer_load_forwards_supported_extra_args():
    multi = object.__new__(AscendMultiConnector)
    calls = []

    class LegacyConnector:
        def wait_for_layer_load(self, layer_name):
            calls.append(("legacy", layer_name))

    class SparseLatentConnector:
        def wait_for_layer_load(
            self,
            layer_name,
            selected_tokens,
            token_start_index,
            request_ids,
        ):
            calls.append(
                (
                    "sparse",
                    layer_name,
                    selected_tokens,
                    token_start_index,
                    request_ids,
                )
            )

    multi._connectors = [LegacyConnector(), SparseLatentConnector()]

    selected_tokens = object()
    request_ids = ["req-1"]
    multi.wait_for_layer_load(
        "model.layers.0.self_attn",
        selected_tokens,
        7,
        request_ids,
    )

    assert calls == [
        ("legacy", "model.layers.0.self_attn"),
        ("sparse", "model.layers.0.self_attn", selected_tokens, 7, request_ids),
    ]


def test_ascend_multi_waits_for_async_index_load_when_latent_hits():
    multi = object.__new__(AscendMultiConnector)
    latent_connector = MagicMock()
    latent_connector.get_num_new_matched_tokens.return_value = (32, False)
    index_connector = object.__new__(MooncakeDSAIndexConnector)
    multi._connectors = [latent_connector, index_connector]
    multi._requests_to_connector = {}

    request = SimpleNamespace(
        request_id="req-1",
        kv_transfer_params=_remote_index_params(),
    )

    tokens, load_async = multi.get_num_new_matched_tokens(request, 0)

    assert tokens == 32
    assert load_async is True
    assert multi._requests_to_connector == {"req-1": 0}
    assert multi._index_load_async_req_ids == {"req-1"}


def test_ascend_multi_keeps_sync_without_remote_index_blocks():
    multi = object.__new__(AscendMultiConnector)
    latent_connector = MagicMock()
    latent_connector.get_num_new_matched_tokens.return_value = (32, False)
    index_connector = object.__new__(MooncakeDSAIndexConnector)
    multi._connectors = [latent_connector, index_connector]
    multi._requests_to_connector = {}

    request = SimpleNamespace(
        request_id="req-1",
        kv_transfer_params=_remote_index_params(remote_block_ids=[]),
    )

    tokens, load_async = multi.get_num_new_matched_tokens(request, 0)

    assert tokens == 32
    assert load_async is False


def test_ascend_multi_preserves_async_from_chosen_connector():
    multi = object.__new__(AscendMultiConnector)
    latent_connector = MagicMock()
    latent_connector.get_num_new_matched_tokens.return_value = (32, True)
    index_connector = object.__new__(MooncakeDSAIndexConnector)
    multi._connectors = [latent_connector, index_connector]
    multi._requests_to_connector = {}

    request = SimpleNamespace(
        request_id="req-1",
        kv_transfer_params=_remote_index_params(),
    )

    tokens, load_async = multi.get_num_new_matched_tokens(request, 0)

    assert tokens == 32
    assert load_async is True


def test_ascend_multi_skips_latent_zero_update_after_async_index_wait():
    multi = object.__new__(AscendMultiConnector)
    latent_connector = MagicMock()
    index_connector = object.__new__(MooncakeDSAIndexConnector)
    index_connector.update_state_after_alloc = MagicMock()
    multi._connectors = [latent_connector, index_connector]
    multi._requests_to_connector = {"req-1": 0}
    multi._index_load_async_req_ids = {"req-1"}

    request = SimpleNamespace(
        request_id="req-1",
        num_computed_tokens=32,
        kv_transfer_params=_remote_index_params(),
    )

    multi.update_state_after_alloc(request, _blocks(), 0)

    latent_connector.update_state_after_alloc.assert_not_called()
    index_connector.update_state_after_alloc.assert_called_once()
    assert multi._index_load_async_req_ids == set()


def test_ascend_multi_updates_chosen_latent_and_index_connector():
    multi = object.__new__(AscendMultiConnector)
    latent_connector = MagicMock()
    index_connector = object.__new__(MooncakeDSAIndexConnector)
    index_connector.update_state_after_alloc = MagicMock()
    multi._connectors = [latent_connector, index_connector]
    multi._requests_to_connector = {"req-1": 0}

    request = SimpleNamespace(request_id="req-1")
    blocks = _blocks()
    multi.update_state_after_alloc(request, blocks, 32)

    latent_blocks = latent_connector.update_state_after_alloc.call_args.args[1]
    index_blocks = index_connector.update_state_after_alloc.call_args.args[1]

    assert latent_blocks.get_unhashed_block_ids() == [10, 11]
    assert index_blocks is blocks
    index_connector.update_state_after_alloc.assert_called_once_with(
        request,
        blocks,
        32,
    )


def test_ascend_multi_request_finished_all_groups_merges_params():
    multi = object.__new__(AscendMultiConnector)
    latent_connector = MagicMock()
    latent_connector.request_finished.return_value = (False, {"first_tok": 7})
    index_connector = object.__new__(MooncakeDSAIndexConnector)
    index_connector.request_finished_all_groups = MagicMock(
        return_value=(True, {"do_remote_prefill": True, "remote_block_ids": [20]})
    )
    multi._connectors = [latent_connector, index_connector]
    multi._requests_to_connector = {"req-1": 0}
    multi._extra_async_saves = {}
    request = SimpleNamespace(request_id="req-1")

    async_save, params = multi.request_finished_all_groups(request, ([10], [20]))

    assert async_save is True
    assert params == {
        "first_tok": 7,
        "do_remote_prefill": True,
        "remote_block_ids": [20],
    }
    latent_connector.request_finished.assert_called_once_with(request, [10])
    index_connector.request_finished_all_groups.assert_called_once_with(
        request,
        ([10], [20]),
    )


def test_ascend_multi_remote_fill_echo_yields_to_sibling_envelope():
    """Regression: the LMCache remote-fill response echoes the prefiller's
    incoming routing keys, which must not clash with the decoder-directed
    envelope authored by the Mooncake P2P producer child."""

    multi = object.__new__(AscendMultiConnector)
    lmcache_connector = MagicMock()
    lmcache_connector.request_finished.return_value = (
        False,
        {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
            "lmcache.remote_fill": {"terminal": {"outcome": "PERSISTENT_ONLY"}},
        },
    )
    mooncake_connector = MagicMock()
    mooncake_connector.request_finished.return_value = (
        True,
        {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_engine_id": "prefiller",
            "remote_block_ids": [10, 11],
            "remote_host": "7.150.4.174",
            "remote_port": "30000",
            "remote_request_id": "req-1",
            "last_token_id": 42,
            "num_prompt_blocks": 2,
        },
    )
    multi._connectors = [lmcache_connector, mooncake_connector]
    multi._requests_to_connector = {"req-1": 0}
    multi._extra_async_saves = {}
    request = SimpleNamespace(request_id="req-1")

    async_save, params = multi.request_finished_all_groups(request, ([10], [11]))

    assert async_save is True
    assert params == {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_engine_id": "prefiller",
        "remote_block_ids": [10, 11],
        "remote_host": "7.150.4.174",
        "remote_port": "30000",
        "remote_request_id": "req-1",
        "last_token_id": 42,
        "num_prompt_blocks": 2,
        "lmcache.remote_fill": {"terminal": {"outcome": "PERSISTENT_ONLY"}},
    }


def test_ascend_multi_remote_fill_echo_preserved_without_sibling_params():
    """Without a sibling envelope the remote-fill response keeps its echo."""

    multi = object.__new__(AscendMultiConnector)
    lmcache_connector = MagicMock()
    echoed = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "lmcache.remote_fill": {"terminal": {"outcome": "LOCAL_FULL"}},
    }
    lmcache_connector.request_finished.return_value = (False, echoed)
    idle_connector = MagicMock()
    idle_connector.request_finished.return_value = (False, None)
    multi._connectors = [lmcache_connector, idle_connector]
    multi._requests_to_connector = {"req-1": 0}
    multi._extra_async_saves = {}
    request = SimpleNamespace(request_id="req-1")

    async_save, params = multi.request_finished_all_groups(request, ([10],))

    assert async_save is False
    assert params == echoed
