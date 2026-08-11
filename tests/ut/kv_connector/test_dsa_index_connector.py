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

from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheBlocks  # noqa: E402

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import (  # noqa: E402
    AscendMultiConnector,
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


def test_ascend_multi_reads_staged_sfa_metadata_from_capable_child():
    metadata = object()
    multi = object.__new__(AscendMultiConnector)
    multi._connectors = [
        SimpleNamespace(_get_connector_metadata=lambda: object()),
        SimpleNamespace(
            supports_staged_sfa_sparse_load=True,
            uses_layerwise_model_callbacks=True,
            wait_for_layer_load=lambda _layer_name: None,
            _get_connector_metadata=lambda: metadata,
        ),
    ]

    assert multi._get_staged_sfa_connector_metadata() is metadata


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
