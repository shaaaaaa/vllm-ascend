import inspect
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)
from vllm.logger import init_logger
from vllm.utils.func_utils import supports_kw
from vllm.v1.core.kv_cache_manager import KVCacheBlocks

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_index_connector import (
    MooncakeDSAIndexConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnector,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.forward_context import ForwardContext
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _is_single_tensor_kv(kv_cache: Any) -> bool:
    return isinstance(kv_cache, (tuple, list)) and len(kv_cache) < 2


def _block_group_counts(blocks: "KVCacheBlocks") -> tuple[int, ...]:
    return tuple(len(group) for group in blocks.blocks)


def _single_group_blocks(blocks: "KVCacheBlocks", group_idx: int) -> "KVCacheBlocks":
    if group_idx >= len(blocks.blocks):
        raise RuntimeError(
            f"Expected KV cache group {group_idx}, but only "
            f"{len(blocks.blocks)} groups exist."
        )
    return KVCacheBlocks((blocks.blocks[group_idx],))


def _has_remote_prefill_blocks(request: "Request") -> bool:
    params = getattr(request, "kv_transfer_params", None)
    if not isinstance(params, dict):
        return False
    required_keys = (
        "remote_engine_id",
        "remote_host",
        "remote_port",
        "remote_request_id",
    )
    return (
        bool(params.get("do_remote_prefill"))
        and bool(params.get("remote_block_ids"))
        and all(key in params and params[key] is not None for key in required_keys)
    )


def _callable_accepts_args(
    func: Any,
    num_positional_args: int,
    keyword_names: set[str],
) -> bool:
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False

    positional_count = 0
    accepts_varargs = False
    accepts_varkw = False
    accepted_keywords = set()
    for param in params:
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            accepts_varargs = True
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_varkw = True
        elif param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_count += 1
            accepted_keywords.add(param.name)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            accepted_keywords.add(param.name)

    accepts_positional = accepts_varargs or positional_count >= num_positional_args
    accepts_keywords = accepts_varkw or keyword_names.issubset(accepted_keywords)
    return accepts_positional and accepts_keywords


class AscendMultiConnector(MultiConnector, SupportsHMA):
    # DSA unbundle needs the model runner to pass both latent and indexer KV
    # caches so this connector can route them to different children.
    requires_full_dsa_kv_caches = True

    @staticmethod
    def _supports_dsa_compact_load(connector: Any) -> bool:
        capability = getattr(
            connector, "supports_dsa_compact_external_load", False
        )
        return bool(capability() if callable(capability) else capability)

    @property
    def supports_dsa_compact_external_load(self) -> bool:
        # The scheduler reads this immediately after matched-token lookup.
        # Delegate to the child selected for that request, since advertising
        # another child's capability could select compact allocation for an
        # incompatible loader.
        return getattr(self, "_selected_supports_dsa_compact_load", False)

    @property
    def uses_layerwise_model_callbacks(self) -> bool:
        return any(
            getattr(connector, "uses_layerwise_model_callbacks", False)
            for connector in self._connectors
        )

    def _staged_sfa_connector(self):
        return next(
            (
                (index, connector)
                for index, connector in enumerate(self._connectors)
                if getattr(
                    connector, "supports_staged_sfa_sparse_load", False
                )
                and getattr(connector, "uses_layerwise_model_callbacks", False)
                and callable(getattr(connector, "wait_for_layer_load", None))
                and callable(getattr(connector, "_get_connector_metadata", None))
            ),
            None,
        )

    @property
    def supports_staged_sfa_sparse_load(self) -> bool:
        return self._staged_sfa_connector() is not None

    def _unwrap_staged_sfa_connector_metadata(self, metadata):
        entry = self._staged_sfa_connector()
        if entry is None:
            raise RuntimeError(
                "No child connector satisfies the staged-SFA contract"
            )
        if not isinstance(metadata, MultiKVConnectorMetadata):
            raise TypeError(
                "AscendMultiConnector requires MultiKVConnectorMetadata"
            )
        index, _ = entry
        if len(metadata.metadata) != len(self._connectors):
            raise RuntimeError(
                "MultiConnector metadata does not match its child connectors"
            )
        return metadata.metadata[index]

    def get_live_split_results(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for connector in self._connectors:
            get_results = getattr(connector, "get_live_split_results", None)
            if not callable(get_results):
                continue
            for request_id, status in get_results().items():
                previous = merged.setdefault(request_id, status)
                if previous != status:
                    merged[request_id] = "failure"
        backlog = getattr(self, "_live_split_result_backlog", None)
        if backlog is None:
            backlog = self._live_split_result_backlog = {}
        for connector in self._connectors:
            accept_results = getattr(
                connector, "_accept_live_split_results", None
            )
            if not callable(accept_results):
                continue
            provider_id = id(connector)
            pending = backlog.pop(provider_id, {})
            for request_id, status in merged.items():
                previous = pending.setdefault(request_id, status)
                if previous != status:
                    pending[request_id] = "failure"
            if not pending:
                continue
            try:
                accept_results(dict(pending))
            except Exception:
                backlog[provider_id] = pending
                logger.exception(
                    "Live-split result provider failed; preserving status "
                    "for persistent fallback"
                )
        return merged

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        finished = super().get_finished(finished_req_ids)
        self.get_live_split_results()
        return finished

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        KVConnectorBase_V1.__init__(
            self,
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        self._connectors: list[KVConnectorBase_V1] = []
        self._ktc_kv_transfer_config = []
        for connector_cls, temp_config in self._get_connector_classes_and_configs(
            vllm_config
        ):
            if supports_kw(connector_cls, "kv_cache_config"):
                connector = connector_cls(
                    temp_config,
                    role,
                    kv_cache_config=kv_cache_config,
                )
            else:
                connector = connector_cls(temp_config, role)
            self._connectors.append(connector)
            self._ktc_kv_transfer_config.append(temp_config.kv_transfer_config)

        # A mapping from request id to the index of the connector chosen to
        # load the request from (if any).
        self._requests_to_connector: dict[str, int] = {}

        # Tracks additional async saves beyond the first one. This mirrors
        # MultiConnector while allowing legacy child connector constructors.
        self._extra_async_saves: dict[str, int] = {}
        self._index_load_async_req_ids: set[str] = set()
        self._live_split_result_backlog: dict[int, dict[str, str]] = {}
        self._wait_for_layer_load_sig_cache: dict[
            tuple[type, int, tuple[str, ...]], bool
        ] = {}

        logger.info(
            "AscendMultiConnector initialized children: %s",
            [connector.__class__.__name__ for connector in self._connectors],
        )

    def _has_dsa_index_connector(self) -> bool:
        return any(isinstance(c, MooncakeDSAIndexConnector) for c in self._connectors)

    def _blocks_for_connector(
        self,
        connector: Any,
        blocks: "KVCacheBlocks",
    ) -> "KVCacheBlocks":
        if isinstance(connector, SupportsHMA):
            return blocks
        return _single_group_blocks(blocks, 0)

    def _should_receive_alloc_update(self, connector: Any) -> bool:
        return isinstance(
            connector,
            (MooncakeLayerwiseConnector, MooncakeDSAIndexConnector),
        )

    def _merge_kv_transfer_params(
        self,
        merged: dict[str, Any] | None,
        new_params: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if new_params is None:
            return merged
        if merged is None:
            return dict(new_params)
        for key, value in new_params.items():
            if key in merged and merged[key] != value:
                raise RuntimeError(
                    "Multiple connectors produced conflicting KV transfer "
                    f"params for key {key!r}: {merged[key]!r} != {value!r}"
                )
            merged[key] = value
        return merged

    def register_kv_caches(self, kv_caches: dict):
        has_index_connector = self._has_dsa_index_connector()
        latent_only = (
            {
                name: kv
                for name, kv in kv_caches.items()
                if not _is_single_tensor_kv(kv)
            }
            if has_index_connector
            else kv_caches
        )

        if has_index_connector:
            logger.info(
                "AscendMultiConnector DSA KV split: total_layers=%d "
                "latent_layers=%d indexer_layers=%d children=%s",
                len(kv_caches),
                len(latent_only),
                len(kv_caches) - len(latent_only),
                [connector.__class__.__name__ for connector in self._connectors],
            )

        for connector in self._connectors:
            requires_full = isinstance(connector, SupportsHMA) or getattr(
                connector, "requires_full_dsa_kv_caches", False
            )
            connector.register_kv_caches(kv_caches if requires_full else latent_only)

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        late_consumers = [
            connector
            for connector in self._connectors
            if callable(
                getattr(
                    connector, "_accept_live_split_destination_plans", None
                )
            )
            and callable(
                getattr(
                    connector, "_needs_live_split_destination_plans", None
                )
            )
            and connector._needs_live_split_destination_plans()
        ]
        if not late_consumers:
            for connector in self._connectors:
                connector.start_load_kv(forward_context, **kwargs)
            return

        handled_groups: set[int] = set()
        for connector in late_consumers:
            get_groups = getattr(connector, "_live_split_source_groups", None)
            handled_groups.update(
                get_groups() if callable(get_groups) else (0, 1)
            )
        unhandled_groups = {0, 1} - handled_groups

        providers = []
        for connector in self._connectors:
            if connector in late_consumers:
                continue
            connector.start_load_kv(forward_context, **kwargs)
            take_plans = getattr(
                connector, "_take_live_split_destination_plans", None
            )
            if callable(take_plans):
                accepts_groups = _callable_accepts_args(
                    take_plans, 1, set()
                )
                if unhandled_groups and not accepts_groups:
                    logger.warning(
                        "Live-split provider cannot acknowledge unhandled "
                        "groups %s; using persistent fallback",
                        sorted(unhandled_groups),
                    )
                    continue
                providers.append((take_plans, accepts_groups))

        plans: dict[str, Any] = {}
        for take_plans, accepts_groups in providers:
            try:
                provided = (
                    take_plans(tuple(sorted(handled_groups)))
                    if accepts_groups
                    else take_plans()
                )
                if provided:
                    plans.update(provided)
            except Exception:
                logger.exception(
                    "Live-split destination provider failed; using persistent fallback"
                )
                plans.clear()
                break

        for connector in late_consumers:
            connector._accept_live_split_destination_plans(plans)
            connector.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(
        self,
        layer_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        sig_cache = getattr(self, "_wait_for_layer_load_sig_cache", None)
        if sig_cache is None:
            sig_cache = {}
            self._wait_for_layer_load_sig_cache = sig_cache

        for connector in self._connectors:
            wait_for_layer_load = connector.wait_for_layer_load
            cache_key = (connector.__class__, 1 + len(args), tuple(sorted(kwargs)))
            accepts_args = sig_cache.get(cache_key)
            if accepts_args is None:
                accepts_args = _callable_accepts_args(
                    wait_for_layer_load,
                    1 + len(args),
                    set(kwargs),
                )
                sig_cache[cache_key] = accepts_args

            if accepts_args:
                wait_for_layer_load(layer_name, *args, **kwargs)
            else:
                wait_for_layer_load(layer_name)
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        to_return = (0, False)
        chosen_connector = -1
        self._selected_supports_dsa_compact_load = False
        for i, connector in enumerate(self._connectors):
            tokens, load_async = connector.get_num_new_matched_tokens(
                request,
                num_computed_tokens,
            )
            if tokens is None:
                return None, False
            if to_return[0] == 0 and tokens > 0:
                self._requests_to_connector[request.request_id] = i
                chosen_connector = i
                to_return = (tokens, load_async)
                self._selected_supports_dsa_compact_load = (
                    self._supports_dsa_compact_load(connector)
                )

        tokens, load_async = to_return
        if (
            tokens > 0
            and not load_async
            and self._has_dsa_index_connector()
            and _has_remote_prefill_blocks(request)
        ):
            chosen_connector_name = (
                self._connectors[chosen_connector].__class__.__name__
                if 0 <= chosen_connector < len(self._connectors)
                else "none"
            )
            index_load_async_req_ids = getattr(
                self, "_index_load_async_req_ids", None
            )
            if index_load_async_req_ids is None:
                index_load_async_req_ids = set()
                self._index_load_async_req_ids = index_load_async_req_ids
            index_load_async_req_ids.add(request.request_id)
            params = request.kv_transfer_params
            logger.info(
                "AscendMultiConnector scheduling async DSA index load: "
                "request_id=%s external_tokens=%d chosen_connector=%s "
                "remote_index_blocks=%d",
                request.request_id,
                tokens,
                chosen_connector_name,
                len(params["remote_block_ids"]),
            )
            return tokens, True

        return to_return

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        chosen_connector_name = (
            self._connectors[chosen_connector].__class__.__name__
            if 0 <= chosen_connector < len(self._connectors)
            else "none"
        )
        params = getattr(request, "kv_transfer_params", None)
        do_remote_prefill = (
            params.get("do_remote_prefill") if params is not None else None
        )
        do_remote_decode = (
            params.get("do_remote_decode") if params is not None else None
        )
        index_load_async_req_ids = getattr(self, "_index_load_async_req_ids", set())
        skip_chosen_zero_update = (
            num_external_tokens == 0
            and request.request_id in index_load_async_req_ids
            and getattr(request, "num_computed_tokens", 0) > 0
            and chosen_connector >= 0
        )
        if (
            num_external_tokens > 0
            or chosen_connector >= 0
            or do_remote_prefill
            or do_remote_decode
        ):
            logger.info(
                "AscendMultiConnector alloc dispatch: request_id=%s "
                "external_tokens=%d block_groups=%s chosen_connector=%s "
                "do_remote_prefill=%s do_remote_decode=%s",
                request.request_id,
                num_external_tokens,
                _block_group_counts(blocks),
                chosen_connector_name,
                do_remote_prefill,
                do_remote_decode,
            )
        empty_blocks = blocks.new_empty()
        for i, connector in enumerate(self._connectors):
            if skip_chosen_zero_update and i == chosen_connector:
                logger.info(
                    "AscendMultiConnector preserving latent load state "
                    "after async DSA index load: request_id=%s "
                    "skipped_connector=%s",
                    request.request_id,
                    connector.__class__.__name__,
                )
                continue
            should_update = i == chosen_connector or self._should_receive_alloc_update(
                connector
            )
            target_blocks = blocks if should_update else empty_blocks
            target_tokens = num_external_tokens if should_update else 0
            connector.update_state_after_alloc(
                request,
                self._blocks_for_connector(connector, target_blocks),
                target_tokens,
            )
        if skip_chosen_zero_update:
            index_load_async_req_ids.discard(request.request_id)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        logger.info(
            "AscendMultiConnector finish dispatch: request_id=%s "
            "block_groups=%s children=%s",
            request.request_id,
            tuple(len(group) for group in block_ids),
            [connector.__class__.__name__ for connector in self._connectors],
        )
        async_saves = 0
        kv_transfer_params: dict[str, Any] | None = None

        connectors = self._connectors
        params = getattr(request, "kv_transfer_params", None)
        if isinstance(params, dict) and params.get("do_remote_decode"):
            live_providers = []
            for connector in connectors:
                try:
                    capable = self._supports_dsa_compact_load(connector)
                except Exception:
                    logger.exception(
                        "Live-split scheduler capability probe failed"
                    )
                    capable = False
                if capable:
                    live_providers.append(connector)
            if live_providers:
                connectors = live_providers + [
                    connector
                    for connector in connectors
                    if connector not in live_providers
                ]

        for connector in connectors:
            if isinstance(connector, SupportsHMA):
                async_save, txfer_params = connector.request_finished_all_groups(
                    request, block_ids
                )
            else:
                async_save, txfer_params = connector.request_finished(
                    request, block_ids[0]
                )

            if async_save:
                async_saves += 1
            # The compact provider runs before Mooncake. Pass its opaque source
            # layout to Mooncake for registration validation/canonicalization;
            # only Mooncake's canonical descriptor is returned downstream.
            if (
                isinstance(params, dict)
                and isinstance(txfer_params, dict)
                and "ascend_live_split_source_v1" in txfer_params
            ):
                params["ascend_live_split_source_v1"] = txfer_params.pop(
                    "ascend_live_split_source_v1"
                )
            kv_transfer_params = self._merge_kv_transfer_params(
                kv_transfer_params, txfer_params
            )

        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1

        self._requests_to_connector.pop(request.request_id, None)
        index_load_async_req_ids = getattr(self, "_index_load_async_req_ids", None)
        if index_load_async_req_ids is not None:
            index_load_async_req_ids.discard(request.request_id)
        return async_saves > 0, kv_transfer_params
