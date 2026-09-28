# SPDX-License-Identifier: Apache-2.0
"""Opt-in, raw-tensor observations of real PD serving (never a transport shim).

This diagnostic deliberately reads device tensors and requires eager execution.
Each worker owns its recorder; importing this module installs no hooks. Files
contain request data, and incomplete calls remain explicitly incomplete.
"""

import functools
import inspect
import json
import math
import os
import re
import socket
from bisect import bisect_left
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def cpu_tensor(value):
    return value.detach().to(device="cpu", copy=True).contiguous()


def safe_component(value):
    value = str(value)
    if not value or value in (".", "..") or len(value.encode("utf-8")) > 180:
        raise ValueError("Diagnostic request/run identity must be nonempty and at most 180 UTF-8 bytes")
    component = quote(value, safe="")
    if len(component) > 220:
        raise ValueError("Encoded diagnostic identity exceeds filesystem component limit")
    return component


def row_window(rows, actual, cp=None, prefer_local=True):
    if cp is not None:
        start, end = int(cp.local_start), int(cp.local_end)
        capacity = int(cp.local_end_with_pad) - start
        if rows == capacity and (prefer_local or rows != actual):
            return start, max(start, end)
    if rows >= actual:
        return 0, actual
    raise ValueError(f"Cannot align {rows} tensor rows with {actual} scheduled query rows")


def sample_rows(positions, limit, context_length, *, aligned=False):
    """Use fixed request-local 4096-token segments, never random selection.

    KV uses absolute offsets 0..63 in each segment, shared by P and D. Query
    tensors keep the first visible rows per segment so TP shards and one-row
    decode calls remain observable even outside those KV offsets.
    """
    chunk_size = 4096
    indices, segments = [], []
    left = 0
    while left < len(positions):
        base = positions[left] // chunk_size * chunk_size
        right = bisect_left(positions, base + chunk_size, lo=left)
        stop = min(right, left + limit) if limit else right
        if aligned and limit:
            stop = min(stop, bisect_left(positions, base + limit, lo=left, hi=right))
        indices.extend(range(left, stop))
        segments.append(
            dict(
                start=base,
                length=min(chunk_size, context_length - base),
                source_rows=right - left,
                saved_rows=stop - left,
                omitted_position_bounds=[positions[stop], positions[right - 1] + 1] if stop < right else None,
            )
        )
        left = right
    capture = dict(
        strategy="absolute_chunk_prefix" if aligned else "visible_chunk_prefix",
        chunk_size=chunk_size,
        limit=limit,
        source_rows=len(positions),
        saved_rows=len(indices),
        source_indices=indices,
        segments=segments,
    )
    return [positions[i] for i in indices], capture


def sampled_tensor(tensor, capture):
    indices = capture["source_indices"]
    if not indices:
        return tensor[:0]
    if indices[-1] - indices[0] + 1 == len(indices):
        return tensor[indices[0] : indices[-1] + 1]
    return tensor.index_select(0, torch.tensor(indices, dtype=torch.long, device=tensor.device))


def feature_prefix(tensor, kind, name, limit):
    # Keep token/head axes and all discrete routing evidence. Quantized KV
    # contains features too, even when its dtype is int8 rather than floating.
    eligible = (
        tensor.ndim >= 2
        and kind not in ("mapping", "rejection", "logits", "draft")
        and name not in ("topk", "logical_topk", "input_ids", "positions")
        and (tensor.is_floating_point() or kind in ("kv_current", "kv_consumed", "kv_indexer"))
    )
    if not eligible or not limit or tensor.shape[-1] <= limit:
        return tensor, None
    capture = dict(axis=tensor.ndim - 1, start=0, source_size=int(tensor.shape[-1]), saved_size=limit)
    return tensor[..., :limit], capture


def unique_kv_pairs(positions, slots):
    """Vectorized CPU lexicographic deduplication; retain physical aliases.

    torch.unique(dim=0) compares tiny tensor rows and becomes expensive with
    a full prefill query-by-top-k matrix. Never pack addresses into arithmetic
    keys: that can overflow or conflate different (position, slot) pairs.
    """
    pairs = np.stack((positions.numpy(), slots.numpy()), axis=1)
    if not len(pairs):
        return torch.from_numpy(pairs)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    distinct = np.empty(len(pairs), dtype=np.bool_)
    distinct[0] = True
    distinct[1:] = np.any(pairs[1:] != pairs[:-1], axis=1)
    return torch.from_numpy(pairs[distinct])


class RequestArchive:
    def __init__(self, root, metadata):
        self.metadata = dict(metadata)
        worker = (
            f"{safe_component(metadata['host'])}-dp{metadata['dp_rank']}-tp{metadata['tp_rank']}-pid{metadata['pid']}"
        )
        self.root = Path(root) / metadata["role"] / safe_component(metadata["request_id"]) / worker
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "tensors").mkdir()
        self.metadata.update(
            schema_version=1,
            tool="pd_tensor_dump",
            complete=False,
            request_finished=False,
            errors=[],
            records=0,
            calls=[],
        )
        self.calls = {}
        self.flush()

    def flush(self):
        write_json(self.root / "manifest.json", self.metadata)

    def begin(self, info, expected):
        number = len(self.calls)
        call = dict(info, schema=1, call=number, complete=False, expected=list(expected))
        self.calls[number] = call
        self.metadata["calls"].append(number)
        self.metadata["complete"] = False
        write_json(self.root / "calls" / f"{number}.json", call)
        self.flush()
        return call

    def record(
        self,
        tensor,
        call,
        layer,
        kind,
        name,
        positions=None,
        mapping_only=False,
        tensor_layout="rank_local",
        *,
        physical_slots=None,
        row_capture=None,
        feature_capture=None,
    ):
        context = call["context_token_ids"]
        if positions is not None:
            if tensor.ndim == 0 or tensor.shape[0] != len(positions):
                raise ValueError(f"Invalid tensor row mapping: {kind}/{name}")
            if any(p < 0 or p >= len(context) for p in positions):
                raise ValueError("Tensor logical position is outside its recorded input context")
        if physical_slots is not None:
            if positions is None or len(physical_slots) != len(positions) or any(s < 0 for s in physical_slots):
                raise ValueError("Invalid consumed KV physical slot mapping")
        if positions is not None and row_capture is None:
            positions, row_capture = sample_rows(positions, self.metadata.get("max_token_rows", 0), len(context))
            tensor = sampled_tensor(tensor, row_capture)
            if physical_slots is not None:
                physical_slots = [physical_slots[i] for i in row_capture["source_indices"]]
        if feature_capture is None:
            tensor, feature_capture = feature_prefix(tensor, kind, name, self.metadata.get("max_features", 8))
        # Materialize the already narrowed view on device. A strided feature
        # view must not cause a full-width temporary/readback in the backend.
        value = cpu_tensor(tensor.contiguous())
        relative = f"tensors/{self.metadata['records']:08d}.pt"
        temporary = (self.root / relative).with_suffix(".pt.tmp")
        torch.save(value, temporary)
        os.replace(temporary, self.root / relative)
        record = dict(
            schema=1,
            request_id=self.metadata["request_id"],
            model=call.get("model", "main"),
            call=call["call"],
            layer=layer,
            kind=kind,
            name=name,
            path=relative,
            shape=list(value.shape),
            dtype=str(value.dtype),
            positions=positions,
            token_ids=[context[p] for p in positions] if positions is not None else None,
            row_axis=0 if positions is not None else None,
            mapping_only=mapping_only,
            tensor_layout="mapping" if mapping_only else tensor_layout,
        )
        if physical_slots is not None:
            record["physical_slots"] = physical_slots
        if row_capture is not None:
            record["row_capture"] = row_capture
        if feature_capture is not None:
            record["feature_capture"] = feature_capture
        with (self.root / "index.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.metadata["records"] += 1
        return record

    def end(self, call, observed):
        missing = {(e["layer"], e["kind"], e["name"]) for e in call["expected"]} - observed
        if missing:
            self.fail(f"call {call['call']} missing tensors: {sorted(missing)[:8]}")
        if not call["context_complete"]:
            self.fail(f"call {call['call']} has unknown input context")
        call["complete"] = not missing and call["context_complete"]
        write_json(self.root / "calls" / f"{call['call']}.json", call)
        self.metadata["complete"] = not self.metadata["errors"] and all(c["complete"] for c in self.calls.values())
        self.flush()

    def fail(self, error):
        self.metadata["errors"].append(str(error))
        self.metadata["complete"] = False
        self.flush()

    def finish(self):
        self.metadata["request_finished"] = True
        self.flush()


class PDTensorDump:
    """Capture main verification and MTP calls under separate model identities."""

    def __init__(self, runner, root, role, rank_info, get_context, max_token_rows=0, max_features=8):
        if type(max_token_rows) is not int or not 0 <= max_token_rows <= 4096:
            raise ValueError("PD tensor dump max token rows must be an integer in [0, 4096] (0 = all)")
        self.max_token_rows = max_token_rows
        if type(max_features) is not int or max_features < 0:
            raise ValueError("PD tensor dump max features must be a nonnegative integer (0 = all)")
        self.max_features = max_features
        self.runner, self.root, self.role = runner, Path(root), role
        self.rank_info, self.get_context = dict(rank_info), get_context
        self.external_ids = {}
        self.archives = {}
        self.active = None
        self.last = None
        self.sfa_stack = []
        self.sfa_caches = []
        self.handles = []
        self.patches = []
        self.layers = {}
        self.attentions = {}
        self.logical_topk = None
        self.scheduled = {}
        self.mtp_layers = {}
        self.mtp_runtime = None
        self.mtp_last = None
        self.rejection_entries = None

    def observe_scheduler(self, output):
        for req in output.scheduled_new_reqs:
            self.external_ids[req.req_id] = getattr(req, "external_req_id", None)
        for internal in output.finished_req_ids:
            archive = self.archives.pop(internal, None)
            if archive is not None:
                archive.finish()
            self.external_ids.pop(internal, None)
        self.scheduled = dict(output.num_scheduled_tokens)

    def metadata(self, layer=None):
        context = self.get_context()
        values = context.attn_metadata
        if not isinstance(values, dict):
            return None
        for index, name in self.attentions.values():
            is_mtp = self.active is not None and self.active.get("model") == "mtp"
            if (layer is not None and index == layer) or (layer is None and (index in self.mtp_layers) == is_mtp):
                return values.get(name)
        return None

    def patch(self, owner, name, factory):
        original = getattr(owner, name)
        self.patches.append((owner, name, original))
        setattr(owner, name, factory(original))

    def install(self, sfa_class, torch_npu):
        for name, module in self.runner.model.named_modules():
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
            if match and "DecoderLayer" in type(module).__name__:
                layer = int(match[1])
                if layer in self.layers:
                    raise ValueError("Ambiguous main decoder layer inventory")
                self.layers[layer] = module
            impl = getattr(module, "impl", None)
            if isinstance(impl, sfa_class):
                layer_name = module.layer_name
                match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
                if match:
                    self.attentions[id(impl)] = (int(match[1]), layer_name)
        if not self.layers or sorted(self.layers) != sorted(v[0] for v in self.attentions.values()):
            raise ValueError("PD tensor dump requires a complete main decoder/SFA layer inventory")
        config = self.runner.model_config.hf_text_config
        if sorted(self.layers) != list(range(int(config.num_hidden_layers))):
            raise ValueError("PD tensor dump layer inventory differs from model configuration")
        self.install_mtp(sfa_class)
        for layer, module in (self.layers | self.mtp_layers).items():
            self.handles.append(
                module.register_forward_pre_hook(functools.partial(self.decoder_pre, layer), with_kwargs=True)
            )
            self.handles.append(
                module.register_forward_hook(functools.partial(self.decoder_post, layer), with_kwargs=True)
            )
        self.patch(sfa_class, "forward", self.sfa_factory)
        self.patch(sfa_class, "_get_indexcache_topk_indices", self.shared_topk_factory)
        self.patch(torch.ops._C_ascend, "npu_sparse_flash_attention", self.kernel_factory("attention"))
        for owner, name, tuple_result in (
            (torch.ops._C_ascend, "npu_lightning_indexer", False),
            (torch.ops._C_ascend, "npu_lightning_indexer_quant", False),
            (torch_npu, "npu_lightning_indexer", True),
        ):
            if hasattr(owner, name):
                self.patch(owner, name, self.kernel_factory("indexer", tuple_result))
        self.patch(self.runner.model, "compute_logits", self.logits_factory)
        self.install_rejection()

    def install_rejection(self, ascend_module=None, upstream_module=None):
        sampler = getattr(self.runner, "rejection_sampler", None)
        if sampler is None:
            return
        if ascend_module is None:
            from vllm.v1.sample import rejection_sampler as upstream_module

            from vllm_ascend.sample import rejection_sampler as ascend_module

        def forward_factory(original):
            @functools.wraps(original)
            def forward(metadata, draft_probs, logits, sampling_metadata):
                if self.last is None:
                    return original(metadata, draft_probs, logits, sampling_metadata)
                entries = []
                start = 0
                targets = cpu_tensor(metadata.target_logits_indices).long()
                bonuses = cpu_tensor(metadata.bonus_logits_indices).long()
                query_rows = cpu_tensor(metadata.logits_indices).long()
                captured = {e["archive"].metadata["internal_request_id"]: e for e in self.last["entries"]}
                for row, internal in enumerate(self.runner.input_batch.req_ids):
                    count = int(metadata.num_draft_tokens[row])
                    end = start + count
                    if internal in captured:
                        entry = captured[internal]
                        selected = targets[start:end]
                        bonus = bonuses[row : row + 1]
                        context_positions = [self.last["positions"][i] for i in query_rows[selected].tolist()]
                        options = {}
                        for name in (
                            "temperature",
                            "top_k",
                            "top_p",
                            "frequency_penalties",
                            "presence_penalties",
                            "repetition_penalties",
                        ):
                            value = getattr(sampling_metadata, name, None)
                            if isinstance(value, torch.Tensor):
                                options[name] = cpu_tensor(value).reshape(-1)[row].item()
                        greedy = bool(sampling_metadata.all_greedy) or options.get("temperature") == 0
                        entry["call"]["rejection"] = dict(
                            schema=1,
                            num_draft_tokens=count,
                            max_spec_len=int(metadata.max_spec_len),
                            mode="greedy" if greedy else "random",
                            sampling=options,
                            draft_probs_present=draft_probs is not None,
                            placeholder_token_id=int(upstream_module.PLACEHOLDER_TOKEN_ID),
                            target_context_positions=context_positions,
                            draft_positions=[p + 1 for p in context_positions],
                            bonus_context_position=self.last["positions"][int(query_rows[bonus[0]])],
                            row_axis="per-request draft order; output retains sampler padding",
                        )
                        required = [
                            "draft_token_ids",
                            "target_logits",
                            "bonus_token_ids",
                            "kernel_output",
                            "sampler_output",
                        ]
                        if draft_probs is not None:
                            required.append("draft_probs")
                        if not sampling_metadata.all_greedy:
                            required.extend(("uniform_probs", "target_probs", "recovered_token_ids"))
                        entry["call"]["expected"].extend(
                            dict(layer=-1, kind="rejection", name=name) for name in required
                        )
                        entries.append((entry, row, start, end))
                        self.rejection_record(
                            entry, "raw_target_logits", logits.index_select(0, selected.to(logits.device))
                        )
                        self.rejection_record(
                            entry, "raw_bonus_logits", logits.index_select(0, bonus.to(logits.device))
                        )
                        for name, value in (
                            ("target_logits_indices", selected),
                            ("bonus_logits_indices", bonus),
                            ("query_row_indices", query_rows[torch.cat((selected, bonus))]),
                        ):
                            self.rejection_record(entry, name, value, mapping=True)
                        entry["archive"].metadata["rejection_recording"] = True
                    start = end
                self.rejection_entries = entries
                try:
                    result = original(metadata, draft_probs, logits, sampling_metadata)
                    for entry, row, _, _ in entries:
                        self.rejection_record(entry, "sampler_output", result.sampled_token_ids[row])
                        # Save the added tensor inventory before main sampling ends.
                        write_json(entry["archive"].root / "calls" / f"{entry['call']['call']}.json", entry["call"])
                        entry["archive"].flush()
                    return result
                except BaseException as error:
                    for entry, _, _, _ in entries:
                        entry["archive"].fail(f"rejection {type(error).__name__}: {error}")
                    raise
                finally:
                    self.rejection_entries = None

            return forward

        def kernel_factory(original):
            signature = inspect.signature(original)

            @functools.wraps(original)
            def kernel(*args, **kwargs):
                if self.rejection_entries is None:
                    return original(*args, **kwargs)
                values = signature.bind(*args, **kwargs).arguments
                for entry, row, start, end in self.rejection_entries:
                    for name in ("draft_token_ids", "draft_probs", "target_logits"):
                        value = values[name]
                        if value is not None:
                            self.rejection_record(entry, name, value[start:end])
                    self.rejection_record(entry, "bonus_token_ids", values["bonus_token_ids"][row : row + 1])
                result = original(*args, **kwargs)
                for entry, row, _, _ in self.rejection_entries:
                    self.rejection_record(entry, "kernel_output", result[row])
                return result

            return kernel

        def random_factory(name):
            def factory(original):
                signature = inspect.signature(original)

                @functools.wraps(original)
                def call(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if self.rejection_entries is not None:
                        values = signature.bind(*args, **kwargs).arguments
                        for entry, _, start, end in self.rejection_entries:
                            self.rejection_record(entry, name, result[start:end])
                            if name == "recovered_token_ids":
                                self.rejection_record(entry, "target_probs", values["target_probs"][start:end])
                    return result

                return call

            return factory

        self.patch(sampler, "forward", forward_factory)
        # Upstream.forward and Ascend's existing timing wrapper each reference
        # their own module binding. Observe either without changing that path.
        self.patch(upstream_module, "rejection_sample", kernel_factory)
        self.patch(ascend_module, "rejection_sample", kernel_factory)
        self.patch(ascend_module, "generate_uniform_probs", random_factory("uniform_probs"))
        self.patch(ascend_module, "sample_recovered_tokens", random_factory("recovered_token_ids"))

    def rejection_record(self, entry, name, tensor, mapping=False):
        role = dict(layer=-1, kind="rejection", name=name)
        if role not in entry["call"]["expected"]:
            entry["call"]["expected"].append(role)
        self.record(entry, tensor, -1, "rejection", name, mapping_only=mapping)

    def install_mtp(self, sfa_class):
        """Install only on an explicitly enabled recorder; preserve draft execution."""
        drafter = getattr(self.runner, "drafter", None)
        if drafter is None:
            if getattr(self.runner, "speculative_config", None) is not None:
                raise ValueError("MTP is configured but its drafter is unavailable")
            return
        if getattr(drafter, "method", None) != "mtp":
            raise ValueError("PD tensor dump currently supports the MTP drafter")
        model = drafter.model
        if callable(getattr(model, "unwrap", None)):
            model = model.unwrap()
        for name, module in model.named_modules():
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
            if match and "DecoderLayer" in type(module).__name__:
                layer = int(match[1])
                if layer in self.layers or layer in self.mtp_layers:
                    raise ValueError("Ambiguous MTP decoder layer inventory")
                self.mtp_layers[layer] = module
            impl = getattr(module, "impl", None)
            if isinstance(impl, sfa_class):
                match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module.layer_name)
                if match:
                    self.attentions[id(impl)] = (int(match[1]), module.layer_name)
        observed = {i for i, _ in self.attentions.values()} - set(self.layers)
        if not self.mtp_layers or observed != set(self.mtp_layers):
            raise ValueError("Incomplete MTP decoder/SFA inventory")

        def draft_factory(original):
            @functools.wraps(original)
            def draft(model_kwargs, **kwargs):
                if self.last is None:
                    return original(model_kwargs, **kwargs)
                runtime = kwargs["runtime_inputs"]
                self.mtp_runtime = dict(
                    positions=cpu_tensor(drafter._get_positions(int(runtime["num_input_tokens"]))).tolist(),
                    indices=cpu_tensor(runtime["token_indices_to_sample"])
                    .long()
                    .tolist()[: int(runtime["batch_size"])],
                    draft_step=int(kwargs["draft_step"]),
                )
                try:
                    return original(model_kwargs, **kwargs)
                finally:
                    self.mtp_runtime = None

            return draft

        def forward_factory(original):
            signature = inspect.signature(original)

            @functools.wraps(original)
            def forward(*args, **kwargs):
                if self.mtp_runtime is None:
                    return original(*args, **kwargs)
                values = signature.bind(*args, **kwargs).arguments
                previous = self.active, self.logical_topk
                try:
                    self.start_mtp(values)
                    result = original(*args, **kwargs)
                    outputs = result if isinstance(result, (tuple, list)) else (result,)
                    for index, tensor in enumerate(outputs):
                        self.emit(
                            tensor,
                            -1,
                            "model_output",
                            "hidden_states" if index == 0 else str(index),
                            self.metadata(),
                            prefer_local=bool(getattr(self.get_context(), "flash_comm_v1_enabled", False)),
                        )
                    return result
                except BaseException as error:
                    if self.active:
                        for entry in self.active["entries"]:
                            entry["archive"].fail(f"MTP {type(error).__name__}: {error}")
                    raise
                finally:
                    self.active, self.logical_topk = previous

            return forward

        def logits_factory(original):
            @functools.wraps(original)
            def logits(hidden_states, *args, **kwargs):
                result = original(hidden_states, *args, **kwargs)
                state = self.mtp_last
                if state is None:
                    return result
                indices = state["indices"]
                for entry in state["entries"]:
                    rows = [i for i, index in enumerate(indices) if entry["start"] <= index < entry["end"]]
                    positions = [state["positions"][indices[i]] for i in rows]
                    selected = torch.tensor(rows, dtype=torch.long, device=result.device)
                    values = result.index_select(0, selected)
                    self.record(entry, hidden_states.index_select(0, selected), -1, "logits", "input", positions)
                    self.record(entry, values, -1, "logits", "output", positions)
                    self.record(entry, values.argmax(dim=-1), -1, "draft", "token_ids", positions)
                    entry["archive"].end(entry["call"], entry["observed"])
                self.mtp_last = None
                return result

            return logits

        self.patch(drafter, "_run_mtp_draft_layer_with_diagnostics", draft_factory)
        self.patch(model, "forward", forward_factory)
        self.patch(model, "compute_logits", logits_factory)

    def start_mtp(self, values):
        """Map shifted MTP inputs to requests using the actual draft metadata."""
        if self.mtp_last is not None:
            raise ValueError("Previous MTP call has no recorded logits")
        self.active = dict(model="mtp")
        meta = self.metadata()
        actual = int(meta.num_actual_tokens)
        bounds = [int(x) for x in meta.query_start_loc_cpu]
        ids = cpu_tensor(values["input_ids"]).reshape(-1).tolist()[:actual]
        positions = self.mtp_runtime["positions"][:actual]
        if len(ids) != actual or len(positions) != actual:
            raise ValueError("MTP inputs do not cover its actual query rows")
        entries = []
        main_entries = {entry["archive"].metadata["internal_request_id"]: entry for entry in self.last["entries"]}
        for row, internal in enumerate(self.runner.input_batch.req_ids):
            start, end = bounds[row : row + 2]
            if start == end or internal not in main_entries:
                continue
            # Padded drafting executes rejected query rows too, but samples the
            # last accepted row. They are not part of this request's MTP context.
            scheduled_end = end
            sample_index = self.mtp_runtime["indices"][row]
            if not start <= sample_index < end:
                raise ValueError("MTP sample row is outside its request query")
            end = sample_index + 1
            source = main_entries[internal]
            context = list(source["call"]["context_token_ids"])[1:]
            query_positions, query_ids = positions[start:end], ids[start:end]
            needed = max(query_positions) + 1
            context.extend([None] * max(0, needed - len(context)))
            for position, token in zip(query_positions, query_ids):
                context[position] = token
            context = context[:needed]
            archive = source["archive"]
            call = archive.begin(
                dict(
                    model="mtp",
                    parent_call=source["call"]["call"],
                    draft_step=self.mtp_runtime["draft_step"],
                    rejected_query_rows=scheduled_end - end,
                    phase=source["call"]["phase"],
                    positions=query_positions,
                    token_ids=query_ids,
                    context_token_ids=context,
                    context_complete=all(token is not None for token in context),
                    position_convention="MTP position p consumes target token p+1 and target hidden state p",
                ),
                self.expected("mtp"),
            )
            entries.append(dict(archive=archive, call=call, start=start, end=end, row=row, observed=set()))
        self.active = dict(
            model="mtp",
            entries=entries,
            positions=positions,
            token_ids=ids,
            actual=actual,
            indices=self.mtp_runtime["indices"],
        )
        self.mtp_last = self.active
        self.logical_topk = None
        local = bool(getattr(self.get_context(), "flash_comm_v1_enabled", False))
        for name in ("input_ids", "positions", "hidden_states"):
            self.emit(values[name], -1, "model_input", name, meta, prefer_local=local if name != "input_ids" else False)
        self.emit(torch.tensor(positions), -1, "model_input", "logical_positions", meta, prefer_local=False)

    def restore(self):
        for handle in self.handles:
            handle.remove()
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)

    def expected(self, model="main"):
        result = [
            dict(layer=-1, kind=kind, name=name)
            for kind, name in (
                ("model_input", "input_ids"),
                ("model_input", "positions"),
                ("model_output", "hidden_states"),
                ("logits", "output"),
            )
        ]
        if model == "mtp":
            result.extend(
                dict(layer=-1, kind=kind, name=name)
                for kind, name in (
                    ("model_input", "hidden_states"),
                    ("model_input", "logical_positions"),
                    ("logits", "input"),
                    ("draft", "token_ids"),
                )
            )
        for layer in self.mtp_layers if model == "mtp" else self.layers:
            for kind, names in (
                ("decoder", ("input", "output")),
                ("sfa", ("input", "output")),
                ("attention", ("query_nope", "query_rope", "logical_topk", "output")),
                ("kv_current", ("nope", "rope")),
                ("kv_consumed", ("nope", "rope")),
            ):
                result.extend(dict(layer=layer, kind=kind, name=name) for name in names)
        return result

    def start(self, input_ids, positions):
        if self.last is not None:
            raise ValueError("Previous PD tensor call did not finish sampling")
        meta = self.metadata()
        if meta is None or not self.scheduled:
            return False  # Warmup/EP dummy forward: never assign it to a user request.
        req_ids = list(self.runner.input_batch.req_ids)
        actual = int(meta.num_actual_tokens)
        bounds = [int(x) for x in meta.query_start_loc_cpu]
        if len(bounds) < len(req_ids) + 1 or bounds[len(req_ids)] != actual:
            raise ValueError("PD dump cannot align scheduled requests with attention query rows")
        ids = cpu_tensor(input_ids).reshape(-1).tolist()[:actual]
        pos = cpu_tensor(positions).reshape(-1).tolist()[:actual]
        if len(ids) != actual or len(pos) != actual:
            raise ValueError("PD dump needs unsharded, one-dimensional main model token/position inputs")
        entries = []
        for row, internal in enumerate(req_ids):
            start, end = bounds[row : row + 2]
            if start == end:
                continue
            external = self.external_ids.get(internal)
            if external is None:
                raise ValueError("Missing external_req_id; update vLLM and vLLM-Ascend together before PD dump")
            request = self.runner.requests[internal]
            if getattr(request.sampling_params, "n", 1) != 1:
                raise ValueError("PD tensor dump currently requires sampling n=1")
            if request.prompt_token_ids is None:
                raise ValueError("PD tensor dump currently requires text token IDs, not prompt embeddings")
            context = list(request.prompt_token_ids) + list(request.output_token_ids)
            query_positions, query_ids = pos[start:end], ids[start:end]
            if any(p < 0 for p in query_positions) or len(set(query_positions)) != len(query_positions):
                raise ValueError("Ambiguous main model positions for request")
            needed = max(query_positions) + 1
            context.extend([None] * max(0, needed - len(context)))
            for position, token in zip(query_positions, query_ids):
                context[position] = token
            context = context[:needed]
            archive = self.archives.get(internal)
            if archive is None:
                archive = RequestArchive(
                    self.root,
                    dict(
                        self.rank_info,
                        request_id=external,
                        internal_request_id=internal,
                        role=self.role,
                        model_id=str(self.runner.model_config.model),
                        num_layers=len(self.layers),
                        layerwise_prefill=bool(getattr(self.runner, "layerwise_prefill_p_node", False)),
                        scope="eager main backbone, target verification and MTP inputs/KV/logits",
                        mtp_layers=sorted(self.mtp_layers),
                        sampling_params=str(request.sampling_params),
                        prompt_token_ids=list(request.prompt_token_ids),
                        max_token_rows=self.max_token_rows,
                        max_features=self.max_features,
                    ),
                )
                self.archives[internal] = archive
            call = archive.begin(
                dict(
                    model="main",
                    phase="prefill" if min(query_positions) < len(request.prompt_token_ids) else "decode",
                    positions=query_positions,
                    token_ids=query_ids,
                    context_token_ids=context,
                    context_complete=all(x is not None for x in context),
                ),
                self.expected(),
            )
            entries.append(dict(archive=archive, call=call, start=start, end=end, row=row, observed=set()))
        self.active = dict(model="main", entries=entries, positions=pos, token_ids=ids, actual=actual)
        self.last = self.active
        self.logical_topk = None
        self.emit(input_ids, -1, "model_input", "input_ids", meta, prefer_local=False)
        self.emit(positions, -1, "model_input", "positions", meta, prefer_local=False)
        return True

    @contextmanager
    def forward(self, input_ids, positions):
        try:
            self.start(input_ids, positions)
            yield
        except BaseException as error:
            if self.active:
                for entry in self.active["entries"]:
                    entry["archive"].fail(f"{type(error).__name__}: {error}")
            raise
        finally:
            self.active = None

    def record(
        self,
        entry,
        tensor,
        layer,
        kind,
        name,
        positions=None,
        mapping_only=False,
        layout="rank_local",
        *,
        physical_slots=None,
        row_capture=None,
        feature_capture=None,
    ):
        key = (layer, kind, name)
        if key in entry["observed"]:
            raise ValueError(f"Duplicate PD tensor observation {key}")
        entry["archive"].record(
            tensor,
            entry["call"],
            layer,
            kind,
            name,
            positions,
            mapping_only,
            layout,
            physical_slots=physical_slots,
            row_capture=row_capture,
            feature_capture=feature_capture,
        )
        entry["observed"].add(key)

    def emit(self, tensor, layer, kind, name, meta, *, prefer_local=True, mapping_only=False):
        if self.active is None or tensor is None:
            return
        if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
            raise ValueError(f"Unsupported required tensor {kind}/{name}")
        cp = getattr(meta, "dsa_cp_context", None)
        lo, hi = row_window(tensor.shape[0], self.active["actual"], cp, prefer_local)
        for entry in self.active["entries"]:
            start, end = max(lo, entry["start"]), min(hi, entry["end"])
            start, end = max(lo, start), max(start, end)
            positions = self.active["positions"][start:end]
            self.record(
                entry,
                tensor[start - lo : end - lo],
                layer,
                kind,
                name,
                positions,
                mapping_only,
                "sequence_sharded" if cp is not None and prefer_local else "rank_local",
            )

    def decoder_pre(self, layer, module, args, kwargs):
        if self.active is None:
            return
        values = inspect.signature(module.forward).bind(*args, **kwargs).arguments
        for name, value in (("input", values.get("hidden_states")), ("input_residual", values.get("residual"))):
            if value is not None:
                self.emit(value, layer, "decoder", name, self.metadata(layer))

    def decoder_post(self, layer, module, args, kwargs, result):
        if self.active is None:
            return
        for index, value in enumerate(result if isinstance(result, (list, tuple)) else (result,)):
            if value is not None:
                self.emit(value, layer, "decoder", "output" if index == 0 else f"output_{index}", self.metadata(layer))

    def sfa_factory(self, original):
        @functools.wraps(original)
        def forward(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs):
            if self.active is None:
                return original(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs)
            layer, _ = self.attentions[id(impl)]
            self.sfa_stack.append((layer, attn_metadata))
            self.sfa_caches.append(kv_cache)
            try:
                self.emit(hidden_states, layer, "sfa", "input", attn_metadata)
                result = original(impl, layer_name, hidden_states, kv_cache, attn_metadata, *args, **kwargs)
                self.emit(result, layer, "sfa", "output", attn_metadata)
                return result
            finally:
                self.sfa_caches.pop()
                self.sfa_stack.pop()

        return forward

    def shared_topk_factory(self, original):
        @functools.wraps(original)
        def shared_topk(*args, **kwargs):
            result = original(*args, **kwargs)
            if self.active and self.sfa_stack:
                self.capture_topk(result, self.sfa_stack[-1][1])
            return result

        return shared_topk

    def capture_topk(self, value, meta):
        # Post-processing can mutate the top-k tensor in place. Snapshot only
        # the observed query rows now, before physical-address remapping.
        cp = getattr(meta, "dsa_cp_context", None)
        lo, hi = row_window(value.shape[0], self.active["actual"], cp)
        snapshots = {}
        for entry in self.active["entries"]:
            start, end = max(lo, entry["start"]), max(lo, min(hi, entry["end"]))
            end = max(start, end)
            positions, capture = sample_rows(
                self.active["positions"][start:end], self.max_token_rows, len(entry["call"]["context_token_ids"])
            )
            snapshots[entry["archive"].root] = (
                cpu_tensor(sampled_tensor(value[start - lo : end - lo], capture)),
                positions,
                capture,
            )
        self.logical_topk = dict(shape=tuple(value.shape), snapshots=snapshots)

    def current_kv(self, kv_cache, layer, meta):
        # Observe after CP all-gather/cache write, immediately before attention.
        # exec_kv alone runs too early and is bypassed by the MLAPO path.
        slots = cpu_tensor(meta.slot_mapping).long().reshape(-1)
        if len(slots) < self.active["actual"]:
            raise ValueError("Current KV slot mapping misses scheduled query rows")
        for entry in self.active["entries"]:
            start, end = entry["start"], entry["end"]
            positions, capture = sample_rows(
                self.active["positions"][start:end],
                self.max_token_rows,
                len(entry["call"]["context_token_ids"]),
                aligned=self.role == "P",
            )
            selected = sampled_tensor(slots[start:end], capture)
            for cache, name in zip(kv_cache[:2], ("nope", "rope")):
                flat = cache.reshape(-1, *cache.shape[2:])
                flat, features = feature_prefix(flat, "kv_current", name, self.max_features)
                if selected.numel() and (selected.min() < 0 or selected.max() >= flat.shape[0]):
                    raise ValueError("Current KV slot out of bounds")
                self.record(
                    entry,
                    flat.index_select(0, selected.to(cache.device)),
                    layer,
                    "kv_current",
                    name,
                    positions,
                    layout="replicated",
                    row_capture=capture,
                    feature_capture=features,
                )

    def indexer_kv(self, tensor, table, lengths, layer, name):
        table, lengths = cpu_tensor(table).long(), cpu_tensor(lengths).long().reshape(-1)
        block_size = int(tensor.shape[1])
        flat = tensor.reshape(-1, *tensor.shape[2:])
        flat, features = feature_prefix(flat, "kv_indexer", name, self.max_features)
        for entry in self.active["entries"]:
            row = entry["row"]
            if row >= len(lengths):
                raise ValueError("Indexer request/table row missing")
            # P's indexer cache is replicated after CP all-gather/cache write.
            # Its kernel length only describes this TP rank's query window (it
            # can be zero), so archive the computed global prefix through this
            # chunk's end. D records the prefix its kernel actually consumes.
            prefix = len(entry["call"]["context_token_ids"]) if self.role == "P" else int(lengths[row])
            if self.active.get("model") == "mtp":
                prefix = min(prefix, len(entry["call"]["context_token_ids"]))
            sampled, capture = sample_rows(range(prefix), self.max_token_rows, prefix, aligned=True)
            positions = torch.tensor(sampled, dtype=torch.long)
            if positions.numel() and int(positions[-1]) // block_size >= table.shape[1]:
                raise ValueError("Indexer block table does not cover prefix")
            slots = table[row, positions // block_size] * block_size + positions % block_size
            if slots.numel() and (slots.min() < 0 or slots.max() >= flat.shape[0]):
                raise ValueError("Indexer KV physical slot out of bounds")
            self.record(
                entry,
                flat.index_select(0, slots.to(tensor.device)),
                layer,
                "kv_indexer",
                name,
                positions.tolist(),
                layout="replicated",
                row_capture=capture,
                feature_capture=features,
            )

    def attention_kv(self, values, layer, meta):
        selected = values["sparse_indices"]
        if self.logical_topk is None:
            raise ValueError("Attention has no observed logical indexer output")
        table = cpu_tensor(values["block_table"]).long()
        lengths = cpu_tensor(values["actual_seq_lengths_kv"]).long().reshape(-1)
        cumulative = cpu_tensor(values["actual_seq_lengths_query"]).long().reshape(-1)
        cp = getattr(meta, "dsa_cp_context", None)
        lo, hi = row_window(selected.shape[0], self.active["actual"], cp)
        shape = self.logical_topk["shape"]
        if shape[0] != selected.shape[0] or math.prod(shape[1:]) != math.prod(selected.shape[1:]):
            raise ValueError("Attention selection and indexer output shape differ")
        for entry in self.active["entries"]:
            start, end = max(lo, entry["start"]), min(hi, entry["end"])
            end = max(start, end)
            logical, query_positions, query_capture = self.logical_topk["snapshots"][entry["archive"].root]
            self.record(
                entry,
                logical,
                layer,
                "attention",
                "logical_topk",
                query_positions,
                layout="sequence_sharded" if cp is not None else "rank_local",
                row_capture=query_capture,
            )
            a, b = start - lo, end - lo
            physical = cpu_tensor(sampled_tensor(selected[a:b], query_capture)).long().flatten(start_dim=1)
            original = logical.long().flatten(start_dim=1)
            owners = torch.bucketize(
                torch.tensor(query_capture["source_indices"], dtype=torch.long) + a, cumulative, right=True
            )
            positions = torch.tensor(query_positions, dtype=torch.long)
            valid = (physical >= 0) & (original >= 0) & (physical < lengths[owners, None])
            valid &= original <= positions[:, None]
            # Filter before constructing/gathering KV pairs. P's current-KV
            # snapshots and D's consumed KV now use identical absolute offsets.
            if self.max_token_rows:
                valid &= original.remainder(4096) < self.max_token_rows
            block_size = int(values["key"].shape[1])
            blocks = physical.clamp_min(0) // block_size
            if valid.any() and int(blocks[valid].max()) >= table.shape[1]:
                raise ValueError("Attention selection exceeds block table")
            slots = table[owners[:, None], blocks.clamp_max(max(0, table.shape[1] - 1))]
            slots = slots * block_size + physical.clamp_min(0) % block_size
            pairs = unique_kv_pairs(original[valid], slots[valid])
            sampled, capture = sample_rows(
                pairs[:, 0].tolist(), self.max_token_rows, len(entry["call"]["context_token_ids"]), aligned=True
            )
            capture["logical_chunk_filter"] = bool(
                self.max_token_rows and len(entry["call"]["context_token_ids"]) > self.max_token_rows
            )
            capture["query_rows"] = query_capture
            pairs = sampled_tensor(pairs, capture)
            alias_rows = int((pairs[1:, 0] == pairs[:-1, 0]).sum())
            archive = entry["archive"]
            if alias_rows and "first_kv_alias" not in archive.metadata:
                archive.metadata["first_kv_alias"] = dict(
                    call=entry["call"]["call"], layer=layer, extra_rows=alias_rows
                )
                archive.flush()
                print(
                    f"[PD_DUMP] {self.role} tp={self.rank_info['tp_rank']} "
                    f"call={entry['call']['call']} layer={layer} kv_alias_rows={alias_rows}; within captured rows",
                    flush=True,
                )
            for key, name in (("key", "nope"), ("key_rope", "rope")):
                cache = values[key]
                flat = cache.reshape(-1, *cache.shape[2:])
                flat, features = feature_prefix(flat, "kv_consumed", name, self.max_features)
                if pairs.numel() and (pairs[:, 1].min() < 0 or pairs[:, 1].max() >= flat.shape[0]):
                    raise ValueError("Attention KV physical slot out of bounds")
                rows = flat.index_select(0, pairs[:, 1].to(cache.device))
                # Numerical disagreement is evidence, not a recorder failure.
                # Preserve physical aliases in the captured rows, including
                # NaNs. Sampling metadata explicitly marks unobserved rows.
                self.record(
                    entry,
                    rows,
                    layer,
                    "kv_consumed",
                    name,
                    pairs[:, 0].tolist(),
                    layout="replicated",
                    physical_slots=pairs[:, 1].tolist(),
                    row_capture=capture,
                    feature_capture=features,
                )

    def kernel_factory(self, kind, tuple_result=False):
        def factory(original):
            @functools.wraps(original)
            def kernel(*args, **kwargs):
                if not self.active or not self.sfa_stack:
                    return original(*args, **kwargs)
                if args:
                    raise ValueError("PD dump sparse kernel requires the known keyword argument schema")
                layer, meta = self.sfa_stack[-1]
                if kind == "indexer":
                    for key in ("query", "weights", "query_dequant_scale"):
                        if key in kwargs:
                            self.emit(kwargs[key], layer, kind, key, meta)
                    self.indexer_kv(
                        kwargs["key"], kwargs["block_table"], kwargs["actual_seq_lengths_key"], layer, "key"
                    )
                    if "key_dequant_scale" in kwargs:
                        self.indexer_kv(
                            kwargs["key_dequant_scale"],
                            kwargs["block_table"],
                            kwargs["actual_seq_lengths_key"],
                            layer,
                            "scale",
                        )
                else:
                    self.current_kv(self.sfa_caches[-1], layer, meta)
                    self.emit(kwargs["query"], layer, kind, "query_nope", meta)
                    self.emit(kwargs["query_rope"], layer, kind, "query_rope", meta)
                    self.emit(kwargs["sparse_indices"], layer, "mapping", "sparse_indices", meta, mapping_only=True)
                    self.attention_kv(kwargs, layer, meta)
                result = original(*args, **kwargs)
                value = result[0] if tuple_result else result
                self.emit(value, layer, kind, "topk" if kind == "indexer" else "output", meta)
                if kind == "indexer":
                    self.capture_topk(value, meta)
                return result

            return kernel

        return factory

    def logits_factory(self, original):
        @functools.wraps(original)
        def logits(*args, **kwargs):
            try:
                result = original(*args, **kwargs)
                if self.last is None:
                    return result
                if result is None:
                    raise ValueError("PD tensor dump requires observable main model logits on every TP rank")
                indices = cpu_tensor(self.runner.logits_indices).long().reshape(-1).tolist()
                for entry in self.last["entries"]:
                    rows = [i for i, index in enumerate(indices) if entry["start"] <= index < entry["end"]]
                    positions = [self.last["positions"][indices[i]] for i in rows]
                    selected = torch.tensor(rows, dtype=torch.long, device=result.device)
                    self.record(entry, result.index_select(0, selected), -1, "logits", "output", positions)
                    entry["archive"].flush()
                return result
            except BaseException as error:
                if self.last is not None:
                    for entry in self.last["entries"]:
                        entry["archive"].fail(f"logits {type(error).__name__}: {error}")
                raise

        return logits

    def sampled(self, request_ids, token_ids):
        if self.last is None:
            return
        try:
            # The runner supplies accepted CPU tokens AFTER discarding unfinished
            # prefill rows and rejection-sampler padding, never raw sampler output.
            if len(token_ids) != len(request_ids):
                raise ValueError("Sampled request rows differ from the captured batch")
            captured = {entry["archive"].metadata["internal_request_id"] for entry in self.last["entries"]}
            if len(set(request_ids)) != len(request_ids) or not captured.issubset(request_ids):
                raise ValueError("Sampled request identities differ from the captured batch")
            for internal, row in zip(request_ids, token_ids):
                if internal not in captured:
                    continue
                archive = self.archives.get(internal)
                if archive is None:
                    continue
                valid = [int(token) for token in row if 0 <= token < self.runner.input_batch.vocab_size]
                with (archive.root / "sampled.jsonl").open("a", encoding="utf-8") as stream:
                    entry = next(e for e in self.last["entries"] if e["archive"] is archive)
                    stream.write(json.dumps({"after_call": entry["call"]["call"], "token_ids": valid}) + "\n")
            for entry in self.last["entries"]:
                entry["archive"].end(entry["call"], entry["observed"])
        except BaseException as error:
            for entry in self.last["entries"]:
                entry["archive"].fail(f"sampling {type(error).__name__}: {error}")
            raise
        finally:
            self.last = None


def install_pd_tensor_dump(runner, root):
    # Imports stay here: a disabled diagnostic must not initialize LMCache,
    # torch_npu or any recording state as an import side effect.
    import torch_npu
    from lmcache.integration.vllm.utils import lmcache_get_or_create_config
    from vllm.distributed import get_dp_group, get_tp_group
    from vllm.forward_context import get_forward_context

    from vllm_ascend import envs
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

    if not Path(root).is_absolute():
        raise ValueError("VLLM_ASCEND_PD_TENSOR_DUMP_DIR must be an absolute run directory")
    if envs.VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD and envs.VLLM_ASCEND_DSA_OFFLOAD_FREE_PAGED:
        raise ValueError("PD tensor dump does not support the separate FREE_PAGED latent-pool path")
    if not runner.model_config.enforce_eager or int(runner.compilation_config.mode) != 0:
        raise ValueError(
            'PD tensor dump requires --enforce-eager --compilation-config \'{"mode":0,"cudagraph_mode":"NONE"}\''
        )
    if runner.use_async_scheduling:
        raise ValueError("PD tensor dump requires --no-async-scheduling for unambiguous sampled token records")
    parallel = runner.vllm_config.parallel_config
    if any(
        int(getattr(parallel, name, 1)) != 1
        for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
    ):
        raise ValueError("PD tensor dump supports TP/DP/EP with PP=PCP=DCP=1")
    role = {"sender": "P", "receiver": "D"}.get(str(lmcache_get_or_create_config().pd_role))
    if role is None:
        raise ValueError("PD tensor dump needs LMCache pd_role sender or receiver")
    tp, dp = get_tp_group(), get_dp_group()
    probe = PDTensorDump(
        runner,
        root,
        role,
        dict(
            host=socket.gethostname(),
            pid=os.getpid(),
            tp_rank=tp.rank_in_group,
            tp_size=tp.world_size,
            dp_rank=dp.rank_in_group,
            dp_size=dp.world_size,
        ),
        get_forward_context,
        max_token_rows=envs.VLLM_ASCEND_PD_TENSOR_DUMP_MAX_TOKENS,
        max_features=envs.VLLM_ASCEND_PD_TENSOR_DUMP_MAX_FEATURES,
    )
    try:
        probe.install(AscendSFAImpl, torch_npu)
    except BaseException:
        probe.restore()
        raise
    return probe
