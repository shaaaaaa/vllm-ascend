# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only worker for real eight-layer eager/root-replay comparisons.

Selected explicitly by tools/sfa_full_graph_parity.py. Normal serving never
imports this worker or installs its hooks. Hooks on decoder modules contain
only tensor copies/counter increments, traced into the existing target graph;
SFA probes run inside existing opaque SFA ops during capture, not live replay.
"""

import importlib
import inspect
import json
import re
from pathlib import Path
from unittest.mock import patch

import torch
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

from vllm_ascend import envs
from vllm_ascend.attention.sfa_parity import (
    DeviceSnapshot,
    ParityError,
    compare_step,
    coordinated_check,
    gather_sparse_kv,
    residual_trace_report,
    weight_fingerprint,
)
from vllm_ascend.attention.sfa_prefill_checkpoint import (
    CacheBinding,
    assert_same_tree,
    capture_caches,
    copy_tree,
    plan_cache_restore,
    restore_prefill,
    validate_callbacks,
    verify_cache_restore,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.worker.worker import NPUWorker

TARGET_LAYERS = 8
QUERY_WIDTH = 2
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def remap_mtp_quant_description(description: dict, source_start: int, num_mtp_layers: int) -> dict:
    """Move the original MTP quantization namespace into the truncated fixture.

    DeepSeekMTP constructs its layers after the *target* depth (eight here),
    while the checkpoint description still names them after the original
    target depth. Replace the whole destination namespace: ordinary decoder
    layer 8 may have different quantization, including FA/indexer metadata.
    Never modify the caller's description or invent a FLOAT fallback.
    """
    if type(source_start) is not int or source_start < TARGET_LAYERS:
        raise ValueError("Original model num_hidden_layers must be an integer >= 8")
    if type(num_mtp_layers) is not int or num_mtp_layers < 1:
        raise ValueError("The parity fixture requires at least one MTP layer")
    destinations = tuple(f"model.layers.{TARGET_LAYERS + i}." for i in range(num_mtp_layers))
    result = {key: value for key, value in description.items() if not key.startswith(destinations)}
    for offset, destination in enumerate(destinations):
        source = f"model.layers.{source_start + offset}."
        if source + "head.weight" not in description and source + "shared_head.head.weight" not in description:
            raise ValueError(f"Missing original MTP head quantization at {source}; refusing a fallback")
        result.update(
            (destination + key[len(source) :], value) for key, value in description.items() if key.startswith(source)
        )
    return result


def deterministic_dummy_load(original, loader, model, model_config) -> None:
    """Also initialize integer dummy weights, which upstream leaves untouched.

    This is test-only and happens before quantization's post-load processing.
    Both processes use the identical fixture; real checkpoints are not changed.
    """
    original(loader, model, model_config)
    with torch.no_grad():
        # Do not randomize integer buffers (e.g. routing maps or position IDs).
        for value in model.parameters():
            if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
                continue
            generator = torch.Generator(device="cpu").manual_seed(1234)
            if value.ndim == 0:
                value = value.reshape(1)
            row_elements = value[0].numel() if value.shape[0] else 1
            chunk_rows = max(1, HASH_CHUNK_BYTES // max(1, row_elements * 8))
            # CPU generation avoids relying on NPU random_ support for packed
            # integer dtypes. Chunking avoids a whole-model host allocation.
            for chunk in value.split(chunk_rows):
                sample = torch.randint(0, 8, chunk.shape, generator=generator, dtype=torch.int64)
                chunk.copy_(sample.to(dtype=value.dtype))


class LayerSnapshots:
    """Fixed buffers for one complete decoder layer and its sparse attention."""

    def __init__(
        self, layer, impl: AscendSFAImpl, index: int, rows: int, device: torch.device, *, trace_residual: bool = False
    ) -> None:
        self.index = index
        self.trace_residual = trace_residual
        # The first parameter may be packed INT4 weights; activations follow RMSNorm.
        dtype = layer.input_layernorm.weight.dtype
        hidden = layer.input_layernorm.weight.numel()
        self.probes: dict[str, DeviceSnapshot] = {}

        def allocate(name, shape, tensor_dtype=dtype):
            self.probes[name] = DeviceSnapshot(shape, dtype=tensor_dtype, device=device)

        for phase in ("input", "output"):
            for component in ("hidden", "residual"):
                allocate(f"{phase}.{component}", (rows, hidden))
        if trace_residual:
            for name in (
                "input_norm.hidden",
                "input_norm.residual",
                "attention.output",
                "post_norm.input.hidden",
                "post_norm.input.residual",
                "post_norm.output.hidden",
                "post_norm.output.residual",
            ):
                allocate(name, (rows, hidden))
        allocate("topk", (QUERY_WIDTH, impl.index_topk), torch.int32)
        allocate("q_nope", (QUERY_WIDTH, impl.local_num_heads, impl.kv_lora_rank))
        allocate("q_pe", (QUERY_WIDTH, impl.local_num_heads, impl.qk_rope_head_dim))
        allocate("kv_nope", (QUERY_WIDTH, impl.index_topk, impl.kv_lora_rank))
        allocate("kv_pe", (QUERY_WIDTH, impl.index_topk, impl.qk_rope_head_dim))
        allocate("valid", (QUERY_WIDTH, impl.index_topk), torch.bool)
        allocate("invalid_address", (QUERY_WIDTH, impl.index_topk), torch.bool)
        allocate("physical_slots", (QUERY_WIDTH, impl.index_topk), torch.long)
        allocate("remapped_topk", (QUERY_WIDTH, impl.index_topk), torch.int32)
        allocate("boundary", (QUERY_WIDTH,), torch.long)
        allocate("miss_count", (1,), torch.int32)
        allocate("miss_tokens", (1, QUERY_WIDTH * impl.index_topk), torch.int32)
        allocate("target_slots", (1, QUERY_WIDTH * impl.index_topk), torch.long)

        # These hooks execute Python during eager/tracing/capture only. During
        # root replay the recorded copy_ and add_ operations refresh the probes.
        layer.register_forward_pre_hook(self.before_layer, with_kwargs=True)
        layer.register_forward_hook(self.after_layer)
        if trace_residual:
            # These extra observable values can affect compiler fusion. Keep
            # this opt-in diagnosis separate from the original acceptance run.
            layer.input_layernorm.register_forward_hook(self.after_input_norm)
            layer.self_attn.register_forward_hook(self.after_attention)
            layer.post_attention_layernorm.register_forward_pre_hook(self.before_post_norm, with_kwargs=True)
            layer.post_attention_layernorm.register_forward_hook(self.after_post_norm)
        self.install_sfa_probes(impl)

    def before_layer(self, module, args, kwargs) -> None:
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[1]
        residual = kwargs["residual"] if "residual" in kwargs else args[2]
        self.probes["input.hidden"].write(hidden)
        if residual is None:
            self.probes["input.residual"].write_absent()
        else:
            self.probes["input.residual"].write(residual)

    def after_layer(self, module, args, output) -> None:
        hidden, residual = output
        self.probes["output.hidden"].write(hidden)
        self.probes["output.residual"].write(residual)

    def _write_trace(self, name, value) -> None:
        probe = self.probes[name]
        if value.ndim != 2 or value.dtype != probe.value.dtype:
            raise ParityError(f"Residual probe {name} requires a 2D activation of dtype {probe.value.dtype}")
        probe.write(value)

    def after_input_norm(self, module, args, output) -> None:
        if isinstance(output, tuple):
            hidden, residual = output
            self._write_trace("input_norm.residual", residual)
        else:
            # First decoder layer calls RMSNorm without a residual operand.
            hidden = output
            self.probes["input_norm.residual"].write_absent()
        self._write_trace("input_norm.hidden", hidden)

    def after_attention(self, module, args, output) -> None:
        self._write_trace("attention.output", output)

    def before_post_norm(self, module, args, kwargs) -> None:
        hidden = kwargs["x"] if "x" in kwargs else args[0]
        residual = kwargs["residual"] if "residual" in kwargs else args[1]
        self._write_trace("post_norm.input.hidden", hidden)
        self._write_trace("post_norm.input.residual", residual)

    def after_post_norm(self, module, args, output) -> None:
        hidden, residual = output
        self._write_trace("post_norm.output.residual", residual)
        self._write_trace("post_norm.output.hidden", hidden)

    def install_sfa_probes(self, impl: AscendSFAImpl) -> None:
        indexer = impl.indexer_select_post_process
        planner = impl._prepare_decode_sparse_indices
        attention = impl._execute_sparse_flash_attention_process

        def indexer_with_probe(*args, **kwargs):
            result = indexer(*args, **kwargs)
            if result.shape[0] <= QUERY_WIDTH:
                self.probes["topk"].write(result.reshape(result.shape[0], -1))
            return result

        def planner_with_probe(topk, boundary, *args, **kwargs):
            result = planner(topk, boundary, *args, **kwargs)
            if topk.shape[0] <= QUERY_WIDTH:
                self.probes["boundary"].write(boundary.reshape(-1).to(torch.long))
                _, tokens, counts, slots = result
                if tokens is not None and counts is not None and slots is not None:
                    counts = counts.reshape(-1)[:1]
                    valid = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < counts[:, None]
                    self.probes["miss_count"].write(counts)
                    self.probes["miss_tokens"].write_padded(torch.where(valid, tokens[:1], -1))
                    self.probes["target_slots"].write_padded(torch.where(valid, slots[:1], -1))
            return result

        def attention_with_probe(q, q_pe, kv, topk, metadata, query_ends, seq_lens, **kwargs):
            if q.shape[0] <= QUERY_WIDTH:
                table = kwargs.get("block_table_override")
                if table is None:
                    table = metadata.block_table
                if kwargs.get("kv_override") is not None:
                    raise ParityError("Parity worker requires the native SHRINK_LATENT=2 attention path")
                nope, valid, invalid, slots = gather_sparse_kv(kv[0], topk, table, query_ends)
                pe, _, _, _ = gather_sparse_kv(kv[1], topk, table, query_ends)
                for name, value in (
                    ("q_nope", q),
                    ("q_pe", q_pe),
                    ("kv_nope", nope),
                    ("kv_pe", pe),
                    ("valid", valid),
                    ("invalid_address", invalid),
                    ("physical_slots", slots),
                    ("remapped_topk", topk.reshape(topk.shape[0], -1)),
                ):
                    self.probes[name].write(value)
            return attention(q, q_pe, kv, topk, metadata, query_ends, seq_lens, **kwargs)

        impl.indexer_select_post_process = indexer_with_probe
        impl._prepare_decode_sparse_indices = planner_with_probe
        impl._execute_sparse_flash_attention_process = attention_with_probe

    def reset(self) -> None:
        for probe in self.probes.values():
            probe.reset()

    def read(self, rows: int, decode: bool) -> tuple[dict, dict]:
        values, addresses = {}, {}
        # Numerical data in execution order, so the first mismatch is useful.
        names = ["input.hidden", "input.residual"]
        if self.trace_residual:
            names += ["input_norm.residual", "input_norm.hidden"]
        if decode:
            names += [
                "q_nope",
                "q_pe",
                "topk",
                "boundary",
                "miss_count",
                "miss_tokens",
                "valid",
                "invalid_address",
                "kv_nope",
                "kv_pe",
                "physical_slots",
                "remapped_topk",
                "target_slots",
            ]
        if self.trace_residual:
            names += [
                "attention.output",
                "post_norm.input.hidden",
                "post_norm.input.residual",
                "post_norm.output.residual",
                "post_norm.output.hidden",
            ]
        names += ["output.residual", "output.hidden"]
        for name in names:
            label = f"layer={self.index} {name}"
            count = 1 if name in ("miss_count", "miss_tokens", "target_slots") else rows
            value = self.probes[name].read(count, label=label)
            if name == "invalid_address" and value.any():
                raise ParityError(f"{label}: attention is addressing an invalid KV location")
            # A Q1 eager planner and the bounded-Q2 graph planner may have
            # different residency/miss decisions while consuming identical KV.
            # Compare logical top-k and KV, not their allocation strategies.
            if name in ("physical_slots", "remapped_topk", "target_slots", "miss_count", "miss_tokens"):
                addresses[label] = value
            else:
                values[label] = value
        return values, addresses


class SFAParityWorker(NPUWorker):
    """Isolated single-host TP test worker; never enabled in normal serving."""

    def _check(self, check, phase):
        return coordinated_check(check, group=self.parity_group, phase=phase)

    def load_model(self) -> None:
        options = self.vllm_config.additional_config.get("sfa_parity")
        if not isinstance(options, dict):
            raise ValueError("SFAParityWorker must be launched by the parity test driver")
        parallel = self.vllm_config.parallel_config
        if (
            parallel.tensor_parallel_size not in (1, 2, 4, 8)
            or parallel.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.enable_expert_parallel
        ):
            raise ValueError("This parity test requires TP=1/2/4/8, DP=1, PP=1 and EP disabled")
        self.parity_group = get_tp_group()
        self.parity_rank = self.parity_group.rank_in_group
        self.parity_tp_size = self.parity_group.world_size
        self.parity_options = options
        self.parity_step = 0
        self.parity_prefill_steps = 0
        self.parity_prefill_tokens = 0
        self.parity_prefill_model_calls = 0
        self.parity_prefill_imports = 0
        self.parity_draft_prefill_model_calls = 0
        self.parity_draft_prefill_imports = 0
        self.parity_last_target_decode = None
        self.parity_last_target_step = None
        self.parity_decode_steps = 0
        self.parity_decode_observations = 0
        self.parity_q2_steps = 0
        self.parity_draft_calls = 0
        self.parity_transfers = [0] * TARGET_LAYERS
        self.parity_directory = Path(options["reference"]) / f"rank-{self.parity_rank}"
        self.parity_is_graph = options["mode"] == "graph"
        if self.parity_is_graph != bool(envs.VLLM_ASCEND_SFA_FULL_GRAPH):
            raise ValueError("Parity mode and full-graph flag disagree")
        if not self.parity_is_graph and (envs.VLLM_ASCEND_SFA_STAGED_GRAPH or not self.model_config.enforce_eager):
            raise ValueError("The reference must disable both staged/full graph and enforce eager")
        self._check(self._prepare_quant_config, "startup MTP quantization")
        original_load = DummyModelLoader.load_weights

        def load(loader, model, model_config):
            deterministic_dummy_load(original_load, loader, model, model_config)

        with patch.object(DummyModelLoader, "load_weights", load):
            super().load_model()
        # Separate pending post-load kernel failures from fingerprint readback.
        self._check(torch.npu.synchronize, "startup post-load synchronization")
        self._check(self._prepare_probes, "startup weights/probes")
        runner = self.model_runner
        original_forward = runner._model_forward
        signature = inspect.signature(original_forward)

        def forward(*args, **kwargs):
            return self._parity_forward(original_forward, signature, *args, **kwargs)

        original_propose = runner.propose_draft_token_ids

        def propose(*args, **kwargs):
            tokens = original_propose(*args, **kwargs)
            return self._check(lambda: self._parity_draft_tokens(tokens), f"step={self.parity_step} after draft")

        runner._model_forward = forward
        runner.propose_draft_token_ids = propose
        original_draft = runner.drafter._run_mtp_draft_layer_with_diagnostics

        def draft(model_kwargs, **kwargs):
            return self._parity_draft_forward(original_draft, model_kwargs, **kwargs)

        runner.drafter._run_mtp_draft_layer_with_diagnostics = draft

    def _parity_draft_tokens(self, tokens):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim < 1 or tokens.shape[-1] != 1:
            raise ParityError("MTP did not produce one draft token per request")
        self.parity_draft_calls += 1
        if self.parity_options.get("compare_output", False):
            # Output comparison must keep the model's real proposals. Returning
            # a fixed token here could conceal acceptance/output differences.
            return tokens
        # Only the choice is overridden in the original layer-parity test.
        return torch.full_like(tokens, self.parity_options["token_id"])

    def _parity_forward(self, original_forward, signature, *args, **kwargs):
        runner = self.model_runner
        context = get_forward_context()
        live = runner.input_batch.num_reqs > 0 and not getattr(context, "staged_sfa_graph_dummy_run", False)
        if not live:
            # Profiling, compilation and root capture retain the production path.
            return original_forward(*args, **kwargs)
        state = self._check(
            lambda: self._prepare_step(signature.bind(*args, **kwargs).arguments, context),
            f"step={self.parity_step} before target",
        )
        before = runner._sfa_full_graph.replay_count
        self.parity_last_target_decode = state["decode"]
        self.parity_last_target_step = self.parity_step
        if state["decode"]:
            # Leave the production eager/root-replay paths completely intact.
            # No failure collective around NPU/HCCL model execution itself.
            result = original_forward(*args, **kwargs)
        else:
            result = self._prefill_pass(
                "target",
                self.parity_step,
                self.parity_attention_names,
                signature.bind(*args, **kwargs).arguments,
                lambda: original_forward(*args, **kwargs),
            )
            self.parity_prefill_model_calls += int(not self.parity_is_graph)
            self.parity_prefill_imports += int(self.parity_is_graph)
        replays = runner._sfa_full_graph.replay_count - before
        if state["decode"]:
            self._check(lambda: self._observe_step(state, result, replays), f"step={self.parity_step} after target")
        elif replays:
            raise ParityError("Prefill checkpoint import must not replay the target graph")
        if self.parity_is_graph and self.parity_rank == 0:
            if state["decode"]:
                status = "OBSERVED (output-only)" if self.parity_options.get("compare_output", False) else "PASS"
                print(
                    f"[SFA_PARITY] step={self.parity_step} phase=decode rows={state['rows']} "
                    f"ranks={self.parity_tp_size} layers=8 {status} replay_per_rank={replays}",
                    flush=True,
                )
            else:
                print(
                    f"[SFA_PARITY] step={self.parity_step} prefill=IMPORTED rows={state['rows']} "
                    "target_prefill_model_calls=0",
                    flush=True,
                )
        self.parity_step += 1
        self.parity_prefill_steps += int(not state["decode"])
        self.parity_prefill_tokens += state["rows"] if not state["decode"] else 0
        self.parity_decode_steps += int(state["decode"])
        self.parity_q2_steps += int(state["decode"] and state["rows"] == QUERY_WIDTH)
        return result

    def _parity_draft_forward(self, original, model_kwargs, **kwargs):
        # MTP's initial cache is part of the common decode starting point too.
        # Startup dummy/profile runs and every real decode draft are untouched.
        if self.model_runner.input_batch.num_reqs == 0 or self.parity_last_target_decode is not False:
            return original(model_kwargs, **kwargs)
        if kwargs["draft_step"] != 0:
            raise ParityError("Prefill checkpoint requires MTP=1")
        result = self._prefill_pass(
            "draft",
            self.parity_last_target_step,
            self.model_runner.drafter.attn_layer_names,
            {
                "model_kwargs": model_kwargs,
                "draft_step": kwargs["draft_step"],
                "runtime_inputs": kwargs["runtime_inputs"],
            },
            lambda: original(model_kwargs, **kwargs),
        )
        self.parity_draft_prefill_model_calls += int(not self.parity_is_graph)
        self.parity_draft_prefill_imports += int(self.parity_is_graph)
        return result

    def _prefill_bindings(self, context, attention_names):
        bindings, persistent, metadata = {}, {}, {}
        for name in attention_names:
            layer = context.no_compile_layers[name]
            impl = layer.impl
            if not isinstance(impl, AscendSFAImpl) or not impl.dsa_offload_unbundle:
                raise ParityError("Prefill checkpoint requires unbundled SFA caches")
            caches, index_name, enabled = impl._cross_layer_kv_cache(name, layer.kv_cache[context.virtual_engine])
            if len(caches) != 3 or not enabled:
                raise ParityError("Prefill checkpoint requires latent K/PE and LMCache-enabled indexer")
            item = context.attn_metadata[name]
            index_table = item.indexer_block_table if item.indexer_block_table is not None else item.block_table
            bindings[name] = CacheBinding(tuple(caches[:2]), item.block_table[:1])
            bindings[index_name] = CacheBinding((caches[2],), index_table[:1])
            metadata[name] = {
                key: copy_tree(getattr(item, key, None))
                for key in (
                    "num_actual_tokens",
                    "num_decode_tokens",
                    "seq_lens",
                    "cum_query_lens",
                    "resident_state_indices",
                    "resident_state_generations",
                )
            }
            resident = impl._sorted_resident_state
            if resident is not None:
                # These slots are scratch-relative, not physical KV addresses.
                # Do not overwrite the graph-capture-only dummy state rows.
                for field in ("tokens", "slots", "counts", "generations"):
                    persistent[f"{name}.{field}"] = getattr(resident, field)[: resident.dummy_state_base]
        return bindings, persistent, metadata

    def _prefill_pass(self, kind, step, attention_names, inputs, runnable):
        """Compute/export once; the graph process can only import, never compute."""
        context = get_forward_context()
        phase = f"{kind} prefill checkpoint step={step}"
        path = self.parity_directory / f"{kind}-prefill-{step:06d}.pt"
        bindings, persistent, metadata = self._check(
            lambda: self._prefill_bindings(context, attention_names), phase + " bindings"
        )
        header = self._check(
            lambda: {
                "schema": 1,
                "kind": kind,
                "step": step,
                "rank": self.parity_rank,
                "tp_size": self.parity_tp_size,
                "inputs": copy_tree(inputs),
                "metadata": metadata,
            },
            phase + " inputs",
        )
        sfa = importlib.import_module("vllm_ascend.attention.sfa_v1")
        wait, save = sfa.wait_for_kv_layer_from_connector, sfa.maybe_save_kv_layer_to_connector
        if self.parity_is_graph:

            def prepare():
                snapshot = torch.load(path, map_location="cpu", weights_only=True)
                assert_same_tree(snapshot["header"], header, phase + " inputs")
                validate_callbacks(snapshot["callbacks"], bindings)
                plan = plan_cache_restore(snapshot["caches"], bindings)
                if snapshot["persistent"].keys() != persistent.keys():
                    raise ParityError("Prefill resident-state coverage differs")
                for name, value in persistent.items():
                    ref = snapshot["persistent"][name]
                    if ref.shape != value.shape or ref.dtype != value.dtype:
                        raise ParityError(f"{name}: prefill resident-state layout differs")
                return snapshot, plan

            snapshot, plan = self._check(prepare, phase + " validate")
            # No model, indexer/top-k, MLP or attention execution in this branch.
            # Rebuild generators/CPU sources through their real public callbacks.
            restore_prefill(snapshot, bindings, plan, wait=wait, save=save)
            for name, value in persistent.items():
                value.copy_(snapshot["persistent"][name])
            torch.set_rng_state(snapshot["cpu_rng"])
            torch.npu.set_rng_state(snapshot["npu_rng"])
            context.moe_layer_index = snapshot["moe_layer_index"]
            output = copy_tree(snapshot["output"], self.device)

            def verify():
                verify_cache_restore(snapshot["caches"], bindings, plan)
                assert_same_tree(snapshot["persistent"], persistent, phase + " resident state")
                assert_same_tree(snapshot["output"], output, phase + " output")

            self._check(verify, phase + " imported bytes")
            return output

        def unused_checkpoint():
            if path.exists():
                raise ParityError(f"Refusing to compute the same prefill twice: {path.name}")

        self._check(unused_checkpoint, phase + " single computation")

        callbacks = []

        def record_wait(name, *args, **kwargs):
            # Diagnose unsupported sparse-load callbacks AFTER the model returns,
            # not between TP collectives. Normal prefill waits take only a name.
            callbacks.append(("unsupported_wait" if args or kwargs else "wait", name))
            return wait(name, *args, **kwargs)

        def record_save(name, caches):
            callbacks.append(("save", name))
            return save(name, caches)

        with (
            patch.object(sfa, "wait_for_kv_layer_from_connector", record_wait),
            patch.object(sfa, "maybe_save_kv_layer_to_connector", record_save),
        ):
            output = runnable()  # Original eager prefill, exactly once.

        def export():
            torch.npu.synchronize()
            validate_callbacks(callbacks, bindings)
            torch.save(
                {
                    "header": header,
                    "callbacks": callbacks,
                    "caches": capture_caches(bindings),
                    "persistent": copy_tree(persistent),
                    "output": copy_tree(output),
                    "cpu_rng": torch.get_rng_state(),
                    "npu_rng": torch.npu.get_rng_state(),
                    "moe_layer_index": getattr(context, "moe_layer_index", 0),
                },
                path,
            )

        self._check(export, phase + " export")
        return output

    def _prepare_quant_config(self) -> None:
        # Test-worker-only import; normal serving keeps the checkpoint config.
        from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig

        config = self.vllm_config
        if (
            config.load_config.load_format != "dummy"
            or config.model_config.hf_config.num_hidden_layers != TARGET_LAYERS
        ):
            raise ValueError("MTP quantization remapping is only for the eight-layer dummy parity fixture")
        speculative = config.speculative_config
        if speculative is None or speculative.num_speculative_tokens != 1:
            raise ValueError("SFA parity requires MTP=1")
        draft = speculative.draft_model_config.hf_config
        if draft.model_type != "deepseek_mtp" or not isinstance(config.quant_config, AscendModelSlimConfig):
            raise ValueError("SFA parity requires a DeepSeek MTP draft with Ascend ModelSlim quantization")
        original = json.loads((Path(config.model_config.model) / "config.json").read_text(encoding="utf-8"))
        description = remap_mtp_quant_description(
            config.quant_config.quant_description, original["num_hidden_layers"], draft.num_nextn_predict_layers
        )
        # Reconstruct to refresh FA/indexer layer lists and shared-head/packed
        # aliases as well. Existing target layer 0..7 descriptions stay intact.
        config.quant_config = AscendModelSlimConfig(description)

    def _prepare_probes(self):
        runner = self.model_runner
        if runner.speculative_config is None or runner.speculative_config.num_speculative_tokens != 1:
            raise ValueError("SFA parity requires MTP=1")
        model = runner.get_model()
        manifest = {
            "rank": self.parity_rank,
            "tp_size": self.parity_tp_size,
            "target": weight_fingerprint(model),
            "draft": weight_fingerprint(runner.drafter.model),
        }
        manifest_path = self.parity_directory / "weights.json"
        if self.parity_is_graph:
            if json.loads(manifest_path.read_text()) != manifest:
                raise ParityError("Eager/graph weights differ; refusing to compare different models")
        else:
            self.parity_directory.mkdir(parents=True, exist_ok=False)
            manifest_path.write_text(json.dumps(manifest))
        self.parity_layers = []
        self.parity_attention_names = []
        for name, layer in model.named_modules():
            match = re.search(r"(?:^|\.)layers\.(\d+)$", name)
            if not match or not hasattr(layer, "input_layernorm"):
                continue
            implementations = [
                (m.layer_name, m.impl) for m in layer.modules() if isinstance(getattr(m, "impl", None), AscendSFAImpl)
            ]
            if len(implementations) != 1:
                raise ParityError(f"{name}: expected one SFA implementation, got {len(implementations)}")
            attn_name, impl = implementations[0]
            self.parity_attention_names.append(attn_name)
            self.parity_layers.append(
                LayerSnapshots(
                    layer,
                    impl,
                    int(match[1]),
                    self.vllm_config.scheduler_config.max_num_batched_tokens,
                    self.device,
                    trace_residual=getattr(self, "parity_options", {}).get("trace_residual", False),
                )
            )
        if [layer.index for layer in self.parity_layers] != list(range(TARGET_LAYERS)):
            raise ParityError("Parity worker did not find all eight target decoder layers")
        # These persistent allocations happen after load_model's weight memory
        # measurement. Include them when the worker budgets its KV cache.
        snapshot_bytes = sum(
            probe.value.numel() * probe.value.element_size() + probe.writes.numel() * probe.writes.element_size()
            for layer in self.parity_layers
            for probe in layer.probes.values()
        )
        runner.model_memory_usage = getattr(runner, "model_memory_usage", 0) + snapshot_bytes

    def _prepare_step(self, inputs, context):
        if self.model_runner.input_batch.num_reqs != 1:
            raise ParityError("Parity driver must submit one request at a time")
        metadata = context.attn_metadata[self.parity_attention_names[0]]
        rows = metadata.num_actual_tokens
        decode = metadata.num_decode_tokens > 0
        if decode and rows > QUERY_WIDTH:
            raise ParityError("Expected Q1/Q2 singleton decode")
        if not decode and (
            getattr(context, "staged_sfa_graph_key", None) is not None
            or getattr(context, "capturing", False)
            or getattr(context, "sfa_full_graph_active", False)
        ):
            raise ParityError("Live prefill unexpectedly routed to graph capture/replay")
        if decode and self.parity_is_graph and context.skip_compiled:
            raise ParityError("Decode must retain the production compiled root-replay path")
        state = {
            "rank": self.parity_rank,
            "tp_size": self.parity_tp_size,
            "step": self.parity_step,
            "decode": decode,
            "rows": rows,
            "layers": TARGET_LAYERS,
            "trace_residual": self.parity_options.get("trace_residual", False),
            "runtime_mode": str(getattr(context, "cudagraph_runtime_mode", None)),
            "input_ids": inputs["input_ids"][:rows].detach().cpu().clone(),
            "positions": inputs["positions"][:rows].detach().cpu().clone(),
            "seq_lens": metadata.seq_lens[:1].detach().cpu().clone(),
            "query_ends": metadata.cum_query_lens[:1].detach().cpu().clone(),
        }
        for layer in self.parity_layers:
            layer.reset()
        return state

    def _observe_step(self, state, result, replays):
        rows, decode = state["rows"], state["decode"]
        if not decode:
            raise ParityError("Prefill must use checkpoint export/import, never the decode comparator")
        state["root_replays"] = replays
        if decode and replays != int(self.parity_is_graph):
            raise ParityError(
                f"step={self.parity_step}: expected {int(self.parity_is_graph)} root replay, got {replays}"
            )
        torch.npu.current_stream().synchronize()
        state["tensors"], state["addresses"] = {}, {}
        for layer in self.parity_layers:
            tensors, addresses = layer.read(rows, decode)
            state["tensors"].update(tensors)
            state["addresses"].update(addresses)
            if decode:
                self.parity_transfers[layer.index] += int(addresses[f"layer={layer.index} miss_count"].sum())
        final_hidden = result[0] if isinstance(result, tuple) else result
        state["tensors"]["target.final_hidden"] = final_hidden[:rows].detach().cpu().clone()
        if self.parity_options.get("compare_output", False):
            # Free-running target/draft tokens can diverge, so step N need not
            # have the same prefix or speculative rows in the two processes.
            # Do not compare those states or force them back into alignment.
            # Keep observation freshness, address and replay checks above, and
            # reject invalid numerical data independently on each side.
            for name, value in state["tensors"].items():
                if not value.numel():
                    raise ParityError(f"{name}: empty output-mode observation")
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ParityError(f"{name}: output-mode observation contains NaN/Inf")
            self.parity_decode_observations += 1
            return
        path = self.parity_directory / f"step-{self.parity_step:06d}.pt"
        if self.parity_is_graph:
            if not path.is_file():
                raise ParityError(f"step={self.parity_step}: no matching eager step")
            reference = torch.load(path, map_location="cpu", weights_only=True)
            try:
                compare_step(reference, state, atol=self.parity_options["atol"], rtol=self.parity_options["rtol"])
            except ParityError as error:
                if self.parity_options.get("trace_residual", False):
                    # Read/format only after the complete forward. A diagnostic
                    # failure must not replace the original parity failure.
                    try:
                        report = residual_trace_report(
                            reference,
                            state,
                            error,
                            atol=self.parity_options["atol"],
                            rtol=self.parity_options["rtol"],
                        )
                        print("[SFA_TRACE] " + json.dumps(report), flush=True)
                    except Exception as diagnostic_error:
                        print(f"[SFA_TRACE] rank={self.parity_rank} diagnostic failed: {diagnostic_error}", flush=True)
                self._log_mapping_error(error, reference, state)
                raise
        else:
            torch.save(state, path)

    def _log_mapping_error(self, error, reference, state):
        layer_match = re.search(r"layer=(\d+)", error.label)
        if not (layer_match and state["decode"] and error.label.endswith(("topk", "kv_nope", "kv_pe", "valid"))):
            return
        prefix = f"layer={layer_match[1]}"
        row = error.index[0] if error.index else 0
        selected_column = error.index[1] if len(error.index) > 1 else 0
        for mode, snapshot in (("eager", reference), ("graph", state)):
            slots = snapshot["addresses"][f"{prefix} physical_slots"]
            column = min(selected_column, slots.shape[1] - 1)
            print(
                f"[SFA_PARITY] rank={self.parity_rank} {mode} {prefix} query_row={row} topk_column={column} "
                f"logical_token={snapshot['tensors'][f'{prefix} topk'][row, column].item()} "
                f"physical_slot={slots[row, column].item()} "
                f"miss_count={snapshot['addresses'][f'{prefix} miss_count'].tolist()}",
                flush=True,
            )

    def parity_summary(self) -> dict:
        """RPC completion gate: no replay, stale hooks, or no transfers cannot pass."""
        return self._check(self._local_summary, "final coverage")

    def _local_summary(self):
        if self.parity_prefill_steps < 1 or self.parity_prefill_tokens < 1:
            raise ParityError("No single-prefill checkpoint setup was checked")
        if self.parity_step != self.parity_prefill_steps + self.parity_decode_steps:
            raise ParityError("Incomplete prefill/decode phase coverage")
        if self.parity_decode_steps < 2 or self.parity_q2_steps < 2 or self.parity_draft_calls < 2:
            raise ParityError("Insufficient live decode/Q2/MTP coverage")
        if not all(count > 0 for count in self.parity_transfers):
            raise ParityError(f"No historical KV transfer observed in some target layers: {self.parity_transfers}")
        expected_prefills = len(list(self.parity_directory.glob("target-prefill-*.pt")))
        expected_draft_prefills = len(list(self.parity_directory.glob("draft-prefill-*.pt")))
        if expected_prefills != self.parity_prefill_steps or expected_draft_prefills < 1:
            raise ParityError("Incomplete target/MTP prefill checkpoint coverage")
        expected_counts = (
            (0, expected_prefills, 0, expected_draft_prefills)
            if self.parity_is_graph
            else (expected_prefills, 0, expected_draft_prefills, 0)
        )
        actual_counts = (
            self.parity_prefill_model_calls,
            self.parity_prefill_imports,
            self.parity_draft_prefill_model_calls,
            self.parity_draft_prefill_imports,
        )
        if actual_counts != expected_counts:
            raise ParityError(f"Prefill must compute once, then import: {actual_counts} != {expected_counts}")
        compare_output = self.parity_options.get("compare_output", False)
        if compare_output:
            if self.parity_decode_observations != self.parity_decode_steps:
                raise ParityError("Incomplete output-mode decode observation coverage")
        else:
            expected_steps = len(list(self.parity_directory.glob("step-*.pt"))) + expected_prefills
            if self.parity_step != expected_steps:
                raise ParityError(f"Incomplete step comparison: {self.parity_step} != {expected_steps}")
        return {
            "rank": self.parity_rank,
            "tp_size": self.parity_tp_size,
            "steps": self.parity_step,
            "prefill_steps": self.parity_prefill_steps,
            "prefill_tokens": self.parity_prefill_tokens,
            "prefill_model_calls": self.parity_prefill_model_calls,
            "prefill_imports": self.parity_prefill_imports,
            "draft_prefill_model_calls": self.parity_draft_prefill_model_calls,
            "draft_prefill_imports": self.parity_draft_prefill_imports,
            "decode_steps": self.parity_decode_steps,
            "q2_steps": self.parity_q2_steps,
            "draft_calls": self.parity_draft_calls,
            "loaded_tokens_per_layer": self.parity_transfers,
            "compare_output": compare_output,
            "decode_observations": getattr(self, "parity_decode_observations", 0),
        }
