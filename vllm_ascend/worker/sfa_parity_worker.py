# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only worker for real eight-layer eager/root-replay comparisons.

Selected explicitly by tools/sfa_full_graph_parity.py. Normal serving never
imports this worker or installs its hooks. Hooks on decoder modules contain
only tensor copies/counter increments, traced into the existing target graph;
SFA probes run inside existing opaque SFA ops during capture, not live replay.
"""

import hashlib
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
)
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.worker.worker import NPUWorker

TARGET_LAYERS = 8
QUERY_WIDTH = 2
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def weight_fingerprint(model: torch.nn.Module) -> str:
    """Hash the complete state, not a sample, in bounded CPU chunks."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        value = value.detach()
        if value.ndim == 0:
            value = value.reshape(1)
        stride_bytes = value[0].numel() * value.element_size() if value.shape[0] else 1
        chunk_rows = max(1, HASH_CHUNK_BYTES // max(1, stride_bytes))
        for chunk in value.split(chunk_rows):
            digest.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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

    def __init__(self, layer, impl: AscendSFAImpl, index: int, rows: int, device: torch.device) -> None:
        self.index = index
        # The first parameter may be packed INT4 weights; activations follow RMSNorm.
        dtype = layer.input_layernorm.weight.dtype
        hidden = layer.input_layernorm.weight.numel()
        self.probes: dict[str, DeviceSnapshot] = {}

        def allocate(name, shape, tensor_dtype=dtype):
            self.probes[name] = DeviceSnapshot(shape, dtype=tensor_dtype, device=device)

        for phase in ("input", "output"):
            for component in ("hidden", "residual"):
                allocate(f"{phase}.{component}", (rows, hidden))
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
        names += ["output.hidden", "output.residual"]
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
        self.parity_decode_steps = 0
        self.parity_q2_steps = 0
        self.parity_draft_calls = 0
        self.parity_transfers = [0] * TARGET_LAYERS
        self.parity_directory = Path(options["reference"]) / f"rank-{self.parity_rank}"
        self.parity_is_graph = options["mode"] == "graph"
        if self.parity_is_graph != bool(envs.VLLM_ASCEND_SFA_FULL_GRAPH):
            raise ValueError("Parity mode and full-graph flag disagree")
        if not self.parity_is_graph and (envs.VLLM_ASCEND_SFA_STAGED_GRAPH or not self.model_config.enforce_eager):
            raise ValueError("The reference must disable both staged/full graph and enforce eager")
        original_load = DummyModelLoader.load_weights

        def load(loader, model, model_config):
            deterministic_dummy_load(original_load, loader, model, model_config)

        with patch.object(DummyModelLoader, "load_weights", load):
            super().load_model()
        self._check(self._prepare_probes, "startup weights/probes")
        runner = self.model_runner
        original_forward = runner._model_forward
        signature = inspect.signature(original_forward)

        def forward(*args, **kwargs):
            context = get_forward_context()
            live = runner.input_batch.num_reqs > 0 and not getattr(context, "staged_sfa_graph_dummy_run", False)
            if not live:
                return original_forward(*args, **kwargs)
            state = self._check(
                lambda: self._prepare_step(signature.bind(*args, **kwargs).arguments, context),
                f"step={self.parity_step} before target",
            )
            before = runner._sfa_full_graph.replay_count
            # Do not put a failure collective around the model itself: a kernel
            # failure may leave other ranks inside HCCL. vLLM's supervisor owns
            # that failure path. Diagnostic checks are coordinated only outside.
            result = original_forward(*args, **kwargs)
            replays = runner._sfa_full_graph.replay_count - before
            self._check(lambda: self._observe_step(state, result, replays), f"step={self.parity_step} after target")
            if self.parity_is_graph and self.parity_rank == 0:
                print(
                    f"[SFA_PARITY] step={self.parity_step} rows={state['rows']} decode={state['decode']} "
                    f"ranks={self.parity_tp_size} layers=8 PASS replay_per_rank={replays}",
                    flush=True,
                )
            self.parity_step += 1
            self.parity_decode_steps += int(state["decode"])
            self.parity_q2_steps += int(state["decode"] and state["rows"] == QUERY_WIDTH)
            return result

        original_propose = runner.propose_draft_token_ids

        def propose(*args, **kwargs):
            tokens = original_propose(*args, **kwargs)

            def fix_tokens():
                if not isinstance(tokens, torch.Tensor) or tokens.shape[-1] != 1:
                    raise ParityError("MTP did not produce one draft token per request")
                self.parity_draft_calls += 1
                # Only the choice is overridden. Draft computation/KV callbacks run.
                return torch.full_like(tokens, options["token_id"])

            return self._check(fix_tokens, f"step={self.parity_step} after draft")

        runner._model_forward = forward
        runner.propose_draft_token_ids = propose

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
                    layer, impl, int(match[1]), self.vllm_config.scheduler_config.max_num_batched_tokens, self.device
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
        state = {
            "rank": self.parity_rank,
            "tp_size": self.parity_tp_size,
            "step": self.parity_step,
            "decode": decode,
            "rows": rows,
            "layers": TARGET_LAYERS,
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
        path = self.parity_directory / f"step-{self.parity_step:06d}.pt"
        if self.parity_is_graph:
            if not path.is_file():
                raise ParityError(f"step={self.parity_step}: no matching eager step")
            reference = torch.load(path, map_location="cpu", weights_only=True)
            try:
                compare_step(reference, state, atol=self.parity_options["atol"], rtol=self.parity_options["rtol"])
            except ParityError as error:
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
        if self.parity_decode_steps < 2 or self.parity_q2_steps < 2 or self.parity_draft_calls < 2:
            raise ParityError("Insufficient live decode/Q2/MTP coverage")
        if not all(count > 0 for count in self.parity_transfers):
            raise ParityError(f"No historical KV transfer observed in some target layers: {self.parity_transfers}")
        expected_steps = len(list(self.parity_directory.glob("step-*.pt")))
        if self.parity_step != expected_steps:
            raise ParityError(f"Incomplete step comparison: {self.parity_step} != {expected_steps}")
        return {
            "rank": self.parity_rank,
            "tp_size": self.parity_tp_size,
            "steps": self.parity_step,
            "decode_steps": self.parity_decode_steps,
            "q2_steps": self.parity_q2_steps,
            "draft_calls": self.parity_draft_calls,
            "loaded_tokens_per_layer": self.parity_transfers,
        }
