# SPDX-License-Identifier: Apache-2.0
"""Run production o_proj control flow with stdlib mocks, without torch/NPU.

As in the other standalone SFA contracts, compile the relevant production AST
blocks rather than importing the Ascend runtime. These tests check branch
execution and registration side effects, not device math or HCCL performance.
"""

import ast
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock

SOURCE = Path(__file__).resolve().parents[2] / "vllm_ascend/attention/sfa_v1.py"


def method(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSFAImpl")
    return next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)


def assignment_index(nodes, name, *, attribute=False):
    for index, node in enumerate(nodes):
        targets = node.targets if isinstance(node, ast.Assign) else []
        for target in targets:
            if attribute and isinstance(target, ast.Attribute) and target.attr == name:
                return index
            if not attribute and isinstance(target, ast.Name) and target.id == name:
                return index
    raise AssertionError(f"Production assignment not found: {name}")


def execute(nodes, namespace):
    tree = ast.parse("from __future__ import annotations")
    tree.body.extend(deepcopy(nodes))
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), namespace)


class TestLayerShardingInitialization(unittest.TestCase):
    def initialize(self, *, cp=True, sharding=False, full=True, names=None, layers=None):
        obj = NS(
            enable_dsa_cp=cp,
            enable_dsa_cp_with_layer_shard=sharding,
            enable_dsa_cp_with_o_proj_tp=full,
            num_heads=2,
            tp_size=4,
        )
        register = Mock()
        nodes = method("__init__").body
        start = assignment_index(nodes, "layer_sharding_kwargs", attribute=True)
        namespace = dict(
            self=obj,
            kwargs=layers or {},
            get_ascend_config=lambda: NS(layer_sharding=names),
            register_all_layers_to_shard_weight_series=register,
            logger=NS(warning_once=Mock()),
        )
        return obj, register, nodes[start:], namespace

    def test_no_kv_transfer_full_path_initializes_empty_sharding_list(self):
        obj, register, nodes, namespace = self.initialize()
        execute(nodes, namespace)
        self.assertEqual(obj.layer_sharding_kwargs, [])
        self.assertEqual(obj.local_num_heads, 8)
        register.assert_not_called()

    def test_cp_disabled_initializes_empty_sharding_list(self):
        obj, register, nodes, namespace = self.initialize(cp=False, full=False)
        execute(nodes, namespace)
        self.assertEqual(obj.layer_sharding_kwargs, [])
        register.assert_not_called()

    def test_q_b_proj_sharding_and_full_o_proj_can_coexist(self):
        q_b_proj = object()
        obj, register, nodes, namespace = self.initialize(
            sharding=True, names=["q_b_proj"], layers={"q_b_proj": q_b_proj}
        )
        execute(nodes, namespace)
        self.assertEqual(obj.layer_sharding_kwargs, [q_b_proj])
        register.assert_called_once_with([q_b_proj])

    def test_o_proj_dual_ownership_rejected_before_registration(self):
        obj, register, nodes, namespace = self.initialize(
            sharding=True,
            names=["q_b_proj", "o_proj"],
            layers={"q_b_proj": object(), "o_proj": object()},
        )
        with self.assertRaisesRegex(ValueError, "layer_sharding containing 'o_proj'"):
            execute(nodes, namespace)
        register.assert_not_called()
        self.assertEqual(obj.layer_sharding_kwargs, [])

    def test_legacy_o_proj_sharding_registration_is_preserved(self):
        o_proj = object()
        obj, register, nodes, namespace = self.initialize(
            sharding=True, full=False, names=["o_proj"], layers={"o_proj": o_proj}
        )
        execute(nodes, namespace)
        self.assertEqual(obj.layer_sharding_kwargs, [o_proj])
        register.assert_called_once_with([o_proj])


class TestProjectionControl(unittest.TestCase):
    def run_projection(self, *, full_enabled, gathered=False, strict=True, layerwise=False):
        output = MagicMock(name="output")
        activation = MagicMock(name="activation")
        calls = []

        def handle(**kwargs):
            self.assertEqual(kwargs["should_shard_weight"], gathered)
            calls.append("full_projection" if gathered else "handler_redistribution")
            return (output, False, False) if gathered else (activation, True, True)

        obj = NS(
            enable_dsa_cp_with_o_proj_tp=full_enabled,
            enable_dsa_cp_strict_accuracy=strict,
            tp_size=4,
            num_heads=2,
            v_head_dim=2,
            _handle_o_proj_weight_switch_and_forward=Mock(side_effect=handle),
            o_proj=Mock(return_value=(activation,)),
            _layerwise_prefill_p_node=layerwise,
            dsa_shrink_latent=0,
            dsa_offload_unbundle=True,
            _submit_sfa_save_operations=Mock(),
            _finish_sfa_layerwise_transfer_window=Mock(),
        )
        collective = Mock()
        namespace = dict(
            self=obj,
            output=output,
            output_padded=output,
            attn_output=activation,
            full_gather_o_proj_enabled=gathered,
            o_proj_full_handle=None,
            o_proj_full_param_handles=None,
            torch=NS(empty_like=Mock(return_value=activation), distributed=NS(all_to_all_single=collective)),
            get_tp_group=lambda: NS(device_group=object()),
            use_layerwise_transfer_window=layerwise,
            is_last_transfer_layer=layerwise,
            pending_transfers=None,
            pending_transfer_names=[],
            final_transfer_names=["latent", "indexer"],
            save_operations=[("latent", ["K", "V"]), ("indexer", ["I"])] if layerwise else [],
            layer_name="latent",
            index_layer_name="indexer",
            index_lmcache_enabled=True,
            kv_cache=["K", "V", "I"],
            _is_pure_decode=not gathered,
            _decode_window_save_window_size=lambda: 0,
            _dsa_prof=NS(end=Mock()),
            _sfa_t=None,
        )
        nodes = method("forward").body
        start = assignment_index(nodes, "full_o_proj_completed")
        # The original return belongs to forward, not this module execution.
        execute(nodes[start:-1], namespace)
        return obj, collective, calls, namespace

    def test_full_projection_skips_strict_all_to_all_and_still_saves(self):
        obj, collective, calls, _ = self.run_projection(full_enabled=True, gathered=True)
        collective.assert_not_called()
        obj.o_proj.assert_not_called()
        self.assertEqual(calls, ["full_projection"])
        obj._submit_sfa_save_operations.assert_called_once_with(
            [("latent", ["K", "V"]), ("indexer", ["I"])]
        )

    def test_full_projection_still_finishes_last_layer_window(self):
        obj, collective, _, _ = self.run_projection(full_enabled=True, gathered=True, layerwise=True)
        collective.assert_not_called()
        obj.o_proj.assert_not_called()
        obj._submit_sfa_save_operations.assert_not_called()
        obj._finish_sfa_layerwise_transfer_window.assert_called_once_with(
            [("latent", ["K", "V"]), ("indexer", ["I"])], ["latent", "indexer"]
        )

    def test_decode_already_redistributed_does_not_repeat_strict_all_to_all(self):
        obj, collective, calls, _ = self.run_projection(full_enabled=True)
        collective.assert_not_called()
        obj.o_proj.assert_called_once()
        self.assertEqual(calls, ["handler_redistribution"])

    def test_legacy_glm_path_keeps_strict_all_to_all(self):
        obj, collective, calls, _ = self.run_projection(full_enabled=False)
        collective.assert_called_once()
        obj.o_proj.assert_called_once()
        self.assertEqual(calls, [])
        obj._handle_o_proj_weight_switch_and_forward.assert_not_called()

    def test_non_strict_legacy_path_does_not_add_all_to_all(self):
        obj, collective, _, _ = self.run_projection(full_enabled=False, strict=False)
        collective.assert_not_called()
        obj.o_proj.assert_called_once()


if __name__ == "__main__":
    unittest.main()
