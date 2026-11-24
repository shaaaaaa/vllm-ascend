# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ondemand-selected MLA prefill."""

import ast
import ctypes
import itertools
import os
import sys
import threading
import types
import unittest
from contextlib import contextmanager, nullcontext
from dataclasses import make_dataclass
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import regex as re
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_INJECTED_VLLM_MODS: list[str] = []
if find_spec("vllm") is None:
    _b, _sm = "vllm.distributed.kv_transfer.kv_connector.factory", sys.modules
    for _n in ("vllm.logger", "vllm.v1.core.dsa_shared_pool", _b):
        _c = ""
        for _p in _n.split("."):
            _c = f"{_c}.{_p}" if _c else _p
            if _c not in _sm:
                _sm[_c] = types.ModuleType(_c)
                _INJECTED_VLLM_MODS.append(_c)
            _sm[_c].__dict__.setdefault("__path__", [])
    _nop = lambda *a, **k: None
    _sm["vllm.logger"].logger = SimpleNamespace(info=_nop, info_once=_nop, warning=_nop, debug=_nop)
    _sm["vllm.v1.core.dsa_shared_pool"].layerwise_prefill_bank_count = lambda: 2
    _sm["vllm.distributed.kv_transfer"].get_kv_transfer_group = _nop
    _sm[_b].KVConnectorFactory = SimpleNamespace(register_connector=_nop)


def tearDownModule() -> None:
    for _mod in reversed(_INJECTED_VLLM_MODS):
        sys.modules.pop(_mod, None)


import vllm_ascend.core.selected_kv as skv  # noqa: E402
from vllm_ascend import envs as envs_ascend  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import lmcache_selected_kv_source as lms  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import prefill_selected_runtime as psr  # noqa: E402

_Plan = make_dataclass("P", ("chunk", "slot", "chunk_token", "tokens"), namespace={"__len__": lambda s: len(s.slot)})


def _mk_stream(sid: int):
    waited = []
    return SimpleNamespace(npu_stream=sid, waited=waited, wait_event=waited.append)


def _mk_event_cls(cur_fn):
    class _FakeEvent:
        def __init__(self) -> None:
            self.recorded_on: int | None = None

        def record(self, stream=None) -> None:
            self.recorded_on = int((stream or cur_fn()).npu_stream)

        def synchronize(self) -> None:
            pass

        def query(self) -> bool:
            return True

    return _FakeEvent


@contextmanager
def _fake_npu():
    stack = [_mk_stream(1)]
    sid = itertools.count(2)
    ev_cls = _mk_event_cls(lambda: stack[-1])

    @contextmanager
    def _on_stream(stream):
        stack.append(stream)
        try:
            yield stream
        finally:
            stack.pop()

    mod = SimpleNamespace(
        Stream=lambda *a, **k: _mk_stream(next(sid)),
        Event=ev_cls,
        stream=_on_stream,
        current_stream=lambda *a, **k: stack[-1],
        device=lambda d: nullcontext(),
    )
    with patch.object(torch, "npu", mod, create=True):
        yield


def _vcfg(**ov):
    return SimpleNamespace(
        kv_transfer_config=ov.get(
            "tr", SimpleNamespace(kv_role="kv_producer", is_kv_producer=True, kv_connector="LMCacheConnectorV1")
        ),
        model_config=SimpleNamespace(enforce_eager=True, max_model_len=1 << 20),
        scheduler_config=ov.get(
            "sched", SimpleNamespace(async_scheduling=True, max_num_batched_tokens=16384, max_num_seqs=8)
        ),
        parallel_config=ov.get("par", SimpleNamespace(enable_dbo=False, prefill_context_parallel_size=1)),
        speculative_config=ov.get("spec", SimpleNamespace(method="deepseek_mtp", num_speculative_tokens=1)),
    )


def _mk_obj(ptr_base: int, tokens: int):
    state = {"p": 0, "r": 0}
    return SimpleNamespace(
        is_valid=lambda: True,
        is_pinned=True,
        layer_size_bytes=lambda _: tokens * 1152,
        layer_data_ptr=lambda lid: ptr_base + lid * 0x100,
        pin=lambda: state.__setitem__("p", state["p"] + 1),
        unpin=lambda: state.__setitem__("p", state["p"] - 1),
        ref_count_up=lambda: state.__setitem__("r", state["r"] + 1),
        ref_count_down=lambda: state.__setitem__("r", state["r"] - 1),
        s=state,
    )


class SelectedKVTopologyDiagnosticTest(unittest.TestCase):
    def tearDown(self) -> None:
        skv._ACTIVE_RUNTIMES.clear()

    @staticmethod
    def _runner(impl, *, is_draft=False):
        name = "model.layers.0.self_attn.attn"
        ctx = {name: SimpleNamespace(impl=impl)}
        if is_draft:
            ctx["model.layers.1.self_attn.attn"] = SimpleNamespace(
                impl=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048)
            )
        return SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=ctx),
            drafter=SimpleNamespace(attn_layer_names=(name,) if is_draft else ()),
        )

    def test_topology_failure_reports_values_types_and_guards(self) -> None:
        cases = (
            ({"kv_lora_rank": 256}, False, "kv_lora_rank=256 (type=int) (expected 512)"),
            ({"qk_rope_head_dim": 32}, False, "qk_rope_head_dim=32 (type=int) (expected 64)"),
            (
                {"enable_dsa_cp": True, "enable_dsa_cp_with_o_proj_tp": True},
                False,
                "impl.enable_dsa_cp_with_o_proj_tp=True (type=bool)",
            ),
            ({"enable_dsa_cp": True, "is_vl_first_layer": True}, False, "is_vl_first_layer=True"),
            ({"skip_topk": True}, True, "is_draft=True, skip_topk=True"),
            ({"kv_lora_rank": "512"}, False, "kv_lora_rank='512' (type=str) (expected 512)"),
        )
        for overrides, is_draft, detail in cases:
            with self.subTest(overrides=overrides, is_draft=is_draft):
                fields = dict(kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048)
                fields.update(overrides)
                with self.assertRaises(ValueError) as raised:
                    skv._resolve_topology(self._runner(SimpleNamespace(**fields), is_draft=is_draft))
                message = str(raised.exception)
                self.assertTrue(message.startswith("Bad MLA/draft 'model.layers.0.self_attn.attn':"))
                self.assertIn("impl_type=SimpleNamespace", message)
                self.assertIn(detail, message)
                self.assertIn("kv_lora_rank=", message)
                self.assertIn("qk_rope_head_dim=", message)
                self.assertIn(f"is_draft={is_draft}", message)

    def test_dimension_failure_preserves_property_access_order(self) -> None:
        accesses = []

        class _Impl:
            @property
            def skip_topk(self):
                accesses.append("skip_topk")
                return False

            @property
            def kv_lora_rank(self):
                accesses.append("kv_lora_rank")
                return 256

            @property
            def qk_rope_head_dim(self):
                accesses.append("qk_rope_head_dim")
                return 64

            @property
            def enable_dsa_cp(self):
                accesses.append("enable_dsa_cp")
                return True

        with self.assertRaises(ValueError) as raised:
            skv._resolve_topology(self._runner(_Impl()))
        msg = str(raised.exception)
        self.assertIn("kv_lora_rank=256 (type=int) (expected 512)", msg)
        self.assertIn("impl.enable_dsa_cp=True (type=bool)", msg)
        self.assertEqual(
            accesses,
            ["kv_lora_rank", "skip_topk", "enable_dsa_cp", "kv_lora_rank", "qk_rope_head_dim"],
        )

    def test_cp_failure_reads_property_once(self) -> None:
        accesses = []

        class _Impl:
            kv_lora_rank = 512
            qk_rope_head_dim = 64
            enable_dsa_cp_with_o_proj_tp = True

            @property
            def enable_dsa_cp(self):
                accesses.append("enable_dsa_cp")
                return True

        with self.assertRaisesRegex(ValueError, "impl.enable_dsa_cp=True"):
            skv._resolve_topology(self._runner(_Impl()))
        self.assertEqual(accesses, ["enable_dsa_cp"])

    def test_topology_diagnostic_does_not_inspect_unknown_values(self) -> None:
        class _Meta(type):
            @property
            def __name__(cls):
                raise AssertionError("custom type-name property must not be called")

        class _DeviceValue(metaclass=_Meta):
            def __repr__(self):
                raise AssertionError("custom repr must not be called")

            def item(self):
                raise AssertionError("device value must not be read")

            def cpu(self):
                raise AssertionError("device value must not be copied")

        for field in ("kv_lora_rank", "qk_rope_head_dim", "enable_dsa_cp", "enable_dsa_cp_with_o_proj_tp"):
            with self.subTest(field=field):
                fields = dict(kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048)
                fields[field] = _DeviceValue()
                if field == "enable_dsa_cp":
                    fields["enable_dsa_cp_with_o_proj_tp"] = True
                with self.assertRaises(ValueError) as raised:
                    skv._resolve_topology(self._runner(SimpleNamespace(**fields)))
                self.assertIn(f"{field}=<type=_DeviceValue>", str(raised.exception))

    def test_topology_diagnostic_bypasses_class_name_subclass_slicing(self) -> None:
        class _UnsafeName(str):
            def __getitem__(self, key):
                raise RuntimeError("custom class-name slicing must not be called")

        class _Value:
            pass

        class _Impl:
            kv_lora_rank = _Value()
            qk_rope_head_dim = 64

        _Value.__name__ = _UnsafeName("DiagnosticValue")
        _Impl.__name__ = _UnsafeName("DiagnosticImpl")
        with self.assertRaises(ValueError) as raised:
            skv._resolve_topology(self._runner(_Impl()))
        message = str(raised.exception)
        self.assertTrue(message.startswith("Bad MLA/draft 'model.layers.0.self_attn.attn':"))
        self.assertIn("impl_type=DiagnosticImpl", message)
        self.assertIn("kv_lora_rank=<type=DiagnosticValue>", message)

    def test_topology_diagnostic_bounds_builtin_values(self) -> None:
        for value, type_name in (("\\n" * 10000, "str"), (1 << 10000, "int")):
            with self.subTest(type_name=type_name):
                detail = skv._topology_diagnostic_value(value)
                self.assertLess(len(detail), 160)
                self.assertIn(f"type={type_name}", detail)
        for value, expected in ((None, "None (type=NoneType)"), (1.5, "1.5 (type=float)")):
            self.assertEqual(skv._topology_diagnostic_value(value), expected)

    def test_valid_topology_preserves_owners_and_draft(self) -> None:
        names = tuple(f"model.layers.{idx}.self_attn.attn" for idx in range(3))
        ctx = {
            name: SimpleNamespace(
                impl=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048, skip_topk=idx == 1)
            )
            for idx, name in enumerate(names)
        }
        runner = SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=ctx),
            drafter=SimpleNamespace(attn_layer_names=(names[2],)),
        )
        self.assertEqual(
            skv._resolve_topology(runner),
            (names, frozenset((names[2],)), {names[0]: names[0], names[1]: names[0], names[2]: names[2]}, False),
        )

    def test_multimodal_mtp_draft_rejected_under_dsa_cp_tp_gt_1(self) -> None:
        names = ("model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn")
        ctx = {
            name: SimpleNamespace(
                impl=SimpleNamespace(
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    index_topk=2048,
                    enable_dsa_cp=True,
                )
            )
            for name in names
        }

        def _mk_runner(tp_size: int, is_mm: bool) -> Any:
            return SimpleNamespace(
                vllm_config=SimpleNamespace(
                    parallel_config=SimpleNamespace(
                        tensor_parallel_size=tp_size,
                        pipeline_parallel_size=1,
                        prefill_context_parallel_size=1,
                        decode_context_parallel_size=1,
                    ),
                    model_config=SimpleNamespace(is_multimodal_model=is_mm),
                ),
                compilation_config=SimpleNamespace(static_forward_context=ctx),
                drafter=SimpleNamespace(attn_layer_names=(names[1],)),
            )

        self.assertRaisesRegex(
            ValueError,
            "multimodal MTP draft is unsupported under DSA-CP",
            skv._resolve_topology,
            _mk_runner(4, True),
        )
        self.assertEqual(
            skv._resolve_topology(_mk_runner(4, False)),
            (names, frozenset((names[1],)), {names[0]: names[0], names[1]: names[1]}, True),
        )
        self.assertEqual(
            skv._resolve_topology(_mk_runner(1, True)),
            (names, frozenset((names[1],)), {names[0]: names[0], names[1]: names[1]}, True),
        )
        rn_dec = _mk_runner(4, False)
        ctx[names[0]].impl.dsa_cp_decoder = True
        try:
            self.assertRaisesRegex(
                ValueError,
                r"impl\.dsa_cp_decoder=True",
                skv._resolve_topology,
                rn_dec,
            )
        finally:
            ctx[names[0]].impl.dsa_cp_decoder = False


class SelectedKVUnitTest(unittest.TestCase):
    def tearDown(self) -> None:
        skv._ACTIVE_RUNTIMES.clear()

    def test_selected_prefill_rejects_adapter_cache(self) -> None:
        with patch.dict(
            envs_ascend.__dict__,
            {
                "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": True,
                "VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS": 2,
                "VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD": False,
                "VLLM_ASCEND_DSA_USE_ADAPTER_CACHE": True,
            },
        ):
            with self.assertRaisesRegex(ValueError, "cannot be combined with DSA adapter cache"):
                skv.SelectedKVConfig(enabled=True).validate_deployment(_vcfg())
            skv.SelectedKVConfig(enabled=False).validate_deployment(_vcfg())
            with patch.dict(envs_ascend.__dict__, {"VLLM_ASCEND_DSA_USE_ADAPTER_CACHE": False}):
                skv.SelectedKVConfig(enabled=True).validate_deployment(_vcfg())

    def test_selected_prefill_rejects_latent_offload(self) -> None:
        with patch.dict(
            envs_ascend.__dict__,
            {
                "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": True,
                "VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS": 2,
                "VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD": True,
            },
        ):
            with self.assertRaisesRegex(ValueError, "cannot be combined with DSA latent offload"):
                skv.SelectedKVConfig(enabled=True).validate_deployment(_vcfg())
            skv.SelectedKVConfig(enabled=False).validate_deployment(_vcfg())

    def test_selected_prefill_rejects_decode_preemption_checkpoint(self) -> None:
        with patch.dict(
            envs_ascend.__dict__,
            {
                "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": True,
                "VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS": 2,
                "VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD": False,
                "VLLM_ASCEND_DSA_USE_ADAPTER_CACHE": False,
            },
        ):
            direct_tr = {
                "kv_connector": "LMCacheAscendConnectorV1Dynamic",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {"decode_preemption_checkpoint": True},
            }
            multi_tr = {
                "kv_connector": "MultiConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "connectors": [
                        {"kv_connector": "MooncakeConnectorV1", "kv_role": "kv_producer"},
                        {
                            "kv_connector": "LMCacheAscendConnectorV1Dynamic",
                            "kv_role": "kv_producer",
                            "kv_connector_extra_config": {"lmcache_config": {"decode_preemption_checkpoint": True}},
                        },
                    ]
                },
            }
            for bad_tr in (direct_tr, multi_tr):
                with self.assertRaisesRegex(ValueError, "decode_preemption_checkpoint"):
                    skv.SelectedKVConfig(enabled=True).validate_deployment(_vcfg(tr=bad_tr))
                skv.SelectedKVConfig(enabled=False).validate_deployment(_vcfg(tr=bad_tr))
            with patch.dict(os.environ, {"LMCACHE_DECODE_PREEMPTION_CHECKPOINT": "1"}):
                with self.assertRaisesRegex(ValueError, "decode_preemption_checkpoint"):
                    skv.SelectedKVConfig(enabled=True).validate_deployment(_vcfg())
                skv.SelectedKVConfig(enabled=False).validate_deployment(_vcfg())
            bad_impl = SimpleNamespace(
                config=SimpleNamespace(decode_preemption_checkpoint=True),
                lmcache_engine=SimpleNamespace(gpu_connector=SimpleNamespace()),
            )
            with self.assertRaisesRegex(ValueError, "decode_preemption_checkpoint"):
                lms.LMCacheSelectedKVSource(
                    SimpleNamespace(device="cpu"),
                    bad_impl,
                    ("l0",),
                    {"l0": (torch.zeros((8, 128, 1, 512), dtype=torch.bfloat16), None)},
                )

    def test_config_gates_budget_classify_partition_dma_filter_and_mla_only(self) -> None:
        cfg = skv.SelectedKVConfig.from_dict({"enabled": True})
        kw = {"VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": True, "VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS": 2}
        bad_cfgs = (
            ("par", SimpleNamespace(enable_dbo=True)),
            ("par", SimpleNamespace(prefill_context_parallel_size=2)),
            ("sched", SimpleNamespace(async_scheduling=True, max_num_batched_tokens=16385, max_num_seqs=8)),
            ("tr", SimpleNamespace(kv_role="kv_consumer", kv_connector="LMCacheConnectorV1")),
            ("spec", SimpleNamespace(method="deepseek_mtp", num_speculative_tokens=2)),
        )
        rn_bad_tk = SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={
                    "model.layers.0.self_attn": SimpleNamespace(
                        impl=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64, index_topk=1024)
                    )
                }
            )
        )
        with patch.multiple(envs_ascend, **kw):
            self.assertEqual(skv.init_selected_kv_config({"selected_kv_config": {"enabled": True}}, _vcfg()), cfg)
            for k, v in bad_cfgs:
                self.assertRaises(ValueError, cfg.validate_deployment, _vcfg(**{k: v}))
            dt_ns = SimpleNamespace(A2=2, A3=1, _310P=0, A5=3)
            mr = SimpleNamespace(ascend_config=SimpleNamespace(selected_kv_config=cfg), enable_dsa_cp=True)
            for dt, msg in (
                (dt_ns._310P, "A2 or A3 build"),
                (dt_ns.A5, "A2 or A3 build"),
                (dt_ns.A2, "Requires P-node"),
                (dt_ns.A3, "Requires P-node"),
            ):
                u_m = {
                    "vllm_ascend.utils": SimpleNamespace(AscendDeviceType=dt_ns, get_ascend_device_type=lambda t=dt: t)
                }
                with patch.dict(sys.modules, u_m), self.assertRaisesRegex(ValueError, msg):
                    skv.reserve_selected_kv_workspace(SimpleNamespace(vllm_config=_vcfg(), model_runner=mr), 1 << 30)
            self.assertRaisesRegex(ValueError, "TopK=2048", skv._resolve_topology, rn_bad_tk)

        self.assertEqual((psr._i64(1 << 63), psr._i64(None)), (-(1 << 63), 0))
        self.assertGreater(psr.selected_workspace_bytes(16384, 8), 0)
        ctx_ok = {
            "model.layers.0.self_attn": SimpleNamespace(
                impl=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048)
            ),
            "model.layers.1.self_attn": SimpleNamespace(
                impl=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64, skip_topk=True)
            ),
        }
        for dev_type in (dt_ns.A2, dt_ns.A3):
            mr_ok = SimpleNamespace(
                ascend_config=SimpleNamespace(selected_kv_config=cfg),
                enable_dsa_cp=False,
                layerwise_prefill_p_node=True,
                layerwise_prefill_bank_count=2,
                compilation_config=SimpleNamespace(static_forward_context=ctx_ok),
            )
            u_dev = {
                "vllm_ascend.utils": SimpleNamespace(
                    AscendDeviceType=dt_ns, get_ascend_device_type=lambda t=dev_type: t
                )
            }
            with patch.multiple(envs_ascend, **kw), patch.dict(sys.modules, u_dev):
                wk_ok = SimpleNamespace(
                    vllm_config=_vcfg(), model_runner=mr_ok, cache_config=SimpleNamespace(num_gpu_blocks_override=16)
                )
                nb_ok = skv.reserve_selected_kv_workspace(wk_ok, 2 << 30)
                self.assertEqual(nb_ok, mr_ok._selected_kv_budget.native_budget_bytes)
                self.assertLess(nb_ok, 2 << 30)

        p_calls = []

        class _Fn:
            def __init__(self, cb=None):
                self._cb = cb

            def __call__(self, *a):
                return self._cb(*a) if self._cb else 0

        f_lib = SimpleNamespace(
            create_sparse_prefill_fixed_handle=_Fn(
                lambda p, c, hp: (p_calls.append(("cr", c)), setattr(hp._obj, "value", 0xCAFE), 0)[2]
            ),
            destroy_sparse_prefill_fused_handle=_Fn(lambda h: p_calls.append(("ds", getattr(h, "value", h))) or 0),
            launch_sparse_prefill_fused_plan=ctypes.c_void_p(0x1111),
        )
        m_lib = SimpleNamespace(launch_sparse_prefill_bf16_mla_inplace=ctypes.c_void_p(0x2222))
        fake_ops = SimpleNamespace(
            load_library=lambda p: p_calls.append(("ld", Path(p).name)),
            dsa_prefill_queue=SimpleNamespace(
                batch_plan=lambda *a: p_calls.append(("bp", a[6], a[8])),
                mla_inplace_copy=lambda *a: p_calls.append(("mc", a[-1])),
            ),
        )
        with (
            _fake_npu(),
            patch.object(ctypes, "CDLL", side_effect=[f_lib, m_lib]),
            patch.object(torch, "ops", fake_ops),
        ):
            planner = psr.SelectedKVPlanner("cpu", 1024, 16, 2)
            ids_p, st_p = planner.build(torch.zeros((4, 1, 2048), dtype=torch.int32), [16384], [0], [4])
            planner.load_mla(
                ids_p,
                st_p,
                torch.zeros((4,), dtype=torch.uint64),
                torch.zeros((1, 4), dtype=torch.int32),
                torch.zeros((1, 2), dtype=torch.int32),
                torch.zeros((4, 128, 1, 512), dtype=torch.bfloat16),
                torch.zeros((4, 128, 1, 64), dtype=torch.bfloat16),
            )
            planner.release_quiescent()
            planner.close_quiescent()
            self.assertEqual(
                (planner._closed, p_calls[0][0], p_calls[1], p_calls[2], p_calls[3], p_calls[4][0]),
                (True, "ld", ("cr", 1024), ("bp", 1024, 0xCAFE), ("mc", 0x2222), "ds"),
            )
            for err_rc, err_msg in ((-2, ">= 32 vector cores"), (-1, "Fused handle init failed: -1")):
                f_err = SimpleNamespace(
                    create_sparse_prefill_fixed_handle=_Fn(lambda *a, r=err_rc: r),
                    destroy_sparse_prefill_fused_handle=_Fn(),
                    launch_sparse_prefill_fused_plan=ctypes.c_void_p(0x1111),
                )
                with (
                    patch.object(ctypes, "CDLL", side_effect=[f_err, m_lib]),
                    self.assertRaisesRegex(RuntimeError, err_msg),
                ):
                    psr.SelectedKVPlanner("cpu", 1024, 16, 2)

            p_calls.clear()
            f_1m = SimpleNamespace(
                create_sparse_prefill_fixed_handle=_Fn(
                    lambda p, c, hp: (
                        p_calls.append(("cr1m", Path(p.decode()).name, c)),
                        setattr(hp._obj, "value", 0xBEEF),
                        0,
                    )[2]
                ),
                destroy_sparse_prefill_fused_handle=_Fn(
                    lambda h: p_calls.append(("ds1m", getattr(h, "value", h))) or 0
                ),
                launch_sparse_prefill_fused_plan=ctypes.c_void_p(0x1111),
                selected_prefill_fused_max_history=_Fn(lambda: 1048576),
            )
            m_1m = SimpleNamespace(
                launch_sparse_prefill_bf16_mla_inplace=ctypes.c_void_p(0x2222),
                selected_prefill_mla_max_history=_Fn(lambda: 1048576),
            )
            with patch.object(ctypes, "CDLL", side_effect=[f_1m, m_1m]):
                pl_1m = psr.SelectedKVPlanner("cpu", 1048576, 512, 2)
                pl_1m.close_quiescent()
            self.assertEqual(p_calls[-2], ("cr1m", "sparse_prefill_fused_fixed_1m.o", 1048576))
            self.assertRaisesRegex(ValueError, "Plan bounds exceeded", psr._plan_shapes, 1048576 + 128, 512, 2)

            f_stale = SimpleNamespace(
                create_sparse_prefill_fixed_handle=f_1m.create_sparse_prefill_fixed_handle,
                destroy_sparse_prefill_fused_handle=f_1m.destroy_sparse_prefill_fused_handle,
                launch_sparse_prefill_fused_plan=ctypes.c_void_p(0x1111),
                selected_prefill_fused_max_history=_Fn(lambda: 524288),
            )
            m_stale = SimpleNamespace(
                launch_sparse_prefill_bf16_mla_inplace=ctypes.c_void_p(0x2222),
                selected_prefill_mla_max_history=_Fn(lambda: 524288),
            )

            class _StaleCDLL(psr._CDLL_TYPE):
                def __init__(self) -> None:
                    pass

                def __getattr__(self, name: str) -> Any:
                    raise AttributeError(name)

            for pair in ((f_stale, m_1m), (f_1m, m_stale), (_StaleCDLL(), m_1m), (f_1m, _StaleCDLL())):
                with (
                    patch.object(ctypes, "CDLL", side_effect=list(pair)),
                    self.assertRaisesRegex(RuntimeError, "Selected KV native libraries are outdated"),
                ):
                    psr.SelectedKVPlanner("cpu", 524288, 256, 2)

            class _MissingObjCDLL(_StaleCDLL):
                create_sparse_prefill_fixed_handle = f_1m.create_sparse_prefill_fixed_handle
                destroy_sparse_prefill_fused_handle = f_1m.destroy_sparse_prefill_fused_handle
                launch_sparse_prefill_fused_plan = ctypes.c_void_p(0x1111)
                selected_prefill_fused_max_history = _Fn(lambda: 1048576)

            with (
                patch.object(ctypes, "CDLL", side_effect=[_MissingObjCDLL(), m_1m]),
                patch.object(Path, "is_file", return_value=False),
                self.assertRaisesRegex(RuntimeError, "Missing 1M selected-KV device object"),
            ):
                psr.SelectedKVPlanner("cpu", 524288, 256, 2)

        budget = skv.reserve_workspace_bytes(1024 * 1024, 4096, 8192, 64, 4096)
        budget.validate(10)
        self.assertRaises(ValueError, skv.SelectedKVConfig.from_dict, {"bad": 1})
        self.assertRaises(ValueError, budget.validate, budget.maximum_blocks + 1)

        lk = {
            "block_allocation_mode": "prefill_child",
            "save_spec": SimpleNamespace(can_save=True, skip_leading_tokens=8192),
        }
        self.assertEqual(skv.MIN_HISTORY_TOKENS, 49152)
        mk = lambda rc=0, lc=0, ct=49152: SimpleNamespace(
            **lk,
            dsa_cold_compact_load=bool(rc),
            load_spec=SimpleNamespace(can_load=True, lmcache_cached_tokens=ct, dsa_cold_compact_load=bool(lc)),
        )
        cs = (
            (49152, 748, 0, 0, 49152),
            (49152, 749, 0, 0, 49152),
            (49151, 16, 0, 0, 49152),
            (16384, 16, 0, 0, 49152),
            (1048577, 16, 0, 0, 49152),
            (49152, 16, 1, 0, 49152),
            (49152, 16, 0, 1, 49152),
            (49152, 16, 0, 0, 8192),
        )
        self.assertEqual(
            [lms.classify_lmcache_request("r", *c[:2], mk(*c[2:])).selected for c in cs], [True] + [False] * 7
        )
        self.assertEqual(lms.classify_lmcache_request("r6", 49152, 16, mk(), kv_role="kv_both").omit_end_token, 8192)
        ok_1m = lms.classify_lmcache_request("r_1m_ok", 1047552, 1024, mk(ct=1047552))
        self.assertEqual((ok_1m.selected, ok_1m.omit_end_token), (True, 1047552))
        self.assertTrue(lms.classify_lmcache_request("r_1m_ratio", 1032960, 15616, mk(ct=1032960)).selected)
        self.assertTrue(lms.classify_lmcache_request("r_1m_cliff_ok", 1032848, 15728, mk(ct=1032848)).selected)
        self.assertFalse(lms.classify_lmcache_request("r_1m_cliff_bad", 1032847, 15729, mk(ct=1032847)).selected)
        max_h_1m = lms.classify_lmcache_request("r_1m_max_h", 1048575, 1, mk(ct=1048575))
        self.assertEqual((max_h_1m.selected, max_h_1m.omit_end_token), (True, 1047552))
        self.assertFalse(lms.classify_lmcache_request("r_1m_ovf1", 1047552, 1025, mk(ct=1047552)).selected)
        self.assertFalse(lms.classify_lmcache_request("r_1m_ovf2", 1048576, 1, mk(ct=1048576)).selected)

        reqs = [lms.SelectedRequest("a", 49152, 128, True, 49152), lms.SelectedRequest("b", 128, 128, False, 0)]
        blk = (np.array([[*range(385)], [2, 9] + [0] * 383], dtype=np.int32),)
        om, kp = lms.compute_per_bank_block_partition(reqs, blk, domain=512)
        self.assertTrue(1 in om[0] and 2 not in om[0] and 2 in kp[0])
        self.assertRaises(
            ValueError,
            lms.compute_per_bank_block_partition,
            [lms.SelectedRequest("x", 49152, 128, True, 999999)],
            blk,
            domain=512,
        )

        fp = lms.filter_dma_plan(
            _Plan(torch.tensor([2, 5]), torch.tensor([128, 640]), torch.tensor([512, 64]), torch.tensor([384, 256])),
            {2, 5},
        )
        self.assertEqual(
            (fp.chunk.tolist(), fp.slot.tolist(), fp.chunk_token.tolist(), fp.tokens.tolist()),
            ([2, 2, 5], [128, 384, 768], [512, 768, 192], [128, 128, 128]),
        )
        fe = lms.filter_dma_plan(
            _Plan(torch.tensor([2]), torch.tensor([256]), torch.tensor([0]), torch.tensor([128])), {2}
        )
        self.assertEqual(
            (fe.chunk.tolist(), fe.slot.tolist(), fe.chunk_token.tolist(), fe.tokens.tolist()), ([], [], [], [])
        )

        src_txt = "\n".join(
            p.read_text("utf-8")
            for d in (ROOT / "vllm_ascend", ROOT / "csrc" / "sparse_prefill_native")
            for p in list(d.rglob("*.py"))
            + list(d.rglob("*.cpp"))
            + list(d.rglob("*.h"))
            + list(d.rglob("CMakeLists.txt"))
        )
        for dead in (
            "packed_inplace",
            "PACKED_ROW_BYTES",
            "dsa_packed_inplace",
            "selected_kv_binding",
            "prefill_selected_plan",
            "prefill_selected_workspace",
            "core.selected_kv_config",
        ):
            self.assertNotIn(dead, src_txt)
        cmake_m = re.search(
            r'if\s*\(\s*SOC_VERSION\s+MATCHES\s+"([^"]+)"\s*\)\s*include\([^)]*sparse_prefill_native',
            (ROOT / "CMakeLists.txt").read_text("utf-8"),
        )
        self.assertIsNotNone(cmake_m)
        soc_re = re.compile(cmake_m.group(1))
        for soc in (
            "ascend910b1",
            "ascend910b2",
            "ascend910b2c",
            "ascend910b3",
            "ascend910b4",
            "ascend910b4-1",
            "ascend910_9391",
            "ascend910_9392",
        ):
            self.assertRegex(soc, soc_re)
        for bad_soc in ("ascend310p1", "ascend950"):
            self.assertNotRegex(bad_soc, soc_re)

        calls, wp = [], (ROOT / "vllm_ascend/worker/worker.py").read_text("utf-8")
        cn = next(n for n in ast.parse(wp).body if getattr(n, "name", "") == "NPUWorker")
        fns = [n for n in cn.body if getattr(n, "name", "") in ("release_sfa_graph_resources", "shutdown")]
        ns = {
            "logger": SimpleNamespace(warning=lambda *a, **k: None),
            "_B": type("_B", (), {"shutdown": lambda s: calls.append("base")}),
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module([ast.ClassDef("W", [ast.Name("_B", ast.Load())], [], fns, [])], [])
                ),
                "w",
                "exec",
            ),
            ns,
        )
        w = ns["W"]()
        w.model_runner = SimpleNamespace(
            _sfa_full_graph=SimpleNamespace(clear=lambda: calls.append("sfa")),
            selected_kv_runtime=SimpleNamespace(_failed=True, close=lambda: (_ for _ in ()).throw(RuntimeError())),
        )
        self.assertIsNone(w.shutdown())
        self.assertEqual(calls, ["sfa", "base"])

    def test_source_hooks_2bank_2plane_page_bases_bind_and_leases(self) -> None:
        nop = lambda *a, **k: None
        with _fake_npu():
            dvas, binds, rel, ups, reuse = [], [], [], [], [True]
            pl = _Plan(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([49280]))
            o0a, o0b, o1 = _mk_obj(0x2000, 24576), _mk_obj(0x5000, 24576), _mk_obj(0x8000, 49152)
            bd = lambda p, *a, **k: binds.append((p.slot.tolist(), a[6], a[8]))
            bcp = lambda *a: binds.append(("bcp", a[2], a[3]))
            npu_m = SimpleNamespace(bind_incremental_copy_addresses=bd, BoundCopyPrefix=bcp)

            def _bg(st, en, *a, **k):
                yield None
                for _ in (0, 1):
                    _ = npu_m.bind_incremental_copy_addresses(
                        pl, [o1], st, en if st == [0] else [24576, 49280], [1], [576], 2, host_ptr=7, previous="prev"
                    )
                    yield "ok"

            gc = SimpleNamespace(
                _layerwise_prefill_bank_count=2, batched_to_gpu=_bg, release_layerwise_prefill_dma_cache=rel.append
            )
            eng = SimpleNamespace(
                gpu_connector=gc,
                config=SimpleNamespace(chunk_size=1024),
                _prepare_shared_prefill_sources=nop,
                release_shared_cpu_sparse_request=rel.append,
            )
            ls = SimpleNamespace(can_load=True, lmcache_cached_tokens=49152, dsa_cold_compact_load=False)
            cm = SimpleNamespace(
                requests=[
                    SimpleNamespace(
                        req_id=r, is_decode_window_save=False, block_allocation_mode="prefill_child", load_spec=ls
                    )
                    for r in ("r1", "r0")
                ]
            )
            impl = SimpleNamespace(
                lmcache_engine=eng,
                _submit_initial_layerwise_prefill_load=nop,
                submit_layerwise_prefill_load=nop,
                _parent=SimpleNamespace(
                    _get_connector_metadata=lambda: cm, _kv_transfer_config=SimpleNamespace(kv_role="kv_producer")
                ),
            )

            ptrs = [[0x102000, 0x105000], [0x102100, 0x105100]]
            _g = lambda r, k, s=[0], e=[49152]: gc.batched_to_gpu(s, e, req_id=r, kv_group=k)
            _sh = lambda: {"req_id": "r0", "cached_chunk_dev_ptrs": ptrs, "_reuse_shared_dense_prefix": reuse[0]}

            def _st(_):
                gg, g1, g0 = _g("r1", 1), _g("r1", 0), _g("r0", 0, [0, 24576], [24576, 49152])
                _ = next(gg), gg.send([o1]), next(g1), next(g0)
                eng._prepare_shared_prefill_sources([[o0a, o0b], [o0a, o0b]], 0, _sh())
                _ = g1.send([o1]), g0.send([o0a, o0b]), impl.submit_layerwise_prefill_load("l0")
                _ = g1.send([o1]), g0.send([o0a, o0b])

            impl.start_load_kv = _st
            mc = {n: (torch.empty((1550, 1, 1, 1)),) * 2 for n in ("l0", "l1")}
            kw_s = dict(get_device_ptr_fn=lambda p, _: (dvas.append(p), p + 0x100000)[1], npu_connectors_module=npu_m)

            # Startup 8-hook contract validation
            self.assertRaisesRegex(
                RuntimeError,
                "BoundCopyPrefix",
                lms.LMCacheSelectedKVSource,
                SimpleNamespace(device="cpu"),
                impl,
                ("l0", "l1"),
                mc,
                get_device_ptr_fn=kw_s["get_device_ptr_fn"],
                npu_connectors_module=SimpleNamespace(bind_incremental_copy_addresses=bd),
            )
            del impl._submit_initial_layerwise_prefill_load
            self.assertRaisesRegex(
                RuntimeError,
                "hooks",
                lms.LMCacheSelectedKVSource,
                SimpleNamespace(device="cpu"),
                impl,
                ("l0", "l1"),
                mc,
                **kw_s,
            )
            impl._submit_initial_layerwise_prefill_load = nop

            src = lms.LMCacheSelectedKVSource(SimpleNamespace(device="cpu"), impl, ("l0", "l1"), mc, **kw_s)
            fn_w = npu_m.bind_incremental_copy_addresses
            lms._patch_bind_incremental(npu_m)
            self.assertTrue(fn_w._selected_kv_patched and npu_m.bind_incremental_copy_addresses is fn_w)

            o_up = src._record_upload
            src._record_upload = lambda s, lid=None: (ups.append(lid), o_up(s, lid))[1]
            tbl0 = np.vstack([np.arange(386), np.arange(386, 772)])
            tbl1 = np.vstack([np.arange(772, 1158), np.arange(1158, 1544)])
            run_s = lambda: (
                src.begin_runner_step(cm, ("r1", "r0"), (49152, 49152), (0, 16, 32), (tbl0, tbl1)),
                impl.start_load_kv(None),
            )
            run_s()
            s1, done, pf = src.step, [False], []

            # Exact (2, 1543, 2) page_bases verification across disjoint banks, multi-chunk o0a/o0b, and both planes
            exp_pb = np.zeros((2, 1543, 2), dtype=np.uint64)
            p384 = np.arange(384, dtype=np.uint64)
            p192 = np.arange(192, dtype=np.uint64)
            for lid, blk_off in ((0, 0), (1, 772)):
                off = lid * 0x100
                b1, b0a, b0b = 0x108000 + off, 0x102000 + off, 0x105000 + off
                exp_pb[lid, blk_off + p384, 0] = b1 + p384 * 128 * 1024
                exp_pb[lid, blk_off + p384, 1] = b1 + 49152 * 1024 + p384 * 128 * 128
                exp_pb[lid, blk_off + 386 + p192, 0] = b0a + p192 * 128 * 1024
                exp_pb[lid, blk_off + 386 + p192, 1] = b0a + 24576 * 1024 + p192 * 128 * 128
                exp_pb[lid, blk_off + 578 + p192, 0] = b0b + p192 * 128 * 1024
                exp_pb[lid, blk_off + 578 + p192, 1] = b0b + 24576 * 1024 + p192 * 128 * 128
            exp_pb_t = torch.from_numpy(exp_pb)
            self.assertTrue(torch.equal(s1.page_bases("l0"), exp_pb_t[0]))
            self.assertTrue(torch.equal(s1.page_bases("l1"), exp_pb_t[1]))

            s1.prefetch_main("l1", lambda: pf.append(int(s1.page_bases("l1")[772, 0])))
            self.assertEqual((pf, 1 in s1._pending_main), ([], True))
            ev_t1, ev_tail = torch.npu.Event(), torch.npu.Event()
            gc._layerwise_prefill_transfer_state = lambda: (0, 0, 0, {(0, 1): {"c0": ("t", ev_t1), "c1": ev_t1}})
            gc._layerwise_prefill_bank_tail_events = {(0, 1): ("t", ev_tail)}
            src._mark_slots_ready([1])
            st_a, st_b = _mk_stream(10), _mk_stream(11)
            s1.wait_main_slot_ready("l1", st_a)
            s1.wait_main_slot_ready("l1", st_a)
            s1.wait_main_slot_ready("l1", st_b)
            self.assertEqual((st_a.waited, st_b.waited), ([ev_t1, ev_tail], [ev_t1, ev_tail]))
            self.assertEqual(
                (len(dvas), ups, s1._shared_reqs, set(s1._layer_ready_events), s1._shared_ready_event),
                (3, [0, 1], {"r0"}, {0, 1}, None),
            )
            self.assertEqual(
                (binds[:3], len(pf), s1._pending_main),
                ([([0], 7, "prev"), ("bcp", (0,), (49152,)), ([49152], 7, None)], 1, {}),
            )

            # 2-thread _BIND_FILTER_TLS isolation: peer call stays unfiltered while selected generator holds TLS
            bar, thr_err, tls_binds = threading.Barrier(2), [], []
            npu_raw = SimpleNamespace(
                bind_incremental_copy_addresses=lambda p, *a, **k: tls_binds.append(("orig", a[6])),
                BoundCopyPrefix=lambda *a: (bar.wait(timeout=2), tls_binds.append(("bcp", a[2]))),
            )
            lms._patch_bind_incremental(npu_raw)

            def _peer():
                try:
                    bar.wait(timeout=2)
                    npu_raw.bind_incremental_copy_addresses(pl, [o1], [0], [49152], [1], [576], 2, host_ptr=99)
                except BaseException as exc:
                    thr_err.append(exc)

            t_peer = threading.Thread(target=_peer)
            t_peer.start()
            with patch.object(lms._BIND_FILTER_TLS, "active_context", (s1, "r1", 0)):
                npu_raw.bind_incremental_copy_addresses(pl, [o1], [0], [49152], [1], [576], 2, host_ptr=11)
            t_peer.join(timeout=2)
            self.assertEqual((thr_err, sorted(tls_binds)), ([], [("bcp", (0,)), ("orig", 99)]))

            _ = eng.release_shared_cpu_sparse_request("r0"), gc.release_layerwise_prefill_dma_cache("r1")
            self.assertFalse(s1.finish(completion=SimpleNamespace(query=lambda: done[0], synchronize=nop)))
            self.assertEqual((rel, o0a.s, o1.s), ([], {"p": 1, "r": 1}, {"p": 1, "r": 1}))
            done[0] = True
            self.assertTrue(s1.try_retire())
            self.assertEqual((len(rel), o0a.s, o1.s), (2, {"p": 0, "r": 0}, {"p": 0, "r": 0}))

            dvas.clear()
            src._dva_cache.clear()
            reuse[0] = False
            run_s()
            src.step.finish(failed=True)
            eng.release_shared_cpu_sparse_request("r0")
            self.assertFalse(src.step.try_retire())
            self.assertEqual((len(dvas), len(rel), o0a.s), (6, 2, {"p": 1, "r": 1}))
            self.assertRaises(RuntimeError, src.close)
            for restore in reversed(src._restorers):
                restore()

            # Unuploaded layer raises RuntimeError on page_bases() and ready()
            mc3 = {n: (torch.empty((1550, 1, 1, 1)),) * 2 for n in ("l0", "l1", "l2")}
            impl3 = SimpleNamespace(
                lmcache_engine=eng,
                _submit_initial_layerwise_prefill_load=nop,
                submit_layerwise_prefill_load=nop,
                start_load_kv=lambda _: _st(None),
                _parent=impl._parent,
            )
            src3 = lms.LMCacheSelectedKVSource(SimpleNamespace(device="cpu"), impl3, ("l0", "l1", "l2"), mc3, **kw_s)
            src3.begin_runner_step(cm, ("r1", "r0"), (49152, 49152), (0, 16, 32), (tbl0, tbl1))
            impl3.start_load_kv(None)
            self.assertRaisesRegex(RuntimeError, "not uploaded", src3.step.page_bases, "l2")
            self.assertRaisesRegex(RuntimeError, "not uploaded", src3.step.ready, "l2")
            src3.step.finish(failed=True)

    def test_runtime_mla_async_prefetch_draft_retire_and_abort(self) -> None:
        nop = lambda *a, **k: None
        ow = {"l0": "l0", "l1": "l0", "l2": "l2"}
        z = lambda k, dt: torch.zeros((8, 128, 1, k), dtype=dt)
        mt = lambda n=4: SimpleNamespace(
            block_table=torch.zeros((1, 420), dtype=torch.int32), num_actual_tokens=n, block_size=128
        )
        nsrc = lambda: SimpleNamespace(step=None, _failed=False, close=nop)

        def mk_pl():
            calls, loads = [], []
            return SimpleNamespace(
                calls=calls,
                loads=loads,
                build=lambda t, h, s, q: (
                    calls.append(t.shape[0]),
                    (
                        torch.zeros((len(h), 128), dtype=torch.int32),
                        torch.tensor([[0, qv, qv, 0, 0, 0, 0, 0] for qv in q], dtype=torch.int32),
                    ),
                )[1],
                load_mla=lambda *a: (
                    calls.append("load_mla"),
                    loads.append((a, int(torch.npu.current_stream().npu_stream))),
                ),
                release_quiescent=lambda: calls.append("r"),
                close_quiescent=lambda: calls.append("c"),
            )

        def mk_st(cm, ls, evs, auto_pf=True):
            s = SimpleNamespace(
                requests=[lms.SelectedRequest("r0", 49152, 4, True, 49152)],
                connector_metadata=cm,
                _finished=False,
                finished=None,
                _closed=False,
                page_bases={n: torch.zeros((8, 2), dtype=torch.uint64) for n in ls}.get,
                ready=evs.get,
                wait_main_slot_ready=nop,
                prefetch_main=(lambda n, fn: fn()) if auto_pf else nop,
                wait_completion=nop,
                try_retire=lambda: setattr(s, "_closed", True) or True,
            )
            s.finish = lambda *, completion=None, failed=False: (
                setattr(s, "_finished", True),
                setattr(s, "finished", "failed" if failed else 1),
            )[-1]
            return s

        mk = lambda o, m, p, s: psr.SelectedKVRuntime(
            SimpleNamespace(device="cpu", vllm_config=_vcfg()),
            owner_by_layer=o,
            main_caches=m,
            source=s,
            planner=p,
            draft_layer_names=("l2",),
        )

        # Reject 1-plane cache
        with _fake_npu():
            self.assertRaises(ValueError, mk, ow, {n: (z(656, torch.int8),) for n in ow}, mk_pl(), nsrc())

        for dt in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dt), _fake_npu():
                caches = {n: (z(512, dt), z(64, dt)) for n in ow}
                src, pl = nsrc(), mk_pl()
                rt = mk(ow, caches, pl, src)
                evs = {n: torch.npu.Event() for n in ow}
                src.step = step = mk_st("cm", ow, evs)
                acfg = SimpleNamespace(selected_kv_config=skv.SelectedKVConfig(True))
                rn = SimpleNamespace(
                    ascend_config=acfg,
                    selected_kv_runtime=rt,
                    input_batch=SimpleNamespace(req_ids=["r0"], num_reqs=1, num_computed_tokens_cpu=[49152]),
                    query_start_loc=SimpleNamespace(np=[0, 4]),
                )
                metas = {n: mt() for n in ow}
                topk = torch.zeros((6, 1, 2048), dtype=torch.int32)
                c_stream = torch.npu.current_stream()

                with skv.selected_kv_forward(
                    rn, SimpleNamespace(kv_connector_metadata="cm"), metas, defer_finish=True
                ) as b:
                    self.assertIsNotNone(b)
                    b.load("l0", topk, caches["l0"], metas["l0"])
                    b.prefetch_next("l0", metas["l0"])
                    # Before load("l1"): l1 submitted on copy stream 2 (waits ready+stream1); stream 1 not waited yet
                    comp_ev = rt._prefetched["l1"]
                    self.assertEqual((len(pl.loads), pl.loads[1][1], comp_ev.recorded_on), (2, 2, 2))
                    self.assertIn(evs["l1"], rt._copy_stream.waited)
                    self.assertTrue(any(ev.recorded_on == 1 for ev in rt._copy_stream.waited))
                    self.assertNotIn(comp_ev, c_stream.waited)
                    # After load("l1"): no duplicate load_mla, and compute stream 1 waits on comp_ev
                    b.load("l1", topk, caches["l1"], metas["l1"])
                    self.assertEqual(len(pl.loads), 2)
                    self.assertIn(comp_ev, c_stream.waited)

                dm = {"l2": mt(1)}
                qsl = SimpleNamespace(query_start_loc_cpu=[0, 1], seq_lens_cpu=[49153])
                self.assertIsNotNone(skv.bind_selected_kv_draft_metadata(rn, dm, qsl))
                self.assertEqual(rt._dev_meta[0, :1].tolist(), [[0, 49152, 0, 4]])
                self.assertEqual(rt._dev_meta[1, :1].tolist(), [[0, 49152, 0, 1]])
                self.assertEqual(rt.request_metadata.tolist(), [[0, 49152, 0, 1]])
                with skv.selected_kv_draft_forward(rn):
                    dm["l2"].selected_kv.load("l2", topk[:1], caches["l2"], dm["l2"])
                self.assertEqual(pl.calls, [4, "load_mla", "load_mla", 1, "load_mla"])

                # Retire before close
                skv.retire_selected_kv_if_idle(rn)
                self.assertEqual(
                    (step._closed, rt._closed, rt.source_step, "r" in pl.calls, "c" in pl.calls),
                    (True, False, None, True, False),
                )
                skv.close_selected_kv_worker(SimpleNamespace(model_runner=rn))
                self.assertEqual((rt._closed, "c" in pl.calls), (True, True))

        # Deferred prefetch not submitted before load("l1")
        with _fake_npu():
            c2 = {n: (z(512, torch.bfloat16), z(64, torch.bfloat16)) for n in ow}
            rt_d = mk(ow, c2, mk_pl(), nsrc())
            m_d = {n: mt() for n in ow}
            rt_d.bind_step(mk_st("cm_d", ow, {n: None for n in ow}, auto_pf=False), ["r0"], [49152], [0, 4], m_d)
            tk_d = torch.zeros((4, 1, 2048), dtype=torch.int32)
            rt_d.load("l0", tk_d, c2["l0"], m_d["l0"])
            rt_d.prefetch_next("l0", m_d["l0"])
            self.assertRaisesRegex(
                RuntimeError,
                "Native slot readiness did not submit prefetch",
                rt_d.load,
                "l1",
                tk_d,
                c2["l1"],
                m_d["l1"],
            )
            rt_d.fail(rt_d.source_step)

        # Abort active steps, non-selected prepare isolation, and fail() -> bind_step() guard
        with _fake_npu():
            ow2 = {"l0": "l0", "l2": "l2"}
            c3 = {k: (z(512, torch.bfloat16), z(64, torch.bfloat16)) for k in ow2}
            ba = (["r0"], [49152], [0, 4], {k: mt() for k in ow2})
            (rt2 := mk(ow2, c3, mk_pl(), nsrc())).bind_step(s2 := mk_st("cm2", ow2, {k: None for k in ow2}), *ba)
            skv.abort_active_selected_kv_steps(deferred_only=True)
            self.assertIsNone(s2.finished)
            rt2._awaiting_draft = True
            skv.abort_active_selected_kv_steps(deferred_only=True)
            self.assertEqual(s2.finished, "failed")
            skv.close_selected_kv_worker(SimpleNamespace(model_runner=SimpleNamespace(selected_kv_runtime=rt2)))
            self.assertRaises(RuntimeError, rt2.close)

            mk(ow2, c3, mk_pl(), nsrc()).bind_step(s3 := mk_st("cm3", ow2, {k: None for k in ow2}), *ba)
            skv.abort_active_selected_kv_steps(deferred_only=False)
            self.assertEqual(s3.finished, "failed")

            src_p = nsrc()
            rt_p = mk(ow2, c3, mk_pl(), src_p)
            rn_p = SimpleNamespace(
                ascend_config=SimpleNamespace(selected_kv_config=skv.SelectedKVConfig(True)),
                selected_kv_runtime=rt_p,
                input_batch=SimpleNamespace(),
            )
            with (
                self.assertRaises(RuntimeError),
                skv.selected_kv_prepare_context(rn_p, SimpleNamespace(kv_connector_metadata=None)),
            ):
                src_p._staged_step = "stale"
                raise RuntimeError("boom")
            self.assertEqual((src_p._staged_step, rt_p._failed), (None, False))
            rt_p.fail()
            self.assertRaisesRegex(
                RuntimeError, "retains", rt_p.bind_step, mk_st("cm4", ow2, {k: None for k in ow2}), *ba
            )

    def test_multiconnector_unwrap_validation_and_deferred_gating(self) -> None:
        nop = lambda *a, **k: None
        cfg = skv.SelectedKVConfig(True)
        env_kw = {"VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": True, "VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS": 2}
        mc_s = {"kv_connector": "MooncakeConnectorV1", "kv_role": "kv_producer"}
        lmc_s = {"kv_connector": "LMCacheAscendConnectorV1Dynamic", "kv_role": "kv_producer"}
        tr = lambda name, role=None, children=None, mod="": {
            "kv_connector": name,
            **({"kv_role": role} if role is not None else {}),
            **({"kv_connector_extra_config": {"connectors": children}} if children is not None else {}),
            **({"kv_connector_module_path": mod} if mod else {}),
        }
        with patch.multiple(envs_ascend, **env_kw):
            for ok_tr in (
                tr("MultiConnector", "kv_producer", [mc_s, lmc_s]),
                tr("AscendMultiConnector", "kv_producer", [mc_s, lmc_s]),
                tr(
                    "MultiConnector",
                    "kv_producer",
                    [mc_s, tr("MultiConnector", "kv_both", [mc_s, {**lmc_s, "kv_role": "kv_both"}])],
                ),
                tr("LMCacheConnectorV1Dynamic", "kv_producer"),
                tr("LMCacheAscendConnector", "kv_both"),
            ):
                cfg.validate_deployment(_vcfg(tr=ok_tr))
            for bad_tr in (
                tr("MultiConnector", "kv_producer", [mc_s]),
                tr(
                    "MultiConnector",
                    "kv_producer",
                    [{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_consumer"}, lmc_s],
                ),
                tr("MultiConnector", "kv_producer", [mc_s, {**lmc_s, "kv_role": "kv_consumer"}]),
                tr("MultiConnector", "kv_consumer", [mc_s, lmc_s]),
                tr("MultiConnector", "kv_producer", "bad", mod="lmcache.fake"),
                tr("MultiConnector", "kv_producer", [lmc_s, tr("MultiConnector", "kv_producer", "bad")]),
                tr("MultiConnector", "kv_producer", [lmc_s, tr("MultiConnector", "kv_producer", [])]),
                tr("MultiConnector", "kv_producer", [None, lmc_s]),
                tr("MultiConnector", "kv_producer", [mc_s, {"kv_connector": lmc_s["kv_connector"]}]),
            ):
                self.assertRaisesRegex(ValueError, "Unsupported deployment", cfg.validate_deployment, _vcfg(tr=bad_tr))

        lmc_n = SimpleNamespace(_lmcache_engine=SimpleNamespace(selected_kv_source=None))
        impl_r, spec_r = skv._resolve_lmcache_target(
            SimpleNamespace(_connectors=[SimpleNamespace(), SimpleNamespace(_connectors=[lmc_n])])
        )
        self.assertEqual((impl_r is lmc_n._lmcache_engine, spec_r), (True, ((1, 2), (0, 1))))
        self.assertEqual(
            skv._unwrap_connector_metadata(
                SimpleNamespace(metadata=("s0", SimpleNamespace(metadata=("lm",)))),
                SimpleNamespace(_connector_meta_spec=spec_r),
            ),
            "lm",
        )
        for bad_g in (
            SimpleNamespace(_connectors=[SimpleNamespace()]),
            SimpleNamespace(
                _connectors=[lmc_n, SimpleNamespace(_lmcache_engine=SimpleNamespace(selected_kv_source=None))]
            ),
            SimpleNamespace(_connectors=[SimpleNamespace(_lmcache_engine=SimpleNamespace(selected_kv_source="taken"))]),
        ):
            self.assertRaisesRegex(RuntimeError, "Bad LMCache engine", skv._resolve_lmcache_target, bad_g)

        with _fake_npu():
            sib_calls, b_calls, orig_cnt = [], [], [0]
            o1 = _mk_obj(0x8000, 49152)
            fake_npu_m = types.ModuleType("lmcache_ascend.v1.npu_connector.npu_connectors")
            fake_npu_m.bind_incremental_copy_addresses = lambda p, *a, **k: b_calls.append(len(p.slot))
            fake_npu_m.BoundCopyPrefix = lambda *a: b_calls.append(a[2])
            fake_c_ops = types.ModuleType("lmcache_ascend.c_ops")
            fake_c_ops.get_device_ptr = lambda p, _: p + 0x100000
            load_done, bank_tail, gen_h = {}, {}, [None]

            def _bg(st, en, *a, **k):
                yield None
                for _ in range(4):
                    fake_npu_m.bind_incremental_copy_addresses(
                        _Plan(*([torch.tensor([0])] * 4)), [o1], st, en, [1], [576], 2, host_ptr=7
                    )
                    yield "ok"

            gc_e = SimpleNamespace(
                _layerwise_prefill_bank_count=2,
                batched_to_gpu=_bg,
                release_layerwise_prefill_dma_cache=nop,
                _layerwise_prefill_transfer_state=lambda: (None, None, None, load_done),
                _layerwise_prefill_bank_tail_events=bank_tail,
            )
            eng_e = SimpleNamespace(
                gpu_connector=gc_e,
                config=SimpleNamespace(chunk_size=1024),
                _prepare_shared_prefill_sources=nop,
                release_shared_cpu_sparse_request=nop,
            )
            ls_ok = SimpleNamespace(can_load=True, lmcache_cached_tokens=49152, dsa_cold_compact_load=False)
            lmc_meta = SimpleNamespace(
                requests=[
                    SimpleNamespace(
                        req_id="r_lmc",
                        is_decode_window_save=False,
                        block_allocation_mode="prefill_child",
                        load_spec=ls_ok,
                    )
                ]
            )
            sib_meta = SimpleNamespace(req="r_sib")
            outer_meta = SimpleNamespace(metadata=(sib_meta, lmc_meta))
            impl_e = SimpleNamespace(
                lmcache_engine=eng_e,
                use_layerwise=True,
                supports_layerwise_prefill_transfer_window=True,
                _deferred_layerwise_prefill_load_active=True,
                _submit_initial_layerwise_prefill_load=nop,
                submit_layerwise_prefill_load=lambda ln: gen_h[0].send([o1]) if ln in (layers[0], layers[1]) else None,
                start_load_kv=lambda _: (
                    orig_cnt.__setitem__(0, orig_cnt[0] + 1),
                    gen_h.__setitem__(0, gc_e.batched_to_gpu([0], [49152], req_id="r_lmc", kv_group=0)),
                    next(gen_h[0]),
                    gen_h[0].send([o1]),
                    gen_h[0].send([o1]),
                    impl_e._submit_initial_layerwise_prefill_load(),
                ),
            )
            sib_c = SimpleNamespace(
                bind_connector_metadata=lambda m: sib_calls.append(("b", m)),
                start_load_kv=lambda ctx: sib_calls.append("s"),
            )
            lmc_c = SimpleNamespace(
                _lmcache_engine=impl_e,
                _connector_metadata=None,
                _get_connector_metadata=lambda: lmc_c._connector_metadata,
                bind_connector_metadata=lambda m: setattr(lmc_c, "_connector_metadata", m),
                start_load_kv=lambda ctx: impl_e.start_load_kv(ctx),
            )
            impl_e._parent = lmc_c
            mconn = SimpleNamespace(
                _connectors=[sib_c, lmc_c],
                bind_connector_metadata=lambda m: [
                    c.bind_connector_metadata(s) for c, s in zip(mconn._connectors, m.metadata)
                ],
                start_load_kv=lambda ctx: [c.start_load_kv(ctx) for c in mconn._connectors],
            )
            layers = tuple(f"model.layers.{i}.self_attn" for i in range(4))
            sfc = {
                layers[i]: SimpleNamespace(
                    impl=SimpleNamespace(
                        kv_lora_rank=512,
                        qk_rope_head_dim=64,
                        **({"skip_topk": True} if i in (1, 2) else {"index_topk": 2048}),
                    )
                )
                for i in range(4)
            }
            kv_c = {
                n: (
                    torch.zeros((800, 128, 1, 512), dtype=torch.bfloat16),
                    torch.zeros((800, 128, 1, 64), dtype=torch.bfloat16),
                )
                for n in layers
            }
            tbl0 = np.vstack([np.array([0, 200] + [201] * 383, dtype=np.int32), np.arange(385, dtype=np.int32)])
            tbl1 = np.vstack([np.array([300] * 385, dtype=np.int32), np.arange(400, 785, dtype=np.int32)])
            btw = lambda arr: (SimpleNamespace(get_cpu_tensor=lambda a=arr: torch.from_numpy(a)),)
            rn = SimpleNamespace(
                device="cpu",
                vllm_config=_vcfg(tr=tr("MultiConnector", "kv_producer", [mc_s, lmc_s])),
                ascend_config=SimpleNamespace(selected_kv_config=cfg),
                enable_dsa_cp=False,
                layerwise_prefill_p_node=True,
                layerwise_prefill_bank_count=2,
                compilation_config=SimpleNamespace(static_forward_context=sfc),
                drafter=SimpleNamespace(attn_layer_names=(layers[3],)),
                selected_kv_runtime=None,
                input_batch=SimpleNamespace(
                    req_ids=["r_sib", "r_lmc"],
                    num_reqs=2,
                    num_computed_tokens_cpu=np.array([128, 49152], dtype=np.int32),
                    layerwise_prefill_block_tables=(btw(tbl0), btw(tbl1)),
                ),
                query_start_loc=SimpleNamespace(np=np.array([0, 8, 24], dtype=np.int32)),
            )
            wk = SimpleNamespace(vllm_config=rn.vllm_config, model_runner=rn, cache_config=SimpleNamespace())
            p_calls, waited_mc = [], []
            _Fn = lambda cb=None: type("_F", (), {"__call__": lambda s, *a: cb(*a) if cb else 0})()
            fl = SimpleNamespace(
                create_sparse_prefill_fixed_handle=_Fn(lambda p, c, hp: setattr(hp._obj, "value", 0xBEEF) or 0),
                destroy_sparse_prefill_fused_handle=_Fn(lambda h: p_calls.append("ds") or 0),
                launch_sparse_prefill_fused_plan=ctypes.c_void_p(0x1),
            )
            ml = SimpleNamespace(launch_sparse_prefill_bf16_mla_inplace=ctypes.c_void_p(0x2))
            fops = SimpleNamespace(
                load_library=nop,
                dsa_prefill_queue=SimpleNamespace(
                    batch_plan=lambda tk, sc, out, h, s, q, *a: (
                        p_calls.append("bp"),
                        out[1][: len(h)].copy_(
                            torch.tensor([[0, qv, qv, 0, 0, 0, 0, 0] for qv in q], dtype=torch.int32)
                        ),
                    ),
                    mla_inplace_copy=lambda *a: (
                        p_calls.append("mc"),
                        waited_mc.append(list(rn.selected_kv_runtime._copy_stream.waited)),
                    ),
                ),
            )
            mods = {
                "vllm.distributed.kv_transfer": SimpleNamespace(get_kv_transfer_group=lambda: mconn),
                "vllm_ascend.utils": SimpleNamespace(
                    AscendDeviceType=SimpleNamespace(A2=2, A3=1, _310P=0, A5=3), get_ascend_device_type=lambda: 1
                ),
                "lmcache_ascend": types.ModuleType("lmcache_ascend"),
                "lmcache_ascend.v1": types.ModuleType("lmcache_ascend.v1"),
                "lmcache_ascend.v1.npu_connector": types.ModuleType("lmcache_ascend.v1.npu_connector"),
                "lmcache_ascend.v1.npu_connector.npu_connectors": fake_npu_m,
                "lmcache_ascend.c_ops": fake_c_ops,
            }
            orig_st = impl_e.start_load_kv
            with (
                patch.multiple(envs_ascend, **env_kw),
                patch.dict(sys.modules, mods),
                patch.object(ctypes, "CDLL", side_effect=[fl, ml] * 2),
                patch.object(torch, "ops", fops),
            ):
                skv.reserve_selected_kv_workspace(wk, 2 << 30)
                skv.initialize_selected_kv(rn, None, kv_c)
                src = rn.selected_kv_runtime.source
                self.assertEqual((src._connector_meta_spec, impl_e.selected_kv_source is src), (((1, 2),), True))
                am = {
                    n: SimpleNamespace(
                        block_table=torch.zeros((2, 420), dtype=torch.int32), num_actual_tokens=24, block_size=128
                    )
                    for n in layers
                }
                with skv.selected_kv_prepare_context(rn, SimpleNamespace(kv_connector_metadata=outer_meta), am):
                    mconn.bind_connector_metadata(outer_meta)
                    mconn.start_load_kv(None)
                self.assertEqual(
                    (
                        src.step.connector_metadata is lmc_meta,
                        sib_calls,
                        0 in src.step.omitted_by_bank[0],
                        1 in src.step.omitted_by_bank[0],
                    ),
                    (True, [("b", sib_meta), "s"], False, True),
                )
                ev_l2, ev_b0, tk = torch.npu.Event(), torch.npu.Event(), torch.zeros((24, 1, 2048), dtype=torch.int32)
                with skv.selected_kv_forward(
                    rn, SimpleNamespace(kv_connector_metadata=outer_meta), am, defer_finish=True
                ) as b:
                    impl_e.submit_layerwise_prefill_load(-1)
                    b.load(layers[0], tk, kv_c[layers[0]], am[layers[0]])
                    b.prefetch_next(layers[0], am[layers[0]])
                    load_done[(0, 2)], bank_tail[(0, 0)] = {0: (0, ev_l2)}, (0, ev_b0)
                    impl_e.submit_layerwise_prefill_load(layers[0])
                    b.load(layers[1], tk, kv_c[layers[1]], am[layers[1]])
                    b.prefetch_next(layers[1], am[layers[1]])
                    b.load(layers[2], tk, kv_c[layers[2]], am[layers[2]])
                    impl_e.submit_layerwise_prefill_load(layers[1])
                dm = {
                    layers[3]: SimpleNamespace(
                        block_table=am[layers[3]].block_table, num_actual_tokens=2, block_size=128
                    )
                }
                skv.bind_selected_kv_draft_metadata(
                    rn, dm, SimpleNamespace(query_start_loc_cpu=[0, 1, 2], seq_lens_cpu=[129, 49153])
                )
                with skv.selected_kv_draft_forward(rn):
                    dm[layers[3]].selected_kv.load(layers[3], tk[:2], kv_c[layers[3]], dm[layers[3]])
                self.assertEqual(
                    (p_calls, ev_l2 in waited_mc[2], ev_b0 in waited_mc[2]),
                    (["bp", "mc", "mc", "mc", "bp", "mc"], True, True),
                )
                skv.retire_selected_kv_if_idle(rn)

                # (c) None / deferred_ok fallback & malformed outer metadata
                bad_sd = SimpleNamespace(
                    req_id="r_s",
                    is_decode_window_save=False,
                    is_sparse_decode=True,
                    block_allocation_mode="prefill_child",
                    load_spec=ls_ok,
                )
                bad_dn = SimpleNamespace(
                    req_id="r_d",
                    is_decode_window_save=False,
                    is_sparse_decode=False,
                    block_allocation_mode="dense",
                    load_spec=ls_ok,
                )
                for m_in, attr_ov in (
                    (None, None),
                    (SimpleNamespace(metadata=(sib_meta, None)), None),
                    (SimpleNamespace(metadata=(sib_meta, lmc_meta)), ("use_layerwise", False)),
                    (
                        SimpleNamespace(metadata=(sib_meta, lmc_meta)),
                        ("supports_layerwise_prefill_transfer_window", False),
                    ),
                    (
                        SimpleNamespace(metadata=(sib_meta, SimpleNamespace(requests=[lmc_meta.requests[0], bad_sd]))),
                        None,
                    ),
                    (
                        SimpleNamespace(metadata=(sib_meta, SimpleNamespace(requests=[lmc_meta.requests[0], bad_dn]))),
                        None,
                    ),
                ):
                    if attr_ov:
                        setattr(impl_e, attr_ov[0], attr_ov[1])
                    with skv.selected_kv_prepare_context(rn, SimpleNamespace(kv_connector_metadata=m_in), am):
                        mconn.bind_connector_metadata(m_in or SimpleNamespace(metadata=(None, None)))
                        mconn.start_load_kv(None)
                    self.assertEqual((src.step, src._failed), (None, False))
                    with skv.selected_kv_forward(rn, SimpleNamespace(kv_connector_metadata=m_in), am) as bf:
                        self.assertIsNone(bf)
                    if attr_ov:
                        setattr(impl_e, attr_ov[0], True)
                for bad_out in (SimpleNamespace(other=1), SimpleNamespace(metadata=(sib_meta,))):
                    with (
                        self.assertRaisesRegex(RuntimeError, "MultiConnector metadata"),
                        skv.selected_kv_prepare_context(rn, SimpleNamespace(kv_connector_metadata=bad_out), am),
                    ):
                        pass
                    self.assertIsNone(src._staged_step)
                skv.close_selected_kv_worker(wk)
                self.assertEqual(
                    (getattr(impl_e, "selected_kv_source", None), impl_e.start_load_kv is orig_st), (None, True)
                )

                rn.selected_kv_runtime = None
                skv.initialize_selected_kv(rn, None, kv_c)
                src2 = rn.selected_kv_runtime.source
                with skv.selected_kv_prepare_context(rn, SimpleNamespace(kv_connector_metadata=outer_meta), am):
                    mconn.bind_connector_metadata(outer_meta)
                    mconn.start_load_kv(None)
                with (
                    self.assertRaisesRegex(RuntimeError, "Connector metadata mismatch"),
                    skv.selected_kv_forward(
                        rn,
                        SimpleNamespace(
                            kv_connector_metadata=SimpleNamespace(
                                metadata=(sib_meta, SimpleNamespace(requests=lmc_meta.requests))
                            )
                        ),
                        am,
                    ),
                ):
                    pass
                self.assertIsNone(src2._staged_step)

                def _flip_off(_):
                    impl_e._deferred_layerwise_prefill_load_active = False

                lmc_c._connector_metadata = lmc_meta
                src_g = lms.LMCacheSelectedKVSource(
                    SimpleNamespace(device="cpu"),
                    impl_e,
                    layers,
                    kv_c,
                    get_device_ptr_fn=lambda p, _: p + 0x100000,
                    npu_connectors_module=fake_npu_m,
                )
                src_g.begin_runner_step(lmc_meta, ("r_lmc",), (49152,), (0, 16), (tbl0[1:2], tbl1[1:2]))
                with (
                    patch.object(impl_e, "start_load_kv", _flip_off),
                    self.assertRaisesRegex(RuntimeError, "deferred layerwise prefill load"),
                ):
                    src_g._start_load_kv(_flip_off, None)
                self.assertTrue(src_g._failed and src_g.step is None)

    def test_lazy_chunk_info_and_bank_plan_cache(self) -> None:
        nop = lambda *a, **k: None
        lt_calls = [0]

        class _LazyPageObj:
            is_pinned = True

            def __init__(self, ptr_base: int, tokens: int) -> None:
                self._ptr_base = ptr_base
                self._tokens = tokens

            def is_valid(self) -> bool:
                return True

            def layer_size_bytes(self, lid: int) -> int:
                return self._tokens * skv.MAIN_ROW_BYTES

            def layer_data_ptr(self, lid: int) -> int:
                return self._ptr_base + lid * 0x100

            def layer_tensor(self, lid: int):
                lt_calls[0] += 1
                raise AssertionError("eager layer_tensor forbidden")

            @property
            def tensor(self):
                raise RuntimeError("eager tensor forbidden")

            def pin(self) -> None:
                pass

            def unpin(self) -> None:
                pass

            def ref_count_up(self) -> None:
                pass

            def ref_count_down(self) -> None:
                pass

        class _NoLtPageObj(_LazyPageObj):
            layer_tensor = None  # type: ignore[assignment]

        with _fake_npu():
            dva_calls = [0]
            gc = SimpleNamespace(
                _layerwise_prefill_bank_count=2,
                batched_to_gpu=nop,
                release_layerwise_prefill_dma_cache=nop,
                _lmc_plane_num_tokens=lambda t, _k, _lid: t.shape[0],
            )
            eng = SimpleNamespace(
                gpu_connector=gc,
                _prepare_shared_prefill_sources=nop,
                release_shared_cpu_sparse_request=nop,
            )
            impl = SimpleNamespace(
                lmcache_engine=eng,
                _submit_initial_layerwise_prefill_load=nop,
                submit_layerwise_prefill_load=nop,
                start_load_kv=nop,
                _parent=SimpleNamespace(_kv_transfer_config=SimpleNamespace(kv_role="kv_producer")),
            )
            npu_m = SimpleNamespace(bind_incremental_copy_addresses=nop, BoundCopyPrefix=nop)
            layers = tuple(f"l{i}" for i in range(8))
            mc = {n: (torch.empty((160, 1, 1, 1)),) * 2 for n in layers}
            src = lms.LMCacheSelectedKVSource(
                SimpleNamespace(device="cpu"),
                impl,
                layers,
                mc,
                get_device_ptr_fn=lambda ptr, _sz: (
                    dva_calls.__setitem__(0, dva_calls[0] + 1),
                    (1 << 63) | ptr,
                )[1],
                npu_connectors_module=npu_m,
            )
            src._step_host, src._step_dev = src._host_bases, src._dev_bases

            # 1. _chunk_info lazy path + fallback chain + error validation
            p_lazy = _LazyPageObj(0x1000_0000_0003, 4096)
            p_no_lt = _NoLtPageObj(0x2000_0000_0007, 4096)
            self.assertEqual(src._chunk_info(p_lazy, 2), (4096, (1 << 63) | 0x1000_0000_0203))
            self.assertEqual(src._chunk_info(p_no_lt, 1), (4096, (1 << 63) | 0x2000_0000_0107))
            self.assertEqual(lt_calls[0], 0)

            t_fb = torch.empty((256, 576), dtype=torch.bfloat16)
            o_lt = SimpleNamespace(layer_tensor=lambda lid: t_fb, layer_data_ptr=lambda lid: 0x3000 + lid)
            self.assertEqual(src._chunk_info(o_lt, 1), (256, (1 << 63) | 0x3001))
            del gc._lmc_plane_num_tokens
            self.assertEqual(src._chunk_info(o_lt, 2), (256, (1 << 63) | 0x3002))
            self.assertEqual(src._chunk_info(t_fb, 0), (256, (1 << 63) | t_fb.data_ptr()))
            o_sz = SimpleNamespace(tensor=None, get_size=lambda: 512 * skv.MAIN_ROW_BYTES, data_ptr=0x4005)
            self.assertEqual(src._chunk_info(o_sz, 0), (512, (1 << 63) | 0x4005))
            gc._lmc_plane_num_tokens = lambda t, _k, _lid: t.shape[0]

            for bad_obj in (
                SimpleNamespace(layer_size_bytes=lambda _: 0, layer_data_ptr=lambda _: 0x100),
                SimpleNamespace(layer_size_bytes=lambda _: -skv.MAIN_ROW_BYTES, layer_data_ptr=lambda _: 0x100),
                SimpleNamespace(layer_size_bytes=lambda _: skv.MAIN_ROW_BYTES + 1, layer_data_ptr=lambda _: 0x100),
                SimpleNamespace(tensor=None, get_size=lambda: -1, data_ptr=0x100),
            ):
                self.assertRaisesRegex(ValueError, "Invalid MLA source byte size", src._chunk_info, bad_obj, 0)

            src_bad_ptr = lms.LMCacheSelectedKVSource(
                SimpleNamespace(device="cpu"),
                impl,
                layers,
                mc,
                get_device_ptr_fn=lambda _p, _s: 0,
                npu_connectors_module=npu_m,
            )
            self.assertRaisesRegex(
                RuntimeError,
                "Bad DDR ptr",
                src_bad_ptr._chunk_info,
                _LazyPageObj(0x5000, 128),
                0,
            )

            # 2. _bank_plans reuse across 4 layers (l0..l3), partial kept pages, 64-bit high DVA, & mismatch checks
            req = lms.SelectedRequest("r0", 16384, 128, True, 16384)
            # 128 pages total (16384 / 128): chunks [0..32) = 4096 tok, [32..64) = 4096 tok, [64..128) = 8192 tok
            # Bank 0 omits strict subset in chunk 0 and chunk 2 (chunk 1 completely kept -> uniq_idx = [0, 2])
            blks = (np.arange(128, dtype=np.int32) + 10)[None, :]
            b0_mask = np.zeros(128, dtype=bool)
            b0_mask[[0, 5, 31, 64, 99, 127]] = True
            b1_mask = np.zeros(128, dtype=bool)
            b1_mask[[1, 33, 70]] = True
            omitted = (
                frozenset(int(x) for x in blks[0, b0_mask]),
                frozenset(int(x) for x in blks[0, b1_mask]),
            )
            step = lms.LMCacheSelectedLoadStep(
                src, SimpleNamespace(requests=[]), [req], (blks, blks), omitted, src._step_dev
            )

            objs = [
                _LazyPageObj(0x1000_0000_0023, 4096),
                _NoLtPageObj(0x2000_0000_0045, 4096),
                _LazyPageObj(0x3000_0000_0067, 12288),
                _LazyPageObj(0x4000_0000_0089, 4096),
            ]
            starts = [0, 4096, 8192, 16384]
            ends = [4096, 8192, 16384, 20480]
            for lid in (0, 1):
                src._populate_bases(step, "r0", lid, objs, starts, ends)
            plan0, plan1 = step._bank_plans[("r0", 0)], step._bank_plans[("r0", 1)]
            for lid in (2, 3):
                src._populate_bases(step, "r0", lid, objs, starts, ends)

            self.assertEqual((len(step._bank_plans), lt_calls[0]), (2, 0))
            self.assertTrue(step._bank_plans[("r0", 0)] is plan0 and step._bank_plans[("r0", 1)] is plan1)
            host_u64 = src._step_host.numpy().view(np.uint64)
            chunk_pages = [(0, 0, 4096, 4096), (1, 32, 4096, 4096), (2, 64, 8192, 12288)]
            for lid in range(4):
                mask = b0_mask if lid % 2 == 0 else b1_mask
                self.assertTrue(np.all(host_u64[lid, blks[0, ~mask]] == 0))
                for p in np.flatnonzero(mask).tolist():
                    c_idx, p0, _, p_tok = next(
                        (ci, s0, l_tok, pt) for ci, s0, l_tok, pt in chunk_pages if p < s0 + l_tok // 128
                    )
                    base0 = np.uint64((1 << 63) | (objs[c_idx]._ptr_base + lid * 0x100))
                    row = np.uint64((p - p0) * skv.PAGE_ROWS)
                    exp0 = base0 + row * np.uint64(skv.MLA_NOPE_ROW_BYTES)
                    exp1 = base0 + np.uint64(p_tok * skv.MLA_NOPE_ROW_BYTES) + row * np.uint64(skv.MLA_PE_ROW_BYTES)
                    self.assertEqual(
                        (int(host_u64[lid, p + 10, 0]), int(host_u64[lid, p + 10, 1])), (int(exp0), int(exp1))
                    )

            # High-bit 64-bit dvas passthrough bypasses _get_device_ptr_fn and preserves physical plane stride
            dvas_in = [(1 << 63) | 0xA000_0000_0711, (1 << 63) | 0xB000_0000_0722, (1 << 63) | 0xC000_0000_0733]
            dva_before = dva_calls[0]
            src._populate_bases(step, "r0", 7, objs, starts, ends, dvas=dvas_in)
            self.assertEqual(dva_calls[0], dva_before)
            self.assertIs(step._bank_plans[("r0", 1)], plan1)
            self.assertTrue(np.all(host_u64[7, blks[0, ~b1_mask]] == 0))
            for p in np.flatnonzero(b1_mask).tolist():
                c_idx, p0, _, p_tok = next(
                    (ci, s0, l_tok, pt) for ci, s0, l_tok, pt in chunk_pages if p < s0 + l_tok // 128
                )
                base0 = np.uint64(dvas_in[c_idx])
                row = np.uint64((p - p0) * skv.PAGE_ROWS)
                exp0 = base0 + row * np.uint64(skv.MLA_NOPE_ROW_BYTES)
                exp1 = base0 + np.uint64(p_tok * skv.MLA_NOPE_ROW_BYTES) + row * np.uint64(skv.MLA_PE_ROW_BYTES)
                self.assertEqual((int(host_u64[7, p + 10, 0]), int(host_u64[7, p + 10, 1])), (int(exp0), int(exp1)))

            # Mismatch checks on l4/l6 (bank 0) & l5 (bank 1): truncated objs, non-contiguous ranges, small plane_tokens
            self.assertRaisesRegex(
                RuntimeError, "Incomplete DDR", src._populate_bases, step, "r0", 4, objs[:1], starts[:3], ends[:3]
            )
            self.assertRaisesRegex(
                RuntimeError,
                "Bad DDR",
                src._populate_bases,
                step,
                "r0",
                6,
                objs[:3],
                [0, 4224, 8192],
                [4096, 8192, 16384],
            )
            small_objs = [_LazyPageObj(0x1000_0000_0023, 2048), objs[1], objs[2]]
            self.assertRaisesRegex(
                ValueError,
                "Chunk capacity P=2048 < 4096",
                src._populate_bases,
                step,
                "r0",
                5,
                small_objs,
                starts[:3],
                ends[:3],
            )
            step.finish(failed=True)

    def test_domain_padding_alignment_and_tail_generation_filter(self) -> None:
        nop = lambda *a, **k: None
        with _fake_npu():
            o_blk = _mk_obj(0x9000, 49152)

            def _btg(*a, **k):
                while True:
                    yield

            gc = SimpleNamespace(
                _layerwise_prefill_bank_count=2,
                batched_to_gpu=_btg,
                release_layerwise_prefill_dma_cache=nop,
                _layerwise_prefill_transfer_state=lambda: (None, {0: 5}, None, {}),
                _layerwise_prefill_bank_tail_events={},
            )
            eng = SimpleNamespace(
                gpu_connector=gc,
                _prepare_shared_prefill_sources=lambda *a: None,
                release_shared_cpu_sparse_request=nop,
            )
            ls = SimpleNamespace(can_load=True, lmcache_cached_tokens=49152, dsa_cold_compact_load=False)
            cm = SimpleNamespace(
                requests=[
                    SimpleNamespace(
                        req_id="r_cur",
                        is_decode_window_save=False,
                        block_allocation_mode="prefill_child",
                        load_spec=ls,
                    )
                ]
            )
            impl = SimpleNamespace(
                lmcache_engine=eng,
                _submit_initial_layerwise_prefill_load=nop,
                submit_layerwise_prefill_load=nop,
                _parent=SimpleNamespace(
                    _get_connector_metadata=lambda: cm,
                    _kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
                ),
            )

            def _st(_):
                g = gc.batched_to_gpu([0], [49152], req_id="r_cur", kv_group=0)
                next(g)
                g.send([o_blk])
                g.send([o_blk])

            impl.start_load_kv = _st
            mc = {n: (torch.empty((773, 1, 1, 1)),) * 2 for n in ("l0", "l1")}
            npu_m = SimpleNamespace(bind_incremental_copy_addresses=nop, BoundCopyPrefix=nop)
            src = lms.LMCacheSelectedKVSource(
                SimpleNamespace(device="cpu"),
                impl,
                ("l0", "l1"),
                mc,
                get_device_ptr_fn=lambda p, _: p + 0x100000,
                npu_connectors_module=npu_m,
            )
            self.assertEqual(src._host_bases.shape[1], 776)

            # Use max block id 768 so active_domain == 769 (non-multiple of 4) -> stride_domain == 772
            tbl0 = np.arange(384, 769, dtype=np.int32)[None, :]  # max id 768 -> active_domain 769
            tbl1 = np.arange(130, 515, dtype=np.int32)[None, :]
            src.begin_runner_step(cm, ("r_cur",), (49152,), (0, 16), (tbl0, tbl1))
            impl.start_load_kv(None)
            step = src.step
            self.assertIsNotNone(step)
            self.assertEqual(src._step_host.shape[1], 772)
            pb0, pb1 = step.page_bases("l0"), step.page_bases("l1")
            self.assertEqual((tuple(pb0.shape), tuple(pb1.shape)), ((769, 2), (769, 2)))
            self.assertTrue(pb0.is_contiguous() and pb1.is_contiguous())
            diff = pb1.data_ptr() - pb0.data_ptr()
            self.assertEqual((diff % 64, diff), (0, 772 * 16))

            # (b) Stale generation tail event filtered; matching generation waited; () and (True, ev) do not crash
            ev_stale, ev_curr, ev_bool = torch.npu.Event(), torch.npu.Event(), torch.npu.Event()
            gc._layerwise_prefill_bank_tail_events = {
                (0, 0): (4, ev_stale, 0, "r_old"),
                (0, 1): (5, ev_curr, 0, "r_cur"),
            }
            step._slot_events.clear()
            src._mark_slots_ready([0, 1])
            st0, st1 = _mk_stream(20), _mk_stream(21)
            step.wait_main_slot_ready("l0", st0)
            step.wait_main_slot_ready("l1", st1)
            self.assertEqual((st0.waited, st1.waited), ([], [ev_curr]))

            gc._layerwise_prefill_bank_tail_events = {(0, 0): (), (0, 1): (True, ev_bool)}
            step._slot_events.clear()
            step._waited_streams.clear()
            src._mark_slots_ready([0, 1])
            step.wait_main_slot_ready("l0", st0)
            step.wait_main_slot_ready("l1", st1)
            self.assertEqual((st0.waited, step._slot_events[0]), ([], ()))
            self.assertIn(ev_bool, st1.waited)

            # Per-bank generation priority: (0, 1): 6 overrides fallback 0: 5
            ev0, ev1, ev2 = torch.npu.Event(), torch.npu.Event(), torch.npu.Event()
            gc._layerwise_prefill_transfer_state = lambda: (None, {(0, 1): 6, 0: 5}, None, {})
            gc._layerwise_prefill_bank_tail_events = {(0, 0): (5, ev0), (0, 1): (5, ev1)}
            step._slot_events.clear()
            step._waited_streams.clear()
            st0.waited.clear()
            st1.waited.clear()
            src._mark_slots_ready([0, 1])
            step.wait_main_slot_ready("l0", st0)
            step.wait_main_slot_ready("l1", st1)
            self.assertEqual((st0.waited, st1.waited), ([ev0], []))
            gc._layerwise_prefill_bank_tail_events = {(0, 1): (6, ev2)}
            step._slot_events.pop(1, None)
            step._waited_streams.clear()
            src._mark_slots_ready([1])
            step.wait_main_slot_ready("l1", st1)
            self.assertEqual(st1.waited, [ev2])
            step.finish(failed=True)

    def test_dsa_cp_gates_workspace_and_split_mask_draft_empty_padding(self) -> None:
        nop = lambda *a, **k: None
        cfg = skv.SelectedKVConfig.from_dict({"enabled": True})
        kw = {"VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": True, "VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS": 2}
        dt_ns = SimpleNamespace(A2=2, A3=1, _310P=0, A5=3)
        u_a3 = {"vllm_ascend.utils": SimpleNamespace(AscendDeviceType=dt_ns, get_ascend_device_type=lambda: dt_ns.A3)}

        # 1. Deployment & topology gates
        with patch.dict(envs_ascend.__dict__, {**kw, "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": 2}):
            self.assertRaisesRegex(ValueError, "FLASHCOMM2", cfg.validate_deployment, _vcfg())

        ctx_cp = {
            "model.layers.0.self_attn": SimpleNamespace(
                impl=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048, enable_dsa_cp=True)
            ),
            "model.layers.1.self_attn": SimpleNamespace(
                impl=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64, skip_topk=True, enable_dsa_cp=True)
            ),
        }
        vcfg_tp4 = _vcfg(
            par=SimpleNamespace(
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
            )
        )
        with patch.multiple(envs_ascend, **kw), patch.dict(sys.modules, u_a3):
            for bad_axis in ("pcp_size", "dcp_size"):
                mr_bad = SimpleNamespace(
                    ascend_config=SimpleNamespace(selected_kv_config=cfg),
                    layerwise_prefill_p_node=True,
                    layerwise_prefill_bank_count=2,
                    compilation_config=SimpleNamespace(static_forward_context=ctx_cp),
                    **{bad_axis: 2},
                )
                self.assertRaisesRegex(
                    ValueError,
                    "Requires P-node",
                    skv.reserve_selected_kv_workspace,
                    SimpleNamespace(vllm_config=vcfg_tp4, model_runner=mr_bad),
                    2 << 30,
                )

            mr_otp = SimpleNamespace(
                compilation_config=SimpleNamespace(
                    static_forward_context={
                        "model.layers.0.self_attn": SimpleNamespace(
                            impl=SimpleNamespace(
                                kv_lora_rank=512,
                                qk_rope_head_dim=64,
                                index_topk=2048,
                                enable_dsa_cp=True,
                                enable_dsa_cp_with_o_proj_tp=True,
                            )
                        )
                    }
                )
            )
            self.assertRaisesRegex(ValueError, "Bad MLA/draft", skv._resolve_topology, mr_otp)

            mr_mix = SimpleNamespace(
                compilation_config=SimpleNamespace(
                    static_forward_context={
                        "model.layers.0.self_attn": SimpleNamespace(
                            impl=SimpleNamespace(
                                kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048, enable_dsa_cp=True
                            )
                        ),
                        "model.layers.1.self_attn": SimpleNamespace(
                            impl=SimpleNamespace(
                                kv_lora_rank=512, qk_rope_head_dim=64, skip_topk=True, enable_dsa_cp=False
                            )
                        ),
                    }
                )
            )
            self.assertRaisesRegex(ValueError, "agree on enable_dsa_cp", skv._resolve_topology, mr_mix)

            mr_cp_ok = SimpleNamespace(
                ascend_config=SimpleNamespace(selected_kv_config=cfg),
                pcp_size=1,
                dcp_size=1,
                layerwise_prefill_p_node=True,
                layerwise_prefill_bank_count=2,
                compilation_config=SimpleNamespace(static_forward_context=ctx_cp),
            )
            wk_cp = SimpleNamespace(
                vllm_config=vcfg_tp4, model_runner=mr_cp_ok, cache_config=SimpleNamespace(num_gpu_blocks_override=16)
            )
            nb_cp = skv.reserve_selected_kv_workspace(wk_cp, 2 << 30)
            self.assertEqual(nb_cp, mr_cp_ok._selected_kv_budget.native_budget_bytes)
            self.assertTrue(mr_cp_ok._selected_kv_budget._topology[3])

        self.assertEqual(psr.selected_workspace_bytes(8192, 8), 105122304 + 75 * 1024)
        self.assertEqual(
            psr.selected_workspace_bytes(16384, 8, dsa_cp=True, cp_tp_size=4),
            psr.selected_workspace_bytes(4096, 8) + 4 * 4096 * skv.TOPK,
        )

        # Shared helpers for runtime DSA-CP tests
        z = lambda k: torch.zeros((8, 128, 1, k), dtype=torch.bfloat16)
        nsrc = lambda: SimpleNamespace(step=None, _failed=False, close=nop)

        def mk_pl():
            built = []
            loads = []
            return SimpleNamespace(
                built=built,
                loads=loads,
                build=lambda t, h, s, q: (
                    built.append(t),
                    (
                        torch.zeros((len(h), 128), dtype=torch.int32),
                        torch.tensor([[0, qv, qv, 0, 0, 0, 0, 0] for qv in q], dtype=torch.int32),
                    ),
                )[1],
                load_mla=lambda *a: loads.append(a),
                release_quiescent=nop,
                close_quiescent=nop,
            )

        def mk_st(reqs, layers=("l0", "l1")):
            s = SimpleNamespace(
                requests=reqs,
                _finished=False,
                _closed=False,
                page_bases={n: torch.zeros((8, 2), dtype=torch.uint64) for n in layers}.get,
                ready={n: torch.npu.Event() for n in layers}.get,
                wait_main_slot_ready=nop,
                prefetch_main=lambda n, fn: fn(),
                wait_completion=nop,
                try_retire=lambda: setattr(s, "_closed", True) or True,
            )
            s.finish = lambda *, completion=None, failed=False: setattr(s, "_finished", True)
            return s

        def mk_rt(pl, src, *, dsa_cp=True, max_tokens=8192, draft_layers=(), tp=4):
            ow = {"l0": "l0", "l1": "l0"}
            if draft_layers:
                ow["l2"] = "l2"
            caches = {n: (z(512), z(64)) for n in ow}
            vc = _vcfg(
                par=SimpleNamespace(
                    tensor_parallel_size=tp if dsa_cp else 1,
                    pipeline_parallel_size=1,
                    prefill_context_parallel_size=1,
                    decode_context_parallel_size=1,
                ),
                sched=SimpleNamespace(async_scheduling=True, max_num_batched_tokens=max_tokens, max_num_seqs=8),
            )
            return psr.SelectedKVRuntime(
                SimpleNamespace(device="cpu", vllm_config=vc),
                owner_by_layer=ow,
                main_caches=caches,
                source=src,
                planner=pl,
                draft_layer_names=draft_layers,
                dsa_cp=dsa_cp,
            ), caches

        def mk_metas(n, cp_ctx, layer_names=("l0", "l1")):
            return {
                ln: SimpleNamespace(
                    block_table=torch.zeros((8, 1200), dtype=torch.int32),
                    num_actual_tokens=n,
                    block_size=128,
                    dsa_cp_context=cp_ctx,
                )
                for ln in layer_names
            }

        # 2. BS=1, 128K + 1K (TP=4) split-request & _plan_topk_buf masking across Ranks 0..3
        with _fake_npu():
            for rank in range(4):
                src, pl = nsrc(), mk_pl()
                rt, caches = mk_rt(pl, src, dsa_cp=True)
                src.step = step = mk_st([lms.SelectedRequest("r0", 131072, 1024, True, 131072)])
                cp_ctx = SimpleNamespace(
                    local_start=256 * rank,
                    local_end=256 * (rank + 1),
                    local_end_with_pad=256 * (rank + 1),
                    num_tokens_pad=1024,
                )
                metas = mk_metas(1024, cp_ctx, ("l0", "l1"))
                rt.bind_step(step, ("r0",), (131072,), (0, 1024), metas)
                self.assertEqual(rt.request_metadata.tolist(), [[0, 131072, 0, 256]])
                q_end = 131072 + 256 * (rank + 1)
                topk = torch.full((256, 1, skv.TOPK), -1, dtype=torch.int32)
                topk[0, 0, 0] = 0
                topk[0, 0, 1] = 131071
                topk[0, 0, 2] = 131072
                topk[0, 0, 3] = 131072 + 256 * rank
                topk[0, 0, 4] = q_end - 1
                topk[0, 0, 5] = q_end
                topk[0, 0, 6] = -2
                topk_orig = topk.clone()
                rt.load("l0", topk, caches["l0"], metas["l0"])
                rt.prefetch_next("l0", metas["l0"])
                rt.load("l1", topk, caches["l1"], metas["l1"])
                rt.finish_step()
                built = pl.built[0]
                self.assertTrue(torch.equal(topk, topk_orig))
                if rank == 0:
                    self.assertEqual(built.data_ptr(), topk.data_ptr())
                    self.assertEqual(built[0, 0, :7].tolist(), [0, 131071, 131072, 131072, q_end - 1, q_end, -2])
                else:
                    self.assertEqual(built.data_ptr(), rt._plan_topk_buf.data_ptr())
                    self.assertEqual(built[0, 0, :7].tolist(), [0, 131071, -1, -1, -1, q_end, -2])
                rt.close()

        # 2b. BS=1, 1M (H=1,047,552 + Q=1,024, TP=4) bind_step acceptance & H+Q>1M rejection
        with _fake_npu():
            src, pl = nsrc(), mk_pl()
            rt, caches_1m = mk_rt(pl, src, dsa_cp=True)
            cp_ctx = SimpleNamespace(local_start=256, local_end=512, local_end_with_pad=512, num_tokens_pad=1024)
            metas_1m = {
                ln: SimpleNamespace(
                    block_table=torch.zeros((8, 8192), dtype=torch.int32),
                    num_actual_tokens=1024,
                    block_size=128,
                    dsa_cp_context=cp_ctx,
                )
                for ln in ("l0", "l1")
            }
            src.step = step_1m = mk_st([lms.SelectedRequest("r_1m", 1047552, 1024, True, 1047552)])
            rt.bind_step(step_1m, ("r_1m",), (1047552,), (0, 1024), metas_1m)
            self.assertEqual(rt.request_metadata.tolist(), [[0, 1047552, 0, 256]])
            topk_1m = torch.zeros((256, 1, skv.TOPK), dtype=torch.int32)
            rt.load("l0", topk_1m, caches_1m["l0"], metas_1m["l0"])
            rt.prefetch_next("l0", metas_1m["l0"])
            rt.load("l1", topk_1m, caches_1m["l1"], metas_1m["l1"])
            rt.finish_step()
            src.step = step_bad = mk_st([lms.SelectedRequest("r_bad", 1047552, 1025, True, 1047552)])
            with self.assertRaisesRegex(ValueError, "Invalid selected batch H/Q"):
                rt.bind_step(step_bad, ("r_bad",), (1047552,), (0, 1025), metas_1m)
            rt.fail(step_bad)
            self.assertRaisesRegex(RuntimeError, "retained", rt.close)

        # 2c. 1M MTP draft + max reachable H=1,048,575, Q=1 (TP=4, Rank 0, 18-aligned block_table) & draft rejection
        with _fake_npu():
            src, pl = nsrc(), mk_pl()
            rt, caches = mk_rt(pl, src, dsa_cp=True, draft_layers=("l2",))
            cp_r0 = SimpleNamespace(local_start=0, local_end=1, local_end_with_pad=1, num_tokens_pad=4)
            tbl_1m = torch.zeros((8, 8224), dtype=torch.int32)[:, :8208]
            metas_m = {
                ln: SimpleNamespace(
                    block_table=tbl_1m,
                    num_actual_tokens=1,
                    block_size=128,
                    dsa_cp_context=cp_r0,
                )
                for ln in ("l0", "l1", "l2")
            }
            src.step = step_m = mk_st([lms.SelectedRequest("r_m", 1048575, 1, True, 1047552)], ("l0", "l1", "l2"))
            rt.bind_step(step_m, ("r_m",), (1048575,), (0, 1), metas_m)
            self.assertEqual(rt.request_metadata.tolist(), [[0, 1048575, 0, 1]])
            topk_m = torch.zeros((1, 1, skv.TOPK), dtype=torch.int32)
            rt.load("l0", topk_m, caches["l0"], metas_m["l0"])
            rt.prefetch_next("l0", metas_m["l0"])
            rt.load("l1", topk_m, caches["l1"], metas_m["l1"])
            rt.defer_to_draft_step()
            draft_metas = {"l2": metas_m["l2"]}
            rt.bind_draft_step(draft_metas, SimpleNamespace(query_start_loc_cpu=[0, 1], seq_lens_cpu=[1048576]))
            rt.load("l2", topk_m, caches["l2"], draft_metas["l2"])
            rt.finish_step()
            rt._retire_previous()
            self.assertTrue(step_m._closed)
            rt.close()

            src_d, pl_d = nsrc(), mk_pl()
            rt_d, caches_d = mk_rt(pl_d, src_d, dsa_cp=True, draft_layers=("l2",))
            src_d.step = step_d = mk_st([lms.SelectedRequest("r_d", 1048575, 1, True, 1047552)], ("l0", "l1", "l2"))
            rt_d.bind_step(step_d, ("r_d",), (1048575,), (0, 1), metas_m)
            rt_d.load("l0", topk_m, caches_d["l0"], metas_m["l0"])
            rt_d.prefetch_next("l0", metas_m["l0"])
            rt_d.load("l1", topk_m, caches_d["l1"], metas_m["l1"])
            rt_d.defer_to_draft_step()
            with self.assertRaisesRegex(ValueError, "Bad draft history"):
                rt_d.bind_draft_step(
                    {"l2": metas_m["l2"]},
                    SimpleNamespace(query_start_loc_cpu=[0, 2], seq_lens_cpu=[1048577]),
                )
            rt_d.fail(step_d)
            self.assertRaisesRegex(RuntimeError, "retained", rt_d.close)

        # 2d. TP=8 DSA-CP straddling + 1-token MTP + mixed selected/native batch at H+Q=1,048,576 (Rank 7, limit0==1M)
        with _fake_npu():
            src, pl = nsrc(), mk_pl()
            rt, caches = mk_rt(pl, src, dsa_cp=True, draft_layers=("l2",), tp=8)
            tbl_8k = torch.zeros((8, 8224), dtype=torch.int32)[:, :8208]
            cp_r7 = SimpleNamespace(local_start=1008, local_end=1152, local_end_with_pad=1152, num_tokens_pad=1152)
            metas_tp8 = {
                ln: SimpleNamespace(
                    block_table=tbl_8k,
                    num_actual_tokens=1152,
                    block_size=128,
                    dsa_cp_context=cp_r7,
                )
                for ln in ("l0", "l1", "l2")
            }
            reqs_tp8 = [
                lms.SelectedRequest("r0", 1047552, 1024, True, 1047552),
                lms.SelectedRequest("r1", 0, 128, False, 0),
            ]
            src.step = step_tp8 = mk_st(reqs_tp8, ("l0", "l1", "l2"))
            rt.bind_step(step_tp8, ("r0", "r1"), (1047552, 0), (0, 1024, 1152), metas_tp8)
            self.assertEqual(rt.request_metadata.tolist(), [[0, 1047552, 0, 16]])
            topk_r7 = torch.full((144, 1, skv.TOPK), -1, dtype=torch.int32)
            topk_r7[0, 0, :6] = torch.tensor([0, 1047551, 1047552, 1048560, 1048575, 1048576], dtype=torch.int32)
            topk_r7_orig = topk_r7.clone()
            rt.load("l0", topk_r7, caches["l0"], metas_tp8["l0"])
            rt.prefetch_next("l0", metas_tp8["l0"])
            rt.load("l1", topk_r7, caches["l1"], metas_tp8["l1"])
            built = pl.built[0]
            self.assertTrue(torch.equal(topk_r7, topk_r7_orig))
            self.assertEqual(built.data_ptr(), rt._plan_topk_buf.data_ptr())
            self.assertEqual(built[0, 0, :6].tolist(), [0, 1047551, -1, -1, -1, 1048576])
            rt.defer_to_draft_step()

            cp_draft_r7 = SimpleNamespace(local_start=903, local_end=1025, local_end_with_pad=1032, num_tokens_pad=1032)
            draft_metas_tp8 = {
                "l2": SimpleNamespace(
                    block_table=tbl_8k, num_actual_tokens=1025, block_size=128, dsa_cp_context=cp_draft_r7
                )
            }
            rt.bind_draft_step(
                draft_metas_tp8, SimpleNamespace(query_start_loc_cpu=[0, 1024, 1025], seq_lens_cpu=[1048576, 129])
            )
            rt.load("l2", torch.zeros((129, 1, skv.TOPK), dtype=torch.int32), caches["l2"], draft_metas_tp8["l2"])
            rt.finish_step()
            rt._retire_previous()
            self.assertTrue(step_tp8._closed)
            rt.close()

        # 3. BS=8, 8 * (128K + 128) (TP=4) multi-request filtering + MTP bind_draft_step
        with _fake_npu():
            src, pl = nsrc(), mk_pl()
            rt, caches = mk_rt(pl, src, dsa_cp=True, draft_layers=("l2",))
            req_ids = tuple(f"r{i}" for i in range(8))
            src.step = step = mk_st(
                [lms.SelectedRequest(rid, 131072, 128, True, 131072) for rid in req_ids], ("l0", "l1", "l2")
            )
            cp_r1 = SimpleNamespace(local_start=256, local_end=512, local_end_with_pad=512, num_tokens_pad=1024)
            metas = mk_metas(1024, cp_r1, ("l0", "l1", "l2"))
            rt.bind_step(step, req_ids, [131072] * 8, [128 * i for i in range(9)], metas)
            self.assertEqual(rt._active_selected_indices, (2, 3))
            self.assertEqual(rt.request_metadata.tolist(), [[2, 131072, 0, 128], [3, 131072, 128, 128]])
            self.assertEqual(rt._query_offsets, [0, 0])
            topk_r1 = torch.zeros((256, 1, skv.TOPK), dtype=torch.int32)
            rt.load("l0", topk_r1, caches["l0"], metas["l0"])
            self.assertEqual(pl.built[0].data_ptr(), topk_r1.data_ptr())
            rt.prefetch_next("l0", metas["l0"])
            rt.load("l1", topk_r1, caches["l1"], metas["l1"])
            rt.defer_to_draft_step()

            cp_draft = SimpleNamespace(local_start=2, local_end=4, local_end_with_pad=4, num_tokens_pad=8)
            draft_metas = mk_metas(8, cp_draft, ("l2",))
            common_meta = SimpleNamespace(query_start_loc_cpu=list(range(9)), seq_lens_cpu=[131073] * 8)
            rt.bind_draft_step(draft_metas, common_meta)
            self.assertEqual(rt._active_selected_indices, (2, 3))
            self.assertEqual(rt.request_metadata.tolist(), [[2, 131072, 0, 1], [3, 131072, 1, 1]])
            rt.load("l2", torch.zeros((2, 1, skv.TOPK), dtype=torch.int32), caches["l2"], draft_metas["l2"])
            rt.finish_step()
            rt._retire_previous()
            self.assertTrue(step._closed)
            rt.close()

        # 4. Empty rank-local slice (mixed hit/miss and short batch < TP)
        with _fake_npu():
            c_stream = torch.npu.current_stream()
            # 4a. Mixed hit/miss: Rank 2 owns [256, 384) of selected=False req (TP=4, num_tokens_pad=512)
            src, pl = nsrc(), mk_pl()
            rt, caches = mk_rt(pl, src, dsa_cp=True)
            src.step = step = mk_st(
                [lms.SelectedRequest("r0", 49152, 256, True, 49152), lms.SelectedRequest("r1", 128, 256, False, 0)]
            )
            cp_mix = SimpleNamespace(local_start=256, local_end=384, local_end_with_pad=384, num_tokens_pad=512)
            metas = mk_metas(512, cp_mix)
            rt.bind_step(step, ("r0", "r1"), (49152, 128), (0, 256, 512), metas)
            self.assertEqual((rt._active_selected_indices, rt.request_metadata), ((), None))
            topk_mix = torch.zeros((128, 1, skv.TOPK), dtype=torch.int32)
            rt.load("l0", topk_mix, caches["l0"], metas["l0"])
            rt.prefetch_next("l0", metas["l0"])
            rt.load("l1", topk_mix, caches["l1"], metas["l1"])
            rt.finish_step()
            rt._retire_previous()
            self.assertEqual((len(pl.built), len(pl.loads), step._closed), (0, 0, True))
            self.assertEqual(len(c_stream.waited), 2)
            rt.close()

            # 4b. Short batch (num_actual_tokens=2, TP=4): Rank 2 has local_valid_tokens=0, local_topk_rows=1
            c_stream.waited.clear()
            src2, pl2 = nsrc(), mk_pl()
            rt2, caches2 = mk_rt(pl2, src2, dsa_cp=True)
            src2.step = step2 = mk_st([lms.SelectedRequest("r0", 49152, 2, True, 49152)])
            cp_short = SimpleNamespace(local_start=2, local_end=2, local_end_with_pad=3, num_tokens_pad=4)
            metas2 = mk_metas(2, cp_short)
            rt2.bind_step(step2, ("r0",), (49152,), (0, 2), metas2)
            self.assertEqual((rt2._local_valid_tokens, rt2._local_topk_rows, rt2.request_metadata), (0, 1, None))
            topk_short = torch.zeros((1, 1, skv.TOPK), dtype=torch.int32)
            rt2.load("l0", topk_short, caches2["l0"], metas2["l0"])
            rt2.prefetch_next("l0", metas2["l0"])
            rt2.load("l1", topk_short, caches2["l1"], metas2["l1"])
            rt2.finish_step()
            rt2._retire_previous()
            self.assertEqual((len(pl2.built), len(pl2.loads), step2._closed), (0, 0, True))
            self.assertEqual(len(c_stream.waited), 2)
            rt2.close()

            # 4c. Reject mismatched-TP dsa_cp_context (num_tokens_pad != local_rows * cp_tp_size or unaligned start)
            for bad_cp in (
                SimpleNamespace(local_start=256, local_end=512, local_end_with_pad=512, num_tokens_pad=512),
                SimpleNamespace(local_start=64, local_end=192, local_end_with_pad=192, num_tokens_pad=512),
            ):
                src_bad, pl_bad = nsrc(), mk_pl()
                rt_bad, _ = mk_rt(pl_bad, src_bad, dsa_cp=True)
                step_bad = mk_st([lms.SelectedRequest("r0", 49152, 512, True, 49152)])
                self.assertRaisesRegex(
                    ValueError,
                    "Bad dsa_cp_context",
                    rt_bad.bind_step,
                    step_bad,
                    ("r0",),
                    (49152,),
                    (0, 512),
                    mk_metas(512, bad_cp),
                )
                rt_bad.fail(step_bad)
                self.assertRaisesRegex(RuntimeError, "retained", rt_bad.close)

        # 5. Trailing SP padding & fail-closed metadata mismatch
        with _fake_npu():
            src, pl = nsrc(), mk_pl()
            rt, caches = mk_rt(pl, src, dsa_cp=True)
            src.step = step = mk_st([lms.SelectedRequest("r0", 49152, 5, True, 49152)])
            cp_pad = SimpleNamespace(local_start=4, local_end=5, local_end_with_pad=6, num_tokens_pad=8)
            metas = mk_metas(5, cp_pad)
            rt.bind_step(step, ("r0",), (49152,), (0, 5), metas)
            self.assertRaisesRegex(
                ValueError,
                "Invalid selected TopK",
                rt.load,
                "l0",
                torch.zeros((1, 1, skv.TOPK), dtype=torch.int32),
                caches["l0"],
                metas["l0"],
            )
            rt.load("l0", torch.zeros((2, 1, skv.TOPK), dtype=torch.int32), caches["l0"], metas["l0"])
            self.assertEqual(pl.built[0].shape[0], 1)
            rt.fail(step)
            self.assertTrue(step._finished)
            self.assertRaisesRegex(RuntimeError, "retained", rt.close)

            # Mismatch: dsa_cp=True with dsa_cp_context=None
            src_m1, pl_m1 = nsrc(), mk_pl()
            rt_m1, _ = mk_rt(pl_m1, src_m1, dsa_cp=True)
            step_m1 = mk_st([lms.SelectedRequest("r0", 49152, 4, True, 49152)])
            self.assertRaisesRegex(
                ValueError,
                "DSA-CP metadata mismatch",
                rt_m1.bind_step,
                step_m1,
                ("r0",),
                (49152,),
                (0, 4),
                mk_metas(4, None),
            )
            rt_m1.fail(step_m1)
            self.assertRaisesRegex(RuntimeError, "retained", rt_m1.close)

            # Mismatch: dsa_cp=False with dsa_cp_context != None
            src_m2, pl_m2 = nsrc(), mk_pl()
            rt_m2, _ = mk_rt(pl_m2, src_m2, dsa_cp=False)
            step_m2 = mk_st([lms.SelectedRequest("r0", 49152, 4, True, 49152)])
            cp_ok = SimpleNamespace(local_start=0, local_end=4, local_end_with_pad=4, num_tokens_pad=4)
            self.assertRaisesRegex(
                ValueError,
                "DSA-CP metadata mismatch",
                rt_m2.bind_step,
                step_m2,
                ("r0",),
                (49152,),
                (0, 4),
                mk_metas(4, cp_ok),
            )
            rt_m2.fail(step_m2)
            self.assertRaisesRegex(RuntimeError, "retained", rt_m2.close)

            # Inconsistent dsa_cp_context across layers
            src_m3, pl_m3 = nsrc(), mk_pl()
            rt_m3, _ = mk_rt(pl_m3, src_m3, dsa_cp=True)
            step_m3 = mk_st([lms.SelectedRequest("r0", 49152, 8, True, 49152)])
            bad_metas = {
                "l0": SimpleNamespace(
                    block_table=torch.zeros((8, 1200), dtype=torch.int32),
                    num_actual_tokens=8,
                    block_size=128,
                    dsa_cp_context=SimpleNamespace(local_start=0, local_end=2, local_end_with_pad=2, num_tokens_pad=8),
                ),
                "l1": SimpleNamespace(
                    block_table=torch.zeros((8, 1200), dtype=torch.int32),
                    num_actual_tokens=8,
                    block_size=128,
                    dsa_cp_context=SimpleNamespace(local_start=2, local_end=4, local_end_with_pad=4, num_tokens_pad=8),
                ),
            }
            self.assertRaisesRegex(
                ValueError,
                "Inconsistent dsa_cp_context across layers",
                rt_m3.bind_step,
                step_m3,
                ("r0",),
                (49152,),
                (0, 8),
                bad_metas,
            )
            rt_m3.fail(step_m3)
            self.assertRaisesRegex(RuntimeError, "retained", rt_m3.close)

        # 6. Straddling rank boundaries, oversized TopK guards, [B2] topology gates, initialize_selected_kv, draft empty
        with _fake_npu():
            # 6a. Multi-request straddling rank boundary (q_off0 > 0 with len(active_indices) == 2)
            src_s1, pl_s1 = nsrc(), mk_pl()
            rt_s1, caches_s1 = mk_rt(pl_s1, src_s1, dsa_cp=True)
            src_s1.step = step_s1 = mk_st(
                [
                    lms.SelectedRequest("r0", 49152, 300, True, 49152),
                    lms.SelectedRequest("r1", 131072, 212, True, 131072),
                ]
            )
            cp_s1 = SimpleNamespace(local_start=256, local_end=512, local_end_with_pad=512, num_tokens_pad=1024)
            metas_s1 = mk_metas(512, cp_s1)
            rt_s1.bind_step(step_s1, ("r0", "r1"), (49152, 131072), (0, 300, 512), metas_s1)
            self.assertEqual(rt_s1._active_selected_indices, (0, 1))
            self.assertEqual(rt_s1._query_offsets, [256, 0])
            self.assertEqual(rt_s1.request_metadata.tolist(), [[0, 49152, 0, 44], [1, 131072, 44, 212]])
            topk_s1 = torch.full((256, 1, skv.TOPK), -1, dtype=torch.int32)
            topk_s1[0, 0, :4] = torch.tensor([49151, 49152, 49451, 49452], dtype=torch.int32)
            topk_s1[44, 0, :3] = torch.tensor([49152, 49451, 131071], dtype=torch.int32)
            rt_s1.load("l0", topk_s1, caches_s1["l0"], metas_s1["l0"])
            rt_s1.prefetch_next("l0", metas_s1["l0"])
            rt_s1.load("l1", topk_s1, caches_s1["l1"], metas_s1["l1"])
            rt_s1.finish_step()
            rt_s1._retire_previous()
            self.assertEqual(pl_s1.built[0][0, 0, :4].tolist(), [49151, -1, -1, 49452])
            self.assertEqual(pl_s1.built[0][44, 0, :3].tolist(), [49152, 49451, 131071])
            rt_s1.close()

            # 6b. Non-selected request straddling local_start before a selected request
            src_s2, pl_s2 = nsrc(), mk_pl()
            rt_s2, caches_s2 = mk_rt(pl_s2, src_s2, dsa_cp=True)
            src_s2.step = step_s2 = mk_st(
                [
                    lms.SelectedRequest("r0", 128, 300, False, 0),
                    lms.SelectedRequest("r1", 49152, 212, True, 49152),
                ]
            )
            rt_s2.bind_step(step_s2, ("r0", "r1"), (128, 49152), (0, 300, 512), metas_s1)
            self.assertEqual(rt_s2._active_selected_indices, (1,))
            self.assertEqual((rt_s2.starts, rt_s2.queries, rt_s2._query_offsets), ([44], [212], [0]))
            self.assertEqual(rt_s2.request_metadata.tolist(), [[1, 49152, 44, 212]])
            rt_s2.fail(step_s2)
            self.assertRaisesRegex(RuntimeError, "retained", rt_s2.close)

            # 6c. Oversized TopK fail-closed checks (dsa_cp=True and dsa_cp=False)
            src_o1, pl_o1 = nsrc(), mk_pl()
            rt_o1, caches_o1 = mk_rt(pl_o1, src_o1, dsa_cp=True)
            step_o1 = mk_st([lms.SelectedRequest("r0", 49152, 512, True, 49152)])
            cp_o1 = SimpleNamespace(local_start=0, local_end=128, local_end_with_pad=128, num_tokens_pad=512)
            metas_o1 = mk_metas(512, cp_o1)
            rt_o1.bind_step(step_o1, ("r0",), (49152,), (0, 512), metas_o1)
            self.assertRaisesRegex(
                ValueError,
                "Invalid selected TopK",
                rt_o1.load,
                "l0",
                torch.zeros((256, 1, skv.TOPK), dtype=torch.int32),
                caches_o1["l0"],
                metas_o1["l0"],
            )
            rt_o1.fail(step_o1)
            self.assertRaisesRegex(RuntimeError, "retained", rt_o1.close)

            src_o2, pl_o2 = nsrc(), mk_pl()
            rt_o2, caches_o2 = mk_rt(pl_o2, src_o2, dsa_cp=False, max_tokens=16)
            step_o2 = mk_st([lms.SelectedRequest("r0", 49152, 4, True, 49152)])
            metas_o2 = mk_metas(4, None)
            rt_o2.bind_step(step_o2, ("r0",), (49152,), (0, 4), metas_o2)
            self.assertRaisesRegex(
                ValueError,
                "Invalid selected TopK",
                rt_o2.load,
                "l0",
                torch.zeros((17, 1, skv.TOPK), dtype=torch.int32),
                caches_o2["l0"],
                metas_o2["l0"],
            )
            rt_o2.fail(step_o2)
            self.assertRaisesRegex(RuntimeError, "retained", rt_o2.close)

            # 6d. Target-non-empty -> draft-empty rank transition (Rank 2: active target -> empty draft)
            src_d, pl_d = nsrc(), mk_pl()
            rt_d, caches_d = mk_rt(pl_d, src_d, dsa_cp=True, draft_layers=("l2",))
            src_d.step = step_d = mk_st(
                [
                    lms.SelectedRequest("r0", 49152, 2, True, 49152),
                    lms.SelectedRequest("r1", 49152, 2, True, 49152),
                ],
                ("l0", "l1", "l2"),
            )
            cp_tgt = SimpleNamespace(local_start=2, local_end=3, local_end_with_pad=3, num_tokens_pad=4)
            metas_tgt = mk_metas(4, cp_tgt, ("l0", "l1", "l2"))
            rt_d.bind_step(step_d, ("r0", "r1"), (49152, 49152), (0, 2, 4), metas_tgt)
            self.assertEqual(rt_d._active_selected_indices, (1,))
            topk_tgt = torch.zeros((1, 1, skv.TOPK), dtype=torch.int32)
            rt_d.load("l0", topk_tgt, caches_d["l0"], metas_tgt["l0"])
            rt_d.prefetch_next("l0", metas_tgt["l0"])
            rt_d.load("l1", topk_tgt, caches_d["l1"], metas_tgt["l1"])
            rt_d.defer_to_draft_step()

            cp_dr = SimpleNamespace(local_start=2, local_end=2, local_end_with_pad=3, num_tokens_pad=4)
            metas_dr = mk_metas(2, cp_dr, ("l2",))
            common_dr = SimpleNamespace(query_start_loc_cpu=[0, 1, 2], seq_lens_cpu=[49153, 49153])
            rt_d.bind_draft_step(metas_dr, common_dr)
            self.assertEqual((rt_d._active_selected_indices, rt_d.request_metadata), ((), None))
            rt_d.load("l2", torch.zeros((1, 1, skv.TOPK), dtype=torch.int32), caches_d["l2"], metas_dr["l2"])
            rt_d.finish_step()
            rt_d._retire_previous()
            self.assertTrue(step_d._closed)
            rt_d.close()

        # 6e. [B2] gates + end-to-end initialize_selected_kv dsa_cp=True wiring
        vcfg_bad_draft_tp = _vcfg(
            par=SimpleNamespace(
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
            ),
            spec=SimpleNamespace(method="deepseek_mtp", num_speculative_tokens=1, draft_tensor_parallel_size=1),
        )
        with patch.multiple(envs_ascend, **kw):
            self.assertRaisesRegex(ValueError, "Unsupported deployment", cfg.validate_deployment, vcfg_bad_draft_tp)

        for bad_ctx in (
            {
                "model.layers.0.self_attn": SimpleNamespace(
                    impl=SimpleNamespace(
                        kv_lora_rank=512,
                        qk_rope_head_dim=64,
                        index_topk=2048,
                        enable_dsa_cp=True,
                        is_vl_first_layer=True,
                    )
                )
            },
            {
                "model.layers.0.self_attn": SimpleNamespace(
                    impl=SimpleNamespace(
                        kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048, enable_dsa_cp=True, tp_size=4
                    )
                ),
                "model.layers.1.self_attn": SimpleNamespace(
                    impl=SimpleNamespace(
                        kv_lora_rank=512, qk_rope_head_dim=64, skip_topk=True, enable_dsa_cp=True, tp_size=1
                    )
                ),
            },
        ):
            self.assertRaises(
                ValueError,
                skv._resolve_topology,
                SimpleNamespace(
                    vllm_config=vcfg_tp4, compilation_config=SimpleNamespace(static_forward_context=bad_ctx)
                ),
            )

        ctx_tp2 = {
            "model.layers.0.self_attn": SimpleNamespace(
                impl=SimpleNamespace(
                    kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048, enable_dsa_cp=True, tp_size=2
                )
            )
        }
        self.assertRaisesRegex(
            ValueError,
            "tp_size mismatch",
            skv._resolve_topology,
            SimpleNamespace(vllm_config=vcfg_tp4, compilation_config=SimpleNamespace(static_forward_context=ctx_tp2)),
        )

        # End-to-end initialize_selected_kv with dsa_cp=True, tp_size=4
        ctx_init = {
            "model.layers.0.self_attn": SimpleNamespace(
                impl=SimpleNamespace(
                    kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048, enable_dsa_cp=True, tp_size=4
                )
            ),
            "model.layers.1.self_attn": SimpleNamespace(
                impl=SimpleNamespace(
                    kv_lora_rank=512, qk_rope_head_dim=64, skip_topk=True, enable_dsa_cp=True, tp_size=4
                )
            ),
        }
        vcfg_init = _vcfg(
            par=SimpleNamespace(
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
            ),
            sched=SimpleNamespace(async_scheduling=True, max_num_batched_tokens=8192, max_num_seqs=8),
            spec=None,
        )
        rn_init = SimpleNamespace(
            device="cpu",
            vllm_config=vcfg_init,
            ascend_config=SimpleNamespace(selected_kv_config=cfg),
            pcp_size=1,
            dcp_size=1,
            layerwise_prefill_p_node=True,
            layerwise_prefill_bank_count=2,
            compilation_config=SimpleNamespace(static_forward_context=ctx_init),
            selected_kv_runtime=None,
            _selected_kv_budget=None,
        )
        wk_init = SimpleNamespace(vllm_config=vcfg_init, model_runner=rn_init)
        with patch.multiple(envs_ascend, **kw), patch.dict(sys.modules, u_a3):
            skv.reserve_selected_kv_workspace(wk_init, 2 << 30)

        kv_caches_init = {
            "model.layers.0.self_attn": (
                torch.zeros((16, 128, 1, 512), dtype=torch.bfloat16),
                torch.zeros((16, 128, 1, 64), dtype=torch.bfloat16),
            ),
            "model.layers.1.self_attn": (
                torch.zeros((16, 128, 1, 512), dtype=torch.bfloat16),
                torch.zeros((16, 128, 1, 64), dtype=torch.bfloat16),
            ),
        }
        fake_impl = SimpleNamespace(selected_kv_source=None)
        fake_conn = SimpleNamespace(_lmcache_engine=fake_impl)
        fake_src = SimpleNamespace(close=nop)
        fake_planner = mk_pl()
        with (
            _fake_npu(),
            patch.object(sys.modules["vllm.distributed.kv_transfer"], "get_kv_transfer_group", lambda: fake_conn),
            patch.object(lms, "LMCacheSelectedKVSource", return_value=fake_src),
            patch.object(psr, "SelectedKVPlanner", return_value=fake_planner),
        ):
            skv.initialize_selected_kv(rn_init, None, kv_caches_init)
            rt_init = rn_init.selected_kv_runtime
            self.assertIsNotNone(rt_init)
            self.assertIs(fake_impl.selected_kv_source, fake_src)
            self.assertEqual((rt_init._dsa_cp, rt_init._cp_tp_size, rt_init.local_query_capacity), (True, 4, 2048))
            self.assertEqual(tuple(rt_init._plan_topk_buf.shape), (2048, 1, skv.TOPK))
            rt_init.close()

    def _nop(self, *_a: Any, **_k: Any) -> None:
        return None

    class _LayerSrc:
        def __init__(self, pages: Any, lid: int, suffix: Any = ()) -> None:
            self.pages, self.layer_id, self.suffix = pages, lid, suffix

    def _sh_obj(self, pb: int, toks: int, nl: int) -> Any:
        o = _mk_obj(pb, toks)
        o.layer_size, o.num_layers, o.valid = toks * skv.MAIN_ROW_BYTES, nl, True
        return o

    def _ar(self, lo: int, hi: int, *, d2: bool = True) -> np.ndarray:
        a = np.arange(lo, hi, dtype=np.int32)
        return a[None, :] if d2 else a

    def _cm(self, *specs: Any) -> Any:
        NS, reqs = SimpleNamespace, []
        for sp in specs:
            rid, t, ld, sd, vc, lp = (
                (sp + (49152, True, False, None, None)[len(sp) - 1 :])
                if isinstance(sp, tuple)
                else (sp, 49152, True, False, None, None)
            )
            ls = NS(can_load=ld, lmcache_cached_tokens=t, dsa_cold_compact_load=False)
            if vc is not None:
                ls.vllm_cached_tokens = vc
            rk = {"req_id": rid, "is_decode_window_save": False, "is_sparse_decode": sd}
            r = NS(block_allocation_mode="prefill_child", load_spec=ls, **rk)
            if lp is not None:
                r.is_last_prefill = lp
            reqs.append(r)
        return NS(requests=reqs)

    def _am(self, ls: tuple[str, ...], nt: int = 8, bs: int = 1) -> dict[str, Any]:
        bt = torch.zeros((bs, 420), dtype=torch.int32)
        return {n: SimpleNamespace(block_table=bt, num_actual_tokens=nt, block_size=128) for n in ls}

    def _dva_grid(self, dva: int, p_tok: int = 49152) -> torch.Tensor:
        rn, rp = skv.MLA_NOPE_ROW_BYTES, skv.MLA_PE_ROW_BYTES
        return torch.tensor(
            [[dva + r * rn, dva + p_tok * rn + r * rp] for r in range(0, 49152, 128)], dtype=torch.uint64
        )

    def _prep_cm(self, rn: Any, lmc_c: Any, cm: Any, am: dict[str, Any]) -> None:
        with skv.selected_kv_prepare_context(rn, SimpleNamespace(kv_connector_metadata=cm), am):
            lmc_c.bind_connector_metadata(cm)
            lmc_c.start_load_kv(None)

    def _fwd_pair(
        self,
        rn: Any,
        impl: Any,
        ls: tuple[str, ...],
        kvc: dict[str, Any],
        cm: Any,
        am: dict[str, Any],
        tk: torch.Tensor,
        ql: list[int],
        sl: list[int],
        *,
        pre_draft: Any = None,
    ) -> None:
        with skv.selected_kv_forward(rn, SimpleNamespace(kv_connector_metadata=cm), am, defer_finish=True) as b:
            impl.submit_layerwise_prefill_load(-1)
            b.load(ls[0], tk, kvc[ls[0]], am[ls[0]])
            b.prefetch_next(ls[0], am[ls[0]])
            impl.submit_layerwise_prefill_load(ls[0])
            b.load(ls[1], tk, kvc[ls[1]], am[ls[1]])
            impl.submit_layerwise_prefill_load(ls[1])
        qt = ql[-1]
        dm = {ls[2]: SimpleNamespace(block_table=am[ls[2]].block_table, num_actual_tokens=qt, block_size=128)}
        skv.bind_selected_kv_draft_metadata(rn, dm, SimpleNamespace(query_start_loc_cpu=ql, seq_lens_cpu=sl))
        if pre_draft is not None:
            pre_draft()
        with skv.selected_kv_draft_forward(rn):
            dm[ls[2]].selected_kv.load(ls[2], tk[:qt], kvc[ls[2]], dm[ls[2]])

    def _set_batch(self, rn: Any, tbox: list[Any], rids: list[str], ql: list[int], tbl: np.ndarray) -> None:
        rn.input_batch.req_ids, rn.input_batch.num_reqs = rids, len(rids)
        rn.input_batch.num_computed_tokens_cpu = np.array([49152] * len(rids), dtype=np.int32)
        rn.query_start_loc = SimpleNamespace(np=np.array(ql, dtype=np.int32))
        tbox[0] = tbl

    def _close_src(self, src: Any, *, restore: bool = True) -> None:
        try:
            if src.step is not None and not src.step._finished:
                src.step.finish(failed=True)
            src.close()
        finally:
            if restore:
                for fn in reversed(src._restorers):
                    fn()

    def _send(self, gc: Any, impl: Any, rid: str, obj: Any, n: int = 3, *, sub: bool = True) -> Any:
        g = gc.batched_to_gpu([0], [49152], req_id=rid, kv_group=0)
        next(g)
        for _ in range(n):
            g.send([obj])
        if sub:
            impl._submit_initial_layerwise_prefill_load()
        return g

    def _mk_src(
        self,
        nl: int = 4,
        *,
        banks: int = 2,
        mc: int = 1024,
        dva: Any = None,
        ei: dict[str, Any] | None = None,
        eg: dict[str, Any] | None = None,
        ee: dict[str, Any] | None = None,
        track_btg: bool = False,
        make_src: bool = True,
    ) -> Any:
        NS, nop, ls = SimpleNamespace, self._nop, tuple(f"l{i}" for i in range(nl))
        cbox, hbox, btg = [None], [nop], []

        def _btg(*_a: Any, **k: Any) -> Any:
            if track_btg:
                btg.append(str(k.get("req_id", "")))
            while True:
                yield

        gc_k = {"_layerwise_prefill_bank_count": banks, "_prefill_dma_bound_loads": {}, "batched_to_gpu": _btg}
        gc = NS(**(gc_k | {"release_layerwise_prefill_dma_cache": nop, "fence_debug": None} | (eg or {})))
        ek = {"gpu_connector": gc, "_prepare_shared_prefill_sources": nop, "release_shared_cpu_sparse_request": nop}
        eng = NS(**(ek | {"config": NS(chunk_size=1024)} | (ee or {})))
        par = NS(_get_connector_metadata=lambda: cbox[0], _kv_transfer_config=NS(kv_role="kv_producer"))
        ik = {"lmcache_engine": eng, "_layerwise_prefill_dma": True, "_layerwise_prefill_p_node": True, "_parent": par}
        ik.update({"enable_blending": False, "_deferred_layerwise_prefill_load_active": True})
        ik.update({"_initial_layerwise_prefill_load_submitted": True, "submit_layerwise_prefill_load": nop})
        ik["_submit_initial_layerwise_prefill_load"] = nop
        ik["_should_skip_group0_prefill_dma"] = nop
        im = NS(**(ik | {"start_load_kv": lambda c: hbox[0](c)} | (ei or {})))
        mc_m = {n: (torch.empty((mc, 1, 1, 1)),) * 2 for n in ls}
        npu_m = NS(bind_incremental_copy_addresses=nop, BoundCopyPrefix=nop)
        dva_fn = dva or (lambda p, _b: p + 0x100000)
        src = (
            lms.LMCacheSelectedKVSource(
                NS(device="cpu"), im, ls, mc_m, get_device_ptr_fn=dva_fn, npu_connectors_module=npu_m
            )
            if make_src
            else None
        )
        return NS(layers=ls, cm_box=cbox, hook=hbox, gc=gc, eng=eng, impl=im, src=src, btg=btg)

    def _pipe_setup(self, *, track_pending: bool = False) -> tuple[Any, ...]:
        NS, nop, cfg = SimpleNamespace, self._nop, skv.SelectedKVConfig.from_dict({"enabled": True})
        ev_cfg = {"done": True, "fail_on_sync": False}

        class _CtrlEv:
            def __init__(self) -> None:
                self.recorded_on: int | None = None
                self.done, self.fail_on_sync, self.sync_calls = bool(ev_cfg["done"]), bool(ev_cfg["fail_on_sync"]), 0

            def record(self, stream=None) -> None:
                self.recorded_on = int((stream or torch.npu.current_stream()).npu_stream)

            def query(self) -> bool:
                return self.done

            def synchronize(self) -> None:
                if self.fail_on_sync:
                    raise AssertionError("must not sync on handoff")
                self.sync_calls += 1
                self.done = True

        class _Fn:
            def __init__(self, cb=None) -> None:
                self._cb = cb

            def __call__(self, *a):
                return self._cb(*a) if self._cb else 0

        def _build() -> Any:
            ls = tuple(f"model.layers.{i}.self_attn" for i in range(3))
            sfc = {
                ls[i]: NS(
                    impl=NS(
                        kv_lora_rank=512,
                        qk_rope_head_dim=64,
                        **({"skip_topk": True} if i == 1 else {"index_topk": 2048}),
                    )
                )
                for i in range(3)
            }
            kvc = {
                n: (
                    torch.zeros((900, 128, 1, 512), dtype=torch.bfloat16),
                    torch.zeros((900, 128, 1, 64), dtype=torch.bfloat16),
                )
                for n in ls
            }
            rel: list[tuple[str, str]] = []
            ld, bt = {}, {}
            eg = {
                "release_layerwise_prefill_dma_cache": lambda r: rel.append(("dma", r)),
                "_layerwise_prefill_transfer_state": lambda: (None, {0: 0}, None, ld),
                "_layerwise_prefill_bank_tail_events": bt,
            }
            h = self._mk_src(
                eg=eg,
                ee={"release_shared_cpu_sparse_request": lambda r: rel.append(("cpu", r))},
                ei={"use_layerwise": True, "supports_layerwise_prefill_transfer_window": lambda: True},
                make_src=False,
            )
            lc = NS(
                _lmcache_engine=h.impl,
                _connector_metadata=None,
                _get_connector_metadata=lambda: lc._connector_metadata,
                bind_connector_metadata=lambda m: setattr(lc, "_connector_metadata", m),
                start_load_kv=lambda c: h.impl.start_load_kv(c),
            )
            h.impl._parent = lc
            tb = [self._ar(0, 420)]
            gbt = (NS(get_cpu_tensor=lambda: torch.from_numpy(tb[0])),)
            ib = NS(
                req_ids=["r0"],
                num_reqs=1,
                num_computed_tokens_cpu=np.array([49152], dtype=np.int32),
                layerwise_prefill_block_tables=(gbt, gbt),
            )
            rn = NS(device="cpu", vllm_config=_vcfg(), ascend_config=NS(selected_kv_config=cfg), enable_dsa_cp=False)
            rn.layerwise_prefill_p_node, rn.layerwise_prefill_bank_count = True, 2
            rn.compilation_config = NS(static_forward_context=sfc)
            rn.drafter, rn.selected_kv_runtime = NS(attn_layer_names=(ls[2],)), None
            rn.query_start_loc, rn.input_batch = NS(np=np.array([0, 8], dtype=np.int32)), ib
            wk = NS(vllm_config=rn.vllm_config, model_runner=rn, cache_config=NS())
            fl = NS(
                create_sparse_prefill_fixed_handle=_Fn(lambda p, c, hp: setattr(hp._obj, "value", 0xBEEF) or 0),
                destroy_sparse_prefill_fused_handle=_Fn(),
                launch_sparse_prefill_fused_plan=ctypes.c_void_p(0x1),
            )
            ml = NS(launch_sparse_prefill_bf16_mla_inplace=ctypes.c_void_p(0x2))

            def _bp(tk, sc, out, h_, s, q, *a):
                if track_pending:
                    rn.selected_kv_runtime.selection_planner.pending.append(torch.zeros((2,), dtype=torch.int32))
                out[1][: len(h_)].copy_(torch.tensor([[0, qv, qv, 0, 0, 0, 0, 0] for qv in q], dtype=torch.int32))

            fops = NS(load_library=nop, dsa_prefill_queue=NS(batch_plan=_bp, mla_inplace_copy=nop))
            mods = {
                "vllm.distributed.kv_transfer": NS(get_kv_transfer_group=lambda: lc),
                "vllm_ascend.utils": NS(
                    AscendDeviceType=NS(A2=2, A3=1, _310P=0, A5=3), get_ascend_device_type=lambda: 1
                ),
                "lmcache_ascend": types.ModuleType("lmcache_ascend"),
                "lmcache_ascend.v1": types.ModuleType("lmcache_ascend.v1"),
                "lmcache_ascend.v1.npu_connector": types.ModuleType("lmcache_ascend.v1.npu_connector"),
                "lmcache_ascend.v1.npu_connector.npu_connectors": NS(
                    bind_incremental_copy_addresses=nop, BoundCopyPrefix=nop
                ),
                "lmcache_ascend.c_ops": NS(get_device_ptr=lambda p, _: p + 0x100000),
            }
            d = {"layers": ls, "kv_c": kvc, "rel": rel, "gc": h.gc, "eng": h.eng, "impl": h.impl, "lmc_c": lc}
            d.update(tbl_box=tb, rn=rn, wk=wk, hook=h.hook, fl=fl, ml=ml, fops=fops, mods=mods)
            return NS(**d)

        def _run_step(
            e: Any,
            rid: str,
            *,
            bb: int = 0,
            hold: bool = True,
            fos: bool = False,
            qd: int = 1,
            sm: Any = None,
            rel: bool = False,
            wio: Any = None,
            rio: Any = None,
        ) -> Any:
            rt, obj = e.rn.selected_kv_runtime, _mk_obj(0x9000 + bb, 49152)
            self._set_batch(e.rn, e.tbl_box, [rid], [0, 8], self._ar(bb, bb + 390))
            cm = self._cm(rid)
            e.hook[0] = lambda _: (
                wio is not None and wio.append(list(rt.source._metadata_stream.waited)),
                rio is not None and rio.append(len(e.rel)),
                self._send(e.gc, e.impl, rid, obj, 3),
            )
            am, tk = self._am(e.layers), torch.zeros((8, 1, 2048), dtype=torch.int32)
            self._prep_cm(e.rn, e.lmc_c, cm, am)
            with torch.npu.stream(sm) if sm is not None else nullcontext():
                st = rt.source.step
                pre_d = lambda: ev_cfg.update(done=not hold, fail_on_sync=fos)
                self._fwd_pair(e.rn, e.impl, e.layers, e.kv_c, cm, am, tk, [0, qd], [49152 + qd], pre_draft=pre_d)
            ev_cfg["done"], ev_cfg["fail_on_sync"] = True, False
            if rel:
                e.eng.release_shared_cpu_sparse_request(rid)
                e.gc.release_layerwise_prefill_dma_cache(rid)
            return st, st._completion, obj

        return cfg, _CtrlEv, _build, _run_step

    @contextmanager
    def _init_rt(self, e: Any, avail: int = 2 << 30) -> Any:
        ekw = {"VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": True, "VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS": 2}
        with (
            patch.multiple(envs_ascend, **ekw),
            patch.dict(sys.modules, e.mods),
            patch.object(ctypes, "CDLL", side_effect=[e.fl, e.ml]),
            patch.object(torch, "ops", e.fops),
        ):
            nret = skv.reserve_selected_kv_workspace(e.wk, avail)
            skv.initialize_selected_kv(e.rn, None, e.kv_c)
            yield nret

    def _assert_retained(self, src: Any, impl: Any, cm: Any, rid: str, tb: np.ndarray) -> None:
        self.assertRaisesRegex(RuntimeError, "retained", src.begin_runner_step, cm, (rid,), (49152,), (0, 8), (tb, tb))
        self.assertRaisesRegex(RuntimeError, "retained", impl.start_load_kv, None)

    def _assert_fail_closed_pinned(self, e: Any, st0: Any, ev0: Any, o0: Any) -> None:
        ev0.done = True
        self.assertFalse(st0.try_retire())
        rt, src = e.rn.selected_kv_runtime, e.rn.selected_kv_runtime.source
        self.assertTrue(src._failed and rt._failed)
        self.assertEqual((o0.s, e.rel), ({"p": 1, "r": 1}, []))

    def _attach_skip_hook(self, h: Any, o_sh: Any, toks: int, s_step: Any = None, s_skip: Any = None) -> None:
        def _orig(_: Any) -> None:
            if s_step is not None:
                s_step.append(h.src.step)
            if s_skip is not None:
                s_skip.append(h.impl._selected_skip_group0_dma_req_ids)
            for rm in h.cm_box[0].requests:
                if rm.req_id in h.impl._selected_skip_group0_dma_req_ids:
                    kw: dict[str, Any] = {
                        "req_id": rm.req_id,
                        "_shared_prefill_starts": [0],
                        "_shared_prefill_ends": [toks],
                    }
                else:
                    next(h.gc.batched_to_gpu([0], [toks], req_id=rm.req_id, kv_group=0))
                    kw = {"req_id": rm.req_id}
                h.eng._prepare_shared_prefill_sources([self._LayerSrc((o_sh,), i) for i in range(4)], 0, kw)

        h.hook[0] = _orig

    def test_vectorized_shared_layer_page_source(self) -> None:
        def _homo(pb: int, toks: int, nl: int, raw_sz=None):
            o = _mk_obj(pb, toks)
            o.num_layers, o.valid, o.ls_calls = nl, True, []
            o.layer_size = toks * skv.MAIN_ROW_BYTES if raw_sz is None else raw_sz
            o._tokens_bytes = lambda: o.layer_size if o.layer_size > 0 else 4096 * skv.MAIN_ROW_BYTES
            o.layer_size_bytes = lambda lid: (o.ls_calls.append(lid), o._tokens_bytes())[1]
            return o

        def _check_vec(nb: int) -> None:
            nl, ptr_calls = 6, []
            p0, p1, p2 = _homo(0x1000, 16384, nl), _homo(0x2000, 16384, nl), _homo(0x3000, 16384, nl)
            pages0 = (p0, p1, p2)

            def _get_dva(hp: int, bs: int) -> int:
                ptr_calls.append((hp, bs))
                return (1 << 63) | (hp + 0x100000)

            def _orig_clean(_p, _g, kw_a: dict[str, Any], *_a, **_k) -> None:
                if kw_a.get("_reuse_shared_dense_prefix"):
                    kw_a["cached_chunk_dev_ptrs"].extend(
                        [(1 << 63) | (o.layer_data_ptr(lid) + 0x100000) for o in pages0] for lid in range(nl)
                    )

            h = self._mk_src(nl, banks=nb, mc=900, dva=_get_dva, ee={"_prepare_shared_prefill_sources": _orig_clean})
            cm = self._cm("r0")
            h.cm_box[0] = cm
            tbl0, tbl1 = self._ar(0, 385), self._ar(127, 512)
            tbls = tuple(tbl0 if b % 2 == 0 else tbl1 for b in range(nb))

            def _run(prep, reuse: bool, cdp=None, clr=()):
                for o in clr:
                    o.ls_calls.clear()
                ptr_calls.clear()
                kw: dict[str, Any] = {
                    "req_id": "r0",
                    "cached_chunk_dev_ptrs": cdp if cdp is not None else [],
                    "_reuse_shared_dense_prefix": reuse,
                }
                h.hook[0] = lambda _: (
                    next(h.gc.batched_to_gpu([0, 16384, 32768], [16384, 32768, 49152], req_id="r0", kv_group=0)),
                    h.eng._prepare_shared_prefill_sources(prep, 0, kw),
                )
                h.src.begin_runner_step(cm, ("r0",), (49152,), (0, 16), tbls)
                h.impl.start_load_kv(None)
                return h.src.step

            def _done_eq(st, refs):
                for lid, n in enumerate(h.layers):
                    self.assertTrue(torch.equal(st.page_bases(n).view(torch.uint64), refs[lid]))
                st.finish(completion=torch.npu.Event())
                h.src._prune_retired_steps()

            st_v = _run([self._LayerSrc(pages0, i) for i in range(nl)], True, clr=pages0)
            self.assertIsNotNone(st_v)
            ab = min(nb, nl)
            self.assertEqual((p0.ls_calls, p1.ls_calls, p2.ls_calls), ([0] * (ab + 1), [0] * ab, [0] * ab))
            self.assertEqual((len(ptr_calls), p0.s), (1, {"p": 1, "r": 1}))
            v_bases = [st_v.page_bases(n).view(torch.uint64).clone() for n in h.layers]
            self.assertGreaterEqual(int(v_bases[0][0, 0].item()), 1 << 63)
            _done_eq(st_v, v_bases)
            self.assertEqual(p0.s, {"p": 0, "r": 0})
            _done_eq(_run([tuple(pages0) for _ in range(nl)], True), v_bases)
            p_mm = _homo(0x1000, 16384, nl, raw_sz=123456)
            p_mm._tokens_bytes = lambda: 16384 * skv.MAIN_ROW_BYTES
            h.src._dva_cache.clear()
            st_m = _run(
                [self._LayerSrc((p_mm, p1, p2), i) for i in range(nl)], False, cdp=[[0xDEAD] * 3 for _ in range(nl)]
            )
            self.assertEqual(len(ptr_calls), nl * 3)
            _done_eq(st_m, v_bases)
            ex = _homo(0x3000, 16384, nl)
            st_s = _run([self._LayerSrc((p0, p1), i, suffix=(ex,)) for i in range(nl)], False)
            self.assertEqual(ex.s, {"p": 1, "r": 1})
            _done_eq(st_s, v_bases)
            self.assertEqual(ex.s, {"p": 0, "r": 0})
            cm_sh = self._cm("r_sh", ("r_ns", 0, False), "r_priv")
            h.cm_box[0] = cm_sh
            t_base = self._ar(0, 385, d2=False)
            tbl_sh = np.vstack([t_base, t_base, self._ar(385, 768, d2=False).tolist() + [766, 767]])
            pv = [self._LayerSrc(pages0, i) for i in range(nl)]

            def _sh_hook(_):
                for r in ("r_sh", "r_priv"):
                    next(h.gc.batched_to_gpu([0, 16384, 32768], [16384, 32768, 49152], req_id=r, kv_group=0))
                    kw: dict[str, Any] = {"req_id": r, "cached_chunk_dev_ptrs": [], "_reuse_shared_dense_prefix": True}
                    h.eng._prepare_shared_prefill_sources(pv, 0, kw)

            h.hook[0] = _sh_hook
            h.src.begin_runner_step(cm_sh, ("r_sh", "r_ns", "r_priv"), (49152,) * 3, (0, 8, 16, 24), (tbl_sh,) * nb)
            h.impl.start_load_kv(None)
            st_sh = h.src.step
            self.assertTrue(all(("r_sh", b) not in st_sh._req_bank for b in range(nb)))
            self.assertIn("r_sh", st_sh._shared_reqs)
            self.assertTrue(all(("r_sh", lid) in st_sh._populated_req_layers for lid in range(nl)))
            self.assertTrue(torch.all(st_sh.page_bases("l0")[:385] == 0))
            st_sh.finish(completion=torch.npu.Event())
            h.src.close()

        with _fake_npu():
            _check_vec(2)
            _check_vec(8)

    def test_selective_zeroing_split_upload_and_coalesced_nonshared(self) -> None:
        with _fake_npu():
            o0 = _mk_obj(0x9000, 49152)
            h = self._mk_src(4, mc=600)
            h.cm_box[0] = self._cm("r0")
            h.hook[0] = lambda _: self._send(h.gc, h.impl, "r0", o0, 4, sub=False)
            bref = np.array(
                [self._dva_grid(h.src._chunk_info(o0, lid)[1]).numpy() for lid in range(4)], dtype=np.uint64
            )
            for idx, mx in enumerate((512, 384, 516, 64, 64)):
                b0 = (
                    self._ar(mx - 384, mx + 1, d2=False)
                    if mx >= 384
                    else np.array([*range(mx), *([mx] * (385 - mx))], dtype=np.int32)
                )
                on = 384 if mx >= 384 else mx
                h.src.begin_runner_step(h.cm_box[0], ("r0",), (49152,), (0, 16), (b0[None, :], b0.copy()[None, :]))
                h.impl.start_load_kv(None)
                st = h.src.step
                slot = st._host_slot
                self.assertEqual(slot, idx % 2)
                sd = (((mx + 1) + 3) // 4) * 4
                exp = torch.zeros((4, sd, 2), dtype=torch.int64)
                u64 = exp.numpy().view(np.uint64)
                for lid in range(4):
                    u64[lid, b0[:on]] = bref[lid, :on]
                self.assertTrue(torch.equal(h.src._step_host, exp))
                self.assertTrue(bool(h.src._host_bases_ring[slot].view(-1)[4 * sd * 2 :].eq(0).all()))
                st.finish(completion=torch.npu.Event())
                h.src._prune_retired_steps()
            h.src.close()
            p_src = (self._sh_obj(0xA000, 49152, 6),)
            h6 = self._mk_src(6, mc=500)
            h6.cm_box[0] = self._cm("r0")
            s6 = [self._LayerSrc(p_src, i) for i in range(6)]
            h6.hook[0] = lambda _: (
                next(h6.gc.batched_to_gpu([0], [49152], req_id="r0", kv_group=0)),
                h6.eng._prepare_shared_prefill_sources(s6, 0, {"req_id": "r0"}),
            )
            t256 = self._ar(0, 385)
            h6.src.begin_runner_step(h6.cm_box[0], ("r0",), (49152,), (0, 16), (t256, t256))
            h6.impl.start_load_kv(None)
            st6 = h6.src.step
            self.assertIsNotNone(st6._init_ready_event)
            self.assertIsNotNone(st6._shared_ready_event)
            self.assertIsNot(st6._init_ready_event, st6._shared_ready_event)
            self.assertIs(st6.ready("l0"), st6._init_ready_event)
            self.assertIs(st6.ready("l1"), st6._init_ready_event)
            for i in range(2, 6):
                self.assertIs(st6.ready(f"l{i}"), st6._shared_ready_event)
            st6.finish(completion=torch.npu.Event())
            h6.src.close()
            rids = ("req0", "req1", "req2")
            gsrc = self._mk_src(4, mc=1300)
            gsrc.cm_box[0] = self._cm(*rids)
            up, orec = [], gsrc.src._record_upload
            gsrc.src._record_upload = lambda st, lid=None: (up.append(lid), orec(st, lid))[1]

            def _st3(_):
                gs = [self._send(gsrc.gc, gsrc.impl, r, o0, 0, sub=False) for r in rids]
                for _ in range(2):
                    for g in gs:
                        g.send([o0])
                gsrc.impl._gens = gs

            gsrc.hook[0] = _st3
            tbl3 = np.vstack([self._ar(i * 400, i * 400 + 385, d2=False) for i in range(3)])
            gsrc.src.begin_runner_step(gsrc.cm_box[0], rids, (49152,) * 3, (0, 8, 16, 24), (tbl3, tbl3))
            gsrc.impl.start_load_kv(None)
            st3 = gsrc.src.step
            self.assertEqual(up, [0, 1])
            gsrc.impl._gens[0].send([o0])
            gsrc.impl._gens[1].send([o0])
            self.assertEqual(up, [0, 1])
            self.assertRaisesRegex(RuntimeError, "DDR page_bases not uploaded", st3.page_bases, "l2")
            gsrc.impl._gens[2].send([o0])
            self.assertEqual((up, tuple(st3.page_bases("l2").shape)), ([0, 1, 2], (1185, 2)))
            self.assertIsNotNone(st3.ready("l2"))
            for g in gsrc.impl._gens:
                g.send([o0])
            self.assertEqual(up, [0, 1, 2, 3])
            st3.finish(completion=torch.npu.Event())
            gsrc.src.close()

    def test_pipelined_handoff_overlap_drain_and_interleaved_native(self) -> None:
        _cfg, _CtrlEv, _build_env, _run_step = self._pipe_setup(track_pending=True)
        with _fake_npu(), patch.object(torch.npu, "Event", _CtrlEv):
            e, wio, rio = _build_env(), [], []
            with self._init_rt(e):
                rt, src = e.rn.selected_kv_runtime, e.rn.selected_kv_runtime.source
                st0, ev0, o0 = _run_step(e, "r0", bb=0, fos=True, wio=wio, rio=rio)
                self.assertEqual((o0.s, st0._host_slot, rt._cur_meta_slot), ({"p": 1, "r": 1}, 0, 0))
                am0 = self._am(e.layers)
                with skv.selected_kv_prepare_context(e.rn, SimpleNamespace(kv_connector_metadata=None), am0):
                    self.assertIsNone(src.step)
                with skv.selected_kv_forward(e.rn, SimpleNamespace(kv_connector_metadata=None), am0) as bfe:
                    self.assertIsNone(bfe)
                self.assertIsNone(src.step)
                self.assertEqual(ev0.sync_calls, 0)
                self.assertTrue(any(st is st0 for st, _, _ in rt._inflight_steps))
                alt_s = torch.npu.Stream()
                st1, ev1, o1 = _run_step(e, "r1", bb=10, fos=True, qd=8, sm=alt_s, wio=wio, rio=rio)
                self.assertEqual(ev0.sync_calls, 0)
                self.assertIn(ev0, alt_s.waited)
                self.assertIn(ev0, wio[-1])
                self.assertEqual((o0.s, st1._host_slot, rt._cur_meta_slot), ({"p": 1, "r": 1}, 1, 1))
                self.assertEqual((len(src._unretired_steps), len(rt._inflight_steps)), (2, 1))
                self.assertIs(rt._inflight_steps[0][0], st0)
                saved_s0_pending = rt._inflight_steps[0][2]
                self.assertGreaterEqual(len(saved_s0_pending), 2)
                ev0.fail_on_sync = False
                st2, ev2, o2 = _run_step(e, "req_d", bb=20, wio=wio, rio=rio)
                self.assertEqual((ev0.sync_calls, ev1.sync_calls, saved_s0_pending, st2._host_slot), (1, 0, [], 0))
                self.assertEqual((o0.s, o1.s, o2.s), ({"p": 0, "r": 0}, {"p": 1, "r": 1}, {"p": 1, "r": 1}))
                e.eng.release_shared_cpu_sparse_request("req_d")
                e.gc.release_layerwise_prefill_dma_cache("req_d")
                self.assertEqual(e.rel, [])
                ev1.done = True
                st3, ev3, o3 = _run_step(e, "req_d", bb=30, fos=True, wio=wio, rio=rio)
                self.assertEqual((ev2.sync_calls, rio[-1], o2.s), (1, 2, {"p": 0, "r": 0}))
                self.assertEqual(e.rel, [("cpu", "req_d"), ("dma", "req_d")])
                slot3 = rt._cur_meta_slot
                snap3 = rt._host_meta_ring[slot3].clone()
                am_nat = self._am(e.layers)
                e.hook[0] = self._nop
                self._prep_cm(e.rn, e.lmc_c, None, am_nat)
                with skv.selected_kv_forward(e.rn, SimpleNamespace(kv_connector_metadata=None), am_nat) as bf:
                    self.assertIsNone(bf)
                self.assertEqual(ev3.sync_calls, 0)
                self.assertTrue(any(st is st3 for st, _, _ in rt._inflight_steps))
                self.assertTrue(torch.equal(rt._host_meta_ring[slot3], snap3))
                _, ev4, o4 = _run_step(e, "r4", bb=40, wio=wio, rio=rio)
                self.assertEqual(ev3.sync_calls, 0)
                self.assertNotEqual(rt._cur_meta_slot, slot3)
                self.assertTrue(torch.equal(rt._host_meta_ring[slot3], snap3))
                ev3.fail_on_sync = False
                ev3.done = ev4.done = True
                skv.retire_selected_kv_if_idle(e.rn)
                skv.close_selected_kv_worker(e.wk)
                for obj in (o0, o1, o2, o3, o4):
                    self.assertEqual(obj.s, {"p": 0, "r": 0})
                self.assertTrue(rt._closed and src._closed)

    def _run_mixed_shared_nonshared(self, *, ns_first: bool) -> None:
        o_sh, o_ns = self._sh_obj(0x9000, 49152, 6), _mk_obj(0xD000, 49152)
        h = self._mk_src(6, mc=900)
        h.cm_box[0] = self._cm("r_sh", "r_ns")
        ns_gens, seen_init, up_lids, oup = [], [], [], h.src._record_upload
        h.src._record_upload = lambda st, lid=None: (up_lids.append(lid), oup(st, lid))[1]

        def _hook(_):
            if ns_first:
                ns_gens.append(self._send(h.gc, h.impl, "r_ns", o_ns, 2, sub=False))
                seen_init.append((h.src.step._initializing, list(up_lids)))
            next(h.gc.batched_to_gpu([0], [49152], req_id="r_sh", kv_group=0))
            h.eng._prepare_shared_prefill_sources([self._LayerSrc((o_sh,), i) for i in range(6)], 0, {"req_id": "r_sh"})
            if not ns_first:
                ns_gens.append(self._send(h.gc, h.impl, "r_ns", o_ns, 2, sub=False))

        h.hook[0] = _hook
        t6 = np.vstack([self._ar(0, 385, d2=False), self._ar(390, 775, d2=False)])
        h.src.begin_runner_step(h.cm_box[0], ("r_sh", "r_ns"), (49152, 49152), (0, 8, 16), (t6, t6))
        h.impl.start_load_kv(None)
        st6 = h.src.step
        if ns_first:
            self.assertEqual((seen_init, st6._initializing), ([(True, [])], False))
            self.assertIsNone(st6._shared_ready_event)
        self.assertEqual(up_lids, [0, 1])
        self.assertRaisesRegex(RuntimeError, "DDR page_bases not uploaded", st6.page_bases, "l2")
        for lid in range(2, 6):
            ns_gens[0].send([o_ns])
            if lid == 2:
                self.assertEqual(up_lids, [0, 1, 2])
        self.assertEqual(up_lids, [0, 1, 2, 3, 4, 5])
        for lid in range(6) if ns_first else (2,):
            pb = st6.page_bases(f"l{lid}")
            for off, oref in ((0, o_sh), (390, o_ns)):
                self.assertTrue(torch.equal(pb[off : off + 384], self._dva_grid(oref.layer_data_ptr(lid) + 0x100000)))
        st6.finish(completion=torch.npu.Event())
        h.src.close()

    def test_cross_step_drain_fail_closed_and_mixed_shared(self) -> None:
        cfg, _CtrlEv, _build_env, _run_step = self._pipe_setup()
        with patch.multiple(
            envs_ascend, VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=True, VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS=2
        ):
            for bad in ("VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD", "VLLM_ASCEND_DSA_USE_ADAPTER_CACHE"):
                with patch.object(envs_ascend, bad, True, create=True):
                    self.assertRaisesRegex(
                        ValueError,
                        r"Selected KV prefill cannot be combined with DSA (?:latent offload|adapter cache)",
                        cfg.validate_deployment,
                        _vcfg(),
                    )
        with _fake_npu(), patch.object(torch.npu, "Event", _CtrlEv):
            e = _build_env()
            with self._init_rt(e):
                rt, src = e.rn.selected_kv_runtime, e.rn.selected_kv_runtime.source
                for ck in ("sparse_decode", "non_sel_alone", "non_sel_with_extra"):
                    e.rel.clear()
                    s_t, ev_t, o_t = _run_step(e, "req_d", rel=True)
                    self.assertEqual((e.rel, o_t.s), ([], {"p": 1, "r": 1}))
                    seen: list[tuple[int, list[Any]]] = []
                    if ck == "non_sel_with_extra":
                        ox = _mk_obj(0xB000, 49152)
                        t2 = np.vstack([self._ar(0, 420, d2=False), self._ar(420, 805, d2=False).tolist() + [804] * 35])
                        self._set_batch(e.rn, e.tbl_box, ["req_d", "req_x"], [0, 4096, 4104], t2)
                        cm_n = self._cm("req_d", "req_x")
                        e.hook[0] = lambda _, s=seen, ev=ev_t, o=ox: (
                            s.append((ev.sync_calls, list(e.rel))),
                            self._send(e.gc, e.impl, "req_x", o, 3),
                        )
                        am_n = self._am(e.layers, nt=4104, bs=2)
                        self._prep_cm(e.rn, e.lmc_c, cm_n, am_n)
                        self.assertIsNotNone(src.step)
                        tk3 = torch.zeros((4104, 1, 2048), dtype=torch.int32)
                        self._fwd_pair(e.rn, e.impl, e.layers, e.kv_c, cm_n, am_n, tk3, [0, 1, 2], [53249, 49153])
                    else:
                        if ck == "non_sel_alone":
                            self._set_batch(e.rn, e.tbl_box, ["req_d"], [0, 4096], self._ar(0, 420))
                        cm_n = self._cm(("req_d", 49152, True, ck == "sparse_decode"))
                        e.hook[0] = lambda _, s=seen, ev=ev_t: s.append((ev.sync_calls, list(e.rel)))
                        self._prep_cm(e.rn, e.lmc_c, cm_n, self._am(e.layers, nt=8 if ck == "sparse_decode" else 4096))
                        self.assertIsNone(src.step)
                    self.assertEqual(
                        (seen, o_t.s, s_t._closed),
                        ([(1, [("cpu", "req_d"), ("dma", "req_d")])], {"p": 0, "r": 0}, True),
                    )
                e.rel.clear()
                st0, ev0, o0 = _run_step(e, "req_d")
                self.assertEqual(o0.s, {"p": 1, "r": 1})
                self._set_batch(e.rn, e.tbl_box, ["r1"], [0, 8], self._ar(10, 400))
                cm1, am1 = self._cm("r1"), self._am(e.layers)
                e.hook[0] = lambda _: (_ for _ in ()).throw(RuntimeError("boom_in_orig_start"))
                with self.assertRaisesRegex(RuntimeError, "boom_in_orig_start"):
                    self._prep_cm(e.rn, e.lmc_c, cm1, am1)
                self._assert_fail_closed_pinned(e, st0, ev0, o0)
                self._assert_retained(src, e.impl, self._cm(("r_no", 0, False)), "r_no", e.tbl_box[0])
                self.assertRaisesRegex(RuntimeError, "retains", rt.bind_step, None, (), (), (), am1)
                skv.close_selected_kv_worker(e.wk)
                self.assertEqual(o0.s, {"p": 1, "r": 1})
            e2 = _build_env()
            with self._init_rt(e2):
                rt2 = e2.rn.selected_kv_runtime
                rt2.source._failed = True
                self.assertFalse(rt2._failed)
                skv.abort_active_selected_kv_steps()
                self.assertTrue(rt2._failed)
                skv.close_selected_kv_worker(e2.wk)
        with _fake_npu():
            self._run_mixed_shared_nonshared(ns_first=False)

    def test_fail_closed_budget_and_heterogeneous(self) -> None:
        _cfg, _CtrlEv, _build_env, _run_step = self._pipe_setup()

        with _fake_npu(), patch.object(torch.npu, "Event", _CtrlEv):
            e = _build_env()
            with self._init_rt(e):
                rt, src = e.rn.selected_kv_runtime, e.rn.selected_kv_runtime.source
                st0, ev0, o0 = _run_step(e, "r0", rel=True)
                self.assertEqual((ev0.done, o0.s, e.rel), (False, {"p": 1, "r": 1}, []))
                o1 = _mk_obj(0xB000, 49152)
                self._set_batch(e.rn, e.tbl_box, ["r1"], [0, 8], self._ar(10, 400))
                cm1, am1 = self._cm("r1"), self._am(e.layers)
                e.hook[0] = lambda _: self._send(e.gc, e.impl, "r1", o1, 3)
                self._prep_cm(e.rn, e.lmc_c, cm1, am1)
                step1 = src.step
                self.assertIsNotNone(step1)
                self.assertIn(st0, src._unretired_steps)
                self.assertTrue(any(st is st0 for st, _, _ in rt._inflight_steps))
                self.assertEqual(e.rel, [])
                with (
                    self.assertRaisesRegex(RuntimeError, "fwd boom"),
                    skv.selected_kv_forward(e.rn, SimpleNamespace(kv_connector_metadata=cm1), am1),
                ):
                    raise RuntimeError("fwd boom")
                self._assert_fail_closed_pinned(e, st0, ev0, o0)
                self.assertTrue(step1._failed)
                self._assert_retained(src, e.impl, cm1, "r1", e.tbl_box[0])
                skv.close_selected_kv_worker(e.wk)
                self.assertEqual((o0.s, e.rel), ({"p": 1, "r": 1}, []))
            for fail_mode in ("none", "st0"):
                ef = _build_env()
                with self._init_rt(ef):
                    rt_f = ef.rn.selected_kv_runtime
                    st0_f, ev0_f, o0_f = _run_step(ef, "r0", rel=False)
                    self.assertTrue(st0_f._finished and not st0_f._closed and not ev0_f.done)
                    rt_f._handoff_previous()
                    rt_f.fail(None if fail_mode == "none" else st0_f)
                    ef.eng.release_shared_cpu_sparse_request("r0")
                    self._assert_fail_closed_pinned(ef, st0_f, ev0_f, o0_f)
                    self.assertFalse(st0_f._closed)
                    skv.close_selected_kv_worker(ef.wk)
                    self.assertEqual(o0_f.s, {"p": 1, "r": 1})
            h_max = 1048576
            exp_ws = (
                4 * (8192 * (4096 + 8) + h_max + 2 * 32 * 8 + (h_max // 32) + 8 * h_max + 8 * 8 + min(8, 16384) * 8)
                + 75 * 1024
            )
            self.assertEqual((exp_ws, psr.selected_workspace_bytes(16384, 8)), (172_439_040, exp_ws))
            ew, pb_b, d1 = _build_env(), skv.PAGE_ROWS * skv.MAIN_ROW_BYTES, 16385
            avail = skv.NATIVE_ALIGNMENT + d1 * pb_b + 1024
            with self._init_rt(ew, avail=avail) as nret:
                budget, eb = ew.rn._selected_kv_budget, skv.PAGE_BASES_ENTRY_BYTES * len(ew.layers)
                self.assertEqual((budget.maximum_blocks, budget._fixed), (d1, exp_ws + 3 * eb))
                self.assertEqual(avail - nret - skv.NATIVE_ALIGNMENT, exp_ws + (d1 + 3) * eb)
                budget.validate(nret // pb_b)
                rt_w, src_w = ew.rn.selected_kv_runtime, ew.rn.selected_kv_runtime.source
                self._set_batch(ew.rn, ew.tbl_box, ["r1"], [0, 8], self._ar(0, 390))
                cm_w, o_w = self._cm("r1"), _mk_obj(0xC000, 49152)
                ew.hook[0] = lambda _: self._send(ew.gc, ew.impl, "r1", o_w, 3)
                self._prep_cm(ew.rn, ew.lmc_c, cm_w, self._am(ew.layers))
                am_bad = self._am(ew.layers, bs=skv.MAX_REQUESTS + 1)
                self.assertRaisesRegex(
                    ValueError, "Bad block_table", rt_w.bind_step, src_w.step, ("r1",), (49152,), (0, 8), am_bad
                )
                rt_w.fail(src_w.step)
                skv.close_selected_kv_worker(ew.wk)
        with _fake_npu():
            oh = SimpleNamespace(layer_size=None, num_layers=6, valid=True, is_valid=lambda: True, is_pinned=False)
            oh.layer_size_bytes = lambda i: (49152 + i * 128) * skv.MAIN_ROW_BYTES
            oh.layer_data_ptr = lambda i: 0x9000_0000 + i * 0x10_0000
            ch = [
                [
                    (1 << 63) | (oh.layer_data_ptr(0) + 0x10_0000)
                    if i == 0
                    else (1 << 63) | (oh.layer_data_ptr(i) + 0x20_0000)
                ]
                for i in range(6)
            ]
            hh = self._mk_src(
                6,
                mc=500,
                dva=lambda p, _: (1 << 63) | (p + 0x10_0000),
                ee={"_prepare_shared_prefill_sources": lambda _p, _g, kw, *_a: kw["cached_chunk_dev_ptrs"].extend(ch)},
            )
            hh.cm_box[0] = self._cm("r_het")
            kwh: dict[str, Any] = {"req_id": "r_het", "cached_chunk_dev_ptrs": [], "_reuse_shared_dense_prefix": True}
            hh.hook[0] = lambda _: (
                kwh["cached_chunk_dev_ptrs"].clear(),
                next(hh.gc.batched_to_gpu([0], [49152], req_id="r_het", kv_group=0)),
                hh.eng._prepare_shared_prefill_sources([self._LayerSrc((oh,), i) for i in range(6)], 0, kwh),
            )
            th = self._ar(0, 385)
            hh.src.begin_runner_step(hh.cm_box[0], ("r_het",), (49152,), (0, 8), (th, th))
            hh.impl.start_load_kv(None)
            sth = hh.src.step
            for i in range(6):
                pb, dva = sth.page_bases(f"l{i}"), ch[i][0]
                self.assertEqual((pb.dtype, dva >> 63), (torch.uint64, 1))
                self.assertTrue(torch.equal(pb[:384], self._dva_grid(dva, p_tok=49152 + i * 128)))
            sth.finish(completion=torch.npu.Event())
            hh.src.close()
            self._run_mixed_shared_nonshared(ns_first=True)

    def test_post_orig_start_exception_aborts_retrieve_and_save_steps(self) -> None:
        with _fake_npu():
            ab: list[tuple[str, tuple[Any, ...]]] = []
            ei = {
                "_abort_layerwise_retrieve_step": lambda rq: ab.append(("ret", tuple(r.req_id for r in rq))),
                "_abort_save_step": lambda rq: ab.append(("save", tuple(r.req_id for r in rq))),
            }
            h = self._mk_src(
                4,
                eg={"batched_to_gpu": self._nop},
                ee={"_should_use_shared_layerwise_retrieve": lambda _g: False},
                ei=ei,
            )
            cm1 = self._cm(("r_post", 49152, True, False, 0))
            h.cm_box[0] = cm1
            t1 = self._ar(10, 395)
            h.src.begin_runner_step(cm1, ("r_post",), (49152,), (0, 8), (t1, t1))
            self.assertRaisesRegex(
                RuntimeError, r"Selected KV initial layers .* unpopulated", h.impl.start_load_kv, None
            )
            self.assertEqual(ab, [("ret", ("r_post",)), ("save", ("r_post",))])
            self.assertTrue(h.src._failed)
            for fn in reversed(h.src._restorers):
                fn()

    def test_skip_group0_dma_eligibility_handshake_and_shared_ranges(self) -> None:
        with _fake_npu():
            o_sh, s_skip = self._sh_obj(0x9000, 49152, 4), []
            h = self._mk_src(
                4,
                track_btg=True,
                eg={"prepare_layerwise_prefill_group0_skip": self._nop},
                ee={"_should_use_shared_layerwise_retrieve": lambda g: g == 0},
                ei={"_selected_skip_group0_dma_req_ids": frozenset()},
            )
            self._attach_skip_hook(h, o_sh, 49152, s_skip=s_skip)
            cm1 = self._cm(("r_skip", 49152, True, False, 0))
            h.cm_box[0] = cm1
            t1 = self._ar(10, 395)
            h.src.begin_runner_step(cm1, ("r_skip",), (49152,), (0, 8), (t1, t1))
            h.impl.start_load_kv(None)
            self.assertEqual(
                (s_skip[-1], h.btg, h.impl._selected_skip_group0_dma_req_ids), (frozenset({"r_skip"}), [], frozenset())
            )
            pbs = [h.src.step.page_bases(n).clone() for n in h.layers]
            h.src.step.finish(completion=torch.npu.Event())
            h.impl._layerwise_prefill_dma = False
            h.src.begin_runner_step(cm1, ("r_skip",), (49152,), (0, 8), (t1, t1))
            h.impl.start_load_kv(None)
            self.assertEqual((s_skip[-1], h.btg), (frozenset(), ["r_skip"]))
            for a, b in zip(pbs, [h.src.step.page_bases(n).clone() for n in h.layers]):
                self.assertTrue(torch.equal(a, b))
            h.src.step.finish(completion=torch.npu.Event())
            h.impl._layerwise_prefill_dma = True
            h.btg.clear()
            del h.gc.prepare_layerwise_prefill_group0_skip
            h.src.begin_runner_step(cm1, ("r_skip",), (49152,), (0, 8), (t1, t1))
            h.impl.start_load_kv(None)
            self.assertEqual((s_skip[-1], h.btg), (frozenset(), ["r_skip"]))
            h.src.step.finish(completion=torch.npu.Event())
            h.gc.prepare_layerwise_prefill_group0_skip = self._nop
            cm_off = self._cm(("r_off", 0, False, False, 0))
            h.cm_box[0] = cm_off
            h.src.begin_runner_step(cm_off, ("r_off",), (49152,), (0, 8), (t1, t1))
            self.assertIsNone(h.src.step)
            h.impl.start_load_kv(None)
            self.assertEqual(s_skip[-1], frozenset())
            self._close_src(h.src)

    def test_resident_row_with_stale_can_load_keeps_selected_prefill(self) -> None:
        history = 49152
        with _fake_npu():
            fixture = self._mk_src()
            metadata = self._cm("prefill", ("resident", history, True, True, 0))
            resident = metadata.requests[1]
            resident.load_spec.dsa_full_resident = True
            fixture.cm_box[0] = metadata
            page = self._sh_obj(0x9000, history, 4)

            def load_prefill(_context: Any) -> None:
                next(fixture.gc.batched_to_gpu([0], [history], req_id="prefill", kv_group=0))
                fixture.eng._prepare_shared_prefill_sources(
                    [self._LayerSrc((page,), layer) for layer in range(4)], 0, {"req_id": "prefill"}
                )

            fixture.hook[0] = load_prefill
            blocks = history // skv.PAGE_ROWS + 1
            tables = np.arange(2 * blocks, dtype=np.int32).reshape(2, blocks)
            try:
                # Even a prefill-shaped row must not load full-resident KV.
                for sparse_decode in (True, False):
                    resident.is_sparse_decode = sparse_decode
                    fixture.src.begin_runner_step(
                        metadata, ("prefill", "resident"), (history, history), (0, 8, 9), (tables, tables)
                    )
                    fixture.impl.start_load_kv(None)
                    self.assertIsNotNone(fixture.src.step)
                    self.assertEqual(fixture.src.step.selected_request_ids, {"prefill"})
                    fixture.src.step.finish(completion=torch.npu.Event())
            finally:
                self._close_src(fixture.src)

    def test_recompute_fallback_bs1_cancel_unused_and_next_step(self) -> None:
        with _fake_npu():
            pst, o_r0 = {"rc": 1}, self._sh_obj(0x9000, 49152, 4)
            o_pas = SimpleNamespace(
                layer_data_ptr=lambda lid: 0xA000 + lid * 0x100,
                layer_size_bytes=lambda _t: 49152 * skv.MAIN_ROW_BYTES,
                ref_count_down=lambda: pst.__setitem__("rc", pst["rc"] - 1),
            )
            mr: dict[str, tuple[int, int]] = {"req_a": (0, 49152)}
            h = self._mk_src(
                4,
                eg={
                    "batched_to_gpu": lambda *_a, **_k: iter([None]),
                    "prepare_layerwise_prefill_group0_skip": self._nop,
                },
                ee={"_should_use_shared_layerwise_retrieve": lambda g: g == 0, "_recompute_fallback_req_ids": set()},
            )
            tbl, cap = self._ar(0, 390), []

            def _orig_fb(_):
                sc = h.src.step
                cap.append(sc)
                assert sc is not None
                h.src._retain_objs(sc, [o_r0])
                sc._deferred_cbs.append(lambda: (mr.__setitem__("req_a", (0, 0)), o_pas.ref_count_down()))
                h.eng._recompute_fallback_req_ids.add("req_a")

            h.hook[0] = _orig_fb
            cm_fb = self._cm(("req_a", 49152, True, False, 0))
            h.cm_box[0] = cm_fb
            h.src.begin_runner_step(cm_fb, ("req_a",), (49152,), (0, 8), (tbl, tbl))
            h.impl.start_load_kv(None)
            self.assertEqual(len(cap), 1)
            sfb = cap[0]
            self.assertIsNotNone(sfb)
            self.assertFalse(h.src._failed)
            self.assertIsNone(h.src.step)
            self.assertTrue(sfb._finished and sfb._closed)
            self.assertEqual(
                (h.src._unretired_steps, o_r0.s, pst["rc"], mr["req_a"]), ([], {"p": 0, "r": 0}, 0, (0, 0))
            )
            self.assertEqual(
                (h.eng._recompute_fallback_req_ids, h.impl._selected_skip_group0_dma_req_ids), (set(), frozenset())
            )
            ok_box: list[Any] = []
            cm_ok = self._cm(("req_ok", 49152, True, False, 0))
            h.cm_box[0] = cm_ok
            kw_ok: dict[str, Any] = {
                "req_id": "req_ok",
                "_shared_prefill_starts": [0],
                "_shared_prefill_ends": [49152],
            }
            ok_src = [self._LayerSrc((o_r0,), i) for i in range(4)]
            h.hook[0] = lambda _: (ok_box.append(h.src.step), h.eng._prepare_shared_prefill_sources(ok_src, 0, kw_ok))
            h.src.begin_runner_step(cm_ok, ("req_ok",), (49152,), (0, 8), (tbl, tbl))
            h.impl.start_load_kv(None)
            self.assertEqual(len(ok_box), 1)
            sok = ok_box[0]
            self.assertIs(h.src.step, sok)
            sok.finish(completion=torch.npu.Event())
            self.assertTrue(sok.try_retire())
            self._close_src(h.src)

    def test_mixed_two_req_recompute_fallback_and_nonshared_upload(self) -> None:
        with _fake_npu():
            oa, ob = _mk_obj(0x9000, 49152), _mk_obj(0xB000, 49152)
            gens, up_ev, sbox = {}, [], []
            h = self._mk_src(
                4,
                eg={"prepare_layerwise_prefill_group0_skip": self._nop},
                ee={"_should_use_shared_layerwise_retrieve": lambda _g: False, "_recompute_fallback_req_ids": set()},
            )

            def _orig(_):
                sbox.append(h.src.step)
                gens["a"], gens["b"] = (
                    self._send(h.gc, h.impl, "req_a", oa, 3, sub=False),
                    self._send(h.gc, h.impl, "req_b", ob, 2, sub=False),
                )
                h.eng._recompute_fallback_req_ids.add("req_a")

            h.hook[0] = _orig
            orec = h.src._record_upload
            h.src._record_upload = lambda st, lid=None: (up_ev.append((st._host_slot, lid)), orec(st, lid))[1]
            cm = self._cm(("req_a", 49152, True, False, 0), ("req_b", 49152, True, False, 0))
            h.cm_box[0] = cm
            tbl = np.vstack([self._ar(0, 390, d2=False), self._ar(400, 790, d2=False)])
            h.src.begin_runner_step(cm, ("req_a", "req_b"), (49152, 49152), (0, 8, 16), (tbl, tbl))
            self.assertEqual(up_ev, [])
            h.impl.start_load_kv(None)
            self.assertEqual(len(sbox), 1)
            step = h.src.step
            assert step is not None
            self.assertEqual(step.selected_request_ids, frozenset({"req_b"}))
            for idx in (0, 1):
                self.assertNotIn(("req_a", idx), step._req_bank)
                self.assertNotIn(("req_a", idx), step._bank_plans)
            for bi in range(len(step.omitted_by_bank)):
                for blk in range(0, 384):
                    self.assertNotIn(blk, step.omitted_by_bank[bi])
            self.assertEqual(up_ev, [(step._host_slot, 0), (step._host_slot, 1)])
            up_ev.clear()
            gens["a"].send([oa])
            self.assertEqual(up_ev, [])
            gens["b"].send([ob])
            gens["b"].send([ob])
            self.assertEqual(up_ev, [(step._host_slot, 2), (step._host_slot, 3)])
            step.finish(completion=torch.npu.Event())
            self.assertTrue(step.try_retire())
            self._close_src(h.src)

    def test_chunk_info_dva_cache_liveness_and_generation_invalidation(self) -> None:
        src = object.__new__(lms.LMCacheSelectedKVSource)
        src.gpu_connector, src.engine = SimpleNamespace(), SimpleNamespace(shared_cpu_cache_generation=1)
        src._dva_cache, src._dva_cache_gen, ptr_calls, rb = {}, 1, [], skv.MAIN_ROW_BYTES
        src._get_device_ptr_fn = lambda ptr, sz: (ptr_calls.append((ptr, sz)), ptr + 0x10000)[1]

        def _o(t: int, p: int, al: Any):
            return SimpleNamespace(
                layer_size_bytes=lambda _l, k=t: rb * k, layer_data_ptr=lambda _l, ptr=p: ptr, parent_allocator=al
            )

        live = _o(2, 0x1000, SimpleNamespace(_mapping=SimpleNamespace(_closed=False)))
        self.assertEqual(
            (src._chunk_info(live, 0), src._chunk_info(live, 0), len(ptr_calls)), ((2, 0x11000), (2, 0x11000), 1)
        )
        unreg = _o(3, 0x2000, SimpleNamespace(_unregistered=False))
        self.assertEqual(
            (src._chunk_info(unreg, 0), src._chunk_info(unreg, 0), len(ptr_calls)), ((3, 0x12000), (3, 0x12000), 2)
        )
        src.engine.shared_cpu_cache_generation = 2
        self.assertEqual(
            (src._chunk_info(live, 0), src._dva_cache_gen, len(src._dva_cache), len(ptr_calls)), ((2, 0x11000), 2, 1, 3)
        )
        dm, du = (
            _o(1, 0x3000, SimpleNamespace(_mapping=SimpleNamespace(_closed=True))),
            _o(1, 0x4000, SimpleNamespace(_unregistered=True)),
        )
        na, sa = _o(1, 0x5000, None), _o(1, 0x6000, SimpleNamespace())
        self.assertEqual(
            [src._chunk_info(x, 0) for x in (dm, dm, du, na, sa, sa)],
            [(1, 0x13000), (1, 0x13000), (1, 0x14000), (1, 0x15000), (1, 0x16000), (1, 0x16000)],
        )
        self.assertEqual((len(ptr_calls), len(src._dva_cache)), (9, 1))

    def test_1m_1024_chunk_8192_page_partition_populate_and_dva_cache(self) -> None:
        nop = lambda *a, **k: None
        req = lms.SelectedRequest("r_1m", 1047552, 1024, True, 1047552)
        blk0 = np.arange(8192, dtype=np.int32)[None, :]
        blk1 = (np.arange(8192, dtype=np.int32) + 8192)[None, :]
        om, kp = lms.compute_per_bank_block_partition([req], (blk0, blk1), domain=16384)
        self.assertEqual((len(om[0]), len(kp[0]), len(om[1]), len(kp[1])), (8184, 8, 8184, 8))

        class _Chunk1M:
            num_layers = 2
            valid = True

            def __init__(self, base: int, tokens: int) -> None:
                self._base = base
                self.layer_size = tokens * skv.MAIN_ROW_BYTES

            def layer_size_bytes(self, _lid: int) -> int:
                return self.layer_size

            def layer_data_ptr(self, lid: int) -> int:
                return self._base + lid * 0x100

        with _fake_npu():
            gc = SimpleNamespace(
                _layerwise_prefill_bank_count=2, batched_to_gpu=nop, release_layerwise_prefill_dma_cache=nop
            )
            eng = SimpleNamespace(
                gpu_connector=gc,
                shared_cpu_cache_generation=1,
                _prepare_shared_prefill_sources=nop,
                release_shared_cpu_sparse_request=nop,
            )
            impl = SimpleNamespace(
                lmcache_engine=eng,
                _submit_initial_layerwise_prefill_load=nop,
                submit_layerwise_prefill_load=nop,
                start_load_kv=nop,
                _parent=SimpleNamespace(_kv_transfer_config=SimpleNamespace(kv_role="kv_producer")),
            )
            npu_m = SimpleNamespace(bind_incremental_copy_addresses=nop, BoundCopyPrefix=nop)
            mc = {n: (torch.empty((16384, 1, 1, 1)),) * 2 for n in ("l0", "l1")}
            src = lms.LMCacheSelectedKVSource(
                SimpleNamespace(device="cpu"),
                impl,
                ("l0", "l1"),
                mc,
                get_device_ptr_fn=lambda ptr, _sz: (1 << 63) | ptr,
                npu_connectors_module=npu_m,
            )
            src._step_host, src._step_dev = src._host_bases, src._dev_bases
            step = lms.LMCacheSelectedLoadStep(
                src, SimpleNamespace(requests=[]), [req], (blk0, blk1), om, src._step_dev
            )
            objs = [_Chunk1M(0x1000_0000_0000 + i * 0x1_0000_0000, 1024) for i in range(1024)]
            starts = [i * 1024 for i in range(1024)]
            ends = [(i + 1) * 1024 for i in range(1024)]
            src._populate_bases(step, "r_1m", 0, objs, starts, ends)
            src._populate_bases(step, "r_1m", 1, objs, starts, ends)
            host_u64 = src._step_host.numpy().view(np.uint64)
            for lid, base_blk in ((0, 0), (1, 8192)):
                for p in (0, 8183):
                    base0 = np.uint64((1 << 63) | (0x1000_0000_0000 + (p // 8) * 0x1_0000_0000 + lid * 0x100))
                    row = np.uint64((p % 8) * skv.PAGE_ROWS)
                    exp0 = base0 + row * np.uint64(skv.MLA_NOPE_ROW_BYTES)
                    exp1 = base0 + np.uint64(1024 * skv.MLA_NOPE_ROW_BYTES) + row * np.uint64(skv.MLA_PE_ROW_BYTES)
                    self.assertEqual(
                        (int(host_u64[lid, base_blk + p, 0]), int(host_u64[lid, base_blk + p, 1])),
                        (int(exp0), int(exp1)),
                    )
                self.assertTrue(np.all(host_u64[lid, base_blk + 8184 : base_blk + 8192] == 0))

            # Vectorized _prepare_shared (LayerPageSource path) across 1,024 chunks with > 4 GiB slab offsets
            src._step_host.zero_()
            step_vec = lms.LMCacheSelectedLoadStep(
                src, SimpleNamespace(requests=[]), [req], (blk0, blk1), om, src._step_dev
            )
            src.step = step_vec
            pages_1m = tuple(objs)
            prep_vec = [SimpleNamespace(pages=pages_1m, suffix=(), layer_id=lid) for lid in range(2)]
            eng._prepare_shared_prefill_sources(
                prep_vec, 0, {"req_id": "r_1m", "_shared_prefill_starts": starts, "_shared_prefill_ends": ends}
            )
            for lid, base_blk in ((0, 0), (1, 8192)):
                for p in (0, 8183):
                    base0 = np.uint64((1 << 63) | (0x1000_0000_0000 + (p // 8) * 0x1_0000_0000 + lid * 0x100))
                    row = np.uint64((p % 8) * skv.PAGE_ROWS)
                    exp0 = base0 + row * np.uint64(skv.MLA_NOPE_ROW_BYTES)
                    exp1 = base0 + np.uint64(1024 * skv.MLA_NOPE_ROW_BYTES) + row * np.uint64(skv.MLA_PE_ROW_BYTES)
                    self.assertEqual(
                        (int(host_u64[lid, base_blk + p, 0]), int(host_u64[lid, base_blk + p, 1])),
                        (int(exp0), int(exp1)),
                    )
                self.assertTrue(np.all(host_u64[lid, base_blk + 8184 : base_blk + 8192] == 0))
            src.step = None

            # > 65,536 _dva_cache lookups retain cleanly; boundary == MAX_DVA_CACHE_ENTRIES clears on next insert
            ptr_calls = [0]
            src._dva_cache.clear()
            src._get_device_ptr_fn = lambda ptr, _sz: (ptr_calls.__setitem__(0, ptr_calls[0] + 1), ptr + 0x10000)[1]
            live_al = SimpleNamespace(_mapping=SimpleNamespace(_closed=False))
            cur_ptr = [0]
            probe_obj = SimpleNamespace(
                layer_size_bytes=lambda _l: skv.MAIN_ROW_BYTES,
                layer_data_ptr=lambda _l: cur_ptr[0],
                parent_allocator=live_al,
            )
            for i in range(70000):
                cur_ptr[0] = 0x1000_0000 + i * 0x1000
                src._chunk_info(probe_obj, 0)
            self.assertEqual((len(src._dva_cache), ptr_calls[0]), (70000, 70000))
            self.assertLess(len(src._dva_cache), lms.MAX_DVA_CACHE_ENTRIES)
            for i in range(70000):
                cur_ptr[0] = 0x1000_0000 + i * 0x1000
                src._chunk_info(probe_obj, 0)
            self.assertEqual(ptr_calls[0], 70000)

            gen = src._dva_cache_gen
            rb = skv.MAIN_ROW_BYTES
            for j in range(lms.MAX_DVA_CACHE_ENTRIES - len(src._dva_cache)):
                src._dva_cache[(gen, 0, 0x8000_0000 + j * 0x1000, rb)] = 0x9000_0000 + j * 0x1000
            self.assertEqual(len(src._dva_cache), lms.MAX_DVA_CACHE_ENTRIES)
            cur_ptr[0] = 0xF000_0000
            self.assertEqual(src._chunk_info(probe_obj, 0), (1, 0xF001_0000))
            self.assertEqual(len(src._dva_cache), 1)

    def test_kv_both_full_resident_policy_deselects_for_dense_decode(self) -> None:
        with _fake_npu():
            htok, qtok = 131072, 1024
            o_sh, s_st, s_sk, kv_cfg = self._sh_obj(0x9000, htok, 4), [], [], SimpleNamespace(kv_role="kv_both")
            ei = {
                "enable_sparse_attention": True,
                "_dsa_kv_policy_threshold": 1000000,
                "_dsa_scratch_capacity": 0,
                "_selected_skip_group0_dma_req_ids": frozenset(),
            }
            h = self._mk_src(
                4,
                mc=2048,
                track_btg=True,
                eg={"prepare_layerwise_prefill_group0_skip": self._nop},
                ee={"_should_use_shared_layerwise_retrieve": lambda g: g == 0},
                ei=ei,
            )
            h.impl._parent = SimpleNamespace(_get_connector_metadata=lambda: h.cm_box[0], _kv_transfer_config=kv_cfg)
            self._attach_skip_hook(h, o_sh, htok, s_step=s_st, s_skip=s_sk)
            tbl = self._ar(10, 10 + (htok + qtok) // skv.PAGE_ROWS + 4)
            try:
                for rid, role, sa, th, sc, esel in (
                    ("r_both_thresh", "kv_both", True, 1000000, 0, False),
                    ("r_both_dense_attn", "kv_both", False, 0, 0, False),
                    ("r_both_sel", "kv_both", True, 0, 2048, True),
                    ("r_prod_sel", "kv_producer", True, 1000000, 0, True),
                ):
                    h.btg.clear()
                    kv_cfg.kv_role, h.impl.enable_sparse_attention = role, sa
                    h.impl._dsa_kv_policy_threshold, h.impl._dsa_scratch_capacity = th, sc
                    cm = self._cm((rid, htok, True, False, 0, True))
                    h.cm_box[0] = cm
                    h.src.begin_runner_step(cm, (rid,), (htok,), (0, qtok), (tbl, tbl))
                    h.impl.start_load_kv(None)
                    if esel:
                        self.assertIsNotNone(s_st[-1])
                        self.assertEqual(
                            (s_st[-1].selected_request_ids, s_sk[-1], h.btg), (frozenset({rid}), frozenset({rid}), [])
                        )
                        h.src.step.finish(completion=torch.npu.Event())
                    else:
                        self.assertIsNone(s_st[-1])
                        self.assertIsNone(h.src.step)
                        self.assertEqual((s_sk[-1], h.btg), (frozenset(), [rid]))
            finally:
                self._close_src(h.src)

    def test_online_hardening_guards(self) -> None:
        req_apc = SimpleNamespace(
            block_allocation_mode="prefill_child",
            save_spec=SimpleNamespace(can_save=True, skip_leading_tokens=8192),
            load_spec=SimpleNamespace(can_load=True, lmcache_cached_tokens=49152, vllm_cached_tokens=256),
        )
        self.assertFalse(lms.classify_lmcache_request("r_apc", 49152, 128, req_apc).selected)

        _cfg, _CtrlEv, _build_env, _run_step = self._pipe_setup()
        with _fake_npu(), patch.object(torch.npu, "Event", _CtrlEv):
            e = _build_env()
            with self._init_rt(e):
                rt = e.rn.selected_kv_runtime
                st0, ev0, _ = _run_step(e, "r_fence", bb=0, hold=True, fos=False)
                alt_s = torch.npu.Stream(device="cpu")
                with torch.npu.stream(alt_s):
                    got = rt.bind_step(None, (), (), (), self._am(e.layers))
                self.assertIsNone(got)
                self.assertIn(ev0, alt_s.waited)
                ev0.done = True
                skv.retire_selected_kv_if_idle(e.rn)
                skv.close_selected_kv_worker(e.wk)

            ef = _build_env()
            with self._init_rt(ef):
                rtf = ef.rn.selected_kv_runtime
                _stf, evf, _ = _run_step(ef, "r_shut", bb=5, hold=True, fos=True)
                rtf._handoff_previous()
                rtf.fail(_stf)
                rtf.device = SimpleNamespace(type="npu")
                with patch.object(
                    torch.npu, "synchronize", side_effect=RuntimeError("sync hung"), create=True
                ) as mock_sync:
                    skv.close_selected_kv_worker(ef.wk)
                mock_sync.assert_called_once_with(rtf.device)
                self.assertTrue(getattr(ef.eng, "_retain_unfenced_external_dma", False))
                evf.fail_on_sync = False
                evf.done = True

            ef_ok = _build_env()
            with self._init_rt(ef_ok):
                rtf_ok = ef_ok.rn.selected_kv_runtime
                _st_ok, ev_ok, _ = _run_step(ef_ok, "r_shut_ok", bb=5, hold=True, fos=True)
                rtf_ok._handoff_previous()
                rtf_ok.fail(_st_ok)
                rtf_ok.device = SimpleNamespace(type="npu")
                with patch.object(torch.npu, "synchronize", return_value=None, create=True) as mock_sync_ok:
                    skv.close_selected_kv_worker(ef_ok.wk)
                mock_sync_ok.assert_called_once_with(rtf_ok.device)
                self.assertFalse(getattr(ef_ok.eng, "_retain_unfenced_external_dma", False))
                ev_ok.fail_on_sync = False
                ev_ok.done = True

        with _fake_npu():
            ei_skew = {
                "enable_sparse_attention": True,
                "_selected_skip_group0_dma_req_ids": frozenset(),
                "_should_skip_group0_prefill_dma": None,
            }
            hs = self._mk_src(
                4,
                mc=500,
                track_btg=True,
                eg={"prepare_layerwise_prefill_group0_skip": self._nop},
                ee={"_should_use_shared_layerwise_retrieve": lambda g: g == 0},
                ei=ei_skew,
            )
            s_sk_skew: list[frozenset[str]] = []
            hs.cm_box[0] = self._cm("r_skew")
            self._attach_skip_hook(hs, self._sh_obj(0x9000, 49152, 4), 49152, s_skip=s_sk_skew)
            ts = self._ar(0, 390)
            hs.src.begin_runner_step(hs.cm_box[0], ("r_skew",), (49152,), (0, 8), (ts, ts))
            hs.impl.start_load_kv(None)
            self.assertEqual((s_sk_skew[-1], hs.impl._selected_skip_group0_dma_req_ids), (frozenset(), frozenset()))
            hs.src.step.finish(completion=torch.npu.Event())
            self._close_src(hs.src)

            o0, o1 = _mk_obj(0x1000, 24576), _mk_obj(0x2000, 24576)
            ptrs_seen: list[tuple[int, int]] = []

            def _dva_tail(hp: int, bs: int) -> int:
                ptrs_seen.append((hp, bs))
                return (1 << 63) | (hp + 0x100000)

            ht = self._mk_src(
                4,
                banks=2,
                mc=500,
                dva=_dva_tail,
            )
            ht.cm_box[0] = self._cm("r_tail")
            cached_bad = [[(1 << 63) | (o0.layer_data_ptr(lid) + 0x100000), 0xDEAD] for lid in range(4)]
            kw_t = {"req_id": "r_tail", "cached_chunk_dev_ptrs": cached_bad, "_reuse_shared_dense_prefix": True}
            ht.hook[0] = lambda _: (
                next(ht.gc.batched_to_gpu([0, 24576], [24576, 49152], req_id="r_tail", kv_group=0)),
                ht.eng._prepare_shared_prefill_sources([self._LayerSrc((o0, o1), i) for i in range(4)], 0, kw_t),
            )
            tt = self._ar(0, 385)
            ht.src.begin_runner_step(ht.cm_box[0], ("r_tail",), (49152,), (0, 8), (tt, tt))
            ht.impl.start_load_kv(None)
            st_t = ht.src.step
            exp_tail = (1 << 63) | (o1.layer_data_ptr(0) + 0x100000)
            self.assertIn(exp_tail, st_t.page_bases("l0")[:, 0].view(torch.uint64).tolist())
            self.assertGreaterEqual(len(ptrs_seen), 2)
            done_flag = [False]
            ev_hold = SimpleNamespace(query=lambda: done_flag[0], synchronize=lambda: done_flag.__setitem__(0, True))
            self.assertIsNotNone(st_t.connector_metadata)
            self.assertFalse(st_t.finish(completion=ev_hold))
            self.assertIsNotNone(st_t.connector_metadata)
            done_flag[0] = True
            self.assertTrue(st_t.try_retire())
            self.assertIsNone(st_t.connector_metadata)
            ht.src.begin_runner_step(ht.cm_box[0], ("r_tail",), (49152,), (0, 8), (tt, tt))
            ht.impl.start_load_kv(None)
            st_c = ht.src.step
            self.assertIsNotNone(st_c.connector_metadata)
            st_c.cancel_unused()
            self.assertIsNone(st_c.connector_metadata)
            ht.src.step = None
            self._close_src(ht.src)

            o0_3, o1_3, o2_3 = _mk_obj(0x1000, 16384), _mk_obj(0x2000, 16384), _mk_obj(0x3000, 16384)
            ptrs_seen_3: list[tuple[int, int]] = []

            def _dva_mid(hp: int, bs: int) -> int:
                ptrs_seen_3.append((hp, bs))
                return (1 << 63) | (hp + 0x100000)

            h3 = self._mk_src(4, banks=2, mc=500, dva=_dva_mid)
            h3.cm_box[0] = self._cm(("r_mid_tamper", 49152))
            cached_mid_bad = [
                [
                    (1 << 63) | (o0_3.layer_data_ptr(lid) + 0x100000),
                    0xDEAD,
                    (1 << 63) | (o2_3.layer_data_ptr(lid) + 0x100000),
                ]
                for lid in range(4)
            ]
            kw_3 = {
                "req_id": "r_mid_tamper",
                "cached_chunk_dev_ptrs": cached_mid_bad,
                "_reuse_shared_dense_prefix": True,
            }
            h3.hook[0] = lambda _: (
                next(h3.gc.batched_to_gpu([0, 16384, 32768], [16384, 32768, 49152], req_id="r_mid_tamper", kv_group=0)),
                h3.eng._prepare_shared_prefill_sources(
                    [self._LayerSrc((o0_3, o1_3, o2_3), i) for i in range(4)], 0, kw_3
                ),
            )
            tt3 = self._ar(0, 385)
            h3.src.begin_runner_step(h3.cm_box[0], ("r_mid_tamper",), (49152,), (0, 8), (tt3, tt3))
            h3.impl.start_load_kv(None)
            st_3 = h3.src.step
            exp_mid = (1 << 63) | (o1_3.layer_data_ptr(0) + 0x100000)
            self.assertIn(exp_mid, st_3.page_bases("l0")[:, 0].view(torch.uint64).tolist())
            self.assertGreaterEqual(len(ptrs_seen_3), 3)
            st_3.finish(completion=torch.npu.Event())
            h3.src.step = None
            self._close_src(h3.src)

            h_no = self._mk_src(2)
            self.assertTrue(callable(getattr(h_no.gc, "synchronize_selected_group0_skip", None)))
            h_no.src.device = SimpleNamespace(type="npu")
            meta_sync_calls: list[str] = []
            copy_sync_calls: list[str] = []
            h_no.src._metadata_stream = SimpleNamespace(synchronize=lambda: meta_sync_calls.append("meta"))

            class _FakeRt:
                def __init__(self, source: Any, copy_stream: Any) -> None:
                    self.source = source
                    self._copy_stream = copy_stream

            fake_rt = _FakeRt(
                h_no.src,
                SimpleNamespace(synchronize=lambda: copy_sync_calls.append("copy")),
            )
            skv._ACTIVE_RUNTIMES.add(fake_rt)
            try:
                with patch.object(torch.npu, "synchronize", create=True) as mock_npu_sync:
                    h_no.gc.synchronize_selected_group0_skip()
            finally:
                skv._ACTIVE_RUNTIMES.discard(fake_rt)
            self.assertEqual(meta_sync_calls, ["meta"])
            self.assertEqual(copy_sync_calls, ["copy"])
            mock_npu_sync.assert_called_once_with(h_no.src.device)
            self._close_src(h_no.src, restore=False)
            self.assertFalse(hasattr(h_no.gc, "synchronize_selected_group0_skip"))

            orig_sync = lambda: "orig"
            h_had = self._mk_src(2, eg={"synchronize_selected_group0_skip": orig_sync})
            self.assertIsNot(h_had.gc.synchronize_selected_group0_skip, orig_sync)
            self._close_src(h_had.src, restore=False)
            self.assertIs(h_had.gc.synchronize_selected_group0_skip, orig_sync)

    def test_nonvectorized_dva_tamper_and_skew_fallback_retention(self) -> None:
        with _fake_npu():
            nl = 4
            o0, o1, o2 = _mk_obj(0x1000, 16384), _mk_obj(0x2000, 16384), _mk_obj(0x3000, 16384)
            pages_src = (o0, o1, o2)
            for o, toks in ((o0, 16384), (o1, 16384), (o2, 16384)):
                o.num_layers = nl
                o.valid = True
                o.layer_size = toks * skv.MAIN_ROW_BYTES
                o.layer_size_bytes = lambda _lid, sz=o.layer_size: sz

            dva_calls: list[tuple[int, int]] = []

            def _dva_fn(hp: int, bs: int) -> int:
                dva_calls.append((hp, bs))
                return (1 << 63) | (hp + 0x100000)

            h = self._mk_src(nl, banks=2, mc=600, dva=_dva_fn)
            tbl0, tbl1 = self._ar(0, 385), self._ar(127, 512)
            cm_gold = self._cm(("r_gold", 49152))
            h.cm_box[0] = cm_gold
            h.hook[0] = lambda _: (
                next(h.gc.batched_to_gpu([0, 16384, 32768], [16384, 32768, 49152], req_id="r_gold", kv_group=0)),
                h.eng._prepare_shared_prefill_sources(
                    [self._LayerSrc(pages_src, i) for i in range(nl)],
                    0,
                    {"req_id": "r_gold", "cached_chunk_dev_ptrs": [], "_reuse_shared_dense_prefix": False},
                ),
            )
            h.src.begin_runner_step(cm_gold, ("r_gold",), (49152,), (0, 8), (tbl0, tbl1))
            h.impl.start_load_kv(None)
            st_gold = h.src.step
            golden = [st_gold.page_bases(n).view(torch.uint64).clone() for n in h.layers]
            st_gold.finish(completion=torch.npu.Event())
            h.src._prune_retired_steps()

            # 1a) Non-vectorized path (distinct MemoryObj instances per layer => objs[k] is not objs0[k])
            h.src._dva_cache.clear()
            dva_calls.clear()
            nv_layers: list[tuple[Any, Any, Any]] = []
            for lid in range(nl):
                lo0 = _mk_obj(0x10000 * (lid + 1) + 0x1000, 16384)
                lo1 = _mk_obj(0x10000 * (lid + 1) + 0x2000, 16384)
                lo2 = _mk_obj(0x10000 * (lid + 1) + 0x3000, 16384)
                for lo, toks in ((lo0, 16384), (lo1, 16384), (lo2, 16384)):
                    lo.num_layers = nl
                    lo.valid = True
                    lo.layer_size = toks * skv.MAIN_ROW_BYTES
                    lo.layer_size_bytes = lambda _lid, sz=lo.layer_size: sz
                nv_layers.append((lo0, lo1, lo2))

            cm_nv_gold = self._cm(("r_nv_gold", 49152))
            h.cm_box[0] = cm_nv_gold
            h.hook[0] = lambda _: (
                next(h.gc.batched_to_gpu([0, 16384, 32768], [16384, 32768, 49152], req_id="r_nv_gold", kv_group=0)),
                h.eng._prepare_shared_prefill_sources(
                    [self._LayerSrc(nv_layers[i], i) for i in range(nl)],
                    0,
                    {"req_id": "r_nv_gold", "cached_chunk_dev_ptrs": [], "_reuse_shared_dense_prefix": False},
                ),
            )
            h.src.begin_runner_step(cm_nv_gold, ("r_nv_gold",), (49152,), (0, 8), (tbl0, tbl1))
            h.impl.start_load_kv(None)
            st_nv_gold = h.src.step
            nv_golden = [st_nv_gold.page_bases(n).view(torch.uint64).clone() for n in h.layers]
            st_nv_gold.finish(completion=torch.npu.Event())
            h.src._prune_retired_steps()

            h.src._dva_cache.clear()
            dva_calls.clear()
            cached_nv = [[(1 << 63) | (o.layer_data_ptr(lid) + 0x100000) for o in nv_layers[lid]] for lid in range(nl)]
            cached_nv[1][1] = 0xDEAD
            cm_nv = self._cm(("r_nv", 49152))
            h.cm_box[0] = cm_nv
            h.hook[0] = lambda _: (
                next(h.gc.batched_to_gpu([0, 16384, 32768], [16384, 32768, 49152], req_id="r_nv", kv_group=0)),
                h.eng._prepare_shared_prefill_sources(
                    [self._LayerSrc(nv_layers[i], i) for i in range(nl)],
                    0,
                    {"req_id": "r_nv", "cached_chunk_dev_ptrs": cached_nv, "_reuse_shared_dense_prefix": True},
                ),
            )
            h.src.begin_runner_step(cm_nv, ("r_nv",), (49152,), (0, 8), (tbl0, tbl1))
            h.impl.start_load_kv(None)
            st_nv = h.src.step
            for lid, n in enumerate(h.layers):
                self.assertTrue(torch.equal(st_nv.page_bases(n).view(torch.uint64), nv_golden[lid]))
            self.assertIn((nv_layers[1][1].layer_data_ptr(1), nv_layers[1][1].layer_size_bytes(1)), dva_calls)
            st_nv.finish(completion=torch.npu.Event())
            h.src._prune_retired_steps()

            # 1b) Vectorized path (shared pages_src tuple => can_vectorize == True) with ptrs[1][1] = 0 out-of-range
            h.src._dva_cache.clear()
            dva_calls.clear()
            cached_vec = [[(1 << 63) | (o.layer_data_ptr(lid) + 0x100000) for o in pages_src] for lid in range(nl)]
            cached_vec[1][1] = 0
            cm_vec = self._cm(("r_vec", 49152))
            h.cm_box[0] = cm_vec
            h.hook[0] = lambda _: (
                next(h.gc.batched_to_gpu([0, 16384, 32768], [16384, 32768, 49152], req_id="r_vec", kv_group=0)),
                h.eng._prepare_shared_prefill_sources(
                    [self._LayerSrc(pages_src, i) for i in range(nl)],
                    0,
                    {"req_id": "r_vec", "cached_chunk_dev_ptrs": cached_vec, "_reuse_shared_dense_prefix": True},
                ),
            )
            h.src.begin_runner_step(cm_vec, ("r_vec",), (49152,), (0, 8), (tbl0, tbl1))
            h.impl.start_load_kv(None)
            st_vec = h.src.step
            for lid, n in enumerate(h.layers):
                self.assertTrue(torch.equal(st_vec.page_bases(n).view(torch.uint64), golden[lid]))
            self.assertIn((o1.layer_data_ptr(1), o1.layer_size_bytes(1)), dva_calls)
            st_vec.finish(completion=torch.npu.Event())
            h.src._prune_retired_steps()
            self._close_src(h.src)

            # 2) Rolling-upgrade skew: impl lacks _sync_recompute_fallback_from_engine
            o_sk = self._sh_obj(0x9000, 49152, nl)
            h_sk = self._mk_src(
                nl,
                eg={
                    "batched_to_gpu": lambda *_a, **_k: iter([None]),
                    "prepare_layerwise_prefill_group0_skip": self._nop,
                },
                ee={"_should_use_shared_layerwise_retrieve": lambda g: g == 0, "_recompute_fallback_req_ids": set()},
            )
            self.assertFalse(hasattr(h_sk.impl, "_sync_recompute_fallback_from_engine"))
            tbl_sk = self._ar(0, 390)

            def _orig_skew(_):
                sc = h_sk.src.step
                assert sc is not None
                h_sk.src._retain_objs(sc, [o_sk])
                h_sk.eng._recompute_fallback_req_ids.add("req_skew")

            h_sk.hook[0] = _orig_skew
            cm_sk = self._cm(("req_skew", 49152, True, False, 0))
            h_sk.cm_box[0] = cm_sk
            h_sk.src.begin_runner_step(cm_sk, ("req_skew",), (49152,), (0, 8), (tbl_sk, tbl_sk))
            h_sk.impl.start_load_kv(None)
            self.assertEqual(h_sk.eng._recompute_fallback_req_ids, set())
            self.assertEqual(h_sk.impl._step_recompute_fallback_req_ids, {"req_skew"})

            cm_next = self._cm(("req_next", 49152, True, False, 0))
            h_sk.cm_box[0] = cm_next
            kw_next: dict[str, Any] = {
                "req_id": "req_next",
                "_shared_prefill_starts": [0],
                "_shared_prefill_ends": [49152],
            }
            next_src = [self._LayerSrc((o_sk,), i) for i in range(nl)]
            h_sk.hook[0] = lambda _: h_sk.eng._prepare_shared_prefill_sources(next_src, 0, kw_next)
            h_sk.src.begin_runner_step(cm_next, ("req_next",), (49152,), (0, 8), (tbl_sk, tbl_sk))
            h_sk.impl.start_load_kv(None)
            self.assertEqual(h_sk.impl._step_recompute_fallback_req_ids, set())
            s_next = h_sk.src.step
            self.assertIsNotNone(s_next)
            s_next.finish(completion=torch.npu.Event())
            self.assertTrue(s_next.try_retire())
            self._close_src(h_sk.src)

    def test_inference_mode_storer_overlap_guard_and_omit_token_handoff(self) -> None:
        def _outer_gen_dec(fn: Any) -> Any:
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                inner = fn(*args, **kwargs)
                yield from inner

            return _wrapped

        @_outer_gen_dec
        @torch.inference_mode()
        def _suspended_store_layer(starts: list[int]) -> Any:
            yield starts
            yield None

        with _fake_npu():
            # 1a) Positive path: kv_group==0 starts=[49152] >= omit_end_token=49152 passes,
            #     kv_group==1 starts=[0] is ignored, and impl._selected_group0_omit_end_tokens is populated.
            storers: dict[tuple[Any, ...], Any] = {}
            seen_omit_maps: list[dict[str, int]] = []
            h = self._mk_src(
                4,
                eg={"prepare_layerwise_prefill_group0_skip": self._nop},
                ee={"_should_use_shared_layerwise_retrieve": lambda g: g == 0},
                ei={"_layerwise_save_storers": storers},
            )
            o_pos = self._sh_obj(0x9000, 49152, 4)
            tbl = self._ar(0, 390)
            cm_pos = self._cm(("req_pos", 49152, True, False, 0))
            h.cm_box[0] = cm_pos

            def _orig_pos(_: Any) -> None:
                seen_omit_maps.append(dict(h.impl._selected_group0_omit_end_tokens))
                st_g0 = _suspended_store_layer([49152])
                st_g1 = _suspended_store_layer([0])
                next(st_g0)
                next(st_g1)
                storers[("req_pos", "normal_save", 0, 0, 53248)] = st_g0
                storers[("req_pos", "normal_save", 1, 0, 53248)] = st_g1
                h.eng._prepare_shared_prefill_sources(
                    [self._LayerSrc((o_pos,), i) for i in range(4)],
                    0,
                    {"req_id": "req_pos", "_shared_prefill_starts": [0], "_shared_prefill_ends": [49152]},
                )

            h.hook[0] = _orig_pos
            h.src.begin_runner_step(cm_pos, ("req_pos",), (49152,), (0, 8), (tbl, tbl))
            h.impl.start_load_kv(None)
            self.assertEqual(seen_omit_maps, [{"req_pos": 49152}])
            for g in storers.values():
                g.close()
            storers.clear()
            st_pos = h.src.step
            self.assertIsNotNone(st_pos)
            st_pos.finish(completion=torch.npu.Event())
            h.src._prune_retired_steps()
            self._close_src(h.src)

            # 1b) Fail-closed path: kv_group==0 @torch.inference_mode() storer with starts=[49024] < 49152 raises,
            #     invokes retrieve/save abort hooks, and clears _selected_group0_omit_end_tokens.
            aborts: list[tuple[str, tuple[str, ...]]] = []
            storers_ov: dict[tuple[Any, ...], Any] = {}
            h_ov = self._mk_src(
                4,
                eg={"prepare_layerwise_prefill_group0_skip": self._nop},
                ee={"_should_use_shared_layerwise_retrieve": lambda g: g == 0},
                ei={
                    "_layerwise_save_storers": storers_ov,
                    "_abort_layerwise_retrieve_step": lambda rq: aborts.append(
                        ("ret", tuple(str(r.req_id) for r in rq))
                    ),
                    "_abort_save_step": lambda rq: aborts.append(("save", tuple(str(r.req_id) for r in rq))),
                },
            )
            o_ov = self._sh_obj(0x9000, 49152, 4)
            cm_ov = self._cm(("req_ov", 49152, True, False, 0))
            h_ov.cm_box[0] = cm_ov

            def _orig_ov(_: Any) -> None:
                st_bad = _suspended_store_layer([49024])
                next(st_bad)
                storers_ov[("req_ov", "normal_save", 0, 0, 53248)] = st_bad
                h_ov.eng._prepare_shared_prefill_sources(
                    [self._LayerSrc((o_ov,), i) for i in range(4)],
                    0,
                    {"req_id": "req_ov", "_shared_prefill_starts": [0], "_shared_prefill_ends": [49152]},
                )

            h_ov.hook[0] = _orig_ov
            h_ov.src.begin_runner_step(cm_ov, ("req_ov",), (49152,), (0, 8), (tbl, tbl))
            with self.assertRaisesRegex(RuntimeError, r"Save overlaps omitted: req_ov"):
                h_ov.impl.start_load_kv(None)
            self.assertEqual(aborts, [("ret", ("req_ov",)), ("save", ("req_ov",))])
            self.assertEqual(h_ov.impl._selected_group0_omit_end_tokens, {})
            for g in storers_ov.values():
                g.close()
            for fn in reversed(h_ov.src._restorers):
                fn()

            # 2) Rolling-upgrade skew: impl lacks _supports_selected_group0_omit_skip.
            def _orig_prep(req: Any, _save_spec: Any, _kv_group: int, *_a: Any, **_kw: Any) -> tuple[Any, ...]:
                toks = list(req.token_ids)
                smap = torch.arange(len(toks), dtype=torch.int64)
                ones = torch.ones(len(toks), dtype=torch.bool)
                return (toks, smap, ones, 0, {"_prefill_metadata_cache": object()}, False)

            h_rw = self._mk_src(
                4,
                mc=1600,
                eg={"prepare_layerwise_prefill_group0_skip": self._nop},
                ee={
                    "_should_use_shared_layerwise_retrieve": lambda g: g == 0,
                    "_recompute_fallback_req_ids": set(),
                },
                ei={"_prepare_layerwise_store_inputs": _orig_prep, "kv_role": "kv_producer"},
            )
            self.assertFalse(bool(getattr(type(h_rw.impl), "_supports_selected_group0_omit_skip", False)))
            self.assertIsNot(h_rw.impl._prepare_layerwise_store_inputs, _orig_prep)

            o_rw = self._sh_obj(0x9000, 49152, 4)
            cm_rw = self._cm(
                ("req_rw", 49152, True, False, 0),
                ("req_short", 49152, True, False, 0),
                ("req_fb", 49152, True, False, 0),
                ("req_fb_step", 49152, True, False, 0),
            )
            h_rw.cm_box[0] = cm_rw
            tbl_rw = np.stack([np.arange(i * 400, i * 400 + 385, dtype=np.int32) for i in range(4)])
            captured: dict[str, Any] = {}

            def _orig_rw(_: Any) -> None:
                h_rw.eng._recompute_fallback_req_ids.add("req_fb")
                for rid in ("req_rw", "req_short", "req_fb_step"):
                    h_rw.eng._prepare_shared_prefill_sources(
                        [self._LayerSrc((o_rw,), i) for i in range(4)],
                        0,
                        {"req_id": rid, "_shared_prefill_starts": [0], "_shared_prefill_ends": [49152]},
                    )
                req_rw = SimpleNamespace(
                    req_id="req_rw",
                    token_ids=list(range(53248)),
                    is_sparse_decode=False,
                    is_decode_window_save=False,
                )
                req_short = SimpleNamespace(
                    req_id="req_short",
                    token_ids=list(range(49152)),
                    is_sparse_decode=False,
                    is_decode_window_save=False,
                )
                req_fb = SimpleNamespace(
                    req_id="req_fb",
                    token_ids=list(range(53248)),
                    is_sparse_decode=False,
                    is_decode_window_save=False,
                )
                req_fb_step = SimpleNamespace(
                    req_id="req_fb_step",
                    token_ids=list(range(53248)),
                    is_sparse_decode=False,
                    is_decode_window_save=False,
                )
                captured["g0"] = h_rw.impl._prepare_layerwise_store_inputs(req_rw, None, 0)
                captured["g1"] = h_rw.impl._prepare_layerwise_store_inputs(req_rw, None, 1)
                captured["short"] = h_rw.impl._prepare_layerwise_store_inputs(req_short, None, 0)
                captured["fb"] = h_rw.impl._prepare_layerwise_store_inputs(req_fb, None, 0)
                h_rw.impl._step_recompute_fallback_req_ids = {"req_fb_step"}
                captured["fb_step"] = h_rw.impl._prepare_layerwise_store_inputs(req_fb_step, None, 0)

            h_rw.hook[0] = _orig_rw
            h_rw.src.begin_runner_step(
                cm_rw,
                ("req_rw", "req_short", "req_fb", "req_fb_step"),
                (49152, 49152, 49152, 49152),
                (0, 8, 16, 24, 32),
                (tbl_rw, tbl_rw),
            )
            h_rw.impl.start_load_kv(None)
            toks_0, _smap_0, smask_0, skip_0, skw_0, _win_0 = captured["g0"]
            self.assertEqual(len(toks_0), 53248)
            self.assertEqual(skip_0, 49152)
            self.assertFalse(bool(smask_0[:49152].any()))
            self.assertTrue(bool(smask_0[49152:].all()))
            self.assertEqual(skw_0["_prefill_skip_tokens"], 49152)
            self.assertEqual(skw_0["_unverified_prefix_skip_tokens"], 49152)
            self.assertEqual(captured["g1"][3], 0)
            self.assertTrue(bool(captured["g1"][2][:49152].all()))
            self.assertIsNone(captured["short"])
            self.assertEqual(captured["fb"][3], 0)
            self.assertTrue(bool(captured["fb"][2][:49152].all()))
            self.assertEqual(captured["fb_step"][3], 0)
            self.assertTrue(bool(captured["fb_step"][2][:49152].all()))

            st_rw = h_rw.src.step
            self.assertIsNotNone(st_rw)
            st_rw.finish(completion=torch.npu.Event())
            h_rw.src._prune_retired_steps()
            self._close_src(h_rw.src)
            self.assertIs(h_rw.impl._prepare_layerwise_store_inputs, _orig_prep)

    def test_dsa_cp_multi_row_plan_topk_buf_causality_masking(self) -> None:
        nop = self._nop
        z = lambda k: torch.zeros((8, 128, 1, k), dtype=torch.bfloat16)
        ow = {"l0": "l0", "l1": "l0"}
        caches = {n: (z(512), z(64)) for n in ow}
        built: list[torch.Tensor] = []
        loads: list[tuple[Any, ...]] = []
        pl = SimpleNamespace(
            built=built,
            loads=loads,
            build=lambda t, h, s, q: (
                built.append(t.clone()),
                built.append(t),
                (
                    torch.zeros((len(h), 128), dtype=torch.int32),
                    torch.tensor([[0, qv, qv, 0, 0, 0, 0, 0] for qv in q], dtype=torch.int32),
                ),
            )[2],
            load_mla=lambda *a: loads.append(a),
            release_quiescent=nop,
            close_quiescent=nop,
        )
        vc = _vcfg(
            par=SimpleNamespace(
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
            ),
            sched=SimpleNamespace(async_scheduling=True, max_num_batched_tokens=8192, max_num_seqs=8),
        )
        src = SimpleNamespace(step=None, _failed=False, close=nop)

        def _ev():
            return SimpleNamespace(record=nop, query=lambda: True, synchronize=nop)

        st = SimpleNamespace(
            requests=[
                lms.SelectedRequest("r0", 49152, 134, True, 49152),
                lms.SelectedRequest("r1", 49152, 3, True, 49152),
            ],
            _finished=False,
            _closed=False,
            page_bases=lambda n: torch.zeros((8, 2), dtype=torch.uint64),
            ready=lambda n: torch.npu.Event(),
            wait_main_slot_ready=nop,
            prefetch_main=lambda n, fn: fn(),
            wait_completion=nop,
            finish=lambda *, completion=None, failed=False: setattr(st, "_finished", True),
            try_retire=lambda: setattr(st, "_closed", True) or True,
        )
        src.step = st
        with _fake_npu():
            rt = psr.SelectedKVRuntime(
                SimpleNamespace(device="cpu", vllm_config=vc),
                owner_by_layer=ow,
                main_caches=caches,
                source=src,
                planner=pl,
                dsa_cp=True,
            )
            # TP=4, total=137, pad=512 (local_rows=128). Rank 1 owns [128, 137):
            # r0 has q_off0=128, q0=6 at starts[0]=0; r1 has q1=3 at starts[1]=6.
            cp_ctx = SimpleNamespace(local_start=128, local_end=137, local_end_with_pad=256, num_tokens_pad=512)
            metas = {
                ln: SimpleNamespace(
                    block_table=torch.zeros((8, 420), dtype=torch.int32),
                    num_actual_tokens=137,
                    block_size=128,
                    dsa_cp_context=cp_ctx,
                )
                for ln in ow
            }
            rt.bind_step(st, ("r0", "r1"), (49152, 49152), (0, 134, 137), metas)
            self.assertEqual(
                (rt._active_selected_indices, rt._query_offsets, rt.starts, rt.queries),
                ((0, 1), [128, 0], [0, 6], [6, 3]),
            )
            h0, h1, limit0 = 49152, 49152, 49152 + 134
            topk = torch.full((128, 1, skv.TOPK), -1, dtype=torch.int32)
            for row in range(6):
                topk[row, 0, :8] = torch.tensor(
                    [0, h0 - 1, h0, h0 + 128, h0 + 128 + row, limit0 - 1, limit0, -2], dtype=torch.int32
                )
            for row in range(6, 9):
                topk[row, 0, :7] = torch.tensor(
                    [h0, h0 + 128, limit0 - 1, h1 - 1, h1, h1 + (row - 6), limit0], dtype=torch.int32
                )
            topk[9:12, 0, 0] = 777
            topk_orig = topk.clone()
            plan_buf_ptr = rt._plan_topk_buf.data_ptr()
            rt.load("l0", topk, caches["l0"], metas["l0"])
            rt.prefetch_next("l0", metas["l0"])
            rt.load("l1", topk, caches["l1"], metas["l1"])
            rt.finish_step()
            rt.close()

            self.assertTrue(torch.equal(topk, topk_orig))
            self.assertEqual(built[1].data_ptr(), plan_buf_ptr)
            plan_snap = built[0]
            self.assertEqual(tuple(plan_snap.shape), (9, 1, skv.TOPK))
            for row in range(6):
                self.assertEqual(
                    plan_snap[row, 0, :8].tolist(),
                    [0, h0 - 1, -1, -1, -1, -1, limit0, -2],
                )
            for row in range(6, 9):
                self.assertTrue(torch.equal(plan_snap[row], topk_orig[row]))

    def test_dsa_cp_target_empty_to_draft_active_with_host_meta_ring(self) -> None:
        nop = self._nop
        z = lambda k: torch.zeros((8, 128, 1, k), dtype=torch.bfloat16)
        ow = {"l0": "l0", "l1": "l0", "l2": "l2"}
        caches = {n: (z(512), z(64)) for n in ow}
        vc = _vcfg(
            par=SimpleNamespace(
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
            ),
            sched=SimpleNamespace(async_scheduling=True, max_num_batched_tokens=8192, max_num_seqs=8),
        )
        built: list[torch.Tensor] = []
        loads: list[Any] = []
        pl = SimpleNamespace(
            built=built,
            loads=loads,
            build=lambda t, h, s, q: (
                built.append(t),
                (
                    torch.zeros((len(h), 128), dtype=torch.int32),
                    torch.tensor([[0, qv, qv, 0, 0, 0, 0, 0] for qv in q], dtype=torch.int32),
                ),
            )[1],
            load_mla=lambda *a: loads.append(a),
            release_quiescent=nop,
            close_quiescent=nop,
        )
        src = SimpleNamespace(step=None, _failed=False, close=nop)

        def _mk_step():
            s = SimpleNamespace(
                requests=[
                    lms.SelectedRequest("r0", 128, 4, False, 0),
                    lms.SelectedRequest("r1", 49152, 4, True, 49152),
                ],
                _finished=False,
                _closed=False,
                page_bases=lambda n: torch.zeros((8, 2), dtype=torch.uint64),
                ready=lambda n: torch.npu.Event(),
                wait_main_slot_ready=nop,
                prefetch_main=lambda n, fn: fn(),
                wait_completion=nop,
                finish=lambda *, completion=None, failed=False: setattr(s, "_finished", True),
                try_retire=lambda: setattr(s, "_closed", True) or True,
            )
            return s

        with _fake_npu():
            rt = psr.SelectedKVRuntime(
                SimpleNamespace(device="cpu", vllm_config=vc),
                owner_by_layer=ow,
                main_caches=caches,
                source=src,
                planner=pl,
                draft_layer_names=("l2",),
                dsa_cp=True,
            )
            cp_tgt = SimpleNamespace(local_start=2, local_end=4, local_end_with_pad=4, num_tokens_pad=8)
            cp_dr = SimpleNamespace(local_start=1, local_end=2, local_end_with_pad=2, num_tokens_pad=4)
            tgt_metas = {
                ln: SimpleNamespace(
                    block_table=torch.zeros((8, 420), dtype=torch.int32),
                    num_actual_tokens=8,
                    block_size=128,
                    dsa_cp_context=cp_tgt,
                )
                for ln in ow
            }
            dr_metas = {
                "l2": SimpleNamespace(
                    block_table=torch.zeros((8, 420), dtype=torch.int32),
                    num_actual_tokens=4,
                    block_size=128,
                    dsa_cp_context=cp_dr,
                )
            }
            for step_idx in range(2):
                src.step = st = _mk_step()
                rt.bind_step(st, ("r0", "r1"), (128, 49152), (0, 4, 8), tgt_metas)
                self.assertEqual((rt._active_selected_indices, rt.request_metadata), ((), None))
                topk_tgt = torch.zeros((2, 1, skv.TOPK), dtype=torch.int32)
                rt.load("l0", topk_tgt, caches["l0"], tgt_metas["l0"])
                rt.prefetch_next("l0", tgt_metas["l0"])
                rt.load("l1", topk_tgt, caches["l1"], tgt_metas["l1"])
                self.assertEqual((len(built), len(loads)), (step_idx, step_idx))
                rt.defer_to_draft_step()

                rt.bind_draft_step(
                    dr_metas,
                    SimpleNamespace(query_start_loc_cpu=[0, 1, 4], seq_lens_cpu=[129, 49155]),
                )
                slot = step_idx % 2
                self.assertEqual((rt._active_selected_indices, rt._cur_meta_slot), ((1,), slot))
                self.assertIs(rt._host_meta, rt._host_meta_ring[slot])
                self.assertEqual(rt.request_metadata.tolist(), [[1, 49152, 0, 1]])
                self.assertEqual(rt._host_meta_ring[slot][1, :1].tolist(), [[1, 49152, 0, 1]])
                self.assertIsNotNone(rt._meta_upload_events[slot])
                rt.load("l2", torch.zeros((1, 1, skv.TOPK), dtype=torch.int32), caches["l2"], dr_metas["l2"])
                rt.finish_step()
                self.assertEqual((len(built), len(loads)), (step_idx + 1, step_idx + 1))
                rt._retire_previous()
                self.assertTrue(st._closed)
            rt.close()

    def test_dsa_cp_4rank_lockstep_inflight2_m2_vec_omit_clamp_and_admission_retirement(self) -> None:
        cfg, _CtrlEv, _build_env, _run_step = self._pipe_setup()
        ev_done = {"s0": False, "s1": False}
        cur_step_key: list[str | None] = [None]

        class _StepEv:
            def __init__(self, key: str | None = None) -> None:
                self._k = cur_step_key[0] if key is None else key
                self.recorded_on: int | None = None

            def record(self, stream=None) -> None:
                self.recorded_on = int((stream or torch.npu.current_stream()).npu_stream)

            def query(self) -> bool:
                return True if self._k is None else bool(ev_done[self._k])

            def synchronize(self) -> None:
                if self._k is not None:
                    ev_done[self._k] = True

        ekw = {"VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": True, "VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS": 2}
        with _fake_npu(), patch.object(torch.npu, "Event", _StepEv), patch.multiple(envs_ascend, **ekw):
            ranks = []
            for rank in range(4):
                e = _build_env()
                for m in e.rn.compilation_config.static_forward_context.values():
                    m.impl.enable_dsa_cp = True
                    m.impl.tp_size = 4
                e.rn.enable_dsa_cp = True
                e.rn.vllm_config = _vcfg(
                    par=SimpleNamespace(
                        tensor_parallel_size=4,
                        pipeline_parallel_size=1,
                        prefill_context_parallel_size=1,
                        decode_context_parallel_size=1,
                    ),
                    sched=SimpleNamespace(async_scheduling=True, max_num_batched_tokens=8192, max_num_seqs=8),
                )
                btg_calls: list[str] = []

                def _btg(*_a: Any, _calls: list[str] = btg_calls, **k: Any) -> Any:
                    _calls.append(str(k.get("req_id", "")))
                    while True:
                        yield

                def _orig_prep(req: Any, _save_spec: Any, _kv_group: int, *_a: Any, **_kw: Any) -> tuple[Any, ...]:
                    toks = list(req.token_ids)
                    return (
                        toks,
                        torch.arange(len(toks), dtype=torch.int64),
                        torch.ones(len(toks), dtype=torch.bool),
                        0,
                        {},
                        False,
                    )

                e.gc.batched_to_gpu = _btg
                e.gc.prepare_layerwise_prefill_group0_skip = self._nop
                e.eng._should_use_shared_layerwise_retrieve = lambda g: g == 0
                e.impl._should_skip_group0_prefill_dma = self._nop
                e.impl._prepare_layerwise_store_inputs = _orig_prep
                e.impl.kv_role = "kv_producer"
                e.impl._layerwise_save_storers = {}
                e.btg_calls = btg_calls

                with (
                    patch.dict(sys.modules, e.mods),
                    patch.object(ctypes, "CDLL", side_effect=[e.fl, e.ml]),
                    patch.object(torch, "ops", e.fops),
                ):
                    skv.reserve_selected_kv_workspace(e.wk, 2 << 30)
                    skv.initialize_selected_kv(e.rn, None, e.kv_c)
                ranks.append(e)

            try:
                # Step 0 (inflight slot 0): r_hit (49152 hist, 200 query) straddles Rank 0/1;
                # r_miss (selected=False, 128 hist, 312 query) spans Rank 1..3. Total=512, TP=4 -> 128/rank.
                o_sh0 = self._sh_obj(0x9000, 49152, 3)
                o_sh0.ls_calls = []
                o_sh0.layer_size_bytes = lambda lid, _o=o_sh0: (_o.ls_calls.append(lid), 49152 * skv.MAIN_ROW_BYTES)[1]
                ptrs0 = tuple([o_sh0.layer_data_ptr(lid) + 0x100000] for lid in range(3))
                tbl0 = np.stack([self._ar(0, 390, d2=False), self._ar(400, 790, d2=False)])
                skip_maps, omit_maps, clamp_results = [], [], []
                steps0 = []

                for rank, e in enumerate(ranks):
                    rt, src = e.rn.selected_kv_runtime, e.rn.selected_kv_runtime.source
                    self._set_batch(e.rn, e.tbl_box, ["r_hit", "r_miss"], [0, 200, 512], tbl0)
                    e.rn.input_batch.num_computed_tokens_cpu = np.array([49152, 128], dtype=np.int32)
                    cm0 = self._cm(("r_hit", 49152, True, False, 0), ("r_miss", 128, True, False, 0))

                    pages0 = (o_sh0,)

                    def _hook0(_: Any, _e: Any = e, _r: int = rank, _p0: tuple[Any, ...] = pages0) -> None:
                        skip_maps.append(frozenset(_e.impl._selected_skip_group0_dma_req_ids))
                        omit_maps.append(dict(_e.impl._selected_group0_omit_end_tokens))
                        _e.eng._prepare_shared_prefill_sources(
                            [self._LayerSrc(_p0, i) for i in range(3)],
                            0,
                            {
                                "req_id": "r_hit",
                                "_shared_prefill_starts": [0],
                                "_shared_prefill_ends": [49152],
                                "cached_chunk_dev_ptrs": ptrs0,
                                "_reuse_shared_dense_prefix": True,
                            },
                        )
                        next(_e.gc.batched_to_gpu([0], [312], req_id="r_miss", kv_group=0))
                        if _r == 0:
                            req0 = SimpleNamespace(
                                req_id="r_hit",
                                token_ids=list(range(49152 + 200)),
                                is_sparse_decode=False,
                                is_decode_window_save=False,
                            )
                            clamp_results.append(_e.impl._prepare_layerwise_store_inputs(req0, None, 0))

                    e.hook[0] = _hook0
                    cp_ctx = SimpleNamespace(
                        local_start=128 * rank,
                        local_end=128 * (rank + 1),
                        local_end_with_pad=128 * (rank + 1),
                        num_tokens_pad=512,
                    )
                    am0 = {
                        ln: SimpleNamespace(
                            block_table=torch.zeros((8, 420), dtype=torch.int32),
                            num_actual_tokens=512,
                            block_size=128,
                            dsa_cp_context=cp_ctx,
                        )
                        for ln in e.layers
                    }
                    self._prep_cm(e.rn, e.lmc_c, cm0, am0)
                    st0 = src.step
                    cur_step_key[0] = "s0"
                    with skv.selected_kv_forward(
                        e.rn, SimpleNamespace(kv_connector_metadata=cm0), am0, defer_finish=True
                    ) as b:
                        if rank in (0, 1):
                            self.assertIn(0, rt._active_selected_indices)
                        else:
                            self.assertEqual(rt._active_selected_indices, ())
                            self.assertIsNone(rt.request_metadata)
                        tk = torch.zeros((128, 1, 2048), dtype=torch.int32)
                        b.load(e.layers[0], tk, e.kv_c[e.layers[0]], am0[e.layers[0]])
                        b.prefetch_next(e.layers[0], am0[e.layers[0]])
                        b.load(e.layers[1], tk, e.kv_c[e.layers[1]], am0[e.layers[1]])
                    # Total draft tokens = 2 -> Ranks 0,1 active (1 token each), Ranks 2,3 empty
                    dr_ctx = SimpleNamespace(
                        local_start=rank,
                        local_end=min(rank + 1, 2),
                        local_end_with_pad=rank + 1,
                        num_tokens_pad=4,
                    )
                    dm0 = {
                        e.layers[2]: SimpleNamespace(
                            block_table=am0[e.layers[2]].block_table,
                            num_actual_tokens=2,
                            block_size=128,
                            dsa_cp_context=dr_ctx,
                        )
                    }
                    skv.bind_selected_kv_draft_metadata(
                        e.rn, dm0, SimpleNamespace(query_start_loc_cpu=[0, 1, 2], seq_lens_cpu=[49153, 129])
                    )
                    with skv.selected_kv_draft_forward(e.rn):
                        tk_d = torch.zeros((1, 1, 2048), dtype=torch.int32)
                        dm0[e.layers[2]].selected_kv.load(e.layers[2], tk_d, e.kv_c[e.layers[2]], dm0[e.layers[2]])
                    cur_step_key[0] = None
                    self.assertTrue(st0._finished)
                    self.assertFalse(st0._closed)
                    steps0.append(st0)
                    self.assertEqual(src.unretired_selected_request_ids(), frozenset({"r_hit"}))

                self.assertEqual(skip_maps, [frozenset({"r_hit"})] * 4)
                self.assertEqual(omit_maps, [{"r_hit": 49152}] * 4)
                self.assertTrue(all(e.btg_calls == ["r_miss"] for e in ranks))
                self.assertEqual(o_sh0.ls_calls, [0] * 12)  # 4 ranks * (1 row-dva check + 2 banks), no lid 1/2 calls
                self.assertEqual(clamp_results[0][3], 49152)
                self.assertFalse(bool(clamp_results[0][2][:49152].any()))

                # Step 1 (inflight slot 1 while Step 0 still unretired!):
                # r_s1 (49152 hist, 64 q) + r_ns1 (128 hist, 192 q), total=256, TP=4.
                # Only Rank 0 owns r_s1; Ranks 1..3 have empty selected slice.
                o_sh1 = self._sh_obj(0xB000, 49152, 3)
                tbl1 = np.stack([self._ar(0, 390, d2=False), self._ar(400, 790, d2=False)])
                steps1 = []
                for rank, e in enumerate(ranks):
                    rt, src = e.rn.selected_kv_runtime, e.rn.selected_kv_runtime.source
                    self._set_batch(e.rn, e.tbl_box, ["r_s1", "r_ns1"], [0, 64, 256], tbl1)
                    e.rn.input_batch.num_computed_tokens_cpu = np.array([49152, 128], dtype=np.int32)
                    cm1 = self._cm(("r_s1", 49152, True, False, 0), ("r_ns1", 128, True, False, 0))
                    e.hook[0] = lambda _arg, _e=e: (
                        _e.eng._prepare_shared_prefill_sources(
                            [self._LayerSrc((o_sh1,), i) for i in range(3)],
                            0,
                            {"req_id": "r_s1", "_shared_prefill_starts": [0], "_shared_prefill_ends": [49152]},
                        ),
                        next(_e.gc.batched_to_gpu([0], [192], req_id="r_ns1", kv_group=0)),
                    )
                    cp1 = SimpleNamespace(
                        local_start=64 * rank,
                        local_end=64 * (rank + 1),
                        local_end_with_pad=64 * (rank + 1),
                        num_tokens_pad=256,
                    )
                    am1 = {
                        ln: SimpleNamespace(
                            block_table=torch.zeros((8, 420), dtype=torch.int32),
                            num_actual_tokens=256,
                            block_size=128,
                            dsa_cp_context=cp1,
                        )
                        for ln in e.layers
                    }
                    self._prep_cm(e.rn, e.lmc_c, cm1, am1)
                    st1 = src.step
                    self.assertEqual(steps0[rank]._host_slot, 0)
                    self.assertEqual(st1._host_slot, 1)
                    self.assertEqual(src.unretired_selected_request_ids(), frozenset({"r_hit", "r_s1"}))
                    cur_step_key[0] = "s1"
                    with skv.selected_kv_forward(
                        e.rn, SimpleNamespace(kv_connector_metadata=cm1), am1, defer_finish=True
                    ) as b:
                        tk1 = torch.zeros((64, 1, 2048), dtype=torch.int32)
                        b.load(e.layers[0], tk1, e.kv_c[e.layers[0]], am1[e.layers[0]])
                        b.prefetch_next(e.layers[0], am1[e.layers[0]])
                        b.load(e.layers[1], tk1, e.kv_c[e.layers[1]], am1[e.layers[1]])
                    dr1 = SimpleNamespace(
                        local_start=rank,
                        local_end=min(rank + 1, 2),
                        local_end_with_pad=rank + 1,
                        num_tokens_pad=4,
                    )
                    dm1 = {
                        e.layers[2]: SimpleNamespace(
                            block_table=am1[e.layers[2]].block_table,
                            num_actual_tokens=2,
                            block_size=128,
                            dsa_cp_context=dr1,
                        )
                    }
                    skv.bind_selected_kv_draft_metadata(
                        e.rn, dm1, SimpleNamespace(query_start_loc_cpu=[0, 1, 2], seq_lens_cpu=[49153, 129])
                    )
                    with skv.selected_kv_draft_forward(e.rn):
                        dm1[e.layers[2]].selected_kv.load(
                            e.layers[2],
                            torch.zeros((1, 1, 2048), dtype=torch.int32),
                            e.kv_c[e.layers[2]],
                            dm1[e.layers[2]],
                        )
                    cur_step_key[0] = None
                    self.assertTrue(st1._finished)
                    self.assertFalse(st1._closed)
                    steps1.append(st1)

                # Retire Step 0 only -> Step 1 stays unretired
                ev_done["s0"] = True
                for rank, e in enumerate(ranks):
                    e.eng.release_shared_cpu_sparse_request("r_hit")
                    e.gc.release_layerwise_prefill_dma_cache("r_hit")
                    self.assertEqual(
                        e.rn.selected_kv_runtime.source.unretired_selected_request_ids(), frozenset({"r_s1"})
                    )

                # On Ranks 0..2 finish retiring Step 1 cleanly; on Rank 3 simulate quarantined failure
                ev_done["s1"] = True
                for rank in range(3):
                    e = ranks[rank]
                    e.eng.release_shared_cpu_sparse_request("r_s1")
                    e.gc.release_layerwise_prefill_dma_cache("r_s1")
                    self.assertEqual(e.rn.selected_kv_runtime.source.unretired_selected_request_ids(), frozenset())
                    self.assertTrue(steps1[rank]._closed)
                ranks[3].rn.selected_kv_runtime.fail(steps1[3])
                self.assertTrue(ranks[3].rn.selected_kv_runtime.source._failed)
                self.assertFalse(steps1[3]._closed)
                self.assertEqual(
                    ranks[3].rn.selected_kv_runtime.source.unretired_selected_request_ids(), frozenset({"r_s1"})
                )
            finally:
                for rank, e in enumerate(ranks):
                    src = e.rn.selected_kv_runtime.source
                    try:
                        if rank < 3:
                            skv.close_selected_kv_worker(SimpleNamespace(model_runner=e.rn))
                        else:
                            self.assertRaisesRegex(RuntimeError, "retained", e.rn.selected_kv_runtime.close)
                            self.assertRaisesRegex(RuntimeError, "retained", src.close)
                    finally:
                        for fn in reversed(src._restorers):
                            fn()

    def test_dsa_cp_unretired_retains_inflight_step0_when_step1_start_load_kv_raises(self) -> None:
        _, _, _build_env, _ = self._pipe_setup()
        ev_done = {"s0": False}
        cur_step_key: list[str | None] = [None]

        class _StepEv:
            def __init__(self, key: str | None = None) -> None:
                self._k = cur_step_key[0] if key is None else key
                self.recorded_on: int | None = None

            def record(self, stream=None) -> None:
                self.recorded_on = int((stream or torch.npu.current_stream()).npu_stream)

            def query(self) -> bool:
                return True if self._k is None else bool(ev_done[self._k])

            def synchronize(self) -> None:
                if self._k is not None:
                    ev_done[self._k] = True

        ekw = {"VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": True, "VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS": 2}
        with _fake_npu(), patch.object(torch.npu, "Event", _StepEv), patch.multiple(envs_ascend, **ekw):
            e = _build_env()
            for m in e.rn.compilation_config.static_forward_context.values():
                m.impl.enable_dsa_cp = True
                m.impl.tp_size = 4
            e.rn.enable_dsa_cp = True
            e.rn.vllm_config = _vcfg(
                par=SimpleNamespace(
                    tensor_parallel_size=4,
                    pipeline_parallel_size=1,
                    prefill_context_parallel_size=1,
                    decode_context_parallel_size=1,
                ),
                sched=SimpleNamespace(async_scheduling=True, max_num_batched_tokens=8192, max_num_seqs=8),
            )
            with (
                patch.dict(sys.modules, e.mods),
                patch.object(ctypes, "CDLL", side_effect=[e.fl, e.ml]),
                patch.object(torch, "ops", e.fops),
            ):
                skv.reserve_selected_kv_workspace(e.wk, 2 << 30)
                skv.initialize_selected_kv(e.rn, None, e.kv_c)

            src = e.rn.selected_kv_runtime.source
            try:
                # Step 0 ("r_s0"): executed and still unclosed (completion query() == False).
                o_sh0 = self._sh_obj(0x9000, 49152, 3)
                tbl0 = np.stack([self._ar(0, 390, d2=False)])
                self._set_batch(e.rn, e.tbl_box, ["r_s0"], [0, 128], tbl0)
                cm0 = self._cm(("r_s0", 49152, True, False, 0))
                e.hook[0] = lambda _arg: e.eng._prepare_shared_prefill_sources(
                    [self._LayerSrc((o_sh0,), i) for i in range(3)],
                    0,
                    {"req_id": "r_s0", "_shared_prefill_starts": [0], "_shared_prefill_ends": [49152]},
                )
                cp_ctx = SimpleNamespace(local_start=0, local_end=32, local_end_with_pad=32, num_tokens_pad=128)
                am0 = {
                    ln: SimpleNamespace(
                        block_table=torch.zeros((8, 420), dtype=torch.int32),
                        num_actual_tokens=128,
                        block_size=128,
                        dsa_cp_context=cp_ctx,
                    )
                    for ln in e.layers
                }
                self._prep_cm(e.rn, e.lmc_c, cm0, am0)
                st0 = src.step
                cur_step_key[0] = "s0"
                with skv.selected_kv_forward(
                    e.rn, SimpleNamespace(kv_connector_metadata=cm0), am0, defer_finish=True
                ) as b:
                    tk0 = torch.zeros((32, 1, 2048), dtype=torch.int32)
                    b.load(e.layers[0], tk0, e.kv_c[e.layers[0]], am0[e.layers[0]])
                    b.prefetch_next(e.layers[0], am0[e.layers[0]])
                    b.load(e.layers[1], tk0, e.kv_c[e.layers[1]], am0[e.layers[1]])
                dm0 = {
                    e.layers[2]: SimpleNamespace(
                        block_table=am0[e.layers[2]].block_table,
                        num_actual_tokens=4,
                        block_size=128,
                        dsa_cp_context=SimpleNamespace(
                            local_start=0, local_end=1, local_end_with_pad=1, num_tokens_pad=4
                        ),
                    )
                }
                skv.bind_selected_kv_draft_metadata(
                    e.rn, dm0, SimpleNamespace(query_start_loc_cpu=[0, 4], seq_lens_cpu=[49156])
                )
                with skv.selected_kv_draft_forward(e.rn):
                    dm0[e.layers[2]].selected_kv.load(
                        e.layers[2],
                        torch.zeros((1, 1, 2048), dtype=torch.int32),
                        e.kv_c[e.layers[2]],
                        dm0[e.layers[2]],
                    )
                cur_step_key[0] = None
                self.assertTrue(st0._finished)
                self.assertFalse(st0._closed)
                self.assertEqual(src.unretired_selected_request_ids(), frozenset({"r_s0"}))

                # Step 1 ("r_s1"): raises in _start_load_kv -> source._failed becomes True,
                # yet both unclosed Step 0 ("r_s0") and appended Step 1 ("r_s1") remain retained.
                o_sh1 = self._sh_obj(0xB000, 49152, 3)
                tbl1 = np.stack([self._ar(0, 390, d2=False)])
                self._set_batch(e.rn, e.tbl_box, ["r_s1"], [0, 128], tbl1)
                cm1 = self._cm(("r_s1", 49152, True, False, 0))

                def _fail_hook(_arg: Any) -> None:
                    e.eng._prepare_shared_prefill_sources(
                        [self._LayerSrc((o_sh1,), i) for i in range(3)],
                        0,
                        {"req_id": "r_s1", "_shared_prefill_starts": [0], "_shared_prefill_ends": [49152]},
                    )
                    raise RuntimeError("injected _start_load_kv failure")

                e.hook[0] = _fail_hook
                with self.assertRaisesRegex(RuntimeError, "injected _start_load_kv failure"):
                    self._prep_cm(e.rn, e.lmc_c, cm1, am0)
                self.assertTrue(src._failed)
                self.assertFalse(st0._closed)
                self.assertEqual(src.unretired_selected_request_ids(), frozenset({"r_s0", "r_s1"}))
            finally:
                try:
                    self.assertRaisesRegex(RuntimeError, "retained", src.close)
                finally:
                    for fn in reversed(src._restorers):
                        fn()

    def test_defer_release_detach_prevents_req_id_reuse_clobber_both_modes(self) -> None:
        _, _CtrlEv, _build_env, _run_step = self._pipe_setup()
        for use_detach in (True, False):
            with _fake_npu(), patch.object(torch.npu, "Event", _CtrlEv):
                e = _build_env()
                e.eng._shared_cpu_request_leases = {}

                def _rel_cpu(rid: str, eng: Any = e.eng) -> None:
                    lease = eng._shared_cpu_request_leases.pop(rid, None)
                    if lease is not None:
                        lease.close()

                e.eng.release_shared_cpu_sparse_request = _rel_cpu
                if use_detach:

                    def _det_cpu(rid: str, eng: Any = e.eng) -> Any:
                        lease = eng._shared_cpu_request_leases.pop(rid, None)
                        if lease is None:
                            return None
                        lease.active = False
                        dseq = getattr(eng, "_det_seq", 0) + 1
                        eng._det_seq = dseq
                        dk = f"{rid}#detached#{dseq}"
                        eng._shared_cpu_request_leases[dk] = lease
                        return lambda: _rel_cpu(dk)

                    e.eng.detach_shared_cpu_sparse_request = _det_cpu

                e.gc._layerwise_prefill_load_owners = {}
                e.gc._prefill_dma_bound_loads = {}

                def _rel_dma(rid: str, gc: Any = e.gc) -> None:
                    owns = gc._layerwise_prefill_load_owners.pop(rid, None)
                    if owns:
                        for own in owns.values():
                            if own.is_valid():
                                own.ref_count_down()
                    gc._prefill_dma_bound_loads.pop(rid, None)

                e.gc.release_layerwise_prefill_dma_cache = _rel_dma
                if use_detach:

                    def _det_dma(rid: str, gc: Any = e.gc) -> Any:
                        gc._prefill_dma_bound_loads.pop(rid, None)
                        owns = gc._layerwise_prefill_load_owners.pop(rid, None)
                        if owns is None:
                            return None
                        dseq = getattr(gc, "_det_seq", 0) + 1
                        gc._det_seq = dseq
                        dk = f"{rid}#detached#{dseq}"
                        gc._layerwise_prefill_load_owners[dk] = owns
                        return lambda: _rel_dma(dk)

                    e.gc.detach_layerwise_prefill_dma_cache = _det_dma

                def _mk_lease() -> Any:
                    lease: Any = SimpleNamespace(active=True, closed=0)
                    lease.close = lambda: setattr(lease, "closed", lease.closed + 1)
                    return lease

                def _mk_owner() -> Any:
                    own: Any = SimpleNamespace(ref_count=1)
                    own.is_valid = lambda: own.ref_count > 0
                    own.ref_count_down = lambda: setattr(own, "ref_count", own.ref_count - 1)
                    return own

                with self._init_rt(e):
                    # (a) CPU lease overlap on same req_id ("req_1")
                    st_cpu, ev_cpu, _ = _run_step(e, "req_1", bb=0, hold=True)
                    lease_prefill, lease_decode = _mk_lease(), _mk_lease()
                    e.eng._shared_cpu_request_leases["req_1"] = lease_prefill
                    e.eng.release_shared_cpu_sparse_request("req_1")
                    e.eng._shared_cpu_request_leases["req_1"] = lease_decode
                    self.assertEqual((lease_prefill.closed, lease_prefill.active), (0, False))
                    self.assertEqual(lease_decode.closed, 0)
                    self.assertIs(e.eng._shared_cpu_request_leases["req_1"], lease_decode)
                    ev_cpu.done = True
                    self.assertTrue(st_cpu.try_retire())
                    self.assertEqual((lease_prefill.closed, lease_decode.closed), (1, 0))
                    self.assertIs(e.eng._shared_cpu_request_leases["req_1"], lease_decode)

                    # (b) DMA owner/bound overlap on same req_id ("req_d")
                    st_dma, ev_dma, _ = _run_step(e, "req_d", bb=130, hold=True)
                    owner_0, owner_1 = _mk_owner(), _mk_owner()
                    e.gc._layerwise_prefill_load_owners["req_d"] = {"l0": owner_0}
                    e.gc._prefill_dma_bound_loads["req_d"] = ("bound_0",)
                    e.gc.release_layerwise_prefill_dma_cache("req_d")
                    self.assertNotIn("req_d", e.gc._prefill_dma_bound_loads)
                    e.gc._layerwise_prefill_load_owners["req_d"] = {"l0": owner_1}
                    e.gc._prefill_dma_bound_loads["req_d"] = ("bound_1",)
                    self.assertEqual((owner_0.ref_count, owner_1.ref_count), (1, 1))
                    ev_dma.done = True
                    self.assertTrue(st_dma.try_retire())
                    self.assertEqual((owner_0.ref_count, owner_1.ref_count), (0, 1))
                    self.assertIs(e.gc._layerwise_prefill_load_owners["req_d"]["l0"], owner_1)
                    self.assertEqual(e.gc._prefill_dma_bound_loads["req_d"], ("bound_1",))

    def test_npu_worker_shutdown_fault_isolation_and_storage_guard(self) -> None:
        worker_path = ROOT / "vllm_ascend/worker/worker.py"
        tree = ast.parse(worker_path.read_text(encoding="utf-8"))
        worker_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUWorker")
        shutdown_fn = next(n for n in worker_cls.body if isinstance(n, ast.FunctionDef) and n.name == "shutdown")

        calls: list[str] = []
        fail_state: list[str | None] = [None]

        class _Base:
            def shutdown(self) -> None:
                calls.append("super_shutdown")
                if fail_state[0] == "super_shutdown":
                    raise RuntimeError("super_shutdown_boom")

        cls_ast = ast.ClassDef(
            name="NPUWorker",
            bases=[ast.Name(id="_Base", ctx=ast.Load())],
            keywords=[],
            body=[shutdown_fn],
            decorator_list=[],
        )
        ns: dict[str, Any] = {"_Base": _Base}
        mod_ast = ast.fix_missing_locations(ast.Module(body=[cls_ast], type_ignores=[]))
        exec(compile(mod_ast, str(worker_path), "exec"), ns)
        dummy_worker_cls = ns["NPUWorker"]

        def _run_case(*, fail_on: str | None) -> tuple[list[str], BaseException | None]:
            calls.clear()
            fail_state[0] = fail_on

            class _Storage:
                def close(self) -> None:
                    calls.append("storage_close")

            w = dummy_worker_cls()
            w._decoder_storage_bootstrap = _Storage()

            def _rel_graph() -> None:
                calls.append("release_graph")
                if fail_on == "release_graph":
                    raise RuntimeError("release_graph_boom")

            w.release_sfa_graph_resources = _rel_graph

            def _close_skv(target: Any) -> None:
                calls.append("close_skv")
                if fail_on == "close_skv":
                    raise RuntimeError("close_skv_boom")

            caught: BaseException | None = None
            with patch.object(skv, "close_selected_kv_worker", side_effect=_close_skv):
                try:
                    w.shutdown()
                except BaseException as exc:
                    caught = exc
            return list(calls), caught

        calls_ok, err_ok = _run_case(fail_on=None)
        self.assertIsNone(err_ok)
        self.assertEqual(calls_ok, ["close_skv", "release_graph", "super_shutdown", "storage_close"])

        calls_skv, err_skv = _run_case(fail_on="close_skv")
        self.assertIsInstance(err_skv, RuntimeError)
        self.assertIn("close_skv_boom", str(err_skv))
        self.assertEqual(calls_skv, ["close_skv", "release_graph", "super_shutdown", "storage_close"])

        calls_grp, err_grp = _run_case(fail_on="release_graph")
        self.assertIsInstance(err_grp, RuntimeError)
        self.assertIn("release_graph_boom", str(err_grp))
        self.assertEqual(calls_grp, ["close_skv", "release_graph", "super_shutdown"])

        calls_sup, err_sup = _run_case(fail_on="super_shutdown")
        self.assertIsInstance(err_sup, RuntimeError)
        self.assertIn("super_shutdown_boom", str(err_sup))
        self.assertEqual(calls_sup, ["close_skv", "release_graph", "super_shutdown"])


if __name__ == "__main__":
    unittest.main()
