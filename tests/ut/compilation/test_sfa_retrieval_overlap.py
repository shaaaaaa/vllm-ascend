# SPDX-License-Identifier: Apache-2.0
"""CPU ordering/layout contracts; native capture has separate NPU coverage."""
import ast
import gc
import symtable
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from sfa_test_support import extract, load_module
from test_sfa_full_graph import graph_module as graph_fixture

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/attention/sfa_v1.py"


def test_runner_constructor_reads_module_environment_without_local_shadowing():
    source = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    table = symtable.symtable(source.read_text(encoding="utf-8"), str(source), "exec")
    runner = next(t for t in table.get_children() if t.get_name() == "NPUModelRunner")
    constructor = next(t for t in runner.get_children() if t.get_name() == "__init__")
    environment = constructor.lookup("envs_ascend")
    # Inspect the ENTIRE function: a later import makes even earlier reads local.
    assert environment.is_global() and not environment.is_local()
    assert table.lookup("envs_ascend").is_imported()


@pytest.fixture
def graph_module(monkeypatch):
    return graph_fixture.__wrapped__(monkeypatch)


@pytest.fixture
def runtime(monkeypatch):
    log = []

    class Event:
        def __init__(self, enable_timing):
            assert not enable_timing
            self.recorded = False

        def record(self, stream):
            self.recorded = True
            log.append(("record", stream.name))

    class Stream:
        def __init__(self, device=None, name="transfer"):
            self.name = name

        def wait_event(self, event):
            assert event.recorded, "waiting on an unrecorded event"
            log.append(("wait", self.name))

    main = Stream(name="main")
    current = [main]

    @contextmanager
    def switch(stream):
        previous = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch, "npu", NS(Event=Event, Stream=Stream, stream=switch,
                                        current_stream=lambda: current[0]), raising=False)
    module = load_module(ROOT / "vllm_ascend/compilation/sfa_retrieval_overlap.py", "overlap_test", monkeypatch)
    monkeypatch.setattr(module, "CaptureProbe", Mock(return_value=NS(verify=Mock())))
    monkeypatch.setattr(torch.ops._C_ascend, "batch_matmul_transpose", Mock(), raising=False)
    return module, log


@dataclass(frozen=True)
class Key:
    request_capacity: int = 2
    token_capacity: int = 4
    query_profile: str = "bounded"


def topology(lengths=(4,), dtype=torch.bfloat16):
    groups, impls, caches = [], {}, {}
    for g, length in enumerate(lengths):
        group = NS(members=tuple(f"g{g}l{i}" for i in range(length)), active=True)
        groups.append(group)
        for i, name in enumerate(group.members):
            caches[name] = (torch.empty((2, 8, 1, 4), dtype=dtype), torch.empty((2, 8, 1, 2), dtype=dtype))
            impls[name] = NS(skip_topk=i > 0, shared_resident_plan=group,
                             _full_graph_transfers={2: NS(load=Mock(), request_capacity=2)},
                             _staged_sfa_bridge_buffers=(torch.empty(1, dtype=dtype),),
                             _staged_sfa_capture_state=NS(runtime=(None, caches[name])))
    return groups, impls, caches


@pytest.mark.parametrize("lengths", [(1,), (2,), (4,), (2, 3)])
def test_edges_never_cross_groups_and_keep_one_stream(runtime, lengths):
    module, log = runtime
    groups, impls, caches = topology(lengths)
    owner = module.SharedRetrievalOverlap()
    owner.configure(tuple(impls.items()), groups, caches)
    assert owner.stream is None
    for key in (Key(), Key(query_profile="fixed")):
        owner.prepare(key, tuple(impls.items()), groups, 1024)
        incoming, outgoing = owner.edges(key)
        assert len(incoming) == len(outgoing) == sum(n - 1 for n in lengths)
        for group in groups:
            assert group.members[0] not in incoming
            assert group.members[-1] not in outgoing
            for before, after in zip(group.members, group.members[1:]):
                assert outgoing[before] is incoming[after]
                assert incoming[after].stream is owner.stream
    module.CaptureProbe.assert_called_once() if owner.pairs else module.CaptureProbe.assert_not_called()
    previous = list(log)
    owner.prepare(Key(), tuple(impls.items()), groups, 1024)
    assert log == previous  # No per-key reinitialization.


@pytest.mark.parametrize("dtype,tokens", [(torch.float32, 4), (torch.bfloat16, 2048)])
def test_ineligible_paths_have_no_stream_or_probe(runtime, dtype, tokens):
    module, log = runtime
    groups, impls, caches = topology(dtype=dtype)
    owner = module.SharedRetrievalOverlap()
    owner.configure(tuple(impls.items()), groups, caches)
    key = Key(token_capacity=tokens)
    owner.prepare(key, tuple(impls.items()), groups, 1024)
    assert owner.edges(key) == ({}, {})
    assert owner.stream is None and not log
    module.CaptureProbe.assert_not_called()


def test_layout_accepts_disjoint_views_and_rejects_real_alias(runtime):
    module, _ = runtime
    backing = torch.zeros(64)
    caches = {"producer": (backing[:16], backing[16:32]), "consumer": (backing[32:48], backing[48:])}
    module.validate_destinations(caches, {"consumer"})
    caches["consumer"] = (backing[30:46], backing[48:])
    with pytest.raises(ValueError, match="ranges overlap"):
        module.validate_destinations(caches, {"consumer"})
    caches["consumer"] = (backing[32:48:2], backing[48:])
    with pytest.raises(ValueError, match="contiguous"):
        module.validate_destinations(caches, {"consumer"})


@pytest.mark.parametrize("fault", ["missing", "destination", "probe"])
def test_setup_failure_cannot_be_reused(runtime, fault):
    module, _ = runtime
    groups, impls, caches = topology()
    owner = module.SharedRetrievalOverlap()
    owner.configure(tuple(impls.items()), groups, caches)
    if fault == "missing":
        impls["g0l1"]._full_graph_transfers.clear()
    elif fault == "destination":
        impls["g0l1"]._staged_sfa_capture_state.runtime = (None, tuple(t.clone() for t in caches["g0l1"]))
    else:
        module.CaptureProbe.return_value.verify.side_effect = RuntimeError("native capture failed")
    with pytest.raises(RuntimeError):
        owner.prepare(Key(), tuple(impls.items()), groups, 1024)
    assert owner.failed
    if fault == "probe":
        assert owner.probe is module.CaptureProbe.return_value
    with pytest.raises(RuntimeError, match="restart"):
        owner.prepare(Key(), tuple(impls.items()), groups, 1024)


def test_actual_query_dtype_controls_cube_eligibility(runtime):
    module, log = runtime
    groups, impls, caches = topology()
    for impl in impls.values():
        impl._staged_sfa_bridge_buffers = (torch.empty(1, dtype=torch.float32),)
    owner = module.SharedRetrievalOverlap()
    owner.configure(tuple(impls.items()), groups, caches)
    owner.prepare(Key(), tuple(impls.items()), groups, 1024)
    assert owner.edges(Key()) == ({}, {})
    assert owner.stream is None and not log


def test_edge_uses_explicit_payload_and_refuses_substituted_destination(runtime):
    module, log = runtime
    destination = torch.zeros(4)
    transfer = NS(load=Mock())
    edge = module.RetrievalEdge(torch.npu.Stream(), transfer, (destination,))
    selected, counts, slots = torch.arange(4), torch.tensor([4]), torch.arange(4)
    log.clear()
    edge.launch(selected, counts, slots, (destination.view(4),))
    edge.join()
    transfer.load.assert_called_once_with(selected, counts, slots)
    assert log == [("record", "main"), ("wait", "transfer"), ("record", "transfer"), ("wait", "main")]
    with pytest.raises(RuntimeError, match="destination changed"):
        edge.launch(selected, counts, slots, (destination.clone(),))
    with pytest.raises(RuntimeError, match="payload storage changed"):
        edge.launch(selected.clone(), counts, slots, (destination,))
    assert transfer.load.call_count == 1


def test_edge_retains_actual_payload_until_cleanup(runtime):
    module, _ = runtime
    edge = module.RetrievalEdge(torch.npu.Stream(), NS(load=lambda *args: None), (torch.zeros(4),))
    selected, counts, slots = torch.arange(4), torch.tensor([4]), torch.arange(4)
    refs = [weakref.ref(t) for t in (selected, counts, slots)]
    edge.launch(selected, counts, slots, edge.destinations)
    del selected, counts, slots
    assert all(ref() is not None for ref in refs)
    del edge  # The root owner synchronizes before dropping edges in production.
    assert all(ref() is None for ref in refs)


def test_capture_join_requires_a_launch_not_just_a_primed_event(runtime):
    module, log = runtime
    edge = module.RetrievalEdge(torch.npu.Stream(), NS(load=Mock()), (torch.zeros(4),))
    log.clear()
    with pytest.raises(RuntimeError, match="no producer launch"):
        edge.join()
    assert not log  # Reject before submitting an invalid native event wait.
    selected, counts, slots = torch.arange(4), torch.tensor([4]), torch.arange(4)
    for _ in range(2):
        edge.launch(selected, counts, slots, edge.destinations)
        with pytest.raises(RuntimeError, match="twice"):
            edge.launch(selected, counts, slots, edge.destinations)
        edge.join()
        with pytest.raises(RuntimeError, match="no producer launch"):
            edge.join()


def test_no_owner_cycle_with_gc_disabled(runtime):
    module, _ = runtime
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        owner = module.SharedRetrievalOverlap()
        groups, impls, caches = topology()
        owner.configure(tuple(impls.items()), groups, caches)
        owner.prepare(Key(), tuple(impls.items()), groups, 1024)
        ref = weakref.ref(owner)
        del owner, impls
        assert ref() is None
    finally:
        if was_enabled:
            gc.enable()


def test_actual_post_starts_prefetch_between_attention_and_value_projection():
    log = []
    value = torch.ones(2)
    ns = dict(torch=torch, get_weight_prefetch_method=lambda: NS(
        maybe_prefetch_mla_or_sla_weight_in_current_stream=lambda **kw: None), MAX_O_PROJ_PREFETCH_SIZE=0)
    post = extract(SOURCE, "_cross_layer_post_compute", ns)
    proj = Mock(side_effect=lambda x: (log.append("output") or x,))
    impl = NS(_execute_sparse_flash_attention_process=lambda *a, **kw: log.append("sfa") or value,
              _v_up_proj=lambda x: log.append("cube") or x, o_proj=proj)
    edge = NS(launch=lambda *args: log.append("prefetch"))
    post(impl, value, value, value, value, value, value, value, value, torch.empty_like(value),
         trace_label="test", prefetch=(edge, value, value, value, (value,)))
    assert log == ["sfa", "prefetch", "cube", "output"]


def test_pre_joins_before_any_consumer_cache_access():
    log = []
    context = NS(staged_sfa_graph_key=Key())
    pre = extract(SOURCE, "cross_layer_graph_pre", {"get_forward_context": lambda: context})
    def cache(*args):
        log.append("cache")
        raise RuntimeError("stop after first access")
    impl = NS(_retrieval_overlap=object(),
              _shared_retrieval_edges=lambda: ({"consumer": NS(join=lambda: log.append("join"))}, {}),
              _cross_layer_kv_cache=cache)
    with pytest.raises(RuntimeError, match="first access"):
        pre(impl, "consumer", torch.ones(1), (), NS(), False, torch.empty(1))
    assert log == ["join", "cache"]


def test_serial_retrieve_suppressed_only_for_incoming_full_graph_edge():
    context = NS(staged_sfa_graph_key=Key(), sfa_full_graph_active=True)
    retrieve = extract(SOURCE, "cross_layer_lmcache_retrieve",
                       {"_staged_sfa_profile_scope": lambda _: nullcontext()})
    load = Mock()
    impl = NS(_retrieval_overlap=object(), _shared_retrieval_edges=lambda: ({"consumer": object()}, {}),
              _full_graph_transfer=NS(load=load))
    data = torch.ones(2, 4)
    retrieve(impl, "producer", "consumer", data, data, data, NS(), context)
    retrieve(impl, "consumer", "", data, data, data, NS(), context)
    assert load.call_count == 1
    impl._retrieval_overlap = None
    retrieve(impl, "consumer", "", data, data, data, NS(), context)
    assert load.call_count == 2


@pytest.mark.parametrize("active", [False, True])
def test_edges_are_only_used_in_root_graph(active):
    context = NS(sfa_full_graph_active=active, staged_sfa_graph_key=Key())
    get_edges = extract(SOURCE, "_shared_retrieval_edges", {"get_forward_context": lambda: context})
    owner = NS(edges=Mock(return_value=({"in": 1}, {"out": 2})))
    impl = NS(_retrieval_overlap=owner, shared_resident_plan=NS(active=True))
    assert get_edges(impl) == (({"in": 1}, {"out": 2}) if active else ({}, {}))
    assert owner.edges.call_count == int(active)


def test_private_post_operator_passes_call_payload_not_hidden_plan():
    selected, counts, slots = torch.ones(4, 8), torch.ones(4), torch.ones(4, 8)
    dest = [torch.empty(4)]
    edge = NS(transfer=NS(request_capacity=2), destinations=dest)
    impl = NS(_shared_retrieval_edges=lambda: ({}, {"producer": edge}), cross_layer_graph_post=Mock())
    op = extract(ROOT / "vllm_ascend/ops/mla.py", "sfa_forward_post_prefetch",
                 {"_mla_runtime_state": lambda name: (impl, "producer", (), NS())})
    op(selected, selected, selected, selected, counts, slots, selected, "wrapper")
    actual = impl.cross_layer_graph_post.call_args.kwargs["prefetch"]
    assert actual[0] is edge and actual[-1] is dest
    for sliced, source in zip(actual[1:4], (selected, counts, slots)):
        assert sliced.data_ptr() == source.data_ptr() and sliced.shape[0] == 2


def test_prefetch_operator_preserves_output_effect_ordering():
    tree = ast.parse((ROOT / "vllm_ascend/ops/mla.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "direct_register_custom_op"]
    matches = [n for n in calls if any(k.arg == "op_name" and isinstance(k.value, ast.Constant)
                                      and k.value.value == "sfa_forward_post_prefetch" for k in n.keywords)]
    assert len(matches) == 1
    writes = next(k.value for k in matches[0].keywords if k.arg == "mutates_args")
    assert ast.literal_eval(writes) == ["output"]


def test_partial_capture_failure_keeps_owner_until_synchronized_clear(graph_module, monkeypatch):
    module, context, _, _ = graph_module
    root = module.SFAFullGraph()
    events = []
    owner = NS(stream=object(), clear=lambda: events.append("clear"))
    root.retrieval_overlap = owner
    monkeypatch.setattr(torch.npu, "synchronize", lambda: events.append("sync"))
    with pytest.raises(RuntimeError, match="partial submission"):
        root.run(Mock(side_effect=RuntimeError("partial submission")))
    assert root._submission_failed and root.retrieval_overlap is owner
    assert not context.sfa_full_graph_active and not context.capturing
    with pytest.raises(RuntimeError):
        root.run(Mock())
    root.clear()
    assert events == ["sync", "clear"]


def test_live_replay_never_reenters_overlap_python(graph_module):
    module, context, _, _ = graph_module
    root = module.SFAFullGraph()
    root.retrieval_overlap = NS(stream=object(), clear=Mock())
    launch = Mock()
    root.run(launch)
    root.seal((context.staged_sfa_graph_key,))
    context.staged_sfa_graph_dummy_run = False
    for _ in range(8):
        root.run(launch)
    assert launch.call_count == 1
    assert root.replay_count == 8


def test_lease_retirement_waits_for_root_completion(graph_module):
    module, _, _, _ = graph_module
    root = module.SFAFullGraph()
    lease, completion = NS(close=Mock()), NS(query=Mock(return_value=False))
    root.retired_sources.append(NS(lease=lease, completion=completion))
    root.collect_retired_sources()
    lease.close.assert_not_called()
    completion.query.return_value = True
    root.collect_retired_sources()
    lease.close.assert_called_once()


@pytest.mark.parametrize("active", [0, 1, 8])
def test_each_layer_copies_once_with_layer_specific_values(runtime, active):
    module, log = runtime
    groups, impls, caches = topology()
    selected, counts, slots = torch.arange(8), torch.tensor([active]), torch.arange(8)
    copies = []
    for layer, name in enumerate(groups[0].members):
        def load(indices, lengths, destinations, *, layer=layer, name=name, max_aiv_cores=0):
            assert max_aiv_cores == (0 if layer == 0 else 12)
            copies.append(name)
            for p, tensor in enumerate(caches[name]):
                tensor.view(-1, tensor.shape[-1])[:active].fill_(layer + p + 1)
        impls[name]._full_graph_transfers[2].load = load
        for tensor in caches[name]:
            tensor.fill_(-7)
    owner = module.SharedRetrievalOverlap()
    owner.configure(tuple(impls.items()), groups, caches)
    owner.prepare(Key(), tuple(impls.items()), groups, 1024)
    incoming, outgoing = owner.edges(Key())
    for layer, name in enumerate(groups[0].members):
        if name in incoming:
            incoming[name].join()
        else:
            impls[name]._full_graph_transfers[2].load(selected, counts, slots)
        for p, tensor in enumerate(caches[name]):
            flat = tensor.view(-1, tensor.shape[-1])
            assert torch.all(flat[:active] == layer + p + 1)
            assert torch.all(flat[active:] == -7)
        if name in outgoing:
            outgoing[name].launch(selected, counts, slots, outgoing[name].destinations)
    assert copies == list(groups[0].members)


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("shared_storage", [False, True])
def test_aot_resolves_late_destinations_inside_prefetch_operator(v2, shared_storage):
    import functools
    from torch._dynamo.backends.common import aot_autograd
    from torch._inductor import config
    from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple
    from torch._inductor.decomposition import select_decomp_table

    data = torch.ones(2, 4)
    backing = torch.zeros(12)
    destinations = ([backing[:8].view(2, 4), backing[8:].view(2, 2)] if shared_storage
                    else [torch.zeros(2, 4), torch.zeros(2, 2)])
    seen = []
    edge = NS(transfer=NS(request_capacity=2), destinations=destinations)
    active = NS(value=False)
    def post(*args, prefetch):
        args[-1].copy_(args[1])
        if prefetch is None:
            return
        buffers = prefetch[-1]
        seen.append([t.data_ptr() for t in buffers])
        for buf in buffers:
            buf.add_(1)
        args[-1].copy_(args[1])
    impl = NS(_shared_retrieval_edges=lambda: ({}, {"producer": edge} if active.value else {}),
              cross_layer_graph_post=post)
    ns = {"torch": torch, "_mla_runtime_state": lambda name: (impl, "producer", (), NS())}
    op_fn = extract(ROOT / "vllm_ascend/ops/mla.py", "sfa_forward_post_prefetch", ns)
    fake = extract(ROOT / "vllm_ascend/ops/mla.py", "sfa_forward_post_prefetch_fake", {})
    namespace = f"prefetch_island_{int(v2)}_{int(shared_storage)}"
    lib = torch.library.Library(namespace, "DEF")
    lib.define("post" + torch.library.infer_schema(op_fn, mutates_args=["output"]))
    lib.impl("post", op_fn, "CPU")
    lib._register_fake("post", fake)
    op = getattr(torch.ops, namespace).post
    compiler = dict(functools=functools, aot_autograd=aot_autograd,
                    graph_returns_tuple=graph_returns_tuple, make_graph_return_tuple=make_graph_return_tuple,
                    select_decomp_table=select_decomp_table, COMPILATION_PASS_KEY="passes",
                    torch=NS(ops=NS(vllm=NS(sfa_forward_post_prefetch=op), higher_order=torch.ops.higher_order)))
    source = ROOT / "vllm_ascend/compilation/compiler_interface.py"
    for name in ("compile_fx", "_reuse_shared_resident_buffers", "fusion_pass_compile"):
        compiler[name] = extract(source, name, compiler)
    def backend(graph, inputs):
        return compiler["fusion_pass_compile"](graph, inputs, {"passes": lambda g: g}, None)[0]
    def forward(x):
        output = torch.empty_like(x)
        op(x, x, x, x, x, x, output, "producer")
        return output
    with config.patch(enable_auto_functionalized_v2=v2):
        compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
        torch.testing.assert_close(compiled(data), data)  # Initial model profiling: no edges.
        assert not seen
        active.value = True  # KV setup after compilation, before capture.
        for _ in range(2):
            torch.testing.assert_close(compiled(data), data)
    assert seen == [[t.data_ptr() for t in destinations]] * 2
    assert all(t.eq(2).all() for t in destinations)


def test_compiler_ignores_unrelated_list_arguments():
    fn = extract(ROOT / "vllm_ascend/compilation/compiler_interface.py", "_reuse_shared_resident_buffers",
                 {"torch": torch})
    graph = torch.fx.symbolic_trace(lambda x: torch.cat([x, x]))
    before = str(graph.graph)
    assert not fn(graph)
    assert str(graph.graph) == before


def test_disabled_mla_keeps_original_operator_sequence():
    events = []
    x = torch.ones(2, 4)
    bridge = (x, x, x, x, x, x)
    ops = NS(
        sfa_forward_pre=Mock(side_effect=lambda *args: events.append("pre") or bridge),
        sfa_forward_pre_shared=Mock(side_effect=AssertionError("unexpected shared operator")),
        sfa_lmcache_retrieve=Mock(side_effect=lambda *args: events.append("retrieve")),
        sfa_forward_post=Mock(side_effect=lambda *args: events.append("post")),
        sfa_forward_post_prefetch=Mock(side_effect=AssertionError("unexpected prefetch")),
    )
    source = ROOT / "vllm_ascend/ops/mla.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMultiHeadLatentAttention")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    module = ast.parse("from __future__ import annotations")
    module.body.append(fn)
    ns = {"torch": NS(empty=torch.empty, ops=NS(vllm=ops)), "_EXTRA_CTX": NS(flash_comm_v1_enabled=False)}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), ns)
    impl = NS(shared_resident_plan=None, local_num_heads=1, kv_lora_rank=4, qk_rope_head_dim=2,
              index_topk=2048, _staged_sfa_graph_capture_sizes=(4,), decode_threshold=2)
    layer = NS(use_cross_layer_sfa=True, use_retrieval_overlap=False, target_sfa_debug=False, mla_attn=NS(impl=impl),
               prefix="layer", next_layer_name="next")
    ns["forward"](layer, x, x)
    assert events == ["pre", "retrieve", "post"]


@pytest.mark.parametrize("full,shared", [(False, False), (False, True), (True, False), (True, True)])
def test_configuration_requires_both_prerequisites(full, shared):
    source = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    block = next(n for n in init.body if isinstance(n, ast.If)
                 and "VLLM_ASCEND_SFA_SHARED_RETRIEVAL_OVERLAP" in ast.unparse(n.test))
    # Execute validation only; allocation/import is covered by the owner tests.
    validate = block.body[0]
    ns = {"sfa_full_graph_enabled": lambda cfg: full, "vllm_config": None,
          "envs_ascend": NS(VLLM_ASCEND_SFA_SHARED_RESIDENT_PLAN=shared)}
    code = compile(ast.fix_missing_locations(ast.Module(body=[validate], type_ignores=[])), str(source), "exec")
    if full and shared:
        exec(code, ns)
    else:
        with pytest.raises(ValueError, match="requires full graphs"):
            exec(code, ns)


def test_profiling_cleanup_releases_old_cache_before_final_allocation(runtime):
    module, _ = runtime
    owner = module.SharedRetrievalOverlap()
    groups, impls, caches = topology()
    owner.configure(tuple(impls.items()), groups, caches)
    references = [weakref.ref(t) for planes in caches.values() for t in planes]

    class Parent:
        def _cleanup_profiling_kv_cache(self):
            for impl in impls.values():
                impl._staged_sfa_capture_state.runtime = None
            caches.clear()

    source = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "_cleanup_profiling_kv_cache")
    cls = ast.ClassDef(name="Runner", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                       keywords=[], body=[method], decorator_list=[])
    ns = {"Parent": Parent, "envs_ascend": NS(VLLM_ASCEND_SFA_FULL_GRAPH=True)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), ns)
    runner = ns["Runner"]()
    runner._sfa_full_graph = NS(retrieval_overlap=owner)
    runner._reset_staged_sfa_startup_capture = owner.clear
    runner._collect_staged_sfa_impls = lambda: tuple(impls.items())
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        runner._cleanup_profiling_kv_cache()
        assert all(ref() is None for ref in references), "overlap retained discarded profiling KV"
    finally:
        if was_enabled:
            gc.enable()


def test_graph_reset_retains_real_layout_but_profile_release_requires_clear(runtime):
    module, _ = runtime
    owner = module.SharedRetrievalOverlap()
    groups, impls, caches = topology()
    owner.configure(tuple(impls.items()), groups, caches)
    owner.prepare(Key(), tuple(impls.items()), groups, 1024)
    with pytest.raises(RuntimeError, match="clear overlap graphs"):
        owner.release_layout(tuple(impls.items()))
    destinations = owner.pairs["g0l0"][1]
    owner.clear()
    assert owner.pairs["g0l0"][1] is destinations
    owner.prepare(Key(), tuple(impls.items()), groups, 1024)
    assert owner.edges(Key())[1]["g0l0"].destinations == tuple(destinations)
    owner.clear()
    owner.release_layout(tuple(impls.items()))
    assert not owner.pairs
    assert impls["g0l0"]._retrieval_overlap is None
    # Final cache allocation uses the same layer objects but different storage.
    fresh = {name: tuple(t.clone() for t in values) for name, values in caches.items()}
    owner.configure(tuple(impls.items()), groups, fresh)
    assert owner.pairs["g0l0"][1][0] is fresh["g0l1"][0]


def test_compile_before_kv_initialization_keeps_prefetch_dispatch():
    """Model profiling compiles without KV; later execution bypasses guards."""
    from torch.fx.experimental.proxy_tensor import make_fx

    def pre(*args):
        return (args[0],) * 6

    def serial_post(*args):
        args[6].copy_(args[0] + 1)

    def overlap_post(*args):
        args[6].copy_(args[0] + 2)

    ops = NS(sfa_forward_pre=pre, sfa_forward_pre_shared=pre,
             sfa_lmcache_retrieve=lambda *args: None,
             sfa_forward_post=serial_post, sfa_forward_post_prefetch=overlap_post)
    source = ROOT / "vllm_ascend/ops/mla.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMultiHeadLatentAttention")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    mod = ast.parse("from __future__ import annotations")
    mod.body.append(fn)
    ns = {"torch": NS(empty=torch.empty, ops=NS(vllm=ops)), "_EXTRA_CTX": NS(flash_comm_v1_enabled=False)}
    exec(compile(ast.fix_missing_locations(mod), str(source), "exec"), ns)
    impl = NS(shared_resident_plan=None, local_num_heads=1, kv_lora_rank=4, qk_rope_head_dim=2,
              index_topk=2048, _staged_sfa_graph_capture_sizes=(4,), decode_threshold=2)
    layer = NS(use_cross_layer_sfa=True, use_retrieval_overlap=True, target_sfa_debug=False,
               mla_attn=NS(impl=impl), prefix="layer", next_layer_name="next")
    x = torch.ones(2, 4)
    compiled = make_fx(lambda value: ns["forward"](layer, value, value))(x)
    impl._retrieval_overlap = NS()
    # Execute the already-compiled callable, as the no-guards runner does.
    torch.testing.assert_close(compiled(x), x + 2)
