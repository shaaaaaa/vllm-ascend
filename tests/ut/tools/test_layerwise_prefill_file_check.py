# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the real connector -> file SDK boundary and orchestration."""

import asyncio
import importlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    workspace = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(workspace / "LMCache"))
    monkeypatch.syspath_prepend(str(workspace / "vllm-ascend/tools"))
    return NS(
        sdk=importlib.import_module("layerwise_prefill_file_store"),
        tool=importlib.import_module("layerwise_prefill_file_check"),
    )


@pytest.fixture
def run_root(tmp_path):
    for name in ("prefill", "decode", "baseline", "store"):
        (tmp_path / name).mkdir()
    return tmp_path


def make_store(modules, root, stage, *owners):
    store = modules.sdk.FileStore(root, stage)
    store.memory.bind(owners)
    store.setup()
    return store


@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16, torch.float32])
def test_real_tensor_offsets_multi_buffer_order_and_restart(modules, run_root, dtype):
    source = torch.arange(40).to(dtype)
    left, right = source[3:11], source[20:26]
    sizes = [x.numel() * x.element_size() for x in (left, right)]
    p = make_store(modules, run_root, "prefill", source)
    key = "unchanged/key@8@0@abc@bfloat16@0@3"
    assert p.batch_put_from_multi_buffers([key], [[left.data_ptr(), right.data_ptr()]], [sizes]) == [0]
    modules.tool.seal_store(run_root)
    p.close()
    source.zero_()  # Store must own bytes, not retain references to these values.
    target = torch.full((sum(sizes) + 10,), 255, dtype=torch.uint8)
    d = make_store(modules, run_root, "decode", target)
    assert d.batch_is_exist([key, "absent"]) == [1, 0]
    assert d.batch_get_into([key], [target[5:].data_ptr()], [sum(sizes)]) == [sum(sizes)]
    expected = torch.cat((torch.arange(3, 11), torch.arange(20, 26))).to(dtype).view(torch.uint8)
    assert torch.equal(target[5:-5], expected)
    assert target[:5].tolist() == target[-5:].tolist() == [255] * 5


@pytest.mark.parametrize("mode", ["missing", "short", "too_small", "corrupt", "wrong_key"])
def test_missing_short_and_corrupt_reads_are_not_silently_successful(modules, run_root, mode):
    source = torch.tensor([8, 6, 7], dtype=torch.uint8)
    target = torch.full((5,), 99, dtype=torch.uint8)
    p = make_store(modules, run_root, "prefill", source)
    p.batch_put_from(["key"], [source.data_ptr()], [3])
    modules.tool.seal_store(run_root)
    d = make_store(modules, run_root, "decode", target)
    if mode == "corrupt":
        with d.path("key").open("ab") as stream:
            stream.write(b"bad")
    if mode == "wrong_key":
        d.path("key").replace(d.path("wrong"))
    key = "absent" if mode == "missing" else "wrong" if mode == "wrong_key" else "key"
    if mode in ("corrupt", "wrong_key"):
        with pytest.raises(RuntimeError, match="corrupt"):
            d.batch_get_into([key], [target.data_ptr()], [5])
    else:
        size = 2 if mode == "too_small" else 5
        assert d.batch_get_into([key], [target.data_ptr()], [size]) == ([3] if mode == "short" else [-1])
    assert target.tolist() == ([8, 6, 7, 99, 99] if mode == "short" else [99] * 5)


def test_read_scatter_respects_each_destination_extent(modules, run_root):
    source = torch.arange(8, dtype=torch.uint8)
    dest = torch.full((20,), 255, dtype=torch.uint8)
    p = make_store(modules, run_root, "prefill", source)
    p.batch_put_from(["key"], [source.data_ptr()], [8])
    modules.tool.seal_store(run_root)
    d = make_store(modules, run_root, "decode", dest)
    assert d.batch_get_into_multi_buffers(["key"], [[dest[2:].data_ptr(), dest[12:].data_ptr()]], [[3, 5]]) == [8]
    assert dest[2:5].tolist() == [0, 1, 2]
    assert dest[12:17].tolist() == [3, 4, 5, 6, 7]
    assert dest[5:12].tolist() == [255] * 7


def test_unknown_pointer_and_readonly_store_fail(modules, run_root):
    p = make_store(modules, run_root, "prefill")
    with pytest.raises(ValueError, match="untracked buffer"):
        p.batch_put_from(["bad"], [1234], [64])
    assert not list((run_root / "store").iterdir())
    with pytest.raises(RuntimeError, match="never called"):
        modules.tool.seal_store(run_root)
    with pytest.raises(RuntimeError, match="cannot start"):
        make_store(modules, run_root, "decode")
    modules.tool.write_json(run_root / "store-sealed.json", {})
    with pytest.raises(RuntimeError, match="unsealed P"):
        p.batch_put_from(["bad"], [1234], [64])
    d = make_store(modules, run_root, "decode")
    with pytest.raises(RuntimeError, match="unsealed P"):
        d.batch_put_from(["bad"], [1234], [64])


def test_parallel_puts_publish_only_complete_objects(modules, run_root):
    source = torch.arange(200, dtype=torch.uint8)
    p = make_store(modules, run_root, "prefill", source)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: p.batch_put_from([f"key{i}"], [source.data_ptr()], [200]), range(32)))
    assert results == [[0]] * 32
    modules.tool.seal_store(run_root)
    assert len(json.loads((run_root / "store-sealed.json").read_text())) == 32
    assert not list((run_root / "store").glob("*.pending"))


def test_observing_owners_calls_original_methods_once(modules, run_root):
    calls = []

    class Connector:
        def _register_cpu_buffer(self):
            calls.append("cpu")
            return 42

        def _register_external_owners(self, owners):
            calls.append(owners)
            return 43

    modules.sdk.observe_buffers(Connector)
    modules.sdk.observe_buffers(Connector)
    obj = Connector()
    obj.store = make_store(modules, run_root, "prefill")
    slab, external = torch.zeros(20), torch.ones(10)
    obj.local_cpu_backend = NS(memory_allocator=NS(pin_allocator=NS(buffer=slab)))
    assert obj._register_cpu_buffer() == 42
    owners = (external,)
    assert obj._register_external_owners(owners) == 43
    assert calls == ["cpu", owners]
    assert set(obj.store.memory.storages) == {slab.data_ptr(), external.data_ptr()}


@pytest.mark.parametrize("stage", ["prefill", "decode", "baseline"])
def test_environment_keeps_production_connector_and_isolates_old_debug(modules, run_root, stage, monkeypatch):
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", "wrong.yaml")
    monkeypatch.setenv("LMCACHE_EXTRA_CONFIG", '{"prefill_check_routing": {}}')
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "wrong.json")
    args = modules.tool.parser().parse_args([])
    env = modules.tool.child_environment(args, run_root, stage)
    extra = json.loads(env["LMCACHE_EXTRA_CONFIG"])
    assert "LMCACHE_CONFIG_FILE" not in env and "MOONCAKE_CONFIG_PATH" not in env
    assert "prefill_check_routing" not in extra and "validation_archive" not in extra
    assert extra["save_chunk_meta"] is False
    assert extra["mooncake_page_first_multi_buffer"] and extra["mooncake_layer_merged_page_objects"]
    assert ("prefill_check_file_sdk" in extra) == (stage != "baseline")
    assert env.get("LMCACHE_REMOTE_URL") == (None if stage == "baseline" else "mooncakestore://127.0.0.1:1/")
    assert env["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] == str(stage == "prefill").lower()
    opts = modules.tool.engine_options(args, 9587, stage)
    assert opts["gpu_memory_utilization"] == 0.96 and opts["max_model_len"] == 16384
    assert bool(opts.get("enforce_eager")) == (stage == "prefill")
    assert "hf_overrides" not in opts and "speculative_config" not in opts


def test_private_bootstrap_covers_new_interpreters_without_native_mooncake(modules, run_root):
    modules.tool.prepare_bootstrap(run_root)
    args = modules.tool.parser().parse_args([])
    env = modules.tool.child_environment(args, run_root, "prefill")
    code = (
        "import mooncake.store as s; import mooncake.engine as e; "
        "assert s.file_check; assert s.ReplicateConfig().replica_num == 1; "
        "assert e.TransferEngine().initialize('unused') == 0; "
        "assert not hasattr(e.TransferEngine(), 'batch_transfer_sync_write'); print('file-sdk-only')"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "file-sdk-only"


def test_import_does_not_patch_normal_serving(modules, monkeypatch):
    monkeypatch.delenv("LMCACHE_EXTRA_CONFIG", raising=False)
    before = {k: v for k, v in sys.modules.items() if k.startswith("mooncake")}
    modules.sdk.install()
    assert {k: v for k, v in sys.modules.items() if k.startswith("mooncake")} == before


def test_real_mooncake_connector_page_put_and_direct_destination_get(modules, run_root, monkeypatch):
    # Real connector code, real page keys and buffer address ordering; only SDK
    # and allocator/model fixtures are replaced. No native Mooncake is imported.
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.storage_backend.connector.mooncakestore_connector import MooncakestoreConnector

    source = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    p = make_store(modules, run_root, "prefill", source)
    connector = MooncakestoreConnector.__new__(MooncakestoreConnector)
    connector.store = p
    connector.config = NS(transfer_timeout=5)
    connector._inflight_put_tasks = set()
    connector._page_first_multi_buffer = True
    connector._page_group_layer_counts = (2, 2)
    connector.local_cpu_backend = NS(metadata=NS(chunk_size=2))
    connector._metadata_for_raw_key = lambda key: (None, None, None, 4)
    connector.replica_config = modules.sdk.ReplicateConfig()
    chunk = CacheEngineKey("model", 8, 0, 0x123, torch.bfloat16, kv_group=1)
    keys = chunk.split_layers(2)
    refs = []
    objs = [
        NS(
            data_ptr=row.data_ptr(),
            get_size=lambda: 8,
            ref_count_up=lambda: refs.append(1),
            ref_count_down=lambda: refs.append(-1),
        )
        for row in source
    ]
    # Deliberately supply layers reversed: the real connector orders the page.
    asyncio.run(connector._batched_put_zero_copy(keys[::-1], objs[::-1]))
    assert sum(refs) == 0
    modules.tool.seal_store(run_root)
    dest = torch.zeros_like(source)
    connector.store = make_store(modules, run_root, "decode")
    connector.save_chunk_meta = False
    connector._external_put_lock = asyncio.Lock()
    connector._external_native_hard_timeout_seconds = 5
    connector._external_buffers = {}
    connector._shared_global_te = None
    # Patch registration only, just like the test process; body stays original.
    register = connector._register_external_owners

    def observed_registration(owners):
        connector.store.memory.bind(owners)
        return register(owners)

    monkeypatch.setattr(connector, "_register_external_owners", observed_registration)
    asyncio.run(
        connector.batched_get_external_pages([chunk], [[row.data_ptr() for row in dest]], [[8, 8]], (dest,), "request")
    )
    assert torch.equal(dest, source)
    gets = modules.tool.io_records(run_root, "decode")
    assert gets[-1]["method"] == "batch_get_into_multi_buffers"
    assert gets[-1]["key"].startswith("__lmcache_page_v1__@2@model@8@0@")


def evidence(modules, root, *, read=True):
    source = torch.arange(6, dtype=torch.uint8)
    p = make_store(modules, root, "prefill", source)
    keys = [f"__lmcache_page_v1__@2@model@8@0@abc@bfloat16@{group}" for group in (0, 1)]
    p.batch_put_from(keys, [source.data_ptr()] * 2, [6] * 2)
    modules.tool.seal_store(root)
    d = make_store(modules, root, "decode", source)
    if read:
        d.batch_get_into(keys, [source.data_ptr()] * 2, [6] * 2)
    modules.tool.write_json(root / "prompt.json", {"length": 9587})
    for stage, tokens, cached in (("baseline", [1, 2], 0), ("prefill", [3], 0), ("decode", [4, 5], 9216)):
        modules.tool.write_json(root / stage / "output.json", {"token_ids": tokens, "num_cached_tokens": cached})


@pytest.mark.parametrize("read", [True, False])
def test_summary_requires_actual_sdk_calls_and_reports_not_rejects_token_difference(modules, run_root, read):
    evidence(modules, run_root, read=read)
    if read:
        modules.tool.analyse(run_root, with_baseline=True)
    else:
        with pytest.raises(RuntimeError, match="validation incomplete"):
            modules.tool.analyse(run_root)
    summary = json.loads((run_root / "summary.json").read_text())
    if read:
        assert summary["decode_read_groups"] == [0, 1]
        assert summary["decode_first_difference"] == 0 and not summary["prefill_first_token_equal"]
        assert not summary["errors"]
    else:
        assert summary["errors"]


def test_model_lifecycle_is_only_prefill_finish_seal_then_decode(modules, run_root, monkeypatch):
    calls = []
    args = modules.tool.parser().parse_args([])
    monkeypatch.setattr(
        modules.tool,
        "start_logged_process",
        lambda cmd, *rest, **kwargs: calls.append(cmd[cmd.index("--child") + 1]) or NS(wait=lambda: 0),
    )
    monkeypatch.setattr(modules.tool, "finish_child", lambda proc: calls.append("finish"))
    monkeypatch.setattr(modules.tool, "seal_store", lambda root: calls.append("seal"))
    modules.tool.run_stages(args, run_root)
    assert calls == ["prefill", "finish", "seal", "decode", "finish"]


def test_prefill_failure_cannot_start_decode(modules, run_root, monkeypatch):
    calls = []
    monkeypatch.setattr(modules.tool, "start_logged_process", lambda *args, **kwargs: NS(wait=lambda: 1))
    monkeypatch.setattr(modules.tool, "finish_child", lambda proc: calls.append("finish"))
    monkeypatch.setattr(modules.tool, "seal_store", lambda root: calls.append("seal"))
    with pytest.raises(RuntimeError, match="prefill failed"):
        modules.tool.run_stages(modules.tool.parser().parse_args([]), run_root)
    assert calls == ["finish"]


@pytest.mark.parametrize("stage", ["baseline", "prefill", "decode"])
def test_generation_flush_and_shutdown_keep_original_role_and_token_limits(modules, run_root, monkeypatch, stage):
    calls = []
    args = modules.tool.parser().parse_args(["--output-tokens", "321"])
    args.run_dir, args.child = run_root, stage
    modules.tool.write_json(run_root / "prompt.json", {"token_ids": [11, 22], "length": 9587})
    monkeypatch.delenv("LMCACHE_EXTRA_CONFIG", raising=False)

    class LLM:
        def __init__(self, **options):
            self.options = options
            self.llm_engine = NS(engine_core=NS(shutdown=lambda: calls.append("shutdown")))
            calls.append(options)

        def generate(self, prompt, sampling, **kwargs):
            assert prompt == {"prompt_token_ids": [11, 22]}
            assert sampling.max_tokens == (1 if stage == "prefill" else 321)
            assert not hasattr(sampling, "ignore_eos") and not hasattr(sampling, "min_tokens")
            calls.append("generate")
            return [NS(outputs=[NS(text="answer", token_ids=[5], finish_reason="stop")], num_cached_tokens=0)]

        def collective_rpc(self, method, **kwargs):
            calls.append(method)
            return [True] * 8

    monkeypatch.setitem(sys.modules, "vllm", NS(LLM=LLM, SamplingParams=NS))
    modules.tool.run_model(args)
    expected = ["generate", "prefill_check_flush_store", "shutdown"] if stage == "prefill" else ["generate", "shutdown"]
    assert calls[1:] == expected
    assert bool(calls[0].get("enforce_eager")) == (stage == "prefill")
    assert (
        "worker_extension_cls" not in calls[0]
        if stage == "baseline"
        else calls[0]["worker_extension_cls"].endswith("FileStoreWorker")
    )


def test_actual_store_factory_patches_before_connector_registration(modules, run_root, monkeypatch):
    from lmcache.v1.storage_backend.connector.mooncakestore_connector import MooncakestoreConnector

    # Restore these class/module changes on teardown so normal tests are untouched.
    for name in ("_register_cpu_buffer", "_register_external_owners"):
        monkeypatch.setattr(MooncakestoreConnector, name, getattr(MooncakestoreConnector, name))
    monkeypatch.setattr(MooncakestoreConnector, "_file_check_observed", False, raising=False)
    for name in ("mooncake", "mooncake.store", "mooncake.engine"):
        monkeypatch.setitem(sys.modules, name, None)
        del sys.modules[name]
    monkeypatch.setenv(
        "LMCACHE_EXTRA_CONFIG", json.dumps({"prefill_check_file_sdk": {"root": str(run_root), "stage": "prefill"}})
    )
    modules.sdk.install()
    sdk_module = sys.modules["mooncake.store"]
    modules.sdk.install()
    assert sys.modules["mooncake.store"] is sdk_module
    store = sdk_module.MooncakeDistributedStore()
    connector = MooncakestoreConnector.__new__(MooncakestoreConnector)
    slab = torch.arange(32, dtype=torch.uint8)
    connector.store = store
    connector.local_cpu_backend = NS(memory_allocator=NS(pin_allocator=NS(buffer=slab)))
    connector._shared_global_te = None
    connector._register_cpu_buffer()
    assert connector.registered_buffer_size == 32
    assert store.memory.read(slab[7:].data_ptr(), 4) == bytes([7, 8, 9, 10])
