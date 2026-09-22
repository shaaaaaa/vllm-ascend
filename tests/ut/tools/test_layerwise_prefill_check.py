# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the opt-in three-pass diagnostic, without vLLM/NPU imports."""

import abc
import ast
import asyncio
import ctypes
import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property, wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sortedcontainers import SortedList

TOOLS = Path(__file__).resolve().parents[3] / "tools"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK = load_tool("layerwise_prefill_check")
PROBE = load_tool("layerwise_prefill_probe")


def args():
    return CHECK.parser().parse_args([])


def test_server_launcher_matches_validation_memory_limits():
    script = (TOOLS / "serve_glm52_baseline.sh").read_text(encoding="utf-8")
    assert "--max-model-len 16384" in script
    assert "--gpu-memory-utilization 0.96" in script
    assert "--enforce-eager" not in script


def test_default_prompt_is_committed_article_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = args().prompt_file
    assert path.is_absolute()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("请阅读")
    assert "END OF ARTICLE" in text
    assert 7500 < len(text.split()) < 9000
    assert "fourteen percent" in text and "six percent" in text
    assert text.count("Field record ") == 36


@pytest.fixture
def fake_tokenizer(monkeypatch):
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = list(range(12288))
    module = ModuleType("transformers")
    module.AutoTokenizer = Mock()
    module.AutoTokenizer.from_pretrained.return_value = tokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)
    return tokenizer


def test_fixed_prompt_tokenized_once_and_preserved(tmp_path, fake_tokenizer, capsys):
    options = args()
    text = options.prompt_file.read_text(encoding="utf-8")
    assert CHECK.prepare_prompt(options, tmp_path) == 12288
    fake_tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    assert (tmp_path / "prompt.txt").read_text(encoding="utf-8") == text
    record = json.loads((tmp_path / "prompt.json").read_text(encoding="utf-8"))
    assert record["token_ids"] == list(range(12288))
    assert record["source"] == str(options.prompt_file.resolve())
    output = capsys.readouterr().out
    assert "loading tokenizer" in output and "prompt_tokens=12288" in output


def test_reuse_previous_baseline_without_copying_prompt_or_kv_trace(tmp_path):
    previous = tmp_path / "previous"
    (previous / "baseline").mkdir(parents=True)
    CHECK.write_json(
        previous / "prompt.json",
        {"token_ids": [1, 2, 3], "sha256": "same", "length": 3},
    )
    CHECK.write_json(previous / "baseline" / "output.json", {"stage": "baseline"})

    current = tmp_path / "current"
    current.mkdir()
    old_root, old_stage = CHECK.validate_baseline_run(
        previous,
        {"token_ids": [1, 2, 3], "sha256": "same", "length": 3},
    )
    assert old_root == previous.resolve()
    assert old_stage == (previous / "baseline").resolve()
    mode = CHECK.reuse_baseline_stage(current, old_stage)
    assert mode in ("symlink", "copy")
    assert (current / "baseline" / "output.json").is_file()


def test_reused_baseline_rejects_different_prompt(tmp_path):
    previous = tmp_path / "previous"
    (previous / "baseline").mkdir(parents=True)
    CHECK.write_json(previous / "prompt.json", {"sha256": "old"})
    CHECK.write_json(previous / "baseline" / "output.json", {})
    with pytest.raises(ValueError, match="does not match"):
        CHECK.validate_baseline_run(previous, {"sha256": "new"})


@pytest.mark.parametrize(
    "encoded",
    [
        [7, 9, 11],
        (7, 9, 11),
        [[7, 9, 11]],
        {"input_ids": [7, 9, 11], "attention_mask": [1, 1, 1]},
        {"input_ids": [[7, 9, 11]], "attention_mask": [[1, 1, 1]]},
        torch.tensor([7, 9, 11]),
        torch.tensor([[7, 9, 11]]),
        {"input_ids": torch.tensor([[7, 9, 11]])},
    ],
)
def test_normalize_tokenizer_result_preserves_actual_ids(encoded):
    assert CHECK.normalize_prompt_token_ids(encoded) == [7, 9, 11]


@pytest.mark.parametrize(
    "encoded", [[], [[]], "rendered text", {"attention_mask": [1]}, [[1, 2], [3, 4]], [1.0], [True], [-1]]
)
def test_invalid_tokenizer_payload_is_not_silently_counted(encoded):
    with pytest.raises(ValueError, match="Tokenizer"):
        CHECK.normalize_prompt_token_ids(encoded)


def test_prepare_prompt_handles_custom_tokenizer_returning_mapping(tmp_path, fake_tokenizer):
    fake_tokenizer.apply_chat_template.return_value = {
        "input_ids": list(range(12288)),
        "attention_mask": [1] * 12288,
    }
    assert CHECK.prepare_prompt(args(), tmp_path) == 12288
    record = json.loads((tmp_path / "prompt.json").read_text(encoding="utf-8"))
    assert record["token_ids"] == list(range(12288))
    assert fake_tokenizer.apply_chat_template.call_count == 1


def test_real_hf_tokenizer_mapping_and_prepare_prompt(tmp_path, monkeypatch):
    # Fully offline: use a real HF chat template and BatchEncoding with a tiny
    # local vocabulary. No model download, vLLM import, or NPU is required.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    raw = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        model_input_names=["input_ids", "attention_mask"],
        chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
    )
    messages = [{"role": "user", "content": "hello hello hello"}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True)
    assert len(encoded) == 2  # Fields, not tokens: reproduce the reported bug.
    assert CHECK.normalize_prompt_token_ids(encoded) == [1, 1, 1]
    calls = Mock(wraps=tokenizer.apply_chat_template)
    monkeypatch.setattr(tokenizer, "apply_chat_template", calls)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    source = tmp_path / "article.txt"
    source.write_text("hello " * 5000, encoding="utf-8")
    options = CHECK.parser().parse_args(["--prompt-file", str(source)])
    assert CHECK.prepare_prompt(options, tmp_path) == 5000
    assert calls.call_count == 1
    assert calls.call_args.kwargs["return_dict"] is False
    record = json.loads((tmp_path / "prompt.json").read_text(encoding="utf-8"))
    assert record["token_ids"] == [1] * 5000


@pytest.mark.parametrize(
    "length, minimum, message", [(4096, None, "multiple"), (16384, None, "exceeds"), (5000, 6000, "below")]
)
def test_fixed_prompt_not_padded_or_truncated(tmp_path, fake_tokenizer, length, minimum, message):
    fake_tokenizer.apply_chat_template.return_value = list(range(length))
    options = args()
    options.prompt_tokens = minimum
    with pytest.raises(ValueError, match=message):
        CHECK.prepare_prompt(options, tmp_path)
    assert fake_tokenizer.apply_chat_template.call_count == 1
    assert not (tmp_path / "prompt.json").exists()


def test_prompt_file_override_and_empty_input(tmp_path, fake_tokenizer):
    source = tmp_path / "custom.txt"
    source.write_text("自定义文章\n", encoding="utf-8")
    options = CHECK.parser().parse_args(["--prompt-file", str(source)])
    CHECK.prepare_prompt(options, tmp_path)
    assert fake_tokenizer.apply_chat_template.call_args.args[0][0]["content"] == "自定义文章\n"
    source.write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty prompt"):
        CHECK.prepare_prompt(options, tmp_path)


def test_startup_and_help_do_not_import_ml_dependencies():
    script = str(TOOLS / "layerwise_prefill_check.py")
    code = f"""
import importlib.abc
import runpy
import sys
class NoMLImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch', 'transformers', 'vllm', 'torch_npu'}}:
            raise RuntimeError('Unexpected early import: ' + fullname)
sys.meta_path.insert(0, NoMLImports())
sys.argv = [{script!r}, '--help']
runpy.run_path({script!r}, run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "[PREFILL_CHECK] starting" in result.stdout
    assert "--prompt-file" in result.stdout


def test_main_records_prompt_path_and_reuses_parent_tokenization(tmp_path, monkeypatch, fake_tokenizer):
    root = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["check", "--run-dir", str(root)])
    monkeypatch.setattr(CHECK, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(CHECK.shutil, "disk_usage", lambda path: SimpleNamespace(free=16 * 1024**3))
    stages = []
    monkeypatch.setattr(CHECK, "run_stage", lambda options, directory, stage: stages.append(stage))
    monkeypatch.setattr(CHECK, "seal_archive", lambda directory: None)
    monkeypatch.setattr(CHECK, "analyse", lambda directory, ranks, workers: None)
    CHECK.main()
    assert stages == list(CHECK.STAGES)
    assert fake_tokenizer.apply_chat_template.call_count == 1
    record = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert record["prompt_file"] == str(CHECK.DEFAULT_PROMPT_FILE)
    assert record["actual_prompt_tokens"] == 12288


def test_full_model_three_separate_roles():
    for stage, role in zip(CHECK.STAGES, ("kv_both", "kv_producer", "kv_consumer"), strict=True):
        options = CHECK.engine_options(args(), 12000, stage)
        assert options["load_format"] == "safetensors"
        assert "hf_overrides" not in options
        assert "speculative_config" not in options
        assert options["tensor_parallel_size"] == 8
        assert options["gpu_memory_utilization"] == 0.97
        assert options["max_model_len"] == 9000
        assert options["kv_transfer_config"]["kv_role"] == role
        if stage == "prefill":
            assert options["enforce_eager"]
        else:
            assert "enforce_eager" not in options
            assert options["compilation_config"]["cudagraph_mode"] == "PIECEWISE"
            assert options["compilation_config"]["cudagraph_capture_sizes"] == [1]


def test_long_output_default_and_article_not_limited_to_short_summary():
    assert args().output_tokens == CHECK.DEFAULT_OUTPUT_TOKENS == 256
    text = args().prompt_file.read_text(encoding="utf-8")
    assert "约200字" not in text
    CHECK.validate_sequence_length(8744, 256)
    with pytest.raises(ValueError, match="shorter --prompt-file"):
        CHECK.validate_sequence_length(8745, 256)


@pytest.mark.parametrize("stage", CHECK.STAGES)
def test_fixed_model_length_boundary(stage):
    options = CHECK.engine_options(args(), 9000 - args().output_tokens, stage)
    assert options["max_model_len"] == 9000
    assert options["gpu_memory_utilization"] == 0.97
    with pytest.raises(ValueError, match="exceeds max_model_len=9000"):
        CHECK.engine_options(args(), 9000 - args().output_tokens + 1, stage)


def test_stages_only_read_prefill_archive_and_isolate_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", "/old/mooncake.yaml")
    monkeypatch.setenv("VLLM_ASCEND_SFA_FULL_GRAPH", "1")
    for stage in CHECK.STAGES:
        env = CHECK.stage_environment(args(), tmp_path, stage)
        assert "LMCACHE_CONFIG_FILE" not in env
        assert env["VLLM_ASCEND_SFA_FULL_GRAPH"] == "0"
        assert env["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] == str(stage == "prefill").lower()
        assert env["LMCACHE_LAYERWISE_PREFILL_DMA"] == ("1" if stage == "prefill" else "0")
        assert env["LMCACHE_SHARED_CPU_TRACE"] == "1"
        extra = json.loads(env["LMCACHE_EXTRA_CONFIG"])
        source = "prefill" if stage == "decode" else stage
        assert Path(extra["validation_archive"]) == tmp_path / source / "archive"
        assert extra["validation_read_only"] == (stage == "decode")


def test_statistics_are_population_variance_and_abs_diff():
    stats = CHECK.tensor_statistics(torch.tensor([1.0, -3.0]), torch.tensor([2.0, -1.0]))
    assert stats["baseline"]["mean"] == -1
    assert stats["baseline"]["variance"] == 4
    assert stats["candidate_abs"]["mean"] == 1.5
    assert stats["diff"]["mean"] == 1.5
    assert stats["abs_diff"]["mean"] == 1.5
    assert stats["abs_diff"]["variance"] == 0.25


def test_tiny_difference_reported_not_rejected():
    base = torch.ones(10, dtype=torch.float64)
    stats = CHECK.tensor_statistics(base, base + 1e-9)
    assert 0 < stats["abs_diff"]["mean"] < 1e-7


def test_bfloat_subtraction_widens_before_stats():
    base = torch.tensor([0.5, 1], dtype=torch.bfloat16)
    other = torch.tensor([0.50390625, 1.0078125], dtype=torch.bfloat16)
    assert CHECK.tensor_statistics(base, other)["abs_diff"]["mean"] == 0.005859375


def test_nonfinite_count_preserved_and_json_safe():
    stats = CHECK.tensor_statistics(torch.tensor([1.0, float("nan")]), torch.tensor([float("inf"), 2.0]))
    assert stats["baseline"]["nonfinite"] == 1
    assert stats["abs_diff"]["nonfinite"] == 2
    assert stats["abs_diff"]["mean"] is None
    json.dumps(stats, allow_nan=False)


def test_streaming_moments_match_whole_tensor():
    values = torch.tensor([1000.0, 1000.125, 1000.25, -4.0], dtype=torch.float64)
    moments = CHECK.Moments()
    moments.add(values[:1])
    moments.add(values[1:])
    assert moments.result()["mean"] == pytest.approx(float(values.mean()))
    assert moments.result()["variance"] == pytest.approx(float(values.var(correction=0)))


def test_token_divergence_and_length():
    assert CHECK.first_difference([1, 2], [1, 2]) is None
    assert CHECK.first_difference([1, 2], [1, 3]) == 1
    assert CHECK.first_difference([1], [1, 2]) == 1


def save_rows(path, positions, values):
    torch.save({"positions": torch.tensor(positions), "values": torch.tensor(values)}, path)
    return path


def test_compare_matches_logical_positions_not_record_order(tmp_path):
    a = save_rows(tmp_path / "a.pt", [0, 1, 2], [[1.0], [2.0], [3.0]])
    b = save_rows(tmp_path / "b.pt", [2, 0], [[3.1], [1.0]])
    report = CHECK.compare_rows([a], [b], 0, 3)
    assert report["matched_rows"] == 2
    assert report["baseline_only_rows"] == 1
    assert report["stats"]["abs_diff"]["mean"] == pytest.approx(0.05)


def test_no_decode_comparison_after_input_divergence(tmp_path):
    a = save_rows(tmp_path / "a.pt", [10, 11], [[1.0], [200.0]])
    b = save_rows(tmp_path / "b.pt", [10, 11], [[1.0], [-500.0]])
    report = CHECK.compare_rows([a], [b], 10, 11)
    assert report["matched_rows"] == 1
    assert report["stats"]["abs_diff"]["max"] == 0


def test_duplicate_writes_are_not_silently_overwritten(tmp_path):
    file = save_rows(tmp_path / "a.pt", [0, 0], [[1.0], [2.0]])
    with pytest.raises(RuntimeError, match="Duplicate"):
        CHECK.load_rows([file], 0, 2)


def test_probe_uses_physical_slots_and_independent_tables():
    cache = torch.arange(24).reshape(3, 2, 1, 4)
    slots = PROBE.slots_for_positions(cache, torch.tensor([[2, 0, 1]]), torch.tensor([0, 1, 2]))
    assert slots.tolist() == [4, 5, 0]
    assert PROBE.rows_from_slots(cache, slots).flatten().tolist() == list(range(16, 24)) + list(range(4))


def test_probe_rejects_invalid_live_slot():
    with pytest.raises(RuntimeError, match="invalid live slots"):
        PROBE.rows_from_slots(torch.zeros(2, 2, 1, 4), torch.tensor([-1]))


def test_loaded_rows_dedup_but_current_writes_retained(tmp_path):
    rec = PROBE.KVRecorder(tmp_path, 0, 3)
    cache = torch.arange(16).reshape(2, 2, 1, 4)
    for _ in range(2):
        rec.save("layer0", "nope", "loaded", torch.tensor([0, 1, 3]), cache, torch.tensor([0, 1, 3]))
    rec.save("layer0", "nope", "current", torch.tensor([3]), cache, torch.tensor([3]))
    rec.flush()
    index = CHECK.read_index(tmp_path)
    loaded = index[("rank0", "layer0", "nope", "loaded")]
    assert len(loaded) == 1
    assert CHECK.load_rows(loaded, 0, 4)[0].tolist() == [0, 1]
    assert len(index[("rank0", "layer0", "nope", "current")]) == 1


def test_reload_cannot_pass_by_recomputing_prompt(tmp_path):
    decode = tmp_path / "decode"
    decode.mkdir()
    CHECK.write_json(decode / "output.json", {"num_cached_tokens": 0})
    with pytest.raises(RuntimeError, match="BOTH"):
        CHECK.validate_reload(tmp_path, 1024)
    with (decode / "archive_reads_1.jsonl").open("w") as file:
        for group in (0, 1):
            file.write(json.dumps({"key": str(group), "kv_group": group}) + "\n")
    with pytest.raises(RuntimeError, match="too short"):
        CHECK.validate_reload(tmp_path, 1024)
    CHECK.write_json(decode / "output.json", {"num_cached_tokens": 1023})
    assert CHECK.validate_reload(tmp_path, 1024)["cached_tokens"] == 1023


@pytest.fixture
def connector_module(monkeypatch):
    """Load real connector with CPU allocator/interface substitutes only."""
    for name in ("lmcache.v1.memory_management", "lmcache.v1.storage_backend.connector.base_connector"):
        module = ModuleType(name)
        if name.endswith("memory_management"):
            module.MemoryFormat = lambda value: value
        else:
            module.RemoteConnector = object
        monkeypatch.setitem(sys.modules, name, module)
    return load_tool("layerwise_prefill_store")


class Memory:
    def __init__(self, raw):
        self.raw_data = raw
        self.meta = SimpleNamespace(valid_tokens=2, cached_positions=torch.tensor([3, 4]))
        self.released = 0

    @property
    def byte_array(self):
        return memoryview(self.raw_data.numpy())

    def get_shapes(self):
        return [torch.Size([self.raw_data.numel()])]

    def get_dtypes(self):
        return [torch.uint8]

    def get_memory_format(self):
        return SimpleNamespace(value=1)

    def ref_count_down(self):
        self.released += 1


def make_connector(module, tmp_path, read_only=False, allocation_delta=0):
    extra = {
        "validation_archive": str(tmp_path / "archive"),
        "validation_stage_dir": str(tmp_path / "stage"),
        "validation_read_only": read_only,
    }
    allocations = []

    def allocate(shapes, dtypes, fmt, busy_loop):
        obj = Memory(torch.empty(shapes[0].numel() + allocation_delta, dtype=torch.uint8))
        allocations.append(obj)
        return obj

    config = SimpleNamespace(get_extra_config_value=lambda k, default=None: extra.get(k, default))
    connector = module.ValidationFileConnector(None, SimpleNamespace(allocate=allocate), config)
    key = SimpleNamespace(to_string=lambda: "model@8@0@abcdef@bf16@1@7", kv_group=1, layer_id=7, worker_id=0)
    return connector, key, allocations


def test_archive_roundtrip_preserves_flat_valid_tokens_and_bytes(connector_module, tmp_path):
    conn, key, allocated = make_connector(connector_module, tmp_path)
    source = Memory(torch.arange(12, dtype=torch.uint8))
    asyncio.run(conn.put(key, source))
    assert source.released == 0  # InstrumentedRemoteConnector owns the release.
    result = asyncio.run(conn.get(key))
    assert torch.equal(result.raw_data, source.raw_data)
    assert result.meta.valid_tokens == 2
    assert result.meta.cached_positions.tolist() == [3, 4]
    assert result is allocated[0]
    assert conn.exists_sync(key)
    assert conn.requires_put_completion()


def test_archive_range_positions_are_weights_only_loadable(connector_module, tmp_path):
    conn, key, _ = make_connector(connector_module, tmp_path)
    source = Memory(torch.arange(12, dtype=torch.uint8))
    source.meta.cached_positions = range(10, 12)
    asyncio.run(conn.put(key, source))
    restored = asyncio.run(conn.get(key))
    assert restored.meta.cached_positions.tolist() == [10, 11]


def test_readonly_archive_cannot_be_overwritten(connector_module, tmp_path):
    conn, key, _ = make_connector(connector_module, tmp_path, read_only=True)
    source = Memory(torch.zeros(12, dtype=torch.uint8))
    with pytest.raises(RuntimeError, match="must not modify"):
        asyncio.run(conn.put(key, source))
    assert source.released == 0  # Including on the error path.
    assert not list((tmp_path / "archive").glob("*.pt"))


def test_corrupted_archive_is_rejected(connector_module, tmp_path):
    conn, key, _ = make_connector(connector_module, tmp_path)
    asyncio.run(conn.put(key, Memory(torch.arange(12, dtype=torch.uint8))))
    path = conn.path_for(key)
    payload = torch.load(path, weights_only=True)
    payload["raw"][0] = 255
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="checksum"):
        asyncio.run(conn.get(key))


def test_reload_allocation_failure_releases_memory(connector_module, tmp_path):
    conn, key, allocated = make_connector(connector_module, tmp_path, allocation_delta=1)
    asyncio.run(conn.put(key, Memory(torch.arange(12, dtype=torch.uint8))))
    with pytest.raises(RuntimeError, match="allocation size"):
        asyncio.run(conn.get(key))
    assert allocated[0].released == 1


@pytest.fixture(scope="module")
def real_cpu_memory():
    """Execute production allocators/objects unchanged, without service imports."""
    root = TOOLS.parents[1] / "LMCache/lmcache"
    namespace = dict(
        __name__=__name__,
        abc=abc,
        ctypes=ctypes,
        dataclass=dataclass,
        Enum=Enum,
        auto=auto,
        cached_property=cached_property,
        wraps=wraps,
        threading=threading,
        torch=torch,
        SortedList=SortedList,
        LMCStatsMonitor=Mock(),
        logger=Mock(),
        _lmcache_nvtx_annotate=lambda f: f,
    )
    for path, names in (
        (root / "integration/vllm/utils.py", {"get_size_bytes"}),
        (
            root / "v1/memory_management.py",
            {
                "_group_prefix_sums",
                "synchronized",
                "MemoryFormat",
                "FreeBlock",
                "MemoryObjMetadata",
                "_TensorAllocationBatch",
                "MemoryObj",
                "TensorMemoryObj",
                "MemoryAllocatorInterface",
                "AddressManager",
                "TensorMemoryAllocator",
            },
        ),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if getattr(n, "name", None) in names]
        assert {n.name for n in nodes} == names
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("tokens", [1, 17, 32, 97, 255, 256])
@pytest.mark.parametrize("layout", ["latent", "indexer", "mixed_dtype"])
@pytest.mark.parametrize("legacy", [False, True])
def test_real_batched_store_single_reload_tail_padding(
    connector_module,
    real_cpu_memory,
    tmp_path,
    tokens,
    layout,
    legacy,
):
    api = real_cpu_memory
    connector_module.MemoryFormat = api.MemoryFormat
    allocator = api.TensorMemoryAllocator(torch.zeros(2 * 1024**2, dtype=torch.uint8))
    if layout == "latent":
        shapes, dtypes = [torch.Size([tokens * 576])], [torch.bfloat16]
        fmt = api.MemoryFormat.KV_MLA_LATENT_FMT
    elif layout == "indexer":
        shapes, dtypes = [torch.Size([tokens * 128])], [torch.bfloat16]
        fmt = api.MemoryFormat.KV_DSA_INDEX_FMT
    else:
        shapes = [torch.Size([tokens * 128]), torch.Size([tokens])]
        dtypes, fmt = [torch.int8, torch.float32], api.MemoryFormat.KV_DSA_INDEX_FMT
    sources = allocator.batched_allocate(shapes, dtypes, batch_size=2, fmt=fmt)
    source = sources[0]
    source.meta.valid_tokens = tokens
    source.meta.cached_positions = torch.arange(37 * 256, 37 * 256 + tokens)
    logical = api.get_size_bytes(shapes, dtypes)
    expected = torch.arange(logical, dtype=torch.int64).remainder(251).to(torch.uint8)
    source.raw_data.fill_(255)  # Padding must not become part of the saved KV.
    source.raw_data[:logical].copy_(expected)
    conn, key, _ = make_connector(connector_module, tmp_path)
    conn.local_cpu_backend.allocate = lambda s, d, f, busy_loop: allocator.allocate(s, d, f)
    asyncio.run(conn.put(key, source))
    path = conn.path_for(key)
    payload = torch.load(path, weights_only=True)
    assert payload["logical_bytes"] == logical
    assert torch.equal(payload["raw"], expected)
    if legacy:
        del payload["logical_bytes"]
        payload["raw"] = torch.frombuffer(bytearray(source.byte_array), dtype=torch.uint8).clone()
        payload["sha256"] = hashlib.sha256(payload["raw"].numpy().tobytes()).hexdigest()
        torch.save(payload, path)
    original_file = path.read_bytes()
    restored = asyncio.run(conn.get(key))
    assert restored.get_shapes() == shapes and restored.get_dtypes() == dtypes
    assert restored.get_num_tokens() == tokens
    assert torch.equal(restored.meta.cached_positions, source.meta.cached_positions)
    assert torch.equal(restored.raw_data, expected)
    assert source.raw_data.numel() == allocator.address_manager.compute_aligned_size(logical)
    assert restored.raw_data.numel() == logical
    if logical % api.AddressManager.ALIGN_BYTES:
        # This exact mismatch tripped the old connector's dst == raw assertion.
        assert source.raw_data.numel() > restored.raw_data.numel()
    assert path.read_bytes() == original_file  # No rewriting the sealed P archive.
    read_log = next(conn.stage_dir.glob("archive_reads_*.jsonl"))
    read = json.loads(read_log.read_text(encoding="utf-8"))
    assert read["sha256"] == payload["sha256"]
    assert read["logical_bytes"] == logical
    restored.ref_count_down()
    for obj in sources:
        obj.ref_count_down()
    assert allocator.total_allocated_size == 0


def test_reload_accepts_padded_destination_with_correct_metadata(connector_module, tmp_path):
    conn, key, _ = make_connector(connector_module, tmp_path)
    asyncio.run(conn.put(key, Memory(torch.arange(12, dtype=torch.uint8))))
    dst = Memory(torch.full((4096,), 255, dtype=torch.uint8))
    dst.get_shapes = lambda: [torch.Size([12])]
    conn.local_cpu_backend.allocate = lambda *a, **kw: dst
    restored = asyncio.run(conn.get(key))
    assert torch.equal(restored.raw_data[:12], torch.arange(12, dtype=torch.uint8))
    assert torch.all(restored.raw_data[12:] == 255)


def test_legacy_padding_is_still_checksum_checked(connector_module, tmp_path):
    conn, key, _ = make_connector(connector_module, tmp_path)
    asyncio.run(conn.put(key, Memory(torch.arange(12, dtype=torch.uint8))))
    path = conn.path_for(key)
    payload = torch.load(path, weights_only=True)
    del payload["logical_bytes"]
    payload["raw"] = torch.cat([payload["raw"], torch.zeros(4, dtype=torch.uint8)])
    payload["sha256"] = hashlib.sha256(payload["raw"].numpy().tobytes()).hexdigest()
    payload["raw"][-1] = 255
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="checksum"):
        asyncio.run(conn.get(key))


def test_truncated_source_rejected_before_publication(connector_module, tmp_path):
    conn, key, _ = make_connector(connector_module, tmp_path)
    source = Memory(torch.zeros(11, dtype=torch.uint8))
    source.get_shapes = lambda: [torch.Size([12])]
    with pytest.raises(RuntimeError, match="source is truncated"):
        asyncio.run(conn.put(key, source))
    assert not conn.exists_sync(key)
    assert source.released == 0


@pytest.mark.parametrize("legacy", [False, True])
def test_truncated_archive_cannot_be_hidden_by_padding_fix(connector_module, tmp_path, legacy):
    conn, key, allocated = make_connector(connector_module, tmp_path)
    asyncio.run(conn.put(key, Memory(torch.arange(12, dtype=torch.uint8))))
    path = conn.path_for(key)
    payload = torch.load(path, weights_only=True)
    if legacy:
        del payload["logical_bytes"]
    payload["raw"] = payload["raw"][:-1]
    payload["sha256"] = hashlib.sha256(payload["raw"].numpy().tobytes()).hexdigest()
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="truncated"):
        asyncio.run(conn.get(key))
    assert allocated == []


def test_archive_size_metadata_mismatch_rejected(connector_module, tmp_path):
    conn, key, allocated = make_connector(connector_module, tmp_path)
    asyncio.run(conn.put(key, Memory(torch.arange(12, dtype=torch.uint8))))
    path = conn.path_for(key)
    payload = torch.load(path, weights_only=True)
    payload["logical_bytes"] = 11
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="logical size metadata"):
        asyncio.run(conn.get(key))
    assert allocated == []


@pytest.mark.parametrize("read_only", [False, True])
def test_real_instrumented_put_releases_exactly_once(connector_module, tmp_path, read_only):
    source_path = TOOLS.parents[1] / "LMCache/lmcache/v1/storage_backend/connector/instrumented_connector.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "put")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {"time": SimpleNamespace(perf_counter=lambda: 0), "logger": Mock()}
    exec(compile(ast.fix_missing_locations(module), "<actual-instrumented-put>", "exec"), namespace)
    conn, key, _ = make_connector(connector_module, tmp_path, read_only)
    source = Memory(torch.arange(8, dtype=torch.uint8))
    source.get_size = lambda: 8
    wrapper = SimpleNamespace(_connector=conn, _stats_monitor=Mock(), name="test")
    if read_only:
        with pytest.raises(RuntimeError):
            asyncio.run(namespace["put"](wrapper, key, source))
    else:
        asyncio.run(namespace["put"](wrapper, key, source))
    assert source.released == 1


@pytest.mark.parametrize("request_ids", [None, ["r"]])
def test_real_worker_hooks_observe_writes_and_loads(monkeypatch, tmp_path, request_ids):
    sfa = ModuleType("vllm_ascend.attention.sfa_v1")
    sfa._dsa_indexer_layer_name = lambda name: name + ".index"
    sfa.wait_for_kv_layer_from_connector = lambda name, **kwargs: None
    index = torch.arange(4, dtype=torch.float32).reshape(2, 2, 1, 1) + 200
    parts = tuple(torch.arange(4, dtype=torch.float32).reshape(2, 2, 1, 1) + i * 100 for i in range(2))

    class Impl:
        has_indexer = True
        dsa_shrink_latent = 0

        def forward(self, layer, hidden, caches, meta, **kwargs):
            sfa.wait_for_kv_layer_from_connector(layer)
            sfa.wait_for_kv_layer_from_connector(layer + ".index")
            caches[0].view(-1)[3] = 90
            caches[1].view(-1)[3] = 91
            index.view(-1)[2] = 92
            return "output"

    sfa.AscendSFAImpl = Impl
    attention = ModuleType("vllm_ascend.attention")
    attention.sfa_v1 = sfa
    context = ModuleType("vllm.forward_context")
    context.get_forward_context = lambda: SimpleNamespace(
        virtual_engine=0, no_compile_layers={"layer.index": SimpleNamespace(kv_cache=[index])}
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention", attention)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", context)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    worker = PROBE.PrefillValidationWorker()
    worker.rank = 0
    worker.install_prefill_validation(str(tmp_path), 3)
    meta = SimpleNamespace(
        # P uses SHRINK_LATENT=0, so its real metadata has no request IDs.
        req_ids=request_ids,
        num_actual_tokens=1,
        query_start_loc_cpu=torch.tensor([0, 1]),
        seq_lens_cpu=torch.tensor([3]),
        slot_mapping=torch.tensor([3]),
        indexer_slot_mapping=torch.tensor([2]),
        block_table=torch.tensor([[0, 1]]),
        indexer_block_table=torch.tensor([[1, 0]]),
    )
    assert Impl().forward("layer", None, parts, meta) == "output"
    worker.finish_prefill_validation()
    manifest = CHECK.read_index(tmp_path)
    for part, expected in (("nope", 90), ("pe", 91), ("index", 92)):
        positions, values = CHECK.load_rows(manifest[("rank0", "layer", part, "current")], 0, 3)
        assert positions.tolist() == [2]
        assert values.reshape(-1).tolist() == [expected]
    positions, values = CHECK.load_rows(manifest[("rank0", "layer", "index", "loaded")], 0, 3)
    assert positions.tolist() == [0, 1]
    assert values.reshape(-1).tolist() == [202, 203]


@pytest.mark.parametrize("missing_prefill_trace", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_three_pass_report_keeps_small_differences_and_real_output_divergence(
    connector_module, tmp_path, missing_prefill_trace, workers
):
    CHECK.write_json(tmp_path / "run.json", {"prefill_chunk_tokens": 2})
    for stage, tokens in (("baseline", [7, 8]), ("prefill", [7]), ("decode", [7, 9])):
        directory = tmp_path / stage
        directory.mkdir()
        CHECK.write_json(
            directory / "output.json",
            {"token_ids": tokens, "prompt_sha256": "same", "num_hidden_layers": 1, "num_cached_tokens": 3},
        )
        if stage == "prefill" and missing_prefill_trace:
            continue
        rec = PROBE.KVRecorder(directory, 0, 4)
        for part in ("nope", "pe", "index"):
            cache = torch.ones(3, 2, 1, 1, dtype=torch.float64)
            if stage == "prefill":
                cache += 1e-9
            positions = torch.arange(4) if stage != "decode" else torch.tensor([3])
            rec.save("layer", part, "current", positions, cache, positions)
            if stage == "decode":
                rec.save("layer", part, "loaded", torch.arange(3), cache, torch.arange(3))
            if stage == "prefill":
                rec.save("layer", part, "loaded", torch.arange(2), cache, torch.arange(2))
            if stage != "prefill":
                rec.save("layer", part, "current", torch.tensor([4]), cache, torch.tensor([4]))
        rec.flush()
        if stage in ("baseline", "prefill"):
            io = PROBE.LayerIORecorder(directory, 0)
            io.record(
                "layer",
                0,
                0,
                4,
                {"sha256": "same", "shape": [4], "dtype": "torch.float32", "numel": 4},
                torch.ones(4, dtype=torch.float32),
            )
            io.flush()
    CHECK.write_json(tmp_path / "prompt.json", {"length": 4, "sha256": "same"})
    for stage in ("baseline", "prefill"):
        conn, key, _ = make_connector(connector_module, tmp_path / stage)
        for group in (0, 1):
            key.kv_group = group
            key.to_string = lambda group=group: f"group{group}"
            asyncio.run(conn.put(key, Memory(torch.arange(12, dtype=torch.uint8))))
    CHECK.seal_archive(tmp_path)
    sealed = json.loads((tmp_path / "prefill/archive_manifest.json").read_text())
    with (tmp_path / "decode/archive_reads_1.jsonl").open("w") as file:
        for group in (0, 1):
            file.write(
                json.dumps({"key": f"group{group}", "sha256": sealed[f"group{group}"], "kv_group": group}) + "\n"
            )
    if workers > 1:
        # Exercise real spawn/pickle/CPU workers through the public CLI, not a
        # mocked executor. Saved artifacts suffice: no model or NPU is loaded.
        process = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "layerwise_prefill_check.py"),
                "--analyse-only",
                "--run-dir",
                str(tmp_path),
                "--devices",
                "0",
                "--analysis-workers",
                str(workers),
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert process.returncode == int(missing_prefill_trace), process.stdout + process.stderr
        assert "analyse kv:" in process.stdout and "analyse archive:" in process.stdout
        result = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        progress = [json.loads(line) for line in (tmp_path / "analysis_progress.jsonl").read_text().splitlines()]
        assert {item["phase"] for item in progress} == {"kv", "archive"}
        assert all(item["pid"] > 0 and item["seconds"] >= 0 for item in progress)
        assert len([item for item in progress if item["phase"] == "kv"]) == 3
        if missing_prefill_trace:
            with pytest.raises(RuntimeError, match="Trace coverage incomplete; report saved"):
                CHECK.analyse(tmp_path, 1, workers=1)
            serial = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        else:
            serial = CHECK.analyse(tmp_path, 1, workers=1)
        for field in ("kv", "kv_by_prefill_chunk", "persisted_kv", "structural_errors", "reload", "tokens_equal"):
            assert result[field] == serial[field]
    elif missing_prefill_trace:
        with pytest.raises(RuntimeError, match="Trace coverage incomplete; report saved"):
            CHECK.analyse(tmp_path, 1, workers=1)
        result = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    else:
        result = CHECK.analyse(tmp_path, 1, workers=1)
    assert result["analysis_status"] == "complete"
    assert result["analysis_workers"] == workers
    assert result["tokens_equal"] is False
    assert result["first_different_output_token_index"] == 1
    assert result["decode_compare_position_exclusive"] == 5
    assert (tmp_path / "kv_statistics.csv").is_file()
    assert (tmp_path / "kv_chunk_statistics.csv").is_file()
    assert result["prefill_chunk_tokens"] == 2
    assert result["prefill_chunk_size_source"] == "run.json"
    reloads = [row for row in result["kv"] if row["comparison"] == "prefill_reloaded"]
    assert len(reloads) == 3
    if missing_prefill_trace:
        errors = result["structural_errors"]
        assert any("Incomplete prefill trace" in error for error in errors)
        assert any("Missing P reference trace" in error for error in errors)
        assert not any("No observed NPU reload" in error for error in errors)
        assert result["persisted_kv"]["common_keys"] == 2
        assert all(row["stats"]["abs_diff"]["max"] == 0 for row in result["persisted_kv"]["per_layer"])
        assert all(row["status"] == "not_observed" and "stats" not in row for row in reloads)
        return
    assert result["structural_errors"] == []
    written = [row for row in result["kv"] if row["comparison"] == "prefill_written"]
    assert all(0 < row["stats"]["abs_diff"]["mean"] < 1e-7 for row in written)
    assert all(row["matched_rows"] == 2 and row["stats"]["abs_diff"]["max"] == 0 for row in reloads)
    assert all(row["status"] == "compared" for row in reloads)
    last_band_reload = [
        row
        for row in result["kv_by_prefill_chunk"]
        if row["comparison"] == "prefill_reloaded" and row["chunk_index"] == 1
    ]
    assert all(row["status"] == "not_observed" and "stats" not in row for row in last_band_reload)


def test_decode_probe_batches_files_without_losing_rows_or_duplicate_writes(tmp_path):
    recorder = PROBE.KVRecorder(tmp_path, 0, 4, flush_rows=3)
    cache = torch.arange(6).reshape(3, 2, 1, 1)
    for pos in (1, 2, 2, 3):
        recorder.save("layer", "nope", "current", torch.tensor([pos]), cache, torch.tensor([pos]))
    assert recorder.count == 1
    recorder.flush()
    recorder.flush()
    assert recorder.count == 2
    files = CHECK.read_index(tmp_path)[("rank0", "layer", "nope", "current")]
    payloads = [torch.load(path, weights_only=True) for path in files]
    assert torch.cat([payload["positions"] for payload in payloads]).tolist() == [1, 2, 2, 3]
    assert torch.cat([payload["values"].reshape(-1) for payload in payloads]).tolist() == [1, 2, 2, 3]


@pytest.mark.parametrize(
    "mode, splits, valid",
    [
        ("PIECEWISE", ["vllm::mla_forward"], True),
        ("PIECEWISE", ["vllm::sfa_lmcache_retrieve"], False),
        ("FULL", [], False),
    ],
)
def test_probe_graph_mode_must_preserve_per_layer_callback(mode, splits, valid):
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name=mode), splitting_ops=splits),
    )
    if valid:
        PROBE.validate_probe_graph_mode(config)
    else:
        with pytest.raises(RuntimeError, match="mla_forward"):
            PROBE.validate_probe_graph_mode(config)
    config.model_config.enforce_eager = True
    PROBE.validate_probe_graph_mode(config)
