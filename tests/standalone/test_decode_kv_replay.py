# SPDX-License-Identifier: Apache-2.0
"""CPU round trips through the actual replay SDK, including partial chunks."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
import decode_kv_replay as R  # noqa: E402
import decode_kv_replay_store as S  # noqa: E402


def capture(root, rank=0, size=1, count=5):
    directory = root / "request" / f"rank{rank}"
    directory.mkdir(parents=True)
    tokens = list(range(count))
    (directory / "tokens.json").write_text(json.dumps(tokens))
    files = []
    for group, layers in enumerate((2, 2)):
        for layer in range(layers):
            for component in range(2):
                # Planar K/V or C8 key/scale. Intentionally non-uniform rows.
                name = f"g{group}-l{layer}-c{component}.bin"
                values = bytes((group * 30 + layer * 10 + component * 3 + t) % 256 for t in tokens)
                (directory / name).write_bytes(values)
                files.append(
                    dict(
                        group=group,
                        layer=layer,
                        component=component,
                        row_bytes=1,
                        dtype="torch.uint8",
                        path=name,
                        start=0,
                        end=count,
                        bytes=count,
                    )
                )
    (directory / "manifest.json").write_text(
        json.dumps(
            dict(
                format="decode-kv-raw-v1",
                status="complete",
                request_id="request",
                tp_rank=rank,
                tp_size=size,
                tokens=count,
                byteorder=sys.byteorder,
                groups=[2, 2],
                layer_names=[["l0", "l1"], ["i0", "i1"]],
                files=files,
                replay_file="replay.json",
            )
        )
    )
    (directory / "replay.json").write_text(
        json.dumps(
            dict(
                request=dict(schema=1, prompt_token_ids=tokens + [99], sampling={"temperature": 0}, cache_salt=None),
                runtime=dict(
                    engine_options=dict(
                        model="model",
                        tokenizer="model",
                        tensor_parallel_size=size,
                        enforce_eager=False,
                        max_model_len=count + 1000,
                        additional_config={},
                    ),
                    environment={},
                    chunk_size=4,
                ),
            )
        )
    )
    return directory


def make_plan(root, output, size=1, count=5):
    for rank in range(size):
        capture(root, rank, size, count)
    output.mkdir()
    plan = R.build_plan([root], "request", output)
    path = output / "plan.json"
    path.write_text(json.dumps(plan))
    return path, plan


class Key:
    def __init__(self, group, start, end, worker=0, layer=None):
        self.group, self.start, self.end, self.worker, self.layer = group, start, end, worker, layer

    def to_string(self):
        return f"{self.group}:{self.start}:{self.end}:{self.worker}:{self.layer}"

    def with_new_worker_id(self, worker):
        return Key(self.group, self.start, self.end, worker)

    def split_layers(self, count):
        return [Key(self.group, self.start, self.end, self.worker, i) for i in range(count)]


def bind_store(monkeypatch, path, rank=0):
    class Database:
        def __init__(self, config, metadata):
            self.size, self.rank = config.chunk_size, metadata.worker_id

        def process_tokens(self, tokens, kv_group):
            for start in range(0, len(tokens), self.size):
                end = min(len(tokens), start + self.size)
                yield start, end, Key(kv_group, start, end, self.rank)

    monkeypatch.setitem(sys.modules, "lmcache.v1.mooncake_layout", NS(mooncake_page_key=lambda key, n: key.to_string()))
    monkeypatch.setitem(sys.modules, "lmcache.v1.token_database", NS(ChunkedTokenDatabase=Database))
    store = S.SnapshotStore(str(path))
    store.bind(
        NS(chunk_size=4),
        NS(
            worker_id=rank,
            runtime_kv_group_layer_counts=(2, 2),
            runtime_kv_group_layer_names=(("l0", "l1"), ("i0", "i1")),
        ),
    )
    return store


def test_raw_components_to_multi_buffer_roundtrip_and_partial_chunk(tmp_path, monkeypatch):
    path, plan = make_plan(tmp_path / "data", tmp_path / "out", size=2)
    store = bind_store(monkeypatch, path, rank=1)
    for group in (0, 1):
        for start, end in ((0, 4), (4, 5)):
            key = Key(group, start, end).to_string()
            expected = b"".join(
                Path(row["absolute_path"]).read_bytes()[start:end]
                for row in plan["workers"]["1"]["files"]
                if row["group"] == group
            )
            # Split through a plane boundary, as real scatter destinations can.
            buffers = [
                torch.full((3,), 255, dtype=torch.uint8),
                torch.full((len(expected) - 3,), 255, dtype=torch.uint8),
            ]
            store.memory.bind(buffers)
            result = store.batch_get_into_multi_buffers(
                [key], [[x.data_ptr() for x in buffers]], [[x.numel() for x in buffers]]
            )
            assert result == [len(expected)]
            assert b"".join(x.numpy().tobytes() for x in buffers) == expected
    assert store.is_exist(Key(0, 5, 6).to_string()) == 0
    with pytest.raises(ValueError, match="layout mismatch"):
        store.batch_get_into_multi_buffers([Key(0, 0, 4).to_string()], [[buffers[0].data_ptr()]], [[3]])
    with pytest.raises(RuntimeError, match="Only unsealed P"):
        store.batch_put_from(["key"], [buffers[0].data_ptr()], [3])


def test_layer_key_only_reads_that_layer(tmp_path, monkeypatch):
    path, _ = make_plan(tmp_path / "data", tmp_path / "out")
    store = bind_store(monkeypatch, path)
    target = torch.zeros(8, dtype=torch.uint8)
    store.memory.bind([target])
    assert store.batch_get_into([Key(1, 0, 4, layer=1).to_string()], [target.data_ptr()], [8]) == [8]
    assert target.tolist() == [40, 41, 42, 43, 43, 44, 45, 46]


def test_truncated_payload_after_validation_fails(tmp_path, monkeypatch):
    path, plan = make_plan(tmp_path / "data", tmp_path / "out")
    store = bind_store(monkeypatch, path)
    Path(plan["workers"]["0"]["files"][0]["absolute_path"]).write_bytes(b"x")
    target = torch.zeros(16, dtype=torch.uint8)
    store.memory.bind([target])
    with pytest.raises(ValueError, match="truncated"):
        store.batch_get_into([Key(0, 0, 4).to_string()], [target.data_ptr()], [16])


def test_missing_replay_metadata_and_tp_disagreement_fail(tmp_path):
    first = capture(tmp_path, 0, 2)
    second = capture(tmp_path, 1, 2)
    payload = second / "g0-l0-c0.bin"
    payload.write_bytes(b"BAD!!")
    with pytest.raises(ValueError, match="latent bytes differ"):
        R.build_plan([tmp_path], "request", tmp_path / "out")
    meta = json.loads((first / "manifest.json").read_text())
    meta.pop("replay_file")
    (first / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="predates replay"):
        R.build_plan([tmp_path], "request", tmp_path / "out")


def test_500k_prefix_and_config_never_retokenizes_or_calls_P(tmp_path):
    path, plan = make_plan(tmp_path / "data", tmp_path / "out", count=500000)
    args = NS(devices=None, cpu_cache_gb=None, timeout=7200, model=None, gpu_memory_utilization=0.9, eager=False)
    options, env = R.configure(plan, args)
    assert len(plan["received_tokens"]) == 500000
    assert options["enforce_eager"] is False
    assert options["kv_transfer_config"]["kv_role"] == "kv_consumer"
    assert options["kv_transfer_config"]["kv_connector_extra_config"] == {"skip_last_n_tokens": 1}
    assert options["data_parallel_size"] == 1
    assert env["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] == "false"
    assert "LMCACHE_DECODE_KV_SNAPSHOT" not in env
    assert json.loads(env["LMCACHE_EXTRA_CONFIG"])["decode_snapshot_replay"]["plan"] == str(path)


def test_bind_hook_applies_to_first_connector(tmp_path, monkeypatch):
    path, _ = make_plan(tmp_path / "data", tmp_path / "out")
    events = []

    class Connector:
        def __init__(self, local_cpu_backend, lmcache_config=None, lmcache_metadata=None):
            self.store = NS(bind=lambda config, metadata: events.append((config, metadata)))

    monkeypatch.setattr(S, "observe_buffers", lambda cls: None)
    S.bind_connector(Connector)
    Connector(NS(config="config", metadata="metadata"))
    assert events == [("config", "metadata")]


def test_install_does_not_import_lmcache_early(tmp_path):
    code = (
        "import os,sys,json; "
        f"sys.path.insert(0,{str(TOOLS)!r}); "
        "os.environ['LMCACHE_EXTRA_CONFIG']=json.dumps({'decode_snapshot_replay':{'plan':'unused'}}); "
        "from decode_kv_replay_store import install; install(); install(); "
        "assert not any(x=='lmcache' or x.startswith('lmcache.') for x in sys.modules); "
        "assert len([x for x in sys.meta_path if type(x).__name__=='ConnectorHook'])==1"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_lazy_import_hook_binds_the_first_store_instance(tmp_path):
    path, _ = make_plan(tmp_path / "data", tmp_path / "out")
    package = tmp_path / "modules"
    location = package
    for part in ("lmcache", "v1", "storage_backend", "connector"):
        location = location / part
        location.mkdir(parents=True)
        (location / "__init__.py").write_text("")
    (location / "mooncakestore_connector.py").write_text(
        "from mooncake.store import MooncakeDistributedStore\n"
        "class MooncakestoreConnector:\n"
        " def __init__(self, local_cpu_backend, lmcache_config, lmcache_metadata):\n"
        "  self.store=MooncakeDistributedStore()\n"
        " def _register_cpu_buffer(self): pass\n"
        " def _register_external_owners(self, owners): pass\n"
    )
    code = (
        "import os,sys,json; "
        f"sys.path[:0]=[{str(package)!r},{str(TOOLS)!r}]; "
        f"os.environ['LMCACHE_EXTRA_CONFIG']=json.dumps({{'decode_snapshot_replay':{{'plan':{str(path)!r}}}}}); "
        "import decode_kv_replay_store as s; s.install(); "
        "s.SnapshotStore.bind=lambda self,c,m: setattr(self,'bound',(c,m)); "
        "from lmcache.v1.storage_backend.connector.mooncakestore_connector import MooncakestoreConnector; "
        "first=MooncakestoreConnector(None,'config','metadata'); "
        "assert first.store.bound==('config','metadata'); "
        "assert not any(type(x).__name__=='ConnectorHook' for x in sys.meta_path)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
