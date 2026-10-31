# SPDX-License-Identifier: Apache-2.0
"""Exercise replay keys against the actual Ascend-patched lookup functions.

Load CPU definitions without importing the NPU plugin's initialization hooks.
The serving hash input, key construction and serialization are not mocked.
"""

import ast
import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import prefill_kv_seed as seed  # noqa: E402


def load_module(path, name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def definitions(path, names):
    selected = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if getattr(node, "name", None) in names or (
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        ):
            selected.append(node)
    return selected


@pytest.fixture
def serving(monkeypatch):
    workspace = ROOT.parent
    lmcache = workspace / "LMCache/lmcache"
    ascend_hash = workspace / "LMCache-Ascend/lmcache_ascend/v1/tokens_hash.py"
    vllm_hash = workspace / "vllm/vllm/utils/hashing.py"
    if not all(p.exists() for p in (lmcache, ascend_hash, vllm_hash)):
        pytest.skip("Requires the four sibling source repositories")
    hashing = load_module(vllm_hash, "vllm.utils.hashing", monkeypatch)
    patched = load_module(ascend_hash, "replay_serving_hash", monkeypatch)
    module = ModuleType("replay_serving_keys")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(torch=torch, dataclass=dataclasses.dataclass, field=dataclasses.field)
    tree = ast.parse("from __future__ import annotations")
    tree.body.extend(definitions(lmcache / "utils.py", {"CacheEngineKey", "TORCH_DTYPE_TO_STR_DTYPE"}))
    tree.body.extend(
        definitions(
            lmcache / "v1/mooncake_layout.py",
            {
                "mooncake_page_key",
                "MOONCAKE_PAYLOAD_LAYOUT_TAG",
                "MOONCAKE_VALID_TOKENS_TAG",
            },
        )
    )
    tree.body.extend(
        definitions(lmcache / "v1/token_database.py", {"DSA_INDEX_CACHE_SCHEMA_TAG", "DSA_INDEX_CACHE_SCHEMA"})
    )
    classes = definitions(lmcache / "v1/token_database.py", {"TokenDatabase", "ChunkedTokenDatabase"})
    methods = {"_make_key_by_hash", "_get_init_hash", "_chunk_tokens", "_prefix_hash", "process_tokens"}
    # Use the concrete process_tokens, not the abstract base declaration.
    selected = {}
    for cls in classes:
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name in methods:
                node.decorator_list = []
                selected[node.name] = node
    tree.body.append(
        ast.ClassDef(name="Database", bases=[], keywords=[], body=list(selected.values()), decorator_list=[])
    )
    exec(compile(ast.fix_missing_locations(tree), str(lmcache), "exec"), module.__dict__)
    module.Database._hash_tokens = patched._hash_tokens
    for name in ("lmcache.utils", "lmcache.v1.mooncake_layout", "lmcache.v1.token_database"):
        monkeypatch.setitem(sys.modules, name, module)
    return module, hashing


@pytest.mark.parametrize("algorithm", ["sha256", "sha256_cbor", "sha256_cbor_64bit", "xxhash", "xxhash_cbor"])
@pytest.mark.parametrize("mixed_c8", [False, True])
@pytest.mark.parametrize("decode_receive", [False, True])
def test_500k_seed_matches_real_ascend_lookup_and_old_keys_miss(serving, algorithm, mixed_c8, decode_receive):
    api, hashing = serving
    hash_fn = hashing.get_hash_fn_by_name(algorithm.replace("sha256_cbor_64bit", "sha256_cbor"))
    api.NONE_HASH = b"serving-root"
    chunk = 1024
    tokens = [i % 100003 for i in range(500095 if decode_receive else 499712)]
    dtypes = (torch.bfloat16, torch.uint8 if mixed_c8 else torch.bfloat16)
    counts = (79, 22)
    signature = "a1b2c3d4e5f67890"
    db = api.Database()
    db.hash_func = hash_fn
    db.config = SimpleNamespace(chunk_size=chunk, save_unfull_chunk=decode_receive, dsa_two_groups=True)
    db.chunk_size = chunk
    db.mooncake_payload_layout = signature
    db.save_only_first_rank = True
    db.metadata = SimpleNamespace(model_name="/model", world_size=4, worker_id=0, get_dtypes=lambda: dtypes)
    plan = dict(
        model_path="/model",
        contract=dict(
            algorithm=algorithm,
            hash_root=repr(api.NONE_HASH),
            signature=signature,
            schema=dict(
                chunk_size=chunk,
                dtype="torch.bfloat16",
                runtime_kv_groups=[(n, []) for n in counts],
                indexer_execution={"layout": "mixed"} if mixed_c8 else None,
            ),
        ),
    )
    request_configs = None
    if decode_receive:
        plan.update(seed_mode="decode_receive", replay_tag="fresh-D-replay")
        request_configs = {"lmcache.tag.decode_replay": plan["replay_tag"]}
    injected = {key: (group, start, end) for key, group, start, end in seed.page_keys(plan, tokens)}
    looked_up = {
        api.mooncake_page_key(key, counts[group]): (group, start, end)
        for group in (0, 1)
        for start, end, key in db.process_tokens(tokens=tokens, kv_group=group, request_configs=request_configs)
    }
    assert len(injected) == (978 if decode_receive else 976)
    assert injected == looked_up
    if decode_receive:
        assert (0, 499712, 500095) in injected.values()
        assert all("@decode_replay%fresh-D-replay" in key for key in injected)
        assert sum("@internal.valid_tokens%383" in key for key in injected) == 2
        untagged = {
            api.mooncake_page_key(key, counts[group])
            for group in (0, 1)
            for _, _, key in db.process_tokens(tokens=tokens, kv_group=group)
        }
        assert untagged.isdisjoint(injected)

    # Reproduce the original failure: native put/get could agree on all of
    # these wrong keys, yet the actual P lookup misses starting at page zero.
    old_keys = set()
    previous = api.NONE_HASH
    for start in range(0, len(tokens), chunk):
        previous = hash_fn((previous, tuple(tokens[start : start + chunk]), ()))
        for group in (0, 1):
            old_keys.add(api.mooncake_page_key(db._make_key_by_hash(previous, kv_group=group), counts[group]))
    assert old_keys.isdisjoint(looked_up)


def test_seed_rejects_partial_history(serving):
    plan = dict(
        model_path="/model",
        contract=dict(
            algorithm="sha256_cbor",
            hash_root=repr(b"root"),
            signature="test",
            schema=dict(chunk_size=4, dtype="torch.bfloat16", runtime_kv_groups=[(79, []), (22, [])]),
        ),
    )
    with pytest.raises(ValueError, match="Only full historical"):
        list(seed.page_keys(plan, [1, 2, 3]))
