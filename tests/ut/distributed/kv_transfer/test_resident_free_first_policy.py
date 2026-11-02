# SPDX-License-Identifier: Apache-2.0
"""Dependency-free policy checks; these do NOT execute AscendC or NPU graphs.

Run directly with Python to avoid the NPU-specific pytest conftest:
    python tests/ut/distributed/kv_transfer/test_resident_free_first_policy.py

The reference is extracted from the NPU test by AST without importing torch.
Thus local checks and device tests use the same expected allocation policy.
"""

import ast
import itertools
import random
import sys
import unittest
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[4]
DEVICE_TEST = ROOT / "tests/e2e/nightly/single_node/ops/singlecard_ops/test_resident_sorted_cache.py"


def load_function(path, name, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.parse("from __future__ import annotations")
    module.body.append(node)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


reference_step = load_function(DEVICE_TEST, "_reference_step")


def make_shards(queries, shard_count):
    """CPU model of historical MTP union, not a substitute for the NPU union."""
    unique = set(itertools.chain.from_iterable(queries))
    return [sorted(t for t in unique if t % shard_count == s) for s in range(shard_count)]


def independent_step(shards, old, capacity):
    """Allocate from explicit free-slot/victim queues instead of index arithmetic."""
    selected = set(itertools.chain.from_iterable(shards))
    available = [slot for slot in range(capacity) if slot not in old.values()]
    victims = sorted(old.keys() - selected, key=lambda t: (t % len(shards), t))
    updated, misses, slots = old.copy(), [], []
    for token in itertools.chain.from_iterable(shards):
        if token in old:
            continue
        if available:
            slot = available.pop(0)
        else:
            slot = updated.pop(victims.pop(0))
        updated[token] = slot
        misses.append(token)
        slots.append(slot)
    return updated, misses, slots


class FreeFirstPolicyTest(unittest.TestCase):
    def check_step(self, shards, old, capacity):
        updated, misses, slots = reference_step(shards, old, capacity)
        self.assertEqual((updated, misses, slots), independent_step(shards, old, capacity))
        selected = set(itertools.chain.from_iterable(shards))
        self.assertEqual(set(misses), selected - old.keys())
        self.assertEqual(len(misses), len(set(misses)))
        self.assertEqual(len(slots), len(set(slots)))
        self.assertEqual(len(updated), min(capacity, len(old) + len(misses)))
        self.assertEqual(set(updated.values()), set(range(len(updated))))
        self.assertTrue(selected <= updated.keys())
        for token in selected & old.keys():
            self.assertEqual(updated[token], old[token])
        evicted = old.keys() - updated.keys()
        self.assertEqual(len(evicted), max(0, len(misses) - (capacity - len(old))))
        # Reconstruct state update from each shard's selected eviction prefix,
        # independently of target-slot overwrite used by reference_step.
        rebuilt = old.copy()
        for shard in range(len(shards)):
            candidates = sorted(t for t in old.keys() - selected if t % len(shards) == shard)
            count = sum(t % len(shards) == shard for t in evicted)
            self.assertEqual(set(candidates[:count]), {t for t in evicted if t % len(shards) == shard})
            for token in candidates[:count]:
                del rebuilt[token]
        rebuilt.update(zip(misses, slots, strict=True))
        self.assertEqual(rebuilt, updated)
        return updated, misses, slots

    def test_free_slot_preserves_old_kv_for_next_query(self):
        old = {2: 0, 6: 1, 7: 2}
        updated, misses, slots = self.check_step(make_shards([[6, 9]], 2), old, 4)
        self.assertEqual((misses, slots), ([9], [3]))
        self.assertEqual(updated[2], 0)
        _, misses, _ = self.check_step(make_shards([[2, 9]], 2), updated, 4)
        self.assertEqual(misses, [])

    def test_partial_space_evicts_only_the_shortfall(self):
        old = {2: 0, 6: 1, 7: 2}
        updated, misses, slots = self.check_step(make_shards([[6, 9, 10]], 1), old, 4)
        self.assertEqual((misses, slots), ([9, 10], [3, 0]))
        self.assertEqual(updated[7], 2)

    def test_full_cache_uses_cross_shard_victim(self):
        old = {0: 0, 1: 1, 2: 2, 3: 3}
        updated, misses, slots = self.check_step(make_shards([[1, 2, 3, 5]], 4), old, 4)
        self.assertEqual((misses, slots), ([5], [0]))
        self.assertNotIn(0, updated)

    def test_mtp_duplicates_have_one_destination(self):
        queries = [[6, 9, 12], [9, 12, 6]]
        updated, misses, slots = self.check_step(make_shards(queries, 4), {6: 0}, 6)
        self.assertEqual(len(misses), 2)
        self.assertEqual(len(slots), 2)
        remapped = [[updated[t] for t in query] for query in queries]
        self.assertEqual(remapped[0][1], remapped[1][0])
        self.assertEqual(remapped[0][2], remapped[1][1])

    def test_no_miss_keeps_unselected_residents(self):
        old = {2: 0, 6: 1, 7: 2}
        updated, misses, slots = self.check_step(make_shards([[6]], 4), old, 4)
        self.assertEqual(updated, old)
        self.assertEqual((misses, slots), ([], []))

    def test_empty_selection_keeps_old_records(self):
        old = {2: 0, 6: 1}
        self.assertEqual(self.check_step(make_shards([[]], 4), old, 4), (old, [], []))

    def test_full_cold_load_then_reuse(self):
        shards = make_shards([[0, 1, 2, 3]], 4)
        updated, misses, _ = self.check_step(shards, {}, 4)
        self.assertEqual(len(misses), 4)
        self.assertEqual(self.check_step(shards, updated, 4)[1], [])

    def test_exhaustive_small_cache(self):
        subsets = [c for n in range(5) for c in itertools.combinations(range(6), n)]
        checked = 0
        for old_tokens, selected, shards in itertools.product(subsets, subsets, (1, 2, 4)):
            old = {t: slot for slot, t in enumerate(reversed(old_tokens))}
            self.check_step(make_shards([selected], shards), old, 4)
            checked += 1
        self.assertEqual(checked, 9747)

    def test_random_mtp_sequences(self):
        rng = random.Random(23)
        for capacity, shards in itertools.product((4, 8, 32), (1, 2, 4, 8)):
            resident = {}
            for _ in range(200):
                selected = rng.sample(range(capacity * 4), rng.randrange(capacity + 1))
                rng.shuffle(selected)
                queries = [selected, list(reversed(selected[::2]))]
                resident, _, _ = self.check_step(make_shards(queries, shards), resident, capacity)

    def test_shared_plan_model_preserves_each_layers_payload(self):
        # Model-only verification: the hardware test checks real SharedResidentPlan.
        resident, caches = {}, [{} for _ in range(3)]
        for selected in ([2, 6], [6, 9], [2, 9], [1, 5, 7, 9], [1, 5, 7, 9]):
            resident, misses, slots = self.check_step(make_shards([selected], 2), resident, 4)
            for layer, cache in enumerate(caches):
                cache.update({slot: (layer, token) for token, slot in zip(misses, slots, strict=True)})
                for token in selected:
                    self.assertEqual(cache[resident[token]], (layer, token))

    def test_coordinator_passes_static_capacity_without_device_readback(self):
        calls = []
        fake_op = lambda *args: calls.append(args)
        fake_torch = SimpleNamespace(ops=SimpleNamespace(_C_ascend=SimpleNamespace(
            npu_dsa_resident_finalize_coordinator_=fake_op)))
        function = load_function(
            ROOT / "vllm_ascend/distributed/kv_transfer/sparse_offload/resident_sorted_cache.py",
            "coordinate_sorted_resident_finalize_", {"torch": fake_torch})
        # These objects expose no .item(), .cpu() or runtime contents.
        workspace = SimpleNamespace(shard_counts=object(), miss_counts=object(),
                                    miss_tokens=SimpleNamespace(shape=(2, 4096)))
        function(workspace)
        self.assertEqual(calls, [(workspace.shard_counts, workspace.miss_counts, 4096)])

    def test_device_entrypoints_configure_after_loading_before_tests(self):
        fixtures = (
            (DEVICE_TEST, "_load_sorted_resident_ops"),
            (DEVICE_TEST.with_name("test_shared_resident_plan.py"), "_load_shared_resident_ops"),
        )
        for path, name in fixtures:
            node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
            decorator = node.decorator_list[0]
            self.assertEqual(ast.unparse(decorator.func), "pytest.fixture")
            options = {kw.arg: ast.literal_eval(kw.value) for kw in decorator.keywords}
            self.assertEqual(options, {"scope": "module", "autouse": True})
            for loaded in (False, True):
                with self.subTest(fixture=name, loaded=loaded):
                    events = []

                    def enable():
                        events.append("load")
                        return loaded

                    configure = Mock(side_effect=lambda: events.append("configure"))
                    fake_torch = Mock()
                    fixture = load_function(path, name, {
                        "pytest": SimpleNamespace(fixture=lambda **kw: lambda fn: fn, fail=self.fail),
                        "torch": fake_torch,
                        "enable_custom_op": enable,
                        "configure_resident_kernels": configure,
                    })
                    if loaded:
                        fixture()
                        configure.assert_called_once_with()
                        self.assertEqual(events, ["load", "configure"])
                    else:
                        with self.assertRaises(AssertionError):
                            fixture()
                        configure.assert_not_called()
                    # Setup must not launch a planner before kernel selection.
                    self.assertEqual(fake_torch.mock_calls, [])

    def test_kernel_configuration_forwards_both_modes_once(self):
        # Execute the actual Python selector with a mocked C++ binding. This
        # verifies flag forwarding/caching, not execution of either NPU kernel.
        for enabled in (False, True):
            with self.subTest(vectorized=enabled):
                setter = Mock()
                fake_torch = SimpleNamespace(ops=SimpleNamespace(_C_ascend=SimpleNamespace(
                    configure_dsa_resident_exact_kernels=setter)))
                package = ModuleType("vllm_ascend")
                package.envs = SimpleNamespace(VLLM_ASCEND_DSA_RESIDENT_EXACT_KERNELS=enabled)
                configure = load_function(
                    ROOT / "vllm_ascend/distributed/kv_transfer/sparse_offload/resident_sorted_cache.py",
                    "configure_resident_kernels", {"torch": fake_torch, "lru_cache": lru_cache})
                with patch.dict(sys.modules, {"vllm_ascend": package}):
                    configure()
                    configure()
                setter.assert_called_once_with(enabled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
