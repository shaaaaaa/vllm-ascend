# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from vllm_ascend.attention.selected_prefill_policy import (
    packed_main_c8_enabled,
    selected_prefill_ineligibility_reason,
    selected_prefill_requested,
)


class TestSelectedPrefillPolicy(unittest.TestCase):
    def setUp(self):
        self.supported = {
            "requested": True,
            "layerwise_prefill_p_node": True,
            "enforce_eager": True,
            "tensor_parallel_size": 1,
            "kv_group_count": 2,
            "dsa_shared_pool": True,
            "is_lmcache_connector": True,
            "main_kv_layout_supported": True,
            "registered_source_ready": True,
            "bank_writer_ready": True,
            "reader_retirement_ready": True,
        }

    def test_only_explicit_true_requests_selected_prefill(self):
        self.assertFalse(selected_prefill_requested({}))
        for value in (False, None, "true", 1):
            with self.subTest(value=value):
                self.assertFalse(selected_prefill_requested({"enable_selected_prefill": value}))
        self.assertTrue(selected_prefill_requested({"enable_selected_prefill": True}))

    def test_native_remains_selected_without_opt_in(self):
        facts = {**self.supported, "requested": selected_prefill_requested({})}
        self.assertEqual(selected_prefill_ineligibility_reason(**facts), "Selected prefill is disabled.")

    def test_each_missing_requirement_retains_native(self):
        unsupported = (
            ("layerwise_prefill_p_node", False, "P node"),
            ("enforce_eager", False, "eager"),
            ("tensor_parallel_size", 8, "parallel size 1"),
            ("kv_group_count", 1, "two KV groups"),
            ("dsa_shared_pool", False, "shared pool"),
            ("is_lmcache_connector", False, "LMCache connector"),
            ("main_kv_layout_supported", False, "main KV layout"),
            ("registered_source_ready", False, "registered LMCache sources"),
            ("bank_writer_ready", False, "bank writer"),
            ("reader_retirement_ready", False, "reader retirement"),
        )
        for field, value, expected_reason in unsupported:
            with self.subTest(field=field):
                reason = selected_prefill_ineligibility_reason(**{**self.supported, field: value})
                self.assertIsNotNone(reason)
                self.assertIn(expected_reason, reason)

    def test_connector_evidence_is_required(self):
        missing_evidence = (
            ("is_lmcache_connector", "LMCache connector"),
            ("main_kv_layout_supported", "main KV layout"),
            ("registered_source_ready", "registered LMCache sources"),
            ("bank_writer_ready", "bank writer"),
            ("reader_retirement_ready", "reader retirement"),
        )
        for field, expected_reason in missing_evidence:
            with self.subTest(field=field):
                facts = {**self.supported}
                del facts[field]
                reason = selected_prefill_ineligibility_reason(**facts)
                self.assertIsNotNone(reason)
                self.assertIn(expected_reason, reason)

    def test_validated_capabilities_allow_step_preparation(self):
        self.assertIsNone(selected_prefill_ineligibility_reason(**self.supported))

    def test_packed_main_marker_defaults_to_disabled(self):
        self.assertFalse(packed_main_c8_enabled(SimpleNamespace()))
        self.assertFalse(packed_main_c8_enabled(SimpleNamespace(cache_sparse_sfa_c8=False)))
        self.assertTrue(packed_main_c8_enabled(SimpleNamespace(cache_sparse_sfa_c8=True)))

    def test_packed_main_marker_rejects_non_boolean_values(self):
        for marker in (0, 1, None, "false"):
            with self.subTest(marker=marker):
                with self.assertRaisesRegex(ValueError, "marker must be a boolean"):
                    packed_main_c8_enabled(SimpleNamespace(cache_sparse_sfa_c8=marker))
