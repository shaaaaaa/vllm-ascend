import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    ProfilerConfig,
    VllmConfig,
)

from tests.ut.base import TestBase

init_cached_hf_modules_path = "vllm.utils.import_utils.init_cached_hf_modules"


class TestNPUWorker(TestBase):
    def setUp(self):
        """Setup test environment"""
        # Create configuration mocks
        self.cache_config_mock = MagicMock(spec=CacheConfig)
        self.cache_config_mock.cache_dtype = "auto"

        self.model_config_mock = MagicMock(spec=ModelConfig)
        self.model_config_mock.dtype = torch.float16
        self.model_config_mock.trust_remote_code = False

        self.hf_config_mock = MagicMock()
        self.hf_config_mock.model_type = "test_model"
        if hasattr(self.hf_config_mock, "index_topk"):
            delattr(self.hf_config_mock, "index_topk")

        self.model_config_mock.hf_config = self.hf_config_mock

        self.parallel_config_mock = MagicMock(spec=ParallelConfig)

        self.vllm_config_mock = MagicMock(spec=VllmConfig)
        self.vllm_config_mock.cache_config = self.cache_config_mock
        self.vllm_config_mock.model_config = self.model_config_mock
        self.vllm_config_mock.parallel_config = self.parallel_config_mock
        self.vllm_config_mock.additional_config = None
        self.vllm_config_mock.load_config = None
        self.vllm_config_mock.scheduler_config = None
        self.vllm_config_mock.device_config = None
        self.vllm_config_mock.compilation_config = None

        self.local_rank = 0
        self.rank = 0
        self.distributed_init_method = "tcp://localhost:12345"
        self.is_driver_worker = False

    def test_mooncake_placement_info_reports_tp0_with_routable_dp_rank(self):
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        worker.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_rank=3)
        )
        worker.get_kv_connector_handshake_metadata = MagicMock(
            return_value={
                0: SimpleNamespace(
                    tp_rank=0,
                    local_ip="7.150.7.133",
                    te_rpc_port=12345,
                )
            }
        )

        self.assertEqual(
            worker.get_mooncake_placement_info(),
            {"dp_rank": 3, "segment": "7.150.7.133:12345"},
        )

        worker.vllm_config.parallel_config.local_engines_only = True
        worker.vllm_config.parallel_config.data_parallel_rank_local = 0
        self.assertEqual(
            worker.get_mooncake_placement_info(),
            {"dp_rank": 0, "segment": "7.150.7.133:12345"},
        )

    def test_mooncake_placement_info_ignores_non_tp0_and_invalid_metadata(self):
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        worker.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_rank=3)
        )
        for metadata in (
            None,
            SimpleNamespace(tp_rank=1, local_ip="host", te_rpc_port=1),
            SimpleNamespace(tp_rank=0, local_ip="", te_rpc_port=1),
            SimpleNamespace(tp_rank=0, local_ip="host", te_rpc_port=0),
        ):
            worker.get_kv_connector_handshake_metadata = MagicMock(
                return_value=None if metadata is None else {0: metadata}
            )
            with self.subTest(metadata=metadata):
                self.assertIsNone(worker.get_mooncake_placement_info())

    def test_remote_fill_placement_uses_its_routable_dp_identity(self):
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        worker.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_rank=1,
                data_parallel_rank_local=0,
                data_parallel_index=5,
                local_engines_only=True,
            )
        )
        worker.get_kv_connector_handshake_metadata = MagicMock(
            return_value=None
        )
        remote_fill = {
            "enabled": True,
            "dp_rank": 5,
            "tp_rank": 0,
        }
        connector = SimpleNamespace(
            get_remote_fill_placement_info=lambda: remote_fill
        )

        with (
            patch(
                "vllm_ascend.worker.worker.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_ascend.worker.worker.get_kv_transfer_group",
                return_value=connector,
            ),
        ):
            self.assertEqual(
                worker.get_mooncake_placement_info(),
                {
                    "dp_rank": 5,
                    "api_dp_rank": 0,
                    "segment": None,
                    "remote_fill": remote_fill,
                },
            )

    def test_remote_fill_fatal_latch_exits_before_health_probe(self):
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        connector = SimpleNamespace(
            remote_fill_requires_paired_restart=lambda: True
        )
        with (
            patch(
                "vllm_ascend.worker.worker.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_ascend.worker.worker.get_kv_transfer_group",
                return_value=connector,
            ),
            patch("subprocess.run") as run,
            self.assertRaises(SystemExit) as raised,
        ):
            worker.check_health()
        self.assertEqual(raised.exception.code, 86)
        run.assert_not_called()

    def test_remote_fill_fatal_latched_during_health_probe_still_exits(self):
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        worker.local_rank = 0
        fatal = MagicMock(side_effect=(False, True))
        connector = SimpleNamespace(
            remote_fill_requires_paired_restart=fatal
        )
        result = SimpleNamespace(returncode=0, stdout="Health : OK", stderr="")
        with (
            patch(
                "vllm_ascend.worker.worker.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_ascend.worker.worker.get_kv_transfer_group",
                return_value=connector,
            ),
            patch("subprocess.run", return_value=result),
            self.assertRaises(SystemExit),
        ):
            worker.check_health()
        self.assertEqual(fatal.call_count, 2)

    def test_execute_model_converts_new_remote_fill_fatal_to_system_exit(self):
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        worker._pp_send_work = []
        worker.model_runner = MagicMock()
        worker.model_runner.execute_model.side_effect = RuntimeError("finalize failed")
        fatal = MagicMock(side_effect=(False, True))
        connector = SimpleNamespace(
            remote_fill_requires_paired_restart=fatal
        )
        scheduler_output = SimpleNamespace(total_num_scheduled_tokens=0)
        with (
            patch(
                "vllm_ascend.worker.worker.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_ascend.worker.worker.get_kv_transfer_group",
                return_value=connector,
            ),
            self.assertRaises(SystemExit),
        ):
            worker.execute_model(scheduler_output)

        worker.model_runner.abort_kv_connector_finalize.assert_called_once_with()
        self.assertEqual(fatal.call_count, 2)

    def test_execute_model_checks_remote_fill_fatal_after_success(self):
        from vllm.v1.outputs import ModelRunnerOutput

        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        worker._pp_send_work = []
        worker.model_runner = MagicMock()
        worker.model_runner.execute_model.return_value = MagicMock(
            spec=ModelRunnerOutput
        )
        fatal = MagicMock(side_effect=(False, True))
        connector = SimpleNamespace(
            remote_fill_requires_paired_restart=fatal
        )
        scheduler_output = SimpleNamespace(total_num_scheduled_tokens=0)
        with (
            patch(
                "vllm_ascend.worker.worker.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_ascend.worker.worker.get_kv_transfer_group",
                return_value=connector,
            ),
            self.assertRaises(SystemExit),
        ):
            worker.execute_model(scheduler_output)

        self.assertEqual(fatal.call_count, 2)

    def test_execute_model_traces_only_first_nonempty_batch(self):
        from vllm.v1.outputs import ModelRunnerOutput

        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        worker._pp_send_work = []
        worker._raise_if_remote_fill_restart_required = MagicMock()
        worker.model_runner = MagicMock()
        output = worker.model_runner.execute_model.return_value = MagicMock(
            spec=ModelRunnerOutput
        )
        first = SimpleNamespace(
            total_num_scheduled_tokens=1,
            num_scheduled_tokens={"first": 1},
            kv_connector_metadata=None,
        )
        second = SimpleNamespace(
            total_num_scheduled_tokens=1,
            num_scheduled_tokens={"second": 1},
            kv_connector_metadata=None,
        )

        with (
            patch(
                "vllm_ascend.worker.worker.cold_perf_enabled",
                return_value=True,
            ),
            patch("vllm_ascend.worker.worker.mark_cold_perf_requests") as mark,
            patch(
                "vllm_ascend.worker.worker.is_cold_perf_request",
                side_effect=lambda req_id: req_id == "first",
            ),
            patch("vllm_ascend.worker.worker.log_cold_perf_event") as log,
            patch(
                "vllm_ascend.worker.worker.faulthandler.dump_traceback_later"
            ) as arm,
            patch(
                "vllm_ascend.worker.worker.faulthandler.cancel_dump_traceback_later"
            ) as cancel,
            patch(
                "vllm_ascend.worker.worker.get_pp_group",
                return_value=SimpleNamespace(is_first_rank=True),
            ),
            patch(
                "vllm_ascend.worker.worker.get_tp_group",
                return_value=SimpleNamespace(rank_in_group=3),
            ),
        ):
            self.assertIs(worker.execute_model(first), output)
            self.assertIs(worker.execute_model(second), output)

        mark.assert_called_once_with(("first",))
        self.assertEqual(
            [call.args[0] for call in log.call_args_list],
            [
                "decoder_execute_rpc_entry",
                "decoder_execute_stall_watchdog_armed",
                "decoder_execute_rpc_return",
            ],
        )
        arm.assert_called_once_with(75)
        cancel.assert_called_once_with()
        self.assertEqual(log.call_args_list[0].kwargs["tp_rank"], 3)
        self.assertEqual(log.call_args_list[1].kwargs["timeout_seconds"], 75)

    def test_execute_model_reports_only_slow_execution_and_submission_gaps(self):
        from vllm.v1.outputs import ModelRunnerOutput

        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        worker._pp_send_work = []
        worker._raise_if_remote_fill_restart_required = MagicMock()
        worker.model_runner = MagicMock()
        worker.model_runner.execute_model.return_value = MagicMock(
            spec=ModelRunnerOutput
        )
        scheduler_output = SimpleNamespace(
            total_num_scheduled_tokens=1,
            num_scheduled_tokens={"warm": 1},
            kv_connector_metadata=None,
        )

        with (
            patch(
                "vllm_ascend.worker.worker.cold_perf_enabled",
                return_value=True,
            ),
            patch(
                "vllm_ascend.worker.worker.is_cold_perf_request",
                return_value=False,
            ),
            patch(
                "vllm_ascend.worker.worker.time.perf_counter",
                side_effect=(1.0, 1.1, 2.0, 2.8),
            ),
            patch(
                "vllm_ascend.worker.worker.log_cold_perf_process_event"
            ) as log,
            patch(
                "vllm_ascend.worker.worker.get_tp_group",
                return_value=SimpleNamespace(rank_in_group=2),
            ),
            patch(
                "vllm_ascend.worker.worker.get_pp_group",
                return_value=SimpleNamespace(is_first_rank=True),
            ),
        ):
            worker.execute_model(scheduler_output)
            worker.execute_model(scheduler_output)

        self.assertEqual(
            [call.args[0] for call in log.call_args_list],
            ["decoder_submission_gap", "decoder_execute_slow"],
        )
        self.assertEqual(log.call_args_list[0].kwargs["gap_ms"], 900.0)
        self.assertEqual(log.call_args_list[1].kwargs["elapsed_ms"], 800.0)
        self.assertEqual(log.call_args_list[1].kwargs["tp_rank"], 2)

    def test_sample_tokens_checks_fatal_latch_after_success_and_failure(self):
        from vllm_ascend.worker.worker import NPUWorker

        for sample_error in (None, RuntimeError("sample failed")):
            with self.subTest(sample_error=sample_error):
                worker = NPUWorker.__new__(NPUWorker)
                worker.model_runner = MagicMock()
                if sample_error is None:
                    worker.model_runner.sample_tokens.return_value = object()
                else:
                    worker.model_runner.sample_tokens.side_effect = sample_error
                fatal = MagicMock(side_effect=(False, True))
                connector = SimpleNamespace(
                    remote_fill_requires_paired_restart=fatal
                )
                with (
                    patch(
                        "vllm_ascend.worker.worker.has_kv_transfer_group",
                        return_value=True,
                    ),
                    patch(
                        "vllm_ascend.worker.worker.get_kv_transfer_group",
                        return_value=connector,
                    ),
                    self.assertRaises(SystemExit),
                ):
                    worker.sample_tokens(MagicMock())

                if sample_error is None:
                    worker.model_runner.abort_kv_connector_finalize.assert_not_called()
                else:
                    worker.model_runner.abort_kv_connector_finalize.assert_called_once_with()
                self.assertEqual(fatal.call_count, 2)

    def test_sample_tokens_traces_marked_cold_request(self):
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        worker.model_runner = MagicMock()
        worker.model_runner._cold_perf_current_req_ids = ("cold",)
        worker.model_runner.sample_tokens.return_value = "output"
        worker._raise_if_remote_fill_restart_required = MagicMock()

        with (
            patch("vllm_ascend.worker.worker.log_cold_perf_event") as log,
            patch(
                "vllm_ascend.worker.worker.faulthandler.dump_traceback_later"
            ) as arm,
            patch(
                "vllm_ascend.worker.worker.faulthandler.cancel_dump_traceback_later"
            ) as cancel,
            patch("vllm_ascend.worker.worker.forget_cold_perf_request") as forget,
            patch(
                "vllm_ascend.worker.worker.get_tp_group",
                return_value=SimpleNamespace(rank_in_group=2),
            ),
        ):
            output = worker.sample_tokens(MagicMock())

        self.assertEqual(output, "output")
        self.assertEqual(
            [call.args[0] for call in log.call_args_list],
            [
                "decoder_sample_rpc_entry",
                "decoder_sample_stall_watchdog_armed",
                "decoder_sample_rpc_return",
            ],
        )
        arm.assert_called_once_with(70)
        cancel.assert_called_once_with()
        forget.assert_called_once_with("cold")
        self.assertEqual(log.call_args_list[0].kwargs["tp_rank"], 2)
        self.assertEqual(log.call_args_list[1].kwargs["timeout_seconds"], 70)

    def test_sample_tokens_propagates_followup_trace_without_watchdog(self):
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        worker.model_runner = MagicMock()
        worker.model_runner._cold_perf_current_req_ids = ()
        worker.model_runner._cold_perf_sample_trace_req_ids = ("followup",)
        output = SimpleNamespace()
        worker.model_runner.sample_tokens.return_value = output
        worker._raise_if_remote_fill_restart_required = MagicMock()

        with (
            patch("vllm_ascend.worker.worker.log_cold_perf_event") as log,
            patch(
                "vllm_ascend.worker.worker.faulthandler.dump_traceback_later"
            ) as arm,
            patch(
                "vllm_ascend.worker.worker.faulthandler.cancel_dump_traceback_later"
            ) as cancel,
            patch("vllm_ascend.worker.worker.forget_cold_perf_request") as forget,
        ):
            result = worker.sample_tokens(MagicMock())

        self.assertIs(result, output)
        self.assertEqual(output._ascend_cold_perf_request_ids, ("followup",))
        self.assertIsInstance(output._ascend_cold_perf_sample_return_ns, int)
        log.assert_not_called()
        arm.assert_not_called()
        cancel.assert_not_called()
        forget.assert_not_called()

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.ops")
    @patch("vllm_ascend.worker.worker._register_atb_extensions")
    @patch("vllm_ascend.worker.worker.register_ascend_customop")
    @patch("vllm_ascend.worker.worker.get_ascend_config")
    @patch("vllm_ascend.worker.worker.init_ascend_config")
    @patch("vllm_ascend.worker.worker.check_ascend_device_type")
    @patch(init_cached_hf_modules_path, create=True)
    @patch("vllm_ascend.worker.worker.NPUWorker._create_profiler")
    def test_init_npu_worker_normal_case(
        self,
        mock_create_profiler,
        mock_init_cached_hf_modules,
        mock_check_ascend_device_type,
        mock_init_ascend_config,
        mock_get_ascend_config,
        mock_register_ascend_customop,
        mock_register_atb_extensions,
        mock_ops,
        mock_adapt_patch,
    ):
        """Test NPUWorker normal initialization"""
        # Setup mock behavior
        mock_ops.register_dummy_fusion_op.return_value = None
        mock_ascend_config = MagicMock()
        mock_ascend_config.enable_cpu_binding = True
        mock_get_ascend_config.return_value = mock_ascend_config

        # Import and create NPUWorker instance
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker(
            vllm_config=self.vllm_config_mock,
            local_rank=self.local_rank,
            rank=self.rank,
            distributed_init_method=self.distributed_init_method,
            is_driver_worker=self.is_driver_worker,
        )

        # Verify initialization call order
        mock_adapt_patch.assert_called_once()
        mock_ops.register_dummy_fusion_op.assert_called_once()
        mock_register_atb_extensions.assert_called_once()
        mock_register_ascend_customop.assert_called_once()
        mock_init_ascend_config.assert_called_once_with(self.vllm_config_mock)
        mock_check_ascend_device_type.assert_called_once()

        # Verify cache_dtype setting
        self.assertEqual(worker.cache_dtype, torch.float16)
        # Profiler is lazily initialized - not created during __init__ (RFC #6954)
        mock_create_profiler.assert_not_called()

        # Verify init_cached_hf_modules is not called (trust_remote_code=False)
        mock_init_cached_hf_modules.assert_not_called()

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.ops")
    @patch("vllm_ascend.worker.worker._register_atb_extensions")
    @patch("vllm_ascend.worker.worker.register_ascend_customop")
    @patch("vllm_ascend.worker.worker.get_ascend_config")
    @patch("vllm_ascend.worker.worker.init_ascend_config")
    @patch("vllm_ascend.worker.worker.check_ascend_device_type")
    @patch(init_cached_hf_modules_path, create=True)
    @patch("vllm_ascend.worker.worker.NPUWorker._create_profiler")
    def test_init_npu_worker_with_trust_remote_code(
        self,
        mock_create_profiler,
        mock_init_cached_hf_modules,
        mock_check_ascend_device_type,
        mock_init_ascend_config,
        mock_get_ascend_config,
        mock_register_ascend_customop,
        mock_register_atb_extensions,
        mock_ops,
        mock_adapt_patch,
    ):
        """Test NPUWorker initialization with trust_remote_code=True"""
        # Set trust_remote_code=True
        self.model_config_mock.trust_remote_code = True
        mock_ops.register_dummy_fusion_op.return_value = None
        mock_ascend_config = MagicMock()
        mock_ascend_config.enable_cpu_binding = True
        mock_get_ascend_config.return_value = mock_ascend_config

        # Create NPUWorker instance
        from vllm_ascend.worker.worker import NPUWorker

        _ = NPUWorker(
            vllm_config=self.vllm_config_mock,
            local_rank=self.local_rank,
            rank=self.rank,
            distributed_init_method=self.distributed_init_method,
            is_driver_worker=self.is_driver_worker,
        )

        # Verify init_cached_hf_modules is called (trust_remote_code=True)
        mock_init_cached_hf_modules.assert_not_called()

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.ops")
    @patch("vllm_ascend.worker.worker._register_atb_extensions")
    @patch("vllm_ascend.worker.worker.register_ascend_customop")
    @patch("vllm_ascend.worker.worker.get_ascend_config")
    @patch("vllm_ascend.worker.worker.init_ascend_config")
    @patch("vllm_ascend.worker.worker.check_ascend_device_type")
    @patch(init_cached_hf_modules_path, create=True)
    @patch("vllm_ascend.worker.worker.NPUWorker._create_profiler")
    def test_init_npu_worker_with_custom_cache_dtype(
        self,
        mock_create_profiler,
        mock_init_cached_hf_modules,
        mock_check_ascend_device_type,
        mock_init_ascend_config,
        mock_get_ascend_config,
        mock_register_ascend_customop,
        mock_register_atb_extensions,
        mock_ops,
        mock_adapt_patch,
    ):
        """Test NPUWorker initialization with custom cache_dtype"""
        # Set custom cache_dtype
        self.cache_config_mock.cache_dtype = "float32"
        mock_ops.register_dummy_fusion_op.return_value = None
        mock_ascend_config = MagicMock()
        mock_ascend_config.enable_cpu_binding = True
        mock_get_ascend_config.return_value = mock_ascend_config

        # Create NPUWorker instance
        from vllm_ascend.worker.worker import NPUWorker

        with patch("vllm.utils.torch_utils.STR_DTYPE_TO_TORCH_DTYPE", {"float32": torch.float32}):
            worker = NPUWorker(
                vllm_config=self.vllm_config_mock,
                local_rank=self.local_rank,
                rank=self.rank,
                distributed_init_method=self.distributed_init_method,
                is_driver_worker=self.is_driver_worker,
            )

        # Verify cache_dtype is set to custom value
        self.assertEqual(worker.cache_dtype, torch.float32)

    def test_initialize_cache(self):
        """Test initialize_cache method"""
        from vllm_ascend.worker.worker import NPUWorker

        # Create a simple worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.cache_config = MagicMock()

            # Test initialize_cache
            worker.initialize_cache(100, 50)

            # Verify parameter setting
            self.assertEqual(worker.cache_config.num_gpu_blocks, 100)
            self.assertEqual(worker.cache_config.num_cpu_blocks, 50)

    @patch("vllm_ascend.worker.worker.CaMemAllocator")
    @patch.dict("os.environ", {"VLLM_ASCEND_ENABLE_NZ": "0"})
    def test_wake_up_mode_enabled(self, mock_allocator_class):
        """Test wake_up method when sleep mode is enabled"""
        from vllm_ascend.worker.worker import NPUWorker

        # Setup mock
        mock_allocator = MagicMock()
        mock_allocator_class.get_instance.return_value = mock_allocator

        mock_hidden_size = MagicMock()
        mock_hf_config = MagicMock()
        mock_hf_config.hidden_size = mock_hidden_size
        mock_model_config = MagicMock()
        mock_model_config.hf_config = mock_hf_config
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config = mock_model_config

        mock_model_runner = MagicMock()
        mock_model_runner.model = MagicMock()

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = mock_model_runner
            worker.vllm_config = mock_vllm_config
            worker._sleep_saved_buffers = {}
            # Test wake_up method
            worker.wake_up(tags=["test_tag"])

            mock_allocator.wake_up.assert_called_once_with(tags=["test_tag"])

    @patch("vllm_ascend.worker.worker.NPUWorker._init_worker_distributed_environment")
    @patch("vllm_ascend.worker.worker.init_device_properties_triton")
    @patch("vllm_ascend.worker.worker.get_ascend_config")
    @patch("vllm_ascend.worker.worker.MemorySnapshot")
    @patch("torch.npu.set_device")
    @patch("torch.npu.empty_cache")
    def test_init_device(
        self,
        mock_empty_cache,
        mock_set_device,
        mock_memory_snapshot,
        mock_get_ascend_config,
        mock_init_triton,
        mock_init_dist_env,
    ):
        """Test _init_device method"""
        from vllm_ascend.worker.worker import NPUWorker

        snapshot = SimpleNamespace(free_memory=1000, total_memory=2000)
        mock_memory_snapshot.return_value = snapshot
        mock_get_ascend_config.return_value.enable_cpu_binding = False

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.local_rank = 1
            worker.model_config = MagicMock()
            worker.parallel_config = MagicMock()
            worker.parallel_config.local_world_size = 0
            worker.parallel_config.data_parallel_size = 1
            worker.model_config.seed = 42
            worker.cache_config = SimpleNamespace(gpu_memory_utilization=0.4)

            # Test _init_device
            result = worker._init_device()

            mock_set_device.assert_called_once()
            self.assertEqual(str(mock_set_device.call_args.args[0]), "npu:1")
            mock_empty_cache.assert_called_once()
            mock_memory_snapshot.assert_called_once_with()
            mock_init_dist_env.assert_called_once()
            mock_init_triton.assert_called_once()

            # Verify return value is a torch.device object
            self.assertEqual(str(result), "npu:1")
            self.assertIs(worker.init_snapshot, snapshot)
            self.assertEqual(worker.requested_memory, 800)

    def test_profile_start_stop(self):
        """Test profile method start and stop"""
        from vllm_ascend.worker.worker import NPUWorker

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.profiler_config = profiler_config
            worker.rank = 0
            mock_profiler = MagicMock()
            worker.profiler = mock_profiler

            worker.profile(is_start=True)
            mock_profiler.start.assert_called_once()

            worker.profile(is_start=False)
            mock_profiler.stop.assert_called_once()

    def test_profile_no_profiler_raises_error(self):
        """Test profile method raises exception when profiler is not available"""
        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock - profiler_config indicates profiling disabled
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.profiler = None
            worker.profiler_config = ProfilerConfig(profiler=None, torch_profiler_dir="")

            # Test should raise exception
            with self.assertRaises(RuntimeError) as cm:
                worker.profile()

            self.assertIn("Profiling is not enabled", str(cm.exception))

    def test_profile_with_prefix_uses_trace_name(self):
        """[RFC #6954] profile() accepts profile_prefix and passes trace_name to _create_profiler"""
        from vllm_ascend.worker.worker import NPUWorker

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        vllm_config_mock = MagicMock()
        vllm_config_mock.profiler_config = profiler_config

        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.profiler_config = profiler_config
            worker.profiler = None
            worker.rank = 0

        with (
            patch(
                "vllm.distributed.utils.get_worker_rank_suffix",
                return_value="dp0_pp0_tp0_dcp0_ep0_rank0",
            ),
            patch.object(
                NPUWorker,
                "_create_profiler",
                return_value=MagicMock(),
            ) as mock_create,
        ):
            worker.profile(is_start=True, profile_prefix="warmup")

            mock_create.assert_called_once_with("warmup_dp0_pp0_tp0_dcp0_ep0_rank0")

    def test_profile_lazy_init(self):
        """[RFC #6954] Profiler is lazily created on first profile(is_start=True) call"""
        from vllm_ascend.worker.worker import NPUWorker

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        vllm_config_mock = MagicMock()
        vllm_config_mock.profiler_config = profiler_config

        with patch.object(NPUWorker, "_create_profiler", return_value=MagicMock()) as mock_create:
            with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
                worker = NPUWorker()
                worker.profiler_config = profiler_config
                worker.profiler = None
                worker.rank = 0

            self.assertIsNone(worker.profiler)
            mock_create.assert_not_called()

            with patch("vllm.distributed.utils.get_worker_rank_suffix", return_value="dp0_pp0_tp0_dcp0_ep0_rank0"):
                worker.profile(is_start=True)

            mock_create.assert_called_once()
            self.assertIsNotNone(worker.profiler)

    def test_profile_restart_reuses_existing_profiler(self):
        """[RFC #6954] Restarting profiling reuses the existing profiler."""
        from vllm_ascend.worker.worker import NPUWorker

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        mock_profiler = MagicMock()

        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.profiler_config = profiler_config
            worker.profiler = None
            worker.rank = 0

        with (
            patch(
                "vllm.distributed.utils.get_worker_rank_suffix",
                return_value="dp0_pp0_tp0_dcp0_ep0_rank0",
            ),
            patch.object(
                NPUWorker,
                "_create_profiler",
                return_value=mock_profiler,
            ) as mock_create,
        ):
            worker.profile(is_start=True, profile_prefix="session1")
            mock_create.assert_called_once_with("session1_dp0_pp0_tp0_dcp0_ep0_rank0")

            worker.profile(is_start=False)
            worker.profile(is_start=True)  # Restart without new prefix
            # Should NOT create new profiler, just restart existing
            mock_create.assert_called_once()

    def test_trace_handler_uses_worker_name(self):
        """[RFC #6954] _create_profiler passes worker_name to tensorboard_trace_handler"""
        from vllm_ascend.worker.worker import NPUWorker

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
        )
        vllm_config_mock = MagicMock()
        vllm_config_mock.profiler_config = profiler_config

        with patch("vllm_ascend.worker.worker.envs_ascend") as mock_envs:
            mock_envs.MSMONITOR_USE_DAEMON = 0
            with patch("torch_npu.profiler.tensorboard_trace_handler") as mock_handler:
                with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
                    worker = NPUWorker()
                    worker.profiler_config = profiler_config
                    worker.vllm_config = vllm_config_mock

                worker._create_profiler("warmup_dp0_pp0_tp0_dcp0_ep0_rank0")

                mock_handler.assert_called_once_with(
                    "/path/to/traces",
                    worker_name="warmup_dp0_pp0_tp0_dcp0_ep0_rank0",
                    analyse_flag=False,
                )

    @patch("vllm_ascend.worker.worker.envs_ascend")
    def test_profile_and_msmonitor_both_enabled_raises_error(self, mock_envs_ascend):
        """Test _create_profiler raises when both profiler and msmonitor are enabled"""
        from vllm_ascend.worker.worker import NPUWorker

        mock_envs_ascend.MSMONITOR_USE_DAEMON = 1

        profiler_config = ProfilerConfig(profiler="torch", torch_profiler_dir="/path/to/traces")
        vllm_config_mock = MagicMock()
        vllm_config_mock.profiler_config = profiler_config

        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.profiler_config = profiler_config
            worker.vllm_config = vllm_config_mock

            with self.assertRaises(RuntimeError) as cm:
                _ = worker._create_profiler("test_trace")

            self.assertIn(
                "MSMONITOR_USE_DAEMON and torch profiler cannot be both enabled at the same time.", str(cm.exception)
            )

    def test_lora_methods(self):
        """Test LoRA related methods"""
        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            mock_model_runner = MagicMock()
            worker.model_runner = mock_model_runner

            # Set return values
            mock_model_runner.add_lora.return_value = True
            mock_model_runner.remove_lora.return_value = True
            mock_model_runner.list_loras.return_value = {1, 2, 3}
            mock_model_runner.pin_lora.return_value = True

            # Test each method
            mock_request = MagicMock()
            self.assertTrue(worker.add_lora(mock_request))
            mock_model_runner.add_lora.assert_called_once_with(mock_request)

            self.assertTrue(worker.remove_lora(1))
            mock_model_runner.remove_lora.assert_called_once_with(1)

            self.assertEqual(worker.list_loras(), {1, 2, 3})
            mock_model_runner.list_loras.assert_called_once()

            self.assertTrue(worker.pin_lora(2))
            mock_model_runner.pin_lora.assert_called_once_with(2)

    def test_get_methods(self):
        """Test various get methods"""
        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            mock_model_runner = MagicMock()
            worker.model_runner = mock_model_runner

            # Set return values
            mock_model = MagicMock()
            mock_kv_cache_spec = {"test": "spec"}
            mock_pooling_tasks = ["task1", "task2"]
            mock_supported_tasks = ("task1", "task2")

            mock_model_runner.get_model.return_value = mock_model
            mock_model_runner.get_kv_cache_spec.return_value = mock_kv_cache_spec
            mock_model_runner.get_supported_pooling_tasks.return_value = mock_pooling_tasks
            mock_model_runner.get_supported_tasks.return_value = mock_supported_tasks

            # Test each get method
            self.assertEqual(worker.get_model(), mock_model)
            self.assertEqual(worker.get_kv_cache_spec(), mock_kv_cache_spec)
            self.assertEqual(worker.get_supported_pooling_tasks(), mock_pooling_tasks)
            self.assertEqual(worker.get_supported_tasks(), mock_supported_tasks)

    def test_execute_dummy_batch(self):
        """Test execute_dummy_batch method"""
        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.compilation_config = MagicMock()
            worker.compilation_config.cudagraph_mode = MagicMock()
            mock_model_runner = MagicMock()
            mock_decode_token_per_req = mock_model_runner.decode_token_per_req
            worker.model_runner = mock_model_runner

            # Test execute_dummy_batch
            worker.execute_dummy_batch()

            # Verify call
            mock_model_runner._dummy_run.assert_called_once_with(
                num_tokens=mock_decode_token_per_req, uniform_decode=True
            )

    @patch("vllm_ascend.worker.worker.envs_ascend")
    @patch("torch_npu.profiler._ExperimentalConfig")
    @patch("torch_npu.profiler.profile")
    @patch("torch_npu.profiler.tensorboard_trace_handler")
    @patch("torch_npu.profiler.ExportType")
    @patch("torch_npu.profiler.ProfilerLevel")
    @patch("torch_npu.profiler.AiCMetrics")
    @patch("torch_npu.profiler.ProfilerActivity")
    def test_create_profiler_enabled(
        self,
        mock_profiler_activity,
        mock_aic_metrics,
        mock_profiler_level,
        mock_export_type,
        mock_trace_handler,
        mock_profile,
        mock_experimental_config,
        mock_envs_ascend,
    ):
        """Test _create_profiler - profiler enabled with worker_name for trace naming (RFC #6954)"""
        from vllm_ascend.worker.worker import NPUWorker

        mock_envs_ascend.MSMONITOR_USE_DAEMON = 0

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="/path/to/traces",
            torch_profiler_with_stack=True,
            torch_profiler_with_memory=True,
        )
        vllm_config_mock = MagicMock()
        vllm_config_mock.profiler_config = profiler_config

        mock_export_type.Text = "Text"
        mock_profiler_level.Level1 = "Level1"
        mock_aic_metrics.PipeUtilization = "PipeUtilization"
        mock_profiler_activity.CPU = "CPU"
        mock_profiler_activity.NPU = "NPU"

        mock_experimental_config_instance = MagicMock()
        mock_experimental_config.return_value = mock_experimental_config_instance
        mock_trace_handler_instance = MagicMock()
        mock_trace_handler.return_value = mock_trace_handler_instance
        mock_profiler_instance = MagicMock()
        mock_profile.return_value = mock_profiler_instance

        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.profiler_config = profiler_config
            worker.vllm_config = vllm_config_mock

            result = worker._create_profiler("warmup_dp0_pp0_tp0_dcp0_ep0_rank0")

            mock_experimental_config.assert_called_once()
            config_call = mock_experimental_config.call_args
            config_kwargs = config_call.kwargs
            expected_config = {
                "export_type": "Text",
                "profiler_level": "Level1",
                "msprof_tx": False,
                "aic_metrics": "PipeUtilization",
                "l2_cache": False,
                "op_attr": False,
                "data_simplification": True,
                "record_op_args": False,
                "gc_detect_threshold": None,
            }
            for key, expected_value in expected_config.items():
                self.assertEqual(config_kwargs[key], expected_value)

            # Verify trace handler called with worker_name (RFC #6954)
            mock_trace_handler.assert_called_once_with(
                "/path/to/traces",
                worker_name="warmup_dp0_pp0_tp0_dcp0_ep0_rank0",
                analyse_flag=False,
            )

            mock_profile.assert_called_once()
            profile_kwargs = mock_profile.call_args.kwargs
            expected_activities = ["CPU", "NPU"]
            self.assertEqual(profile_kwargs["activities"], expected_activities)
            self.assertTrue(profile_kwargs["profile_memory"])
            self.assertEqual(profile_kwargs["on_trace_ready"], mock_trace_handler_instance)
            self.assertEqual(result, mock_profiler_instance)

    def test_create_profiler_disabled(self):
        """Test _create_profiler raises when profiler disabled"""
        from vllm_ascend.worker.worker import NPUWorker

        profiler_config = ProfilerConfig(profiler=None, torch_profiler_dir="")

        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.profiler_config = profiler_config

            with self.assertRaises(RuntimeError) as cm:
                worker._create_profiler("test_trace")
            self.assertIn("Unrecognized profiler: None", str(cm.exception))

    def test_create_profiler_empty_dir(self):
        """Test _create_profiler raises when torch_profiler_dir is empty/falsy"""
        from vllm_ascend.worker.worker import NPUWorker

        # Use MagicMock to bypass ProfilerConfig validation (empty dir not allowed)
        profiler_config = MagicMock()
        profiler_config.profiler = "torch"
        profiler_config.torch_profiler_dir = ""

        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.profiler_config = profiler_config

            with self.assertRaises(RuntimeError) as cm:
                worker._create_profiler("test_trace")
            self.assertIn("torch_profiler_dir cannot be empty", str(cm.exception))

    @staticmethod
    def _build_memory_profile_worker(
        requested_memory,
        initial_free_memory=8500,
    ):
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        worker.init_snapshot = SimpleNamespace(free_memory=initial_free_memory)
        worker.requested_memory = requested_memory
        worker.model_runner = MagicMock()
        worker.model_runner.model_memory_usage = 1000
        worker.vllm_config = MagicMock()
        return worker

    @staticmethod
    def _memory_profile_result(free_memory, non_kv_cache_memory):
        return SimpleNamespace(
            after_profile=SimpleNamespace(free_memory=free_memory),
            non_kv_cache_memory=non_kv_cache_memory,
        )

    @staticmethod
    def _memory_profile_envs_without_optional_reserves():
        return SimpleNamespace(
            VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD=False,
            VLLM_ASCEND_DSA_USE_ADAPTER_CACHE=False,
        )

    def test_determine_available_memory_normal_case(self):
        """Test the current memory-profiling result is used for KV capacity."""
        import vllm_ascend.worker.worker as worker_module

        worker = self._build_memory_profile_worker(requested_memory=8000)
        profile_result = self._memory_profile_result(
            free_memory=6000,
            non_kv_cache_memory=3000,
        )

        with (
            patch.object(
                worker_module,
                "memory_profiling",
            ) as mock_memory_profiling,
            patch.object(
                worker_module,
                "envs_ascend",
                self._memory_profile_envs_without_optional_reserves(),
            ),
            patch.object(worker_module, "logger") as mock_logger,
        ):
            mock_memory_profiling.return_value.__enter__.return_value = profile_result
            result = worker.determine_available_memory()

        mock_memory_profiling.assert_called_once_with(
            worker.init_snapshot,
            weights_memory=1000,
        )
        worker.model_runner.profile_run.assert_called_once_with()
        self.assertEqual(result, 5000)
        self.assertEqual(worker.available_kv_cache_memory_bytes, 5000)
        mock_logger.info_once.assert_called_once()

    def test_determine_available_memory_with_non_torch_allocations(self):
        """Test non-torch usage folded into non_kv_cache_memory."""
        import vllm_ascend.worker.worker as worker_module

        worker = self._build_memory_profile_worker(requested_memory=9000)
        profile_result = self._memory_profile_result(
            free_memory=5000,
            non_kv_cache_memory=5500,
        )

        with (
            patch.object(
                worker_module,
                "memory_profiling",
            ) as mock_memory_profiling,
            patch.object(
                worker_module,
                "envs_ascend",
                self._memory_profile_envs_without_optional_reserves(),
            ),
        ):
            mock_memory_profiling.return_value.__enter__.return_value = profile_result
            result = worker.determine_available_memory()

        worker.model_runner.profile_run.assert_called_once_with()
        self.assertEqual(result, 3500)

    def test_staged_graph_memory_is_reserved_before_kv_sizing(self):
        import vllm_ascend.worker.worker as worker_module

        worker = self._build_memory_profile_worker(
            requested_memory=512 << 20,
        )
        worker.device = torch.device("cpu")
        profile_result = SimpleNamespace(
            after_profile=SimpleNamespace(free_memory=6000),
            before_profile=SimpleNamespace(torch_peak=100),
            non_torch_increase=1000,
            torch_peak_increase=0,
            weights_memory=1000,
            non_kv_cache_memory=3000,
        )
        worker.model_runner.profile_cudagraph_memory.return_value = 32 << 20

        with (
            patch.object(
                worker_module,
                "memory_profiling",
            ) as mock_memory_profiling,
            patch.object(
                worker_module,
                "staged_sfa_graph_configured",
                return_value=True,
            ),
            patch.object(
                worker_module.torch.accelerator,
                "memory_stats",
                return_value={"allocated_bytes.all.peak": 2100},
            ),
            patch.object(
                worker_module,
                "envs_ascend",
                self._memory_profile_envs_without_optional_reserves(),
            ),
        ):
            mock_memory_profiling.return_value.__enter__.return_value = (
                profile_result
            )
            result = worker.determine_available_memory()

        reservation = (32 + 64) << 20
        self.assertEqual(
            result,
            (512 << 20) - 1000 - 2000 - 1000 - reservation,
        )
        worker.model_runner.profile_cudagraph_memory.assert_called_once_with()
        self.assertEqual(
            worker_module._staged_sfa_graph_memory_reservation(1 << 30),
            int(1.1 * (1 << 30)),
        )

    def test_determine_available_memory_memory_profiling_error(self):
        """Test an increase in free memory during profiling is rejected."""
        import vllm_ascend.worker.worker as worker_module

        worker = self._build_memory_profile_worker(requested_memory=8000)
        profile_result = self._memory_profile_result(
            free_memory=9000,
            non_kv_cache_memory=1000,
        )

        with patch.object(
            worker_module,
            "memory_profiling",
        ) as mock_memory_profiling:
            mock_memory_profiling.return_value.__enter__.return_value = profile_result
            with self.assertRaises(AssertionError) as cm:
                worker.determine_available_memory()

        worker.model_runner.profile_run.assert_called_once_with()
        self.assertIn("Error in memory profiling", str(cm.exception))

    def test_determine_available_memory_negative_result(self):
        """A negative budget is propagated for downstream validation."""
        import vllm_ascend.worker.worker as worker_module

        worker = self._build_memory_profile_worker(requested_memory=8000)
        profile_result = self._memory_profile_result(
            free_memory=2000,
            non_kv_cache_memory=10000,
        )

        with (
            patch.object(
                worker_module,
                "memory_profiling",
            ) as mock_memory_profiling,
            patch.object(
                worker_module,
                "envs_ascend",
                self._memory_profile_envs_without_optional_reserves(),
            ),
        ):
            mock_memory_profiling.return_value.__enter__.return_value = profile_result
            result = worker.determine_available_memory()

        worker.model_runner.profile_run.assert_called_once_with()
        self.assertEqual(result, -2000)
        self.assertEqual(worker.available_kv_cache_memory_bytes, -2000)

    def test_execute_model_first_rank(self):
        """Test execute_model method - first rank case"""
        from vllm.v1.outputs import ModelRunnerOutput

        from vllm_ascend.worker.worker import NPUWorker

        with (
            patch.object(
                NPUWorker,
                "__init__",
                lambda x, **kwargs: None,
            ),
            patch("vllm_ascend.worker.worker.get_pp_group") as mock_get_pp_group,
        ):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = "ray"
            worker._pp_send_work = []

            mock_pp_group = MagicMock()
            mock_pp_group.is_first_rank = True
            mock_pp_group.is_last_rank = True
            mock_get_pp_group.return_value = mock_pp_group

            mock_scheduler_output = MagicMock()
            mock_scheduler_output.total_num_scheduled_tokens = 1
            mock_model_output = MagicMock(spec=ModelRunnerOutput)
            worker.model_runner.execute_model.return_value = mock_model_output

            result = worker.execute_model(mock_scheduler_output)

            worker.model_runner.execute_model.assert_called_once_with(
                mock_scheduler_output,
                None,
            )
            self.assertEqual(result, mock_model_output)

    @patch(
        "vllm_ascend.worker.worker.enable_sp",
        return_value=False,
    )
    @patch("vllm_ascend.worker.worker.get_pp_group")
    @patch("vllm_ascend.worker.worker.get_tp_group")
    def test_execute_model_middle_rank(
        self,
        mock_get_tp_group,
        mock_get_pp_group,
        mock_enable_sp,
    ):
        """Test execute_model method - middle rank case"""
        from vllm.sequence import IntermediateTensors

        from vllm_ascend.worker.worker import NPUWorker

        with patch.object(
            NPUWorker,
            "__init__",
            lambda x, **kwargs: None,
        ):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = "ray"
            worker._pp_send_work = []

            mock_pp_group = MagicMock()
            mock_pp_group.is_first_rank = False
            mock_pp_group.is_last_rank = False
            mock_pp_group.irecv_tensor_dict.return_value = (
                {"tensor": "data"},
                [],
                [],
            )
            mock_get_pp_group.return_value = mock_pp_group

            mock_intermediate_output = MagicMock(spec=IntermediateTensors)
            mock_intermediate_output.tensors = {"output_tensor": "data"}
            mock_intermediate_output.kv_connector_output = None
            worker.model_runner.execute_model.return_value = mock_intermediate_output

            mock_scheduler_output = MagicMock()
            mock_scheduler_output.total_num_scheduled_tokens = 1

            result = worker.execute_model(mock_scheduler_output)

            mock_pp_group.irecv_tensor_dict.assert_called_once_with(all_gather_group=mock_get_tp_group.return_value)
            worker.model_runner.execute_model.assert_called_once()
            args, kwargs = worker.model_runner.execute_model.call_args
            self.assertEqual(args[0], mock_scheduler_output)
            self.assertIsInstance(args[1], IntermediateTensors)
            self.assertEqual(kwargs, {})
            mock_pp_group.isend_tensor_dict.assert_called_once_with(
                {"output_tensor": "data"},
                all_gather_group=mock_get_tp_group.return_value,
            )
            self.assertIsNone(result)

    def test_execute_model_external_launcher(self):
        """Test execute_model method - external_launcher mode"""
        from vllm.v1.outputs import ModelRunnerOutput

        from vllm_ascend.worker.worker import NPUWorker

        with (
            patch.object(
                NPUWorker,
                "__init__",
                lambda x, **kwargs: None,
            ),
            patch("vllm_ascend.worker.worker.get_pp_group") as mock_get_pp_group,
        ):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = "external_launcher"
            worker._pp_send_work = []

            mock_pp_group = MagicMock()
            mock_pp_group.is_first_rank = True
            mock_pp_group.is_last_rank = False
            mock_get_pp_group.return_value = mock_pp_group

            mock_scheduler_output = MagicMock()
            mock_scheduler_output.total_num_scheduled_tokens = 1
            mock_model_output = MagicMock(spec=ModelRunnerOutput)
            worker.model_runner.execute_model.return_value = mock_model_output

            result = worker.execute_model(mock_scheduler_output)

            self.assertEqual(result, mock_model_output)

    @patch("vllm_ascend.worker.worker.CaMemAllocator")
    def test_load_model_with_sleep_mode(self, mock_allocator_class):
        """Test load_model method - with sleep mode enabled"""
        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.model_config = MagicMock()
            worker.vllm_config.model_config.enable_sleep_mode = True

            # Setup allocator mock
            mock_allocator = MagicMock()
            mock_allocator.get_current_usage.return_value = 0
            mock_context = MagicMock()
            mock_allocator.use_memory_pool.return_value = mock_context
            mock_allocator_class.get_instance.return_value = mock_allocator

            # Test load_model
            worker.load_model()

            # Verify calls
            mock_allocator_class.get_instance.assert_called_once()
            mock_allocator.get_current_usage.assert_called_once()
            mock_allocator.use_memory_pool.assert_called_once_with(tag="weights")
            worker.model_runner.load_model.assert_called_once()

    def test_load_model_without_sleep_mode(self):
        """Test load_model method - without sleep mode enabled"""
        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.model_config = MagicMock()
            worker.vllm_config.model_config.enable_sleep_mode = False

            # Test load_model
            worker.load_model()

            # Verify calls
            worker.model_runner.load_model.assert_called_once()

    @patch("vllm_ascend.worker.worker.CaMemAllocator")
    def test_load_model_sleep_mode_assertion_error(self, mock_allocator_class):
        """Test load_model method - assertion error in sleep mode"""
        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.model_config = MagicMock()
            worker.vllm_config.model_config.enable_sleep_mode = True

            # Setup allocator mock - current usage is not 0
            mock_allocator = MagicMock()
            mock_allocator.get_current_usage.return_value = 100  # Non-zero value
            mock_allocator_class.get_instance.return_value = mock_allocator

            # Test should throw assertion error
            with self.assertRaises(AssertionError) as cm:
                worker.load_model()

            self.assertIn("Sleep mode can only be", str(cm.exception))

    @patch("vllm_ascend.worker.worker.logger")
    @patch("vllm_ascend.worker.worker.NPUWorker._warm_up_atb")
    def test_compile_or_warm_up_model_with_eager_mode(self, mock_warm_up_atb, mock_logger):
        """Test compile_or_warm_up_model method - eager mode"""
        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.model_config = MagicMock()
            worker.model_config.enforce_eager = True
            worker.model_config.seed = 12345

            # Setup compilation config
            worker.vllm_config.compilation_config = MagicMock()
            worker.vllm_config.compilation_config.compile_sizes = [1, 4, 8, 16]
            worker.vllm_config.compilation_config.cudagraph_capture_sizes = [4, 8]

            # Test compile_or_warm_up_model
            worker.compile_or_warm_up_model()

            # Verify _dummy_run call count and order (by size descending)
            expected_calls = [
                unittest.mock.call(16),
                unittest.mock.call(8),
                unittest.mock.call(4),
                unittest.mock.call(1),
            ]
            worker.model_runner._dummy_run.assert_has_calls(expected_calls)

            # Should not call capture_model in eager mode
            worker.model_runner.capture_model.assert_not_called()

            # Verify log output
            self.assertEqual(mock_logger.info.call_count, 4)

            # Verify atb warm up
            mock_warm_up_atb.assert_called_once()

    @patch("vllm_ascend.worker.worker.log_cold_perf_process_event")
    @patch("vllm_ascend.worker.worker.cold_perf_enabled", return_value=True)
    @patch(
        "vllm_ascend.worker.worker.staged_sfa_graph_configured",
        return_value=False,
    )
    @patch("vllm_ascend.worker.worker.logger")
    @patch("vllm_ascend.worker.worker.NPUWorker._warm_up_atb")
    def test_compile_or_warm_up_model_with_graph_capture(
        self,
        mock_warm_up_atb,
        mock_logger,
        _mock_staged_sfa_graph_configured,
        _mock_cold_perf_enabled,
        mock_perf_log,
    ):
        """Test compile_or_warm_up_model method - with graph capture enabled"""
        from vllm_ascend.worker.worker import NPUWorker

        # Create worker mock
        with patch.object(NPUWorker, "__init__", lambda x, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.model_config = MagicMock()
            worker.model_config.enforce_eager = False  # Enable graph capture
            worker.model_config.seed = 67890

            # Setup compilation config
            worker.vllm_config.compilation_config = MagicMock()
            worker.vllm_config.compilation_config.compile_sizes = [1, 4, 8, 16]
            worker.vllm_config.compilation_config.cudagraph_capture_sizes = [4, 8]

            # Test compile_or_warm_up_model
            worker.compile_or_warm_up_model()

            # Verify only call _dummy_run for sizes not in cudagraph_capture_sizes
            expected_calls = [unittest.mock.call(16), unittest.mock.call(1)]
            worker.model_runner._dummy_run.assert_has_calls(expected_calls)

            # Should call capture_model in non-eager mode
            worker.model_runner.capture_model.assert_called_once()

            self.assertEqual(
                [call.args[0] for call in mock_perf_log.call_args_list],
                [
                    "decoder_graph_capture_start",
                    "decoder_graph_capture_complete",
                ],
            )
            self.assertEqual(
                mock_perf_log.call_args_list[0].kwargs["capture_sizes"],
                [4, 8],
            )
            self.assertIn("elapsed_ms", mock_perf_log.call_args_list[1].kwargs)

            # Verify atb warm up
            mock_warm_up_atb.assert_called_once()

    @patch("vllm_ascend.worker.worker.CaMemAllocator")
    def test_initialize_from_config_with_sleep_mode(
        self,
        mock_allocator_class,
    ):
        """Test initialize_from_config method - with sleep mode enabled"""
        from vllm_ascend.worker.worker import NPUWorker

        with (
            patch.object(
                NPUWorker,
                "__init__",
                lambda x, **kwargs: None,
            ),
            patch("vllm_ascend.worker.worker.ensure_kv_transfer_initialized") as mock_ensure_kv_transfer_initialized,
        ):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.model_config = MagicMock()
            worker.vllm_config.model_config.enable_sleep_mode = True

            mock_allocator = MagicMock()
            mock_context = MagicMock()
            mock_allocator.use_memory_pool.return_value = mock_context
            mock_allocator_class.get_instance.return_value = mock_allocator
            mock_kv_cache_config = MagicMock()

            worker.initialize_from_config(mock_kv_cache_config)

            mock_ensure_kv_transfer_initialized.assert_called_once_with(
                worker.vllm_config,
                mock_kv_cache_config,
            )
            mock_allocator_class.get_instance.assert_called_once_with()
            mock_allocator.use_memory_pool.assert_called_once_with(tag="kv_cache")
            worker.model_runner.initialize_kv_cache.assert_called_once_with(mock_kv_cache_config)

    def test_initialize_from_config_without_sleep_mode(self):
        """Test initialize_from_config method - without sleep mode enabled"""
        from vllm_ascend.worker.worker import NPUWorker

        with (
            patch.object(
                NPUWorker,
                "__init__",
                lambda x, **kwargs: None,
            ),
            patch("vllm_ascend.worker.worker.ensure_kv_transfer_initialized") as mock_ensure_kv_transfer_initialized,
        ):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.model_config = MagicMock()
            worker.vllm_config.model_config.enable_sleep_mode = False
            mock_kv_cache_config = MagicMock()

            worker.initialize_from_config(mock_kv_cache_config)

            mock_ensure_kv_transfer_initialized.assert_called_once_with(
                worker.vllm_config,
                mock_kv_cache_config,
            )
            worker.model_runner.initialize_kv_cache.assert_called_once_with(mock_kv_cache_config)

    @patch(
        "vllm_ascend.worker.worker.enable_sp",
        return_value=False,
    )
    @patch("vllm_ascend.worker.worker.get_pp_group")
    @patch("vllm_ascend.worker.worker.get_tp_group")
    @patch("vllm_ascend.worker.worker.EMPTY_MODEL_RUNNER_OUTPUT")
    def test_execute_model_kv_connector_not_finished(
        self,
        mock_empty_output,
        mock_get_tp_group,
        mock_get_pp_group,
        mock_enable_sp,
    ):
        """Test unfinished KV connector output on a middle PP rank."""
        from vllm.sequence import IntermediateTensors

        from vllm_ascend.worker.worker import NPUWorker

        with patch.object(
            NPUWorker,
            "__init__",
            lambda x, **kwargs: None,
        ):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = "ray"
            worker._pp_send_work = []

            mock_pp_group = MagicMock()
            mock_pp_group.is_first_rank = False
            mock_pp_group.is_last_rank = False
            mock_pp_group.irecv_tensor_dict.return_value = (
                {"tensor": "data"},
                [],
                [],
            )
            mock_get_pp_group.return_value = mock_pp_group

            mock_kv_connector_output = MagicMock()
            mock_kv_connector_output.finished_sending = False
            mock_kv_connector_output.finished_recving = False

            mock_intermediate_output = MagicMock(spec=IntermediateTensors)
            mock_intermediate_output.tensors = {"output_tensor": "data"}
            mock_intermediate_output.kv_connector_output = mock_kv_connector_output
            worker.model_runner.execute_model.return_value = mock_intermediate_output

            mock_scheduler_output = MagicMock()
            mock_scheduler_output.total_num_scheduled_tokens = 1

            result = worker.execute_model(mock_scheduler_output)

            mock_pp_group.irecv_tensor_dict.assert_called_once_with(all_gather_group=mock_get_tp_group.return_value)
            mock_pp_group.isend_tensor_dict.assert_called_once_with(
                {"output_tensor": "data"},
                all_gather_group=mock_get_tp_group.return_value,
            )
            self.assertEqual(result, mock_empty_output)
