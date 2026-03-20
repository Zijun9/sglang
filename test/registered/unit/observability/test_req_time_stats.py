"""Unit tests for srt/observability/req_time_stats.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import unittest

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    ReqTimeStatsBase,
    RequestStage,
    RequestStageConfig,
    SchedulerReqTimeStats,
    calibrate_time_diff,
    convert_time_cross_thread,
    convert_time_to_realtime,
    convert_time_to_realtime_ns,
    global_diff_realtime_monotonic,
    monotonic_time,
    real_time,
)
from sglang.srt.observability.trace import TraceNullContext
from sglang.test.test_utils import CustomTestCase


class TestTimeConversionFunctions(CustomTestCase):

    def test_convert_time_to_realtime(self):
        """Test that realtime = perf_counter_value + global diff."""
        import sglang.srt.observability.req_time_stats as rts

        diff = rts.global_diff_realtime_monotonic
        result = convert_time_to_realtime(100.0)
        self.assertAlmostEqual(result, 100.0 + diff)

    def test_convert_time_to_realtime_ns_returns_int(self):
        """Test that nanosecond conversion returns an integer close to expected."""
        import sglang.srt.observability.req_time_stats as rts

        diff = rts.global_diff_realtime_monotonic
        result = convert_time_to_realtime_ns(1.5)
        expected = int((1.5 + diff) * 1e9)
        self.assertIsInstance(result, int)
        self.assertAlmostEqual(result, expected, delta=10000)

    def test_convert_time_cross_thread_same_diff(self):
        """Test that same old_diff and new_diff leaves the value unchanged."""
        self.assertAlmostEqual(convert_time_cross_thread(100.0, 5.0, 5.0), 100.0)

    def test_convert_time_cross_thread_adjusts(self):
        """Test formula: time_value + old_diff - new_diff."""
        self.assertAlmostEqual(convert_time_cross_thread(100.0, 10.0, 7.0), 103.0)

    def test_convert_time_cross_thread_negative_adjustment(self):
        """Test that new_diff > old_diff produces a smaller result."""
        self.assertAlmostEqual(convert_time_cross_thread(100.0, 5.0, 8.0), 97.0)

    def test_calibrate_time_diff_updates_global(self):
        """Test that calibrate_time_diff refreshes the module-level diff value."""
        import sglang.srt.observability.req_time_stats as rts

        old = rts.global_diff_realtime_monotonic
        calibrate_time_diff()
        self.assertAlmostEqual(rts.global_diff_realtime_monotonic, old, delta=1.0)


class TestRequestStageConfig(CustomTestCase):

    def test_stage_config_attributes(self):
        """Test that constructor arguments are stored correctly."""
        cfg = RequestStageConfig("test_stage", level=2, metrics_is_observed=True)
        self.assertEqual(cfg.stage_name, "test_stage")
        self.assertEqual(cfg.level, 2)
        self.assertTrue(cfg.metrics_is_observed)

    def test_stage_config_defaults(self):
        """Test that level defaults to 0 and metrics_is_observed to False."""
        cfg = RequestStageConfig("minimal")
        self.assertEqual(cfg.level, 0)
        self.assertFalse(cfg.metrics_is_observed)

    def test_predefined_stages(self):
        """Test that key predefined stages have expected names and levels."""
        self.assertEqual(RequestStage.TOKENIZE.stage_name, "tokenize")
        self.assertEqual(RequestStage.TOKENIZE.level, 1)
        self.assertEqual(RequestStage.PREFILL_FORWARD.stage_name, "prefill_forward")
        self.assertTrue(RequestStage.PREFILL_FORWARD.metrics_is_observed)
        self.assertEqual(RequestStage.DECODE_LOOP.level, 3)
        self.assertEqual(RequestStage.ANONYMOUS.stage_name, "")


class TestReqTimeStatsBase(CustomTestCase):

    def test_disagg_mode_str_null(self):
        """Test that NULL mode maps to 'unified'."""
        base = ReqTimeStatsBase(disagg_mode=DisaggregationMode.NULL)
        self.assertEqual(base.disagg_mode_str(), "unified")

    def test_disagg_mode_str_prefill(self):
        """Test that PREFILL mode maps to 'prefill'."""
        base = ReqTimeStatsBase(disagg_mode=DisaggregationMode.PREFILL)
        self.assertEqual(base.disagg_mode_str(), "prefill")

    def test_disagg_mode_str_decode(self):
        """Test that DECODE mode maps to 'decode'."""
        base = ReqTimeStatsBase(disagg_mode=DisaggregationMode.DECODE)
        self.assertEqual(base.disagg_mode_str(), "decode")

    def test_getstate_disables_metrics(self):
        """Test that serialization always sets enable_metrics=False."""
        base = ReqTimeStatsBase(enable_metrics=True)
        state = base.__getstate__()
        self.assertFalse(state["enable_metrics"])

    def test_setstate_converts_fields_ending_in_time(self):
        """Test that __setstate__ applies cross-thread conversion to all *time keys."""
        import sglang.srt.observability.req_time_stats as rts

        cur_global = rts.global_diff_realtime_monotonic
        old_diff = 100.0

        base = ReqTimeStatsBase.__new__(ReqTimeStatsBase)
        state = {
            "created_time": 50.0,
            "disagg_mode": DisaggregationMode.NULL,
            "enable_metrics": False,
            "trace_ctx": TraceNullContext(),
            "diff_realtime_monotonic": old_diff,
        }
        base.__setstate__(state)
        expected = convert_time_cross_thread(50.0, old_diff, cur_global)
        self.assertAlmostEqual(base.created_time, expected)

    def test_setstate_does_not_convert_non_time_fields(self):
        """Test that keys not ending in 'time' are left unchanged."""
        import sglang.srt.observability.req_time_stats as rts

        base = ReqTimeStatsBase.__new__(ReqTimeStatsBase)
        state = {
            "disagg_mode": DisaggregationMode.NULL,
            "enable_metrics": False,
            "trace_ctx": TraceNullContext(),
            "diff_realtime_monotonic": 100.0,
        }
        base.__setstate__(state)
        self.assertEqual(base.disagg_mode, DisaggregationMode.NULL)


class TestAPIServerReqTimeStats(CustomTestCase):

    def _make(self, **kwargs):
        """Helper: create APIServerReqTimeStats with overridden fields."""
        stats = APIServerReqTimeStats()
        for k, v in kwargs.items():
            setattr(stats, k, v)
        return stats

    def test_get_first_token_latency(self):
        """Test first_token_time - created_time."""
        stats = self._make(created_time=100.0, first_token_time=100.5)
        self.assertAlmostEqual(stats.get_first_token_latency(), 0.5)

    def test_get_e2e_latency(self):
        """Test finished_time - created_time."""
        stats = self._make(created_time=100.0, finished_time=102.0)
        self.assertAlmostEqual(stats.get_e2e_latency(), 2.0)

    def test_get_decode_latency(self):
        """Test finished_time - first_token_time."""
        stats = self._make(first_token_time=100.5, finished_time=102.0)
        self.assertAlmostEqual(stats.get_decode_latency(), 1.5)

    def test_convert_to_output_meta_info_decode_throughput(self):
        """Test that decode_throughput = completion_tokens / decode_latency."""
        stats = self._make(
            created_time=100.0, finished_time=102.0, first_token_time=101.0, last_time=101.0
        )
        meta = stats.convert_to_output_meta_info(completion_tokens=20)
        self.assertAlmostEqual(meta["decode_throughput"], 20.0)

    def test_convert_to_output_meta_info_zero_tokens_no_throughput(self):
        """Test that decode_throughput is absent when completion_tokens=0."""
        stats = self._make(
            created_time=100.0, finished_time=102.0, first_token_time=101.0, last_time=101.0
        )
        meta = stats.convert_to_output_meta_info(completion_tokens=0)
        self.assertNotIn("decode_throughput", meta)

    def test_convert_to_output_meta_info_with_scheduler_stats(self):
        """Test that inference_time = finished_time - scheduler.forward_entry_time."""
        stats = self._make(
            created_time=100.0, finished_time=102.0, first_token_time=101.0, last_time=101.0
        )
        sched_stats = SchedulerReqTimeStats(forward_entry_time=100.5)
        meta = stats.convert_to_output_meta_info(
            scheduler_time_stats=sched_stats, completion_tokens=5
        )
        self.assertAlmostEqual(meta["inference_time"], 1.5)

    def test_convert_to_gen_ai_span_attrs_full(self):
        """Test all span attributes when all timestamps are set."""
        stats = self._make(
            created_time=100.0,
            finished_time=103.0,
            first_token_time=101.0,
            api_server_dispatch_finish_time=100.2,
        )
        attrs = stats.convert_to_gen_ai_span_attrs()
        self.assertAlmostEqual(attrs["gen_ai.latency.time_to_first_token"], 1.0)
        self.assertAlmostEqual(attrs["gen_ai.latency.e2e"], 3.0)
        self.assertAlmostEqual(attrs["gen_ai.latency.time_in_model_decode"], 2.0)
        self.assertAlmostEqual(attrs["gen_ai.latency.time_in_model_inference"], 2.8)
        self.assertAlmostEqual(attrs["gen_ai.latency.time_in_model_prefill"], 0.8)

    def test_convert_to_gen_ai_span_attrs_partial(self):
        """Test that missing timestamps produce fewer span attributes."""
        stats = self._make(created_time=100.0, finished_time=103.0)
        attrs = stats.convert_to_gen_ai_span_attrs()
        self.assertIn("gen_ai.latency.e2e", attrs)
        self.assertNotIn("gen_ai.latency.time_to_first_token", attrs)


class TestSchedulerReqTimeStats(CustomTestCase):

    def _make(self, mode=DisaggregationMode.NULL, **kwargs):
        """Helper: create SchedulerReqTimeStats with overridden fields."""
        stats = SchedulerReqTimeStats(disagg_mode=mode)
        for k, v in kwargs.items():
            setattr(stats, k, v)
        return stats

    def test_get_queueing_time(self):
        """Test forward_entry_time - wait_queue_entry_time."""
        stats = self._make(wait_queue_entry_time=100.0, forward_entry_time=105.0)
        self.assertAlmostEqual(stats.get_queueing_time(), 5.0)

    def test_get_prefill_waiting_latency_with_value(self):
        """Test prefill_run_batch_start_time - forward_entry_time."""
        stats = self._make(forward_entry_time=100.0, prefill_run_batch_start_time=100.3)
        self.assertAlmostEqual(stats.get_prefill_waiting_latency(), 0.3)

    def test_get_prefill_waiting_latency_returns_none(self):
        """Test that None is returned when prefill_run_batch_start_time is 0."""
        stats = self._make(forward_entry_time=100.0)
        self.assertIsNone(stats.get_prefill_waiting_latency())

    def test_get_prefill_launch_latency_with_values(self):
        """Test prefill_run_batch_end_time - prefill_run_batch_start_time."""
        stats = self._make(
            prefill_run_batch_start_time=100.0, prefill_run_batch_end_time=100.5
        )
        self.assertAlmostEqual(stats.get_prefill_launch_latency(), 0.5)

    def test_get_prefill_launch_latency_returns_none_when_partial(self):
        """Test that None is returned if either start or end time is zero."""
        self.assertIsNone(
            self._make(prefill_run_batch_start_time=100.0).get_prefill_launch_latency()
        )
        self.assertIsNone(
            self._make(prefill_run_batch_end_time=100.5).get_prefill_launch_latency()
        )

    def test_format_duration(self):
        """Test that duration in seconds is formatted as milliseconds."""
        stats = SchedulerReqTimeStats()
        self.assertEqual(stats.format_duration(0.0), "0.00ms")
        self.assertEqual(stats.format_duration(0.001), "1.00ms")
        self.assertEqual(stats.format_duration(0.5), "500.00ms")
        self.assertEqual(stats.format_duration(1.0), "1000.00ms")
        self.assertEqual(stats.format_duration(0.12345), "123.45ms")

    def test_convert_to_duration_unified(self):
        """Test NULL (unified) mode: queue + forward durations."""
        stats = self._make(
            mode=DisaggregationMode.NULL,
            wait_queue_entry_time=100.0,
            forward_entry_time=102.0,
            completion_time=107.0,
        )
        result = stats.convert_to_duration()
        self.assertIn("queue_duration=2000.00ms", result)
        self.assertIn("forward_duration=5000.00ms", result)
        self.assertIn("start_time=100.000", result)

    def test_convert_to_duration_prefill_without_bootstrap(self):
        """Test PREFILL mode without bootstrap_done_time sub-phase."""
        stats = self._make(
            mode=DisaggregationMode.PREFILL,
            prefill_bootstrap_queue_entry_time=100.0,
            wait_queue_entry_time=103.0,
            forward_entry_time=105.0,
            completion_time=110.0,
        )
        result = stats.convert_to_duration()
        self.assertIn("bootstrap_queue_duration(3000.00ms)", result)
        self.assertIn("queue_duration=2000.00ms", result)
        self.assertIn("forward_duration=5000.00ms", result)
        self.assertIn("start=100.000", result)
        self.assertIn("#retries=0", result)
        self.assertNotIn("bootstrap(", result)

    def test_convert_to_duration_prefill_with_bootstrap(self):
        """Test PREFILL mode with bootstrap_done_time breakdown."""
        stats = self._make(
            mode=DisaggregationMode.PREFILL,
            prefill_bootstrap_queue_entry_time=100.0,
            bootstrap_done_time=101.5,
            wait_queue_entry_time=103.0,
            forward_entry_time=105.0,
            completion_time=110.0,
        )
        result = stats.convert_to_duration()
        self.assertIn("bootstrap(1500.00ms)", result)
        self.assertIn("alloc_wait(1500.00ms)", result)

    def test_convert_to_duration_prefill_retry_count(self):
        """Test that prefill_retry_count appears in the PREFILL duration string."""
        stats = self._make(
            mode=DisaggregationMode.PREFILL,
            prefill_bootstrap_queue_entry_time=100.0,
            wait_queue_entry_time=103.0,
            forward_entry_time=105.0,
            completion_time=110.0,
            prefill_retry_count=3,
        )
        self.assertIn("#retries=3", stats.convert_to_duration())

    def test_convert_to_duration_decode_without_bootstrap(self):
        """Test DECODE mode without bootstrap_done_time sub-phase."""
        stats = self._make(
            mode=DisaggregationMode.DECODE,
            decode_prealloc_queue_entry_time=100.0,
            decode_transfer_queue_entry_time=102.0,
            wait_queue_entry_time=104.0,
            forward_entry_time=106.0,
            completion_time=110.0,
        )
        result = stats.convert_to_duration()
        self.assertIn("prealloc_queue_duration(2000.00ms)", result)
        self.assertIn("transfer_duration=2000.00ms", result)
        self.assertIn("queue_duration=2000.00ms", result)
        self.assertIn("forward_duration=4000.00ms", result)
        self.assertIn("start=100.000", result)
        self.assertNotIn("bootstrap(", result)

    def test_convert_to_duration_decode_with_bootstrap(self):
        """Test DECODE mode with bootstrap_done_time breakdown."""
        stats = self._make(
            mode=DisaggregationMode.DECODE,
            decode_prealloc_queue_entry_time=100.0,
            bootstrap_done_time=101.0,
            decode_transfer_queue_entry_time=102.0,
            wait_queue_entry_time=104.0,
            forward_entry_time=106.0,
            completion_time=110.0,
        )
        result = stats.convert_to_duration()
        self.assertIn("bootstrap(1000.00ms)", result)
        self.assertIn("alloc_wait(1000.00ms)", result)

    def test_convert_to_duration_unknown_mode(self):
        """Test that an unrecognized disagg_mode returns the fallback string."""
        stats = SchedulerReqTimeStats()
        stats.disagg_mode = "invalid"
        self.assertEqual(stats.convert_to_duration(), "Unknown Time Stats")

    def test_getstate_metrics_disabled_returns_empty(self):
        """Test that __getstate__ returns {} when enable_metrics is False."""
        stats = self._make(wait_queue_entry_time=100.0, forward_entry_time=105.0)
        stats.enable_metrics = False
        self.assertEqual(stats.__getstate__(), {})

    def test_getstate_metrics_enabled_includes_times(self):
        """Test that __getstate__ includes time fields when enable_metrics is True."""
        stats = self._make(
            wait_queue_entry_time=100.0,
            forward_entry_time=105.0,
            prefill_finished_time=105.0,
        )
        stats.enable_metrics = True
        state = stats.__getstate__()
        self.assertEqual(state["wait_queue_entry_time"], 100.0)
        self.assertEqual(state["forward_entry_time"], 105.0)
        self.assertIn("diff_realtime_monotonic", state)

    def test_convert_to_output_meta_info(self):
        """Test that scheduler meta info includes queue_time and latencies."""
        stats = self._make(
            forward_entry_time=100.0,
            prefill_finished_time=100.5,
            wait_queue_entry_time=99.0,
            prefill_run_batch_start_time=100.1,
            prefill_run_batch_end_time=100.4,
        )
        meta = stats.convert_to_output_meta_info()
        self.assertAlmostEqual(meta["queue_time"], 1.0)
        self.assertAlmostEqual(meta["prefill_waiting_latency"], 0.1)
        self.assertAlmostEqual(meta["prefill_launch_latency"], 0.3)
        self.assertIn("forward_entry_time", meta)
        self.assertIn("prefill_finished_time", meta)

    def test_compute_kv_transfer_metrics_basic(self):
        """Test KV transfer speed/size calculation with known inputs."""
        from sglang.srt.disaggregation.utils import kv_to_page_num

        stats = self._make(
            prefill_transfer_queue_entry_time=100.0,
            completion_time=101.0,
        )
        result = stats.compute_and_observe_kv_transfer_metrics(
            num_tokens=128, page_size=16, bytes_per_page_all_layers=1024 * 1024
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["latency_ms"], 1000.0)
        num_pages = kv_to_page_num(128, 16)
        expected_mb = (1024 * 1024 * num_pages) / (1024 * 1024)
        self.assertAlmostEqual(result["total_mb"], expected_mb)
        self.assertAlmostEqual(stats.transfer_total_mb, expected_mb)
        expected_speed = (expected_mb / 1024) / 1.0
        self.assertAlmostEqual(result["speed_gb_s"], expected_speed)

    def test_compute_kv_transfer_metrics_with_bootstrap(self):
        """Test that bootstrap_ms and alloc_ms are computed when sub-phase times exist."""
        stats = self._make(
            prefill_transfer_queue_entry_time=100.0,
            completion_time=101.0,
            prefill_bootstrap_queue_entry_time=98.0,
            bootstrap_done_time=99.0,
            wait_queue_entry_time=99.5,
        )
        result = stats.compute_and_observe_kv_transfer_metrics(
            num_tokens=64, page_size=16, bytes_per_page_all_layers=512
        )
        self.assertAlmostEqual(result["bootstrap_ms"], 1000.0)
        self.assertAlmostEqual(result["alloc_ms"], 500.0)

    def test_compute_kv_transfer_metrics_returns_none_when_no_times(self):
        """Test that None is returned when timestamps are all zero."""
        stats = self._make()
        result = stats.compute_and_observe_kv_transfer_metrics(
            num_tokens=128, page_size=16, bytes_per_page_all_layers=1024
        )
        self.assertIsNone(result)

    def test_compute_kv_transfer_zero_latency(self):
        """Test that zero transfer latency produces speed_gb_s=0 (no division by zero)."""
        stats = self._make(
            prefill_transfer_queue_entry_time=100.0, completion_time=100.0
        )
        result = stats.compute_and_observe_kv_transfer_metrics(
            num_tokens=128, page_size=16, bytes_per_page_all_layers=1024
        )
        self.assertAlmostEqual(result["latency_ms"], 0.0)
        self.assertAlmostEqual(result["speed_gb_s"], 0.0)


class TestTimeFunctions(CustomTestCase):

    def test_real_time_returns_positive(self):
        """Test that real_time() returns a current wall-clock timestamp."""
        self.assertGreater(real_time(), 0)

    def test_monotonic_time_returns_positive(self):
        """Test that monotonic_time() returns a current perf_counter value."""
        self.assertGreater(monotonic_time(), 0)


class TestReqTimeStatsBaseLifecycle(CustomTestCase):

    def test_new_from_obj_with_none(self):
        """Test that new_from_obj(None) returns a fresh default instance."""
        new = SchedulerReqTimeStats.new_from_obj(None)
        self.assertEqual(new.wait_queue_entry_time, 0.0)
        self.assertEqual(new.forward_entry_time, 0.0)

    def test_new_from_obj_copies_fields(self):
        """Test that new_from_obj copies matching fields from the source object."""
        old = SchedulerReqTimeStats()
        old.wait_queue_entry_time = 42.0
        old.forward_entry_time = 43.0
        new = SchedulerReqTimeStats.new_from_obj(old)
        self.assertAlmostEqual(new.wait_queue_entry_time, 42.0)
        self.assertAlmostEqual(new.forward_entry_time, 43.0)

    def test_set_metrics_collector_enables_metrics(self):
        """Test that set_metrics_collector flips enable_metrics to True."""
        from unittest.mock import MagicMock

        base = ReqTimeStatsBase()
        self.assertFalse(base.enable_metrics)
        collector = MagicMock()
        base.set_metrics_collector(collector)
        self.assertTrue(base.enable_metrics)
        self.assertIs(base.metrics_collector, collector)

    def test_set_metrics_collector_none_does_nothing(self):
        """Test that set_metrics_collector(None) keeps metrics disabled."""
        base = ReqTimeStatsBase()
        base.set_metrics_collector(None)
        self.assertFalse(base.enable_metrics)

    def test_init_trace_ctx_creates_null_context(self):
        """Test that init_trace_ctx creates TraceNullContext when OTel is off."""
        base = ReqTimeStatsBase()
        base.init_trace_ctx(rid="test-rid", bootstrap_room=None)
        self.assertIsInstance(base.trace_ctx, TraceNullContext)

    def test_disagg_mode_str_unknown(self):
        """Test that an unrecognized mode returns 'unknown'."""
        base = ReqTimeStatsBase()
        base.disagg_mode = "something_else"
        self.assertEqual(base.disagg_mode_str(), "unknown")


class TestAPIServerSetters(CustomTestCase):

    def _make(self, **kwargs):
        """Helper: create APIServerReqTimeStats with overridden fields."""
        stats = APIServerReqTimeStats()
        for k, v in kwargs.items():
            setattr(stats, k, v)
        return stats

    def test_set_created_time_explicit(self):
        """Test that set_created_time stores the explicit timestamp."""
        stats = APIServerReqTimeStats()
        stats.set_created_time(ts=10.0)
        self.assertEqual(stats.created_time, 10.0)

    def test_set_created_time_auto(self):
        """Test that set_created_time auto-generates a positive timestamp."""
        stats = APIServerReqTimeStats()
        stats.set_created_time()
        self.assertGreater(stats.created_time, 0)

    def test_set_finished_time(self):
        """Test that set_finished_time stores the timestamp."""
        stats = APIServerReqTimeStats()
        stats.set_finished_time(ts=20.0)
        self.assertEqual(stats.finished_time, 20.0)

    def test_set_first_token_time_also_sets_last_time(self):
        """Test that set_first_token_time updates both first_token_time and last_time."""
        stats = APIServerReqTimeStats()
        stats.set_first_token_time(ts=15.0)
        self.assertEqual(stats.first_token_time, 15.0)
        self.assertEqual(stats.last_time, 15.0)

    def test_set_last_time(self):
        """Test that set_last_time stores the timestamp."""
        stats = APIServerReqTimeStats()
        stats.set_last_time(ts=16.0)
        self.assertEqual(stats.last_time, 16.0)

    def test_set_tokenize_finish_time(self):
        """Test that set_tokenize_finish_time stores the timestamp."""
        stats = APIServerReqTimeStats()
        stats.set_tokenize_finish_time(ts=11.0)
        self.assertEqual(stats.tokenize_finish_time, 11.0)

    def test_set_api_server_dispatch_times(self):
        """Test that dispatch start/finish setters store timestamps."""
        stats = APIServerReqTimeStats()
        stats.set_api_server_dispatch_time(ts=12.0)
        self.assertEqual(stats.api_server_dispatch_time, 12.0)
        stats.set_api_server_dispatch_finish_time(ts=13.0)
        self.assertEqual(stats.api_server_dispatch_finish_time, 13.0)

    def test_set_response_sent_to_client_time(self):
        """Test that set_response_sent_to_client_time stores the timestamp."""
        stats = APIServerReqTimeStats()
        stats.set_response_sent_to_client_time(ts=25.0)
        self.assertEqual(stats.response_sent_to_client_time, 25.0)

    def test_get_interval(self):
        """Test that get_interval returns time since last_time."""
        stats = self._make(last_time=monotonic_time())
        import time

        time.sleep(0.01)
        self.assertGreater(stats.get_interval(), 0.005)

    def test_get_response_sent_to_client_realtime(self):
        """Test that realtime conversion is applied to the client timestamp."""
        stats = self._make(response_sent_to_client_time=100.0)
        result = stats.get_response_sent_to_client_realtime()
        self.assertAlmostEqual(result, convert_time_to_realtime(100.0))

    def test_getstate_includes_base_state(self):
        """Test that API __getstate__ merges with base __getstate__."""
        stats = APIServerReqTimeStats()
        state = stats.__getstate__()
        self.assertIn("enable_metrics", state)
        self.assertFalse(state["enable_metrics"])

    def test_convert_to_output_meta_info_all_timestamps(self):
        """Test meta info with all possible timestamp fields set."""
        stats = self._make(
            created_time=100.0,
            finished_time=103.0,
            first_token_time=101.0,
            last_time=101.0,
            api_server_dispatch_finish_time=100.5,
            response_sent_to_client_time=103.5,
        )
        meta = stats.convert_to_output_meta_info(completion_tokens=10)
        self.assertIn("request_received_ts", meta)
        self.assertIn("api_server_dispatch_finish_ts", meta)
        self.assertIn("response_sent_to_client_ts", meta)
        self.assertIn("request_finished_ts", meta)
        self.assertAlmostEqual(meta["decode_throughput"], 5.0)


class TestDPControllerReqTimeStats(CustomTestCase):

    def test_set_dp_dispatch_time(self):
        """Test that dispatch timestamp is stored."""
        stats = DPControllerReqTimeStats()
        stats.set_dp_dispatch_time(ts=50.0)
        self.assertEqual(stats.dc_dispatch_time, 50.0)

    def test_set_dp_dispatch_finish_time(self):
        """Test that dispatch finish timestamp is stored."""
        stats = DPControllerReqTimeStats()
        stats.set_dp_dispatch_finish_time(ts=55.0)
        self.assertEqual(stats.dc_dispatch_finish_time, 55.0)

    def test_getstate_includes_base_state(self):
        """Test that DP __getstate__ merges with base __getstate__."""
        stats = DPControllerReqTimeStats()
        state = stats.__getstate__()
        self.assertIn("enable_metrics", state)


class TestSchedulerReqTimeStatsSetters(CustomTestCase):

    def _make(self, mode=DisaggregationMode.NULL, **kwargs):
        """Helper: create SchedulerReqTimeStats with overridden fields."""
        stats = SchedulerReqTimeStats(disagg_mode=mode)
        for k, v in kwargs.items():
            setattr(stats, k, v)
        return stats

    def test_set_scheduler_recv_time(self):
        """Test that recv time is stored and calibrate_time_diff is called."""
        stats = SchedulerReqTimeStats()
        stats.set_scheduler_recv_time(ts=60.0)
        self.assertEqual(stats.scheduler_recv_time, 60.0)

    def test_set_scheduler_recv_time_auto(self):
        """Test that auto-timestamp produces a positive value."""
        stats = SchedulerReqTimeStats()
        stats.set_scheduler_recv_time()
        self.assertGreater(stats.scheduler_recv_time, 0)

    def test_set_retract_time_resets_fields(self):
        """Test that set_retract_time zeros out all tracking timestamps."""
        stats = self._make(
            last_forward_entry_time=1.0,
            last_prefill_finished_time=2.0,
            last_chunked_prefill_finish_time=3.0,
            last_decode_finish_time=4.0,
            last_decode_scheduled_time=5.0,
        )
        stats.set_retract_time(ts=99.0)
        self.assertEqual(stats.last_forward_entry_time, 0.0)
        self.assertEqual(stats.last_prefill_finished_time, 0.0)
        self.assertEqual(stats.last_chunked_prefill_finish_time, 0.0)
        self.assertEqual(stats.last_decode_finish_time, 0.0)
        self.assertEqual(stats.last_decode_scheduled_time, 0.0)

    def test_set_wait_queue_entry_time_first_call(self):
        """Test that first call stores the timestamp when wait_queue == 0."""
        stats = SchedulerReqTimeStats()
        stats.set_wait_queue_entry_time(ts=70.0)
        self.assertEqual(stats.wait_queue_entry_time, 70.0)

    def test_set_wait_queue_entry_time_second_call_triggers_retract(self):
        """Test that second call triggers set_retract_time logic."""
        stats = self._make(
            wait_queue_entry_time=70.0,
            last_forward_entry_time=1.0,
            last_prefill_finished_time=2.0,
        )
        stats.set_wait_queue_entry_time(ts=80.0)
        self.assertEqual(stats.wait_queue_entry_time, 80.0)
        self.assertEqual(stats.last_forward_entry_time, 0.0)

    def test_set_forward_entry_time_first_call(self):
        """Test that first call sets both forward_entry_time and last_forward_entry_time."""
        stats = self._make(wait_queue_entry_time=70.0)
        stats.set_forward_entry_time(ts=75.0)
        self.assertEqual(stats.forward_entry_time, 75.0)
        self.assertEqual(stats.last_forward_entry_time, 75.0)

    def test_set_forward_entry_time_after_retract(self):
        """Test that after retract (forward!=0, last_forward==0) only last is set."""
        stats = self._make(forward_entry_time=75.0, last_forward_entry_time=0.0)
        stats.set_forward_entry_time(ts=80.0)
        self.assertEqual(stats.forward_entry_time, 75.0)
        self.assertEqual(stats.last_forward_entry_time, 80.0)

    def test_set_prefill_run_batch_times(self):
        """Test that batch start/end timestamps are stored."""
        stats = SchedulerReqTimeStats()
        stats.set_prefill_run_batch_start_time(ts=100.0)
        stats.set_prefill_run_batch_end_time(ts=101.0)
        self.assertEqual(stats.prefill_run_batch_start_time, 100.0)
        self.assertEqual(stats.prefill_run_batch_end_time, 101.0)

    def test_set_completion_time(self):
        """Test that set_completion_time stores timestamp."""
        stats = SchedulerReqTimeStats()
        stats.set_completion_time(ts=200.0)
        self.assertEqual(stats.completion_time, 200.0)

    def test_set_quick_finish_time(self):
        """Test that quick_finish sets both completion and forward_entry."""
        stats = SchedulerReqTimeStats()
        stats.set_quick_finish_time(ts=150.0)
        self.assertEqual(stats.completion_time, 150.0)
        self.assertEqual(stats.forward_entry_time, 150.0)

    def test_set_prefill_bootstrap_queue_entry_time(self):
        """Test that bootstrap queue entry timestamp is stored."""
        stats = self._make(scheduler_recv_time=50.0)
        stats.set_prefill_bootstrap_queue_entry_time(ts=55.0)
        self.assertEqual(stats.prefill_bootstrap_queue_entry_time, 55.0)

    def test_set_prefill_transfer_queue_entry_time(self):
        """Test that transfer queue entry timestamp is stored."""
        stats = SchedulerReqTimeStats()
        stats.set_prefill_transfer_queue_entry_time(ts=60.0)
        self.assertEqual(stats.prefill_transfer_queue_entry_time, 60.0)

    def test_set_prefill_kv_transfer_finish_time(self):
        """Test that KV transfer finish timestamp is stored."""
        stats = self._make(prefill_transfer_queue_entry_time=60.0)
        stats.set_prefill_kv_transfer_finish_time(ts=65.0)
        self.assertEqual(stats.prefill_kv_transfer_finish_time, 65.0)

    def test_set_decode_prealloc_queue_entry_time(self):
        """Test that decode prealloc queue entry timestamp is stored."""
        stats = self._make(scheduler_recv_time=40.0)
        stats.set_decode_prealloc_queue_entry_time(ts=45.0)
        self.assertEqual(stats.decode_prealloc_queue_entry_time, 45.0)

    def test_set_decode_transfer_queue_entry_time(self):
        """Test that decode transfer queue entry timestamp is stored."""
        stats = self._make(decode_prealloc_queue_entry_time=45.0)
        stats.set_decode_transfer_queue_entry_time(ts=48.0)
        self.assertEqual(stats.decode_transfer_queue_entry_time, 48.0)

    def test_set_bootstrap_done_time_first_call_only(self):
        """Test that bootstrap_done_time is only set on the first call."""
        stats = SchedulerReqTimeStats()
        stats.set_bootstrap_done_time(ts=50.0)
        self.assertEqual(stats.bootstrap_done_time, 50.0)
        stats.set_bootstrap_done_time(ts=60.0)
        self.assertEqual(stats.bootstrap_done_time, 50.0)

    def test_set_decode_prebuilt_finish_time(self):
        """Test that decode prebuilt finish timestamp is stored."""
        stats = self._make(last_forward_entry_time=80.0)
        stats.set_decode_prebuilt_finish_time(ts=85.0)
        self.assertEqual(stats.decode_prebuilt_finish_time, 85.0)

    def test_set_prefill_finished_time_first_call(self):
        """Test that first call sets both prefill_finished and last_prefill_finished."""
        stats = self._make(last_forward_entry_time=70.0)
        stats.set_prefill_finished_time(ts=80.0)
        self.assertEqual(stats.prefill_finished_time, 80.0)
        self.assertEqual(stats.last_prefill_finished_time, 80.0)

    def test_set_prefill_finished_time_after_retract(self):
        """Test that after retract (prefill!=0, last_prefill==0) only last is set."""
        stats = self._make(
            prefill_finished_time=80.0,
            last_prefill_finished_time=0.0,
            last_forward_entry_time=70.0,
        )
        stats.set_prefill_finished_time(ts=90.0)
        self.assertEqual(stats.prefill_finished_time, 80.0)
        self.assertEqual(stats.last_prefill_finished_time, 90.0)


def _ensure_otel():
    """Initialize real OpenTelemetry tracing (once per process)."""
    import sglang.srt.observability.trace as tr

    if not tr.opentelemetry_initialized:
        tr.process_tracing_init("localhost:4317", "unit-test-server")


class TestSettersWithTracing(CustomTestCase):
    """Test setter methods with real OpenTelemetry tracing active."""

    @classmethod
    def setUpClass(cls):
        _ensure_otel()

    def _make_traced(self, mode=DisaggregationMode.NULL, **kwargs):
        """Helper: create SchedulerReqTimeStats with a live trace context."""
        from sglang.srt.observability.trace import TraceReqContext

        stats = SchedulerReqTimeStats(disagg_mode=mode)
        stats.trace_ctx = TraceReqContext(
            rid="traced-req", role=stats.disagg_mode_str(), module_name="request"
        )
        stats.trace_ctx.trace_req_start()
        for k, v in kwargs.items():
            setattr(stats, k, v)
        return stats

    def _finish(self, stats):
        if hasattr(stats, "trace_ctx") and stats.trace_ctx.tracing_enable:
            stats.trace_ctx.trace_req_finish()

    def test_set_wait_queue_entry_unified(self):
        """Test set_wait_queue_entry_time in unified mode with tracing."""
        stats = self._make_traced(scheduler_recv_time=100.0)
        stats.set_wait_queue_entry_time(ts=105.0)
        self.assertEqual(stats.wait_queue_entry_time, 105.0)
        self._finish(stats)

    def test_set_wait_queue_entry_prefill(self):
        """Test set_wait_queue_entry_time in PREFILL mode exercises PREFILL_BOOTSTRAP."""
        stats = self._make_traced(
            mode=DisaggregationMode.PREFILL,
            prefill_bootstrap_queue_entry_time=100.0,
        )
        stats.set_wait_queue_entry_time(ts=105.0)
        self.assertEqual(stats.wait_queue_entry_time, 105.0)
        self._finish(stats)

    def test_set_wait_queue_entry_decode(self):
        """Test set_wait_queue_entry_time in DECODE mode exercises DECODE_TRANSFERRED."""
        stats = self._make_traced(
            mode=DisaggregationMode.DECODE,
            decode_transfer_queue_entry_time=100.0,
        )
        stats.set_wait_queue_entry_time(ts=105.0)
        self.assertEqual(stats.wait_queue_entry_time, 105.0)
        self._finish(stats)

    def test_set_forward_entry_time_unified_with_tracing(self):
        """Test set_forward_entry_time in unified mode creates prefill slice."""
        stats = self._make_traced(wait_queue_entry_time=100.0)
        stats.set_forward_entry_time(ts=105.0)
        self.assertEqual(stats.forward_entry_time, 105.0)
        self.assertEqual(stats.last_forward_entry_time, 105.0)
        self._finish(stats)

    def test_set_forward_entry_time_decode_with_tracing(self):
        """Test set_forward_entry_time in DECODE mode creates decode waiting slice."""
        stats = self._make_traced(
            mode=DisaggregationMode.DECODE,
            wait_queue_entry_time=100.0,
        )
        stats.set_forward_entry_time(ts=105.0)
        self.assertEqual(stats.forward_entry_time, 105.0)
        self._finish(stats)

    def test_set_prefill_finished_with_tracing(self):
        """Test set_prefill_finished_time closes the prefill forward slice."""
        stats = self._make_traced(wait_queue_entry_time=100.0)
        stats.set_forward_entry_time(ts=105.0)
        stats.set_prefill_finished_time(ts=110.0)
        self.assertEqual(stats.prefill_finished_time, 110.0)
        self._finish(stats)

    def test_set_prefill_finished_with_chunked(self):
        """Test set_prefill_finished_time when chunked prefill was tracked."""
        stats = self._make_traced(wait_queue_entry_time=100.0)
        stats.set_forward_entry_time(ts=105.0)
        stats.set_last_chunked_prefill_finish_time(ts=107.0)
        stats.set_prefill_finished_time(ts=110.0)
        self.assertEqual(stats.prefill_finished_time, 110.0)
        self._finish(stats)

    def test_set_last_chunked_prefill_with_tracing(self):
        """Test set_last_chunked_prefill_finish_time creates chunked prefill slice."""
        stats = self._make_traced(last_forward_entry_time=100.0)
        stats.set_last_chunked_prefill_finish_time(ts=102.0)
        self.assertEqual(stats.last_chunked_prefill_finish_time, 102.0)
        self._finish(stats)

    def test_set_last_decode_finish_with_tracing(self):
        """Test set_last_decode_finish_time creates decode loop slice."""
        stats = self._make_traced(
            last_prefill_finished_time=100.0,
            last_decode_scheduled_time=100.0,
        )
        stats.set_last_decode_finish_time(ts=101.0)
        self.assertEqual(stats.last_decode_finish_time, 101.0)
        self.assertEqual(stats.decode_ct, 1)
        self._finish(stats)

    def test_set_last_decode_finish_decode_mode(self):
        """Test set_last_decode_finish_time in DECODE mode uses decode_prebuilt_finish."""
        stats = self._make_traced(
            mode=DisaggregationMode.DECODE,
            decode_prebuilt_finish_time=100.0,
        )
        stats.set_last_decode_finish_time(ts=101.0)
        self.assertEqual(stats.decode_ct, 1)
        self._finish(stats)

    def test_set_completion_time_calls_abort(self):
        """Test that set_completion_time triggers trace abort."""
        stats = self._make_traced()
        stats.set_completion_time(ts=200.0)
        self.assertEqual(stats.completion_time, 200.0)


class TestAPIServerSettersWithTracing(CustomTestCase):
    """Test the complete API server timing lifecycle with real tracing."""

    @classmethod
    def setUpClass(cls):
        _ensure_otel()

    def test_full_api_server_lifecycle(self):
        """Test the full API server lifecycle: init → setters → finish → verify."""
        stats = APIServerReqTimeStats()
        stats.init_trace_ctx(rid="api-lifecycle", bootstrap_room=None)
        self.assertNotIsInstance(stats.trace_ctx, TraceNullContext)

        stats.set_created_time(ts=100.0)
        stats.set_tokenize_finish_time(ts=101.0)
        stats.set_api_server_dispatch_time(ts=101.5)
        stats.set_api_server_dispatch_finish_time(ts=102.0)
        stats.set_first_token_time(ts=103.0)
        stats.set_last_time(ts=104.0)
        stats.set_response_sent_to_client_time(ts=104.5)
        stats.set_finished_time(ts=105.0)

        self.assertAlmostEqual(stats.get_e2e_latency(), 5.0)
        self.assertAlmostEqual(stats.get_first_token_latency(), 3.0)


class TestSettersAutoTimestamp(CustomTestCase):
    """Test that setter ts=None branch auto-generates via time.perf_counter()."""

    def test_api_server_setters_auto(self):
        """Test that all API server setters produce positive timestamps when ts=None."""
        stats = APIServerReqTimeStats()
        stats.set_created_time()
        self.assertGreater(stats.created_time, 0)
        stats.set_tokenize_finish_time()
        self.assertGreater(stats.tokenize_finish_time, 0)
        stats.set_api_server_dispatch_time()
        self.assertGreater(stats.api_server_dispatch_time, 0)
        stats.set_api_server_dispatch_finish_time()
        self.assertGreater(stats.api_server_dispatch_finish_time, 0)
        stats.set_first_token_time()
        self.assertGreater(stats.first_token_time, 0)
        stats.set_last_time()
        self.assertGreater(stats.last_time, 0)
        stats.set_response_sent_to_client_time()
        self.assertGreater(stats.response_sent_to_client_time, 0)
        stats.set_finished_time()
        self.assertGreater(stats.finished_time, 0)

    def test_dp_controller_setters_auto(self):
        """Test that DP controller setters produce positive timestamps when ts=None."""
        stats = DPControllerReqTimeStats()
        stats.set_dp_dispatch_time()
        self.assertGreater(stats.dc_dispatch_time, 0)
        stats.set_dp_dispatch_finish_time()
        self.assertGreater(stats.dc_dispatch_finish_time, 0)

    def test_scheduler_setters_auto(self):
        """Test that scheduler setters produce positive timestamps when ts=None."""
        stats = SchedulerReqTimeStats()
        stats.set_scheduler_recv_time()
        self.assertGreater(stats.scheduler_recv_time, 0)
        stats.set_wait_queue_entry_time()
        self.assertGreater(stats.wait_queue_entry_time, 0)
        stats.set_forward_entry_time()
        self.assertGreater(stats.forward_entry_time, 0)
        stats.set_prefill_run_batch_start_time()
        self.assertGreater(stats.prefill_run_batch_start_time, 0)
        stats.set_prefill_run_batch_end_time()
        self.assertGreater(stats.prefill_run_batch_end_time, 0)
        stats.set_completion_time()
        self.assertGreater(stats.completion_time, 0)


class TestSetLastScheduledTimeWithTracing(CustomTestCase):
    """Test set_last_scheduled_time with real tracing for decode-forward slice creation."""

    @classmethod
    def setUpClass(cls):
        _ensure_otel()

    def _make_traced(self, mode=DisaggregationMode.NULL, **kwargs):
        from sglang.srt.observability.trace import TraceReqContext

        stats = SchedulerReqTimeStats(disagg_mode=mode)
        stats.trace_ctx = TraceReqContext(
            rid="sched-test", role=stats.disagg_mode_str(), module_name="request"
        )
        stats.trace_ctx.trace_req_start()
        for k, v in kwargs.items():
            setattr(stats, k, v)
        return stats

    def test_decode_creates_waiting_slice(self):
        """Test that first decode schedule in unified mode creates DECODE_WAITING slice."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        stats = self._make_traced(
            last_prefill_finished_time=100.0,
            last_decode_scheduled_time=0.0,
        )
        stats.set_last_scheduled_time(ForwardMode.DECODE, ts=105.0)
        self.assertEqual(stats.last_decode_scheduled_time, 105.0)
        self.assertEqual(stats.last_decode_finish_time, 105.0)
        stats.trace_ctx.trace_req_finish()

    def test_extend_does_not_create_decode_slice(self):
        """Test that EXTEND mode does not trigger decode slice creation."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        stats = self._make_traced(last_prefill_finished_time=100.0)
        stats.set_last_scheduled_time(ForwardMode.EXTEND, ts=105.0)
        self.assertEqual(stats.last_decode_scheduled_time, 0.0)
        stats.trace_ctx.trace_req_finish()


class TestSetTimeBatch(CustomTestCase):
    """Test the set_time_batch utility function."""

    def test_empty_list_is_noop(self):
        """Test that empty/None lists return immediately."""
        from sglang.srt.observability.req_time_stats import set_time_batch

        set_time_batch(None, "set_completion_time")
        set_time_batch([], "set_completion_time")

    def test_calls_setter_on_each_req(self):
        """Test that set_time_batch calls the named setter on each request."""
        from types import SimpleNamespace

        from sglang.srt.observability.req_time_stats import set_time_batch

        stats1 = SchedulerReqTimeStats()
        stats2 = SchedulerReqTimeStats()
        reqs = [
            SimpleNamespace(time_stats=stats1),
            SimpleNamespace(time_stats=stats2),
        ]
        set_time_batch(reqs, "set_completion_time")
        self.assertGreater(stats1.completion_time, 0)
        self.assertGreater(stats2.completion_time, 0)


class TestSchedulerSettersAutoTimestamp(CustomTestCase):
    """Test remaining scheduler setters with ts=None to cover auto-timestamp branches."""

    def test_set_retract_time_auto(self):
        """Test set_retract_time without explicit ts generates a timestamp."""
        stats = SchedulerReqTimeStats()
        stats.last_forward_entry_time = 1.0
        stats.set_retract_time()
        self.assertEqual(stats.last_forward_entry_time, 0.0)

    def test_set_quick_finish_time_auto(self):
        """Test set_quick_finish_time without explicit ts."""
        stats = SchedulerReqTimeStats()
        stats.set_quick_finish_time()
        self.assertGreater(stats.completion_time, 0)
        self.assertGreater(stats.forward_entry_time, 0)

    def test_set_prefill_bootstrap_queue_entry_auto(self):
        """Test set_prefill_bootstrap_queue_entry_time without explicit ts."""
        stats = SchedulerReqTimeStats(scheduler_recv_time=100.0)
        stats.set_prefill_bootstrap_queue_entry_time()
        self.assertGreater(stats.prefill_bootstrap_queue_entry_time, 0)

    def test_set_prefill_transfer_queue_entry_auto(self):
        """Test set_prefill_transfer_queue_entry_time without explicit ts."""
        stats = SchedulerReqTimeStats()
        stats.set_prefill_transfer_queue_entry_time()
        self.assertGreater(stats.prefill_transfer_queue_entry_time, 0)

    def test_set_prefill_kv_transfer_finish_auto(self):
        """Test set_prefill_kv_transfer_finish_time without explicit ts."""
        stats = SchedulerReqTimeStats(prefill_transfer_queue_entry_time=100.0)
        stats.set_prefill_kv_transfer_finish_time()
        self.assertGreater(stats.prefill_kv_transfer_finish_time, 0)

    def test_set_decode_prealloc_queue_entry_auto(self):
        """Test set_decode_prealloc_queue_entry_time without explicit ts."""
        stats = SchedulerReqTimeStats(scheduler_recv_time=100.0)
        stats.set_decode_prealloc_queue_entry_time()
        self.assertGreater(stats.decode_prealloc_queue_entry_time, 0)

    def test_set_decode_transfer_queue_entry_auto(self):
        """Test set_decode_transfer_queue_entry_time without explicit ts."""
        stats = SchedulerReqTimeStats(decode_prealloc_queue_entry_time=100.0)
        stats.set_decode_transfer_queue_entry_time()
        self.assertGreater(stats.decode_transfer_queue_entry_time, 0)

    def test_set_bootstrap_done_time_auto(self):
        """Test set_bootstrap_done_time without explicit ts."""
        stats = SchedulerReqTimeStats()
        stats.set_bootstrap_done_time()
        self.assertGreater(stats.bootstrap_done_time, 0)

    def test_set_decode_prebuilt_finish_auto(self):
        """Test set_decode_prebuilt_finish_time without explicit ts."""
        stats = SchedulerReqTimeStats(last_forward_entry_time=100.0)
        stats.set_decode_prebuilt_finish_time()
        self.assertGreater(stats.decode_prebuilt_finish_time, 0)

    def test_set_prefill_finished_auto(self):
        """Test set_prefill_finished_time without explicit ts."""
        stats = SchedulerReqTimeStats(last_forward_entry_time=100.0)
        stats.set_prefill_finished_time()
        self.assertGreater(stats.prefill_finished_time, 0)

    def test_set_last_decode_finish_auto(self):
        """Test set_last_decode_finish_time without explicit ts."""
        stats = SchedulerReqTimeStats(last_prefill_finished_time=100.0)
        stats.set_last_decode_finish_time()
        self.assertGreater(stats.last_decode_finish_time, 0)

    def test_set_last_chunked_prefill_auto(self):
        """Test set_last_chunked_prefill_finish_time without explicit ts."""
        stats = SchedulerReqTimeStats(last_forward_entry_time=100.0)
        stats.set_last_chunked_prefill_finish_time()
        self.assertGreater(stats.last_chunked_prefill_finish_time, 0)


if __name__ == "__main__":
    unittest.main()
