"""Unit tests for srt/observability/metrics_collector.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-cpu-only")

import dataclasses
import os
import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.metrics_collector import (
    DPCooperationInfo,
    ExpertDispatchCollector,
    QueueCount,
    RadixCacheMetricsCollector,
    SchedulerMetricsCollector,
    SchedulerStats,
    StorageMetricsCollector,
    TokenizerMetricsCollector,
    compute_routing_key_stats,
    get_histogram_conf_from_env,
)
from sglang.test.test_utils import CustomTestCase


class TestGetHistogramConfFromEnv(CustomTestCase):

    def setUp(self):
        self._env_key = "SGLANG_TEST_HISTOGRAM_CONF"
        self._orig = os.environ.get(self._env_key)

    def tearDown(self):
        if self._orig is None:
            os.environ.pop(self._env_key, None)
        else:
            os.environ[self._env_key] = self._orig

    def test_not_set_returns_none(self):
        """Test that missing env var returns None."""
        os.environ.pop(self._env_key, None)
        self.assertIsNone(get_histogram_conf_from_env(self._env_key))

    def test_empty_value_returns_none(self):
        """Test that empty string env var returns None."""
        os.environ[self._env_key] = ""
        self.assertIsNone(get_histogram_conf_from_env(self._env_key))

    def test_parses_csv_floats(self):
        """Test that comma-separated values are parsed to list of floats."""
        os.environ[self._env_key] = "0.1,0.5,1.0,5.0"
        result = get_histogram_conf_from_env(self._env_key)
        self.assertEqual(result, [0.1, 0.5, 1.0, 5.0])

    def test_single_value(self):
        """Test that a single value is parsed correctly."""
        os.environ[self._env_key] = "42.0"
        self.assertEqual(get_histogram_conf_from_env(self._env_key), [42.0])

    def test_integer_strings_become_floats(self):
        """Test that integer strings like '1,2,3' are converted to floats."""
        os.environ[self._env_key] = "1,2,3"
        self.assertEqual(get_histogram_conf_from_env(self._env_key), [1.0, 2.0, 3.0])


class TestQueueCount(CustomTestCase):

    def test_default_values(self):
        """Test that QueueCount defaults to total=0, by_priority=None."""
        qc = QueueCount()
        self.assertEqual(qc.total, 0)
        self.assertIsNone(qc.by_priority)

    def test_explicit_values(self):
        """Test that constructor arguments are stored correctly."""
        qc = QueueCount(total=10, by_priority={0: 5, 1: 5})
        self.assertEqual(qc.total, 10)
        self.assertEqual(qc.by_priority, {0: 5, 1: 5})

    def test_from_reqs_without_priority(self):
        """Test from_reqs with priority scheduling disabled returns None by_priority."""
        reqs = [SimpleNamespace(priority=0) for _ in range(5)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=False)
        self.assertEqual(qc.total, 5)
        self.assertIsNone(qc.by_priority)

    def test_from_reqs_with_priority(self):
        """Test from_reqs with priority scheduling enabled counts per-priority."""
        reqs = [
            SimpleNamespace(priority=0),
            SimpleNamespace(priority=0),
            SimpleNamespace(priority=1),
            SimpleNamespace(priority=2),
            SimpleNamespace(priority=1),
        ]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)
        self.assertEqual(qc.total, 5)
        self.assertEqual(qc.by_priority, {0: 2, 1: 2, 2: 1})

    def test_from_reqs_empty_list(self):
        """Test from_reqs with empty list returns total=0."""
        qc = QueueCount.from_reqs([], enable_priority_scheduling=True)
        self.assertEqual(qc.total, 0)
        self.assertEqual(qc.by_priority, {})

    def test_from_reqs_none_priority(self):
        """Test from_reqs when requests have priority=None."""
        reqs = [SimpleNamespace(priority=None), SimpleNamespace(priority=None)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)
        self.assertEqual(qc.total, 2)
        self.assertEqual(qc.by_priority, {None: 2})


class TestSchedulerStats(CustomTestCase):

    def test_default_construction(self):
        """Test that SchedulerStats defaults are all zeros."""
        stats = SchedulerStats()
        self.assertEqual(stats.num_used_tokens, 0)
        self.assertEqual(stats.token_usage, 0.0)
        self.assertIsInstance(stats.num_running_reqs, QueueCount)

    def test_queue_count_fields_are_independent(self):
        """Test that each QueueCount field is a separate instance."""
        stats = SchedulerStats()
        stats.num_running_reqs.total = 10
        self.assertEqual(stats.num_queue_reqs.total, 0)


class TestComputeRoutingKeyStats(CustomTestCase):

    def test_basic(self):
        """Test that unique count and per-key counts are correct."""
        keys = ["modelA", "modelA", "modelB", "modelC", "modelB"]
        num_unique, counts = compute_routing_key_stats(keys)
        self.assertEqual(num_unique, 3)
        self.assertEqual(sorted(counts), [1, 2, 2])

    def test_none_keys_excluded(self):
        """Test that None keys are filtered out before counting."""
        keys = [None, "modelA", None, "modelA"]
        num_unique, counts = compute_routing_key_stats(keys)
        self.assertEqual(num_unique, 1)
        self.assertEqual(counts, [2])

    def test_all_none(self):
        """Test that all-None input returns 0 unique keys."""
        num_unique, counts = compute_routing_key_stats([None, None])
        self.assertEqual(num_unique, 0)
        self.assertEqual(counts, [])

    def test_empty_list(self):
        """Test that empty list returns 0 unique keys."""
        num_unique, counts = compute_routing_key_stats([])
        self.assertEqual(num_unique, 0)
        self.assertEqual(counts, [])


class TestDPCooperationInfo(CustomTestCase):

    def test_create_all_extend(self):
        """Test that all EXTEND modes count as prefill ranks."""
        modes = [ForwardMode.EXTEND.value, ForwardMode.EXTEND.value]
        info = DPCooperationInfo.create(modes)
        self.assertEqual(info.num_prefill_ranks, 2)

    def test_create_all_decode(self):
        """Test that all DECODE modes count as 0 prefill ranks."""
        modes = [ForwardMode.DECODE.value, ForwardMode.DECODE.value]
        info = DPCooperationInfo.create(modes)
        self.assertEqual(info.num_prefill_ranks, 0)

    def test_create_mixed(self):
        """Test mix of extend and decode modes."""
        modes = [
            ForwardMode.EXTEND.value,
            ForwardMode.DECODE.value,
            ForwardMode.EXTEND.value,
        ]
        info = DPCooperationInfo.create(modes)
        self.assertEqual(info.num_prefill_ranks, 2)

    def test_create_empty(self):
        """Test that empty forward_modes produces 0 prefill ranks."""
        info = DPCooperationInfo.create([])
        self.assertEqual(info.num_prefill_ranks, 0)

    def test_to_labels(self):
        """Test that to_labels returns a dict suitable for Prometheus labels."""
        info = DPCooperationInfo(num_prefill_ranks=3)
        self.assertEqual(info.to_labels(), {"num_prefill_ranks": 3})


def _make_server_args():
    """Helper: create a minimal ServerArgs bypassing __post_init__ model download."""
    import dataclasses as dc

    from sglang.srt.server_args import ServerArgs

    sa = object.__new__(ServerArgs)
    for f in dc.fields(ServerArgs):
        val = (
            f.default
            if f.default is not dc.MISSING
            else f.default_factory()
            if f.default_factory is not dc.MISSING
            else None
        )
        setattr(sa, f.name, val)
    sa.prefill_delayer_max_delay_passes = 10
    sa.kv_transfer_config = None
    return sa


_LABELS = {"model_name": "test", "moe_ep_rank": "0"}

# --- Module-level singletons (Prometheus forbids duplicate metric names) ---
_sa = _make_server_args()
_sa.prompt_tokens_buckets = None
_sa.generation_tokens_buckets = None

_SCHED = SchedulerMetricsCollector(
    labels=_LABELS, server_args=_sa, enable_lora=True, enable_hierarchical_cache=True
)
_TOK = TokenizerMetricsCollector(
    labels=_LABELS, server_args=_sa, collect_tokens_histogram=True
)
_STORAGE = StorageMetricsCollector(labels=_LABELS)
_RADIX = RadixCacheMetricsCollector(labels=_LABELS)
_EXPERT = ExpertDispatchCollector(ep_size=8)


# --- Helpers for reading Prometheus metric values ---


def _counter_val(counter, **extra):
    """Read current counter value for the test labels."""
    return counter.labels(**{**_LABELS, **extra})._value.get()


def _gauge_val(gauge):
    """Read current gauge value for the test labels."""
    return gauge.labels(**_LABELS)._value.get()


def _hist_sum(hist, **extra):
    """Read histogram sum of observed values for the test labels."""
    return hist.labels(**{**_LABELS, **extra})._sum.get()


class TestSchedulerMetricsCollector(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.c = _SCHED

    def test_log_stats_basic(self):
        """Test log_stats sets gauge values from SchedulerStats."""
        self.c.log_stats(
            SchedulerStats(
                num_running_reqs=QueueCount(total=5),
                num_used_tokens=1000,
                token_usage=0.5,
                gen_throughput=10.0,
                num_queue_reqs=QueueCount(total=3),
                max_total_num_tokens=2000,
                cache_hit_rate=0.8,
            )
        )
        self.assertAlmostEqual(_gauge_val(self.c.num_used_tokens), 1000)
        self.assertAlmostEqual(_gauge_val(self.c.token_usage), 0.5)
        self.assertAlmostEqual(_gauge_val(self.c.gen_throughput), 10.0)
        self.assertAlmostEqual(_gauge_val(self.c.cache_hit_rate), 0.8)
        self.assertAlmostEqual(_gauge_val(self.c.max_total_num_tokens), 2000)

    def test_log_stats_with_optional_fields(self):
        """Test log_stats with speculative decoding and SLO gauge values."""
        self.c.log_stats(
            SchedulerStats(
                num_running_reqs=QueueCount(total=1),
                num_queue_reqs=QueueCount(total=0),
                max_total_num_tokens=2000,
                max_running_requests_under_SLO=50,
                engine_load_weights_time=1.5,
                num_retracted_reqs=2,
                spec_accept_rate=0.9,
                spec_accept_length=3.5,
            )
        )
        self.assertAlmostEqual(_gauge_val(self.c.spec_accept_rate), 0.9)
        self.assertAlmostEqual(_gauge_val(self.c.spec_accept_length), 3.5)
        self.assertAlmostEqual(_gauge_val(self.c.max_running_requests_under_SLO), 50)

    def test_log_stats_with_disagg_queues(self):
        """Test log_stats with disaggregation queue fields populated."""
        self.c.log_stats(
            SchedulerStats(
                num_running_reqs=QueueCount(total=2),
                num_queue_reqs=QueueCount(total=1),
                max_total_num_tokens=3000,
                num_prefill_prealloc_queue_reqs=QueueCount(total=1),
                num_decode_prealloc_queue_reqs=QueueCount(total=2),
                num_decode_transfer_queue_reqs=QueueCount(total=1),
                kv_transfer_speed_gb_s=1.5,
                kv_transfer_latency_ms=100.0,
            )
        )
        self.assertAlmostEqual(_gauge_val(self.c.max_total_num_tokens), 3000)

    def test_observe_per_stage_req_latency(self):
        """Test observe_per_stage_req_latency records to the histogram."""
        before = _hist_sum(self.c.per_stage_req_latency_seconds, stage="prefill_forward")
        self.c.observe_per_stage_req_latency("prefill_forward", 0.1)
        self.assertAlmostEqual(
            _hist_sum(self.c.per_stage_req_latency_seconds, stage="prefill_forward")
            - before,
            0.1,
        )

    def test_observe_queue_time(self):
        """Test observe_queue_time records to the histogram."""
        before = _hist_sum(self.c.queue_time)
        self.c.observe_queue_time(0.05)
        self.assertAlmostEqual(_hist_sum(self.c.queue_time) - before, 0.05)

    def test_observe_kv_transfer_metrics(self):
        """Test observe_kv_transfer_metrics records all three histograms."""
        b_lat = _hist_sum(self.c.kv_transfer_latency_ms)
        b_mb = _hist_sum(self.c.kv_transfer_total_mb)
        b_spd = _hist_sum(self.c.kv_transfer_speed_gb_s)
        self.c.observe_kv_transfer_metrics(
            latency_ms=100.0, total_mb=8.0, speed_gb_s=0.08
        )
        self.assertAlmostEqual(_hist_sum(self.c.kv_transfer_latency_ms) - b_lat, 100.0)
        self.assertAlmostEqual(_hist_sum(self.c.kv_transfer_total_mb) - b_mb, 8.0)
        self.assertAlmostEqual(
            _hist_sum(self.c.kv_transfer_speed_gb_s) - b_spd, 0.08
        )

    def test_observe_kv_transfer_bootstrap(self):
        """Test observe_kv_transfer_bootstrap records both histograms."""
        b_bs = _hist_sum(self.c.kv_transfer_bootstrap_ms)
        b_al = _hist_sum(self.c.kv_transfer_alloc_ms)
        self.c.observe_kv_transfer_bootstrap(bootstrap_ms=50.0, alloc_ms=20.0)
        self.assertAlmostEqual(
            _hist_sum(self.c.kv_transfer_bootstrap_ms) - b_bs, 50.0
        )
        self.assertAlmostEqual(_hist_sum(self.c.kv_transfer_alloc_ms) - b_al, 20.0)

    def test_increment_bootstrap_failed_reqs(self):
        """Test that bootstrap failure counter is incremented by 1."""
        before = _counter_val(self.c.num_bootstrap_failed_reqs)
        self.c.increment_bootstrap_failed_reqs()
        self.assertEqual(_counter_val(self.c.num_bootstrap_failed_reqs), before + 1)

    def test_increment_transfer_failed_reqs(self):
        """Test that transfer failure counter is incremented by 1."""
        before = _counter_val(self.c.num_transfer_failed_reqs)
        self.c.increment_transfer_failed_reqs()
        self.assertEqual(_counter_val(self.c.num_transfer_failed_reqs), before + 1)

    def test_increment_prefill_retries_positive(self):
        """Test that prefill retries counter increments by count when count > 0."""
        before = _counter_val(self.c.num_prefill_retries_total)
        self.c.increment_prefill_retries(3)
        self.assertEqual(_counter_val(self.c.num_prefill_retries_total), before + 3)

    def test_increment_prefill_retries_zero_is_noop(self):
        """Test that count=0 does not increment the counter."""
        before = _counter_val(self.c.num_prefill_retries_total)
        self.c.increment_prefill_retries(0)
        self.assertEqual(_counter_val(self.c.num_prefill_retries_total), before)

    def test_observe_prefill_delayer_allow_and_execute(self):
        """Test prefill delayer records wait histograms when allow=True, execution=True."""
        b_passes = _hist_sum(self.c.prefill_delayer_wait_forward_passes)
        b_secs = _hist_sum(self.c.prefill_delayer_wait_seconds)
        self.c.observe_prefill_delayer_outcome(
            forward_passes=5,
            wait_seconds=0.1,
            input_estimation="low",
            output_allow=True,
            output_reason="ready",
            actual_execution=True,
        )
        self.assertAlmostEqual(
            _hist_sum(self.c.prefill_delayer_wait_forward_passes) - b_passes, 5
        )
        self.assertAlmostEqual(
            _hist_sum(self.c.prefill_delayer_wait_seconds) - b_secs, 0.1
        )

    def test_observe_prefill_delayer_deny(self):
        """Test that deny outcome skips wait histograms."""
        b_passes = _hist_sum(self.c.prefill_delayer_wait_forward_passes)
        self.c.observe_prefill_delayer_outcome(
            forward_passes=0,
            wait_seconds=0.0,
            input_estimation="high",
            output_allow=False,
            output_reason="delayed",
            actual_execution=False,
        )
        self.assertAlmostEqual(
            _hist_sum(self.c.prefill_delayer_wait_forward_passes), b_passes
        )

    def test_increment_retracted_reqs(self):
        """Test that all three retraction counters are incremented."""
        b_reqs = _counter_val(self.c.num_retracted_reqs_total)
        b_inp = _counter_val(self.c.num_retracted_input_tokens_total)
        b_out = _counter_val(self.c.num_retracted_output_tokens_total)
        self.c.increment_retracted_reqs(
            num_retracted_reqs=2,
            num_retracted_input_tokens=100,
            num_retracted_output_tokens=50,
        )
        self.assertEqual(_counter_val(self.c.num_retracted_reqs_total), b_reqs + 2)
        self.assertEqual(
            _counter_val(self.c.num_retracted_input_tokens_total), b_inp + 100
        )
        self.assertEqual(
            _counter_val(self.c.num_retracted_output_tokens_total), b_out + 50
        )

    def test_increment_decode_cuda_graph_pass(self):
        """Test decode CUDA graph pass counter for both True and False."""
        b_cg = _counter_val(self.c.cuda_graph_passes_total, mode="decode_cuda_graph")
        b_no = _counter_val(self.c.cuda_graph_passes_total, mode="decode_none")
        self.c.increment_decode_cuda_graph_pass(True)
        self.c.increment_decode_cuda_graph_pass(False)
        self.assertEqual(
            _counter_val(self.c.cuda_graph_passes_total, mode="decode_cuda_graph"),
            b_cg + 1,
        )
        self.assertEqual(
            _counter_val(self.c.cuda_graph_passes_total, mode="decode_none"),
            b_no + 1,
        )

    def test_increment_prefill_cuda_graph_pass(self):
        """Test prefill CUDA graph pass counter for both True and False."""
        b_cg = _counter_val(
            self.c.cuda_graph_passes_total, mode="prefill_cuda_graph"
        )
        b_no = _counter_val(self.c.cuda_graph_passes_total, mode="prefill_none")
        self.c.increment_prefill_cuda_graph_pass(True)
        self.c.increment_prefill_cuda_graph_pass(False)
        self.assertEqual(
            _counter_val(
                self.c.cuda_graph_passes_total, mode="prefill_cuda_graph"
            ),
            b_cg + 1,
        )
        self.assertEqual(
            _counter_val(self.c.cuda_graph_passes_total, mode="prefill_none"),
            b_no + 1,
        )

    def test_increment_realtime_tokens_without_dp(self):
        """Test realtime token counters without DP cooperation."""
        b_pf = _counter_val(self.c.realtime_tokens_total, mode="prefill_compute")
        b_dc = _counter_val(self.c.realtime_tokens_total, mode="decode")
        self.c.increment_realtime_tokens(
            dp_cooperation_info=None,
            prefill_compute_tokens=10,
            decode_tokens=5,
        )
        self.assertEqual(
            _counter_val(self.c.realtime_tokens_total, mode="prefill_compute"),
            b_pf + 10,
        )
        self.assertEqual(
            _counter_val(self.c.realtime_tokens_total, mode="decode"), b_dc + 5
        )

    def test_increment_realtime_tokens_with_dp(self):
        """Test realtime token counters with DP cooperation info."""
        dp = DPCooperationInfo(num_prefill_ranks=1)
        b_pf = _counter_val(self.c.realtime_tokens_total, mode="prefill_compute")
        b_dp = _counter_val(
            self.c.dp_cooperation_realtime_tokens_total,
            mode="prefill_compute",
            num_prefill_ranks=1,
        )
        self.c.increment_realtime_tokens(
            dp_cooperation_info=dp,
            prefill_compute_tokens=10,
            decode_tokens=5,
        )
        self.assertEqual(
            _counter_val(self.c.realtime_tokens_total, mode="prefill_compute"),
            b_pf + 10,
        )
        self.assertEqual(
            _counter_val(
                self.c.dp_cooperation_realtime_tokens_total,
                mode="prefill_compute",
                num_prefill_ranks=1,
            ),
            b_dp + 10,
        )

    def test_increment_gpu_overlap_wait_seconds(self):
        """Test GPU overlap wait seconds counter."""
        before = _counter_val(
            self.c.gpu_overlap_wait_seconds_total, category="forward"
        )
        self.c.increment_gpu_overlap_wait_seconds("forward", 0.01, None)
        self.assertAlmostEqual(
            _counter_val(self.c.gpu_overlap_wait_seconds_total, category="forward"),
            before + 0.01,
        )

    def test_increment_gpu_execution_seconds_without_dp(self):
        """Test GPU execution seconds counter without DP cooperation."""
        before = _counter_val(
            self.c.gpu_execution_seconds_total, category="forward"
        )
        self.c.increment_gpu_execution_seconds("forward", 0.05, None)
        self.assertAlmostEqual(
            _counter_val(self.c.gpu_execution_seconds_total, category="forward"),
            before + 0.05,
        )

    def test_increment_gpu_execution_seconds_with_dp(self):
        """Test GPU execution seconds counter with DP cooperation."""
        dp = DPCooperationInfo(num_prefill_ranks=2)
        b_main = _counter_val(
            self.c.gpu_execution_seconds_total, category="forward"
        )
        b_dp = _counter_val(
            self.c.dp_cooperation_gpu_execution_seconds_total,
            category="forward",
            num_prefill_ranks=2,
        )
        self.c.increment_gpu_execution_seconds("forward", 0.05, dp)
        self.assertAlmostEqual(
            _counter_val(self.c.gpu_execution_seconds_total, category="forward"),
            b_main + 0.05,
        )
        self.assertAlmostEqual(
            _counter_val(
                self.c.dp_cooperation_gpu_execution_seconds_total,
                category="forward",
                num_prefill_ranks=2,
            ),
            b_dp + 0.05,
        )


class TestTokenizerMetricsCollector(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.c = _TOK

    def test_observe_one_finished_request(self):
        """Test observe_one_finished_request increments counters and histograms."""
        b_prompt = _counter_val(self.c.prompt_tokens_total)
        b_gen = _counter_val(self.c.generation_tokens_total)
        b_req = _counter_val(self.c.num_requests_total)
        b_e2e = _hist_sum(self.c.histogram_e2e_request_latency)
        self.c.observe_one_finished_request(
            labels=_LABELS,
            prompt_tokens=100,
            generation_tokens=50,
            cached_tokens=20,
            e2e_latency=1.5,
            has_grammar=False,
            retraction_count=0,
        )
        self.assertEqual(_counter_val(self.c.prompt_tokens_total), b_prompt + 100)
        self.assertEqual(_counter_val(self.c.generation_tokens_total), b_gen + 50)
        self.assertEqual(_counter_val(self.c.num_requests_total), b_req + 1)
        self.assertAlmostEqual(
            _hist_sum(self.c.histogram_e2e_request_latency) - b_e2e, 1.5
        )

    def test_observe_one_finished_request_with_grammar(self):
        """Test that has_grammar=True increments the structured output counter."""
        before = _counter_val(self.c.num_so_requests_total)
        self.c.observe_one_finished_request(
            labels=_LABELS,
            prompt_tokens=50,
            generation_tokens=10,
            cached_tokens=0,
            e2e_latency=0.5,
            has_grammar=True,
            retraction_count=1,
        )
        self.assertEqual(_counter_val(self.c.num_so_requests_total), before + 1)

    def test_observe_one_finished_request_with_cached_details(self):
        """Test cached token reporting with detailed source breakdown."""
        b_dev = _counter_val(self.c.cached_tokens_total, cache_source="device")
        b_host = _counter_val(self.c.cached_tokens_total, cache_source="host")
        self.c.observe_one_finished_request(
            labels=_LABELS,
            prompt_tokens=80,
            generation_tokens=20,
            cached_tokens=30,
            e2e_latency=1.0,
            has_grammar=False,
            retraction_count=0,
            cached_tokens_details={"device": 20, "host": 10},
        )
        self.assertEqual(
            _counter_val(self.c.cached_tokens_total, cache_source="device"),
            b_dev + 20,
        )
        self.assertEqual(
            _counter_val(self.c.cached_tokens_total, cache_source="host"),
            b_host + 10,
        )

    def test_observe_one_finished_request_with_storage_backend(self):
        """Test cached token reporting with storage backend breakdown."""
        b_dev = _counter_val(self.c.cached_tokens_total, cache_source="device")
        b_stor = _counter_val(self.c.cached_tokens_total, cache_source="storage_redis")
        self.c.observe_one_finished_request(
            labels=_LABELS,
            prompt_tokens=80,
            generation_tokens=20,
            cached_tokens=50,
            e2e_latency=1.0,
            has_grammar=False,
            retraction_count=0,
            cached_tokens_details={
                "device": 20,
                "host": 10,
                "storage": 20,
                "storage_backend": "redis",
            },
        )
        self.assertEqual(
            _counter_val(self.c.cached_tokens_total, cache_source="device"),
            b_dev + 20,
        )
        self.assertEqual(
            _counter_val(self.c.cached_tokens_total, cache_source="storage_redis"),
            b_stor + 20,
        )

    def test_observe_time_to_first_token(self):
        """Test observe_time_to_first_token records to the histogram."""
        before = _hist_sum(self.c.histogram_time_to_first_token)
        self.c.observe_time_to_first_token(_LABELS, 0.2)
        self.assertAlmostEqual(
            _hist_sum(self.c.histogram_time_to_first_token) - before, 0.2
        )

    def test_observe_inter_token_latency(self):
        """Test observe_inter_token_latency records interval to the histogram."""
        before = _hist_sum(self.c.histogram_inter_token_latency)
        self.c.observe_inter_token_latency(_LABELS, 0.03, num_new_tokens=3)
        self.assertAlmostEqual(
            _hist_sum(self.c.histogram_inter_token_latency) - before, 0.03
        )

    def test_observe_one_aborted_request(self):
        """Test that aborted request counter is incremented."""
        before = _counter_val(self.c.num_aborted_requests_total)
        self.c.observe_one_aborted_request(_LABELS)
        self.assertEqual(_counter_val(self.c.num_aborted_requests_total), before + 1)

    def test_check_straggler_under_threshold(self):
        """Test that straggler check returns False with < 100 observations."""
        self.assertFalse(self.c.check_time_to_first_token_straggler(10.0))


class TestStorageMetricsCollector(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.c = _STORAGE

    def test_log_prefetched_tokens_positive(self):
        """Test that prefetched token counter increments when > 0."""
        before = _counter_val(self.c.prefetched_tokens_total)
        self.c.log_prefetched_tokens(100)
        self.assertEqual(_counter_val(self.c.prefetched_tokens_total), before + 100)

    def test_log_prefetched_tokens_zero_is_noop(self):
        """Test that prefetched_tokens=0 does not increment the counter."""
        before = _counter_val(self.c.prefetched_tokens_total)
        self.c.log_prefetched_tokens(0)
        self.assertEqual(_counter_val(self.c.prefetched_tokens_total), before)

    def test_log_backuped_tokens_positive(self):
        """Test that backuped token counter increments when > 0."""
        before = _counter_val(self.c.backuped_tokens_total)
        self.c.log_backuped_tokens(50)
        self.assertEqual(_counter_val(self.c.backuped_tokens_total), before + 50)

    def test_log_backuped_tokens_zero_is_noop(self):
        """Test that backuped_tokens=0 does not increment the counter."""
        before = _counter_val(self.c.backuped_tokens_total)
        self.c.log_backuped_tokens(0)
        self.assertEqual(_counter_val(self.c.backuped_tokens_total), before)


class TestRadixCacheMetricsCollector(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.c = _RADIX

    def test_increment_eviction_num_tokens(self):
        """Test that eviction token counter is incremented."""
        before = _counter_val(self.c.eviction_num_tokens)
        self.c.increment_eviction_num_tokens(500)
        self.assertEqual(_counter_val(self.c.eviction_num_tokens), before + 500)

    def test_increment_load_back_num_tokens(self):
        """Test that load-back token counter is incremented."""
        before = _counter_val(self.c.load_back_num_tokens)
        self.c.increment_load_back_num_tokens(200)
        self.assertEqual(_counter_val(self.c.load_back_num_tokens), before + 200)

    def test_observe_eviction_duration(self):
        """Test that eviction duration is observed in the histogram."""
        before = _hist_sum(self.c.eviction_duration_seconds)
        self.c.observe_eviction_duration(0.005)
        self.assertAlmostEqual(
            _hist_sum(self.c.eviction_duration_seconds) - before, 0.005
        )

    def test_observe_load_back_duration(self):
        """Test that load-back duration is observed in the histogram."""
        before = _hist_sum(self.c.load_back_duration_seconds)
        self.c.observe_load_back_duration(0.01)
        self.assertAlmostEqual(
            _hist_sum(self.c.load_back_duration_seconds) - before, 0.01
        )


class TestExpertDispatchCollector(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.c = _EXPERT

    def test_collector_created(self):
        """Test that ExpertDispatchCollector was constructed with ep_size=8."""
        self.assertIsNotNone(self.c.eplb_gpu_physical_count)


class TestSchedulerMetricsCollectorLoraHicache(CustomTestCase):
    """Test SchedulerMetricsCollector with enable_lora and enable_hierarchical_cache."""

    @classmethod
    def setUpClass(cls):
        cls.c = _SCHED

    def test_lora_gauges_created(self):
        """Test that enable_lora=True creates LoRA gauge metrics."""
        self.assertTrue(hasattr(self.c, "lora_pool_slots_used"))
        self.assertTrue(hasattr(self.c, "lora_pool_slots_total"))
        self.assertTrue(hasattr(self.c, "lora_pool_utilization"))

    def test_hicache_gauges_created(self):
        """Test that enable_hierarchical_cache=True creates HiCache gauge metrics."""
        self.assertTrue(hasattr(self.c, "hicache_host_used_tokens"))
        self.assertTrue(hasattr(self.c, "hicache_host_total_tokens"))

    def test_log_stats_with_lora(self):
        """Test that log_stats records LoRA pool metrics when lora is enabled."""
        self.c.log_stats(
            SchedulerStats(
                num_running_reqs=QueueCount(total=1),
                num_queue_reqs=QueueCount(total=0),
                max_total_num_tokens=2000,
                lora_pool_slots_used=3,
                lora_pool_slots_total=8,
                lora_pool_utilization=0.375,
            )
        )
        self.assertAlmostEqual(_gauge_val(self.c.lora_pool_slots_used), 3)
        self.assertAlmostEqual(_gauge_val(self.c.lora_pool_slots_total), 8)
        self.assertAlmostEqual(_gauge_val(self.c.lora_pool_utilization), 0.375)

    def test_log_stats_with_hicache(self):
        """Test that log_stats records HiCache metrics when hierarchical cache is enabled."""
        self.c.log_stats(
            SchedulerStats(
                num_running_reqs=QueueCount(total=1),
                num_queue_reqs=QueueCount(total=0),
                max_total_num_tokens=2000,
                hicache_host_used_tokens=500,
                hicache_host_total_tokens=1000,
            )
        )
        self.assertAlmostEqual(_gauge_val(self.c.hicache_host_used_tokens), 500)
        self.assertAlmostEqual(_gauge_val(self.c.hicache_host_total_tokens), 1000)


class TestLogGrammarStats(CustomTestCase):
    """Test log_grammar_stats with real grammar stats objects."""

    @classmethod
    def setUpClass(cls):
        cls.c = _SCHED

    def test_log_grammar_stats_full(self):
        """Test log_grammar_stats with all fields set."""
        grammar_stats = SimpleNamespace(
            compilation_time=0.05,
            schema_count=3,
            ebnf_size=100,
            tree_traversal_time=[0.01, 0.02, 0.03],
            is_cache_hit=True,
            is_grammar_aborted=False,
            num_timeout=0,
        )
        self.c.log_grammar_stats(grammar_stats)

    def test_log_grammar_stats_aborted_and_timeout(self):
        """Test log_grammar_stats with abort and timeout flags."""
        grammar_stats = SimpleNamespace(
            compilation_time=None,
            schema_count=None,
            ebnf_size=None,
            tree_traversal_time=[],
            is_cache_hit=False,
            is_grammar_aborted=True,
            num_timeout=2,
        )
        self.c.log_grammar_stats(grammar_stats)

    def test_emit_cache_config_info(self):
        """Test that emit_cache_config_info sets the info gauge."""
        self.c.emit_cache_config_info(page_size=16, num_pages=1000)


class TestTokenizerCollectorWithHistogram(CustomTestCase):
    """Test TokenizerMetricsCollector with collect_tokens_histogram=True."""

    @classmethod
    def setUpClass(cls):
        cls.c = _TOK

    def test_histograms_created(self):
        """Test that token histograms are created when collect_tokens_histogram=True."""
        self.assertTrue(hasattr(self.c, "prompt_tokens_histogram"))
        self.assertTrue(hasattr(self.c, "generation_tokens_histogram"))

    def test_observe_records_to_histograms(self):
        """Test that observe_one_finished_request records to the token histograms."""
        b_prompt = _hist_sum(self.c.prompt_tokens_histogram)
        b_gen = _hist_sum(self.c.generation_tokens_histogram)
        self.c.observe_one_finished_request(
            labels=_LABELS, prompt_tokens=200, generation_tokens=80,
            cached_tokens=0, e2e_latency=1.0, has_grammar=False, retraction_count=0,
        )
        self.assertAlmostEqual(
            _hist_sum(self.c.prompt_tokens_histogram) - b_prompt, 200.0
        )
        self.assertAlmostEqual(
            _hist_sum(self.c.generation_tokens_histogram) - b_gen, 80.0
        )


class TestStragglerDetection(CustomTestCase):
    """Test check_time_to_first_token_straggler with >= 100 observations."""

    @classmethod
    def setUpClass(cls):
        cls.c = _TOK

    def test_straggler_detected(self):
        """Test that a value at the tail is detected as straggler after 100+ obs."""
        for _ in range(150):
            self.c.observe_time_to_first_token(_LABELS, 0.1)
        # 0.1s is the "normal" value; a 100s value should be a straggler
        self.assertTrue(self.c.check_time_to_first_token_straggler(100.0))

    def test_non_straggler(self):
        """Test that a value within the normal range is not flagged."""
        self.assertFalse(self.c.check_time_to_first_token_straggler(0.1))


class TestStorageLogStorageMetrics(CustomTestCase):
    """Test StorageMetricsCollector.log_storage_metrics with real StorageMetrics."""

    @classmethod
    def setUpClass(cls):
        cls.c = _STORAGE

    def test_log_storage_metrics_none_is_noop(self):
        """Test that log_storage_metrics(None) returns immediately."""
        self.c.log_storage_metrics(None)

    def test_log_storage_metrics_with_data(self):
        """Test log_storage_metrics records all histogram values."""
        from sglang.srt.observability.metrics_collector import StorageMetrics

        sm = StorageMetrics(
            prefetch_pgs=[1, 2, 3],
            backup_pgs=[4, 5],
            prefetch_bandwidth=[1.0, 2.0],
            backup_bandwidth=[0.5],
        )
        b_pf = _hist_sum(self.c.histogram_prefetch_pgs)
        b_bp = _hist_sum(self.c.histogram_backup_pgs)
        self.c.log_storage_metrics(sm)
        self.assertAlmostEqual(_hist_sum(self.c.histogram_prefetch_pgs) - b_pf, 6.0)
        self.assertAlmostEqual(_hist_sum(self.c.histogram_backup_pgs) - b_bp, 9.0)


if __name__ == "__main__":
    unittest.main()
