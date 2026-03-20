"""Unit tests for srt/observability/startup_func_log_and_timer.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import time
import unittest

from prometheus_client import CollectorRegistry, Gauge

from sglang.srt.observability import startup_func_log_and_timer as st
from sglang.srt.observability.startup_func_log_and_timer import (
    get_max_duration,
    reset_startup_timers,
    set_startup_metric,
    startup_timer,
    time_startup_latency,
)
from sglang.test.test_utils import CustomTestCase


class TestStartupTimerMaxTracking(CustomTestCase):

    def setUp(self):
        reset_startup_timers()
        self._orig_enabled = st.enable_startup_metrics
        st.enable_startup_metrics = False

    def tearDown(self):
        st.enable_startup_metrics = self._orig_enabled
        reset_startup_timers()

    def test_context_manager_records_positive_duration(self):
        """Test that startup_timer records a measurable positive duration."""
        with startup_timer("block_a"):
            time.sleep(0.02)
        self.assertGreater(get_max_duration("block_a"), 0.01)

    def test_larger_duration_overwrites(self):
        """Test that a longer execution replaces the stored maximum."""
        with startup_timer("max_test"):
            time.sleep(0.01)
        first = get_max_duration("max_test")

        with startup_timer("max_test"):
            time.sleep(0.04)
        self.assertGreater(get_max_duration("max_test"), first)

    def test_smaller_duration_does_not_overwrite(self):
        """Test that a shorter execution does not replace a larger stored maximum."""
        with startup_timer("keep_max"):
            time.sleep(0.04)
        big = get_max_duration("keep_max")

        with startup_timer("keep_max"):
            pass
        self.assertAlmostEqual(get_max_duration("keep_max"), big, places=2)

    def test_exception_still_records_in_finally(self):
        """Test that duration is captured even when the block raises."""
        with self.assertRaises(ValueError):
            with startup_timer("err_block"):
                time.sleep(0.01)
                raise ValueError("fail")
        self.assertGreater(get_max_duration("err_block"), 0)

    def test_separate_contexts_are_independent(self):
        """Test that different context names track independently."""
        with startup_timer("ctx_x"):
            time.sleep(0.01)
        with startup_timer("ctx_y"):
            time.sleep(0.03)
        self.assertGreater(get_max_duration("ctx_y"), get_max_duration("ctx_x"))


class TestSetStartupMetric(CustomTestCase):

    def setUp(self):
        reset_startup_timers()
        self._orig_enabled = st.enable_startup_metrics
        st.enable_startup_metrics = False

    def tearDown(self):
        st.enable_startup_metrics = self._orig_enabled
        reset_startup_timers()

    def test_noop_when_metrics_disabled(self):
        """Test that set_startup_metric does nothing when enable_startup_metrics is False."""
        set_startup_metric("m", 1.0, should_log=False)
        self.assertIsNone(get_max_duration("m"))


class TestResetStartupTimers(CustomTestCase):

    def test_clears_all_durations(self):
        """Test that reset_startup_timers empties _max_durations."""
        st._max_durations["a"] = 1.0
        st._max_durations["b"] = 2.0
        reset_startup_timers()
        self.assertIsNone(get_max_duration("a"))
        self.assertIsNone(get_max_duration("b"))
        self.assertEqual(len(st._max_durations), 0)


class TestTimeStartupLatencyDecorator(CustomTestCase):

    def setUp(self):
        reset_startup_timers()
        self._orig_enabled = st.enable_startup_metrics
        st.enable_startup_metrics = False

    def tearDown(self):
        st.enable_startup_metrics = self._orig_enabled
        reset_startup_timers()

    def test_decorator_records_duration(self):
        """Test that @time_startup_latency records a positive duration."""

        @time_startup_latency
        def init_stuff():
            time.sleep(0.01)
            return "done"

        self.assertEqual(init_stuff(), "done")
        self.assertGreater(get_max_duration("init_stuff"), 0)

    def test_custom_name_overrides_function_name(self):
        """Test that name= kwarg is used instead of __name__."""

        @time_startup_latency(name="custom_init")
        def whatever():
            return 42

        self.assertEqual(whatever(), 42)
        self.assertIsNotNone(get_max_duration("custom_init"))
        self.assertIsNone(get_max_duration("whatever"))

    def test_decorator_tracks_max_across_calls(self):
        """Test that repeated calls store only the maximum duration."""

        @time_startup_latency(name="repeat")
        def work(dur):
            time.sleep(dur)

        work(0.01)
        first = get_max_duration("repeat")
        work(0.03)
        second = get_max_duration("repeat")
        self.assertGreater(second, first)

        work(0.001)
        self.assertAlmostEqual(get_max_duration("repeat"), second, places=2)

    def test_preserves_function_name(self):
        """Test that @wraps keeps __name__ of the original function."""

        @time_startup_latency
        def original():
            pass

        self.assertEqual(original.__name__, "original")

    def test_exception_still_records(self):
        """Test that duration is captured even when the function raises."""

        @time_startup_latency
        def fail_init():
            raise RuntimeError("init failed")

        with self.assertRaises(RuntimeError):
            fail_init()
        self.assertIsNotNone(get_max_duration("fail_init"))


class TestStartupTimerPrometheus(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.registry = CollectorRegistry()
        cls.gauge = Gauge(
            "test_startup_latency_max",
            "Test startup latency",
            labelnames=["context"],
            registry=cls.registry,
        )

    def setUp(self):
        reset_startup_timers()
        self._orig_enabled = st.enable_startup_metrics
        self._orig_gauge = st.STARTUP_LATENCY_SECONDS
        st.enable_startup_metrics = True
        st.STARTUP_LATENCY_SECONDS = self.__class__.gauge

    def tearDown(self):
        st.enable_startup_metrics = self._orig_enabled
        st.STARTUP_LATENCY_SECONDS = self._orig_gauge
        reset_startup_timers()

    def _get_gauge(self, context):
        """Helper: read gauge value from the isolated registry."""
        return self.registry.get_sample_value(
            "test_startup_latency_max", {"context": context}
        )

    def test_context_manager_updates_gauge(self):
        """Test that startup_timer sets the real Prometheus gauge."""
        with startup_timer("gauge_test"):
            time.sleep(0.01)
        self.assertIsNotNone(self._get_gauge("gauge_test"))
        self.assertGreater(self._get_gauge("gauge_test"), 0)

    def test_log_only_tracks_max_but_skips_gauge(self):
        """Test that log_only=True updates _max_durations but not the gauge."""

        @time_startup_latency(name="log_only_ctx", log_only=True)
        def func():
            time.sleep(0.01)

        func()
        self.assertIsNotNone(get_max_duration("log_only_ctx"))
        self.assertIsNone(self._get_gauge("log_only_ctx"))

    def test_gauge_only_updated_on_new_max(self):
        """Test that a smaller duration does not re-set the gauge."""
        with startup_timer("max_gauge"):
            time.sleep(0.03)
        first_gauge = self._get_gauge("max_gauge")

        with startup_timer("max_gauge"):
            pass
        self.assertAlmostEqual(self._get_gauge("max_gauge"), first_gauge, places=3)

    def test_set_startup_metric_only_records_new_max(self):
        """Test that set_startup_metric only updates on new maximum, not downgrade."""
        set_startup_metric("m", 1.0, should_log=False)
        self.assertEqual(get_max_duration("m"), 1.0)

        set_startup_metric("m", 0.5, should_log=False)
        self.assertEqual(get_max_duration("m"), 1.0)

        set_startup_metric("m", 2.0, should_log=False)
        self.assertEqual(get_max_duration("m"), 2.0)

    def test_set_startup_metric_with_logging(self):
        """Test that should_log=True (default) still records correctly."""
        set_startup_metric("logged", 1.5)
        self.assertIsNotNone(self._get_gauge("logged"))

    def test_decorator_with_prometheus_enabled(self):
        """Test that the decorator updates the real gauge when metrics are on."""

        @time_startup_latency(name="prom_deco")
        def init():
            time.sleep(0.01)

        init()
        self.assertIsNotNone(self._get_gauge("prom_deco"))
        self.assertGreater(self._get_gauge("prom_deco"), 0)


class TestEnableStartupTimer(CustomTestCase):

    def test_enable_startup_timer_sets_globals(self):
        """Test that enable_startup_timer creates a real Gauge and flips the flag."""
        from sglang.srt.observability.startup_func_log_and_timer import (
            enable_startup_timer,
        )

        if not st.enable_startup_metrics:
            enable_startup_timer()
        self.assertTrue(st.enable_startup_metrics)
        self.assertIsNotNone(st.STARTUP_LATENCY_SECONDS)


if __name__ == "__main__":
    unittest.main()
