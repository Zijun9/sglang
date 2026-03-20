"""Unit tests for srt/observability/func_timer.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-cpu-only")

import asyncio
import time
import unittest

from prometheus_client import CollectorRegistry, Histogram

from sglang.srt.observability import func_timer
from sglang.srt.observability.func_timer import time_func_latency
from sglang.test.test_utils import CustomTestCase


class TestTimeFuncLatencyDisabled(CustomTestCase):

    def setUp(self):
        self._orig = func_timer.enable_metrics
        func_timer.enable_metrics = False

    def tearDown(self):
        func_timer.enable_metrics = self._orig

    def test_sync_return_value_preserved(self):
        """Test that sync decorated function returns the correct value."""

        @time_func_latency
        def add(a, b):
            return a + b

        self.assertEqual(add(3, 4), 7)

    def test_async_return_value_preserved(self):
        """Test that async decorated function returns the correct value."""

        @time_func_latency
        async def double(x):
            return x * 2

        self.assertEqual(asyncio.run(double(5)), 10)

    def test_sync_exception_propagates(self):
        """Test that sync function exception is not swallowed by the decorator."""

        @time_func_latency
        def fail():
            raise ValueError("sync error")

        with self.assertRaises(ValueError):
            fail()

    def test_async_exception_propagates(self):
        """Test that async function exception is not swallowed by the decorator."""

        @time_func_latency
        async def fail():
            raise RuntimeError("async error")

        with self.assertRaises(RuntimeError):
            asyncio.run(fail())

    def test_decorator_with_name_kwarg(self):
        """Test that @time_func_latency(name=...) syntax works."""

        @time_func_latency(name="custom")
        def work():
            return 99

        self.assertEqual(work(), 99)

    def test_wraps_preserves_sync_metadata(self):
        """Test that @wraps keeps __name__ and __doc__ of the original function."""

        @time_func_latency
        def my_func():
            """my docstring"""
            pass

        self.assertEqual(my_func.__name__, "my_func")
        self.assertEqual(my_func.__doc__, "my docstring")

    def test_sync_function_uses_sync_wrapper(self):
        """Test that a regular function gets sync_wrapper, not async_wrapper."""

        @time_func_latency
        def regular():
            return "sync_result"

        self.assertEqual(regular(), "sync_result")

    def test_async_function_detected_as_coroutine(self):
        """Test that an async function stays a coroutine function after decoration."""

        @time_func_latency
        async def coro():
            return "async"

        self.assertTrue(asyncio.iscoroutinefunction(coro))


class TestTimeFuncLatencyEnabled(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.registry = CollectorRegistry()
        cls.histogram = Histogram(
            "test_func_latency_seconds",
            "Test function latency",
            labelnames=["name"],
            registry=cls.registry,
        )

    def setUp(self):
        self._orig_enabled = func_timer.enable_metrics
        self._orig_latency = func_timer.FUNC_LATENCY
        func_timer.enable_metrics = True
        func_timer.FUNC_LATENCY = self.__class__.histogram

    def tearDown(self):
        func_timer.enable_metrics = self._orig_enabled
        func_timer.FUNC_LATENCY = self._orig_latency

    def _get_count(self, name):
        """Helper: read observation count from histogram for a given label."""
        return (
            self.registry.get_sample_value(
                "test_func_latency_seconds_count", {"name": name}
            )
            or 0.0
        )

    def _get_sum(self, name):
        """Helper: read cumulative observed time from histogram for a given label."""
        return (
            self.registry.get_sample_value(
                "test_func_latency_seconds_sum", {"name": name}
            )
            or 0.0
        )

    def test_sync_records_to_histogram(self):
        """Test that sync function execution is recorded in the real histogram."""

        @time_func_latency
        def timed_sync():
            time.sleep(0.01)
            return "ok"

        before = self._get_count("timed_sync")
        self.assertEqual(timed_sync(), "ok")
        self.assertEqual(self._get_count("timed_sync"), before + 1)
        self.assertGreater(self._get_sum("timed_sync"), 0)

    def test_async_records_to_histogram(self):
        """Test that async function execution is recorded in the real histogram."""

        @time_func_latency
        async def timed_async():
            await asyncio.sleep(0.01)
            return "async_ok"

        before = self._get_count("timed_async")
        self.assertEqual(asyncio.run(timed_async()), "async_ok")
        self.assertEqual(self._get_count("timed_async"), before + 1)

    def test_custom_name_used_as_label(self):
        """Test that name= kwarg overrides __name__ in the histogram label."""

        @time_func_latency(name="custom_label")
        def anything():
            pass

        before = self._get_count("custom_label")
        anything()
        self.assertEqual(self._get_count("custom_label"), before + 1)

    def test_sync_exception_still_records_via_finally(self):
        """Test that try/finally ensures metric is observed even on exception."""

        @time_func_latency
        def sync_fail():
            raise ValueError("boom")

        before = self._get_count("sync_fail")
        with self.assertRaises(ValueError):
            sync_fail()
        self.assertEqual(self._get_count("sync_fail"), before + 1)

    def test_async_exception_still_records_via_finally(self):
        """Test that async try/finally ensures metric is observed even on exception."""

        @time_func_latency
        async def async_fail():
            raise RuntimeError("async boom")

        before = self._get_count("async_fail")
        with self.assertRaises(RuntimeError):
            asyncio.run(async_fail())
        self.assertEqual(self._get_count("async_fail"), before + 1)

    def test_timing_reflects_actual_duration(self):
        """Test that observed duration is in a plausible range for a 20ms sleep."""

        @time_func_latency
        def sleep_20ms():
            time.sleep(0.02)

        sum_before = self._get_sum("sleep_20ms")
        sleep_20ms()
        elapsed = self._get_sum("sleep_20ms") - sum_before
        self.assertGreater(elapsed, 0.015)
        self.assertLess(elapsed, 0.5)


class TestEnableFuncTimer(CustomTestCase):

    def test_enable_func_timer_sets_globals(self):
        """Test that enable_func_timer creates a real Histogram and flips the flag."""
        from sglang.srt.observability.func_timer import enable_func_timer

        if not func_timer.enable_metrics:
            enable_func_timer()
        self.assertTrue(func_timer.enable_metrics)
        self.assertIsNotNone(func_timer.FUNC_LATENCY)


if __name__ == "__main__":
    unittest.main()
