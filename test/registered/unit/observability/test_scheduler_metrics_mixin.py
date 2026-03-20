"""Unit tests for srt/observability/scheduler_metrics_mixin.py — no server, no model loading.

The circular import between scheduler.py and scheduler_metrics_mixin.py is resolved
by importing sglang.srt.managers.scheduler first, which forces both modules to load
in the correct order.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import unittest
from types import SimpleNamespace

# Resolve the circular import by importing scheduler first
import sglang.srt.managers.scheduler  # noqa: F401
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.observability.metrics_collector import QueueCount, SchedulerStats
from sglang.srt.observability.scheduler_metrics_mixin import (
    KvMetrics,
    PrefillStats,
    SchedulerMetricsMixin,
)
from sglang.test.test_utils import CustomTestCase


class TestPrefillStats(CustomTestCase):

    def test_construction(self):
        """Test that PrefillStats stores all constructor arguments."""
        ps = PrefillStats(
            log_input_tokens=100,
            log_hit_tokens=80,
            new_token_ratio=0.2,
            num_running_reqs=QueueCount(total=5),
            num_new_seqs=3,
        )
        self.assertEqual(ps.log_input_tokens, 100)
        self.assertEqual(ps.log_hit_tokens, 80)
        self.assertAlmostEqual(ps.new_token_ratio, 0.2)
        self.assertEqual(ps.num_running_reqs.total, 5)
        self.assertEqual(ps.num_new_seqs, 3)


class TestKvMetrics(CustomTestCase):

    def test_default_construction(self):
        """Test that KvMetrics defaults all fields to None."""
        kv = KvMetrics()
        self.assertIsNone(kv.request_active_slots)
        self.assertIsNone(kv.kv_active_blocks)
        self.assertIsNone(kv.gpu_cache_usage_perc)
        self.assertIsNone(kv.data_parallel_rank)

    def test_explicit_values(self):
        """Test that fields can be set after construction."""
        kv = KvMetrics()
        kv.request_active_slots = 10
        kv.kv_total_blocks = 1000
        kv.gpu_cache_usage_perc = 0.75
        self.assertEqual(kv.request_active_slots, 10)
        self.assertEqual(kv.kv_total_blocks, 1000)
        self.assertAlmostEqual(kv.gpu_cache_usage_perc, 0.75)


class TestCalculateUtilization(CustomTestCase):
    """Test the calculate_utilization mixin method with real SchedulerStats."""

    def _make_fake_scheduler(self, mode=DisaggregationMode.NULL, **stat_overrides):
        """Helper: create a minimal namespace with just the fields calculate_utilization needs."""
        stats = SchedulerStats()
        for k, v in stat_overrides.items():
            setattr(stats, k, v)
        obj = SimpleNamespace(disaggregation_mode=mode, stats=stats)
        return obj

    def test_prefill_mode_returns_minus_one(self):
        """Test that PREFILL mode sets utilization to -1."""
        sched = self._make_fake_scheduler(mode=DisaggregationMode.PREFILL)
        SchedulerMetricsMixin.calculate_utilization(sched)
        self.assertEqual(sched.stats.utilization, -1)

    def test_unified_mode_no_slo(self):
        """Test that without max_running_requests_under_SLO, utilization stays 0."""
        sched = self._make_fake_scheduler(
            mode=DisaggregationMode.NULL,
            max_running_requests_under_SLO=None,
            token_usage=0.5,
        )
        SchedulerMetricsMixin.calculate_utilization(sched)
        self.assertAlmostEqual(sched.stats.utilization, 0.0)

    def test_unified_mode_with_slo_req_dominant(self):
        """Test utilization when running reqs / SLO is the dominant factor."""
        sched = self._make_fake_scheduler(
            mode=DisaggregationMode.NULL,
            num_running_reqs=QueueCount(total=40),
            max_running_requests_under_SLO=50,
            token_usage=0.1,  # 0.1 / 0.9 ≈ 0.111
        )
        SchedulerMetricsMixin.calculate_utilization(sched)
        # max(40/50=0.8, 0.1/0.9=0.111) = 0.8
        self.assertAlmostEqual(sched.stats.utilization, 0.8)

    def test_unified_mode_with_slo_token_dominant(self):
        """Test utilization when token_usage / 0.9 is the dominant factor."""
        sched = self._make_fake_scheduler(
            mode=DisaggregationMode.NULL,
            num_running_reqs=QueueCount(total=5),
            max_running_requests_under_SLO=100,
            token_usage=0.85,  # 0.85 / 0.9 ≈ 0.944
        )
        SchedulerMetricsMixin.calculate_utilization(sched)
        # max(5/100=0.05, 0.85/0.9≈0.944) = 0.944
        self.assertAlmostEqual(sched.stats.utilization, 0.85 / 0.9, places=3)

    def test_decode_mode_with_slo(self):
        """Test that DECODE mode also computes utilization (same as unified)."""
        sched = self._make_fake_scheduler(
            mode=DisaggregationMode.DECODE,
            num_running_reqs=QueueCount(total=20),
            max_running_requests_under_SLO=40,
            token_usage=0.3,
        )
        SchedulerMetricsMixin.calculate_utilization(sched)
        # max(20/40=0.5, 0.3/0.9≈0.333) = 0.5
        self.assertAlmostEqual(sched.stats.utilization, 0.5)

    def test_slo_zero_skips_calculation(self):
        """Test that SLO=0 does not trigger division by zero."""
        sched = self._make_fake_scheduler(
            mode=DisaggregationMode.NULL,
            max_running_requests_under_SLO=0,
            token_usage=0.5,
        )
        SchedulerMetricsMixin.calculate_utilization(sched)
        self.assertAlmostEqual(sched.stats.utilization, 0.0)


class TestUpdateSpecMetrics(CustomTestCase):
    """Test the update_spec_metrics mixin method."""

    def _make(self):
        return SimpleNamespace(
            spec_num_accepted_tokens=0,
            spec_num_forward_ct=0,
            num_generated_tokens=0,
        )

    def test_accumulates_accepted_tokens(self):
        """Test that accepted tokens are accumulated as num_accepted_tokens + bs."""
        sched = self._make()
        SchedulerMetricsMixin.update_spec_metrics(sched, bs=4, num_accepted_tokens=10)
        # accepted_tokens += 10 + 4 = 14
        self.assertEqual(sched.spec_num_accepted_tokens, 14)
        # forward_ct += bs = 4
        self.assertEqual(sched.spec_num_forward_ct, 4)
        # generated_tokens += num_accepted_tokens only = 10
        self.assertEqual(sched.num_generated_tokens, 10)

    def test_multiple_updates_accumulate(self):
        """Test that multiple calls accumulate correctly."""
        sched = self._make()
        SchedulerMetricsMixin.update_spec_metrics(sched, bs=2, num_accepted_tokens=5)
        SchedulerMetricsMixin.update_spec_metrics(sched, bs=3, num_accepted_tokens=8)
        self.assertEqual(sched.spec_num_accepted_tokens, (5 + 2) + (8 + 3))
        self.assertEqual(sched.spec_num_forward_ct, 2 + 3)
        self.assertEqual(sched.num_generated_tokens, 5 + 8)


class TestResetMetrics(CustomTestCase):
    """Test the reset_metrics mixin method."""

    def test_resets_all_counters_to_zero(self):
        """Test that reset_metrics zeros all tracking counters."""
        sched = SimpleNamespace(
            forward_ct_decode=100,
            num_generated_tokens=5000,
            spec_num_accepted_tokens=200,
            spec_num_forward_ct=50,
            spec_total_num_accepted_tokens=10000,
            spec_total_num_forward_ct=2000,
        )
        SchedulerMetricsMixin.reset_metrics(sched)
        self.assertEqual(sched.forward_ct_decode, 0)
        self.assertEqual(sched.num_generated_tokens, 0)
        self.assertEqual(sched.spec_num_accepted_tokens, 0)
        self.assertEqual(sched.spec_num_forward_ct, 0)
        self.assertEqual(sched.spec_total_num_accepted_tokens, 0)
        self.assertEqual(sched.spec_total_num_forward_ct, 0)


if __name__ == "__main__":
    unittest.main()
