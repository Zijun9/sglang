"""Unit tests for srt/observability/label_transform.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")

import unittest

from sglang.srt.observability.label_transform import (
    UNKNOWN_PRIORITY_VALUE,
    _HIGH_PRIORITY_VALUE,
    _LOW_PRIORITY_VALUE,
    _PRIORITY_MAX,
    _PRIORITY_MIN,
    transform_priority,
)
from sglang.test.test_utils import CustomTestCase


class TestTransformPriority(CustomTestCase):

    def test_none_returns_unknown(self):
        """Test that None priority maps to UNKNOWN."""
        self.assertEqual(transform_priority(None), UNKNOWN_PRIORITY_VALUE)

    def test_all_in_range_values_return_string_repr(self):
        """Test that every value in [0, 31) maps to its string representation."""
        for p in range(_PRIORITY_MIN, _PRIORITY_MAX):
            self.assertEqual(transform_priority(p), str(p))

    def test_boundary_max_uses_ge_not_gt(self):
        """Test that priority == 31 maps to HIGH (>= check), while 30 is still in range."""
        self.assertEqual(transform_priority(_PRIORITY_MAX), _HIGH_PRIORITY_VALUE)
        self.assertEqual(transform_priority(_PRIORITY_MAX - 1), str(_PRIORITY_MAX - 1))

    def test_negative_returns_low(self):
        """Test that negative priorities are clamped to LOW."""
        self.assertEqual(transform_priority(-1), _LOW_PRIORITY_VALUE)
        self.assertEqual(transform_priority(-100), _LOW_PRIORITY_VALUE)

    def test_large_positive_returns_high(self):
        """Test that priorities far above max are clamped to HIGH."""
        self.assertEqual(transform_priority(100), _HIGH_PRIORITY_VALUE)
        self.assertEqual(transform_priority(10000), _HIGH_PRIORITY_VALUE)

    def test_output_cardinality_bounded(self):
        """Test that total distinct outputs <= 35 (32 in-range + LOW + HIGH + UNKNOWN)."""
        outputs = {transform_priority(p) for p in range(-50, 200)}
        outputs.add(transform_priority(None))
        self.assertLessEqual(len(outputs), 35)
        self.assertIn(_LOW_PRIORITY_VALUE, outputs)
        self.assertIn(_HIGH_PRIORITY_VALUE, outputs)
        self.assertIn(UNKNOWN_PRIORITY_VALUE, outputs)


if __name__ == "__main__":
    unittest.main()
