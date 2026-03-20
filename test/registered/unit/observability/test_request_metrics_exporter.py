"""Unit tests for srt/observability/request_metrics_exporter.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-cpu-only")

import asyncio
import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

from sglang.srt.observability.request_metrics_exporter import (
    ALWAYS_EXCLUDE_FIELDS,
    FileRequestMetricsExporter,
    RequestMetricsExporterManager,
    create_request_metrics_exporters,
)
from sglang.test.test_utils import CustomTestCase


@dataclass
class _TestReqInput:
    """Lightweight dataclass to exercise _format_output_data without heavy deps."""

    rid: str = "test-123"
    text: str = "hello world"
    temperature: float = 0.7
    max_tokens: int = 100
    image_data: Optional[str] = "base64img"
    video_data: Optional[str] = "mp4data"
    audio_data: Optional[str] = "wavdata"
    input_embeds: Optional[list] = None
    optional_none: Optional[str] = None


def _make_server_args(tmpdir, export=True):
    """Helper: create a minimal server-args-like namespace."""
    return SimpleNamespace(
        export_metrics_to_file_dir=tmpdir,
        export_metrics_to_file=export,
    )


class TestAlwaysExcludeFields(CustomTestCase):

    def test_contains_non_serializable_fields(self):
        """Test that the exclusion set covers all media/tensor fields."""
        self.assertEqual(
            ALWAYS_EXCLUDE_FIELDS, {"image_data", "video_data", "audio_data", "input_embeds"}
        )


class TestFormatOutputData(CustomTestCase):

    def _make_exporter(self, obj_skip=None, out_skip=None):
        """Helper: create a FileRequestMetricsExporter backed by a temp dir."""
        self._tmpdir = tempfile.mkdtemp()
        return FileRequestMetricsExporter(
            _make_server_args(self._tmpdir), obj_skip, out_skip
        )

    def test_excludes_always_exclude_fields(self):
        """Test that image_data, video_data, audio_data, input_embeds are filtered."""
        exporter = self._make_exporter()
        result = exporter._format_output_data(_TestReqInput(), {"meta_info": {}})
        params = json.loads(result["request_parameters"])
        for excluded in ALWAYS_EXCLUDE_FIELDS:
            self.assertNotIn(excluded, params)

    def test_includes_normal_fields(self):
        """Test that non-excluded, non-None fields appear in request_parameters."""
        exporter = self._make_exporter()
        obj = _TestReqInput(text="test prompt", temperature=0.9)
        result = exporter._format_output_data(obj, {"meta_info": {}})
        params = json.loads(result["request_parameters"])
        self.assertEqual(params["text"], "test prompt")
        self.assertEqual(params["temperature"], 0.9)
        self.assertEqual(params["rid"], "test-123")

    def test_excludes_none_values(self):
        """Test that fields with value None are not included."""
        exporter = self._make_exporter()
        result = exporter._format_output_data(
            _TestReqInput(optional_none=None), {"meta_info": {}}
        )
        params = json.loads(result["request_parameters"])
        self.assertNotIn("optional_none", params)

    def test_obj_skip_names_excluded(self):
        """Test that fields listed in obj_skip_names are filtered out."""
        exporter = self._make_exporter(obj_skip={"temperature", "max_tokens"})
        result = exporter._format_output_data(_TestReqInput(), {"meta_info": {}})
        params = json.loads(result["request_parameters"])
        self.assertNotIn("temperature", params)
        self.assertNotIn("max_tokens", params)
        self.assertIn("text", params)

    def test_out_skip_names_filtered_from_meta(self):
        """Test that keys in out_skip_names are removed from meta_info output."""
        exporter = self._make_exporter(out_skip={"secret_key"})
        out_dict = {"meta_info": {"public_metric": 42, "secret_key": "hidden"}}
        result = exporter._format_output_data(_TestReqInput(), out_dict)
        self.assertEqual(result["public_metric"], 42)
        self.assertNotIn("secret_key", result)

    def test_request_parameters_is_json_string(self):
        """Test that request_parameters value is a JSON-encoded string, not a dict."""
        exporter = self._make_exporter()
        result = exporter._format_output_data(_TestReqInput(), {"meta_info": {}})
        self.assertIsInstance(result["request_parameters"], str)
        self.assertIsInstance(json.loads(result["request_parameters"]), dict)


class TestFileRequestMetricsExporter(CustomTestCase):

    def test_creates_export_dir(self):
        """Test that the constructor creates the export directory if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "metrics_out")
            FileRequestMetricsExporter(_make_server_args(subdir), None, None)
            self.assertTrue(os.path.isdir(subdir))

    def test_ensure_file_handler_creates_file(self):
        """Test that _ensure_file_handler opens a log file for the given hour."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = FileRequestMetricsExporter(
                _make_server_args(tmpdir), None, None
            )
            exporter._ensure_file_handler("20260319_10")
            self.assertEqual(exporter._current_hour_suffix, "20260319_10")
            self.assertIsNotNone(exporter._current_file_handler)
            expected = os.path.join(tmpdir, "sglang-request-metrics-20260319_10.log")
            self.assertTrue(os.path.exists(expected))
            exporter.close()

    def test_file_rotation_on_hour_change(self):
        """Test that switching to a new hour opens a new file and closes the old one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = FileRequestMetricsExporter(
                _make_server_args(tmpdir), None, None
            )
            exporter._ensure_file_handler("20260319_10")
            exporter._ensure_file_handler("20260319_11")
            self.assertEqual(exporter._current_hour_suffix, "20260319_11")
            files = set(os.listdir(tmpdir))
            self.assertIn("sglang-request-metrics-20260319_10.log", files)
            self.assertIn("sglang-request-metrics-20260319_11.log", files)
            exporter.close()

    def test_same_hour_does_not_reopen(self):
        """Test that calling _ensure_file_handler with same hour reuses the handle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = FileRequestMetricsExporter(
                _make_server_args(tmpdir), None, None
            )
            exporter._ensure_file_handler("20260319_10")
            handler1 = exporter._current_file_handler
            exporter._ensure_file_handler("20260319_10")
            self.assertIs(exporter._current_file_handler, handler1)
            exporter.close()

    def test_close_clears_state(self):
        """Test that close() nullifies the handler and hour suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = FileRequestMetricsExporter(
                _make_server_args(tmpdir), None, None
            )
            exporter._ensure_file_handler("20260319_10")
            exporter.close()
            self.assertIsNone(exporter._current_file_handler)
            self.assertIsNone(exporter._current_hour_suffix)

    def test_close_idempotent(self):
        """Test that calling close() twice does not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = FileRequestMetricsExporter(
                _make_server_args(tmpdir), None, None
            )
            exporter.close()
            exporter.close()


class TestWriteRecord(CustomTestCase):

    def test_write_record_creates_json_line(self):
        """Test that write_record appends a valid JSON line to the log file."""
        with tempfile.TemporaryDirectory() as tmpdir:

            async def _run():
                exporter = FileRequestMetricsExporter(
                    _make_server_args(tmpdir), None, None
                )
                await exporter.write_record(
                    _TestReqInput(rid="write-test", text="prompt"),
                    {"meta_info": {"latency": 0.5}},
                )
                exporter.close()

            asyncio.run(_run())

            files = os.listdir(tmpdir)
            self.assertEqual(len(files), 1)
            with open(os.path.join(tmpdir, files[0])) as f:
                data = json.loads(f.readline())
            self.assertIn("request_parameters", data)
            self.assertAlmostEqual(data["latency"], 0.5)

    def test_health_check_filtered(self):
        """Test that requests with HEALTH_CHECK in rid are silently skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:

            async def _run():
                exporter = FileRequestMetricsExporter(
                    _make_server_args(tmpdir), None, None
                )
                await exporter.write_record(
                    _TestReqInput(rid="HEALTH_CHECK_001"), {"meta_info": {}}
                )
                exporter.close()

            asyncio.run(_run())

            self.assertEqual(len(os.listdir(tmpdir)), 0)


class TestCreateRequestMetricsExporters(CustomTestCase):

    def test_no_exporters_when_disabled(self):
        """Test that no exporters are created when export_metrics_to_file is False."""
        server_args = SimpleNamespace(export_metrics_to_file=False)
        self.assertEqual(len(create_request_metrics_exporters(server_args)), 0)

    def test_file_exporter_created_when_enabled(self):
        """Test that enabling file export creates a FileRequestMetricsExporter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporters = create_request_metrics_exporters(_make_server_args(tmpdir))
            self.assertEqual(len(exporters), 1)
            self.assertIsInstance(exporters[0], FileRequestMetricsExporter)


class TestRequestMetricsExporterManager(CustomTestCase):

    def test_exporter_enabled(self):
        """Test that exporter_enabled() returns True when file export is on."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RequestMetricsExporterManager(_make_server_args(tmpdir))
            self.assertTrue(manager.exporter_enabled())

    def test_exporter_disabled(self):
        """Test that exporter_enabled() returns False when file export is off."""
        server_args = SimpleNamespace(export_metrics_to_file=False)
        manager = RequestMetricsExporterManager(server_args)
        self.assertFalse(manager.exporter_enabled())

    def test_write_record_delegates_to_all_exporters(self):
        """Test that manager.write_record calls all configured exporters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RequestMetricsExporterManager(_make_server_args(tmpdir))

            async def _run():
                await manager.write_record(
                    _TestReqInput(rid="mgr-test"), {"meta_info": {"v": 1}}
                )

            asyncio.run(_run())
            files = os.listdir(tmpdir)
            self.assertEqual(len(files), 1)


class TestFileExporterEdgeCases(CustomTestCase):

    def test_ensure_file_handler_open_failure(self):
        """Test that open failure sets handler and suffix to None and raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = FileRequestMetricsExporter(
                _make_server_args(tmpdir), None, None
            )
            # Point export_dir to a non-existent path
            exporter.export_dir = "/nonexistent/dir/that/does/not/exist"
            with self.assertRaises(Exception):
                exporter._ensure_file_handler("20260319_10")
            self.assertIsNone(exporter._current_file_handler)
            self.assertIsNone(exporter._current_hour_suffix)

    def test_write_record_non_string_rid_not_filtered(self):
        """Test that a non-string rid (e.g. int) does not trigger health check filter."""
        with tempfile.TemporaryDirectory() as tmpdir:

            async def _run():
                exporter = FileRequestMetricsExporter(
                    _make_server_args(tmpdir), None, None
                )
                await exporter.write_record(
                    _TestReqInput(rid=12345), {"meta_info": {}}
                )
                exporter.close()

            asyncio.run(_run())
            self.assertEqual(len(os.listdir(tmpdir)), 1)

    def test_write_record_multiple_records(self):
        """Test that multiple records are appended as separate JSON lines."""
        with tempfile.TemporaryDirectory() as tmpdir:

            async def _run():
                exporter = FileRequestMetricsExporter(
                    _make_server_args(tmpdir), None, None
                )
                for i in range(3):
                    await exporter.write_record(
                        _TestReqInput(rid=f"req-{i}"), {"meta_info": {"i": i}}
                    )
                exporter.close()

            asyncio.run(_run())
            files = os.listdir(tmpdir)
            self.assertEqual(len(files), 1)
            with open(os.path.join(tmpdir, files[0])) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 3)
            for i, line in enumerate(lines):
                data = json.loads(line)
                self.assertEqual(data["i"], i)


    def test_rotation_close_failure_handled(self):
        """Test that exception during old handler close does not prevent rotation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = FileRequestMetricsExporter(
                _make_server_args(tmpdir), None, None
            )
            exporter._ensure_file_handler("20260319_10")
            # Sabotage the current handler so close() raises
            exporter._current_file_handler.close()  # close the real fd
            # Now rotation should handle the exception and open the new file
            exporter._ensure_file_handler("20260319_11")
            self.assertEqual(exporter._current_hour_suffix, "20260319_11")
            self.assertIsNotNone(exporter._current_file_handler)
            exporter.close()

    def test_close_exception_handled(self):
        """Test that close() does not raise even if the underlying fd is already closed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = FileRequestMetricsExporter(
                _make_server_args(tmpdir), None, None
            )
            exporter._ensure_file_handler("20260319_10")
            exporter._current_file_handler.close()  # sabotage
            exporter.close()  # should not raise
            self.assertIsNone(exporter._current_file_handler)

    def test_write_record_with_null_handler(self):
        """Test that write_record returns silently when file handler is None."""
        with tempfile.TemporaryDirectory() as tmpdir:

            async def _run():
                exporter = FileRequestMetricsExporter(
                    _make_server_args(tmpdir), None, None
                )
                # Force handler to None (simulates failed open)
                exporter._current_hour_suffix = "20260319_10"
                exporter._current_file_handler = None
                await exporter.write_record(
                    _TestReqInput(rid="null-handler"), {"meta_info": {}}
                )
                exporter.close()

            asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
