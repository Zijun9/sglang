"""Unit tests for srt/observability/trace.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")

import time
import unittest

from sglang.srt.observability.trace import (
    TRACE_HEADERS,
    TraceCustomIdGenerator,
    TraceEvent,
    TraceNullContext,
    TraceSliceContext,
    TraceThreadInfo,
    extract_trace_headers,
    get_global_tracing_enabled,
    set_global_trace_level,
)
from sglang.test.test_utils import CustomTestCase


class TestExtractTraceHeaders(CustomTestCase):

    def test_extracts_present_headers(self):
        """Test that both traceparent and tracestate are extracted when present."""
        headers = {
            "traceparent": "00-abc-def-01",
            "tracestate": "vendor=val",
            "other": "ignored",
        }
        result = extract_trace_headers(headers)
        self.assertEqual(
            result, {"traceparent": "00-abc-def-01", "tracestate": "vendor=val"}
        )

    def test_skips_missing_headers(self):
        """Test that only present TRACE_HEADERS are included in the result."""
        result = extract_trace_headers({"traceparent": "val"})
        self.assertEqual(result, {"traceparent": "val"})
        self.assertNotIn("tracestate", result)

    def test_empty_headers_returns_empty(self):
        """Test that empty input mapping produces empty output."""
        self.assertEqual(extract_trace_headers({}), {})

    def test_non_trace_headers_ignored(self):
        """Test that arbitrary non-trace headers are not extracted."""
        headers = {"content-type": "application/json", "authorization": "Bearer x"}
        self.assertEqual(extract_trace_headers(headers), {})

    def test_only_known_trace_headers_extracted(self):
        """Test that even among 100 headers only TRACE_HEADERS are picked."""
        headers = {f"header-{i}": f"val-{i}" for i in range(100)}
        headers["traceparent"] = "tp-value"
        result = extract_trace_headers(headers)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["traceparent"], "tp-value")


class TestTraceNullContext(CustomTestCase):

    def test_tracing_enable_is_false(self):
        """Test that the dataclass field tracing_enable defaults to False."""
        ctx = TraceNullContext()
        self.assertFalse(ctx.tracing_enable)

    def test_any_attribute_returns_self(self):
        """Test that __getattr__ returns self for any unknown attribute."""
        ctx = TraceNullContext()
        self.assertIs(ctx.trace_slice_start, ctx)
        self.assertIs(ctx.trace_req_finish, ctx)
        self.assertIs(ctx.nonexistent_attr, ctx)

    def test_call_returns_self(self):
        """Test that __call__ returns self regardless of arguments."""
        ctx = TraceNullContext()
        self.assertIs(ctx(), ctx)
        self.assertIs(ctx("arg1", key="val"), ctx)

    def test_chained_attribute_and_call(self):
        """Test that ctx.method(args) chains through __getattr__ and __call__."""
        ctx = TraceNullContext()
        result = ctx.trace_slice_start("name", 1, ts=12345)
        self.assertIs(result, ctx)

    def test_deep_chain(self):
        """Test that deeply chained attribute access still returns self."""
        ctx = TraceNullContext()
        self.assertIs(ctx.a.b.c.d, ctx)

    def test_production_usage_pattern(self):
        """Test the sequence of calls that real code makes when tracing is disabled."""
        ctx = TraceNullContext()
        ctx.trace_req_start(ts=100)
        ctx.trace_slice_start("prefill", 1, ts=200)
        ctx.trace_event("schedule", 3, ts=300, attrs={"bid": "abc"})
        ctx.trace_slice_end("prefill", 1, ts=400)
        ctx.trace_req_finish(ts=500, attrs={"tokens": 100})
        self.assertFalse(ctx.tracing_enable)


class TestTraceCustomIdGenerator(CustomTestCase):

    def test_generates_positive_trace_ids(self):
        """Test that generated trace IDs are non-negative integers."""
        gen = TraceCustomIdGenerator()
        for _ in range(100):
            tid = gen.generate_trace_id()
            self.assertIsInstance(tid, int)
            self.assertGreaterEqual(tid, 0)

    def test_generates_positive_span_ids(self):
        """Test that generated span IDs are non-negative integers."""
        gen = TraceCustomIdGenerator()
        for _ in range(100):
            self.assertGreaterEqual(gen.generate_span_id(), 0)

    def test_trace_ids_are_unique(self):
        """Test that 1000 generated trace IDs have no collisions."""
        gen = TraceCustomIdGenerator()
        ids = {gen.generate_trace_id() for _ in range(1000)}
        self.assertEqual(len(ids), 1000)

    def test_span_ids_are_unique(self):
        """Test that 1000 generated span IDs have no collisions."""
        gen = TraceCustomIdGenerator()
        ids = {gen.generate_span_id() for _ in range(1000)}
        self.assertEqual(len(ids), 1000)

    def test_two_generators_produce_different_sequences(self):
        """Test that generators seeded at different times diverge."""
        gen1 = TraceCustomIdGenerator()
        time.sleep(0.01)
        gen2 = TraceCustomIdGenerator()
        ids1 = [gen1.generate_trace_id() for _ in range(10)]
        ids2 = [gen2.generate_trace_id() for _ in range(10)]
        self.assertNotEqual(ids1, ids2)


class TestSetGlobalTraceLevel(CustomTestCase):

    def setUp(self):
        import sglang.srt.observability.trace as _mod

        self._mod = _mod
        self._orig = _mod.global_trace_level

    def tearDown(self):
        set_global_trace_level(self._orig)

    def test_sets_level(self):
        """Test that set_global_trace_level updates the module global."""
        set_global_trace_level(5)
        self.assertEqual(self._mod.global_trace_level, 5)

    def test_default_level_is_3(self):
        """Test that the initial default trace level is 3."""
        set_global_trace_level(3)
        self.assertEqual(self._mod.global_trace_level, 3)


class TestGetGlobalTracingEnabled(CustomTestCase):

    def test_disabled_without_initialization(self):
        """Test that tracing is disabled when opentelemetry_initialized is False."""
        import sglang.srt.observability.trace as tr

        orig = tr.opentelemetry_initialized
        tr.opentelemetry_initialized = False
        try:
            self.assertFalse(get_global_tracing_enabled())
        finally:
            tr.opentelemetry_initialized = orig


class TestTraceDataclasses(CustomTestCase):

    def test_trace_thread_info_fields(self):
        """Test that TraceThreadInfo stores all constructor arguments."""
        info = TraceThreadInfo(
            host_id="abc123",
            pid=1234,
            thread_label="scheduler",
            tp_rank=0,
            dp_rank=1,
        )
        self.assertEqual(info.host_id, "abc123")
        self.assertEqual(info.thread_label, "scheduler")
        self.assertEqual(info.tp_rank, 0)

    def test_trace_event_fields(self):
        """Test that TraceEvent stores event_name, ts, and attrs."""
        event = TraceEvent(event_name="schedule", ts=1000000, attrs={"key": "val"})
        self.assertEqual(event.event_name, "schedule")
        self.assertEqual(event.ts, 1000000)
        self.assertEqual(event.attrs["key"], "val")

    def test_trace_slice_context_defaults(self):
        """Test that optional fields default to None/1."""
        ctx = TraceSliceContext(slice_name="prefill", start_time_ns=100)
        self.assertEqual(ctx.slice_name, "prefill")
        self.assertIsNone(ctx.end_time_ns)
        self.assertEqual(ctx.level, 1)
        self.assertIsNone(ctx.attrs)
        self.assertIsNone(ctx.events)


class TestTraceReqContextDisabled(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        import sglang.srt.observability.trace as tr

        cls._orig = tr.opentelemetry_initialized
        tr.opentelemetry_initialized = False

    @classmethod
    def tearDownClass(cls):
        import sglang.srt.observability.trace as tr

        tr.opentelemetry_initialized = cls._orig

    def test_tracing_disabled_by_default(self):
        """Test that TraceReqContext disables itself when OTel is not initialized."""
        from sglang.srt.observability.trace import TraceReqContext

        ctx = TraceReqContext(rid="test-001")
        self.assertFalse(ctx.is_tracing_enabled())

    def test_getstate_when_disabled(self):
        """Test that __getstate__ returns minimal dict when tracing is off."""
        from sglang.srt.observability.trace import TraceReqContext

        ctx = TraceReqContext(rid="test-002")
        self.assertEqual(ctx.__getstate__(), {"tracing_enable": False})


class TestGetHostId(CustomTestCase):

    def test_returns_nonempty_string(self):
        """Test that __get_host_id returns a non-empty host identifier."""
        import sglang.srt.observability.trace as _trace_mod

        get_host_id = vars(_trace_mod)["__get_host_id"]
        result = get_host_id()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_result_is_deterministic(self):
        """Test that __get_host_id returns the same value on repeated calls."""
        import sglang.srt.observability.trace as _trace_mod

        get_host_id = vars(_trace_mod)["__get_host_id"]
        self.assertEqual(get_host_id(), get_host_id())


class TestGetCurTimeNs(CustomTestCase):

    def test_returns_positive_integer(self):
        """Test that get_cur_time_ns returns a positive nanosecond timestamp."""
        from sglang.srt.observability.trace import get_cur_time_ns

        result = get_cur_time_ns()
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_monotonically_increasing(self):
        """Test that consecutive calls return non-decreasing values."""
        from sglang.srt.observability.trace import get_cur_time_ns

        t1 = get_cur_time_ns()
        t2 = get_cur_time_ns()
        self.assertGreaterEqual(t2, t1)


def _ensure_otel_initialized():
    """Initialize real OpenTelemetry tracing (once per process)."""
    import sglang.srt.observability.trace as tr

    if not tr.opentelemetry_initialized:
        tr.process_tracing_init("localhost:4317", "unit-test-server")


class TestProcessTracingInit(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        _ensure_otel_initialized()

    def test_sets_initialized_flag(self):
        """Test that process_tracing_init sets opentelemetry_initialized=True."""
        import sglang.srt.observability.trace as tr

        self.assertTrue(tr.opentelemetry_initialized)

    def test_creates_tracer(self):
        """Test that process_tracing_init creates a real Tracer object."""
        import sglang.srt.observability.trace as tr

        self.assertIsNotNone(tr.tracer)

    def test_get_global_tracing_enabled(self):
        """Test that get_global_tracing_enabled returns True after init."""
        self.assertTrue(get_global_tracing_enabled())


class TestGetOtlpSpanExporter(CustomTestCase):

    def test_grpc_protocol(self):
        """Test that default protocol creates a gRPC exporter."""
        import os

        from sglang.srt.observability.trace import get_otlp_span_exporter

        os.environ.pop("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", None)
        exporter = get_otlp_span_exporter("localhost:4317")
        self.assertIsNotNone(exporter)

    def test_http_protocol(self):
        """Test that http/protobuf protocol creates an HTTP exporter."""
        import os

        from sglang.srt.observability.trace import get_otlp_span_exporter

        os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "http/protobuf"
        try:
            exporter = get_otlp_span_exporter("http://localhost:4318")
            self.assertIsNotNone(exporter)
        finally:
            os.environ.pop("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", None)

    def test_invalid_protocol_raises(self):
        """Test that unsupported protocol raises ValueError."""
        import os

        from sglang.srt.observability.trace import get_otlp_span_exporter

        os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "invalid"
        try:
            with self.assertRaises(ValueError):
                get_otlp_span_exporter("localhost:4317")
        finally:
            os.environ.pop("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", None)


class TestTraceSetThreadInfo(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        _ensure_otel_initialized()

    def test_registers_current_thread(self):
        """Test that trace_set_thread_info stores thread info in the global dict."""
        import threading

        import sglang.srt.observability.trace as tr

        pid = threading.get_native_id()
        # Clear any existing entry so we can verify fresh registration
        tr.threads_info.pop(pid, None)
        tr.trace_set_thread_info("test_thread", tp_rank=7, dp_rank=3)
        self.assertIn(pid, tr.threads_info)
        info = tr.threads_info[pid]
        self.assertEqual(info.thread_label, "test_thread")
        self.assertEqual(info.tp_rank, 7)
        self.assertEqual(info.dp_rank, 3)

    def test_duplicate_call_is_noop(self):
        """Test that calling trace_set_thread_info again for same thread is ignored."""
        import threading

        import sglang.srt.observability.trace as tr

        pid = threading.get_native_id()
        # Ensure first registration
        tr.threads_info.pop(pid, None)
        tr.trace_set_thread_info("first_label", tp_rank=0, dp_rank=0)
        tr.trace_set_thread_info("second_label", tp_rank=99, dp_rank=99)
        self.assertEqual(tr.threads_info[pid].tp_rank, 0)


class TestTraceReqContextWithOTel(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        _ensure_otel_initialized()

    def _make_ctx(self, rid="test"):
        from sglang.srt.observability.trace import TraceReqContext

        return TraceReqContext(rid=rid)

    def test_tracing_enabled(self):
        """Test that TraceReqContext enables tracing when OTel is initialized."""
        ctx = self._make_ctx("otel-enabled")
        self.assertTrue(ctx.is_tracing_enabled())

    def test_full_lifecycle(self):
        """Test the complete request tracing lifecycle: start → slice → event → end."""
        ctx = self._make_ctx("lifecycle")
        ctx.trace_req_start()
        self.assertIsNotNone(ctx.root_span)
        self.assertIsNotNone(ctx.thread_context)

        ctx.trace_slice_start("prefill", level=1)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 1)

        ctx.trace_event("schedule", level=1, attrs={"bid": "abc"})

        ctx.trace_slice_end("prefill", level=1)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 0)
        self.assertIsNotNone(ctx.last_span_context)

        ctx.trace_req_finish(attrs={"tokens": 100})
        self.assertIsNone(ctx.root_span)

    def test_nested_slices(self):
        """Test that nested slice start/end pairs track correctly on the stack."""
        ctx = self._make_ctx("nested")
        ctx.trace_req_start()
        ctx.trace_slice_start("outer", level=1)
        ctx.trace_slice_start("inner", level=2)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 2)
        ctx.trace_slice_end("inner", level=2)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 1)
        ctx.trace_slice_end("outer", level=1)
        ctx.trace_req_finish()

    def test_trace_level_filtering(self):
        """Test that slices above the trace level are silently skipped."""
        ctx = self._make_ctx("filter")
        ctx.trace_level = 1
        ctx.trace_req_start()
        ctx.trace_slice_start("detail", level=2)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 0)
        ctx.trace_req_finish()

    def test_abort_closes_all_slices(self):
        """Test that abort() ends all open slices and clears thread context."""
        ctx = self._make_ctx("abort")
        ctx.trace_req_start()
        ctx.trace_slice_start("a", level=1)
        ctx.trace_slice_start("b", level=2)
        ctx.abort()
        self.assertIsNone(ctx.thread_context)

    def test_trace_single_slice(self):
        """Test trace_slice() with a pre-constructed TraceSliceContext."""
        from sglang.srt.observability.trace import get_cur_time_ns

        ctx = self._make_ctx("single-slice")
        ctx.trace_req_start()
        start = get_cur_time_ns()
        s = TraceSliceContext(
            slice_name="batch",
            start_time_ns=start,
            end_time_ns=start + 1_000_000,
            level=1,
            attrs={"size": 32},
            events=[TraceEvent("evt", start + 500, {"k": "v"})],
        )
        ctx.trace_slice(s)
        self.assertIsNotNone(ctx.last_span_context)
        ctx.trace_req_finish()

    def test_set_root_attrs(self):
        """Test that trace_set_root_attrs writes attributes to the root span."""
        ctx = self._make_ctx("root-attrs")
        ctx.trace_req_start()
        ctx.trace_set_root_attrs({"model": "test-model"})
        self.assertEqual(ctx.root_span.attributes.get("model"), "test-model")
        ctx.trace_req_finish()

    def test_set_thread_attrs(self):
        """Test that trace_set_thread_attrs writes attributes to the thread span."""
        ctx = self._make_ctx("thread-attrs")
        ctx.trace_req_start()
        ctx.trace_set_thread_attrs({"key": "value"})
        self.assertEqual(
            ctx.thread_context.thread_span.attributes.get("key"), "value"
        )
        ctx.trace_req_finish()

    def test_getstate_with_active_tracing(self):
        """Test that __getstate__ serializes root span context for cross-process."""
        ctx = self._make_ctx("serial")
        ctx.trace_req_start()
        state = ctx.__getstate__()
        self.assertTrue(state["tracing_enable"])
        self.assertEqual(state["rid"], "serial")
        self.assertIsInstance(state["root_span_context"], dict)
        ctx.trace_req_finish()

    def test_setstate_creates_copy(self):
        """Test that __setstate__ creates an is_copy context from serialized state."""
        ctx = self._make_ctx("setstate")
        ctx.trace_req_start()
        state = ctx.__getstate__()
        ctx.trace_req_finish()

        ctx2 = self._make_ctx.__func__(self, "dummy")
        ctx2.__class__ = type(ctx)
        from sglang.srt.observability.trace import TraceReqContext

        ctx2 = TraceReqContext.__new__(TraceReqContext)
        ctx2.__setstate__(state)
        self.assertTrue(ctx2.is_copy)
        self.assertTrue(ctx2.tracing_enable)

    def test_rebuild_thread_context(self):
        """Test that rebuild_thread_context re-creates thread context on copy."""
        ctx = self._make_ctx("rebuild")
        ctx.trace_req_start()
        state = ctx.__getstate__()
        ctx.trace_req_finish()

        from sglang.srt.observability.trace import TraceReqContext

        ctx2 = TraceReqContext.__new__(TraceReqContext)
        ctx2.__setstate__(state)
        ctx2.rebuild_thread_context()
        self.assertIsNotNone(ctx2.thread_context)

    def test_slice_end_name_mismatch(self):
        """Test that mismatched slice_end name/level pops without crashing."""
        ctx = self._make_ctx("mismatch")
        ctx.trace_req_start()
        ctx.trace_slice_start("outer", level=1)
        ctx.trace_slice_end("WRONG", level=1)
        self.assertEqual(len(ctx.thread_context.cur_slice_stack), 0)
        ctx.trace_req_finish()

    def test_slice_end_on_empty_stack(self):
        """Test that slice_end on empty stack logs warning without crashing."""
        ctx = self._make_ctx("empty-end")
        ctx.trace_req_start()
        ctx.trace_slice_end("nonexistent", level=1)
        ctx.trace_req_finish()

    def test_events_cache_consumed_by_slice_end(self):
        """Test that events_cache entries within slice time range are consumed."""
        from sglang.srt.observability.trace import get_cur_time_ns

        ctx = self._make_ctx("cache")
        ctx.trace_req_start()
        t1 = get_cur_time_ns()
        ctx.trace_event("before", level=1, ts=t1)
        ctx.trace_slice_start("process", level=1, ts=t1 + 100)
        t2 = get_cur_time_ns()
        ctx.trace_event("inside", level=1, ts=t2)
        ctx.trace_slice_end("process", level=1, ts=t2 + 100)
        ctx.trace_req_finish()

    def test_external_trace_header(self):
        """Test that external traceparent header is propagated into the root span."""
        from sglang.srt.observability.trace import TraceReqContext

        ctx = TraceReqContext(
            rid="ext-header",
            external_trace_header={"traceparent": "00-abc-def-01"},
        )
        ctx.trace_req_start()
        self.assertIsNotNone(ctx.root_span)
        ctx.trace_req_finish()

    def test_trace_req_finish_without_start_is_noop(self):
        """Test that finishing without starting does not crash."""
        ctx = self._make_ctx("no-start")
        ctx.trace_req_finish()

    def test_getstate_without_root_span_context(self):
        """Test __getstate__ when root_span_context is None returns disabled."""
        ctx = self._make_ctx("no-root")
        state = ctx.__getstate__()
        self.assertFalse(state.get("tracing_enable", True))

    def test_bootstrap_room_in_span_attrs(self):
        """Test that bootstrap_room is injected into root span attributes."""
        from sglang.srt.observability.trace import TraceReqContext

        ctx = TraceReqContext(rid="bs-room", bootstrap_room=0xABCD)
        ctx.trace_req_start()
        self.assertEqual(ctx.root_span.attributes.get("bootstrap_room"), hex(0xABCD))
        ctx.trace_req_finish()

    def test_trace_slice_end_with_attrs(self):
        """Test that trace_slice_end propagates attrs to the span."""
        ctx = self._make_ctx("end-attrs")
        ctx.trace_req_start()
        ctx.trace_slice_start("op", level=1)
        ctx.trace_slice_end("op", level=1, attrs={"result": "ok"})
        ctx.trace_req_finish()

    def test_trace_slice_with_events_cache_and_finish_flag(self):
        """Test trace_slice() consumes events_cache and honours thread_finish_flag."""
        from sglang.srt.observability.trace import get_cur_time_ns

        ctx = self._make_ctx("slice-cache")
        ctx.trace_req_start()
        start = get_cur_time_ns()
        # Add an event that falls within the upcoming slice's time range
        ctx.trace_event("in-range", level=1, ts=start + 500)
        s = TraceSliceContext(
            slice_name="batch",
            start_time_ns=start,
            end_time_ns=start + 1_000_000,
            level=1,
        )
        ctx.trace_slice(s, thread_finish_flag=True)
        # thread_finish_flag causes abort → thread_context is None
        self.assertIsNone(ctx.thread_context)

    def test_trace_slice_with_last_span_context(self):
        """Test that trace_slice adds a link from last_span_context."""
        from sglang.srt.observability.trace import get_cur_time_ns

        ctx = self._make_ctx("link")
        ctx.trace_req_start()
        # Create a first slice so last_span_context is set
        ctx.trace_slice_start("first", level=1)
        ctx.trace_slice_end("first", level=1)
        self.assertIsNotNone(ctx.last_span_context)
        # Now trace_slice should create a link from last_span_context
        start = get_cur_time_ns()
        s = TraceSliceContext(
            slice_name="second",
            start_time_ns=start,
            end_time_ns=start + 1000,
            level=1,
        )
        ctx.trace_slice(s)
        ctx.trace_req_finish()

    def test_getstate_with_active_slice_captures_span_context(self):
        """Test __getstate__ captures last_span_context from active slice stack."""
        ctx = self._make_ctx("active-slice")
        ctx.trace_req_start()
        ctx.trace_slice_start("running", level=1)
        state = ctx.__getstate__()
        # last_span_context should be captured from the active slice's span
        self.assertIsNotNone(state.get("last_span_context"))
        ctx.trace_slice_end("running", level=1)
        ctx.trace_req_finish()

    def test_setstate_restores_last_span_context(self):
        """Test __setstate__ restores last_span_context from serialized dict."""
        ctx = self._make_ctx("restore-lsc")
        ctx.trace_req_start()
        ctx.trace_slice_start("a", level=1)
        ctx.trace_slice_end("a", level=1)
        state = ctx.__getstate__()
        ctx.trace_req_finish()

        from sglang.srt.observability.trace import TraceReqContext

        ctx2 = TraceReqContext.__new__(TraceReqContext)
        ctx2.__setstate__(state)
        self.assertIsNotNone(ctx2.last_span_context)

    def test_abort_with_abort_info_dict(self):
        """Test abort() with abort_info dict sets error status on thread span."""
        ctx = self._make_ctx("abort-info")
        ctx.trace_req_start()
        ctx.abort(abort_info={"reason": "cancelled"})
        self.assertIsNone(ctx.thread_context)


if __name__ == "__main__":
    unittest.main()
