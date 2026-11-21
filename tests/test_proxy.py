import asyncio

import aiohttp
import orjson
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from inferobs.proxy.sse import EventKind, SSEParser, StreamMeter
from tests.conftest import start_proxy

LBL = {"gen_ai_operation_name": "chat", "gen_ai_provider_name": "sim", "gen_ai_request_model": "sim-llm"}


def sample(proxy, name, **labels):
    return proxy.metrics.registry.get_sample_value(name, {**LBL, **labels}) or 0.0


async def stream_chat(url, body, headers=None):
    events, meter = [], StreamMeter(t_start=0.0)
    async with (
        aiohttp.ClientSession() as s,
        s.post(f"{url}/v1/chat/completions", json=body, headers=headers or {}) as r,
    ):
        status = r.status
        parser = SSEParser()
        async for chunk in r.content.iter_any():
            for _raw, data in parser.feed(chunk):
                events.append((meter.on_event(data, 0.0), data))
    return status, events, meter


def chat_body(n_words=12, max_tokens=8, **kw):
    return {
        "model": "sim-llm",
        "messages": [{"role": "user", "content": " ".join(["the"] * n_words)}],
        "max_tokens": max_tokens,
        "ignore_eos": True,
        **kw,
    }


async def test_streaming_is_forwarded_and_metered(sim):
    proxy, server = await start_proxy(sim.url)
    try:
        status, events, _meter = await stream_chat(proxy.url, chat_body(stream=True))
        assert status == 200
        kinds = [k for k, _ in events]
        # usage was injected upstream but the client did not ask for it, so it must not see it
        assert EventKind.USAGE_ONLY not in kinds
        assert kinds.count(EventKind.CONTENT) == 8 and kinds[-1] is EventKind.DONE
        assert sample(proxy, "gen_ai_client_operation_duration_seconds_count", error_type="") == 1
        assert sample(proxy, "gen_ai_client_operation_time_to_first_chunk_seconds_count") == 1
        assert sample(proxy, "gen_ai_client_operation_time_per_output_chunk_seconds_count") == 7
        assert sample(proxy, "inferobs_time_per_output_token_seconds_count") == 1
        assert sample(proxy, "gen_ai_client_token_usage_sum", gen_ai_token_type="input") == 12
        assert sample(proxy, "gen_ai_client_token_usage_sum", gen_ai_token_type="output") == 8
        ttfc = sample(proxy, "gen_ai_client_operation_time_to_first_chunk_seconds_sum")
        dur = sample(proxy, "gen_ai_client_operation_duration_seconds_sum", error_type="")
        assert 0 < ttfc < dur
    finally:
        await server.close()


async def test_ttfc_is_observed_when_the_first_chunk_arrives(sim):
    """A long generation must not delay its TTFC observation until the stream ends."""
    proxy, server = await start_proxy(sim.url)
    try:
        async with aiohttp.ClientSession() as s:
            r = await s.post(f"{proxy.url}/v1/chat/completions", json=chat_body(stream=True, max_tokens=400))
            await r.content.readany()
            await asyncio.sleep(0.05)
            assert sample(proxy, "gen_ai_client_operation_time_to_first_chunk_seconds_count") == 1
            assert sample(proxy, "gen_ai_client_operation_duration_seconds_count", error_type="") == 0
            await r.read()
        assert sample(proxy, "gen_ai_client_operation_duration_seconds_count", error_type="") == 1
        assert sample(proxy, "gen_ai_client_operation_time_per_output_chunk_seconds_count") == 399
    finally:
        await server.close()


async def test_client_requested_usage_is_kept(sim):
    proxy, server = await start_proxy(sim.url)
    try:
        _, events, meter = await stream_chat(
            proxy.url, chat_body(stream=True, stream_options={"include_usage": True})
        )
        assert EventKind.USAGE_ONLY in [k for k, _ in events]
        assert meter.completion_tokens == 8
    finally:
        await server.close()


async def test_non_streaming_records_usage_without_ttfc(sim):
    proxy, server = await start_proxy(sim.url)
    try:
        async with (
            aiohttp.ClientSession() as s,
            s.post(f"{proxy.url}/v1/chat/completions", json=chat_body()) as r,
        ):
            body = await r.json()
        assert body["usage"]["completion_tokens"] == 8
        assert sample(proxy, "gen_ai_client_operation_duration_seconds_count", error_type="") == 1
        assert sample(proxy, "gen_ai_client_operation_time_to_first_chunk_seconds_count") == 0
        assert sample(proxy, "gen_ai_client_token_usage_sum", gen_ai_token_type="output") == 8
    finally:
        await server.close()


async def test_upstream_error_status_is_passed_through_and_labelled(sim):
    proxy, server = await start_proxy(sim.url)
    try:
        sim.engine.faults.error_rate = 1.0
        sim.engine.faults.error_status = 503
        status, _, _ = await stream_chat(proxy.url, chat_body(stream=True))
        assert status == 503
        assert sample(proxy, "gen_ai_client_operation_duration_seconds_count", error_type="503") == 1
        assert sample(proxy, "gen_ai_client_operation_duration_seconds_count", error_type="") == 0
    finally:
        await server.close()


async def test_stalled_upstream_times_out_inside_the_stream(sim):
    proxy, server = await start_proxy(sim.url, idle_timeout=0.3)
    try:
        sim.engine.faults.hang = True
        status, _events, meter = await stream_chat(proxy.url, chat_body(stream=True))
        assert status == 200  # headers were already sent when the stall began
        assert meter.stream_error == "timeout"
        assert sample(proxy, "gen_ai_client_operation_duration_seconds_count", error_type="timeout") == 1
    finally:
        sim.engine.faults.hang = False
        await server.close()


async def test_unreachable_upstream_is_502_and_marked_down():
    proxy, server = await start_proxy("http://127.0.0.1:9")
    try:
        status, _, _ = await stream_chat(proxy.url, chat_body(stream=True))
        assert status == 502
        assert (
            sample(proxy, "gen_ai_client_operation_duration_seconds_count", error_type="connection_error")
            == 1
        )
        assert (
            proxy.metrics.registry.get_sample_value("inferobs_upstream_up", {"gen_ai_provider_name": "sim"})
            == 0
        )
    finally:
        await server.close()


async def test_client_disconnect_aborts_upstream_request(sim):
    proxy, server = await start_proxy(sim.url)
    try:
        async with aiohttp.ClientSession() as s:
            r = await s.post(f"{proxy.url}/v1/chat/completions", json=chat_body(stream=True, max_tokens=5000))
            await r.content.readany()
            r.close()  # drop the connection mid-stream
        for _ in range(100):
            await asyncio.sleep(0.02)
            if sample(proxy, "gen_ai_client_operation_duration_seconds_count", error_type="cancelled"):
                break
        assert sample(proxy, "gen_ai_client_operation_duration_seconds_count", error_type="cancelled") == 1
        for _ in range(100):
            await asyncio.sleep(0.02)
            if not sim.engine.running:
                break
        assert not sim.engine.running, "engine kept generating for a client that left"
        aborted = sim.metrics.registry.get_sample_value(
            "vllm:request_success_total", {"model_name": "sim-llm", "engine": "0", "finished_reason": "abort"}
        )
        assert aborted == 1
    finally:
        await server.close()


async def test_spans_follow_semconv_and_exemplars_carry_trace_ids(sim):
    exporter = InMemorySpanExporter()
    proxy, server = await start_proxy(sim.url, exporter=exporter)
    try:
        parent_trace = "4bf92f3577b34da6a3ce929d0e0e4736"
        tp = f"00-{parent_trace}-00f067aa0ba902b7-01"
        await stream_chat(proxy.url, chat_body(stream=True), headers={"traceparent": tp})
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        s = spans[0]
        assert s.name == "chat sim-llm"
        assert format(s.context.trace_id, "032x") == parent_trace
        a = s.attributes
        assert a["gen_ai.operation.name"] == "chat" and a["gen_ai.provider.name"] == "sim"
        assert a["gen_ai.usage.input_tokens"] == 12 and a["gen_ai.usage.output_tokens"] == 8
        assert a["gen_ai.response.finish_reasons"] == ("length",)
        assert any(e.name == "gen_ai.first_chunk" for e in s.events)
        async with (
            aiohttp.ClientSession() as cs,
            cs.get(f"{proxy.url}/metrics", headers={"Accept": "application/openmetrics-text"}) as r,
        ):
            text = await r.text()
        assert f'trace_id="{parent_trace}"' in text
    finally:
        await server.close()


async def test_unknown_model_names_are_bucketed(sim):
    proxy, server = await start_proxy(sim.url)
    try:
        status, _, _ = await stream_chat(proxy.url, {**chat_body(stream=True), "model": "made-up"})
        assert status == 404
        v = proxy.metrics.registry.get_sample_value(
            "gen_ai_client_operation_duration_seconds_count",
            {**LBL, "gen_ai_request_model": "_OTHER", "error_type": "404"},
        )
        assert v == 1
    finally:
        await server.close()


async def test_upstream_dropping_mid_stream_is_stream_interrupted():
    async def broken(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i in range(2):
            obj = {"id": "x", "model": "sim-llm", "choices": [{"index": 0, "delta": {"content": f"t{i}"}}]}
            await resp.write(b"data: " + orjson.dumps(obj) + b"\n\n")
        request.transport.close()
        return resp

    async def models(_r):
        return web.json_response({"data": [{"id": "sim-llm"}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", broken)
    app.router.add_get("/v1/models", models)
    up = TestServer(app)
    await up.start_server()
    proxy, server = await start_proxy(str(up.make_url("")).rstrip("/"))
    try:
        _, _events, meter = await stream_chat(proxy.url, chat_body(stream=True))
        assert meter.content_chunks == 2
        assert meter.stream_error == "stream_interrupted"
        assert (
            sample(proxy, "gen_ai_client_operation_duration_seconds_count", error_type="stream_interrupted")
            == 1
        )
    finally:
        await server.close()
        await up.close()


@pytest.mark.parametrize("layout", ["fine", "semconv"])
async def test_bucket_layouts(sim, layout):
    proxy, server = await start_proxy(sim.url, layout=layout)
    try:
        await stream_chat(proxy.url, chat_body(stream=True))
        assert proxy.metrics.layout == layout
    finally:
        await server.close()
