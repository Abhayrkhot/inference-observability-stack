"""OpenAI-compatible metering proxy.

Sits in front of one or more inference engines (vLLM, llama.cpp server, the simulator), forwards
requests unchanged and measures what a caller actually experiences: time to first chunk, gaps
between chunks, mean time per output token, total duration and token usage. Each request gets an
OpenTelemetry span (W3C trace context is propagated to the engine) and a trace-id exemplar on its
histogram observations, plus an optional JSON access-log line.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field

import aiohttp
import orjson
from aiohttp import web
from opentelemetry import context as otel_context
from opentelemetry import propagate, trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from prometheus_client import exposition

from inferobs.proxy.metrics import ProxyMetrics
from inferobs.proxy.sse import EventKind, SSEParser, StreamMeter
from inferobs.proxy.tracing import make_tracer_provider, tracer_from

log = logging.getLogger("inferobs.proxy")

OTHER_MODEL = "_OTHER"
FORWARD_HEADERS = ("authorization", "content-type", "accept", "x-request-id", "openai-organization")


@dataclass
class ProxyConfig:
    upstreams: list[tuple[str, str]]  # (provider name, base URL)
    layout: str = "fine"
    connect_timeout: float = 5.0
    idle_timeout: float = 60.0  # longest allowed silence from the upstream, before or between chunks
    health_interval: float = 5.0
    models_interval: float = 30.0
    access_log: str | None = None
    access_log_gaps: bool = False
    otlp_endpoint: str | None = None
    sample_ratio: float = 1.0
    service_name: str = "inferobs-proxy"
    max_connections: int = 2048
    inject_usage: bool = True  # ask for stream usage; the extra chunk is hidden from clients that didn't


@dataclass
class Upstream:
    provider: str
    base: str
    models: set[str] = field(default_factory=set)
    up: bool = False


class AccessLog:
    """Buffered JSON-lines writer flushed by a background task."""

    def __init__(self, path: str) -> None:
        self._f = open(path, "ab", buffering=1 << 16)  # noqa: SIM115
        self._task: asyncio.Task | None = None

    def write(self, rec: dict) -> None:
        self._f.write(orjson.dumps(rec) + b"\n")

    async def run(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            self._f.flush()

    def close(self) -> None:
        self._f.flush()
        self._f.close()


class Proxy:
    def __init__(self, cfg: ProxyConfig, metrics: ProxyMetrics | None = None, tracer_provider=None) -> None:
        if not cfg.upstreams:
            raise ValueError("at least one upstream is required")
        self.cfg = cfg
        self.metrics = metrics or ProxyMetrics(layout=cfg.layout)
        self.tracer_provider = tracer_provider or make_tracer_provider(
            cfg.service_name, cfg.otlp_endpoint, cfg.sample_ratio
        )
        self.tracer = tracer_from(self.tracer_provider)
        self.upstreams = [Upstream(p, u.rstrip("/")) for p, u in cfg.upstreams]
        self.session: aiohttp.ClientSession | None = None
        self.access_log = AccessLog(cfg.access_log) if cfg.access_log else None
        self._tasks: list[asyncio.Task] = []
        for u in self.upstreams:
            self.metrics.upstream_up.labels(u.provider).set(0)
            self.metrics.inflight.labels(u.provider).set(0)

    # ---------- lifecycle ----------
    def app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/chat/completions", self.handle_generate)
        app.router.add_post("/v1/completions", self.handle_generate)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_get("/metrics", self.handle_metrics)
        app.router.add_get("/health", self.handle_health)
        app.on_startup.append(self._startup)
        app.on_cleanup.append(self._cleanup)
        return app

    async def _startup(self, _app: web.Application) -> None:
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.cfg.max_connections, keepalive_timeout=60),
            timeout=aiohttp.ClientTimeout(
                total=None, connect=self.cfg.connect_timeout, sock_read=self.cfg.idle_timeout
            ),
            auto_decompress=False,
        )
        await asyncio.gather(*(self._probe(u) for u in self.upstreams))
        await asyncio.gather(*(self._refresh_models(u) for u in self.upstreams))
        self._tasks.append(asyncio.create_task(self._health_loop()))
        self._tasks.append(asyncio.create_task(self._models_loop()))
        if self.access_log:
            self._tasks.append(asyncio.create_task(self.access_log.run()))

    async def _cleanup(self, _app: web.Application) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        if self.session:
            await self.session.close()
        if self.access_log:
            self.access_log.close()
        if self.tracer_provider is not None:
            self.tracer_provider.force_flush(5000)

    # ---------- upstream health and model discovery ----------
    async def _probe(self, u: Upstream) -> None:
        assert self.session is not None
        t0 = time.perf_counter()
        ok = False
        try:
            async with self.session.get(
                f"{u.base}/health", timeout=aiohttp.ClientTimeout(total=max(1.0, self.cfg.health_interval))
            ) as r:
                await r.read()
                ok = r.status == 200
        except (TimeoutError, aiohttp.ClientError, OSError):
            ok = False
        u.up = ok
        self.metrics.upstream_up.labels(u.provider).set(1 if ok else 0)
        self.metrics.upstream_probe.labels(u.provider).set(time.perf_counter() - t0)

    async def _refresh_models(self, u: Upstream) -> None:
        assert self.session is not None
        try:
            async with self.session.get(f"{u.base}/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 200:
                    data = orjson.loads(await r.read())
                    u.models = {m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m}
        except (TimeoutError, aiohttp.ClientError, OSError, orjson.JSONDecodeError):
            pass

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.health_interval)
            await asyncio.gather(*(self._probe(u) for u in self.upstreams))

    async def _models_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.models_interval)
            await asyncio.gather(*(self._refresh_models(u) for u in self.upstreams))

    def route(self, model: str) -> Upstream:
        if len(self.upstreams) == 1:
            return self.upstreams[0]
        for u in self.upstreams:
            if model in u.models:
                return u
        for u in self.upstreams:
            if u.up:
                return u
        return self.upstreams[0]

    def model_label(self, u: Upstream, model: str) -> str:
        # Bound label cardinality: client-supplied model names are only used once the upstream lists them.
        if not u.models or model in u.models:
            return model or OTHER_MODEL
        return OTHER_MODEL

    # ---------- simple endpoints ----------
    async def handle_models(self, _request: web.Request) -> web.Response:
        seen: dict[str, dict] = {}
        for u in self.upstreams:
            for m in sorted(u.models):
                seen.setdefault(m, {"id": m, "object": "model", "owned_by": u.provider})
        return web.json_response({"object": "list", "data": list(seen.values())}, dumps=_dumps)

    async def handle_metrics(self, request: web.Request) -> web.Response:
        encoder, ctype = exposition.choose_encoder(request.headers.get("Accept", ""))
        body = encoder(self.metrics.registry)
        return web.Response(body=body, headers={"Content-Type": ctype})

    async def handle_health(self, _request: web.Request) -> web.Response:
        up = [u.provider for u in self.upstreams if u.up]
        return web.json_response(
            {"status": "ok" if up else "degraded", "upstreams_up": up}, status=200 if up else 503
        )

    # ---------- the metered path ----------
    async def handle_generate(self, request: web.Request) -> web.StreamResponse:
        t0 = time.perf_counter()
        wall0 = time.time()
        assert self.session is not None
        raw_body = await request.read()
        try:
            req = orjson.loads(raw_body)
            if not isinstance(req, dict):
                raise ValueError("body must be a JSON object")
        except (orjson.JSONDecodeError, ValueError) as e:
            return web.json_response(
                {"error": {"message": f"invalid JSON body: {e}", "type": "invalid_request"}}, status=400
            )
        op = "chat" if request.path.endswith("/chat/completions") else "text_completion"
        model = str(req.get("model", ""))
        stream = bool(req.get("stream"))
        u = self.route(model)
        labels = (op, u.provider, self.model_label(u, model))

        injected = False
        if stream and self.cfg.inject_usage:
            so = req.get("stream_options")
            if not isinstance(so, dict) or not so.get("include_usage"):
                req["stream_options"] = {**(so if isinstance(so, dict) else {}), "include_usage": True}
                injected = True
        body = orjson.dumps(req) if injected else raw_body

        headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_HEADERS}
        headers.setdefault("Content-Type", "application/json")
        parent = propagate.extract(request.headers)
        span = self.tracer.start_span(
            f"{op} {model}",
            context=parent,
            kind=SpanKind.CLIENT,
            start_time=time.time_ns(),
            attributes=_request_attributes(op, u, model, req),
        )
        ctx = trace.set_span_in_context(span, parent)
        propagate.inject(headers, context=ctx)

        meter = StreamMeter(t_start=t0)
        status = 0
        error_type = ""
        resp: web.StreamResponse | None = None
        m = self.metrics
        m.inflight.labels(u.provider).inc()
        token = otel_context.attach(ctx)
        try:
            async with self.session.post(f"{u.base}{request.path}", data=body, headers=headers) as up:
                status = up.status
                if up.status != 200 or not stream:
                    data = await up.read()
                    if up.status != 200:
                        error_type = str(up.status)
                    else:
                        _usage_into(meter, data)
                    resp = web.Response(
                        body=data,
                        status=up.status,
                        headers={"Content-Type": up.headers.get("Content-Type", "application/json")},
                    )
                    return resp
                resp = web.StreamResponse(
                    status=200,
                    headers={
                        "Content-Type": up.headers.get("Content-Type", "text/event-stream"),
                        "Cache-Control": "no-cache",
                    },
                )
                await resp.prepare(request)
                parser = SSEParser()
                client_gone = False
                # TTFC and chunk gaps are observed as they happen, not when the request ends: a
                # 30 s generation would otherwise report its TTFC 30 s late and delay alerts.
                h_gap = m.tpoc.labels(*labels)
                async for chunk in up.content.iter_any():
                    now = time.perf_counter()
                    for raw, data in parser.feed(chunk):
                        kind = meter.on_event(data, now)
                        if kind is EventKind.CONTENT:
                            if meter.content_chunks == 1:
                                span.add_event("gen_ai.first_chunk")
                                m.ttfc.labels(*labels).observe(now - t0, exemplar=_exemplar(span))
                            else:
                                h_gap.observe(meter.gaps[-1])
                        if injected and kind is EventKind.USAGE_ONLY:
                            continue
                        try:
                            await resp.write(raw)
                        except ConnectionResetError:
                            client_gone = True
                            break
                    if client_gone:
                        break
                if client_gone:
                    # Leaving the `async with` closes the upstream connection, which makes the engine
                    # abort the request instead of generating tokens nobody reads.
                    error_type = "cancelled"
                    return resp
                tail = parser.pending()
                if tail:
                    with contextlib.suppress(ConnectionResetError):
                        await resp.write(tail)
                if meter.stream_error:
                    error_type = meter.stream_error
                elif not (meter.done or meter.finish_reasons):
                    error_type = "stream_interrupted"
                with contextlib.suppress(ConnectionResetError):
                    await resp.write_eof()
                return resp
        except TimeoutError:
            error_type = "timeout"
            return await self._fail(request, resp, 504, error_type, "upstream timed out")
        except aiohttp.ServerDisconnectedError:
            error_type = "stream_interrupted" if resp is not None else "connection_error"
            return await self._fail(request, resp, 502, error_type, "upstream disconnected")
        except (aiohttp.ClientError, OSError) as e:
            error_type = "connection_error" if resp is None else "stream_interrupted"
            return await self._fail(request, resp, 502, error_type, f"upstream error: {type(e).__name__}")
        except asyncio.CancelledError:
            error_type = "cancelled"
            raise
        finally:
            otel_context.detach(token)
            m.inflight.labels(u.provider).dec()
            self._record(labels, meter, error_type, stream, span, u, model, status, wall0)

    async def _fail(self, request, resp, status, error_type, message) -> web.StreamResponse:
        payload = {"error": {"message": message, "type": error_type}}
        if resp is None:
            return web.json_response(payload, status=status)
        # Headers were already sent: report the failure inside the stream, then end it.
        with contextlib.suppress(ConnectionResetError, RuntimeError):
            await resp.write(b"data: " + orjson.dumps(payload) + b"\n\n")
            await resp.write_eof()
        return resp

    def _record(
        self,
        labels,
        meter: StreamMeter,
        error_type: str,
        stream: bool,
        span,
        u: Upstream,
        model: str,
        status: int,
        wall0: float,
    ) -> None:
        m = self.metrics
        t_end = time.perf_counter()
        duration = t_end - meter.t_start
        ex = _exemplar(span)
        m.duration.labels(*labels, error_type).observe(duration, exemplar=ex)
        ttfc = meter.ttfc
        tpot = meter.tpot()
        if not error_type:
            if stream and tpot is not None:
                m.tpot.labels(*labels).observe(tpot, exemplar=ex)
            if meter.prompt_tokens is not None:
                m.tokens.labels(*labels, "input").observe(meter.prompt_tokens)
            m.tokens.labels(*labels, "output").observe(meter.output_tokens)

        if span.is_recording():
            attrs = {
                "gen_ai.usage.output_tokens": meter.output_tokens,
                "inferobs.content_chunks": meter.content_chunks,
            }
            if meter.prompt_tokens is not None:
                attrs["gen_ai.usage.input_tokens"] = meter.prompt_tokens
            if meter.response_id:
                attrs["gen_ai.response.id"] = meter.response_id
            if meter.response_model:
                attrs["gen_ai.response.model"] = meter.response_model
            if meter.finish_reasons:
                attrs["gen_ai.response.finish_reasons"] = meter.finish_reasons
            if ttfc is not None:
                attrs["inferobs.time_to_first_chunk_s"] = round(ttfc, 6)
            if tpot is not None:
                attrs["inferobs.time_per_output_token_s"] = round(tpot, 6)
            if meter.gaps:
                attrs["inferobs.max_chunk_gap_s"] = round(max(meter.gaps), 6)
            if status:
                attrs["http.response.status_code"] = status
            if error_type:
                attrs["error.type"] = error_type
                if error_type != "cancelled":
                    span.set_status(Status(StatusCode.ERROR, error_type))
            span.set_attributes(attrs)
        span.end()

        if self.access_log:
            rec = {
                "ts": round(wall0, 6),
                "trace_id": ex["trace_id"] if ex else None,
                "provider": u.provider,
                "model": model,
                "op": labels[0],
                "stream": stream,
                "status": status,
                "error_type": error_type,
                "prompt_tokens": meter.prompt_tokens,
                "completion_tokens": meter.output_tokens,
                "chunks": meter.content_chunks,
                "duration_s": round(duration, 6),
                "ttfc_s": None if ttfc is None else round(ttfc, 6),
                "tpot_s": None if tpot is None else round(tpot, 6),
                "max_gap_s": round(max(meter.gaps), 6) if meter.gaps else None,
            }
            if self.cfg.access_log_gaps:
                rec["gaps_us"] = [round(g * 1e6) for g in meter.gaps]
            self.access_log.write(rec)


def _exemplar(span) -> dict | None:
    sc = span.get_span_context()
    return {"trace_id": format(sc.trace_id, "032x")} if sc.is_valid and sc.trace_flags.sampled else None


def _dumps(obj) -> str:
    return orjson.dumps(obj).decode()


def _usage_into(meter: StreamMeter, data: bytes) -> None:
    try:
        obj = orjson.loads(data)
    except orjson.JSONDecodeError:
        return
    if not isinstance(obj, dict):
        return
    usage = obj.get("usage") or {}
    meter.prompt_tokens = usage.get("prompt_tokens")
    meter.completion_tokens = usage.get("completion_tokens")
    meter.response_id = obj.get("id")
    meter.response_model = obj.get("model")
    meter.finish_reasons = [
        c.get("finish_reason")
        for c in obj.get("choices") or []
        if isinstance(c, dict) and c.get("finish_reason")
    ]
    meter.done = True


def _request_attributes(op: str, u: Upstream, model: str, req: dict) -> dict:
    attrs = {
        "gen_ai.operation.name": op,
        "gen_ai.provider.name": u.provider,
        "gen_ai.request.model": model,
        "server.address": u.base,
    }
    for key, attr in (
        ("max_tokens", "gen_ai.request.max_tokens"),
        ("max_completion_tokens", "gen_ai.request.max_tokens"),
        ("temperature", "gen_ai.request.temperature"),
        ("top_p", "gen_ai.request.top_p"),
    ):
        v = req.get(key)
        if isinstance(v, int | float) and not isinstance(v, bool):
            attrs[attr] = v
    attrs["inferobs.request.stream"] = bool(req.get("stream"))
    return attrs


def run(cfg: ProxyConfig, host: str, port: int) -> None:
    try:
        import uvloop  # type: ignore[import-not-found]

        uvloop.install()
    except ImportError:
        pass
    proxy = Proxy(cfg)
    log.info("proxy on %s:%d -> %s", host, port, ", ".join(f"{p}={u}" for p, u in cfg.upstreams))
    web.run_app(proxy.app(), host=host, port=port, access_log=None, print=None)
