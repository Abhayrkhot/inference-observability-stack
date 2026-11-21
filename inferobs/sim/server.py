"""OpenAI-compatible HTTP front end for the simulated engine.

Exposes ``/v1/chat/completions``, ``/v1/completions`` (streaming and not), ``/v1/models``,
``/health`` and ``/metrics`` with the same metric names, labels and default buckets as vLLM 0.34,
so the dashboards and alert rules work unchanged against either. ``/sim/fault`` switches faults.

Token accounting: prompt tokens are whitespace-separated words (the load generator builds prompts
from single-token words); each streamed chunk carries one output token.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from dataclasses import asdict

import orjson
from aiohttp import web
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, exposition

from inferobs.sim.engine import EngineHooks, SimConfig, SimEngine, SimRequest
from inferobs.vllm_buckets import VLLM_ITL, VLLM_LATENCY, VLLM_TTFT


class VllmStyleMetrics(EngineHooks):
    def __init__(self, model: str, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        ln = ("model_name", "engine")
        self.lv = (model, "0")
        self.running = Gauge("vllm:num_requests_running", "Requests in the running batch.", ln, registry=r)
        self.waiting = Gauge("vllm:num_requests_waiting", "Requests waiting to be scheduled.", ln, registry=r)
        self.kv = Gauge("vllm:kv_cache_usage_perc", "KV-cache usage, 1 = full.", ln, registry=r)
        self.prompt = Counter("vllm:prompt_tokens", "Prefill tokens processed.", ln, registry=r)
        self.gen = Counter("vllm:generation_tokens", "Generation tokens produced.", ln, registry=r)
        self.success = Counter(
            "vllm:request_success", "Finished requests.", (*ln, "finished_reason"), registry=r
        )
        self.ttft = Histogram(
            "vllm:time_to_first_token_seconds",
            "Engine-side time to first token.",
            ln,
            buckets=VLLM_TTFT,
            registry=r,
        )
        self.itl = Histogram(
            "vllm:inter_token_latency_seconds",
            "Engine-side inter-token latency.",
            ln,
            buckets=VLLM_ITL,
            registry=r,
        )
        self.e2e = Histogram(
            "vllm:e2e_request_latency_seconds",
            "Engine-side end-to-end latency.",
            ln,
            buckets=VLLM_LATENCY,
            registry=r,
        )
        self.queue = Histogram(
            "vllm:request_queue_time_seconds",
            "Time spent waiting to be scheduled.",
            ln,
            buckets=VLLM_LATENCY,
            registry=r,
        )
        self.prefill = Histogram(
            "vllm:request_prefill_time_seconds",
            "Scheduled to first token.",
            ln,
            buckets=VLLM_LATENCY,
            registry=r,
        )
        self.decode = Histogram(
            "vllm:request_decode_time_seconds",
            "First token to last token.",
            ln,
            buckets=VLLM_LATENCY,
            registry=r,
        )
        self.engine: SimEngine | None = None
        for reason in ("stop", "length", "abort"):
            self.success.labels(*self.lv, reason)

    def refresh(self) -> None:
        if self.engine is not None:
            self.running.labels(*self.lv).set(len(self.engine.running))
            self.waiting.labels(*self.lv).set(len(self.engine.waiting))
            self.kv.labels(*self.lv).set(self.engine.kv_usage)

    def on_admit(self, r: SimRequest, now: float) -> None:
        self.queue.labels(*self.lv).observe(now - r.arrival)

    def on_first_token(self, r: SimRequest, now: float) -> None:
        self.ttft.labels(*self.lv).observe(now - r.arrival)
        self.prompt.labels(*self.lv).inc(r.prompt_tokens)
        self.gen.labels(*self.lv).inc()
        if r.t_scheduled is not None:
            self.prefill.labels(*self.lv).observe(now - r.t_scheduled)

    def on_token(self, r: SimRequest, now: float, gap: float) -> None:
        self.itl.labels(*self.lv).observe(gap)
        self.gen.labels(*self.lv).inc()

    def on_finish(self, r: SimRequest, now: float, reason: str) -> None:
        self.success.labels(*self.lv, reason).inc()
        if reason != "abort":
            self.e2e.labels(*self.lv).observe(now - r.arrival)
            if r.t_first is not None:
                self.decode.labels(*self.lv).observe(now - r.t_first)


def count_prompt_tokens(req: dict, chat: bool) -> int:
    if chat:
        n = 0
        for msg in req.get("messages") or []:
            c = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(c, str):
                n += len(c.split())
            elif isinstance(c, list):
                n += sum(len(p.get("text", "").split()) for p in c if isinstance(p, dict))
        return n
    p = req.get("prompt", "")
    if isinstance(p, list):
        p = " ".join(x for x in p if isinstance(x, str))
    return len(str(p).split())


class SimServer:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.metrics = VllmStyleMetrics(cfg.model)
        self.engine = SimEngine(cfg, hooks=self.metrics)
        self.metrics.engine = self.engine
        self._rng = random.Random(cfg.seed + 1)
        self.created = int(time.time())

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.generate)
        app.router.add_post("/v1/completions", self.generate)
        app.router.add_get("/v1/models", self.models)
        app.router.add_get("/health", self.health)
        app.router.add_get("/metrics", self.metrics_endpoint)
        app.router.add_get("/sim/fault", self.get_fault)
        app.router.add_post("/sim/fault", self.set_fault)
        app.on_startup.append(self._startup)
        app.on_cleanup.append(self._cleanup)
        return app

    async def _startup(self, _app) -> None:
        self.engine.start()

    async def _cleanup(self, _app) -> None:
        await self.engine.stop()

    async def models(self, _r: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.cfg.model,
                        "object": "model",
                        "created": self.created,
                        "owned_by": "inferobs-sim",
                        "max_model_len": self.cfg.max_model_len,
                    }
                ],
            }
        )

    async def health(self, _r: web.Request) -> web.Response:
        if self.engine.faults.hang_health:
            await asyncio.sleep(3600)
        return web.Response(text="")

    async def metrics_endpoint(self, request: web.Request) -> web.Response:
        self.metrics.refresh()
        encoder, ctype = exposition.choose_encoder(request.headers.get("Accept", ""))
        return web.Response(body=encoder(self.metrics.registry), headers={"Content-Type": ctype})

    async def get_fault(self, _r: web.Request) -> web.Response:
        return web.json_response(asdict(self.engine.faults))

    async def set_fault(self, request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else {}
        f = self.engine.faults
        if body.get("reset"):
            f.__init__()  # type: ignore[misc]
        for k, v in body.items():
            if hasattr(f, k):
                setattr(f, k, type(getattr(f, k))(v))
        if "kv_capacity_tokens" in body:
            self.cfg.kv_capacity_tokens = int(body["kv_capacity_tokens"])
        self.engine._wake.set()
        return web.json_response(asdict(f))

    async def generate(self, request: web.Request) -> web.StreamResponse:
        chat = request.path.endswith("/chat/completions")
        try:
            req = orjson.loads(await request.read())
        except orjson.JSONDecodeError:
            return _error(400, "invalid JSON", "BadRequestError")
        model = req.get("model") or self.cfg.model
        if model != self.cfg.model:
            return _error(404, f"The model `{model}` does not exist.", "NotFoundError")
        f = self.engine.faults
        if f.error_rate > 0 and self._rng.random() < f.error_rate:
            return _error(f.error_status, "injected fault", "InternalServerError")
        max_tokens = int(
            req.get("max_completion_tokens") or req.get("max_tokens") or self.cfg.default_max_tokens
        )
        try:
            r = self.engine.submit(count_prompt_tokens(req, chat), max_tokens, bool(req.get("ignore_eos")))
        except ValueError as e:
            return _error(400, str(e), "BadRequestError")
        rid = ("chatcmpl-" if chat else "cmpl-") + r.id
        created = int(time.time())
        obj = "chat.completion.chunk" if chat else "text_completion"
        resp: web.StreamResponse | None = None
        try:
            if not req.get("stream"):
                reason = None
                while reason is None:
                    _, reason = await r.out.get()
                text = " ".join(f"t{i}" for i in range(r.generated))
                choice = (
                    {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": reason}
                    if chat
                    else {"index": 0, "text": text, "finish_reason": reason}
                )
                return web.json_response(
                    {
                        "id": rid,
                        "object": "chat.completion" if chat else "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [choice],
                        "usage": _usage(r),
                    }
                )
            include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
            resp = web.StreamResponse(
                headers={"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache"}
            )
            await resp.prepare(request)
            first = True
            reason = None
            while reason is None:
                i, reason = await r.out.get()
                if chat:
                    delta = {"content": f"t{i} "}
                    if first:
                        delta = {"role": "assistant", "content": f"t{i} "}
                    choice = {"index": 0, "delta": delta, "finish_reason": reason}
                else:
                    choice = {"index": 0, "text": f"t{i} ", "finish_reason": reason}
                first = False
                await resp.write(
                    _sse({"id": rid, "object": obj, "created": created, "model": model, "choices": [choice]})
                )
            if include_usage:
                await resp.write(
                    _sse(
                        {
                            "id": rid,
                            "object": obj,
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": _usage(r),
                        }
                    )
                )
            await resp.write(b"data: [DONE]\n\n")
            with contextlib.suppress(ConnectionResetError):
                await resp.write_eof()
            return resp
        except ConnectionResetError:
            # Client disconnected mid-stream; the finally block aborts the request like vLLM does.
            assert resp is not None
            return resp
        finally:
            if not r.finished:
                self.engine.cancel(r)


def _usage(r: SimRequest) -> dict:
    return {
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.generated,
        "total_tokens": r.prompt_tokens + r.generated,
    }


def _sse(obj: dict) -> bytes:
    return b"data: " + orjson.dumps(obj) + b"\n\n"


def _error(status: int, message: str, typ: str) -> web.Response:
    return web.json_response({"error": {"message": message, "type": typ, "code": status}}, status=status)


def run(cfg: SimConfig, host: str, port: int) -> None:
    with contextlib.suppress(ImportError):
        import uvloop  # type: ignore[import-not-found]

        uvloop.install()
    web.run_app(SimServer(cfg).app(), host=host, port=port, access_log=None, print=None)
