"""A continuous-batching inference engine simulator.

The scheduler mirrors the shape of vLLM's: FCFS admission while there is room in the running batch
and KV-cache budget, chunked prefill under a per-step token budget, and one decode token per running
sequence per step. Step time follows a linear cost model

    step = (base + per_seq * sequences_in_step + per_prefill_token * prefill_tokens) * slowdown

whose defaults can be fitted to a real engine (``inferobs sim-calibrate``). Faults (slowdown, error
rate, hang) can be switched on at runtime so dashboards and alerts can be exercised without a GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    model: str = "sim-llm"
    step_base_s: float = 0.006
    step_per_seq_s: float = 0.0002
    prefill_per_token_s: float = 0.00003
    max_num_seqs: int = 64
    max_batched_tokens: int = 2048
    kv_capacity_tokens: int = 65536
    max_model_len: int = 8192
    default_max_tokens: int = 128
    seed: int = 0


@dataclass
class Faults:
    slowdown: float = 1.0  # multiply every step's duration
    error_rate: float = 0.0  # fraction of requests rejected at admission
    error_status: int = 500
    hang: bool = False  # scheduler stops stepping; in-flight requests stall
    hang_health: bool = False  # /health stops answering too


@dataclass
class SimRequest:
    id: str
    prompt_tokens: int
    max_tokens: int
    target_tokens: int
    arrival: float
    out: asyncio.Queue = field(default_factory=asyncio.Queue)
    prefilled: int = 0
    generated: int = 0
    t_scheduled: float | None = None
    t_first: float | None = None
    t_last: float | None = None
    cancelled: bool = False
    finished: bool = False

    @property
    def kv_need(self) -> int:
        return self.prompt_tokens + self.max_tokens


class EngineHooks:
    """Callbacks the server uses to export metrics; all no-ops by default."""

    def on_admit(self, r: SimRequest, now: float) -> None: ...
    def on_first_token(self, r: SimRequest, now: float) -> None: ...
    def on_token(self, r: SimRequest, now: float, gap: float) -> None: ...
    def on_finish(self, r: SimRequest, now: float, reason: str) -> None: ...
    def on_step(self, engine: SimEngine, dt: float, tokens: int) -> None: ...


class SimEngine:
    def __init__(self, cfg: SimConfig, hooks: EngineHooks | None = None) -> None:
        self.cfg = cfg
        self.faults = Faults()
        self.hooks = hooks or EngineHooks()
        self.waiting: deque[SimRequest] = deque()
        self.running: list[SimRequest] = []
        self.kv_used = 0
        self._wake = asyncio.Event()
        self._ids = itertools.count()
        self._rng = random.Random(cfg.seed)
        self._task: asyncio.Task | None = None
        self.steps = 0

    # ---------- public API ----------
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def submit(self, prompt_tokens: int, max_tokens: int, ignore_eos: bool) -> SimRequest:
        prompt_tokens = max(1, prompt_tokens)
        max_tokens = max(1, max_tokens)
        if prompt_tokens + max_tokens > min(self.cfg.max_model_len, self.cfg.kv_capacity_tokens):
            raise ValueError(
                f"prompt ({prompt_tokens}) + max_tokens ({max_tokens}) exceeds the context limit "
                f"({min(self.cfg.max_model_len, self.cfg.kv_capacity_tokens)})"
            )
        target = max_tokens if ignore_eos else self._rng.randint(max(1, max_tokens // 4), max_tokens)
        loop = asyncio.get_running_loop()
        r = SimRequest(
            id=f"sim-{next(self._ids)}",
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            target_tokens=target,
            arrival=loop.time(),
        )
        self.waiting.append(r)
        self._wake.set()
        return r

    def cancel(self, r: SimRequest) -> None:
        r.cancelled = True
        self._wake.set()

    @property
    def kv_usage(self) -> float:
        return self.kv_used / self.cfg.kv_capacity_tokens

    # ---------- scheduler ----------
    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        next_t = loop.time()
        while True:
            self._reap_cancelled(loop.time())
            if not self.running and not self.waiting:
                self._wake.clear()
                await self._wake.wait()
                next_t = loop.time()
                continue
            if self.faults.hang:
                await asyncio.sleep(0.05)
                next_t = loop.time()
                continue
            now = loop.time()
            self._admit(now)
            decode = [r for r in self.running if r.prefilled >= r.prompt_tokens]
            budget = self.cfg.max_batched_tokens - len(decode)
            prefill: list[tuple[SimRequest, int]] = []
            for r in self.running:
                if r.prefilled >= r.prompt_tokens:
                    continue
                if budget <= 0:
                    break
                chunk = min(budget, r.prompt_tokens - r.prefilled)
                prefill.append((r, chunk))
                budget -= chunk
            n_prefill_tok = sum(c for _, c in prefill)
            dt = (
                self.cfg.step_base_s
                + self.cfg.step_per_seq_s * (len(decode) + len(prefill))
                + self.cfg.prefill_per_token_s * n_prefill_tok
            ) * max(self.faults.slowdown, 0.0)
            # Sleep to an absolute deadline so timer overshoot does not accumulate across steps.
            next_t = max(next_t + dt, loop.time())
            await asyncio.sleep(max(0.0, next_t - loop.time()))
            if self.faults.hang:
                continue
            now = loop.time()
            self.steps += 1
            emitted = 0
            for r in decode:
                if not r.cancelled:
                    self._emit(r, now)
                    emitted += 1
            for r, c in prefill:
                r.prefilled += c
                if r.prefilled >= r.prompt_tokens and not r.cancelled:
                    self._emit(r, now)  # the prefill step produces the first token
                    emitted += 1
            self.hooks.on_step(self, dt, emitted + n_prefill_tok)
            self._reap_finished(now)

    def _admit(self, now: float) -> None:
        while self.waiting and len(self.running) < self.cfg.max_num_seqs:
            r = self.waiting[0]
            if r.cancelled:
                self.waiting.popleft()
                continue
            if self.kv_used + r.kv_need > self.cfg.kv_capacity_tokens:
                break
            self.waiting.popleft()
            self.kv_used += r.kv_need
            r.t_scheduled = now
            self.running.append(r)
            self.hooks.on_admit(r, now)

    def _emit(self, r: SimRequest, now: float) -> None:
        r.generated += 1
        if r.t_first is None:
            r.t_first = now
            self.hooks.on_first_token(r, now)
        else:
            self.hooks.on_token(r, now, now - (r.t_last or now))
        r.t_last = now
        reason = None
        if r.generated >= r.target_tokens:
            reason = "length" if r.generated >= r.max_tokens else "stop"
            r.finished = True
        r.out.put_nowait((r.generated, reason))

    def _reap_finished(self, now: float) -> None:
        keep = []
        for r in self.running:
            if r.finished:
                self.kv_used -= r.kv_need
                self.hooks.on_finish(r, now, "length" if r.generated >= r.max_tokens else "stop")
            else:
                keep.append(r)
        self.running = keep

    def _reap_cancelled(self, now: float) -> None:
        if any(r.cancelled for r in self.running):
            keep = []
            for r in self.running:
                if r.cancelled and not r.finished:
                    self.kv_used -= r.kv_need
                    self.hooks.on_finish(r, now, "abort")
                else:
                    keep.append(r)
            self.running = keep
