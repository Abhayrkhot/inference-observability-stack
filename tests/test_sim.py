import asyncio
import itertools

import pytest

from inferobs.sim.engine import SimConfig, SimEngine


async def drain(r):
    times = []
    loop = asyncio.get_running_loop()
    while True:
        _, reason = await r.out.get()
        times.append(loop.time())
        if reason:
            return times


async def test_single_request_follows_cost_model():
    cfg = SimConfig(step_base_s=0.01, step_per_seq_s=0.001, prefill_per_token_s=0.0001)
    e = SimEngine(cfg)
    e.start()
    try:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        r = e.submit(prompt_tokens=200, max_tokens=10, ignore_eos=True)
        times = await drain(r)
        ttft = times[0] - t0
        # prefill step: 10 ms + 1 ms + 200 * 0.1 ms = 31 ms; decode steps: 11 ms each
        assert ttft == pytest.approx(0.031, abs=0.006)
        gaps = [b - a for a, b in itertools.pairwise(times)]
        assert len(gaps) == 9
        assert sum(gaps) / len(gaps) == pytest.approx(0.011, abs=0.002)
        assert r.generated == 10
    finally:
        await e.stop()


async def test_kv_budget_limits_admission_and_queues_the_rest():
    cfg = SimConfig(
        step_base_s=0.002,
        step_per_seq_s=0.0,
        prefill_per_token_s=0.0,
        kv_capacity_tokens=300,
        max_model_len=300,
    )
    e = SimEngine(cfg)
    e.start()
    try:
        reqs = [e.submit(prompt_tokens=50, max_tokens=50, ignore_eos=True) for _ in range(5)]
        await asyncio.sleep(0.01)
        assert len(e.running) == 3 and len(e.waiting) == 2  # 3 x 100 tokens fill the 300-token cache
        await asyncio.gather(*(drain(r) for r in reqs))
        assert e.kv_used == 0 and not e.running and not e.waiting
    finally:
        await e.stop()


async def test_slowdown_fault_stretches_steps():
    cfg = SimConfig(step_base_s=0.004, step_per_seq_s=0.0, prefill_per_token_s=0.0)
    e = SimEngine(cfg)
    e.start()
    try:
        e.faults.slowdown = 3.0
        r = e.submit(prompt_tokens=1, max_tokens=6, ignore_eos=True)
        times = await drain(r)
        gaps = [b - a for a, b in itertools.pairwise(times)]
        assert sum(gaps) / len(gaps) == pytest.approx(0.012, abs=0.003)
    finally:
        await e.stop()


async def test_oversized_request_is_rejected():
    e = SimEngine(SimConfig(kv_capacity_tokens=100, max_model_len=100))
    with pytest.raises(ValueError):
        e.submit(prompt_tokens=90, max_tokens=20, ignore_eos=True)
