import asyncio

import pytest

from inferobs.loadgen.runner import Dist, LoadConfig, Stage, parse_stages, run_load


def test_dist_parsing_and_bounds():
    import random

    rng = random.Random(0)
    d = Dist.parse("lognormal:100:0.5:10:300")
    xs = [d.sample(rng) for _ in range(5000)]
    assert min(xs) >= 10 and max(xs) <= 300
    assert sum(xs) / len(xs) == pytest.approx(100, rel=0.08)
    assert Dist.parse("fixed:7").sample(rng) == 7
    assert parse_stages(None, None, 1, "r=2:10,c=4:5") == [Stage(10, rate=2), Stage(5, concurrency=4)]


async def test_open_loop_run_against_sim(sim):
    cfg = LoadConfig(
        url=sim.url, model="sim-llm", stages=[Stage(1.5, rate=20)], prompt="fixed:32", output="fixed:16"
    )
    results, summ = await run_load(cfg)
    assert summ["errors"] == 0
    assert 15 <= summ["requests"] <= 50
    assert all(r.completion_tokens == 16 and r.prompt_tokens == 32 for r in results)
    assert all(len(r.gaps) == 15 for r in results)
    assert 0 < summ["ttft_s"]["p50"] < summ["e2e_s"]["p50"]


async def test_latency_counts_from_scheduled_arrival(sim):
    """A 0.5 s engine stall must show up in TTFT for requests that arrived during it."""
    cfg = LoadConfig(
        url=sim.url, model="sim-llm", stages=[Stage(1.2, rate=25)], prompt="fixed:8", output="fixed:4"
    )

    async def stall():
        await asyncio.sleep(0.3)
        sim.engine.faults.hang = True
        await asyncio.sleep(0.5)
        sim.engine.faults.hang = False

    (results, _), _ = await asyncio.gather(run_load(cfg), stall())
    hit = [r for r in results if 0.3 <= r.t_sched <= 0.5]
    assert hit and min(r.ttft_s for r in hit) > 0.25
