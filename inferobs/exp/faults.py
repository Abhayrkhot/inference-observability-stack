"""Which signal notices which fault, and how fast?

A steady background load runs through the faults overlay (proxy -> toxiproxy -> engine) with the
compressed-window SLO rules loaded. Each scenario injects one fault, waits until its expected alerts
fire (or a time limit), reverts it and waits for the windows to clear. Detection time is read from
Prometheus' ALERTS series (first sample in state "firing" after injection) and from the alert sink
(when Alertmanager's notification arrived).

Scenarios:
  engine_crash    docker kill the engine container, restart it afterwards
  engine_hang     docker pause the engine (frozen process: connections open, nothing answers)
  net_latency     +1.5 s on every byte from the engine (toxiproxy); the engine's own TTFT is unchanged
  stream_cut      toxiproxy closes engine connections after 4 KB: streams end mid-generation
  overload        an extra open-loop load of several times the background rate
  slow_decode     sim: slowdown x4; vLLM: a co-located GPU job contends for the GPU
"""

from __future__ import annotations

import contextlib
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field

from inferobs.exp.common import (
    Prom,
    background,
    docker,
    http_json,
    loadavg,
    loadgen_cmd,
    results_dir,
    runs_dir,
    wait_http,
    write_csv,
    write_meta,
)

TOXI = "http://127.0.0.1:28474"
SINK = "http://127.0.0.1:28095"

# Alerts each scenario is expected to raise; the run ends early once all have fired.
EXPECTED = {
    "engine_crash": [
        "SLOErrorBudgetBurnFast:availability",
        "InferenceUpstreamUnhealthy",
        "InferenceEngineScrapeDown",
    ],
    "engine_hang": ["InferenceUpstreamUnhealthy", "InferenceEngineScrapeDown"],
    "net_latency": ["SLOErrorBudgetBurnFast:ttfc", "CallerTTFCHigh"],
    "stream_cut": ["SLOErrorBudgetBurnFast:availability"],
    "overload": ["SLOErrorBudgetBurnFast:ttfc", "EngineQueueBacklog"],
    "slow_decode": ["SLOErrorBudgetBurnFast:tpot"],
}


def alert_key(labels: dict) -> str:
    name = labels["alertname"]
    if name.startswith("SLOErrorBudgetBurn"):
        return f"{name}:{labels.get('slo')}" + (
            "" if labels.get("long_window") in ("5m", "1h") else f"@{labels.get('long_window')}"
        )
    return name


@dataclass
class Ctx:
    engine_kind: str  # "sim" or "vllm"
    engine_container: str
    engine_url: str  # direct (host port), for sim fault API and health
    proxy_url: str
    model: str
    rate: float
    prompt: str
    output: str
    run_dir: object
    gpu_burner: str = field(default="")
    extra: list = field(default_factory=list)


def _toxic(name: str, typ: str, attrs: dict, stream: str = "downstream") -> None:
    http_json(
        "POST",
        f"{TOXI}/proxies/engine/toxics",
        {"name": name, "type": typ, "stream": stream, "toxicity": 1.0, "attributes": attrs},
    )


def _untoxic(name: str) -> None:
    with contextlib.suppress(Exception):
        http_json("DELETE", f"{TOXI}/proxies/engine/toxics/{name}")


def inject(ctx: Ctx, scenario: str) -> None:
    if scenario == "engine_crash":
        docker("kill", ctx.engine_container)
    elif scenario == "engine_hang":
        docker("pause", ctx.engine_container)
    elif scenario == "net_latency":
        _toxic("lat", "latency", {"latency": 1500, "jitter": 0})
    elif scenario == "stream_cut":
        _toxic("cut", "limit_data", {"bytes": 4096})
    elif scenario == "overload":
        cmd = loadgen_cmd(
            ctx.proxy_url,
            ctx.model,
            rate=ctx.rate * 4,
            duration=600,
            prompt=ctx.prompt,
            output=ctx.output,
            seed=99,
        )
        log = (ctx.run_dir / f"overload-{int(time.time())}.log").open("w")
        ctx.extra.append(
            subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        )
    elif scenario == "slow_decode":
        if ctx.engine_kind == "sim":
            http_json("POST", f"{ctx.engine_url}/sim/fault", {"slowdown": 4.0})
        else:
            docker(
                "run",
                "-d",
                "--rm",
                "--name",
                "inferobs-gpu-burner",
                "--gpus",
                "all",
                "--entrypoint",
                "python3",
                ctx.gpu_burner,
                "-c",
                "import torch\na=torch.randn(4096,4096,device='cuda',dtype=torch.half)\n"
                "while True:\n  b=a@a\n  torch.cuda.synchronize()",
            )
    else:
        raise ValueError(scenario)


def revert(ctx: Ctx, scenario: str) -> float | None:
    """Undo the fault; for a crash, return seconds until the engine served health checks again."""
    if scenario == "engine_crash":
        t = time.time()
        docker("start", ctx.engine_container)
        wait_http(f"{ctx.engine_url}/health", 900)
        return time.time() - t
    if scenario == "engine_hang":
        docker("unpause", ctx.engine_container)
    elif scenario == "net_latency":
        _untoxic("lat")
    elif scenario == "stream_cut":
        _untoxic("cut")
    elif scenario == "overload":
        import os
        import signal

        for p in ctx.extra:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
                p.wait(30)
        ctx.extra.clear()
    elif scenario == "slow_decode":
        if ctx.engine_kind == "sim":
            http_json("POST", f"{ctx.engine_url}/sim/fault", {"reset": True})
        else:
            docker("rm", "-f", "inferobs-gpu-burner", check=False)
    return None


def first_firing(prom: Prom, start: float, end: float) -> dict[str, float]:
    """First time (unix) each alert was firing in [start, end], from the ALERTS series."""
    out: dict[str, float] = {}
    for series in prom.range('ALERTS{alertstate="firing"}', start, end, step=1.0):
        k = alert_key(series["metric"])
        t = float(series["values"][0][0])
        out[k] = min(out.get(k, t), t)
    return out


def first_notified(start: float) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        recs = http_json("GET", f"{SINK}/alerts")
    except Exception:
        return out
    for r in recs:
        if r["status"] == "firing" and r["received_unix"] >= start:
            k = alert_key(r["labels"])
            out[k] = min(out.get(k, r["received_unix"]), r["received_unix"])
    return out


def active_alerts(prom: Prom) -> list[str]:
    return sorted({alert_key(r["metric"]) for r in prom.query("ALERTS")})


def wait_quiet(prom: Prom, max_s: float, ignore: tuple[str, ...] = ()) -> float:
    t0 = time.time()
    while time.time() - t0 < max_s:
        act = [a for a in active_alerts(prom) if not a.startswith(ignore)]
        if not act:
            return time.time() - t0
        time.sleep(5)
    return time.time() - t0


def run(
    engine_kind: str,
    engine_container: str,
    engine_url: str,
    proxy_url: str,
    model: str,
    rate: float,
    prompt: str,
    output: str,
    scenarios: list[str],
    repeats: int,
    max_fault_s: float,
    cooldown_s: float,
    warmup_s: float,
    prom: Prom,
    gpu_burner: str = "vllm/vllm-openai:v0.34.0",
    tag: str | None = None,
    label: str = "faults",
) -> None:
    rd = runs_dir(label)
    ctx = Ctx(
        engine_kind, engine_container, engine_url, proxy_url, model, rate, prompt, output, rd, gpu_burner
    )
    for s in scenarios:
        if s not in EXPECTED:
            raise SystemExit(f"unknown scenario {s}")
    for name in ("lat", "cut"):
        _untoxic(name)
    with contextlib.suppress(Exception):
        http_json("DELETE", f"{SINK}/alerts")
    bg = loadgen_cmd(
        proxy_url, model, rate=rate, duration=86400, prompt=prompt, output=output, seed=7, timeout=120
    )
    rows: list[dict] = []
    with background(bg, rd / "background-load.log"):
        print(f"[{label}] warming up {warmup_s:.0f}s", flush=True)
        time.sleep(warmup_s)
        for rep in range(repeats):
            for scenario in scenarios:
                quiet = wait_quiet(prom, cooldown_s * 2)
                t0 = time.time()
                inject(ctx, scenario)
                print(f"[{label}] rep {rep} {scenario}: injected", flush=True)
                expected = EXPECTED[scenario]
                fired: dict[str, float] = {}
                while time.time() - t0 < max_fault_s:
                    time.sleep(2)
                    fired = first_firing(prom, t0, time.time())
                    if all(e in fired for e in expected):
                        break
                t_fault_end = time.time()
                recovery_s = revert(ctx, scenario)
                t_rev = time.time()
                time.sleep(cooldown_s)
                fired = first_firing(prom, t0, t_fault_end + 5)
                notified = first_notified(t0)
                for key in sorted(set(fired) | set(expected)):
                    rows.append(
                        {
                            "scenario": scenario,
                            "repeat": rep,
                            "alert": key,
                            "expected": key in expected,
                            "detected": key in fired,
                            "detect_s": round(fired[key] - t0, 1) if key in fired else None,
                            "notified_s": round(notified[key] - t0, 1) if key in notified else None,
                            "fault_duration_s": round(t_fault_end - t0, 1),
                            "engine_recovery_s": round(recovery_s, 1) if recovery_s is not None else None,
                            "quiet_wait_s": round(quiet, 1),
                            "loadavg": loadavg(),
                            "injected_unix": round(t0, 1),
                            "reverted_unix": round(t_rev, 1),
                        }
                    )
                summary = {k: round(v - t0, 1) for k, v in sorted(fired.items(), key=lambda kv: kv[1])}
                print(f"[{label}] rep {rep} {scenario}: fired {json.dumps(summary)}", flush=True)
    out = results_dir(tag) / f"{label}.csv"
    write_csv(out, rows)
    write_meta(
        out,
        engine=engine_kind,
        model=model,
        rate=rate,
        prompt=prompt,
        output=output,
        scenarios=scenarios,
        repeats=repeats,
        max_fault_s=max_fault_s,
        cooldown_s=cooldown_s,
        warmup_s=warmup_s,
        run_dir=rd.name,
        rules="deploy/lab-rules/slo-fast.yaml (window_scale 1/12)",
        scrape_interval="5s",
        evaluation_interval="5s",
        argv=" ".join(map(shlex.quote, sys.argv)),
    )
