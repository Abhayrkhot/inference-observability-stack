"""What does metering cost? Direct-to-engine vs through the proxy, with and without tracing.

Two proxy processes are started next to the stack (same host, host networking) so their CPU time can
be read from /proc: one without tracing and one exporting every span to Tempo. Each closed-loop
concurrency level runs against all three targets in a rotated order per repeat, so slow drift on a
shared machine does not favour one target.
"""

from __future__ import annotations

import statistics
import sys
import time

from inferobs.exp.common import (
    background,
    cpu_seconds,
    loadavg,
    results_dir,
    run_loadgen,
    runs_dir,
    wait_http,
    write_csv,
    write_meta,
)


def run(
    engine_url: str,
    model: str,
    provider: str,
    levels: list[int],
    duration: float,
    repeats: int,
    prompt: str,
    output: str,
    otlp: str,
    port_base: int = 28010,
    tag: str | None = None,
    label: str = "overhead",
) -> None:
    rd = runs_dir(label)
    py = sys.executable
    plain = (
        f"{py} -m inferobs proxy --upstream {provider}={engine_url} --port {port_base} --health-interval 5"
    )
    traced = (
        f"{py} -m inferobs proxy --upstream {provider}={engine_url} --port {port_base + 1} "
        f"--health-interval 5 --otlp-endpoint {otlp} --service-name inferobs-proxy-exp"
    )
    targets = {
        "direct": engine_url,
        "proxy": f"http://127.0.0.1:{port_base}",
        "proxy+tracing": f"http://127.0.0.1:{port_base + 1}",
    }
    rows: list[dict] = []
    with background(plain, rd / "proxy.log") as p_plain, background(traced, rd / "proxy-traced.log") as p_tr:
        pids = {"proxy": p_plain.pid, "proxy+tracing": p_tr.pid}
        for name in ("proxy", "proxy+tracing"):
            wait_http(f"{targets[name]}/health", 60)
        # Warm the engine (CUDA graphs, allocator) and the proxies' connection pools.
        for name, url in targets.items():
            run_loadgen(
                url,
                model,
                rd,
                f"warmup-{name.replace('+', '_')}",
                concurrency=8,
                duration=15,
                prompt=prompt,
                output=output,
            )
        names = list(targets)
        for rep in range(repeats):
            order = names[rep % len(names) :] + names[: rep % len(names)]
            for level in levels:
                for name in order:
                    pid = pids.get(name)
                    c0 = cpu_seconds(pid) if pid else None
                    t0 = time.time()
                    s = run_loadgen(
                        targets[name],
                        model,
                        rd,
                        f"{name.replace('+', '_')}-c{level}-r{rep}",
                        concurrency=level,
                        duration=duration,
                        prompt=prompt,
                        output=output,
                        seed=rep,
                    )
                    wall = time.time() - t0
                    cpu = (cpu_seconds(pid) - c0) if pid else None
                    ok = s["requests"] - s["errors"]
                    rows.append(
                        {
                            "target": name,
                            "concurrency": level,
                            "repeat": rep,
                            "requests": s["requests"],
                            "errors": s["errors"],
                            "output_tok_per_s": s["output_tok_per_s"],
                            "req_per_s": s["req_per_s"],
                            "ttft_p50_s": s["ttft_s"]["p50"],
                            "ttft_p99_s": s["ttft_s"]["p99"],
                            "tpot_p50_s": s["tpot_s"]["p50"],
                            "itl_p99_s": s["itl_s"]["p99"],
                            "e2e_p50_s": s["e2e_s"]["p50"],
                            "e2e_p99_s": s["e2e_s"]["p99"],
                            "proxy_cpu_s": cpu,
                            "proxy_cpu_util": (cpu / wall) if cpu is not None else None,
                            "proxy_cpu_ms_per_req": (cpu * 1e3 / ok) if cpu is not None and ok else None,
                            "proxy_cpu_us_per_token": (cpu * 1e6 / s["output_tokens"])
                            if cpu is not None and s["output_tokens"]
                            else None,
                            "loadavg": loadavg(),
                        }
                    )
                    print(
                        f"[{label}] rep {rep} c={level} {name}: ttft p50 {s['ttft_s']['p50'] * 1e3:.1f} ms "
                        f"tok/s {s['output_tok_per_s']:.0f}",
                        flush=True,
                    )
    out = results_dir(tag) / f"{label}_runs.csv"
    write_csv(out, rows)
    write_csv(results_dir(tag) / f"{label}.csv", summarize(rows))
    write_meta(
        results_dir(tag) / f"{label}.csv",
        engine=engine_url,
        model=model,
        levels=levels,
        duration_s=duration,
        repeats=repeats,
        prompt=prompt,
        output=output,
        run_dir=str(rd.name),
    )


def summarize(rows: list[dict]) -> list[dict]:
    """Median over repeats per (target, level), plus the proxy's delta against direct."""
    out = []
    levels = sorted({r["concurrency"] for r in rows})
    for level in levels:
        base = {}
        for target in ("direct", "proxy", "proxy+tracing"):
            rs = [r for r in rows if r["target"] == target and r["concurrency"] == level]
            if not rs:
                continue
            med = {
                k: statistics.median([r[k] for r in rs if r[k] is not None])
                for k in (
                    "output_tok_per_s",
                    "ttft_p50_s",
                    "ttft_p99_s",
                    "tpot_p50_s",
                    "itl_p99_s",
                    "e2e_p50_s",
                    "e2e_p99_s",
                )
            }
            for k in ("proxy_cpu_util", "proxy_cpu_ms_per_req", "proxy_cpu_us_per_token"):
                vals = [r[k] for r in rs if r[k] is not None]
                med[k] = statistics.median(vals) if vals else None
            if target == "direct":
                base = med
            row = {
                "concurrency": level,
                "target": target,
                "repeats": len(rs),
                "errors": sum(r["errors"] for r in rs),
                **med,
            }
            if base and target != "direct":
                row["ttft_p50_delta_ms"] = (med["ttft_p50_s"] - base["ttft_p50_s"]) * 1e3
                row["ttft_p99_delta_ms"] = (med["ttft_p99_s"] - base["ttft_p99_s"]) * 1e3
                row["tpot_p50_delta_us"] = (med["tpot_p50_s"] - base["tpot_p50_s"]) * 1e6
                row["e2e_p50_delta_ms"] = (med["e2e_p50_s"] - base["e2e_p50_s"]) * 1e3
                row["throughput_delta_pct"] = (med["output_tok_per_s"] / base["output_tok_per_s"] - 1) * 100
            out.append(row)
    return out
