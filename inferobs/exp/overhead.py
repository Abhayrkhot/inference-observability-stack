"""What does metering cost? Direct-to-engine vs through the proxy, with and without tracing.

Two proxy processes are started next to the stack (same host, host networking) so their CPU time can
be read from /proc: one without tracing and one exporting every span to Tempo. Each closed-loop
concurrency level runs against all three targets in a rotated order per repeat, so slow drift on a
shared machine does not favour one target.
"""

from __future__ import annotations

import json
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


TARGETS = ("direct", "proxy", "proxy+tracing")


def _boot_delta(
    a: list[float], b: list[float], q: float, n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap interval for quantile(b) - quantile(a)."""
    import random

    from inferobs.analysis.stats import quantile

    rng = random.Random(seed)
    ds = sorted(
        quantile([rng.choice(b) for _ in b], q) - quantile([rng.choice(a) for _ in a], q) for _ in range(n)
    )
    return ds[int(0.025 * n)], ds[int(0.975 * n)]


def run_interleaved(
    engine_url: str,
    model: str,
    provider: str,
    levels: list[int],
    duration: float,
    prompt: str,
    output: str,
    otlp: str,
    port_base: int = 28010,
    tag: str | None = None,
    label: str = "overhead",
    settle_s: float = 10.0,
) -> None:
    """Paired overhead: every run rotates requests through direct, proxy and proxy+tracing.

    On a laptop GPU whose clocks drift with power and temperature, separate runs per target differ by
    more than the proxy costs. Interleaving request by request gives every target the same engine state,
    so the per-target difference isolates the proxy hop.
    """
    rd = runs_dir(label)
    py = sys.executable
    plain = (
        f"{py} -m inferobs proxy --upstream {provider}={engine_url} --port {port_base} --health-interval 5"
    )
    traced = (
        f"{py} -m inferobs proxy --upstream {provider}={engine_url} --port {port_base + 1} "
        f"--health-interval 5 --otlp-endpoint {otlp} --service-name inferobs-proxy-exp"
    )
    urls = [engine_url, f"http://127.0.0.1:{port_base}", f"http://127.0.0.1:{port_base + 1}"]
    runs: dict[str, dict] = {}
    with background(plain, rd / "proxy.log") as p_plain, background(traced, rd / "proxy-traced.log") as p_tr:
        pids = {"proxy": p_plain.pid, "proxy+tracing": p_tr.pid}
        for u in urls[1:]:
            wait_http(f"{u}/health", 60)
        run_loadgen(
            ",".join(urls), model, rd, "warmup", concurrency=8, duration=20, prompt=prompt, output=output
        )
        for level in levels:
            c0 = {k: cpu_seconds(p) for k, p in pids.items()}
            t0 = time.time()
            s = run_loadgen(
                ",".join(urls),
                model,
                rd,
                f"c{level}",
                concurrency=level,
                duration=duration,
                prompt=prompt,
                output=output,
                seed=level,
            )
            wall = time.time() - t0
            cpu = {k: cpu_seconds(p) - c0[k] for k, p in pids.items()}
            runs[str(level)] = _cpu_fields(rd / f"c{level}.csv", cpu, wall, s)
    (rd / "levels.json").write_text(json.dumps(runs, indent=2) + "\n")
    _write_interleaved(
        rd,
        tag,
        label,
        settle_s,
        engine=engine_url,
        model=model,
        levels=levels,
        duration_s=duration,
        prompt=prompt,
        output=output,
    )


def _cpu_fields(path, cpu: dict, wall: float, summary: dict) -> dict:
    """Per-proxy CPU over the whole run (all requests), plus run-level stats."""
    import csv as _csv

    with open(path) as f:
        reqs = [r for r in _csv.DictReader(f) if not r["error"]]
    out = {
        "errors": summary["errors"],
        "run_output_tok_per_s": summary["output_tok_per_s"],
        "loadavg": loadavg(),
    }
    for i, t in enumerate(TARGETS):
        if t == "direct":
            continue
        rs = [r for r in reqs if int(r["target"]) == i]
        toks = sum(int(r["completion_tokens"] or 0) for r in rs)
        out[t] = {
            "proxy_cpu_util": cpu[t] / wall,
            "proxy_cpu_us_per_token": cpu[t] * 1e6 / toks if toks else None,
            "proxy_cpu_ms_per_req": cpu[t] * 1e3 / len(rs) if rs else None,
        }
    return out


def analyze_interleaved(rd, settle_s: float) -> list[dict]:
    """Latency per target from each level's raw CSV, skipping requests scheduled in the first settle_s.

    Closed-loop workers all start at once, so the first wave of requests queues behind a synchronized
    prefill burst; which target lands in a later prefill wave is decided by sub-millisecond arrival
    order, not by the proxy. Skipping the start measures steady state.
    """
    import csv as _csv
    from pathlib import Path

    from inferobs.analysis.stats import quantile

    rd = Path(rd)
    runs = json.loads((rd / "levels.json").read_text())
    rows: list[dict] = []
    for level in sorted(runs, key=int):
        info = runs[level]
        with (rd / f"c{level}.csv").open() as f:
            reqs = [r for r in _csv.DictReader(f) if not r["error"] and float(r["t_sched"]) >= settle_s]
        by = {t: [r for r in reqs if int(r["target"]) == i] for i, t in enumerate(TARGETS)}
        ttft = {t: [float(r["ttft_s"]) for r in rs if r["ttft_s"]] for t, rs in by.items()}
        tpot = {t: [float(r["tpot_s"]) for r in rs if r["tpot_s"]] for t, rs in by.items()}
        e2e = {t: [float(r["e2e_s"]) for r in rs] for t, rs in by.items()}
        for t in TARGETS:
            row = {
                "concurrency": int(level),
                "target": t,
                "requests": len(by[t]),
                "errors": info["errors"] if t == "direct" else None,
                "ttft_p50_s": quantile(ttft[t], 0.5),
                "ttft_p99_s": quantile(ttft[t], 0.99),
                "tpot_p50_s": quantile(tpot[t], 0.5),
                "e2e_p50_s": quantile(e2e[t], 0.5),
                "e2e_p99_s": quantile(e2e[t], 0.99),
                "run_output_tok_per_s": info["run_output_tok_per_s"],
                "loadavg": info["loadavg"],
            }
            if t != "direct":
                for q, name in ((0.5, "p50"), (0.99, "p99")):
                    lo, hi = _boot_delta(ttft["direct"], ttft[t], q)
                    row[f"ttft_{name}_delta_ms"] = (quantile(ttft[t], q) - quantile(ttft["direct"], q)) * 1e3
                    row[f"ttft_{name}_delta_ci_ms"] = f"{lo * 1e3:.2f}..{hi * 1e3:.2f}"
                row["tpot_p50_delta_us"] = (quantile(tpot[t], 0.5) - quantile(tpot["direct"], 0.5)) * 1e6
                row["e2e_p50_delta_ms"] = (quantile(e2e[t], 0.5) - quantile(e2e["direct"], 0.5)) * 1e3
                row.update(info[t])
            rows.append(row)
    return rows


def _write_interleaved(rd, tag, label: str, settle_s: float, **meta) -> None:
    rows = analyze_interleaved(rd, settle_s)
    for r in rows:
        if r["target"] == "proxy":
            print(
                f"[{label}] c={r['concurrency']}: proxy ttft p50 delta {r['ttft_p50_delta_ms']:.2f} ms "
                f"({r['ttft_p50_delta_ci_ms']}), p99 delta {r['ttft_p99_delta_ms']:.2f} ms "
                f"({r['ttft_p99_delta_ci_ms']})",
                flush=True,
            )
    out = results_dir(tag) / f"{label}.csv"
    write_csv(out, rows)
    write_meta(
        out,
        method="interleaved: requests rotate direct, proxy, proxy+tracing within each run",
        settle_s=settle_s,
        run_dir=rd.name,
        **meta,
    )


def reanalyze(run_dir: str, tag: str | None, label: str, settle_s: float) -> None:
    """Rebuild results/<tag>/<label>.csv from a run directory, keeping the original run's metadata."""
    from pathlib import Path

    rd = Path(run_dir)
    old = results_dir(tag) / f"{label}.meta.json"
    meta = json.loads(old.read_text()) if old.exists() else {}
    keep = {k: meta[k] for k in ("engine", "model", "levels", "duration_s", "prompt", "output") if k in meta}
    git_sha = meta.get("git_sha")
    _write_interleaved(rd, tag, label, settle_s, **keep)
    if git_sha:  # the measurement ran at this commit; the reanalysis only filtered its raw data
        new = json.loads(old.read_text())
        new["measured_at_git_sha"] = git_sha
        new["reanalyzed_at_git_sha"] = new.pop("git_sha")
        new["git_sha"] = git_sha
        old.write_text(json.dumps(new, indent=2) + "\n")
