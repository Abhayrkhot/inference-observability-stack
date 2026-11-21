"""Where does the engine stop meeting the SLOs? An open-loop rate ramp through the proxy.

Each stage reports the caller's percentiles (from the load generator) next to what the engine's
own metrics showed over the same window (queue depth, KV usage, throughput), so the dashboard's
saturation signals can be read against the user-visible effect.
"""

from __future__ import annotations

import csv
import time

from inferobs.analysis.stats import quantile
from inferobs.exp.common import Prom, results_dir, run_loadgen, runs_dir, write_csv, write_meta


def run(
    proxy_url: str,
    model: str,
    rates: list[float],
    stage_s: float,
    prompt: str,
    output: str,
    prom: Prom,
    ttft_slo: float = 1.0,
    tpot_slo: float = 0.05,
    tag: str | None = None,
    label: str = "saturation",
    settle_s: float = 30.0,
) -> None:
    rd = runs_dir(label)
    stages = ",".join(f"r={r:g}:{stage_s:g}" for r in rates)
    s = run_loadgen(proxy_url, model, rd, "ramp", stages=stages, prompt=prompt, output=output, timeout=600)
    time.sleep(12)
    t_start = s["started_unix"]
    with (rd / "ramp.csv").open() as f:
        reqs = list(csv.DictReader(f))
    rows = []
    for i, rate in enumerate(rates):
        a, b = t_start + i * stage_s, t_start + (i + 1) * stage_s
        # Requests that *arrived* in the stage, skipping the first settle_s while queues adjust.
        rs = [r for r in reqs if int(r["stage"]) == i and float(r["t_sched"]) >= i * stage_s + settle_s]
        ok = [r for r in rs if not r["error"]]
        ttft = [float(r["ttft_s"]) for r in ok if r["ttft_s"]]
        tpot = [float(r["tpot_s"]) for r in ok if r["tpot_s"]]
        toks = sum(int(r["completion_tokens"] or 0) for r in ok)
        w = max(1, round(b - a - settle_s))
        lo = a + settle_s
        row = {
            "offered_rps": rate,
            "requests": len(rs),
            "errors": len(rs) - len(ok),
            "ttft_p50_s": quantile(ttft, 0.5),
            "ttft_p99_s": quantile(ttft, 0.99),
            "tpot_p50_s": quantile(tpot, 0.5),
            "tpot_p99_s": quantile(tpot, 0.99),
            "goodput_tok_per_s": toks / (stage_s - settle_s),
            "engine_running_avg": prom.scalar(f"avg_over_time(sum(vllm:num_requests_running)[{w}s:1s])", b),
            "engine_waiting_avg": prom.scalar(f"avg_over_time(sum(vllm:num_requests_waiting)[{w}s:1s])", b),
            "engine_kv_max": prom.scalar(f"max_over_time(max(vllm:kv_cache_usage_perc)[{w}s:1s])", b),
            "engine_gen_tok_per_s": prom.scalar(
                f"sum(increase(vllm:generation_tokens_total[{w}s])) / {w}", b
            ),
            "window_start_unix": round(lo, 1),
        }
        row["meets_slo"] = bool(
            ttft
            and tpot
            and row["ttft_p99_s"] <= ttft_slo
            and row["tpot_p99_s"] <= tpot_slo
            and not row["errors"]
        )
        rows.append(row)
        print(
            f"[{label}] {rate:g} rps: ttft p99 {row['ttft_p99_s']:.3f}s "
            f"tpot p99 {row['tpot_p99_s'] * 1e3:.1f}ms waiting {row['engine_waiting_avg']}",
            flush=True,
        )
    out = results_dir(tag) / f"{label}.csv"
    write_csv(out, rows)
    write_meta(
        out,
        proxy=proxy_url,
        model=model,
        rates=rates,
        stage_s=stage_s,
        settle_s=settle_s,
        prompt=prompt,
        output=output,
        ttft_slo_s=ttft_slo,
        tpot_slo_s=tpot_slo,
        run_dir=rd.name,
    )
