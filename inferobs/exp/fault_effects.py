"""What did each injected fault do to callers? Before/during comparison for every fault window.

Reads results/<tag>/faults.csv for the injection and revert times, then queries Prometheus for the
caller-side (proxy) and engine-side (vLLM) signals over the fault window and over the 2 minutes
before it. Writes results/<tag>/fault_effects.csv. Run while Prometheus still holds the data.
"""

from __future__ import annotations

import csv

from inferobs.exp.common import Prom, results_dir, write_csv, write_meta

SETTLE_S = 30  # skip the first seconds of the fault window
BEFORE_S = 60


def _q(prom: Prom, expr: str, end: float) -> float | None:
    return prom.scalar(expr, end)


def _window(prom: Prom, start: float, end: float) -> dict:
    w = max(10, round(end - start))
    ttfc = "gen_ai_client_operation_time_to_first_chunk_seconds_bucket"
    gap = "gen_ai_client_operation_time_per_output_chunk_seconds_bucket"
    dur = "gen_ai_client_operation_duration_seconds_count"
    return {
        "caller_req_per_s": _q(prom, f"sum(increase({dur}[{w}s])) / {w}", end),
        "caller_error_ratio": _q(
            prom,
            f'(sum(increase({dur}{{error_type!~"|cancelled"}}[{w}s])) or vector(0)) / '
            f'sum(increase({dur}{{error_type!="cancelled"}}[{w}s]))',
            end,
        ),
        "caller_ttfc_p50_s": _q(prom, f"histogram_quantile(0.5, sum by (le) (increase({ttfc}[{w}s])))", end),
        "caller_ttfc_p99_s": _q(prom, f"histogram_quantile(0.99, sum by (le) (increase({ttfc}[{w}s])))", end),
        "caller_gap_p50_s": _q(prom, f"histogram_quantile(0.5, sum by (le) (increase({gap}[{w}s])))", end),
        "engine_ttft_p99_s": _q(
            prom,
            "histogram_quantile(0.99, sum by (le) "
            f"(increase(vllm:time_to_first_token_seconds_bucket[{w}s])))",
            end,
        ),
        "engine_up_min": _q(prom, f'min_over_time(up{{job="engine"}}[{w}s])', end),
    }


def run(prom: Prom, tag: str | None = None, label: str = "fault_effects") -> None:
    src = results_dir(tag) / "faults.csv"
    with src.open() as f:
        rows = list(csv.DictReader(f))
    runs = {}
    for r in rows:
        runs.setdefault(
            (r["scenario"], r["repeat"]),
            (float(r["injected_unix"]), float(r["reverted_unix"]), float(r["fault_duration_s"])),
        )
    out = []
    for (scenario, rep), (inj, _rev, dur) in sorted(runs.items(), key=lambda kv: kv[1][0]):
        fault_end = inj + dur  # the fault was active until revert started
        before = _window(prom, inj - BEFORE_S, inj - 5)
        during = (
            _window(prom, inj + SETTLE_S, fault_end) if dur > SETTLE_S + 10 else _window(prom, inj, fault_end)
        )
        row = {"scenario": scenario, "repeat": int(rep), "fault_duration_s": dur}
        for k, v in before.items():
            row[f"before_{k}"] = v
        for k, v in during.items():
            row[f"during_{k}"] = v
        out.append(row)
        print(
            f"[{label}] {scenario} rep {rep}: ttfc p50 {before['caller_ttfc_p50_s']} -> "
            f"{during['caller_ttfc_p50_s']}, gap p50 {before['caller_gap_p50_s']} -> "
            f"{during['caller_gap_p50_s']}, "
            f"errors {before['caller_error_ratio']} -> {during['caller_error_ratio']}",
            flush=True,
        )
    path = results_dir(tag) / f"{label}.csv"
    write_csv(path, out)
    write_meta(
        path,
        source="faults.csv",
        settle_s=SETTLE_S,
        before_s=BEFORE_S,
        note="caller = proxy histograms, engine = vLLM histograms, over each fault window",
    )
