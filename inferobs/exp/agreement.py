"""Do the caller, the proxy, Prometheus and the engine agree on latency?

One open-loop run through the stack's proxy. Sources compared per metric and quantile:

- ``client``: the load generator's own measurement (from the scheduled arrival time).
- ``proxy exact``: the proxy's access log (every request, every chunk gap).
- ``proxy histogram``: Prometheus histogram_quantile over the run window (the dashboard's number).
- ``engine histogram``: vLLM's own histograms over the same window.

The same run feeds the bucket-layout analysis: the exact per-request and per-gap values from the
access log are re-bucketed into vLLM's default layout, the OpenTelemetry advisory layout and the
proxy's fine layout, and histogram_quantile's error is measured against the exact quantile.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import orjson

from inferobs.analysis.stats import bucket_counts, histogram_quantile, quantile
from inferobs.exp.common import (
    Prom,
    docker,
    results_dir,
    run_loadgen,
    runs_dir,
    write_csv,
    write_meta,
)
from inferobs.proxy import metrics as pm
from inferobs.vllm_buckets import VLLM_ITL, VLLM_LATENCY, VLLM_TTFT

QS = (0.5, 0.9, 0.99)


def _load_access_log(container: str, dest: Path, t0: float, t1: float) -> list[dict]:
    raw = docker("exec", container, "cat", "/logs/access.jsonl", timeout=120)
    dest.write_text(raw + "\n")
    out = []
    for line in raw.splitlines():
        rec = orjson.loads(line)
        if t0 - 1 <= rec["ts"] <= t1 and (rec.get("request_id") or "").startswith("lg-"):
            out.append(rec)
    return out


def _client_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [r for r in csv.DictReader(f) if not r["error"]]


def _client_gaps(path: Path) -> list[float]:
    gaps: list[float] = []
    for line in path.read_text().splitlines():
        gaps += [g / 1e6 for g in orjson.loads(line)["gaps_us"]]
    return gaps


def run(
    proxy_url: str,
    model: str,
    rate: float,
    duration: float,
    prompt: str,
    output: str,
    prom: Prom,
    proxy_container: str = "inferobs-proxy-1",
    tag: str | None = None,
    label: str = "agreement",
    engine_label: str = "engine histogram (vLLM)",
) -> None:
    rd = runs_dir(label)
    prom.wait_ready()
    time.sleep(12)  # let any earlier traffic age out of the scrape before the window opens
    t0 = time.time()
    s = run_loadgen(
        proxy_url,
        model,
        rd,
        "load",
        rate=rate,
        duration=duration,
        prompt=prompt,
        output=output,
        gaps=rd / "load.gaps.jsonl",
    )
    t1 = time.time()
    time.sleep(15)  # two scrapes after the last request
    t_eval = time.time()
    window = (t0 - 2, t_eval)

    access = _load_access_log(proxy_container, rd / "access.jsonl", t0, t1)
    ok = [a for a in access if not a["error_type"]]
    client = _client_rows(rd / "load.csv")
    cgaps = _client_gaps(rd / "load.gaps.jsonl")
    pgaps = [g / 1e6 for a in ok for g in a.get("gaps_us") or []]
    if not pgaps:
        raise SystemExit("access log has no chunk gaps: start the stack with INFEROBS_ACCESS_LOG_GAPS=1")

    exact = {
        ("ttft", "client"): [float(r["ttft_s"]) for r in client if r["ttft_s"]],
        ("ttft", "proxy exact"): [a["ttfc_s"] for a in ok if a["ttfc_s"] is not None],
        ("itl", "client"): cgaps,
        ("itl", "proxy exact"): pgaps,
        ("tpot", "client"): [float(r["tpot_s"]) for r in client if r["tpot_s"]],
        ("tpot", "proxy exact"): [a["tpot_s"] for a in ok if a["tpot_s"] is not None],
        ("e2e", "client"): [float(r["e2e_s"]) for r in client if r["e2e_s"]],
        ("e2e", "proxy exact"): [a["duration_s"] for a in ok],
    }
    hist = {
        ("ttft", "proxy histogram"): ("gen_ai_client_operation_time_to_first_chunk_seconds", ""),
        ("itl", "proxy histogram"): ("gen_ai_client_operation_time_per_output_chunk_seconds", ""),
        ("tpot", "proxy histogram"): ("inferobs_time_per_output_token_seconds", ""),
        ("e2e", "proxy histogram"): ("gen_ai_client_operation_duration_seconds", 'error_type=""'),
        ("ttft", engine_label): ("vllm:time_to_first_token_seconds", ""),
        ("itl", engine_label): ("vllm:inter_token_latency_seconds", ""),
        ("tpot", engine_label): ("vllm:request_time_per_output_token_seconds", ""),
        ("e2e", engine_label): ("vllm:e2e_request_latency_seconds", ""),
    }
    rows = []
    for (metric, source), values in exact.items():
        for q in QS:
            rows.append(
                {
                    "metric": metric,
                    "source": source,
                    "quantile": q,
                    "value_s": quantile(values, q),
                    "n": len(values),
                }
            )
    for (metric, source), (name, sel) in hist.items():
        bounds, cum = prom.buckets(name, sel, *window)
        if not cum or cum[-1] == 0:
            continue
        for q in QS:
            rng = max(1, round(window[1] - window[0]))
            promq = prom.scalar(
                f"histogram_quantile({q}, sum by (le) (increase({name}_bucket{{{sel}}}[{rng}s])))", window[1]
            )
            rows.append(
                {
                    "metric": metric,
                    "source": source,
                    "quantile": q,
                    "value_s": promq if promq is not None else histogram_quantile(q, bounds, cum),
                    "n": round(cum[-1]),
                    "buckets": len(bounds),
                }
            )
    # Per-request differences, matched by X-Request-Id.
    by_id = {a["request_id"]: a for a in ok}
    diffs_ttft, diffs_e2e = [], []
    for r in client:
        a = by_id.get(f"lg-{r['id']}")
        if a and r["ttft_s"] and a["ttfc_s"] is not None:
            diffs_ttft.append(float(r["ttft_s"]) - a["ttfc_s"])
            diffs_e2e.append(float(r["e2e_s"]) - a["duration_s"])
    for name, vals in (("ttft client - proxy", diffs_ttft), ("e2e client - proxy", diffs_e2e)):
        for q in QS:
            rows.append(
                {
                    "metric": name,
                    "source": "per-request diff",
                    "quantile": q,
                    "value_s": quantile(vals, q),
                    "n": len(vals),
                }
            )

    out_dir = results_dir(tag)
    write_csv(out_dir / f"{label}.csv", rows)
    meta = dict(
        proxy=proxy_url,
        model=model,
        rate=rate,
        duration_s=duration,
        prompt=prompt,
        output=output,
        requests=s["requests"],
        errors=s["errors"],
        matched=len(diffs_ttft),
        run_dir=rd.name,
        window_s=round(window[1] - window[0], 1),
    )
    write_meta(out_dir / f"{label}.csv", **meta)
    write_csv(out_dir / f"{label}_buckets.csv", bucket_accuracy(ok, pgaps))
    write_meta(out_dir / f"{label}_buckets.csv", **meta)
    print(f"[{label}] {s['requests']} requests, {len(ok)} in access log, {len(pgaps)} gaps", flush=True)


def bucket_accuracy(ok: list[dict], gaps: list[float]) -> list[dict]:
    values = {
        "ttft": [a["ttfc_s"] for a in ok if a["ttfc_s"] is not None],
        "itl": gaps,
        "tpot": [a["tpot_s"] for a in ok if a["tpot_s"] is not None],
        "e2e": [a["duration_s"] for a in ok],
    }
    layouts = {
        "vLLM default": {"ttft": VLLM_TTFT, "itl": VLLM_ITL, "tpot": VLLM_ITL, "e2e": VLLM_LATENCY},
        "OTel semconv": {
            "ttft": pm.SEMCONV_TTFT,
            "itl": pm.SEMCONV_TPOT,
            "tpot": pm.SEMCONV_TPOT,
            "e2e": pm.SEMCONV_DURATION,
        },
        "inferobs fine": {
            "ttft": pm.FINE_TTFT,
            "itl": pm.FINE_TPOT,
            "tpot": pm.FINE_TPOT,
            "e2e": pm.FINE_DURATION,
        },
    }
    rows = []
    for metric, vals in values.items():
        for q in QS:
            ex = quantile(vals, q)
            for layout, fams in layouts.items():
                b = list(fams[metric])
                est = histogram_quantile(q, b, bucket_counts(vals, b))
                rows.append(
                    {
                        "metric": metric,
                        "quantile": q,
                        "layout": layout,
                        "buckets": len(b),
                        "exact_s": ex,
                        "estimate_s": est,
                        "rel_error": (est - ex) / ex if ex else None,
                        "n": len(vals),
                    }
                )
    return rows
