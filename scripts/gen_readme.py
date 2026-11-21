#!/usr/bin/env python3
"""Fill the generated blocks of README.md from results/<tag>/*.csv.

Hand-written text lives outside ``<!-- BEGIN GENERATED: name -->`` / ``<!-- END GENERATED: name -->``
markers; everything between them is rebuilt from result files, so the README cannot drift from the
measurements. ``--check`` fails (CI) if README.md is not what this script would write.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
GPU_TAG = "box-3050ti"


def rows(tag: str, name: str) -> list[dict]:
    p = ROOT / "results" / tag / f"{name}.csv"
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def meta(tag: str, name: str) -> dict:
    p = ROOT / "results" / tag / f"{name}.meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def f(v, scale=1.0, digits=1, unit="") -> str:
    if v in (None, "", "None"):
        return "n/a"
    x = float(v) * scale
    return f"{x:,.{digits}f}{unit}"


def ms(v, digits=1) -> str:
    return f(v, 1e3, digits, " ms")


def pct(v, digits=1) -> str:
    if v in (None, "", "None"):
        return "n/a"
    return f"{float(v) * 100:+.{digits}f}%"


def table(header: list[str], body: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def missing(name: str) -> str:
    return f"_Not measured yet: run the `{name}` experiment (see Reproducing)._"


def src(tag: str, name: str) -> str:
    m = meta(tag, name)
    bits = [f"`results/{tag}/{name}.csv`"]
    if m.get("git_sha"):
        bits.append(f"commit `{m['git_sha']}`")
    return "Source: " + ", ".join(bits) + "."


# ---------------------------------------------------------------- blocks
def block_setup() -> str:
    m = meta(GPU_TAG, "overhead") or meta(GPU_TAG, "agreement")
    gpu = "NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB)"
    return (
        f"Measured on one {gpu}, vLLM 0.34.0 serving Qwen2.5-0.5B-Instruct (bf16), on a shared "
        f"20-thread i7-12700H laptop ({m.get('cpus', 20)} CPUs visible). Prometheus scrapes every 5 s. "
        "Workload: lognormal prompts (mean 256 words) and outputs (mean 128 tokens, `ignore_eos`)."
    )


def block_headline() -> str:
    ov = rows(GPU_TAG, "overhead")
    ag = {(r["metric"], r["source"], r["quantile"]): r for r in rows(GPU_TAG, "agreement")}
    bk = rows(GPU_TAG, "agreement_buckets")
    fl = rows(GPU_TAG, "faults")
    sat = rows(GPU_TAG, "saturation")
    out = []
    p1 = next((r for r in ov if r["target"] == "proxy" and r["concurrency"] == "1"), None)
    p32 = next((r for r in ov if r["target"] == "proxy+tracing" and r["concurrency"] == "32"), None)
    if p1:
        out.append(
            f"- **Metering costs {f(p1['ttft_p50_delta_ms'], digits=2)} ms of TTFT at p50** "
            f"(single stream) and {pct(float(p32['throughput_delta_pct']) / 100) if p32 else 'n/a'} "
            f"throughput at 32 concurrent streams with every request traced."
        )
    k = ("ttft", "proxy histogram", "0.99")
    e = ("ttft", "proxy exact", "0.99")
    if k in ag and e in ag:
        err = float(ag[k]["value_s"]) / float(ag[e]["value_s"]) - 1
        out.append(
            f"- **The dashboard's TTFT p99 is within {abs(err) * 100:.1f}% of the exact value** "
            "computed from every request in the access log."
        )
    vb = next(
        (
            r
            for r in bk
            if r["metric"] == "tpot" and r["quantile"] == "0.99" and r["layout"] == "vLLM default"
        ),
        None,
    )
    fb = next(
        (
            r
            for r in bk
            if r["metric"] == "tpot" and r["quantile"] == "0.99" and r["layout"] == "inferobs fine"
        ),
        None,
    )
    if vb and fb:
        out.append(
            f"- **vLLM's default buckets misreport TPOT p99 by {pct(vb['rel_error'], 0)}** on this GPU "
            f"(true value {ms(vb['exact_s'])}, below the lowest 10 ms bound); the proxy's layout is off by "
            f"{pct(fb['rel_error'])}."
        )
    if fl:
        lat = [r for r in fl if r["scenario"] == "net_latency"]
        eng = [r for r in lat if r["alert"] == "EngineTTFTHigh"]
        cal = [r for r in lat if r["alert"] == "SLOErrorBudgetBurnFast:ttfc" and r["detected"] == "True"]
        if cal:
            det = sorted(float(r["detect_s"]) for r in cal)
            out.append(
                f"- **A 3 s network delay is caught by the caller-side TTFC SLO in "
                f"{det[len(det) // 2]:.0f} s (median); the engine's own TTFT never moves**"
                f"{' (EngineTTFTHigh did not fire)' if not any(r['detected'] == 'True' for r in eng) else ''}."
            )
    if sat:
        ok = [r for r in sat if r["meets_slo"] == "True"]
        if ok:
            best = max(ok, key=lambda r: float(r["offered_rps"]))
            out.append(
                f"- **Capacity at SLO: {f(best['offered_rps'], digits=1)} req/s** "
                f"({f(best['goodput_tok_per_s'], digits=0)} output tok/s) before TTFT p99 passes 1 s or "
                "TPOT p99 passes 50 ms."
            )
    return "\n".join(out) if out else "_Results pending._"


def block_overhead() -> str:
    r = rows(GPU_TAG, "overhead")
    if not r:
        return missing("overhead")
    body = []
    for x in r:
        body.append(
            [
                x["concurrency"],
                x["target"],
                ms(x["ttft_p50_s"], 2),
                ms(x["ttft_p99_s"], 2),
                ms(x["tpot_p50_s"], 2),
                f(x["output_tok_per_s"], digits=0),
                "" if x["target"] == "direct" else f(x.get("ttft_p50_delta_ms"), digits=2, unit=" ms"),
                "" if x["target"] == "direct" else pct(float(x["throughput_delta_pct"]) / 100),
                "" if x["target"] == "direct" else f(x.get("proxy_cpu_util"), 100, 0, "%"),
            ]
        )
    m = meta(GPU_TAG, "overhead")
    return (
        table(
            [
                "streams",
                "target",
                "TTFT p50",
                "TTFT p99",
                "TPOT p50",
                "output tok/s",
                "TTFT p50 delta",
                "throughput delta",
                "proxy CPU",
            ],
            body,
        )
        + f"\n\nMedians of {m.get('repeats', '?')} repeats of {f(m.get('duration_s'), digits=0)} s closed-loop "
        "runs per cell, target order rotated each repeat. Proxy CPU is the proxy process's CPU time over "
        "wall time (100% = one core). " + src(GPU_TAG, "overhead")
    )


def block_agreement() -> str:
    r = rows(GPU_TAG, "agreement")
    if not r:
        return missing("agreement")
    idx = {(x["metric"], x["source"], x["quantile"]): x for x in r}
    sources = ["client", "proxy exact", "proxy histogram", "engine histogram (vLLM)"]
    body = []
    for metric, label in (
        ("ttft", "TTFT"),
        ("tpot", "TPOT (per request)"),
        ("itl", "gap between chunks"),
        ("e2e", "end to end"),
    ):
        for q in ("0.5", "0.99"):
            cells = [
                ms(idx[(metric, s, q)]["value_s"], 2) if (metric, s, q) in idx else "n/a" for s in sources
            ]
            body.append([label, f"p{round(float(q) * 100)}", *cells])
    d = [idx.get(("ttft client - proxy", "per-request diff", q)) for q in ("0.5", "0.99")]
    m = meta(GPU_TAG, "agreement")
    note = ""
    if all(d):
        note = (
            f"\n\nPer request (matched by `X-Request-Id`), the client saw TTFT {ms(d[0]['value_s'], 2)} later "
            f"than the proxy at p50 and {ms(d[1]['value_s'], 2)} at p99: the loopback hop and the client's own "
            "event loop. "
        )
    return (
        table(["metric", "q", "client", "proxy (exact)", "proxy (Prometheus)", "vLLM (Prometheus)"], body)
        + note
        + f"\n\n{m.get('requests', '?')} requests at {m.get('rate', '?')} req/s for "
        f"{f(m.get('duration_s'), digits=0)} s. " + src(GPU_TAG, "agreement")
    )


def block_buckets() -> str:
    r = rows(GPU_TAG, "agreement_buckets")
    if not r:
        return missing("agreement")
    body = []
    for metric, label in (
        ("ttft", "TTFT"),
        ("tpot", "TPOT"),
        ("itl", "gap between chunks"),
        ("e2e", "end to end"),
    ):
        for q in ("0.5", "0.99"):
            xs = {x["layout"]: x for x in r if x["metric"] == metric and x["quantile"] == q}
            if not xs:
                continue
            ex = next(iter(xs.values()))["exact_s"]
            body.append(
                [label, f"p{round(float(q) * 100)}", ms(ex, 2)]
                + [
                    pct(xs[lay]["rel_error"]) if lay in xs else "n/a"
                    for lay in ("vLLM default", "OTel semconv", "inferobs fine")
                ]
            )
    nb = {x["layout"]: x["buckets"] for x in r if x["metric"] == "itl"}
    return (
        table(
            [
                "metric",
                "q",
                "exact",
                f"vLLM default ({nb.get('vLLM default', '?')} buckets)",
                f"OTel semconv ({nb.get('OTel semconv', '?')})",
                f"inferobs fine ({nb.get('inferobs fine', '?')})",
            ],
            body,
        )
        + "\n\nError of PromQL `histogram_quantile` against the exact quantile of the same observations "
        "(every request and chunk gap from the agreement run, re-bucketed per layout with a reimplementation "
        "of PromQL's interpolation, `inferobs/analysis/stats.py`). " + src(GPU_TAG, "agreement_buckets")
    )


def block_saturation() -> str:
    r = rows(GPU_TAG, "saturation")
    if not r:
        return missing("saturation")
    body = [
        [
            f(x["offered_rps"], digits=1),
            ms(x["ttft_p50_s"]),
            ms(x["ttft_p99_s"]),
            ms(x["tpot_p99_s"], 2),
            f(x["goodput_tok_per_s"], digits=0),
            f(x["engine_running_avg"], digits=1),
            f(x["engine_waiting_avg"], digits=1),
            f(x["engine_kv_max"], 100, 0, "%"),
            "yes" if x["meets_slo"] == "True" else "**no**",
        ]
        for x in r
    ]
    m = meta(GPU_TAG, "saturation")
    return (
        table(
            [
                "offered req/s",
                "TTFT p50",
                "TTFT p99",
                "TPOT p99",
                "goodput tok/s",
                "running",
                "waiting",
                "KV peak",
                "meets SLOs",
            ],
            body,
        )
        + f"\n\nOpen-loop Poisson arrivals, {f(m.get('stage_s'), digits=0)} s per stage, first "
        f"{f(m.get('settle_s'), digits=0)} s of each stage excluded. Caller columns from the load generator, "
        "engine columns from vLLM's metrics over the same window. " + src(GPU_TAG, "saturation")
    )


def block_faults() -> str:
    r = rows(GPU_TAG, "faults")
    if not r:
        return missing("faults")
    scen = []
    for x in r:
        if x["scenario"] not in scen:
            scen.append(x["scenario"])
    body = []
    for s in scen:
        xs = [x for x in r if x["scenario"] == s]
        alerts = []
        for x in xs:
            if x["alert"] not in alerts:
                alerts.append(x["alert"])
        for a in alerts:
            ys = [x for x in xs if x["alert"] == a and x.get("already_firing") != "True"]
            if not ys:
                continue
            det = sorted(float(y["detect_s"]) for y in ys if y["detected"] == "True")
            exp = {"fire": "fire", "silent": "stay silent"}.get(ys[0]["expected"], "")
            reps = len(ys)
            if det:
                cell = f"{det[len(det) // 2]:.0f} s" + (
                    f" ({det[0]:.0f} to {det[-1]:.0f})" if len(det) > 1 else ""
                )
            else:
                cell = "not fired"
            body.append([s, a, exp, f"{len(det)}/{reps}", cell])
    rec = [
        float(x["engine_recovery_s"]) for x in r if x["scenario"] == "engine_crash" and x["engine_recovery_s"]
    ]
    m = meta(GPU_TAG, "faults")
    note = f" vLLM took {min(rec):.0f} to {max(rec):.0f} s to serve again after a restart." if rec else ""
    return (
        table(["fault", "alert", "should", "fired", "detection (median, range)"], body)
        + f"\n\nDetection is the first Prometheus evaluation with the alert firing after injection, with the "
        f"compressed-window SLO rules (window_scale 1/12: the 1 h/5 m fast-burn page becomes 5 m/25 s) and "
        f"a background load of {m.get('rate', '?')} req/s.{note} " + src(GPU_TAG, "faults")
    )


def block_trace() -> str:
    p = ROOT / "results" / GPU_TAG / "trace_example.json"
    if not p.exists():
        return missing("trace")
    t = json.loads(p.read_text())["traces"][0]
    lines = []
    for s in sorted(t["spans"], key=lambda s: s["parent_span_id"] or ""):
        a = s["attributes"]
        keys = [
            k
            for k in (
                "gen_ai.latency.time_in_queue",
                "gen_ai.latency.time_to_first_token",
                "gen_ai.latency.time_in_model_prefill",
                "gen_ai.latency.time_in_model_decode",
                "inferobs.time_to_first_chunk_s",
                "gen_ai.usage.output_tokens",
            )
            if k in a
        ]
        attrs = ", ".join(
            f"{k}={float(a[k]):.4g}"
            if isinstance(a[k], (int, float, str)) and str(a[k]).replace(".", "", 1).isdigit()
            else f"{k}={a[k]}"
            for k in keys
        )
        lines.append(f"{s['service']:<16} {s['name']:<28} {s['duration_ms']:8.1f} ms  {attrs}")
    return (
        "```text\n"
        + "\n".join(lines)
        + "\n```\n\n"
        + (
            "One request, one trace: the proxy's client span and vLLM's server span share the trace ID the caller "
            "sent, and vLLM's span carries its own queue/prefill/decode split. "
            f"Source: `results/{GPU_TAG}/trace_example.json`."
        )
    )


BLOCKS = {
    "setup": block_setup,
    "headline": block_headline,
    "overhead": block_overhead,
    "agreement": block_agreement,
    "buckets": block_buckets,
    "saturation": block_saturation,
    "faults": block_faults,
    "trace": block_trace,
}


def render(text: str) -> str:
    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in BLOCKS:
            raise SystemExit(f"unknown generated block {name!r}")
        return f"<!-- BEGIN GENERATED: {name} -->\n{BLOCKS[name]()}\n<!-- END GENERATED: {name} -->"

    return re.sub(r"<!-- BEGIN GENERATED: (\w+) -->.*?<!-- END GENERATED: \1 -->", sub, text, flags=re.S)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = README.read_text()
    new = render(text)
    if a.check:
        if new != text:
            print("README.md is stale; run scripts/gen_readme.py", file=sys.stderr)
            return 1
        return 0
    README.write_text(new)
    print("README.md updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
