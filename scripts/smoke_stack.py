#!/usr/bin/env python3
"""End-to-end check of a running stack: load through the proxy, then confirm the data arrived.

Checks that every Prometheus target is up, client and engine recording rules have values, exemplars
were stored, the exemplar's trace exists in Tempo, Grafana has the dashboard, and every panel query
of the dashboard parses in Prometheus. Stdlib only (runs in CI before anything is installed).

    python scripts/smoke_stack.py --model sim-llm
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def wait(url: str, secs: float) -> None:
    deadline = time.time() + secs
    while True:
        try:
            urllib.request.urlopen(url, timeout=3).read()
            return
        except Exception:
            if time.time() > deadline:
                raise
            time.sleep(1)


def promq(base: str, expr: str) -> list:
    return get(f"{base}/api/v1/query?query={urllib.parse.quote(expr)}")["data"]["result"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="http://127.0.0.1:28000")
    ap.add_argument("--prometheus", default="http://127.0.0.1:28090")
    ap.add_argument("--tempo", default="http://127.0.0.1:28200")
    ap.add_argument("--grafana", default="http://127.0.0.1:28300")
    ap.add_argument("--model", default="sim-llm")
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--rate", type=float, default=5)
    ap.add_argument("--loadgen", default=f"{sys.executable} -m inferobs loadgen")
    a = ap.parse_args()
    fails: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            fails.append(what)

    for u in (f"{a.proxy}/health", f"{a.prometheus}/-/ready", f"{a.tempo}/ready", f"{a.grafana}/api/health"):
        wait(u, 180)
    cmd = (
        f"{a.loadgen} --url {a.proxy} --model {a.model} --rate {a.rate} --duration {a.duration} "
        f"--prompt lognormal:64:0.5:8:256 --output lognormal:32:0.5:4:128"
    )
    out = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, cwd=ROOT).stdout
    s = json.loads(out)
    check(s["requests"] > 0 and s["errors"] == 0, f"loadgen: {s['requests']} requests, {s['errors']} errors")
    time.sleep(15)  # two scrape + rule-evaluation cycles

    targets = get(f"{a.prometheus}/api/v1/targets")["data"]["activeTargets"]
    health = {t["labels"]["job"]: t["health"] for t in targets}
    check(all(h == "up" for h in health.values()), f"targets up: {health}")
    for expr in (
        "provider_model:gen_ai_ttfc_seconds:p99_1m",
        "provider_model:gen_ai_output_tokens:rate1m",
        "provider_model:gen_ai_errors:ratio_rate1m",
        "model:vllm_ttft_seconds:p99_1m",
    ):
        r = promq(a.prometheus, expr)
        check(bool(r), f"{expr} = {[round(float(x['value'][1]), 4) for x in r]}")

    now = time.time()
    ex = get(
        f"{a.prometheus}/api/v1/query_exemplars?query="
        f"{urllib.parse.quote('gen_ai_client_operation_time_to_first_chunk_seconds_bucket')}"
        f"&start={now - 600:.0f}&end={now:.0f}"
    )["data"]
    ids = [e["labels"].get("trace_id") for series in ex for e in series["exemplars"]]
    check(bool(ids), f"exemplars stored: {len(ids)}")
    if ids:
        found = False
        for _ in range(20):
            try:
                tr = get(f"{a.tempo}/api/v2/traces/{ids[-1]}")
                found = bool(tr.get("trace", {}).get("resourceSpans"))
            except Exception:
                found = False
            if found:
                break
            time.sleep(2)
        check(found, f"exemplar trace {ids[-1]} found in Tempo")

    dash = get(f"{a.grafana}/api/dashboards/uid/inferobs-llm-serving")["dashboard"]
    check(len(dash["panels"]) > 10, f"Grafana dashboard provisioned ({len(dash['panels'])} panels)")
    bad = []
    for p in dash["panels"]:
        for t in p.get("targets", []):
            expr = re.sub(r"\$__rate_interval", "1m", t["expr"])
            expr = expr.replace("$provider", ".*").replace("$model", ".*")
            try:
                get(f"{a.prometheus}/api/v1/query?query={urllib.parse.quote(expr)}")
            except Exception as e:
                bad.append((p["title"], str(e)))
    check(not bad, f"all dashboard queries parse ({bad[:2]})")
    print("SMOKE", "FAILED" if fails else "PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
