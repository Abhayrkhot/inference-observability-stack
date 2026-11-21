#!/usr/bin/env python3
"""Generate deploy/grafana/dashboards/llm-serving.json.

python scripts/build_dashboard.py           # write
python scripts/build_dashboard.py --check   # fail if the committed JSON is stale (CI)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "deploy/grafana/dashboards/llm-serving.json"
F = 'gen_ai_provider_name=~"$provider", gen_ai_request_model=~"$model"'
DS = {"type": "prometheus", "uid": "prometheus"}
RI = "$__rate_interval"

_ids = iter(range(1, 1000))


def target(expr: str, legend: str, ref: str, exemplar: bool = False) -> dict:
    return {
        "datasource": DS,
        "expr": expr,
        "legendFormat": legend,
        "refId": ref,
        "exemplar": exemplar,
        "range": True,
    }


def ts(
    title, targets, unit, x, y, w=8, h=8, desc="", stack=False, minimum=None, maximum=None, overrides=None
) -> dict:
    defaults = {
        "unit": unit,
        "custom": {
            "lineWidth": 1,
            "fillOpacity": 10 if stack else 0,
            "showPoints": "never",
            "stacking": {"mode": "normal" if stack else "none"},
        },
    }
    if minimum is not None:
        defaults["min"] = minimum
    if maximum is not None:
        defaults["max"] = maximum
    return {
        "id": next(_ids),
        "type": "timeseries",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(e, lg, chr(65 + i), ex) for i, (e, lg, ex) in enumerate(targets)],
        "fieldConfig": {"defaults": defaults, "overrides": overrides or []},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom"},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def stat(title, expr, unit, x, y, w=4, h=4, desc="", thresholds=None, decimals=None) -> dict:
    d = {
        "unit": unit,
        "thresholds": {"mode": "absolute", "steps": thresholds or [{"color": "green", "value": None}]},
    }
    if decimals is not None:
        d["decimals"] = decimals
    return {
        "id": next(_ids),
        "type": "stat",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(expr, "", "A")],
        "fieldConfig": {"defaults": d, "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "area"},
    }


def row(title, y) -> dict:
    return {
        "id": next(_ids),
        "type": "row",
        "title": title,
        "collapsed": False,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "panels": [],
    }


def q(p: float, metric: str, extra: str = "") -> str:
    sel = F + (", " + extra if extra else "")
    return f"histogram_quantile({p}, sum by (le) (rate({metric}_bucket{{{sel}}}[{RI}])))"


def build() -> dict:
    ok = 'label_replace({e}, "error_type", "ok", "error_type", "^$")'
    req = f"sum by (error_type) (rate(gen_ai_client_operation_duration_seconds_count{{{F}}}[{RI}]))"
    err_ratio = (
        f'(sum(rate(gen_ai_client_operation_duration_seconds_count{{{F}, error_type!~"|cancelled"}}[{RI}])) '
        f"or vector(0)) / "
        f'sum(rate(gen_ai_client_operation_duration_seconds_count{{{F}, error_type!="cancelled"}}[{RI}]))'
    )
    ttfc = "gen_ai_client_operation_time_to_first_chunk_seconds"
    tpot = "inferobs_time_per_output_token_seconds"
    gap = "gen_ai_client_operation_time_per_output_chunk_seconds"
    dur = "gen_ai_client_operation_duration_seconds"
    vm = 'model_name=~".+"'
    panels = []
    y = 0
    panels.append(row("Caller experience (measured at the proxy)", y))
    y += 1
    panels += [
        stat("Requests/s", f"sum(rate({dur}_count{{{F}}}[{RI}]))", "reqps", 0, y),
        stat(
            "Error ratio",
            err_ratio,
            "percentunit",
            4,
            y,
            decimals=2,
            desc="Failed / all requests, client cancellations excluded.",
            thresholds=[
                {"color": "green", "value": None},
                {"color": "orange", "value": 0.005},
                {"color": "red", "value": 0.02},
            ],
        ),
        stat("TTFC p99", q(0.99, ttfc), "s", 8, y, decimals=3),
        stat("TPOT p99", q(0.99, tpot), "s", 12, y, decimals=4),
        stat(
            "Output tok/s",
            f'sum(rate(gen_ai_client_token_usage_sum{{{F}, gen_ai_token_type="output"}}[{RI}]))',
            "short",
            16,
            y,
            decimals=0,
        ),
        stat("In flight", "sum(inferobs_inflight_requests)", "short", 20, y),
    ]
    y += 4
    panels += [
        ts(
            "Requests/s by outcome",
            [(ok.format(e=req), "{{error_type}}", False)],
            "reqps",
            0,
            y,
            stack=True,
            desc="error_type is empty for success, an HTTP status for upstream errors, or timeout / "
            "connection_error / stream_interrupted / cancelled.",
        ),
        ts(
            "Time to first chunk",
            [(q(0.5, ttfc), "p50", False), (q(0.95, ttfc), "p95", False), (q(0.99, ttfc), "p99", True)],
            "s",
            8,
            y,
            desc="Proxy receipt to first content-bearing chunk. Diamonds are exemplars: click one to open "
            "its trace in Tempo.",
        ),
        ts(
            "Time per output token",
            [(q(0.5, tpot), "p50", False), (q(0.99, tpot), "p99", True)],
            "s",
            16,
            y,
            desc="Per request: (last chunk - first chunk) / (output tokens - 1).",
        ),
    ]
    y += 8
    panels += [
        ts(
            "Gap between chunks (stalls)",
            [(q(0.99, gap), "p99", False), (q(0.999, gap), "p99.9", False)],
            "s",
            0,
            y,
            desc="Every inter-chunk gap is one observation, so this exposes decode stalls that a "
            "per-request mean hides.",
        ),
        ts(
            "Tokens/s",
            [
                (
                    f'sum(rate(gen_ai_client_token_usage_sum{{{F}, gen_ai_token_type="output"}}[{RI}]))',
                    "output",
                    False,
                ),
                (
                    f'sum(rate(gen_ai_client_token_usage_sum{{{F}, gen_ai_token_type="input"}}[{RI}]))',
                    "input",
                    False,
                ),
            ],
            "short",
            8,
            y,
        ),
        ts(
            "Request duration (success)",
            [(q(0.5, dur, 'error_type=""'), "p50", False), (q(0.99, dur, 'error_type=""'), "p99", True)],
            "s",
            16,
            y,
        ),
    ]
    y += 8
    panels.append(row("Engine (vLLM metrics)", y))
    y += 1
    panels += [
        ts(
            "Scheduler queue",
            [
                (f"sum(vllm:num_requests_running{{{vm}}})", "running", False),
                (f"sum(vllm:num_requests_waiting{{{vm}}})", "waiting", False),
            ],
            "short",
            0,
            y,
        ),
        ts(
            "KV-cache usage",
            [(f"max(vllm:kv_cache_usage_perc{{{vm}}})", "used", False)],
            "percentunit",
            8,
            y,
            minimum=0,
            maximum=1,
        ),
        ts(
            "TTFT p99: caller vs engine",
            [
                (q(0.99, ttfc), "caller (proxy)", False),
                (
                    f"histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{{{vm}}}[{RI}])))",
                    "engine (vLLM)",
                    False,
                ),
            ],
            "s",
            16,
            y,
            desc="The gap is time the engine's own histogram cannot see: HTTP, the API server, the network and "
            "any proxy hops.",
        ),
    ]
    y += 8
    panels += [
        ts(
            "Engine queue and prefill time p99",
            [
                (
                    f"histogram_quantile(0.99, sum by (le) (rate(vllm:request_queue_time_seconds_bucket{{{vm}}}[{RI}])))",
                    "queue",
                    False,
                ),
                (
                    f"histogram_quantile(0.99, sum by (le) (rate(vllm:request_prefill_time_seconds_bucket{{{vm}}}[{RI}])))",
                    "prefill",
                    False,
                ),
            ],
            "s",
            0,
            y,
        ),
        ts(
            "Generation tokens/s: engine vs caller",
            [
                (f"sum(rate(vllm:generation_tokens_total{{{vm}}}[{RI}]))", "engine", False),
                (
                    f'sum(rate(gen_ai_client_token_usage_sum{{{F}, gen_ai_token_type="output"}}[{RI}]))',
                    "caller",
                    False,
                ),
            ],
            "short",
            8,
            y,
        ),
        ts(
            "Upstream health and preemptions",
            [
                ("min(inferobs_upstream_up)", "upstream up", False),
                (f"sum(rate(vllm:num_preemptions_total{{{vm}}}[{RI}]))", "preemptions/s", False),
            ],
            "short",
            16,
            y,
        ),
    ]
    y += 8
    panels.append(row("SLOs (multi-window burn rates)", y))
    y += 1
    burn = "slo:sli_error:ratio_rate{w} / on (slo) group_left () (1 - slo:objective:ratio)"
    panels += [
        ts(
            "Burn rate, fast windows",
            [(burn.format(w="5m"), "{{slo}} 5m", False), (burn.format(w="1h"), "{{slo}} 1h", False)],
            "short",
            0,
            y,
            w=12,
            desc="1 = spending the error budget exactly over the SLO period. Page when both windows exceed "
            "14.4 (2% of a 30-day budget in an hour).",
        ),
        ts(
            "Error budget remaining (30d)",
            [
                (
                    "1 - (slo:sli_error:ratio_rate30d / on (slo) group_left () (1 - slo:objective:ratio))",
                    "{{slo}}",
                    False,
                )
            ],
            "percentunit",
            12,
            y,
            w=12,
            maximum=1,
        ),
    ]
    return {
        "uid": "inferobs-llm-serving",
        "title": "LLM serving: caller vs engine",
        "tags": ["llm", "inference", "inferobs"],
        "timezone": "browser",
        "refresh": "10s",
        "time": {"from": "now-30m", "to": "now"},
        "schemaVersion": 41,
        "version": 1,
        "editable": True,
        "templating": {
            "list": [
                {
                    "name": "provider",
                    "type": "query",
                    "datasource": DS,
                    "label": "provider",
                    "query": {
                        "query": "label_values(gen_ai_client_operation_duration_seconds_count, gen_ai_provider_name)",
                        "refId": "p",
                    },
                    "includeAll": True,
                    "multi": True,
                    "current": {"text": "All", "value": "$__all"},
                    "refresh": 2,
                },
                {
                    "name": "model",
                    "type": "query",
                    "datasource": DS,
                    "label": "model",
                    "query": {
                        "query": 'label_values(gen_ai_client_operation_duration_seconds_count{gen_ai_provider_name=~"$provider"}, gen_ai_request_model)',
                        "refId": "m",
                    },
                    "includeAll": True,
                    "multi": True,
                    "current": {"text": "All", "value": "$__all"},
                    "refresh": 2,
                },
            ]
        },
        "annotations": {
            "list": [
                {
                    "name": "Alerts",
                    "datasource": DS,
                    "enable": True,
                    "iconColor": "red",
                    "expr": 'ALERTS{alertstate="firing"}',
                    "titleFormat": "{{alertname}}",
                    "step": "15s",
                }
            ]
        },
        "panels": panels,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = json.dumps(build(), indent=2) + "\n"
    if a.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT} is stale; run scripts/build_dashboard.py", file=sys.stderr)
            return 1
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
