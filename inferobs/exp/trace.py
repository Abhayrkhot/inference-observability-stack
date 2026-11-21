"""Check that one request yields one distributed trace: the proxy's span plus the engine's span.

Sends a few streaming requests through the proxy with a known W3C traceparent, then fetches each
trace from Tempo and records which services contributed spans, the parent/child links and the
engine's latency breakdown attributes (vLLM: queue, prefill, decode). Writes trace_example.json.
"""

from __future__ import annotations

import json
import secrets
import time
import urllib.request

from inferobs.exp.common import results_dir, write_meta


def _post(url: str, body: dict, headers: dict) -> None:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)


def _attrs(span: dict) -> dict:
    out = {}
    for a in span.get("attributes", []):
        v = a["value"]
        out[a["key"]] = next(iter(v.values())) if v else None
    return out


def run(proxy_url: str, tempo_url: str, model: str, n: int = 3, tag: str | None = None) -> dict:
    traces = []
    for i in range(n):
        trace_id = secrets.token_hex(16)
        parent = secrets.token_hex(8)
        body = {
            "model": model,
            "stream": True,
            "max_tokens": 32,
            "ignore_eos": True,
            "messages": [{"role": "user", "content": f"Count to ten, attempt {i}."}],
        }
        _post(f"{proxy_url}/v1/chat/completions", body, {"traceparent": f"00-{trace_id}-{parent}-01"})
        traces.append(trace_id)
    time.sleep(8)
    found = []
    for tid in traces:
        data = None
        for _ in range(15):
            try:
                data = _get(f"{tempo_url}/api/v2/traces/{tid}")
                if data.get("trace", {}).get("resourceSpans"):
                    break
            except Exception:
                pass
            time.sleep(2)
        spans = []
        for rs in (data or {}).get("trace", {}).get("resourceSpans", []):
            svc = next(
                (
                    a["value"].get("stringValue")
                    for a in rs["resource"]["attributes"]
                    if a["key"] == "service.name"
                ),
                "?",
            )
            for ss in rs["scopeSpans"]:
                for sp in ss["spans"]:
                    spans.append(
                        {
                            "service": svc,
                            "name": sp["name"],
                            "kind": sp.get("kind"),
                            "span_id": sp["spanId"],
                            "parent_span_id": sp.get("parentSpanId"),
                            "duration_ms": (int(sp["endTimeUnixNano"]) - int(sp["startTimeUnixNano"])) / 1e6,
                            "attributes": _attrs(sp),
                        }
                    )
        services = sorted({s["service"] for s in spans})
        ids = {s["span_id"] for s in spans}
        linked = any(s["parent_span_id"] in ids for s in spans)
        found.append({"trace_id": tid, "services": services, "spans": spans, "engine_span_is_child": linked})
    out = {
        "traces": found,
        "all_linked": all(t["engine_span_is_child"] and len(t["services"]) >= 2 for t in found),
    }
    path = results_dir(tag) / "trace_example.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    write_meta(path, proxy=proxy_url, tempo=tempo_url, model=model, n=n)
    print(f"[trace] services per trace: {[t['services'] for t in found]}; linked: {out['all_linked']}")
    return out
