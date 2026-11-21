"""Command-line entry point: ``inferobs <proxy|sim|loadgen|slo|...>``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys


def _env(name: str, default=None):
    return os.environ.get(name, default)


def cmd_proxy(a: argparse.Namespace) -> None:
    from inferobs.proxy.app import ProxyConfig, run

    ups = []
    for spec in a.upstream:
        name, _, url = spec.partition("=")
        if not url:
            raise SystemExit(f"--upstream wants NAME=URL, got {spec!r}")
        ups.append((name, url))
    cfg = ProxyConfig(
        upstreams=ups,
        layout=a.buckets,
        idle_timeout=a.idle_timeout,
        health_interval=a.health_interval,
        access_log=a.access_log,
        access_log_gaps=a.access_log_gaps,
        otlp_endpoint=a.otlp_endpoint,
        sample_ratio=a.sample_ratio,
        service_name=a.service_name,
        inject_usage=not a.no_inject_usage,
    )
    run(cfg, a.host, a.port)


def cmd_sim(a: argparse.Namespace) -> None:
    from inferobs.sim.engine import SimConfig
    from inferobs.sim.server import run

    cfg = SimConfig(
        model=a.model,
        step_base_s=a.step_base_ms / 1e3,
        step_per_seq_s=a.step_per_seq_ms / 1e3,
        prefill_per_token_s=a.prefill_per_token_us / 1e6,
        max_num_seqs=a.max_num_seqs,
        max_batched_tokens=a.max_batched_tokens,
        kv_capacity_tokens=a.kv_capacity_tokens,
        max_model_len=a.max_model_len,
        seed=a.seed,
    )
    run(cfg, a.host, a.port)


def cmd_loadgen(a: argparse.Namespace) -> None:
    from inferobs.loadgen.runner import LoadConfig, parse_stages, run_load, write_results

    extra = json.loads(a.extra_body) if a.extra_body else {}
    cfg = LoadConfig(
        url=a.url,
        model=a.model,
        stages=parse_stages(a.rate, a.concurrency, a.duration, a.stages),
        endpoint=a.endpoint,
        prompt=a.prompt,
        output=a.output,
        seed=a.seed,
        ignore_eos=not a.no_ignore_eos,
        stream=not a.no_stream,
        timeout_s=a.timeout,
        extra_body=extra,
    )
    results, summ = asyncio.run(run_load(cfg))
    if a.out:
        write_results(results, a.out, a.gaps_out)
    if a.summary_out:
        with open(a.summary_out, "w") as f:
            json.dump(summ, f, indent=2)
    s = {k: v for k, v in summ.items() if k != "config"}
    print(json.dumps(s, indent=2))


def cmd_slo_gen(a: argparse.Namespace) -> None:
    from pathlib import Path

    from inferobs.slo.gen import render

    text = render(a.spec)
    out = Path(a.out)
    if a.check:
        if not out.exists() or out.read_text() != text:
            raise SystemExit(f"{out} is stale; regenerate with: inferobs slo gen --spec {a.spec} --out {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def cmd_exp(a: argparse.Namespace) -> None:
    from inferobs.exp.common import Prom

    prom = Prom(a.prometheus)
    if a.exp_cmd == "overhead":
        from inferobs.exp import overhead

        if a.interleave:
            overhead.run_interleaved(
                a.engine_url,
                a.model,
                a.provider,
                [int(x) for x in _floats(a.levels)],
                a.duration,
                a.prompt,
                a.output,
                a.otlp,
                port_base=a.port_base,
                tag=a.tag,
                label=a.label,
            )
            return
        overhead.run(
            a.engine_url,
            a.model,
            a.provider,
            [int(x) for x in _floats(a.levels)],
            a.duration,
            a.repeats,
            a.prompt,
            a.output,
            a.otlp,
            tag=a.tag,
            label=a.label,
        )
    elif a.exp_cmd == "agreement":
        from inferobs.exp import agreement

        agreement.run(
            a.proxy_url,
            a.model,
            a.rate,
            a.duration,
            a.prompt,
            a.output,
            prom,
            tag=a.tag,
            label=a.label,
            engine_label=a.engine_label,
        )
    elif a.exp_cmd == "saturation":
        from inferobs.exp import saturation

        saturation.run(
            a.proxy_url,
            a.model,
            _floats(a.rates),
            a.stage_s,
            a.prompt,
            a.output,
            prom,
            tag=a.tag,
            label=a.label,
        )
    elif a.exp_cmd == "trace":
        from inferobs.exp import trace

        trace.run(a.proxy_url, a.tempo_url, a.model, a.n, tag=a.tag)
    elif a.exp_cmd == "faults":
        from inferobs.exp import faults

        faults.run(
            a.engine_kind,
            a.engine_container,
            a.engine_url,
            a.proxy_url,
            a.model,
            a.rate,
            a.prompt,
            a.output,
            a.scenarios.split(","),
            a.repeats,
            a.max_fault_s,
            a.cooldown_s,
            a.warmup_s,
            prom,
            tag=a.tag,
            label=a.label,
        )


def cmd_alert_sink(a: argparse.Namespace) -> None:
    from inferobs.alertsink import run

    run(a.host, a.port, a.out)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="inferobs")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("proxy", help="run the metering proxy")
    pp.add_argument("--upstream", action="append", default=[], help="NAME=URL (repeatable)")
    pp.add_argument("--host", default="0.0.0.0")
    pp.add_argument("--port", type=int, default=9000)
    pp.add_argument("--buckets", default=_env("INFEROBS_BUCKETS", "fine"), choices=["fine", "semconv"])
    pp.add_argument("--idle-timeout", type=float, default=60.0)
    pp.add_argument("--health-interval", type=float, default=5.0)
    pp.add_argument("--access-log", default=_env("INFEROBS_ACCESS_LOG"))
    pp.add_argument(
        "--access-log-gaps",
        action="store_true",
        default=_env("INFEROBS_ACCESS_LOG_GAPS", "0") == "1",
        help="include every chunk gap in the access log (for bucket-accuracy analysis)",
    )
    pp.add_argument("--otlp-endpoint", default=_env("OTEL_EXPORTER_OTLP_ENDPOINT"))
    pp.add_argument("--sample-ratio", type=float, default=float(_env("INFEROBS_SAMPLE_RATIO", "1.0")))
    pp.add_argument("--service-name", default=_env("OTEL_SERVICE_NAME", "inferobs-proxy"))
    pp.add_argument("--no-inject-usage", action="store_true")
    pp.set_defaults(fn=cmd_proxy)

    ps = sub.add_parser("sim", help="run the simulated OpenAI-compatible engine")
    ps.add_argument("--host", default="0.0.0.0")
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--model", default="sim-llm")
    ps.add_argument("--step-base-ms", type=float, default=6.0)
    ps.add_argument("--step-per-seq-ms", type=float, default=0.2)
    ps.add_argument("--prefill-per-token-us", type=float, default=30.0)
    ps.add_argument("--max-num-seqs", type=int, default=64)
    ps.add_argument("--max-batched-tokens", type=int, default=2048)
    ps.add_argument("--kv-capacity-tokens", type=int, default=65536)
    ps.add_argument("--max-model-len", type=int, default=8192)
    ps.add_argument("--seed", type=int, default=0)
    ps.set_defaults(fn=cmd_sim)

    pl = sub.add_parser("loadgen", help="generate streaming load and record client-side ground truth")
    pl.add_argument("--url", required=True)
    pl.add_argument("--model", required=True)
    pl.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    pl.add_argument("--rate", type=float, help="open-loop Poisson arrivals per second")
    pl.add_argument("--concurrency", type=int, help="closed-loop workers")
    pl.add_argument("--duration", type=float, default=60.0)
    pl.add_argument("--stages", help='e.g. "r=2:60,r=4:60" or "c=8:30"')
    pl.add_argument("--prompt", default="lognormal:256:0.6:16:2048")
    pl.add_argument("--output", default="lognormal:128:0.6:8:512")
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--timeout", type=float, default=300.0)
    pl.add_argument("--no-ignore-eos", action="store_true")
    pl.add_argument("--no-stream", action="store_true")
    pl.add_argument("--extra-body", help="JSON merged into every request body")
    pl.add_argument("--out", help="per-request CSV")
    pl.add_argument("--gaps-out", help="per-request chunk gaps (JSON lines)")
    pl.add_argument("--summary-out", help="summary JSON")
    pl.set_defaults(fn=cmd_loadgen)

    pslo = sub.add_parser("slo", help="SLO tooling")
    slo_sub = pslo.add_subparsers(dest="slo_cmd", required=True)
    pg = slo_sub.add_parser("gen", help="generate recording + burn-rate alert rules from an SLO spec")
    pg.add_argument("--spec", required=True)
    pg.add_argument("--out", required=True)
    pg.add_argument("--check", action="store_true", help="fail if --out differs from the generated rules")
    pg.set_defaults(fn=cmd_slo_gen)

    pe = sub.add_parser("exp", help="experiments that write results/<host>/*.csv")
    exp_sub = pe.add_subparsers(dest="exp_cmd", required=True)

    def common(x, label):
        x.add_argument("--model", required=True)
        x.add_argument("--prompt", default="lognormal:256:0.6:16:1024")
        x.add_argument("--output", default="lognormal:128:0.6:8:512")
        x.add_argument("--prometheus", default="http://127.0.0.1:28090")
        x.add_argument("--tag", help="results subdirectory (default: host name)")
        x.add_argument("--label", default=label, help="output file stem")
        x.set_defaults(fn=cmd_exp)

    eo = exp_sub.add_parser("overhead", help="direct vs proxy vs proxy+tracing")
    common(eo, "overhead")
    eo.add_argument("--engine-url", default="http://127.0.0.1:28001")
    eo.add_argument("--provider", default="vllm")
    eo.add_argument("--levels", default="1,8,32")
    eo.add_argument("--duration", type=float, default=60)
    eo.add_argument("--repeats", type=int, default=3)
    eo.add_argument("--otlp", default="http://127.0.0.1:28318")
    eo.add_argument("--interleave", action="store_true", help="rotate targets per request within each run")
    eo.add_argument("--port-base", type=int, default=28010, help="ports for the two experiment proxies")

    ea = exp_sub.add_parser("agreement", help="client vs proxy vs Prometheus vs engine latency")
    common(ea, "agreement")
    ea.add_argument("--proxy-url", default="http://127.0.0.1:28000")
    ea.add_argument("--rate", type=float, required=True)
    ea.add_argument("--duration", type=float, default=300)
    ea.add_argument("--engine-label", default="engine histogram (vLLM)")

    es = exp_sub.add_parser("saturation", help="open-loop rate ramp")
    common(es, "saturation")
    es.add_argument("--proxy-url", default="http://127.0.0.1:28000")
    es.add_argument("--rates", required=True, help="comma-separated arrival rates (req/s)")
    es.add_argument("--stage-s", type=float, default=90)

    et = exp_sub.add_parser("trace", help="check proxy and engine spans join into one trace")
    common(et, "trace")
    et.add_argument("--proxy-url", default="http://127.0.0.1:28000")
    et.add_argument("--tempo-url", default="http://127.0.0.1:28200")
    et.add_argument("--n", type=int, default=3)

    ef = exp_sub.add_parser("faults", help="fault injection and alert detection times")
    common(ef, "faults")
    ef.add_argument("--engine-kind", choices=["sim", "vllm"], required=True)
    ef.add_argument("--engine-container", default="inferobs-engine-1")
    ef.add_argument("--engine-url", default="http://127.0.0.1:28001")
    ef.add_argument("--proxy-url", default="http://127.0.0.1:28000")
    ef.add_argument("--rate", type=float, required=True)
    ef.add_argument(
        "--scenarios", default="engine_crash,engine_hang,net_latency,stream_cut,overload,slow_decode"
    )
    ef.add_argument("--repeats", type=int, default=2)
    ef.add_argument("--max-fault-s", type=float, default=240)
    ef.add_argument("--cooldown-s", type=float, default=330)
    ef.add_argument("--warmup-s", type=float, default=330)

    pa = sub.add_parser("alert-sink", help="Alertmanager webhook receiver that timestamps notifications")
    pa.add_argument("--host", default="0.0.0.0")
    pa.add_argument("--port", type=int, default=9095)
    pa.add_argument("--out", help="append received alerts as JSON lines")
    pa.set_defaults(fn=cmd_alert_sink)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
