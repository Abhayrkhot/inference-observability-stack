"""Prometheus metrics for the metering proxy.

Names follow the OpenTelemetry GenAI semantic conventions (client side, v1.41.0) converted to
Prometheus form (dots to underscores, unit suffix). The proxy is a client of the inference engine,
so it uses the ``gen_ai.client.*`` instruments; the engine's own ``gen_ai.server.*`` style metrics
(vLLM's ``vllm:*``) are scraped separately and compared against these.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Gauge, Histogram

# Advisory bucket boundaries from the semantic conventions.
SEMCONV_DURATION = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92)
SEMCONV_TTFT = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
SEMCONV_TPOT = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5)
SEMCONV_TOKENS = (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864)


def geometric(lo: float, hi: float, factor: float) -> tuple[float, ...]:
    out, v = [], lo
    while v <= hi * (1 + 1e-9):
        out.append(round(v, 6))
        v *= factor
    return tuple(out)


# Round thresholds an SLO is likely to use. A latency SLI needs its threshold to be a bucket bound.
ROUND_SECONDS = (0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 30.0, 60.0)


def with_round_bounds(base: tuple[float, ...], tol: float = 0.06) -> tuple[float, ...]:
    """Add ROUND_SECONDS inside base's range, dropping base bounds within ``tol`` of a round one."""
    lo, hi = base[0], base[-1]
    rounds = [r for r in ROUND_SECONDS if lo <= r <= hi]
    kept = [b for b in base if all(abs(b - r) / r > tol for r in rounds)]
    return tuple(sorted(set(kept) | set(rounds)))


# "fine" layout: geometric buckets (ratio 1.2 to 1.25) sized for small/fast models where decode steps
# take a few ms, plus round SLO thresholds. Compared with the semconv layout in the bucket-accuracy
# experiment.
FINE_DURATION = with_round_bounds(geometric(0.005, 120.0, 1.25))
FINE_TTFT = with_round_bounds(geometric(0.002, 30.0, 1.2))
FINE_TPOT = with_round_bounds(geometric(0.0005, 5.0, 1.2))

LAYOUTS = {
    "semconv": {"duration": SEMCONV_DURATION, "ttft": SEMCONV_TTFT, "tpot": SEMCONV_TPOT},
    "fine": {"duration": FINE_DURATION, "ttft": FINE_TTFT, "tpot": FINE_TPOT},
}

BASE_LABELS = ("gen_ai_operation_name", "gen_ai_provider_name", "gen_ai_request_model")


class ProxyMetrics:
    def __init__(self, registry: CollectorRegistry | None = None, layout: str = "fine") -> None:
        if layout not in LAYOUTS:
            raise ValueError(f"unknown bucket layout {layout!r}; choose from {sorted(LAYOUTS)}")
        b = LAYOUTS[layout]
        self.layout = layout
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.duration = Histogram(
            "gen_ai_client_operation_duration_seconds",
            "GenAI operation duration measured at the proxy, request received to last byte sent.",
            (*BASE_LABELS, "error_type"),
            buckets=b["duration"],
            registry=r,
        )
        self.ttfc = Histogram(
            "gen_ai_client_operation_time_to_first_chunk_seconds",
            "Request received at the proxy to first content-bearing chunk, observed when it arrives.",
            BASE_LABELS,
            buckets=b["ttft"],
            registry=r,
        )
        self.tpoc = Histogram(
            "gen_ai_client_operation_time_per_output_chunk_seconds",
            "Gap between consecutive content-bearing chunks (one observation per gap).",
            BASE_LABELS,
            buckets=b["tpot"],
            registry=r,
        )
        self.tpot = Histogram(
            "inferobs_time_per_output_token_seconds",
            "Per-request mean time per output token after the first: (t_last - t_first) / (tokens - 1).",
            BASE_LABELS,
            buckets=b["tpot"],
            registry=r,
        )
        self.tokens = Histogram(
            "gen_ai_client_token_usage",
            "Tokens per operation, by gen_ai_token_type (input or output), from upstream usage.",
            (*BASE_LABELS, "gen_ai_token_type"),
            buckets=SEMCONV_TOKENS,
            registry=r,
        )
        self.inflight = Gauge(
            "inferobs_inflight_requests",
            "Requests currently being proxied.",
            ("gen_ai_provider_name",),
            registry=r,
        )
        self.upstream_up = Gauge(
            "inferobs_upstream_up",
            "1 if the upstream health probe succeeded on the last attempt.",
            ("gen_ai_provider_name",),
            registry=r,
        )
        self.upstream_probe = Gauge(
            "inferobs_upstream_probe_duration_seconds",
            "Duration of the last upstream health probe.",
            ("gen_ai_provider_name",),
            registry=r,
        )
