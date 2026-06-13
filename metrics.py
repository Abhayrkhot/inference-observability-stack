"""Metrics collection for LLM inference observability.

This module defines a lightweight metrics collector that records request
counts, error counts, per‑request latencies and total tokens.  It computes
percentiles and throughput and exposes the metrics in Prometheus exposition
format.
"""

from __future__ import annotations

import json
import math
import time
from typing import Iterable, List


def compute_percentile(latencies: List[float], percentile: float) -> float:
    """Compute a percentile value from a list of latencies.

    Parameters
    ----------
    latencies: List[float]
        A list of latencies in milliseconds.
    percentile: float
        The percentile to compute (0–100).

    Returns
    -------
    float
        The latency at the given percentile, or 0 if the list is empty.
    """
    if not latencies:
        return 0.0
    sorted_lat = sorted(latencies)
    k = (len(sorted_lat) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_lat[int(k)]
    d0 = sorted_lat[int(f)] * (c - k)
    d1 = sorted_lat[int(c)] * (k - f)
    return d0 + d1


class MetricsCollector:
    """Collects metrics for LLM inference requests."""

    def __init__(self) -> None:
        self.total_requests: int = 0
        self.total_errors: int = 0
        self.latencies_ms: List[float] = []
        self.total_tokens: int = 0
        self.start_time: float = time.time()

    def record(self, latency_ms: float, tokens: int, error: bool = False) -> None:
        """Record a request.

        Parameters
        ----------
        latency_ms: float
            End‑to‑end latency of the request in milliseconds.
        tokens: int
            Number of tokens generated.
        error: bool, optional
            Whether the request resulted in an error.
        """
        self.total_requests += 1
        self.latencies_ms.append(latency_ms)
        self.total_tokens += tokens
        if error:
            self.total_errors += 1

    def p95_latency(self) -> float:
        return compute_percentile(self.latencies_ms, 95.0)

    def p99_latency(self) -> float:
        return compute_percentile(self.latencies_ms, 99.0)

    def tokens_per_second(self) -> float:
        elapsed = max(time.time() - self.start_time, 1e-6)
        return self.total_tokens / elapsed

    def to_prometheus(self) -> str:
        """Render metrics in Prometheus exposition format."""
        lines = []
        # Counters
        lines.append("# HELP llm_requests_total Total number of requests processed")
        lines.append("# TYPE llm_requests_total counter")
        lines.append(f"llm_requests_total {self.total_requests}")
        lines.append("# HELP llm_errors_total Total number of failed requests")
        lines.append("# TYPE llm_errors_total counter")
        lines.append(f"llm_errors_total {self.total_errors}")
        # Latency summary
        lines.append("# HELP llm_latency_ms Latency in milliseconds")
        lines.append("# TYPE llm_latency_ms summary")
        lines.append(f"llm_latency_ms_count {self.total_requests}")
        lines.append(f"llm_latency_ms_sum {sum(self.latencies_ms)}")
        # Percentile samples (approximate summary representation)
        lines.append(f"llm_latency_ms{{quantile=\"0.95\"}} {self.p95_latency()}")
        lines.append(f"llm_latency_ms{{quantile=\"0.99\"}} {self.p99_latency()}")
        # Token counters
        lines.append("# HELP llm_tokens_total Total tokens generated")
        lines.append("# TYPE llm_tokens_total counter")
        lines.append(f"llm_tokens_total {self.total_tokens}")
        # Tokens per second gauge
        lines.append("# HELP llm_tokens_per_second Tokens processed per second")
        lines.append("# TYPE llm_tokens_per_second gauge")
        lines.append(f"llm_tokens_per_second {self.tokens_per_second()}")
        return "\n".join(lines) + "\n"