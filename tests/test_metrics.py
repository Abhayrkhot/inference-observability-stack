"""Tests for the inference observability metrics collector."""

import os
import sys

import pytest

# Adjust path to import the module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa

from metrics import compute_percentile, MetricsCollector


def test_compute_percentile():
    data = [10, 20, 30, 40, 50]
    assert compute_percentile(data, 50) == 30
    # 90th percentile interpolates between 40 and 50 -> 46.0
    assert compute_percentile(data, 90) == pytest.approx(46.0)
    assert compute_percentile(data, 100) == 50
    # Edge case: empty list
    assert compute_percentile([], 50) == 0.0


def test_metrics_collector_record_and_percentiles():
    collector = MetricsCollector()
    # Record latencies (ms) and tokens
    collector.record(100.0, tokens=10)
    collector.record(200.0, tokens=20)
    collector.record(300.0, tokens=30)
    assert collector.total_requests == 3
    assert collector.total_tokens == 60
    # p95 of [100,200,300] is 290 (linear interpolation between 200 and 300)
    assert collector.p95_latency() == pytest.approx(290.0)
    # p99 is closer to the largest value
    assert collector.p99_latency() == pytest.approx(298.0, abs=1.0)
    # tokens per second should be positive
    assert collector.tokens_per_second() > 0


def test_to_prometheus_format():
    collector = MetricsCollector()
    collector.record(50.0, tokens=5)
    metrics_text = collector.to_prometheus()
    # Check that essential metrics appear
    assert "llm_requests_total" in metrics_text
    assert "llm_latency_ms" in metrics_text
    assert "llm_tokens_total" in metrics_text