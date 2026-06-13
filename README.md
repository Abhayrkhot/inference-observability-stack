# Inference Observability Stack

## Overview

Observability is critical for operating large language model (LLM) inference
systems in production.  Inference latency — the time from sending a request to
receiving the first token or the full response — directly shapes user
experience, cost and scalability【341601092433499†L225-L231】.  Throughput
measures how much work the system does per unit time, but high throughput
without latency insights can hide poor user responsiveness【341601092433499†L244-L272】.
Effective observability means tracking **time to first token (TTFT)**, **end‑to‑end
latency**, **tokens per second (TPS)**, **requests per second (RPS)**, and
latency percentiles (P90, P95, P99)【341601092433499†L252-L254】.  Reporting only
averages hides tail latency that users actually experience【341601092433499†L252-L254】.

This project implements a minimal **observability stack** for LLM inference.
It defines a metrics collector that records request counts, error counts and
latencies, computes percentiles and throughput, and exposes them in the
Prometheus exposition format via a simple HTTP server.  It also includes a
simulation harness that generates synthetic request latencies and updates
metrics in real time.

## Why this matters

LLM inference workloads have complex performance characteristics.  The prefill
phase dominates TTFT, while output length dominates end‑to‑end latency【341601092433499†L262-L281】.
Queueing and batching trade throughput for latency【341601092433499†L269-L347】.  Without
instrumentation, teams cannot pinpoint bottlenecks or measure the impact of
optimizations.  Observability surfaces tail latency (P95, P99) and trends
tokens per second so that operators can tune batch sizes, scheduling and
hardware choices to achieve low latency and high throughput.

## Features

* **Metrics collector:** Record the number of requests, errors, total tokens and
  per‑request latencies.  Compute TTFT, end‑to‑end latency, percentiles and
  throughput.
* **Prometheus exporter:** Serve metrics on an HTTP endpoint in the
  [Prometheus exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/)
  for integration with Grafana dashboards.
* **Simulation harness:** Generate synthetic request latencies and update
  metrics periodically to demonstrate metric collection in action.
* **Unit tests:** Validate percentile calculations and metrics exposure.

## Installation

This project requires Python 3.8+.  Install the optional [`prometheus_client`]
package if you wish to use Prometheus helper utilities.  For development
dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

Run the metrics server and a simulation of 200 synthetic requests:

```bash
python server.py --simulate --num-requests 200 --seed 0
```

Visit `http://localhost:8000/metrics` to see metrics in Prometheus format.
Example output:

```
# HELP llm_requests_total Total number of requests processed
# TYPE llm_requests_total counter
llm_requests_total 200
# HELP llm_errors_total Total number of failed requests
# TYPE llm_errors_total counter
llm_errors_total 0
# HELP llm_latency_ms Latency in milliseconds
# TYPE llm_latency_ms summary
llm_latency_ms_count 200
llm_latency_ms_sum 11000.0
llm_latency_ms{quantile="0.95"} 80.0
llm_latency_ms{quantile="0.99"} 90.0
# HELP llm_tokens_total Total tokens generated
# TYPE llm_tokens_total counter
llm_tokens_total 12000
# HELP llm_tokens_per_second Tokens processed per second
# TYPE llm_tokens_per_second gauge
llm_tokens_per_second 450.0
```

## Running the tests

Execute the test suite with:

```bash
pytest
```

The tests cover percentile calculation and metrics formatting.

## Limitations and future work

This observability stack is simplified.  It does not record per‑token
interarrival times or queue depth.  In production, you would integrate with
frameworks like `prometheus_client` or `OpenTelemetry` and include logs and
traces.  Real latency measurement should separate TTFT and inter‑token
latency and capture queue wait times.  Extending the simulation to model
batching and continuous scheduling would provide deeper insights.

## Resume bullet

> Built a lightweight LLM observability stack tracking TTFT, end‑to‑end
> latency, percentiles (P95, P99), requests per second and tokens per
> second; exported metrics via HTTP in Prometheus format and validated
> percentile calculations【341601092433499†L252-L254】.
