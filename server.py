"""HTTP server exposing LLM inference metrics.

This script starts a simple HTTP server that serves Prometheus metrics
collected by :class:`~metrics.MetricsCollector`.  Optionally it can run a
simulation to generate synthetic requests before starting the server.
"""

from __future__ import annotations

import argparse
import http.server
import json
import random
import socketserver
import threading
import time
from typing import Tuple

from metrics import MetricsCollector


def simulate_requests(collector: MetricsCollector, num_requests: int, seed: int = 0) -> None:
    """Generate synthetic requests and record their metrics.

    Each request has a random token count and latency.  Latency is sampled
    such that TTFT and output length contributions vary.
    """
    rng = random.Random(seed)
    for _ in range(num_requests):
        # Random tokens between 5 and 100
        tokens = rng.randint(5, 100)
        # Latency model: TTFT ~ 50–150 ms + per‑token 2–4 ms
        ttf = rng.uniform(50.0, 150.0)
        per_token = rng.uniform(2.0, 4.0)
        latency_ms = ttf + tokens * per_token
        collector.record(latency_ms=latency_ms, tokens=tokens, error=False)
        # Simulate real time delay so throughput calculation is meaningful
        time.sleep(0.005)


class MetricsRequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that exposes metrics on /metrics."""

    collector: MetricsCollector  # Set this attribute before serving

    def do_GET(self) -> None:  # type: ignore
        if self.path == "/metrics":
            body = self.collector.to_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int, collector: MetricsCollector) -> None:
    """Run the HTTP server indefinitely on the given port."""
    handler_class = MetricsRequestHandler
    handler_class.collector = collector
    with socketserver.TCPServer(("", port), handler_class) as httpd:
        print(f"Serving metrics on port {port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Shutting down server.")


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a simple metrics server for LLM inference observability.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve metrics on")
    parser.add_argument("--simulate", action="store_true", help="Generate synthetic requests before serving")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of synthetic requests to simulate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for simulation")
    args = parser.parse_args(argv)
    collector = MetricsCollector()
    if args.simulate:
        print(f"Simulating {args.num_requests} requests...")
        simulate_requests(collector, num_requests=args.num_requests, seed=args.seed)
        print("Simulation complete.")
    run_server(args.port, collector)


if __name__ == "__main__":
    cli()