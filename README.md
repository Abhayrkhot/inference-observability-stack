# inference-observability-stack

Observability for LLM inference serving, measured against a real engine.

> Work in progress. This README is replaced by a generated one (every number comes from files in
> `results/`) once the experiments land.

## What is here so far

- **`inferobs proxy`**: an OpenAI-compatible metering proxy that sits in front of vLLM, llama.cpp
  server or the simulator. It forwards streams unchanged and records what callers actually see:
  time to first chunk, gaps between chunks, mean time per output token, duration and token usage.
  Metric names follow the OpenTelemetry GenAI semantic conventions (client side, v1.41.0). Every
  request gets a span (W3C trace context is forwarded to the engine), a trace-id exemplar on its
  histogram samples and an optional JSON access-log line.
- **`inferobs sim`**: a continuous-batching engine simulator (FCFS admission, chunked prefill,
  KV-cache budget, linear step-cost model) with vLLM 0.34's metric names and buckets, plus runtime
  fault injection (slowdown, error rate, hang). It lets CI exercise dashboards and alerts without a GPU.
- **`inferobs loadgen`**: an open/closed-loop streaming load generator. It records per-request
  ground truth and measures latency from the scheduled arrival time, so client-side queueing cannot
  hide engine stalls (coordinated omission).

## Quick start

```bash
pip install -e '.[dev]'
inferobs sim --port 8000 &
inferobs proxy --upstream sim=http://127.0.0.1:8000 --port 9000 &
inferobs loadgen --url http://127.0.0.1:9000 --model sim-llm --rate 20 --duration 30
curl -s localhost:9000/metrics | grep gen_ai_client
pytest -q
```

## License

MIT
