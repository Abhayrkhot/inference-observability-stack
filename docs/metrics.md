# Metrics reference

## Proxy (`inferobs proxy`, job `proxy`)

Names follow the OpenTelemetry GenAI semantic conventions v1.41.0 (client side), converted to
Prometheus form. Common labels: `gen_ai_operation_name` (`chat`, `text_completion`),
`gen_ai_provider_name` (the upstream's name, e.g. `vllm`), `gen_ai_request_model` (the requested
model, or `_OTHER` if the upstream does not list it, which bounds cardinality).

| metric | type | what it measures |
|---|---|---|
| `gen_ai_client_operation_duration_seconds` | histogram | request received at the proxy to last byte sent; extra label `error_type` (empty on success) |
| `gen_ai_client_operation_time_to_first_chunk_seconds` | histogram | request received to first content-bearing chunk, observed when that chunk arrives (streaming only) |
| `gen_ai_client_operation_time_per_output_chunk_seconds` | histogram | every gap between consecutive content chunks, observed as it happens |
| `inferobs_time_per_output_token_seconds` | histogram | per request: (last chunk - first chunk) / (output tokens - 1), successful streams |
| `gen_ai_client_token_usage` | histogram | tokens per request from the upstream's usage report; label `gen_ai_token_type` = `input` / `output` |
| `inferobs_inflight_requests` | gauge | requests being proxied, per provider |
| `inferobs_upstream_up` | gauge | 1 if the last `/health` probe to the upstream succeeded |
| `inferobs_upstream_probe_duration_seconds` | gauge | duration of the last probe |

`error_type` values: an HTTP status (`500`, `503`, ...) passed through from the upstream, `timeout`
(no bytes from the upstream for `--idle-timeout`), `connection_error` (no response), `stream_interrupted`
(the stream ended without `[DONE]` or a finish reason), an error `type` sent inside the stream, or
`cancelled` (the client disconnected; excluded from the availability SLO).

Histograms carry a `trace_id` exemplar when the request's span is sampled. Exemplars are only
exposed in the OpenMetrics format, which Prometheus 3 negotiates by default.

### Bucket layouts (`--buckets`)

- `fine` (default): geometric buckets (ratio 1.2 to 1.25) from 0.5 ms (TPOT) / 2 ms (TTFC) / 5 ms
  (duration) plus round SLO thresholds (0.05, 0.1, 0.25, 0.5, 1, 2, 2.5, 5, 10 s ...). 51 to 57 buckets.
- `semconv`: the conventions' advisory boundaries (13 to 16 buckets).

The bucket-accuracy experiment measures the quantile error of each layout on real traffic.

## Engine (job `engine`)

vLLM 0.34 exposes `vllm:*` metrics with labels `model_name` and `engine`; the simulator exposes the
subset the dashboard and alerts use, with the same names and default buckets:
`vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`,
`vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, `vllm:request_success_total`,
`vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`,
`vllm:e2e_request_latency_seconds`, `vllm:request_queue_time_seconds`,
`vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds`.

## Recording rules

`deploy/prometheus/rules/recording.yaml`: per provider/model request rate, error ratio, token rates
and TTFC/TPOT/chunk-gap/duration quantiles over 1 m (prefix `provider_model:`), and the engine's
TTFT/ITL/queue quantiles and generation rate (prefix `model:`).

`deploy/prometheus/rules/slo.yaml` (generated): `slo:objective:ratio{slo}` and
`slo:sli_error:ratio_rate<window>{slo}` for every alert window plus the 30 d period.

## Spans

One `CLIENT` span per request named `{operation} {model}` with `gen_ai.operation.name`,
`gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.request.max_tokens`,
`gen_ai.request.temperature`, `gen_ai.response.id`, `gen_ai.response.model`,
`gen_ai.response.finish_reasons`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
`error.type`, an event `gen_ai.first_chunk`, and `inferobs.time_to_first_chunk_s`,
`inferobs.time_per_output_token_s`, `inferobs.max_chunk_gap_s`. The W3C `traceparent` is forwarded
to the engine, so vLLM's own span (started with `--otlp-traces-endpoint`) joins the same trace.

## Access log

With `--access-log PATH`, one JSON line per request: timestamp, `request_id` (from `X-Request-Id`),
trace ID, provider, model, status, `error_type`, token counts, chunk count, duration, TTFC, TPOT and
the largest chunk gap. `--access-log-gaps` adds every chunk gap in microseconds (used by the
agreement and bucket experiments).
