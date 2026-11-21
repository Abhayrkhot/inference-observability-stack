"""OpenTelemetry tracing setup for the proxy (OTLP/HTTP export)."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

import inferobs


def make_tracer_provider(
    service_name: str,
    otlp_endpoint: str | None,
    sample_ratio: float = 1.0,
    exporter: SpanExporter | None = None,
) -> TracerProvider | None:
    """Return a TracerProvider exporting to ``otlp_endpoint`` (or ``exporter``), or None if neither."""
    if not otlp_endpoint and exporter is None:
        return None
    set_global_textmap(TraceContextTextMapPropagator())
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name, "service.version": inferobs.__version__}),
        sampler=ParentBased(TraceIdRatioBased(sample_ratio)),
    )
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        url = otlp_endpoint.rstrip("/")  # type: ignore[union-attr]
        if not url.endswith("/v1/traces"):
            url += "/v1/traces"
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=url), max_queue_size=8192, max_export_batch_size=512)
        )
    return provider


def tracer_from(provider: TracerProvider | None) -> trace.Tracer:
    if provider is None:
        return trace.NoOpTracer()
    return provider.get_tracer("inferobs.proxy", inferobs.__version__)
