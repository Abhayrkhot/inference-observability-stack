import pytest
from aiohttp.test_utils import TestServer

from inferobs.proxy.app import Proxy, ProxyConfig
from inferobs.sim.engine import SimConfig
from inferobs.sim.server import SimServer

FAST = dict(step_base_s=0.002, step_per_seq_s=0.0001, prefill_per_token_s=0.00001)


@pytest.fixture
async def sim():
    s = SimServer(SimConfig(model="sim-llm", **FAST))
    server = TestServer(s.app())
    await server.start_server()
    s.url = str(server.make_url("")).rstrip("/")
    yield s
    await server.close()


async def start_proxy(upstream_url: str, **kw) -> tuple[Proxy, TestServer]:
    exporter = kw.pop("exporter", None)
    tp = None
    if exporter is not None:
        from inferobs.proxy.tracing import make_tracer_provider

        tp = make_tracer_provider("test", None, exporter=exporter)
    cfg = ProxyConfig(upstreams=[("sim", upstream_url)], health_interval=0.2, **kw)
    proxy = Proxy(cfg, tracer_provider=tp)
    server = TestServer(proxy.app())
    await server.start_server()
    proxy.url = str(server.make_url("")).rstrip("/")
    return proxy, server
