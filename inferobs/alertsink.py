"""Alertmanager webhook receiver that records when each notification arrived (for detection-time tests)."""

from __future__ import annotations

import time

import orjson
from aiohttp import web


class AlertSink:
    def __init__(self, path: str | None) -> None:
        self.path = path
        self.records: list[dict] = []

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/alerts", self.receive)
        app.router.add_get("/alerts", self.list)
        app.router.add_delete("/alerts", self.clear)
        app.router.add_get("/health", lambda _r: web.Response(text="ok"))
        return app

    async def receive(self, request: web.Request) -> web.Response:
        now = time.time()
        body = await request.json()
        for a in body.get("alerts", []):
            rec = {
                "received_unix": now,
                "status": a.get("status"),
                "alertname": a.get("labels", {}).get("alertname"),
                "labels": a.get("labels", {}),
                "starts_at": a.get("startsAt"),
                "ends_at": a.get("endsAt"),
            }
            self.records.append(rec)
            if self.path:
                with open(self.path, "ab") as f:
                    f.write(orjson.dumps(rec) + b"\n")
        return web.Response(text="ok")

    async def list(self, _r: web.Request) -> web.Response:
        return web.Response(body=orjson.dumps(self.records), content_type="application/json")

    async def clear(self, _r: web.Request) -> web.Response:
        self.records.clear()
        return web.Response(text="cleared")


def run(host: str, port: int, path: str | None) -> None:
    web.run_app(AlertSink(path).app(), host=host, port=port, access_log=None, print=None)
