"""Shared helpers for experiments: Prometheus queries, metadata, subprocesses, CSV output."""

from __future__ import annotations

import csv
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def host_tag() -> str:
    return os.environ.get("IOS_HOST") or socket.gethostname().split(".")[0]


def results_dir(tag: str | None = None) -> Path:
    d = ROOT / "results" / (tag or host_tag())
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir(name: str) -> Path:
    d = ROOT / "runs" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def gpu_info() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        name, mem, drv = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"gpu": name, "gpu_memory": mem, "driver": drv}
    except Exception:
        return {}


def loadavg() -> float | None:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except OSError:
        return None


def write_meta(path: Path, **extra) -> None:
    meta = {
        "git_sha": os.environ.get("IOS_GIT_SHA", "unknown"),
        "host": host_tag(),
        "cpus": os.cpu_count(),
        "loadavg_end": loadavg(),
        "finished_unix": round(time.time(), 1),
        "python": sys.version.split()[0],
        **extra,
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


@dataclass
class Prom:
    base: str = "http://127.0.0.1:28090"

    def _get(self, path: str, **params) -> dict:
        url = f"{self.base}{path}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=30) as r:
            body = json.load(r)
        if body.get("status") != "success":
            raise RuntimeError(f"prometheus error for {params}: {body}")
        return body["data"]

    def query(self, expr: str, t: float | None = None) -> list[dict]:
        params = {"query": expr}
        if t is not None:
            params["time"] = f"{t:.3f}"
        return self._get("/api/v1/query", **params)["result"]

    def scalar(self, expr: str, t: float | None = None) -> float | None:
        r = self.query(expr, t)
        if not r:
            return None
        v = float(r[0]["value"][1])
        return None if v != v else v  # NaN -> None

    def range(self, expr: str, start: float, end: float, step: float = 1.0) -> list[dict]:
        return self._get(
            "/api/v1/query_range", query=expr, start=f"{start:.3f}", end=f"{end:.3f}", step=f"{step}"
        )["result"]

    def buckets(
        self, metric: str, selector: str, start: float, end: float
    ) -> tuple[list[float], list[float]]:
        """Cumulative bucket counts accumulated over [start, end] (increase), summed over series."""
        rng = max(1, round(end - start))
        res = self.query(f"sum by (le) (increase({metric}_bucket{{{selector}}}[{rng}s]))", end)
        pairs = []
        for r in res:
            le = r["metric"]["le"]
            pairs.append((float("inf") if le == "+Inf" else float(le), float(r["value"][1])))
        pairs.sort()
        bounds = [b for b, _ in pairs if b != float("inf")]
        counts = [c for _, c in pairs]
        return bounds, counts

    def wait_ready(self, secs: float = 120) -> None:
        deadline = time.time() + secs
        while True:
            try:
                urllib.request.urlopen(f"{self.base}/-/ready", timeout=3).read()
                return
            except Exception:
                if time.time() > deadline:
                    raise
                time.sleep(1)


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 10.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw and raw[:1] in b"[{" else raw.decode()


def wait_http(url: str, secs: float, ok_status=(200,)) -> float:
    """Block until url answers with an ok status; return seconds waited."""
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status in ok_status:
                    return time.time() - t0
        except Exception:
            pass
        if time.time() - t0 > secs:
            raise TimeoutError(f"{url} not ready after {secs}s")
        time.sleep(0.5)


@contextmanager
def background(cmd: str, log: Path, env: dict | None = None):
    """Run a shell command in the background for the duration of the block."""
    with log.open("w") as f:
        p = subprocess.Popen(
            cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env={**os.environ, **(env or {})},
            start_new_session=True,
        )
    try:
        yield p
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGINT)
            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()


def cpu_seconds(pid: int) -> float:
    """utime + stime of a process and its children, from /proc (Linux)."""
    clk = os.sysconf("SC_CLK_TCK")
    total = 0.0
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        total += (int(f[11]) + int(f[12]) + int(f[13]) + int(f[14])) / clk
    except (OSError, IndexError):
        pass
    try:
        for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split():
            total += cpu_seconds(int(child))
    except OSError:
        pass
    return total


def loadgen_cmd(
    url: str,
    model: str,
    *,
    out: Path | None = None,
    gaps: Path | None = None,
    summary: Path | None = None,
    **kw,
) -> str:
    parts = [sys.executable, "-m", "inferobs", "loadgen", "--url", url, "--model", model]
    for k, v in kw.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        parts += [flag] if v is True else [flag, str(v)]
    if out:
        parts += ["--out", str(out)]
    if gaps:
        parts += ["--gaps-out", str(gaps)]
    if summary:
        parts += ["--summary-out", str(summary)]
    return " ".join(shlex.quote(p) for p in parts)


def run_loadgen(url: str, model: str, run_dir: Path, name: str, **kw) -> dict:
    """Run the load generator to completion and return its summary."""
    out, summ = run_dir / f"{name}.csv", run_dir / f"{name}.summary.json"
    cmd = loadgen_cmd(url, model, out=out, summary=summ, **kw)
    subprocess.run(cmd, shell=True, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return json.loads(summ.read_text())


def docker(*args: str, check: bool = True, timeout: float = 120) -> str:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=check, timeout=timeout
    ).stdout.strip()
