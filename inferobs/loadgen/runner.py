"""Streaming load generator that records client-side ground truth.

Open-loop mode schedules Poisson arrivals and measures every latency from the *scheduled* arrival
time, so a backed-up client cannot hide server queueing (coordinated omission). Closed-loop mode runs
a fixed number of workers back to back. Stages let one run sweep several rates for saturation tests.
"""

from __future__ import annotations

import asyncio
import csv
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp
import orjson

from inferobs.analysis.stats import summarize
from inferobs.proxy.sse import EventKind, SSEParser, StreamMeter

# Common English words that are single tokens (with a leading space) in Llama/Qwen/GPT BPE vocabularies.
WORDS = (
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "was",
    "for",
    "on",
    "that",
    "with",
    "as",
    "by",
    "at",
    "from",
    "his",
    "her",
    "it",
    "an",
    "are",
    "this",
    "be",
    "or",
    "had",
    "not",
    "but",
    "have",
    "they",
    "which",
    "one",
    "you",
    "were",
    "all",
    "we",
    "she",
    "can",
    "their",
    "there",
    "been",
    "has",
    "more",
    "when",
    "will",
    "would",
    "who",
    "so",
    "no",
    "if",
    "out",
    "up",
    "into",
    "do",
    "time",
    "only",
    "new",
    "some",
    "could",
    "these",
    "two",
    "may",
    "first",
    "then",
    "other",
    "than",
    "its",
    "like",
    "over",
    "also",
    "after",
    "most",
    "made",
    "many",
    "before",
    "must",
    "through",
    "back",
    "years",
    "where",
    "much",
    "way",
    "well",
    "down",
    "should",
    "because",
    "each",
    "just",
    "those",
    "people",
    "how",
    "too",
    "little",
    "state",
    "good",
    "very",
    "make",
    "world",
    "still",
    "own",
    "see",
    "men",
    "work",
    "long",
    "here",
    "get",
    "both",
    "between",
    "life",
    "being",
    "under",
    "never",
    "day",
    "same",
    "another",
    "know",
    "while",
    "last",
    "might",
    "us",
    "great",
    "old",
    "year",
    "off",
    "come",
    "since",
    "against",
    "go",
    "came",
    "right",
    "used",
    "take",
    "three",
    "states",
    "himself",
    "few",
    "house",
    "use",
    "during",
    "without",
    "again",
    "place",
    "around",
    "however",
    "home",
    "small",
    "found",
    "thought",
    "went",
    "say",
    "part",
    "once",
    "general",
    "high",
    "upon",
    "school",
    "every",
    "don",
    "does",
    "got",
    "united",
    "left",
    "number",
    "course",
    "war",
    "until",
    "always",
    "away",
    "something",
    "fact",
    "though",
    "water",
    "less",
    "public",
    "put",
    "think",
    "almost",
    "hand",
    "enough",
    "far",
    "took",
    "head",
    "yet",
    "government",
    "system",
    "better",
    "set",
    "told",
    "nothing",
    "night",
    "end",
    "why",
    "called",
    "didn",
    "eyes",
    "find",
    "going",
    "look",
    "asked",
    "later",
    "knew",
    "point",
    "next",
    "program",
    "city",
    "business",
)


@dataclass
class Dist:
    kind: str
    a: float
    b: float = 0.0
    lo: int = 1
    hi: int = 1 << 20

    @staticmethod
    def parse(spec: str) -> Dist:
        """Parse ``fixed:N``, ``uniform:LO:HI`` or ``lognormal:MEAN:SIGMA[:LO:HI]`` (MEAN = mean)."""
        p = spec.split(":")
        if p[0] == "fixed":
            return Dist("fixed", float(p[1]))
        if p[0] == "uniform":
            return Dist("uniform", float(p[1]), float(p[2]))
        if p[0] == "lognormal":
            d = Dist("lognormal", float(p[1]), float(p[2]))
            if len(p) >= 5:
                d.lo, d.hi = int(p[3]), int(p[4])
            return d
        raise ValueError(f"bad distribution spec {spec!r}")

    def sample(self, rng: random.Random) -> int:
        if self.kind == "fixed":
            v = self.a
        elif self.kind == "uniform":
            v = rng.uniform(self.a, self.b)
        else:
            mu = math.log(self.a) - self.b**2 / 2
            v = rng.lognormvariate(mu, self.b)
        return int(min(max(round(v), self.lo, 1), self.hi))


@dataclass
class Stage:
    duration_s: float
    rate: float | None = None  # open loop arrivals/s
    concurrency: int | None = None  # closed loop workers


@dataclass
class LoadConfig:
    url: str
    model: str
    stages: list[Stage]
    endpoint: str = "chat"
    prompt: str = "lognormal:256:0.6:16:2048"
    output: str = "lognormal:128:0.6:8:512"
    seed: int = 0
    ignore_eos: bool = True
    stream: bool = True
    timeout_s: float = 300.0
    max_inflight: int = 4096
    extra_body: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)


@dataclass
class Result:
    id: int
    stage: int
    t_sched: float  # scheduled arrival, seconds since run start
    t_send: float
    status: int = 0
    error: str = ""
    ttfb_s: float | None = None
    ttft_s: float | None = None
    e2e_s: float | None = None
    tpot_s: float | None = None
    max_gap_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    requested_prompt_words: int = 0
    requested_output_tokens: int = 0
    chunks: int = 0
    gaps: list[float] = field(default_factory=list, repr=False)


CSV_FIELDS = [f for f in Result.__dataclass_fields__ if f != "gaps"]


def make_prompt(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


async def run_load(cfg: LoadConfig, on_result=None) -> tuple[list[Result], dict]:
    rng = random.Random(cfg.seed)
    pd, od = Dist.parse(cfg.prompt), Dist.parse(cfg.output)
    results: list[Result] = []
    path = "/v1/chat/completions" if cfg.endpoint == "chat" else "/v1/completions"
    url = cfg.url.rstrip("/") + path
    sem = asyncio.Semaphore(cfg.max_inflight)
    timeout = aiohttp.ClientTimeout(total=cfg.timeout_s)
    connector = aiohttp.TCPConnector(limit=0, force_close=False)
    ids = iter(range(1 << 62))
    t_run0 = time.perf_counter()
    wall0 = time.time()

    def body_for(n_prompt: int, n_out: int) -> bytes:
        text = make_prompt(rng, n_prompt)
        b: dict = {"model": cfg.model, "max_tokens": n_out, "stream": cfg.stream, "temperature": 0.0}
        if cfg.endpoint == "chat":
            b["messages"] = [{"role": "user", "content": text}]
        else:
            b["prompt"] = text
        if cfg.stream:
            b["stream_options"] = {"include_usage": True}
        if cfg.ignore_eos:
            b["ignore_eos"] = True
        b.update(cfg.extra_body)
        return orjson.dumps(b)

    async def one(session: aiohttp.ClientSession, stage: int, t_sched: float) -> None:
        n_prompt, n_out = pd.sample(rng), od.sample(rng)
        body = body_for(n_prompt, n_out)
        async with sem:
            t_send = time.perf_counter() - t_run0
            r = Result(
                id=next(ids),
                stage=stage,
                t_sched=t_sched,
                t_send=t_send,
                requested_prompt_words=n_prompt,
                requested_output_tokens=n_out,
            )
            t0_abs = t_run0 + t_sched  # latencies are measured from the scheduled arrival
            meter = StreamMeter(t_start=t0_abs)
            try:
                hdrs = {"Content-Type": "application/json", "X-Request-Id": f"lg-{r.id}", **cfg.headers}
                async with session.post(url, data=body, headers=hdrs) as resp:
                    r.status = resp.status
                    r.ttfb_s = time.perf_counter() - t0_abs
                    if resp.status != 200:
                        r.error = f"http_{resp.status}"
                        await resp.read()
                    elif not cfg.stream:
                        obj = orjson.loads(await resp.read())
                        u = obj.get("usage") or {}
                        r.prompt_tokens, r.completion_tokens = (
                            u.get("prompt_tokens"),
                            u.get("completion_tokens"),
                        )
                    else:
                        parser = SSEParser()
                        async for chunk in resp.content.iter_any():
                            now = time.perf_counter()
                            for _raw, data in parser.feed(chunk):
                                if meter.on_event(data, now) is EventKind.ERROR:
                                    r.error = meter.stream_error or "stream_error"
                        if not r.error and not (meter.done or meter.finish_reasons):
                            r.error = "stream_interrupted"
            except TimeoutError:
                r.error = "timeout"
            except aiohttp.ClientError as e:
                r.error = type(e).__name__
            r.e2e_s = time.perf_counter() - t0_abs
            if cfg.stream:
                r.ttft_s = meter.ttfc
                r.tpot_s = meter.tpot()
                r.gaps = meter.gaps
                r.max_gap_s = max(meter.gaps) if meter.gaps else None
                r.chunks = meter.content_chunks
                r.prompt_tokens = meter.prompt_tokens
                r.completion_tokens = meter.completion_tokens
            results.append(r)
            if on_result:
                on_result(r)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks: list[asyncio.Task] = []
        t_stage = 0.0
        for si, st in enumerate(cfg.stages):
            end = t_stage + st.duration_s
            if st.rate:
                t = t_stage
                while True:
                    t += rng.expovariate(st.rate)
                    if t >= end:
                        break
                    delay = t - (time.perf_counter() - t_run0)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    tasks.append(asyncio.create_task(one(session, si, t)))
            elif st.concurrency:

                async def worker(si=si, end=end):
                    while time.perf_counter() - t_run0 < end:
                        await one(session, si, time.perf_counter() - t_run0)

                workers = [asyncio.create_task(worker()) for _ in range(st.concurrency)]
                await asyncio.gather(*workers)
            delay = end - (time.perf_counter() - t_run0)
            if delay > 0:
                await asyncio.sleep(delay)
            t_stage = end
        if tasks:
            await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t_run0
    return results, summary(results, elapsed, wall0, cfg)


def summary(results: list[Result], elapsed: float, wall0: float, cfg: LoadConfig) -> dict:
    ok = [r for r in results if not r.error]
    out_tok = sum(r.completion_tokens or 0 for r in ok)
    gaps = [g for r in ok for g in r.gaps]
    return {
        "started_unix": round(wall0, 3),
        "elapsed_s": round(elapsed, 3),
        "requests": len(results),
        "errors": len(results) - len(ok),
        "error_kinds": _count(r.error for r in results if r.error),
        "output_tokens": out_tok,
        "output_tok_per_s": out_tok / elapsed if elapsed else 0.0,
        "req_per_s": len(ok) / elapsed if elapsed else 0.0,
        "ttft_s": summarize([r.ttft_s for r in ok if r.ttft_s is not None]),
        "tpot_s": summarize([r.tpot_s for r in ok if r.tpot_s is not None]),
        "itl_s": summarize(gaps),
        "e2e_s": summarize([r.e2e_s for r in ok if r.e2e_s is not None]),
        "config": {**asdict(cfg), "stages": [asdict(s) for s in cfg.stages]},
    }


def _count(xs) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def write_results(results: list[Result], csv_path: str | Path, gaps_path: str | Path | None = None) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in sorted(results, key=lambda r: r.id):
            d = asdict(r)
            d.pop("gaps")
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()})
    if gaps_path:
        with Path(gaps_path).open("wb") as f:
            for r in sorted(results, key=lambda r: r.id):
                f.write(orjson.dumps({"id": r.id, "gaps_us": [round(g * 1e6) for g in r.gaps]}) + b"\n")


def parse_stages(
    rate: float | None, concurrency: int | None, duration: float, stages: str | None
) -> list[Stage]:
    """``--stages "r=2:60,r=4:60,c=8:30"`` (r = Poisson rate, c = closed-loop workers) or a single stage."""
    if stages:
        out = []
        for part in stages.split(","):
            kind_val, dur = part.split(":")
            kind, val = kind_val.split("=")
            if kind == "r":
                out.append(Stage(float(dur), rate=float(val)))
            elif kind == "c":
                out.append(Stage(float(dur), concurrency=int(val)))
            else:
                raise ValueError(f"bad stage {part!r}")
        return out
    if rate:
        return [Stage(duration, rate=rate)]
    return [Stage(duration, concurrency=concurrency or 1)]
