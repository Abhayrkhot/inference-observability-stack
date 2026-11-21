"""Incremental Server-Sent Events framing and OpenAI stream-chunk accounting."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import orjson


class SSEParser:
    """Split a byte stream into complete SSE events.

    ``feed`` returns ``(raw, data)`` pairs: ``raw`` is the event exactly as received (terminator
    included) so it can be forwarded unchanged, ``data`` is the joined ``data:`` payload or ``None``
    for events without one (comments, keep-alives).
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> list[tuple[bytes, bytes | None]]:
        self._buf += chunk
        out: list[tuple[bytes, bytes | None]] = []
        while True:
            end, sep = self._boundary()
            if end < 0:
                return out
            raw = bytes(self._buf[: end + sep])
            del self._buf[: end + sep]
            out.append((raw, _data_of(raw[:end])))

    def pending(self) -> bytes:
        """Bytes received after the last complete event (a truncated event at stream end)."""
        return bytes(self._buf)

    def _boundary(self) -> tuple[int, int]:
        best, sep = -1, 0
        for pat in (b"\n\n", b"\r\n\r\n", b"\r\r"):
            i = self._buf.find(pat)
            if i >= 0 and (best < 0 or i < best):
                best, sep = i, len(pat)
        return best, sep


def _data_of(block: bytes) -> bytes | None:
    parts = []
    for line in block.replace(b"\r\n", b"\n").split(b"\n"):
        if line.startswith(b"data:"):
            v = line[5:]
            parts.append(v[1:] if v.startswith(b" ") else v)
    return b"\n".join(parts) if parts else None


class EventKind(enum.Enum):
    CONTENT = "content"  # carries generated text, reasoning or tool-call deltas
    EMPTY = "empty"  # role-only first chunk, finish chunk without text, etc.
    USAGE_ONLY = "usage_only"  # the extra chunk produced by stream_options.include_usage
    DONE = "done"  # data: [DONE]
    ERROR = "error"  # an error object sent inside the stream
    OTHER = "other"  # not JSON / not an OpenAI chunk


@dataclass
class StreamMeter:
    """Per-request timing and token accounting for an OpenAI-compatible streaming response.

    Times are ``time.perf_counter()`` seconds supplied by the caller, so the meter itself is
    deterministic and testable.
    """

    t_start: float
    t_first: float | None = None
    t_last: float | None = None
    content_chunks: int = 0
    gaps: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reasons: list[str] = field(default_factory=list)
    response_id: str | None = None
    response_model: str | None = None
    done: bool = False
    stream_error: str | None = None

    def on_event(self, data: bytes | None, now: float) -> EventKind:
        if data is None:
            return EventKind.OTHER
        if data.strip() == b"[DONE]":
            self.done = True
            return EventKind.DONE
        try:
            obj = orjson.loads(data)
        except orjson.JSONDecodeError:
            return EventKind.OTHER
        if not isinstance(obj, dict):
            return EventKind.OTHER
        if "error" in obj and "choices" not in obj:
            err = obj["error"]
            self.stream_error = (
                str(err.get("type") or err.get("code") or "stream_error")
                if isinstance(err, dict)
                else "stream_error"
            )
            return EventKind.ERROR
        if self.response_id is None:
            self.response_id = obj.get("id")
            self.response_model = obj.get("model")
        usage = obj.get("usage")
        if isinstance(usage, dict):
            self.prompt_tokens = usage.get("prompt_tokens", self.prompt_tokens)
            self.completion_tokens = usage.get("completion_tokens", self.completion_tokens)
        choices = obj.get("choices") or []
        if not choices:
            return EventKind.USAGE_ONLY if isinstance(usage, dict) else EventKind.EMPTY
        has_content = False
        for c in choices:
            if not isinstance(c, dict):
                continue
            fr = c.get("finish_reason")
            if fr:
                self.finish_reasons.append(fr)
            delta = c.get("delta")
            if isinstance(delta, dict):
                if (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or delta.get("tool_calls")
                ):
                    has_content = True
            elif c.get("text"):
                has_content = True
        if not has_content:
            return EventKind.EMPTY
        if self.t_first is None:
            self.t_first = now
        else:
            self.gaps.append(now - self.t_last)  # type: ignore[operator]
        self.t_last = now
        self.content_chunks += 1
        return EventKind.CONTENT

    @property
    def ttfc(self) -> float | None:
        return None if self.t_first is None else self.t_first - self.t_start

    @property
    def output_tokens(self) -> int:
        """Usage-reported completion tokens, falling back to the number of content chunks."""
        return self.completion_tokens if self.completion_tokens is not None else self.content_chunks

    def tpot(self) -> float | None:
        """Mean time per output token after the first: (t_last - t_first) / (tokens - 1)."""
        n = self.output_tokens
        if self.t_first is None or self.t_last is None or n < 2:
            return None
        return (self.t_last - self.t_first) / (n - 1)
