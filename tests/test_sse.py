import orjson
import pytest

from inferobs.proxy.sse import EventKind, SSEParser, StreamMeter


def chunk(content=None, finish=None, usage=None, choices=True, text=None):
    obj = {"id": "c1", "model": "m", "object": "chat.completion.chunk"}
    if choices:
        c = {"index": 0, "finish_reason": finish}
        if text is not None:
            c["text"] = text
        else:
            c["delta"] = {} if content is None else {"content": content}
        obj["choices"] = [c]
    else:
        obj["choices"] = []
    if usage:
        obj["usage"] = usage
    return b"data: " + orjson.dumps(obj) + b"\n\n"


def data(*args, **kw):
    """The JSON payload of a chunk, as SSEParser hands it to the meter."""
    return chunk(*args, **kw)[len(b"data: ") : -2]


def test_parser_handles_split_and_crlf_events():
    p = SSEParser()
    stream = chunk("a") + b": keep-alive\n\n" + chunk("b").replace(b"\n\n", b"\r\n\r\n") + b"data: [DONE]\n\n"
    events = []
    for i in range(0, len(stream), 7):  # arbitrary split points, including inside separators
        events += p.feed(stream[i : i + 7])
    assert [e[0] for e in events] == [
        chunk("a"),
        b": keep-alive\n\n",
        chunk("b").replace(b"\n\n", b"\r\n\r\n"),
        b"data: [DONE]\n\n",
    ]
    assert events[1][1] is None
    assert events[3][1] == b"[DONE]"
    assert p.pending() == b""


def test_parser_joins_multiline_data_and_keeps_truncated_tail():
    p = SSEParser()
    ev = p.feed(b'data: line1\ndata: line2\n\ndata: {"partial')
    assert ev == [(b"data: line1\ndata: line2\n\n", b"line1\nline2")]
    assert p.pending() == b'data: {"partial'


def test_meter_timing_tokens_and_kinds():
    m = StreamMeter(t_start=10.0)
    assert m.on_event(data(None), 10.05) is EventKind.EMPTY  # role-only chunk does not count as first
    assert m.on_event(data("hello"), 10.2) is EventKind.CONTENT
    assert m.on_event(data(" world"), 10.25) is EventKind.CONTENT
    assert m.on_event(data("!", finish="length"), 10.32) is EventKind.CONTENT
    usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert m.on_event(data(choices=False, usage=usage), 10.33) is EventKind.USAGE_ONLY
    assert m.on_event(b"[DONE]", 10.34) is EventKind.DONE
    assert m.ttfc == pytest.approx(0.2)
    assert m.gaps == pytest.approx([0.05, 0.07])
    assert m.tpot() == pytest.approx(0.06)  # (10.32 - 10.2) / (3 - 1)
    assert (m.prompt_tokens, m.output_tokens) == (7, 3)
    assert m.finish_reasons == ["length"] and m.done


def test_meter_completions_text_and_error_events():
    m = StreamMeter(t_start=0.0)
    assert m.on_event(data(text="x"), 1.0) is EventKind.CONTENT
    assert m.output_tokens == 1  # no usage: falls back to content chunks
    assert m.tpot() is None
    err = orjson.dumps({"error": {"type": "overloaded", "message": "busy"}})
    assert m.on_event(err, 1.5) is EventKind.ERROR
    assert m.stream_error == "overloaded"
    assert m.on_event(b"not json", 2.0) is EventKind.OTHER


def test_meter_counts_reasoning_and_tool_call_deltas():
    m = StreamMeter(t_start=0.0)
    r = orjson.dumps({"choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}}]})
    t = orjson.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0}]}}]})
    assert m.on_event(r, 0.1) is EventKind.CONTENT
    assert m.on_event(t, 0.2) is EventKind.CONTENT
    assert m.content_chunks == 2
