# SPDX-License-Identifier: Apache-2.0 OR MIT
"""run_stream: event order, one terminal event, cancel, errors."""

import json
import threading
import time

import pytest

import latai
from conftest import PACK, strip_timing, text_req


def terminals(events):
    return [e for e in events if e["event"] in ("final", "error")]


def test_events_are_deltas_then_exactly_one_final(rt):
    events = list(rt.run_stream(text_req("stream me", 8)))
    assert events, "events"
    *deltas, last = events
    assert last["event"] == "final"
    assert len(terminals(events)) == 1
    assert deltas and all(e["event"] == "delta" for e in deltas)
    for d in deltas:
        assert isinstance(d["text"], str) and d["ms"] >= 0
    text = "".join(d["text"] for d in deltas)
    assert text == last["response"]["output"]["text"]
    assert last["response"]["finish"] in ("stop", "length")


def test_stream_final_equals_the_blocking_response(rt):
    req = text_req("compare", 6)
    blocking = rt.run(req)
    events = list(rt.run_stream(req))
    assert strip_timing(events[-1]["response"]) == strip_timing(blocking)


def test_events_round_trip_as_json(rt):
    for ev in rt.run_stream(text_req("json", 4)):
        assert json.loads(json.dumps(ev)) == ev


def test_stream_object_exposes_request_id_and_final(rt):
    s = rt.run_stream(text_req("ids", 4), request_id="py-stream-1")
    assert s.request_id == "py-stream-1"
    events = list(s)
    assert s.final == events[-1]["response"]
    s.close()  # closing a finished stream is harmless
    s.close()


def test_generated_request_id_when_none_given(rt):
    s = rt.run_stream(text_req("ids", 2))
    try:
        assert isinstance(s.request_id, str) and s.request_id
    finally:
        s.close()


def test_closing_the_iterator_cancels_the_run(rt):
    s = rt.run_stream(text_req("a long answer", 400))
    it = iter(s)
    first = next(it)
    assert first["event"] == "delta"
    s.close()
    # the run ends cancelled and the runtime is free again for the next request
    t0 = time.monotonic()
    resp = rt.run(text_req("after cancel", 3))
    assert resp["finish"] in ("stop", "length")
    assert time.monotonic() - t0 < 60
    if s.final is not None:
        assert s.final["finish"] == "cancelled"


def test_context_manager_cancels_on_early_exit(rt):
    n = 0
    with rt.run_stream(text_req("context", 400)) as s:
        for ev in s:
            n += 1
            if n == 2:
                break
    assert n == 2
    assert rt.run(text_req("ok", 2))["finish"] in ("stop", "length")


def test_cancel_by_request_id_ends_with_finish_cancelled(rt):
    s = rt.run_stream(text_req("cancel by id", 400), request_id="py-cancel-1")
    it = iter(s)
    assert next(it)["event"] == "delta"
    rt.cancel("py-cancel-1")
    rest = list(it)  # a cancelled run is a final, not an exception
    assert rest[-1]["event"] == "final"
    assert rest[-1]["response"]["finish"] == "cancelled"
    assert len(terminals(rest)) == 1


def test_stream_cancel_method(rt):
    s = rt.run_stream(text_req("cancel method", 400))
    it = iter(s)
    assert next(it)["event"] == "delta"
    s.cancel()
    rest = list(it)
    assert rest and rest[-1]["event"] == "final"
    assert rest[-1]["response"]["finish"] == "cancelled"


def test_cancel_from_another_thread(rt):
    s = rt.run_stream(text_req("thread cancel", 400), request_id="py-cancel-2")
    it = iter(s)
    assert next(it)["event"] == "delta"
    t = threading.Thread(target=lambda: rt.cancel("py-cancel-2"))
    t.start()
    t.join(30)
    rest = list(it)
    assert rest[-1]["response"]["finish"] == "cancelled"


def test_slow_consumer_with_a_tiny_queue_loses_no_event(rt):
    req = text_req("backpressure", 10)
    want = [e for e in rt.run_stream(req, queue_size=64)]
    got = []
    for ev in rt.run_stream(req, queue_size=1):
        time.sleep(0.01)
        got.append(ev)
    assert [e["event"] for e in got] == [e["event"] for e in want]
    assert [e.get("text") for e in got[:-1]] == [e.get("text") for e in want[:-1]]


def test_unknown_pack_streams_one_terminal_event_and_does_not_raise(rt):
    seen = list(rt.run_stream(text_req("x", 2, pack="nope")))
    assert len(terminals(seen)) == 1, seen
    last = seen[-1]
    err = last["error"] if last["event"] == "error" else last["response"]["error"]
    assert err["code"] == "E0316" and err["hint"]


def test_raise_on_error_true_raises_after_the_terminal_event(rt):
    seen = []
    with pytest.raises(latai.UnknownPack) as ei:
        for ev in rt.run_stream(text_req("x", 2, pack="nope"), raise_on_error=True):
            seen.append(ev)
    assert len(terminals(seen)) == 1, seen
    assert ei.value.code == "E0316"


def test_bad_json_is_one_error_event_E0301(rt):
    seen = list(rt.run_stream("{not json"))
    assert [e["event"] for e in seen] == ["error"]
    assert seen[0]["error"]["code"] == "E0301"
    with pytest.raises(latai.BadRequest):
        list(rt.run_stream("{not json", raise_on_error=True))


def test_many_short_streams_each_have_one_terminal(rt):
    for i in range(15):
        events = list(rt.run_stream(text_req(f"n{i}", 2)))
        assert len(terminals(events)) == 1 and events[-1]["event"] == "final"
