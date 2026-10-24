# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Memory hygiene and thread safety on the real library."""

import gc
import threading

import pytest

import latai
from conftest import PACK, rss_kb, strip_timing, text_req

LOOP = 1000
RSS_CAP_KB = 64 * 1024  # generous: a leak of one response per call would be far above it


def counters():
    c = latai.debug_counters()
    return c["taken"], c["freed"]


def test_1000_runs_leave_no_buffer_out_and_rss_flat(rt):
    req = text_req("memory", 1)
    for _ in range(50):  # warm up: caches, allocator arenas
        rt.run(req)
    gc.collect()
    taken0, freed0 = counters()
    out0 = rt.library.outstanding_strings
    base = rss_kb()
    for i in range(LOOP):
        resp = rt.run(req)
        assert resp["finish"] in ("stop", "length")
    gc.collect()
    taken1, freed1 = counters()
    assert taken1 - taken0 >= LOOP, "every run takes a response string"
    assert taken1 - taken0 == freed1 - freed0, "every string taken was freed"
    assert rt.library.outstanding_strings == out0
    growth = rss_kb() - base
    assert growth < RSS_CAP_KB, f"RSS grew {growth} KiB over {LOOP} runs"


def test_error_paths_free_their_buffers_too(rt):
    gc.collect()
    taken0, freed0 = counters()
    for _ in range(200):
        with pytest.raises(latai.UnknownPack):
            rt.run(text_req("x", 1, pack="nope"))
        with pytest.raises(latai.BadRequest):
            rt.run("{bad")
        with pytest.raises(latai.UnknownPack):
            rt.inspect("nope")
    taken1, freed1 = counters()
    assert taken1 - taken0 == freed1 - freed0
    assert rt.library.outstanding_strings == 0 or rt.library.outstanding_strings == taken0 - freed0


def test_streams_free_everything_even_when_cancelled(rt):
    gc.collect()
    taken0, freed0 = counters()
    base = rss_kb()
    for i in range(100):
        s = rt.run_stream(text_req("s", 50))
        it = iter(s)
        next(it)
        s.close()
    for i in range(100):
        list(rt.run_stream(text_req("s", 2)))
    gc.collect()
    taken1, freed1 = counters()
    assert taken1 - taken0 == freed1 - freed0
    assert rss_kb() - base < RSS_CAP_KB
    # no worker threads left behind
    names = [t.name for t in threading.enumerate()]
    assert len(names) < 20, names


def test_two_threads_run_at_the_same_time(rt):
    results, errors = {}, []

    def work(name, text):
        try:
            results[name] = [strip_timing(rt.run(text_req(text, 4))) for _ in range(8)]
        except Exception as e:  # pragma: no cover
            errors.append((name, e))

    ts = [threading.Thread(target=work, args=(n, t)) for n, t in (("a", "alpha"), ("b", "beta"))]
    [t.start() for t in ts]
    [t.join(300) for t in ts]
    assert not errors, errors
    for name in ("a", "b"):
        assert len(results[name]) == 8
        assert all(r == results[name][0] for r in results[name])  # greedy: each thread repeatable
    assert results["a"][0] != results["b"][0]  # different prompts, different answers


def test_run_while_another_thread_streams(rt):
    out = {}

    def stream():
        out["events"] = list(rt.run_stream(text_req("streaming", 12)))

    t = threading.Thread(target=stream)
    t.start()
    resp = rt.run(text_req("blocking", 4))
    t.join(300)
    assert resp["finish"] in ("stop", "length")
    assert out["events"][-1]["event"] == "final"


def test_cancel_from_a_thread_while_main_waits_in_run(rt):
    def later():
        import time

        time.sleep(0.3)
        rt.cancel("py-block-1")

    t = threading.Thread(target=later)
    t.start()
    resp = rt.run(text_req("long", 500), request_id="py-block-1")
    t.join(60)
    assert resp["finish"] in ("cancelled", "length", "stop")


def test_info_from_many_threads(rt):
    errs = []

    def work():
        try:
            for _ in range(20):
                assert rt.info()["version"]
        except Exception as e:  # pragma: no cover
            errs.append(e)

    ts = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in ts]
    [t.join(120) for t in ts]
    assert not errs
