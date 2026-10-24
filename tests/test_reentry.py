# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Re-entry rules: from inside an event callback the calls that wait for a run are E0303 and leave
everything as it was. Own module: one runtime at a time."""

import pytest

import latai
from conftest import SPEECH
from test_listen import OPEN, pcm  # noqa: F401  (the fixture)


def test_calls_from_the_callback_are_E0303_and_the_session_goes_on(speech_rt, pcm):
    codes = {}
    s = None

    def probe(ev):
        if codes:
            return None
        for name, call in {
            "run": lambda: speech_rt.run({"pack": SPEECH, "inputs": {"type": "text", "text": "a"}}),
            "push": lambda: s.push(b"\x00\x00"),
            "listen": lambda: speech_rt.listen({"pack": SPEECH}),
        }.items():
            try:
                call()
                codes[name] = "ok"
            except latai.LataiError as e:
                codes[name] = e.code
        # allowed from inside: cancel, info, list
        speech_rt.cancel("nobody")
        speech_rt.info()
        speech_rt.list_packs()
        return None

    s = speech_rt.listen({**OPEN}, on_event=probe)
    for i in range(0, len(pcm), 3200):
        s.push(pcm[i : i + 3200])
    s.close()
    assert codes, "the callback ran"
    assert codes == {k: "E0303" for k in codes}, codes
    assert not speech_rt.closed
    # the refused push did not end the stream: it still ends normally
    assert s.final is not None and s.final["finish"] == "stop"


def test_close_session_from_the_callback_is_E0303_and_the_handle_stays_open(speech_rt, pcm):
    out = {}

    def probe(ev):
        if out:
            return None
        try:
            s.close()
            out["code"] = "ok"
        except latai.LataiError as e:
            out["code"] = e.code
        out["closed_flag"] = s.closed
        return None

    s = speech_rt.listen({**OPEN}, on_event=probe)
    for i in range(0, len(pcm), 3200):
        s.push(pcm[i : i + 3200])
    assert out["code"] == "E0303"
    assert out["closed_flag"] is False, "a refused close must not mark the session closed"
    events = s.close()
    assert events[-1]["event"] == "final" and events[-1]["response"]["finish"] == "stop"


def test_close_runtime_from_the_callback_is_E0303_and_the_runtime_stays_open(speech_rt, pcm):
    """Last in the module: if the SDK marks the runtime closed anyway, nothing after it can run."""
    out = {}

    def probe(ev):
        if out:
            return None
        try:
            speech_rt.close()
            out["code"] = "ok"
        except latai.LataiError as e:
            out["code"] = e.code
        out["closed_flag"] = speech_rt.closed
        return None

    s = speech_rt.listen({**OPEN}, on_event=probe)
    try:
        for i in range(0, len(pcm), 3200):
            s.push(pcm[i : i + 3200])
        s.close()
    except latai.LataiError:
        pass
    assert out["code"] == "E0303"
    assert out["closed_flag"] is False, "a refused close must not mark the runtime closed"
    assert speech_rt.info()["version"]
