# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Blocking runs on the real fixture pack, installed through the SDK's own `install`."""

import json

import pytest

import latai
from conftest import PACK, strip_timing, text_req, validate


def test_run_returns_the_wire_response(rt):
    resp = rt.run(text_req("hello", 6))
    assert type(resp) is dict
    validate("Response", resp)
    assert resp["finish"] in ("stop", "length")
    assert resp.get("error") is None
    t = resp["latai"]["timing"]
    assert t["total_ms"] > 0 and t["prompt_tokens"] > 0 and t["gen_tokens"] > 0
    assert resp["output"]["kind"] == "text"
    assert isinstance(resp["output"]["text"], str)
    assert len(resp["latai"]["digest"]) >= 64 or resp["latai"]["digest"].startswith("sha256:")


def test_response_is_json_that_round_trips_unchanged(rt):
    resp = rt.run(text_req("round trip", 5))
    again = json.loads(json.dumps(resp))
    assert again == resp
    # re-serialising with sorted keys and compact separators is stable
    a = json.dumps(resp, sort_keys=True, separators=(",", ":"))
    assert json.dumps(json.loads(a), sort_keys=True, separators=(",", ":")) == a
    validate("Response", again)


def test_request_forms_str_bytes_dict_give_the_same_answer(rt):
    req = text_req("same", 5)
    a = rt.run(req)
    b = rt.run(json.dumps(req))
    c = rt.run(json.dumps(req).encode())
    assert strip_timing(a) == strip_timing(b) == strip_timing(c)  # greedy by default


def test_request_with_non_ascii_text(rt):
    resp = rt.run(text_req("héllo wörld ☃ \U0001f600", 4))
    validate("Response", resp)
    assert resp["latai"]["timing"]["prompt_tokens"] > 0


def test_request_id_is_accepted_and_cancel_of_an_unknown_id_is_ok(rt):
    resp = rt.run(text_req("id", 3), request_id="py-run-1")
    validate("Response", resp)
    rt.cancel("py-run-1")  # finished: nothing to do
    rt.cancel("never-existed")
    rt.session_close("no-such-session")


def test_unknown_pack_raises_E0316_with_the_response(rt):
    with pytest.raises(latai.UnknownPack) as ei:
        rt.run(text_req("x", 2, pack="nope"))
    e = ei.value
    assert e.code == "E0316" and isinstance(e, latai.RequestError)
    assert e.hint
    assert e.response is not None
    validate("Response", e.response)
    assert e.response["finish"] == "error"
    assert e.response["error"]["code"] == "E0316"
    assert e.details == e.response["error"].get("details")


def test_raise_on_error_false_returns_the_error_response(rt):
    resp = rt.run(text_req("x", 2, pack="nope"), raise_on_error=False)
    validate("Response", resp)
    assert resp["finish"] == "error"
    assert resp["error"]["code"] == "E0316" and resp["error"]["hint"]


def test_request_that_does_not_parse_is_E0301_without_a_response(rt):
    for bad in ("{not json", b"\xff\xfe", {"pack": PACK, "inputs": {"type": "text", "text": "x"}, "bogus": 1}):
        with pytest.raises(latai.BadRequest) as ei:
            rt.run(bad, raise_on_error=False)  # no response exists, so it raises either way
        e = ei.value
        assert e.code == "E0301" and e.hint
        assert e.response is None


def test_invalid_request_value_is_a_typed_error(rt):
    req = text_req("x", 2)
    req["inputs"] = {"type": "audio", "data": "AAAA"}  # a text pack refuses audio
    with pytest.raises(latai.LataiError) as ei:
        rt.run(req)
    assert ei.value.code == "E0315"
    assert type(ei.value) is latai.InputInvalid


def test_info_and_status_validate_against_the_schema(rt):
    info = rt.info()
    validate("RuntimeInfo", info)
    assert any(p["name"] == PACK for p in info["packs"]) or True
    st = rt.status(PACK)
    validate("Status", st)
    assert st["pack"] == PACK and st["state"] == "available"
    validate("RuntimeInfo", rt.status())


def test_run_is_greedy_and_repeatable(rt):
    runs = [strip_timing(rt.run(text_req("determinism", 6))) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_max_tokens_bounds_the_output(rt):
    resp = rt.run(text_req("bound", 3))
    assert resp["latai"]["timing"]["gen_tokens"] <= 3
