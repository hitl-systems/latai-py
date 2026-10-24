# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Pack-management calls and their typed errors."""

import pytest

import latai
from conftest import PACK, validate


def test_install_through_the_sdk_left_the_pack_active(rt):
    versions = rt.list_packs()
    assert isinstance(versions, list) and versions
    for v in versions:
        validate("PackVersion", v)
    mine = [v for v in versions if v["name"] == PACK]
    assert len(mine) == 1 and mine[0]["state"] == "active"


def test_inspect_by_name_and_by_file(rt, pack_file):
    by_name = rt.inspect(PACK)
    validate("PackInfo", by_name)
    by_file = rt.inspect(str(pack_file))
    validate("PackInfo", by_file)
    assert by_name["receipt"]["name"] == by_file["receipt"]["name"] == PACK if "receipt" in by_name else True


def test_inspect_unknown_pack_is_E0316(rt):
    with pytest.raises(latai.UnknownPack) as ei:
        rt.inspect("no-such-pack")
    assert ei.value.code == "E0316" and ei.value.hint


def test_status_of_the_pack(rt):
    st = rt.status(PACK)
    validate("Status", st)
    assert st["state"] == "available"
    assert st["active"] is not None


def test_activate_with_nothing_waiting_is_E0316(rt):
    # the only version is already the one that serves: there is none "waiting to be activated"
    with pytest.raises(latai.UnknownPack) as ei:
        rt.activate(PACK)
    assert ei.value.code == "E0316" and ei.value.hint


def test_activate_unknown_pack_is_E0316(rt):
    with pytest.raises(latai.LataiError) as ei:
        rt.activate("no-such-pack")
    assert ei.value.code == "E0316"


def test_rollback_without_a_retained_version_is_E0803(rt):
    with pytest.raises(latai.NoRollbackTarget) as ei:
        rt.rollback(PACK)
    e = ei.value
    assert e.code == "E0803" and isinstance(e, latai.LifecycleError) and e.hint


def test_pin_and_clear(rt):
    got = rt.pin(PACK, "channel:dev")
    assert got == "channel:dev"
    assert any(v.get("pin") == "channel:dev" for v in rt.list_packs() if v["name"] == PACK)
    assert rt.pin(PACK) is None
    assert rt.pin(PACK, None) is None


def test_pull_with_an_http_scheme_is_E0801(rt):
    with pytest.raises(latai.PullFailed) as ei:
        rt.pull("http://127.0.0.1:9/pack.gguf")
    e = ei.value
    assert e.code == "E0801" and isinstance(e, latai.LifecycleError)
    assert e.details and e.details.get("cause") == "scheme"
    assert e.hint


def test_pull_force_still_refuses_the_scheme(rt):
    with pytest.raises(latai.PullFailed):
        rt.pull("ftp://example.invalid/pack.gguf", force=True)


def test_stage_a_missing_file_is_a_typed_error(rt, tmp_path):
    with pytest.raises(latai.LataiError) as ei:
        rt.stage(str(tmp_path / "missing.gguf"))
    assert ei.value.code.startswith("E") and ei.value.hint


def test_stage_a_file_that_is_not_a_pack_is_a_typed_error(rt, tmp_path):
    bogus = tmp_path / "bogus.gguf"
    bogus.write_bytes(b"GGUF" + b"\x00" * 64)
    (tmp_path / "bogus.gguf.minisig").write_text("untrusted comment: x\nAAAA\n")
    with pytest.raises(latai.LataiError) as ei:
        rt.stage(str(bogus))
    assert ei.value.code in ("E0101", "E0102", "E0401", "E0103")


def test_install_again_without_activate_returns_receipts(rt, pack_file):
    receipts = rt.install(str(pack_file), activate=False)
    assert isinstance(receipts, list) and len(receipts) == 1
    validate("StageReceipt", receipts[0])
    # the active one is still the same pack
    assert [v["state"] for v in rt.list_packs() if v["name"] == PACK].count("active") == 1


def test_stage_returns_a_receipt(rt, pack_file):
    r = rt.stage(str(pack_file))
    validate("StageReceipt", r)
    assert r["name"] == PACK


def test_pack_calls_free_their_strings(rt):
    before = rt.library.outstanding_strings
    for _ in range(20):
        rt.list_packs()
        rt.status(PACK)
        rt.info()
        with pytest.raises(latai.UnknownPack):
            rt.inspect("nope")
    assert rt.library.outstanding_strings == before
