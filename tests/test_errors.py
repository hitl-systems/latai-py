# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Every code of schema/errors.json maps to an exception class with .code, .hint, .details."""

import pytest

import latai
from conftest import errors_json

ERRORS = errors_json()["errors"]
CATEGORY_BASES = {
    "E01": "PackError",
    "E02": "CompatError",
    "E03": "RequestError",
    "E04": "TrustError",
    "E05": "EngineError",
    "E06": "AccessError",
    "E07": "BusyError",
    "E08": "LifecycleError",
    "E09": "InternalError",
}


def test_the_table_covers_every_code():
    assert set(latai.ERROR_CLASSES) >= {e["code"] for e in ERRORS}
    assert len(ERRORS) >= 44


@pytest.mark.parametrize("entry", ERRORS, ids=lambda e: e["code"])
def test_each_code_has_a_class(entry):
    code = entry["code"]
    cls = latai.ERROR_CLASSES[code]
    assert issubclass(cls, latai.LataiError) and issubclass(cls, Exception)
    assert cls.__name__ == entry["name"]
    assert getattr(latai, entry["name"]) is cls
    base = getattr(latai, CATEGORY_BASES[entry["category"]])
    assert issubclass(cls, base)
    assert issubclass(base, latai.LataiError)

    # from a wire error
    wire = {"code": code, "message": "m of " + code, "hint": "h of " + code}
    err = latai.error_for(wire)
    assert type(err) is cls
    assert err.code == code
    assert err.message == "m of " + code
    assert err.hint == "h of " + code
    assert err.details in (None, {})
    assert err.category == entry["category"]
    assert err.retryable is entry["retryable"]
    assert code in str(err)

    # with details
    d = {"x": 1}
    err = latai.error_for({**wire, "details": d})
    assert err.details == d

    # from the bare code: the hint comes from the table and is never empty
    bare = latai.error_for(code)
    assert type(bare) is cls
    assert bare.code == code
    assert bare.hint and bare.hint.strip()


@pytest.mark.parametrize("entry", ERRORS, ids=lambda e: e["code"])
def test_default_hint_is_the_registry_hint(entry):
    err = latai.error_for(entry["code"])
    assert err.hint == entry["hint"]


def test_unknown_code_is_a_plain_latai_error_that_keeps_the_code():
    err = latai.error_for({"code": "E9999", "message": "future", "hint": "update the sdk"})
    assert type(err) is latai.LataiError
    assert err.code == "E9999" and err.hint == "update the sdk"


def test_errors_are_picklable_enough_to_log_and_compare():
    err = latai.error_for({"code": "E0316", "message": "no pack", "hint": "install it", "details": {"pack": "x"}})
    assert isinstance(err, latai.UnknownPack)
    assert err.details == {"pack": "x"}
    assert "E0316" in repr(err) or "E0316" in str(err)


def test_catching_by_category_and_by_code():
    with pytest.raises(latai.RequestError):
        raise latai.error_for("E0316")
    with pytest.raises(latai.UnknownPack):
        raise latai.error_for("E0316")
    with pytest.raises(latai.BusyError):
        raise latai.error_for("E0701")
    assert not issubclass(latai.UnknownPack, latai.BusyError)


def test_library_not_found_is_not_a_wire_code():
    assert "" not in latai.ERROR_CLASSES
    assert issubclass(latai.LibraryNotFoundError, latai.LataiError)
