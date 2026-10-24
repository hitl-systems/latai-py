# SPDX-License-Identifier: Apache-2.0 OR MIT
"""open/close lifecycle on the real library (one runtime at a time: every test here opens and
closes its own, and nothing else holds one open in this module)."""

import threading

import pytest

import latai
from conftest import lib_file, text_req, validate


def test_library_wrapper_reports_version_and_streams(lib):
    library = latai.Library(lib)
    assert library.api_version >= 2
    assert library.has_streams is True
    assert library.outstanding_strings == 0


def test_open_info_close(text_data, lib):
    rt = latai.Runtime.open(data_dir=str(text_data), lib=lib)
    try:
        assert rt.closed is False
        info = rt.info()
        validate("RuntimeInfo", info)
        assert info["data_dir"].rstrip("/") == str(text_data).rstrip("/")
        assert [p["name"] for p in info["packs"]] in ([], ["x4r-l8"])
    finally:
        rt.close()
    assert rt.closed is True


def test_constructor_and_open_are_the_same(text_data, lib):
    rt = latai.Runtime(data_dir=str(text_data), lib=lib)
    try:
        assert rt.closed is False
    finally:
        rt.close()


def test_double_close_is_harmless(text_data, lib):
    rt = latai.Runtime.open(data_dir=str(text_data), lib=lib)
    rt.close()
    rt.close()
    assert rt.closed


def test_use_after_close_is_a_typed_error(text_data, lib):
    rt = latai.Runtime.open(data_dir=str(text_data), lib=lib)
    rt.close()
    calls = [
        lambda: rt.run(text_req()),
        lambda: rt.info(),
        lambda: rt.status(),
        lambda: rt.list_packs(),
        lambda: rt.cancel("x"),
        lambda: rt.session_close("x"),
        lambda: rt.inspect("x4r-l8"),
        lambda: rt.run_stream(text_req()).__iter__().__next__(),
    ]
    for call in calls:
        with pytest.raises(latai.LataiError) as ei:
            call()
        e = ei.value
        # the stale handle is E0302 on the wire; the SDK may refuse earlier with its own class
        assert e.code in ("E0302", ""), (e.code, e)
        assert str(e)
        assert e.hint or e.code == ""


def test_context_manager_closes(text_data, lib):
    with latai.Runtime.open(data_dir=str(text_data), lib=lib) as rt:
        assert not rt.closed
        saved = rt
    assert saved.closed
    with pytest.raises(latai.LataiError):
        saved.info()


def test_context_manager_closes_on_exception(text_data, lib):
    saved = None
    with pytest.raises(ZeroDivisionError):
        with latai.Runtime.open(data_dir=str(text_data), lib=lib) as rt:
            saved = rt
            1 / 0
    assert saved.closed
    # and the process can open again
    with latai.Runtime.open(data_dir=str(text_data), lib=lib):
        pass


def test_second_open_in_one_process_is_E0901_with_the_class(text_data, lib):
    with latai.Runtime.open(data_dir=str(text_data), lib=lib) as first:
        with pytest.raises(latai.LataiError) as ei:
            latai.Runtime.open(data_dir=str(text_data), lib=lib)
        e = ei.value
        assert e.code == "E0901"
        assert type(e) is latai.ERROR_CLASSES["E0901"]
        assert isinstance(e, latai.InternalError)
        assert e.hint
        # the failed open did not disturb the first one
        validate("RuntimeInfo", first.info())
    # closed: open works again
    with latai.Runtime.open(data_dir=str(text_data), lib=lib) as again:
        validate("RuntimeInfo", again.info())


def test_second_open_from_another_thread_is_E0901_too(text_data, lib):
    got = []
    with latai.Runtime.open(data_dir=str(text_data), lib=lib):

        def other():
            try:
                latai.Runtime.open(data_dir=str(text_data), lib=lib)
            except latai.LataiError as e:
                got.append(e.code)

        t = threading.Thread(target=other)
        t.start()
        t.join(30)
    assert got == ["E0901"]


def test_open_on_a_file_instead_of_a_folder_is_a_typed_error(tmp_path, lib):
    f = tmp_path / "not-a-dir"
    f.write_text("x")
    with pytest.raises(latai.LataiError) as ei:
        latai.Runtime.open(data_dir=str(f), lib=lib)
    assert ei.value.code.startswith("E")
    # a failed open leaves the process able to open
    with latai.Runtime.open(data_dir=str(tmp_path / "fresh"), lib=lib):
        pass


def test_closing_a_runtime_stops_its_live_stream(text_data, lib):
    rt = latai.Runtime.open(data_dir=str(text_data), lib=lib)
    try:
        stream = rt.run_stream(text_req("hi", max_tokens=200))
        it = iter(stream)
        assert next(it)["event"] == "delta"
    finally:
        rt.close()
    # whatever is left ends with one terminal event or a typed error, never a hang
    events = []
    try:
        for ev in it:
            events.append(ev)
    except latai.LataiError:
        pass
    assert sum(1 for e in events if e["event"] in ("final", "error")) <= 1


def test_explicit_lib_path_equals_environment_discovery(text_data, monkeypatch):
    monkeypatch.setenv("LATAI_LIB", lib_file())
    with latai.Runtime.open(data_dir=str(text_data)) as rt:
        assert rt.library.api_version >= 2
