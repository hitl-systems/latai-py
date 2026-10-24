# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Library discovery: pure functions (no filesystem) plus the real loader paths."""

import os
import shutil

import pytest

import latai
from conftest import lib_file


def test_file_names_per_platform():
    assert latai.library_file_names("linux") == ["liblatai.so"]
    assert latai.library_file_names("linux2") == ["liblatai.so"]
    assert latai.library_file_names("darwin") == ["liblatai.dylib"]
    win = latai.library_file_names("win32")
    assert win[0] == "latai.dll" and "liblatai.dll" in win


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_candidates_order_explicit_then_env_then_package(platform):
    sep = "\\" if platform == "win32" else "/"
    explicit = sep.join(["", "opt", "x", "explicit"]) if platform != "win32" else r"C:\opt\x\explicit.dll"
    lib_env = "/env/lib" if platform != "win32" else r"D:\env\lib"
    dir_env = "/env/dir" if platform != "win32" else r"D:\env\dir"
    env = {"LATAI_LIB": lib_env, "LATAI_LIB_DIR": dir_env}
    c = latai.library_candidates(explicit, environ=env, platform=platform, package_dir="/pkg/latai")
    assert isinstance(c, list) and all(isinstance(x, str) for x in c)
    assert len(c) == len(set(c)), "no duplicates"

    def first(prefix):
        for i, x in enumerate(c):
            if x.startswith(prefix):
                return i
        raise AssertionError(f"no candidate starts with {prefix!r}: {c}")

    assert c[0] == explicit
    assert first(explicit) < first(lib_env) < first(dir_env)
    # the package directory comes after the environment
    assert first(dir_env) < first("/pkg/latai") if True else None
    names = latai.library_file_names(platform)
    # the LATAI_LIB_DIR entry is a folder: the candidates are the folder plus a file name
    assert any(x.startswith(dir_env) and any(x.endswith(n) for n in names) for x in c), c
    # bare names for the dynamic loader come last
    assert c[-1] in names or any(c[-1].endswith(n) for n in names)


def test_candidates_without_environment_has_no_env_entries():
    c = latai.library_candidates(None, environ={}, platform="linux", package_dir="/p/latai")
    assert c, "there are always candidates"
    assert not any("env" in x for x in c)
    assert "liblatai.so" in c  # bare name for the loader


def test_candidates_are_pure_no_filesystem_and_no_os_environ(monkeypatch):
    monkeypatch.setenv("LATAI_LIB", "/should/not/appear/anywhere")
    c = latai.library_candidates(None, environ={}, platform="linux", package_dir="/p")
    assert not any("should/not/appear" in x for x in c)
    # nonexistent paths are returned without error
    c = latai.library_candidates("/no/such/dir/liblatai.so", environ={}, platform="linux", package_dir="/p")
    assert c[0] == "/no/such/dir/liblatai.so"


def test_windows_paths_keep_drive_and_backslashes():
    c = latai.library_candidates(
        r"C:\Program Files\latai\latai.dll", environ={}, platform="win32", package_dir=r"C:\py\latai"
    )
    assert c[0] == r"C:\Program Files\latai\latai.dll"
    assert any(x.startswith(r"C:\py\latai") for x in c)
    # the Windows names, never the Unix ones
    assert not any(x.endswith("liblatai.so") for x in c)


def test_macos_names_never_linux():
    c = latai.library_candidates(None, environ={"LATAI_LIB_DIR": "/Users/me/latai/lib"}, platform="darwin", package_dir="/p")
    assert any(x.endswith("liblatai.dylib") for x in c)
    assert not any(x.endswith("liblatai.so") for x in c)
    assert any(x.startswith("/Users/me/latai/lib") for x in c)


def test_find_library_prefers_explicit_existing_file(tmp_path):
    real = lib_file()
    got = latai.find_library(real, environ={}, platform=None)
    assert os.path.samefile(got, real)


def test_find_library_env_file_and_env_dir(tmp_path):
    real = lib_file()
    got = latai.find_library(environ={"LATAI_LIB": real})
    assert os.path.samefile(got, real)
    # a folder holding the library
    d = tmp_path / "libs"
    d.mkdir()
    shutil.copy(real, d / os.path.basename(real))
    got = latai.find_library(environ={"LATAI_LIB_DIR": str(d)})
    assert os.path.samefile(got, d / os.path.basename(real))
    got = latai.find_library(environ={"LATAI_LIB": str(d)})
    assert os.path.samefile(got, d / os.path.basename(real))


def test_missing_library_is_a_clear_typed_error(tmp_path):
    """Run in a copy of the package, away from the checkout's target/, with a clean environment, so
    that nothing can be found."""
    import subprocess
    import sys

    from conftest import SDK

    pkg = tmp_path / "isolated" / "deep" / "site" / "latai"
    shutil.copytree(SDK / "latai", pkg, ignore=shutil.ignore_patterns("__pycache__"))
    code = (
        "import sys, json; sys.path.insert(0, sys.argv[1]); import latai\n"
        "try:\n"
        "    latai.find_library(sys.argv[2], environ={'LATAI_LIB_DIR': sys.argv[3]})\n"
        "except latai.LibraryNotFoundError as e:\n"
        "    assert isinstance(e, latai.LataiError) and e.code == ''\n"
        "    print(json.dumps({'tried': e.tried, 'text': str(e)}))\n"
        "else:\n"
        "    print('FOUND')\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LATAI_", "LD_LIBRARY", "DYLD_"))}
    env["PATH"] = os.environ.get("PATH", "")
    nope = str(tmp_path / "nope" / "liblatai.so")
    r = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "isolated" / "deep" / "site"), nope, str(tmp_path / "empty")],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() != "FOUND", "a library was found where none should be (is one installed system-wide?)"
    import json

    got = json.loads(r.stdout)
    assert got["tried"] and all(isinstance(t, str) for t in got["tried"])
    assert got["tried"][0] == nope
    text = got["text"]
    assert "LATAI_LIB" in text and nope in text, text


def isolated(tmp_path, body: str):
    """Run `body` in a copy of the package away from the checkout, with no LATAI_* environment."""
    import subprocess
    import sys

    from conftest import SDK

    pkg = tmp_path / "iso" / "deep" / "site" / "latai"
    if not pkg.exists():
        shutil.copytree(SDK / "latai", pkg, ignore=shutil.ignore_patterns("__pycache__"))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LATAI_", "LD_LIBRARY", "DYLD_"))}
    code = "import sys; sys.path.insert(0, sys.argv[1]); import latai\n" + body
    return subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "iso" / "deep" / "site"), str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_opening_with_a_missing_library_raises_the_same_error(tmp_path):
    r = isolated(
        tmp_path,
        "import os\n"
        "missing = os.path.join(sys.argv[2], 'missing', 'liblatai.so')\n"
        "for call in (lambda: latai.Runtime.open(data_dir=sys.argv[2], lib=missing), lambda: latai.Library(missing)):\n"
        "    try:\n"
        "        call()\n"
        "    except latai.LibraryNotFoundError as e:\n"
        "        assert isinstance(e, latai.LataiError) and e.tried\n"
        "    else:\n"
        "        raise SystemExit('no error')\n",
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_file_that_is_not_a_library_is_a_typed_error_not_a_crash(tmp_path):
    bogus = tmp_path / "liblatai.so"
    bogus.write_bytes(b"not an elf")
    with pytest.raises(latai.LataiError):
        latai.Library(str(bogus))


def test_explicit_path_wins_over_environment(tmp_path, monkeypatch):
    real = lib_file()
    monkeypatch.setenv("LATAI_LIB", str(tmp_path / "wrong.so"))
    lib = latai.Library(real)
    assert lib.api_version >= 2
    assert lib.has_streams


def test_environment_lib_is_used_by_default(monkeypatch):
    monkeypatch.setenv("LATAI_LIB", lib_file())
    lib = latai.Library()
    assert lib.api_version >= 2


def test_the_core_a_platform_wheel_carries_comes_first_in_the_package():
    c = latai.library_candidates(None, environ={}, platform="linux", package_dir="/p/latai")
    package = [x for x in c if x.startswith("/p/latai")]
    assert package[0] == "/p/latai/_core/liblatai.so", package
