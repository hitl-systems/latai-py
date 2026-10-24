# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Shared by the Python SDK tests (phase 14, tester 14A).

The suite runs against a real `liblatai` and the real fixture packs; nothing here builds anything.
Environment (all optional when the defaults below resolve):

* ``LATAI_LIB``              the built library (default: ``<repo>/target/debug/liblatai.so``)
* ``LATAI_FIXTURES_DIR``     the fixture cache (``tests/fixtures/cache``)
* ``LATAI_TEST_PACK``        the text pack file (default ``$LATAI_FIXTURES_DIR/packs/x4r-l8.gguf``)
* ``LATAI_TEST_SPEECH_PACK`` the speech pack file (``speech-packs/speech-qwen2a.gguf``)
* ``LATAI_TEST_STAGE_CHILD`` the ``stage_child`` binary (default ``<repo>/target/debug/stage_child``),
  or ``core`` for the core's own ``latai`` beside liblatai (a core archive, a platform wheel)
* ``LATAI_REQUIRE_FIXTURES=1`` a missing fixture fails the test instead of skipping it.

Only one runtime may be open per process, so the runtime fixtures are module scoped and the
modules run one after the other; a test that opens its own runtime must not use ``rt``.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import struct
import sys
import wave
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SDK = TESTS.parent
REPO = SDK.parent.parent

# The package under test is the checkout's, whatever else is installed.
if os.environ.get("LATAI_SDK_INSTALLED") != "1":
    sys.path.insert(0, str(SDK))

PACK = "x4r-l8"
SPEECH = "speech-qwen2a"


def require_fixtures() -> bool:
    return os.environ.get("LATAI_REQUIRE_FIXTURES") == "1"


def missing(what: str):
    """A fixture is missing: a failure when the run requires fixtures, else a loud skip."""
    msg = f"fixture missing: {what} (set LATAI_FIXTURES_DIR / LATAI_REQUIRE_FIXTURES=1)"
    if require_fixtures():
        pytest.fail(msg, pytrace=False)
    pytest.skip(msg)


def fixtures_dir() -> Path | None:
    d = os.environ.get("LATAI_FIXTURES_DIR")
    if d:
        return Path(d)
    default = REPO / "tests" / "fixtures" / "cache"
    return default if default.is_dir() else None


def fixture_file(env: str, rel: str) -> Path:
    explicit = os.environ.get(env)
    if explicit:
        p = Path(explicit)
    else:
        base = fixtures_dir()
        if base is None:
            missing(rel)
        p = base / rel
    if not p.is_file():
        missing(str(p))
    return p


def lib_file() -> str:
    explicit = os.environ.get("LATAI_LIB")
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            for n in ("liblatai.so", "liblatai.dylib", "latai.dll"):
                if (p / n).is_file():
                    return str(p / n)
        if p.is_file():
            return str(p)
        pytest.fail(f"LATAI_LIB={explicit} is not a built liblatai (cargo build -p latai-capi)", pytrace=False)
    for profile in ("debug", "release"):
        for n in ("liblatai.so", "liblatai.dylib", "latai.dll"):
            p = REPO / "target" / profile / n
            if p.is_file():
                return str(p)
    pytest.fail("liblatai is not built: run `cargo build -p latai-capi` and set LATAI_LIB", pytrace=False)


def stage_child() -> Path:
    p = Path(os.environ.get("LATAI_TEST_STAGE_CHILD") or REPO / "target" / "debug" / "stage_child")
    if not p.is_file():
        pytest.fail(f"stage_child binary not found at {p}: build latai-conformance first", pytrace=False)
    return p


def make_data_dir(root: Path, tag: str) -> Path:
    """An empty data folder that trusts the test key, uses the `dev` channel and stages through the
    conformance stage child (which needs the library path put back: it gets an empty environment)."""
    d = root / f"data-{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "keys" / "packs").mkdir(parents=True)
    key = os.environ.get("LATAI_TEST_PUB") or REPO / "tests" / "fixtures" / "keys" / "test.pub"
    shutil.copy(key, d / "keys" / "packs" / "test.pub")
    if os.environ.get("LATAI_TEST_STAGE_CHILD") == "core":
        # The core's own `latai` beside liblatai stages, as in a shipped app (a core archive, a
        # platform wheel): no stage_child line.
        (d / "latai.toml").write_text('[runtime]\nchannel = "dev"\n', encoding="utf-8")
        return d
    libdirs = [str(Path(lib_file()).parent)]
    old = os.environ.get("LD_LIBRARY_PATH") or os.environ.get("DYLD_LIBRARY_PATH")
    if old:
        libdirs.append(old)
    else:  # run by hand, outside cargo: the llama libraries the build left (the newest build)
        found = sorted(
            (REPO / "target").glob("*/build/llama-cpp-sys-2-*/out/lib"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        libdirs += [str(p) for p in found[:1]]
    if sys.platform == "win32":
        # No /bin/sh to run a wrapper, and none needed: Windows finds the llama DLLs beside the
        # executable (the sys crate links them into target/<profile>, where stage_child.exe is).
        program = stage_child()
    else:
        var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        program = d / "stage-child.sh"
        program.write_text(f"#!/bin/sh\n{var}='{os.pathsep.join(libdirs)}' exec '{stage_child()}'\n")
        program.chmod(program.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # A TOML basic string: JSON's escaping is valid TOML, and a Windows path (`C:\Users\...`) needs it
    # (`\U` is a unicode escape there).
    (d / "latai.toml").write_text(
        f'[runtime]\nstage_child = {json.dumps(str(program))}\nchannel = "dev"\n', encoding="utf-8"
    )
    return d


# ------------------------------------------------------------------------------ session fixtures


@pytest.fixture(scope="session")
def work(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("latai-sdk-py")


@pytest.fixture(scope="session")
def lib() -> str:
    return lib_file()


@pytest.fixture(scope="session")
def pack_file() -> Path:
    return fixture_file("LATAI_TEST_PACK", "packs/x4r-l8.gguf")


@pytest.fixture(scope="session")
def speech_pack_file() -> Path:
    fixture_file("LATAI_TEST_SPEECH_PACK", "speech-packs/speech-qwen2a.mmproj.gguf") if not os.environ.get(
        "LATAI_TEST_SPEECH_PACK"
    ) else None
    return fixture_file("LATAI_TEST_SPEECH_PACK", "speech-packs/speech-qwen2a.gguf")


def _installed(work: Path, tag: str, files: list[Path]) -> Path:
    """A data folder with `files` installed through the SDK's own `install`, then closed."""
    import latai

    data = make_data_dir(work, tag)
    with latai.Runtime.open(data_dir=str(data), lib=lib_file()) as rt:
        for f in files:
            receipts = rt.install(str(f))
            assert isinstance(receipts, list) and len(receipts) == 1, receipts
    return data


@pytest.fixture(scope="session")
def text_template(work, pack_file) -> Path:
    """The template data folder with `x4r-l8` installed and active (copied per module)."""
    return _installed(work, "text-template", [pack_file])


@pytest.fixture(scope="session")
def speech_template(work, speech_pack_file) -> Path:
    return _installed(work, "speech-template", [speech_pack_file])


def copy_data(template: Path, work: Path, tag: str) -> Path:
    dst = work / f"data-{tag}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(template, dst, symlinks=True)
    # the wrapper script has an absolute path in latai.toml of the template: still valid
    return dst


@pytest.fixture(scope="module")
def text_data(text_template, work, request) -> Path:
    return copy_data(text_template, work, "t-" + request.module.__name__)


def _close(r) -> None:
    """Close the module's runtime for sure: a test that found a bug may have left the SDK believing it
    is closed while the library still has it open (which would fail every later module with E0901)."""
    r.close()
    leaked = getattr(r, "_test_handle", None)
    if leaked is not None:
        try:
            r.library.close(leaked)  # E0302 when it was closed properly: ignored
        except Exception:
            pass


@pytest.fixture(scope="module")
def rt(text_data, lib):
    """The one runtime of the module, on a copy of the template with `x4r-l8` installed."""
    import latai

    r = latai.Runtime.open(data_dir=str(text_data), lib=lib)
    r._test_handle = getattr(r, "_rt", None)
    try:
        yield r
    finally:
        _close(r)


@pytest.fixture(scope="module")
def speech_rt(speech_template, work, lib, request):
    import latai

    data = copy_data(speech_template, work, "s-" + request.module.__name__)
    r = latai.Runtime.open(data_dir=str(data), lib=lib)
    r._test_handle = getattr(r, "_rt", None)
    try:
        yield r
    finally:
        _close(r)


def text_req(text: str = "hello", max_tokens: int = 4, pack: str = PACK, **extra) -> dict:
    req = {"pack": pack, "inputs": {"type": "text", "text": text}, "output": {"max_tokens": max_tokens}}
    req.update(extra)
    return req


# ------------------------------------------------------------------------------ audio helpers


def sine_pcm16(seconds: float = 1.0, rate: int = 16000, freq: float = 440.0, amp: int = 8000) -> bytes:
    """Deterministic raw 16-bit little-endian mono samples (integer arithmetic only)."""
    import math

    n = int(seconds * rate)
    return b"".join(struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / rate))) for i in range(n))


def write_wav(path: Path, pcm: bytes, rate: int = 16000, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return path


def speech_clip_pcm16(name: str = "sine") -> bytes:
    """A fixture clip (f32 samples at 16 kHz) as pcm16, the way the conformance tests convert it."""
    path = fixture_file("LATAI_TEST_CLIP_" + name.upper(), f"speech/clip-{name}.f32le")
    raw = path.read_bytes()
    n = len(raw) // 4
    out = bytearray()
    for (v,) in struct.iter_unpack("<f", raw[: n * 4]):
        s = int(round(max(-1.0, min(1.0, v)) * 32767))
        out += struct.pack("<h", s)
    return bytes(out)


# The wire schema: latai's schema/, or testdata/schema in the SDK's own public repository.
SCHEMA_DIR = REPO / "schema" if (REPO / "schema").is_dir() else SDK / "testdata" / "schema"


def schema() -> dict:
    return json.loads((SCHEMA_DIR / "latai.schema.json").read_text())


def errors_json() -> dict:
    return json.loads((SCHEMA_DIR / "errors.json").read_text())


def validate(name: str, instance) -> None:
    """Validate against `$defs/<name>` of the wire schema (jsonschema when importable, else the
    required keys and the closed key set of the definition)."""
    s = schema()
    try:
        import jsonschema
    except ImportError:
        d = s["$defs"][name]
        if d.get("type") == "object":
            assert isinstance(instance, dict), instance
            for k in d.get("required", []):
                assert k in instance, f"{name}: missing {k}"
            if d.get("additionalProperties") is False:
                extra = set(instance) - set(d.get("properties", {}))
                assert not extra, f"{name}: unknown keys {extra}"
        return
    jsonschema.validate(instance, {"$ref": f"#/$defs/{name}", "$defs": s["$defs"]})


def rss_kb() -> int:
    if sys.platform == "win32":
        return _windows_working_set_kb()
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)
    except OSError:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r // 1024 if sys.platform == "darwin" else r


def _windows_working_set_kb() -> int:
    # The working set (Windows' resident size) from K32GetProcessMemoryInfo: Windows has neither
    # /proc nor the `resource` module (ci-rca.md 32).
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    if not kernel32.K32GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.WorkingSetSize // 1024


def strip_timing(resp: dict) -> dict:
    out = json.loads(json.dumps(resp))
    out.get("latai", {}).pop("timing", None)
    return out
