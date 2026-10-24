# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Find and load ``liblatai`` and call its C table (``include/latai.h``) through ``ctypes``.

This module is the whole FFI layer. Rules of the C ABI that it keeps (docs/c-abi.md):

* JSON goes in as ``(pointer, length)``; names, paths, ids and urls as NUL-terminated UTF-8.
* A string comes out in ``*out`` / ``*out_len`` (NUL-terminated, the NUL not counted), is copied
  into ``bytes`` and released **at once with the table's own ``free``** (never the C library's).
* ``info`` and ``last_error`` follow the NULL-buffer rule (ask for the size, then fill).
* ``last_error`` is **per thread**: every wrapper reads it on the thread that made the failing
  call, right after it.
* A callback runs on the thread that made the call, with a string that is valid only during the
  call; the callback objects are kept alive until the call returns.

Nothing here raises for a non-zero return code: the wrappers return ``(rc, value)`` and
``latai.runtime`` turns them into exceptions.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import sys
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .errors import LibraryNotFoundError, UnsupportedError

__all__ = [
    "API_VERSION",
    "EVENT_CALLBACK",
    "Library",
    "debug_counters",
    "find_library",
    "library_candidates",
    "library_file_names",
    "load_library",
]

#: The version of the table this SDK is written for.
API_VERSION = 2

#: ``int32_t (*LataiEventCallback)(void *user_data, const char *event_json, size_t len)``. The
#: string is taken as a raw pointer and read with its length (never as a C string).
EVENT_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)

_c_int32 = ctypes.c_int32
_c_size_t = ctypes.c_size_t
_c_void_p = ctypes.c_void_p
_c_char_p = ctypes.c_char_p
_pp = ctypes.POINTER(ctypes.c_void_p)
_psize = ctypes.POINTER(ctypes.c_size_t)


class _Api(ctypes.Structure):
    """``struct LataiApi``: the field order is the order of ``include/latai.h`` (append-only)."""

    _fields_ = [
        ("size", _c_size_t),
        ("version", ctypes.c_uint32),
        ("open", ctypes.CFUNCTYPE(_c_int32, _c_char_p, _c_size_t, _pp)),
        ("close", ctypes.CFUNCTYPE(_c_int32, _c_void_p)),
        ("info", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_void_p, _psize)),
        ("run", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_char_p, _c_size_t, _c_char_p, _pp, _psize)),
        ("run_stream", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_char_p, _c_size_t, _c_char_p, EVENT_CALLBACK, _c_void_p)),
        ("cancel", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_char_p)),
        ("session_close", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_char_p)),
        ("inspect", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_char_p, _pp, _psize)),
        ("stage", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_char_p, _pp, _psize)),
        ("pull", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_char_p, _c_int32, _pp, _psize)),
        ("install", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_char_p, _c_int32, _pp, _psize)),
        ("activate", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_char_p, _c_char_p, _pp, _psize)),
        ("rollback", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_char_p, _pp, _psize)),
        ("list", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _pp, _psize)),
        ("pin", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_char_p, _c_char_p, _pp, _psize)),
        ("status", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_char_p, _pp, _psize)),
        ("stage_child", ctypes.CFUNCTYPE(_c_int32, _c_char_p, _c_size_t, _pp, _psize)),
        ("free", ctypes.CFUNCTYPE(None, _c_void_p)),
        ("last_error", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _psize)),
        # Version 2.
        ("stream_open", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_char_p, _c_size_t, EVENT_CALLBACK, _c_void_p, _pp)),
        ("stream_push", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_char_p, _c_size_t)),
        ("stream_cancel", ctypes.CFUNCTYPE(_c_int32, _c_void_p)),
        ("stream_close", ctypes.CFUNCTYPE(_c_int32, _c_void_p, _c_int32)),
    ]


# ---- discovery -------------------------------------------------------------------------------


def library_file_names(platform: Optional[str] = None) -> List[str]:
    """The file names ``liblatai`` has on a platform (``sys.platform`` text)."""
    p = platform if platform is not None else sys.platform
    if p.startswith(("win", "cygwin", "msys")):
        return ["latai.dll", "liblatai.dll"]
    if p == "darwin":
        return ["liblatai.dylib"]
    return ["liblatai.so"]


def _join(directory: str, names: List[str], platform: str) -> List[str]:
    sep = "\\" if platform.startswith("win") else "/"
    return [directory.rstrip("/\\") + sep + n for n in names]


def library_candidates(
    explicit: Optional[str] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
    package_dir: Optional[str] = None,
) -> List[str]:
    """Where ``liblatai`` is looked for, in order. Pure: nothing is read from the file system.

    1. ``explicit`` (a file, or a folder holding the library);
    2. ``$LATAI_LIB`` (a file, or a folder);
    3. ``$LATAI_LIB_DIR`` (a folder);
    4. next to the package: ``<package>/_core/`` (the core a platform wheel carries),
       ``<package>/`` and ``<package>/lib/``;
    5. a source checkout: ``<repo>/target/debug`` and ``<repo>/target/release`` (where
       ``cargo build -p latai-capi`` leaves it; the package then lives in ``<repo>/sdk/python``);
    6. the usual system folders (``/usr/local/lib``, ``/usr/lib``, Homebrew's ``/opt/homebrew/lib``);
    7. the bare file names, which the dynamic loader resolves (``LD_LIBRARY_PATH``,
       ``DYLD_LIBRARY_PATH``, ``PATH`` on Windows).

    A candidate that is a folder is expanded to the file names of the platform inside it.
    """
    env = os.environ if environ is None else environ
    plat = platform if platform is not None else sys.platform
    names = library_file_names(plat)
    out: List[str] = []

    def add(item: str) -> None:
        if item and item not in out:
            out.append(item)

    def add_path_or_dir(path: str) -> None:
        add(path)
        for c in _join(path, names, plat):
            add(c)

    if explicit:
        add_path_or_dir(os.fspath(explicit))
    if env.get("LATAI_LIB"):
        add_path_or_dir(env["LATAI_LIB"])
    if env.get("LATAI_LIB_DIR"):
        for c in _join(env["LATAI_LIB_DIR"], names, plat):
            add(c)
    pkg = package_dir if package_dir is not None else os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.join(pkg, "_core"), pkg, os.path.join(pkg, "lib")):
        for c in _join(d, names, plat):
            add(c)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(pkg)))
    for profile in ("debug", "release"):
        for c in _join(os.path.join(repo, "target", profile), names, plat):
            add(c)
    if not plat.startswith("win"):
        for d in ("/usr/local/lib", "/usr/lib", "/opt/homebrew/lib"):
            for c in _join(d, names, plat):
                add(c)
    for n in names:
        add(n)
    return out


def find_library(
    explicit: Optional[str] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
) -> str:
    """The path (or loader name) of ``liblatai``: the file named by ``explicit`` or ``$LATAI_LIB``
    (an error if nothing is there), else the first candidate that exists, else the first bare name
    the loader can open, else what ``ctypes.util.find_library`` says.

    Raises ``LibraryNotFoundError`` listing what was tried.
    """
    env = os.environ if environ is None else environ
    plat = platform if platform is not None else sys.platform
    # A path you name (the argument, or $LATAI_LIB) is not a hint: if nothing is there, say so
    # instead of loading some other liblatai.
    for what, given in (("the library path", explicit), ("$LATAI_LIB", env.get("LATAI_LIB"))):
        if given:
            own = [given] + _join(os.fspath(given), library_file_names(plat), plat)
            for c in own:
                if os.path.isfile(c):
                    return os.path.abspath(c)
            raise LibraryNotFoundError(f"{what} {given!r} does not hold liblatai (tried: {', '.join(own)})", own)
    cands = library_candidates(explicit, environ=environ, platform=platform)
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    for c in cands:
        if os.sep not in c and "/" not in c and "\\" not in c:
            try:
                ctypes.CDLL(c)
                return c
            except OSError:
                continue
    found = ctypes.util.find_library("latai")
    if found:
        return found
    raise LibraryNotFoundError("liblatai was not found (tried: " + ", ".join(cands) + ")", cands)


# ---- counters --------------------------------------------------------------------------------

_counter_lock = threading.Lock()
_taken = 0
_freed = 0


def debug_counters() -> Dict[str, int]:
    """Strings the library handed out through ``*out`` (``taken``) and strings given back to its
    ``free`` (``freed``) in this process. They are equal whenever no call is in progress; a test
    can assert it after a workload to catch a leak of the ABI's strings."""
    with _counter_lock:
        return {"taken": _taken, "freed": _freed}


# ---- the library -----------------------------------------------------------------------------


class Library:
    """A loaded ``liblatai``: the C table with its memory rules applied.

    Thread-safe: every method may be called from any thread (the table says so). Use
    ``load_library`` to share one per path. ``outstanding_strings`` is this instance's
    ``taken - freed``.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = find_library(path)
        try:
            if sys.platform.startswith("win") and os.path.isabs(self.path):
                os.add_dll_directory(os.path.dirname(self.path))  # type: ignore[attr-defined]
            self._cdll = ctypes.CDLL(self.path)
        except OSError as exc:
            raise LibraryNotFoundError(f"cannot load {self.path}: {exc}", [self.path]) from exc
        try:
            get_api = self._cdll.latai_get_api
        except AttributeError as exc:
            raise LibraryNotFoundError(f"{self.path} has no latai_get_api: not liblatai", [self.path]) from exc
        get_api.restype = ctypes.c_void_p
        get_api.argtypes = [ctypes.c_uint32]
        ptr = get_api(API_VERSION) or get_api(1)
        if not ptr:
            raise LibraryNotFoundError(f"{self.path} does not serve an API table we know", [self.path])
        self._table = ctypes.cast(ptr, ctypes.POINTER(_Api)).contents
        self.api_version: int = int(self._table.version)
        #: Whether the table has the four ``stream_*`` entries (API version 2).
        self.has_streams: bool = int(self._table.size) >= ctypes.sizeof(_Api)
        self._lock = threading.Lock()
        self._taken = 0
        self._freed = 0

    # -- strings ------------------------------------------------------------------------------

    @property
    def outstanding_strings(self) -> int:
        """Strings taken from the library and not yet freed (0 outside a call)."""
        with self._lock:
            return self._taken - self._freed

    def _take(self, out: ctypes.c_void_p, out_len: ctypes.c_size_t) -> Optional[bytes]:
        """Copy ``*out`` and release it with the library's ``free``; ``None`` when it is NULL."""
        global _taken, _freed
        if not out.value:
            return None
        with self._lock:
            self._taken += 1
        with _counter_lock:
            _taken += 1
        try:
            return ctypes.string_at(out.value, out_len.value)
        finally:
            self._table.free(out.value)
            with self._lock:
                self._freed += 1
            with _counter_lock:
                _freed += 1

    def _out_call(self, fn: Callable[..., int], *args: Any) -> Tuple[int, Optional[bytes]]:
        out = ctypes.c_void_p()
        out_len = ctypes.c_size_t(0)
        rc = fn(*args, ctypes.byref(out), ctypes.byref(out_len))
        return int(rc), self._take(out, out_len)

    def _buffer_call(self, fn: Callable[..., int], *args: Any) -> Tuple[int, bytes]:
        """The NULL-buffer rule: ask for the size, then fill, again if it grew in between."""
        need = ctypes.c_size_t(0)
        rc = fn(*args, None, ctypes.byref(need))
        if rc != 0:
            return int(rc), b""
        for _ in range(8):
            cap = need.value
            buf = ctypes.create_string_buffer(max(cap, 1))
            n = ctypes.c_size_t(cap)
            rc = fn(*args, buf, ctypes.byref(n))
            if rc != 0:
                return int(rc), b""
            if n.value <= cap:
                return 0, buf.raw[: max(n.value - 1, 0)]  # drop the terminating NUL
            need = n
        return 0, b""

    def last_error(self) -> Optional[Dict[str, Any]]:
        """The wire error of the last failing call **on this thread**, or ``None``."""
        rc, raw = self._buffer_call(self._table.last_error)
        if rc != 0 or not raw:
            return None
        value = json.loads(raw.decode("utf-8", "replace"))
        return value if isinstance(value, dict) else None

    # -- runtime ------------------------------------------------------------------------------

    def open(self, config: Optional[bytes]) -> Tuple[int, Optional[int]]:
        rt = ctypes.c_void_p()
        if config:
            rc = self._table.open(config, len(config), ctypes.byref(rt))
        else:
            rc = self._table.open(None, 0, ctypes.byref(rt))
        return int(rc), (rt.value if rc == 0 else None)

    def close(self, rt: int) -> int:
        return int(self._table.close(rt))

    def info(self, rt: int) -> Tuple[int, bytes]:
        return self._buffer_call(self._table.info, rt)

    def run(self, rt: int, request: bytes, request_id: Optional[str]) -> Tuple[int, Optional[bytes]]:
        rid = request_id.encode("utf-8") if request_id is not None else None
        return self._out_call(self._table.run, rt, request, len(request), rid)

    def run_stream(self, rt: int, request: bytes, request_id: Optional[str], cb: Any) -> int:
        """Blocks until the run ends; ``cb`` is an ``EVENT_CALLBACK`` instance that the caller keeps
        alive. Runs ``cb`` on the calling thread."""
        rid = request_id.encode("utf-8") if request_id is not None else None
        return int(self._table.run_stream(rt, request, len(request), rid, cb, None))

    def cancel(self, rt: int, request_id: str) -> int:
        return int(self._table.cancel(rt, request_id.encode("utf-8")))

    def session_close(self, rt: int, session: str) -> int:
        return int(self._table.session_close(rt, session.encode("utf-8")))

    def _by_name(self, name: str, rt: int, *args: Any) -> Tuple[int, Optional[bytes]]:
        return self._out_call(getattr(self._table, name), rt, *args)

    def inspect(self, rt: int, target: str) -> Tuple[int, Optional[bytes]]:
        return self._by_name("inspect", rt, target.encode("utf-8"))

    def stage(self, rt: int, path: str) -> Tuple[int, Optional[bytes]]:
        return self._by_name("stage", rt, path.encode("utf-8"))

    def pull(self, rt: int, url: str, force: bool) -> Tuple[int, Optional[bytes]]:
        return self._by_name("pull", rt, url.encode("utf-8"), 1 if force else 0)

    def install(self, rt: int, source: str, activate: bool) -> Tuple[int, Optional[bytes]]:
        return self._by_name("install", rt, source.encode("utf-8"), 1 if activate else 0)

    def activate(self, rt: int, name: str, version: Optional[str]) -> Tuple[int, Optional[bytes]]:
        v = version.encode("utf-8") if version is not None else None
        return self._by_name("activate", rt, name.encode("utf-8"), v)

    def rollback(self, rt: int, name: str) -> Tuple[int, Optional[bytes]]:
        return self._by_name("rollback", rt, name.encode("utf-8"))

    def list(self, rt: int) -> Tuple[int, Optional[bytes]]:
        return self._by_name("list", rt)

    def pin(self, rt: int, name: str, pin: Optional[str]) -> Tuple[int, Optional[bytes]]:
        p = pin.encode("utf-8") if pin is not None else None
        return self._by_name("pin", rt, name.encode("utf-8"), p)

    def status(self, rt: int, name: Optional[str]) -> Tuple[int, Optional[bytes]]:
        n = name.encode("utf-8") if name is not None else None
        return self._by_name("status", rt, n)

    def stage_child(self, request: bytes) -> Tuple[int, Optional[bytes]]:
        """For ``latai __stage-child`` only: it limits the address space of **this process** and
        ends it when stdin ends. Never call it from a program that means to go on running."""
        return self._out_call(self._table.stage_child, request, len(request))

    # -- streams (API version 2) --------------------------------------------------------------

    def _need_streams(self) -> None:
        if not self.has_streams:
            raise UnsupportedError(f"{self.path} is API version {self.api_version}: no stream_* entries")

    def stream_open(self, rt: int, open_json: bytes, cb: Any) -> Tuple[int, Optional[int]]:
        self._need_streams()
        handle = ctypes.c_void_p()
        rc = self._table.stream_open(rt, open_json, len(open_json), cb, None, ctypes.byref(handle))
        return int(rc), (handle.value if rc == 0 else None)

    def stream_push(self, stream: int, data: bytes) -> int:
        self._need_streams()
        return int(self._table.stream_push(stream, data, len(data)))

    def stream_cancel(self, stream: int) -> int:
        self._need_streams()
        return int(self._table.stream_cancel(stream))

    def stream_close(self, stream: int, flush: bool) -> int:
        self._need_streams()
        return int(self._table.stream_close(stream, 1 if flush else 0))


_libraries: Dict[str, Library] = {}
_libraries_lock = threading.Lock()


def load_library(path: Optional[str] = None) -> Library:
    """The shared ``Library`` for ``path`` (discovery as in ``find_library``), loaded once."""
    resolved = find_library(path)
    with _libraries_lock:
        lib = _libraries.get(resolved)
        if lib is None:
            lib = _libraries[resolved] = Library(resolved)
        return lib
