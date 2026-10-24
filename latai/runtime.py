# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The runtime: open, run, stream, manage packs, close.

Rules of the C ABI that this module keeps for you (docs/c-abi.md):

* **One runtime per process.** A second ``Runtime.open`` before ``close`` is ``E0901``.
* **Any call from any thread.** ``Runtime`` methods are thread-safe; requests run one at a time
  in the library's queue (a full queue is ``BusyError`` / ``E0701``, retryable).
* **Callbacks run on the thread that made the call.** ``run_stream`` makes its C call on a worker
  thread and hands the events to *your* thread through a bounded queue, so your loop body is
  never inside the callback and ``E0303`` (re-entry) cannot happen from it. The sessions of
  ``listen`` do deliver on the pushing thread (see ``latai.listen``): do not call ``run``,
  ``close`` or the pack calls from an ``on_event`` there.
* **Strings out** are copied and released with the table's ``free`` before the call returns.
* **Errors.** A failing call leaves its wire error for the calling thread; it is read on that
  thread and raised as a typed exception (``latai.errors``).
* **``close``** stops requests in flight and answers queued ones ``cancelled``; live ``listen``
  sessions are abandoned first. Close every stream before the runtime.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import queue
import threading
import uuid
import weakref
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Union

from ._json import JsonInput, loads, to_json_bytes, with_stream
from ._lib import EVENT_CALLBACK, Library, load_library
from .errors import LataiError, NullArgument, error_for
from .listen import ListenSession, listen_iter

if TYPE_CHECKING:  # pragma: no cover
    from .types import (
        Event,
        PackInfo,
        PackVersion,
        Request,
        Response,
        RuntimeInfo,
        StageReceipt,
        Status,
        StreamEvent,
        StreamOpen,
    )

__all__ = ["EventStream", "Runtime"]

_END = object()

# Streams and runtimes still alive when the interpreter exits are closed first: a daemon worker
# that is inside the C library when Python finalizes would be killed with a forced unwind through
# the library's frames (glibc: "exception not rethrown"), so none may be left running.
_live_streams: "weakref.WeakSet[EventStream]" = weakref.WeakSet()
_live_runtimes: "weakref.WeakSet[Runtime]" = weakref.WeakSet()


@atexit.register
def _close_all_at_exit() -> None:  # pragma: no cover - exercised at interpreter exit
    for stream in list(_live_streams):
        try:
            stream.close()
        except Exception:
            pass
    for runtime in list(_live_runtimes):
        try:
            runtime.close()
        except Exception:
            pass


def _synthetic(rc: int) -> Dict[str, Any]:
    return {"code": rc}


class EventStream:
    """The events of one ``run_stream`` call, as an iterator of event dicts.

    ``{"event":"delta","text":..,"ms":..}`` events, then exactly one ``final`` (``response``) or
    ``error`` event. The C call runs on a worker thread; its callback puts each event on a bounded
    queue (so a slow consumer slows the run down; no event is dropped or merged) and returns 0.

    * ``close()`` (also on leaving a ``with`` block, on ``break`` + garbage collection) stops the
      run: the callback starts returning non-zero and ``cancel`` is called, so the run ends with
      ``finish: "cancelled"``; events not yet read are discarded.
    * ``cancel()`` only asks the library to cancel the request; the iteration goes on and ends
      with the ``final`` event (``finish: "cancelled"``).
    * **Events are never raised.** An ``error`` event, or a ``final`` with ``finish: "error"``, is
      yielded like any event and the iterator then ends normally; look at the event (``final`` /
      ``code``). Only a call that fails *before any terminal event* (``E0302`` closed runtime,
      ``E0303`` re-entry: the library delivers no event) raises. ``raise_on_error=True`` also
      raises the typed error of a failed terminal event after yielding it (``.response`` set).
    * ``final`` is the response of the ``final`` event once it was read; ``request_id`` is the id
      used for ``cancel``.
    """

    def __init__(
        self,
        runtime: "Runtime",
        request: bytes,
        request_id: str,
        raise_on_error: bool,
        queue_size: int,
    ) -> None:
        self._runtime = runtime
        self._request = request
        self.request_id = request_id
        self._raise = raise_on_error
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._finished = False
        self._lock = threading.Lock()
        self._terminal = False
        self.final: Optional["Response"] = None
        #: The return code of the C call once the stream is done (``None`` before).
        self.code: Optional[int] = None
        _live_streams.add(self)

    # -- worker -------------------------------------------------------------------------------

    def _put(self, item: Any) -> bool:
        while True:
            try:
                self._queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                if self._stop.is_set():
                    return False

    def _worker(self) -> None:
        lib = self._runtime.library
        failure: List[BaseException] = []

        def callback(_user: Any, ptr: Any, length: int) -> int:
            try:
                if self._stop.is_set():
                    return 1
                event = loads(ctypes.string_at(ptr, length))
                return 0 if self._put(event) else 1
            except BaseException as exc:  # never let an exception cross into C
                failure.append(exc)
                return 1

        cb = EVENT_CALLBACK(callback)  # alive until run_stream returns
        rc = 0
        wire: Optional[Dict[str, Any]] = None
        try:
            rc = lib.run_stream(self._runtime._handle(), self._request, self.request_id, cb)
            if rc != 0:
                wire = lib.last_error() or _synthetic(rc)  # this thread made the call
        except BaseException as exc:
            failure.append(exc)
        self._put((_END, rc, wire, failure[0] if failure else None))

    def _start(self) -> None:
        with self._lock:
            if self._thread is None and not self._stop.is_set():
                self._thread = threading.Thread(target=self._worker, name="latai-run-stream", daemon=True)
                self._thread.start()

    # -- iterator -----------------------------------------------------------------------------

    def __iter__(self) -> "EventStream":
        return self

    def __next__(self) -> "Event":
        if self._finished:
            raise StopIteration
        self._start()
        if self._thread is None:  # closed before it started
            self._finished = True
            raise StopIteration
        item = self._queue.get()
        if isinstance(item, tuple) and item and item[0] is _END:
            _, rc, wire, failure = item
            self._finished = True
            self.code = rc
            self._join()
            if failure is not None:
                raise failure
            if rc != 0 and (self._raise or not self._terminal):
                raise error_for(wire, response=self.final)
            raise StopIteration
        if item.get("event") == "final":
            self.final = item.get("response")
        if item.get("event") in ("final", "error"):
            self._terminal = True
        return item  # type: ignore[no-any-return]

    def _join(self) -> None:
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def cancel(self) -> None:
        """Ask the library to cancel the request; the stream ends with a ``cancelled`` ``final``."""
        try:
            self._runtime.cancel(self.request_id)
        except LataiError:
            pass

    def close(self) -> None:
        """Stop the run now and release the worker thread. Safe to call twice."""
        started = self._thread is not None
        self._stop.set()
        if started and not self._finished:
            self.cancel()
            while True:  # drain, so the worker is never blocked on a full queue
                try:
                    self._queue.get(timeout=0.05)
                except queue.Empty:
                    if not (self._thread and self._thread.is_alive()):
                        break
        self._finished = True
        self._join()

    def __enter__(self) -> "EventStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            if self._thread is not None and not self._finished:
                self.close()
        except Exception:
            pass


class Runtime:
    """An opened latai runtime. Make one with ``Runtime.open``; use it as a context manager.

    >>> with Runtime.open(data_dir="/var/lib/latai") as rt:            # doctest: +SKIP
    ...     rt.run({"pack": "x4r-l8", "inputs": {"type": "text", "text": "hi"}})["output"]
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        bundle_dir: Optional[str] = None,
        *,
        lib: Union[None, str, Library] = None,
    ) -> None:
        self.library: Library = lib if isinstance(lib, Library) else load_library(lib)
        config: Dict[str, str] = {}
        if data_dir is not None:
            config["data_dir"] = str(data_dir)
        if bundle_dir is not None:
            config["bundle_dir"] = str(bundle_dir)
        raw = json.dumps(config).encode("utf-8") if config else None
        rc, handle = self.library.open(raw)
        if rc != 0:
            raise error_for(self.library.last_error() or _synthetic(rc))
        self._rt: Optional[int] = handle
        self._lock = threading.RLock()
        self._sessions: "weakref.WeakSet[ListenSession]" = weakref.WeakSet()
        _live_runtimes.add(self)

    @classmethod
    def open(
        cls,
        data_dir: Optional[str] = None,
        bundle_dir: Optional[str] = None,
        *,
        lib: Union[None, str, Library] = None,
    ) -> "Runtime":
        """Open the runtime of this process.

        ``data_dir`` and ``bundle_dir`` default to ``LATAI_DATA_DIR`` / ``latai.toml`` / the OS
        default (docs/config.md). ``lib`` is the path of ``liblatai`` (see
        ``latai.library_candidates``) or a ``Library``.
        """
        return cls(data_dir, bundle_dir, lib=lib)

    # -- lifecycle ----------------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._rt is None

    def _handle(self) -> int:
        rt = self._rt
        if rt is None:
            raise NullArgument("E0302", "the runtime is closed", "Open a new Runtime.", {"arg": "rt"})
        return rt

    def _forget_session(self, session: ListenSession) -> None:
        self._sessions.discard(session)

    def close(self) -> None:
        """Close the runtime (idempotent): live ``listen`` sessions are abandoned, requests in
        flight are stopped, queued ones answered ``cancelled``.

        The runtime counts as closed only when the library's ``close`` succeeded. Called from
        inside an ``on_event`` callback it raises ``Reentrant`` (``E0303``) and nothing changes:
        the runtime and its sessions stay open.
        """
        with self._lock:
            rt = self._rt
            if rt is None:
                return
            for session in list(self._sessions):
                session.close(flush=False)  # E0303 from a callback propagates; nothing is closed
            rc = self.library.close(rt)
            if rc != 0:
                raise error_for(self.library.last_error() or _synthetic(rc))
            self._rt = None

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    # -- plumbing -----------------------------------------------------------------------------

    def _json(self, rc: int, raw: Optional[bytes]) -> Any:
        if rc != 0:
            raise error_for(self.library.last_error() or _synthetic(rc))
        return loads(raw) if raw else None

    # -- running ------------------------------------------------------------------------------

    def run(
        self,
        request: Union["Request", JsonInput],
        *,
        request_id: Optional[str] = None,
        raise_on_error: bool = True,
    ) -> "Response":
        """Run a request (docs/request.md) and wait for the response.

        ``request`` is a dict, JSON text or JSON bytes. ``request_id`` names it for ``cancel``
        from another thread. A response that ended ``finish: "error"`` raises the error of
        ``response.error`` (``.response`` holds the whole response, ``.response["latai"]["started"]
        is False`` for an error before the run began); with ``raise_on_error=False`` it is
        returned instead. ``cancelled`` and ``length`` are normal returns. A request that does not
        parse (``BadRequest``, ``E0301``) always raises.
        """
        rc, raw = self.library.run(self._handle(), to_json_bytes(request), request_id)
        wire = self.library.last_error() if rc != 0 else None  # same thread as the call
        if raw is None:
            raise error_for(wire or _synthetic(rc))
        response = loads(raw)
        if rc != 0 and raise_on_error:
            err = response.get("error") or wire or _synthetic(rc)
            raise error_for(err, response=response)
        return response  # type: ignore[no-any-return]

    def run_stream(
        self,
        request: Union["Request", JsonInput],
        *,
        request_id: Optional[str] = None,
        raise_on_error: bool = False,
        queue_size: int = 64,
    ) -> EventStream:
        """Run a request and iterate its events (``delta``..., then one ``final`` or ``error``).

        ``"stream": true`` is set on the request for you. Nothing starts until
        the first ``next()``. See ``EventStream`` for cancelling and closing and for why errors are
        events here (``raise_on_error=False``, unlike ``run``). ``request_id`` is generated when
        not given.
        """
        rid = request_id if request_id is not None else "py-" + uuid.uuid4().hex
        return EventStream(self, with_stream(request), rid, raise_on_error, queue_size)

    def cancel(self, request_id: str) -> None:
        """Cancel a request by id, from any thread; unknown or finished ids do nothing."""
        self._json(self.library.cancel(self._handle(), request_id), None)

    def session_close(self, session: str) -> None:
        """End a ``session`` (a KV-reuse hint): its next request runs cold. Unknown ids do nothing."""
        self._json(self.library.session_close(self._handle(), session), None)

    # -- streams of audio ---------------------------------------------------------------------

    def listen(
        self,
        open_request: Union["StreamOpen", JsonInput],
        *,
        on_event: Optional[Any] = None,
        raise_on_error: bool = False,
    ) -> ListenSession:
        """Open a stream session on a speech pack. ``open_request`` is a ``StreamOpen`` dict
        (``{"pack": "asr", "format": "pcm16", ...}``), its JSON, or just a pack name.
        See ``ListenSession``."""
        if isinstance(open_request, str) and not open_request.lstrip().startswith("{"):
            open_request = {"pack": open_request}
        session = ListenSession(self, open_request, on_event, raise_on_error)
        self._sessions.add(session)
        return session

    def listen_iter(
        self, open_request: Union["StreamOpen", JsonInput], chunks: Iterable[Any]
    ) -> Iterator["StreamEvent"]:
        """Push each of ``chunks`` (bytes) into a new session and yield every event, the ``final``
        last. Runs on the calling thread."""
        if isinstance(open_request, str) and not open_request.lstrip().startswith("{"):
            open_request = {"pack": open_request}
        return listen_iter(self, open_request, chunks)

    # -- information and packs ----------------------------------------------------------------

    def info(self) -> "RuntimeInfo":
        """Version, thread budget, engine variant, data directory, loaded packs, plug-ins."""
        rc, raw = self.library.info(self._handle())
        if rc != 0:
            raise error_for(self.library.last_error() or _synthetic(rc))
        return loads(raw)  # type: ignore[no-any-return]

    def status(self, name: Optional[str] = None) -> Union["Status", "RuntimeInfo"]:
        """The state of a pack from local state only; ``None`` gives the ``RuntimeInfo`` of ``info``."""
        return self._json(*self.library.status(self._handle(), name))  # type: ignore[no-any-return]

    def inspect(self, target: str) -> "PackInfo":
        """What a pack says about itself, without reading a tensor; ``target`` is a pack file or
        the name of an installed pack."""
        return self._json(*self.library.inspect(self._handle(), str(target)))  # type: ignore[no-any-return]

    def stage(self, path: str) -> "StageReceipt":
        """Check a pack file and its ``.minisig`` and store it as a version that is not active."""
        return self._json(*self.library.stage(self._handle(), str(path)))  # type: ignore[no-any-return]

    def pull(self, url: str, force: bool = False) -> "StageReceipt":
        """Download the pack at ``url`` (and ``<url>.minisig``) and stage it; ``force`` ignores the
        ``[pull]`` rules (``E0802``)."""
        return self._json(*self.library.pull(self._handle(), url, force))  # type: ignore[no-any-return]

    def install(self, source: str, activate: bool = True) -> List["StageReceipt"]:
        """Stage a pack file, or every pack file of a folder, and (unless ``activate`` is false)
        make it serve. Returns one receipt per file."""
        return self._json(*self.library.install(self._handle(), str(source), activate))  # type: ignore[no-any-return]

    def activate(self, name: str, version: Optional[str] = None) -> "PackVersion":
        """Make a version of ``name`` the one that serves; ``version`` (a serial, a folder name or
        a sha256) defaults to the newest one the pin allows."""
        return self._json(*self.library.activate(self._handle(), name, version))  # type: ignore[no-any-return]

    def rollback(self, name: str) -> "PackVersion":
        """Bring the retained version of ``name`` back (``NoRollbackTarget``, ``E0803``, if none)."""
        return self._json(*self.library.rollback(self._handle(), name))  # type: ignore[no-any-return]

    def list_packs(self) -> List["PackVersion"]:
        """Every installed version with its state and pin."""
        return self._json(*self.library.list(self._handle()))  # type: ignore[no-any-return]

    def pin(self, name: str, pin: Optional[str] = None) -> Optional[str]:
        """Pin ``name`` to a sha256 or ``"channel:NAME"``; ``pin=None`` clears it. Returns the pin
        now in force, or ``None`` after a clear."""
        return self._json(*self.library.pin(self._handle(), name, pin))  # type: ignore[no-any-return]
