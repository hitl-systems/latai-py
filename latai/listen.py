# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Stream sessions: audio in chunks, ``partial`` / ``committed`` events out (docs/audio.md).

A session is a thin object over ``stream_open`` / ``stream_push`` / ``stream_cancel`` /
``stream_close``. The C library has no thread for it: the events of a call are delivered **on the
thread that calls ``push`` or ``close``**, before the call returns, so ``push`` simply *returns*
the events that call produced. A producer faster than the engine waits inside ``push``
(backpressure); nothing is queued without bound.
"""

from __future__ import annotations

import ctypes
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional

from ._json import JsonInput, loads, to_json_bytes
from ._lib import EVENT_CALLBACK
from .errors import LataiError, StreamState, error_for

if TYPE_CHECKING:  # pragma: no cover
    from .runtime import Runtime
    from .types import Response, StreamEvent

#: The most bytes the C ABI takes in one ``stream_push``; a larger buffer is split.
MAX_PUSH_BYTES = 1 << 20

__all__ = ["ListenSession", "MAX_PUSH_BYTES"]


class ListenSession:
    """An open stream session on a speech pack. Made by ``Runtime.listen``.

    * ``push(data)`` gives audio (raw 16-bit little-endian mono samples, or WAV bytes when the
      stream was opened with ``"format": "wav"``) and returns the events delivered in that call.
    * ``close(flush=True)`` ends the audio (the last utterance is decoded; ``finish`` is
      ``stop``) and returns the remaining events including the ``final``; ``flush=False``
      abandons the stream (``cancelled``). Always close: it releases the handle.
    * ``cancel()`` may be called from any thread, also while a ``push`` is running there.
    * ``on_event`` (optional) is called for every event on the pushing thread. Do not call
      ``push``, ``close``, ``Runtime.run`` or the pack calls from it (``E0303``); return a truthy
      value to end the stream (it still ends with one ``final``, ``cancelled``).

    **Events are never raised.** A call that ends the stream with an error (a cap ``E0312``, bad
    audio ``E0318``, a failed pass) *returns* the events it delivered, the ``final`` with
    ``finish: "error"`` among them; ``session.ended`` is true and ``session.error`` holds the wire
    error. Only a call that delivers no terminal event raises (``StreamState`` ``E0320`` after the
    end, ``E0302`` / ``E0303``), and so does ``Runtime.listen`` when the stream cannot open (the
    error is also its one ``error`` event, in ``.events``). With ``raise_on_error=True`` a failed
    terminal event raises the typed error after the call, ``.events`` holding the events. Thread-safety:
    calls on one session are serialised by a lock here (and by the library); give one thread the
    pushes if audio order matters to you.
    """

    def __init__(
        self,
        runtime: "Runtime",
        open_request: JsonInput,
        on_event: Optional[Callable[[Dict[str, Any]], Any]] = None,
        raise_on_error: bool = False,
    ) -> None:
        self._runtime = runtime
        self._raise = raise_on_error
        #: The wire error that ended the stream (a failed ``final`` or an ``error`` event), if any.
        self.error: Optional[Dict[str, Any]] = None
        self._lib = runtime.library
        self._on_event = on_event
        self._call_lock = threading.RLock()
        self._events: List[Dict[str, Any]] = []
        self._cb_error: Optional[BaseException] = None
        self._handle: Optional[int] = None
        self._closed = False
        #: Whether the stream has ended (a ``final`` was seen, or a call failed); the handle must
        #: still be closed.
        self.ended = False
        #: The response of the ``final`` event, once seen.
        self.final: Optional["Response"] = None
        self._cb = EVENT_CALLBACK(self._callback)  # kept alive for the life of the stream
        raw = to_json_bytes(open_request)
        with self._call_lock:
            rc, handle = self._lib.stream_open(runtime._handle(), raw, self._cb)
            if rc != 0:
                err = self._error(rc)
                self._closed = True
                raise err
            self._handle = handle

    # -- the callback -------------------------------------------------------------------------

    def _callback(self, _user: Any, ptr: Any, length: int) -> int:
        try:
            event = loads(ctypes.string_at(ptr, length))
            self._events.append(event)
            kind = event.get("event")
            if kind == "final":
                self.ended = True
                self.final = event.get("response")
                self.error = (self.final or {}).get("error") or self.error
            elif kind == "error":
                self.ended = True
                self.error = event.get("error")
            if self._on_event is not None and self._on_event(event):
                return 1
            return 0
        except BaseException as exc:  # never let an exception cross into C
            self._cb_error = exc
            return 1

    def _take_events(self) -> List[Dict[str, Any]]:
        events, self._events = self._events, []
        return events

    def _error(self, rc: int) -> LataiError:
        # Same thread as the failing call: last_error is per thread.
        wire = self._lib.last_error() or {"code": rc}
        return error_for(wire, events=self._take_events())

    def _after_call(self, rc: int) -> List[Dict[str, Any]]:
        cb_error, self._cb_error = self._cb_error, None
        if rc != 0:
            # Whether the stream ended is told by the events (`_callback` sets `ended` on a
            # `final` or `error`): a refused call (E0302, E0303, E0320) ends nothing.
            if cb_error is not None:
                self._take_events()
                raise cb_error
            events = self._take_events()
            # The events of the call are the answer when a terminal one is among them, unless the
            # caller asked for exceptions; no terminal event (E0302, E0303, E0320) is an exception.
            if not self._raise and any(e.get("event") in ("final", "error") for e in events):
                return events
            wire = self._lib.last_error() or {"code": rc}
            raise error_for(wire, events=events)
        if cb_error is not None:
            self._take_events()
            raise cb_error
        return self._take_events()

    # -- calls --------------------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def transcript(self) -> str:
        """The transcript of the ``final`` (empty before it)."""
        if not self.final:
            return ""
        out = self.final.get("output") or {}
        return str(out.get("text") or "")

    def push(self, data: Any) -> List["StreamEvent"]:
        """Give the stream audio; return the events delivered during this call (maybe none).

        ``data`` is any bytes-like object. A buffer above 1 MiB is split into pushes of at most
        1 MiB. Raises ``StreamState`` (``E0320``) after the stream ended or the session closed.
        """
        view = memoryview(data).cast("B")
        events: List[Dict[str, Any]] = []
        with self._call_lock:
            if self._closed:
                raise StreamState("E0320", "the stream session is closed", "Open a new session.", {"state": "closed"})
            for start in range(0, len(view), MAX_PUSH_BYTES) or [0]:
                chunk = bytes(view[start : start + MAX_PUSH_BYTES])
                rc = self._lib.stream_push(self._handle, chunk)
                try:
                    events += self._after_call(rc)
                except LataiError as exc:
                    exc.events = events + exc.events
                    raise
                if self.ended:
                    break
        return events  # type: ignore[return-value]

    def cancel(self) -> None:
        """Stop the stream from any thread; its ``final`` (``cancelled``) arrives with the push or
        close that is running or comes next."""
        handle = self._handle
        if handle is not None and not self._closed:
            self._lib.stream_cancel(handle)

    def close(self, flush: bool = True) -> List["StreamEvent"]:
        """End the stream and release its handle; return the events of this call.

        With ``flush`` the audio is over: the last utterance is decoded and the ``final`` has
        ``finish: "stop"``. Without it the stream is abandoned (``cancelled``). Closing twice
        returns ``[]``.
        """
        with self._call_lock:
            if self._closed:
                return []
            rc = self._lib.stream_close(self._handle, flush)
            if rc == 303 or (rc != 0 and (self._lib.last_error() or {}).get("code") == "E0303"):
                # Refused: called from inside an event callback. The handle is still open.
                raise error_for(self._lib.last_error() or {"code": rc}, events=self._take_events())
            self._closed = True
            self._runtime._forget_session(self)
            return self._after_call(rc)  # type: ignore[return-value]

    def __enter__(self) -> "ListenSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # A clean exit ends the audio; an exception abandons the stream. The events of the final
        # close are in ``self.final`` (call ``close()`` yourself to get them).
        try:
            self.close(flush=exc_type is None)
        except LataiError:
            if exc_type is None:
                raise

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            if not self._closed and self._handle is not None:
                self.close(flush=False)
        except Exception:
            pass


def listen_iter(
    runtime: "Runtime", open_request: JsonInput, chunks: Iterable[Any]
) -> Iterator["StreamEvent"]:
    """Push ``chunks`` into a new session and yield every event, the ``final`` last.

    Runs on the calling thread: the iterator drives the pushes, so it is as slow as the engine.
    Closing the generator early abandons the stream.
    """
    session = ListenSession(runtime, open_request)
    finished = False
    try:
        for chunk in chunks:
            try:
                events = session.push(chunk)
            except LataiError as exc:
                yield from exc.events
                raise
            yield from events
            if session.ended:
                break
        finished = True
        try:
            yield from session.close(flush=True)
        except LataiError as exc:
            yield from exc.events
            raise
    finally:
        if not finished or not session.closed:
            try:
                session.close(flush=False)
            except LataiError:
                pass
