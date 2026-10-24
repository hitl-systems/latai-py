# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The base class of every error latai raises (the generated subclasses are in `_errors_gen`)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class LataiError(Exception):
    """A latai error: the wire error ``{"code", "message", "hint", "details"?}`` as an exception.

    Attributes:
        code: the stable error code as text (``"E0316"``); empty for errors that did not come
            from the library (``LibraryNotFoundError``). Programs branch on this, or on the class.
        message: what went wrong, for people.
        hint: what the caller can do about it.
        details: the typed fields of this code (``{"name": ...}`` for ``E0316``), or ``None``.
        retryable: whether trying the same call again later can work (from ``schema/errors.json``).
        category: the two-digit group (``"E03"`` is a bad request).
        response: when a run ended with ``finish: "error"``, the whole response (with its
            ``latai.timing`` and ``latai.started``); else ``None``.
        events: for a stream call that failed, the events that call delivered (the ``final`` is
            among them when the stream got that far); else an empty list.
    """

    #: The code of the class (set on the generated per-code subclasses).
    default_code: str = ""
    #: The category of the class, ``"E03"`` for a request error (set on the category classes).
    category: str = ""
    retryable: bool = False

    def __init__(
        self,
        code: Optional[str] = None,
        message: str = "",
        hint: str = "",
        details: Optional[Dict[str, Any]] = None,
        *,
        response: Optional[Dict[str, Any]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        self.code: str = code if code is not None else self.default_code
        self.message: str = message
        self.hint: str = hint
        self.details: Optional[Dict[str, Any]] = details
        if retryable is not None:
            self.retryable = retryable
        self.response = response
        self.events: List[Dict[str, Any]] = list(events) if events else []
        super().__init__(self.code, message, hint)

    def __str__(self) -> str:
        head = f"{self.code}: {self.message}" if self.code else self.message
        return f"{head} (hint: {self.hint})" if self.hint else head

    def to_wire(self) -> Dict[str, Any]:
        """The wire form: ``{"code", "message", "hint", "details"?}``."""
        out: Dict[str, Any] = {"code": self.code, "message": self.message, "hint": self.hint}
        if self.details is not None:
            out["details"] = self.details
        return out
