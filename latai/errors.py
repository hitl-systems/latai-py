# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Typed errors: one class per error category and per error code.

``LataiError`` is the base. ``RequestError`` (``E03xx``), ``BusyError`` (``E07xx``) and the other
category classes group the codes by range; ``UnknownPack`` (``E0316``) and the other classes are
the codes themselves, so ``except UnknownPack`` and ``except RequestError`` both work. The
classes come from ``schema/errors.json`` (``sdk/tools/gen_types.py``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Type, Union

from ._errors_base import LataiError
from ._errors_gen import *  # noqa: F401,F403
from ._errors_gen import CATEGORY_CLASSES, ERROR_CLASSES, ERROR_HINTS, RETRYABLE_CODES
from ._errors_gen import __all__ as _generated

__all__ = [
    "LataiError",
    "LibraryNotFoundError",
    "UnsupportedError",
    "error_for",
    "code_text",
    *_generated,
]


class LibraryNotFoundError(LataiError):
    """``liblatai`` was not found, or was found but could not be loaded.

    ``tried`` lists the candidates that were looked at. Set ``LATAI_LIB`` to the library (or the
    folder that holds it) or pass ``lib=`` to ``Runtime.open``.
    """

    def __init__(self, message: str, tried: Optional[List[str]] = None, hint: str = "") -> None:
        super().__init__(
            "",
            message,
            hint or "Set LATAI_LIB to the path of liblatai (or its folder), or pass lib= to Runtime.open.",
        )
        self.tried: List[str] = list(tried or [])


class UnsupportedError(LataiError):
    """The loaded library is older than the call needs (for example no stream entries: API v1)."""

    def __init__(self, message: str) -> None:
        super().__init__("", message, "Use a liblatai of API version 2 or later.")


def code_text(code: Union[int, str]) -> str:
    """The text of a code: ``316`` and ``"E0316"`` are both ``"E0316"``."""
    if isinstance(code, int):
        return f"E{code:04d}"
    return code


def error_for(
    wire: Union[Mapping[str, Any], int, str, None],
    *,
    response: Optional[Dict[str, Any]] = None,
    events: Optional[List[Dict[str, Any]]] = None,
) -> LataiError:
    """Build the exception for a wire error ``{"code","message","hint","details"?}``.

    The class is the one of the code, else the one of its category, else ``LataiError``. A bare
    code (``316`` or ``"E0316"``) gives an error whose hint is the one of ``schema/errors.json``
    and whose message only names the code.
    """
    if wire is None:
        wire = {}
    if isinstance(wire, (int, str)):
        wire = {"code": code_text(wire)}
    code = code_text(wire.get("code") or "")
    cls: Type[LataiError] = ERROR_CLASSES.get(code) or CATEGORY_CLASSES.get(code[:3]) or LataiError
    message = wire.get("message") or (f"latai call failed with {code}" if code else "latai call failed")
    return cls(
        code,
        message,
        wire.get("hint") or ERROR_HINTS.get(code, ""),
        wire.get("details"),
        response=response,
        events=events,
        retryable=code in RETRYABLE_CODES,
    )
