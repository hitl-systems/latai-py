# SPDX-License-Identifier: Apache-2.0 OR MIT
"""JSON in and out, shared by the runtime and the stream session."""

from __future__ import annotations

import json
from typing import Any, Mapping, Union

JsonInput = Union[Mapping[str, Any], str, bytes, bytearray, memoryview]


def to_json_bytes(value: JsonInput) -> bytes:
    """A request given as a dict (or any JSON-able mapping), JSON text or UTF-8 bytes, as bytes."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def loads(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"))


def with_stream(value: JsonInput) -> bytes:
    """The request as JSON bytes with ``"stream": true``, which ``run_stream`` needs for the run
    to send ``delta`` events (the other fields are kept as they are)."""
    if isinstance(value, Mapping):
        return to_json_bytes({**value, "stream": True})
    raw = to_json_bytes(value)
    try:
        obj = loads(raw)
    except ValueError:
        return raw  # not JSON: let the library answer E0301
    if isinstance(obj, dict):
        obj["stream"] = True
        return to_json_bytes(obj)
    return raw  # not an object: let the library answer E0301
