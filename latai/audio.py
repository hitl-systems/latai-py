# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Audio helpers: WAV / raw PCM16 to the ``audio`` input of a request (docs/request.md#audio).

latai takes **16-bit PCM, one channel, at the sample rate the pack declares** (16 kHz for the
fixture packs) and never resamples (another rate is ``E0319``). These helpers do not resample
either; they only read, check and wrap the bytes.
"""

from __future__ import annotations

import base64
import os
import struct
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Union

__all__ = [
    "AudioError",
    "WavClip",
    "audio_input",
    "audio_request",
    "pcm16_chunks",
    "read_wav",
    "text_request",
]

AudioSource = Union[str, "os.PathLike[str]", bytes, bytearray, memoryview]


class AudioError(ValueError):
    """The bytes are not audio latai takes (before any call is made)."""


class WavClip(NamedTuple):
    """A parsed WAV file: the samples and what the header said."""

    pcm: bytes  #: raw 16-bit little-endian samples, one channel
    sample_rate: int
    channels: int
    bits: int

    @property
    def seconds(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate if self.sample_rate else 0.0


def _bytes_of(source: AudioSource) -> bytes:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    with open(os.fspath(source), "rb") as fh:
        return fh.read()


def read_wav(source: AudioSource) -> WavClip:
    """Parse a RIFF/WAVE file (a path or its bytes): 16-bit PCM, one channel (also as
    ``WAVE_FORMAT_EXTENSIBLE`` PCM). A data size of 0 or ``0xFFFFFFFF`` (live writers) means
    until the end. Raises ``AudioError`` for anything else (the same cases as ``E0318``)."""
    data = _bytes_of(source)
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("not a RIFF/WAVE file")
    pos = 12
    fmt: Optional[Dict[str, int]] = None
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = pos + 8
        if cid == b"fmt ":
            if size < 16 or body + 16 > len(data):
                raise AudioError("truncated fmt chunk")
            tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", data, body)
            if tag == 0xFFFE and size >= 26 and body + 26 <= len(data):  # WAVE_FORMAT_EXTENSIBLE: the sub-format starts the GUID
                tag = struct.unpack_from("<H", data, body + 24)[0]
            fmt = {"tag": tag, "channels": channels, "rate": rate, "bits": bits}
        elif cid == b"data":
            if fmt is None:
                raise AudioError("data chunk before fmt chunk")
            if fmt["tag"] != 1 or fmt["bits"] != 16:
                raise AudioError(f"only 16-bit PCM is accepted (format tag {fmt['tag']}, {fmt['bits']} bits)")
            if fmt["channels"] != 1:
                raise AudioError(f"only one channel is accepted (got {fmt['channels']})")
            end = len(data) if size in (0, 0xFFFFFFFF) else min(body + size, len(data))
            pcm = data[body:end]
            if len(pcm) % 2:
                raise AudioError("odd number of PCM bytes")
            return WavClip(pcm, fmt["rate"], fmt["channels"], fmt["bits"])
        pos = body + size + (size & 1)
    raise AudioError("no data chunk")


def audio_input(
    source: AudioSource,
    *,
    format: Optional[str] = None,
    sample_rate: Optional[int] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    prompt_before: Optional[str] = None,
    outputs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """The ``audio`` input of a request: ``{"type": "audio", "data": <base64>, ...}``.

    ``source`` is a path or bytes. A WAV file is sent whole (``format`` ``"wav"``; detected from
    the ``RIFF....WAVE`` magic); anything else is raw 16-bit little-endian samples (``"pcm16"``)
    and wants ``sample_rate`` unless the pack's rate is right. ``outputs`` is a list of
    ``"timestamps"``, ``"language"``, ``"embeddings"`` (each must be declared by the pack).
    Clips are limited to 4 MiB of audio and 120 s by the library.
    """
    raw = _bytes_of(source)
    fmt = format or ("wav" if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE" else "pcm16")
    if fmt not in ("wav", "pcm16"):
        raise AudioError(f"format must be 'wav' or 'pcm16', not {fmt!r}")
    inp: Dict[str, Any] = {"type": "audio", "data": base64.b64encode(raw).decode("ascii")}
    if fmt != "wav":
        inp["format"] = fmt
    if sample_rate is not None:
        inp["sample_rate"] = int(sample_rate)
    if language is not None:
        inp["language"] = language
    if prompt is not None:
        inp["prompt"] = prompt
    if prompt_before is not None:
        inp["prompt_before"] = prompt_before
    if outputs:
        inp["outputs"] = list(outputs)
    return inp


def audio_request(pack: str, source: AudioSource, **options: Any) -> Dict[str, Any]:
    """A request that transcribes a clip: ``{"pack", "inputs": audio_input(source, ...)}``.

    Keyword arguments of ``audio_input`` go to the input; ``output`` (a dict), ``deadline_ms``,
    ``expect_digest``, ``session`` and ``then`` go to the request.
    """
    request_keys = ("output", "deadline_ms", "expect_digest", "session", "then", "stream")
    request_part = {k: options.pop(k) for k in request_keys if k in options}
    request: Dict[str, Any] = {"pack": pack, "inputs": audio_input(source, **options)}
    request.update(request_part)
    return request


def text_request(
    pack: str,
    text: str,
    *,
    max_tokens: Optional[int] = None,
    session: Optional[str] = None,
    deadline_ms: Optional[int] = None,
    **output: Any,
) -> Dict[str, Any]:
    """A request with a ``text`` input; ``max_tokens`` and other keywords fill ``output``."""
    request: Dict[str, Any] = {"pack": pack, "inputs": {"type": "text", "text": text}}
    out = dict(output)
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    if out:
        request["output"] = out
    if session is not None:
        request["session"] = session
    if deadline_ms is not None:
        request["deadline_ms"] = deadline_ms
    return request


def pcm16_chunks(source: AudioSource, chunk_bytes: int = 3200) -> Iterator[bytes]:
    """Cut raw samples (bytes or a path) into pieces of ``chunk_bytes`` (3200 = 100 ms at 16 kHz),
    as a microphone would deliver them, for ``ListenSession.push``. A WAV file is not unwrapped
    here: push its bytes to a stream opened with ``"format": "wav"``, or give
    ``read_wav(path).pcm`` to this function."""
    if chunk_bytes < 1 or chunk_bytes > 1 << 20:
        raise ValueError("chunk_bytes is 1 to 1048576")
    if isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source)
        for start in range(0, len(raw), chunk_bytes):
            yield raw[start : start + chunk_bytes]
        return
    with open(os.fspath(source), "rb") as fh:
        while True:
            piece = fh.read(chunk_bytes)
            if not piece:
                return
            yield piece
