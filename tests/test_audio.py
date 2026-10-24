# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The audio helpers: a WAV or PCM16 file becomes the same request the CLI builds for the file
(`latai run PACK --audio FILE`, bins/latai/src/cli/run.rs: base64 of the file's bytes, `format`
`wav` when the file starts with RIFF and `pcm16` otherwise, the options as given)."""

import base64
import json
import struct
import wave

import pytest

import latai
from latai import audio
from conftest import sine_pcm16, write_wav


def norm(request):
    """`format` is `wav` when absent (schema, docs/request.md): the SDK leaves the default out where the
    CLI writes it, and the two requests mean the same. Compare after filling the default in."""
    out = json.loads(json.dumps(request))
    out["inputs"].setdefault("format", "wav")
    return out


def cli_request(pack, data: bytes, fmt=None, **opts):
    """What the CLI builds for the file."""
    f = fmt or ("wav" if data.startswith(b"RIFF") else "pcm16")
    inputs = {"type": "audio", "data": base64.b64encode(data).decode(), "format": f}
    for k in ("sample_rate", "language", "prompt", "prompt_before"):
        if opts.get(k) is not None:
            inputs[k] = opts[k]
    if opts.get("outputs"):
        inputs["outputs"] = list(opts["outputs"])
    return {"pack": pack, "inputs": inputs}


def test_wav_file_request_equals_the_cli_request(tmp_path):
    pcm = sine_pcm16(0.5)
    path = write_wav(tmp_path / "a.wav", pcm)
    data = path.read_bytes()
    assert data.startswith(b"RIFF")
    want = cli_request("asr", data)
    assert norm(audio.audio_request("asr", str(path))) == want
    assert norm(audio.audio_request("asr", path)) == want  # a Path too
    assert norm(audio.audio_request("asr", data)) == want  # bytes of a WAV file
    assert norm({"inputs": audio.audio_input(str(path))})["inputs"] == want["inputs"]
    # the same JSON, key order aside
    assert json.dumps(norm(audio.audio_request("asr", str(path))), sort_keys=True) == json.dumps(want, sort_keys=True)


def test_raw_pcm16_file_request_equals_the_cli_request(tmp_path):
    pcm = sine_pcm16(0.25)
    path = tmp_path / "a.pcm"
    path.write_bytes(pcm)
    want = cli_request("asr", pcm)
    assert want["inputs"]["format"] == "pcm16"
    assert audio.audio_request("asr", str(path)) == want
    assert audio.audio_request("asr", pcm, format="pcm16") == want


def test_options_are_carried_like_the_cli_flags(tmp_path):
    path = write_wav(tmp_path / "a.wav", sine_pcm16(0.1))
    data = path.read_bytes()
    kw = dict(language="en", prompt="after", prompt_before="before", outputs=["timestamps", "language"])
    want = cli_request("asr", data, **kw)
    assert norm(audio.audio_request("asr", str(path), **kw)) == want
    want = cli_request("asr", sine_pcm16(0.1), fmt="pcm16", sample_rate=8000)
    assert audio.audio_input(sine_pcm16(0.1), format="pcm16", sample_rate=8000) == want["inputs"]
    # nothing optional that was not given (format is the default `wav` and may be left out)
    inp = audio.audio_input(str(path))
    assert set(inp) - {"format"} == {"type", "data"}


def test_base64_is_standard_padded_and_decodes_back(tmp_path):
    for n in (1, 2, 3, 4, 5, 1000):
        raw = bytes(range(256)) * 4
        inp = audio.audio_input(raw[:n], format="pcm16")
        assert base64.b64decode(inp["data"], validate=True) == raw[:n]
        assert "\n" not in inp["data"]


def test_read_wav_gives_pcm_and_format(tmp_path):
    pcm = sine_pcm16(0.2, rate=16000)
    clip = audio.read_wav(write_wav(tmp_path / "a.wav", pcm))
    assert clip.pcm == pcm
    assert (clip.sample_rate, clip.channels, clip.bits) == (16000, 1, 16)
    clip2 = audio.read_wav((tmp_path / "a.wav").read_bytes())
    assert clip2.pcm == pcm
    # other rates are read as they are (the library refuses them with E0319, the helper never resamples)
    other = audio.read_wav(write_wav(tmp_path / "r.wav", pcm, rate=8000))
    assert other.sample_rate == 8000
    # the library takes one channel only: the reader refuses two before any call is made
    with pytest.raises(audio.AudioError):
        audio.read_wav(write_wav(tmp_path / "s.wav", pcm * 2, rate=8000, channels=2))


def test_read_wav_rejects_garbage_and_non_pcm16(tmp_path):
    with pytest.raises(Exception):
        audio.read_wav(b"this is not a wav file at all")
    with pytest.raises(Exception):
        audio.read_wav(b"RIFF")
    p = tmp_path / "eight.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(16000)
        w.writeframes(b"\x80" * 100)
    # an 8-bit file is not 16-bit PCM: either refused by the reader or reported by .bits
    try:
        clip = audio.read_wav(p)
    except Exception:
        return
    assert clip.bits == 8


def test_missing_audio_file_is_an_error(tmp_path):
    with pytest.raises((OSError, latai.LataiError)):
        audio.audio_input(str(tmp_path / "missing.wav"))


def test_pcm16_chunks_cover_the_data_exactly():
    data = sine_pcm16(1.0)
    chunks = list(audio.pcm16_chunks(data))
    assert b"".join(chunks) == data
    assert all(len(c) == 3200 for c in chunks[:-1])
    assert 0 < len(chunks[-1]) <= 3200
    small = list(audio.pcm16_chunks(data, chunk_bytes=1001))  # odd sizes may split a sample
    assert b"".join(small) == data
    assert all(len(c) <= 1001 for c in small)
    assert list(audio.pcm16_chunks(b"")) == []


def test_text_request_shape():
    assert audio.text_request("p", "hi") == {"pack": "p", "inputs": {"type": "text", "text": "hi"}}
    r = audio.text_request("p", "hi", max_tokens=8)
    assert r["output"] == {"max_tokens": 8}
    r = audio.text_request("p", "hi", max_tokens=8, stop=["x"])
    assert r["output"] == {"max_tokens": 8, "stop": ["x"]}
    json.dumps(r)
