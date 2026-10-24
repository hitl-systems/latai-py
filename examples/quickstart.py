#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Install a pack, run it, stream it, cancel a stream, and (with a speech pack) transcribe a stream.

    python sdk/python/examples/quickstart.py PACK.gguf [SPEECH_PACK.gguf]

* ``LATAI_DATA_DIR`` is the data directory (the pack key list and ``latai.toml`` live there; see
  docs/quickstart.md). A pack is installed only if its signature is from a key in the list.
* ``LATAI_LIB`` points to ``liblatai`` when it is not found by itself (``latai.library_candidates``).
* ``PACK.gguf.minisig`` must sit next to the pack, as for ``latai install``.

The script prints one line per step and ``ok`` at the end; any failure raises (exit status 1).
Run from a source checkout with ``PYTHONPATH=sdk/python`` or after ``pip install sdk/python``.
"""

import math
import struct
import sys

import latai


def synthetic_clip(rate: int = 16_000) -> bytes:
    """Half a second of silence, a tone, a pause of 1.5 s (an utterance ends there), a second tone
    and a silent tail, as raw 16-bit little-endian samples: what ``cargo xtask examples`` uses."""

    def tone(seconds: float, hz: float):
        return [
            int(8000 * (1 + 0.5 * math.sin(i / rate * 17)) * math.sin(2 * math.pi * hz * i / rate))
            for i in range(int(seconds * rate))
        ]

    silence = lambda seconds: [0] * int(seconds * rate)  # noqa: E731
    samples = silence(0.5) + tone(2.5, 300) + silence(1.5) + tone(2.0, 520) + silence(1.5)
    return struct.pack(f"<{len(samples)}h", *samples)


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    pack_file = argv[1]
    speech_file = argv[2] if len(argv) > 2 else None

    # One runtime per process. The context manager closes it (and any open stream) on exit.
    with latai.Runtime.open() as rt:
        # Install: checks the signature, measures the pack in an isolated process, makes it serve.
        receipts = rt.install(pack_file)
        pack = receipts[0]["name"]
        print(f"installed: {pack}")

        # A run: a dict in, a response dict out. Errors are typed exceptions (latai.UnknownPack ...).
        response = rt.run(latai.text_request(pack, "hello", max_tokens=8))
        print(f"run: {response['output']['kind']} finish={response['finish']} "
              f"total_ms={response['latai']['timing']['total_ms']:.1f}")

        try:
            rt.run(latai.text_request("no-such-pack", "hello"))
        except latai.UnknownPack as err:
            print(f"error: {err.code} {err.details}")

        # A stream: deltas, then exactly one final. Here we cancel after five deltas; the run still
        # ends with one final whose finish is "cancelled".
        deltas = 0
        stream = rt.run_stream(latai.text_request(pack, "hello", max_tokens=64))
        with stream:
            for event in stream:
                if event["event"] == "delta":
                    deltas += 1
                    if deltas == 5:
                        stream.cancel()
        print(f"stream: {deltas} deltas, final {stream.final['finish']}")

        if speech_file:
            speech = rt.install(speech_file)[0]["name"]
            # Audio in chunks of 100 ms, as a microphone would deliver them; the events (partial,
            # committed, utterance, ..., final) come out as they happen.
            kinds = [e["event"] for e in rt.listen_iter({"pack": speech}, latai.pcm16_chunks(synthetic_clip()))]
            assert kinds[-1] == "final", kinds
            print(f"listen: {len(kinds)} events, {kinds.count('utterance')} utterances, "
                  f"{kinds.count('committed')} committed")

    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
