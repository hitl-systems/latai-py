# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Stream sessions (open/push/close) and clip runs on the speech stand-in pack."""

import threading

import pytest

import latai
from latai import audio
from conftest import PACK, SPEECH, speech_clip_pcm16, strip_timing, validate, write_wav

OPEN = {"pack": SPEECH, "hop_ms": 1000, "pause_ms": 0, "window_seconds": 3}


@pytest.fixture(scope="module")
def pcm():
    """Four seconds of the fixture clips, back to back (16 kHz mono pcm16)."""
    return speech_clip_pcm16("sine") + speech_clip_pcm16("chirp") + speech_clip_pcm16("noise") + speech_clip_pcm16("chirp")


def terminals(events):
    return [e for e in events if e["event"] in ("final", "error")]


def check_rules(events, pcm_len=None):
    """The event rules of docs/audio.md that hold on every interface."""
    assert events, "no events"
    assert len(terminals(events)) == 1, [e["event"] for e in events]
    assert events[-1]["event"] == "final"
    last_at = 0
    committed = []
    for e in events[:-1]:
        assert e["event"] in ("partial", "committed", "utterance", "then"), e
        if e["event"] in ("partial", "committed"):
            assert e["at_ms"] >= last_at, "at_ms never goes back"
            last_at = e["at_ms"]
        if e["event"] == "committed":
            committed.append(e["text"])
    final = events[-1]["response"]
    validate("Response", final)
    assert final["output"]["kind"] == "transcript"
    if pcm_len is not None:
        st = final["latai"]["stream"]
        assert st["bytes"] == pcm_len
        assert st["audio_ms"] == pcm_len // 2 * 1000 // 16000
    return final, committed


def run_session(rt, pcm, chunk=3200, **opts):
    s = rt.listen({**OPEN, **opts})
    events = []
    for i in range(0, len(pcm), chunk):
        events += s.push(pcm[i : i + chunk])
    events += s.close()
    return s, events


def stripped(events):
    out = []
    for e in events:
        e = latai_copy(e)
        if e["event"] == "final":
            e["response"]["latai"]["timing"] = 0
            e["response"]["latai"]["stream"]["buffered_samples_peak"] = 0
        out.append(e)
    return out


def latai_copy(e):
    import json

    return json.loads(json.dumps(e))


def test_clip_run_from_a_wav_file_and_from_raw_pcm_agree(speech_rt, pcm, tmp_path):
    wav = write_wav(tmp_path / "clip.wav", pcm)
    a = speech_rt.run(audio.audio_request(SPEECH, str(wav)))
    raw = tmp_path / "clip.pcm"
    raw.write_bytes(pcm)
    b = speech_rt.run(audio.audio_request(SPEECH, str(raw)))
    validate("Response", a)
    assert a["output"]["kind"] == "transcript"
    assert strip_timing(a)["output"] == strip_timing(b)["output"]


def test_clip_with_an_undeclared_output_is_E0315(speech_rt, pcm, tmp_path):
    """The stand-in pack declares no extra outputs: the option reaches the library, which refuses it."""
    wav = write_wav(tmp_path / "clip.wav", pcm[:64000])
    for name in ("timestamps", "language", "embeddings"):
        with pytest.raises(latai.InputInvalid) as ei:
            speech_rt.run(audio.audio_request(SPEECH, str(wav), outputs=[name]))
        assert ei.value.code == "E0315" and name in ei.value.message


def test_audio_to_a_text_pack_is_refused(speech_rt, tmp_path):
    speech_rt.install(str(__import__("conftest").fixture_file("LATAI_TEST_PACK", "packs/x4r-l8.gguf")))
    wav = write_wav(tmp_path / "c.wav", b"\x00\x01" * 1600)
    with pytest.raises(latai.InputInvalid) as ei:
        speech_rt.run(audio.audio_request(PACK, str(wav)))
    assert ei.value.code == "E0315"


def test_session_events_follow_the_rules(speech_rt, pcm):
    s, events = run_session(speech_rt, pcm)
    final, committed = check_rules(events, len(pcm))
    assert final["finish"] == "stop"
    assert s.ended is True
    assert s.final == final
    assert s.transcript == " ".join(committed) or s.transcript == final["output"]["text"]
    assert final["latai"]["stream"]["passes"] >= 4


def test_same_events_whatever_the_chunk_size(speech_rt, pcm):
    _, a = run_session(speech_rt, pcm, chunk=3200)
    _, b = run_session(speech_rt, pcm, chunk=16000)
    _, c = run_session(speech_rt, pcm, chunk=1001)  # chunks may end in the middle of a sample
    assert stripped(a) == stripped(b) == stripped(c)


def test_listen_iter_yields_the_same_events(speech_rt, pcm):
    _, manual = run_session(speech_rt, pcm)
    it = list(speech_rt.listen_iter({**OPEN}, audio.pcm16_chunks(pcm)))
    assert stripped(it) == stripped(manual)


def test_wav_format_stream(speech_rt, pcm, tmp_path):
    wav = write_wav(tmp_path / "s.wav", pcm).read_bytes()
    s = speech_rt.listen({**OPEN, "format": "wav"})
    events = []
    for i in range(0, len(wav), 3200):
        events += s.push(wav[i : i + 3200])
    events += s.close()
    final, _ = check_rules(events)
    assert final["finish"] == "stop"


def test_on_event_runs_on_the_thread_that_pushes(speech_rt, pcm):
    seen = []
    main = threading.get_ident()
    s = speech_rt.listen({**OPEN}, on_event=lambda ev: seen.append((threading.get_ident(), ev["event"])) and None)
    pusher_ids = []

    def pusher():
        pusher_ids.append(threading.get_ident())
        for i in range(0, len(pcm) // 2, 3200):
            s.push(pcm[i : i + 3200])

    t = threading.Thread(target=pusher)
    t.start()
    t.join(300)
    assert not t.is_alive()
    n_before = len(seen)
    assert n_before > 0
    assert {tid for tid, _ in seen} == {pusher_ids[0]}
    # the rest from the main thread, and the close too
    for i in range(len(pcm) // 2, len(pcm), 3200):
        s.push(pcm[i : i + 3200])
    s.close()
    mine = [tid for tid, _ in seen[n_before:]]
    assert mine and set(mine) == {main}
    assert [e for _, e in seen].count("final") == 1 and seen[-1][1] == "final"


def test_pushes_from_different_threads_in_turn(speech_rt, pcm):
    s = speech_rt.listen({**OPEN})
    events = []
    lock = threading.Lock()
    chunks = [pcm[i : i + 3200] for i in range(0, len(pcm), 3200)]
    errors = []

    def worker(part):
        try:
            for c in part:
                with lock:  # one thread at a time, in audio order
                    events.extend(s.push(c))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    half = len(chunks) // 2
    for part in (chunks[:half], chunks[half:]):
        t = threading.Thread(target=worker, args=(part,))
        t.start()
        t.join(300)
    events += s.close()
    assert not errors
    check_rules(events, len(pcm))


def test_on_event_true_ends_the_stream_cancelled(speech_rt, pcm):
    count = []

    def stop(ev):
        count.append(ev["event"])
        return True

    s = speech_rt.listen({**OPEN}, on_event=stop)
    ended = None
    for i in range(0, len(pcm), 3200):
        try:
            s.push(pcm[i : i + 3200])
        except latai.StreamState as e:
            ended = e
            break
        if s.ended:
            break
    assert s.final is not None and s.final["finish"] == "cancelled"
    assert count.count("final") == 1 and count[-1] == "final"
    # a push after the end is E0320
    with pytest.raises(latai.StreamState) as ei:
        s.push(pcm[:3200])
    assert ei.value.code == "E0320"
    try:
        s.close()
    except latai.LataiError:
        pass


def test_exception_in_on_event_does_not_crash_and_ends_the_stream_cleanly(speech_rt, pcm):
    def boom(ev):
        raise RuntimeError("boom")

    s = speech_rt.listen({**OPEN}, on_event=boom)
    try:
        for i in range(0, len(pcm), 3200):
            s.push(pcm[i : i + 3200])
        s.close()
    except (RuntimeError, latai.LataiError):
        pass
    assert speech_rt.info()["version"]
    try:
        s.close()
    except latai.LataiError:
        pass


def test_abandon_with_close_flush_false_is_cancelled(speech_rt, pcm):
    s = speech_rt.listen({**OPEN})
    s.push(pcm[:32000])
    events = s.close(flush=False)
    assert events[-1]["event"] == "final"
    assert events[-1]["response"]["finish"] == "cancelled"
    assert len(terminals(events)) == 1


def test_cancel_then_the_next_call_delivers_the_final(speech_rt, pcm):
    s = speech_rt.listen({**OPEN})
    s.push(pcm[:32000])
    s.cancel()
    try:
        evs = s.push(pcm[32000:64000])
    except latai.StreamState:
        evs = []
    try:
        evs += s.close()
    except latai.LataiError:
        pass
    assert s.final is not None and s.final["finish"] == "cancelled"


def test_use_after_close_is_a_typed_error(speech_rt, pcm):
    s = speech_rt.listen({**OPEN})
    s.close()
    assert s.closed
    with pytest.raises(latai.StreamState) as ei:
        s.push(pcm[:3200])
    assert ei.value.code == "E0320" and ei.value.hint
    s.cancel()  # nothing to cancel: harmless
    assert s.close() == []  # a second close does not crash


def test_context_manager_flushes_and_ends(speech_rt, pcm):
    with speech_rt.listen({**OPEN}) as s:
        s.push(pcm[:32000])
    assert s.ended is True
    assert s.final is not None


def test_open_errors_are_typed(speech_rt):
    with pytest.raises(latai.UnknownPack) as ei:
        speech_rt.listen({"pack": "nope"})
    assert ei.value.code == "E0316"
    with pytest.raises(latai.InputInvalid):
        speech_rt.listen({**OPEN, "hop_ms": 5})
    with pytest.raises(latai.InputInvalid):
        speech_rt.listen({"pack": PACK})  # the text pack installed above is not a speech pack
    with pytest.raises(latai.BadRequest):
        speech_rt.listen("{not json")


def test_odd_byte_end_is_an_error_final_not_an_exception(speech_rt, pcm):
    s = speech_rt.listen({**OPEN})
    s.push(pcm[:3201])  # ends in the middle of a sample
    events = s.close()  # stream events are returned, never raised
    assert events[-1]["event"] == "final" and events[-1]["response"]["finish"] == "error"
    assert events[-1]["response"]["error"]["code"] == "E0318"
    assert len(terminals(events)) == 1
    assert s.ended and s.error["code"] == "E0318"
    assert s.final["finish"] == "error"


def test_raise_on_error_true_raises_the_typed_error_with_the_events(speech_rt, pcm):
    s = speech_rt.listen({**OPEN}, raise_on_error=True)
    s.push(pcm[:3201])
    with pytest.raises(latai.AudioInvalid) as ei:
        s.close()
    assert ei.value.code == "E0318"
    assert ei.value.events and ei.value.events[-1]["event"] == "final"


def test_a_push_above_1_mib_is_split_not_refused(speech_rt):
    s = speech_rt.listen({**OPEN})
    events = s.push(b"\x00" * (1024 * 1024 + 2))  # silence: the gate skips it, no event
    assert events == []
    events = s.close(flush=False)
    assert events[-1]["response"]["latai"]["stream"]["bytes"] == 1024 * 1024 + 2


def test_two_streams_at_once(speech_rt, pcm):
    a = speech_rt.listen({**OPEN})
    b = speech_rt.listen({**OPEN})
    ea, eb = [], []
    for i in range(0, 64000, 3200):
        ea += a.push(pcm[i : i + 3200])
        eb += b.push(pcm[i : i + 3200])
    ea += a.close()
    eb += b.close()
    assert stripped(ea) == stripped(eb)
