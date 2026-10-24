# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Closing the runtime under live sessions. Own module: it closes the runtime."""

import pytest

import latai
from conftest import SPEECH
from test_listen import OPEN, pcm  # noqa: F401


def test_runtime_close_ends_live_sessions(speech_rt, pcm):
    s = speech_rt.listen({**OPEN})
    s.push(pcm[:32000])
    speech_rt.close()
    assert speech_rt.closed
    assert s.ended is True and s.closed is True
    with pytest.raises(latai.LataiError):
        s.push(pcm[:3200])
    with pytest.raises(latai.LataiError):
        speech_rt.listen({**OPEN})
