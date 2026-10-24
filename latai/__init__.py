# SPDX-License-Identifier: Apache-2.0 OR MIT
"""latai: Python binding of the C ABI of ``liblatai`` (``include/latai.h``).

Pure Python (``ctypes``), no compiled extension, no runtime dependencies. Find the library with
``LATAI_LIB`` (a file or a folder), ``LATAI_LIB_DIR``, or ``Runtime.open(lib=...)``; see
``library_candidates`` for the whole search. Quick tour::

    import latai
    with latai.Runtime.open() as rt:
        rt.install("x4r-l8.gguf")
        print(rt.run(latai.text_request("x4r-l8", "hello", max_tokens=8))["output"])
        for event in rt.run_stream(latai.text_request("x4r-l8", "hello")):
            if event["event"] == "delta":
                print(event["text"], end="")

Docs: ``docs/sdk.md`` and ``sdk/README.md`` in the repository.
"""

from __future__ import annotations

from . import types
from ._lib import (
    API_VERSION,
    Library,
    debug_counters,
    find_library,
    library_candidates,
    library_file_names,
    load_library,
)
from .audio import (
    AudioError,
    WavClip,
    audio_input,
    audio_request,
    pcm16_chunks,
    read_wav,
    text_request,
)
from .errors import *  # noqa: F401,F403  (LataiError, the category and per-code classes, error_for)
from .errors import (
    CATEGORY_CLASSES,
    ERROR_CLASSES,
    LataiError,
    LibraryNotFoundError,
    UnsupportedError,
    code_text,
    error_for,
)
from .listen import ListenSession
from .runtime import EventStream, Runtime

__version__ = "0.1.0"

__all__ = [
    "API_VERSION",
    "AudioError",
    "CATEGORY_CLASSES",
    "ERROR_CLASSES",
    "EventStream",
    "LataiError",
    "Library",
    "LibraryNotFoundError",
    "ListenSession",
    "Runtime",
    "UnsupportedError",
    "WavClip",
    "__version__",
    "audio_input",
    "audio_request",
    "code_text",
    "debug_counters",
    "error_for",
    "find_library",
    "library_candidates",
    "library_file_names",
    "load_library",
    "pcm16_chunks",
    "read_wav",
    "text_request",
    "types",
]
from .errors import __all__ as _error_names  # noqa: E402

__all__ += [n for n in _error_names if n not in __all__]
