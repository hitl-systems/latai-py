# latai (Python)

Python binding of the latai C ABI: run packs, stream tokens, stream audio, manage packs. Pure Python
(`ctypes`), Python 3.9 or later, no runtime dependencies, typed (`py.typed`). Guide:
[docs/sdk.md](../../docs/sdk.md).

```text
pip install ./sdk/python           # or: PYTHONPATH=sdk/python
```

`liblatai` is found through `LATAI_LIB` (a file or its folder), `LATAI_LIB_DIR`, `Runtime.open(lib=...)`,
the package folder, a source checkout's `target/debug`, the system folders, or the loader path.

```python
import latai

with latai.Runtime.open(data_dir=None) as rt:
    rt.install("x4r-l8.gguf")
    print(rt.run(latai.text_request("x4r-l8", "hello", max_tokens=8))["output"])
    for event in rt.run_stream(latai.text_request("x4r-l8", "hello")):
        ...
```

## Surface

| Name | What |
|---|---|
| `Runtime.open(data_dir=None, bundle_dir=None, *, lib=None)` | the runtime; a context manager; `close()` |
| `rt.run(request, *, request_id=None, raise_on_error=True)` | one response (`Response`) |
| `rt.run_stream(request, *, request_id=None, raise_on_error=False, queue_size=64)` | `EventStream`: iterator of events (errors are events); `close()`, `cancel()`, `final`, `code` |
| `rt.cancel(id)`, `rt.session_close(session)` | |
| `rt.listen(open, *, on_event=None, raise_on_error=False)` | `ListenSession`: `push(bytes)`, `close(flush=True)`, `cancel()`, `final`, `error`, `transcript` |
| `rt.listen_iter(open, chunks)` | generator of stream events |
| `rt.info()`, `status(name=None)`, `inspect(target)`, `stage(path)`, `pull(url, force=False)`, `install(source, activate=True)`, `activate(name, version=None)`, `rollback(name)`, `list_packs()`, `pin(name, pin=None)` | packs |
| `latai.text_request`, `audio_request`, `audio_input`, `read_wav`, `pcm16_chunks` | request and audio helpers |
| `LataiError` and a class per category / code (`RequestError`, `UnknownPack`, ...) | `.code .message .hint .details .retryable .response .events` |
| `library_candidates`, `find_library`, `Library`, `debug_counters` | the FFI layer |
| `latai.types` | generated `TypedDict`s of the wire types |

Regenerate the types with `python sdk/tools/gen_types.py` (`--check` for drift). Example:
[`examples/quickstart.py`](examples/quickstart.py).
