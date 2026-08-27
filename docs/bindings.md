# Python bindings

`bindings/` is a [nanobind](https://github.com/wjakob/nanobind) wrapper that exposes chimera's header-only OOP layer ([`chimera.hpp`](../src/chimera/chimera.hpp)) as a Python module named `chimera`. Load a model once, call it many times:

```python
import chimera

llm = chimera.Llama("models/Qwen3-1.7B-Q4_0.gguf")
llm.options.n_predict = 128
print(llm.generate("What is the capital of France?"))
```

This binds **chimera's OOP handles** — one static binary, shared ggml, the persistent-handle semantics — which is distinct from the sibling [cyllama](https://github.com/shakfu/cyllama) project that binds upstream llama.cpp / whisper.cpp / sd.cpp directly via Cython. The design rationale for the wrapped classes lives in [`docs/dev/oop-layer.md`](dev/oop-layer.md); this document is the build-and-use guide for the Python layer.

## Prerequisites

The module links chimera's three prebuilt static archives (it does **not** rebuild the C++ side), so build them first:

```bash
make build                              # produces libchimera*.a + runs combine_archives.py
```

This is a **local / dev install**: the archives in `build/` are not bundled, so `uv build` / `pip wheel` to a redistributable wheel won't work without bundling them. Install from the repo checkout.

## Build & install

### make (recommended — no Python setup needed)

```bash
make bindings        # builds bindings/build/chimera.*.so
make test-bindings   # builds + runs the smoke test
```

When `uv` is available, these provision nanobind + scikit-build-core into a local venv (`bindings/.venv`) pinned to the project Python (`$(PYTHON)`), so the module's ABI matches the interpreter you actually use. No system-Python changes. Override the interpreter with `make PYTHON=python3.13 bindings`, or point at your own nanobind-equipped Python with `make BINDINGS_PY=/path/to/python bindings`. Import the result by putting `bindings/build` on `PYTHONPATH`.

### uv (installs into an environment)

```bash
# one-shot build + install (build isolation auto-installs the toolchain):
uv pip install ./bindings

# editable dev install -- recompiles on import after you edit chimera_ext.cpp:
uv sync --group build
uv pip install --no-build-isolation -e ./bindings

# run the smoke test in the env:
CHIMERA_SMOKE_MODEL=models/Llama-3.2-1B-Instruct-Q8_0.gguf \
    uv run python bindings/smoke_test.py
```

Plain `pip` works identically (`pip install ./bindings`).

### Configuration

Pass CMake defines through the installer (uv/pip) with `-C cmake.define.<X>=<v>`:

```bash
uv pip install ./bindings -C cmake.define.CHIMERA_BUILD_ROOT=/abs/path/to/build
uv pip install ./bindings -C cmake.define.CHIMERA_HAS_SD=OFF
```

`CHIMERA_HAS_SD` / `CHIMERA_HAS_WHISPER` must match the modalities the archives were built with; the SD / Whisper classes are compiled in only when their modality is present.

## Usage

```python
import chimera, sys

# Text generation (persistent handle: model loads once, reused across calls).
llm = chimera.Llama("models/Qwen3-1.7B-Q4_0.gguf")
llm.options.n_predict = 128          # full option coverage; mutate live
print(llm.generate("Hello"))

# Streaming: per-token callback (the GIL is re-acquired for each call).
llm.generate("Tell me a story.", on_token=lambda piece: sys.stdout.write(piece))

# Tokenizer / embeddings.
tok = chimera.Tokenizer("models/Qwen3-1.7B-Q4_0.gguf")
ids = tok.encode("hello"); text = tok.decode(ids)

emb = chimera.Embedder("models/bge-small-en-v1.5-q8_0.gguf")
vec = emb.embed("hello world")        # list[float]; np.asarray(vec) for numpy

# OpenAI-compatible server (blocks until SIGINT / programmatic stop).
srv = chimera.Server(chimera.ServeOptions())
srv.options.model = "models/Qwen3-1.7B-Q4_0.gguf"
srv.options.port = 8080
srv.run()

# Image generation (when built with SD).
sd = chimera.SD("models/sd-model.gguf")
sd.options.steps = 20
imgs = sd.generate("a red fox in snow")   # list[PixelImage]; .width/.height/.channels/.pixels

# Transcription (when built with Whisper).
w = chimera.Whisper("models/ggml-base.en.bin")
res = w.transcribe("audio.wav")            # .text, .segments, .detected_language
```

## API notes

- **Errors are exceptions.** `chimera::fail()` (C++) surfaces as `chimera.ChimeraError` carrying the message; the interpreter never sees a process exit. `chimera.ExitCode` is also exposed.

- **Options are strings, not enums.** Every wrapper exposes its full option struct (`Llama.options`, `Embedder`, `Server`, `SD`, `Whisper` — all fields bound). Choice fields (`sample_method`, `scheduler`, `rng`, `prediction`, `pooling`, `rope_scaling`, `split_mode`, ...) are plain strings that the engine converts internally — so `ExitCode` is the only enum on the surface and you set e.g. `sd.options.sample_method = "euler_a"`.

- **GIL and thread safety.** `generate()` / `embed()` / `run()` / `transcribe()` release the GIL for the compute. Because `generate()` mutates one shared `llama_context`, **two Python threads calling `generate()` on the same object is a data race** — use one object per thread or a lock. Distinct objects are independent.

- **Return types.** `Embedder.embed` returns `list[float]`; `SD.generate` returns `PixelImage` objects (`.pixels` is `bytes` of size `width*height*channels`); `Whisper.transcribe` returns a `TranscribeResult` with `Segment`s. (A zero-copy numpy variant via `nb::ndarray` is a possible enhancement — see below.)

## Limitations / possible enhancements

- Local/dev install only (archives are not bundled into a portable wheel).

- `command_chat` (the interactive REPL) is not wrapped, matching `chimera.hpp`.

- `Embedder.embed` / `PixelImage.pixels` could expose `nb::ndarray` for zero-copy numpy instead of `list` / `bytes`.

- Typed Python `enum.Enum`s mapping to the canonical choice strings could be layered on for discoverability (not required — the fields take strings).

## See also

- [`bindings/README.md`](../bindings/README.md) — the package README (same material, lives next to the code and is the `pyproject.toml` readme).

- [`docs/dev/oop-layer.md`](dev/oop-layer.md) — `chimera.hpp` design the bindings wrap (persistent-handle semantics, streaming hook, upstream-drift guards).

- [`docs/dev/combine_archives.md`](dev/combine_archives.md) — the three-archive link contract the bindings rely on.
