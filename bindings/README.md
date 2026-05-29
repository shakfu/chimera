# chimera Python bindings (nanobind)

A [nanobind](https://github.com/wjakob/nanobind) wrapper over the header-only
OOP layer in [`src/chimera/chimera.hpp`](../src/chimera/chimera.hpp). It exposes
the persistent-handle classes (`Llama`, `Embedder`, `Tokenizer`, `Server`, and,
when the underlying archive was built with them, `SD` / `Whisper`) to Python.

This binds **chimera's OOP handles** (one binary, shared ggml, persistent-handle
semantics), which is distinct from the sibling
[cyllama](https://github.com/shakfu/cyllama) project that binds upstream
llama.cpp / whisper.cpp / sd.cpp directly via Cython. Decide whether the two
should converge before investing heavily here.

## Prerequisites

The bindings link chimera's three prebuilt static archives, so build them first:

```bash
make build                              # produces libchimera*.a in build/
python3 scripts/combine_archives.py     # (run by `make build`; re-run if needed)
pip install nanobind scikit-build-core
```

## Build

> This is a **local/host build**: the extension statically links the prebuilt
> archives in `<repo>/build`, so build them first (`make build &&
> python3 scripts/combine_archives.py`). The archives are *not* bundled into
> the sdist, but they *are* linked into the `.so`, so the resulting wheel is
> self-contained at runtime -- it just inherits the archives' host-tuned ggml
> (`GGML_NATIVE`, single backend) and so runs only on this machine (or a
> bit-identical one). Cross-machine / PyPI portability is a non-goal of the
> archive layer (see `docs/dev/combine_archives.md` S1); the sibling
> [inferna](https://github.com/shakfu/inferna) project owns the portable-wheel
> story.
>
> To produce a `.whl`: `make wheel` (writes `bindings/dist/`). It runs
> `uv build --wheel`, which builds straight from the source tree. A *plain*
> `uv build` does **not** work: it builds the wheel from an extracted sdist
> where the archives + repo headers (both outside `bindings/`) are absent.

### uv (recommended)

```bash
# one-shot build + install into the active env (build isolation auto-installs
# scikit-build-core + nanobind):
uv pip install ./bindings

# editable dev install -- recompiles on import after you edit chimera_ext.cpp:
uv sync --group build
uv pip install --no-build-isolation -e ./bindings

# run the smoke test in the env:
CHIMERA_SMOKE_MODEL=models/Llama-3.2-1B-Instruct-Q8_0.gguf \
    uv run python bindings/smoke_test.py

# archives elsewhere? / drop a modality:
uv pip install ./bindings -C cmake.define.CHIMERA_BUILD_ROOT=/abs/path/to/build
uv pip install ./bindings -C cmake.define.CHIMERA_HAS_SD=OFF
```

Plain `pip` works identically (`pip install ./bindings -C cmake.define...`).

### Standalone CMake (no Python packaging; what `make bindings` runs)

```bash
make bindings                      # cmake -S bindings -B bindings/build + build
PYTHONPATH=bindings/build python3 bindings/smoke_test.py
# or directly:
cmake -S bindings -B bindings/build -DCHIMERA_BUILD_ROOT=$PWD/build
cmake --build bindings/build
```

## Usage

```python
import chimera

llm = chimera.Llama("models/Qwen3-1.7B-Q4_0.gguf")
llm.options.n_predict = 128
print(llm.generate("What is the capital of France?"))

# streaming
import sys
llm.generate("Tell me a story.", on_token=lambda piece: sys.stdout.write(piece))

emb = chimera.Embedder("models/bge-small-en-v1.5-q8_0.gguf")
vec = emb.embed("hello world")          # list[float]; np.asarray(vec) for numpy

srv = chimera.Server(chimera.ServeOptions())
srv.options.model = "models/Qwen3-1.7B-Q4_0.gguf"
srv.options.port = 8080
srv.run()                                # blocks until SIGINT
```

## Smoke test

`smoke_test.py` mirrors `tests/external/hpp_smoke.cpp`: an import/construction
probe (no model needed) plus an optional inference probe gated on
`CHIMERA_SMOKE_MODEL`.

## pytest suite

`tests/` holds a pytest suite that supersedes the standalone smoke test for
finer-grained coverage: `test_module.py` (no model needed -- import, class
export, option round-trips, exception translation) and `test_inference.py`
(model-gated end-to-end tests for tokenize, generate + streaming callback,
embeddings, and the optional SD / Whisper modalities).

```bash
make test-bindings-pytest                 # build the module + run the suite
make test-bindings-pytest PYTEST_ARGS="-k tokenizer -v"

# or directly against a built module:
PYTHONPATH=bindings/build pytest bindings/tests
```

Model-dependent tests SKIP (never FAIL) when their model is absent. Each
resolves a path from a `CHIMERA_TEST_*` env override, else the repo's
`models/` directory; `conftest.py` documents the variables
(`CHIMERA_TEST_LLAMA_MODEL`, `CHIMERA_TEST_EMBED_MODEL`, `CHIMERA_TEST_SD_MODEL`,
`CHIMERA_TEST_WHISPER_MODEL`, `CHIMERA_TEST_WHISPER_WAV`). `pytest` is the `dev`
dependency group in `pyproject.toml` (`uv sync --group dev`).

## Known limits / TODO (this is a scaffold)

- **Option coverage is complete** for `LlamaOptions`, `EmbedOptions`,
  `ServeOptions`, `SdOptions`, and `WhisperOptions` (every field is bound;
  verified by an auto-generated member-pointer compile check against
  `chimera.hpp`). `TokenizeOptions` is intentionally not bound -- the
  `Tokenizer` class takes a path + `use_mmap` directly.
- **No enum bindings are needed beyond `ExitCode`.** chimera's option structs
  use `std::string` (not C++ enums) for every user-facing choice field --
  `sample_method`, `scheduler`, `rng`, `prediction`, `lora_apply_mode`,
  `wtype`, `pooling`, `rope_scaling`, `split_mode`, etc. The CLI/engine
  converts those strings to the underlying engine enums internally
  (`str_to_sample_method`, ...). Pass the string value (e.g.
  `sd.options.sample_method = "euler_a"`). Valid values are noted in inline
  comments in `chimera_ext.cpp`; an optional typed-enum convenience layer
  (Python `enum.Enum` mapping to the canonical strings) could be added on top
  if discoverability matters, but it is not required for correctness.
- **Thread safety.** `generate()` releases the GIL and mutates one shared
  `llama_context`. Two Python threads calling `generate()` on the *same* object
  is a data race -- use one object per thread or add a lock. Distinct objects
  are independent.
- **Return types.** `Embedder.embed` returns `list[float]`; switch to
  `nb::ndarray<float>` (with a capsule owner) for zero-copy numpy.
  `PixelImage.pixels` is `bytes`; an `nb::ndarray<uint8_t>` shaped `(h, w, c)`
  is friendlier for PIL/numpy.
- **`command_chat`** (interactive REPL) is not wrapped, matching `chimera.hpp`.
- **Version** in `pyproject.toml` is pinned manually; keep it in sync with
  `CHIMERA_VERSION` in `scripts/manage.py`.
- **OpenSSL** is linked only if found, since the default archive build is
  OpenSSL-free (`CHIMERA_OPENSSL=OFF`). A `CHIMERA_OPENSSL=ON` archive needs it
  present at link time.
