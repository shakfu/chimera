# chimera

A single statically-linked C++ executable that bundles [llama.cpp](https://github.com/ggml-org/llama.cpp), [whisper.cpp](https://github.com/ggml-org/whisper.cpp), and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) into a busybox-style multitool. The same binary handles text generation, interactive chat, speech-to-text, and text-to-image, all sharing a single ggml backend set.

If you want the same capabilities from Python instead of a native binary, see [**cyllama**](https://github.com/shakfu/cyllama) — chimera's sibling project, which exposes llama.cpp / whisper.cpp / stable-diffusion.cpp as Cython bindings with a high-level Python API.

## Subcommands

| Command    | Purpose                                                                |
|------------|------------------------------------------------------------------------|
| `gen`      | One-shot llama text generation (text + optional images via `--mmproj`) |
| `chat`     | Interactive chat with persistent KV cache across turns                 |
| `tokenize` | Print token ids (or `id<TAB>piece`) for a prompt                       |
| `embed`    | Emit a single pooled embedding vector for a prompt                     |
| `whisper`  | Transcribe a WAV file (streaming, segment-by-segment)                  |
| `sd`       | Text-to-image / img2img / inpaint with stable-diffusion.cpp            |

A top-level `-v,--verbose` flag re-enables native backend logging (silenced by
default).

## Build

```bash
make build
```

This will:

1. Run `python scripts/manage.py build --all --deps-only --sd-shared-ggml`,
   which clones and builds llama.cpp, whisper.cpp, and stable-diffusion.cpp
   into `thirdparty/<project>/{include,lib}`. The `--sd-shared-ggml` flag is
   load-bearing: stable-diffusion.cpp normally vendors its own copy of ggml,
   which would collide with llama.cpp's at link time. Building all three
   projects against the single ggml set is what makes the static binary
   possible.
2. Configure with `cmake -S . -B build -DSD_USE_VENDORED_GGML=OFF`.
3. Build the `chimera` target.

Output: `build/chimera`.

Run `make deps` alone if you just want to (re)build the third-party libs, or
`make rebuild` after touching only chimera source.

## Install

```bash
make install                       # /usr/local/bin/chimera (may need sudo)
make install PREFIX=$HOME/.local   # ~/.local/bin/chimera
make install DESTDIR=/tmp/stage PREFIX=/usr   # for packaging
```

`make uninstall` removes the binary from the same `$PREFIX/bin/`.

## Test

```bash
make smoke    # CLI plumbing only -- no model files needed
make test     # smoke + end-to-end runs gated on models/ presence
```

`scripts/test.sh` skips end-to-end checks when the matching model file is
absent (see the script for the lookup paths), so a fresh clone reports
SKIP rather than FAIL.

### Backends

On macOS, Metal is enabled by default. Other backends are opt-in via
environment variables when invoking `make deps`:

```bash
GGML_CUDA=1   make deps    # NVIDIA CUDA
GGML_VULKAN=1 make deps    # Vulkan (cross-platform)
GGML_HIP=1    make deps    # AMD HIP/ROCm
```

Then re-run `cmake` with the matching `-DGGML_<BACKEND>=ON` to pick up the
right libraries.

## Usage

```bash
./build/chimera gen -m models/Qwen3-4B-Q8_0.gguf -p "Why did..."
./build/chimera chat -m models/Qwen3-4B-Q8_0.gguf
./build/chimera tokenize -m models/Qwen3-4B-Q8_0.gguf -p "hello world" --pieces
./build/chimera embed -m models/bge-small-en-v1.5-q8_0.gguf -p "a quick brown fox"
./build/chimera whisper -m models/ggml-base.en.bin -i audio.wav
./build/chimera sd -m models/sd-v1-5.gguf -p "a cat" -o out.png
```

`gen` and `tokenize`/`embed` accept `-f <file>` instead of `-p` (use `-f -`
for stdin); `chat` accepts `--system-prompt-file <file>`.

Use `--help` on any subcommand to see its options.

### Vision input (`gen --mmproj --image`)

`gen` accepts one or more `--image` paths when paired with a vision
projector (`--mmproj`). The image is encoded by the projector, threaded
into the prompt at the default media marker, and the resulting chunks
are evaluated into the llama context via `mtmd_helper_eval_chunks`.

```bash
./build/chimera gen \
  -m models/gemma-4-E4B-it-Q4_K_M.gguf \
  --mmproj models/mmproj-gemma-4-E4B-it-BF16.gguf \
  --image photo.png \
  -p "Describe this image in one sentence." -n 64
```

Notes:

- The prompt is auto-wrapped in the model's chat template (VL models
  are almost always instruct-tuned and otherwise stall on turn 0).
- If your prompt does not already contain the media marker
  (`<__media__>` by default), one is prepended per `--image` so images
  appear before the text. To interleave, place the marker yourself.
- `--image` may be repeated; each gets its own marker.
- The vision encoder runs on the default backend (Metal on macOS),
  independent of `--gpu-layers` (which only controls LLM offload).
- Vision input is supported on `gen` only; multi-turn vision in `chat`
  is not yet wired up (see `TODO.md`).

### Image-to-image / inpainting (`sd --init-image`)

`sd` accepts an `--init-image` for img2img and an optional `--mask-image`
for inpainting. Both must match `-W,-H` (the SD pipeline does not
resize internally).

```bash
# img2img: re-render an input image guided by a new prompt
./build/chimera sd -m models/sd-v1-5.gguf \
  --init-image input.png --strength 0.6 \
  -p "the same scene but at night" -W 512 -H 512 -s 20 -o out.png

# inpaint: only repaint regions where the mask is non-zero
./build/chimera sd -m models/sd-v1-5.gguf \
  --init-image input.png --mask-image mask.png \
  -p "a hat on the person's head" -W 512 -H 512 -s 20 -o out.png
```

`--strength` ranges 0..1 (0 preserves the init image, 1 = full noise =
text-to-image). The SD context is automatically built with
`vae_decode_only=false` whenever `--init-image` is supplied.

### Line editing in `chat` (linenoise)

If [linenoise](https://github.com/shakfu/linenoise) is present under
`thirdparty/`, interactive `chat` sessions get readline-style line
editing, history (↑/↓, `Ctrl-R`), and basic editing keys. History
persists at `$CHIMERA_HISTORY` (override) or
`$HOME/.chimera_chat_history`. The integration is opt-out:

```bash
# probe automatically (default; links if liblinenoise.a is present)
cmake -S . -B build -DCHIMERA_LINENOISE=AUTO

# require linenoise (configure fails if missing)
cmake -S . -B build -DCHIMERA_LINENOISE=ON

# skip linenoise entirely (chat falls back to plain getline)
cmake -S . -B build -DCHIMERA_LINENOISE=OFF
```

Build the lib with `python scripts/manage.py build -L`. Piped /
redirected stdin always falls back to `getline`, so scripts and the
test suite are unaffected by this option.

### Exit codes

| Code  | Meaning                                                           |
|-------|-------------------------------------------------------------------|
| 0     | success                                                           |
| 1     | generic runtime error                                             |
| 2     | bad input (missing / invalid file or argument)                    |
| 3     | model-load failure (model not found, mmproj incompatible, etc.)   |
| 4     | generation / inference failure                                    |
| >= 100| CLI11 parse error (forwarded from CLI11's own exit codes)         |

## Source layout

```
CMakeLists.txt                  Top-level: defines LIB_*, SYSTEM_LIBS, etc.,
                                then add_subdirectory(src/chimera).
Makefile                        deps + cmake wrapper.
scripts/manage.py               Third-party fetch/build driver.
thirdparty/CLI11.hpp            Single-header CLI parser.
src/chimera/
  chimera.h                     CLI option structs + cross-TU declarations
  chimera.cpp                   main(), llama gen/chat, log silencing entry
  chimera_whisper.cpp           whisper transcription + whisper_log_set
  chimera_sd.cpp                stable-diffusion text-to-image + sd_set_log_callback
  llama_build_info_shim.cpp     stubs the symbols libllama-common.a expects
  stb_impl.cpp                  stb_image_write implementation
  CMakeLists.txt                chimera target (consumes parent-scope vars)
```

### Why three translation units

`llama.cpp/include/ggml.h` and `whisper.cpp/include/ggml.h` ship slightly
different versions of the same header and define overlapping enums (e.g.
`ggml_scale_flag`, `ggml_sort_order`). Including both in one TU is a hard
compile error. The split is:

- `chimera.cpp` includes `llama.h` / `ggml.h` (llama.cpp's copy).
- `chimera_whisper.cpp` includes only `whisper.h` (which pulls whisper.cpp's
  ggml.h).
- `chimera_sd.cpp` includes only `stable-diffusion.h`.

`chimera.h` declares forward-declared C++ types only — no native headers — so
all three TUs can include it freely.

### Log silencing

`main()` calls `silence_all_logging()` *before* `app.parse(argc, argv)`,
installing no-op callbacks for `llama_log_set`, `ggml_log_set`,
`whisper_log_set` (via `chimera_silence_whisper_log()` in the whisper TU), and
`sd_set_log_callback` (via `chimera_silence_sd_log()` in the SD TU). The
verbose flag re-installs `nullptr` callbacks (the upstream defaults) once
parsing is complete.

The whisper/sd silencers cannot live in `chimera.cpp` because their headers
would re-introduce the ggml collision above.

### Late backend init

`llama_backend_init()` is called **after** `app.parse()` returns, so `--help`
and parse errors do not trigger `ggml_load_backends`. `llama_backend_free()`
runs on every exit path (success, `CLI::ParseError`, `std::exception`).

## Origin

chimera was originally developed inside the [cyllama](https://github.com/shakfu/cyllama) Python project, sharing its
`scripts/manage.py` and `thirdparty/` build infrastructure. It was extracted
into this repository so that it could be developed independently with its own release cadence.

## License

MIT. The vendored third-party libraries carry their own licenses (all MIT or
permissive equivalents) — see their respective `thirdparty/<project>/LICENSE`
files after running `make deps`.
