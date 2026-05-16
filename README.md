# chimera

A single statically-linked C++ executable that bundles [llama.cpp](https://github.com/ggml-org/llama.cpp), [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), [SQLite](https://sqlite.org), and [sqlite-vec](https://github.com/asg017/sqlite-vec) into a busybox-style multitool. The same binary handles text generation, interactive chat with persistent history, speech-to-text, text-to-image, a personal RAG / vector store, and an OpenAI-compatible HTTP server — all sharing a single ggml backend set and one SQLite database.

If you want the same capabilities from Python instead of a native binary, see [**cyllama**](https://github.com/shakfu/cyllama) — chimera's sibling project, which exposes llama.cpp / whisper.cpp / stable-diffusion.cpp as Cython bindings with a high-level Python API.

## Subcommands

| Command    | Purpose                                                                |
|------------|------------------------------------------------------------------------|
| `gen`      | One-shot llama text generation (text + optional images via `--mmproj`) |
| `chat`     | Interactive chat with persistent KV cache across turns; optional save-to-DB |
| `tokenize` | Print token ids (or `id<TAB>piece`) for a prompt                       |
| `embed`    | Emit a single pooled embedding vector for a prompt                     |
| `whisper`  | Transcribe a WAV file (streaming, segment-by-segment)                  |
| `sd`       | Text-to-image / img2img / inpaint with stable-diffusion.cpp            |
| `serve`    | OpenAI-compatible HTTP server (text + audio + image + RAG)             |
| `index`    | Vector-store collections (create / ingest / list / stats / drop)       |
| `search`   | KNN search over a vector-store collection                              |
| `db`       | Embedded SQLite management (status; future: backup / vacuum)           |
| `info`     | Print versions + ggml backends/devices + CPU features (useful for bug reports) |

A top-level `-v,--verbose` flag re-enables native backend logging (silenced by default).

See [`doc/cheatsheet.md`](doc/cheatsheet.md) for a one-page command reference, and [`doc/serve.md`](doc/serve.md) for the HTTP server.

## Build

```bash
make build
```

This will:

1. Run `python scripts/manage.py build --all --deps-only --sd-shared-ggml`, which clones and builds llama.cpp, whisper.cpp, and stable-diffusion.cpp into `thirdparty/<project>/{include,lib}` and vendors the SQLite + sqlite-vec amalgamations into `thirdparty/{sqlite,sqlite-vec}/`. The `--sd-shared-ggml` flag is load-bearing: stable-diffusion.cpp normally vendors its own copy of ggml, which would collide with llama.cpp's at link time. Building all three projects against the single ggml set is what makes the static binary possible.
2. Configure with `cmake -S . -B build -DSD_USE_VENDORED_GGML=OFF`.
3. Build the `chimera` target.

Output: `build/chimera`.

Run `make deps` alone if you just want to (re)build the third-party libs, or `make rebuild` after touching only chimera source.

### System dependencies

OpenSSL is required at link time (cpp-httplib uses it for TLS support inside the bundled HTTP server). On macOS this also pulls in the system Security and CoreFoundation frameworks. Install OpenSSL via your package manager (`brew install openssl@3` on macOS; `apt install libssl-dev` on Debian/Ubuntu) before running `make build`.

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

`scripts/test.sh` skips end-to-end checks when the matching model file is absent (see the script for the lookup paths), so a fresh clone reports SKIP rather than FAIL.

### Backends

On macOS, Metal is enabled by default. Other backends are opt-in via environment variables when invoking `make deps`:

```bash
GGML_CUDA=1   make deps    # NVIDIA CUDA
GGML_VULKAN=1 make deps    # Vulkan (cross-platform)
GGML_HIP=1    make deps    # AMD HIP/ROCm
```

Then re-run `cmake` with the matching `-DGGML_<BACKEND>=ON` to pick up the right libraries.

## Usage

```bash
./build/chimera gen -m models/Qwen3-4B-Q8_0.gguf -p "Why did..."
./build/chimera chat -m models/Qwen3-4B-Q8_0.gguf
./build/chimera tokenize -m models/Qwen3-4B-Q8_0.gguf -p "hello world" --pieces
./build/chimera embed -m models/bge-small-en-v1.5-q8_0.gguf -p "a quick brown fox"
./build/chimera whisper -m models/ggml-base.en.bin -i audio.wav
./build/chimera sd -m models/sd-v1-5.gguf -p "a cat" -o out.png
./build/chimera serve -m models/Qwen3-4B-Q8_0.gguf            # OpenAI-compatible HTTP server
./build/chimera index create -n notes -e models/bge-small-en-v1.5-q8_0.gguf
./build/chimera search -n notes -q "how does X work" -k 5
./build/chimera db status
```

`gen` and `tokenize`/`embed` accept `-f <file>` instead of `-p` (use `-f -` for stdin); `chat` accepts `--system-prompt-file <file>`.

Use `--help` on any subcommand to see its options.

### Server (`serve`)

`chimera serve` exposes an OpenAI-compatible HTTP API. With no extra flags it serves the text-LLM endpoints; opt-in flags enable additional surfaces that load the corresponding model alongside the LLM in the same process:

```bash
chimera serve -m model.gguf                                  # text-only
chimera serve -m model.gguf --embeddings                     # +/v1/embeddings
chimera serve -m model.gguf --enable-audio whisper.gguf      # +/v1/audio/transcriptions
chimera serve -m model.gguf --enable-image sd.gguf           # +/v1/images/*
chimera serve -m model.gguf --enable-rag    embed.gguf       # +/v1/vector_stores/*
chimera serve -m model.gguf --persist-chats                  # save every chat to DB
```

Point any OpenAI client at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-used")
```

Supported endpoints, by default: `/v1/chat/completions`, `/v1/completions`, `/v1/messages` + `/v1/messages/count_tokens` (Anthropic compat), `/v1/responses`, `/v1/models`, `/v1/embeddings`, `/infill`, `/tokenize`, `/detokenize`, `/apply-template`, `/health`, `/metrics`, `/props`. Opt-in endpoints add `/v1/audio/transcriptions`, `/v1/images/{generations,edits,variations}`, and `/v1/vector_stores/*`. See [`doc/serve.md`](doc/serve.md) for the full surface and [`doc/dev/server.md`](doc/dev/server.md) for the implementation notes (what's bound, what's deliberately not, why).

### Vector store / RAG (`index`, `search`)

A personal RAG index lives entirely in the local SQLite file (vendored SQLite + sqlite-vec). No external service required.

```bash
chimera index create -n notes -e models/bge-small-en-v1.5-q8_0.gguf
chimera index ingest -n notes -f path/to/doc.md
chimera index ingest -n notes -g 'docs/**/*.md'
chimera search       -n notes -q "how does X work?" -k 5
```

The same index is queryable over HTTP when `chimera serve --enable-rag` is running (`POST /v1/vector_stores/:name/search`). See [`doc/dev/sqlite.md`](doc/dev/sqlite.md) for the schema, migration model, and the phased plan.

### Persistent chat history

`chimera chat --persist` saves every turn to the SQLite DB; later you can list, search, or resume past chats from any chimera invocation:

```bash
chimera chat -m model.gguf --persist            # save turns as they happen
chimera chat --resume 42                        # resume saved chat #42
chimera chat --resume last                      # most recent
chimera chat --list                             # list saved chats (no model load)
chimera chat --search "secret password"         # FTS5 over messages (no model load)
```

The DB location defaults to `$CHIMERA_DB` then to the platform XDG path (`~/Library/Application Support/chimera/chimera.db` on macOS, `~/.local/share/chimera/chimera.db` on Linux, `%LOCALAPPDATA%\chimera\` on Windows). Override with `--db <path>` on any chat / index / search / db subcommand.

### Vision input (`gen --mmproj --image`)

`gen` accepts one or more `--image` paths when paired with a vision projector (`--mmproj`). The image is encoded by the projector, threaded into the prompt at the default media marker, and the resulting chunks are evaluated into the llama context via `mtmd_helper_eval_chunks`.

```bash
./build/chimera gen \
  -m models/gemma-4-E4B-it-Q4_K_M.gguf \
  --mmproj models/mmproj-gemma-4-E4B-it-BF16.gguf \
  --image photo.png \
  -p "Describe this image in one sentence." -n 64
```

Notes:

- The prompt is auto-wrapped in the model's chat template (VL models are almost always instruct-tuned and otherwise stall on turn 0).
- If your prompt does not already contain the media marker (`<__media__>` by default), one is prepended per `--image` so images appear before the text. To interleave, place the marker yourself.
- `--image` may be repeated; each gets its own marker.
- The vision encoder runs on the default backend (Metal on macOS), independent of `--gpu-layers` (which only controls LLM offload).
- Vision input is supported on `gen` only; multi-turn vision in `chat` is not yet wired up (see `TODO.md`).

### Image-to-image / inpainting (`sd --init-image`)

`sd` accepts an `--init-image` for img2img and an optional `--mask-image` for inpainting. Both must match `-W,-H` (the SD pipeline does not resize internally).

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

`--strength` ranges 0..1 (0 preserves the init image, 1 = full noise = text-to-image). The SD context is automatically built with `vae_decode_only=false` whenever `--init-image` is supplied.

### Line editing in `chat` (linenoise)

If [linenoise](https://github.com/shakfu/linenoise) is present under `thirdparty/`, interactive `chat` sessions get readline-style line editing, history (↑/↓, `Ctrl-R`), and basic editing keys. History persists at `$CHIMERA_HISTORY` (override) or `$HOME/.chimera_chat_history`. The integration is opt-out:

```bash
# probe automatically (default; links if liblinenoise.a is present)
cmake -S . -B build -DCHIMERA_LINENOISE=AUTO

# require linenoise (configure fails if missing)
cmake -S . -B build -DCHIMERA_LINENOISE=ON

# skip linenoise entirely (chat falls back to plain getline)
cmake -S . -B build -DCHIMERA_LINENOISE=OFF
```

Build the lib with `python scripts/manage.py build -L`. Piped / redirected stdin always falls back to `getline`, so scripts and the test suite are unaffected by this option.

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
thirdparty/
  CLI11.hpp                     Single-header CLI parser.
  rang.hpp                      Single-header ANSI color.
  llama.cpp/, whisper.cpp/,
  stable-diffusion.cpp/,
  linenoise/, sqlite/,
  sqlite-vec/                   Populated by scripts/manage.py.
src/chimera/
  chimera.h                     CLI option structs + cross-TU declarations
  chimera.cpp                   main(), llama gen, chat REPL, log silencing
  chimera_embed.{h,cpp}         Embedder helper (CLI embed + RAG ingest)
  chimera_whisper.{h,cpp}       whisper transcription + ASR HTTP handler
  chimera_sd.{h,cpp}            stable-diffusion + image HTTP handlers
  chimera_serve.cpp             OpenAI-compatible HTTP server (LLM, audio,
                                image, RAG, chat persistence)
  chimera_db.{h,cpp}            SQLite connection, XDG paths, migrations
  chimera_chat_store.{h,cpp}    CRUD over chats + messages + messages_fts
  chimera_vector_store.{h,cpp}  CRUD over collections + documents + vec0
  llama_build_info_shim.cpp     stubs the symbols libllama-common.a expects
  stb_impl.cpp                  stb_image_write implementation
  CMakeLists.txt                chimera target (consumes parent-scope vars)
doc/
  serve.md, cheatsheet.md       User-facing prose + one-page reference.
  dev/server.md, dev/sqlite.md  Internal notes: what's bound, schema model,
                                phased plans, things-to-watch-for.
```

### Why so many translation units

`llama.cpp/include/ggml.h` and `whisper.cpp/include/ggml.h` ship slightly different versions of the same header and define overlapping enums (e.g. `ggml_scale_flag`, `ggml_sort_order`). Including both in one TU is a hard compile error. The split:

- `chimera.cpp` (+ `chimera_embed.cpp`, `chimera_chat_store.cpp`, `chimera_vector_store.cpp`, `chimera_db.cpp`, `chimera_serve.cpp`) include `llama.h` / `ggml.h` (llama.cpp's copy) — the LLM, embedding, and HTTP-server side.
- `chimera_whisper.cpp` includes only `whisper.h` (which pulls whisper.cpp's ggml.h).
- `chimera_sd.cpp` includes only `stable-diffusion.h`.

`chimera.h` and the per-modality headers (`chimera_whisper.h`, `chimera_sd.h`, `chimera_db.h`, etc.) declare forward-declared C++ types only — no native headers — so all TUs can include them freely.

The vendored `sqlite3.c` and `sqlite-vec.c` are pulled into the chimera target directly (single-TU amalgamations), and `server-http.cpp` from llama.cpp's `tools/server/` is shipped under `thirdparty/llama.cpp/src-aux/` and compiled into chimera the same way (upstream doesn't expose it as a library).

### Log silencing

`main()` calls `silence_all_logging()` *before* `app.parse(argc, argv)`, installing no-op callbacks for `llama_log_set`, `ggml_log_set`, `whisper_log_set` (via `chimera_silence_whisper_log()` in the whisper TU), and `sd_set_log_callback` (via `chimera_silence_sd_log()` in the SD TU). The verbose flag re-installs `nullptr` callbacks (the upstream defaults) once parsing is complete.

The whisper/sd silencers cannot live in `chimera.cpp` because their headers would re-introduce the ggml collision above.

### Late backend init

`llama_backend_init()` is called **after** `app.parse()` returns, so `--help` and parse errors do not trigger `ggml_load_backends`. `llama_backend_free()` runs on every exit path (success, `CLI::ParseError`, `std::exception`).

## Documentation

| File                                         | Audience      | Contents                                                                                  |
|----------------------------------------------|---------------|-------------------------------------------------------------------------------------------|
| [`doc/cheatsheet.md`](doc/cheatsheet.md)     | user          | One-page command + curl reference.                                                        |
| [`doc/serve.md`](doc/serve.md)               | user          | `chimera serve` walkthrough: flags, endpoints, errors, SDK setup.                         |
| [`doc/dev/server.md`](doc/dev/server.md)     | maintainer    | What's bound on the HTTP surface, what's deliberately not, threading model, gotchas.      |
| [`doc/dev/sqlite.md`](doc/dev/sqlite.md)     | maintainer    | Schema, migration discipline, phased RAG + chat-persistence plan, open design questions.  |
| [`CHANGELOG.md`](CHANGELOG.md)               | everyone      | Per-release feature notes.                                                                |

## Origin

chimera was originally developed inside the [cyllama](https://github.com/shakfu/cyllama) Python project, sharing its `scripts/manage.py` and `thirdparty/` build infrastructure. It was extracted into this repository so that it could be developed independently with its own release cadence.

## License

MIT. The vendored third-party libraries carry their own licenses (all MIT or permissive equivalents) — see their respective `thirdparty/<project>/LICENSE` files after running `make deps`.
