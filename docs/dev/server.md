# `chimera serve` — implementation notes

`chimera serve` is chimera's OpenAI-compatible HTTP server. It is not a
greenfield server; it is a thin chimera-side translation layer over
**llama.cpp's `server-context` STATIC library** (the engine that powers
`llama-server`), plus opt-in handlers that bind chimera's own whisper.cpp
and stable-diffusion.cpp wrappers to the OpenAI audio and image routes.

This document is the maintainer's view: what is wired up, what was
deliberately left out, the seams that bit during implementation, and the
work that is still on the table.

---

## 1. Architecture at a glance

```
                                  ┌───────────────────────────────────────┐
HTTP request                      │ command_serve() in chimera_serve.cpp  │
─────────────────────────────────▶│                                       │
                                  │   server_http_context (cpp-httplib)   │◀── /health, /v1/models
                                  │             │                         │   /v1/chat/completions
                                  │             │                         │   /v1/completions
                                  │             │                         │   /v1/embeddings
                                  │             ▼                         │
                                  │   server_routes (pre-built lambdas)   │
                                  │             │                         │
                                  │             ▼                         │
                                  │   server_context (the engine)         │
                                  │   ├─ slot scheduler                   │
                                  │   ├─ chat-template handling           │
                                  │   ├─ mtmd integration (vision/audio)  │
                                  │   ├─ streaming SSE                    │
                                  │   ├─ tool-call parsing                │
                                  │   └─ KV cache + sampler               │
                                  │                                       │
                                  │   chimera-owned handlers (optional)   │
                                  │   ├─ /v1/audio/transcriptions ───▶ chimera_whisper::transcribe
                                  │   └─ /v1/images/*             ───▶ chimera_sd::generate
                                  └───────────────────────────────────────┘
```

Three concrete consequences of this shape:

1. **The LLM endpoints are not chimera code.** We bind handlers that already
   exist on `server_routes`. Streaming, slot scheduling, tool calls,
   chat-template plumbing, KV reuse across requests — all come from
   `libserver-context.a`.

2. **Audio and image endpoints are chimera code,** running in the same
   process. They are registered on the *same* `server_http_context` via
   `ctx_http.post(path, handler)` — the documented extension point that
   `llama-server` itself uses for its CORS proxy, GCP/Vertex compat, and
   built-in tools.

3. **One process, three model backends.** The LLM, whisper, and SD models
   coexist if all three flags are passed. cpp-httplib serves requests on
   its own thread pool; the LLM engine runs on the main thread via
   `ctx_server.start_loop()`; audio and image work runs synchronously
   inside whichever httplib worker thread caught the request.

---

## 2. Build pipeline

### `scripts/manage.py`

The `LlamaCppBuilder` was already cloning llama.cpp and producing
`libllama.a`, `libllama-common.a`, `libmtmd.a` plus headers. Phase 1
added the server pieces:

```python
# extra_libs reported as available + cmake_build_targets entries
extra_libs = ["llama", "llama-common", "mtmd",
              "server-context", "cpp-httplib"]

# CMake configure step
LLAMA_BUILD_SERVER = True   # required so tools/server/CMakeLists.txt
                            # is included and the server-context target
                            # is defined
LLAMA_BUILD_WEBUI  = False  # skip baking the ~11 MB SvelteKit bundle
                            # into the static lib

# Targets built (note: llama-server *executable* is intentionally absent;
# we only need the static lib and cpp-httplib)
cmake_build_targets(targets=["llama", "llama-common", "mtmd",
                             "server-context", "cpp-httplib"], ...)

# Artifacts copied to thirdparty/llama.cpp/
copy_lib("tools/server",       "server-context", lib_dir)
copy_lib("vendor/cpp-httplib", "cpp-httplib",    lib_dir)
glob_copy("tools/server",  include_dir,            patterns=["server-*.h"])
glob_copy("vendor/cpp-httplib", include_dir/"cpp-httplib",
          patterns=["httplib.h"])   # subdir matters; server-http.cpp
                                    # includes it as <cpp-httplib/httplib.h>

# src-aux/: auxiliary sources we ship as code rather than as a library
src_aux = prefix/"src-aux"
glob_copy("tools/server", src_aux, patterns=["server-http.cpp"])
```

The `src-aux/` directory is a new concept and exists because **server-http
is not a separate library upstream** — its `.cpp` is on `llama-server`'s
own source list, not `libserver-context.a`'s. Rather than build our own
library out of one file, chimera's `CMakeLists.txt` pulls
`thirdparty/llama.cpp/src-aux/server-http.cpp` into the `chimera`
executable's `target_sources`. The compilation cost is the same; the
build graph is one library short.

### Top-level `CMakeLists.txt`

```cmake
static_lib(LIB_SERVER_CONTEXT "${LLAMACPP_LIB}" "server-context")
static_lib(LIB_CPP_HTTPLIB    "${LLAMACPP_LIB}" "cpp-httplib")
set(LLAMACPP_SRC_AUX "${LLAMACPP_DIR}/src-aux")

# OpenSSL is now required: cpp-httplib was built with LLAMA_OPENSSL=ON,
# so libcpp-httplib.a references X509_*, SSL_*, EVP_*, etc.
find_package(OpenSSL REQUIRED)
list(APPEND SYSTEM_LIBS OpenSSL::SSL OpenSSL::Crypto)

# macOS-only: cpp-httplib reads trust anchors from the system keychain
# via SecCertificateCopyData / SecTrustCopyAnchorCertificates.
if(APPLE)
    list(APPEND SYSTEM_LIBS "-framework Security" "-framework CoreFoundation")
endif()
```

`src/chimera/CMakeLists.txt` adds `chimera_serve.cpp` to the target's
source list, pulls in `server-http.cpp` from `src-aux/`, and links
`LIB_SERVER_CONTEXT` + `LIB_CPP_HTTPLIB` on every platform.

### `llama_build_info_shim.cpp`

`libserver-context.a` (specifically `server-task.cpp`'s
`to_json_oaicompat_chat_stream`) and `libllama-common.a` (download.cpp,
hf-cache.cpp) both reference `llama_build_info()`. Upstream generates
that string from `common/build-info.cpp.in` at configure time. chimera's
shim now returns the literal `"chimera"`; that string surfaces as the
`system_fingerprint` field in OpenAI responses.

---

## 3. Module layout

| File | Responsibility |
|------|---------------|
| `src/chimera/chimera_serve.cpp` | `command_serve()`, route registration, audio + image handlers, JSON-field coercion, base64, signal handling. |
| `src/chimera/chimera_whisper.h` | Public API: `load_model`, `load_wav_bytes`, `load_wav_file`, `resample_linear`, `transcribe`, `format_timestamp_10ms`. |
| `src/chimera/chimera_whisper.cpp` | Implementation. `command_whisper` (CLI) and the HTTP handler both call into this. |
| `src/chimera/chimera_sd.h` | Public API: `load_model`, `decode_image_bytes`, `decode_image_file`, `encode_png`, `save_png_file`, `generate`. `PixelImage` / `GenerateRequest` types. |
| `src/chimera/chimera_sd.cpp` | Implementation. `command_sd` (CLI) and the three image HTTP handlers both call into this. |
| `thirdparty/llama.cpp/src-aux/server-http.cpp` | Vendored from upstream by `manage.py`; compiled directly into the chimera target. Implements `server_http_context` over cpp-httplib. |

The whisper and SD modules grew dedicated headers as part of phases 2
and 3 specifically so the CLI subcommands and the HTTP handlers could
share one implementation. Before the refactor, the relevant helpers
lived in anonymous namespaces inside the `.cpp` files.

---

## 4. Routes — implemented, scope-limited, and skipped

### 4.1 Always exposed

| Route | Source | Notes |
|-------|--------|-------|
| `GET /health`, `GET /v1/health` | `routes.get_health` | Liveness probe; pre-built lambda from server_routes. |
| `GET /v1/models` | `routes.get_models` | Returns `{ "data": [...], "object": "list" }` plus ollama-compat fields. |
| `GET /metrics` | `routes.get_metrics` | Prometheus-style telemetry. `params.endpoint_metrics` is forced to `true` in `build_common_params`, so this works regardless of CLI flags. |
| `GET /props` | `routes.get_props` | Read-only introspection — current chat template, mmproj capabilities, generation defaults. |
| `POST /chat/completions`, `POST /v1/chat/completions` | `routes.post_chat_completions` | Streaming + non-streaming SSE. Tool calls, mtmd inputs, reasoning_content all work. The unprefixed legacy path is bound for older OpenAI clients. |
| `POST /v1/completions` | `routes.post_completions_oai` | Legacy text completion. |
| `POST /v1/embeddings` | `routes.post_embeddings_oai` | Only returns success when the model was loaded with `--embeddings`. Without it, upstream's handler returns HTTP 501 with the right message. |
| `POST /v1/messages`, `POST /v1/messages/count_tokens` | `routes.post_anthropic_messages` + `post_anthropic_count_tokens` | Anthropic Messages API compat — lets Anthropic SDK / claude-code-shaped clients point at chimera serve. |
| `POST /infill` | `routes.post_infill` | Fill-in-the-middle for code models. Returns 501 on models without FIM tokens, which is the right behavior. |
| `POST /tokenize`, `POST /detokenize` | `routes.post_tokenize` + `post_detokenize` | Vocab helpers; useful for clients that don't bundle a tokenizer (e.g. token counting before sending). |
| `POST /apply-template` | `routes.post_apply_template` | Renders the chat template against a `messages[]` array without generating. Pure debugging value. |
| `POST /v1/responses` | `routes.post_responses_oai` | OpenAI Responses API. **Stateful within a single chimera serve invocation** — server-context holds the conversation thread state in-process; lost on restart. With `--persist-chats` the underlying chat-completions traffic is still saved to the chats table. |

### 4.2 Opt-in via `--enable-audio` (shipped)

| Route | Handler | Notes |
|-------|---------|-------|
| `POST /v1/audio/transcriptions` | `make_audio_transcribe_handler` | chimera-owned. Routes through `chimera_whisper::transcribe`. |

Supported `response_format` values: `json` (default → `{"text":"..."}`),
`text` (raw `text/plain`), `verbose_json` (full structure with
`task` / `language` / `duration` / `segments[]`), `srt`, `vtt`.

Ignored request fields (accepted but no effect): `model`, `temperature`,
`timestamp_granularities[]`.

Why we do NOT bind `routes.post_transcriptions_oai`: that upstream
handler feeds audio through **mtmd's audio mmproj** (LLM-with-audio-tokens),
which is a fundamentally different pipeline from dedicated ASR. Two
different things — both have a place; one chooses based on the model.
We expose the ASR pipeline; the mtmd path is reachable via
`/v1/chat/completions` with an audio mmproj-aware model.

### 4.3 Opt-in via `--enable-image` (shipped)

| Route | Handler | Notes |
|-------|---------|-------|
| `POST /v1/images/generations` | `make_image_generations_handler` | application/json body. txt2img. |
| `POST /v1/images/edits` | `make_image_edits_handler` | multipart (`image` + optional `mask` + form fields). img2img / inpaint. |
| `POST /v1/images/variations` | `make_image_variations_handler` | multipart (`image`). img2img with no prompt. |

### 4.2b Opt-in via `--persist-chats` (shipped)

`--persist-chats` doesn't add new routes; it **wraps** the upstream
chat-completions handler. `make_persisting_chat_handler` decorates the
`routes.post_chat_completions` lambda so each successful exchange is
saved to the SQLite `chats` + `messages` tables.

Mechanics:
- The wrapper captures `ChatPersistContext *` (per-server state with
  the DB path + a write mutex) and a copy of the inner handler.
- On each request it parses the request body for the `messages` array
  + system prompt + model alias.
- For **non-streaming** responses, after the inner handler returns it
  parses `choices[0].message.{content,reasoning_content}` + `usage`
  and saves.
- For **streaming** responses, it replaces `res->next` with a wrapper
  that mirrors every chunk into a `shared_ptr<std::string>` buffer
  while still returning it to the client. When `next` returns false
  (stream finished) the buffered SSE is parsed event-by-event,
  `delta.content` is concatenated, and the result is saved.
- Persistence errors are caught + logged to stderr; they never affect
  the user's HTTP response.

One chat row per request — the OpenAI API has no chat-id concept, so
multi-turn clients (which resend the full conversation each request)
produce N rows with overlapping content. The `X-Chimera-Chat-Id`
request/response header (shipped) consolidates a multi-turn exchange
into a single row when the client threads it through; see
`chimera_serve.cpp` around the `X-Chimera-Chat-Id` block.

### 4.3a Opt-in via `--enable-rag` (shipped)

| Route | Handler | Notes |
|-------|---------|-------|
| `GET /v1/vector_stores`              | `make_vs_list_handler`   | List collections. |
| `POST /v1/vector_stores`             | `make_vs_create_handler` | Create. Body: `{"name": "..."}`. |
| `GET /v1/vector_stores/:name`        | `make_vs_get_handler`    | Stats. |
| `POST /v1/vector_stores/:name/delete` | `make_vs_delete_handler` | Drop. POST because `server_http_context` only wraps GET/POST. |
| `POST /v1/vector_stores/:name/files`  | `make_vs_ingest_handler` | Chunk + embed + insert. multipart `file` upload or JSON `{"text": ..., "source_uri": ...}`. |
| `POST /v1/vector_stores/:name/search` | `make_vs_search_handler` | KNN. JSON `{"query": ..., "k": ...}`. |

Shared per-server state is a `RagContext` struct (the loaded
`Embedder`, a `std::mutex` serializing embed calls, the db path, and
the embedding model name). All six handlers capture a pointer to it.
SQLite connections are opened per request (cheap in WAL mode); no
pool. Errors that would normally be 404 ("no such collection")
return 400 instead so the upstream `set_error_handler` 404-body
override (`server-http.cpp:140`) doesn't swallow the message.

Supported `response_format`: `b64_json` only (the default). `url` returns
HTTP 400 with an explicit "no static-file backend" message.

Supported SD-specific JSON fields (alongside OpenAI's `prompt`, `n`,
`size`, `response_format`): `negative_prompt`, `steps`, `cfg_scale`,
`seed`, `sample_method`, `scheduler`, `strength`.

Ignored: `model`, `user`, `quality`, `style`.

There is no server-context handler for image generation upstream; this
pipeline is entirely chimera's.

### 4.4 Deliberately NOT bound

Every one of these is a one-line `ctx_http.post(...)` away. The omission
is a scope choice, not a capability gap:

- `POST /completion`, `POST /completions` — legacy (non-/v1) llama.cpp completion shape, *different* from `/v1/completions`. Practically nobody calls it in 2025.
- `POST /embedding`, `POST /embeddings` — non-/v1 embeddings variants; redundant with `/v1/embeddings`.
- `POST /rerank`, `POST /v1/rerank` — document reranking via cross-encoder models. Niche; bind on request.
- `GET /slots`, `POST /slots/:id_slot` — slot save/load (KV cache snapshots). Bind on request.
- `GET /lora-adapters`, `POST /lora-adapters` — LoRA hot-swap. Bind on request.
- `POST /props` — mutating server props at runtime conflicts with chimera serve's "CLI is the config" stance. Read (`GET /props`) is bound; write is not.

Server-mode features not wired up:

- Router / multi-model mode (`is_router_server` branch in `llama-server`'s `server.cpp`). chimera serve loads one LLM.
- Built-in tool plugins (`--server-tools`). EXPERIMENTAL upstream.
- MCP CORS proxy (`--webui-mcp-proxy`). EXPERIMENTAL upstream.
- GCP / Vertex AI compat (`ctx_http.register_gcp_compat()`).
- Embedded Web chat UI. `manage.py` passes `LLAMA_BUILD_WEBUI=OFF` to skip baking the ~11 MB SvelteKit bundle.
- Child-server / parent-process sleeping notifications.
- SSL / TLS direct serving. Run behind a reverse proxy (nginx, caddy) for HTTPS.

---

## 5. Threading model

```
main thread ────────────────────── ctx_server.start_loop()  [blocks until shutdown]

cpp-httplib worker thread #1 ──── handler(req)  ──┐
cpp-httplib worker thread #2 ──── handler(req)  ──┤
cpp-httplib worker thread #N ──── handler(req)  ──┘
                                                  │
                          ┌───────────────────────┴───────────────────┐
                          │                                           │
                          ▼                                           ▼
            LLM routes: enqueue server_task,             chimera-owned routes:
            wait on response_reader.                      hold per-modality mutex,
            server_context's slot scheduler              call into chimera_whisper
            drives concurrency on the LLM side.          / chimera_sd synchronously.
```

**Why one mutex per modality:**
- `whisper_full` mutates the context's internal state; concurrent calls
  on the same context corrupt that state.
- `generate_image` is not thread-safe on a shared `sd_ctx_t` (both the
  diffusion graph and the VAE allocator are owned by the context).

This serializes audio and image requests across the server. For the LLM
side, `server_context`'s slot scheduler handles real parallelism via
KV-cache slots; we don't add a mutex there.

If audio or image throughput becomes a bottleneck the right answer is
*not* a finer-grained lock — it's holding multiple contexts (one per
worker thread). Both whisper.cpp and stable-diffusion.cpp can be loaded
multiple times; the cost is GPU memory.

---

## 6. Issues caught during implementation

### 6.1 Form fields arrive as JSON strings, not typed numbers

`server-http.cpp:485` collects multipart form text fields into a JSON
object where every value is `field.content` — *always a string*. For
application/json bodies, nlohmann/json preserves the original numeric
types. So the same field-reading code can't use `fields.value<int>("n", 1)`
across both — it works for `/v1/images/generations` (JSON body) and
throws `type_error.302` for `/v1/images/edits` (multipart).

Fix: `coerce_int`, `coerce_int64`, `coerce_float`, `coerce_string` in
`chimera_serve.cpp`. Each accepts the JSON-native type *or* a string
that parses to that type. All numeric reads in the image handlers go
through these. Worth re-using when adding new routes that may be reached
by either body type.

### 6.2 Linenoise width math vs. ANSI in the prompt

Embedded SGR escapes inside the prompt string passed to
`linenoise_read()` confuse `utf8_str_width` and corrupt the cursor on
multi-line edits. Fix (in `chimera_chat`, but the lesson stands): emit
the SGR escape to stdout *before* calling `linenoise_read`, pass a plain
prompt to linenoise, and emit a reset after. SGR state persists across
linenoise's own cursor moves.

This is not a server-mode bug but landed during the same arc of work.
If the server ever grows an interactive console it inherits the same
constraint.

### 6.3 `whisper_full_params.detect_language = true` is a different mode

It runs language identification and returns *without transcribing*. To
auto-detect language and transcribe, set `params.language = "auto"`
(or `nullptr`/`""`) and leave `detect_language = false`. The bug was
unreachable from the CLI (whose default is `language="en"`), so existing
tests never caught it; the HTTP handler exercised it by default because
the OpenAI spec's default is autodetect. Documented and fixed in
`chimera_whisper::transcribe`.

### 6.4 `cfg_scale=1.0` + `sample_method=euler` crashes the VAE on SDXL-Turbo

This is a stable-diffusion.cpp issue: certain `(cfg_scale, sample_method)`
combinations trigger `GGML_ASSERT(buft) failed` inside `VAE::encode`.
The crash reproduces from the CLI with the same params; it is not
HTTP-specific.

We currently surface this as a generic `image generation failed` HTTP
500. Two follow-ups worth considering:

- A pre-flight sanity check in `chimera_sd::generate` that rejects
  known-bad combinations with a 400 — but the combination space is
  model-specific and hard to enumerate.
- Capturing the `GGML_ASSERT` log line and bubbling it back into the
  HTTP error body. Would require installing a custom `ggml_log` handler
  in the SD code path and a small ring buffer to retrieve the last few
  lines on failure.

For now: document in the changelog, leave the broad error message in
place.

### 6.5 OpenSSL is a hard new dependency

cpp-httplib was already a transitive dep, but `LLAMA_OPENSSL=ON` in
`manage.py` means we now pull in libssl/libcrypto symbols. On macOS that
also drags in the system Security and CoreFoundation frameworks
(`SecCertificateCopyData` / `SecTrustCopyAnchorCertificates`) because
cpp-httplib reads the trust store from the keychain.

For Linux distros this is essentially zero friction. For Windows the
OpenSSL search path may need `OPENSSL_ROOT_DIR` set explicitly.

If we ever want a no-OpenSSL build, the move is to flip
`LLAMA_OPENSSL=OFF` in `manage.py` and accept that HTTPS-fetch features
(LoRA URLs, HF download via `common`) stop working.

### 6.6 Double-init of llama_backend

`command_serve` initially called `llama_backend_init()` itself, but
`main()` already does. The second call is harmless on most builds but
felt fragile; we removed it from `command_serve` and kept only
`llama_numa_init(params.numa)` (which depends on per-subcommand
`common_params`). The cleanup at end of `command_serve` similarly does
*not* call `llama_backend_free` — `main()` does that after
`command_serve` returns.

---

## 7. Things to watch out for

**Server-context API stability.** chimera tracks a pinned llama.cpp
commit via `LLAMACPP_VERSION` in `scripts/manage.py`. The
`server_context`, `server_routes`, and `server_http_context` types are
**not** part of upstream's public API; they shift with llama.cpp's
internal refactors.

The `make bump-check` target automates the diff. It fetches
`tools/server/server-context.h` and `tools/server/server-http.h` from
the upstream tag (defaults to the currently pinned version; override
with `make bump-check LLAMA_VERSION=bXXXX`) and compares them against
the headers currently vendored under
`thirdparty/llama.cpp/include/`. The output lists added/removed
top-level symbols (struct/class/enum names, `handler_t` fields,
function signatures) and a unified diff capped at 120 lines. Exit
code is 0 when clean, 2 when any header changed.

Recommended bump workflow:

```sh
# 1. See what's changing before touching anything.
make bump-check LLAMA_VERSION=<new_ref>

# 2. If anything came back, audit the call sites — chimera_serve.cpp
#    route bindings (especially `handler_t` fields on `server_routes`),
#    src/chimera/CMakeLists.txt link order / archive groups, and the
#    server-http.cpp source copy under thirdparty/llama.cpp/src-aux/.

# 3. Edit LLAMACPP_VERSION in scripts/manage.py, rebuild deps + chimera:
python scripts/manage.py build --llama-cpp --llama-version <new_ref>
make build
make test
```

Once `LLAMACPP_VERSION` is bumped and `make build` re-vendors the
headers, `make bump-check` will report "clean" again — the check is
**a pre-bump audit step**, not a CI guard.

**Sleeping / hibernation.** `server_context::on_sleeping_changed` exists
upstream for the router-server protocol; we don't wire it up. If we
ever add `chimera serve` to a router topology, this is the integration
point.

**Memory growth across many turns.** server-context manages its own
KV-cache lifetime via slots; we trust its behavior. If a deployment
shows unbounded memory growth, it is *likely* a server-context issue
rather than a chimera one — file upstream first.

**Web UI assets size.** `LLAMA_BUILD_WEBUI=OFF` keeps the chimera
binary ~11 MB smaller than `llama-server` by default. Both Web UI
delivery paths now ship as opt-in: the bundle can be baked into the
binary via `-DCHIMERA_WEBUI_EMBED=ON` (upstream's `xxd.cmake`
machinery; binary size up ~6 MB stripped) or served from any static
directory at runtime via `--public-path <dir>`. See
[`doc/dev/webui.md`](webui.md) for the full picture.

**Single SD context per process.** Loading multiple SD models at once
would multiply VRAM. For now the design is "one image model per
process"; if a deployment needs to switch models, restart the server.

**Single whisper context per process.** Same as SD. The model name in
OpenAI's transcription request is currently ignored; if we ever respect
it, we'd need a small registry of preloaded whisper contexts.

---

## 8. Future work

Ordered roughly by ROI per implementation effort. None of these is
blocking.

### Deliberately not supported

**Non-WAV audio in `/v1/audio/transcriptions`.** Currently only
RIFF/WAVE is accepted; mp3, m4a, mp4, mpeg, webm all return 415 with a
message telling the caller to transcode. Every viable path adds
dependencies that don't pull their weight: single-header decoders
(`dr_mp3.h` + `dr_flac.h`) only cover two of the formats users actually
send, and a complete solution drags in libavcodec / FFmpeg, which is
multi-megabyte and brings its own license + CVE surface. Clients
already have `ffmpeg` available far more often than they have an
in-process audio codec library, so we punt the decode to them. **Do
not revisit without a concrete user request.**

### Longer-term

9. **Web chat UI.** Either:
   - Re-enable `LLAMA_BUILD_WEBUI=ON` in `manage.py` and serve the
     baked-in assets. Binary grows ~11 MB. Simplest.
   - Ship assets next to the binary and route `GET /` plus
     `GET /{index.html, bundle.js, bundle.css}` through cpp-httplib's
     static-file machinery. Smaller binary; install complexity.

10. **SSE for image generation progress.** OpenAI's spec doesn't define
    SSE for `/v1/images/*`; some clients invent extensions. SD reports
    progress step-by-step via its callback (which we currently route to
    stderr). Adding an `event: progress` SSE stream alongside the final
    `data:` payload is technically straightforward — the question is
    which client convention to follow.

11. **Multi-tenancy.** Multiple LLMs loadable simultaneously, route by
    request's `model` field. Today we set `model_alias` to the loaded
    name and ignore `model`. Doing this properly is the
    `is_router_server` path in upstream's `server.cpp`; effectively a
    rewrite of `command_serve`.

12. **HTTPS direct serving.** `--ssl-cert-file` / `--ssl-key-file`. The
    machinery is in cpp-httplib; would mean wiring it through
    `server_http_context::init`. Most deployments will use a reverse
    proxy instead.

13. **Authentication beyond `--api-key`.** The current single-key
    bearer-token check is enough for "behind a VPN"; multi-tenant
    deployments will want JWT or per-key rate limiting.

14. **Reorganize `chimera_serve.cpp`** if it keeps growing. The current
    ~600 LOC handles five LLM routes + one audio route + three image
    routes + signal wiring + helpers. Threshold for splitting: when a
    new modality lands and the file crosses ~1000 LOC, split into
    `chimera_serve.cpp` (route registration + lifecycle) plus
    per-modality handler files (`serve_audio.cpp`, `serve_images.cpp`).

---

## 9. Useful references

- [`build/llama.cpp/tools/server/server.cpp`](../../build/llama.cpp/tools/server/server.cpp) — upstream's 363-line `llama-server` main. Our `command_serve` is structurally a stripped-down copy of this.
- [`build/llama.cpp/tools/server/server-context.h`](../../build/llama.cpp/tools/server/server-context.h) — `server_context`, `server_routes`, `server_context_meta` declarations.
- [`build/llama.cpp/tools/server/server-http.h`](../../build/llama.cpp/tools/server/server-http.h) — `server_http_context`, `server_http_req`, `server_http_res`, `uploaded_file`.
- OpenAI API docs: <https://platform.openai.com/docs/api-reference>.
- llama.cpp server README: <https://github.com/ggml-org/llama.cpp/tree/master/tools/server>.
