# Maintaining chimera across upstream changes

chimera bundles three large C++ projects (llama.cpp, whisper.cpp,
stable-diffusion.cpp) plus SQLite + sqlite-vec + cpp-httplib + a handful of
single-header utilities. Every bump in any of those is a potential
breakage. This doc surveys where breakage actually lands, what we
already do to contain it, and the highest-leverage things still
missing.

## Where breakage actually lands

Five categories, in roughly decreasing order of how much they hurt:

| Category | Symptom | Cost |
|---|---|---|
| **1. Link / compile drift in llama.cpp internals** | `undefined reference to common_speculative_*`; `PHANDLER_ROUTINE not found` on Windows; types in `server-context.h` quietly grow `handler_t` fields | Worst. Bites at link time after a version bump. Server bindings can break silently when upstream renames. |
| **2. API drift in stable APIs** (`llama_*`, `whisper_*`, `sd_*`) | Function rename, parameter added, enum value changed | Compile-time, usually easy to fix locally. |
| **3. Runtime / behavior drift** | Same headers, different semantics — e.g. upstream changes default pooling or chat-template handling | Hardest to detect. Tests pass but output silently changes. |
| **4. Schema / format drift** | Old GGUFs stop loading; old `chimera.db` breaks on new chimera | User-visible failure on existing installs. |
| **5. Build-system drift** | `LLAMA_BUILD_WEBUI=OFF` gets renamed; CMake target names change; backend flags shift | Per-bump fixup in `scripts/manage.py`. |

## What chimera already does well

- **Pinned versions, vendored headers.** `LLAMACPP_VERSION = "b9119"`,
  headers copied into `thirdparty/llama.cpp/include/` by
  `manage.py`. We control when an upstream change lands.
- **Per-modality TU isolation.** `chimera_whisper.cpp`,
  `chimera_sd.cpp`, `chimera_embed.cpp`. One upstream project → one
  chimera file, mostly localized.
- **`make bump-check`.** Diffs the vendored
  `server-context.h` / `server-http.h` against a target ref before you
  change the pin; lists added/removed top-level symbols. See
  `doc/dev/server.md` § 7.
- **`make test-db-migrate`.** Builds a v1 chimera.db in a tempdir and
  asserts it upgrades to current cleanly with all pre-existing rows
  preserved.
- **`make test`** — 23 e2e cases hitting every subcommand.
- **`chimera info`** — single command captures every version + backend
  at runtime; useful for bug reports.
- **`doc/dev/server.md` § 7** — explicit list of upstream types we
  depend on.
- **Link surgery contained.** `-Wl,--start-group` on Linux,
  `-Wl,-force_load` on macOS, MSVC `/WHOLEARCHIVE` on Windows — all in
  `src/chimera/CMakeLists.txt`, not scattered.

## What's still weak, ranked by leverage

### 1. ~~HTTP-response golden tests~~ — DONE

Implemented as `scripts/test_golden.py` + `tests/golden/`, driven by
`make test-golden`. Spawns chimera serve against fixed models on a
free port, hits 9 routes (`/v1/health`, `/v1/models`, `/props`,
`/tokenize`, `/detokenize`, `/apply-template`, `/v1/embeddings`,
`/v1/chat/completions`, `/v1/completions`), runs each response
through a route-specific normalizer (redact volatile / shape-check
generated text), and diffs against checked-in JSON. Pinned to CPU
(`--gpu-layers 0`) so the goldens stay portable across machines.
`UPDATE_GOLDEN=1` to refresh after a deliberate shape change.

### 2. ~~Widen `bump-check`~~ — DONE

`make bump-check` now covers `include/llama.h`, `common/common.h`,
`common/arg.h`, `common/chat.h`, `tools/mtmd/mtmd.h`, plus the
original `tools/server/server-{context,http}.h`. whisper.h and
stable-diffusion.h aren't covered yet (different upstream repos;
extension is straightforward if a bump bites).

### 3. ~~Compile-time pin assertions~~ — DONE

`src/chimera/chimera_pin_check.cpp` for the llama.cpp surface
(every `server_routes` handler_t field, `common_params` field types,
`LLAMA_POOLING_TYPE_*` enum values, key `llama_*` function
signatures). A matching per-modality block lives in
`chimera_whisper.cpp` for whisper signatures (ggml.h collisions mean
whisper / sd can't share a TU with llama).

### 4. Adapter shim between chimera_serve and `server_routes`

`chimera_serve.cpp` directly references upstream lambda names. Wrap
each binding in a one-line indirection —
`chimera::routes::chat_completions(routes)` returning
`routes.post_chat_completions` — so when upstream renames
`post_chat_completions` → `chat_completions_post`, you change one line
in a shim header rather than greping through `chimera_serve.cpp`.
Modest churn, big localization win. Could land incrementally as
bindings break.

### 5. Bump PR template / checklist

A `.github/PULL_REQUEST_TEMPLATE.md` (or a one-page doc) listing the
steps for a version bump: run `make bump-check`, run `make test`, run
`make test-db-migrate`, eyeball golden diffs, update CHANGELOG. Cheap,
prevents the "I forgot" failure mode.

### 6. Pin compatibility window in CI

A nightly job that builds against `LLAMACPP_VERSION` *and* against
`LLAMACPP_VERSION - 100` (some rolling old pin), then runs
`make smoke`. If we accidentally start using a symbol only present in
the new pin, the old-pin build flags it. Costs CI minutes; catches
"subtle dependency widening" before a user hits it. Optional — only
worth the cost if you want a real back-compat window.

### 7. Cross-component invariants

The one runtime cross-component invariant today is
`GGML_MAX_NAME=128` (SD requires it; llama.cpp defaults to 64; both
must agree because they share ggml). Today the constraint lives in
`manage.py` via `SD_USE_VENDORED_GGML=0` +
`CMAKE_CXX_FLAGS=-DGGML_MAX_NAME=128`. A runtime assert at startup
(`chimera info` already prints `ggml_version`; add a check that the
two libraries report the same string) would surface a ggml mismatch
before it manifests as silent tensor corruption. Tiny diff.

### 8. Reduce coupling to internal-API headers where possible

Some upstream features have *both* a public C API and an internal C++
API (`server_routes` is purely the latter). Where we have a choice,
prefer the C API — `llama.h` is far more stable than `common.h` or
`server-context.h`. Long-term refactor; only do this when a particular
`server_routes` field becomes annoying enough to relocate.

## Status

Items 1–3 above (golden tests, widened bump-check, pin assertions) +
the bump PR checklist (`.github/PULL_REQUEST_TEMPLATE.md`) all landed.
Items 4 (adapter shim) and beyond stay on the radar — only worth the
churn when a specific bump actually hurts in a way the above don't
already catch.

## The pattern

Keep the **boundary** of "upstream code we link against" small and
explicit. **Test the behavior at the boundary** (golden files,
compile asserts, schema fixtures). Almost everything else falls out
of that.

## What's bespoke vs vendored — quick map

Useful when triaging "is this our bug or theirs":

**Vendored (upstream owns the code)**
- llama.cpp: text LLM engine, chat templating, mtmd, OpenAI-API
  handlers in `server_routes`.
- whisper.cpp: ASR.
- stable-diffusion.cpp: txt2img / img2img / VAE.
- cpp-httplib (via llama.cpp): HTTP server.
- SQLite + sqlite-vec: amalgamation drop-ins. FTS5, vec0 virtual
  tables.
- OpenSSL: TLS + `EVP_sha256` for embedding-cache fingerprints.
- Header-only: nlohmann/json, CLI11, rang, linenoise, stb_image.

**Bespoke (chimera owns)**
- CLI surface: every subcommand and its flags.
- `chimera serve` wiring: which upstream routes get bound, with what
  overrides, all the per-modality handler factories
  (`make_audio_transcribe_handler`, `make_image_*_handler`,
  `make_vs_*_handler`), the SSE-aware persistence wrapper, the
  `SecondaryServerCtx` machinery for `--enable-embeddings` /
  `--reranking`.
- `chimera_db` (schema + migrations), `chimera_vector_store`,
  `chimera_embed` + `chimera_embed_cache`, `chimera_chat_store`,
  `chimera_whisper` (WAV decode, resample, word timestamps),
  `chimera_sd` (GenerateRequest, PNG i/o, log ring).
- Chat REPL UX: slash commands, linenoise integration, ANSI color
  layer, spinner, tok/sec stats.
- Build system: `scripts/manage.py`, `CMakeLists.txt` glue, link-order
  surgery, `bump-check`, release workflow.
- Tests + docs: `scripts/test.sh`, `scripts/test_db_migrate.py`,
  `doc/`, `doc/dev/`.

**Mental shortcut**: if it's about *running a model* (decode, sample,
embed, transcribe, generate-image) it's upstream. If it's about *what
subcommand does what, which routes are bound, what gets persisted,
how chunks are sized, how the binary is built and shipped, or how the
REPL feels* — that's chimera.
