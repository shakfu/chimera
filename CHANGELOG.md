# Changelog

All notable changes to chimera will be documented in this file. Format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.4]

### Added

- Sentence-aware chunking for `chimera index ingest` and the equivalent serve route. The previous fixed token-window splitter (with overlap measured in tokens) is replaced by `chimera_embed::chunk_by_sentences`: text is split on sentence terminators (`.`, `?`, `!`) and paragraph breaks (blank lines), then greedy-packed into chunks bounded by the collection's `chunk_tokens` budget. Overlap is carried as whole-sentence tails of the previous chunk re-prepended to the next. Pathological inputs (run-on sentences longer than the budget, source code, base64) fall back to `chunk_by_tokens` for the offending span, so ingestion never refuses input. Empirically this improves retrieval quality on prose because the embedded text now corresponds to complete thoughts rather than arbitrary mid-sentence cuts; for structured/non-prose input the behavior degenerates to the old splitter via the fallback.

- Hybrid retrieval with reciprocal-rank fusion. New `documents_fts` virtual table (FTS5 over `documents.text`) is created in schema v5 and back-populated from existing rows via `INSERT INTO documents_fts(documents_fts) VALUES('rebuild')`; insert/delete/update triggers keep it in sync with `documents`. `chimera_vector_store::SearchMode` selects between three retrieval paths:
  - `Semantic` — vec0 KNN on the collection's chosen distance metric (unchanged behavior).
  - `Lexical` — FTS5 BM25 over the chunk text. Falls back to phrase-quoted form on FTS5 syntax errors (e.g. user-typed queries with stray parens or quotes) so search never returns HTTP 500 for input it could otherwise tolerate.
  - `Hybrid` *(default)* — pulls top-`max(k, 30)` from each leg and merges by `rrf_score = Σ 1 / (60 + rank_i)`. Hits gain `semantic_rank`, `lexical_rank`, and `rrf_score` fields; output sorted by RRF score descending.

  Surface area: `chimera search --mode {semantic|lexical|hybrid}` (default `hybrid`); `POST /v1/vector_stores/:name/search` body accepts `"mode"` with the same values and echoes the chosen mode on the response. Lexical-only mode short-circuits the embedding model load — a 100+ MB GGUF doesn't need to be paged in to do a BM25 lookup. The choice to make `hybrid` the default changes search results for existing collections; collections created before this release continue to work because the FTS5 table is rebuilt from existing rows during the v5 migration.

- `X-Chimera-Chat-Id` request/response header on `/v1/chat/completions` for multi-turn server clients. Closes the gap where `--persist-chats` previously produced one chat row per request (because the OpenAI API has no chat id). Behavior:
  - Request without the header: the wrapper creates a chats row *before* delegating to the inner handler so its id is known in time to be set on the streaming response. Response echoes `X-Chimera-Chat-Id: <new-id>`. All messages in the request plus the assistant reply are appended (unchanged).
  - Request with `X-Chimera-Chat-Id: <existing-id>`: only the **last** message in the body plus the assistant reply are appended; prior turns are assumed to already be on disk from earlier calls. Response echoes the same id.
  - Request with an unknown id: HTTP 404, inner handler not invoked.
  - Request with a non-integer id: HTTP 400.

  Header lookup is case-insensitive (HTTP semantics). Persistence still runs *after* the stream finishes, so a failed persist never breaks the user's request. New helpers in `chimera_serve.cpp`: `create_chat_row_for_request`, `chat_id_exists`, plus a case-insensitive header-map lookup.

- Ctrl-C in `chimera chat --persist` now persists the in-flight assistant turn instead of discarding it. A SIGINT handler installed only for the duration of `chat_sample_loop` (RAII via `ChatSigintGuard`) flips an atomic that the per-token loop polls; on trip the loop exits early, the streamed content so far is written with a new `partial=1` flag, and the REPL prints `[interrupted — partial response saved]` before returning to the prompt. Handler is uninstalled outside generation so Ctrl-C at the prompt still goes to linenoise. POSIX uses `sigaction` without `SA_RESTART` (so blocked syscalls EINTR cleanly); Windows uses `std::signal`.

  Schema v4 adds `messages.partial INTEGER NOT NULL DEFAULT 0`; existing rows backfill to 0 (interpretation: every previously-persisted message was complete). `chimera_chat_store::append_message` takes an optional `partial` arg; `StoredMessage.partial` and `Chat.partial_count` (populated by `list_chats` via a correlated subquery) are exposed.

  Surfacing: `chimera chat --list` shows `(N interrupted)` next to chats containing partial turns; `chimera chat --resume <id>` includes the interrupted count in its banner.

### Changed

- Schema v5: `documents_fts` virtual table added with content-table backing on `documents.id`, and three sync triggers (`documents_ai/ad/au`). Existing collections get a `rebuild` of the FTS5 index during migration so hybrid/lexical search works against previously-ingested documents without re-ingest.

- Default retrieval mode flipped from semantic-only to hybrid. Pre-hybrid behavior is recoverable via `--mode semantic` (CLI) or `"mode": "semantic"` (HTTP). Justified because hybrid is a superset of semantic for prose: keyword recall is added (helping proper nouns and rare-term queries) without losing the conceptual matches that semantic-only delivered. Cost is one additional FTS5 SELECT per request.

- `scripts/test.sh` end-to-end suite is now 32 tests (was 23). New coverage:
  - Hybrid retrieval (4 tests): a small corpus with a deliberately rare proper-noun-like token asserts that `--mode lexical` surfaces it via BM25, `--mode hybrid` produces `rrf=`-annotated output, omitting `--mode` falls through to hybrid (default-mode regression guard), and `--mode <bogus>` exits with BadInput.
  - `X-Chimera-Chat-Id` header (5 tests): spawns `serve --persist-chats` on a free port, asserts the response header on a no-header request, echo-id-reuse, unknown id → 404, malformed id → 400, and a final DB-state check (1 chats row + 4 messages rows after the four-case sequence — proving the unknown/malformed cases never invoked the inner handler).
  - Skipped on CI containers without `python3` + `curl` + `sqlite3` (SKIP, not FAIL).

- Privacy documentation for persistence features. New "Privacy / data on disk" section in `doc/serve.md` enumerating exactly what `serve --persist-chats` and `chat --persist` record (message content, reasoning spans, model path, token counts, timestamps, media paths; explicitly *not* client IPs, headers, API keys, or HTTP bodies), where the DB lives per platform, and how to wipe persisted state. A matching summary table in `doc/cheatsheet.md` covers all five write-to-disk surfaces (chat persist, serve persist-chats, RAG ingest, embedding cache, linenoise history). Closes the gap where opt-in persistence shipped without a user-facing privacy note.

### Removed

- `TODO.md` pruned of items that have shipped (sentence-aware chunking, hybrid search, `X-Chimera-Chat-Id`, Ctrl-C partial-turn persistence, privacy notes). Four items moved to a new "Out of scope (wontfix)" section with brief rationale rather than silently deleted, so the same proposals aren't re-litigated from scratch:
  - `POST /props` runtime config mutation (conflicts with "CLI is the config").
  - Multi-tenancy / `is_router_server` (wrong deployment shape — one process = one model is core to the busybox identity; users needing multi-model routing should run one chimera per model behind a reverse proxy).
  - HTTPS direct serving via `--ssl-cert-file` / `--ssl-key-file` (reverse-proxy territory).
  - Auth beyond `--api-key` — JWT, per-key rate limiting, multi-tenant auth (reverse-proxy / API-gateway territory).

  The web chat UI item remains in TODO but was restructured as two independent opt-in variants (`CHIMERA_WEBUI_EMBED` = xxd-baked single binary; `CHIMERA_WEBUI_PATH` = `--public-path` static-file serving). Both gated on a new `manage.py build --webui` step that pulls in the Node toolchain only when requested.

## [0.1.3]

### Added

- `chimera serve --enable-embeddings <model.gguf>`: load a dedicated embedding model alongside the LLM and route `/v1/embeddings` to it. Same opt-in pattern as `--enable-audio` / `--enable-image`. When set, the primary LLM stays in generative mode and the embeddings endpoint is served by a secondary `server_context` running its own task loop on a worker thread. If both `--embeddings` (single-model embed mode on the primary) and `--enable-embeddings` are passed, the dedicated model wins.

- `chimera serve --reranking <model.gguf>`: load a cross-encoder reranker and bind `POST /v1/rerank` (and `/rerank`). Same secondary-context machinery as `--enable-embeddings`, but with `pooling_type=LLAMA_POOLING_TYPE_RANK` — matches the toggle llama-server uses upstream. Natural follow-up to the RAG vector search: top-N hits from `/v1/vector_stores/:name/search` → rerank → top-k → LLM.

  Both flags share a new `SecondaryServerCtx` helper in `chimera_serve.cpp`: heap-allocated, non-movable (because `server_routes` keeps a `const common_params &`), one worker thread per secondary running its `start_loop()`. Shutdown handler terminates secondaries before the primary so the loops unblock cleanly; their threads are joined after `ctx_http.thread`.

- Word-level timestamps in `/v1/audio/transcriptions`. Pass `timestamp_granularities=["word"]` (multipart `timestamp_granularities[]=word` works too) with `response_format=verbose_json` and the response gains a top-level `words[]` array of `{start, end, word}` entries. Implementation walks `whisper_full_get_token_data` per segment, filters tokens at or above `whisper_token_beg` (timestamp specials like `[_BEG_]` / `[_TT_550]`), and groups remaining pieces into words by leading-space token boundaries. New fields: `chimera_whisper::Word`, `Segment::words`, `TranscribeRequest::word_timestamps`.

- `POST /v1/audio/translations`. Trivial bind of the existing transcription handler with `translate=true` in `TranscribeRequest` — whisper does the to-English translation inline, output goes through the same response-format pipeline as transcription (`json` / `text` / `srt` / `vtt` / `verbose_json`).

- Token-based chunking for `chimera index ingest` + `POST /v1/vector_stores/:name/files`. The previous character-window splitter (with sentence-boundary nudge) is gone; chunks are now sized in tokens of the loaded embedding model's vocab via `chimera_embed::chunk_by_tokens`. Default 512 tokens with 64-token overlap — matches the input limit of common encoders (bge-small, gte-small). Tokens are decoded once per source via `llama_tokenize`; each chunk is `detokenize`'d back to text. Eliminates the 400–800-token variance of the old proxy and ensures chunks always fit through `embed()` without truncation.

- Per-collection chunk + distance knobs. `collections` gains three new columns (`distance`, `chunk_tokens`, `chunk_overlap`) in schema v3, set at `chimera index create` time and read by `chimera index ingest` / the equivalent serve route unless the caller overrides on the command line. New flags:
  - `chimera index create --distance cosine|l2|l1` — picks the sqlite-vec `distance_metric` baked into the per-collection `vec_<id>` virtual table. Default cosine (right for L2-normalized embeddings, which is also the default `chimera embed --normalize=true`). Validated at create time; invalid values get a `BadInput` error.
  - `chimera index create --chunk-tokens N --chunk-overlap N` — collection-wide defaults.
  - `chimera index ingest --chunk-tokens N --chunk-overlap N` — per-call overrides; otherwise use whatever the collection row recorded.
  - `POST /v1/vector_stores` body now accepts `distance`, `chunk_tokens`, `chunk_overlap`; `/v1/vector_stores/:name` and `GET /v1/vector_stores` surface them under `meta`.

  Schema v3 backfills existing rows with cosine/512/64 via `ALTER TABLE … ADD COLUMN … NOT NULL DEFAULT …`, so users upgrading from v1 / v2 DBs get sensible defaults without touching the DB.

  New public API:
  - `chimera_embed::Embedder::tokenize / detokenize / n_ctx`
  - `chimera_embed::TokenChunk` + `chunk_by_tokens(text, embedder, chunk_tokens, overlap_tokens)`
  - `chimera_vector_store::CreateOptions`, `is_valid_distance`

- Persistent embedding cache. New `--cache-embeddings` flag on `chimera embed`, `chimera index ingest`, `chimera search`, and `chimera serve` memoizes `embed(text) -> vector` to SQLite (the `embedding_cache` table added in schema v2) so repeated work skips the model. Key: `(model_id, sha256(text))`. `model_id` is a fast fingerprint of the embedding model file (SHA-256 of `size || first 64 KB || last 64 KB`); GGUFs store metadata in the header, so this catches re-quantization, re-training, and architecture swaps without our code tracking model names. Vectors are stored as raw little-endian float32 blobs and round-trip bit-identical. Default OFF — cache rows take real disk (`dim*4 + 32` per row; 384-dim float32 ≈ 1.5 kB; 100k entries ≈ 150 MB) so users opt in.

  Wiring: new `chimera_embed_cache` module (`Cache`, `compute_model_id`, `sha256_bytes`); `chimera_embed::Embedder` grows a `set_cache(...)` setter and consults the cache before tokenize/decode, writing the final (normalized, if enabled) vector back on miss. `chimera serve --cache-embeddings` reuses `--rag-db` and attaches the cache to the RAG `Embedder`; CLI subcommands take `--cache-db` (`chimera embed`) or reuse `--db` (`chimera index ingest`, `chimera search`). Smoke test added: two embed calls with the same input produce bit-identical output.

- `make bump-check` + `scripts/manage.py bump_check`: fetch the upstream `tools/server/server-context.h` and `tools/server/server-http.h` at a target llama.cpp ref (defaults to the currently pinned `LLAMACPP_VERSION`) and diff them against the headers vendored under `thirdparty/llama.cpp/include/`. Output lists added/removed top-level symbols (struct/class/enum names, `handler_t` fields, function signatures) plus a unified diff capped at 120 lines. Exit code 0 when clean, 2 when any header changed. These headers are NOT part of upstream's stable API — last week's CI failure was caused by a new symbol appearing in `libllama-common` that our `--start-group` fix happened to catch; this script surfaces the surprise at bump time. The check is a pre-bump audit step (run before changing `LLAMACPP_VERSION`), not a CI guard. Documented in `doc/dev/server.md` § 7.

- SD log capture for image-generation error bodies. `chimera_sd.cpp` now mirrors every log line emitted via `sd_set_log_callback` (which sd.cpp uses to forward ggml log lines too) into a 64-entry mutex-protected ring buffer. The HTTP image handler clears the ring before calling `chimera_sd::generate`, and on throw appends up to 8 of the most recent lines under a `recent SD log:` header in the error body. The underlying generic message (`image generation failed`) stays the same; the new tail carries the descriptive line sd.cpp emitted — buft failures, scheduler/sampler-name rejections, ggml backend errors — which previously only reached stderr. New public API: `chimera_sd::recent_log_lines(max)` + `chimera_sd::clear_log_buffer()`. Does not help with cases where sd.cpp aborts the process via `GGML_ASSERT` (e.g. the SDXL-Turbo `cfg_scale=1.0`+`euler` crash); those still terminate the server.

### Changed

- Release workflow (`.github/workflows/release.yml`) packages each platform's binary as a compressed archive instead of uploading the raw executable. Unix targets ship `chimera-<version>-<target>.tar.gz`; Windows ships `chimera-<version>-windows-x86_64.zip`. Inside, the binary keeps its plain name (`chimera` / `chimera.exe`). The `.sha256` sidecar files were dropped — the GitHub release UI already shows a SHA-256 next to every asset.

- Windows release matrix target renamed `windows-x86_64.exe` → `windows-x86_64` so the archive name no longer ends in `…windows-x86_64.exe.zip`. No effect on the unzipped binary, which is still `chimera.exe`.

- Non-WAV audio in `/v1/audio/transcriptions` is now framed as a deliberate non-feature rather than a temporary limitation. Error body on a non-WAV upload points the user at `ffmpeg -i in.<ext> -ar 16000 -ac 1 out.wav`. Bundling audio codecs is out of scope: every viable path (single-header `dr_mp3` + `dr_flac` for partial coverage, or libavcodec / FFmpeg for the full set) adds dependencies that don't pull their weight. Updated `doc/serve.md` § "What's not supported" + `doc/dev/server.md` § 8 with a "do not revisit without a concrete user request" note.

- `scripts/test.sh` end-to-end suite is now 23 tests (was 22): added an embedding-cache round-trip check that asserts two `chimera embed --cache-embeddings` calls with the same input produce bit-identical output.

- `make test-golden` target + `scripts/test_golden.py` + `tests/golden/`. Spawns `chimera serve` against fixed models (`Llama-3.2-1B-Instruct-Q8_0.gguf` + `bge-small-en-v1.5-q8_0.gguf`) on a free port, hits 9 routes (`/v1/health`, `/v1/models`, `/props`, `/tokenize`, `/detokenize`, `/apply-template`, `/v1/embeddings`, `/v1/chat/completions`, `/v1/completions`) with fixed payloads, runs each response through a route-specific normalizer (redacting `created` / `id` / port-dependent / model-path-dependent fields; shape-checking generated text and embedding floats), and diffs against checked-in JSON goldens. Catches *runtime drift* in upstream `server_routes` lambdas — the class of regression where `make test`'s did-it-work smoke can't see a JSON key being renamed or dropped. `UPDATE_GOLDEN=1 make test-golden` regenerates the goldens after a deliberate shape change. Backend-pinned via `--gpu-layers 0` so the goldens stay portable across dev machines and CI.

- Widened `make bump-check`: in addition to `tools/server/server-{context,http}.h`, the diff now covers `include/llama.h`, `common/common.h`, `common/arg.h`, `common/chat.h`, and `tools/mtmd/mtmd.h` — the full set of upstream-internal headers chimera actually links / compiles against. Same script, same exit-code semantics.

- Compile-time pin assertions (`src/chimera/chimera_pin_check.cpp` + a per-modality block in `chimera_whisper.cpp`). `static_assert`s on the upstream surface chimera depends on: every `server_routes::*` field bound in `chimera_serve.cpp` is asserted to be a `handler_t`; `common_params` fields we poke are asserted to keep their types; `LLAMA_POOLING_TYPE_*` enum values are pinned by integer; key `llama_*` and `whisper_*` function signatures are pinned via typed function pointers. If a llama.cpp bump renames a handler or changes a `common_params` field type, this file fails to compile with a labeled error (`common_params::embedding changed type.`) instead of cascading into a cryptic instantiation failure deep in `chimera_serve.cpp`. Verified to catch the failure class: a deliberate `embedding: bool -> int` flip produced the labeled `static assertion failed` at `chimera_pin_check.cpp`.

- `.github/PULL_REQUEST_TEMPLATE.md` with a Dependency-bump checklist for the `*_VERSION` lines in `scripts/manage.py` (run `make bump-check` before bumping; verify `make test`, `make test-db-migrate`, `make test-golden` after; audit `chimera_pin_check.cpp`, route bindings, CMake link order). Generic PR sections live above the checklist for non-bump PRs.

- `doc/dev/maintenance.md` — strategy doc covering where breakage lands across upstream changes, what we have, what's still weak, and the bespoke-vs-vendored map for triage. Companion to `doc/dev/server.md` § 7.

- `make test-db-migrate` target + `scripts/test_db_migrate.py`: builds a v1-schema chimera.db in a temp dir, seeds it with one row per pre-existing table, drives `chimera db status` against it (which calls `open_and_migrate`), and asserts that (a) `PRAGMA user_version` advances to the current latest, (b) pre-existing rows survive, (c) v2 / v3 additions are in place (`embedding_cache` table; `collections.distance` / `chunk_tokens` / `chunk_overlap` backfilled to `'cosine'` / `512` / `64`). Verified to catch regressions — a deliberate default-value change in the migration SQL produced `FAIL: collections.chunk_tokens default: got 999, want 512`. Pre-empts the class of bug where a new migration silently breaks the v1 → latest upgrade path for users still on the original schema.

### Removed

- `chimera index ingest --chunk-chars` (character-window chunker). Replaced by `--chunk-tokens` (token-window chunker; see "Token-based chunking" above). Pre-existing scripts using `--chunk-chars` need to switch flag names; the unit also changed (chars → tokens), so a literal `--chunk-chars 2048` is closer to `--chunk-tokens 512` for English. `--chunk-overlap` survives but its unit is now tokens, not chars.

## [0.1.2]

### Added

- `chimera info` subcommand: prints chimera version + platform tag, then one block per bundled component:
  - **llama.cpp**: version, ggml version + commit, the primary backend chimera was built with, the full list of registered ggml backends, enumerated ggml devices with type tags (`GPU` / `CPU` / `ACCEL` / `IGPU`), and the four `llama_supports_*` capability flags (GPU offload, MMAP, MLOCK, RPC).
  - **whisper.cpp**: version, ggml version, parsed CPU features from `whisper_print_system_info()`.
  - **stable-diffusion.cpp**: version, ggml version, parsed CPU features from `sd_get_system_info()`.
  - **sqlite + sqlite-vec**: versions.

  Output shape matches cyllama's `info` subcommand so users hopping between the native binary and the Python sibling see one familiar format. Useful artifact for bug reports — a single command captures every version and backend chimera saw at link time and at startup.

  Two new helper headers (`chimera_whisper.h::whispercpp_version` etc., `chimera_sd.h::sdcpp_version` etc.) expose the per-library accessors through the per-modality TUs, avoiding the ggml.h cross-library collision. `chimera_sd.cpp` forward-declares `ggml_version` rather than including ggml.h.

- `chimera serve` phase 5: server-side chat persistence + the OpenAI Responses API.
  - `--persist-chats` opts the chat-completions endpoint into per-request DB writes. Implementation is a wrapper around `routes.post_chat_completions` (`make_persisting_chat_handler`) that doesn't change the response — clients see exactly the same bytes as before — but saves a copy to the `chats` + `messages` tables after each successful exchange.
  - **Streaming SSE is handled**, not just non-streaming JSON. The wrapper replaces `res->next` with a closure that mirrors each chunk into a `shared_ptr<std::string>` buffer while still returning it to the client. When `next` returns false (stream end) the buffered SSE is parsed event-by-event, `delta.content` + `delta.reasoning_content` are concatenated, and the result is persisted.
  - Errors in the persistence path are caught + logged to stderr; they never break the client's HTTP response.
  - The DB is shared with the CLI's `chat --persist` and with `--enable-rag` — same `$CHIMERA_DB` / platform default; override with `--chat-db`.
  - **`POST /v1/responses` is now bound** (`routes.post_responses_oai`). This was deferred in the original "group A" server work because chimera serve was stateless across requests; phase 5 partially lifts that constraint. The Responses API itself is still stateful *within a single chimera serve invocation* — server-context holds thread state in-process and loses it on restart — but the underlying chat-completions traffic is persisted when `--persist-chats` is on, so audit-log use cases work.
  - One chat row per request, by design. The OpenAI API has no chat id concept, so multi-turn clients (which resend the full conversation each request) produce N overlapping rows. The duplication is the cost of staying API-compatible; phase 6+ may add an `X-Chimera-Chat-Id` header.

- `chimera serve` phase 4: OpenAI-shaped vector-store / RAG routes backed by the SQLite + sqlite-vec layer from phases 1–2. Opt-in via `--enable-rag <embedding.gguf>`; the SQLite DB is shared with the CLI (`$CHIMERA_DB` or the platform default; override with `--rag-db`). Six new routes on the same `server_http_context` that hosts the LLM + audio + image endpoints:
  - `GET  /v1/vector_stores` — list
  - `POST /v1/vector_stores` — create. Body: `{"name": "..."}`.
  - `GET  /v1/vector_stores/:name` — stats (doc count, dim, embedding model, created_at).
  - `POST /v1/vector_stores/:name/delete` — drop. **POST**, not DELETE — see below.
  - `POST /v1/vector_stores/:name/files` — ingest. Accepts either a multipart `file` upload (treated as UTF-8 text; `filename` becomes the `source_uri`) or a JSON body `{"text": "...", "source_uri": "..."}`. Chunks via the same character-window splitter the CLI uses (2048/256 default), embeds each chunk through the shared `Embedder`, inserts into `documents` + the per-collection `vec_<id>` virtual table.
  - `POST /v1/vector_stores/:name/search` — KNN. Body `{"query": "...", "k": 5}`. Returns top-k hits with `text`, `source_uri`, `chunk_index`, `distance`.

  Implementation:
  - One `Embedder` per server, serialized on `RagContext::embedder_mutex` (same pattern as whisper / SD). SQLite connections are **opened per request** rather than pooled — open is microseconds in WAL mode, so the pool ceremony isn't worth it.
  - One embedding model per server in this cut. If a collection's recorded `embedding_model` doesn't match what `--enable-rag` loaded, ingest and search return a clear 400 pointing at the flag. Multi-model would be Phase 4.1.
  - `chimera_serve.cpp` got a `RagContext` struct, six handler factories (`make_vs_*_handler`), and a local `serve_chunk_text` duplicating the CLI's chunker (one extra call site isn't worth promoting the helper to a shared header).

  **Two server-http warts surfaced and worked around:**
  - `server_http_context` only exposes `get()` and `post()` (the wrapped subset of cpp-httplib). Adding DELETE would mean patching our vendored `server-http.cpp`, which becomes a per-llama.cpp-version maintenance burden. So drop is `POST :name/delete`. OpenAI SDK clients sending `DELETE /v1/vector_stores/{id}` need to be reconfigured; documented in `doc/dev/server.md` § 4.3a and `doc/serve.md`.
  - Upstream's `server-http.cpp:140` `set_error_handler` unconditionally overwrites response bodies on status 404 with a generic `"File Not Found"` payload. To keep our specific error messages visible ("no such collection: 'missing'"), those errors return **400** (`invalid_request_error`) instead. Defensible semantically — the URL pattern matched, the named resource inside it didn't.

- Persistent chat history (phase 3 of `doc/dev/sqlite.md`). The secondary driver for embedding SQLite — `chimera chat` sessions can now be saved, listed, searched, and resumed across invocations. The data lives in the same `chimera.db` as the vector store from phase 2.

  New flags on the `chat` subcommand:
  - `--persist` opts a session into per-turn DB writes. Off by default so existing users see no behavior change. Each turn (user, then assistant) becomes one row in `messages`, with reasoning content (text between `<think>...</think>`) captured into the `messages.reasoning` column. The chat row is created on session start with `model_path`, `model_alias`, and the system prompt.
  - `--resume <id|last>` loads a previously-saved chat by id or grabs the most recently-updated one. The full message history is replayed into the in-memory `history` vector before the main loop, so the first turn after resume picks up where the previous session ended. If `-m` is omitted, the model path is taken from the saved chat row.
  - `--list` (print-and-exit) shows recently-active chats with message counts and model aliases. No model load required.
  - `--search QUERY` (print-and-exit) runs FTS5 over `messages_fts` with `[word]`-marked snippet highlights. No model load required.
  - `--db <path>` overrides the default DB location for any of the above; otherwise `$CHIMERA_DB` or the XDG default applies.

  Slash-command interactions in persistent mode:
  - `/clear` starts a fresh chat row instead of wiping the active one. The old chat stays in the DB; you can find it via `--list`.
  - `/regen` deletes the trailing assistant message(s) from the DB so the next attempt replaces them cleanly.

  Implementation:
  - New `src/chimera/chimera_chat_store.{h,cpp}` is the SQL layer: `create_chat`, `append_message`, `delete_last_message`, `load_chat`, `load_messages`, `list_chats`, `latest_chat`, `set_chat_title`, `touch_chat`, `search_messages`. Each multi-statement write is wrapped in a transaction.
  - `chat_sample_loop` gained an optional `std::string * out_reasoning` output parameter so reasoning can be persisted; existing callers that don't pass it are unaffected.
  - `chimera chat -m` is no longer marked `required()`; the print-and-exit and `--resume` paths fill the gap with their own validation.
  - `scripts/test.sh` exercises `--persist` → `--list` → `--search` in one go with a unique-token query.

  Phase 3 scope limits (also listed in `doc/dev/sqlite.md` § 11):
  - **Ctrl-C mid-stream**: interrupted assistant turns are not saved. The DB stays consistent (no partial-message row), but the in-flight text is lost.
  - **Media on resume**: attached image / audio paths are serialized into `messages.media_json` for forensics, but `--resume` does not auto-reattach them. The replayed conversation has the text
    - media-marker placeholders; the model sees only the text.
  - **Cross-model resume**: warned, not blocked. If you resume a chat under a different model than it was saved with, a `note:` line prints and the session continues anyway. The chat template is taken from the *new* model.
  - **Reasoning tokens count toward the next turn**: chat history stores only `content` for replay (`messages.reasoning` is for record-keeping, not for re-priming the KV cache).

- Vector store / RAG (phase 2 of `doc/dev/sqlite.md`). The primary driver feature for embedding SQLite — chimera can now build a personal RAG index entirely against local models, no server required. New CLI surface:
  - `chimera index create -n <name> -e <embedding.gguf>` loads the embedding model briefly to discover its `n_embd` and records a collection with that dim plus the model path. Creates a per-collection `vec_<id>` virtual table at the right size.
  - `chimera index ingest -n <name> -f <file>` (repeatable) or `-g <glob>` chunks each input with a character-window fixed-overlap chunker (defaults: 2048/256, configurable via `--chunk-chars` / `--chunk-overlap`; sentence-boundary nudge picks pleasant split points). One `Embedder` is reused across the whole batch so model load happens once, not per file. Each chunk lands as one row in `documents` and one row in the `vec_<id>` virtual table, with the same rowid so KNN results join trivially.
  - `chimera index list / stats / drop` for browsing and teardown. Drop is FK-cascading on documents + explicit DROP for the vec0 table inside one transaction.
  - `chimera search -n <name> -q <text> -k <n>` loads the collection's recorded embedding model, embeds the query, runs the vec0 KNN query, and prints top-k chunks with distance + source URI + chunk index.
  - `scripts/test.sh` ingests a three-passage corpus and confirms the top-1 hit on a targeted query — verifies end-to-end that sqlite-vec
    - the embedding model + the chunker + the SQL plumbing all line up.

  Refactors under the hood:
  - `src/chimera/chimera_embed.{h,cpp}` extracts the embedding loop from `command_embed`'s inline implementation into a reusable `Embedder` class. The CLI `embed` subcommand still passes its existing test; the new code path runs the same inference logic via the new class. Per-call `llama_memory_seq_rm(... 0, -1)` clears the previous sequence so reusing an `Embedder` across many chunks doesn't accumulate state.
  - `src/chimera/chimera_vector_store.{h,cpp}` is the SQL layer: create/drop/find/list/insert_document/search. Each function finalizes prepared statements via a tiny `StmtGuard` RAII helper, wraps multi-statement work in transactions, and binds vectors as raw float blobs (cheaper than the JSON-string form sqlite-vec also accepts).

  Phase 2 scope limits, all in `doc/dev/sqlite.md`:
  - Chunking is character-based, not token-based. The 2048-char default lands at ~500 tokens for English; bigger if your text is code-heavy. Token-based chunking deferred (§ 11.4).
  - One embedding model per collection — recorded at create time and enforced at ingest. Re-indexing with a different model means dropping and recreating; the dim mismatch is caught with a clear error.
  - No re-ingest deduplication — running `ingest` twice on the same file produces duplicate chunks. Caller-managed.

- Embedded SQLite + sqlite-vec (phase 1 of the plan in `doc/dev/sqlite.md`). No user-visible behavior change yet beyond a new diagnostic subcommand; this phase lays the rails for the vector-store + RAG features (phase 2) and the cross-session chat history feature (phase 3).
  - `scripts/manage.py` learns two new builders (`SqliteBuilder`, `SqliteVecBuilder`) that vendor the SQLite amalgamation (pinned `3.47.0`, downloaded from sqlite.org) and sqlite-vec's single-translation-unit source (pinned `v0.1.6`, fetched from asg017/sqlite-vec). Both land under `thirdparty/<name>/{include,src-aux}/` following the same pattern used for `server-http.cpp`; neither is built as a separate library — chimera compiles `sqlite3.c` and `sqlite-vec.c` directly into its own target.
  - CMakeLists adds SQLite compile-time tuning flags (`SQLITE_DQS=0`, `SQLITE_THREADSAFE=2`, `SQLITE_ENABLE_FTS5`, etc. — full list with rationale in `doc/dev/sqlite.md` § 2). `SQLITE_CORE` is set so sqlite-vec.h uses the in-process-API include path rather than the loadable-extension function-pointer routing.
  - `target_compile_options(chimera PRIVATE ${CXX_COMPILE_OPTIONS})` was gated to `$<COMPILE_LANGUAGE:CXX>` — `-std=c++17` is now applied only to .cpp sources so the .c files (sqlite3, sqlite-vec) build cleanly.
  - New `src/chimera/chimera_db.{h,cpp}` exposes a small public API: `Connection` (RAII wrapper), `default_path()` (XDG-compliant resolver honoring `$CHIMERA_DB`, `$XDG_DATA_HOME`, then `~/Library/Application Support/chimera/` on macOS, `~/.local/share/chimera/` on Linux, `%LOCALAPPDATA%\chimera\` on Windows), `open_and_migrate()` (opens with WAL + foreign keys + sqlite-vec auto-extension and walks `PRAGMA user_version` forward), `latest_schema_version()`, `list_tables()`, and version-string accessors.
  - V1 schema lands: `chats` + `messages` + `messages_fts` (FTS5 mirror with INSERT/UPDATE/DELETE triggers) + `collections` + `documents`. Per-collection `vec0` tables are deliberately *not* created in this migration — they're created on demand in phase 2 when collections are first declared via `chimera index create`.
  - New `chimera db status` subcommand opens the DB, runs migrations, and prints a human-readable summary: path, compile-time vs runtime sqlite-vec version (the runtime check actually executes `SELECT vec_version()` to verify the extension loaded — proves `sqlite3_auto_extension(sqlite3_vec_init)` worked, not just that the symbol linked), schema version, table list.
  - `chimera --version` adds two lines for sqlite + sqlite-vec versions next to the existing llama.cpp / whisper.cpp / sd.cpp lines.
  - Binary size grows ~1.5 MB (sqlite3.c) + ~60 KB (sqlite-vec.c).

- `chimera serve`: bind the "group A" server-context routes that were previously deferred. All are pre-built lambdas on `server_routes`; this is a route-registration change with no new handler logic. Newly always available, regardless of `--enable-*` flags:
  - `GET /metrics` — Prometheus-style telemetry. `params.endpoint_metrics` is forced to `true` in `build_common_params` so the route works without extra flags.
  - `GET /props` — read-only introspection (chat template, mmproj caps, default sampling params). `POST /props` deliberately *not* bound — runtime mutation of server state conflicts with chimera's "the CLI is the config" stance.
  - `POST /chat/completions` — legacy unprefixed path, same handler as `/v1/chat/completions`. Free compat for older OpenAI clients.
  - `POST /v1/messages`, `POST /v1/messages/count_tokens` — Anthropic Messages API compat. Lets the Anthropic Python SDK and claude-code-shaped clients point at chimera serve unchanged.
  - `POST /infill` — fill-in-the-middle for code models (continue.dev, llama.vim). Returns 501 on chat models without FIM tokens, which is the right behavior.
  - `POST /tokenize`, `POST /detokenize` — vocab helpers. Useful for clients that don't bundle a tokenizer (token counting before send) and for general debugging.
  - `POST /apply-template` — render the chat template against a `messages[]` array without generating. Pure debugging value.

  The "deliberately NOT bound" list in `doc/dev/server.md` § 4.4 now contains only the routes that have a real reason to be skipped: legacy non-`/v1` completion/embedding variants (redundant), `POST /props` (stateful), `POST /v1/responses` (stateful by design), `/rerank`, `/slots*`, `/lora-adapters*` (niche; bind on request).

- `chimera serve` phase 3: `POST /v1/images/{generations,edits,variations}` via stable-diffusion.cpp. New CLI flag `--enable-image <sd.gguf>` on `serve`; when supplied, an SD context (`vae_decode_only=false` so the encode path is available for img2img) is loaded alongside the LLM and the three image routes are bound. All three share a generate-and-encode core: each output `sd_image_t` is encoded to PNG via `stbi_write_png_to_func` and base64-encoded into OpenAI's `{"data":[{"b64_json":"..."}]}` envelope. Concurrent requests are serialized on a per-server mutex (SD's `generate_image` is not thread-safe on a shared context).

  **Routes:**
  - `POST /v1/images/generations` — txt2img. Application/json body with `prompt` (required), `n`, `size` (`"<W>x<H>"`), `response_format`, plus SD-specific fields (`negative_prompt`, `steps`, `cfg_scale`, `seed`, `sample_method`, `scheduler`).
  - `POST /v1/images/edits` — img2img / inpaint. Multipart upload with `image` (required) and optional `mask`; same JSON fields as `generations` plus `strength`.
  - `POST /v1/images/variations` — img2img with no prompt. Multipart upload with `image` only.

  **Architecture note**: like phase 2 (audio), this does NOT bind a server-context handler — there isn't one for image generation upstream. We register chimera-owned handlers on the shared `server_http_context` alongside the LLM routes, the same way llama-server registers its own non-LLM routes (CORS proxy, /tools, GCP compat).

  **Phase 3 scope limits**:
  - **response_format**: `b64_json` only (the default). `url` returns HTTP 400 with a clear message — chimera serve has no static-file backend to host generated PNGs from.
  - **Request fields ignored**: `model` (we have one loaded, no selection), `user` (OpenAI uses this for abuse tracking; we don't), `quality`/`style` (DALL-E-3 fields without obvious SD analogs).
  - **No prompt streaming**: SD step-by-step progress goes to stderr (as in the CLI), not to the HTTP client. The OpenAI spec doesn't define SSE for images either.

  **Refactor under the hood**: same shape as the phase-2 whisper refactor. `src/chimera/chimera_sd.h` (new) exposes a public API in the `chimera_sd::` namespace: `load_model`, `decode_image_bytes`, `decode_image_file`, `encode_png`, `save_png_file`, plus `GenerateRequest`/`PixelImage` types and the `generate` entry point. `chimera_sd.cpp` was rewritten as the implementation of those helpers; `command_sd` is now a thin caller of `chimera_sd::generate` that adds the CLI-only conveniences (numbered output paths, stderr progress). CLI behavior is preserved — both `sd` and `sd img2img round-trip` tests still pass.

  **Implementation footprint**: ~350 LOC in `chimera_serve.cpp` for the three handlers + JSON-field coercion + base64 + the SRT/VTT formatters that landed in phase 2; ~120 LOC in `chimera_sd.h`; `chimera_sd.cpp` reorganized in place.

- `chimera serve` phase 2: `POST /v1/audio/transcriptions` via whisper.cpp. New CLI flag `--enable-audio <whisper.gguf>` on `serve`; when supplied, a whisper context is loaded alongside the LLM and the `/v1/audio/transcriptions` route is bound. The handler reads multipart uploads (`file` field) and text form fields (`language`, `prompt`, `response_format`, plus OpenAI's other documented fields, ignored when not yet meaningful — see in-code comment). Supports response formats `json` (default; `{"text": "..."}`), `text` (raw text/plain), `verbose_json` (full structure: task/language/duration/segments[]), `srt`, and `vtt`. Concurrent transcription requests are serialized on a per-server mutex because `whisper_full` mutates the context state.

  **Architecture note**: we do NOT bind server-context's own `post_transcriptions_oai` handler. That route in llama-server feeds audio through mtmd's audio mmproj — a fundamentally different pipeline (LLM-with-audio-tokens vs dedicated ASR). chimera's handler uses whisper.cpp directly, same engine as the `chimera whisper` CLI subcommand.

  **Phase 2 scope limits**:
  - **Audio formats**: WAV (RIFF/WAVE) only. OpenAI's spec also accepts mp3, mp4, mpeg, mpga, m4a, webm — those need a real decoder (libsndfile + libavcodec, or single-header dr_mp3/dr_flac). Requests with non-WAV uploads return HTTP 415 with a descriptive error.
  - **Request fields ignored**: `model` (we have one loaded, no selection), `temperature` (whisper doesn't expose a comparable knob), `timestamp_granularities[]` (segment-level timing is always returned in `verbose_json`; word-level would require enabling `params.token_timestamps`).
  - **Translation route**: `POST /v1/audio/translations` is not yet bound. Trivial follow-up — set `translate=true` in the `TranscribeRequest`. Currently the only way to translate is via the CLI's `chimera whisper --translate`.

  **Refactor under the hood**: `chimera_whisper.cpp` now exposes a public API in the new `chimera_whisper.h` (`load_model`, `load_wav_file`, `load_wav_bytes`, `resample_linear`, `transcribe`, `format_timestamp_10ms`) inside the `chimera_whisper::` namespace. `command_whisper` was rewritten as a thin caller of those helpers, preserving its existing streaming output behavior under `--timestamps`. While extracting the transcribe loop I noticed and fixed a latent bug: setting `whisper_full_params.detect_language = true` puts whisper into a language-id-only mode that returns without transcribing — `language="auto"` now correctly resolves via `params.language = "auto"` alone. The bug was unreachable from the CLI's default `language="en"`, so no existing test caught it; the HTTP handler exercises it by default.

- `chimera serve`: OpenAI-compatible HTTP server, phase 1 (LLM only). Links llama.cpp's `server-context` STATIC library — the same engine that powers llama-server — and exposes a curated subset of its routes through the vendored cpp-httplib (via `server_http_context` from llama-server's `server-http.{cpp,h}`, compiled directly into the chimera target since upstream does not ship it as a separate library). New CLI flags on `serve`: `-m/--model` (required), `--mmproj`, `--host`, `--port`, `-c/--ctx-size`, `-b/--batch-size`, `--ubatch-size`, `-t/--threads`, `--gpu-layers`, `--parallel`, `--api-key`, `--embeddings`.

  **Exposed routes:**
  - `GET /health`, `GET /v1/health` — liveness probe
  - `GET /v1/models` — list loaded model + aliases
  - `POST /v1/chat/completions` — Chat Completions, streaming + non-streaming SSE
  - `POST /v1/completions` — legacy OpenAI text completions
  - `POST /v1/embeddings` — embeddings (requires `--embeddings`)

  **Deliberately NOT exposed in phase 1** (all live on `server_routes` and could be enabled with a one-line `ctx_http.post(...)` call — the omission is a scope choice, not a missing capability): `GET /metrics`, `GET/POST /props`, `POST /completion`/`/completions` (legacy non-/v1 variants), `POST /chat/completions` (non-/v1), `POST /v1/responses`, `POST /v1/audio/transcriptions`, `POST /v1/messages` + `/count_tokens` (Anthropic compat), `POST /infill`, `POST /embedding`/`/embeddings` (non- /v1 variants), `POST /rerank` + `/v1/rerank`, `POST /tokenize` + `/detokenize`, `POST /apply-template`, `GET/POST /slots*`, `GET/POST /lora-adapters`. Server-mode features also skipped: router/multi-model mode, built-in tools (EXPERIMENTAL upstream), MCP CORS proxy (EXPERIMENTAL), GCP/Vertex AI compat, embedded Web UI (manage.py passes `LLAMA_BUILD_WEBUI=OFF` to skip baking the ~11 MB asset bundle), child-server sleeping notifications, SSL/TLS (run behind a reverse proxy).

  **Phase 2/3 follow-ups** (additive — won't refactor this code): `--enable-audio <whisper.gguf>` will add `POST /v1/audio/transcriptions` wired to chimera's whisper.cpp machinery; `--enable-image <sd.gguf>` will add `POST /v1/images/{generations,edits,variations}` wired to chimera's sd.cpp machinery. Both register on the same `server_http_context` alongside the LLM routes — same pattern llama-server uses for its own non-LLM endpoints (CORS proxy, /tools, GCP compat).

  **Build-system changes** to support this:
  - `scripts/manage.py` now passes `LLAMA_BUILD_SERVER=ON`, `LLAMA_BUILD_WEBUI=OFF` so the `server-context` static lib target is defined without baking the Web UI. Builds the new `server-context` + `cpp-httplib` targets, copies `libserver-context.a` + `libcpp-httplib.a` to `thirdparty/llama.cpp/lib/`, copies `tools/server/server-*.h` headers to `thirdparty/llama.cpp/include/`, copies cpp-httplib's single header into `thirdparty/llama.cpp/include/cpp-httplib/` (server-http.cpp includes it as `<cpp-httplib/httplib.h>`), and stashes `server-http.cpp` under a new `thirdparty/llama.cpp/src-aux/` directory that chimera's CMake compiles into its own target.
  - Top-level `CMakeLists.txt` adds `find_package(OpenSSL REQUIRED)` (cpp-httplib is built with TLS support) and links `OpenSSL::SSL OpenSSL::Crypto`. On macOS the Security and CoreFoundation frameworks are linked alongside (cpp-httplib reads trust anchors from the system keychain via `SecCertificateCopyData` / `SecTrustCopyAnchorCertificates`).
  - `src/chimera/llama_build_info_shim.cpp` gains `llama_build_info()` (returns `"chimera"`); previously only the `llama_build_number` / `llama_commit` / `llama_compiler` / `llama_build_target` symbols were stubbed. server-context and llama-common both reference it.

## [0.1.1]

### Added

- `chat`: slash commands, multimodal input, tab completion, color, and a background spinner during model load. New commands: `/help` (lists available commands), `/regen` (drops trailing assistant turns and re-samples), `/clear` (resets history + KV + attached media), `/read <file>` and `/glob <pattern>` (attach text to the next user message), and `/image <file>` / `/audio <file>` (attach media when `--mmproj` is provided). `/exit` and `/quit` remain. Banner at startup prints `build`, `model`, and `modalities`, then a hint pointing at `/help`.
- `chat`: tab completion via linenoise's completion callback. Completes the slash-command word; for `/read`, `/glob`, `/image`, `/audio` it falls through to filesystem completion against the partial path. `/image` and `/audio` are only listed and completed when the loaded mmproj advertises the corresponding modality (`mtmd_support_vision` / `mtmd_support_audio`).
- `chat`: multimodal turns. `--mmproj <gguf>` enables `/image` and `/audio`; pending media markers (one `mtmd_default_marker()` per attachment) are inserted before the next user line. Once any media is attached, the loop switches from the text-only KV-prefix-reuse path to a "rebuild every turn" mtmd path (the entire templated conversation is re-tokenized as `mtmd_input_chunks` and re-evaluated via `mtmd_helper_eval_chunks` on each turn — correct but O(history)).
- `chat`: ANSI color via the new [rang](https://github.com/agauniyal/rang) single-header dep at `thirdparty/rang.hpp`. Controlled by `--color {auto,always,never}` (default `auto`, falls through to `rang::setControlMode`). Concrete colors are routed through a semantic-tag layer (`enum class Sem { Reset, User, Cmd, Think, Stats, Info, Err }` plus one `operator<<` switch), so re-skinning chat is a single-site edit. The `>` prompt and user-typed input render green (SGR emitted around `linenoise_read`, not inside the prompt string — ANSI bytes in the prompt break linenoise's `utf8_str_width` math and corrupt the cursor under multi-line edits), `/help` references are cyan, info notices dim, errors bold red. Per-turn `[ Prompt: X t/s | Generation: Y t/s ]` line in magenta, timed with `std::chrono::steady_clock` around the prompt decode and the sample loop.
- `chat`: thinking-text rendered grey. Replies stream through `common_chat_parse` with `reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK`; after each sampled token the running text is re-parsed and diffed via `common_chat_msg_diff::compute_diffs`, so `reasoning_content_delta` (text inside `<think>...</think>`) prints grey while `content_delta` prints in the default color. Only the content portion is stored in chat history — the next turn's templating does not reinject the model's prior thinking. Matches llama-cli's behavior.
- `chat`: background spinner on stderr during `load_llama_model` and optional `mtmd_init_from_file`. Auto-disables when stderr is not a TTY so piped logs stay clean.
- Optional [linenoise](https://github.com/shakfu/linenoise) integration for `chat`. When `liblinenoise.a` is present under `thirdparty/`, interactive sessions get readline-style line editing, history (↑/↓, `Ctrl-R`), and basic editing keys (`Ctrl-A`/`Ctrl-E`/word jumps). History persists at `$CHIMERA_HISTORY` (override) or `$HOME/.chimera_chat_history`. Engaged only on a TTY — piped/redirected stdin falls back to `getline`, so scripts and the test suite are unaffected. Controlled by CMake option `CHIMERA_LINENOISE` (`AUTO` / `ON` / `OFF`; default `AUTO`); when ON and the lib is missing, configure fails with a pointer to `manage.py build -L`. Linenoise source is pinned at `master` (commit `06552a1de6` at integration time).
- `gen` multimodal input via `--mmproj <path>` + `--image <path>` (repeatable). Uses `mtmd_init_from_file` + `mtmd_helper_eval_chunks` to evaluate text + image chunks into the llama context, then runs the existing sample loop. Auto-prepends the default media marker once per `--image` if the user did not place it themselves; auto-wraps the prompt in the model's chat template (VL models are typically instruct-tuned and stall without it). The vision encoder runs on the default backend (Metal on macOS) -- ~16 s end-to-end for a 512x512 image on gemma-4-E4B + its mmproj.
- `chat`: persistent KV cache across turns. The llama_context and sampler are now built once at session start; each turn re-templates the conversation, finds the longest common prefix against what's already resident in the KV cache, rewinds via `llama_memory_seq_rm`, and only decodes the tail. The previous implementation rebuilt the context per turn, paying full prompt re-decoding cost each time.
- `sd` img2img and inpainting via stable-diffusion.cpp's existing `init_image` / `mask_image` fields. New flags: `--init-image <path>`, `--mask-image <path>` (requires --init-image), `--strength` (default 0.75). When `--init-image` is supplied, the SD context is built with `vae_decode_only=false` so the encode path is available. Image dimensions must match `-W,-H` (no internal resizing). Verified via a test that round-trips a freshly-generated image through img2img.
- `make install` / `make uninstall` targets honoring `PREFIX` (default `/usr/local`) and `DESTDIR` (for staged packaging). `make rebuild` added as a `--target chimera` shortcut that skips `make deps`.
- CI workflow (`.github/workflows/ci.yml`): builds + smokes on `macos-14` (arm64, Metal), `ubuntu-latest` (x86_64, CPU), and `windows-latest` (x86_64, MSVC; marked `continue-on-error: true` while the path is shaken out) on every push and PR to `main`. Uploads per-platform binaries + `.sha256` companion files as workflow artifacts. Caches `thirdparty/` keyed on `scripts/manage.py` + top-level CMake files. Default shell is bash on every leg so the Makefile / `scripts/test.sh` work uniformly via git-bash on Windows.
- Release workflow (`.github/workflows/release.yml`): triggered on `v*` tags (or manual `workflow_dispatch`). Rebuilds the same matrix and attaches `chimera-macos-arm64`, `chimera-linux-x86_64`, plus `.sha256` checksums to a GitHub Release.
- `tokenize` subcommand: emit token ids for a prompt (one per line, or `--pieces` for `id<TAB>piece` rows). Reads prompt from `-p`, `-f <file>`, or `-f -` (stdin). Useful for debugging vocab and template behavior without running generation.
- `embed` subcommand: emit a single embedding vector (space-separated floats) for a prompt via a GGUF embedding model. Options: `--pooling` (mean/cls/last/none, default mean), `--no-normalize`, `-c/--ctx-size`, `-b/--batch-size`, `-t/--threads`, `--gpu-layers`. Verified against `bge-small-en-v1.5-q8_0.gguf`.
- `-f, --prompt-file <path>` on `gen` and `tokenize` / `embed`. Reads the prompt from a file; `-` reads stdin. Mutually exclusive with `-p`.
- `--system-prompt-file <path>` on `chat`. Reads the system prompt from a file; mutually exclusive with `--system`.
- Structured exit codes (`ExitCode` enum + `ChimeraError`): `1` runtime, `2` bad input, `3` model-load failure, `4` generation failure. CLI11 parse errors still return CLI11's own codes (>= 100). Documented at the top of `chimera.h`.
- Whisper streaming: each finalized segment is printed as whisper.cpp produces it (via `new_segment_callback` + `whisper_full_*_from_state` accessors), instead of buffering until `whisper_full` returns.
- SD progress callback: prints `sd: step N/M` to stderr (one carriage return per update, newline on completion). Stdout still receives only the produced PNG paths, so pipelines stay clean.
- `make test` / `make smoke` targets backed by `scripts/test.sh`. Smoke tier exercises `--version`, `--help` on the root and every subcommand, and confirms that `gen` without `-m` exits non-zero. End-to-end tier runs `gen` (Llama-3.2-1B), `whisper` (ggml-base.en + whisper.cpp's bundled `jfk.wav`), and `sd` (sd_xl_turbo) when those model files are present under `models/`; missing models are reported as SKIP, not FAIL.
- `REVIEW.md`: architecture / feature / usability / best-practices review of the 0.1.0 baseline.

### Fixed

- `whisper` defaulted `language = nullptr` + `detect_language = true` when `-l` was omitted, which sometimes mis-detected English-only `.en` models as Azerbaijani (p < 0.02) and produced empty output. Now leave whisper.cpp's default (`"en"`) in place unless the user passes `--language auto` or an explicit code.
- `whisper` crashed with `error: vector` when `--threads` was left at its default of `-1`. The value was forwarded to `params.n_threads`, which whisper.cpp then cast to `size_t` to size an internal `std::vector`, triggering `std::length_error("vector")`. Now leave `params.n_threads` at `whisper_full_default_params`'s value unless the user passed a positive override.

### Changed

- `stb_impl.cpp` no longer defines `STB_IMAGE_IMPLEMENTATION`. Reason: libmtmd's `mtmd-helper.o` ships its own non-static `stbi_load`, which would duplicate-symbol on link the moment any `mtmd_helper_*` is referenced. `chimera_sd.cpp`'s `stbi_load` / `stbi_image_free` calls now resolve against libmtmd's copy (libmtmd is unconditionally linked for the `gen` mtmd path). `stb_image_write` is still ours.
- `fail()` and `trim()` were duplicated across `chimera.cpp`, `chimera_whisper.cpp`, and `chimera_sd.cpp`. Pulled into `chimera.h` as inline helpers so all three TUs share one definition. The helpers don't pull in `ggml.h`, so the three-TU isolation is preserved.
- `scripts/manage.py` trimmed of cyllama-specific code (wheel building, Cython artifact cleanup, `profile` / `bench` / `bump` / `bins` / `check-vendor` / `fix-macos-vulkan-wheel` / `write-build-config` / `status` / `test` subcommands, dynamic-wheel `build_shared` / `download_release` / macOS dylib rpath sanitization / MSVC import-lib generation, dynamic-lib path machinery, `pip_install` / `apt_install` / `brew_install` helpers, `STABLE_BUILD` env split). ~3170 -> ~1210 lines. Retained subcommands: `build`, `info`, `clean`, `download`. `-D/--deps-only` kept as a no-op for Makefile compatibility.
- `--help` output: compact spacing (short + long flags packed together via `long_option_alignment_ratio(0.0f)`; explicit `usage()` string and a `CompactFormatter` that trims `make_usage`'s trailing `"\n\n"` to `"\n"` so section breaks are single blank lines).
- Top-level description tightened to `chimera - {llama,whisper,stable-diffusion}.cpp multitool`.

## [0.1.0]

### Added

- Initial repository, extracted from [cyllama](https://github.com/shakfu/cyllama).
- Static multitool executable bundling llama.cpp, whisper.cpp, and stable-diffusion.cpp against a single shared ggml backend set.
- Subcommands: `gen` (one-shot completion), `chat` (interactive), `whisper` (WAV transcription), `sd` (text-to-image).
- Top-level `-v,--verbose` flag; native backend logging silenced by default.
- Three-TU layout (`chimera.cpp` / `chimera_whisper.cpp` / `chimera_sd.cpp`) to isolate the colliding `ggml.h` headers shipped by llama.cpp and whisper.cpp.
- Late `llama_backend_init()`: deferred until after CLI parsing so `--help` and parse errors do not trigger `ggml_load_backends`.
- `scripts/manage.py` build driver (forked from cyllama, trimmed of Python-binding-specific code paths).
- `make deps` / `make build` / `make clean` / `make reset` wrappers.
- Verified end-to-end on macOS arm64 with Metal backend (Llama-3.2-1B Q8_0 model).

### Known issues

- Only macOS arm64 + Metal is verified. Linux, Windows, and non-Metal backends are believed to work via the inherited cyllama build matrix but have not been re-validated post-split.
- `whisper` and `sd` subcommands build cleanly but have not been exercised end-to-end in this repo yet.
