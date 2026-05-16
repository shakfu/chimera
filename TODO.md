# TODO

Forward-looking work for chimera. Loosely prioritized within each section.
Detailed rationale and design discussion lives in `doc/dev/server.md`
(§ 6 issues, § 7 watch-fors, § 8 future work) and `doc/dev/sqlite.md`
(§ 9 phase 6+, § 10 risks, § 11 open design questions).

## Validation

- [ ] CI matrix for Vulkan + CUDA on Linux x86_64 (currently CPU only).
- [ ] Validate non-Metal macOS backends (Vulkan, CPU-only).
- [ ] Promote the Windows MSVC leg out of `experimental: true` once it
      passes consistently across a release or two.
- [ ] `make test-db-migrate` target that fixtures an older-version
      SQLite DB and asserts the migration to current succeeds. Pre-empts
      schema-migration regressions across version bumps.
- [ ] Bump-process check: every `LLAMACPP_VERSION` change should diff
      `tools/server/server-context.h` + `tools/server/server-http.h`
      against the prior pin. Those types are not part of upstream's
      public API and shift with internal refactors.

## Build / packaging

- [ ] Trim remaining cyllama-specific code from `scripts/manage.py`
      (wheel builder, profile/benchmark commands, lingering `cyllama`
      references in log lines).
- [ ] Homebrew tap (or formula PR to homebrew-core once the project is
      stable enough).

## Server — deferred routes / features

Near-term:

- [ ] Non-WAV audio in `/v1/audio/transcriptions`. mp3 / m4a / mp4 /
      mpeg / webm all return 415 today. Options: single-header
      decoders (`dr_mp3.h`, `dr_flac.h`) for the common cases, or
      libsndfile + libavcodec / FFmpeg for full coverage.
- [ ] Better SD error messages: capture `ggml_log` into a small ring
      buffer and pipe the last few lines into the HTTP response body
      when generation fails. Today users see a generic 500 even when
      the underlying assertion was descriptive ("buft failed", etc.).
      Same idea covers the SDXL-Turbo `cfg_scale=1.0`+`euler` crash.

Medium-term:

- [ ] `POST /v1/embeddings` without `--embeddings`: add
      `--enable-embeddings <bge.gguf>` to load a small embedding model
      alongside the LLM, same opt-in pattern as `--enable-audio` and
      `--enable-image`. ~80 LOC.
- [ ] `--reranking <model.gguf>` + bind `POST /v1/rerank`. Cross-encoder
      reranking is the natural follow-up to RAG (top-50 from vec search
      → rerank → top-5 → LLM).
- [ ] Bind `POST /props` (mutating server props at runtime). Currently
      only `GET /props` is bound; the write side conflicts with
      "CLI is the config" but real deployments may want it.
- [ ] Bind `GET /slots` + `POST /slots/:id` (KV-cache snapshots) and
      `GET/POST /lora-adapters` (LoRA hot-swap). Both are one-line
      `ctx_http.post(...)` calls each; the work is the UX around them
      (where do slot files live, how is LoRA hotswap parameterized).

Longer-term:

- [ ] Web chat UI: either re-enable `LLAMA_BUILD_WEBUI=ON` in
      `manage.py` (binary grows ~11 MB; uses upstream's `xxd.cmake`
      machinery) or ship assets next to the binary + add a
      `--public-path` flag.
- [ ] SSE for image generation progress on `/v1/images/*`. Pick a
      client convention; OpenAI's spec doesn't define one. SD already
      reports step-by-step progress to the callback we currently route
      to stderr.
- [ ] Multi-tenancy: multiple LLMs simultaneously, route by request's
      `model` field. This is the `is_router_server` path in upstream's
      `server.cpp`; effectively a rewrite of `command_serve`. Also
      unlocks multi-model embedding/rerank/audio/SD configurations.
- [ ] HTTPS direct serving via `--ssl-cert-file` / `--ssl-key-file`.
      cpp-httplib supports it; needs to be threaded through
      `server_http_context::init`. Most deployments will continue to
      front chimera with a reverse proxy.
- [ ] Authentication beyond `--api-key`: JWT or per-key rate limiting
      for multi-tenant deployments.
- [ ] Split `chimera_serve.cpp` into modality-specific files
      (`serve_audio.cpp`, `serve_images.cpp`, ...) when it crosses
      ~1000 LOC. Currently ~600.

## RAG / SQLite

Near-term:

- [ ] Token-based chunking in `chimera index ingest` (currently
      character-window with a sentence-boundary nudge). Token-based
      gives accurate per-chunk sizes; costs a `llama_tokenize` call per
      chunk.
- [ ] `--distance` flag on `chimera index create` to pick L2 / cosine /
      inner-product. Default cosine is right for normalized embeddings;
      only override needs the flag.
- [ ] Per-collection chunk-size overrides recorded in the collection
      row, not just per-CLI-call.

Medium-term:

- [ ] Embedding cache: `embed(text) -> vector` memoized to a small KV
      table keyed by `sha256(text) || model_id`. Speeds up ingestion
      of partially-updated corpora and repeated query embedding.
- [ ] Smarter chunking (sentence-aware, semantic boundaries).
- [ ] Hybrid search: combine FTS5 + sqlite-vec with reciprocal-rank
      fusion or weighted scoring.
- [ ] `chimera db backup` / `chimera db vacuum` subcommands. Backup
      must include the WAL + SHM siblings or use `VACUUM INTO`.
- [ ] Save partial assistant turns on `Ctrl-C` in `chimera chat
      --persist`, with a `partial=1` column on `messages` so resume
      can mark them as incomplete.

Longer-term:

- [ ] `X-Chimera-Chat-Id` request header on `/v1/chat/completions` so
      multi-turn clients can persist into one chat row instead of
      duplicating per request.
- [ ] Auto-reattach media on `chimera chat --resume`. Today
      `media_json` records the paths but resume doesn't reload them.
- [ ] Streaming progress for `POST /v1/vector_stores/:name/files`
      ingestion (a 10 MB text file takes many seconds; the client
      gets no feedback until done).
- [ ] `chimera serve --enable-rag` audit table for ingest/search
      calls.

## Documentation

- [ ] Example gallery: short transcripts of each subcommand against a
      well-known small model.
- [ ] Backend matrix table: which subcommand × backend combinations
      have been tested.
- [ ] Document the model formats each subcommand accepts (and which
      common GGUF variants don't work).
- [ ] Privacy note in the user-facing docs (`doc/serve.md`,
      `doc/cheatsheet.md`) about what `--persist` and `--persist-chats`
      record. The DB lives in user-private XDG paths but this should be
      called out explicitly.
