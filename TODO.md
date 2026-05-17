# TODO


## Validation

- [ ] CI matrix for Vulkan + CUDA on Linux x86_64 (currently CPU only).
- [ ] Validate non-Metal macOS backends (Vulkan, CPU-only).
- [ ] Promote the Windows MSVC leg out of `experimental: true` once it
      passes consistently across a release or two.

## Build / packaging

- [ ] Homebrew tap (or formula PR to homebrew-core once the project is
      stable enough).

## Server — deferred routes / features

Longer-term:

- [ ] Web chat UI — opt-in only, **not** in the default release.
      Two independent variants to wire up; users pick one (or neither)
      at build time. Mirrors the `CHIMERA_LINENOISE=AUTO|ON|OFF`
      pattern.

      Variant A — **embedded webui** (`CHIMERA_WEBUI_EMBED=ON`):
      re-enable `LLAMA_BUILD_WEBUI=ON` in `manage.py`, run upstream's
      `xxd.cmake` machinery to bake the built bundle into the binary
      as a C header, bind a `GET /` handler that serves it. Binary
      grows ~11 MB. Pro: single artifact, drop-and-run. Con: UI is
      pinned to whatever chimera version was cut; updating the UI
      requires a chimera rebuild.

      Variant B — **public-path** (`CHIMERA_WEBUI_PATH=ON`): compile in
      a static-file handler keyed off a new `chimera serve
      --public-path <dir>` flag. Webui bundle ships as a separate
      tarball (or the user points at any directory). Pro: UI lifecycle
      decoupled from chimera; opens a door to shipping a
      chimera-specific UI later that surfaces RAG mode,
      `X-Chimera-Chat-Id`, etc. Con: two artifacts to ship, runtime
      flag required to enable.

      Both need: `manage.py build --webui` to drive the upstream JS
      toolchain (adds a Node dependency at build time, opt-in only),
      and a decision on whether the route is auth-gated by `--api-key`
      like the JSON routes are.
- [ ] SSE for image generation progress on `/v1/images/*`. Pick a
      client convention; OpenAI's spec doesn't define one. SD already
      reports step-by-step progress to the callback we currently route
      to stderr.
- [ ] Split `chimera_serve.cpp` into modality-specific files
      (`serve_audio.cpp`, `serve_images.cpp`, ...) when it crosses
      ~1000 LOC. Currently ~600.

## RAG / SQLite

Medium-term:

- [ ] `chimera db backup` / `chimera db vacuum` subcommands. Backup
      must include the WAL + SHM siblings or use `VACUUM INTO`.

Longer-term:

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

## Out of scope (wontfix)

Recorded so the same proposals aren't re-litigated from scratch. Each
was considered, weighed against chimera's "single static busybox-style
binary" identity, and consciously rejected. Reopen the discussion only
if a concrete user request shows up.

- **`POST /props`** — runtime mutation of server props. Conflicts with
  "CLI is the config." Configuration belongs in flags and the DB, not
  a write API that lets clients reshape the server out from under
  other clients. Wontfix.
- **Multi-tenancy / `is_router_server`** — multiple LLMs in one
  process, routed by the request's `model` field. Effectively a
  rewrite of `command_serve` for a deployment shape chimera isn't
  aimed at (one process = one model is core to the busybox identity).
  Users who need multi-model routing should run one chimera per model
  behind a real reverse proxy. Wontfix.
- **HTTPS direct serving** (`--ssl-cert-file` / `--ssl-key-file`) —
  cpp-httplib supports it but a reverse proxy (nginx, caddy, Cloudflare,
  etc.) is the right place to terminate TLS for any deployment that
  cares. Adding it here means rebuilding chimera every time a cert
  rotates. Wontfix.
- **Auth beyond `--api-key`** — JWT, per-key rate limiting,
  multi-tenant auth. Same reasoning as HTTPS: reverse proxies and API
  gateways solve this an order of magnitude better than chimera can.
  Wontfix.
