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

- [ ] Web chat UI — **Variant A (embedded webui) shipped** as an
      experimental opt-in: configure with `-DCHIMERA_WEBUI_EMBED=ON`
      and the chimera binary serves GET / + /bundle.{js,css} from
      upstream's prebuilt bundle (~6 MB stripped, ~7 MB unstripped; no Node
      toolchain required). Disable at runtime with `serve --no-webui`.
      Caveat: the baked UI is pinned to the vendored llama.cpp version,
      so the UI lifecycle is coupled to the chimera release cadence.

      Still open:
      - Variant B — **public-path** (`CHIMERA_WEBUI_PATH=ON`): compile
        in a static-file handler keyed off a new `chimera serve
        --public-path <dir>` flag. Pro: UI lifecycle decoupled from
        chimera; opens a door to shipping a chimera-specific UI later
        that surfaces RAG mode, `X-Chimera-Chat-Id`, etc.
      - Decide whether GET / + /bundle.* should be `--api-key`-gated
        like the JSON routes are. Currently the webui routes follow
        upstream's middleware (unauth-gated when no api-key is set;
        any-bearer-token-accepting when one is). Browsers can't easily
        send custom Authorization headers on a top-level navigation.
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
  behind a real reverse proxy. Wontfix. See
  [`doc/dev/server-router-mode.md`](doc/dev/server-router-mode.md)
  for the full decision record (architecture, port cost, the
  concurrency / single-GPU analysis, when to revisit).
- **HTTPS direct serving** (`--ssl-cert-file` / `--ssl-key-file`) —
  cpp-httplib supports it but a reverse proxy (nginx, caddy, Cloudflare,
  etc.) is the right place to terminate TLS for any deployment that
  cares. Adding it here means rebuilding chimera every time a cert
  rotates. Wontfix.
- **Auth beyond `--api-key`** — JWT, per-key rate limiting,
  multi-tenant auth. Same reasoning as HTTPS: reverse proxies and API
  gateways solve this an order of magnitude better than chimera can.
  Wontfix.
