# TODO


## Validation

- [~] CI matrix for Vulkan + CUDA on Linux x86_64. Compile-only legs
      exist in `.github/workflows/ci-gpu.yml` (workflow_dispatch-only):
      they build each backend and assert its ggml registration symbol
      linked, but do not execute kernels. Remaining: (a) promote to a
      nightly `schedule:` once stable, (b) real runtime coverage needs a
      self-hosted / paid GPU runner (catches silent runtime backend
      non-registration, which compile-only is blind to).
- [ ] Validate non-Metal macOS backends (Vulkan, CPU-only).
- [ ] Promote the Windows MSVC leg out of `experimental: true` once it
      passes consistently across a release or two.
- [ ] `combine_archives.py` Windows path support. The combined static-lib
      output (`libchimera*.a`/`.lib`) has never built on Windows: the
      inventory looks for `build/llama.cpp/build/src/libllama.lib` but MSVC
      emits `.../src/Release/llama.lib` (no `lib` prefix, extra per-config
      `Release/` subdir). collect() needs to insert the config subdir and
      drop the `lib` prefix on the `windows` target. Until then the CI
      `combined-archive link contract` + `python bindings` legs are gated
      to Linux + macOS. Needs an MSVC box to validate (untestable from
      a Unix host).

## Build / packaging

- [ ] Homebrew tap (or formula PR to homebrew-core once the project is
      stable enough).

## Server — deferred routes / features

Longer-term:

- [ ] SSE for image generation progress on `/v1/images/*`. Pick a
      client convention; OpenAI's spec doesn't define one. SD already
      reports step-by-step progress to the callback we currently route
      to stderr.

## RAG / SQLite

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
  [`docs/dev/server-router-mode.md`](docs/dev/server-router-mode.md)
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
