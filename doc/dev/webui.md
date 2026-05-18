# Embedded web UI — developer guide

The chimera binary can optionally bake llama.cpp's prebuilt web chat UI
into itself. This document covers how it is wired up, how to enable it
locally, and the seams that bit during implementation.

It is the maintainer's view. End-user-facing notes (the `--no-webui`
flag, what the UI is) live in [`doc/serve.md`](../serve.md). The wider
`chimera serve` implementation notes are in
[`doc/dev/server.md`](server.md); this file is the webui-specific
addendum.

> **Status — experimental.** Variant A (embedded) is the shipped path
> documented here. The `--public-path <dir>` flag also shipped (parity
> with llama-server) but the chimera-specific UI prototyped against it
> was abandoned; see [§ 6](#6-variant-b--attempted-and-dumped) for the
> post-mortem and re-open conditions.

---

## 1. The one fact that makes the wiring make sense

Upstream's webui route binding lives in `tools/server/server-http.cpp`,
gated by the compile-time `LLAMA_BUILD_WEBUI` macro and the runtime
`params.webui` boolean. Crucially, **upstream does NOT compile
`server-http.cpp` into `libserver-context.a`** — it compiles it directly
into the `llama-server` executable. So flipping `LLAMA_BUILD_WEBUI=ON`
in the llama.cpp build does nothing for chimera; we link the static
library, not the executable.

Chimera already worked around this for the non-webui parts of
server-http: `manage.py` copies `server-http.cpp` into
`thirdparty/llama.cpp/src-aux/`, and `src/chimera/CMakeLists.txt`
compiles it as part of the `chimera` target (see
`src/chimera/CMakeLists.txt` line ~52). To get the webui we extend the
same trick:

1. Stage the four prebuilt assets + `xxd.cmake` into `src-aux/webui/`.
2. When the user configures with `-DCHIMERA_WEBUI_EMBED=ON`, the chimera
   CMake runs `xxd.cmake` against each asset to produce `<asset>.hpp`
   files in the chimera build dir.
3. Those `.hpp` paths are added to the chimera target's source list, the
   build dir is added to the include path, and `LLAMA_BUILD_WEBUI` is
   defined on the chimera target so the `#ifdef`s in our local copy of
   `server-http.cpp` activate.

The route binding then happens automatically inside
`server_http_context::init(params)`, which `chimera_serve.cpp` was
already calling.

---

## 2. Enabling the build

One-shot from a clean checkout:

```
make build-with-webui
```

That target is just `make deps` followed by `cmake … -DCHIMERA_WEBUI_EMBED=ON`
and a chimera-target build. Equivalent to (and useful for understanding what
it does):

```
make deps
cmake -S . -B build -DSD_USE_VENDORED_GGML=OFF -DCMAKE_BUILD_TYPE=Release \
                   -DCHIMERA_WEBUI_EMBED=ON
make rebuild
```

The CMake option mirrors the existing `CHIMERA_LINENOISE` ON/OFF/AUTO
pattern, so you can also drive it directly:

```
# AUTO — links only if thirdparty/llama.cpp/src-aux/webui/index.html exists
cmake -S . -B build -DCHIMERA_WEBUI_EMBED=AUTO
make rebuild

# OFF (the default for `make build`)
cmake -S . -B build       # or -DCHIMERA_WEBUI_EMBED=OFF
make rebuild
```

`make deps` always stages the assets, regardless of the option — they
sit at ~7 MB on disk in the dev tree, only inflate the binary when the
option flips them in. This decouples the staging step from the binary
flip so toggling on a previously-built tree is a one-line `cmake`
reconfigure + rebuild, not a full `make deps` cycle.

**Measured binary cost** (Apple Silicon, Release build, May 2026):

| | unstripped | `strip`-ped |
|---|---:|---:|
| `CHIMERA_WEBUI_EMBED=OFF` | 34 MB | 31 MB |
| `CHIMERA_WEBUI_EMBED=ON`  | 41 MB | 37 MB |
| **net webui cost** | **+7 MB** | **+6 MB** |

The asset bytes (6.6 MB `bundle.js` + 505 KB `bundle.css` + ~7 KB
`index.html` + ~268 B `loading.html`) sit in the binary's data section
as plain `unsigned char` arrays, which `strip` cannot touch — so the
asset payload passes through ~1:1. The strip-recoverable delta (~1 MB)
is debug info on the surrounding chimera + server-http code, not on the
xxd'd assets.

Stripped is what gets shipped; the +6 MB figure is the one to quote in
user-facing material. See § 6.4 for the only viable size-reduction path
(serve the assets gzip-compressed and skip storing the uncompressed
copy in the binary).

**No Node toolchain at build time.** The bundles ship pre-built in
`build/llama.cpp/tools/server/public/`, and `xxd.cmake` is pure CMake.
Rebuilding the JS bundle (e.g. to ship a chimera-customized UI) would
require Node + the upstream `webui/` Vite project — out of scope for the
current experiment.

---

## 3. Where things live

| Path | Purpose |
|------|---------|
| `build/llama.cpp/tools/server/public/{index,bundle.{js,css},loading}.html` | Upstream-shipped prebuilt assets. |
| `build/llama.cpp/scripts/xxd.cmake` | Pure-CMake byte-array generator (no `xxd(1)` dependency). |
| `scripts/manage.py` :: `LlamaCppBuilder._copy_headers` | Stages assets + `xxd.cmake` into `src-aux/webui/` and `src-aux/xxd.cmake`. |
| `thirdparty/llama.cpp/src-aux/webui/` | Where the staged assets land (gitignored). |
| `thirdparty/llama.cpp/src-aux/xxd.cmake` | Staged copy of the helper. |
| `CMakeLists.txt` (top-level) | Defines `CHIMERA_WEBUI_EMBED` option; resolves AUTO; exports `CHIMERA_LINK_WEBUI`. |
| `src/chimera/CMakeLists.txt` | When `CHIMERA_LINK_WEBUI` is ON: runs xxd, adds `.hpp` to sources, sets include dir, defines `LLAMA_BUILD_WEBUI`. |
| `src/chimera/chimera_serve.cpp` | Sets `params.webui = opts.webui`. The actual route binding is in our locally-compiled `server-http.cpp`. |
| `src/chimera/chimera.h` :: `ServeOptions::webui` | Default true; flipped by `--no-webui`. |
| `src/chimera/chimera.cpp` | `--no-webui` CLI flag declaration. |
| `build/<chimera-build>/src/chimera/<asset>.hpp` | Generated at build time; one `unsigned char[]` + length per asset. |

---

## 4. Verifying it worked

After building with `-DCHIMERA_WEBUI_EMBED=ON`:

```
ls build/src/chimera/*.hpp
# expect: bundle.css.hpp  bundle.js.hpp  index.html.hpp  loading.html.hpp

ls -la build/chimera
# expect: ~7 MB larger than an OFF build (~6 MB after `strip`)
```

Run-time smoke:

```
PORT=8080
./build/chimera serve -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
    --host 127.0.0.1 --port $PORT &

# Wait for /health to go 200, then:
curl -o /dev/null -w "%{http_code} %{content_type} size=%{size_download}\n" \
    http://127.0.0.1:$PORT/
# expect: 200 text/html; charset=utf-8  size=~6900

curl -o /dev/null -w "%{http_code} size=%{size_download}\n" \
    http://127.0.0.1:$PORT/bundle.js
# expect: 200  size=~6.6 MB

curl -o /dev/null -w "%{http_code}\n" http://127.0.0.1:$PORT/bundle.css
# expect: 200
```

The startup banner should also include a `webui:` line. If
`--no-webui` is passed, the banner reads `built in but disabled by
--no-webui` and `GET /` returns 404.

---

## 5. Seams to watch out for

These are the things that aren't obvious from the code, ordered by how
likely they are to bite a future maintainer.

### 5.1. `LLAMA_BUILD_WEBUI` is a chimera-side flag now, not a llama.cpp flag

`scripts/manage.py` keeps `LLAMA_BUILD_WEBUI=False` in the llama.cpp
build. Resist the urge to flip it. Even ON, it would only bake the
assets into the `llama-server` executable, which chimera doesn't ship —
and would not affect `libserver-context.a`. The chimera-side wiring is
the only thing that matters.

If a future llama.cpp bump moves the route-binding block out of
`server-http.cpp` and into `libserver-context.a`, this whole scheme
collapses to "just enable upstream's flag" — at which point the chimera
CMake gymnastics in `src/chimera/CMakeLists.txt` should be deleted, not
preserved alongside it. Pin-check (`chimera_pin_check.cpp`) won't catch
that move because the binding is plain inline code, not a typed handler
field; you'll discover it when a webui-on build either double-binds
`GET /` (httplib errors) or silently stops binding it (route 404s).
Smoke test in [§ 7](#7-testing) defends against the second case.

### 5.2. The UI is pinned to the vendored llama.cpp version

The bundles in `build/llama.cpp/tools/server/public/` are
prebuilt-and-committed in upstream's tree. Updating the UI means
bumping `LLAMACPP_VERSION` in `scripts/manage.py` and rebuilding.
There is no "ship a UI patch without a chimera rebuild" story.

This is the headline trade-off of Variant A. Variant B (see § 6) would
fix it.

### 5.3. The route binding ignores `--api-key`

Upstream's webui block (`server-http.cpp` around line 320) registers
`GET /` and `GET /bundle.{js,css}` *outside* the
`middleware_validate_api_key` chain — those routes are served regardless
of whether `--api-key` is set. The XHR/fetch calls the UI makes
afterwards (`/v1/chat/completions`, etc.) do go through the middleware
and require the bearer token, which the UI prompts the user for in its
settings panel.

This is consistent with upstream's behavior and probably the right
default (a browser can't easily send `Authorization` on a top-level
navigation), but it does mean **anyone who can reach the port can load
the page** even on an auth-gated chimera. The data exposure is zero
(every actual API call is still auth-gated), but it's worth knowing
before you publish a chimera instance on a non-private network.

### 5.4. The COEP / COOP headers on `GET /`

Upstream sets `Cross-Origin-Embedder-Policy: require-corp` and
`Cross-Origin-Opener-Policy: same-origin` on the `/` response, required
by the webui's Pyodide (Python-in-browser) feature. If you load chimera
behind a reverse proxy that strips or rewrites those headers, Pyodide
breaks; non-Pyodide use is unaffected. The headers are set inside
upstream's lambda, not in chimera — we can't change them without
patching `server-http.cpp`.

### 5.5. The webui hits a few endpoints chimera *doesn't* expose

The UI was designed against the full `llama-server` route surface;
chimera exposes a curated subset. The actual API surface the webui
talks to is small — sourced from
`tools/server/webui/src/lib/constants/api-endpoints.ts` plus a grep
of `fetch()` call sites in `src/lib/services/*.ts`:

| Endpoint                            | Method      | Used by                          | Chimera? |
|-------------------------------------|-------------|----------------------------------|----------|
| `./v1/chat/completions`             | POST        | chat (streamed + non-streamed)   | ✓ bound  |
| `./v1/models`                       | GET         | model picker / model info        | ✓ bound  |
| `./props`                           | GET         | server capability detection      | ✓ bound (GET only — `POST /props` for runtime mutation is wontfix per `chimera_serve.cpp` top-of-file list) |
| `./slots` (+ optional `?model=`)    | GET         | slot status panel                | ✓ bound  |
| `/models/load`                      | POST        | router-mode model switch         | ✗ — chimera is single-model by design (wontfix) |
| `/models/unload`                    | POST        | router-mode model switch         | ✗ — same |
| `/tools`                            | GET + POST  | built-in tool plugins panel      | ✗ — upstream `--server-tools` is EXPERIMENTAL; not exposed |
| `/cors-proxy`                       | proxy       | webui's MCP client               | ✗ — upstream `--webui-mcp-proxy` is EXPERIMENTAL; not exposed |

**Net effect.** Core chat works: send messages, get streamed responses,
see the loaded model, inspect slot status. Three feature areas degrade:

1. **Model switcher / "load this model" panel.** No-op in chimera — one
   process, one model. The UI may show an empty list or an error toast;
   nothing else breaks. The "list" half (`/v1/models`) does work and
   reports the single loaded model.
2. **MCP (Model Context Protocol) integration.** The webui can talk to
   external MCP servers, but only when chimera proxies cross-origin
   requests via `/cors-proxy`. Since chimera doesn't expose that route,
   MCP either works only against same-origin servers (rare) or fails
   outright. Pyodide (Python-in-browser, which is client-side only) is
   unaffected.
3. **Built-in tool plugins panel.** The webui's tool-plugin UI talks to
   `/tools`. That endpoint isn't bound. Note: tool *calling* — the
   model emitting tool-use blocks inside chat completions — still works
   because the parsing lives in `/v1/chat/completions` server-side. The
   missing route is only for the panel that *configures* tools.

Before binding any of the missing routes in response to a user
complaint, re-read the "deliberately not exposed" list at the top of
`chimera_serve.cpp` — most of those omissions were reasoned through
(`POST /completion` legacy shape, `POST /props` mutation, router-mode,
the experimental tools + MCP plugins). Adding them piecemeal in
response to "the UI panel is empty" reports is how chimera's busybox
identity gets diluted.

### 5.6. What chimera exposes that the webui doesn't surface

The mirror-image gap: chimera binds several routes the upstream webui
has no UI for. They still work over HTTP for code clients; the chat
UI just doesn't know about them.

- **`/v1/messages` and `/v1/messages/count_tokens`** (Anthropic Messages
  API compat) — the webui speaks OpenAI shape only. Clients pointed at
  these routes directly (the Anthropic Python SDK, `claude-code`-shaped
  tools) work fine; the webui simply doesn't expose them as an option.
- **`/v1/vector_stores/*`** (RAG / vector store routes, when
  `--enable-rag` is set) — no UI surface at all. Users have to drive
  ingest + search via the CLI (`chimera index`, `chimera search`) or
  raw HTTP.
- **`X-Chimera-Chat-Id` header** (chat persistence consolidation, when
  `--persist-chats` is set) — the webui doesn't send it, so each
  message produces a new chats row instead of being grouped into one
  conversation. The DB still records everything; it just doesn't get
  the multi-turn consolidation the header was designed for. Worth
  knowing before promising "persisted chat history" in the same
  breath as "the webui works."
- **`/v1/audio/{transcriptions,translations}`** (when `--enable-audio`)
  — no UI.
- **`/v1/images/{generations,edits,variations}`** (when
  `--enable-image`) — no UI.
- **`/v1/rerank`** (when `--reranking`) — no UI.
- **`/v1/embeddings`** when chimera was started with `--enable-embeddings`
  routing to a dedicated embedding model — the webui has no embedding
  UI.
- **`/lora-adapters` POST** for hot-swap — no UI (the webui doesn't know
  about LoRAs in chimera's loaded-at-startup-then-rescaled shape).

This is the §6.1 argument in concrete form: making the chimera-specific
surface usable from a browser is **only** tractable under Variant B
(separate-bundle, `--public-path`), where we can ship a UI that knows
about these routes. Variant A pins us to upstream's pinned UI which by
definition only knows about upstream's routes.

### 5.7. cmake configure cache vs. asset regeneration

`add_custom_command(OUTPUT <asset>.hpp DEPENDS <asset> ...)` is what
triggers re-xxd when the asset content changes. If `make deps` re-stages
the assets (e.g. after a `LLAMACPP_VERSION` bump), the `.hpp` files
should regenerate on the next chimera build automatically. They have not
been observed to go stale, but if you see the bundle.js in the binary
not matching the bundle.js on disk after a llama.cpp bump, suspect this
mechanism and `rm -rf build/src/chimera/*.hpp` to force.

### 5.8. Asset size and download timing

`bundle.js` is ~6.6 MB uncompressed. cpp-httplib serves it
uncompressed (no gzip). On a slow network the first page load is
multi-second; the browser caches aggressively after that, but each
distinct chimera version invalidates the cache (no `Cache-Control:
immutable` + content-hashed filenames, just the static path). Not a
problem on localhost; might be worth knowing if a user reports "the UI
takes forever to load over SSH tunnel."

§ 6.4 sketches the gzip-at-build-time fix that would address both the
on-disk and over-the-wire size (same change, two wins).

---

## 6. Variant B — attempted and dumped

The `--public-path <dir>` flag shipped (see CHANGELOG `[Unreleased]`),
giving chimera true API parity with llama-server on the static-serve
side. A chimera-specific UI built against that flag was prototyped and
dumped. This section records what was tried, why it was abandoned, and
when (if ever) to revisit.

### 6.1. What was tried

Stack: vanilla JS + Alpine 3.x + htmx 2.x + Pico 2 CSS + marked +
highlight.js + KaTeX, with Inter / JetBrains Mono self-hosted webfonts.
~470 lines of JS in a single `app.js`, ~290 lines of HTML, ~180 lines of
CSS. Vendored via a `scripts/fetch_webui_vendor.sh` that pulled ~50 pinned
files (~1.4 MB on disk). Distribution: separate tarball mounted via
`chimera serve --public-path <dir>` so chimera's default binary stayed
untouched. Scope: chat with `X-Chimera-Chat-Id` round-trip, chat history
sidebar (list + FTS5 search via the new `/v1/chats*` endpoints), vector
store CRUD + ingest + search, markdown + code highlight + math rendering,
settings (endpoint URL + API key + sampling defaults) in localStorage.

Make targets `webui-vendor` and `webui-serve` wrapped the workflow. The
HTML/CSS/JS code lived in `webui/`. None of it ships anymore — see git
history at the commit that introduced and the commit that removed.

### 6.2. Why it was dumped

Five reasons, ordered by weight:

1. **Identity mismatch.** Chimera's defining property is "one static
   busybox-style binary that does many things, no extra install steps."
   Owning a frontend is a different skill set on a different release
   cadence with a different bug surface. The friction showed up fast:
   the first manual smoke test of the UI surfaced bugs (sub-stores
   calling `this.$nextTick` outside Alpine's scope; marked 14's removed
   options; a MutationObserver feedback loop with KaTeX) that needed
   browser-driven iteration to confirm fixed. That iteration loop is
   not where chimera's effort should go.
2. **Existing UIs already work.** chimera is OpenAI-compatible. Open
   WebUI, LibreChat, Jan, LM Studio, and the upstream llama.cpp webui
   (now opt-in via `CHIMERA_WEBUI_EMBED=ON` per § 1–5 of this doc) all
   work today. The chimera-specific surfacing — RAG panel, persisted
   chat browsing, `X-Chimera-Chat-Id`, the Anthropic Messages form —
   is real but narrow value relative to the ongoing maintenance ask.
3. **Demo code rots fast.** A "lightweight demo" looks small (~470 LOC
   + vendor) but it gates a permanent tax: each chimera capability
   becomes optional UI work, each vendor lib bump needs re-testing,
   each browser API change is a new bug surface. Within 2–3 chimera
   releases without active UI use, it would be visibly broken.
4. **A broken demo is worse than no demo.** First user who runs
   `make webui-serve` and hits a blank page or a stale-API regression
   forms a "chimera is half-finished" impression. The risk is
   asymmetric: working UI is "fine, expected"; broken UI is "this
   project doesn't care about polish."
5. **The chimera-specific routes don't actually need a UI.** RAG ingest
   + search is one `chimera index` / `chimera search` invocation. Chat
   history is `chimera chat --list --search`. The Anthropic Messages
   shape is what the Anthropic SDK posts; you don't need a UI to use it.
   Power users have CLIs; casual users have third-party OpenAI UIs.

### 6.3. What stays vs. what went away

Stays (the genuinely valuable byproducts of the experiment):

- `chimera serve --public-path <dir>` flag — parity with llama-server,
  works for any static directory, useful regardless of whether chimera
  ever ships its own UI.
- `GET /v1/chats`, `GET /v1/chats/:id`, `GET /v1/chats/search`
  endpoints — close the read-side gap on `--persist-chats` for any
  HTTP client, not just a UI. Useful for scripts, exporters,
  third-party tools.
- The 6 new smoke tests covering those endpoints + `--public-path`.
- The "llama-server parity vs chimera-owned surface" table at the top
  of `chimera_serve.cpp` — categorizes every route by drift risk.

Removed:

- `webui/` directory (the HTML / JS / CSS).
- `scripts/fetch_webui_vendor.sh` (vendor fetcher).
- Makefile targets `webui-vendor` and `webui-serve`.

### 6.4. When (if ever) to revisit

Re-open this **only** if one of these specific signals shows up. Generic
"a chimera UI would be nice" is not a trigger — see § 6.2 point 5.

- **Concrete user request** for a chimera-specific UI feature that
  *cannot* be expressed through an existing OpenAI-compatible UI. Bar:
  the user is currently hitting the gap, not "might want this later."
- **chimera capability that has no CLI access path.** Today every
  chimera-only feature is reachable from the CLI; if that ever stops
  being true (rare modality, multi-modal composition that's awkward
  in a terminal, etc.), the UI argument strengthens.
- **External contributor wants to own the UI.** If someone who actually
  enjoys frontend work shows up and commits to maintaining it, the
  effort-vs-value math changes. The shipped server-side API (especially
  the `/v1/chats*` set + `X-Chimera-Chat-Id`) is the launching pad.

Until then: chimera does the server, third-party UIs do the UI.

## 7. Other follow-ups

Two smaller things, kept around because they apply regardless of UI
decisions:

1. **Smoke test in `scripts/test.sh` for Variant A.** A conditional
   block that probes whether the current binary has the embedded
   webui baked in (e.g. `GET /` → 200 + `text/html` vs. 404), and
   asserts the working case when so. No-op when
   `CHIMERA_WEBUI_EMBED=OFF`. Defends against the "upstream moved
   the binding" failure mode in § 5.1.

2. **Pre-gzip the embedded bundles + serve with
   `Content-Encoding: gzip`.** Currently the binary stores the full
   ~7 MB uncompressed bytes (see the size table in § 2) *and*
   cpp-httplib serves them uncompressed over the wire. JS/CSS gzip
   ratio is typically 3–4×, so this would cut the in-binary footprint
   from ~6 MB stripped to ~2 MB *and* the over-the-wire payload by
   the same factor. Requires:
   - A small CMake step to gzip each asset before invoking
     `xxd.cmake` (or replace `xxd.cmake` with a variant that gzips
     inline).
   - A chimera-side patch to `server-http.cpp`'s webui handlers to set
     `Content-Encoding: gzip` on the response. Browsers handle this
     transparently; no UI change.
   - A fallback path for clients that pass `Accept-Encoding: identity`
     (rare; the patch can just 406, every browser since 2010 sends
     `Accept-Encoding: gzip` by default).

   Worth doing if Variant A users complain about size.

---

## 8. Testing

There are currently no automated tests for the webui path. The
`scripts/test.sh` end-to-end suite tests against an `OFF` build (the
default), so the webui code paths are not exercised by `make test`.

The minimum useful addition would be the conditional smoke test
sketched in § 7 item 1: probe whether the binary has the webui, and if
so assert the three endpoints respond correctly. Keep it short — a full
browser-driven test (Playwright, etc.) is overkill for an opt-in
experimental feature.

Until that lands, the manual verification recipe in § 4 is what we have.

---

## 9. Useful references

- `tools/server/server-http.cpp` in the vendored llama.cpp tree — the
  authoritative source for the route binding, the COEP/COOP headers,
  and the api-key middleware ordering.
- `tools/server/CMakeLists.txt` in the vendored llama.cpp tree — shows
  how upstream itself invokes `xxd.cmake` for the same assets. Our
  chimera-side CMake mirrors that block.
- `tools/server/webui/` in the vendored llama.cpp tree — the Svelte
  source the bundles are built from.
- [`doc/serve.md`](../serve.md) — user-facing notes (the `--no-webui`
  flag and the build-time opt-in mention).
- [`doc/dev/server.md`](server.md) — the broader `chimera serve`
  developer guide; § 4 ("Routes") and § 7 ("Things to watch out for")
  pair well with this document.
- Git history at the commit that removed `webui/` — the only place the
  prototyped chimera-specific UI source survives. Useful starting point
  if Variant C ever materializes per § 6.4.
