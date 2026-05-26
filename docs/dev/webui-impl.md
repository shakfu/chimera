# Extended WebUI — implementation plan

Companion to [`docs/dev/webui.md`](webui.md). That document records
the history (Variant A shipped, Variant B dumped, § 6.4 re-open
conditions). This one is the forward plan for the `webui` branch off
chimera 0.2.0: a sidecar showcase UI that demonstrates and exercises
the chimera-specific routes the upstream `llama-ui` does not know
about.

Status: experimental, branch-only. Promotion to `main` is gated on the
acceptance criteria in § 9.

> **Related consumer surface (not this doc).** This plan is about an
> HTTP-driven sidecar UI (`chimera serve --public-path` + static JS). A
> separate, programmatic way to drive chimera is the Python bindings
> (`bindings/`, nanobind over `chimera.hpp`) -- documented in
> [`docs/dev/oop-layer.md`](oop-layer.md) § "Python bindings". If a future
> UI wants an in-process Python backend instead of HTTP, that is the layer
> to build on; it is orthogonal to the sidecar plan below.

---

## 1. Recommendation

**Path 3 — sidecar UI mounted via `--public-path` at `/chimera/`,
alongside the upstream `llama-ui` at `/`.**

The three candidates in `webui.md` § "Can you extend ..." were:

1. Fork the upstream Svelte source.
2. JS injection on top of the served `index.html`.
3. Sidecar UI via `--public-path`.

**Why Path 3, ranked against the stated goal ("showcase chimera
features + test them"):**

- *Showcase fit.* A sidecar with one page per feature is **better** for
  a showcase than grafting features into a chat UI. Each feature gets
  dedicated screen real estate + narrative, instead of hiding in a
  panel a casual visitor may not open. Path 1 would scatter features
  through the upstream UI's chrome; Path 3 makes them the headline.
- *Decoupled from upstream churn.* Path 3 never touches the vendored
  Svelte source. A llama.cpp bump cannot break it. Path 1's headline
  cost (manual rebase per `LLAMACPP_VERSION` bump, plus a Node
  toolchain in the build) does not apply. Path 2's failure mode
  (Svelte's compile-time obfuscated class names silently breaking
  selectors after every bump, with no pin-check defense — see
  `webui.md` § 5.1) does not apply either.
- *Failure containment.* If a sidecar page breaks, the upstream chat
  UI at `/` is unaffected. Path 2 actively risks breaking the chat UI
  that users rely on.
- *Scaffolding already shipped.* `--public-path`, the `/v1/chats*`
  endpoints, the smoke tests, and the route-categorization table at
  the top of `chimera_serve.cpp` all survived Variant B's teardown
  specifically because they're useful regardless of UI ownership
  (`webui.md` § 6.3). Path 3 reuses every line.
- *Bounded maintenance ceiling.* `rm -rf webui/` removes the
  experiment cleanly. Path 1's removal means undoing CMake + Node +
  `manage.py` changes.
- *Maps to a re-open trigger.* `webui.md` § 6.4 names three triggers;
  the "concrete user request for a chimera-specific feature no
  third-party UI can express" trigger is the one being invoked here,
  scoped to **showcase / test harness** rather than production
  everyday UX.

**The honest counter to Path 3.** Deep inline integration — e.g.,
RAG citations rendered *inside* the chat transcript, or LoRA pickers
in the model selector — needs Path 1. The sidecar cannot do that.
For *showcase*, dedicated pages are a feature, not a limitation. If a
real production "RAG-inline-in-chat" use case appears later, that's a
separate Variant D conversation; do not pre-build for it.

**The honest counter to "do nothing."** Every § 6.2 objection from the
Variant B post-mortem still applies, at lower weight, to Path 3. The
mitigations are: scope (showcase pages only, never a chat UI), tech
stack discipline (§ 5), explicit kill switches (§ 10).

---

## 2. Scope

**In:**

- One sidecar HTML/CSS/JS application served at `/chimera/` via
  chimera's existing `--public-path` mount.
- One showcase page per chimera-specific feature surface enumerated
  in `webui.md` § 6.5.
- Capability detection: landing page probes which optional surfaces
  are enabled and grays out / annotates pages whose backing flag is
  off.
- A `make webui-serve` target that starts chimera with the demo's
  required flags.
- Smoke tests asserting each page returns 200 and the backing routes
  are reachable.

**Out:**

- A chat UI. The upstream `llama-ui` at `/` is the chat surface.
- Replacing or overlaying the upstream UI's chrome.
- Production-grade UX (settings persistence, themes, accessibility
  audits, i18n, mobile-first layouts). Functional clarity > polish.
- Embedding the sidecar into the binary (no xxd). The whole point of
  staying on `--public-path` is that the experiment can be deleted
  without touching the build.
- A Node toolchain. No npm, no bundler. Pure vanilla.
- Any new HTTP endpoints. The plan binds to what `chimera_serve.cpp`
  already exposes; new endpoints are tracked in § 8 as deferred work.

---

## 3. Architecture

```
chimera serve --public-path webui/ \
              --persist-chats --enable-rag --enable-audio \
              --enable-image --reranking --enable-embeddings <e.gguf>

   /                    -> upstream llama-ui (xxd-baked, Variant A)
   /chimera/            -> sidecar landing page (this plan)
   /chimera/chats/      -> persisted chat browser
   /chimera/rag/        -> RAG collections + ingest + search
   /chimera/audio/      -> transcription / translation demo
   /chimera/image/      -> txt2img + img2img + inpaint
   /chimera/rerank/     -> cross-encoder rerank demo
   /chimera/lora/       -> LLM LoRA hot-swap
   /chimera/anthropic/  -> /v1/messages playground
   /chimera/slots/      -> KV-slot snapshot demo
   /chimera/lib/        -> shared JS / CSS / vendored deps
```

`webui/` is a flat static directory. `chimera serve` mounts it via
cpp-httplib's `set_mount_point`, which is already wired through
`common_params.public_path` (see § 4.1 of `webui.md` — same path
upstream uses for its own webui). No new server code.

The sidecar is purely client-side; every page makes `fetch()` calls
to the chimera HTTP API at the same origin (no CORS, no proxy).

---

## 4. Tech stack

**Decision: vanilla JS + Pico 2 CSS + `marked` for markdown.**
Everything else proven necessary at page-build time, not in advance.

The Variant B post-mortem (`webui.md` § 6.2 point 1) called out three
specific bugs in the Alpine + htmx + KaTeX stack: Alpine `$nextTick`
scoping, marked 14 removed options, KaTeX/MutationObserver feedback
loop. The lesson is **fewer moving parts**, not "use a different
framework." Specifically:

- **No Alpine / htmx / Preact.** Showcase pages are mostly forms +
  fetch + render-result. State machines are overkill; a 50-line
  vanilla-JS page beats a 30-line Alpine page with a 40 KB runtime
  cost and a debugging tax.
- **No KaTeX.** None of the chimera-only demos generate math.
  Re-introduce only if a specific page needs it.
- **No highlight.js initially.** Add per-page if a demo benefits
  (the Anthropic playground might want JSON syntax highlight for
  request / response).
- **Pico 2 CSS** for default form / typography. Single CSS file,
  no JS dependency, drop-in classless.
- **`marked`** (pinned) for rendering message content in the chat
  history browser. Pinned to a known-good version to avoid the
  marked-14 surprise.

Vendor everything under `webui/lib/vendor/` with pinned filenames
including version. No CDN at runtime (offline demo must work).

---

## 5. Page-by-page wiring

Each page is a single `index.html` + same-folder `app.js` + same-folder
`style.css` (only for page-specific overrides; Pico defaults dominate).
All pages share `webui/lib/{chimera.js,capabilities.js,format.js}`.

### 5.1. Landing — `/chimera/`

- Static `index.html` with a card grid: one card per page below.
- On load, calls `capabilities.js` which fetches:
  - `GET /health` — must succeed; otherwise show "chimera serve not
    running."
  - `GET /props` — pulled for model name, sampler defaults (echo to
    each page's defaults).
  - `GET /v1/models` — confirms loaded model + aliases.
  - Probes each chimera-specific route head-only (`HEAD` not always
    supported; fall back to a cheap `GET` and treat 404 as "feature
    off, flag not passed").
- Cards for disabled features are grayed out with the exact CLI flag
  to enable them ("start chimera with `--enable-rag` to enable this
  demo").

### 5.2. Persisted chats — `/chimera/chats/`

Backs `--persist-chats`. The killer chimera feature this surfaces.

- **List view** — `GET /v1/chats?limit=50` populates a left rail
  (chat id, first user message preview, updated_at, message count).
- **Detail view** — clicking a row calls `GET /v1/chats/:id` and
  renders the messages in a read-only transcript using `marked`.
- **Search box** — `GET /v1/chats/search?q=<term>` with the FTS5
  `[word]`-highlighted snippets rendered in result rows.
- **"Continue this chat" link** — opens the upstream UI at `/` with a
  hint banner ("the upstream UI does not send X-Chimera-Chat-Id, so
  continued turns will land in a new chat row"). This is the precise
  demonstration of the consolidation gap in `webui.md` § 5.6.

No write endpoints (chats are written by `/v1/chat/completions`,
not by a UI gesture).

### 5.3. RAG — `/chimera/rag/`

Backs `--enable-rag`. Combines three sub-panels on one page.

- **Collections panel** — `GET /v1/vector_stores` lists; row actions
  call `POST /v1/vector_stores` (create), `GET /v1/vector_stores/:n`
  (stats), `POST /v1/vector_stores/:n/delete` (drop).
- **Ingest panel** — file picker + textarea. File path goes to
  `POST /v1/vector_stores/:n/files` as multipart; textarea goes to
  the same endpoint as JSON `{"text": "..."}`. Show row count
  delta after ingest.
- **Search panel** — `POST /v1/vector_stores/:n/search` with
  `{"query": "...", "k": 5}`. Render hits with score + snippet.

### 5.4. RAG-augmented chat — `/chimera/rag-chat/`

Composition demo: shows that chimera does RAG and chat in one
process. Single page, no new endpoints.

- Pick a collection (`GET /v1/vector_stores`).
- Type a question; UI runs `POST /v1/vector_stores/:n/search` first,
  injects the top-k as a `system` message, then calls
  `POST /v1/chat/completions` (non-streaming first cut; streaming
  later if the page survives).
- Renders the answer **with the retrieved snippets as visible
  citations next to the answer**. This is the chimera-specific value
  proposition that the upstream UI cannot demonstrate.

### 5.5. Audio — `/chimera/audio/`

Backs `--enable-audio`.

- File picker (WAV only — match server limitation, surface it in
  copy).
- Two buttons: "transcribe" → `POST /v1/audio/transcriptions`,
  "translate to English" → `POST /v1/audio/translations`.
- Render result + timing.
- Mic capture: deferred. Browser MediaRecorder produces WebM/Opus
  by default; server is WAV-only. Add only after the server grows
  multi-format support.

### 5.6. Image — `/chimera/image/`

Backs `--enable-image`.

- Three tabs: txt2img, img2img, inpaint.
- txt2img — `POST /v1/images/generations`. Form fields: prompt,
  negative prompt, width, height, steps, cfg, seed, sampler.
- img2img — `POST /v1/images/edits` with init image upload +
  strength slider.
- Inpaint — same as img2img + mask upload.
- LoRA picker sub-section — `GET /v1/images/lora-adapters` populates
  a multi-select; chosen names go into the `loras` array on the
  request.
- ControlNet / PhotoMaker fields — start collapsed under an
  "advanced" disclosure. Showcase, not daily-driver UX.

### 5.7. Rerank — `/chimera/rerank/`

Backs `--reranking`.

- Query textarea + N document textareas (add/remove rows).
- "Rerank" → `POST /v1/rerank`.
- Render the ranked output with scores; visualize the reordering vs.
  input order.

### 5.8. LoRA hot-swap — `/chimera/lora/`

Always-on (no flag required if any LoRA was loaded at startup).

- `GET /lora-adapters` lists registered adapters.
- Per-row scale slider (0..1).
- "Apply" → `POST /lora-adapters` with the full vector of
  id/scale pairs.
- Tiny side-by-side: generate one completion before and after with
  the same prompt + seed to make the effect visible.

### 5.9. Anthropic Messages — `/chimera/anthropic/`

Always-on (route is unconditionally bound).

- Two-pane: raw request JSON on the left, raw response on the right.
- "Send" → `POST /v1/messages`.
- "Count tokens" → `POST /v1/messages/count_tokens`.
- Pre-fill with a working example. Purpose: prove the Anthropic-shape
  compat works against the same loaded model.

### 5.10. KV slots — `/chimera/slots/`

Conditional on `--slot-save-path` being set.

- `GET /slots` lists current slot state.
- Per-slot save / restore / erase via `POST /slots/:id_slot` with the
  action body.
- Pair with a tiny chat-against-this-slot box that uses `/infill` or
  `/v1/completions` so the snapshot effect is observable.

---

## 6. Shared library — `webui/lib/`

- `chimera.js` — `fetchJSON(path, opts)` wrapper that handles bearer
  auth (read from `localStorage["chimera.apiKey"]`), pulls
  `X-Chimera-Chat-Id` out of response headers when present, throws
  typed errors on non-2xx.
- `capabilities.js` — probe the chimera surface once on landing,
  cache the result on `sessionStorage`, expose
  `capabilities.has("rag")` / `.has("audio")` / etc. so each page
  can early-exit with a friendly message.
- `format.js` — markdown render via `marked` (pinned), timestamp
  formatting, byte sizes, score bars.
- `vendor/` — pinned third-party files; one subdir per lib + version.

---

## 7. Build & distribution

**Branch-local, no binary involvement.**

- `webui/` lives at the repo root (matches Variant B's location for
  consistency with git history).
- `make webui-serve` is the demo entrypoint:

  ```make
  webui-serve:
      ./build/chimera serve \
          -m $(MODEL) \
          --public-path webui \
          --persist-chats \
          --enable-rag $(EMBED_MODEL) \
          --enable-audio $(WHISPER_MODEL) \
          --enable-image $(SD_MODEL) \
          --reranking $(RERANK_MODEL) \
          --slot-save-path .chimera-slots \
          --host 127.0.0.1 --port 8080
  ```

  Variables default to common paths under `models/`; user can
  override.

- `make webui-vendor` re-fetches the pinned third-party files into
  `webui/lib/vendor/` (Pico, marked). Pure shell, no Node. Vendored
  files commit to the repo (~150 KB).

- **No CMake changes.** No `CHIMERA_WEBUI_*` option. No xxd. The
  binary is identical to a non-webui-branch build.

**If this graduates** (per § 9 acceptance), then revisit whether to
embed the sidecar too. Not now — the build cost of "make a
sidecar embeddable" is exactly the kind of work that contributed to
Variant B's death-by-maintenance.

### 7.1. Note: upstream's embed mechanism changed at b9318

This plan layers a sidecar on top of the existing Variant A embed path
(`CHIMERA_WEBUI_EMBED`) and the `--public-path` mount. Both still apply,
but the *mechanism* underneath Variant A was rebuilt by llama.cpp b9318
and is worth knowing before touching anything embed-adjacent:

- `scripts/xxd.cmake` and the static `tools/ui/ui.h` were deleted
  upstream. A host generator `tools/ui/embed.cpp` now produces `ui.cpp` +
  `ui.h`, and `server-http.cpp` gates its routes on the generated
  `LLAMA_UI_HAS_ASSETS` define + the runtime `params.ui` flag, not on
  `LLAMA_BUILD_WEBUI`. Full analysis: [`webui.md` § 10](webui.md).
- chimera now *always* generates and links `ui.cpp`/`ui.h` (a nullptr
  stub when embed is OFF, the baked assets when ON), because
  `server-http.cpp` references `llama_ui_find_asset()` unconditionally.

Implications for this sidecar plan:

- The "**No xxd**" line under § 7 is now trivially true — xxd is gone
  upstream regardless. The stronger claim still holds: this sidecar adds
  **no CMake changes and no `CHIMERA_WEBUI_*` coupling**; it ships as
  static files mounted via `--public-path`, independent of the generated
  `ui.cpp`.
- If § 9 graduation ever revisits embedding the sidecar, do *not*
  resurrect an xxd step — reuse the same `ui-embed` generator chimera
  already builds (extend it with the sidecar's asset names), so there is
  one embed path, not two.
- The `make webui-serve` recipe in § 7 is unaffected: `--public-path`
  mounting takes precedence over the embedded `GET /` handler and does
  not depend on the embed mechanism at all.

---

## 8. Testing

Mirror `webui.md` § 7 / § 8 — minimum useful smoke, no Playwright.

- Extend `scripts/test.py` with a conditional block: when
  `webui/index.html` exists on disk, start `chimera serve
  --public-path webui` and probe each page returns 200 +
  `text/html`. Skip when the directory is absent (matches the
  existing "skip when fixture missing" pattern).
- One end-to-end smoke per page that goes beyond "200 OK": e.g.,
  hit `/v1/chats` directly to confirm the route the chats page
  depends on responds with valid JSON. These tests are server-side
  smokes for the routes; the page itself is tested by hand against
  a browser.
- No browser automation. The pages are simple enough that manual
  smoke during page authorship is sufficient; adding Playwright /
  Selenium is exactly the kind of dependency creep § 6.2 warned
  against.

**Shipped after this doc was written:**

- `GET /v1/chimera/info` — JSON form of `chimera info` (versions,
  built/loaded backends, devices, GPU/mmap/mlock/RPC flags, whisper/sd
  linkage + CPU features, SQLite versions, build flags). Implemented
  in `chimera_serve_meta.cpp`, always bound. Use this for a
  "what is this binary capable of" page in the landing chrome.
- `GET /v1/chimera/db` — JSON form of `chimera db status` (path,
  size, schema version, table list, per-table row counts). Use this
  for the persisted-chats page footer.
- `POST /v1/chimera/shutdown` — graceful exit. Returns 202 then
  triggers SIGINT-equivalent teardown 150 ms later. Was added with
  chimera-desktop's wrapper-process cleanup in mind, but useful for
  any sidecar setup that wants a clean shutdown signal.

---

## 9. Acceptance criteria for promotion to `main`

The `webui` branch merges back only if all of:

1. The pages in § 5.1–5.4 (landing, chats, RAG, RAG-chat) all
   demonstrably work end-to-end against a `make webui-serve`
   session, including the four chimera-only shapes flagged in
   `webui.md` § 6.5: `/v1/chats*`, `X-Chimera-Chat-Id`
   consolidation behavior, RAG ingest+search, and at least one
   image-side LoRA / ControlNet / PhotoMaker field round-trip.
   (Audio, image, rerank, LoRA hot-swap, Anthropic, slots can ship
   incomplete or come later.)
2. `make webui-serve` works on a fresh clone with no manual setup
   beyond models being present (same SKIP-when-missing pattern as
   `make test`).
3. The smoke tests in § 8 are wired into `make test` and pass.
4. `make build` is unchanged (no new CMake option, no binary size
   delta, no Node toolchain dependency).
5. `webui.md` § 6.4 / § 6.5 cross-reference this document so a
   future maintainer reading the post-mortem finds the live
   experiment.
6. A README in `webui/` documents what each page demonstrates and
   which `chimera serve` flag enables it.

Anything not on this list is out of scope for the first merge.

---

## 10. Kill switches

The branch is abandoned (or merged but later deleted) if any of:

- **Maintenance creep.** Tracking weekly time spent on the sidecar
  for one release cycle. If it exceeds ~10% of release effort
  without delivering new showcase value, archive.
- **Silent breakage across releases.** A page broken across 2+
  chimera releases without anyone noticing means nobody uses it —
  delete the page (or the whole sidecar if the pattern repeats).
- **No external validation.** Six months after merge with zero
  external user feedback referencing a sidecar page, treat as
  evidence the showcase intent is not landing and archive the
  experiment. Reuse the surviving infrastructure (the smoke tests,
  any new endpoints added) exactly as `webui.md` § 6.3 captured
  the Variant B leftovers.
- **A sidecar page starts pulling for a chat UI.** If a page wants
  to render streaming chat or duplicate features the upstream UI
  already provides, that is the Variant B failure mode resurfacing.
  Stop and re-read `webui.md` § 6.2 point 2 before adding it.

The kill switches exist because the maintenance trap that killed
Variant B is structural, not specific to that stack. Surviving the
trap means committing in advance to the conditions under which the
experiment ends.

---

## 11. Open questions

These are deliberately unresolved; resolve during page authorship,
not in this plan:

- **Streaming on the RAG-chat page.** SSE parsing adds JS surface;
  worth it only if non-streaming feels slow during demos.
- **Image gallery / history.** Generated images vanish on reload
  unless we persist them. SQLite has no images table today; out of
  scope for this branch.
- **Settings persistence.** localStorage for API key + endpoint URL
  is enough for v1; do not build a settings panel until something
  needs configurable beyond those two.
- **Embedding the sidecar in the binary.** Decide only after § 9
  acceptance; the answer probably stays "no" because the friction
  of `make webui-vendor` + commit is exactly what keeps the
  experiment honest about what's actually in use.
