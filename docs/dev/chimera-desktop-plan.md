# chimera-desktop — design plan

Companion to [`docs/dev/webui.md`](webui.md) and
[`docs/dev/webui-impl.md`](webui-impl.md). Those two scope the
in-tree webui story (Variant A shipped, Variant B dumped, sidecar
showcase on the `webui` branch). This document scopes a **separate
downstream project** — `chimera-desktop` — that consumes chimera and
delivers a fully integrated desktop application.

Status: planning only. Not a chimera deliverable. Lives in this
repo's docs because the design decisions interact tightly with
chimera's HTTP surface and library artifacts; the project itself
would live in its own repository.

---

## 1. Why a separate project unlocks options chimera rejects

`webui.md` § 6.2 enumerated five reasons Variant B (a chimera-owned
UI) was killed. Re-evaluated under the chimera-desktop split:

| § 6.2 objection (in chimera) | Status in chimera-desktop |
|---|---|
| Identity mismatch (chimera = busybox binary, not a frontend) | Inverted — frontend ownership *is* the identity |
| Existing OpenAI-compatible UIs already cover chat | Differentiator is the integrated non-chat surfaces, not chat itself |
| Demo code rots fast | True everywhere; lands on chimera-desktop maintainers, not chimera's |
| Broken demo worse than no demo | Same, scoped to chimera-desktop's release cadence |
| Chimera-specific routes don't need a UI | Re-evaluated: an *integrated* product needs one even if power users don't |

The objections do not vanish; they get re-parented. chimera-desktop's
maintenance tax is its own burden, not a drag on chimera's release
cadence. That re-parenting is the whole point of the split — without
it, the right answer is still "don't do this."

Net effect: **Path 1 (fork upstream Svelte source)** and a new
**Path 4 (greenfield SvelteKit app)** become viable, where they
were not for the in-tree case.

---

## 2. Recommendation summary

- **Packager: Tauri 2.x.** Rust shell + system webview + first-class
  sidecar binary tooling. ~15 MB shell vs. Electron's ~150 MB
  baseline. Future path to FFI-link `libchimera.a` in-process via
  Rust bindings if the sidecar IPC ever becomes a bottleneck.
- **UI: SvelteKit, forking upstream `llama-ui` components.** Same
  framework as upstream so the forked chat pane drops in
  mechanically; rebases on llama.cpp bumps stay near-mechanical too.
  Tailwind for shell styling (matches upstream).
- **Integration model: spawn chimera as a sidecar process.** v1
  ships chimera as a bundled per-platform binary launched on app
  start with all opt-in flags enabled, bound to a random localhost
  port. v2 may switch to in-process linkage; defer until measured.
- **Distribution: per-OS signed installers with auto-update.**
  Apple notarization, Windows Authenticode, Linux AppImage / deb /
  rpm via Tauri's bundler.

The remainder of this document expands each decision.

---

## 3. Scope

**In (v1):**

- Tauri 2.x desktop application, cross-platform (macOS, Windows,
  Linux).
- Bundled per-platform `chimera` binary (initially CPU-only; backend
  matrix discussion in § 7).
- SvelteKit UI shell with:
  - Chat pane (forked from upstream `llama-ui` components, patched
    to send `X-Chimera-Chat-Id` for persistence consolidation).
  - Persisted-chat browser (left rail), backed by `/v1/chats*`.
  - RAG panel: collection CRUD + ingest + search.
  - Audio panel: file-upload transcription / translation.
  - Image panel: txt2img + img2img + inpaint + LoRA picker.
  - Rerank panel.
  - LLM LoRA hot-swap.
  - Settings: model picker, sampler defaults, API key (for remote
    chimera mode).
  - "About" pane surfacing the data `chimera info` prints.
- First-run UX: model picker (HF download, local file pick, default
  suggestions per modality).
- Auto-update via Tauri updater + signed manifests.

**Out (v1):**

- In-process `libchimera.a` linkage. Sidecar process only.
- Per-backend installers (CUDA, ROCm, Metal). CPU-only v1; backend
  matrix is a v1.x discussion.
- App-store distribution (Mac App Store, Microsoft Store). Direct
  download + auto-update only.
- Remote chimera mode (connect to a chimera serve on another host).
  Wire later; defer first-run complexity.
- Telemetry. Opt-in crash reporting only if any.
- Mobile (iOS / Android). Tauri 2 supports it, but the chimera
  binary is desktop-shaped today.

**Landed in chimera (was previously deferred):**

- `GET /v1/chimera/info` — JSON form of `chimera info`. Versions,
  built / loaded backends, devices, GPU/mmap/mlock/RPC flags,
  whisper/sd linkage + CPU features, SQLite versions, build flags.
  Consumed by chimera-desktop's About pane at `/#/chimera/about`.
- `GET /v1/chimera/db` — JSON form of `chimera db status`. Path,
  size, schema version, table list, per-table row counts. Consumed
  by chimera-desktop's chats-page footer.
- `POST /v1/chimera/shutdown` — graceful exit endpoint. Returns 202
  then triggers SIGINT-equivalent teardown on a detached thread
  150 ms later. Consumed by chimera-desktop's `sidecar::kill()`,
  which prefers it to SIGKILL when terminating the bundled child.

All three live in `src/chimera/chimera_serve_meta.cpp`; bound
unconditionally (no opt-in flag) since the data they expose is
read-only and useful to any downstream client. See chimera's
CHANGELOG `[Unreleased]` for the full design rationale.

---

## 4. Architecture

```text
+--- chimera-desktop (Tauri app) ---------------------------+
|                                                            |
|  +----------------------+     +-------------------------+  |
|  |  Tauri main process  |     |  webview (SvelteKit)    |  |
|  |  (Rust)              |<--->|  - shell chrome         |  |
|  |  - spawn sidecar     | IPC |  - forked chat pane     |  |
|  |  - lifecycle mgmt    |     |  - native panels        |  |
|  |  - native menus      |     |  - settings             |  |
|  |  - file dialogs      |     +-----------+-------------+  |
|  |  - updater           |                 |                |
|  +----------+-----------+                 |                |
|             |                             |                |
|             | spawns                      | fetch()        |
|             v                             v                |
|  +------------------------------------------------------+  |
|  |  bundled `chimera serve` (sidecar)                   |  |
|  |  --persist-chats --enable-rag --enable-audio         |  |
|  |  --enable-image --reranking                          |  |
|  |  --host 127.0.0.1 --port <random>                    |  |
|  +------------------------------------------------------+  |
|                                                            |
+------------------------------------------------------------+
```

Key invariants:

- The webview never talks to anything but `127.0.0.1:<port>`. The
  sidecar is localhost-only.
- The Tauri main process owns the sidecar's lifecycle. Quit-app
  signals SIGTERM, with a SIGKILL fallback after a timeout.
- The port is chosen at launch (bind to `:0`, read back). No
  fixed-port collisions across multiple app instances.
- Models live outside the app bundle — under the OS-conventional
  data directory (macOS: `~/Library/Application Support/chimera/
  models/`; Linux: `$XDG_DATA_HOME/chimera/models/`; Windows:
  `%LOCALAPPDATA%\chimera\models\`). Same path family chimera's DB
  already uses, so chimera CLI + chimera-desktop share state.
- The chimera SQLite DB at the default XDG path is the *same* DB
  the CLI uses. A user can run `chimera chat --persist` from the
  terminal and see the chats appear in chimera-desktop's left rail
  on next refresh.

---

## 5. Packager decision: Tauri

Ranked against the four candidates the user named:

### 5.1. Tauri 2.x (recommended)

- **Sidecar binary support is first-class.** `tauri.conf.json`'s
  `bundle.externalBin` array bundles per-platform binaries that the
  Rust side can spawn via `Command::new_sidecar`. Lifecycle hooks
  for clean shutdown.
- **~15 MB baseline shell.** Matters when the chimera CUDA build is
  already ~500 MB; an additional 150 MB of Electron Chromium on
  top reads as poor stewardship.
- **System webview** (WebView2 on Win 10+, WKWebView on macOS,
  WebKitGTK on Linux). Cross-webview compatibility is a real risk
  but the chimera-desktop UI does not need exotic CSS or
  experimental web APIs.
- **Future libchimera linkage path.** Rust FFI to chimera's C
  surface (`chimera/*.h`) via `bindgen` is straightforward. If v1's
  sidecar IPC ever measures as a bottleneck, in-process linkage is
  a v2 option without re-platforming.
- **Tauri's auto-updater** is signed-manifest-based and OS-native
  installer compatible.

### 5.2. Electron

- Most mature, largest contributor pool, no webview-consistency
  surprises (Chromium everywhere).
- **~150 MB baseline.** This is the deciding factor against.
  chimera's whole positioning is "lean single binary"; a chimera
  desktop that ships 150 MB of Chromium before chimera's bytes
  contradicts the upstream value proposition.
- Pick this only if "team can hire frontend folks who already know
  Electron" outweighs the binary-size argument. For a single-
  maintainer project, no.

### 5.3. Wails (Go)

- Reasonable engineering choice; similar size profile to Tauri.
- Go adds nothing chimera-desktop specifically needs — the sidecar
  is a C++ binary, not Go. The v2 in-process linkage path is
  awkwarder from Go than from Rust (cgo overhead, ABI friction).
- No clear advantage over Tauri.

### 5.4. Electrobun (Bun + system webview)

- Promising tech, very small community.
- For a project that has to be reliable on three operating systems
  with code signing and auto-update, leading with a not-yet-mature
  packager is asking for trouble.
- Revisit in 2 years.

### 5.5. The honest dependency cost of Tauri

Rust toolchain, Node toolchain, Tauri CLI, Apple developer
certificate ($99/yr), Windows code signing certificate ($200-400/yr
or use a service like SignPath), CI on three OSes. None of this
matters for chimera; all of it matters for chimera-desktop. Plan
for ~2 weeks of pure build-infrastructure work before the first
shippable installer.

---

## 6. UI architecture

Three sub-decisions.

### 6.1. Fork upstream Svelte components, do not iframe

The naive "embed upstream `llama-ui` in an iframe" approach loses
exactly what makes chimera-desktop worth building:

- The upstream UI does not send `X-Chimera-Chat-Id`, so chat
  persistence stays in the broken-state described in `webui.md`
  § 5.6 (one chats row per request instead of consolidated
  multi-turn).
- Inline RAG citations are not possible from outside the chat
  rendering code.
- Settings, model picker, themes all live in upstream's chrome,
  which conflicts with chimera-desktop's own shell.
- Cross-frame messaging for the surrounding panels (chat history
  rail, RAG, etc.) is brittle and ugly.

The right approach: **vendor upstream's `tools/server/webui/src/`
SvelteKit source into chimera-desktop's repo**, patch the chat
components to send `X-Chimera-Chat-Id`, drop them into
chimera-desktop's shell as a `<ChatPane>` component.

### 6.2. SvelteKit for the shell, matching upstream's framework

Same framework as upstream means:

- The vendored chat components drop in without re-platforming.
- Rebasing on llama.cpp bumps is "diff their SvelteKit, apply"
  rather than "port their Svelte to React again."
- The styling vocabulary (Tailwind, upstream's design tokens)
  carries over.

Trade-off: ties chimera-desktop's framework choice to upstream's.
If upstream ever migrates off SvelteKit, chimera-desktop has to
decide between following or freezing the fork. Acceptable risk —
SvelteKit migration is a multi-quarter event with public signal.

### 6.3. Shell wraps chat; chimera-specific panels are native shell

```text
+------------------------------------------------------------+
| chimera-desktop                                  [_][o][x] |
+------+-----------------------------------------------+-----+
|      |                                               |     |
|  ┌───┤  ChatPane (vendored, patched llama-ui)        ├───┐ |
|  │   │                                               │   │ |
|  │ p │  - sends X-Chimera-Chat-Id                    │ R │ |
|  │ e │  - renders inline RAG citations               │ A │ |
|  │ r │  - markdown, code highlight, math, branching  │ G │ |
|  │ s │  - parallel conversations                     │   │ |
|  │ i │  - all client-side features from upstream     │ A │ |
|  │ s │                                               │ u │ |
|  │ t │                                               │ d │ |
|  │   │                                               │ i │ |
|  │ c │                                               │ o │ |
|  │ h │                                               │   │ |
|  │ a │                                               │ I │ |
|  │ t │                                               │ m │ |
|  │ s │                                               │ a │ |
|  │   │                                               │ g │ |
|  └───┤                                               ├─e─┘ |
|      |                                               |     |
+------+-----------------------------------------------+-----+
| status bar: model | backend | slot util | DB size          |
+------------------------------------------------------------+
```

The chat pane is upstream's IP, lightly patched. The shell, left
rail, right tab strip, and status bar are chimera-desktop's IP.
Boundary is clean: upstream-tracked vs. chimera-desktop-owned.

---

## 7. Backend matrix problem

chimera's CPU build is ~34 MB. CUDA is multi-hundred MB. ROCm,
Metal, Vulkan each have their own footprints. chimera-desktop has
to choose:

- **A. CPU-only installer.** Smallest (~50 MB), works everywhere,
  slow on a GPU machine. v1 default.
- **B. Per-backend installers.** One installer per (OS × backend)
  combination. Six+ artifacts per release. Forces the user to know
  what GPU they have. UX disaster.
- **C. Detect-and-download at first run.** Ship a thin installer
  that detects GPU and pulls the right chimera build from a CDN on
  first launch. Better UX, requires CDN infra and signed builds
  per backend on chimera's side.
- **D. Bundle everything.** Multi-GB installer. Strictly worse than
  C unless air-gapped distribution matters.

**v1 picks A.** Honest, simple, ships. Add C in v1.x with a
"download GPU acceleration" button in settings.

Note this puts a soft requirement on chimera: per-backend builds
need to be reliably produced and hosted somewhere chimera-desktop's
first-run downloader can fetch them. Today this is manual; making
it automated is a chimera-side ask.

---

## 8. Project structure

Lives in a separate repo (`chimera-desktop`), not under
`shakfu/chimera`. Sketch:

```text
chimera-desktop/
  src-tauri/                 Rust shell
    src/
      main.rs                spawn sidecar, lifecycle, menus
      sidecar.rs             chimera child-process mgmt
      commands.rs            Tauri commands exposed to JS
    Cargo.toml
    tauri.conf.json          externalBin entries per platform
    binaries/
      chimera-x86_64-apple-darwin
      chimera-aarch64-apple-darwin
      chimera-x86_64-pc-windows-msvc.exe
      chimera-x86_64-unknown-linux-gnu
      ...
  src/                       SvelteKit UI
    routes/
      +layout.svelte         shell chrome
      +page.svelte           main view
      settings/+page.svelte
      about/+page.svelte
    lib/
      vendored/llama-ui/     forked from upstream tools/server/webui
      components/            chimera-desktop shell components
      api/                   typed fetch client for chimera HTTP
      stores/                Svelte stores for app state
  package.json
  vite.config.ts
  README.md
  docs/
    architecture.md
    upstream-rebase.md       how to re-vendor llama-ui on bump
    backend-matrix.md        the § 7 problem and its current state
```

---

## 9. Build & distribution

- **Cross-platform CI.** GitHub Actions matrix on
  `{macos-latest, ubuntu-latest, windows-latest}`. Each builds the
  Tauri bundle for its host.
- **Code signing.**
  - macOS: Apple Developer ID Application certificate, notarization
    via `notarytool`. Required for users to launch without
    Gatekeeper warnings.
  - Windows: Authenticode cert (EV preferred for SmartScreen
    reputation skip).
  - Linux: GPG signing for `.deb` / `.rpm` repos; AppImage
    optionally signed via `appimagetool --sign`.
- **Installer formats.**
  - macOS: `.dmg` with drag-to-Applications.
  - Windows: `.msi` (Tauri default) + `.exe` NSIS option.
  - Linux: `.AppImage` (no install required) + `.deb` + `.rpm`.
- **Auto-update.** Tauri updater with a signed `latest.json`
  manifest hosted on GitHub Releases. Delta updates not supported;
  full re-download per update. Acceptable given the 15 MB shell.
- **Release cadence.** Independent of chimera. Pin to a specific
  chimera version per release; bump deliberately, not
  automatically.

---

## 10. Upstream rebase discipline

Two upstreams to track: chimera itself, and the vendored portion of
`llama-ui`.

- **Chimera bumps.** Pin chimera version in CI. Update the
  bundled-binary set per release. Re-run the smoke suite against
  the new binary. The pin-check infrastructure
  (`chimera_pin_check.cpp`) defends chimera's HTTP shape against
  silent llama.cpp drift — chimera-desktop inherits that protection
  for free by depending on chimera's HTTP API rather than
  llama.cpp's internal types.
- **`llama-ui` bumps.** Document the rebase recipe in
  `docs/upstream-rebase.md`: which files were vendored, which
  patches applied, how to re-apply on the next pull. Suggest one
  rebase per major chimera bump (when `LLAMACPP_VERSION` moves in
  `scripts/manage.py`), not per chimera-desktop release.

The chimera-desktop equivalent of `make bump-check` is the
manual-but-disciplined rebase. Automating it is the long-term goal;
not v1 work.

---

## 11. Testing

- **Smoke at sidecar launch.** Tauri main process waits for
  `GET /health` to return 200 before showing the webview. Failure
  surfaces as an actionable error dialog ("chimera failed to
  start: <stderr tail>") rather than a blank window.
- **Playwright against the running app.** Tauri exposes a WebDriver
  endpoint via `tauri-driver`; Playwright drives the real webview.
  Cover the four headline flows: send a chat message, ingest a
  document into RAG, transcribe an audio file, generate an image.
- **CI matrix.** Same OS matrix as builds. Headless on Linux
  (Xvfb); native on macOS / Windows.
- **Manual QA** for visual regressions; no screenshot diffing in
  v1 (it adds CI flakiness disproportionate to value).

---

## 12. Acceptance criteria for v1.0

The first public release ships when all of:

1. Installer for all three OSes downloads, installs, and launches
   cleanly on a fresh VM without developer-mode workarounds.
2. First-run UX successfully points to a working model file
   (downloaded or local) and reaches a usable chat state.
3. Chat with persistence consolidation works end-to-end
   (`X-Chimera-Chat-Id` round-trips, multi-turn lands in one
   chats row, the persisted-chat browser shows it).
4. RAG ingest + search + retrieve-augmented-chat works on at least
   one collection.
5. Audio transcription and image generation each demonstrably work
   against bundled-default sample models.
6. Auto-update from v0.x to v1.0 has been smoke-tested on at least
   one OS.
7. The "About" pane surfaces chimera's `info` data via the
   `/v1/chimera/info` endpoint (shipped — see § 3 "Landed in
   chimera"). The chats panel footer surfaces DB size + row
   counts via `/v1/chimera/db`.
8. Code signing on macOS and Windows; AppImage on Linux.
9. `docs/upstream-rebase.md` documents the `llama-ui` vendoring
   so a second contributor can perform the next rebase from notes
   alone.

Anything beyond this list (per-backend installers, remote chimera
mode, mobile, app-store distribution) is v1.x or later.

---

## 13. Kill switches

chimera-desktop is archived if any of:

- **Maintainer burnout signal.** A release slips by more than 3x
  the previous cadence with no external contributors filling the
  gap. The Variant B post-mortem (`webui.md` § 6.2) is the warning
  shot.
- **Upstream rebase becomes intractable.** If `llama-ui` undergoes
  a breaking framework migration (e.g., off SvelteKit) and
  re-vendoring would require a multi-month port, freeze the chat
  pane at the last working revision and let chimera-desktop fall
  behind, OR archive. Do not pretend a frozen fork is current.
- **Chimera HTTP surface destabilizes.** If chimera starts churning
  its HTTP routes faster than chimera-desktop can adapt, that is a
  conversation to have with chimera maintainers, not a problem to
  paper over with a compatibility shim layer.
- **The "we should just make this an Electron wrapper around the
  upstream `llama-ui`" temptation wins.** That product already
  exists (sort of — see Jan, LM Studio). chimera-desktop's
  reason-to-exist is the integrated non-chat surfaces. Losing that
  focus means archive.

---

## 14. The honest framing

chimera-desktop is a **6–9 month project to v1**, not a weekend.
Scope is closer to LM Studio or Jan than to "extend the chimera
webui." The positioning is genuinely differentiated: no current
desktop LLM app ships LLM + audio + image + RAG + persistent chat
in one shared-ggml process — that combination is chimera's
contribution, and a polished desktop wrapper would make it visible
to users who would never run a CLI.

But it is a separate product with a separate release cadence, a
separate skill set, separate distribution infrastructure, and a
separate maintenance burden. Starting it without committing to those
costs reproduces Variant B's failure mode at a larger scale.

The recommended sequencing:

1. **First, ship the in-tree sidecar** per `webui-impl.md`. It
   surfaces the integration gaps cheaply (does
   `X-Chimera-Chat-Id` round-trip cleanly? do the RAG ingest
   endpoints accept what a UI sends? does the LoRA list endpoint
   include the data a picker needs?) without committing to the
   desktop build.
2. ~~**Bind the `/v1/chimera/info` and `/v1/chimera/db` endpoints**
   in chimera.~~ Shipped — see § 3 "Landed in chimera". chimera
   also ships `POST /v1/chimera/shutdown` for graceful child
   termination from a wrapper process.
3. **Watch the chimera serve `make webui-serve` traffic.** If
   anyone external uses it and asks for a desktop version, that is
   the § 6.4 trigger that justifies the chimera-desktop investment.
4. **Then start chimera-desktop**, with this plan as the starting
   point and the in-tree sidecar's painful spots as the punch list
   of what to fix.

Skipping straight to chimera-desktop is the high-variance bet; the
sidecar-first path is the high-evidence one.

---

## 15. Open questions

- **Remote chimera mode.** Should chimera-desktop be able to point
  at a `chimera serve` running on another host (the user's home
  server, a workstation, a VPS)? Useful but complicates first-run
  UX and security model (API key prompt, TLS, port forwarding).
  v1.x.
- **In-process libchimera linkage.** Worth it only if measured IPC
  overhead is a real bottleneck. Defer until v1 is in users'
  hands.
- **Plugin / extension model.** Tempting (let users add their own
  panels), almost always premature. Do not build until at least
  one concrete external request exists.
- **Mobile.** Tauri 2 supports iOS / Android but chimera does not
  build for those platforms today. Out of scope until chimera
  itself does.
- **Telemetry.** Opt-in crash reporting (Sentry) is reasonable.
  Anything beyond that (feature usage, model selections) is a
  trust-erosion risk for a local-first AI app. Default to none.
- **Monetization.** Not a question to leave to v1.0 surprise. If
  the project is MIT-licensed and free, say so upfront; if a paid
  tier is ever planned, design the open-source boundary now.
