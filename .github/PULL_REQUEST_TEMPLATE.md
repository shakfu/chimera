<!--
Generic PR template. Most sections are skippable; fill in what
applies.

If this PR bumps a vendored dependency (llama.cpp, whisper.cpp,
stable-diffusion.cpp, sqlite, sqlite-vec, linenoise) — i.e. any
`*_VERSION` line in scripts/manage.py — work through the
"Dependency-bump checklist" near the bottom.
-->

## Summary

<!-- 1-3 bullet points describing what changed and why. -->

## Test plan

- [ ] `make build` succeeds locally
- [ ] `make test` passes (23+ e2e cases)
- [ ] `make test-db-migrate` passes (v1 → latest still upgrades cleanly)
- [ ] Manual smoke for any user-facing change (paste curl / CLI output)

## Notes for review

<!--
Optional. Surface anything reviewers should look at first: a tricky
edge case, a deliberate trade-off, a follow-up TODO, etc.
-->

---

## Dependency-bump checklist

<!--
Only fill this in if any *_VERSION line in scripts/manage.py changed.
Skip the whole section otherwise.

Context: chimera links against several upstream-internal C++ headers
(server-context.h, server-http.h, common.h, ...) that are not part of
upstream's stable API. They shift with refactors. The steps below are
the audit trail that makes a version bump safe to land. See
doc/dev/maintenance.md for the full rationale.
-->

**Which dependency(ies) are bumping?**

- [ ] llama.cpp (`LLAMACPP_VERSION`) — old: `?` → new: `?`
- [ ] whisper.cpp (`WHISPERCPP_VERSION`) — old: `?` → new: `?`
- [ ] stable-diffusion.cpp (`SDCPP_VERSION`) — old: `?` → new: `?`
- [ ] sqlite (`SQLITE_VERSION`) — old: `?` → new: `?`
- [ ] sqlite-vec (`SQLITE_VEC_VERSION`) — old: `?` → new: `?`
- [ ] linenoise (`LINENOISE_VERSION`) — old: `?` → new: `?`

**Pre-bump audit** (run *before* editing the `*_VERSION` line)

- [ ] `make bump-check LLAMA_VERSION=<new_ref>` — diffs the seven
      vendored upstream-internal headers (llama.h, common.h, arg.h,
      chat.h, mtmd.h, server-context.h, server-http.h) against the
      target ref. If anything is flagged, audit:
  - `chimera_serve.cpp` route bindings (especially `handler_t` fields
    on `server_routes`).
  - `src/chimera/CMakeLists.txt` link order / archive groups.
  - `src/chimera/chimera_pin_check.cpp` — does each removed/renamed
    symbol have a matching `static_assert`? If not, add one.
  - `thirdparty/llama.cpp/src-aux/server-http.cpp` — chimera compiles
    this directly into its own target; check for any new
    `#include` it expects.

**After bumping the version line + `make build`**

- [ ] `make bump-check` — should report "clean" (the vendored copies
      are now the new pin's copies).
- [ ] `make test` — 23+ e2e cases pass.
- [ ] `make test-db-migrate` — v1 → latest still upgrades cleanly.
- [ ] `chimera info` reflects the new versions.
- [ ] CHANGELOG entry under `[Unreleased]` mentioning the bump.

**Optional but recommended**

- [ ] Pin assertions in `chimera_pin_check.cpp` (and the per-modality
      pin blocks in `chimera_whisper.cpp` / similar) still cover every
      `server_routes::*` field bound in `chimera_serve.cpp`. Add asserts
      for any *newly* bound route.
- [ ] If `LLAMA_BUILD_*` / SDCPP CMake options changed shape upstream,
      update `scripts/manage.py`.
