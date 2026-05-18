# Combining chimera + dependencies into one static library

This is a planning document, not a built feature. It captures the
analysis behind turning chimera into a redistributable static library
and the per-platform bundling strategy implemented in
`scripts/combine_archives.py`.

---

## 1. Why

Today `src/chimera/CMakeLists.txt` produces a single executable that
statically links ~15 archives (llama.cpp, ggml + per-backend ggml-*,
whisper.cpp, stable-diffusion.cpp + vendored libwebp/libwebm,
mtmd, server-context, cpp-httplib, linenoise, plus the
sqlite3/sqlite-vec amalgamations compiled directly into the target).

Two motivating use cases for a library form:

1. **Embedding chimera in another C++ application.** Today the only
   way to reuse chimera's chat-store / RAG / serve plumbing is to
   spawn the binary and talk HTTP. A `libchimera_core.a` would let a
   host binary call into it directly.
2. **Distributing chimera as a single drop-in archive.** A consumer
   wanting "give me one `.a` to link against" cannot get that today;
   they would have to reproduce the full transitive archive list and
   per-platform whole-archive flags themselves.

These are separable. (1) only requires the `add_library` conversion.
(2) additionally requires the archive-bundling step.

---

## 2. Step 1 - convert chimera to a static library

Mechanically straightforward. Four changes to
`src/chimera/CMakeLists.txt:35`:

1. **Separate `main()` from library code.** `chimera.cpp` currently
   contains both the CLI entry point (CLI11 wiring, signal handlers,
   subcommand dispatch) and reusable library code. Extract the
   `main()` and CLI11 layer into a thin `chimera_main.cpp`. The rest
   of the existing TU list stays unchanged.
2. **Switch the target.** Replace `add_executable(chimera ...)` with
   `add_library(chimera_core STATIC ...)` plus a small
   `add_executable(chimera chimera_main.cpp)` that
   `target_link_libraries(chimera PRIVATE chimera_core)`. All
   existing `target_*` calls move to `chimera_core`.
3. **Promote select properties to `PUBLIC` or `INTERFACE`.** Anything
   a consumer needs visible through chimera's own headers must be
   reachable transitively:
   - SQLite tuning defines (`SQLITE_CORE`, `SQLITE_ENABLE_FTS5`,
     `SQLITE_THREADSAFE=2`, ...) only if a consumer includes
     `sqlite3.h` through a chimera header. Otherwise keep `PRIVATE`.
   - Include directories for any header chimera re-exports.
4. **Hoist the platform link flags to `INTERFACE`.** This is the
   only non-mechanical part. The `--whole-archive` (Linux,
   `src/chimera/CMakeLists.txt:248-251`) and `-force_load` (macOS,
   `src/chimera/CMakeLists.txt:273-299`) blocks exist because ggml's
   CUDA / Metal / Vulkan / etc. backends self-register via static
   initializers - if the linker drops those `.o` members, the
   backend silently disappears at runtime. A static library cannot
   force this on its consumer's behalf. Attach the flags as
   `target_link_options(chimera_core INTERFACE ...)` and the ggml
   archives as `INTERFACE` link libraries so any executable linking
   `chimera_core` inherits the whole-archive wrapping. Same for the
   GNU `--start-group` / `--end-group` block.

After this step:
- `libchimera_core.a` exists as a usable library;
- the existing `chimera` executable still builds, now as a thin
  shim;
- consumers can link against `chimera_core` and inherit the right
  link flags automatically, but they must still provide every
  transitive archive (llama, ggml, whisper, sd, mtmd, server-context,
  httplib, vendored libwebp/libwebm, linenoise).

`WITH_DYLIB` (gate at `src/chimera/CMakeLists.txt:22`) suggests a
sibling shared-library path may already exist in the top-level
build. Check that before duplicating the pattern, so the static and
dynamic paths share whatever target the bulk of the TU list compiles
into.

---

## 3. Step 2 - bundle dependencies into one fat archive

To produce a single drop-in `.a` / `.lib`, the per-platform archive
tools differ. `scripts/combine_archives.py` implements all three in
one stdlib-only script.

### 3.1 Archive inventory

The script hardcodes the same archive list that
`src/chimera/CMakeLists.txt` links, gated by the same per-modality
and per-backend toggles. One **critical** filter: three sibling ggml
builds exist under `build/` (one each from llama.cpp, whisper.cpp,
stable-diffusion.cpp). chimera only links the llama.cpp build (see
top-level `CMakeLists.txt:306-307`). Bundling more than one ggml
guarantees duplicate symbols. The script picks llama.cpp's ggml and
does not look at the other two.

### 3.2 macOS - `libtool -static`

Apple's `libtool` natively merges archive members. Duplicate symbol
names produce a warning, not an error - acceptable because we
already filtered duplicate ggml builds upstream. Handles Mach-O
quirks and fat archives that GNU `ar` mishandles. This is what Xcode
uses internally for static framework builds.

### 3.3 Linux - extract + repack via GNU `ar`

GNU `ar` has two ways to merge archives:

- **MRI script** (`CREATE` + `ADDLIB` + `SAVE`). Simpler, but several
  distros ship a broken thin-archive interaction that produces a
  zero-byte output without error.
- **Extract and repack** (`ar x` each input, then `ar crs` the
  union). Bulletproof, with two footguns the script handles:
  1. `.o` name collisions across archives silently overwrite during
     extraction. Solved by extracting each input into its own
     subdirectory, then renaming each `.o` to a globally-unique
     basename before the final pack.
  2. `ARG_MAX` overflow on large dep sets. Solved by batching `ar
     crs` invocations at 500 files (mode `crs` for the first batch,
     `rs` to append for subsequent batches).

The script uses the extract-and-repack path for both reasons. Final
`ranlib` regenerates the symbol index.

### 3.4 Windows - `lib.exe /OUT:`

MSVC's `lib.exe` natively accepts multiple `.lib` inputs and writes a
merged `.lib`. Must run from an "x64 Native Tools Command Prompt for
VS" (or equivalent `vcvars` invocation) so `lib.exe` is on `PATH`
and links against the matching CRT. No extraction step needed.

---

## 4. What the bundle does NOT do

A bundled static library does **not** relieve consumers of the need
to pass platform link flags. ggml's GPU backends register at static
init time via constructor symbols in `.o` files. Without
`-force_load` (macOS), `--whole-archive` (GNU ld), or
`/WHOLEARCHIVE` (link.exe), the linker drops those `.o` members and
the backend silently disappears at runtime - no error, the backend
just is not in the registry.

The bundle shrinks the consumer's link line from ~15 archives to 1.
It does not change the linker contract. The script prints the
required consumer flag on success as a reminder:

| Platform | Required consumer flag                                                 |
|----------|------------------------------------------------------------------------|
| macOS    | `-Wl,-force_load,libchimera_bundle.a`                                  |
| Linux    | `-Wl,--whole-archive libchimera_bundle.a -Wl,--no-whole-archive`       |
| Windows  | `/WHOLEARCHIVE:chimera_bundle.lib`                                     |

The corollary: if the consumer is itself a static library rather
than an executable, the flag must move further out to *their*
consumer. There is no way to bake "please whole-archive me" into an
archive's own metadata.

---

## 5. Open questions

1. **Should `chimera_main.cpp` move out of `src/chimera/`?** Pure
   layering would put `chimera_core` under `src/chimera/` and
   `chimera_main.cpp` under `src/chimera_cli/`. Probably overkill for
   one file; flag it if the CLI grows.
2. **What is `WITH_DYLIB` already doing?** If a shared-library path
   exists, the new static path should share its source list. Audit
   before implementing, not after.
3. **Should the sqlite3 / sqlite-vec / server-http amalgamations
   stay inside `libchimera_core.a`?** Fine for single-consumer use.
   If a consumer also links its own SQLite or its own llama-server,
   they will hit duplicate-symbol errors. If multi-consumer use is
   expected, split each into an `OBJECT` library so consumers can
   opt out.
4. **Do we want a CMake `install()` target that calls
   `combine_archives.py` as a post-build step?** Or keep it a
   manual `python3 scripts/combine_archives.py` invocation?

---

## 6. Validation plan

After Step 1 lands:

- `make build` still produces a working `chimera` binary (same
  CLI, same routes, same tests pass).
- A trivial external CMake project can `find_library(chimera_core)`,
  link it, and call one public function (e.g. open a chat store, run
  one embed). This proves the include / link / INTERFACE-flag
  contract.

After Step 2 lands:

- On each of macOS / Linux / Windows: `python3
  scripts/combine_archives.py` produces `libchimera_bundle.{a,lib}`.
- The same trivial external project can link only the bundle (plus
  the consumer flag from the table above) and pass the same smoke
  test. This proves the bundle is symbol-complete.
- A second test that exercises a GPU backend (Metal on macOS, CUDA
  on Linux if available) confirms the whole-archive flag is doing
  its job.
