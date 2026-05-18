# Splitting chimera into three static libraries

This document captures the design and current state of the work that
turned chimera into three redistributable static libraries:

- `libchimera.a` - chimera's own code, produced by `make build`.
- `libchimera_thirdparty.a` - the third-party C++ stack (llama,
  llama-common, mtmd, server-context, cpp-httplib, whisper, sd,
  vendored libwebp / libwebm, linenoise) merged into one archive,
  produced by `scripts/combine_archives.py`. Normal-linked.
- `libchimera_ggml.a` - just the ggml core + per-backend ggml
  archives, also produced by `scripts/combine_archives.py`.
  **Must be whole-archived** by the consumer or GPU backends silently
  fail to register at runtime.

The split between the two deps archives is not cosmetic - it is
forced by the linker contract. See §4 for the duplicate-symbol
analysis that made a single deps archive untenable.

Status: all three archives are implemented. An external smoke test
under `tests/external/` links the three archives as a non-CMake
consumer would and runs successfully on macOS. Linux and Windows
paths exist in the script but have not been exercised on those
platforms.

---

## 1. Scope

This work is for **host-optimized static builds**. Each install
picks one ggml backend matched to the local hardware (Metal on Apple
Silicon, CUDA for one specific NVIDIA toolkit + arch, etc.) and
links it statically. `GGML_NATIVE=1` is the expected configuration.

Explicit non-goal: pre-built redistributable binaries that work
across machines (the cyllama wheel-distribution problem). There is
no runtime backend dispatch, no fat binary covering many GPU
generations, no PyPI-shaped portability. A bundle built on machine
A is shippable to machine B only if B is bit-identical for the
relevant ISA extensions, CUDA toolkit, and driver.

The corollary: anyone else builds from source via `manage.py build`
plus `combine_archives.py`. This is the same contract as a
`--prefix=/opt/local`-style native package.

---

## 2. Why

Today (well, before Step 1 landed) `src/chimera/CMakeLists.txt`
produced a single executable that statically linked ~15 archives
(llama.cpp, ggml + per-backend ggml-*, whisper.cpp, sd.cpp +
vendored libwebp/libwebm, mtmd, server-context, cpp-httplib,
linenoise, plus the sqlite3/sqlite-vec amalgamations compiled
directly into the target).

Two motivating use cases for a library form:

1. **Embedding chimera in another C++ application.** Previously the
   only way to reuse chimera's chat-store / RAG / serve plumbing was
   to spawn the binary and talk HTTP. A `libchimera.a` lets a host
   binary call into it directly through the `command_serve` /
   `command_whisper` / `command_sd` entry points already declared in
   `chimera.h`.
2. **Distributing chimera with a manageable link line.** A consumer
   previously had to reproduce the full transitive archive list
   (~19 archives) and per-platform whole-archive flags themselves.
   Reducing the deps to two grouped archives shrinks the link line
   to three entries with one whole-archive wrapper on the ggml
   group.

These are separable. (1) only requires the `add_library`
conversion. (2) additionally requires the deps-bundling step.

Two deliberate non-choices:

- We do NOT merge `libchimera.a` into either deps archive. Doing
  so would force a full deps rebuild on every chimera edit, and
  create a "did I link both and double-define" footgun.
- We do NOT merge `libchimera_ggml.a` and `libchimera_thirdparty.a`
  into one. A single deps archive forced to whole-archive would
  pull duplicate definitions of common helpers (e.g. `trim()`,
  defined independently in server-context's `util.cpp` and
  whisper-common's `common.cpp`) and the consumer link would fail.
  See §4.1.

---

## 3. Step 1 - convert chimera to a static library (DONE)

The split point was easier than expected: `chimera.cpp` is the CLI
shell (CLI11 wiring, REPL, signal handlers, subcommand dispatch),
and every other TU exposes its functionality through `chimera.h` as
the library API. So no source-splitting was required - `chimera.cpp`
becomes the executable's only source, and every other TU goes into
the library.

`src/chimera/CMakeLists.txt` was rewritten to produce two targets:

| CMake target | Output filename       | Purpose                                                                 |
|--------------|-----------------------|-------------------------------------------------------------------------|
| `chimera_lib` | `libchimera.a`       | Static library. `OUTPUT_NAME chimera` makes the file `libchimera.a` despite the internal target name. |
| `chimera`    | `chimera` (or `.exe`) | CLI executable. Just `chimera.cpp` linked against `chimera_lib`.        |

Internal target names need to be unique inside CMake but the output
filenames do not collide: `libchimera.a` and `chimera` are distinct
on every platform.

Key property changes vs. the old executable-only target:

- **Link libraries promoted to `PUBLIC`.** Anything `chimera_lib`
  links against (every transitive archive: llama, ggml, whisper, sd,
  mtmd, server-context, httplib, vendored libwebp/libwebm,
  linenoise, system libs) propagates to consumers. The executable
  inherits this automatically; an external consumer of
  `libchimera.a` also does, so they do not have to reproduce the
  archive list.
- **Link options promoted to `INTERFACE`.** The platform-specific
  `--whole-archive` (Linux) / `-force_load` (macOS) / (eventually)
  `/WHOLEARCHIVE` (Windows) wrappers exist because ggml's GPU
  backends self-register via static initializers in `.o` files - if
  the linker drops those members, the backend silently disappears
  at runtime. A static library cannot force these flags on itself,
  but `INTERFACE` propagation makes every consumer inherit them.
- **Modality / linenoise / version defines promoted to `PUBLIC`.**
  Compiled into the library and visible through `chimera.h` to
  consumers, so `chimera info` rendering and the `CHIMERA_HAS_*`
  gates work the same from inside chimera and from external code.
- **SQLite, webui, and payload-cap defines stayed `PRIVATE`.** Only
  the amalgamation TUs and server-http.cpp (all inside the library)
  read them.

The executable target is gated on a new
`option(CHIMERA_BUILD_EXECUTABLE ON)`. A downstream consumer who
only wants the library configures with `-DCHIMERA_BUILD_EXECUTABLE=OFF`
to skip the `chimera.cpp` TU and CLI11 link.

Dead code removed: the `WITH_DYLIB` sentinel in the top-level
`CMakeLists.txt` and the corresponding early-return guard at the
top of `src/chimera/CMakeLists.txt`. It was a vestige from sharing
the slice with cyllama and never fired in chimera's tree.

**Verified:** `make build` produces both artifacts (4.2 MB
`libchimera.a`, 34 MB `chimera`); `make test` passes 44/44.

---

## 4. Step 2 - bundle dependencies into two grouped archives (DONE)

`scripts/combine_archives.py` produces two output archives per run:
`libchimera_thirdparty.a` (normal-link group) and
`libchimera_ggml.a` (whole-archive group). The per-platform archive
tool differs (libtool on macOS, ar on Linux, lib.exe on Windows);
the inventory and the split are platform-independent.

### 4.1 Why two archives, not one

The naive design was one merged `libchimera_deps.a`. The chimera
executable build force-loads only the ggml archives (see
`src/chimera/CMakeLists.txt:273-299` for the macOS block); every
other dep is normal-linked, so the linker prunes unused members
before they collide.

Once merged into a single archive, the consumer has two choices,
both bad:

- **Whole-archive the whole thing.** Pulls in duplicate symbols
  from helpers that exist in two independent upstream codebases.
  Concretely, both `tools/server/util.cpp` (in server-context) and
  `examples/common.cpp` (in whisper-common) define a free `trim()`
  function; whole-archive forces both `.o` members in, and the
  link fails with `duplicate symbol 'trim(...)'`. Other latent
  duplicates may exist.
- **Normal-link the whole thing.** ggml backend constructors live
  in `.o` files that have no externally-referenced symbols, so the
  linker drops them, and `ggml_backend_dev_count()` returns 0 at
  runtime. The backend silently disappears.

Splitting along the whole-archive boundary makes both groups
self-consistent: the thirdparty group is normal-linked (so latent
duplicates stay pruned), the ggml group is whole-archived (so
backend initializers run). This mirrors the contract the chimera
executable build already has.

### 4.2 Archive inventory

The script hardcodes the same transitive archive list that
`src/chimera/CMakeLists.txt` links, gated by the same per-modality
and per-backend toggles. `libchimera.a` itself is deliberately
excluded from both groups.

| Group | Contents |
|-------|----------|
| `libchimera_thirdparty.a` | libllama, libllama-common, libmtmd, libserver-context, libcpp-httplib, libwhisper + libwhisper-common (if SD/whisper on), libstable-diffusion + libwebp/libsharpyuv/libwebpmux/libwebm (if SD on), liblinenoise (if on) |
| `libchimera_ggml.a` | libggml, libggml-base, libggml-cpu, plus libggml-blas / libggml-metal / libggml-cuda / libggml-vulkan / libggml-hip / libggml-sycl / libggml-opencl as enabled |

One **critical** filter: three sibling ggml builds exist under
`build/` (one each from llama.cpp, whisper.cpp, sd.cpp). chimera
only links the llama.cpp build (top-level `CMakeLists.txt:306-307`).
Bundling more than one ggml guarantees duplicate symbols. The
script hardcodes the llama.cpp ggml path and ships a runtime
assertion (`Inventory.assert_single_ggml`) that fails loudly if a
future edit accidentally inserts a whisper or sd ggml path.

### 4.3 macOS - `libtool -static`

Apple's `libtool` natively merges archive members. Duplicate symbol
names produce a warning, not an error - acceptable because we
already filtered duplicate ggml builds upstream. Handles Mach-O
quirks and fat archives that GNU `ar` mishandles. This is what
Xcode uses internally for static framework builds.

Smoke-tested end-to-end on macOS with the current build: produces a
45.3 MB `libchimera_thirdparty.a` (14 inputs) plus a 2.9 MB
`libchimera_ggml.a` (5 inputs, with Metal + BLAS).

### 4.4 Linux - extract + repack via GNU `ar`

GNU `ar` has two ways to merge archives:

- **MRI script** (`CREATE` + `ADDLIB` + `SAVE`). Simpler, but
  several distros ship a broken thin-archive interaction that
  produces a zero-byte output without error.
- **Extract and repack** (`ar x` each input, then `ar crs` the
  union). Bulletproof, with two footguns the script handles:
  1. `.o` name collisions across archives silently overwrite
     during extraction. Solved by extracting each input into its
     own subdirectory, then renaming each `.o` to a globally-unique
     basename before the final pack.
  2. `ARG_MAX` overflow on large dep sets. Solved by batching `ar
     crs` invocations at 500 files (mode `crs` for the first batch,
     `rs` to append for subsequent batches).

The script uses the extract-and-repack path for both reasons. Final
`ranlib` regenerates the symbol index.

### 4.5 Windows - `lib.exe /OUT:`

MSVC's `lib.exe` natively accepts multiple `.lib` inputs and writes
a merged `.lib`. Must run from an "x64 Native Tools Command Prompt
for VS" (or equivalent `vcvars` invocation) so `lib.exe` is on
`PATH` and links against the matching CRT. No extraction step
needed.

Note: on Windows the archives are `chimera.lib`,
`chimera_thirdparty.lib`, `chimera_ggml.lib` (no `lib` prefix),
unlike Unix's `libchimera.a` / `libchimera_thirdparty.a` /
`libchimera_ggml.a`. The script handles this by selecting the right
basename based on the target platform.

---

## 5. What the deps archives do NOT do

The deps archives do **not** relieve consumers of the need to pass
platform link flags. ggml's GPU backends register at static init
time via constructor symbols in `.o` files. Without `-force_load`
(macOS), `--whole-archive` (GNU ld), or `/WHOLEARCHIVE` (link.exe)
wrapping `libchimera_ggml.a`, the linker drops those `.o` members
and the backend silently disappears at runtime - no error, the
backend just is not in the registry.

The deps archives shrink the consumer's link line from ~19 archives
to 3 (`libchimera.a` + `libchimera_thirdparty.a` +
`libchimera_ggml.a`). They do not change the linker contract.

The whole-archive wrapper applies **only** to `libchimera_ggml.a`.
Wrapping `libchimera_thirdparty.a` triggers the duplicate-symbol
errors described in §4.1; `libchimera.a` does not contain backend
initializers and does not need it.

The script prints the required consumer link line on success:

| Platform | Consumer link line                                                                                                            |
|----------|-------------------------------------------------------------------------------------------------------------------------------|
| macOS    | `libchimera.a libchimera_thirdparty.a -Wl,-force_load,libchimera_ggml.a`                                                      |
| Linux    | `libchimera.a libchimera_thirdparty.a -Wl,--whole-archive libchimera_ggml.a -Wl,--no-whole-archive`                           |
| Windows  | `chimera.lib chimera_thirdparty.lib /WHOLEARCHIVE:chimera_ggml.lib`                                                           |

The corollary: if the consumer is itself a static library rather
than an executable, the flag must move further out to *their*
consumer. There is no way to bake "please whole-archive me" into an
archive's own metadata.

---

## 6. Open questions

1. **Should `chimera.cpp` move out of `src/chimera/`?** Pure
   layering would put the CLI under `src/chimera_cli/`. Probably
   overkill for one file; flag it if the CLI grows.
2. **Should the sqlite3 / sqlite-vec / server-http amalgamations
   stay inside `libchimera.a`?** Fine for single-consumer use. If a
   consumer also links its own SQLite or its own llama-server, they
   will hit duplicate-symbol errors. If multi-consumer use becomes
   real, split each into an `OBJECT` library so consumers can opt
   out.
3. **Do we want a CMake `install()` target that calls
   `combine_archives.py` as a post-build step?** Or keep it a
   manual `python3 scripts/combine_archives.py` invocation?

---

## 7. Validation plan

Step 1 (done):

- `make build` produces a working `chimera` binary (same CLI, same
  routes). [PASSING]
- `make test` passes 44/44. [PASSING]
- A trivial external CMake project can `find_library(chimera)`,
  link it, and call one public function (e.g. open a chat store,
  run one embed). [NOT YET EXERCISED]

Step 2:

- `python3 scripts/combine_archives.py` on macOS produces a 45.3 MB
  `libchimera_thirdparty.a` plus a 2.9 MB `libchimera_ggml.a`.
  [PASSING]
- Equivalent run on Linux. [NOT YET EXERCISED]
- Equivalent run on Windows from a VS dev prompt. [NOT YET EXERCISED]
- `tests/external/` is an external CMake project that links the
  three archives as a non-CMake consumer would, calls
  `ggml_backend_dev_count()`, `llama_print_system_info()`, and a
  chimera helper, and prints PASS. On macOS this reports
  `ggml_backend_dev_count = 3` (CPU + Metal + Accelerate),
  confirming the ggml whole-archive contract is working end-to-end.
  [PASSING on macOS, NOT YET EXERCISED on Linux/Windows]
- Inference probe: setting `CHIMERA_SMOKE_MODEL=<path/to/.gguf>`
  before running the smoke test triggers a full model load +
  tokenize + `llama_decode` round-trip with finite-output and
  non-zero-magnitude assertions. Verified with granite-4.0-h-tiny
  on macOS: 100,352 logits produced, all finite, max|x|=11.55.
  Proves the linked library actually computes, not just that
  symbols resolve. [PASSING on macOS]
