# Changelog

All notable changes to chimera will be documented in this file. Format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.15]

### Fixed

- **`-v` did not reach stable-diffusion, leaving every sd run diagnosable only by its errors.** `sd_log_callback()` hard-filtered at `SD_LOG_WARN`, and sd.cpp installs no printer of its own (`log_printf()` hands every level to the callback and returns), so `chimera sd` dropped the whole INFO/DEBUG tier no matter how it was invoked. That tier is where the lines that explain a low-VRAM failure live: `Found N backend devices`, `total params memory size = ... (VRAM x, RAM y)`, and the `model_manager` `prepared params backend buffer` / `staged compute params to CUDA0` / `releasing compute params` sequence. Without them an OOM from weights pinned to the compute backend -- sd's default, since `SDBackendManager::params_backend()` falls back to `runtime_backend()` when no `--params-backend` spec is given -- is indistinguishable from one caused by a placement flag that did not take effect, and the two have opposite fixes. The threshold is now settable (`chimera_set_sd_log_verbose()`), and `-v` drops it to `SD_LOG_DEBUG`. Warnings and errors are unaffected when the flag is absent.

### Changed

- **Global flags now bind after the subcommand as well as before it.** `app.fallthrough()`, so `chimera sd -v ...` works alongside `chimera -v sd ...`; previously the former exited with `The following argument was not expected: -v`. Subcommands bind their own options first, so fallthrough only catches what would otherwise have been an error.

## [0.2.14]

Audit of chimera against the sibling cyllama (`0.4.0`--`0.4.2`) and inferna (`0.2.0`) releases, which fixed a set of defects in the same shared upstreams. Most of what those projects fixed chimera either fixed first (`GGML_MAX_NAME`, `0.2.12`) or never had (the placement flags were never routed onto `max_vram`), but five carried over. Two are sd.cpp defects chimera cannot reach from its own code, which is what the new patch machinery is for; see Added.

### Fixed

- **The DRY sampler was disabled outright, and `--repeat-last-n -1` silently disabled the repeat penalty.** llama.cpp `v0.3.0` removed the `-1 = context size` sentinel from `llama_sampler_init_penalties()` and `llama_sampler_init_dry()`: a negative window is now `std::max(n, 0)`, and `0` is each sampler's *disabled* value (`dry_enabled` explicitly requires `!= 0`). Upstream resolved `-1` one layer up and moved both `common_params_sampling` defaults to `64` when it dropped the sentinel; chimera kept `dry_penalty_last_n = -1`, so every DRY invocation since `0.2.13` has been a no-op, and `--repeat-last-n -1` -- documented as "scan the whole context" -- turned the repeat penalty off. `make_sampler()` now resolves a negative window to the context it is about to create (`opts.n_ctx`, falling back to the model's training context) before handing it to `common_sampler_init`, so both flags mean what their help text says. `0` still disables. Nothing in the pin-check machinery can catch this class: the fields kept their names and types, only the meaning of one value changed.

- **`_sync_ggml_abi()` left stale objects behind, so a llama.cpp bump could link two ggml generations together.** It replaces stable-diffusion.cpp's vendored `ggml/` with llama.cpp's, but left `build/stable-diffusion.cpp/build` in place -- and `copytree` preserves mtimes, so the incoming sources are never newer than the objects already built from the tree just replaced and make sees nothing to redo. SD then compiles against the new ggml and links objects built from the old one. The cmake tree is now dropped along with the swap. To avoid paying a full SD rebuild on every single build, the swap is skipped entirely when the ggml already in place is the one that would be copied: `_ggml_tree_stamp()` fingerprints the source tree (relative path, size, mtime) and the result is recorded in the destination, so only a real change -- a bump, or a `ggml-*.patch` landing on llama.cpp's copy -- re-copies and invalidates.

- **A failed stable-diffusion text-encoder graph aborted the process** (`scripts/patches/stable-diffusion.cpp-conditioner-compute-failure.patch`). `GGMLRunner::compute()` reports failure as an empty `std::optional`, `take_or_empty()` flattens it to an empty tensor, and the conditioners `GGML_ASSERT(!hidden_states.empty())` on the result, so any budget that pushed the text encoder itself through the graph-cut segmented path (e.g. `--max-vram 3.5`) killed `chimera sd` -- and killed the whole server on a `chimera serve` image route rather than failing one request. Several of `compute()`'s failure paths also log nothing, so the abort arrived with no cause. The patch logs the dropped failure and propagates it through the LLM conditioner's existing error channel, so generation fails cleanly.

- **`--max-vram` budgets ignored VRAM that was already in use** (`scripts/patches/stable-diffusion.cpp-graph-cut-budget-clamp.patch`). The graph-cut planner clamps its budget to the VRAM actually free at plan time, but only under `--stream-layers`. Modules are budgeted once at init from *free* VRAM and a params storage block is only reclaimed when its tensors are disk-backed, so by the time the diffusion model plans its graph the text encoder's weights are still resident and several GiB of the init-time budget no longer exist. The patch ungates the clamp so each module plans against what is genuinely free.

- **The MSVC stable-diffusion build lost `/EHsc`.** `/bigobj` (needed since `src/stable-diffusion.cpp` crossed the COFF 65,279-section limit at master-795) was supplied as `-DCMAKE_CXX_FLAGS=/bigobj`, mirroring upstream's own workflow. A command-line `-D` pre-seeds the cache, so CMake's platform init never runs and that *replaces* the MSVC defaults instead of appending, silently dropping `/EHsc` and `/GR` from every SD translation unit. `/bigobj` now goes into sd.cpp's own `if (MSVC)` block next to the `/MP` and `/utf-8` it already sets (`scripts/patches/stable-diffusion.cpp-msvc-bigobj.patch`). chimera's own TUs were never affected -- `CMakeLists.txt` sets the full option list.

- **Metal shader compilation depended on the host binary, not the machine** (`scripts/patches/ggml-metal-pin-msl-version{,-perkind}.patch`). ggml compiles its embedded shader library with a bare `[MTLCompileOptions new]`, so Metal derives the MSL version from the SDK the *host process* linked against rather than from the running OS. Below MSL 3.1 the library fails to compile outright (`no matching constructor for initialization of 'threadgroup metal::half4x4[512]'`) and the Metal backend never initializes; below 3.1 the bf16 kernels are also `#undef`'d while `props.has_bfloat` stays true. Both patches install the same `@available` ladder (3.2 on macOS 15+, 3.1 on 14+, 3.0 on 13+), deliberately stopping below MSL 4.0 so the Metal 4 tensor kernels stay off. They differ only in context: `-perkind` matches llama.cpp `v0.3.0`'s per-kind Metal library split, the plain one matches whisper.cpp `v1.9.2`; exactly one applies per tree and the other self-disables.

### Added

- **Source-patch machinery** (`GgmlBuilder._apply_source_patches()`). `scripts/patches/*.patch` are applied to the cloned trees on every build, `ggml-*.patch` against all three and `<project>-*.patch` against the matching upstream. Each is applied with `git apply -p1` and is idempotent and self-disabling: already-applied and no-longer-applies are logged and skipped, never fatal, so a version bump that lands an equivalent upstream fix does not break the build -- but it also stops fixing silently, so the build log is worth reading after a bump. `scripts/patches/README.md` carries the analysis for each. This covers failures the wrapper cannot reach at all: a `GGML_ASSERT` that aborts the process, and a compile flag that has to be set inside upstream's own `if (MSVC)` block. It replaces nothing -- the hand-rolled `_patch_server_http_payload_cap()` search-replace stays as it is.

- **`--backend`, `--params-backend` and `--auto-fit` (`sd`), mirrored as `--sd-*` on `serve`.** The first two pass `sd_ctx_params_t::backend` / `::params_backend` through verbatim: `backend` assigns *compute* per module, `params_backend` assigns *weight residency* and additionally accepts `cpu` and `disk`. Both take a bare target (`cuda0`, `cpu`) or comma-separated per-module assignments (`diffusion=cuda0,te=cpu`), with upstream's module keys. They are appended after whatever `--offload-to-cpu` / `--clip-on-cpu` / `--vae-on-cpu` / `--control-net-cpu` compose, because sd's parser is last-wins per key -- so an explicit spec overrides the coarse booleans rather than the other way round. Neither is validated chimera-side: sd parses them and reports its own errors, and mirroring the module list would add a second place to drift. `--auto-fit` sets `sd_ctx_params_t::auto_fit`, letting sd derive both specs from the model sizes and the available VRAM; it fits where an explicit placement would not, but on a single-GPU box it tends to resolve to all-CPU, so prefer an explicit placement when one is known.

- **`--load-mode auto` and `--load-mode mmap+mlock`.** llama.cpp `v0.3.0` reshaped `llama_load_mode`: it added `LLAMA_LOAD_MODE_AUTO` (-1, and upstream's new default -- the loader picks a strategy from the device's capabilities) and split the old combined value into a plain `MLOCK` (lock *without* mmap) and `MMAP_MLOCK`, moving `DIRECT_IO` from 3 to 4. chimera names the enumerators rather than their values, so nothing broke at compile time and only the parser needed widening. `--mlock` keeps selecting plain `MLOCK`, which is what upstream's own deprecated-flag shim does; `--load-mode mmap+mlock` is the pre-`v0.3.0` meaning.

### Changed

- **`SDCPP_VERSION` gains a ceiling note.** From master-817 on, stable-diffusion.cpp calls `ggml_mul_mat_i8_tensorwise` and `ggml_quantize_i8_convrot`, which exist only in leejet's ggml fork; chimera compiles SD against llama.cpp's ggml, where they are undeclared, so every SD translation unit fails from 817 on. `master-816-487de75` is the last commit that builds. Observed by cyllama at its `0.4.1` and recorded here so the next bump does not walk into it; not re-verified against 817 directly.


## [0.2.13]

### Changed

- **Update llama.cpp to v0.3.0** (from b10107), whisper.cpp to **v1.9.2** (from v1.9.1), and stable-diffusion.cpp to **master-816-487de75** (from master-795-87a0177). llama.cpp has moved from `bNNNN` build tags to semver, so `LLAMACPP_VERSION` changes shape with it. whisper.cpp and sd.cpp built clean; every adaptation below is llama.cpp's.

### Fixed

- **Adapt to llama.cpp replacing `nlohmann::json` with `common_json`.** `json_schema_to_grammar` takes the new type, so `--json-schema` parses through `common_json::parse`. Separately, `server-common.h` now declares a global `using json = common_json`, which is ambiguous against `chimera_serve::json` inside `command_serve` -- that function pulls the namespace in with a `using`-directive, so both names land in one scope. The one affected site is qualified rather than renaming chimera's alias, which is used throughout the serve TUs.

- **Adapt to `common_chat_params::thinking_end_tag` becoming `thinking_end_tags`,** and `common_params_sampling::reasoning_budget_end` becoming a list of token sequences. chimera propagates the whole list instead of taking the first tag, so a template advertising several end tags still stops on all of them; the first also forms the forced sequence, as upstream specifies. `ReasoningBudgetParams::thinking_end_tag` is renamed and retyped to match, which is source-breaking for any external consumer that set it.

- **Link llama.cpp's new `vendor-hash` archive.** `hash_sha256_hex`, which backs mtmd-helper's content-addressed bitmap ids, moved out of `libmtmd.a` into its own `libvendor-hash.a`; without it the final link fails on an undefined symbol. Added to the `manage.py` target list and copy step, all three link paths in `src/chimera/CMakeLists.txt`, and `combine_archives.py` so the redistributable `libchimera_thirdparty.a` carries it too.

- **Resolve `cpuparams.n_threads` before handing `common_params` to llama.cpp.** `common_init_from_params()` now always builds a threadpool, and ggml sizes the worker array directly from `n_threads`. chimera fills `common_params` by hand and so never runs the arg-parser postprocess where upstream turns the `-1` sentinel into a real count; ggml computed a ~2^64-byte allocation, it returned null, and the `memset` of that null segfaulted `chimera serve` before it logged a single line. Both construction sites in `chimera_serve.cpp` now call upstream's own `postprocess_cpu_params()`. The image routes failed the same way, reporting `attempted to allocate 17592186044416.00 MB`. Nothing in the pin-check machinery can catch this class: the field was already there and still compiles, it just started being read.

### Testing

- `make test` 69 pass / 0 fail / 6 skip. `make test-external-smoke` and `make bump-check` clean.

- **Not covered:** CUDA, ROCm, SYCL and Vulkan were not built, and `CHIMERA_WEBUI_EMBED=ON` was not exercised.

## [0.2.12]

### Fixed

- **`GGML_MAX_NAME` had diverged across the shared-ggml link, corrupting the heap on every SD run.** The macro sizes a field inside `struct ggml_tensor`, so it sets `sizeof(ggml_tensor)` and the offset of every field after `name`. llama.cpp and chimera compiled at 128, whisper.cpp at the 64 default, sd.cpp at 160; sd.cpp then wrote past the end of tensors ggml had allocated. All four now compile with one value. sd.cpp's own guard is a `>=` minimum, so nothing failed at build time, and the corruption only manifested when an unrelated `malloc` walked the damaged free list.

- **Apply `GGML_MAX_NAME` to the CUDA and HIP languages, not just C/C++.** `CMAKE_C_FLAGS` / `CMAKE_CXX_FLAGS` do not reach `.cu` files, so `ggml-cuda` disagreed with `ggml-base` in the same build. This predates the SD divergence and was present in prior releases. The HIP leg is reasoned-correct but was not exercised on ROCm hardware.

- **Pin whisper.cpp's `GGML_MAX_NAME` at all.** Its builder never passed the define. No whisper failure was ever observed, so this closes a latent defect rather than fixing an observed one.

- **Fetch stable-diffusion.cpp before reading `GGML_MAX_NAME` off it.** On a clean tree the value was read before the SD clone existed and silently fell back to a constant, which would hand llama.cpp and whisper.cpp a stale value the first time upstream moves it -- reintroducing the ABI split this machinery exists to prevent. Both fallback paths now warn.

### Changed

- **Derive `GGML_MAX_NAME` from stable-diffusion.cpp instead of pinning it.** The value belongs to upstream SD, which has moved it before, and a stale copy fails silently at runtime rather than loudly at build time. Configure reports the resolved value so a future divergence shows in the build log. `GGML_MAX_NAME_FALLBACK = 160` covers only an unreadable checkout. Both parsers are inert under `SD_USE_VENDORED_GGML=1`, where no agreement is required.

  To audit agreement across a build tree, every `flags.make` must report the same number:

  ```text
  find build/{llama.cpp,whisper.cpp,stable-diffusion.cpp}/build build/src \
       -name flags.make | xargs grep -ho 'GGML_MAX_NAME=[0-9]*' | sort -u
  ```

### Testing

- Verified on the Z-Image Turbo case that exposed it (`scripts/case/z_turbo.sh`), CUDA and Vulkan, four runs each to rule out nondeterminism. `make test` unchanged at 68 pass / 0 fail / 7 skip on both.

- Every `flags.make` across all four trees reports the same value. The one legitimate exception is `vulkan-shaders-gen`, a host GLSL compiler that never includes `ggml.h`.

- **Not covered:** ROCm/HIP and SYCL were not built.

## [0.2.11]

### Changed

- **Update llama.cpp to b10107** (from b9979) and stable-diffusion.cpp to **master-795-87a0177** (from master-775-b5d8120); whisper.cpp stays at **v1.9.1**. Both ranges broke the build the same way: a pair of booleans on a params struct collapsed into one richer field. The pin-check `static_assert`s fired alongside the call sites, which is what they exist for.

### Fixed

- **Adapt to llama.cpp replacing `use_mmap` / `use_mlock` with `llama_load_mode`.** chimera keeps its two booleans -- they are what `--no-mmap` and `--mlock` bind to -- and translates at the llama boundary in a new leaf header, `chimera_llama_load_mode.h`, kept dependency-light so `chimera_embed.cpp` can share it.

- **Adapt to sd.cpp folding the reference-image booleans into `ref_image_args`.** chimera keeps `--increase-ref-index` and `--no-auto-resize-ref-image` and composes the string upstream's way. The flags are appended after the caller's passthrough string because sd resolves repeated keys last-wins, and the result is left empty when nothing is set so the architecture preset's defaults stand.

- **Restore the Windows build: compile sd.cpp with `/bigobj`.** master-795 pushed `stable-diffusion.cpp` past MSVC's 65535-section cap. Upstream works around this in its own workflow rather than its CMakeLists, so every downstream consumer inherits the break. GCC and Clang have no equivalent limit, which is why it was Windows-only.

- **Pre-empt the same limit in chimera's own MSVC build.** `/bigobj` added to the MSVC flag list for chimera's C++ TUs. `tests/external/` deliberately does not get the flag: it models an ordinary downstream link, so if `chimera.hpp` ever needs it, that test should reveal it.

- **Quote multi-token CMake flag values in `manage.py`.** Values containing whitespace were split by the shell into a broken `-D` argument.

### Added

- **`--load-mode` (gen / chat / embed).** Accepts `none | mmap | mlock | dio`, naming `llama_load_mode` directly and exposing `dio`, which chimera's booleans cannot reach. An explicit `--load-mode` overrides `--no-mmap` / `--mlock`; precedence is fixed rather than positional because the booleans and the enum live in separate option fields. Upstream's `-lm` short form is not mirrored -- CLI11 rejects multi-character short names.

- **`--ref-image-args` (`sd`).** Passes `sd_img_gen_params_t::ref_image_args` through verbatim, opening the reference-image surface master-795 added. Not validated chimera-side: sd warns and continues on unknown keys, so a typo degrades rather than aborts, and mirroring the key list would add a second place to drift.

### Testing

- Model-free probes pin both translation layers: a truth table for `chimera_llama_load_mode()` and one for `chimera_sd::build_ref_image_args()` in `tests/external/smoke.cpp`, plus seven CLI-level checks in `scripts/test.py`. The sd composition was extracted into a pure function specifically so it could be tested without a diffusion model.

## [0.2.10]

### Changed

- **Update llama.cpp to b9979** (from b9804), whisper.cpp to **v1.9.1** (from v1.8.6), and stable-diffusion.cpp to **master-775-b5d8120** (from master-721-8caa3f9). Whisper built clean. The other two each required one adaptation, both caught by the build and tests rather than by a pre-flip header audit.

### Fixed

- **Adapt to sd.cpp resignaturing `generate_image`** to return `bool` with the image array and count as out-params. The copy loop now iterates the returned count instead of the requested batch count.

- **Adapt to mtmd making `mtmd_input_text::text_len` structural.** `mtmd_tokenize` no longer falls back to `strlen`, so chimera's never-populated `text_len` left the marker scan empty and the vision pipeline returned nothing. Both call sites now set it.

## [0.2.9]

### Changed

- **Update llama.cpp to b9804** (from b9741) and stable-diffusion.cpp to **master-721-8caa3f9** (from master-709-92a3b73); whisper.cpp stays at v1.8.6. Both built clean -- the upstream surface chimera links against was unaffected. Deferred as unwrapped: `llama_model_n_layer_nextn()`, the additive mtmd load-progress callback, and the `/models/sse` progress feed (tied to upstream's model downloader, which chimera does not use). sd.cpp's new `LOGIT_NORMAL_SCHEDULER` is already reachable because `--scheduler` delegates to upstream's parser.

### Added

- **`--eager-load` (`sd`) / `--sd-eager-load` (serve).** sd.cpp defaults to lazy weight residency, so the first generation pays a warmup cost. The flag loads every weight up front, trading load time for steady-state latency from the first request. The useful case is `chimera serve`; for one-shot `chimera sd` it only shifts when the cost is paid.

### Changed (docs)

- `docs/dev/cli-api-coverage.md` gains rows for `--eager-load` and the previously-undocumented `--stream-layers`. Separately, `CHIMERA_WEBUI_EMBED=ON` was re-verified on this pin (`make test` only ever builds the default `OFF`): the bump shifted upstream's Vite output to `tools/ui/dist/`, which manage.py's probe list already covered.

## [0.2.8]

### Changed

- **Update llama.cpp to b9741** (from b9631) and stable-diffusion.cpp to **master-709-92a3b73** (from master-700-c2df4e1); whisper.cpp stays at v1.8.6. Both required chimera-side changes; all three were caught at compile time.

### Fixed

- **Adapt to sd.cpp retyping `sd_ctx_params_t::max_vram` from `float` to `const char *`.** The field now also accepts a per-backend assignment spec. chimera's `--max-vram` stays a float GiB budget and is formatted at the boundary, so there is no behavior change and no new syntax exposed.

- **Drop the removed `common_params::webui` alias.** `--no-webui` now sets only `params.ui`.

- **Adapt to `common_params_model::name` becoming the `get_name()` accessor.** This is field-to-accessor drift, which `decltype` pin-checks cannot guard, so it surfaced as a plain compile error.

### Changed (docs)

- `docs/dev/maintenance.md` documents the field-to-accessor pin-check blind spot: a pin can assert a member's type but not that it stayed a data member. The `bump-check` `common.h` diff is the catch, and the PR template gains a matching pre-bump audit bullet.

## [0.2.7]

### Changed

- **Update llama.cpp to b9631** (from b9592) and stable-diffusion.cpp to **master-700-c2df4e1** (from master-685-19bdfe2); whisper.cpp stays at v1.8.6. The llama.cpp range was audited by diffing upstream headers between the two refs directly -- `make bump-check` only confirms the vendored headers match the currently-pinned ref, so it cannot report drift since the previous pin. Only sd.cpp required code changes.

### Fixed

- **Adapt to sd.cpp's `sd_ctx_params_t` offload rework.** The per-component offload booleans became backend-assignment spec strings. chimera's existing offload flags now translate into those strings the way the upstream example CLI does, so the public knobs are unchanged. `vae_decode_only` has no replacement -- upstream always builds a full encode+decode VAE -- so the field is retained as an accepted no-op.

- **Repair `CHIMERA_WEBUI_EMBED=ON`, broken by the llama.cpp bump.** Upstream's asset-embed helper switched from a fixed list of name/path pairs to a recursive directory, and the prebuilt UI became a SvelteKit tree with content-hashed assets served dynamically. chimera still assumed the old flat four-file model, so staging silently declined and an explicit `ON` build failed at configure. Invisible to `make test`, which only builds the default `OFF`.

## [0.2.6]

### Changed

- **Update llama.cpp to b9592** (from b9528), whisper.cpp to **v1.8.6** (from v1.8.4), and stable-diffusion.cpp to **master-685-19bdfe2** (from master-672-1f9ee88). The two breaking changes in this range lived in headers `bump-check` did not watch (`sampling.h`, `mtmd-helper.h`), so the compile caught them instead. The watch list now covers both.

- **Adapt to two breaking llama.cpp API changes.** `common_sampler_types_from_names` dropped its `allow_alt_names` parameter, and `mtmd_helper_bitmap_init_from_file` gained a trailing parameter and now returns a wrapper struct carrying a video-decode context.

### Added

- **Video input on `chimera gen` and `chimera chat`.** b9592 bundles upstream's video decoder, already compiled into chimera's vendored libmtmd.

  - **`gen --video <path>`** (repeatable, requires `--mmproj`), distinct from `--image`: `--image` keeps upstream's content-based auto-dispatch with default decode params, while `--video` always routes through the video decoder and honors `--video-fps`, `--video-timestamp-ms`, and `--ffmpeg-dir`. ffmpeg/ffprobe are required at runtime.

  - **chat `/video <file>`**, with help, tab-completion, and the same session knobs.

  - Honoring custom decode params needed a chimera-side helper, because upstream's bitmap loader hardcodes default video params internally.

  - A guard rejects video when the build or mmproj lacks video support. End-to-end decode is not CI-covered -- it needs a video-capable mmproj fixture (see `TODO.md`).

### Fixed

- **Video-decode context was leaked when adopting the new mtmd wrapper API,** and video lazy bitmaps are single-use. The context is now RAII-owned for the lifetime of its bitmap, and chat rebuilds each video from its source path before every turn -- the REPL re-tokenizes the whole media history each turn, so reusing a consumed context made the video silently vanish after the first one.

## [0.2.5]

### Changed

- **Update llama.cpp to b9528** (from b9318). The wrapper audit found that every changed, deprecated, or renamed symbol in this range is one chimera does not call, so the clean build was not masking a semantic break. The vendored server stack picked up a new task type, a mid-struct `server_routes` member, and a `task_params::stream` default flip; none affect chimera, which constructs `server_routes` via the upstream constructor and binds routes by member name.

- **Update stable-diffusion.cpp to master-672-1f9ee88** (from master-650-1ceb5bd). Two fields were inserted mid-`sd_ctx_params_t`. This is layout-safe for chimera because `load_model` initializes via upstream's `sd_ctx_params_init()` rather than aggregate init, so both receive their correct defaults -- a plain `{}` would have silently defaulted `vae_format` to the wrong enumerator. Both are now exposed as knobs.

### Added

- **SD VAE-format override (`--vae-format` / `--sd-vae-format`),** accepting `auto | flux | sd3 | flux2`. Upstream's string-to-enum mapper is file-static in the example CLI, so it is reimplemented chimera-side.

- **SD CPU weight-streaming toggle (`--stream-layers` / `--sd-stream-layers`).** Only engages alongside `--max-vram > 0`; sd.cpp disables it otherwise and logs the reason, so no extra guard is added.

- **HTTP server timeout knobs on `chimera serve` (`--http-timeout`, `--sse-ping-interval`).** Both default to `0` = leave the upstream default in place.

- **Realtime reasoning termination on `chimera serve` (`POST /v1/chat/completions/control`).** A client that opened a streaming completion with `"reasoning_control": true` can make the model stop thinking and answer immediately without aborting the stream. The chat handler already passed the flag through, so binding the route is the whole integration.

- **Interactive reasoning termination in `chimera chat` (`--reasoning-control`).** With the flag set, the first `Ctrl-C` while the model is inside its think block ends the reasoning instead of aborting the reply; a second one, or one while answering, aborts as before. The reasoning-budget sampler is armed with no fixed budget so it never auto-fires -- only the runtime force ends thinking. A non-reasoning template warns and the flag becomes inert.

- **Python bindings track the new option fields,** keeping `LlamaOptions` / `SdOptions` / `ServeOptions` at full parity with the C++ structs. Binding coverage is hand-maintained with the round-trip tests as the regression net; `bindings/README.md` previously claimed a compile-time check that does not exist.

## [0.2.4]

### Added

- **CI now covers the combined-archive layer, schema migrations, and the Python bindings.** The matrix previously ran only `make build` + `make smoke`, which exercise the executable and not the three redistributable archives -- the gap that let the 0.2.3 `combine_archives.py` regression ship silently. Three model-free legs close it: `make test-external-smoke` (all OSes, asserts the link contract and ggml backend self-registration), `make test-db-migrate` (all OSes), and `make test-bindings-pytest` (Linux + macOS; gated off Windows because the recipe assumes the POSIX venv layout).

- **Compile-only GPU backend CI (`.github/workflows/ci-gpu.yml`).** A `workflow_dispatch`-only workflow that builds CUDA and Vulkan on Linux and asserts the backend's ggml registration symbol actually linked -- catching a silently-dropped backend without a GPU. It does not validate runtime registration or kernel correctness; that needs a self-hosted GPU runner. Each leg uploads a stripped binary so users can try a GPU build on their own hardware. Manual-only until it proves stable.

- **One-shot artifact bundle in both CI workflows.** A final `collect` job merges the per-leg artifacts over the GitHub API so every platform can be pulled in one click. Per-leg artifacts are kept, and the job runs `if: always()` so a green leg is still bundled when a sibling fails.

- **Compiler-launcher (ccache) forwarding on both configures.** CMake does not read `CMAKE_*_COMPILER_LAUNCHER` from the environment, so a wrapper was previously ignored on both the deps build (where the heavy nvcc compile lives) and chimera's own. Expands to nothing when unset.

- **`DEPS_EXTRA` Makefile passthrough on the `deps` target,** so CI can inject flags like `--no-sd-examples` without a dedicated target.

- **pytest suite for the Python bindings (`bindings/tests/`)** -- 11 no-model tests plus model-gated inference tests. `conftest.py` makes `import chimera` work from the uninstalled standalone build and resolves model paths from env overrides, skipping rather than failing when a model is absent.

- **`make test-bindings-pytest`** runs the suite under the bindings venv interpreter so the ABI matches the freshly-built module. **`make clean-bindings`** is a bindings-scoped clean.

### Fixed

- **`combine_archives.py` dropped object members with duplicate basenames, breaking the combined archives on Linux and Windows.** A single static lib can contain several members with the same basename -- `libggml-cpu.a` ships two `quants.c.o`, `libllama.a` two `llama.cpp.o` -- and the Linux path used a plain `ar x`, which extracts them to the same file, so the later silently overwrote the earlier and consumers hit undefined references. macOS was unaffected because `libtool -static` keeps all members -- which is why this stayed hidden until the new Linux CI leg exercised the combine path.

- **`combine_archives.py` unconditionally required `libggml-blas.a`,** which a default Linux/Windows CPU build never produces. Now include-if-present, same as libwebp/libwebm.

- **`make bindings` no longer prints a spurious `uv venv` failure on re-runs.**

## [0.2.3]

### Changed

- **Update llama.cpp to b9318** (from b9284).

- **Update stable-diffusion.cpp to master-650-1ceb5bd** (from master-645-645e6e9). New model support behind the existing API; `stable-diffusion.h` is unchanged, so there is no drift to adopt and the new models work through the existing path.

- **WebUI embedding rewritten to track the llama.cpp b9318 restructure.** Upstream replaced the asset-embed mechanism with a host generator, and `server-http.cpp` -- which chimera compiles itself -- now includes the generated header unconditionally. This broke even the default `WEBUI_EMBED=OFF` build. chimera now always builds the generator and links a stub when embedding is off, and `manage.py` stages the whole `dist/` tree instead of four named files. `make bump-check` learned to recognize the new layout so the next rearrangement fails loudly instead of crashing `make build`. Upstream no longer ships prebuilt assets, so `ON` requires an `npm run build` first; full write-up in [`docs/dev/webui.md`](docs/dev/webui.md).

### Fixed

- **`chimera serve --no-webui` was silently ignored on b9318 embedded builds.** Upstream switched to reading `params.ui`, of which `params.webui` is now only a default-initializer alias. chimera sets both.

- **`make build-with-webui` failure mode + stale guidance.** The CMake hint and Makefile comment claimed no Node toolchain was required, which is now false, and pointed at an insufficient recovery command.

- **`combine_archives.py` required libwebp/libwebm archives that the default build no longer produces,** so the combine step and every consumer of the archives failed. It went unnoticed because `make test` exercises the executable, not the archives. Both are now include-if-present.

### Added

- **Field-level pin-check expansion across all three engine wrappers.** Previously only llama.cpp had a meaningful compile-time contract. Whisper (38 asserts) and sd (93) now get the same discipline, so a bump that silently retypes a field chimera assigns fails at a labeled line instead of misbehaving at runtime. The whisper and sd blocks stay in their own TUs per the ggml enum-collision isolation.

- **Python bindings scaffold (`bindings/`) over the `chimera.hpp` OOP layer,** via [nanobind](https://github.com/wjakob/nanobind). Exposes `Llama`, `Embedder`, `Tokenizer`, `Server`, and modality-gated `SD` / `Whisper`, with exception translation, GIL release on the long compute calls, and full field coverage of all five option structs. Coverage is hand-maintained with the round-trip tests as the regression net. `make bindings` / `make test-bindings` auto-provision the toolchain when `uv` is available. Documented in [`docs/bindings.md`](docs/bindings.md).

## [0.2.2]

### Changed

- **Update llama.cpp to b9284** and stable-diffusion.cpp to **master-645-645e6e9**.

- **WebP and WebM disabled in the sd.cpp build by default.** sd.cpp auto-detects its vendored submodules and compiles 128 extra translation units into archives chimera never links, since it reads and writes PNG via stb_image. Saves 3-5 minutes of cold dependency build time; no binary-size change. Re-enable with `SD_WEBP=1` / `SD_WEBM=1`.

- **Binary size note.** The release binary is ~44 MB, up from ~31 MB on 0.2.1, entirely from code growth in the pinned upstreams (49 sd.cpp commits, 165 llama.cpp commits). All reachable code with dead-strip already active, so there is no easy chimera-side reduction. The `0.2.0.1` / `0.2.1.1` portable backports carry the same OpenSSL-free fix against the lighter pins.

- **OpenSSL is now opt-in and statically linked.** The release binary previously picked up homebrew's libssl/libcrypto as dynamic dependencies, making it non-portable across macOS machines. The new `CHIMERA_OPENSSL` option defaults `OFF`, and the matching environment variable must agree with it or the link fails with undefined SSL symbols. `chimera serve` is HTTP-only by default; TLS via a reverse proxy is the recommended deployment.

### Fixed

- **SD inference regression on macOS Metal after the sd.cpp master-645 bump.** Every SD-touching test slowed 3-7x. The cause was upstream's new memory-mapped weights, which default off upstream but were switched on during the dev sync: the Metal encoder fails to resolve buffer IDs for mmap-backed tensors and sd.cpp falls through to per-tensor host-to-device copies. sd.cpp's own backend-aware check reports true on Apple Silicon's unified memory but the encoder fails anyway, so the upstream gating is necessary but not sufficient. mmap is now forced off on Apple and honors `--no-mmap` elsewhere, where it is the win it was designed to be. Full investigation log, including the failed bisects and the measurement artifacts that nearly diverted the fix, in [`docs/dev/regression-b9284-investigation.md`](docs/dev/regression-b9284-investigation.md).

### Added

- **Per-run test-timing capture and diff tooling.** `scripts/test.py --timings-out` writes per-test wall-clock plus a version header; `--timings-baseline` annotates each line with the delta. The regression threshold is `abs > 0.5s AND rel > 20%` so trivial flips do not crowd the output.

- **`scripts/test_diff.py`** -- offline diff between two timings JSONs, sorted by absolute delta. Exits 1 on a regression past both thresholds, making it usable as a CI gate.

- **`make test-bench` / `make test-bench-fast`.** Capture a baseline on a known-good ref, capture current on a suspect ref, diff.

## [0.2.1]

### Added

- **Three chimera-specific `/v1/chimera/*` endpoints,** bound unconditionally. Designed as the read surface a downstream desktop wrapper consumes for its About pane, DB footer, and graceful-exit path, but useful to any HTTP client.

  - **`GET /v1/chimera/info`** -- JSON form of `chimera info`: versions, backends, enumerated ggml devices, capability flags, and build flags (emitted only when set, so the object is empty rather than all-empty-strings on a CPU build).

  - **`GET /v1/chimera/db`** -- JSON form of `chimera db status`. Table names are validated before being spliced into the row-count SQL. The endpoint runs migrations so a fresh response on a never-used host is populated rather than a 500.

  - **`POST /v1/chimera/shutdown`** -- graceful exit. Returns `202` and then runs the same teardown as the SIGINT handler after a short delay; without the delay the listener can close before the response is flushed, so the client sees a closed connection instead of the 202.

### Changed

- **`make test-fast` / `make test-slow`** split the suite by wall-clock cost. Three tests account for most of the runtime; `test-fast` runs the rest for tight iteration on the LLM / embed / RAG / chat paths, `test-slow` runs only those three for SD and vision work. The set is centralised in `SLOW_TEST_RE`; update it when a new test crosses ~20 seconds.

- **`scripts/test.py --slow-only` / `--no-slow` / `--exclude`.** `--exclude REGEX` is the general complement to `--filter`; the two compose. The other two are aliases wiring `SLOW_TEST_RE` into each.

## [0.2.0]

### Added

- **CTest registration for external smoke tests,** with labels so `make test-external-oop` can filter to the OOP lane. The "inference probe: SKIP" line maps to ctest's `Skipped` outcome so CI can distinguish "passed with fixture" from "ran without one".

- **`src/chimera/chimera_llama.{h,cpp}`** -- moves the llama.cpp glue and the `command_prompt` / `command_embed` / `command_tokenize` entrypoints out of the CLI shell and into the library, which previously had no direct text-generation entrypoint. `command_chat` deliberately stays in the CLI because it owns terminal I/O, signal handling, linenoise, and color streaming.

- **`src/chimera/chimera.hpp`** -- optional header-only OOP layer over the procedural surface. Persistent-handle classes load the model once and reuse it across calls. Not compiled into the archive; consumers include it at their call site. See [`docs/dev/oop-layer.md`](docs/dev/oop-layer.md).

- **`tests/external/hpp_smoke.cpp`** -- parallel external smoke test through `chimera.hpp`, asserting persistent-handle behavior: the cached context is stable across calls, `reset()` preserves it, `reset(rebuild=true)` drops it.

### Changed (OOP)

- **`chimera::Llama` caches the `llama_context` across `generate()` calls.** Semantics are unchanged -- the KV cache is cleared at the start of each call -- so this is an internal optimization. Sampler and LoRA are rebuilt per call so those knobs take effect immediately; context-creation fields silently no-op until `reset(rebuild=true)`.

- **Streaming callback on `chimera::Llama::generate`.** Library consumers feeding a WebSocket, notebook cell, or log file now have a hook instead of inheriting stdout streaming. The `bool stream` overload is preserved and routes through the callback path.

- **Path-only convenience constructors** on every persistent-handle wrapper, delegating to the full-options constructor. The lazy-context design means callers can still override defaults between construction and first call.

- **`chimera::Whisper` and `chimera::SD` are now persistent-handle** -- no model reload between calls. The CLI and OOP paths share one post-load helper each, so behavior is identical between the subcommand and the wrapper. Load-time fields silently no-op after construction; `reset(reload=true)` honors new values.

### Changed

- **stable-diffusion.cpp pin: `master-596-90e87bc` -> `master-637-ef92a00`.** Bump-check flagged one changed header; the pin-check asserts all held. The API drift on chimera's call path is additive (new sampler methods, schedulers, types, and struct fields), and the two breaking signature changes -- `generate_video` and `new_upscaler_ctx` -- are functions chimera does not call, verified by grep. Upstream also added a libwebp dependency, which builds cleanly but lengthens the cold SD build.

  - **Runtime slowdown (known, accepted on the dev branch).** SD-touching tests are 4-8x slower on this pin. The cause is a ggml/Metal ABI mismatch in the shared-ggml build path, not graph-cut offload: every leaf tensor fails Metal buffer-id lookup and falls through to a slow path. Output is still correct. Resolves once upstream llama.cpp ships a matching ggml-Metal change, sd.cpp reverts, or chimera flips to a vendored ggml (~3 MB binary cost). Main branch unaffected.

- **Upstream-drift defenses widened to cover stable-diffusion.cpp and the post-b9200 `webui` -> `ui` rename.**

  - **`make bump-check` covers two repos in one run,** with `--skip-llama` / `--skip-sd` to scope it. SD's `master-<count>-<sha>` format is auto-reduced to the SHA that raw.githubusercontent.com expects.

  - **SD pin-check assertions in `chimera_sd.cpp`** -- struct field types, sentinel enum counts, and seven function signatures. Kept in that TU rather than the central pin-check because SD's ggml enums collide with llama's.

  - **`common_params::ui` pin-check.** Upstream still has `webui` as an alias; asserting the new canonical field means that when the alias is dropped, the failure points at the replacement rather than at the disappearing name.

- **`.github/workflows/release.yml` -- fail-fast tag-existence guard.** Runs before checkout and reports the missing tag with a copy-pasteable fix and a listing of the tags that do exist, catching typos like `0.17` vs `0.1.7`. Replaces three opaque checkout retries.

- **`scripts/test.py` -- distinguish subprocess timeouts from wrong-exit-code,** and raise the budget on the two tests that run backend init before reaching their check path. CI macos-metal runners have no real GPU, so the Metal probe falls through over 5-10s and exceeded the old budget -- a timeout that masqueraded as a signal-kill in the 0.1.7 release CI.

- **llama.cpp pin: `b9119` -> `b9264`.** Header drift in six files plus build-system drift not visible in headers; no rename hit chimera's call surface.

  - **Graceful webui staging across upstream layouts.** Around b9200 upstream moved the webui to a Vite project with no prebuilt assets in the source tree, and the hard-coded copy raised. `manage.py` now probes three candidate directories in order and skips with a log when none has assets.

  - **Define both `LLAMA_BUILD_WEBUI` and `LLAMA_BUILD_UI`,** so `CHIMERA_WEBUI_EMBED=ON` works on either upstream generation.

  - **`make bump-check` extended with a build-system drift probe.** Header diffs miss non-header files -- exactly the failure mode that bit this bump, where the webui asset directory disappeared with no header change. A missing required path now fails bump-check instead of surfacing as an opaque `make build` failure later.

  - **`docs/dev/webui.md`** documents the manual npm build now required for embedding, and why requiring Node as a hard build dependency would be a regression.

- **Documentation sweep addressing a comprehensive audit** -- 11 files, no code change. Fixed stale test counts and script names, corrected `doc/` -> `docs/` path labels that were lying while the relative links happened to resolve, moved an orphaned image-generation section under its heading, reconciled the "deliberately not bound" route lists against what had shipped, resolved a routes-table contradiction in `docs/dev/sqlite.md`, and grew the README docs table from 5 rows to 12 with a new-contributor reading order. Deliberately left alone: CHANGELOG history, the webui Variant B post-mortem, the router-mode decision record, and the TODO wontfix block -- these are institutional memory, and pruning them is a regression in discoverability for the next maintainer asking why something was not done.

## [0.1.7]

### Changed

- **Converted `scripts/test.sh` to `scripts/test.py`.** Same coverage, ~1100 LOC down from ~1500. Python because it is cross-platform and the stdlib covers everything the bash shelled out to `curl` / `sqlite3` / `python3 -c` for, so the runner no longer needs those on PATH. Per-test wall-clock timing is collected and printed slowest-first.

  - **New flags:** `--smoke`, `--filter REGEX`, `--no-color`, `--verbose`, `--no-timing`.

  - **Refactor win:** eight nearly-identical spawn/poll/teardown blocks now route through one `chimera_serve()` context manager handling port selection, readiness polling, graceful teardown, and log capture.

  - **Behavior changes:** the SKIP path honors `--filter`, so filter-excluded tests no longer emit fixture-missing noise; and the two SD CLI tests each run their own generation instead of sharing one output, costing ~10s but decoupling them so a regression in one no longer cascades into a SKIP of the other.

### Added

- **`scripts/test_fixtures.py`** -- local runner for the six adapter / aux-model success-path tests that `make test` leaves as SKIPs. Probes the env vars, validates paths up front, and forwards to `scripts/test.py --filter`. Partial or invalid configuration exits 2 rather than skipping, matching the misuse-not-SKIP contract used internally.

  - **Why local, not CI:** this replaces a deleted draft workflow. Adapter fixtures are individually licensed, mirrors come and go, and the runs are gigabytes long. Chasing moving availability for hosted CI did not pay off.

- **`POST /v1/audio/detect-language`** -- exit-after-detect probe. Returns `{"language", "duration"}` from the same multipart `file` field as transcription; other transcription knobs are not honored because they are irrelevant for detect-only.

  - **Why a separate endpoint, not a query parameter:** OpenAI's audio surface has no detect-only concept, and conflating the two would force every transcription client to handle a silently-suppressed-transcription path.

  - The integration test asserts a well-formed response, not `language=="en"`: the bundled fixture is an English-only fine-tune whose language-id pass is noise. Verifying detection correctness needs a multilingual fixture chimera does not ship; the wiring is what gets caught here.

- **Opt-in fixture-driven success-path tests for the SD adapter surfaces** -- three on the CLI, three on serve, covering LoRA, ControlNet, and PhotoMaker. Gating tests already covered the 400-class responses, but actual generation output went unverified because chimera does not ship adapter fixtures. Env-var gated: unset skips, set-but-missing fails, since partial configuration is misuse rather than absence. Each test is independent, and each serve test spawns its own server -- coupling four orthogonal opt-ins behind one gate would let partial fixture coverage silently skip everything. The bar is exit 0 with a non-empty PNG, or HTTP 200 with non-empty base64: the value is catching wiring regressions, not validating adapter math.

- **`chimera serve` per-request LoRA selection** on all three image endpoints, plus `GET /v1/images/lora-adapters`. Closes the final numbered item of the server parity roadmap.

  - **Server-init:** repeatable `--sd-lora <name>=<path>` registers a closed allowlist. SD reloads LoRA tensors per generate, so the map is pure metadata and no files are opened at startup. Malformed specs fail at startup, not on first request.

  - **Closed-set by design.** Requests reference adapters by name only and cannot supply filesystem paths -- the safer default for any deployment beyond personal localhost use. A `--sd-allow-lora-paths` opt-in could gate path mode later.

  - **Per-request shape is an array of objects,** `[{"name","scale"}]`, chosen over the CLI's `"name:scale"` string form because future per-adapter fields extend cleanly without inventing parsing rules.

  - **Precise 400s** for a non-array body, a non-object element, a missing name, an unknown alias (listing the known ones so the client can self-correct), and a non-numeric scale.

- **`chimera serve` per-request PhotoMaker** on all three image endpoints, closing the last server-init gap on the SD side.

  - **Server-init:** `--sd-photo-maker`, `--sd-pm-id-dir` (each subdirectory becomes one named identity set), and `--sd-pm-id-embed-path`. The identity directory is scanned eagerly at startup so a misconfigured set fails fast; non-image files are skipped silently while an empty subdirectory is a configuration error.

  - **JSON-base64 shape sidesteps the multipart-plural design question.** OpenAI's body shape has no precedent for repeatable image uploads, so identity images travel as a base64 array rather than an invented multipart convention. Elements accept a raw payload or a `data:` URI.

  - **Two request shapes with explicit precedence:** fully per-request `pm_id_images` wins over the admin-curated `pm_id_image_set`, matching precedence elsewhere in chimera.

  - **Precise 400s** for every PM field against a server missing the corresponding flag, an unknown set name (listing the known ones), a non-array or non-string element, a decode failure, and an undecodable image (415).

- **`chimera serve` SD perf/offload flag family** -- 16 `--sd-*` flags, each mirroring a `chimera sd` flag. With this, the full `chimera_sd::LoadParams` surface is exposed on serve (30 flags total). The four enum flags are CLI11-validated so a bogus value exits before model load with the accepted set listed. `--sd-threads` is guarded on `> 0` so the chimera-side "leave default" sentinel does not override sd's thread auto-pick.

- **`chimera serve` SD split-checkpoint flag family** -- 13 `--sd-*` path flags, so `chimera sd` command lines port 1:1. Gating broadened from requiring `--enable-image` to accepting either a combined checkpoint or a diffusion model, the same allowance the CLI uses. Unlocks serving Flux, SD3, Z-Image, and Qwen-Image; the existing per-request fields work for them without extra wiring because the engine path is shared.

- **`chimera serve --sd-control-net` + per-request `control_image` / `control_strength`** on all three image endpoints -- the highest-impact server-init gap on the SD side. This is where the full `chimera_sd::LoadParams` was first wired through, making the two flag families above one line per flag. A request supplying `control_image` without the flag returns 400 naming it.

  - **Honest correction:** the roadmap framed this step as an unblocking trio. `--sd-upscale-model` unblocks nothing -- the upscale model is per-request and loaded fresh each generate, already covered -- and was removed. `--sd-photo-maker` is half a feature without its per-request bundle and was split out.

- **`chimera serve --audio-*` flags + per-request VAD bundle.** Plumbs `chimera_whisper::LoadParams` through the serve path and unblocks per-request `vad=true`. The VAD model path is server-init-only, deliberately not accepted from clients, to avoid letting external callers direct the server to read arbitrary filesystem paths. Existing `--enable-audio` deployments are unaffected; VAD is opt-in on both sides.

- **Audio wave 2** -- four decoder-fail thresholds on the transcription endpoints. The sentinel is NaN, not a negative number, because `logprob_thold`'s own upstream default is negative and a negative sentinel could not distinguish "leave the default" from "explicitly requested the default". `no_fallback=true` still wins when both arrive, mirroring the CLI. The per-request audio surface is now closed; the remaining CLI flags all need either a model at startup or a dedicated endpoint.

- **Image wave 2** -- hires-fix and cache/SCM bundles, 13 fields into the shared filler so all three endpoints pick them up at once. Validation runs through the same parser the CLI uses, so HTTP errors are byte-identical, and fires before generate so a typo does not waste an inference pass. The `Model` upscaler accepts the field but fails downstream because the serve path had no upscale model at startup -- blocked on the server-init work above; the latent upscalers work.

- **Image wave 1** -- 14 per-request fields into the shared filler: sampler and generation core, VAE tiling, skip-layer guidance, and custom sigmas. Sentinel defaults preserve upstream behavior for omitted keys. `skip_layers` and `sigmas` accept a JSON array or a comma-separated string, matching the CLI's shape, and reject malformed entries with the offending token named.

  - **Unrelated:** a Metal-side crash surfaced when running two image generations back-to-back on one server context. It reproduces without wave-1 fields set -- pre-existing sd.cpp/Metal behavior, not caused by this wiring.

- **Audio wave 1** -- nine per-request fields, all mapping to existing engine fields: decoding strategy, region of audio, segment shaping, and stereo diarization. `temperature` was previously parsed and discarded. Diarization mirrors the CLI's energy-ratio classifier and stamps both the structured `Segment.speaker` field and the text prefix, so every response format reflects speakers. Mono uploads return 400 before transcription runs.

## [0.1.6]

### Added

- **`chimera whisper --detect-language`** -- exit-after-detect language identification. whisper.cpp short-circuits before any decode pass, so chimera writes just the ISO code and no format files are produced, which is the right behavior for a probe. English-only models do not really detect language and can return a wrong code; that is a model-side artifact, documented in the coverage table.

- **`chimera whisper --diarize`** -- stereo speaker diarization, mirroring whisper-cli's per-segment energy-ratio algorithm and threshold. The WAV reader now keeps a per-channel view alongside the downmixed samples, populated only for multi-channel input. The label flows out two ways: a structured `Segment.speaker` field and a text prefix, so the existing format writers render it without change. Mono input with `--diarize` exits with `BadInput` before model load.

- **`chimera whisper` constrained decoding: `--grammar`, `--grammar-file`, `--grammar-rule`, `--grammar-penalty`.** whisper ships its GBNF parser in `examples/` rather than in the library, so it is vendored verbatim (~450 LOC, MIT) with a header noting the source version for clean diffing on the next bump. `--grammar-file` is a chimera-side ergonomic that whisper-cli lacks. Mutual exclusion and bad-rule validation fire before the WAV load.

- **`chimera gen` / `chimera chat` long-tail closer -- 19 flags,** closing every documented not-implemented row in the coverage table.

  - **Extra samplers:** `--typical`, `--top-nsigma`, `--xtc-probability` / `--xtc-threshold`, `--dynatemp-range` / `--dynatemp-exp`.

  - **Sampler chain ordering:** `--samplers`, same shape as llama-cli's flag. `--sampler-seq` is not added separately because it is redundant.

  - **Perf:** `--threads-batch`, `--swa-full`. **Vision-token budget:** `--image-min-tokens`, `--image-max-tokens`.

  - **MoE expert offload:** `--cpu-moe`, `--n-cpu-moe N`. Both stack with `--override-tensor`.

  - **Manual overrides:** `--override-tensor` (upstream's parser is `static` so it could not be reused; the buffer-type lookup reports available names on a typo) and `--override-kv` (reuses upstream's parser so the grammar is exact).

  - **Activation steering:** `--control-vector`, `--control-vector-scaled`, `--control-vector-layer-start` / `-end`. Layer defaults mirror upstream.

  - **Reclassified out of scope,** documented in place rather than silently skipped: `--keep` is architecture-mismatched (it drives upstream's context-shift loop, which chimera's KV-prefix chat path does not run), and `--ctx-checkpoints` / `--checkpoint-every-n-tokens` / `--cache-ram` are server-only fields the CLI path never reads.

- **`chimera chat --reasoning-budget N` is now enforced** at the sampler level, replacing the "parsed but not yet enforced" warning. The earlier claim that this required restructuring the sample loop was wrong: the integration point is entirely upstream of `common_sampler_init`. The active template's thinking tags are probed once at startup, and `--reasoning-budget-message` is tokenized as the forced termination sequence so generation ends cleanly inside the reasoning block instead of cutting mid-thought. A template advertising no thinking tags warns once and the budget is ignored -- warn but do not fail, matching chimera's handling of marginally-wrong input.

- **Doc-only:** `--disable-image-metadata` reclassified out of scope. sd-cli's flag disables a metadata chunk written by a patched `stbi_write_png` in sd's vendored fork; chimera links stock stb_image_write, so its PNGs are already metadata-free and there is nothing to disable. Writing generation params for parity would be net-new functionality, not a port.

- **`chimera sd --embd-dir <dir>`** -- textual-inversion directory, mirroring sd-cli: non-recursive scan, filename stem becomes the prompt token. Invalid paths exit before model load.

- **`chimera sd` cache / SCM bundle** -- `--cache-mode`, `--cache-option`, `--scm-mask`, `--scm-policy`, mirroring sd-cli's two-flag surface exactly so command lines port 1:1. `--cache-option` uses sd-cli's `key=value` grammar with the same per-mode key mapping. Validation runs before model load so a typo does not waste one.

- **`chimera sd` coverage closer -- 33 flags across six rounds,** closing all remaining unforced gaps. All defaults preserve existing behavior: floats use negative sentinels because the upstream defaults are `INFINITY` or specific positive numbers, ints use 0 because that is upstream's own "leave default".

  - **Perf / offload:** `--fa`, `--no-mmap`, `--max-vram`, `--clip-on-cpu`, `--vae-on-cpu`, `--control-net-cpu`, `--force-sdxl-vae-conv-scale`.

  - **Sampler core:** `--img-cfg-scale`, `--eta`, `--timestep-shift`, `--sigmas`, `--prediction`, `--lora-apply-mode`.

  - **Model-loading completers:** `--taesd`, `--clip-vision`, `--llm-vision`, `--tensor-type-rules`, `--photo-maker`.

  - **PhotoMaker generation bundle:** `--pm-id-images-dir` (an empty scan result is an error so a typo does not silently disable PM), `--pm-id-embed-path`, `--pm-style-strength`.

  - **Reference images:** `--ref-image` (repeatable), `--increase-ref-index`, `--no-auto-resize-ref-image`.

  - **Hires-fix bundle:** `--hires`, `--hires-upscaler`, `--upscale-model`, `--hires-width` / `-height` / `-scale` / `-steps` / `-denoising-strength` / `-upscale-tile-size`.

  - **Out of scope, documented:** video-only flags, standalone-mode flags, shell features, and chroma/qwen-specific tuning.

- **`chimera whisper` coverage closer -- 22 flags.** Every numeric knob uses a sentinel that leaves the upstream field untouched, so existing invocations are byte-identical.

  - Region of audio: `--offset`, `--duration`. The sample-offset form is internal to whisper-cli's WAV reader and not exposed by the params struct, so only the ms forms are wired.

  - VAD: `--vad` plus `--vad-model` (required when `--vad` is set) and six tuning knobs inheriting whisper's defaults.

  - Segment shaping: `--max-len`, `--max-tokens`, `--split-on-word`. Decoder-fail thresholds: `--temperature-inc`, `--entropy-thold`, `--logprob-thold`, `--no-speech-thold` (NaN sentinel, since `logprob_thold`'s default is itself negative).

  - Perf: `--audio-ctx`, `--tinydiarize`. Token suppression: `--suppress-regex`, `--suppress-nst`. Context params: `--flash-attn`, `--no-gpu`, `--device`.

  - Parallel decode: `--processors N`. Upstream warns this degrades accuracy at chunk seams, so the serial default is intentional.

- **`chimera sd` skip-layer guidance + high-noise model slot.** `--skip-layers` and its three scalars; an empty list disables SLG regardless of the scalars, and a non-integer entry fails rather than being silently dropped. `--high-noise-diffusion-model` exposes only the model-loading slot: the rest of the `--high-noise-*` family is video-only in sd.cpp and chimera-sd is image-gen only.

- **`chimera sd` perf + RNG knobs:** `--diffusion-conv-direct`, `--vae-conv-direct` (measurable win on modern dGPUs), and `--rng` / `--sampler-rng`. `--sampler-rng cpu` is what matches ComfyUI seeds across implementations.

- **`chimera embed` output-shape gaps:** `--embd-output-format` (`''` preserves current output byte-for-byte; `array`, `json`, `raw` added), `--embd-separator` for one vector per split piece, `--attention causal|non-causal` (a required override for some encoder checkpoints), and `--pooling rank` so the existing subcommand can drive cross-encoder rerankers.

- **`chimera sd` ControlNet + VAE tiling + model-loading completers.** ControlNet: `--control-net`, `--control-image`, `--control-strength`. VAE tiling lets large outputs render without OOM at a small quality cost. Completers: `--clip-g`, `--type`, `--lora` (repeatable), `--lora-model-dir` -- closing the asymmetry where `--lora` had landed on serve/gen/chat but not sd.

- **`chimera sd --guidance` / `--flow-shift`.** Closes the Flux/SD3 generation-side gap that paralleled the earlier Z-Image model-loading fix.

- **`chimera whisper` decoding-strategy flags:** `--prompt` (the most-requested missing flag), `--carry-initial-prompt`, `--beam-size`, `--best-of`, `--temperature`, `--no-fallback`.

- **`chimera info --list-devices`** prints one ggml device name per line, suitable for piping into `--device`.

- **Whisper output formats:** `--output-file` plus `--output-txt` / `-srt` / `-vtt` / `-json` / `-json-full` / `-csv` / `-lrc`, combinable. Segment-level timestamps are auto-enabled whenever any format file is requested -- independent of `--timestamps`, which still controls inline streaming -- so the writers do not emit garbage times. CLI11 forbids multi-char short flags, so upstream's `-osrt`-style aliases are long-only here.

- **Broad llama.cpp CLI coverage uplift on `gen` / `chat` / `embed`** -- 20 flag groups, ~40 options, driven by the coverage audit: the sampler long tail (grammar, JSON schema, penalties, mirostat, the DRY family, logit bias), performance and context (`--flash-attn`, KV cache types, `--ubatch-size`), `--lora` (closing the asymmetry with serve), the RoPE/YaRN family, multi-GPU and device selection, `--no-mmap` / `--mlock` (mmap was previously hard-coded), and chat-only template and reasoning flags. `--list-devices` was skipped as a better fit for `chimera info`.

- **Vulkan build validated end-to-end.** Both an AMD iGPU and an NVIDIA dGPU enumerate on the same host, so dispatch is not vendor-locked. The binary is 92M on Linux x86_64, roughly half the CUDA build, because Vulkan ships compact SPIR-V rather than per-arch cubins -- the smaller-footprint option when raw CUDA perf is not required.

- **Split-checkpoint and Z-Image support on `chimera sd`.** `--diffusion-model`, `--vae`, `--clip-l`, `--t5xxl`, `--llm`, `--offload-to-cpu`, `--diffusion-fa`. `-m` is no longer required: pass it for combined checkpoints, or use component paths for split layouts (Z-Image, Flux, SD3). Validated end-to-end on Z-Image-Turbo on a CUDA build.

### Fixed

- **Windows build works end-to-end across CPU, CUDA, and Vulkan.** Two regressions blocked it:

  - The MSVC branch of the library link never added the GGML backend libs, so any GPU build failed to resolve the backend registration symbol that `ggml-backend-reg.obj` calls directly. The Linux branch already had the per-backend append.

  - The Makefile's Python autodetect picked the Microsoft Store `python3.exe` shim, which exists on PATH but only prints an install prompt and exits non-zero. Candidates are now probed with `--version` instead of `command -v`.

## [0.1.5]

### Added

- **KV-cache slot snapshots and LoRA hot-swap routes** -- four upstream handlers previously called out as deliberately not exposed.

  - `GET /slots` -- per-slot status.

  - `POST /slots/:id_slot?action={save,restore,erase}` -- KV-cache snapshot I/O; `save` and `restore` require `--slot-save-path`. Skipping prefill on a multi-thousand-token system prompt or RAG context turns a multi-second first-token latency into a sub-second restore.

  - `GET /lora-adapters` and `POST /lora-adapters` -- list and re-weight adapters without a model reload. Adapters must be on the startup `--lora` list; the route can re-weight but cannot register new files at runtime.

- **`--slot-save-path <dir>` and repeatable `--lora <path[:scale]>` on `chimera serve`.** The scale parser uses the rightmost colon and falls back to treating the whole string as a path, so Windows drive-letter paths round-trip. `--slot-save-path` is normalized to end with a separator, which upstream's own parser does but chimera bypasses.

- **Pin-check asserts for the four new route handlers,** so a bump that renames one fails that TU first instead of cascading through `chimera_serve.cpp`.

- **Chat history read endpoints over HTTP,** bound only when `--persist-chats` is on so there is no orphan endpoint without a write path producing data. `GET /v1/chats` lists, `GET /v1/chats/:id` returns metadata and ordered messages including interrupted turns, `GET /v1/chats/search` runs FTS5 with highlighted snippets. Unknown ids return 400 rather than 404 because cpp-httplib's default error handler overwrites 404 bodies. Closes the read-side gap: until now `--persist-chats` was write-only.

- **`chimera serve --public-path <dir>`** mounts an external directory at `GET /`, true parity with llama-server's flag. Independent of `CHIMERA_WEBUI_EMBED`; when both apply the mount point wins, and the startup banner reports both so precedence is visible. Lets users point chimera at their own UI or a third-party frontend without baking it into the binary. A chimera-specific UI was prototyped against this flag and abandoned as a poor fit for the busybox identity; see `docs/dev/webui.md`.

- **`chimera db backup --to <path>`** -- snapshot via `VACUUM INTO`, producing a single defragmented file with no WAL/SHM siblings to copy. Refuses to overwrite. **`chimera db vacuum`** defragments in place; its error message names the lock failure mode explicitly, since that is the only realistic failure once the migration ran.

- **Build-time modularity for whisper.cpp and stable-diffusion.cpp** via `CHIMERA_WITH_WHISPER` / `CHIMERA_WITH_SD` (`AUTO` default, following the existing tri-state pattern). When OFF the modality disappears completely: subcommand, serve flags, routes, `chimera info` line, and translation units. `gen --mmproj --image` stays available regardless because it routes through libmtmd, not chimera_sd. A text-only Apple-silicon build drops from ~34 MB to ~12 MB. Caveat: the `whisper --help` / `sd --help` smoke lines pass spuriously on OFF builds because CLI11 falls through to the global help.

- **Static library form for chimera.** `libchimera.a` is now produced alongside the executable, which becomes a thin shim over the CLI TU. Link libraries are `PUBLIC` and the platform whole-archive options are `INTERFACE`, so a downstream CMake consumer inherits both the transitive archive list and the wrappers that keep backend static initializers alive. Compile defines split by audience: modality gating and version strings propagate `PUBLIC` so consumers see the same gating chimera's own code does. Opens a use case that did not exist -- Python bindings, plugin hosts, or non-CMake apps can embed chimera's serve / chat-store / RAG plumbing in-process instead of spawning the binary and parsing output.

- **`scripts/combine_archives.py`** bundles chimera's ~19 transitive archives into two grouped outputs: `libchimera_thirdparty.a` (normal-linked) and `libchimera_ggml.a` (whole-archived). The split is forced by the linker contract, not aesthetics: a single merged archive would force-load duplicate definitions of generic helpers defined independently in two upstreams, and the consumer link would fail. Splitting along the whole-archive boundary keeps duplicates pruned while backend constructors still run. Includes a guard that fails loudly if an edit ever pulls whisper's or sd's sibling ggml builds into the inventory, which would produce silent duplicate-symbol corruption. Scope is host-optimized builds, not cross-machine binary distribution.

- **`tests/external/`** -- standalone consumer smoke test that deliberately does not use chimera's own CMake targets, linking the three archives by raw path the way a non-CMake consumer would. Four probes: backend device count (proves the whole-archive wrapper was applied -- silently 0 without it), a thirdparty symbol, a chimera symbol, and an optional model-gated inference probe asserting non-zero finite logits. New `make combine` and `make test-external-smoke` targets.

- **`docs/dev/combine_archives.md`** -- design and status doc for the three-archive split: scope, motivation, why the deps had to split into two, the per-platform bundling tools and the GNU-ar footguns the script avoids, what the bundle does not do (consumers still owe the whole-archive wrapper), deferred questions, and a validation plan with per-platform status markers.

- **Experimental: embedded llama.cpp web chat UI,** opt-in via `-DCHIMERA_WEBUI_EMBED=ON`. Costs ~6 MB on a stripped build -- the baked byte arrays live in the data section, which `strip` cannot drop. Disable at runtime with `chimera serve --no-webui`. The UI is pinned to whichever llama.cpp version chimera vendored, so UI updates require a rebuild. `docs/dev/webui.md` covers the wiring rationale and the seams worth watching.

### Changed

- **`src/` layout split** along the library / CLI boundary the CMake targets already enforced: the CLI shell moved to `src/chimera_cli/`, everything else stays in `src/chimera/`, which now contains library code only. `src/chimera/` keeps the project name because it produces the artifact users link; the CLI is the late addition. Pure structural refactor, no source edits to any moved file.

- **`chimera_serve.cpp` split into per-modality translation units** (2249 LOC -> 871), leaving lifecycle, `build_common_params`, and the secondary-context helpers behind. New TUs: `chimera_serve_audio.cpp`, `_images.cpp`, `_rag.cpp`, `_chat_persist.cpp`, `_chats_read.cpp`, plus `chimera_serve_internal.h` declaring the seams. Mechanical, no behavior change. Closes a self-prescribed TODO whose "currently ~600 LOC" estimate was stale by a factor of four.

- **`chimera.cpp::main()` extracted** from 475 lines of inline CLI11 wiring to a 46-line parse-and-dispatch driver: a `ParsedCli` struct, eleven `bind_*_cmd` helpers, and a `dispatch_cli`. Mechanical; no behavior or `--help` change.

- **Stale "Phase N" labels stripped** from docs and from the last inline comments in the source tree. The labels referred to a one-time delivery roadmap and have been noise to fresh readers since each feature shipped. Also corrected two doc claims that had been overtaken by shipped work.

- **`README.md`** gains a "Dropping modalities at build time" section with a table of what disappears per modality and the binary-size delta.

- **Security hardening: per-request payload cap.** New `CHIMERA_HTTP_PAYLOAD_MAX_BYTES` (default 256 MB). Upstream's default is unbounded, so a multi-gigabyte upload to the transcription or vector-store routes could OOM the process; requests are now rejected with 413 before any handler runs. Implemented as a chimera-local patch to the vendored `server-http.cpp`, reapplied idempotently by `manage.py` so it survives every `make deps`.

- **FTS5 metacharacter robustness in `GET /v1/chats/search`.** Queries containing FTS5 operators or unbalanced quotes returned 500 with the raw SQLite error. The handler now mirrors the vector-store pattern: retry once with the input wrapped as a literal phrase. Both search surfaces now guarantee that a free-text query never errors -- it matches the literal phrase.

- **Pin-check coverage for seven previously-unpinned `common_params` fields,** each touched by `build_common_params`. A bump that renames or retypes one now fails with a chimera-specific message naming the dependent call site.

- **Error messages distinguish "model file not found" from "loaded but not parseable".** `load_llama_model` stats the path first and reports three distinct failures -- missing, directory, unparseable -- where all three previously collapsed into one generic line.

- **`chat_id_exists` no longer swallows DB-open failures as 404.** A bare catch-all silently converted a corrupt DB or failed migration into "no such chat", serving a 404 for an operator-actionable 500-class problem.

- **`persist_non_streaming` and `persist_streaming` log instead of swallowing exceptions.** Operators now see persistence failures rather than deducing them from missing assistant content.

- **`scripts/test.sh`** gains 12 tests covering the slots and LoRA routes (including a save-to-disk-and-restore round trip) and the chat history endpoints plus `--public-path`, seeded with a deliberately rare token for unambiguous FTS5 assertions. Suite total is now 44.

- **`docs/serve.md`** documents the new routes and flags; **`docs/dev/server-router-mode.md`** is a new decision record on why chimera does not implement llama-server's router mode. The useful part is the concurrency analysis: router mode buys model-residency parallelism (several distinct models hot at once) but not compute parallelism on a single GPU, since kernel submissions serialize at the device. Three concrete revisit triggers are recorded.

### Fixed

- **`ggml_dev_type_label` now handles the fifth `GGML_BACKEND_DEVICE_TYPE_*` enumerator** added upstream, silencing a `-Wswitch` warning on every build.

- **`chats-search` test assertion was over-specific** in two compounding ways: it checked only the first hit, though FTS5 rank order between two matching messages is implementation-defined, and it compared case-sensitively though FTS5 matches case-insensitively. The endpoint was correct all along; the test's narrower assumption was the bug.

- **`chat-kv-cache` token budget raised from 32 to 64.** The 1B model often prepends a preamble before echoing the recall target, pushing it past the cutoff and producing intermittent failures.

## [0.1.4]

### Added

- **Sentence-aware chunking for `chimera index ingest`** and the equivalent serve route, replacing the fixed token-window splitter. Text splits on sentence terminators and paragraph breaks, then greedy-packs into the collection's token budget, with overlap carried as whole-sentence tails. Pathological input (run-on sentences, source code, base64) falls back to the token splitter for the offending span, so ingestion never refuses input. Improves retrieval on prose because embedded text now corresponds to complete thoughts rather than arbitrary mid-sentence cuts.

- **Hybrid retrieval with reciprocal-rank fusion.** A new FTS5 table over document text is created in schema v5 and back-populated from existing rows, with triggers keeping it in sync. Three modes: `semantic` (unchanged KNN), `lexical` (BM25, falling back to a phrase-quoted query on FTS5 syntax errors so user-typed input never 500s), and `hybrid` (default; merges the top hits from both legs by RRF score). Exposed as `chimera search --mode` and a `"mode"` request field. Lexical-only short-circuits the embedding model load -- a BM25 lookup does not need a 100+ MB GGUF paged in. Making hybrid the default changes results for existing collections, which keep working because the FTS5 index is rebuilt during migration.

- **`X-Chimera-Chat-Id` request/response header on `/v1/chat/completions`.** Closes the gap where `--persist-chats` produced one chat row per request, since the OpenAI API has no chat id. Without the header, a chat row is created before delegating (so its id is known in time for a streaming response) and echoed back; with a known id, only the last message plus the reply are appended; an unknown id returns 404 and a non-integer 400. Persistence still runs after the stream finishes, so a failed persist never breaks the request.

- **Ctrl-C in `chimera chat --persist` now persists the in-flight assistant turn** instead of discarding it. A SIGINT handler installed only for the duration of generation flips an atomic the token loop polls; the streamed content is written with a new `partial` flag and the REPL prints an interrupted notice. Outside generation, Ctrl-C still goes to linenoise. Schema v4 adds the flag, backfilling existing rows to complete. `--list` and `--resume` show the interrupted count.

### Changed

- **Schema v5** adds the documents FTS5 table with content-table backing and three sync triggers, rebuilding the index during migration so hybrid search works against previously-ingested documents without re-ingest.

- **Default retrieval mode flipped from semantic-only to hybrid.** Recoverable via `--mode semantic`. Justified because hybrid is a superset for prose: keyword recall is added, helping proper nouns and rare terms, without losing conceptual matches. Cost is one extra FTS5 select per request.

- **`scripts/test.sh` is now 32 tests** (was 23), adding four hybrid-retrieval tests (including a default-mode regression guard) and five `X-Chimera-Chat-Id` tests ending in a DB-state check that proves the unknown and malformed cases never invoked the inner handler. Skipped, not failed, where `python3` / `curl` / `sqlite3` are absent.

- **Privacy documentation for persistence features.** A new section in `docs/serve.md` enumerates exactly what each persistence flag records -- and explicitly what it does not: client IPs, headers, API keys, HTTP bodies -- where the DB lives per platform, and how to wipe it. A matching table in the cheatsheet covers all five write-to-disk surfaces. Closes the gap where opt-in persistence shipped without a user-facing privacy note.

### Removed

- **`TODO.md` pruned of shipped items.** Four were moved to a new "Out of scope (wontfix)" section with brief rationale rather than silently deleted, so the same proposals are not re-litigated from scratch: `POST /props` runtime mutation (conflicts with "the CLI is the config"), multi-tenancy / router mode (one process = one model is core to the busybox identity), HTTPS direct serving, and auth beyond `--api-key` (both reverse-proxy territory). The web chat UI item was restructured as two independent opt-in variants.

## [0.1.3]

### Added

- **`chimera serve --enable-embeddings <model.gguf>`** loads a dedicated embedding model alongside the LLM and routes `/v1/embeddings` to it, keeping the primary LLM generative. Wins over `--embeddings` if both are passed.

- **`chimera serve --reranking <model.gguf>`** loads a cross-encoder and binds `POST /v1/rerank`, matching the pooling toggle llama-server uses. Natural follow-up to vector search: top-N hits, rerank, top-k, LLM.

  Both share a `SecondaryServerCtx` helper -- heap-allocated and non-movable because `server_routes` holds a reference to its params -- with one worker thread each. Shutdown terminates secondaries before the primary so their loops unblock cleanly.

- **Word-level timestamps in `/v1/audio/transcriptions`.** Pass `timestamp_granularities=["word"]` with `response_format=verbose_json` for a top-level `words[]` array. The implementation filters whisper's timestamp special tokens and groups the rest into words by leading-space boundaries.

- **`POST /v1/audio/translations`** -- the existing transcription handler bound with `translate=true`; whisper does the to-English translation inline.

- **Token-based chunking for ingestion,** replacing the character-window splitter. Chunks are sized in tokens of the loaded embedding model's vocab, defaulting to 512 with 64 overlap to match the input limit of common encoders. Eliminates the 400-800 token variance of the old proxy and guarantees chunks fit through `embed()` without truncation.

- **Per-collection chunk and distance knobs** in schema v3, set at `index create` and read at ingest unless overridden per call: `--distance cosine|l2|l1` (default cosine, right for the L2-normalized embeddings chimera produces by default), `--chunk-tokens`, `--chunk-overlap`. Also accepted on the vector-store create route and reported under `meta`. Existing rows backfill to sensible defaults.

- **Persistent embedding cache** via `--cache-embeddings` on embed / ingest / search / serve, memoizing `embed(text) -> vector` to SQLite so repeated work skips the model. The key is the model fingerprint plus a hash of the text; the fingerprint hashes the file size and its head and tail, and since GGUFs store metadata in the header this catches re-quantization, re-training, and architecture swaps without tracking model names. Vectors round-trip bit-identical. Default off, because cache rows take real disk (~1.5 kB per 384-dim row; ~150 MB for 100k entries).

- **`make bump-check`** fetches upstream's server headers at a target llama.cpp ref and diffs them against the vendored copies, listing added and removed symbols. These headers are not part of upstream's stable API -- a CI failure the week before was caused by a new symbol appearing in one -- and this makes the surprise visible at bump time. It is a pre-bump audit step, not a CI guard.

- **SD log capture for image-generation error bodies.** sd.cpp's log lines are mirrored into a small ring buffer and the most recent are appended to the HTTP error body on failure, so the descriptive line -- buft failures, rejected sampler names, ggml backend errors -- reaches the client instead of only stderr. Does not help when sd.cpp aborts the process via `GGML_ASSERT`.

### Changed

- **Release workflow packages each binary as a compressed archive** instead of a raw executable. The `.sha256` sidecars were dropped because the release UI already shows a checksum per asset. The Windows matrix target was renamed so the archive name no longer ends in `.exe.zip`.

- **Non-WAV audio in `/v1/audio/transcriptions` is now framed as a deliberate non-feature** rather than a temporary limitation, with the error body pointing at an `ffmpeg` one-liner. Bundling codecs is out of scope: single-header decoders give partial coverage, and libavcodec does not pull its weight. Marked "do not revisit without a concrete user request".

- **`make test-golden`** spawns `chimera serve` against fixed models, hits nine routes with fixed payloads, normalizes the volatile fields, and diffs against checked-in goldens. Catches runtime drift in upstream's route lambdas -- the class of regression where a smoke test cannot see a JSON key being renamed or dropped. Backend-pinned so the goldens stay portable across dev machines and CI.

- **Widened `make bump-check`** to cover `llama.h`, `common.h`, `arg.h`, `chat.h`, and `mtmd.h` alongside the server headers -- the full set chimera links or compiles against.

- **Compile-time pin assertions (`chimera_pin_check.cpp`).** Every route handler chimera binds, every `common_params` field it pokes, the pooling-type enum values, and key function signatures are asserted. A bump that renames a handler or retypes a field now fails with a labeled error instead of cascading into a cryptic instantiation failure deep in `chimera_serve.cpp`. Verified against a deliberate type flip.

- **`.github/PULL_REQUEST_TEMPLATE.md`** with a dependency-bump checklist, and **`docs/dev/maintenance.md`**, a strategy doc covering where breakage lands across upstream changes and the bespoke-vs-vendored map for triage.

- **`make test-db-migrate`** builds a v1-schema DB, drives migration, and asserts the version advances, pre-existing rows survive, and later additions are backfilled. Verified to catch regressions. Pre-empts the class of bug where a new migration silently breaks the upgrade path for users still on the original schema.

### Removed

- **`chimera index ingest --chunk-chars`,** replaced by `--chunk-tokens`. The unit changed too, so a literal `--chunk-chars 2048` is closer to `--chunk-tokens 512` for English. `--chunk-overlap` survives with its unit changed to tokens.

## [0.1.2]

### Added

- **`chimera info`** prints chimera's version and platform, then one block per bundled component: llama.cpp (versions, backends, enumerated devices with type tags, capability flags), whisper.cpp and stable-diffusion.cpp (versions, parsed CPU features), and sqlite + sqlite-vec versions. The output shape matches cyllama's `info` so users hopping between the two see one familiar format. A single command captures every version and backend chimera saw at link time and startup, which makes it a useful bug-report artifact.

- **`chimera serve` phase 5: server-side chat persistence + the OpenAI Responses API.**

  - `--persist-chats` wraps the chat-completions handler with per-request DB writes. Clients see exactly the same bytes; a copy is saved after each exchange. Streaming SSE is handled, not just non-streaming JSON. Persistence errors are logged and never break the client's response. The DB is shared with the CLI and with `--enable-rag`.

  - **`POST /v1/responses` is now bound.** It was deferred in the original server work because serve was stateless across requests. The API is still stateful only within one invocation -- server-context holds thread state in-process -- but the underlying chat traffic is persisted, so audit-log use cases work.

  - One chat row per request, by design: the OpenAI API has no chat id, so multi-turn clients that resend the full conversation produce overlapping rows. The duplication is the cost of staying API-compatible.

- **`chimera serve` phase 4: OpenAI-shaped vector-store / RAG routes** over the SQLite + sqlite-vec layer, opt-in via `--enable-rag <embedding.gguf>` and sharing the CLI's DB. Six routes: list, create, stats, delete, ingest (multipart upload or JSON body), and KNN search.

  - One `Embedder` per server, serialized on a mutex like whisper and SD. SQLite connections are opened per request rather than pooled -- open is microseconds in WAL mode, so the pool ceremony is not worth it.

  - One embedding model per server in this cut; a collection recorded with a different model returns a clear 400.

  - **Two server-http warts worked around:** the wrapped cpp-httplib subset exposes only GET and POST, so drop is `POST :name/delete` rather than `DELETE` -- adding the verb would mean patching the vendored source and carrying that per llama.cpp version. And upstream's error handler unconditionally overwrites 404 bodies with a generic payload, so not-found errors return 400 to keep chimera's specific message visible. Defensible: the URL pattern matched, the named resource inside it did not.

- **Persistent chat history (phase 3),** the secondary driver for embedding SQLite. New `chat` flags: `--persist` (off by default; each turn becomes a row, with reasoning captured to its own column), `--resume <id|last>` (replays history into the in-memory conversation, and takes the model path from the saved row if `-m` is omitted), `--list` and `--search` (print-and-exit, no model load), and `--db`. In persistent mode `/clear` starts a fresh chat row rather than wiping the active one, and `/regen` deletes the trailing assistant messages so the next attempt replaces them cleanly.

  Scope limits: interrupted turns are not saved (the DB stays consistent, the in-flight text is lost); attached media paths are recorded for forensics but `--resume` does not reattach them; cross-model resume warns rather than blocks, and the template comes from the new model; and stored reasoning is for record-keeping, not for re-priming the KV cache.

- **Vector store / RAG (phase 2),** the primary driver for embedding SQLite -- a personal RAG index built entirely against local models, no server required. `chimera index create` records the embedding model's dimension and creates a per-collection vector table; `index ingest` chunks and embeds files or globs, reusing one `Embedder` across the batch so the model loads once; `index list / stats / drop`; and `chimera search` embeds the query and prints top-k chunks with distance and source.

  Under the hood, the embedding loop was extracted from `command_embed` into a reusable `Embedder` class, and the SQL lives in a new vector-store module.

  Scope limits: chunking is character-based (token-based deferred); one embedding model per collection, enforced at ingest with a clear dim-mismatch error; and no re-ingest deduplication, so ingesting the same file twice duplicates chunks.

- **Embedded SQLite + sqlite-vec (phase 1).** No user-visible change beyond a diagnostic subcommand; this lays the rails for RAG and chat history. Both are vendored as single translation units at pinned versions and compiled directly into chimera's target rather than built as separate libraries. New `chimera_db` module exposes an RAII connection, an XDG-compliant default path resolver, and `open_and_migrate`. The v1 schema lands chats, messages with an FTS5 mirror, collections, and documents; per-collection vector tables are deliberately created on demand in phase 2 instead. New `chimera db status` verifies the extension actually loaded by executing a query against it, not merely that the symbol linked. Binary grows ~1.5 MB.

- **`chimera serve`: bind the "group A" server-context routes** previously deferred -- a registration change with no new handler logic. Newly always available: `GET /metrics` (with the upstream flag forced on so it works without extra flags), `GET /props`, legacy unprefixed `POST /chat/completions`, the Anthropic Messages compat pair (so Anthropic-SDK-shaped clients can point at chimera unchanged), `POST /infill`, `POST /tokenize` / `/detokenize`, and `POST /apply-template`. `POST /props` is deliberately not bound: runtime mutation of server state conflicts with chimera's "the CLI is the config" stance.

- **`chimera serve` phase 3: `POST /v1/images/{generations,edits,variations}`** via stable-diffusion.cpp, opt-in with `--enable-image <sd.gguf>`. Output is PNG-encoded and base64'd into OpenAI's envelope. Concurrent requests are serialized because sd.cpp's generate is not thread-safe on a shared context.

  Like the audio phase, this does not bind an upstream handler -- there is none for image generation -- so chimera registers its own on the shared HTTP context, the same way llama-server registers its non-LLM routes.

  Scope limits: `response_format` is `b64_json` only, since chimera serve has no static-file backend to host URLs from; `model`, `user`, `quality`, and `style` are ignored; and step progress goes to stderr, since the OpenAI spec defines no SSE for images.

  `chimera_sd.cpp` was reorganized behind a public `chimera_sd::` API, with `command_sd` becoming a thin caller that adds the CLI-only conveniences.

- **`chimera serve` phase 2: `POST /v1/audio/transcriptions`** via whisper.cpp, opt-in with `--enable-audio <whisper.gguf>`. Response formats: `json`, `text`, `verbose_json`, `srt`, `vtt`. Requests are serialized because `whisper_full` mutates the context.

  chimera deliberately does not bind upstream's own transcription handler: that route feeds audio through mtmd's audio mmproj, a fundamentally different pipeline from dedicated ASR. chimera's handler uses whisper.cpp directly, the same engine as the CLI subcommand.

  Scope limits: WAV only (other formats need a real decoder; non-WAV returns 415), and `model` / `temperature` / `timestamp_granularities[]` are ignored in this cut.

  The refactor exposed a latent bug: setting `detect_language = true` puts whisper into a language-id-only mode that returns without transcribing, so `language="auto"` now resolves through the language parameter alone. Unreachable from the CLI's default, so no existing test caught it; the HTTP handler exercises it by default.

- **`chimera serve`: OpenAI-compatible HTTP server, phase 1 (LLM only).** Links llama.cpp's `server-context` -- the same engine behind llama-server -- and exposes a curated route subset through the vendored cpp-httplib. Compiles `server-http.cpp` directly into the chimera target, since upstream does not ship it as a separate library. Exposed: health, `/v1/models`, chat completions (streaming and not), legacy completions, and embeddings.

  Everything else on `server_routes` is a scope choice rather than a missing capability -- each could be enabled with a one-line binding. Server-mode features skipped outright: router mode, built-in tools, the MCP CORS proxy, GCP compat, the embedded web UI, and SSL (reverse-proxy territory).

## [0.1.1]

### Added

- **`chat`: slash commands, multimodal input, tab completion, color, and a load spinner.** New commands: `/help`, `/regen`, `/clear`, `/read` and `/glob` (attach text to the next message), `/image` and `/audio` (attach media when `--mmproj` is given). Tab completion covers the command word and falls through to filesystem completion for the path-taking commands; `/image` and `/audio` are only offered when the loaded mmproj advertises that modality.

- **`chat`: multimodal turns.** Once any media is attached, the loop switches from the text-only KV-prefix-reuse path to re-tokenizing and re-evaluating the whole conversation each turn -- correct, but O(history).

- **`chat`: ANSI color** via the [rang](https://github.com/agauniyal/rang) single-header dep, controlled by `--color {auto,always,never}`. Concrete colors route through a semantic-tag layer so re-skinning is a single-site edit. The prompt's color codes are emitted around the line-read call rather than inside the prompt string, because ANSI bytes in the prompt break linenoise's width math and corrupt the cursor under multi-line edits.

- **`chat`: thinking text rendered grey.** Replies are re-parsed and diffed per token so reasoning content prints grey while the answer prints normally. Only the content portion is stored in history -- the next turn does not reinject the model's prior thinking. Matches llama-cli.

- **`chat`: background spinner during model load,** auto-disabled when stderr is not a TTY so piped logs stay clean.

- **Optional [linenoise](https://github.com/shakfu/linenoise) integration for `chat`** -- line editing, history, and editing keys, with history persisted at `$CHIMERA_HISTORY` or a home-directory default. Engaged only on a TTY, so scripts and the test suite are unaffected. Controlled by `CHIMERA_LINENOISE` (`AUTO` / `ON` / `OFF`).

- **`gen` multimodal input** via `--mmproj` + repeatable `--image`. Auto-prepends the media marker if the user did not place it, and auto-wraps the prompt in the model's chat template, since VL models are typically instruct-tuned and stall without it.

- **`chat`: persistent KV cache across turns.** The context and sampler are built once per session; each turn finds the longest common prefix already resident, rewinds, and decodes only the tail. The previous implementation rebuilt the context per turn and paid full prompt re-decoding every time.

- **`sd` img2img and inpainting** via `--init-image`, `--mask-image`, `--strength`. Image dimensions must match `-W`/`-H`; there is no internal resizing.

- **`make install` / `make uninstall`** honoring `PREFIX` and `DESTDIR`, and `make rebuild` as a deps-skipping shortcut.

- **CI workflow** building and smoking on macOS arm64/Metal, Linux x86_64/CPU, and Windows x86_64/MSVC (initially non-blocking), uploading per-platform binaries and caching `thirdparty/`. **Release workflow** on `v*` tags rebuilds the same matrix and attaches binaries plus checksums to a GitHub Release.

- **`tokenize` subcommand** -- token ids for a prompt, or `id<TAB>piece` rows with `--pieces`. Useful for debugging vocab and template behavior without running generation.

- **`embed` subcommand** -- a single embedding vector via a GGUF embedding model, with pooling, normalization, and the usual context knobs.

- **`-f, --prompt-file`** on `gen` / `tokenize` / `embed` and **`--system-prompt-file`** on `chat`, each reading `-` as stdin and mutually exclusive with the inline form.

- **Structured exit codes** (`ExitCode` enum + `ChimeraError`): 1 runtime, 2 bad input, 3 model-load failure, 4 generation failure. CLI11 parse errors keep CLI11's own codes.

- **Whisper streaming:** each finalized segment prints as whisper.cpp produces it, instead of buffering until the call returns. **SD progress** goes to stderr, so stdout still receives only the produced PNG paths and pipelines stay clean.

- **`make test` / `make smoke`.** Smoke exercises `--version` and `--help` on every subcommand; the end-to-end tier runs gen, whisper, and sd when the model files are present, reporting missing models as SKIP rather than FAIL.

- **`REVIEW.md`** -- architecture, feature, usability, and best-practices review of the 0.1.0 baseline.

### Fixed

- **`whisper` mis-detected language when `-l` was omitted,** defaulting to auto-detect and occasionally identifying English-only models as Azerbaijani, producing empty output. Now leaves whisper.cpp's own default in place unless the user asks for auto or an explicit code.

- **`whisper` crashed when `--threads` was left at its default of -1.** The value reached a field whisper.cpp casts to `size_t` to size a vector. The default is now left untouched unless the user passes a positive override.

### Changed

- **`stb_impl.cpp` no longer defines `STB_IMAGE_IMPLEMENTATION`,** because libmtmd ships its own non-static `stbi_load` that would duplicate-symbol on link as soon as any mtmd helper is referenced. The image-write implementation is still chimera's.

- **`fail()` and `trim()` deduplicated** into `chimera.h` as inline helpers. They pull in no ggml headers, so the three-TU isolation is preserved.

- **`scripts/manage.py` trimmed of cyllama-specific code** -- wheel building, dynamic-library machinery, and a dozen unused subcommands. ~3170 lines down to ~1210. Retained: `build`, `info`, `clean`, `download`.

- **`--help` output compacted** -- short and long flags packed together, explicit usage string, single blank lines between sections.

- **Top-level description tightened** to `chimera - {llama,whisper,stable-diffusion}.cpp multitool`.

## [0.1.0]

### Added

- Initial repository, extracted from [cyllama](https://github.com/shakfu/cyllama).

- Static multitool executable bundling llama.cpp, whisper.cpp, and stable-diffusion.cpp against a single shared ggml backend set.

- Subcommands: `gen`, `chat`, `whisper`, `sd`.

- Top-level `-v,--verbose`; native backend logging silenced by default.

- Three-TU layout to isolate the colliding `ggml.h` headers shipped by llama.cpp and whisper.cpp.

- Late `llama_backend_init()`, deferred until after CLI parsing so `--help` and parse errors do not trigger backend loading.

- `scripts/manage.py` build driver and the `make deps` / `build` / `clean` / `reset` wrappers.

- Verified end-to-end on macOS arm64 with Metal.

### Known issues

- Only macOS arm64 + Metal is verified. Other platforms are believed to work via the inherited cyllama build matrix but have not been re-validated post-split.

- `whisper` and `sd` build cleanly but have not been exercised end-to-end in this repo yet.
