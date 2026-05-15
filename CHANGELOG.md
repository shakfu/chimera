# Changelog

All notable changes to chimera will be documented in this file. Format is
loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Optional [linenoise](https://github.com/shakfu/linenoise) integration
  for `chat`. When `liblinenoise.a` is present under `thirdparty/`,
  interactive sessions get readline-style line editing, history (↑/↓,
  `Ctrl-R`), and basic editing keys (`Ctrl-A`/`Ctrl-E`/word jumps).
  History persists at `$CHIMERA_HISTORY` (override) or
  `$HOME/.chimera_chat_history`. Engaged only on a TTY — piped/redirected
  stdin falls back to `getline`, so scripts and the test suite are
  unaffected. Controlled by CMake option `CHIMERA_LINENOISE` (`AUTO` /
  `ON` / `OFF`; default `AUTO`); when ON and the lib is missing,
  configure fails with a pointer to `manage.py build -L`. Linenoise
  source is pinned at `master` (commit `06552a1de6` at integration
  time).
- `gen` multimodal input via `--mmproj <path>` + `--image <path>`
  (repeatable). Uses `mtmd_init_from_file` + `mtmd_helper_eval_chunks`
  to evaluate text + image chunks into the llama context, then runs the
  existing sample loop. Auto-prepends the default media marker once per
  `--image` if the user did not place it themselves; auto-wraps the
  prompt in the model's chat template (VL models are typically instruct-
  tuned and stall without it). The vision encoder runs on the default
  backend (Metal on macOS) -- ~16 s end-to-end for a 512x512 image on
  gemma-4-E4B + its mmproj.
- `chat`: persistent KV cache across turns. The llama_context and sampler
  are now built once at session start; each turn re-templates the
  conversation, finds the longest common prefix against what's already
  resident in the KV cache, rewinds via `llama_memory_seq_rm`, and only
  decodes the tail. The previous implementation rebuilt the context per
  turn, paying full prompt re-decoding cost each time.
- `sd` img2img and inpainting via stable-diffusion.cpp's existing
  `init_image` / `mask_image` fields. New flags: `--init-image <path>`,
  `--mask-image <path>` (requires --init-image), `--strength` (default
  0.75). When `--init-image` is supplied, the SD context is built with
  `vae_decode_only=false` so the encode path is available. Image
  dimensions must match `-W,-H` (no internal resizing). Verified via a
  test that round-trips a freshly-generated image through img2img.
- `make install` / `make uninstall` targets honoring `PREFIX` (default
  `/usr/local`) and `DESTDIR` (for staged packaging). `make rebuild`
  added as a `--target chimera` shortcut that skips `make deps`.
- CI workflow (`.github/workflows/ci.yml`): builds + smokes on
  `macos-14` (arm64, Metal) and `ubuntu-latest` (x86_64, CPU) on every
  push and PR to `main`. Uploads per-platform binaries +
  `.sha256` companion files as workflow artifacts. Caches
  `thirdparty/` keyed on `scripts/manage.py` + top-level CMake files.
- Release workflow (`.github/workflows/release.yml`): triggered on
  `v*` tags (or manual `workflow_dispatch`). Rebuilds the same matrix
  and attaches `chimera-macos-arm64`, `chimera-linux-x86_64`, plus
  `.sha256` checksums to a GitHub Release.
- `tokenize` subcommand: emit token ids for a prompt (one per line, or
  `--pieces` for `id<TAB>piece` rows). Reads prompt from `-p`,
  `-f <file>`, or `-f -` (stdin). Useful for debugging vocab and
  template behavior without running generation.
- `embed` subcommand: emit a single embedding vector (space-separated
  floats) for a prompt via a GGUF embedding model. Options: `--pooling`
  (mean/cls/last/none, default mean), `--no-normalize`, `-c/--ctx-size`,
  `-b/--batch-size`, `-t/--threads`, `--gpu-layers`. Verified against
  `bge-small-en-v1.5-q8_0.gguf`.
- `-f, --prompt-file <path>` on `gen` and `tokenize` / `embed`. Reads
  the prompt from a file; `-` reads stdin. Mutually exclusive with
  `-p`.
- `--system-prompt-file <path>` on `chat`. Reads the system prompt from
  a file; mutually exclusive with `--system`.
- Structured exit codes (`ExitCode` enum + `ChimeraError`): `1` runtime,
  `2` bad input, `3` model-load failure, `4` generation failure. CLI11
  parse errors still return CLI11's own codes (>= 100). Documented at
  the top of `chimera.h`.
- Whisper streaming: each finalized segment is printed as
  whisper.cpp produces it (via `new_segment_callback` +
  `whisper_full_*_from_state` accessors), instead of buffering until
  `whisper_full` returns.
- SD progress callback: prints `sd: step N/M` to stderr (one carriage
  return per update, newline on completion). Stdout still receives only
  the produced PNG paths, so pipelines stay clean.
- `make test` / `make smoke` targets backed by `scripts/test.sh`. Smoke
  tier exercises `--version`, `--help` on the root and every subcommand,
  and confirms that `gen` without `-m` exits non-zero. End-to-end tier
  runs `gen` (Llama-3.2-1B), `whisper` (ggml-base.en + whisper.cpp's
  bundled `jfk.wav`), and `sd` (sd_xl_turbo) when those model files are
  present under `models/`; missing models are reported as SKIP, not FAIL.
- `REVIEW.md`: architecture / feature / usability / best-practices
  review of the 0.1.0 baseline.

### Fixed

- `whisper` defaulted `language = nullptr` + `detect_language = true`
  when `-l` was omitted, which sometimes mis-detected English-only
  `.en` models as Azerbaijani (p < 0.02) and produced empty output.
  Now leave whisper.cpp's default (`"en"`) in place unless the user
  passes `--language auto` or an explicit code.
- `whisper` crashed with `error: vector` when `--threads` was left at
  its default of `-1`. The value was forwarded to `params.n_threads`,
  which whisper.cpp then cast to `size_t` to size an internal
  `std::vector`, triggering `std::length_error("vector")`. Now leave
  `params.n_threads` at `whisper_full_default_params`'s value unless
  the user passed a positive override.

### Changed

- `stb_impl.cpp` no longer defines `STB_IMAGE_IMPLEMENTATION`. Reason:
  libmtmd's `mtmd-helper.o` ships its own non-static `stbi_load`, which
  would duplicate-symbol on link the moment any `mtmd_helper_*` is
  referenced. `chimera_sd.cpp`'s `stbi_load` / `stbi_image_free` calls
  now resolve against libmtmd's copy (libmtmd is unconditionally linked
  for the `gen` mtmd path). `stb_image_write` is still ours.
- `fail()` and `trim()` were duplicated across `chimera.cpp`,
  `chimera_whisper.cpp`, and `chimera_sd.cpp`. Pulled into `chimera.h`
  as inline helpers so all three TUs share one definition. The helpers
  don't pull in `ggml.h`, so the three-TU isolation is preserved.

- `scripts/manage.py` trimmed of cyllama-specific code (wheel building,
  Cython artifact cleanup, `profile` / `bench` / `bump` / `bins` /
  `check-vendor` / `fix-macos-vulkan-wheel` / `write-build-config` /
  `status` / `test` subcommands, dynamic-wheel `build_shared` /
  `download_release` / macOS dylib rpath sanitization / MSVC import-lib
  generation, dynamic-lib path machinery, `pip_install` / `apt_install` /
  `brew_install` helpers, `STABLE_BUILD` env split). ~3170 -> ~1210
  lines. Retained subcommands: `build`, `info`, `clean`, `download`.
  `-D/--deps-only` kept as a no-op for Makefile compatibility.
- `--help` output: compact spacing (short + long flags packed together
  via `long_option_alignment_ratio(0.0f)`; explicit `usage()` string and
  a `CompactFormatter` that trims `make_usage`'s trailing `"\n\n"` to
  `"\n"` so section breaks are single blank lines).
- Top-level description tightened to
  `chimera - {llama,whisper,stable-diffusion}.cpp multitool`.

## [0.1.0]

### Added

- Initial repository, extracted from [cyllama](https://github.com/shakfu/cyllama).
- Static multitool executable bundling llama.cpp, whisper.cpp, and
  stable-diffusion.cpp against a single shared ggml backend set.
- Subcommands: `gen` (one-shot completion), `chat` (interactive),
  `whisper` (WAV transcription), `sd` (text-to-image).
- Top-level `-v,--verbose` flag; native backend logging silenced by default.
- Three-TU layout (`chimera.cpp` / `chimera_whisper.cpp` / `chimera_sd.cpp`)
  to isolate the colliding `ggml.h` headers shipped by llama.cpp and
  whisper.cpp.
- Late `llama_backend_init()`: deferred until after CLI parsing so `--help`
  and parse errors do not trigger `ggml_load_backends`.
- `scripts/manage.py` build driver (forked from cyllama, trimmed of
  Python-binding-specific code paths).
- `make deps` / `make build` / `make clean` / `make reset` wrappers.
- Verified end-to-end on macOS arm64 with Metal backend
  (Llama-3.2-1B Q8_0 model).

### Known issues

- Only macOS arm64 + Metal is verified. Linux, Windows, and non-Metal
  backends are believed to work via the inherited cyllama build matrix but
  have not been re-validated post-split.
- `whisper` and `sd` subcommands build cleanly but have not been exercised
  end-to-end in this repo yet.
