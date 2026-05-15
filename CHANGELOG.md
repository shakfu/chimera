# Changelog

All notable changes to chimera will be documented in this file. Format is
loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `make test` / `make smoke` targets backed by `scripts/test.sh`. Smoke
  tier exercises `--version`, `--help` on the root and every subcommand,
  and confirms that `gen` without `-m` exits non-zero. End-to-end tier
  runs `gen` (Llama-3.2-1B), `whisper` (ggml-base.en + whisper.cpp's
  bundled `jfk.wav`), and `sd` (sd_xl_turbo) when those model files are
  present under `models/`; missing models are reported as SKIP, not FAIL.
- `REVIEW.md`: architecture / feature / usability / best-practices
  review of the 0.1.0 baseline.

### Fixed

- `whisper` crashed with `error: vector` when `--threads` was left at
  its default of `-1`. The value was forwarded to `params.n_threads`,
  which whisper.cpp then cast to `size_t` to size an internal
  `std::vector`, triggering `std::length_error("vector")`. Now leave
  `params.n_threads` at `whisper_full_default_params`'s value unless
  the user passed a positive override.

### Changed

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
