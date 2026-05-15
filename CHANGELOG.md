# Changelog

All notable changes to chimera will be documented in this file. Format is
loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
