# TODO

Forward-looking work for chimera. Loosely prioritized.

## Validation

- [ ] Verify build on Linux x86_64 (CPU + Vulkan + CUDA).
- [ ] Verify build on Windows (MSVC).
- [ ] End-to-end test `whisper` subcommand against a real WAV + ggml model.
- [ ] End-to-end test `sd` subcommand against a real diffusion model.
- [ ] Validate non-Metal macOS backends (Vulkan, CPU-only).

## Build system

- [ ] Trim more cyllama-specific code from `scripts/manage.py`
      (wheel builder, profile/benchmark commands, `cyllama` references in
      log lines).
- [ ] Add a `make install` target that drops `build/chimera` into a
      `PREFIX`-controlled location.
- [ ] CI workflow that builds release binaries for macOS arm64, Linux
      x86_64, and Linux arm64, and attaches them to GitHub Releases.
- [ ] Homebrew tap (or formula PR to homebrew-core once the project is
      stable enough).

## Features

- [ ] `embed` subcommand — emit embedding vectors for one or more inputs.
- [ ] `tokenize` subcommand — debug helper for prompt tokenization.
- [ ] Streaming output for `whisper` (currently buffered until completion).
- [ ] Per-turn sampler state in `chat` (currently the sampler is rebuilt
      every turn; persistent state would improve coherence).
- [ ] `--system-prompt-file` for `chat`.
- [ ] Image-to-image and inpainting modes for `sd`.
- [ ] Multimodal (`mtmd`) image input for `gen` / `chat` — the static lib
      is already linked.

## Quality

- [ ] Replace ad-hoc WAV parser in `chimera_whisper.cpp` with a real one
      (currently rejects anything that isn't 16-bit PCM mono).
- [ ] Exit codes: distinguish parse errors, model-load failures, and
      generation failures.
- [ ] Document the model formats each subcommand accepts.
- [ ] Add a `--version` flag that reports the bundled llama.cpp /
      whisper.cpp / sd.cpp commits.

## Documentation

- [ ] Example gallery: short transcripts of each subcommand against a
      well-known small model.
- [ ] Backend matrix table: which subcommand × backend combinations are
      tested.
