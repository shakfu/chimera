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

- [x] `embed` subcommand — emit embedding vectors for one or more inputs.
- [x] `tokenize` subcommand — debug helper for prompt tokenization.
- [x] Streaming output for `whisper` (segment-callback driven).
- [x] Per-turn sampler state in `chat` (sampler + KV cache now persistent
      across turns).
- [x] `--system-prompt-file` for `chat`.
- [x] Image-to-image and inpainting modes for `sd`.
- [ ] Multimodal (`mtmd`) image input for `gen` / `chat`. Blocked on
      `stbi_load` symbol clash between `libmtmd.a` and our `stb_impl.cpp`
      once any `mtmd_helper_*` is referenced. Resolve by dropping
      `STB_IMAGE_IMPLEMENTATION` from `stb_impl.cpp` and relying on
      libmtmd's copy (couples chimera_sd's image read to mtmd, which is
      acceptable if mtmd is always linked). Then add `--mmproj` + repeated
      `--image` flags and route through `mtmd_helper_eval_chunks`.

## Quality

- [ ] Replace ad-hoc WAV parser in `chimera_whisper.cpp` with a real one
      (currently rejects anything that isn't 16-bit PCM mono).
- [x] Exit codes: distinguish parse errors, model-load failures, and
      generation failures (`ExitCode` enum + `ChimeraError`).
- [ ] Document the model formats each subcommand accepts.
- [x] `--version` reports bundled llama.cpp / whisper.cpp / sd.cpp pins.

## Documentation

- [ ] Example gallery: short transcripts of each subcommand against a
      well-known small model.
- [ ] Backend matrix table: which subcommand × backend combinations are
      tested.
