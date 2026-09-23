# Local source patches

Unified diffs applied to the cloned upstream trees on every build by
`GgmlBuilder._apply_source_patches()` (`scripts/manage.py`). They cover defects
chimera cannot work around from its own code — an upstream `GGML_ASSERT` that
aborts the process, or a compile flag that has to be set inside upstream's own
`if (MSVC)` block.

## How they are applied

- `ggml-*.patch` go to llama.cpp's and whisper.cpp's trees, which vendor
  upstream ggml. Not to stable-diffusion.cpp: shared-ggml mode compiles
  llama.cpp's patched tree (`SD_GGML_SOURCE_DIR`), and `SD_USE_VENDORED_GGML=1`
  compiles leejet's fork, whose layout they do not match. A vendored Metal build
  therefore ships sd.cpp without the MSL pin. `<project>-*.patch` go only to the
  matching tree.
- Applied with `git apply -p1` from the tree root, in sorted filename order.
  Where two patches touch the same file, the later one's hunk offsets already
  account for the earlier one.
- An already-applied patch is skipped. A patch that no longer applies fails the
  build and prints git's reason. Rebase it, or delete it if upstream fixed the
  defect, and record which under "Retired patches". A skipped patch ships the
  build without its fix; chimera 0.3.0 and 0.3.1 shipped without the Metal MSL
  pin this way.

Trees are wiped and re-fetched by `make reset`, so nothing here is persistent
state — the `.patch` files are the single source of truth, and double as the
payload for the corresponding upstream PR.

## Current patches

### `ggml-metal-pin-msl-version-set-lang.patch`

Metal derives the MSL version from the SDK the *host process* linked against
whenever `MTLCompileOptions.languageVersion` is unset, so shader compilation
depended on the binary rather than on the machine. Below MSL 3.1 the embedded
library fails to compile outright (`no matching constructor for ... 'threadgroup
metal::half4x4[512]'`) and the Metal backend never initializes; below 3.1 the
bf16 kernels are also `#undef`'d while `props.has_bfloat` stays true. The patch
pins an `@available` ladder (3.2 on macOS 15+, 3.1 on 14+, 3.0 on 13+) inside
`ggml_metal_compile_options_set_lang()`.

It stops below 4.0. llama.cpp v0.4.0+ otherwise requests MSL 4.0 on
tensor-capable GPUs (M5/M6/A19/A20), and the MSL 4.0 tensor kernels blank
stable-diffusion output. It matches llama.cpp v0.4.1 and whisper.cpp v1.9.4.

## Retired patches

The three stable-diffusion.cpp patches were dropped at `master-898-2bb7294`,
each fixed upstream. The two older Metal patches were replaced.

- `conditioner-compute-failure`: `LLMEmbedder` returns an empty condition
  instead of asserting, and the pipeline fails the generation.
- `graph-cut-budget-clamp`: segmentation is now planned per run against live
  free VRAM ([#1878](https://github.com/leejet/stable-diffusion.cpp/pull/1878)).
- `msvc-bigobj`: upstream adds `/bigobj` to the `stable-diffusion` target.
- `ggml-metal-pin-msl-version{,-perkind}.patch`: stopped matching at llama.cpp
  v0.4.0 and whisper.cpp v1.9.4, which moved both compile sites into
  `ggml_metal_compile_options_set_lang()`. Replaced by `-set-lang`.
