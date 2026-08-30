# Local source patches

Unified diffs applied to the cloned upstream trees on every build by
`GgmlBuilder._apply_source_patches()` (`scripts/manage.py`). They cover defects
chimera cannot work around from its own code — an upstream `GGML_ASSERT` that
aborts the process, or a compile flag that has to be set inside upstream's own
`if (MSVC)` block.

## How they are applied

- `ggml-*.patch` go to **all three** upstreams (llama.cpp, whisper.cpp,
  stable-diffusion.cpp); `<project>-*.patch` go only to the matching one.
- Applied with `git apply -p1` from the tree root, in sorted filename order.
  Where two patches touch the same file, the later one's hunk offsets already
  account for the earlier one.
- Idempotent and self-disabling. `git apply --check` decides; a patch that is
  already applied, or that no longer applies (upstream landed an equivalent fix,
  or refactored the context away), is logged and skipped, never fatal. A version
  bump therefore cannot break the build on a stale patch — but it can silently
  stop fixing something, so check the build log after every bump.
- For stable-diffusion.cpp they run *after* `_sync_ggml_abi()`, so `ggml-*.patch`
  land on whichever ggml is actually compiled. Since that sync copies llama.cpp's
  already-patched ggml over SD's, the ggml patches normally register as
  already-applied on the SD pass.

Trees are wiped and re-fetched by `make reset`, so nothing here is persistent
state — the `.patch` files are the single source of truth, and double as the
payload for the corresponding upstream PR.

## Current patches

### `stable-diffusion.cpp-conditioner-compute-failure.patch`

A failed text-encoder graph aborted the process. `GGMLRunner::compute()` reports
failure as an empty `std::optional`, `take_or_empty()` flattens that to an empty
tensor, and the conditioners `GGML_ASSERT(!hidden_states.empty())` on the result
— so any budget that pushes the text encoder through the graph-cut segmented
path (e.g. `--max-vram 3.5`) killed `chimera sd`, and killed the whole server on
a `chimera serve` image route rather than failing one request. Several of
`compute()`'s failure paths also log nothing, so the abort arrived with no cause.
The patch logs the dropped failure and propagates it through the LLM
conditioner's existing error channel, so generation fails cleanly.

### `stable-diffusion.cpp-graph-cut-budget-clamp.patch`

`--max-vram` / `--sd-max-vram` budgets ignored VRAM that was already in use. The
graph-cut planner clamps its budget to the VRAM actually free at plan time, but
only under `--stream-layers`. Modules are budgeted once at init from *free* VRAM
and a params storage block is only reclaimed when its tensors are disk-backed, so
by the time the diffusion model plans its graph the text encoder's weights are
still resident and several GiB of the init-time budget no longer exist. The patch
ungates the clamp, so each module plans against what is genuinely free.

### `stable-diffusion.cpp-msvc-bigobj.patch`

`src/stable-diffusion.cpp` has grown past the COFF 65,279-section limit
(`error C1128`). The flag has to go inside upstream's own `if (MSVC)` block:
passing `-DCMAKE_CXX_FLAGS=/bigobj` on the command line pre-seeds the cache so
CMake's platform init never runs, which *replaces* the MSVC defaults instead of
appending and silently drops `/EHsc` from every SD translation unit.

### `ggml-metal-pin-msl-version.patch` / `ggml-metal-pin-msl-version-perkind.patch`

Metal derives the MSL version from the SDK the *host process* linked against
whenever `MTLCompileOptions.languageVersion` is unset, so shader compilation
depended on the binary rather than on the machine. Below MSL 3.1 the embedded
library fails to compile outright (`no matching constructor for ... 'threadgroup
metal::half4x4[512]'`) and the Metal backend never initializes; below 3.1 the
bf16 kernels are also `#undef`'d while `props.has_bfloat` stays true. Both
variants install the same `@available` ladder (3.2 on macOS 15+, 3.1 on 14+, 3.0
on 13+, deliberately stopping below 4.0 so the Metal 4 tensor kernels stay off);
they differ only in the surrounding context. `-perkind` matches trees that have
llama.cpp's per-kind Metal library split (llama.cpp `v0.3.0`), the plain one
matches trees that have not (whisper.cpp `v1.9.2`). Exactly one applies per tree;
the other self-disables.
