# Releasing chimera

chimera ships prebuilt single-file binaries as GitHub release assets. Two
workflows produce them and attach them to the **same** release for a tag:

| Workflow | Builds | Notes owner |
|---|---|---|
| `.github/workflows/release.yml` | portable CPU + macOS Metal | yes (CHANGELOG) |
| `.github/workflows/release-gpu.yml` | CUDA / Vulkan / ROCm / SYCL | no (assets only) |

This doc covers the GPU workflow and how the two fit together. For the
GPU *build* mechanics themselves (containers, SDKs, toolchains, per-backend
link assertions) the authoritative reference is the compile-only CI sibling,
`.github/workflows/ci-gpu.yml`, whose per-leg comments `release-gpu.yml`
points back to rather than duplicating.

## Cutting a release

Releases are tag-driven. Both workflows trigger on the same tag patterns
(bare semver `0.2.4` or `v0.2.4`):

```bash
# from a clean main with CHANGELOG.md updated for the version
git tag 0.2.4
git push origin 0.2.4
```

That one push fans out to both workflows. Each can also be re-run manually
via `workflow_dispatch` with a `tag` input (the tag must already exist on
origin -- `release-gpu.yml`'s `preflight` job verifies this before any
expensive build starts, failing fast with a copy-pasteable fix instead of an
opaque checkout error).

## How the two workflows share one release

`release.yml` is fast (CPU/Metal builds finish in minutes) and **owns the
release body**: its `publish` job extracts the version's section from
`CHANGELOG.md` via `scripts/release_notes.py` and sets it as the release
notes (falling back to GitHub's auto-generated notes when no section is
found).

`release-gpu.yml` is slow (the cold Windows-CUDA leg alone is ~75 min) and
**only attaches its GPU archives**. Its `publish` job calls
`softprops/action-gh-release` with `files:` but deliberately *no* `body`,
`body_path`, or `generate_release_notes`. action-gh-release leaves an
existing release's body untouched when none of those are set, so:

- The CHANGELOG notes are published exactly once, by `release.yml`.
- `release-gpu.yml` never writes or regenerates them -- no duplication, no
  body race between the two workflows.
- Ordering is irrelevant: in the normal case `release.yml` finishes first and
  writes the notes; if the GPU legs somehow win, action-gh-release creates a
  bodyless release and `release.yml` fills in the notes afterward.

This no-notes invariant is structural -- do not add `body_path` /
`generate_release_notes` to `release-gpu.yml`'s publish step. The `with:` keys
there are intended to be exactly `tag_name`, `files`, `fail_on_unmatched_files`.

## Asset naming

Both workflows use the same scheme so a release's asset list reads uniformly:

```text
chimera-<version>-<target>.<ext>
```

The archive's inner binary is always the bare `chimera` (or `chimera.exe`),
matching what users expect after extraction. `<ext>` is `.tar.gz` everywhere
except Windows, which gets `.zip`. Targets:

| Source | Target | Asset |
|---|---|---|
| release.yml | macos-arm64 | `chimera-<v>-macos-arm64.tar.gz` |
| release.yml | linux-x86_64 (CPU) | `chimera-<v>-linux-x86_64.tar.gz` |
| release.yml | windows-x86_64 (CPU) | `chimera-<v>-windows-x86_64.zip` |
| release-gpu.yml | linux-x86_64 + CUDA | `chimera-<v>-linux-x86_64-cuda.tar.gz` |
| release-gpu.yml | linux-x86_64 + Vulkan | `chimera-<v>-linux-x86_64-vulkan.tar.gz` |
| release-gpu.yml | linux-x86_64 + ROCm | `chimera-<v>-linux-x86_64-rocm.tar.gz` |
| release-gpu.yml | linux-x86_64 + SYCL | `chimera-<v>-linux-x86_64-sycl.tar.gz` |
| release-gpu.yml | windows-x86_64 + CUDA | `chimera-<v>-windows-x86_64-cuda.zip` |
| release-gpu.yml | windows-x86_64 + Vulkan | `chimera-<v>-windows-x86_64-vulkan.zip` |

A complete release therefore carries nine assets. Worked example: the 0.2.4
release shipped all nine, with a single release body.

## release-gpu.yml job graph

```text
preflight (verify tag on origin)
   |
   +-- build-cuda            (container: nvidia/cuda:12.4.1-devel)
   +-- build-vulkan          (ubuntu-latest + apt vulkan/glslc/spirv-headers)
   +-- build-vulkan-windows  (windows-latest + choco vulkan-sdk + MSVC)
   +-- build-cuda-windows    (windows-latest + Jimver/cuda-toolkit + MSVC)
   +-- build-rocm            (container: rocm/dev-ubuntu-22.04:6.4.4)
   +-- build-sycl            (container: intel/oneapi-basekit:2025.3.2)
        |
        v
   publish  (needs ALL six legs; attaches archives to the release)
```

Each build leg, in order:

1. checks out the release tag (`ref: ${{ inputs.tag || github.ref }}`),
2. installs its toolchain/SDK and restores the per-backend `thirdparty`
   cache (keyed on `hashFiles('scripts/manage.py')` only -- chimera-side
   CMake edits do not invalidate the vendored deps),
3. builds via the matching `make build-<backend>` target with
   `DEPS_EXTRA=--no-sd-examples`,
4. **asserts the backend actually linked** before staging (a release must
   never ship a binary where the `ggml-<backend>` archive silently dropped):
   `nm` for a registration symbol on Linux CUDA/Vulkan, `readelf -d` NEEDED
   for ROCm (`libamdhip64`/rocBLAS) and SYCL (`libsycl`), `dumpbin
   //DEPENDENTS` for the Windows DLL imports,
5. stages the versioned archive (strip on unix; no strip on Windows -- the
   `.exe` carries no debug info and the runner has no `strip`),
6. uploads it as a per-job artifact for the `publish` job to collect.

## Two policy knobs to know before relying on this

These are intentional, documented in the workflow header, and worth a
conscious decision per project phase:

- **Single GPU arch.** `CMAKE_CUDA_ARCHITECTURES=75` and
  `CMAKE_HIP_ARCHITECTURES=gfx1030` are inherited from `ci-gpu.yml` to bound
  the (very slow) nvcc/hipcc compile. A shipped binary built for one arch is
  narrower than users may expect; widening these envs multiplies compile time.
  The SYCL/Vulkan legs are not arch-pinned this way.
- **Strict publish (no partial GPU release).** `publish` `needs` every leg and
  uses `fail_on_unmatched_files: true`, mirroring `release.yml`'s "no partial
  releases" stance: if any one leg fails (most likely the ~75-min cold
  Windows-CUDA leg), *no* GPU assets ship. To allow a partial GPU release,
  drop the flaky leg from `publish.needs`, add `if: always()` to `publish`,
  and set `fail_on_unmatched_files: false`.

## SYCL build note

The SYCL leg depends on the `if(GGML_SYCL)` block in the top-level
`CMakeLists.txt`, which strips `-fsycl` out of the imported
`IntelSYCL::SYCL_CXX` / `MKL` / `oneDNN` targets'
`INTERFACE_COMPILE_OPTIONS` so it never lands on chimera's own C
amalgamations (icx otherwise reinterprets `.c` as C++ and the SQLite/
sqlite-vec amalgamations fail to compile). `-fsycl` is re-added to the link
line only. If the SYCL leg ever regresses with "treating 'c' input as 'c++'",
that block -- not the workflow -- is where to look; the leg's build step
comment explains how to re-enable `VERBOSE=1` to capture the offending `icx`
command line.

## Validating workflow edits locally

The CI/release YAML is checked with [`actionlint`](https://github.com/rhysd/actionlint)
(which runs `shellcheck` over each `run:` block). All four workflows
(`ci.yml`, `release.yml`, `ci-gpu.yml`, `release-gpu.yml`) lint clean. The one
deliberate suppression is `# shellcheck disable=SC2012` on the Windows
Vulkan-SDK version glob (`ls -d /c/VulkanSDK/* | sort -V`), present identically
in both `ci-gpu.yml` and `release-gpu.yml`.
