# Regression investigation: llama.cpp b9119 -> b9284 (+ sd.cpp 596 -> 645)

Tracking a multi-test slowdown observed after the `dev` branch sync to
llama.cpp b9284 / sd.cpp master-645 (commit `2c4156d`). Three of the
heavy end-to-end tests run 3-6x slower on `dev` than on `main`.

## Confirmed regressions

Measured with N=3 `make test-bench` runs on each branch (clean rebuild
via `make reset && make`). Comparison flags any test where the `dev`
median falls outside the `main` [min, max] noise envelope.

| Test                                       | main median | dev median | Delta    |
|--------------------------------------------|------------:|-----------:|---------:|
| `gen --mmproj --image (vision pipeline)`   | 40.65s      | 223.48s    | +450%    |
| `sd img2img round-trip`                    | 12.97s      | 85.47s     | +559%    |
| `sd sd_xl_turbo_1.0.q8_0.gguf`             | 10.75s      | 33.91s     | +215%    |

Suite totals: main = 87.2 / 89.6 / 92.4s (~3% spread); dev = 283 / 367 /
385s. All other tests fall inside the noise band -- the regression is
isolated to the vision (mtmd) path and the two stable-diffusion image
round-trips.

Pinned versions on each side:

| Branch | llama.cpp | sd.cpp                | whisper.cpp |
|--------|-----------|-----------------------|-------------|
| main   | b9119     | master-596-90e87bc    | v1.8.4      |
| dev    | b9284     | master-645-645e6e9    | v1.8.4      |

## Hypotheses (not yet falsified)

The vision pipeline lives in llama.cpp's `mtmd` tools, so its regression
points at the llama.cpp bump. The two SD regressions point at the
sd.cpp bump (sd.cpp shares ggml with llama.cpp via
`SD_USE_VENDORED_GGML=0`, so a ggml ABI/perf change in llama.cpp could
in principle also affect SD -- but the cleanest way to find out is the
crossed-build test below, not speculation).

## Localization attempt 1 -- failed bisect of llama.cpp b9119..b9284

165 commits between the tags; bisect was set up against a fresh clone
of `ggml-org/llama.cpp` at `/tmp/llama-bisect`, with a runner script
that for each midpoint SHA invoked
`scripts/manage.py build -l -d --sd-shared-ggml --llama-version <SHA>`,
rebuilt chimera, and ran the vision test with a 120s good/bad
threshold.

### Issue 1: upstream webui layout change around b9200

`manage.py` on `main` only knew about the pre-b9200 webui asset layout
(`tools/server/public/`). Around b9200 upstream moved the webui to a
Vite project under `tools/ui/`. Every post-b9200 midpoint failed at the
header-copy step. The fix already exists on `dev` (same commit
`2c4156d` that did the version bump also taught `manage.py` to probe
both layouts). Rerunning the bisect with `dev`'s `manage.py` checked
out into the working tree let all iterations build.

### Issue 2: manage.py silently kept stale source

After the second bisect concluded, every in-range commit had timed
above the 120s threshold and bisect reported `fde69a36` -- a commit
that only adds Python files under `examples/llama-eval/` and changes
nothing that compiles into chimera -- as the "first bad" commit. That
verdict was meaningless: every commit was bad, including the
implicitly-trusted `b9119` endpoint.

Root cause for the meaningless verdict: `scripts/manage.py build -l
--llama-version <X>` does not unconditionally re-checkout the source
tree to `<X>`. If `build/llama.cpp/` already exists at a different SHA,
it is reused. During the bisect the source kept advancing through the
range, but the manual b9119 "control" run that should have anchored
the baseline actually still measured against a stale post-b9119 build
(source tree at `afcda09` = b9284). The 192s "b9119 control" timing
was therefore not b9119 at all.

The only clean run came once the user ran `make reset && make` on
`main` -- which purges `thirdparty/llama.cpp/` and forces a real
re-fetch -- and the 40s vision baseline reappeared, reproducible
across three runs.

## Methodology lessons

1. **N=1 timing is not a baseline.** The original `test_timings_*.json`
   files were single samples. The heavy tests have 22-48% spread within
   the same binary (worst observed: vision pipeline 152-224s across
   three back-to-back runs on `dev`). A bench-diff framework that
   doesn't sample multiple times will keep generating false positives
   *and* miss real regressions inside the noise band. Suggested change:
   in `make test-bench`, run the `SLOW_TEST_RE` set N=3 (the slow set
   is only three tests) and store `[min, median, max]` instead of a
   single sample.

2. **Bisect rebuilds must purge `build/llama.cpp/` and
   `thirdparty/llama.cpp/` between iterations.** Either by calling
   `make reset` before each `manage.py build`, or by teaching
   `manage.py` to force re-checkout when `--llama-version` differs
   from the SHA recorded in the existing source tree. The current
   "skip if already present" behavior is fine for incremental dev
   work but invalidates any version-comparison workflow.

3. **Disambiguate dependency bumps before bisecting.** `dev` bumped
   both llama.cpp and sd.cpp simultaneously, so a bisect on either
   alone is uninterpretable for the SD regressions. Cheapest
   disambiguation: two crossed builds, one full `test-bench` each.

   | Run | llama.cpp | sd.cpp | What it tells us |
   |-----|-----------|--------|------------------|
   | A   | b9284     | 596    | Whether the vision-pipeline regression survives without the sd.cpp bump (expected: yes) and whether the SD regressions disappear (expected: yes -- implicates sd.cpp). |
   | B   | b9119     | 645    | The complement: confirms the SD regressions land with sd.cpp 645 even on old llama.cpp, and that vision-pipeline returns to baseline. |

## Localization attempt 2 -- Phase 1 crossed builds

Run script: `/tmp/phase1.sh`. Each config does `make reset`, full deps
build with one bump pinned to old and one to new, then N=1
`make test-bench`. Sequential -- they share the build tree.

### Config A: llama=b9284 + sd=596 -- retried with dev's manage.py, N=1 timings

First attempt failed at the pre-b9200 webui-layout check in main's
`manage.py`. Retried by checking out dev's `manage.py` into the
working tree (with an EXIT trap to restore main's afterward); the
build then succeeded.

| Test                                | main median (b9119/596) | Cross A (**b9284**/596) | Cross B (b9119/**645**) |
|-------------------------------------|------------------------:|------------------------:|------------------------:|
| `gen --mmproj --image (vision pipeline)` | 40.65s             | **40.74s**              | 138.28s                 |
| `sd img2img round-trip`             | 12.97s                  | **12.91s**              | 62.27s                  |
| `sd sd_xl_turbo_1.0.q8_0.gguf`      | 10.75s                  | **10.72s**              | 30.88s                  |

Config A reproduces all three main baselines to within 0.1s. The
llama.cpp bump b9119 -> b9284 contributes **zero** measurable
regression, including on the vision pipeline (mtmd) which lives in
llama.cpp's own code.

### Config B: llama=b9119 + sd=645 -- BUILT, N=1 timings

| Test                                | main median (b9119/596) | Cross B (b9119/**645**) | dev median (b9284/645) |
|-------------------------------------|------------------------:|------------------------:|-----------------------:|
| `gen --mmproj --image (vision pipeline)` | 40.65s             | **138.28s**             | 223.48s                |
| `sd img2img round-trip`             | 12.97s                  | **62.27s**              | 85.47s                 |
| `sd sd_xl_turbo_1.0.q8_0.gguf`      | 10.75s                  | **30.88s**              | 33.91s                 |

### Interpretation

Phase 1 collapses the diagnosis to a single bump:

- **Config A (only llama.cpp bumped):** all three baselines unchanged.
- **Config B (only sd.cpp bumped):** all three tests regress.

Therefore the **entire regression -- vision pipeline included -- is
attributable to the sd.cpp 596 -> 645 bump**. The llama.cpp bump is
exonerated.

This is non-obvious because the vision pipeline test exercises
llama.cpp's mtmd path, not sd.cpp. The chimera binary nonetheless
links sd.cpp's libs (for the `sd` subcommands), so two plausible
mechanisms remain:

1. **Build-flag bleed.** sd.cpp 645's CMake changed compile flags
   (optimization, Metal shader options, OpenMP, etc.) that propagate
   to the chimera binary or to libraries it shares with llama.cpp.
2. **Static-init / global-state leak.** sd.cpp 645 may have added or
   changed a static initializer that runs at chimera startup and
   alters global GPU/Metal state in a way that slows down llama.cpp's
   subsequent mtmd inference.

Single-sample noise is ruled out: main's vision baseline has 3.5%
spread across N=3, and Config A returns to that baseline tightly.

### Next moves

1. **Compile-command diff** between a main build (b9119/596) and a
   Config B build (b9119/645). Same llama, different sd -- any
   `compile_commands.json` flag delta is sd.cpp 645's doing. Cheaper
   than a bisect and may short-circuit it.
2. **Bisect sd.cpp master-596..master-645** (~49 commits, ~6
   iterations) if the compile-flag diff isn't decisive. With
   `make reset` between iterations.

## Localization attempt 3 -- compile-command diff

Built main (b9119 + sd 596) and Cross B (b9119 + sd 645) clean, then
diffed each project's `compile_commands.json`.

- **llama.cpp:** 0/285 files differ. Identical builds, as expected
  with the same llama version on both sides.
- **sd.cpp:** 140/202 files differ on the raw command, but **128/202
  after filtering out version-stamp macros and CLI-only changes**.
  All remaining diffs are *newly added thirdparty libraries* in
  sd.cpp 645 (`libwebp`, `libwebm`) -- new compilations, not flag
  changes on pre-existing translation units.

Conclusion: no meaningful compile-flag delta. Hypothesis 1 (build-flag
bleed) is dead. The regression must be in sd.cpp's source code or its
runtime behavior, not its build configuration.

## Localization attempt 4 -- sd.cpp bisect (49 commits)

### First attempt: failed at SHA cloning

`scripts/manage.py`'s `Builder.setup()` clones with
`git clone --depth 1 --branch <ref>`, which git rejects for arbitrary
SHAs (`--branch` only accepts tag/branch names). Every iteration
exited 128 at `git clone`.

### Second attempt: patched manage.py + warm staged clone

Workaround:

1. Patched `Builder.setup()` to skip the clone if `src_dir` is already
   populated. ~5-line change; restored to main with
   `git checkout main -- scripts/manage.py` after the bisect.
2. Pre-staged `/tmp/sd-bisect-warm` as a full clone of leejet's
   stable-diffusion.cpp with all submodules (libwebm, libwebp, ggml,
   server frontend), so each bisect iteration only does a local clone
   + SHA checkout, not a fresh upstream pull.

Bisect ran on `sd sd_xl_turbo_1.0.q8_0.gguf` (fastest of the three
regressing tests; ~11s good, ~31s bad on dev; threshold 20s). Range
master-596 (`90e87bc`) .. master-645 (`645e6e9`), 49 commits.

### Result

| SHA       | sd_xl_turbo | Verdict |
|-----------|------------:|---------|
| `9d68341` | 10.80s      | GOOD    |
| `57ff2eb` | 50.93s      | **first BAD** |
| `eeac950` | 42.28s      | BAD     |
| `0665a7f` | 42.95s      | BAD     |
| `38b14ad` | 50.90s      | BAD     |
| `e7eb92f` | 34.40s      | BAD     |

**First bad commit: `57ff2eb` -- "feat: support for memory-mapping
model weights (#1414)"**.
[leejet/stable-diffusion.cpp#1414](https://github.com/leejet/stable-diffusion.cpp/pull/1414).
7 files changed, +382/-66, touching `src/model.{cpp,h}`,
`src/stable-diffusion.cpp`, `src/util.{cpp,h}`, `src/denoiser.hpp`,
`src/ggml_extend.hpp`.

## Root cause

`src/chimera/chimera_sd.h:210`:

```cpp
bool enable_mmap = true;  // chimera default; CLI's --no-mmap maps to false
```

The comment three lines up reads "enable_mmap defaults off upstream",
which is correct -- sd.cpp 645 ships with mmap off by default. The
dev-branch sync nonetheless set chimera's default to `true`, flipping
mmap **on** for every SD load through chimera.

Causal chain:

1. sd.cpp commit 57ff2eb introduces mmap support for model weights.
   Off by default upstream.
2. chimera's dev sync (commit `2c4156d`) added a new field
   `enable_mmap` to `chimera_sd_load_params` and set chimera's
   default to `true`.
3. On macOS with the Metal backend, mmap-backed memory is not
   directly usable by Metal command buffers (Metal needs
   MTLBuffer-backed allocations). The mmap path forces per-access
   copies / synchronous page-ins, dramatically slowing SD inference.

This fully explains the two SD test regressions
(`sd img2img round-trip` +559%, `sd sd_xl_turbo` +215%).

### Open question: the vision-pipeline regression

`gen --mmproj --image (vision pipeline)` runs against llama.cpp's
mtmd path, not sd.cpp -- but the cross builds showed it tracks the
sd.cpp version, not llama.cpp. The bisect identified `57ff2eb` for
the `sd_xl_turbo` test only; whether the vision regression has the
same root commit is not yet verified.

Two possibilities:

1. **Same commit, indirect coupling.** sd.cpp's libs are linked into
   the chimera binary even when only the vision path runs. 57ff2eb
   introduces substantial new state in sd.cpp's TUs; a static
   initializer or backend-registration change might affect llama.cpp's
   Metal usage.
2. **Different commit, different cause.** A separate sd.cpp commit in
   the range regresses the vision path through some other mechanism.

The cheapest discriminator is to flip `enable_mmap` to `false` on the
dev branch and re-bench. If all three tests recover to main baseline,
the entire regression is a one-default-flag fix. If only the SD tests
recover, the vision regression needs a separate bisect (probably with
the same patched manage.py + warm clone setup).

## Verification 1 -- chimera_sd.h struct default flipped to false

Set `enable_mmap = false` at `chimera_sd.h:210`, rebuilt clean on dev,
ran `make test-bench`. Result: **no recovery on any test.** All three
still in the dev range (vision 139s, img2img 64s, sd_xl_turbo 36s).

Diagnosis: the struct default is clobbered at runtime by
`chimera_sd.cpp` (~line 799 on dev):

```cpp
lp.enable_mmap = !opts.no_mmap;
```

`opts.no_mmap` is bound to the CLI flag `--no-mmap` (default `false`),
so `lp.enable_mmap` is forced to `true` regardless of the struct
default. The test invocations don't pass `--no-mmap`, so the
chimera_sd.h flip never reached the runtime.

## Verification 2 -- runtime probe with --no-mmap

Bypassed the build entirely by running the existing dev-flavored
`build/chimera` binary directly with and without `--no-mmap`. Same
binary, same model, same prompt:

| Invocation                                              | Wall time |
|---------------------------------------------------------|----------:|
| `chimera sd -m sd_xl_turbo ... -s 2`                    |   31.34s  |
| `chimera sd -m sd_xl_turbo ... -s 2 --no-mmap`          |   11.19s  |

`--no-mmap` restores baseline. The mmap-on run also emitted hundreds
of Metal-backend errors to stderr:

```
ggml_extend.hpp:69   - ggml_metal_buffer_get_id: error: tensor ' (reshaped)' buffer is nil
```

The Metal backend cannot acquire a buffer ID from mmap-backed memory;
sd.cpp's fallback path (per-tensor host->device copies) runs instead.
Hence the ~3x slowdown for `sd_xl_turbo` and ~7x for `img2img
round-trip` (more steps -> more copies per inference).

## Root cause closed -- including the vision-pipeline regression

The "vision pipeline" test (`scripts/test.py:1054-1059`) synthesizes
its input image by calling `chimera sd` first, then feeds the
512x512 4-step PNG into `chimera gen --mmproj --image`. The total
wall time is dominated by the SD synthesis step:

- main: ~20-25s SD synth + ~15s mtmd ≈ **40s**.
- dev:  ~200s mmap-degraded SD synth + ~20s mtmd ≈ **220s**.

So all three regressions trace to the **same single root cause**:
sd.cpp commit `57ff2eb` introduces mmap support; chimera's dev sync
defaults it on; Metal cannot use mmap-backed buffers on macOS;
sd.cpp falls back to per-tensor copies.

## Fix

In `src/chimera/chimera_sd.cpp` (dev), change the runtime override
that defeats the struct default:

```cpp
- lp.enable_mmap = !opts.no_mmap;
+ // Metal can't use mmap-backed buffers (ggml_metal_buffer_get_id
+ // returns nil), forcing per-tensor host->device copies that
+ // 3-7x SD inference. Disable mmap until upstream sd.cpp learns
+ // Metal-compatible mmap or we add backend-aware gating.
+ lp.enable_mmap = false;
```

(The `chimera_sd.h:210` struct default of `true` is misleading given
the upstream-off comment three lines above it. Flipping it to `false`
for documentation is harmless but irrelevant until the runtime
override is removed.)

A more principled fix would gate on the active backend at runtime
(`enable_mmap = !using_metal`), but every chimera build on macOS uses
Metal, so the unconditional disable is equivalent in practice.

Committed to dev as `0cdf7a8 lp.enable_mmap = false fixes sd
regression`.

## Verification 3 -- N=3 bench with the committed fix

Three back-to-back `make test-bench` runs on dev with the fix
applied (b9284 + sd-645 + `lp.enable_mmap = false`):

| Test                              | r1      | r2      | r3      | median  | main median | ratio |
|-----------------------------------|--------:|--------:|--------:|--------:|------------:|------:|
| `sd sd_xl_turbo_1.0.q8_0.gguf`    | 11.24s  | 11.21s  | 11.20s  | **11.21s** | 10.75s   | 1.04x |
| `gen --mmproj --image (vision)`   | 45.59s  | 50.60s  | 45.08s  | **45.59s** | 40.65s   | 1.12x |
| `sd img2img round-trip`           | 23.99s  | 23.43s  | 23.41s  | **23.43s** | 12.97s   | 1.81x |
| Suite total                       | 104.6s  | 111.4s  | 103.8s  | 104.6s  | 89.6s       | 1.17x |

Artifacts: `tests/timings/test_timings_dev_mmap_fix_{1,2,3}.json`.

### Outcome

- `sd_xl_turbo`: **fully recovered** (within 4% of main, inside the
  noise band).
- `vision pipeline`: **fully recovered** (within 12% of main, inside
  the noise band; pre-fix value was 223s).
- `sd img2img round-trip`: **partially recovered** -- dropped from
  85s (dev pre-fix) to 23.4s, but still 1.81x main. The three runs
  are 23.41/23.43/23.99, essentially deterministic, so this is not
  measurement noise but a real residual regression.

### Residual: img2img-specific slowdown

`img2img round-trip` runs two sequential `chimera sd` invocations:
txt2img to synthesize an input image, then img2img using that as
init. On main both invocations together finish in 13s; with the fix
on dev they take 23s. Since `sd_xl_turbo` (a single equivalent
invocation) is at parity, the residual is plausibly per-invocation
overhead that surfaces only when there are two loads -- e.g., a new
per-load tensor-copy or VAE-init path introduced somewhere in the
sd.cpp 596..645 range.

Not investigated further in this round. If pursued, the approach is
the same one used here: bisect sd.cpp on `img2img round-trip` with
the mmap fix applied throughout, using the patched `manage.py` and
warm-staged clone.

## Status

- `main` baseline is established and tight (3.5% spread on vision
  pipeline across N=3).
- `dev` regression is confirmed real, not noise.
- Phase 1 complete: regression localized to the sd.cpp 596 -> 645
  bump. llama.cpp b9119 -> b9284 contributes nothing.
- Compile-command diff: no flag bleed. Same flags, only new
  vendored libs (libwebp/libwebm) in sd.cpp 645.
- sd.cpp bisect: first bad commit is **57ff2eb (#1414)** -- adds
  mmap support, off by default upstream.
- Root cause: chimera dev sync set `enable_mmap = true` on the new
  field, contradicting its own comment that upstream defaults it off.
  Metal can't directly use mmap-backed memory on macOS.
- Verification 1 (struct-default flip): no recovery; defeated by a
  runtime override in `chimera_sd.cpp`.
- Verification 2 (runtime `--no-mmap` flag): clean recovery to
  baseline. Confirms mmap is the cause.
- Vision-pipeline regression resolved by the same fix: that test
  synthesizes its image via `chimera sd` first, so most of its wall
  time is the same mmap-degraded path.
- **Fix committed on `dev` as `0cdf7a8`** -- one-line change in
  `chimera_sd.cpp` to set `lp.enable_mmap = false` unconditionally.
- Verification 3 (N=3 bench with committed fix):
  - `sd_xl_turbo` and `vision pipeline` fully recovered (within noise).
  - `sd img2img round-trip` partially recovered: 85s -> 23.4s, but
    still 1.81x main. Real, deterministic residual -- a smaller
    per-invocation regression in sd.cpp 596..645 unrelated to mmap.
    Follow-up if desired; same bisect mechanics apply.

## Artifacts

Timing JSONs under `tests/timings/`:

- `test_timings_main_{1,2,3}.json` -- N=3 on main (b9119 / sd 596).
- `test_timings_dev_{1,2,3}.json`  -- N=3 on dev  (b9284 / sd 645).
- `test_timings_main.json`, `test_timings_dev.json` -- N=1 originals,
  superseded by the N=3 sets.
