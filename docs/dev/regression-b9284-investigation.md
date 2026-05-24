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

## Status

- `main` baseline is established and tight (3.5% spread on vision
  pipeline across N=3).
- `dev` regression is confirmed real, not noise.
- No commit-level localization yet. Next step is the crossed-build
  table above, then a clean bisect on whichever side(s) own the
  regression -- with `make reset` between iterations.

## Artifacts

Timing JSONs (in repo root, gitignored):

- `test_timings_main_{1,2,3}.json` -- N=3 on main (b9119 / sd 596).
- `test_timings_dev_{1,2,3}.json`  -- N=3 on dev  (b9284 / sd 645).
- `test_timings_main.json`, `test_timings_dev.json` -- N=1 originals,
  superseded by the N=3 sets.
