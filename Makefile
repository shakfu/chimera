.PHONY: deps build remake build-with-webui build-cuda build-rocm \
		build-sycl build-vulkan rebuild clean reset \
		test test-fast test-slow test-bench test-bench-fast smoke \
		install uninstall release-notes bump-check test-db-migrate \
		test-golden combine test-external-smoke test-external-oop \
		bindings test-bindings wheel

# Override with `make PYTHON=<cmd>` for
# unusual environments (e.g. `PYTHON="uv run python"`).
PYTHON ?= $(shell \
    if python3 --version >/dev/null 2>&1; then echo python3;  \
    elif python  --version >/dev/null 2>&1; then echo python; \
    elif py -3   --version >/dev/null 2>&1; then echo py -3;  \
    else echo python3; fi)
BUILD_DIR ?= build
PREFIX ?= /usr/local
DESTDIR ?=

build: deps
	@cmake -S . -B $(BUILD_DIR) -DSD_USE_VENDORED_GGML=OFF -DCMAKE_BUILD_TYPE=Release
	@cmake --build $(BUILD_DIR) --target chimera --config Release -j

remake: reset build test

# build-with-webui: same as `build`, but flips the experimental
# CHIMERA_WEBUI_EMBED option ON so the chimera binary bakes upstream's
# prebuilt web UI bundle into itself (GET / + /bundle.{js,css}) via the
# generated ui.cpp. Adds ~7 MB to the binary.
#
# IMPORTANT (llama.cpp b9318+): upstream no longer ships prebuilt assets
# in the source tree, so `deps` alone does NOT stage them and this target
# will FATAL-error at configure. Produce + stage the assets first:
#   (cd build/llama.cpp/tools/ui && npm install && npm run build)
#   python3 scripts/manage.py build --llama-cpp --sd-shared-ggml
# then run `make build-with-webui`. A Node toolchain is required for that
# one-time asset build. See docs/dev/webui.md sections 2.1 and 10 for the
# full picture. Pass --no-webui at runtime to suppress per-invocation.
build-with-webui: deps
	@cmake -S . -B $(BUILD_DIR) -DSD_USE_VENDORED_GGML=OFF -DCMAKE_BUILD_TYPE=Release -DCHIMERA_WEBUI_EMBED=ON
	@cmake --build $(BUILD_DIR) --target chimera --config Release -j

# GPU-backend build targets. Each one builds the third-party deps with the
# matching GGML_<BACKEND>=1 env var (picked up by scripts/manage.py, which
# forwards it to llama.cpp / whisper.cpp / stable-diffusion.cpp configures)
# and then configures chimera with -DGGML_<BACKEND>=ON so the resulting
# binary links the backend's static archive. Backend toolkit (CUDA Toolkit,
# ROCm/HIP, oneAPI, Vulkan SDK) must already be installed on the host.
#
# Override architectures via env vars:
#   CMAKE_CUDA_ARCHITECTURES=89          (Ada/RTX 40xx; default builds many)
#   CMAKE_HIP_ARCHITECTURES=gfx1100      (RDNA3; default builds many — ROCm
#                                         uses HIP under the hood, so the
#                                         env var keeps its upstream name)
# Override toolchains via env vars when not on PATH:
#   CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
# CUDA perf knobs (forwarded by manage.py): GGML_CUDA_FORCE_MMQ,
# GGML_CUDA_FORCE_CUBLAS, GGML_CUDA_FA_ALL_QUANTS, GGML_CUDA_PEER_MAX_BATCH_SIZE.
# ROCm knob: GGML_HIP_ROCWMMA_FATTN=1 for rocWMMA flash attention.
#
# Verify the resulting binary picked up the backend with `./build/chimera info`.
build-cuda:
	@GGML_CUDA=1 $(MAKE) deps
	@cmake -S . -B $(BUILD_DIR) -DSD_USE_VENDORED_GGML=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
	@cmake --build $(BUILD_DIR) --target chimera --config Release -j

build-rocm:
	@GGML_HIP=1 $(MAKE) deps
	@cmake -S . -B $(BUILD_DIR) -DSD_USE_VENDORED_GGML=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON
	@cmake --build $(BUILD_DIR) --target chimera --config Release -j

build-sycl:
	@GGML_SYCL=1 $(MAKE) deps
	@cmake -S . -B $(BUILD_DIR) -DSD_USE_VENDORED_GGML=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON
	@cmake --build $(BUILD_DIR) --target chimera --config Release -j

build-vulkan:
	@GGML_VULKAN=1 $(MAKE) deps
	@cmake -S . -B $(BUILD_DIR) -DSD_USE_VENDORED_GGML=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
	@cmake --build $(BUILD_DIR) --target chimera --config Release -j

rebuild:
	@cmake --build $(BUILD_DIR) --target chimera --config Release -j

deps:
	@SD_USE_VENDORED_GGML=0 $(PYTHON) scripts/manage.py build --all --deps-only --sd-shared-ggml

test:
	@$(PYTHON) scripts/test.py

# test-fast: same suite as `test` but skip the three vision / SD round-trips
# that dominate wall-clock time (vision pipeline ~165s, sd img2img ~75s,
# sd_xl_turbo ~30s). Tight inner-loop tier for edits that don't touch the
# vision or SD paths. The slow set is centralised in SLOW_TEST_RE in
# scripts/test.py — update there when a new test crosses the ~20s mark.
test-fast:
	@$(PYTHON) scripts/test.py --no-slow

# test-slow: complement of test-fast. Runs only the heavy vision / SD
# round-trips. Useful when iterating on the SD pipeline or the mtmd
# vision integration without paying for the rest of the suite.
test-slow:
	@$(PYTHON) scripts/test.py --slow-only

# test-bench: run the full suite and write per-test timings to a JSON
# file for later comparison. Override the destination with
# `make test-bench BENCH=mybench.json`. Companion to scripts/test_diff.py,
# which diffs two of these files and flags regressions:
#
#   git checkout main && make build && make test-bench BENCH=main.json
#   git checkout dev  && make build && make test-bench BENCH=dev.json
#   scripts/test_diff.py main.json dev.json
#
# `test-bench-fast` runs the same with --no-slow for a quick check that
# omits the heavy vision / SD round-trips.
BENCH ?= $(BUILD_DIR)/test_timings.json
test-bench:
	@$(PYTHON) scripts/test.py --timings-out $(BENCH)

test-bench-fast:
	@$(PYTHON) scripts/test.py --no-slow --timings-out $(BENCH)

smoke:
	@$(PYTHON) scripts/test.py --smoke

install: $(BUILD_DIR)/chimera
	@install -d "$(DESTDIR)$(PREFIX)/bin"
	@install -m 755 "$(BUILD_DIR)/chimera" "$(DESTDIR)$(PREFIX)/bin/chimera"
	@echo "installed $(DESTDIR)$(PREFIX)/bin/chimera"

uninstall:
	@rm -f "$(DESTDIR)$(PREFIX)/bin/chimera"
	@echo "removed $(DESTDIR)$(PREFIX)/bin/chimera"

$(BUILD_DIR)/chimera:
	$(MAKE) build

clean:
	@rm -rf $(BUILD_DIR)
	@rm -rf bindings/build

reset: clean
	@rm -rf thirdparty/llama.cpp thirdparty/whisper.cpp thirdparty/stable-diffusion.cpp thirdparty/linenoise
	@rm -rf bindings/.venv

# release-notes: write release-notes.md by extracting the current
# CHIMERA_VERSION's section from CHANGELOG.md. Same script the release
# workflow runs; pulls the version straight from scripts/manage.py so
# the file always matches what the binary will report at runtime.
# Override the version with `make release-notes VERSION=X.Y.Z`.
VERSION ?= $(shell $(PYTHON) -c "import re; \
    print(re.search(r'^CHIMERA_VERSION = \"([^\"]+)\"', open('scripts/manage.py').read(), re.M).group(1))")

release-notes:
	@$(PYTHON) scripts/release_notes.py $(VERSION)

# bump-check: diff the currently-vendored upstream-server headers against
# llama.cpp at $(LLAMA_VERSION) (default: whatever scripts/manage.py has
# pinned). server-context.h / server-http.h aren't part of upstream's
# stable API and shift with internal refactors, so a chimera bump that
# bypasses this check can silently break the build or the runtime.
# Run before changing LLAMACPP_VERSION; the script exits non-zero when
# any of the headers changed.
LLAMA_VERSION ?=

bump-check:
	@$(PYTHON) scripts/manage.py bump_check $(if $(LLAMA_VERSION),--llama-version $(LLAMA_VERSION))

# test-db-migrate: build a v1-schema chimera.db in a temp dir, drive
# `chimera db status` against it (which calls open_and_migrate), and
# assert the v1 -> latest migration both advances the schema and
# preserves pre-existing rows. Pre-empts schema-migration regressions
# across version bumps — every new migration we land has to keep
# upgrading old user DBs cleanly. Requires the chimera binary built.
test-db-migrate: $(BUILD_DIR)/chimera
	@$(PYTHON) scripts/test_db_migrate.py

# combine: bundle the third-party static archives chimera transitively
# links into two grouped archives under build/:
#   build/libchimera_thirdparty.a  (normal-linked by consumer)
#   build/libchimera_ggml.a        (consumer MUST whole-archive this)
# Combined with build/libchimera.a (produced by `make build`), these
# are what a non-CMake external consumer links against. See
# doc/dev/combine_archives.md for the link contract.
combine: $(BUILD_DIR)/libchimera.a
	@$(PYTHON) scripts/combine_archives.py

$(BUILD_DIR)/libchimera_thirdparty.a $(BUILD_DIR)/libchimera_ggml.a: $(BUILD_DIR)/libchimera.a
	@$(PYTHON) scripts/combine_archives.py

$(BUILD_DIR)/libchimera.a:
	$(MAKE) build

# test-external-smoke: build and run tests/external/chimera_smoke,
# which links the three chimera archives the way a non-CMake consumer
# would and exercises the link contract end-to-end (ggml backend
# registration, llama + chimera symbol resolution). Set
# CHIMERA_SMOKE_MODEL=<path/to/.gguf> to additionally run a full
# tokenize + llama_decode inference probe. See
# doc/dev/combine_archives.md section 7 for what this validates.
test-external-smoke: tests/external/build/chimera_smoke tests/external/build/chimera_hpp_smoke
	@ctest --test-dir tests/external/build --output-on-failure

# Convenience: run only the OOP-layer lane (chimera_hpp_smoke). Same
# env-var inputs as the full target -- CHIMERA_SMOKE_MODEL gates the
# inference probe; CHIMERA_SMOKE_WHISPER_MODEL + CHIMERA_SMOKE_WHISPER_INPUT
# gate the Whisper persistence probe.
test-external-oop: tests/external/build/chimera_smoke tests/external/build/chimera_hpp_smoke
	@ctest --test-dir tests/external/build -L OOP --output-on-failure

tests/external/build/chimera_smoke tests/external/build/chimera_hpp_smoke: tests/external/smoke.cpp tests/external/hpp_smoke.cpp tests/external/CMakeLists.txt $(BUILD_DIR)/libchimera.a $(BUILD_DIR)/libchimera_thirdparty.a $(BUILD_DIR)/libchimera_ggml.a
	@cmake -S tests/external -B tests/external/build
	@cmake --build tests/external/build

# bindings: build the nanobind Python extension (the `chimera` module) under
# bindings/ against the three prebuilt archives -- same link contract as the
# external smoke tests, but the output is a Python module rather than an
# executable. Depends on the archives, so a from-scratch `make bindings`
# builds chimera + combines the archives first. Output lands at
# bindings/build/chimera.*.so; add bindings/build to PYTHONPATH to import it.
#
# The build needs nanobind + scikit-build-core for the Python that CMake uses.
# When `uv` is available (the default path), this target provisions them into
# a local venv (BINDINGS_VENV, default bindings/.venv) and points CMake at that
# interpreter -- no system-Python pollution, works out of the box.
#
# The venv is pinned to $(PYTHON)'s interpreter (the project's Python, e.g.
# system python3) so the built module matches the Python you actually use --
# NOT uv's default managed interpreter, which `uv venv` would otherwise pick
# (it ignores a bindings/.python-version because this recipe runs from the repo
# root, not bindings/, so the module would silently target uv's default
# version). Build for a different Python with `make PYTHON=python3.13 bindings`.
#
# Override BINDINGS_PY=/path/to/python to use your own interpreter that already
# has nanobind installed (uv is then not used). With neither uv nor BINDINGS_PY,
# it falls back to $(PYTHON), which must have nanobind installed.
# See bindings/README.md.
#
# (Alternatively, skip this target and `uv pip install ./bindings` -- build
# isolation handles the toolchain. See bindings/pyproject.toml.)
BINDINGS_VENV ?= bindings/.venv

bindings: $(BUILD_DIR)/libchimera.a $(BUILD_DIR)/libchimera_thirdparty.a $(BUILD_DIR)/libchimera_ggml.a
	@if [ -n "$(BINDINGS_PY)" ]; then \
	    py="$(BINDINGS_PY)"; \
	elif command -v uv >/dev/null 2>&1; then \
	    pyexe="$$($(PYTHON) -c 'import sys; print(sys.executable)')"; \
	    echo "bindings: provisioning nanobind + scikit-build-core via uv into $(BINDINGS_VENV) (python: $$pyexe)"; \
	    uv venv "$(BINDINGS_VENV)" --python "$$pyexe" >/dev/null; \
	    uv pip install --python "$(BINDINGS_VENV)/bin/python" -q nanobind scikit-build-core; \
	    py="$(abspath $(BINDINGS_VENV))/bin/python"; \
	else \
	    echo "bindings: uv not found; using $(PYTHON) (must have nanobind installed)"; \
	    py="$$($(PYTHON) -c 'import sys; print(sys.executable)')"; \
	fi; \
	cmake -S bindings -B bindings/build -DCHIMERA_BUILD_ROOT="$(abspath $(BUILD_DIR))" -DPython_EXECUTABLE="$$py"
	@cmake --build bindings/build

# test-bindings: build the module + run the Python smoke test. Uses the
# bindings venv interpreter when present (guaranteed ABI match), else
# $(PYTHON). CHIMERA_SMOKE_MODEL gates the optional inference probe (the
# import/exception probe always runs).
test-bindings: bindings
	@py="$(PYTHON)"; \
	[ -x "$(BINDINGS_VENV)/bin/python" ] && py="$(BINDINGS_VENV)/bin/python"; \
	PYTHONPATH=bindings/build $$py bindings/smoke_test.py

# wheel: build an installable .whl for the chimera Python bindings via
# uv + scikit-build-core. Output lands in bindings/dist/.
#
# IMPORTANT -- this is a LOCAL, host-optimized wheel, not a portable one.
# The extension statically links the three prebuilt archives, so the wheel
# is self-contained at RUNTIME (no repo / no archives needed to import it),
# but it inherits the archives' host-tuned ggml build (GGML_NATIVE, single
# backend). It runs only on this machine (or one bit-identical for the
# relevant ISA / GPU toolkit). Cross-machine redistribution is an explicit
# non-goal of the archive layer -- see docs/dev/combine_archives.md S1.
#
# Uses `uv build --wheel` (builds the wheel straight from the source tree),
# NOT a plain `uv build`. A plain `uv build` makes an sdist first and builds
# the wheel from the *extracted* sdist, where the archives + repo headers
# (which live outside bindings/) are absent, so its CMake configure fails.
# The --wheel path keeps the real repo paths intact; CHIMERA_BUILD_ROOT is
# passed explicitly so it resolves regardless of cwd. Build isolation
# auto-provisions scikit-build-core + nanobind.
#
# Build for a specific interpreter with `make PYTHON=python3.13 wheel`;
# change the output dir with `make WHEEL_OUT=dist wheel`.
WHEEL_OUT ?= bindings/dist

wheel: $(BUILD_DIR)/libchimera.a $(BUILD_DIR)/libchimera_thirdparty.a $(BUILD_DIR)/libchimera_ggml.a
	@command -v uv >/dev/null 2>&1 || { \
	    echo "wheel: uv is required (https://docs.astral.sh/uv/); or use 'uv pip install ./bindings'"; \
	    exit 1; }
	@pyexe="$$($(PYTHON) -c 'import sys; print(sys.executable)')"; \
	echo "wheel: building host-optimized wheel for $$pyexe -> $(WHEEL_OUT)/"; \
	uv build --wheel ./bindings --python "$$pyexe" -o "$(WHEEL_OUT)" \
	    -C cmake.define.CHIMERA_BUILD_ROOT="$(abspath $(BUILD_DIR))"
	@echo "wheel: done (host-optimized; not for cross-machine redistribution)"

# test-golden: HTTP response-shape regression tests against fixed
# models. Spawns `chimera serve` on a free port, hits each route with a
# fixed payload, normalizes the response (redacting volatile fields),
# and diffs against checked-in goldens under tests/golden/. Catches the
# class of bug where a llama.cpp bump changes the JSON shape of a route
# we expose -- something `make test`'s did-it-work smoke can't see.
# Pass UPDATE_GOLDEN=1 to refresh the goldens (use when you legitimately
# changed a response shape and the new shape is correct).
test-golden: $(BUILD_DIR)/chimera
	@$(PYTHON) scripts/test_golden.py
