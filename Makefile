.PHONY: deps build build-with-webui build-cuda build-rocm build-sycl build-vulkan rebuild clean reset test smoke install uninstall release-notes bump-check test-db-migrate test-golden combine test-external-smoke

# Python interpreter, picked at make-invocation time by probing the host.
# Order: `python3` (Linux / macOS / Conda / venv), `python` (Windows
# python.org default, Microsoft Store Python, some Linux distros), `py -3`
# (Windows Python Launcher, ships with python.org installers since 3.6).
# Falls back to a literal `python3` if nothing is found so the failure is
# obvious. Requires a POSIX shell — the project already needs bash for
# `scripts/test.sh`, and the Windows CI leg uses git-bash, so this is
# satisfied everywhere we build. Override with `make PYTHON=<cmd>` for
# unusual environments (e.g. `PYTHON="uv run python"`).
PYTHON ?= $(shell \
    if command -v python3 >/dev/null 2>&1; then echo python3;  \
    elif command -v python  >/dev/null 2>&1; then echo python; \
    elif command -v py      >/dev/null 2>&1; then echo py -3;  \
    else echo python3; fi)
BUILD_DIR ?= build
PREFIX ?= /usr/local
DESTDIR ?=

build: deps
	@cmake -S . -B $(BUILD_DIR) -DSD_USE_VENDORED_GGML=OFF -DCMAKE_BUILD_TYPE=Release
	@cmake --build $(BUILD_DIR) --target chimera --config Release -j

# build-with-webui: same as `build`, but flips the experimental
# CHIMERA_WEBUI_EMBED option ON so the chimera binary xxd-bakes
# upstream's prebuilt web UI bundle (GET / + /bundle.{js,css}). Adds
# ~7 MB to the binary; no Node toolchain required. See doc/dev/webui.md
# for the seams. Pass --no-webui at runtime to suppress per-invocation.
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
	@scripts/test.sh

smoke:
	@scripts/test.sh --smoke

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

reset: clean
	@rm -rf thirdparty/llama.cpp thirdparty/whisper.cpp thirdparty/stable-diffusion.cpp thirdparty/linenoise

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
test-external-smoke: tests/external/build/chimera_smoke
	@tests/external/build/chimera_smoke

tests/external/build/chimera_smoke: tests/external/smoke.cpp tests/external/CMakeLists.txt $(BUILD_DIR)/libchimera.a $(BUILD_DIR)/libchimera_thirdparty.a $(BUILD_DIR)/libchimera_ggml.a
	@cmake -S tests/external -B tests/external/build
	@cmake --build tests/external/build

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
