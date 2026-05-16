.PHONY: deps build rebuild clean reset test smoke install uninstall release-notes bump-check test-db-migrate test-golden

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
