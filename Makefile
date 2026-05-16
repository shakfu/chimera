.PHONY: deps build rebuild clean reset test smoke install uninstall

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
