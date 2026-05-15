.PHONY: deps build rebuild clean reset test smoke install uninstall

PYTHON ?= python3
BUILD_DIR ?= build
PREFIX ?= /usr/local
DESTDIR ?=

build: deps
	@cmake -S . -B $(BUILD_DIR) -DSD_USE_VENDORED_GGML=OFF
	@cmake --build $(BUILD_DIR) --target chimera -j

rebuild:
	@cmake --build $(BUILD_DIR) --target chimera -j

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
	@rm -rf thirdparty/llama.cpp thirdparty/whisper.cpp thirdparty/stable-diffusion.cpp
