.PHONY: deps build clean reset test smoke

PYTHON ?= python3
BUILD_DIR ?= build

build: deps
	@cmake -S . -B $(BUILD_DIR) -DSD_USE_VENDORED_GGML=OFF
	@cmake --build $(BUILD_DIR) --target chimera -j

deps:
	@SD_USE_VENDORED_GGML=0 $(PYTHON) scripts/manage.py build --all --deps-only --sd-shared-ggml

test:
	@scripts/test.sh

smoke:
	@scripts/test.sh --smoke

clean:
	@rm -rf $(BUILD_DIR)

reset: clean
	@rm -rf thirdparty/llama.cpp thirdparty/whisper.cpp thirdparty/stable-diffusion.cpp
