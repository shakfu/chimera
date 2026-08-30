#!/usr/bin/env sh

./build/chimera gen \
	-m models/Qwen3-4B-Q8_0.gguf \
	-p "Write a haiku about GPUs." \
	-n 256 \
	--stream \
	--stats
