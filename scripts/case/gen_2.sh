#!/usr/bin/env sh
# gen: larger model, same shape as gen_1. Output streams to stdout as it is
# produced; there is no --stream flag to ask for it.

./build/chimera gen \
	-m models/Qwen3-4B-Q8_0.gguf \
	-p "Write a haiku about GPUs." \
	-n 256 \
	--gpu-layers 99
