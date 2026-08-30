#!/usr/bin/env sh
# gen: one-shot generation, greedy defaults.
# --gpu-layers is required to exercise a GPU build: chimera defaults to 0
# (CPU), unlike sd, which picks up the GPU on its own.

./build/chimera gen \
	-m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
	-p "Explain quantum entanglement in one paragraph." \
	-n 256 \
	--gpu-layers 99
