#!/usr/bin/env sh
# gen: sampler knobs. The temperature flag is --temp (sd-cli spelling),
# not --temperature.

./build/chimera gen \
	-m models/gemma-4-E4B-it-Q5_K_M.gguf \
	-p "List three interesting facts about octopuses." \
	-n 512 \
	--temp 0.7 \
	--top-p 0.95 \
	--gpu-layers 99
