#!/usr/bin/env sh

./build/chimera gen \
	-m models/gemma-4-E4B-it-Q5_K_M.gguf \
	-p "List three interesting facts about octopuses." \
	-n 512 \
	--temperature 0.7 \
	--stream \
	--stats
