#!/usr/bin/env sh

./build/chimera gen \
	-m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
	-p "Explain quantum entanglement in one paragraph." \
	-n 256 \
	--stats
