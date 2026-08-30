#!/usr/bin/env sh

./build/chimera sd \
	--diffusion-model models/z_image_turbo-Q6_K.gguf \
	--vae models/ae.safetensors \
	--llm models/Qwen3-4B-Q8_0.gguf \
	-H 1024 -W 512 \
	-p "a lovely plump cat"