#!/usr/bin/env sh
# embed: emit one vector per input. `embed` has no --similarity/--threshold;
# similarity search over a corpus is `index` + `search` (see
# rag-chat-qwen3.sh). --embd-separator splits the input into several texts.

./build/chimera embed \
	-m models/bge-small-en-v1.5-q8_0.gguf \
	-p "death and dying;grief and mourning;a cheerful summer picnic" \
	--embd-separator ";" \
	--embd-output-format array \
	--pooling mean \
	--gpu-layers 99
