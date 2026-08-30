#!/usr/bin/env sh
# There is no `chimera rag` subcommand. Retrieval over local files is
# `index create` -> `index ingest` -> `search`; the collection records the
# embedding model, so `search` does not take -e. This variant uses the
# default DB ($CHIMERA_DB or the platform default).
#
# For the served, chat-shaped equivalent see:
#   ./build/chimera serve -m models/Qwen3-4B-Q8_0.gguf \
#       --enable-rag models/bge-small-en-v1.5-q8_0.gguf
set -e

./build/chimera index create \
	-n docs \
	-e models/bge-small-en-v1.5-q8_0.gguf \
	--gpu-layers 99

./build/chimera index ingest \
	-n docs \
	-f README.md \
	-f CHANGELOG.md \
	--gpu-layers 99

./build/chimera search \
	-n docs \
	-q "how do I offload weights to the CPU?" \
	-k 5 \
	--mode hybrid \
	--gpu-layers 99
