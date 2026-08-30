#!/usr/bin/env sh
# Same pipeline as rag-chat-qwen3.sh against an explicit DB file. --db must
# be passed to every step -- it is a per-subcommand flag, not a global one.
set -e

DB=vector.db

./build/chimera index create \
	-n docs \
	-e models/bge-small-en-v1.5-q8_0.gguf \
	--db "$DB" \
	--gpu-layers 99

./build/chimera index ingest \
	-n docs \
	-f README.md \
	-f CHANGELOG.md \
	--db "$DB" \
	--gpu-layers 99

./build/chimera search \
	-n docs \
	-q "how do I offload weights to the CPU?" \
	-k 5 \
	--mode hybrid \
	--db "$DB" \
	--gpu-layers 99
