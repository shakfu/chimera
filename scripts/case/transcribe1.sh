#!/usr/bin/env sh
# The subcommand is `whisper`, not `transcribe`, and the audio flag is
# -i/--input, not -f. jfk.wav ships with the vendored whisper.cpp tree
# (`make deps` fetches it).

./build/chimera whisper \
	-m models/ggml-base.en.bin \
	-i build/whisper.cpp/samples/jfk.wav \
	--timestamps
