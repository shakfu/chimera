// chimera_whisper.h — internal public API for whisper.cpp integration.
//
// Both `command_whisper` (the CLI subcommand) and `chimera serve` (the
// HTTP `/v1/audio/transcriptions` route) consume this. Anything declared
// here is the contract between those callers and chimera_whisper.cpp;
// the .cpp can keep private helpers in its own anonymous namespace.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct whisper_context;

struct WhisperContextDeleter {
    void operator()(whisper_context * ctx) const;
};
using WhisperContextPtr = std::unique_ptr<whisper_context, WhisperContextDeleter>;

namespace chimera_whisper {

// A single finalized whisper segment. Times are in 10 ms units (whisper.cpp's
// native unit) when produced by transcribe(); convert via `ms = t * 10`.
struct Segment {
    int64_t     t0;   // start, 10ms units
    int64_t     t1;   // end,   10ms units
    std::string text;
};

struct TranscribeRequest {
    // 16 kHz mono float PCM. Use load_wav_*() + resample_linear() to produce.
    std::vector<float> audio_16k_mono;

    // ISO-639-1 code, "auto" for autodetect, or empty to leave whisper's
    // default (which is "en") in place.
    std::string language = "en";

    bool translate    = false;  // translate to English
    bool no_context   = false;  // disable previous-text conditioning
    bool emit_timestamps = true; // params.no_timestamps = !this

    int  threads      = -1;     // -1 = leave whisper's default
    std::string initial_prompt; // optional priming text (params.initial_prompt)

    // Optional callback invoked from whisper's new_segment_callback for each
    // finalized segment as it arrives. Used by the CLI for streaming output;
    // HTTP handlers can omit this and read TranscribeResult::segments after.
    std::function<void(const Segment &)> on_segment;
};

struct TranscribeResult {
    std::string          text;             // concatenated segment text, trimmed
    std::vector<Segment> segments;         // populated regardless of streaming
    std::string          detected_language; // populated when language=="auto"
    double               audio_duration_s = 0.0;
};

// ---- model lifecycle ---------------------------------------------------

// Load a whisper model. Returns an empty pointer on failure; caller decides
// whether to fail() or just refuse to register the audio route.
WhisperContextPtr load_model(const std::string & path);

// ---- WAV I/O ------------------------------------------------------------

struct WavData {
    int                sample_rate = 0;
    int                channels    = 0;
    std::vector<float> samples;   // interleaved-then-downmixed to mono
};

// Parse a RIFF/WAVE byte buffer (PCM int8/16/24/32 or float32). Throws
// ChimeraError(BadInput) on malformed or unsupported input.
WavData load_wav_bytes(const void * data, size_t size);

// File-path convenience wrapper around load_wav_bytes().
WavData load_wav_file(const std::string & path);

// Linear-interpolation resampler. Sufficient for the bandwidth whisper cares
// about; if you need higher quality (or aliasing-free downsampling from
// 48 kHz), front this with a polyphase filter.
std::vector<float> resample_linear(const std::vector<float> & input,
                                   int from_rate, int to_rate);

// ---- transcription ------------------------------------------------------

// Run whisper_full on the context with the given request. Throws
// ChimeraError(Generate) on whisper_full failure. The context is reusable
// across calls (whisper_full owns its own state).
TranscribeResult transcribe(whisper_context * ctx, const TranscribeRequest & req);

// Same 10ms-units -> "HH:MM:SS.mmm" / "MM:SS.mmm" formatter used by the
// `whisper --timestamps` CLI mode. Exposed so HTTP response formatters
// (SRT, VTT, verbose_json) can produce consistent timestamp strings.
std::string format_timestamp_10ms(int64_t t);

// ---- runtime introspection (for `chimera info`) ------------------------

// Runtime whisper.cpp version string (e.g. "v1.8.4"). Differs from
// the compile-time `CHIMERA_WHISPERCPP_VERSION` macro only if the
// upstream tag we pinned doesn't match what whisper.cpp itself reports.
std::string whispercpp_version();

// `ggml_version()` as visible from this TU. In practice this is the
// ggml that whisper.cpp's own build embedded.
std::string whisper_ggml_version();

// Raw `whisper_print_system_info()` output. Format is
// `WHISPER : COREML = 0 | OPENVINO = 0 | ... | NEON = 1 | METAL = 1 | ...`.
// `chimera info` parses it to extract enabled CPU/backend features.
std::string whisper_system_info_raw();

}  // namespace chimera_whisper
