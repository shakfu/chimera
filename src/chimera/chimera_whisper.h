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
#include <limits>
#include <memory>
#include <string>
#include <vector>

struct whisper_context;
// Forward decl for the grammar element. whisper.h defines this as a
// `typedef struct whisper_grammar_element { ... }` so the struct tag
// is sufficient; consumers that actually look inside the struct
// (chimera_whisper.cpp + chimera_whisper_grammar.cpp) include whisper.h.
struct whisper_grammar_element;

struct WhisperContextDeleter {
    void operator()(whisper_context * ctx) const;
};
using WhisperContextPtr = std::unique_ptr<whisper_context, WhisperContextDeleter>;

namespace chimera_whisper {

// One word inside a segment, populated when `TranscribeRequest::word_timestamps`
// is set. `text` is the user-visible word (no leading space). Tokens that
// don't begin a word (subword continuations) are merged into the previous
// word; special tokens (`<|...|>`) are skipped.
struct Word {
    int64_t     t0;   // start, 10ms units
    int64_t     t1;   // end,   10ms units
    std::string text;
};

// A single finalized whisper segment. Times are in 10 ms units (whisper.cpp's
// native unit) when produced by transcribe(); convert via `ms = t * 10`.
// `words` is populated only when `TranscribeRequest::word_timestamps` is true.
struct Segment {
    int64_t           t0;   // start, 10ms units
    int64_t           t1;   // end,   10ms units
    std::string       text;
    std::vector<Word> words;
    // Optional speaker label, populated by `whisper --diarize` only.
    // Empty for non-diarize transcripts. Format mirrors whisper-cli:
    // "(speaker 0)" / "(speaker 1)" / "(speaker ?)" — including the
    // parens — so it can be prepended verbatim to text.
    std::string speaker;
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
    bool  carry_initial_prompt = false; // params.carry_initial_prompt

    // Decoding strategy: when beam_size > 0, switch to BEAM_SEARCH and
    // use the given beam width; otherwise stay on greedy with best_of.
    // Negative / zero sentinels leave whisper.cpp's defaults in place.
    int   beam_size   = -1;
    int   best_of     = -1;
    float temperature = -1.0f;
    bool  no_fallback = false;  // sets temperature_inc to a negative value

    // When true, whisper produces per-token timing data which transcribe()
    // groups into Word entries on each Segment. Required for OpenAI's
    // `timestamp_granularities=["word"]` response. Adds compute; off by
    // default. Maps to whisper's `params.token_timestamps`.
    bool word_timestamps = false;

    // Region-of-audio selection. Both in milliseconds; 0 leaves whisper's
    // default (process the entire input) in place. Mirrors whisper-cli's
    // -ot / -d.
    int offset_ms   = 0;
    int duration_ms = 0;

    // Voice Activity Detection (whisper.cpp >= v1.7.5). When `vad` is
    // true, whisper loads `vad_model_path` and runs the VAD preprocessor
    // before transcription; otherwise the fields are ignored. The
    // numeric knobs map 1:1 to `whisper_vad_params`; negative sentinels
    // (or empty string for the path) leave the upstream default in
    // place.
    bool        vad = false;
    std::string vad_model_path;
    float       vad_threshold              = -1.0f;
    int         vad_min_speech_duration_ms = -1;
    int         vad_min_silence_duration_ms = -1;
    float       vad_max_speech_duration_s  = -1.0f;
    int         vad_speech_pad_ms          = -1;
    float       vad_samples_overlap        = -1.0f;

    // Segment shaping. Mirrors whisper_full_params.{max_len, max_tokens,
    // split_on_word}. Zeros leave the whisper.cpp defaults.
    int  max_len       = 0;
    int  max_tokens    = 0;
    bool split_on_word = false;

    // Decoder fallback thresholds. NaN sentinels (default) leave the
    // upstream values in place — we cannot use a negative sentinel
    // here because whisper's default `logprob_thold` is itself
    // negative.
    float temperature_inc = std::numeric_limits<float>::quiet_NaN();
    float entropy_thold   = std::numeric_limits<float>::quiet_NaN();
    float logprob_thold   = std::numeric_limits<float>::quiet_NaN();
    float no_speech_thold = std::numeric_limits<float>::quiet_NaN();

    // audio_ctx=0 keeps the model's default audio context size.
    int audio_ctx = 0;

    // Tinydiarize (speaker-turn detection). Requires a tdrz-trained
    // model; setting this on a non-tdrz model is harmless (the flag is
    // ignored upstream).
    bool tinydiarize = false;

    // Token suppression. Empty regex / false flag leave the defaults.
    // The regex is matched against token strings by whisper.cpp.
    std::string suppress_regex;
    bool        suppress_nst = false;

    // Number of parallel processors. >1 routes through whisper_full_parallel
    // which splits the input into N chunks; upstream warns this can degrade
    // accuracy at chunk boundaries, so default 1 keeps the serial path.
    int processors = 1;

    // Optional GBNF grammar constraint. `grammar_rules` is the C-pointer
    // view that transcribe() borrows into whisper_full_params, so its
    // backing parse_state (held by the caller) must outlive the call.
    // `grammar_rule_index` is the start-rule index inside that vector
    // (resolved by name in command_whisper before this struct is built);
    // -1 disables grammar regardless of the other fields.
    const std::vector<const whisper_grammar_element *> * grammar_rules = nullptr;
    int   grammar_rule_index = -1;
    float grammar_penalty    = 100.0f;

    // Exit-after-detect language identification. When true, whisper.cpp
    // runs `whisper_lang_auto_detect_with_state` and returns from
    // `whisper_full` before any decode steps — the resulting TranscribeResult
    // has the language in `.detected_language` and zero segments. Useful
    // as a probe before committing to a full transcription run.
    bool detect_language = false;

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

// Knobs that map to whisper_context_params (applied at context init,
// not per-call). All have whisper.cpp defaults; the fields are set to
// match those defaults so a default-constructed LoadParams reproduces
// the previous `load_model(path)` behavior.
struct LoadParams {
    std::string model;
    bool use_gpu    = true;   // whisper_context_default_params() default
    bool flash_attn = false;  // upstream default
    int  gpu_device = 0;      // CUDA device index
};

// Load a whisper model. Returns an empty pointer on failure; caller decides
// whether to fail() or just refuse to register the audio route.
WhisperContextPtr load_model(const std::string & path);
WhisperContextPtr load_model(const LoadParams & params);

// ---- WAV I/O ------------------------------------------------------------

struct WavData {
    int                sample_rate = 0;
    int                channels    = 0;
    std::vector<float> samples;   // interleaved-then-downmixed to mono
    // Per-channel float view in source order, populated only when
    // channels > 1 (mono inputs leave this empty — `samples` already is
    // the lone channel). Used by `whisper --diarize` for stereo speaker
    // estimation; other consumers can ignore.
    std::vector<std::vector<float>> per_channel;
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
