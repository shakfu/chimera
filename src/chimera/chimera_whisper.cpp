// chimera_whisper.cpp — whisper.cpp wrapper consumed by both the `whisper`
// CLI subcommand and the `serve` POST /v1/audio/transcriptions handler.
//
// The public API lives in chimera_whisper.h. Anything kept private to this
// translation unit (the streaming-segment callback shim, the WAV chunk
// scanner) stays in the anonymous namespace below.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "chimera.h"
#include "chimera_whisper.h"
#include "chimera_whisper_grammar.h"
#include "whisper.h"

void WhisperContextDeleter::operator()(whisper_context * ctx) const {
    if (ctx != nullptr) {
        whisper_free(ctx);
    }
}

// Pin-asserts: upstream signatures we depend on. If whisper.cpp
// renames any of these or changes the prototype, this TU fails to
// compile with a pointer at the broken contract -- before the failure
// cascades into the rest of chimera_whisper.cpp's call sites. See
// doc/dev/maintenance.md and src/chimera/chimera_pin_check.cpp for the
// cross-cutting llama.cpp version.
namespace {
[[maybe_unused]] void chimera_whisper_pin_check() {
    [[maybe_unused]] whisper_token (*p_token_beg)(struct whisper_context *) =
        &whisper_token_beg;
    [[maybe_unused]] int (*p_full_n_tokens)(struct whisper_context *, int) =
        &whisper_full_n_tokens;
    [[maybe_unused]] const char * (*p_full_get_token_text)(struct whisper_context *, int, int) =
        &whisper_full_get_token_text;
    [[maybe_unused]] whisper_token_data (*p_full_get_token_data)(struct whisper_context *, int, int) =
        &whisper_full_get_token_data;

    // ---- persistent-handle dependencies (chimera::Whisper) ----------
    //
    // The OOP wrapper holds a whisper_context across many transcribe()
    // calls. Its correctness rests on the upstream contract that
    // whisper_full is safe to call repeatedly on the same context (the
    // "context is reusable across calls" note in chimera_whisper.h).
    // This is a behavioral contract; we can't static-assert it, but
    // we can pin the signatures so a rename of any load/run/free
    // symbol fails to compile here rather than inside chimera.hpp's
    // template instantiation.
    [[maybe_unused]] struct whisper_context * (*p_whisper_init_from_file_with_params)(
        const char *, struct whisper_context_params) = &whisper_init_from_file_with_params;
    [[maybe_unused]] void (*p_whisper_free)(struct whisper_context *) = &whisper_free;
    [[maybe_unused]] int (*p_whisper_full)(
        struct whisper_context *, struct whisper_full_params,
        const float *, int) = &whisper_full;
}
}  // namespace

static void chimera_silent_whisper_log(enum ggml_log_level, const char *, void *) {}

void chimera_silence_whisper_log() {
    whisper_log_set(chimera_silent_whisper_log, nullptr);
}

void chimera_restore_whisper_log() {
    whisper_log_set(nullptr, nullptr);
}

namespace {

template <typename T>
T read_le(std::istream & in) {
    T value{};
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!in) {
        fail("failed to read input file");
    }
    return value;
}

// Drains a RIFF/WAVE stream into a WavData. Shared by load_wav_file (file
// path) and load_wav_bytes (HTTP upload). Anything format-related that
// callers might want to extend (e.g. RF64, BWF) goes here.
chimera_whisper::WavData parse_wav_stream(std::istream & in, const std::string & origin) {
    using chimera_whisper::WavData;

    char riff[4];
    char wave[4];
    in.read(riff, 4);
    (void) read_le<uint32_t>(in);
    in.read(wave, 4);
    if (std::string_view(riff, 4) != "RIFF" || std::string_view(wave, 4) != "WAVE") {
        fail(ExitCode::BadInput, "unsupported WAV container: " + origin);
    }

    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<char> pcm_bytes;

    while (in) {
        char chunk_id[4];
        in.read(chunk_id, 4);
        if (!in) {
            break;
        }
        const uint32_t chunk_size = read_le<uint32_t>(in);
        const std::string_view id(chunk_id, 4);

        if (id == "fmt ") {
            audio_format = read_le<uint16_t>(in);
            channels = read_le<uint16_t>(in);
            sample_rate = read_le<uint32_t>(in);
            (void) read_le<uint32_t>(in);
            (void) read_le<uint16_t>(in);
            bits_per_sample = read_le<uint16_t>(in);
            if (chunk_size > 16) {
                in.seekg(static_cast<std::streamoff>(chunk_size - 16), std::ios::cur);
            }
        } else if (id == "data") {
            pcm_bytes.resize(chunk_size);
            in.read(pcm_bytes.data(), static_cast<std::streamsize>(chunk_size));
        } else {
            in.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
        }

        if (chunk_size % 2 != 0) {
            in.seekg(1, std::ios::cur);
        }
    }

    if (audio_format == 0 || channels == 0 || sample_rate == 0 || bits_per_sample == 0 || pcm_bytes.empty()) {
        fail(ExitCode::BadInput, "incomplete WAV file: " + origin);
    }

    const size_t bytes_per_sample = bits_per_sample / 8;
    if (bytes_per_sample == 0 || pcm_bytes.size() % bytes_per_sample != 0) {
        fail(ExitCode::BadInput, "invalid WAV sample size: " + origin);
    }

    std::vector<float> interleaved;
    interleaved.reserve(pcm_bytes.size() / bytes_per_sample);

    if (audio_format == 1) {
        if (bits_per_sample == 8) {
            for (unsigned char ch : pcm_bytes) {
                interleaved.push_back((static_cast<int>(ch) - 128) / 128.0f);
            }
        } else if (bits_per_sample == 16) {
            for (size_t i = 0; i < pcm_bytes.size(); i += 2) {
                int16_t sample = static_cast<int16_t>(
                    static_cast<uint8_t>(pcm_bytes[i]) |
                    (static_cast<uint8_t>(pcm_bytes[i + 1]) << 8));
                interleaved.push_back(sample / 32768.0f);
            }
        } else if (bits_per_sample == 24) {
            for (size_t i = 0; i < pcm_bytes.size(); i += 3) {
                int32_t sample =
                    static_cast<int32_t>(static_cast<uint8_t>(pcm_bytes[i])) |
                    (static_cast<int32_t>(static_cast<uint8_t>(pcm_bytes[i + 1])) << 8) |
                    (static_cast<int32_t>(static_cast<int8_t>(pcm_bytes[i + 2])) << 16);
                interleaved.push_back(sample / 8388608.0f);
            }
        } else if (bits_per_sample == 32) {
            for (size_t i = 0; i < pcm_bytes.size(); i += 4) {
                int32_t sample =
                    static_cast<int32_t>(static_cast<uint8_t>(pcm_bytes[i])) |
                    (static_cast<int32_t>(static_cast<uint8_t>(pcm_bytes[i + 1])) << 8) |
                    (static_cast<int32_t>(static_cast<uint8_t>(pcm_bytes[i + 2])) << 16) |
                    (static_cast<int32_t>(static_cast<uint8_t>(pcm_bytes[i + 3])) << 24);
                interleaved.push_back(sample / 2147483648.0f);
            }
        } else {
            fail(ExitCode::BadInput,
                 "unsupported PCM bit depth in WAV: " + std::to_string(bits_per_sample));
        }
    } else if (audio_format == 3 && bits_per_sample == 32) {
        for (size_t i = 0; i < pcm_bytes.size(); i += 4) {
            float sample = 0.0f;
            std::memcpy(&sample, pcm_bytes.data() + i, sizeof(float));
            interleaved.push_back(sample);
        }
    } else {
        fail(ExitCode::BadInput, "unsupported WAV encoding in " + origin);
    }

    std::vector<float> mono;
    std::vector<std::vector<float>> per_channel;
    if (channels == 1) {
        mono = std::move(interleaved);
    } else {
        const size_t frames = interleaved.size() / channels;
        // Split into per-channel float vectors first so the downmix
        // and the diarize path read from the same source data without
        // a second pass over `interleaved`.
        per_channel.assign(channels, std::vector<float>(frames));
        mono.resize(frames, 0.0f);
        for (size_t frame = 0; frame < frames; ++frame) {
            double sum = 0.0;
            for (uint16_t ch = 0; ch < channels; ++ch) {
                const float s = interleaved[frame * channels + ch];
                per_channel[ch][frame] = s;
                sum += s;
            }
            mono[frame] = static_cast<float>(sum / channels);
        }
    }

    WavData out;
    out.sample_rate = static_cast<int>(sample_rate);
    out.channels    = static_cast<int>(channels);
    out.samples     = std::move(mono);
    out.per_channel = std::move(per_channel);
    return out;
}

// Bridges req.on_segment into whisper's C-style new_segment_callback. The
// state object's segments are not yet visible via the ctx-based accessors
// at this point, so we read via *_from_state.
struct StreamingCallbackCtx {
    const std::function<void(const chimera_whisper::Segment &)> * cb;
};

void streaming_segment_cb(struct whisper_context * /*ctx*/, struct whisper_state * state,
                          int n_new, void * user_data) {
    auto * cb_ctx = static_cast<StreamingCallbackCtx *>(user_data);
    if (!cb_ctx || !cb_ctx->cb || !(*cb_ctx->cb)) return;

    const int n_total = whisper_full_n_segments_from_state(state);
    const int n_start = std::max(0, n_total - n_new);
    for (int i = n_start; i < n_total; ++i) {
        chimera_whisper::Segment s;
        s.t0   = whisper_full_get_segment_t0_from_state(state, i);
        s.t1   = whisper_full_get_segment_t1_from_state(state, i);
        s.text = trim(whisper_full_get_segment_text_from_state(state, i));
        (*cb_ctx->cb)(s);
    }
}

}  // namespace

namespace chimera_whisper {

WhisperContextPtr load_model(const std::string & path) {
    LoadParams p;
    p.model = path;
    return load_model(p);
}

WhisperContextPtr load_model(const LoadParams & params) {
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;
    cparams.gpu_device = params.gpu_device;
    WhisperContextPtr ctx(whisper_init_from_file_with_params(params.model.c_str(), cparams));
    return ctx;
}

WavData load_wav_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail(ExitCode::BadInput, "failed to open WAV file: " + path);
    }
    return parse_wav_stream(in, path);
}

WavData load_wav_bytes(const void * data, size_t size) {
    if (data == nullptr || size < 12) {
        fail(ExitCode::BadInput, "WAV buffer is empty or too short");
    }
    // istringstream over a string-copy. ~10 MB / second on the cheap; if
    // this becomes a bottleneck we can swap in a membuf wrapper. For now
    // simplicity > saving one copy.
    std::string blob(static_cast<const char *>(data), size);
    std::istringstream in(std::move(blob), std::ios::binary);
    return parse_wav_stream(in, "<upload>");
}

std::vector<float> resample_linear(const std::vector<float> & input, int from_rate, int to_rate) {
    if (from_rate == to_rate || input.empty()) {
        return input;
    }
    const double ratio = static_cast<double>(from_rate) / to_rate;
    const size_t out_size = static_cast<size_t>(std::max(1.0, std::floor(input.size() / ratio)));

    std::vector<float> output(out_size, 0.0f);
    for (size_t i = 0; i < out_size; ++i) {
        const double src_index = i * ratio;
        const size_t idx0 = static_cast<size_t>(src_index);
        const size_t idx1 = std::min(idx0 + 1, input.size() - 1);
        const double frac = src_index - idx0;
        output[i] = static_cast<float>((1.0 - frac) * input[idx0] + frac * input[idx1]);
    }
    return output;
}

std::string format_timestamp_10ms(int64_t t) {
    int64_t ms = t * 10;
    const int64_t hours = ms / 3600000;
    ms %= 3600000;
    const int64_t minutes = ms / 60000;
    ms %= 60000;
    const int64_t seconds = ms / 1000;
    ms %= 1000;

    char buf[32];
    if (hours > 0) {
        std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld.%03lld",
                      static_cast<long long>(hours),   static_cast<long long>(minutes),
                      static_cast<long long>(seconds), static_cast<long long>(ms));
    } else {
        std::snprintf(buf, sizeof(buf), "%02lld:%02lld.%03lld",
                      static_cast<long long>(minutes), static_cast<long long>(seconds),
                      static_cast<long long>(ms));
    }
    return buf;
}

TranscribeResult transcribe(whisper_context * ctx, const TranscribeRequest & req) {
    if (ctx == nullptr) {
        fail(ExitCode::Runtime, "whisper context is null");
    }

    const whisper_sampling_strategy strat = req.beam_size > 0
        ? WHISPER_SAMPLING_BEAM_SEARCH
        : WHISPER_SAMPLING_GREEDY;
    whisper_full_params params = whisper_full_default_params(strat);
    if (strat == WHISPER_SAMPLING_BEAM_SEARCH) {
        params.beam_search.beam_size = req.beam_size;
    }
    if (req.best_of > 0) {
        params.greedy.best_of = req.best_of;
    }
    if (req.temperature >= 0.0f) {
        params.temperature = req.temperature;
    }
    if (req.no_fallback) {
        // whisper.cpp convention: negative temperature_inc disables the
        // temperature-fallback ladder in whisper_full_with_state.
        params.temperature_inc = -1.0f;
    }
    params.carry_initial_prompt = req.carry_initial_prompt;
    // -1 is "auto" upstream but whisper.cpp's default expects a positive
    // value (otherwise it constructs a std::vector(n_threads) and throws).
    if (req.threads > 0) {
        params.n_threads = req.threads;
    }
    params.translate        = req.translate;
    params.no_context       = req.no_context;
    params.no_timestamps    = !req.emit_timestamps;
    // OpenAI's `timestamp_granularities=["word"]` maps here. Per-token
    // timing is heavier than the segment-level default; only turn it on
    // when the caller actually wants it.
    params.token_timestamps = req.word_timestamps;
    params.print_progress   = false;
    params.print_realtime   = false;
    params.print_timestamps = false;

    // whisper.h:532 -- "for auto-detection, set to nullptr, "" or "auto"".
    // `detect_language = true` is a *different* mode that only runs the
    // language-id pass and returns early without transcribing; we never
    // want that here.
    params.detect_language = false;
    if (req.language == "auto" || req.language.empty()) {
        params.language = "auto";
    } else {
        params.language = req.language.c_str();
    }
    if (!req.initial_prompt.empty()) {
        params.initial_prompt = req.initial_prompt.c_str();
    }

    // Region-of-audio selection. whisper_full's default is 0/0 (process
    // the entire input); writing 0/0 explicitly is harmless and keeps
    // the wiring uniform.
    params.offset_ms   = req.offset_ms;
    params.duration_ms = req.duration_ms;

    // Voice Activity Detection. The model file is required when `vad`
    // is true; without it whisper.cpp will refuse to initialize the VAD
    // context. Tuning knobs default to the upstream values via
    // `whisper_vad_default_params()`; we override only the fields the
    // caller provided (non-sentinel).
    params.vad = req.vad;
    if (req.vad) {
        if (req.vad_model_path.empty()) {
            fail(ExitCode::BadInput,
                 "--vad requires --vad-model (path to a whisper VAD model file)");
        }
        params.vad_model_path = req.vad_model_path.c_str();
        params.vad_params     = whisper_vad_default_params();
        if (req.vad_threshold >= 0.0f)
            params.vad_params.threshold = req.vad_threshold;
        if (req.vad_min_speech_duration_ms >= 0)
            params.vad_params.min_speech_duration_ms = req.vad_min_speech_duration_ms;
        if (req.vad_min_silence_duration_ms >= 0)
            params.vad_params.min_silence_duration_ms = req.vad_min_silence_duration_ms;
        if (req.vad_max_speech_duration_s >= 0.0f)
            params.vad_params.max_speech_duration_s = req.vad_max_speech_duration_s;
        if (req.vad_speech_pad_ms >= 0)
            params.vad_params.speech_pad_ms = req.vad_speech_pad_ms;
        if (req.vad_samples_overlap >= 0.0f)
            params.vad_params.samples_overlap = req.vad_samples_overlap;
    }

    // Segment shaping. Zero leaves the default; `split_on_word` is a
    // plain bool with no sentinel (it only takes effect when max_len>0
    // upstream, so an unintentional `true` here is benign).
    if (req.max_len > 0)    params.max_len    = req.max_len;
    if (req.max_tokens > 0) params.max_tokens = req.max_tokens;
    params.split_on_word = req.split_on_word;

    // Decoder fallback thresholds. NaN means "leave the default".
    // --no-fallback (applied above) sets temperature_inc<0 — if the
    // caller also passed --temperature-inc, the explicit value wins
    // here, but the --no-fallback override below restores the disable.
    if (!std::isnan(req.temperature_inc)) params.temperature_inc = req.temperature_inc;
    if (!std::isnan(req.entropy_thold))   params.entropy_thold   = req.entropy_thold;
    if (!std::isnan(req.logprob_thold))   params.logprob_thold   = req.logprob_thold;
    if (!std::isnan(req.no_speech_thold)) params.no_speech_thold = req.no_speech_thold;
    if (req.no_fallback) {
        // Reassert here so --no-fallback always wins over an explicit
        // --temperature-inc, mirroring whisper-cli's flag precedence.
        params.temperature_inc = -1.0f;
    }

    if (req.audio_ctx > 0)  params.audio_ctx  = req.audio_ctx;
    if (req.tinydiarize)    params.tdrz_enable = true;

    if (!req.suppress_regex.empty()) {
        // suppress_regex is a borrowed const char* — request outlives this call.
        params.suppress_regex = req.suppress_regex.c_str();
    }
    if (req.suppress_nst) params.suppress_nst = true;

    // Exit-after-detect. whisper.cpp's whisper_full short-circuits with
    // a return code of 0 once `state->lang_id` is set, before any decode
    // pass — so segments come back empty and `whisper_full_lang_id`
    // reads the detected id. Implies `language="auto"` for the runtime
    // check inside whisper.cpp, but we don't have to touch req.language
    // because that check is `||`-disjunctive with detect_language.
    if (req.detect_language) params.detect_language = true;

    // Grammar constraint. The vector pointed to by req.grammar_rules
    // (a `std::vector<const whisper_grammar_element *>`) is owned by
    // the caller and must outlive whisper_full / whisper_full_parallel.
    // grammar_rule_index < 0 disables the constraint regardless of the
    // other fields; the index is also bounds-checked here against the
    // rule count so a stale request doesn't index out of range.
    if (req.grammar_rules != nullptr && req.grammar_rule_index >= 0 &&
        !req.grammar_rules->empty()) {
        if (static_cast<size_t>(req.grammar_rule_index) >= req.grammar_rules->size()) {
            fail(ExitCode::BadInput,
                 "grammar_rule_index out of range for the parsed grammar");
        }
        // whisper.h declares grammar_rules as `const whisper_grammar_element **`
        // (mutable outer pointer); our request stores a vector of
        // `const whisper_grammar_element *`, so .data() yields
        // `const whisper_grammar_element * const *`. Strip the outer const —
        // whisper.cpp only reads through the pointer, never mutates.
        params.grammar_rules   = const_cast<const whisper_grammar_element **>(req.grammar_rules->data());
        params.n_grammar_rules = req.grammar_rules->size();
        params.i_start_rule    = static_cast<size_t>(req.grammar_rule_index);
        params.grammar_penalty = req.grammar_penalty;
    }

    StreamingCallbackCtx cb_ctx{&req.on_segment};
    if (req.on_segment) {
        params.new_segment_callback = streaming_segment_cb;
        params.new_segment_callback_user_data = &cb_ctx;
    }

    // When processors > 1, whisper.cpp splits the input across N
    // independent decoder states. Accuracy degrades slightly at chunk
    // boundaries (per upstream docs), so we only use the parallel path
    // when the caller explicitly asks for it.
    const int rc = req.processors > 1
        ? whisper_full_parallel(ctx, params,
                                req.audio_16k_mono.data(),
                                static_cast<int>(req.audio_16k_mono.size()),
                                req.processors)
        : whisper_full(ctx, params,
                       req.audio_16k_mono.data(),
                       static_cast<int>(req.audio_16k_mono.size()));
    if (rc != 0) {
        fail(ExitCode::Generate, "whisper_full failed");
    }

    TranscribeResult result;
    result.audio_duration_s =
        static_cast<double>(req.audio_16k_mono.size()) / static_cast<double>(WHISPER_SAMPLE_RATE);

    const int n_seg = whisper_full_n_segments(ctx);
    result.segments.reserve(static_cast<size_t>(n_seg));
    for (int i = 0; i < n_seg; ++i) {
        Segment s;
        s.t0   = whisper_full_get_segment_t0(ctx, i);
        s.t1   = whisper_full_get_segment_t1(ctx, i);
        s.text = trim(whisper_full_get_segment_text(ctx, i));

        // Per-word timestamps: walk the segment's tokens, skip special
        // tokens (text starts with `<|`...), and group subword pieces
        // into words. Whisper's tokenizer prefixes the first piece of
        // each word with a leading space; later pieces of the same word
        // do not. We use that signal to detect word boundaries. CJK
        // languages without inter-word spacing will produce
        // one-word-per-token, which is still reasonable.
        if (req.word_timestamps) {
            const int n_tok = whisper_full_n_tokens(ctx, i);
            Word cur{};
            bool have_cur = false;
            auto flush = [&]() {
                if (!have_cur) return;
                cur.text = trim(cur.text);
                if (!cur.text.empty()) s.words.push_back(cur);
                have_cur = false;
                cur = Word{};
            };
            // Any token id >= whisper_token_beg is a timestamp marker
            // (rendered as `[_BEG_]`, `[_TT_550]`, ...); skip them. The
            // text-based guards below catch the remaining specials
            // (`<|en|>`, `[_EOT_]`, etc.) which sit below that threshold.
            const whisper_token tok_beg = whisper_token_beg(ctx);
            for (int j = 0; j < n_tok; ++j) {
                const whisper_token_data td =
                    whisper_full_get_token_data(ctx, i, j);
                if (td.id >= tok_beg) continue;

                const char * raw = whisper_full_get_token_text(ctx, i, j);
                if (raw == nullptr) continue;
                const std::string text = raw;
                if (text.empty()) continue;
                if (text.size() >= 2 && text[0] == '<' && text[1] == '|') continue;
                if (text.size() >= 2 && text[0] == '[' && text[1] == '_') continue;
                // A leading-space token (or the first non-special token
                // of the segment) begins a new word.
                const bool starts_word = !have_cur || (!text.empty() && text[0] == ' ');
                if (starts_word) {
                    flush();
                    cur.t0   = td.t0;
                    cur.t1   = td.t1;
                    cur.text = text;
                    have_cur = true;
                } else {
                    cur.t1   = td.t1;
                    cur.text += text;
                }
            }
            flush();
        }

        if (!result.text.empty() && !s.text.empty()) {
            result.text += ' ';
        }
        result.text += s.text;
        result.segments.push_back(std::move(s));
    }
    if (req.language == "auto" || req.detect_language) {
        const int lang_id = whisper_full_lang_id(ctx);
        if (lang_id >= 0) {
            if (const char * code = whisper_lang_str(lang_id)) {
                result.detected_language = code;
            }
        }
    }
    return result;
}

}  // namespace chimera_whisper

// ---- runtime introspection ---------------------------------------------

namespace chimera_whisper {

std::string whispercpp_version() {
    if (const char * v = whisper_version()) return v;
    return "unknown";
}

std::string whisper_ggml_version() {
    if (const char * v = ggml_version()) return v;
    return "unknown";
}

std::string whisper_system_info_raw() {
    if (const char * s = whisper_print_system_info()) return s;
    return "";
}

}  // namespace chimera_whisper

// ---- output-format writers ---------------------------------------------
//
// Each writer takes a fully-realized TranscribeResult and a destination
// stream. Timestamp formatting matches whisper-cli:
//   .srt -> "HH:MM:SS,mmm" (comma decimal)
//   .vtt -> "HH:MM:SS.mmm" (period decimal)
//   .lrc -> "[MM:SS.cc]"   (centiseconds)
//   .csv -> integer milliseconds, text quoted with doubled inner quotes
//   .json -> minimal {language,text,segments[]} (or per-word data for ojf)

namespace {

std::string ts_srt(int64_t t10ms) {
    int64_t ms = t10ms * 10;
    const int64_t h = ms / 3600000; ms %= 3600000;
    const int64_t m = ms / 60000;   ms %= 60000;
    const int64_t s = ms / 1000;    ms %= 1000;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld,%03lld",
                  (long long)h, (long long)m, (long long)s, (long long)ms);
    return buf;
}

std::string ts_vtt(int64_t t10ms) {
    std::string s = ts_srt(t10ms);
    for (char & c : s) if (c == ',') { c = '.'; break; }
    return s;
}

std::string ts_lrc(int64_t t10ms) {
    // whisper.cpp timestamps are in units of 10ms = 1 centisecond.
    const int64_t cs = t10ms;
    const int64_t m  = (cs / 100) / 60;
    const int64_t s  = (cs / 100) % 60;
    const int64_t cc = cs % 100;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld.%02lld",
                  (long long)m, (long long)s, (long long)cc);
    return buf;
}

std::string json_escape(const std::string & in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

std::string csv_quote(const std::string & in) {
    std::string out = "\"";
    for (char c : in) {
        if (c == '"') out += "\"\"";
        else          out += c;
    }
    out += '"';
    return out;
}

void write_txt(std::ostream & out, const chimera_whisper::TranscribeResult & r) {
    for (const auto & s : r.segments) out << s.text << '\n';
}

void write_srt(std::ostream & out, const chimera_whisper::TranscribeResult & r) {
    int i = 1;
    for (const auto & s : r.segments) {
        out << i++ << '\n'
            << ts_srt(s.t0) << " --> " << ts_srt(s.t1) << '\n'
            << s.text << "\n\n";
    }
}

void write_vtt(std::ostream & out, const chimera_whisper::TranscribeResult & r) {
    out << "WEBVTT\n\n";
    for (const auto & s : r.segments) {
        out << ts_vtt(s.t0) << " --> " << ts_vtt(s.t1) << '\n'
            << s.text << "\n\n";
    }
}

void write_csv(std::ostream & out, const chimera_whisper::TranscribeResult & r) {
    out << "start_ms,end_ms,text\n";
    for (const auto & s : r.segments) {
        out << (s.t0 * 10) << ',' << (s.t1 * 10) << ',' << csv_quote(s.text) << '\n';
    }
}

void write_lrc(std::ostream & out, const chimera_whisper::TranscribeResult & r) {
    for (const auto & s : r.segments) {
        out << '[' << ts_lrc(s.t0) << ']' << s.text << '\n';
    }
}

void write_json(std::ostream & out, const chimera_whisper::TranscribeResult & r, bool full) {
    out << "{\n";
    if (!r.detected_language.empty()) {
        out << "  \"language\": \"" << json_escape(r.detected_language) << "\",\n";
    }
    out << "  \"text\": \"" << json_escape(r.text) << "\",\n";
    out << "  \"duration\": " << r.audio_duration_s << ",\n";
    out << "  \"segments\": [\n";
    for (size_t i = 0; i < r.segments.size(); ++i) {
        const auto & s = r.segments[i];
        out << "    {\"start_ms\": " << (s.t0 * 10)
            << ", \"end_ms\": "      << (s.t1 * 10)
            << ", \"text\": \""      << json_escape(s.text) << "\"";
        if (full && !s.words.empty()) {
            out << ", \"words\": [";
            for (size_t j = 0; j < s.words.size(); ++j) {
                const auto & w = s.words[j];
                out << "{\"start_ms\": " << (w.t0 * 10)
                    << ", \"end_ms\": "  << (w.t1 * 10)
                    << ", \"text\": \""  << json_escape(w.text) << "\"}";
                if (j + 1 < s.words.size()) out << ", ";
            }
            out << "]";
        }
        out << "}";
        if (i + 1 < r.segments.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";
}

// Path stem helper: strips trailing extension from a filename.
std::string strip_ext(const std::string & path) {
    const auto slash = path.find_last_of("/\\");
    const auto dot   = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return path;
    }
    return path.substr(0, dot);
}

void emit_format_file(const std::string & path,
                      void (*writer)(std::ostream &, const chimera_whisper::TranscribeResult &),
                      const chimera_whisper::TranscribeResult & r) {
    std::ofstream f(path);
    if (!f) {
        fail(ExitCode::Runtime, "failed to open output file: " + path);
    }
    writer(f, r);
    std::cerr << "wrote " << path << '\n';
}

}  // namespace

// ---- CLI subcommand ----------------------------------------------------

// CLI driver: loads a fresh whisper_context from opts.model and delegates
// to run_whisper(ctx, opts) for the actual pipeline. The OOP wrapper
// (chimera::Whisper) skips this and calls run_whisper directly against
// its persistent ctx.
int command_whisper(const WhisperOptions & opts) {
    if (opts.model.empty()) {
        fail(ExitCode::BadInput, "whisper requires --model");
    }
    chimera_whisper::LoadParams lp;
    lp.model      = opts.model;
    lp.use_gpu    = !opts.no_gpu;
    lp.flash_attn = opts.flash_attn;
    lp.gpu_device = opts.gpu_device;
    auto ctx = chimera_whisper::load_model(lp);
    if (!ctx) {
        fail(ExitCode::Load, "failed to load whisper model: " + opts.model);
    }
    return run_whisper(ctx.get(), opts);
}

int run_whisper(whisper_context * ctx, const WhisperOptions & opts) {
    if (opts.input.empty()) {
        fail(ExitCode::BadInput, "whisper requires --input");
    }
    // Fast checks before paying the WAV-load cost: mutually-exclusive
    // grammar sources. Detailed parse-time errors (bad rule name, GBNF
    // syntax) still surface below — those need the full grammar source.
    if (!opts.grammar.empty() && !opts.grammar_file.empty()) {
        fail(ExitCode::BadInput,
             "use only one of --grammar or --grammar-file");
    }

    auto wav = chimera_whisper::load_wav_file(opts.input);
    auto audio = chimera_whisper::resample_linear(wav.samples, wav.sample_rate, WHISPER_SAMPLE_RATE);

    // Stereo diarization. We need both channels at 16 kHz for the
    // energy-ratio classifier; mono inputs fail with a precise message
    // (mirrors whisper-cli's behavior where --diarize is a stereo-only
    // feature). Each channel is resampled independently so the
    // per-channel sample count matches `audio` and segment-time -> sample
    // arithmetic uses one rate everywhere.
    std::vector<std::vector<float>> diarize_16k;
    if (opts.diarize) {
        if (wav.channels != 2) {
            fail(ExitCode::BadInput,
                 "--diarize requires a 2-channel (stereo) WAV input; got " +
                 std::to_string(wav.channels) +
                 (wav.channels == 1 ? "-channel (mono)" : "-channel") + " audio");
        }
        if (wav.per_channel.size() != 2) {
            // Defensive: WAV parser should populate per_channel whenever
            // channels > 1. If it didn't (parser change?), bail loudly
            // rather than silently producing all-"(speaker ?)" segments.
            fail(ExitCode::Runtime,
                 "internal: stereo WAV has no per_channel data — refusing to diarize");
        }
        diarize_16k.resize(2);
        for (int c = 0; c < 2; ++c) {
            diarize_16k[c] = chimera_whisper::resample_linear(
                wav.per_channel[c], wav.sample_rate, WHISPER_SAMPLE_RATE);
        }
    }
    // Energy-ratio classifier matching whisper-cli's
    // estimate_diarization_speaker: sum |amplitude| over [t0,t1] in 16kHz
    // samples for both channels; 1.1x ratio threshold picks a speaker,
    // otherwise "?". t0/t1 are in 10ms units.
    auto estimate_speaker = [&](int64_t t0, int64_t t1) -> std::string {
        if (diarize_16k.size() != 2) return "";
        const int64_t n_samples = static_cast<int64_t>(diarize_16k[0].size());
        const auto clamp = [&](int64_t t) -> int64_t {
            int64_t s = (t * WHISPER_SAMPLE_RATE) / 100;
            if (s < 0)         s = 0;
            if (s > n_samples) s = n_samples;
            return s;
        };
        const int64_t is0 = clamp(t0);
        const int64_t is1 = clamp(t1);
        double e0 = 0.0, e1 = 0.0;
        for (int64_t j = is0; j < is1; ++j) {
            e0 += std::fabs(diarize_16k[0][j]);
            e1 += std::fabs(diarize_16k[1][j]);
        }
        const char * id = (e0 > 1.1 * e1) ? "0" : (e1 > 1.1 * e0) ? "1" : "?";
        return std::string("(speaker ") + id + ")";
    };

    std::ofstream out_file;
    std::ostream * out = &std::cout;
    if (!opts.output.empty()) {
        out_file.open(opts.output);
        if (!out_file) {
            fail(ExitCode::BadInput, "failed to open output file: " + opts.output);
        }
        out = &out_file;
    }

    const bool any_format = opts.out_txt || opts.out_srt || opts.out_vtt ||
                            opts.out_json || opts.out_json_full ||
                            opts.out_csv || opts.out_lrc;
    // SRT/VTT/CSV/LRC/JSON all need segment-level timestamps. Whisper.cpp
    // returns garbage t0/t1 when no_timestamps=true, so force segment
    // timing on whenever any format file is requested, independent of the
    // user-facing --timestamps flag (which only controls inline streaming).
    chimera_whisper::TranscribeRequest req;
    req.audio_16k_mono  = std::move(audio);
    req.language        = opts.language;
    req.translate       = opts.translate;
    req.no_context      = opts.no_context;
    req.emit_timestamps = opts.timestamps || any_format;
    req.threads         = opts.threads;
    req.initial_prompt        = opts.initial_prompt;
    req.carry_initial_prompt  = opts.carry_initial_prompt;
    req.beam_size             = opts.beam_size;
    req.best_of               = opts.best_of;
    req.temperature           = opts.temperature;
    req.no_fallback           = opts.no_fallback;
    req.offset_ms             = opts.offset_ms;
    req.duration_ms           = opts.duration_ms;
    req.vad                       = opts.vad;
    req.vad_model_path            = opts.vad_model;
    req.vad_threshold             = opts.vad_threshold;
    req.vad_min_speech_duration_ms  = opts.vad_min_speech_duration_ms;
    req.vad_min_silence_duration_ms = opts.vad_min_silence_duration_ms;
    req.vad_max_speech_duration_s   = opts.vad_max_speech_duration_s;
    req.vad_speech_pad_ms           = opts.vad_speech_pad_ms;
    req.vad_samples_overlap         = opts.vad_samples_overlap;
    req.max_len            = opts.max_len;
    req.max_tokens         = opts.max_tokens;
    req.split_on_word      = opts.split_on_word;
    req.temperature_inc    = opts.temperature_inc;
    req.entropy_thold      = opts.entropy_thold;
    req.logprob_thold      = opts.logprob_thold;
    req.no_speech_thold    = opts.no_speech_thold;
    req.audio_ctx          = opts.audio_ctx;
    req.tinydiarize        = opts.tinydiarize;
    req.suppress_regex     = opts.suppress_regex;
    req.suppress_nst       = opts.suppress_nst;
    req.processors         = opts.processors;
    req.detect_language    = opts.detect_language;

    // Grammar constraint. parse_state + c_rules() output both live on
    // this stack frame for the duration of transcribe(). c_rules()
    // builds a vector<const whisper_grammar_element *> by taking the
    // address of each row of parse_state.rules, so parse_state must
    // outlive the borrowed pointer view.
    grammar_parser::parse_state grammar_state;
    std::vector<const whisper_grammar_element *> grammar_c_rules;
    std::string grammar_src = opts.grammar;
    if (grammar_src.empty() && !opts.grammar_file.empty()) {
        std::ifstream f(opts.grammar_file);
        if (!f) {
            fail(ExitCode::BadInput,
                 "failed to open --grammar-file: " + opts.grammar_file);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        grammar_src = ss.str();
    }
    if (!grammar_src.empty()) {
        try {
            grammar_state = grammar_parser::parse(grammar_src.c_str());
        } catch (const std::exception & e) {
            fail(ExitCode::BadInput,
                 std::string("--grammar parse failed: ") + e.what());
        }
        if (grammar_state.rules.empty()) {
            fail(ExitCode::BadInput,
                 "--grammar parsed but produced no rules");
        }
        const auto it = grammar_state.symbol_ids.find(opts.grammar_rule);
        if (it == grammar_state.symbol_ids.end()) {
            fail(ExitCode::BadInput,
                 "--grammar-rule '" + opts.grammar_rule +
                 "' not found in the parsed grammar (top-level rules must "
                 "be declared with `rule ::= ...`)");
        }
        grammar_c_rules    = grammar_state.c_rules();
        req.grammar_rules  = &grammar_c_rules;
        req.grammar_rule_index = static_cast<int>(it->second);
        req.grammar_penalty    = opts.grammar_penalty;
    }

    // -ojf needs per-word timing on top of segment-level timing.
    req.word_timestamps = opts.out_json_full;
    // Stream each finalized segment as soon as whisper.cpp emits it. Same
    // visible behavior as before the refactor.
    req.on_segment = [&](const chimera_whisper::Segment & s) {
        if (opts.timestamps) {
            *out << '['
                 << chimera_whisper::format_timestamp_10ms(s.t0)
                 << " --> "
                 << chimera_whisper::format_timestamp_10ms(s.t1)
                 << "] ";
        }
        if (opts.diarize) {
            // Speaker label is computed once per segment here and reused
            // when stamping result.segments below — the energy-ratio
            // function is fast enough that running it twice would be
            // fine, but the lambda is the only place where this segment
            // is identifiable, so we compute it here and rely on the
            // post-transcribe walk to keep result.segments in sync.
            *out << estimate_speaker(s.t0, s.t1) << ' ';
        }
        *out << s.text << '\n' << std::flush;
    };

    auto result = chimera_whisper::transcribe(ctx, req);

    // --detect-language short-circuit. whisper_full returns before
    // running any decode pass, so result has no segments. Emit just
    // the detected language code (or `?` if detection itself failed)
    // and skip all the streaming + format-file plumbing below. The
    // output goes to the same sink as a normal transcript would
    // (-o file when set, stdout otherwise) so users can pipe it.
    if (opts.detect_language) {
        const std::string code = result.detected_language.empty()
            ? std::string("?") : result.detected_language;
        *out << code << '\n' << std::flush;
        return 0;
    }

    // For --diarize, stamp each finalized segment with its speaker
    // label so the SRT/VTT/JSON/CSV/LRC writers below see it. The label
    // is written into `segment.speaker` (structured) and also prefixed
    // onto `segment.text` (so existing format writers, which only look
    // at .text, render the prefix verbatim without further changes).
    if (opts.diarize && diarize_16k.size() == 2) {
        for (auto & s : result.segments) {
            s.speaker = estimate_speaker(s.t0, s.t1);
            if (!s.speaker.empty()) {
                s.text = s.speaker + ' ' + s.text;
            }
        }
    }

    if (any_format) {
        const std::string base = opts.output_file_base.empty()
            ? strip_ext(opts.input)
            : opts.output_file_base;
        if (opts.out_txt)  emit_format_file(base + ".txt",  write_txt, result);
        if (opts.out_srt)  emit_format_file(base + ".srt",  write_srt, result);
        if (opts.out_vtt)  emit_format_file(base + ".vtt",  write_vtt, result);
        if (opts.out_csv)  emit_format_file(base + ".csv",  write_csv, result);
        if (opts.out_lrc)  emit_format_file(base + ".lrc",  write_lrc, result);
        if (opts.out_json) {
            std::ofstream f(base + ".json");
            if (!f) fail(ExitCode::Runtime, "failed to open output file: " + base + ".json");
            write_json(f, result, /*full=*/false);
            std::cerr << "wrote " << (base + ".json") << '\n';
        }
        if (opts.out_json_full) {
            const std::string path = base + ".json";
            // If both -oj and -ojf are set, the full variant wins because
            // it's a strict superset of -oj's output.
            std::ofstream f(path);
            if (!f) fail(ExitCode::Runtime, "failed to open output file: " + path);
            write_json(f, result, /*full=*/true);
            std::cerr << "wrote " << path << " (full)\n";
        }
    }
    return 0;
}
