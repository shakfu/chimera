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
#include "whisper.h"

void WhisperContextDeleter::operator()(whisper_context * ctx) const {
    if (ctx != nullptr) {
        whisper_free(ctx);
    }
}

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
    if (channels == 1) {
        mono = std::move(interleaved);
    } else {
        const size_t frames = interleaved.size() / channels;
        mono.resize(frames, 0.0f);
        for (size_t frame = 0; frame < frames; ++frame) {
            double sum = 0.0;
            for (uint16_t ch = 0; ch < channels; ++ch) {
                sum += interleaved[frame * channels + ch];
            }
            mono[frame] = static_cast<float>(sum / channels);
        }
    }

    return WavData{static_cast<int>(sample_rate), static_cast<int>(channels), std::move(mono)};
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
    whisper_context_params cparams = whisper_context_default_params();
    WhisperContextPtr ctx(whisper_init_from_file_with_params(path.c_str(), cparams));
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

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    // -1 is "auto" upstream but whisper.cpp's default expects a positive
    // value (otherwise it constructs a std::vector(n_threads) and throws).
    if (req.threads > 0) {
        params.n_threads = req.threads;
    }
    params.translate        = req.translate;
    params.no_context       = req.no_context;
    params.no_timestamps    = !req.emit_timestamps;
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

    StreamingCallbackCtx cb_ctx{&req.on_segment};
    if (req.on_segment) {
        params.new_segment_callback = streaming_segment_cb;
        params.new_segment_callback_user_data = &cb_ctx;
    }

    if (whisper_full(ctx, params,
                     req.audio_16k_mono.data(),
                     static_cast<int>(req.audio_16k_mono.size())) != 0) {
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
        if (!result.text.empty() && !s.text.empty()) {
            result.text += ' ';
        }
        result.text += s.text;
        result.segments.push_back(std::move(s));
    }
    if (req.language == "auto") {
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

// ---- CLI subcommand ----------------------------------------------------

int command_whisper(const WhisperOptions & opts) {
    if (opts.model.empty() || opts.input.empty()) {
        fail(ExitCode::BadInput, "whisper requires --model and --input");
    }

    auto wav = chimera_whisper::load_wav_file(opts.input);
    auto audio = chimera_whisper::resample_linear(wav.samples, wav.sample_rate, WHISPER_SAMPLE_RATE);

    auto ctx = chimera_whisper::load_model(opts.model);
    if (!ctx) {
        fail(ExitCode::Load, "failed to load whisper model: " + opts.model);
    }

    std::ofstream out_file;
    std::ostream * out = &std::cout;
    if (!opts.output.empty()) {
        out_file.open(opts.output);
        if (!out_file) {
            fail(ExitCode::BadInput, "failed to open output file: " + opts.output);
        }
        out = &out_file;
    }

    chimera_whisper::TranscribeRequest req;
    req.audio_16k_mono  = std::move(audio);
    req.language        = opts.language;
    req.translate       = opts.translate;
    req.no_context      = opts.no_context;
    req.emit_timestamps = opts.timestamps;
    req.threads         = opts.threads;
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
        *out << s.text << '\n' << std::flush;
    };

    (void) chimera_whisper::transcribe(ctx.get(), req);
    return 0;
}
