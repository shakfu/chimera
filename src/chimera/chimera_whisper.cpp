#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "chimera.h"
#include "whisper.h"

static void chimera_silent_whisper_log(enum ggml_log_level, const char *, void *) {}

void chimera_silence_whisper_log() {
    whisper_log_set(chimera_silent_whisper_log, nullptr);
}

void chimera_restore_whisper_log() {
    whisper_log_set(nullptr, nullptr);
}

namespace {

struct WavData {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples;
};

struct WhisperContextDeleter {
    void operator()(whisper_context * ctx) const {
        if (ctx != nullptr) {
            whisper_free(ctx);
        }
    }
};

using WhisperContextPtr = std::unique_ptr<whisper_context, WhisperContextDeleter>;

template <typename T>
T read_le(std::istream & in) {
    T value {};
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!in) {
        fail("failed to read input file");
    }
    return value;
}

WavData load_wav_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("failed to open WAV file: " + path);
    }

    char riff[4];
    char wave[4];
    in.read(riff, 4);
    (void) read_le<uint32_t>(in);
    in.read(wave, 4);
    if (std::string_view(riff, 4) != "RIFF" || std::string_view(wave, 4) != "WAVE") {
        fail("unsupported WAV container: " + path);
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
        fail("incomplete WAV file: " + path);
    }

    const size_t bytes_per_sample = bits_per_sample / 8;
    if (bytes_per_sample == 0 || pcm_bytes.size() % bytes_per_sample != 0) {
        fail("invalid WAV sample size: " + path);
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
            fail("unsupported PCM bit depth in WAV: " + std::to_string(bits_per_sample));
        }
    } else if (audio_format == 3 && bits_per_sample == 32) {
        for (size_t i = 0; i < pcm_bytes.size(); i += 4) {
            float sample = 0.0f;
            std::memcpy(&sample, pcm_bytes.data() + i, sizeof(float));
            interleaved.push_back(sample);
        }
    } else {
        fail("unsupported WAV encoding in " + path);
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

    return WavData {static_cast<int>(sample_rate), static_cast<int>(channels), std::move(mono)};
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
        std::snprintf(
            buf,
            sizeof(buf),
            "%02lld:%02lld:%02lld.%03lld",
            static_cast<long long>(hours),
            static_cast<long long>(minutes),
            static_cast<long long>(seconds),
            static_cast<long long>(ms));
    } else {
        std::snprintf(
            buf,
            sizeof(buf),
            "%02lld:%02lld.%03lld",
            static_cast<long long>(minutes),
            static_cast<long long>(seconds),
            static_cast<long long>(ms));
    }
    return buf;
}

} // namespace

namespace {

struct SegmentCallbackCtx {
    std::ostream * out;
    bool timestamps;
    int next_segment;
};

void on_new_segment(struct whisper_context * /*ctx*/, struct whisper_state * state,
                    int n_new, void * user_data) {
    auto * cb = static_cast<SegmentCallbackCtx *>(user_data);
    // Segments are owned by `state` until whisper_full returns. Use the
    // _from_state accessors -- the ctx-based ones are not yet populated.
    const int n_total = whisper_full_n_segments_from_state(state);
    const int n_start = std::max(0, n_total - n_new);
    for (int i = n_start; i < n_total; ++i) {
        const std::string text = trim(whisper_full_get_segment_text_from_state(state, i));
        if (cb->timestamps) {
            *cb->out << "["
                     << format_timestamp_10ms(whisper_full_get_segment_t0_from_state(state, i))
                     << " --> "
                     << format_timestamp_10ms(whisper_full_get_segment_t1_from_state(state, i))
                     << "] ";
        }
        *cb->out << text << '\n' << std::flush;
    }
    cb->next_segment = n_total;
}

}  // namespace

int command_whisper(const WhisperOptions & opts) {
    if (opts.model.empty() || opts.input.empty()) {
        fail(ExitCode::BadInput, "whisper requires --model and --input");
    }

    WavData wav = load_wav_file(opts.input);
    std::vector<float> audio = resample_linear(wav.samples, wav.sample_rate, WHISPER_SAMPLE_RATE);

    whisper_context_params cparams = whisper_context_default_params();
    WhisperContextPtr ctx(whisper_init_from_file_with_params(opts.model.c_str(), cparams));
    if (!ctx) {
        fail(ExitCode::Load, "failed to load whisper model: " + opts.model);
    }

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    // opts.threads defaults to -1 ("auto"); whisper.cpp would then size a
    // std::vector(n_threads) and throw std::length_error("vector"). Leave
    // the default from whisper_full_default_params alone unless the user
    // passed a positive value.
    if (opts.threads > 0) {
        params.n_threads = opts.threads;
    }
    params.translate = opts.translate;
    params.no_context = opts.no_context;
    params.no_timestamps = !opts.timestamps;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;

    // Language defaults to "en" (matches whisper_full_default_params).
    // Pass "auto" or set --language explicitly to override.
    if (opts.language == "auto") {
        params.language = nullptr;
        params.detect_language = true;
    } else if (!opts.language.empty()) {
        params.language = opts.language.c_str();
        params.detect_language = false;
    }

    // Wire up streaming output via the new-segment callback. Each finalized
    // segment is printed as soon as whisper.cpp produces it, instead of
    // buffering everything until whisper_full returns.
    std::ofstream out_file;
    std::ostream * out = &std::cout;
    if (!opts.output.empty()) {
        out_file.open(opts.output);
        if (!out_file) {
            fail(ExitCode::BadInput, "failed to open output file: " + opts.output);
        }
        out = &out_file;
    }

    SegmentCallbackCtx cb_ctx { out, opts.timestamps, 0 };
    params.new_segment_callback = on_new_segment;
    params.new_segment_callback_user_data = &cb_ctx;

    if (whisper_full(ctx.get(), params, audio.data(), static_cast<int>(audio.size())) != 0) {
        fail(ExitCode::Generate, "whisper transcription failed");
    }

    return 0;
}
