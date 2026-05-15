#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>

// Version strings. Overridden at compile time via -DCHIMERA_*_VERSION="..."
// from CMake; fall back to "unknown" when built outside the project's CMake.
#ifndef CHIMERA_VERSION
#define CHIMERA_VERSION "unknown"
#endif
#ifndef CHIMERA_LLAMACPP_VERSION
#define CHIMERA_LLAMACPP_VERSION "unknown"
#endif
#ifndef CHIMERA_WHISPERCPP_VERSION
#define CHIMERA_WHISPERCPP_VERSION "unknown"
#endif
#ifndef CHIMERA_SDCPP_VERSION
#define CHIMERA_SDCPP_VERSION "unknown"
#endif

struct LlamaCommonOptions {
    std::string model;
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 512;
    int threads = -1;
    int gpu_layers = 0;
    int n_predict = 256;
    uint32_t seed = 0xFFFFFFFFu;
    float temp = 0.8f;
    int top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.05f;
    float repeat_penalty = 1.1f;
    bool use_mmap = true;
};

struct WhisperOptions {
    std::string model;
    std::string input;
    std::string output;
    std::string language;
    int threads = -1;
    bool translate = false;
    bool timestamps = false;
    bool no_context = false;
};

struct SdOptions {
    std::string model;
    std::string prompt;
    std::string negative_prompt;
    std::string output = "output.png";
    std::string sample_method;
    std::string scheduler;
    int width = 512;
    int height = 512;
    int steps = 20;
    int batch_count = 1;
    int clip_skip = -1;
    int threads = -1;
    int64_t seed = -1;
    float cfg_scale = 7.0f;
};

int command_whisper(const WhisperOptions & opts);
int command_sd(const SdOptions & opts);

void chimera_silence_whisper_log();
void chimera_restore_whisper_log();
void chimera_silence_sd_log();
void chimera_restore_sd_log();

// Shared utility helpers, inlined here so all three TUs (chimera.cpp,
// chimera_whisper.cpp, chimera_sd.cpp) get one definition without needing
// a separate .cpp -- and without re-introducing the ggml.h collision.

[[noreturn]] inline void fail(const std::string & message) {
    throw std::runtime_error(message);
}

inline std::string trim(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}
