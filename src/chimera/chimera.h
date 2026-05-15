#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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
    std::string mmproj;             // empty = text-only; otherwise mtmd vision projector
    std::vector<std::string> images;  // images to feed alongside the prompt (gen only)
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

struct EmbedOptions {
    std::string model;
    std::string input;
    std::string input_file;
    std::string output;            // empty = stdout
    std::string pooling = "mean";  // mean | cls | last | none
    int threads = -1;
    int gpu_layers = 0;
    uint32_t n_ctx = 0;            // 0 = use model's default training context
    uint32_t n_batch = 512;
    bool normalize = true;
    bool use_mmap = true;
};

struct TokenizeOptions {
    std::string model;
    std::string input;
    std::string input_file;
    bool add_special = true;
    bool parse_special = true;
    bool show_pieces = false;
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
    std::string init_image;  // empty = text-to-image
    std::string mask_image;  // empty = no inpainting mask
    int width = 512;
    int height = 512;
    int steps = 20;
    int batch_count = 1;
    int clip_skip = -1;
    int threads = -1;
    int64_t seed = -1;
    float cfg_scale = 7.0f;
    float strength = 0.75f;  // img2img denoising strength (0 = preserve, 1 = full noise)
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

// Exit codes. CLI11 parse errors come from CLI11 itself (codes >= 100 by
// default); use values <= 10 here so they remain visually distinct.
//   1  generic runtime error (fallback)
//   2  bad input (missing/invalid file, malformed argument)
//   3  model-load failure
//   4  generation / inference failure
enum class ExitCode : int {
    Runtime  = 1,
    BadInput = 2,
    Load     = 3,
    Generate = 4,
};

// Runtime exception that carries a structured exit code. Caught in main()
// to map to the process exit status. Use fail(message) for generic errors
// (Runtime) or fail(code, message) for a specific category.
class ChimeraError : public std::runtime_error {
public:
    ChimeraError(ExitCode code, const std::string & msg)
        : std::runtime_error(msg), code_(code) {}
    ExitCode code() const noexcept { return code_; }

private:
    ExitCode code_;
};

[[noreturn]] inline void fail(const std::string & message) {
    throw ChimeraError(ExitCode::Runtime, message);
}

[[noreturn]] inline void fail(ExitCode code, const std::string & message) {
    throw ChimeraError(code, message);
}

inline std::string trim(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}
