#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "chimera.h"
#include "stable-diffusion.h"
#include "stb_image_write.h"

static void chimera_silent_sd_log(enum sd_log_level_t, const char *, void *) {}

void chimera_silence_sd_log() {
    sd_set_log_callback(chimera_silent_sd_log, nullptr);
}

void chimera_restore_sd_log() {
    sd_set_log_callback(nullptr, nullptr);
}

namespace {

struct SdContextDeleter {
    void operator()(sd_ctx_t * ctx) const {
        if (ctx != nullptr) {
            free_sd_ctx(ctx);
        }
    }
};

using SdContextPtr = std::unique_ptr<sd_ctx_t, SdContextDeleter>;

void sd_log_callback(enum sd_log_level_t level, const char * text, void * user_data) {
    (void) user_data;
    if (level >= SD_LOG_WARN) {
        std::cerr << text;
    }
}

void save_png(const sd_image_t & image, const std::string & path) {
    if (stbi_write_png(
            path.c_str(),
            static_cast<int>(image.width),
            static_cast<int>(image.height),
            static_cast<int>(image.channel),
            image.data,
            static_cast<int>(image.width * image.channel)) == 0) {
        fail("failed to save PNG: " + path);
    }
}

std::string numbered_output_path(const std::string & path, int index, int count) {
    if (count == 1) {
        return path;
    }

    const auto dot = path.find_last_of('.');
    const std::string stem = dot == std::string::npos ? path : path.substr(0, dot);
    const std::string ext = dot == std::string::npos ? ".png" : path.substr(dot);

    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), "_%03d", index + 1);
    return stem + suffix + ext;
}

} // namespace

int command_sd(const SdOptions & opts) {
    if (opts.model.empty() || opts.prompt.empty()) {
        fail("sd requires --model and --prompt");
    }

    sd_set_log_callback(sd_log_callback, nullptr);

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path = opts.model.c_str();
    ctx_params.n_threads = opts.threads;
    ctx_params.enable_mmap = true;

    SdContextPtr ctx(new_sd_ctx(&ctx_params));
    if (!ctx) {
        fail("failed to load stable diffusion model: " + opts.model);
    }
    if (!sd_ctx_supports_image_generation(ctx.get())) {
        fail("loaded model does not support image generation");
    }

    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);
    gen_params.prompt = opts.prompt.c_str();
    gen_params.negative_prompt = opts.negative_prompt.c_str();
    gen_params.width = opts.width;
    gen_params.height = opts.height;
    gen_params.seed = opts.seed;
    gen_params.batch_count = opts.batch_count;
    gen_params.clip_skip = opts.clip_skip;
    gen_params.sample_params.sample_steps = opts.steps;
    gen_params.sample_params.guidance.txt_cfg = opts.cfg_scale;
    gen_params.sample_params.sample_method = opts.sample_method.empty()
        ? SAMPLE_METHOD_COUNT
        : str_to_sample_method(opts.sample_method.c_str());
    gen_params.sample_params.scheduler = opts.scheduler.empty()
        ? SCHEDULER_COUNT
        : str_to_scheduler(opts.scheduler.c_str());

    sd_image_t * images = generate_image(ctx.get(), &gen_params);
    if (images == nullptr) {
        fail("image generation failed");
    }

    for (int i = 0; i < opts.batch_count; ++i) {
        const std::string out = numbered_output_path(opts.output, i, opts.batch_count);
        save_png(images[i], out);
        std::cout << out << '\n';
        std::free(images[i].data);
    }
    std::free(images);

    return 0;
}
