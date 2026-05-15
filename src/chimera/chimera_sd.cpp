#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "chimera.h"
#include "stable-diffusion.h"
#include "stb_image.h"
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
        fail(ExitCode::Runtime, "failed to save PNG: " + path);
    }
}

// Load an image (PNG/JPG/etc.) into an sd_image_t with the requested channel
// count (3 = RGB, 1 = grayscale). If target_w/target_h are non-zero and the
// image dimensions differ, fail loudly -- stable-diffusion.cpp expects the
// init / mask images to match the target generation size exactly. The data
// pointer is owned by stb_image; caller must `stbi_image_free` it.
sd_image_t load_image(const std::string & path, int channels,
                      int target_w, int target_h, const char * role) {
    int w = 0, h = 0, in_channels = 0;
    unsigned char * data = stbi_load(path.c_str(), &w, &h, &in_channels, channels);
    if (data == nullptr) {
        fail(ExitCode::BadInput,
             std::string("failed to load ") + role + " image: " + path +
             " (" + (stbi_failure_reason() ? stbi_failure_reason() : "unknown") + ")");
    }
    if ((target_w > 0 && w != target_w) || (target_h > 0 && h != target_h)) {
        stbi_image_free(data);
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "%s image dimensions (%dx%d) must match --width / --height (%dx%d)",
                      role, w, h, target_w, target_h);
        fail(ExitCode::BadInput, buf);
    }
    sd_image_t img;
    img.width = static_cast<uint32_t>(w);
    img.height = static_cast<uint32_t>(h);
    img.channel = static_cast<uint32_t>(channels);
    img.data = data;
    return img;
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

namespace {
void sd_progress_callback(int step, int steps, float /*time*/, void * /*data*/) {
    if (steps <= 0) return;
    // Render to stderr so the line (stdout = produced PNG paths) stays clean.
    std::fprintf(stderr, "\rsd: step %d/%d", step, steps);
    if (step >= steps) {
        std::fputc('\n', stderr);
    }
    std::fflush(stderr);
}
}  // namespace

int command_sd(const SdOptions & opts) {
    if (opts.model.empty() || opts.prompt.empty()) {
        fail(ExitCode::BadInput, "sd requires --model and --prompt");
    }

    sd_set_log_callback(sd_log_callback, nullptr);
    // Show "step N/M" on stderr; user can silence with `2>/dev/null`.
    sd_set_progress_callback(sd_progress_callback, nullptr);

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path = opts.model.c_str();
    ctx_params.n_threads = opts.threads;
    ctx_params.enable_mmap = true;
    // img2img / inpaint need the VAE encode path. Default ctx params set
    // vae_decode_only=true to save memory on text-to-image runs; flip it
    // off whenever the user supplies an --init-image.
    if (!opts.init_image.empty()) {
        ctx_params.vae_decode_only = false;
    }

    SdContextPtr ctx(new_sd_ctx(&ctx_params));
    if (!ctx) {
        fail(ExitCode::Load, "failed to load stable diffusion model: " + opts.model);
    }
    if (!sd_ctx_supports_image_generation(ctx.get())) {
        fail(ExitCode::Load, "loaded model does not support image generation");
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

    // img2img / inpaint: load init (RGB) and optionally mask (single-channel).
    // Both must match --width / --height; stable-diffusion.cpp does not
    // resize internally. Memory is owned by stb_image and freed below.
    sd_image_t init_image{};
    sd_image_t mask_image{};
    const bool have_init = !opts.init_image.empty();
    const bool have_mask = !opts.mask_image.empty();
    if (have_mask && !have_init) {
        fail(ExitCode::BadInput, "--mask-image requires --init-image (inpaint mode)");
    }
    if (have_init) {
        init_image = load_image(opts.init_image, 3, opts.width, opts.height, "init");
        gen_params.init_image = init_image;
        gen_params.strength = opts.strength;
    }
    if (have_mask) {
        mask_image = load_image(opts.mask_image, 1, opts.width, opts.height, "mask");
        gen_params.mask_image = mask_image;
    }

    sd_image_t * images = generate_image(ctx.get(), &gen_params);
    if (have_init) stbi_image_free(init_image.data);
    if (have_mask) stbi_image_free(mask_image.data);
    if (images == nullptr) {
        fail(ExitCode::Generate, "image generation failed");
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
