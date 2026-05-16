// chimera_sd.cpp — stable-diffusion.cpp wrapper consumed by both the
// `sd` CLI subcommand and the `serve` POST /v1/images/* routes.
//
// Public API in chimera_sd.h. Anything private to this TU (progress
// spinner, numbered-output-path helper) stays in the anonymous namespace.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "chimera.h"
#include "chimera_sd.h"
#include "stable-diffusion.h"

// We can't include ggml.h here without dragging in its enum definitions,
// which collide with the ones in whisper.cpp's ggml. Forward-declare
// just the runtime version accessor we need. With chimera's
// SD_USE_VENDORED_GGML=0 build, this resolves to llama.cpp's ggml symbol.
extern "C" const char * ggml_version(void);
#include "stb_image.h"
#include "stb_image_write.h"

void SdContextDeleter::operator()(sd_ctx_t * ctx) const {
    if (ctx != nullptr) {
        free_sd_ctx(ctx);
    }
}

static void chimera_silent_sd_log(enum sd_log_level_t, const char *, void *) {}

void chimera_silence_sd_log() {
    sd_set_log_callback(chimera_silent_sd_log, nullptr);
}

void chimera_restore_sd_log() {
    sd_set_log_callback(nullptr, nullptr);
}

namespace {

void sd_log_callback(enum sd_log_level_t level, const char * text, void * user_data) {
    (void) user_data;
    if (level >= SD_LOG_WARN) {
        std::cerr << text;
    }
}

void sd_progress_callback(int step, int steps, float /*time*/, void * /*data*/) {
    if (steps <= 0) return;
    // Render to stderr so the line (stdout = produced PNG paths or HTTP body) stays clean.
    std::fprintf(stderr, "\rsd: step %d/%d", step, steps);
    if (step >= steps) {
        std::fputc('\n', stderr);
    }
    std::fflush(stderr);
}

std::string numbered_output_path(const std::string & path, int index, int count) {
    if (count == 1) {
        return path;
    }
    const auto dot = path.find_last_of('.');
    const std::string stem = dot == std::string::npos ? path : path.substr(0, dot);
    const std::string ext  = dot == std::string::npos ? ".png" : path.substr(dot);

    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), "_%03d", index + 1);
    return stem + suffix + ext;
}

// Borrowing PixelImage data to sd_image_t without copying. Caller must keep
// the source PixelImage alive for as long as the sd_image_t is used.
sd_image_t pixel_image_to_sd(const chimera_sd::PixelImage & p) {
    sd_image_t img{};
    img.width   = static_cast<uint32_t>(p.width);
    img.height  = static_cast<uint32_t>(p.height);
    img.channel = static_cast<uint32_t>(p.channels);
    img.data    = const_cast<uint8_t *>(p.pixels.data());
    return img;
}

}  // namespace

namespace chimera_sd {

SdContextPtr load_model(const std::string & path, bool vae_decode_only, int threads) {
    sd_set_log_callback(sd_log_callback, nullptr);

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path       = path.c_str();
    ctx_params.n_threads        = threads;
    ctx_params.enable_mmap      = true;
    ctx_params.vae_decode_only  = vae_decode_only;

    SdContextPtr ctx(new_sd_ctx(&ctx_params));
    return ctx;
}

PixelImage decode_image_bytes(const void * data, size_t size, int channels) {
    if (data == nullptr || size == 0) {
        fail(ExitCode::BadInput, "image buffer is empty");
    }
    int w = 0, h = 0, in_channels = 0;
    unsigned char * pixels = stbi_load_from_memory(
        static_cast<const unsigned char *>(data), static_cast<int>(size),
        &w, &h, &in_channels, channels);
    if (pixels == nullptr) {
        fail(ExitCode::BadInput,
             std::string("failed to decode image: ") +
             (stbi_failure_reason() ? stbi_failure_reason() : "unknown"));
    }
    PixelImage out;
    out.width    = w;
    out.height   = h;
    out.channels = channels;
    out.pixels.assign(pixels, pixels + static_cast<size_t>(w) * h * channels);
    stbi_image_free(pixels);
    return out;
}

PixelImage decode_image_file(const std::string & path, int channels) {
    int w = 0, h = 0, in_channels = 0;
    unsigned char * pixels = stbi_load(path.c_str(), &w, &h, &in_channels, channels);
    if (pixels == nullptr) {
        fail(ExitCode::BadInput,
             std::string("failed to load image: ") + path + " (" +
             (stbi_failure_reason() ? stbi_failure_reason() : "unknown") + ")");
    }
    PixelImage out;
    out.width    = w;
    out.height   = h;
    out.channels = channels;
    out.pixels.assign(pixels, pixels + static_cast<size_t>(w) * h * channels);
    stbi_image_free(pixels);
    return out;
}

std::vector<unsigned char> encode_png(uint32_t              width,
                                      uint32_t              height,
                                      uint32_t              channels,
                                      const unsigned char * pixels) {
    // stbi_write_png_to_func feeds chunks to a callback, which lets us
    // accumulate into a std::vector with a known-correct length. The
    // `_to_mem` overload exists but doesn't return the size in this stb
    // version, so func+callback is the more portable path.
    std::vector<unsigned char> buf;
    struct Sink { std::vector<unsigned char> * out; };
    auto write_cb = [](void * context, void * data, int size) {
        auto * sink  = static_cast<Sink *>(context);
        auto * bytes = static_cast<unsigned char *>(data);
        sink->out->insert(sink->out->end(), bytes, bytes + size);
    };
    Sink sink{&buf};
    if (stbi_write_png_to_func(write_cb, &sink,
                                static_cast<int>(width),
                                static_cast<int>(height),
                                static_cast<int>(channels),
                                pixels,
                                static_cast<int>(width * channels)) == 0) {
        fail(ExitCode::Runtime, "failed to encode PNG");
    }
    return buf;
}

void save_png_file(const std::string &   path,
                   uint32_t              width,
                   uint32_t              height,
                   uint32_t              channels,
                   const unsigned char * pixels) {
    if (stbi_write_png(
            path.c_str(),
            static_cast<int>(width),
            static_cast<int>(height),
            static_cast<int>(channels),
            pixels,
            static_cast<int>(width * channels)) == 0) {
        fail(ExitCode::Runtime, "failed to save PNG: " + path);
    }
}

std::vector<PixelImage> generate(sd_ctx_t * ctx, const GenerateRequest & req) {
    if (ctx == nullptr) {
        fail(ExitCode::Runtime, "sd context is null");
    }
    if (!sd_ctx_supports_image_generation(ctx)) {
        fail(ExitCode::Load, "loaded model does not support image generation");
    }

    sd_set_progress_callback(sd_progress_callback, nullptr);

    sd_img_gen_params_t gp;
    sd_img_gen_params_init(&gp);
    gp.prompt          = req.prompt.c_str();
    gp.negative_prompt = req.negative_prompt.c_str();
    gp.width           = req.width;
    gp.height          = req.height;
    gp.seed            = req.seed;
    gp.batch_count     = req.batch_count;
    gp.clip_skip       = req.clip_skip;
    gp.sample_params.sample_steps    = req.steps;
    gp.sample_params.guidance.txt_cfg = req.cfg_scale;
    gp.sample_params.sample_method = req.sample_method.empty()
        ? SAMPLE_METHOD_COUNT
        : str_to_sample_method(req.sample_method.c_str());
    gp.sample_params.scheduler = req.scheduler.empty()
        ? SCHEDULER_COUNT
        : str_to_scheduler(req.scheduler.c_str());

    sd_image_t init_img{};
    sd_image_t mask_img{};
    const bool have_init = !req.init.pixels.empty();
    const bool have_mask = !req.mask.pixels.empty();
    if (have_mask && !have_init) {
        fail(ExitCode::BadInput, "mask without init image is not supported");
    }
    if (have_init) {
        if (req.init.width != req.width || req.init.height != req.height) {
            fail(ExitCode::BadInput,
                 "init image dimensions must match generation width/height");
        }
        init_img = pixel_image_to_sd(req.init);
        gp.init_image = init_img;
        gp.strength   = req.strength;
    }
    if (have_mask) {
        if (req.mask.width != req.width || req.mask.height != req.height) {
            fail(ExitCode::BadInput,
                 "mask image dimensions must match generation width/height");
        }
        mask_img = pixel_image_to_sd(req.mask);
        gp.mask_image = mask_img;
    }

    sd_image_t * images = generate_image(ctx, &gp);
    if (images == nullptr) {
        fail(ExitCode::Generate, "image generation failed");
    }

    std::vector<PixelImage> out;
    out.reserve(static_cast<size_t>(req.batch_count));
    for (int i = 0; i < req.batch_count; ++i) {
        PixelImage p;
        p.width    = static_cast<int>(images[i].width);
        p.height   = static_cast<int>(images[i].height);
        p.channels = static_cast<int>(images[i].channel);
        const size_t n = static_cast<size_t>(p.width) * p.height * p.channels;
        p.pixels.assign(images[i].data, images[i].data + n);
        std::free(images[i].data);
        out.push_back(std::move(p));
    }
    std::free(images);
    return out;
}

// ---- runtime introspection ---------------------------------------------

std::string sdcpp_version() {
    if (const char * v = sd_version()) return v;
    return "unknown";
}

std::string sd_ggml_version() {
    // Pulled from the ggml that's linked into this TU. With chimera's
    // SD_USE_VENDORED_GGML=0 build mode this is the same ggml used by
    // llama.cpp.
    if (const char * v = ggml_version()) return v;
    return "unknown";
}

std::string sd_system_info_raw() {
    if (const char * s = sd_get_system_info()) return s;
    return "";
}

}  // namespace chimera_sd

// ---- CLI subcommand ----------------------------------------------------

int command_sd(const SdOptions & opts) {
    if (opts.model.empty() || opts.prompt.empty()) {
        fail(ExitCode::BadInput, "sd requires --model and --prompt");
    }

    // VAE encode path is only needed for img2img / inpaint.
    const bool need_encode = !opts.init_image.empty();
    auto ctx = chimera_sd::load_model(opts.model, /*vae_decode_only=*/!need_encode, opts.threads);
    if (!ctx) {
        fail(ExitCode::Load, "failed to load stable diffusion model: " + opts.model);
    }

    chimera_sd::GenerateRequest req;
    req.prompt           = opts.prompt;
    req.negative_prompt  = opts.negative_prompt;
    req.width            = opts.width;
    req.height           = opts.height;
    req.steps            = opts.steps;
    req.batch_count      = opts.batch_count;
    req.clip_skip        = opts.clip_skip;
    req.threads          = opts.threads;
    req.seed             = opts.seed;
    req.cfg_scale        = opts.cfg_scale;
    req.strength         = opts.strength;
    req.sample_method    = opts.sample_method;
    req.scheduler        = opts.scheduler;

    if (!opts.init_image.empty()) {
        req.init = chimera_sd::decode_image_file(opts.init_image, 3);
        if (req.init.width != opts.width || req.init.height != opts.height) {
            fail(ExitCode::BadInput,
                 "init image dimensions must match --width / --height");
        }
    }
    if (!opts.mask_image.empty()) {
        req.mask = chimera_sd::decode_image_file(opts.mask_image, 1);
        if (req.mask.width != opts.width || req.mask.height != opts.height) {
            fail(ExitCode::BadInput,
                 "mask image dimensions must match --width / --height");
        }
    }

    auto images = chimera_sd::generate(ctx.get(), req);

    for (size_t i = 0; i < images.size(); ++i) {
        const std::string out_path = numbered_output_path(
            opts.output, static_cast<int>(i), static_cast<int>(images.size()));
        chimera_sd::save_png_file(out_path,
                                  static_cast<uint32_t>(images[i].width),
                                  static_cast<uint32_t>(images[i].height),
                                  static_cast<uint32_t>(images[i].channels),
                                  images[i].pixels.data());
        std::cout << out_path << '\n';
    }
    return 0;
}
