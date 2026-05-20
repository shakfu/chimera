// chimera_sd.cpp — stable-diffusion.cpp wrapper consumed by both the
// `sd` CLI subcommand and the `serve` POST /v1/images/* routes.
//
// Public API in chimera_sd.h. Anything private to this TU (progress
// spinner, numbered-output-path helper) stays in the anonymous namespace.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
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

// Last-N log line ring shared by the chimera log callback and the
// `recent_log_lines()` accessor. Lines are stored exactly as sd.cpp / ggml
// emit them (already trailing-newline-bearing) so the public API can
// join them for an HTTP error body without re-formatting.
static std::mutex              g_log_mtx;
static std::deque<std::string> g_log_buf;
static constexpr size_t        kLogBufCap = 64;

static void push_log_line(const char * text) {
    if (text == nullptr || *text == '\0') return;
    std::lock_guard<std::mutex> lock(g_log_mtx);
    g_log_buf.emplace_back(text);
    while (g_log_buf.size() > kLogBufCap) {
        g_log_buf.pop_front();
    }
}

static void chimera_silent_sd_log(enum sd_log_level_t, const char * text, void *) {
    // Even in "silenced" mode (CLI path uses this while loading the model
    // to suppress stderr noise) we still want the ring to capture lines —
    // an HTTP error after model load is the whole point of the buffer.
    push_log_line(text);
}

void chimera_silence_sd_log() {
    sd_set_log_callback(chimera_silent_sd_log, nullptr);
}

void chimera_restore_sd_log() {
    sd_set_log_callback(nullptr, nullptr);
}

namespace {

void sd_log_callback(enum sd_log_level_t level, const char * text, void * user_data) {
    (void) user_data;
    push_log_line(text);
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

SdContextPtr load_model(const LoadParams & params) {
    sd_set_log_callback(sd_log_callback, nullptr);

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    // Empty-string -> nullptr so sd.cpp's "is this component present?"
    // checks (which compare against nullptr) behave correctly.
    auto cstr = [](const std::string & s) -> const char * {
        return s.empty() ? nullptr : s.c_str();
    };
    ctx_params.model_path            = cstr(params.model);
    ctx_params.diffusion_model_path  = cstr(params.diffusion_model);
    ctx_params.vae_path              = cstr(params.vae);
    ctx_params.clip_l_path           = cstr(params.clip_l);
    ctx_params.clip_g_path           = cstr(params.clip_g);
    ctx_params.t5xxl_path            = cstr(params.t5xxl);
    ctx_params.llm_path              = cstr(params.llm);
    ctx_params.control_net_path      = cstr(params.control_net);
    ctx_params.n_threads             = params.threads;
    ctx_params.enable_mmap           = true;
    ctx_params.vae_decode_only       = params.vae_decode_only;
    ctx_params.offload_params_to_cpu = params.offload_to_cpu;
    ctx_params.diffusion_flash_attn  = params.diffusion_flash_attn;
    ctx_params.diffusion_conv_direct = params.diffusion_conv_direct;
    ctx_params.vae_conv_direct       = params.vae_conv_direct;
    auto parse_rng = [](const std::string & name, const char * flag) -> rng_type_t {
        const rng_type_t t = str_to_rng_type(name.c_str());
        if (t == RNG_TYPE_COUNT) {
            fail(ExitCode::BadInput,
                 std::string("unknown ") + flag + " value: " + name +
                 " (expected std_default, cuda, or cpu)");
        }
        return t;
    };
    if (!params.rng_type.empty()) {
        ctx_params.rng_type = parse_rng(params.rng_type, "--rng");
    }
    if (!params.sampler_rng_type.empty()) {
        ctx_params.sampler_rng_type = parse_rng(params.sampler_rng_type, "--sampler-rng");
    }
    if (!params.wtype.empty()) {
        const sd_type_t wt = str_to_sd_type(params.wtype.c_str());
        if (wt == SD_TYPE_COUNT) {
            fail(ExitCode::BadInput,
                 "unknown --type value: " + params.wtype +
                 " (expected one of f16/f32/bf16/q8_0/q5_1/q5_0/q4_1/q4_0/q4_k/q3_k/q2_k/iq4_nl/...)");
        }
        ctx_params.wtype = wt;
    }

    SdContextPtr ctx(new_sd_ctx(&ctx_params));
    return ctx;
}

SdContextPtr load_model(const std::string & path, bool vae_decode_only, int threads) {
    LoadParams p;
    p.model            = path;
    p.vae_decode_only  = vae_decode_only;
    p.threads          = threads;
    return load_model(p);
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
    if (req.guidance   >= 0.0f) gp.sample_params.guidance.distilled_guidance = req.guidance;
    if (req.flow_shift >= 0.0f) gp.sample_params.flow_shift                   = req.flow_shift;
    gp.sample_params.sample_method = req.sample_method.empty()
        ? SAMPLE_METHOD_COUNT
        : str_to_sample_method(req.sample_method.c_str());
    gp.sample_params.scheduler = req.scheduler.empty()
        ? SCHEDULER_COUNT
        : str_to_scheduler(req.scheduler.c_str());

    sd_image_t init_img{};
    sd_image_t mask_img{};
    sd_image_t control_img{};
    const bool have_init    = !req.init.pixels.empty();
    const bool have_mask    = !req.mask.pixels.empty();
    const bool have_control = !req.control.pixels.empty();
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
    if (have_control) {
        if (req.control.width != req.width || req.control.height != req.height) {
            fail(ExitCode::BadInput,
                 "control image dimensions must match generation width/height");
        }
        control_img = pixel_image_to_sd(req.control);
        gp.control_image    = control_img;
        gp.control_strength = req.control_strength;
    }

    if (req.vae_tiling) {
        gp.vae_tiling_params.enabled = true;
        if (req.vae_tile_size > 0) {
            gp.vae_tiling_params.tile_size_x = req.vae_tile_size;
            gp.vae_tiling_params.tile_size_y = req.vae_tile_size;
        }
        if (req.vae_relative_tile_size >= 0.0f) {
            gp.vae_tiling_params.rel_size_x = req.vae_relative_tile_size;
            gp.vae_tiling_params.rel_size_y = req.vae_relative_tile_size;
        }
        if (req.vae_tile_overlap >= 0.0f) {
            gp.vae_tiling_params.target_overlap = req.vae_tile_overlap;
        }
    }

    // LoRA adapters. sd_lora_t.path is borrowed from the caller-owned
    // strings inside req.loras; keep the std::vector<sd_lora_t> on this
    // stack frame so it outlives the generate_image call.
    std::vector<sd_lora_t> lora_specs;
    if (!req.loras.empty()) {
        lora_specs.reserve(req.loras.size());
        for (const auto & l : req.loras) {
            sd_lora_t entry{};
            entry.is_high_noise = false;
            entry.multiplier    = l.scale;
            entry.path          = l.path.c_str();
            lora_specs.push_back(entry);
        }
        gp.loras      = lora_specs.data();
        gp.lora_count = static_cast<uint32_t>(lora_specs.size());
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

// ---- log capture --------------------------------------------------------

std::vector<std::string> recent_log_lines(size_t max_lines) {
    std::lock_guard<std::mutex> lock(g_log_mtx);
    std::vector<std::string> out;
    const size_t take = std::min(max_lines, g_log_buf.size());
    out.reserve(take);
    for (auto it = g_log_buf.end() - static_cast<std::ptrdiff_t>(take);
         it != g_log_buf.end(); ++it) {
        out.push_back(*it);
    }
    return out;
}

void clear_log_buffer() {
    std::lock_guard<std::mutex> lock(g_log_mtx);
    g_log_buf.clear();
}

}  // namespace chimera_sd

// ---- CLI subcommand ----------------------------------------------------

int command_sd(const SdOptions & opts) {
    if (opts.prompt.empty()) {
        fail(ExitCode::BadInput, "sd requires --prompt");
    }
    if (opts.model.empty() && opts.diffusion_model.empty()) {
        fail(ExitCode::BadInput,
             "sd requires --model (combined checkpoint) or --diffusion-model "
             "(split layout, e.g. Z-Image / Flux)");
    }

    if (!opts.control_image.empty() && opts.control_net.empty()) {
        fail(ExitCode::BadInput,
             "--control-image requires --control-net (the conditioning model)");
    }

    // VAE encode path is only needed for img2img / inpaint.
    const bool need_encode = !opts.init_image.empty();
    chimera_sd::LoadParams lp;
    lp.model                = opts.model;
    lp.diffusion_model      = opts.diffusion_model;
    lp.vae                  = opts.vae;
    lp.clip_l               = opts.clip_l;
    lp.clip_g               = opts.clip_g;
    lp.t5xxl                = opts.t5xxl;
    lp.llm                  = opts.llm;
    lp.control_net          = opts.control_net;
    lp.wtype                = opts.wtype;
    lp.vae_decode_only      = !need_encode;
    lp.offload_to_cpu        = opts.offload_to_cpu;
    lp.diffusion_flash_attn  = opts.diffusion_fa;
    lp.diffusion_conv_direct = opts.diffusion_conv_direct;
    lp.vae_conv_direct       = opts.vae_conv_direct;
    lp.rng_type              = opts.rng;
    lp.sampler_rng_type      = opts.sampler_rng;
    lp.threads               = opts.threads;
    auto ctx = chimera_sd::load_model(lp);
    if (!ctx) {
        const std::string & shown = opts.model.empty() ? opts.diffusion_model : opts.model;
        fail(ExitCode::Load, "failed to load stable diffusion model: " + shown);
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
    req.guidance         = opts.guidance;
    req.flow_shift       = opts.flow_shift;
    req.control_strength = opts.control_strength;
    req.vae_tiling             = opts.vae_tiling;
    req.vae_tile_size          = opts.vae_tile_size;
    req.vae_relative_tile_size = opts.vae_relative_tile_size;
    req.vae_tile_overlap       = opts.vae_tile_overlap;

    // Resolve --lora "<path>[:<scale>]" specs into LoraEntry. Relative
    // paths are joined against --lora-model-dir when set (mirrors the
    // upstream sd-cli behavior). Filesystem path separator detection is
    // platform-aware: '/' on POSIX, plus '\\' on Windows.
    auto split_lora_spec = [](const std::string & spec) -> std::pair<std::string, float> {
        const auto colon = spec.find_last_of(':');
        if (colon == std::string::npos) return {spec, 1.0f};
        try {
            const float scale = std::stof(spec.substr(colon + 1));
            return {spec.substr(0, colon), scale};
        } catch (const std::exception &) {
            return {spec, 1.0f};
        }
    };
    auto is_absolute_or_explicit = [](const std::string & p) {
        if (p.empty()) return false;
        if (p[0] == '/' || p[0] == '.') return true;
#ifdef _WIN32
        if (p[0] == '\\') return true;
        if (p.size() >= 2 && p[1] == ':') return true; // drive letter
#endif
        return false;
    };
    for (const auto & raw : opts.lora_adapters) {
        auto [path, scale] = split_lora_spec(raw);
        if (!opts.lora_model_dir.empty() && !is_absolute_or_explicit(path)) {
            std::string base = opts.lora_model_dir;
            if (base.back() != '/' && base.back() != '\\') base += '/';
            path = base + path;
        }
        chimera_sd::LoraEntry e;
        e.path  = std::move(path);
        e.scale = scale;
        req.loras.push_back(std::move(e));
    }

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
    if (!opts.control_image.empty()) {
        req.control = chimera_sd::decode_image_file(opts.control_image, 3);
        if (req.control.width != opts.width || req.control.height != opts.height) {
            fail(ExitCode::BadInput,
                 "control image dimensions must match --width / --height");
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
