// chimera_sd.cpp — stable-diffusion.cpp wrapper consumed by both the
// `sd` CLI subcommand and the `serve` POST /v1/images/* routes.
//
// Public API in chimera_sd.h. Anything private to this TU (progress
// spinner, numbered-output-path helper) stays in the anonymous namespace.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
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

// ---- cache / SCM parsing ----------------------------------------------
//
// Mirrors sd-cli's two-flag surface (`--cache-mode` + `--cache-option
// key=value,...`) — same parser, same accepted keys per mode. Kept on
// the chimera side so command_sd can validate up-front before paying
// the model-load cost. The result is stored as plain POD on the
// GenerateRequest; generate() copies fields into sd_cache_params_t.
void parse_cache_options(const std::string & cache_mode,
                         const std::string & cache_option,
                         const std::string & scm_mask,
                         const std::string & scm_policy,
                         GenerateRequest *   req) {
    if (cache_mode.empty() && cache_option.empty() &&
        scm_mask.empty() && scm_policy.empty()) {
        return;
    }
    // Resolve mode first (kv parser branches on it for the `threshold`
    // and `warmup` keys).
    int mode_id = -1;
    if (!cache_mode.empty()) {
        if      (cache_mode == "disabled")   mode_id = 0;
        else if (cache_mode == "easycache")  mode_id = 1;
        else if (cache_mode == "ucache")     mode_id = 2;
        else if (cache_mode == "dbcache")    mode_id = 3;
        else if (cache_mode == "taylorseer") mode_id = 4;
        else if (cache_mode == "cache-dit")  mode_id = 5;
        else if (cache_mode == "spectrum")   mode_id = 6;
        else {
            fail(ExitCode::BadInput,
                 "unknown --cache-mode value: " + cache_mode +
                 " (expected disabled, easycache, ucache, dbcache, "
                 "taylorseer, cache-dit, or spectrum)");
        }
    }
    req->cache_mode_id = mode_id;

    // Parse "k=v,k=v,..." into the matching field. Branching on
    // mode_id matches sd-cli (e.g. `threshold` lives on
    // reuse_threshold for easy/ucache vs residual_diff_threshold for
    // dbcache/taylorseer/cache-dit; `warmup` lives on
    // spectrum_warmup_steps vs max_warmup_steps).
    auto trim = [](std::string & s) {
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))  s.pop_back();
    };
    if (!cache_option.empty()) {
        const std::string & opt = cache_option;
        size_t pos = 0;
        while (pos < opt.size()) {
            size_t comma = opt.find(',', pos);
            std::string tok = opt.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            trim(tok);
            if (!tok.empty()) {
                size_t eq = tok.find('=');
                if (eq == std::string::npos) {
                    fail(ExitCode::BadInput,
                         "--cache-option entry missing '=' separator: '" + tok + "'");
                }
                std::string key = tok.substr(0, eq);
                std::string val = tok.substr(eq + 1);
                trim(key); trim(val);
                try {
                    if (key == "threshold") {
                        // sd-cli branches on mode here. We match.
                        if (mode_id == 1 || mode_id == 2) {  // easycache, ucache
                            req->reuse_threshold = std::stof(val);
                        } else {
                            req->residual_diff_threshold = std::stof(val);
                        }
                    } else if (key == "start") {
                        req->start_percent = std::stof(val);
                    } else if (key == "end") {
                        req->end_percent = std::stof(val);
                    } else if (key == "decay") {
                        req->error_decay_rate = std::stof(val);
                    } else if (key == "relative") {
                        req->use_relative_threshold = (std::stof(val) != 0.0f) ? 1 : 0;
                    } else if (key == "reset") {
                        req->reset_error_on_compute = (std::stof(val) != 0.0f) ? 1 : 0;
                    } else if (key == "Fn" || key == "fn") {
                        req->Fn_compute_blocks = std::stoi(val);
                    } else if (key == "Bn" || key == "bn") {
                        req->Bn_compute_blocks = std::stoi(val);
                    } else if (key == "warmup") {
                        if (mode_id == 6) {  // spectrum
                            req->spectrum_warmup_steps = std::stoi(val);
                        } else {
                            req->max_warmup_steps = std::stoi(val);
                        }
                    } else if (key == "w")      req->spectrum_w             = std::stof(val);
                    else if (key == "m")        req->spectrum_m             = std::stoi(val);
                    else if (key == "lam")      req->spectrum_lam           = std::stof(val);
                    else if (key == "window")   req->spectrum_window_size   = std::stoi(val);
                    else if (key == "flex")     req->spectrum_flex_window   = std::stof(val);
                    else if (key == "stop")     req->spectrum_stop_percent  = std::stof(val);
                    else {
                        fail(ExitCode::BadInput,
                             "unknown --cache-option key: '" + key + "' "
                             "(accepted: threshold/start/end/decay/relative/reset/Fn/Bn/warmup"
                             "/w/m/lam/window/flex/stop)");
                    }
                } catch (const ChimeraError &) {
                    throw;
                } catch (const std::exception &) {
                    fail(ExitCode::BadInput,
                         "--cache-option: invalid value '" + val + "' for key '" + key + "'");
                }
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }

    req->scm_mask = scm_mask;
    if (!scm_policy.empty()) {
        if      (scm_policy == "static")  req->scm_policy_dynamic = 0;
        else if (scm_policy == "dynamic") req->scm_policy_dynamic = 1;
        else {
            fail(ExitCode::BadInput,
                 "unknown --scm-policy value: " + scm_policy +
                 " (expected 'static' or 'dynamic')");
        }
    }
}

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
    ctx_params.high_noise_diffusion_model_path = cstr(params.high_noise_diffusion_model);
    ctx_params.vae_path              = cstr(params.vae);
    ctx_params.clip_l_path           = cstr(params.clip_l);
    ctx_params.clip_g_path           = cstr(params.clip_g);
    ctx_params.t5xxl_path            = cstr(params.t5xxl);
    ctx_params.llm_path              = cstr(params.llm);
    ctx_params.control_net_path      = cstr(params.control_net);
    ctx_params.taesd_path            = cstr(params.taesd);
    ctx_params.clip_vision_path      = cstr(params.clip_vision);
    ctx_params.llm_vision_path       = cstr(params.llm_vision);
    ctx_params.tensor_type_rules     = cstr(params.tensor_type_rules);
    ctx_params.photo_maker_path      = cstr(params.photo_maker);
    ctx_params.n_threads             = params.threads;
    ctx_params.enable_mmap           = params.enable_mmap;
    ctx_params.vae_decode_only       = params.vae_decode_only;
    ctx_params.offload_params_to_cpu = params.offload_to_cpu;
    ctx_params.diffusion_flash_attn  = params.diffusion_flash_attn;
    ctx_params.diffusion_conv_direct = params.diffusion_conv_direct;
    ctx_params.vae_conv_direct       = params.vae_conv_direct;
    ctx_params.flash_attn               = params.flash_attn;
    ctx_params.keep_clip_on_cpu         = params.keep_clip_on_cpu;
    ctx_params.keep_vae_on_cpu          = params.keep_vae_on_cpu;
    ctx_params.keep_control_net_on_cpu  = params.keep_control_net_on_cpu;
    ctx_params.force_sdxl_vae_conv_scale = params.force_sdxl_vae_conv_scale;
    if (params.max_vram > 0.0f) {
        ctx_params.max_vram = params.max_vram;
    }
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
    if (!params.prediction.empty()) {
        const prediction_t pt = str_to_prediction(params.prediction.c_str());
        if (pt == PREDICTION_COUNT) {
            fail(ExitCode::BadInput,
                 "unknown --prediction value: " + params.prediction +
                 " (expected eps, v, edm_v, flow, flux_flow, or flux2_flow)");
        }
        ctx_params.prediction = pt;
    }
    if (!params.lora_apply_mode.empty()) {
        const lora_apply_mode_t lm = str_to_lora_apply_mode(params.lora_apply_mode.c_str());
        if (lm == LORA_APPLY_MODE_COUNT) {
            fail(ExitCode::BadInput,
                 "unknown --lora-apply-mode value: " + params.lora_apply_mode +
                 " (expected auto, immediately, or at_runtime)");
        }
        ctx_params.lora_apply_mode = lm;
    }

    // Textual-inversion embedding directory. Scan non-recursively for
    // .gguf / .safetensors / .pt files, deriving each token name from
    // the filename stem (same convention as sd-cli). The pair vector
    // owns the backing strings; the sd_embedding_t vector borrows
    // pointers from it. Both live for the rest of this function — sd's
    // new_sd_ctx (called below) copies the strings into its own
    // std::map before returning, so stack lifetime is sufficient.
    std::vector<std::pair<std::string, std::string>> embd_kv;
    std::vector<sd_embedding_t>                      embd_vec;
    if (!params.embd_dir.empty()) {
        namespace fs = std::filesystem;
        const fs::path dir = params.embd_dir;
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) {
            fail(ExitCode::BadInput,
                 "--embd-dir is not a directory: " + params.embd_dir);
        }
        std::vector<fs::path> files;
        for (const auto & ent : fs::directory_iterator(dir, ec)) {
            if (!ent.is_regular_file()) continue;
            const std::string ext = ent.path().extension().string();
            if (ext == ".gguf" || ext == ".safetensors" || ext == ".pt") {
                files.push_back(ent.path());
            }
        }
        std::sort(files.begin(), files.end());
        embd_kv.reserve(files.size());
        embd_vec.reserve(files.size());
        for (const auto & p : files) {
            embd_kv.emplace_back(p.stem().string(), p.string());
        }
        // Build the sd_embedding_t vector AFTER embd_kv is fully sized
        // — emplace_back can reallocate and invalidate pointers if we
        // built both vectors in lockstep.
        for (const auto & kv : embd_kv) {
            sd_embedding_t e{};
            e.name = kv.first.c_str();
            e.path = kv.second.c_str();
            embd_vec.push_back(e);
        }
        ctx_params.embeddings      = embd_vec.data();
        ctx_params.embedding_count = static_cast<uint32_t>(embd_vec.size());
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
    if (req.img_cfg_scale >= 0.0f) gp.sample_params.guidance.img_cfg = req.img_cfg_scale;
    if (req.eta            >= 0.0f) gp.sample_params.eta             = req.eta;
    if (req.shifted_timestep  > 0)  gp.sample_params.shifted_timestep = req.shifted_timestep;
    if (!req.custom_sigmas.empty()) {
        // Borrowed for the duration of generate_image; request outlives this call.
        gp.sample_params.custom_sigmas       =
            const_cast<float *>(req.custom_sigmas.data());
        gp.sample_params.custom_sigmas_count =
            static_cast<int>(req.custom_sigmas.size());
    }
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

    // PhotoMaker. id_images borrows from req.pm_id_images for the
    // duration of generate_image; keep the sd_image_t vector on this
    // stack frame so it outlives the call. Empty pm_id_images disables
    // PM regardless of id_embed_path / style_strength.
    std::vector<sd_image_t> pm_id_imgs_sd;
    if (!req.pm_id_images.empty()) {
        pm_id_imgs_sd.reserve(req.pm_id_images.size());
        for (const auto & img : req.pm_id_images) {
            pm_id_imgs_sd.push_back(pixel_image_to_sd(img));
        }
        gp.pm_params.id_images       = pm_id_imgs_sd.data();
        gp.pm_params.id_images_count = static_cast<int>(pm_id_imgs_sd.size());
        if (!req.pm_id_embed_path.empty()) {
            gp.pm_params.id_embed_path = req.pm_id_embed_path.c_str();
        }
        if (req.pm_style_strength >= 0.0f) {
            gp.pm_params.style_strength = req.pm_style_strength;
        }
    }

    // Reference images. Same lifetime contract as PM above.
    std::vector<sd_image_t> ref_imgs_sd;
    if (!req.ref_images.empty()) {
        ref_imgs_sd.reserve(req.ref_images.size());
        for (const auto & img : req.ref_images) {
            ref_imgs_sd.push_back(pixel_image_to_sd(img));
        }
        gp.ref_images       = ref_imgs_sd.data();
        gp.ref_images_count = static_cast<int>(ref_imgs_sd.size());
    }
    gp.increase_ref_index    = req.increase_ref_index;
    // sd defaults auto_resize_ref_image to true; CLI flag flips it off.
    gp.auto_resize_ref_image = !req.disable_auto_resize_ref_image;

    if (!req.skip_layers.empty()) {
        // sd_slg_params_t.layers borrows the int* buffer from the
        // request for the duration of generate_image; the request
        // outlives this call so this is safe.
        gp.sample_params.guidance.slg.layers      =
            const_cast<int *>(req.skip_layers.data());
        gp.sample_params.guidance.slg.layer_count = req.skip_layers.size();
        if (req.slg_scale         >= 0.0f) gp.sample_params.guidance.slg.scale       = req.slg_scale;
        if (req.skip_layer_start  >= 0.0f) gp.sample_params.guidance.slg.layer_start = req.skip_layer_start;
        if (req.skip_layer_end    >= 0.0f) gp.sample_params.guidance.slg.layer_end   = req.skip_layer_end;
    }

    // Cache / SCM. sd_img_gen_params_init has already populated
    // gp.cache via sd_cache_params_init; we override only the fields
    // the caller explicitly set. cache_mode_id < 0 means "leave the
    // default", which is SD_CACHE_DISABLED.
    if (req.cache_mode_id >= 0) {
        gp.cache.mode = static_cast<sd_cache_mode_t>(req.cache_mode_id);
    }
    if (!std::isnan(req.reuse_threshold))         gp.cache.reuse_threshold         = req.reuse_threshold;
    if (!std::isnan(req.residual_diff_threshold)) gp.cache.residual_diff_threshold = req.residual_diff_threshold;
    if (!std::isnan(req.start_percent))           gp.cache.start_percent           = req.start_percent;
    if (!std::isnan(req.end_percent))             gp.cache.end_percent             = req.end_percent;
    if (!std::isnan(req.error_decay_rate))        gp.cache.error_decay_rate        = req.error_decay_rate;
    if (req.use_relative_threshold >= 0)          gp.cache.use_relative_threshold  = (req.use_relative_threshold != 0);
    if (req.reset_error_on_compute >= 0)          gp.cache.reset_error_on_compute  = (req.reset_error_on_compute != 0);
    if (req.Fn_compute_blocks      >= 0)          gp.cache.Fn_compute_blocks       = req.Fn_compute_blocks;
    if (req.Bn_compute_blocks      >= 0)          gp.cache.Bn_compute_blocks       = req.Bn_compute_blocks;
    if (req.max_warmup_steps       >= 0)          gp.cache.max_warmup_steps        = req.max_warmup_steps;
    if (req.spectrum_warmup_steps  >= 0)          gp.cache.spectrum_warmup_steps   = req.spectrum_warmup_steps;
    if (!std::isnan(req.spectrum_w))              gp.cache.spectrum_w              = req.spectrum_w;
    if (req.spectrum_m             >= 0)          gp.cache.spectrum_m              = req.spectrum_m;
    if (!std::isnan(req.spectrum_lam))            gp.cache.spectrum_lam            = req.spectrum_lam;
    if (req.spectrum_window_size   >= 0)          gp.cache.spectrum_window_size    = req.spectrum_window_size;
    if (!std::isnan(req.spectrum_flex_window))    gp.cache.spectrum_flex_window    = req.spectrum_flex_window;
    if (!std::isnan(req.spectrum_stop_percent))   gp.cache.spectrum_stop_percent   = req.spectrum_stop_percent;
    // scm_mask is borrowed for the duration of generate_image.
    if (!req.scm_mask.empty()) gp.cache.scm_mask = req.scm_mask.c_str();
    if (req.scm_policy_dynamic >= 0) gp.cache.scm_policy_dynamic = (req.scm_policy_dynamic != 0);

    if (req.hires_enabled) {
        gp.hires.enabled = true;
        if (!req.hires_upscaler.empty()) {
            const sd_hires_upscaler_t u =
                str_to_sd_hires_upscaler(req.hires_upscaler.c_str());
            if (u == SD_HIRES_UPSCALER_COUNT) {
                fail(ExitCode::BadInput,
                     "unknown --hires-upscaler value: " + req.hires_upscaler +
                     " (expected one of None / Latent / Latent (nearest|nearest-exact|antialiased|bicubic|"
                     "bicubic antialiased) / Lanczos / Nearest / Model)");
            }
            gp.hires.upscaler = u;
        }
        // model_path is borrowed from req; request outlives generate_image.
        if (!req.hires_upscale_model.empty()) {
            gp.hires.model_path = req.hires_upscale_model.c_str();
        }
        if (req.hires_scale > 0.0f)              gp.hires.scale              = req.hires_scale;
        if (req.hires_target_width > 0)          gp.hires.target_width       = req.hires_target_width;
        if (req.hires_target_height > 0)         gp.hires.target_height      = req.hires_target_height;
        if (req.hires_steps > 0)                 gp.hires.steps              = req.hires_steps;
        if (req.hires_denoising_strength >= 0.0f)
            gp.hires.denoising_strength = req.hires_denoising_strength;
        if (req.hires_upscale_tile_size > 0)     gp.hires.upscale_tile_size  = req.hires_upscale_tile_size;
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

    // Validate the cache/SCM bundle up-front so a typo doesn't waste a
    // model load. The temp request is discarded; the real one below
    // re-parses (same input, known-good).
    {
        chimera_sd::GenerateRequest validate_only;
        chimera_sd::parse_cache_options(opts.cache_mode, opts.cache_option,
                                        opts.scm_mask,   opts.scm_policy,
                                        &validate_only);
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
    lp.high_noise_diffusion_model = opts.high_noise_diffusion_model;
    lp.control_net          = opts.control_net;
    lp.wtype                = opts.wtype;
    lp.taesd                = opts.taesd;
    lp.clip_vision          = opts.clip_vision;
    lp.llm_vision           = opts.llm_vision;
    lp.tensor_type_rules    = opts.tensor_type_rules;
    lp.photo_maker          = opts.photo_maker;
    lp.embd_dir             = opts.embd_dir;
    lp.vae_decode_only      = !need_encode;
    lp.offload_to_cpu        = opts.offload_to_cpu;
    lp.diffusion_flash_attn  = opts.diffusion_fa;
    lp.diffusion_conv_direct = opts.diffusion_conv_direct;
    lp.vae_conv_direct       = opts.vae_conv_direct;
    lp.rng_type              = opts.rng;
    lp.sampler_rng_type      = opts.sampler_rng;
    lp.threads               = opts.threads;
    lp.flash_attn                = opts.flash_attn_global;
    lp.enable_mmap               = !opts.no_mmap;
    lp.max_vram                  = opts.max_vram;
    lp.keep_clip_on_cpu          = opts.keep_clip_on_cpu;
    lp.keep_vae_on_cpu           = opts.keep_vae_on_cpu;
    lp.keep_control_net_on_cpu   = opts.keep_control_net_on_cpu;
    lp.force_sdxl_vae_conv_scale = opts.force_sdxl_vae_conv_scale;
    lp.prediction                = opts.prediction;
    lp.lora_apply_mode           = opts.lora_apply_mode;
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
    req.img_cfg_scale    = opts.img_cfg_scale;
    req.eta              = opts.eta;
    req.shifted_timestep = opts.shifted_timestep;
    // Parse --sigmas "0.1,0.2,..." into the float vector for the custom
    // schedule. Same lifetime contract as --skip-layers above; non-float
    // entries fail with BadInput so a typo doesn't silently disable the
    // override.
    if (!opts.sigmas.empty()) {
        const std::string & s = opts.sigmas;
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.front()))) tok.erase(tok.begin());
            while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.back())))  tok.pop_back();
            if (!tok.empty()) {
                try {
                    req.custom_sigmas.push_back(std::stof(tok));
                } catch (const std::exception &) {
                    fail(ExitCode::BadInput,
                         "--sigmas expects a comma-separated list of floats, got: '" + tok + "'");
                }
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
    req.control_strength = opts.control_strength;
    // Parse --skip-layers "7,8,9" into the int vector consumed by SLG.
    // Whitespace around entries is tolerated; non-integer entries fail
    // with BadInput so a typo doesn't get silently dropped.
    if (!opts.skip_layers.empty()) {
        const std::string & s = opts.skip_layers;
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            // strip whitespace
            while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.front()))) tok.erase(tok.begin());
            while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.back())))  tok.pop_back();
            if (!tok.empty()) {
                try {
                    req.skip_layers.push_back(std::stoi(tok));
                } catch (const std::exception &) {
                    fail(ExitCode::BadInput,
                         "--skip-layers expects a comma-separated list of integers, got: '" + tok + "'");
                }
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
    req.slg_scale        = opts.slg_scale;
    req.skip_layer_start = opts.skip_layer_start;
    req.skip_layer_end   = opts.skip_layer_end;
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

    // PhotoMaker ID images. Non-recursive scan of the directory in
    // sorted order (deterministic across filesystems). Anything stb_image
    // can decode is accepted; non-image entries are skipped. Empty dir
    // is an error — if the user set the flag, they intend PM to run.
    if (!opts.pm_id_images_dir.empty()) {
        namespace fs = std::filesystem;
        const fs::path dir = opts.pm_id_images_dir;
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) {
            fail(ExitCode::BadInput,
                 "--pm-id-images-dir is not a directory: " + opts.pm_id_images_dir);
        }
        std::vector<fs::path> files;
        for (const auto & ent : fs::directory_iterator(dir, ec)) {
            if (ent.is_regular_file()) files.push_back(ent.path());
        }
        std::sort(files.begin(), files.end());
        for (const auto & p : files) {
            try {
                req.pm_id_images.push_back(chimera_sd::decode_image_file(p.string(), 3));
            } catch (const ChimeraError &) {
                // Skip files stb_image rejects (README.md etc).
            }
        }
        if (req.pm_id_images.empty()) {
            fail(ExitCode::BadInput,
                 "--pm-id-images-dir contains no decodable images: " + opts.pm_id_images_dir);
        }
    }
    req.pm_id_embed_path  = opts.pm_id_embed_path;
    req.pm_style_strength = opts.pm_style_strength;

    // Reference images. Each --ref-image path is decoded to RGB and
    // borrowed into sd_img_gen_params_t.ref_images.
    for (const auto & path : opts.ref_images) {
        req.ref_images.push_back(chimera_sd::decode_image_file(path, 3));
    }
    req.increase_ref_index            = opts.increase_ref_index;
    req.disable_auto_resize_ref_image = opts.no_auto_resize_ref_image;

    // Cache / SCM validation up-front (before load_model) so a typo
    // doesn't waste a model load. Populates req's parsed-cache fields.
    chimera_sd::parse_cache_options(opts.cache_mode, opts.cache_option,
                                    opts.scm_mask,   opts.scm_policy,
                                    &req);

    req.hires_enabled            = opts.hires_fix;
    req.hires_upscaler           = opts.hires_upscaler;
    req.hires_upscale_model      = opts.hires_upscale_model;
    req.hires_target_width       = opts.hires_width;
    req.hires_target_height      = opts.hires_height;
    req.hires_scale              = opts.hires_scale;
    req.hires_steps              = opts.hires_steps;
    req.hires_denoising_strength = opts.hires_denoising_strength;
    req.hires_upscale_tile_size  = opts.hires_upscale_tile_size;

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
