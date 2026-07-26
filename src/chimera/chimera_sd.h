// chimera_sd.h — internal public API for stable-diffusion.cpp integration.
//
// Consumed by both `command_sd` (the CLI subcommand) and `chimera serve`
// (the HTTP `/v1/images/*` routes). The contract here is what those two
// callers may use; anything else stays private to chimera_sd.cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

struct sd_ctx_t;

struct SdContextDeleter {
    void operator()(sd_ctx_t * ctx) const;
};
using SdContextPtr = std::unique_ptr<sd_ctx_t, SdContextDeleter>;

namespace chimera_sd {

// Raw 8-bit pixel block. Used for both inputs (init/mask uploads) and
// outputs (the generated images we'll PNG-encode for the HTTP response).
struct PixelImage {
    int                       width    = 0;
    int                       height   = 0;
    int                       channels = 0;  // 3 = RGB, 1 = grayscale
    std::vector<unsigned char> pixels;       // size == width*height*channels
};

// LoRA adapter request entry. `path` is the resolved on-disk file; the
// chimera_sd::generate() implementation borrows the string for the
// duration of the call (sd_lora_t.path is const char*), so callers must
// keep the GenerateRequest alive across the call.
struct LoraEntry {
    std::string path;
    float       scale = 1.0f;
};

struct GenerateRequest {
    std::string prompt;
    std::string negative_prompt;
    int  width        = 512;
    int  height       = 512;
    int  steps        = 20;
    int  batch_count  = 1;
    int  clip_skip    = -1;
    int  threads      = -1;
    int64_t seed      = -1;
    float cfg_scale   = 7.0f;
    float strength    = 0.75f;  // only used when init is set (img2img)
    float guidance    = -1.0f;  // distilled guidance (Flux/SD3); -1 = upstream default
    float flow_shift  = -1.0f;  // Flux/SD3 timestep shift;       -1 = upstream default
    float img_cfg_scale    = -1.0f; // sd_guidance_params_t.img_cfg; -1 = leave INFINITY (sd falls back to txt_cfg)
    float eta              = -1.0f; // sd_sample_params_t.eta;       -1 = leave INFINITY
    int   shifted_timestep = 0;     // sd_sample_params_t.shifted_timestep; 0 = no shift (upstream default)
    // Optional custom sigma schedule. Borrowed by generate() into
    // sd_sample_params_t.custom_sigmas for the duration of the call,
    // so it must outlive the request. Empty = no custom schedule.
    std::vector<float> custom_sigmas;
    std::string sample_method;  // empty -> SD default
    std::string scheduler;      // empty -> SD default

    // Optional img2img / inpaint inputs. init must match width/height; if
    // mask is set, init must also be set (mask without init is rejected).
    PixelImage init;   // .pixels.empty() means "skip"
    PixelImage mask;   // .pixels.empty() means "skip"

    // Optional ControlNet conditioning. Requires the context to have been
    // loaded with LoadParams::control_net set. control must match
    // width/height. control_strength is ignored when control is empty.
    PixelImage control;
    float      control_strength = 0.9f;

    // Optional skip-layer guidance (SLG). `skip_layers` is borrowed by
    // generate() into `sd_slg_params_t.layers` for the duration of the
    // call, so it must outlive the request. An empty vector disables
    // SLG regardless of the scalar fields. Negative-one sentinels for
    // the scalars leave the upstream default in place.
    std::vector<int> skip_layers;
    float            slg_scale         = -1.0f;
    float            skip_layer_start  = -1.0f;
    float            skip_layer_end    = -1.0f;

    // Optional VAE tiling (lowers peak VRAM at a small quality cost).
    // Mirrors sd_tiling_params_t. Negative-one sentinels leave the upstream
    // default in place (only `enabled` is checked unconditionally).
    bool  vae_tiling             = false;
    int   vae_tile_size          = -1;    // absolute tile size (applied to both axes)
    float vae_relative_tile_size = -1.0f; // fraction of canvas (applied to both axes)
    float vae_tile_overlap       = -1.0f; // fractional overlap

    // Optional LoRA adapters applied during generation. Each entry's path
    // must already be resolved to a file the sd context can open.
    std::vector<LoraEntry> loras;

    // Round 4 PhotoMaker. `pm_id_images` is borrowed (as sd_image_t) into
    // sd_pm_params_t.id_images for the duration of the call, so the
    // request must outlive generate(). Empty disables PM regardless of
    // the other knobs. `pm_style_strength` < 0 leaves the upstream default.
    std::vector<PixelImage> pm_id_images;
    std::string             pm_id_embed_path;
    float                   pm_style_strength = -1.0f;

    // Round 5 reference images (style/identity conditioning). Same
    // lifetime contract as the PM bundle above.
    std::vector<PixelImage> ref_images;
    bool                    increase_ref_index           = false;
    bool                    disable_auto_resize_ref_image = false;
    // Verbatim sd_img_gen_params_t::ref_image_args passthrough (comma- or
    // semicolon-separated key=value pairs; see sd's resolve_ref_image_params
    // for the accepted keys). The two booleans above are appended after this
    // string, so they override a colliding key set here.
    std::string             ref_image_args;

    // Round 6 hires-fix. Disabled by default; the scalar sentinels
    // (0 / -1) leave sd_hires_params_init's defaults in place.
    // `hires_upscaler` is empty (= default Latent) unless the user
    // picks a sd_hires_upscaler_t name (resolved by str_to_sd_hires_upscaler).
    // `hires_upscale_model` is the file path used when upscaler=Model.
    bool        hires_enabled              = false;
    std::string hires_upscaler;
    std::string hires_upscale_model;
    int         hires_target_width         = 0;
    int         hires_target_height        = 0;
    float       hires_scale                = -1.0f;
    int         hires_steps                = 0;
    float       hires_denoising_strength   = -1.0f;
    int         hires_upscale_tile_size    = 0;

    // Parsed cache/SCM params. cache_mode_id mirrors sd_cache_mode_t as
    // an int (-1 = "leave default", 0 = SD_CACHE_DISABLED, 1..6 =
    // EASYCACHE/UCACHE/DBCACHE/TAYLORSEER/CACHE_DIT/SPECTRUM). NaN /
    // -1 sentinels on the numeric fields mean "leave whichever value
    // sd_cache_params_init wrote". scm_mask is borrowed by generate()
    // into sd_cache_params_t.scm_mask for the duration of the call.
    int   cache_mode_id          = -1;
    float reuse_threshold        = std::numeric_limits<float>::quiet_NaN();
    float residual_diff_threshold = std::numeric_limits<float>::quiet_NaN();
    float start_percent          = std::numeric_limits<float>::quiet_NaN();
    float end_percent            = std::numeric_limits<float>::quiet_NaN();
    float error_decay_rate       = std::numeric_limits<float>::quiet_NaN();
    int   use_relative_threshold = -1;  // -1 unset; 0 false; 1 true
    int   reset_error_on_compute = -1;
    int   Fn_compute_blocks      = -1;
    int   Bn_compute_blocks      = -1;
    int   max_warmup_steps       = -1;
    int   spectrum_warmup_steps  = -1;
    float spectrum_w             = std::numeric_limits<float>::quiet_NaN();
    int   spectrum_m             = -1;
    float spectrum_lam           = std::numeric_limits<float>::quiet_NaN();
    int   spectrum_window_size   = -1;
    float spectrum_flex_window   = std::numeric_limits<float>::quiet_NaN();
    float spectrum_stop_percent  = std::numeric_limits<float>::quiet_NaN();
    std::string scm_mask;
    int   scm_policy_dynamic     = -1;  // -1 unset; 0 static; 1 dynamic
};

// Validate and parse the cache surface up-front so command_sd can fail
// fast (before model load). Maps cache_mode string to a sd_cache_mode_t
// id and parses cache_option's "key=value,..." string into the numeric
// fields of GenerateRequest. Throws ChimeraError(BadInput) on any
// invalid token. Empty strings are no-ops.
void parse_cache_options(const std::string & cache_mode,
                         const std::string & cache_option,
                         const std::string & scm_mask,
                         const std::string & scm_policy,
                         GenerateRequest *   req);

// ---- model lifecycle ---------------------------------------------------

// Inputs for load_model. `model` is the combined single-file checkpoint
// path used by classic SD/SDXL builds; for split layouts (Flux, Z-Image,
// SD3, ...) leave `model` empty and populate the per-component paths
// (`diffusion_model`, `vae`, and one of `clip_l`/`t5xxl`/`llm`).
// `offload_to_cpu` and `diffusion_flash_attn` map to the same-named
// sd_ctx_params_t fields and are the perf knobs Z-Image expects.
struct LoadParams {
    std::string model;
    std::string diffusion_model;
    std::string vae;
    std::string clip_l;
    std::string clip_g;           // SDXL split-checkpoint CLIP-G text encoder
    std::string t5xxl;
    std::string llm;
    std::string high_noise_diffusion_model; // optional second diffusion model for two-stage pairs
    std::string control_net;      // ControlNet model file
    std::string wtype;            // weights type override (empty = upstream default)
    std::string taesd;             // tiny-autoencoder
    std::string clip_vision;       // CLIP-Vision encoder
    std::string llm_vision;        // LLM-Vision encoder
    std::string tensor_type_rules; // per-tensor wtype override rules
    std::string photo_maker;       // PhotoMaker model
    // Textual-inversion / embedding directory. load_model scans it
    // non-recursively for .gguf / .safetensors / .pt files and registers
    // each as a sd_embedding_t { name=stem, path=full } before calling
    // new_sd_ctx. The names become the tokens addressable from prompts.
    std::string embd_dir;
    bool        vae_decode_only       = true;
    bool        offload_to_cpu        = false;
    bool        diffusion_flash_attn  = false;
    bool        diffusion_conv_direct = false;
    bool        vae_conv_direct       = false;
    std::string rng_type;          // empty = upstream default; otherwise via str_to_rng_type
    std::string sampler_rng_type;  // empty = upstream default
    int         threads              = -1;

    // Generic perf / offload (Round 1). flash_attn is the global FA
    // toggle (sd_ctx_params_t.flash_attn); diffusion_flash_attn above
    // only flips the diffusion path. enable_mmap defaults off upstream.
    // max_vram <= 0 leaves the upstream default in place.
    bool  flash_attn                = false;
    bool  enable_mmap               = true;  // chimera default; CLI's --no-mmap maps to false
    float max_vram                  = 0.0f;
    bool  keep_clip_on_cpu          = false;
    bool  keep_vae_on_cpu           = false;
    bool  keep_control_net_on_cpu   = false;
    bool  force_sdxl_vae_conv_scale = false;

    // Stream diffusion weights from CPU during generation (sd_cli's
    // --stream-layers). Only takes effect alongside max_vram > 0; sd.cpp
    // silently disables it otherwise. Default off = upstream default.
    bool  stream_layers             = false;

    // Pre-load all params into the params backend at model-load time
    // instead of lazily on first use (sd_cli's --eager-load). Trades a
    // slower load for no first-generation warmup; independent of max_vram.
    // Default off = upstream default (lazy).
    bool  eager_load                = false;

    // Enum-string knobs resolved via sd.cpp's str_to_* helpers. Empty
    // leaves the upstream default in place; unknown values exit with
    // BadInput from load_model.
    std::string prediction;       // eps | v | edm_v | flow | flux_flow | flux2_flow
    std::string lora_apply_mode;  // auto | immediately | at_runtime
    std::string vae_format;       // auto | flux | sd3 | flux2 (empty = auto)
};

// Loads a stable-diffusion model from a LoadParams. Returns an empty
// pointer on failure.
SdContextPtr load_model(const LoadParams & params);

// Back-compat single-file overload. `vae_decode_only=false` is required
// for img2img / inpaint (which need the VAE encode path); `true` saves
// memory on plain txt2img.
SdContextPtr load_model(const std::string & path,
                        bool                vae_decode_only,
                        int                 threads = -1);

// ---- image I/O ---------------------------------------------------------

// Decode a memory buffer (PNG/JPG/etc., anything stb_image accepts) into
// an 8-bit pixel image with the requested channel count (3 = RGB, 1 =
// grayscale). Throws ChimeraError(BadInput) on failure.
PixelImage decode_image_bytes(const void * data, size_t size, int channels);

// File-path convenience wrapper around decode_image_bytes.
PixelImage decode_image_file(const std::string & path, int channels);

// Encode raw RGB pixels to PNG bytes in memory. Used to produce the bytes
// that get base64-encoded into the OpenAI `b64_json` response. Throws
// ChimeraError(Runtime) on encoder failure.
std::vector<unsigned char> encode_png(uint32_t              width,
                                      uint32_t              height,
                                      uint32_t              channels,
                                      const unsigned char * pixels);

// File-path wrapper around encode_png that writes a PNG to disk.
void save_png_file(const std::string &   path,
                   uint32_t              width,
                   uint32_t              height,
                   uint32_t              channels,
                   const unsigned char * pixels);

// ---- generation --------------------------------------------------------

// Run image generation. Returns batch_count PixelImage entries in RGB
// order. Throws ChimeraError(Generate) on failure. The caller owns the
// returned vector and the pixel data inside each entry.
std::vector<PixelImage> generate(sd_ctx_t * ctx, const GenerateRequest & req);

// Compose sd_img_gen_params_t::ref_image_args from the request: the caller's
// `ref_image_args` passthrough first, then `increase_ref_index` and
// `disable_auto_resize_ref_image` appended as key=value pairs. sd applies the
// pairs left to right, so the two booleans override a colliding key in the
// passthrough string. Returns empty when nothing is set, which leaves sd's
// per-architecture preset defaults in place. Exposed for testing; generate()
// calls it and hands the result to sd.
std::string build_ref_image_args(const GenerateRequest & req);

// ---- runtime introspection (for `chimera info`) ------------------------

// Runtime stable-diffusion.cpp version string. Should match the
// `CHIMERA_SDCPP_VERSION` compile-time macro for pinned builds.
std::string sdcpp_version();

// `ggml_version()` as visible from this TU. With SD_USE_VENDORED_GGML=0
// (chimera's required mode) this is the same ggml linked by llama.cpp.
std::string sd_ggml_version();

// ---- log capture --------------------------------------------------------
//
// SD (and the ggml backend it routes through) emits diagnostic lines via
// `sd_set_log_callback`. The chimera log callback mirrors those into a
// thread-safe ring buffer in addition to stderr, so HTTP handlers can
// attach the last few lines to an error body when generation fails. The
// underlying generic "image generation failed" message is rarely
// actionable on its own — the ring buffer typically carries the
// descriptive line ("buft failed", "unsupported sample method", ...) the
// user actually needs to see.

// Snapshot the most recent log lines (newest last). max_lines caps the
// returned size; the ring itself is fixed-size internally.
std::vector<std::string> recent_log_lines(size_t max_lines = 8);

// Discard whatever the ring currently holds. Call this immediately
// before invoking generate() so the snapshot taken on catch is scoped
// to that request only.
void clear_log_buffer();

// Raw `sd_get_system_info()` output (multi-line: system info, backends,
// CPU features as reported by sd.cpp's own ggml). Caller decides how to
// render it.
std::string sd_system_info_raw();

}  // namespace chimera_sd
