// chimera_sd.h — internal public API for stable-diffusion.cpp integration.
//
// Consumed by both `command_sd` (the CLI subcommand) and `chimera serve`
// (the HTTP `/v1/images/*` routes). The contract here is what those two
// callers may use; anything else stays private to chimera_sd.cpp.
#pragma once

#include <cstddef>
#include <cstdint>
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
};

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
    bool        vae_decode_only       = true;
    bool        offload_to_cpu        = false;
    bool        diffusion_flash_attn  = false;
    bool        diffusion_conv_direct = false;
    bool        vae_conv_direct       = false;
    std::string rng_type;          // empty = upstream default; otherwise via str_to_rng_type
    std::string sampler_rng_type;  // empty = upstream default
    int         threads              = -1;
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
