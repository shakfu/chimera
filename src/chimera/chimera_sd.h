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
    std::string sample_method;  // empty -> SD default
    std::string scheduler;      // empty -> SD default

    // Optional img2img / inpaint inputs. init must match width/height; if
    // mask is set, init must also be set (mask without init is rejected).
    PixelImage init;   // .pixels.empty() means "skip"
    PixelImage mask;   // .pixels.empty() means "skip"
};

// ---- model lifecycle ---------------------------------------------------

// Loads a stable-diffusion model. `vae_decode_only=false` is required for
// img2img / inpaint (which need the VAE encode path); `true` saves memory
// on plain txt2img. Returns an empty pointer on failure.
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

// Raw `sd_get_system_info()` output (multi-line: system info, backends,
// CPU features as reported by sd.cpp's own ggml). Caller decides how to
// render it.
std::string sd_system_info_raw();

}  // namespace chimera_sd
