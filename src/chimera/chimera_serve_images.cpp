// chimera_serve_images.cpp — POST /v1/images/{generations,edits,variations}
// handlers (stable-diffusion.cpp behind the /v1/images surface). Bound
// from chimera_serve.cpp when --enable-image loaded a model. Extracted
// from chimera_serve.cpp.
//
// The coerce_* JSON field helpers defined here are also linked from
// chimera_serve_rag.cpp via the chimera_serve_internal.h declarations — both modalities
// need to accept multipart-string and JSON-number forms for the same
// field names.

#include "chimera_serve_internal.h"

#include <cctype>
#include <cstdint>
#include <ctime>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace chimera_serve {

// Minimal base64 encoder (standard alphabet, with padding). The OpenAI
// `b64_json` response format expects standard base64. nlohmann/json does
// not bundle base64, and pulling in a whole library for ~30 lines of
// code isn't worth it.
namespace {

std::string base64_encode(const unsigned char * data, size_t size) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string out;
    out.reserve((size + 2) / 3 * 4);

    size_t i = 0;
    while (i + 3 <= size) {
        const uint32_t triple = (uint32_t(data[i]) << 16) |
                                (uint32_t(data[i + 1]) << 8) |
                                 uint32_t(data[i + 2]);
        out.push_back(alphabet[(triple >> 18) & 0x3F]);
        out.push_back(alphabet[(triple >> 12) & 0x3F]);
        out.push_back(alphabet[(triple >>  6) & 0x3F]);
        out.push_back(alphabet[ triple        & 0x3F]);
        i += 3;
    }
    if (i < size) {
        const uint32_t a = data[i];
        const uint32_t b = (i + 1 < size) ? data[i + 1] : 0;
        const uint32_t triple = (a << 16) | (b << 8);
        out.push_back(alphabet[(triple >> 18) & 0x3F]);
        out.push_back(alphabet[(triple >> 12) & 0x3F]);
        out.push_back((i + 1 < size) ? alphabet[(triple >> 6) & 0x3F] : '=');
        out.push_back('=');
    }
    return out;
}

}  // namespace

namespace {

// Parse OpenAI's `size` field ("256x256", "512x512", "1024x1024", or
// "WxH"). Returns true on success. We accept any positive WxH; SD will
// reject mismatches against the model's expected dimensions.
bool parse_size(const std::string & s, int & w, int & h) {
    const auto x = s.find('x');
    if (x == std::string::npos || x == 0 || x + 1 >= s.size()) return false;
    try {
        w = std::stoi(s.substr(0, x));
        h = std::stoi(s.substr(x + 1));
        return w > 0 && h > 0;
    } catch (...) {
        return false;
    }
}

// Build the response body for /v1/images/* given a list of generated
// images. Matches OpenAI's shape: { "created": ..., "data": [ { "b64_json":
// "..." } ] }. Only b64_json is supported here (no URL upload backend).
json images_response(const std::vector<chimera_sd::PixelImage> & images) {
    json data_arr = json::array();
    for (const auto & img : images) {
        auto png = chimera_sd::encode_png(static_cast<uint32_t>(img.width),
                                           static_cast<uint32_t>(img.height),
                                           static_cast<uint32_t>(img.channels),
                                           img.pixels.data());
        data_arr.push_back({{ "b64_json", base64_encode(png.data(), png.size()) }});
    }
    return {
        { "created", static_cast<int64_t>(std::time(nullptr)) },
        { "data",    std::move(data_arr) },
    };
}

// Shared core: takes a parsed request, runs SD, returns the response.
// Used by all three /v1/images/* handlers — they differ only in whether
// they read an init image, a mask image, and/or a prompt.
//
// Supported request fields across all three routes:
//   prompt           required for generations + edits; ignored for variations
//   n                1-N images, default 1 (mapped to SD batch_count)
//   size             "WxH", default 512x512
//   response_format  "b64_json" (default; only supported value)
//   user             ignored (OpenAI uses this for abuse tracking; we don't)
//   model            ignored (we have one loaded)
//
// SD-specific extensions exposed via additional JSON fields:
//   steps, cfg_scale, seed, sample_method, scheduler, negative_prompt,
//   strength (edits/variations only).
std::unique_ptr<server_http_res> run_image_generate(
    sd_ctx_t *                                 ctx,
    std::mutex &                               ctx_mutex,
    chimera_sd::GenerateRequest &&             req,
    const json &                                fields) {

    auto err_res = [](int code, const std::string & msg) {
        auto res = std::make_unique<server_http_res>();
        res->status = code;
        res->data = json{{ "error", { { "message", msg }, { "code", code }, { "type", "invalid_request_error" } }}}.dump();
        return res;
    };

    const std::string fmt = fields.contains("response_format")
        ? coerce_string(fields["response_format"]) : std::string("b64_json");
    if (fmt == "url") {
        return err_res(400,
            "response_format=url is not supported (chimera serve has no static-file backend); use 'b64_json'");
    }
    if (fmt != "b64_json") {
        return err_res(400,
            "response_format must be 'b64_json' (got '" + fmt + "')");
    }

    std::vector<chimera_sd::PixelImage> images;
    {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        // Reset the SD log ring so the snapshot we attach to the error
        // body (if generate throws) is scoped to this request only.
        chimera_sd::clear_log_buffer();
        try {
            images = chimera_sd::generate(ctx, req);
        } catch (const ChimeraError & e) {
            const int code = (e.code() == ExitCode::BadInput) ? 400 : 500;
            // Pull the last few SD log lines and append them to the error
            // body. The throws from chimera_sd::generate are intentionally
            // generic ("image generation failed") because sd.cpp delivers
            // the diagnostic detail via the log channel — buft failures,
            // unsupported sampler/scheduler names, assertion strings, the
            // SDXL-Turbo cfg_scale crash text, etc. Without this tail
            // clients only see the generic message.
            std::string msg = std::string("image generation failed: ") + e.what();
            auto tail = chimera_sd::recent_log_lines(/*max_lines=*/8);
            if (!tail.empty()) {
                msg += "\nrecent SD log:\n";
                for (const auto & line : tail) {
                    msg += "  ";
                    msg += line;
                    if (msg.empty() || msg.back() != '\n') msg += '\n';
                }
            }
            return err_res(code, msg);
        }
    }

    auto res = std::make_unique<server_http_res>();
    res->status = 200;
    res->data = images_response(images).dump();
    return res;
}

// Read the OpenAI fields common to all /v1/images/* endpoints into the
// GenerateRequest. Skips prompt (caller-supplied) and init/mask (route-
// specific). Returns false + sets err if any field is malformed.
bool fill_common_image_fields(const json &                  fields,
                              chimera_sd::GenerateRequest & req,
                              std::string &                 err) {
    if (fields.contains("n")) {
        req.batch_count = coerce_int(fields["n"], 1);
    }
    if (req.batch_count < 1) { err = "n must be >= 1"; return false; }

    if (fields.contains("size")) {
        const std::string size_str = coerce_string(fields["size"]);
        if (!parse_size(size_str, req.width, req.height)) {
            err = "size must be '<W>x<H>' (got '" + size_str + "')";
            return false;
        }
    }
    if (fields.contains("negative_prompt")) req.negative_prompt = coerce_string(fields["negative_prompt"]);
    if (fields.contains("steps"))           req.steps         = coerce_int    (fields["steps"],         req.steps);
    if (fields.contains("cfg_scale"))       req.cfg_scale     = coerce_float  (fields["cfg_scale"],     req.cfg_scale);
    if (fields.contains("seed"))            req.seed          = coerce_int64  (fields["seed"],          req.seed);
    if (fields.contains("sample_method"))   req.sample_method = coerce_string (fields["sample_method"]);
    if (fields.contains("scheduler"))       req.scheduler     = coerce_string (fields["scheduler"]);
    if (fields.contains("strength"))        req.strength      = coerce_float  (fields["strength"],      req.strength);

    // Image wave 1 — per-request fields that already exist on
    // GenerateRequest. Sentinel defaults (negative / zero / empty) on
    // the struct mean omitted keys leave the upstream sd default in
    // place; we only override when the caller supplies a value.
    if (fields.contains("clip_skip"))       req.clip_skip      = coerce_int   (fields["clip_skip"],       req.clip_skip);
    if (fields.contains("guidance"))        req.guidance       = coerce_float (fields["guidance"],        req.guidance);
    if (fields.contains("flow_shift"))      req.flow_shift     = coerce_float (fields["flow_shift"],      req.flow_shift);
    if (fields.contains("img_cfg_scale"))   req.img_cfg_scale  = coerce_float (fields["img_cfg_scale"],   req.img_cfg_scale);
    if (fields.contains("eta"))             req.eta            = coerce_float (fields["eta"],             req.eta);
    if (fields.contains("timestep_shift"))  req.shifted_timestep = coerce_int (fields["timestep_shift"],  req.shifted_timestep);
    // VAE tiling. Toggle plus three numeric knobs. The toggle accepts
    // bool / number / string forms (multipart fields arrive as strings).
    if (fields.contains("vae_tiling"))           req.vae_tiling             = coerce_bool (fields["vae_tiling"],           req.vae_tiling);
    if (fields.contains("vae_tile_size"))        req.vae_tile_size          = coerce_int  (fields["vae_tile_size"],        req.vae_tile_size);
    if (fields.contains("vae_relative_tile_size"))req.vae_relative_tile_size= coerce_float(fields["vae_relative_tile_size"],req.vae_relative_tile_size);
    if (fields.contains("vae_tile_overlap"))     req.vae_tile_overlap       = coerce_float(fields["vae_tile_overlap"],     req.vae_tile_overlap);
    // Skip-layer guidance (SLG). `skip_layers` accepts either a JSON
    // array of ints (idiomatic) or a comma-separated string (mirrors
    // the CLI). Empty / absent disables SLG regardless of the scalar
    // knobs.
    if (fields.contains("skip_layers")) {
        const auto & sl = fields["skip_layers"];
        std::vector<int> out;
        if (sl.is_array()) {
            for (const auto & e : sl) {
                if (e.is_number_integer()) out.push_back(e.get<int>());
                else if (e.is_number()) out.push_back(static_cast<int>(e.get<double>()));
                else if (e.is_string()) {
                    try { out.push_back(std::stoi(e.get<std::string>())); }
                    catch (const std::exception &) {
                        err = "skip_layers contains non-integer entry: '" + e.get<std::string>() + "'";
                        return false;
                    }
                } else {
                    err = "skip_layers array entries must be integers";
                    return false;
                }
            }
        } else if (sl.is_string()) {
            const std::string s = sl.get<std::string>();
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.front()))) tok.erase(tok.begin());
                while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.back())))  tok.pop_back();
                if (!tok.empty()) {
                    try { out.push_back(std::stoi(tok)); }
                    catch (const std::exception &) {
                        err = "skip_layers contains non-integer entry: '" + tok + "'";
                        return false;
                    }
                }
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } else {
            err = "skip_layers must be an array of ints or a comma-separated string";
            return false;
        }
        req.skip_layers = std::move(out);
    }
    if (fields.contains("slg_scale"))        req.slg_scale        = coerce_float(fields["slg_scale"],        req.slg_scale);
    if (fields.contains("skip_layer_start")) req.skip_layer_start = coerce_float(fields["skip_layer_start"], req.skip_layer_start);
    if (fields.contains("skip_layer_end"))   req.skip_layer_end   = coerce_float(fields["skip_layer_end"],   req.skip_layer_end);
    // Image wave 2 — hires-fix bundle. Toggle plus the seven scalar
    // knobs that mirror sd_hires_params_t. The Latent* upscalers work
    // without any extra model load; the `Model` upscaler needs an
    // upscale model on disk + the `hires_upscale_model` path — without
    // a server-init plumb-through of LoadParams the `Model` form will
    // fail downstream in generate(). Documented in
    // docs/dev/server-api-coverage.md.
    if (fields.contains("hires"))                req.hires_enabled            = coerce_bool (fields["hires"],                req.hires_enabled);
    if (fields.contains("hires_upscaler"))       req.hires_upscaler           = coerce_string(fields["hires_upscaler"]);
    if (fields.contains("hires_upscale_model")) req.hires_upscale_model      = coerce_string(fields["hires_upscale_model"]);
    if (fields.contains("hires_width"))          req.hires_target_width       = coerce_int  (fields["hires_width"],          req.hires_target_width);
    if (fields.contains("hires_height"))         req.hires_target_height      = coerce_int  (fields["hires_height"],         req.hires_target_height);
    if (fields.contains("hires_scale"))          req.hires_scale              = coerce_float(fields["hires_scale"],          req.hires_scale);
    if (fields.contains("hires_steps"))          req.hires_steps              = coerce_int  (fields["hires_steps"],          req.hires_steps);
    if (fields.contains("hires_denoising_strength")) req.hires_denoising_strength = coerce_float(fields["hires_denoising_strength"], req.hires_denoising_strength);
    if (fields.contains("hires_upscale_tile_size")) req.hires_upscale_tile_size = coerce_int(fields["hires_upscale_tile_size"], req.hires_upscale_tile_size);

    // Cache / SCM bundle. The four-field surface mirrors sd-cli: mode
    // picks the algorithm, option overrides per-mode tunables in
    // KEY=VALUE,... form, scm_mask/scm_policy steer sampler-cached-memory.
    // chimera_sd::parse_cache_options does the validation up-front so
    // a bad cache_option entry fails here with a precise 400 rather
    // than producing silent default behavior.
    {
        const std::string cache_mode   = fields.contains("cache_mode")   ? coerce_string(fields["cache_mode"])   : std::string();
        const std::string cache_option = fields.contains("cache_option") ? coerce_string(fields["cache_option"]) : std::string();
        const std::string scm_mask     = fields.contains("scm_mask")     ? coerce_string(fields["scm_mask"])     : std::string();
        const std::string scm_policy   = fields.contains("scm_policy")   ? coerce_string(fields["scm_policy"])   : std::string();
        if (!cache_mode.empty() || !cache_option.empty() ||
            !scm_mask.empty()   || !scm_policy.empty()) {
            try {
                chimera_sd::parse_cache_options(cache_mode, cache_option, scm_mask, scm_policy, &req);
            } catch (const ChimeraError & e) {
                err = e.what();
                return false;
            }
        }
    }

    // Custom sigma schedule. Same array-or-CSV shape as skip_layers.
    if (fields.contains("sigmas")) {
        const auto & sg = fields["sigmas"];
        std::vector<float> out;
        if (sg.is_array()) {
            for (const auto & e : sg) {
                if (e.is_number()) out.push_back(static_cast<float>(e.get<double>()));
                else if (e.is_string()) {
                    try { out.push_back(std::stof(e.get<std::string>())); }
                    catch (const std::exception &) {
                        err = "sigmas contains non-float entry: '" + e.get<std::string>() + "'";
                        return false;
                    }
                } else {
                    err = "sigmas array entries must be numbers";
                    return false;
                }
            }
        } else if (sg.is_string()) {
            const std::string s = sg.get<std::string>();
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.front()))) tok.erase(tok.begin());
                while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.back())))  tok.pop_back();
                if (!tok.empty()) {
                    try { out.push_back(std::stof(tok)); }
                    catch (const std::exception &) {
                        err = "sigmas contains non-float entry: '" + tok + "'";
                        return false;
                    }
                }
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } else {
            err = "sigmas must be an array of numbers or a comma-separated string";
            return false;
        }
        req.custom_sigmas = std::move(out);
    }
    return true;
}

}  // namespace

// POST /v1/images/generations — txt2img.
// Optional per-request ControlNet conditioning. Multipart file
// `control_image` + JSON field `control_strength`. Gated on the server
// having been started with `--sd-control-net <path>` — a request that
// supplies `control_image` without a ControlNet loaded returns 400
// with the same gating shape used for audio VAD.
//
// Decodes the conditioning image to 3-channel RGB. Dimension matching
// against the request's width/height is enforced inside
// chimera_sd::generate() so callers get a uniform error path.
//
// Returns nullptr on success; otherwise an HTTP error response that
// the caller should propagate.
// Decode a base64 string (with or without a `data:<mime>;base64,` URI
// prefix) into raw bytes. Returns false + sets `err` on any non-base64
// character (after the optional prefix), letting the caller surface a
// precise 400 that names the index of the offending image.
bool base64_decode(const std::string & input, std::vector<uint8_t> & out, std::string & err) {
    out.clear();
    static constexpr int8_t T[256] = {
        // Built ASCII-by-ASCII to keep the constant table self-checking.
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, // 0x00
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, // 0x10
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63, // 0x20 — '+' = 62, '/' = 63
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1, // 0x30 — '0'..'9' = 52..61, '=' = -2 sentinel
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14, // 0x40 — 'A'..'O'
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1, // 0x50 — 'P'..'Z'
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40, // 0x60 — 'a'..'o'
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1, // 0x70 — 'p'..'z'
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    };
    // Strip optional data-URI prefix `data:<anything>;base64,`.
    size_t start = 0;
    if (input.rfind("data:", 0) == 0) {
        const size_t comma = input.find(',', 5);
        if (comma == std::string::npos) {
            err = "base64 data URI is missing the ',' separator";
            return false;
        }
        start = comma + 1;
    }
    uint32_t buf  = 0;
    int      bits = 0;
    for (size_t i = start; i < input.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(input[i]);
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        const int8_t v = T[c];
        if (v == -2) break;  // hit '=' padding; stop
        if (v < 0) {
            err = "invalid base64 character at offset " + std::to_string(i - start);
            return false;
        }
        buf = (buf << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buf >> bits) & 0xff));
        }
    }
    return true;
}

// Optional per-request PhotoMaker. Two shapes accepted:
//   pm_id_images: [base64-data-uri-or-raw, ...]   (option C)
//   pm_id_image_set: "<name>"                     (option E)
// pm_id_images wins if both are present (explicit per-request beats
// admin-curated default — same precedence as other override pairs in
// chimera). pm_style_strength is optional in both cases; if not given,
// the upstream sd default applies. The server-init default embed path
// (--sd-pm-id-embed-path) is injected automatically when the request
// triggers PM and no per-request override is given.
//
// Gating:
//   - PM not loaded (no --sd-photo-maker)         → 400
//   - pm_id_image_set without --sd-pm-id-dir      → 400
//   - pm_id_image_set names an unknown subdir     → 400 (listing known names)
//   - pm_id_images is not an array                → 400
//   - pm_id_images is empty after decode          → 400
//   - any base64 element fails to decode/load     → 400 (naming the index)
//
// Returns nullptr on success (PM not requested, or attached cleanly);
// otherwise an HTTP error response the caller should propagate.
server_http_res_ptr maybe_attach_pm(
    const json &                  fields,
    const PmServeState &          pm,
    chimera_sd::GenerateRequest & sreq) {
    auto err_res = [](int code, const std::string & msg) {
        auto r = std::make_unique<server_http_res>();
        r->status = code;
        r->data = json{{ "error", { { "message", msg }, { "code", code }, { "type", "invalid_request_error" } }}}.dump();
        return r;
    };
    const bool has_imgs = fields.contains("pm_id_images");
    const bool has_set  = fields.contains("pm_id_image_set");
    const bool has_pm_request = has_imgs || has_set;
    if (!has_pm_request) return nullptr;
    if (!pm.model_loaded) {
        return err_res(400,
            "PhotoMaker request fields supplied (pm_id_images / pm_id_image_set) "
            "but the server has no PhotoMaker model loaded; restart chimera serve "
            "with --sd-photo-maker <path> to enable per-request PhotoMaker");
    }
    // Option C: explicit base64 array — wins over option E if both present.
    if (has_imgs) {
        const auto & arr = fields["pm_id_images"];
        if (!arr.is_array()) {
            return err_res(400, "pm_id_images must be a JSON array of base64 image strings");
        }
        std::vector<chimera_sd::PixelImage> imgs;
        imgs.reserve(arr.size());
        for (size_t i = 0; i < arr.size(); ++i) {
            if (!arr[i].is_string()) {
                return err_res(400, "pm_id_images[" + std::to_string(i) + "] is not a string");
            }
            std::vector<uint8_t> bytes;
            std::string err;
            if (!base64_decode(arr[i].get<std::string>(), bytes, err)) {
                return err_res(400, "pm_id_images[" + std::to_string(i) + "]: " + err);
            }
            try {
                imgs.push_back(chimera_sd::decode_image_bytes(bytes.data(), bytes.size(), 3));
            } catch (const ChimeraError & e) {
                return err_res(415,
                    "pm_id_images[" + std::to_string(i) + "] could not be decoded as an image: " + e.what());
            }
        }
        if (imgs.empty()) {
            return err_res(400, "pm_id_images was supplied but contains zero images");
        }
        sreq.pm_id_images = std::move(imgs);
    } else {
        // Option E: named set from --sd-pm-id-dir.
        if (pm.id_sets == nullptr || pm.id_sets->empty()) {
            return err_res(400,
                "pm_id_image_set was supplied but the server has no PhotoMaker "
                "identity directory configured (--sd-pm-id-dir)");
        }
        const std::string name = coerce_string(fields["pm_id_image_set"]);
        const auto it = pm.id_sets->find(name);
        if (it == pm.id_sets->end()) {
            std::string known;
            for (const auto & kv : *pm.id_sets) {
                if (!known.empty()) known += ", ";
                known += kv.first;
            }
            return err_res(400,
                "pm_id_image_set '" + name + "' not found (known sets: " + known + ")");
        }
        // Copy the cached PixelImages into the request. The cache lives
        // for the server's lifetime; the request lives for one call. We
        // copy rather than borrow so that GenerateRequest's owning
        // `std::vector<PixelImage>` shape is preserved (matches the CLI
        // call path). The byte cost is per-request but bounded by the
        // identity set's pixel-buffer total (~8 MB for ten 512² RGB
        // crops — fine for a server doing identity work).
        sreq.pm_id_images = it->second;
    }
    if (fields.contains("pm_style_strength")) {
        sreq.pm_style_strength = coerce_float(fields["pm_style_strength"], sreq.pm_style_strength);
    }
    sreq.pm_id_embed_path = pm.default_id_embed_path;  // server-init default
    return nullptr;
}

server_http_res_ptr maybe_attach_control(
    const server_http_req &       req,
    const json &                  fields,
    bool                          control_net_loaded,
    chimera_sd::GenerateRequest & sreq) {
    auto err_res = [](int code, const std::string & msg) {
        auto r = std::make_unique<server_http_res>();
        r->status = code;
        r->data = json{{ "error", { { "message", msg }, { "code", code }, { "type", "invalid_request_error" } }}}.dump();
        return r;
    };
    auto file_it = req.files.find("control_image");
    const bool has_control_image =
        file_it != req.files.end() && !file_it->second.data.empty();
    if (!has_control_image) return nullptr;
    if (!control_net_loaded) {
        return err_res(400,
            "control_image was supplied but the server has no ControlNet "
            "loaded; restart chimera serve with --sd-control-net <path> "
            "to enable per-request ControlNet conditioning");
    }
    try {
        sreq.control = chimera_sd::decode_image_bytes(
            file_it->second.data.data(), file_it->second.data.size(), 3);
    } catch (const ChimeraError & e) {
        return err_res(415, std::string("could not decode control_image: ") + e.what());
    }
    if (fields.contains("control_strength")) {
        sreq.control_strength = coerce_float(fields["control_strength"], sreq.control_strength);
    }
    return nullptr;
}

// Step 6: per-request LoRA selection via named aliases. Reads
// `loras: [{"name":"foo","scale":0.7}, ...]` from the JSON body,
// looks up each name in the alias map registered via `--sd-lora`,
// and populates GenerateRequest::loras with {path, scale} entries.
//
// Returns nullptr on success (no `loras` field present, or all names
// resolved cleanly); otherwise an HTTP error response the caller
// should propagate.
//
// Gating shape mirrors the other maybe_attach_* helpers — opt-in
// server-init via --sd-lora, precise 400s when a request asks for
// what isn't registered. Specifically:
//   - `loras` against a server with no --sd-lora registered → 400
//     pointing at the missing flag
//   - `loras` is not an array                                → 400
//   - an array element isn't an object                       → 400
//   - the element's `name` is missing / not a string         → 400
//   - the name doesn't match any registered alias            → 400
//     (listing the known names so the client can self-correct)
//   - `scale` is present but not a number                    → 400
//
// Raw filesystem paths (`{"path": "..."}` instead of `{"name": "..."}`)
// are intentionally rejected — closed-set is the safer default. If a
// future need for path-mode emerges, a `--sd-allow-lora-paths` flag
// would gate it; today's behavior is "names only".
server_http_res_ptr maybe_attach_loras(
    const json &                  fields,
    const LoraAliasMap *          aliases,
    chimera_sd::GenerateRequest & sreq) {
    auto err_res = [](int code, const std::string & msg) {
        auto r = std::make_unique<server_http_res>();
        r->status = code;
        r->data = json{{ "error", { { "message", msg }, { "code", code }, { "type", "invalid_request_error" } }}}.dump();
        return r;
    };
    if (!fields.contains("loras")) return nullptr;
    const auto & arr = fields["loras"];
    if (!arr.is_array()) {
        return err_res(400, "loras must be a JSON array of {name, scale} objects");
    }
    if (arr.empty()) return nullptr;  // empty array is a no-op, not an error
    if (aliases == nullptr || aliases->empty()) {
        return err_res(400,
            "loras was supplied but the server has no LoRA aliases registered; "
            "restart chimera serve with --sd-lora <name>=<path> (repeatable) "
            "to enable per-request LoRA selection");
    }
    sreq.loras.reserve(sreq.loras.size() + arr.size());
    for (size_t i = 0; i < arr.size(); ++i) {
        const auto & entry = arr[i];
        if (!entry.is_object()) {
            return err_res(400,
                "loras[" + std::to_string(i) + "] must be an object with `name` and optional `scale`");
        }
        if (!entry.contains("name") || !entry["name"].is_string()) {
            return err_res(400,
                "loras[" + std::to_string(i) + "] is missing a string `name`");
        }
        const std::string name = entry["name"].get<std::string>();
        const auto it = aliases->find(name);
        if (it == aliases->end()) {
            std::string known;
            for (const auto & kv : *aliases) {
                if (!known.empty()) known += ", ";
                known += kv.first;
            }
            return err_res(400,
                "loras[" + std::to_string(i) + "].name '" + name +
                "' is not registered (known aliases: " + known + ")");
        }
        chimera_sd::LoraEntry le;
        le.path  = it->second;
        le.scale = 1.0f;
        if (entry.contains("scale")) {
            const auto & sc = entry["scale"];
            if (!sc.is_number()) {
                return err_res(400,
                    "loras[" + std::to_string(i) + "].scale must be a number");
            }
            le.scale = sc.get<float>();
        }
        sreq.loras.push_back(std::move(le));
    }
    return nullptr;
}

server_http_context::handler_t make_image_generations_handler(
    sd_ctx_t * ctx, std::mutex & ctx_mutex,
    bool control_net_loaded, PmServeState pm, const LoraAliasMap * lora_aliases) {
    return [ctx, &ctx_mutex, control_net_loaded, pm, lora_aliases](const server_http_req & req) -> server_http_res_ptr {
        json fields = json::object();
        if (!req.body.empty()) {
            try { fields = json::parse(req.body); }
            catch (const std::exception & e) {
                auto res = std::make_unique<server_http_res>();
                res->status = 400;
                res->data = json{{ "error", { { "message", std::string("invalid JSON body: ") + e.what() }, { "code", 400 }}}}.dump();
                return res;
            }
        }
        chimera_sd::GenerateRequest sreq;
        sreq.prompt = fields.contains("prompt") ? coerce_string(fields["prompt"]) : "";
        if (sreq.prompt.empty()) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = json{{ "error", { { "message", "prompt is required" }, { "code", 400 }}}}.dump();
            return res;
        }
        std::string err;
        if (!fill_common_image_fields(fields, sreq, err)) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = json{{ "error", { { "message", err }, { "code", 400 }}}}.dump();
            return res;
        }
        if (auto e = maybe_attach_control(req, fields, control_net_loaded, sreq)) {
            return e;
        }
        if (auto e = maybe_attach_pm(fields, pm, sreq)) {
            return e;
        }
        if (auto e = maybe_attach_loras(fields, lora_aliases, sreq)) {
            return e;
        }
        return run_image_generate(ctx, ctx_mutex, std::move(sreq), fields);
    };
}

// POST /v1/images/edits — img2img + optional mask (inpaint). Multipart.
server_http_context::handler_t make_image_edits_handler(
    sd_ctx_t * ctx, std::mutex & ctx_mutex,
    bool control_net_loaded, PmServeState pm, const LoraAliasMap * lora_aliases) {
    return [ctx, &ctx_mutex, control_net_loaded, pm, lora_aliases](const server_http_req & req) -> server_http_res_ptr {
        auto err_res = [](int code, const std::string & msg) {
            auto r = std::make_unique<server_http_res>();
            r->status = code;
            r->data = json{{ "error", { { "message", msg }, { "code", code }}}}.dump();
            return r;
        };
        // server-http folds multipart text fields into req.body (JSON).
        json fields = json::object();
        if (!req.body.empty()) {
            try { fields = json::parse(req.body); }
            catch (...) { fields = json::object(); }
        }
        auto image_it = req.files.find("image");
        if (image_it == req.files.end() || image_it->second.data.empty()) {
            return err_res(400, "missing 'image' field in multipart form");
        }
        chimera_sd::GenerateRequest sreq;
        sreq.prompt = fields.contains("prompt") ? coerce_string(fields["prompt"]) : "";
        if (sreq.prompt.empty()) {
            return err_res(400, "prompt is required for /v1/images/edits");
        }
        std::string err;
        if (!fill_common_image_fields(fields, sreq, err)) {
            return err_res(400, err);
        }
        try {
            sreq.init = chimera_sd::decode_image_bytes(
                image_it->second.data.data(), image_it->second.data.size(), 3);
        } catch (const ChimeraError & e) {
            return err_res(415, std::string("could not decode init image: ") + e.what());
        }
        // If size wasn't specified, fall back to the init image's dims.
        if (!fields.contains("size")) {
            sreq.width  = sreq.init.width;
            sreq.height = sreq.init.height;
        }
        auto mask_it = req.files.find("mask");
        if (mask_it != req.files.end() && !mask_it->second.data.empty()) {
            try {
                sreq.mask = chimera_sd::decode_image_bytes(
                    mask_it->second.data.data(), mask_it->second.data.size(), 1);
            } catch (const ChimeraError & e) {
                return err_res(415, std::string("could not decode mask image: ") + e.what());
            }
        }
        if (auto e = maybe_attach_control(req, fields, control_net_loaded, sreq)) {
            return e;
        }
        if (auto e = maybe_attach_pm(fields, pm, sreq)) {
            return e;
        }
        if (auto e = maybe_attach_loras(fields, lora_aliases, sreq)) {
            return e;
        }
        return run_image_generate(ctx, ctx_mutex, std::move(sreq), fields);
    };
}

// POST /v1/images/variations — img2img with no prompt. We pass an empty
// prompt; SD will produce variations driven by the init latent + noise.
server_http_context::handler_t make_image_variations_handler(
    sd_ctx_t * ctx, std::mutex & ctx_mutex,
    bool control_net_loaded, PmServeState pm, const LoraAliasMap * lora_aliases) {
    return [ctx, &ctx_mutex, control_net_loaded, pm, lora_aliases](const server_http_req & req) -> server_http_res_ptr {
        auto err_res = [](int code, const std::string & msg) {
            auto r = std::make_unique<server_http_res>();
            r->status = code;
            r->data = json{{ "error", { { "message", msg }, { "code", code }}}}.dump();
            return r;
        };
        json fields = json::object();
        if (!req.body.empty()) {
            try { fields = json::parse(req.body); }
            catch (...) { fields = json::object(); }
        }
        auto image_it = req.files.find("image");
        if (image_it == req.files.end() || image_it->second.data.empty()) {
            return err_res(400, "missing 'image' field in multipart form");
        }
        chimera_sd::GenerateRequest sreq;
        sreq.prompt = "";  // variations: no prompt
        std::string err;
        if (!fill_common_image_fields(fields, sreq, err)) {
            return err_res(400, err);
        }
        try {
            sreq.init = chimera_sd::decode_image_bytes(
                image_it->second.data.data(), image_it->second.data.size(), 3);
        } catch (const ChimeraError & e) {
            return err_res(415, std::string("could not decode source image: ") + e.what());
        }
        if (!fields.contains("size")) {
            sreq.width  = sreq.init.width;
            sreq.height = sreq.init.height;
        }
        if (auto e = maybe_attach_control(req, fields, control_net_loaded, sreq)) {
            return e;
        }
        if (auto e = maybe_attach_pm(fields, pm, sreq)) {
            return e;
        }
        if (auto e = maybe_attach_loras(fields, lora_aliases, sreq)) {
            return e;
        }
        return run_image_generate(ctx, ctx_mutex, std::move(sreq), fields);
    };
}

}  // namespace chimera_serve
