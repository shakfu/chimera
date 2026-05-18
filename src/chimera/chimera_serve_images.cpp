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
    return true;
}

}  // namespace

// POST /v1/images/generations — txt2img.
server_http_context::handler_t make_image_generations_handler(
    sd_ctx_t * ctx, std::mutex & ctx_mutex) {
    return [ctx, &ctx_mutex](const server_http_req & req) -> server_http_res_ptr {
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
        return run_image_generate(ctx, ctx_mutex, std::move(sreq), fields);
    };
}

// POST /v1/images/edits — img2img + optional mask (inpaint). Multipart.
server_http_context::handler_t make_image_edits_handler(
    sd_ctx_t * ctx, std::mutex & ctx_mutex) {
    return [ctx, &ctx_mutex](const server_http_req & req) -> server_http_res_ptr {
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
        return run_image_generate(ctx, ctx_mutex, std::move(sreq), fields);
    };
}

// POST /v1/images/variations — img2img with no prompt. We pass an empty
// prompt; SD will produce variations driven by the init latent + noise.
server_http_context::handler_t make_image_variations_handler(
    sd_ctx_t * ctx, std::mutex & ctx_mutex) {
    return [ctx, &ctx_mutex](const server_http_req & req) -> server_http_res_ptr {
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
        return run_image_generate(ctx, ctx_mutex, std::move(sreq), fields);
    };
}

}  // namespace chimera_serve
