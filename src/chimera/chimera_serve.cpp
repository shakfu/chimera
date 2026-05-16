// chimera_serve.cpp — OpenAI-compatible HTTP server.
//
// This subcommand links llama.cpp's `server-context` STATIC library (the
// engine behind llama-server: slot scheduler, chat-template handling, mtmd
// integration, streaming SSE, tool-call parsing) and exposes a curated
// subset of its routes. The HTTP frontend is `server_http_context` from
// llama-server's `server-http.{cpp,h}`, which wraps the vendored cpp-httplib.
//
// ----------------------------------------------------------------------------
// Currently exposed routes
// ----------------------------------------------------------------------------
// LLM (text):
//   GET  /health, /v1/health            liveness probe
//   GET  /v1/models                     list loaded model + aliases
//   GET  /metrics                       Prometheus-style telemetry
//   GET  /props                         read server props (template kwargs, ...)
//   POST /chat/completions
//        + /v1/chat/completions         OpenAI Chat Completions (streaming + non-streaming SSE)
//   POST /v1/completions                OpenAI legacy text Completions
//   POST /v1/embeddings                 OpenAI Embeddings (only when --embeddings)
//   POST /v1/messages                   Anthropic Messages API compat
//   POST /v1/messages/count_tokens      Anthropic token counting
//   POST /infill                        fill-in-the-middle for code models
//   POST /tokenize, /detokenize         vocab helpers
//   POST /apply-template                render the chat template against messages
//   POST /v1/responses                  OpenAI Responses API (server-context's
//                                       built-in handler). Stateful within a
//                                       single chimera serve invocation; state
//                                       is held in-process and lost on restart.
//
// Audio (only when --enable-audio):
//   POST /v1/audio/transcriptions       (whisper.cpp; WAV-only, see handler comment)
//
// Image (only when --enable-image):
//   POST /v1/images/generations         txt2img (stable-diffusion.cpp)
//   POST /v1/images/edits               img2img + optional mask
//   POST /v1/images/variations          img2img with no prompt
//
// Vector store / RAG (only when --enable-rag):
//   GET  /v1/vector_stores              list collections
//   POST /v1/vector_stores              create a collection
//   GET  /v1/vector_stores/:name        collection stats
//   POST /v1/vector_stores/:name/delete drop a collection (POST not DELETE
//                                       because server-http only wraps
//                                       GET/POST; OpenAI SDK clients
//                                       sending DELETE will need to be
//                                       configured to use the POST path)
//   POST /v1/vector_stores/:name/files  ingest text via multipart upload
//                                       or JSON {"text": "..."} body
//   POST /v1/vector_stores/:name/search KNN search; body {"query": "...", "k": N}
//
// ----------------------------------------------------------------------------
// Phase 1 scope — what is DELIBERATELY NOT exposed yet
// ----------------------------------------------------------------------------
// Routes that exist on `server_routes` but are not bound here. Every one is a
// few lines of `ctx_http.post(...)` away if/when we decide to surface it; the
// list is explicit so the omission is a design choice, not an oversight.
//
//   POST /completion, /completions      legacy llama.cpp completion shape
//                                       (different from /v1/completions).
//                                       Practically nobody calls this in 2025.
//   POST /v1/audio/transcriptions       server_routes' built-in handler is NOT
//                                       bound. We expose this path ourselves
//                                       via whisper.cpp when --enable-audio
//                                       is passed. The upstream handler routes
//                                       audio through mtmd's audio mmproj —
//                                       a fundamentally different pipeline
//                                       (LLM-with-audio-tokens vs dedicated ASR).
//   POST /embedding, /embeddings        non-/v1 embeddings variants — redundant.
//   POST /rerank, /v1/rerank            document reranking via cross-encoder
//                                       models. Niche; bind on request.
//   GET  /slots, POST /slots/:id        slot save/load (KV cache snapshots).
//                                       Bind on request.
//   GET  /lora-adapters,
//   POST /lora-adapters                 LoRA hot-swap at request time.
//                                       Bind on request.
//   POST /props                         mutating server props at runtime
//                                       conflicts with chimera serve's
//                                       "CLI is the config" stance. Read
//                                       (GET /props) is bound; write is not.
//
// Server-mode features not wired up:
//
//   - Router/multi-model mode (`is_router_server` branch in llama-server's
//     server.cpp). Single-model only here.
//   - Built-in tool plugins (`--server-tools`). EXPERIMENTAL upstream.
//   - MCP CORS proxy (`--webui-mcp-proxy`). EXPERIMENTAL upstream.
//   - GCP / Vertex AI compat (`ctx_http.register_gcp_compat()`).
//   - Web chat UI (`public/{index.html, bundle.js, bundle.css}`). manage.py
//     passes `LLAMA_BUILD_WEBUI=OFF` to skip baking the ~11 MB asset bundle
//     into libserver-context.a.
//   - Child-server / parent-process sleeping notification.
//   - SSL / TLS (no `--ssl-cert-file` / `--ssl-key-file`). Run behind a
//     reverse proxy for HTTPS.
//
// Planned Phase 2/3 follow-ups (additive — won't refactor this file):
//   - `--enable-audio <whisper.gguf>` adds POST /v1/audio/transcriptions
//     wired to chimera's existing whisper.cpp machinery.
//   - `--enable-image <sd.gguf>` adds POST /v1/images/{generations,edits,
//     variations} wired to chimera's existing sd.cpp machinery.
//
// ----------------------------------------------------------------------------

#include "chimera.h"
#include "chimera_chat_store.h"
#include "chimera_db.h"
#include "chimera_embed.h"
#include "chimera_sd.h"
#include "chimera_vector_store.h"
#include "chimera_whisper.h"

#include "arg.h"      // common_init
#include "common.h"   // common_params
#include "log.h"

#include "server-context.h"
#include "server-http.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using json = nlohmann::ordered_json;

// Mirrors llama-server's ex_wrapper (server.cpp:40). Ensures handlers never
// throw out of the HTTP layer; converts std::invalid_argument to 400 and
// every other exception to 500. Kept private to chimera_serve so we don't
// have to forward-declare it.
server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        int status = 500;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            status = 400;
            message = e.what();
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "unknown error";
        }
        auto res = std::make_unique<server_http_res>();
        res->status = status;
        json body = { { "error", { { "message", message }, { "code", status } } } };
        res->data = body.dump();
        return res;
    };
}

// Signal handling. The shutdown_handler closes the task queue, which causes
// ctx_server.start_loop() to return on the main thread. Hitting Ctrl-C twice
// force-exits in case the loop is wedged. Same pattern as llama-server.
std::function<void(int)> g_shutdown_handler;
std::atomic_flag g_terminating = ATOMIC_FLAG_INIT;

void chimera_serve_signal_handler(int signal) {
    if (g_terminating.test_and_set()) {
        std::fprintf(stderr, "\nreceived second interrupt, exiting immediately.\n");
        std::exit(1);
    }
    if (g_shutdown_handler) {
        g_shutdown_handler(signal);
    }
}

void install_signal_handlers() {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sa{};
    sa.sa_handler = chimera_serve_signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
#elif defined(_WIN32)
    // Lambda-to-PHANDLER_ROUTINE trampoline; mirrors llama-server.
    auto win_handler = +[](unsigned long ctrl_type) -> int {
        if (ctrl_type == 0 /*CTRL_C_EVENT*/) {
            chimera_serve_signal_handler(SIGINT);
            return 1;
        }
        return 0;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(win_handler), TRUE);
#endif
}

// Populate common_params from chimera's ServeOptions. This is the only place
// where chimera's CLI surface meets llama.cpp's giant param struct; keeping
// the mapping centralized makes it easy to add a flag later without hunting
// through the file.
common_params build_common_params(const ServeOptions & opts) {
    common_params params;
    common_init();

    params.model.path           = opts.model;
    params.mmproj.path          = opts.mmproj;
    params.n_ctx                = opts.n_ctx;
    params.n_batch              = opts.n_batch;
    params.n_ubatch             = opts.n_ubatch;
    params.n_gpu_layers         = opts.gpu_layers;
    params.cpuparams.n_threads  = opts.threads;
    params.n_parallel           = opts.parallel;
    params.hostname             = opts.host;
    params.port                 = opts.port;
    params.embedding            = opts.embedding;
    // Server-context's metrics route returns 501 unless this is set.
    // Free to enable: lightweight counters, no external dep on Prometheus.
    params.endpoint_metrics     = true;
    if (!opts.api_key.empty()) {
        params.api_keys.push_back(opts.api_key);
    }

    // The next two blocks mirror llama-server's main() defensively-correct
    // setup. They're not optional: server-context asserts on the embedding
    // batch invariant, and n_parallel<0 means "auto" upstream.

    // Embeddings require all tokens in a single ubatch; if the user picked
    // mismatched values we clamp rather than fail. See llama.cpp #12836.
    if (params.embedding && params.n_batch > params.n_ubatch) {
        params.n_batch = params.n_ubatch;
    }

    if (params.n_parallel < 0) {
        params.n_parallel = 4;
        params.kv_unified = true;
    }

    // For consistency between router-mode and single-model paths upstream
    // sets the same name as alias if not provided. /v1/models reads this.
    if (params.model_alias.empty() && !params.model.name.empty()) {
        params.model_alias.insert(params.model.name);
    }

    return params;
}

// ---- Phase 2: audio transcription helpers ------------------------------

// SRT and VTT format the same segment list with a 4-byte difference (comma
// vs period before the ms field, and an opening "WEBVTT\n\n" header on
// VTT). Both are simple enough to write inline; we avoid pulling in another
// library just for time-string punctuation.

std::string srt_timestamp(int64_t t_10ms) {
    int64_t ms = t_10ms * 10;
    int64_t h = ms / 3600000; ms %= 3600000;
    int64_t m = ms / 60000;   ms %= 60000;
    int64_t s = ms / 1000;    ms %= 1000;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld,%03lld",
                  (long long)h, (long long)m, (long long)s, (long long)ms);
    return buf;
}

std::string vtt_timestamp(int64_t t_10ms) {
    int64_t ms = t_10ms * 10;
    int64_t h = ms / 3600000; ms %= 3600000;
    int64_t m = ms / 60000;   ms %= 60000;
    int64_t s = ms / 1000;    ms %= 1000;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld.%03lld",
                  (long long)h, (long long)m, (long long)s, (long long)ms);
    return buf;
}

std::string format_srt(const chimera_whisper::TranscribeResult & r) {
    std::ostringstream out;
    for (size_t i = 0; i < r.segments.size(); ++i) {
        out << (i + 1) << '\n'
            << srt_timestamp(r.segments[i].t0) << " --> "
            << srt_timestamp(r.segments[i].t1) << '\n'
            << r.segments[i].text << "\n\n";
    }
    return out.str();
}

std::string format_vtt(const chimera_whisper::TranscribeResult & r) {
    std::ostringstream out;
    out << "WEBVTT\n\n";
    for (const auto & s : r.segments) {
        out << vtt_timestamp(s.t0) << " --> "
            << vtt_timestamp(s.t1) << '\n'
            << s.text << "\n\n";
    }
    return out.str();
}

json format_verbose_json(const chimera_whisper::TranscribeResult & r,
                         const std::string & task,
                         const std::string & language) {
    json segs = json::array();
    for (size_t i = 0; i < r.segments.size(); ++i) {
        // OpenAI's spec uses fractional-seconds (double) for start/end.
        // 10ms units -> seconds via /100.0.
        segs.push_back({
            { "id",    static_cast<int>(i) },
            { "seek",  0 },
            { "start", r.segments[i].t0 / 100.0 },
            { "end",   r.segments[i].t1 / 100.0 },
            { "text",  r.segments[i].text },
        });
    }
    return {
        { "task",     task },
        { "language", language },
        { "duration", r.audio_duration_s },
        { "text",     r.text },
        { "segments", segs },
    };
}

// Build the handler bound to /v1/audio/transcriptions. Captures the
// per-server whisper context + mutex by reference; both live in command_serve
// for as long as the handler can be invoked, so the reference is safe.
//
// SUPPORTED audio formats (Phase 2): WAV (RIFF/WAVE, PCM 8/16/24/32-bit
// integer, or 32-bit float). The OpenAI spec also accepts mp3, mp4, mpeg,
// mpga, m4a, webm — those need a real decoder. Phase 2.1 will add either
// libsndfile + libavcodec or a single-header dr_mp3/dr_flac.
//
// SUPPORTED request fields: file (required), model (ignored, we have one),
// language, prompt (-> initial_prompt), response_format (json/text/
// verbose_json/srt/vtt), temperature (currently ignored — whisper doesn't
// expose a temperature knob in the same sense). timestamp_granularities is
// also ignored; segment-level timing is always returned in verbose_json.
server_http_context::handler_t make_audio_transcribe_handler(
    whisper_context * ctx,
    std::mutex      & ctx_mutex) {

    return [ctx, &ctx_mutex](const server_http_req & req) -> server_http_res_ptr {
        auto err_res = [](int code, const std::string & msg) {
            auto res = std::make_unique<server_http_res>();
            res->status = code;
            res->data = json{{ "error", { { "message", msg }, { "code", code }, { "type", "invalid_request_error" }}}}.dump();
            return res;
        };

        // Audio bytes: the multipart "file" field, as deposited by
        // server-http.cpp's MultipartFormDataMap reader.
        auto file_it = req.files.find("file");
        if (file_it == req.files.end() || file_it->second.data.empty()) {
            return err_res(400, "missing 'file' field in multipart form");
        }
        const auto & upload = file_it->second;

        // Text fields: server-http folds them into a JSON object stored in
        // req.body. Tolerate the empty / non-JSON body case (e.g. file-only
        // request) by treating it as an empty object.
        json fields = json::object();
        if (!req.body.empty()) {
            try {
                fields = json::parse(req.body);
            } catch (const std::exception &) {
                fields = json::object();
            }
        }

        const std::string fmt = fields.value("response_format", std::string("json"));
        const std::string lang = fields.value("language", std::string("auto"));
        const std::string prompt = fields.value("prompt", std::string());

        // Decode WAV. Phase 2 limitation: only RIFF/WAVE is supported; mp3,
        // m4a, webm, etc. need a follow-up decoder. The error message names
        // the limitation so callers don't have to read the source.
        chimera_whisper::WavData wav;
        try {
            wav = chimera_whisper::load_wav_bytes(upload.data.data(), upload.data.size());
        } catch (const ChimeraError & e) {
            return err_res(415,
                std::string("unsupported audio: ") + e.what() +
                " (Phase 2 currently accepts WAV / RIFF only; mp3, m4a, webm are not yet implemented)");
        }
        auto audio = chimera_whisper::resample_linear(
            wav.samples, wav.sample_rate, /*WHISPER_SAMPLE_RATE=*/16000);

        chimera_whisper::TranscribeRequest treq;
        treq.audio_16k_mono  = std::move(audio);
        treq.language        = lang;
        treq.translate       = false;
        treq.no_context      = false;
        treq.emit_timestamps = true;   // needed for verbose_json/srt/vtt
        treq.initial_prompt  = prompt;

        chimera_whisper::TranscribeResult result;
        {
            // whisper_full mutates internal state on the shared context, so
            // serialize concurrent transcription requests. Slow path anyway.
            std::lock_guard<std::mutex> lock(ctx_mutex);
            try {
                result = chimera_whisper::transcribe(ctx, treq);
            } catch (const ChimeraError & e) {
                return err_res(500, std::string("transcription failed: ") + e.what());
            }
        }

        // Response language: detected language if "auto" was requested,
        // otherwise echo the user's choice.
        const std::string out_lang =
            result.detected_language.empty() ? lang : result.detected_language;

        auto res = std::make_unique<server_http_res>();
        res->status = 200;

        if (fmt == "text") {
            res->content_type = "text/plain; charset=utf-8";
            res->data = result.text;
        } else if (fmt == "srt") {
            res->content_type = "application/x-subrip; charset=utf-8";
            res->data = format_srt(result);
        } else if (fmt == "vtt") {
            res->content_type = "text/vtt; charset=utf-8";
            res->data = format_vtt(result);
        } else if (fmt == "verbose_json") {
            res->data = format_verbose_json(result, "transcribe", out_lang).dump();
        } else {
            // Default: simple {"text": "..."} per OpenAI spec.
            res->data = json{{ "text", result.text }}.dump();
        }
        return res;
    };
}

// ---- Phase 3: image generation helpers ---------------------------------

// Minimal base64 encoder (no padding skip, standard alphabet). The OpenAI
// `b64_json` response format expects unpadded? no — it expects standard
// base64 (with padding). nlohmann/json does not bundle base64, and pulling
// in a whole library for ~30 lines of code isn't worth it.
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

// JSON-field coercion helpers. multipart text fields arrive as strings in
// the JSON body built by server-http.cpp (everything is `field.content`),
// while application/json bodies preserve their original numeric types.
// These helpers accept either form so the same field-reading code can
// drive both routes.
int coerce_int(const json & v, int dflt) {
    if (v.is_number_integer())  return v.get<int>();
    if (v.is_number_unsigned()) return static_cast<int>(v.get<unsigned>());
    if (v.is_number_float())    return static_cast<int>(v.get<double>());
    if (v.is_string()) {
        try { return std::stoi(v.get<std::string>()); } catch (...) {}
    }
    return dflt;
}

int64_t coerce_int64(const json & v, int64_t dflt) {
    if (v.is_number_integer())  return v.get<int64_t>();
    if (v.is_number_unsigned()) return static_cast<int64_t>(v.get<uint64_t>());
    if (v.is_number_float())    return static_cast<int64_t>(v.get<double>());
    if (v.is_string()) {
        try { return std::stoll(v.get<std::string>()); } catch (...) {}
    }
    return dflt;
}

float coerce_float(const json & v, float dflt) {
    if (v.is_number())  return v.get<float>();
    if (v.is_string()) {
        try { return std::stof(v.get<std::string>()); } catch (...) {}
    }
    return dflt;
}

std::string coerce_string(const json & v, const std::string & dflt = "") {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_null())   return dflt;
    return v.dump();
}

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
        try {
            images = chimera_sd::generate(ctx, req);
        } catch (const ChimeraError & e) {
            const int code = (e.code() == ExitCode::BadInput) ? 400 : 500;
            return err_res(code, std::string("image generation failed: ") + e.what());
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

// ---- Phase 4: vector store / RAG helpers -------------------------------

// State shared by every /v1/vector_stores/* handler. Lifetime is the
// `command_serve` call; handlers capture this by pointer.
struct RagContext {
    std::string                                  db_path;
    std::unique_ptr<chimera_embed::Embedder>     embedder;
    std::mutex                                   embedder_mutex;
    std::string                                  loaded_model;  // basename or full path
};

// chunk_text() lives in chimera.cpp's anonymous namespace; for the
// server we re-implement the same simple character-window chunker
// here. Keeping a copy is cheaper than promoting the helper to a
// header for one extra caller; mirror tunables with the CLI defaults.
struct ServeTextChunk { std::string text; int index; };
std::vector<ServeTextChunk> serve_chunk_text(const std::string & text,
                                              size_t window_chars  = 2048,
                                              size_t overlap_chars = 256) {
    std::vector<ServeTextChunk> chunks;
    if (text.empty()) return chunks;
    size_t pos = 0;
    int idx = 0;
    while (pos < text.size()) {
        size_t end = std::min(pos + window_chars, text.size());
        if (end < text.size()) {
            const size_t backstop = pos + (window_chars * 95) / 100;
            for (const char * pat : {"\n\n", ". ", "! ", "? ", "\n"}) {
                const auto p = text.rfind(pat, end);
                if (p != std::string::npos && p > backstop) {
                    end = p + std::strlen(pat);
                    break;
                }
            }
        }
        std::string slice = text.substr(pos, end - pos);
        // Trim leading/trailing whitespace.
        auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
        slice.erase(slice.begin(), std::find_if(slice.begin(), slice.end(), not_space));
        slice.erase(std::find_if(slice.rbegin(), slice.rend(), not_space).base(), slice.end());
        if (!slice.empty()) chunks.push_back({std::move(slice), idx++});
        if (end >= text.size()) break;
        pos = (end > overlap_chars) ? end - overlap_chars : 0;
    }
    return chunks;
}

// Open one DB connection for a single handler call. SQLite open is
// microseconds in WAL mode; cheaper than running a per-thread pool
// with all the lifetime ceremony that brings.
chimera_db::Connection rag_open_db(RagContext * rag) {
    return chimera_db::open_and_migrate(
        rag->db_path.empty() ? chimera_db::default_path() : rag->db_path);
}

server_http_res_ptr rag_err(int code, const std::string & msg) {
    auto r = std::make_unique<server_http_res>();
    r->status = code;
    r->data = json{{ "error",
        { { "message", msg }, { "code", code }, { "type", "invalid_request_error" } }
    }}.dump();
    return r;
}

// Serialize a Collection into the OpenAI-ish shape we return on
// GET/POST. Fields outside OpenAI's `vector_store` schema (dim,
// embedding_model) are namespaced under `meta` rather than
// silently spliced into the OpenAI shape.
json collection_to_json(const chimera_vector_store::Collection & c) {
    return {
        { "id",        c.name },               // OpenAI keys vector stores by id
        { "object",    "vector_store" },
        { "name",      c.name },
        { "created_at", c.created_at },
        { "file_counts", { { "completed", c.doc_count }, { "total", c.doc_count } } },
        { "meta", {
            { "embedding_model", c.embedding_model },
            { "dim",             c.dim },
        }},
    };
}

// GET /v1/vector_stores → { "object": "list", "data": [...] }
server_http_context::handler_t make_vs_list_handler(RagContext * rag) {
    return [rag](const server_http_req & /*req*/) -> server_http_res_ptr {
        auto conn = rag_open_db(rag);
        const auto cols = chimera_vector_store::list(conn.get());
        json data = json::array();
        for (const auto & c : cols) data.push_back(collection_to_json(c));
        auto res = std::make_unique<server_http_res>();
        res->data = json{{ "object", "list" }, { "data", std::move(data) }}.dump();
        return res;
    };
}

// POST /v1/vector_stores → create. Body: { "name": "...", "embedding_model": "..."? }.
// embedding_model defaults to the model loaded at server start. If a
// different value is supplied we reject — single-model server in this cut.
server_http_context::handler_t make_vs_create_handler(RagContext * rag) {
    return [rag](const server_http_req & req) -> server_http_res_ptr {
        json body = json::object();
        if (!req.body.empty()) {
            try { body = json::parse(req.body); }
            catch (const std::exception & e) {
                return rag_err(400, std::string("invalid JSON body: ") + e.what());
            }
        }
        const std::string name = body.value("name", std::string());
        if (name.empty()) return rag_err(400, "field 'name' is required");
        const std::string model = body.value("embedding_model", rag->loaded_model);
        if (model != rag->loaded_model) {
            return rag_err(400,
                "embedding_model mismatch: server has '" + rag->loaded_model +
                "' loaded; collection requested '" + model + "'. "
                "Restart chimera serve with --enable-rag pointing at that model.");
        }
        auto conn = rag_open_db(rag);
        const int dim = rag->embedder->n_embd();
        auto col = chimera_vector_store::create(conn.get(), name, model, dim);
        auto res = std::make_unique<server_http_res>();
        res->status = 201;
        res->data = collection_to_json(col).dump();
        return res;
    };
}

// GET /v1/vector_stores/:name → stats for one collection.
server_http_context::handler_t make_vs_get_handler(RagContext * rag) {
    return [rag](const server_http_req & req) -> server_http_res_ptr {
        const std::string name = req.get_param("name");
        if (name.empty()) return rag_err(400, "missing :name path param");
        auto conn = rag_open_db(rag);
        // Pull from list() so we get the doc_count via correlated subquery.
        for (const auto & c : chimera_vector_store::list(conn.get())) {
            if (c.name == name) {
                auto res = std::make_unique<server_http_res>();
                res->data = collection_to_json(c).dump();
                return res;
            }
        }
        // Returning 400 (not 404) keeps our error message visible —
        // server-http's set_error_handler unconditionally overwrites
        // any 404 body with the upstream-generic "File Not Found".
        return rag_err(400, "no such collection: '" + name + "'");
    };
}

// POST /v1/vector_stores/:name/delete → drop. (Not DELETE; see comment
// near the route registration.)
server_http_context::handler_t make_vs_delete_handler(RagContext * rag) {
    return [rag](const server_http_req & req) -> server_http_res_ptr {
        const std::string name = req.get_param("name");
        if (name.empty()) return rag_err(400, "missing :name path param");
        auto conn = rag_open_db(rag);
        try {
            chimera_vector_store::drop(conn.get(), name);
        } catch (const ChimeraError & e) {
            return rag_err(400, e.what());
        }
        auto res = std::make_unique<server_http_res>();
        res->data = json{{ "id", name }, { "deleted", true },
                          { "object", "vector_store.deleted" }}.dump();
        return res;
    };
}

// POST /v1/vector_stores/:name/files → chunk + embed + insert.
// Body forms accepted:
//   multipart/form-data with a `file` upload (and optional `source_uri`)
//   application/json: { "text": "...", "source_uri": "..." }
server_http_context::handler_t make_vs_ingest_handler(RagContext * rag) {
    return [rag](const server_http_req & req) -> server_http_res_ptr {
        const std::string name = req.get_param("name");
        if (name.empty()) return rag_err(400, "missing :name path param");

        // Resolve text input + source_uri. server-http folds multipart
        // text fields into req.body as JSON; for application/json the
        // body is whatever the client sent.
        std::string text, source_uri;
        if (!req.files.empty()) {
            auto it = req.files.find("file");
            if (it == req.files.end() || it->second.data.empty()) {
                return rag_err(400, "missing 'file' field in multipart form");
            }
            text.assign(reinterpret_cast<const char *>(it->second.data.data()),
                        it->second.data.size());
            source_uri = it->second.filename;
        }
        if (!req.body.empty()) {
            json b;
            try { b = json::parse(req.body); }
            catch (const std::exception &) { b = json::object(); }
            if (text.empty() && b.contains("text") && b["text"].is_string()) {
                text = b["text"].get<std::string>();
            }
            if (source_uri.empty() && b.contains("source_uri") && b["source_uri"].is_string()) {
                source_uri = b["source_uri"].get<std::string>();
            }
        }
        if (text.empty()) {
            return rag_err(400,
                "request must include either a multipart 'file' upload or a JSON 'text' field");
        }

        auto conn = rag_open_db(rag);
        auto col = chimera_vector_store::find(conn.get(), name);
        if (!col) return rag_err(400, "no such collection: '" + name + "'");
        if (col->embedding_model != rag->loaded_model) {
            return rag_err(400,
                "collection '" + name + "' was indexed with '" + col->embedding_model +
                "'; server has '" + rag->loaded_model + "' loaded.");
        }

        const auto chunks = serve_chunk_text(text);
        if (chunks.empty()) return rag_err(400, "no non-empty chunks produced");

        int64_t inserted = 0;
        for (const auto & c : chunks) {
            std::vector<float> vec;
            {
                std::lock_guard<std::mutex> lk(rag->embedder_mutex);
                vec = rag->embedder->embed(c.text);
            }
            chimera_vector_store::DocumentInput doc;
            doc.source_uri  = source_uri;
            doc.chunk_index = c.index;
            doc.text        = c.text;
            doc.embedding   = std::move(vec);
            chimera_vector_store::insert_document(conn.get(), *col, doc);
            ++inserted;
        }

        auto res = std::make_unique<server_http_res>();
        res->data = json{
            { "object",      "vector_store.file" },
            { "vector_store_id", name },
            { "source_uri",  source_uri },
            { "chunks_inserted", inserted },
        }.dump();
        return res;
    };
}

// POST /v1/vector_stores/:name/search → KNN.
// Body: { "query": "...", "k": 5? }
server_http_context::handler_t make_vs_search_handler(RagContext * rag) {
    return [rag](const server_http_req & req) -> server_http_res_ptr {
        const std::string name = req.get_param("name");
        if (name.empty()) return rag_err(400, "missing :name path param");

        json body = json::object();
        if (!req.body.empty()) {
            try { body = json::parse(req.body); }
            catch (const std::exception & e) {
                return rag_err(400, std::string("invalid JSON body: ") + e.what());
            }
        }
        const std::string query = body.value("query", std::string());
        if (query.empty()) return rag_err(400, "field 'query' is required");
        const int k = coerce_int(body.value("k", json(5)), 5);

        auto conn = rag_open_db(rag);
        auto col = chimera_vector_store::find(conn.get(), name);
        if (!col) return rag_err(400, "no such collection: '" + name + "'");
        if (col->embedding_model != rag->loaded_model) {
            return rag_err(400,
                "collection '" + name + "' was indexed with '" + col->embedding_model +
                "'; server has '" + rag->loaded_model + "' loaded.");
        }

        std::vector<float> qvec;
        {
            std::lock_guard<std::mutex> lk(rag->embedder_mutex);
            qvec = rag->embedder->embed(query);
        }
        const auto hits = chimera_vector_store::search(conn.get(), *col, qvec, k);

        json data = json::array();
        for (const auto & h : hits) {
            data.push_back({
                { "document_id", h.document_id },
                { "source_uri",  h.source_uri },
                { "chunk_index", h.chunk_index },
                { "text",        h.text },
                { "distance",    h.distance },
            });
        }
        auto res = std::make_unique<server_http_res>();
        res->data = json{{ "object", "list" }, { "data", std::move(data) }}.dump();
        return res;
    };
}

// ---- Phase 5: chat persistence shim ----------------------------------

// Per-server state for the /v1/chat/completions persistence wrapper.
// Captured by pointer in the wrapped handler so the SQLite connection
// pool isn't recreated on every request.
struct ChatPersistContext {
    std::string db_path;
    std::mutex  mutex;   // serializes DB writes from concurrent HTTP workers
};

// Write a completed exchange to the DB: create one chats row, then append
// every message from the request plus the new assistant reply.
// `model_alias` is the model the request claimed; fall back to whatever
// the server has loaded.
void persist_completed_chat(ChatPersistContext * cpc,
                            const json &        req_body,
                            const std::string & assistant_content,
                            const std::string & assistant_reasoning,
                            int                 tokens_in,
                            int                 tokens_out) {
    try {
        if (!req_body.contains("messages") || !req_body["messages"].is_array()) return;

        chimera_db::Connection conn = chimera_db::open_and_migrate(
            cpc->db_path.empty() ? chimera_db::default_path() : cpc->db_path);

        // Pull system prompt + model name out of the request for the
        // chats row metadata. OpenAI's `model` may be the server-loaded
        // name, an alias, or "any"; record verbatim.
        std::string model_alias = "any";
        if (req_body.contains("model") && req_body["model"].is_string()) {
            model_alias = req_body["model"].get<std::string>();
        }
        std::string system_prompt;
        for (const auto & m : req_body["messages"]) {
            if (m.contains("role") && m["role"] == "system" &&
                m.contains("content") && m["content"].is_string()) {
                system_prompt = m["content"].get<std::string>();
                break;
            }
        }

        std::lock_guard<std::mutex> lk(cpc->mutex);

        const int64_t chat_id = chimera_chat_store::create_chat(
            conn.get(),
            /*model_path=*/model_alias, model_alias,
            system_prompt, /*source=*/"serve");

        // Every message in the request gets one row. The OpenAI client
        // sends the full conversation on each request, so this captures
        // the whole conversation (with the duplication caveat documented
        // in CHANGELOG: multi-turn chats produce multiple chat rows
        // that share content).
        for (const auto & m : req_body["messages"]) {
            const std::string role = m.value("role", std::string("user"));
            std::string content;
            if (m.contains("content")) {
                if (m["content"].is_string()) {
                    content = m["content"].get<std::string>();
                } else {
                    // Tool calls, image_url parts, etc. — serialize as JSON.
                    content = m["content"].dump();
                }
            }
            chimera_chat_store::append_message(
                conn.get(), chat_id, role, content);
        }
        // The new assistant reply.
        chimera_chat_store::append_message(
            conn.get(), chat_id, "assistant", assistant_content,
            assistant_reasoning, /*media_json=*/"", tokens_in, tokens_out);
    } catch (const std::exception & e) {
        // Persistence failure must never break the user's HTTP request.
        // Log and move on; the response has already been (or will be) sent.
        std::fprintf(stderr, "chimera serve: persist_completed_chat failed: %s\n",
                     e.what());
    }
}

// Parse a non-streaming /v1/chat/completions response body and pull out
// the assistant content + token counts.
void persist_non_streaming(ChatPersistContext * cpc,
                            const json &        req_body,
                            const std::string & res_data) {
    try {
        json r = json::parse(res_data);
        if (!r.contains("choices") || !r["choices"].is_array() || r["choices"].empty()) {
            return;
        }
        const auto & ch = r["choices"][0];
        std::string content, reasoning;
        if (ch.contains("message") && ch["message"].is_object()) {
            const auto & m = ch["message"];
            if (m.contains("content") && m["content"].is_string()) {
                content = m["content"].get<std::string>();
            }
            if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
                reasoning = m["reasoning_content"].get<std::string>();
            }
        }
        int tokens_in = 0, tokens_out = 0;
        if (r.contains("usage") && r["usage"].is_object()) {
            tokens_in  = r["usage"].value("prompt_tokens",     0);
            tokens_out = r["usage"].value("completion_tokens", 0);
        }
        persist_completed_chat(cpc, req_body, content, reasoning, tokens_in, tokens_out);
    } catch (...) {
        // Don't propagate.
    }
}

// Parse a streaming response (buffered SSE chunks already collected) and
// persist the result. SSE shape: `data: {json}\n\n` per chunk, plus a
// `data: [DONE]` trailer.
void persist_streaming(ChatPersistContext * cpc,
                        const json &        req_body,
                        const std::string & sse_buffer) {
    std::string content, reasoning;
    int tokens_in = 0, tokens_out = 0;
    size_t pos = 0;
    while (pos < sse_buffer.size()) {
        // Each SSE event ends with "\n\n". Find the next boundary.
        const size_t end = sse_buffer.find("\n\n", pos);
        if (end == std::string::npos) break;
        const std::string event = sse_buffer.substr(pos, end - pos);
        pos = end + 2;

        // Strip the "data: " prefix on each line of the event. (Multi-line
        // data: events get concatenated; rare here but cheap to handle.)
        std::string payload;
        size_t line_start = 0;
        while (line_start < event.size()) {
            const size_t lend = event.find('\n', line_start);
            const std::string line = event.substr(line_start, lend - line_start);
            if (line.rfind("data: ", 0) == 0) {
                payload += line.substr(6);
            }
            if (lend == std::string::npos) break;
            line_start = lend + 1;
        }
        if (payload.empty() || payload == "[DONE]") continue;
        try {
            json j = json::parse(payload);
            if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
                const auto & delta = j["choices"][0].value("delta", json::object());
                if (delta.contains("content") && delta["content"].is_string()) {
                    content += delta["content"].get<std::string>();
                }
                if (delta.contains("reasoning_content") && delta["reasoning_content"].is_string()) {
                    reasoning += delta["reasoning_content"].get<std::string>();
                }
            }
            // Usage may appear in the final chunk (OpenAI sends it when
            // stream_options.include_usage = true).
            if (j.contains("usage") && j["usage"].is_object()) {
                tokens_in  = j["usage"].value("prompt_tokens",     tokens_in);
                tokens_out = j["usage"].value("completion_tokens", tokens_out);
            }
        } catch (...) {
            // Skip malformed event; keep parsing.
        }
    }
    if (!content.empty() || !reasoning.empty()) {
        persist_completed_chat(cpc, req_body, content, reasoning, tokens_in, tokens_out);
    }
}

// Wraps server_routes::post_chat_completions so each successful exchange
// is saved to the chats + messages tables. Streaming responses are
// passed through chunk-by-chunk and assembled in a buffer for parsing at
// stream end. Persistence runs *after* the chunk is sent to the client;
// any error in persistence is logged but never visible to the caller.
server_http_context::handler_t make_persisting_chat_handler(
    server_http_context::handler_t inner,
    ChatPersistContext *           cpc) {
    return [inner, cpc](const server_http_req & req) -> server_http_res_ptr {
        json req_body;
        try { req_body = json::parse(req.body); }
        catch (...) {
            // Malformed JSON — let the inner handler emit its own 400.
            return inner(req);
        }
        auto res = inner(req);
        if (!res || res->status >= 300) return res;

        if (!res->is_stream()) {
            // Non-streaming: we have the full body in res->data.
            persist_non_streaming(cpc, req_body, res->data);
            return res;
        }

        // Streaming: wrap `next` to mirror each chunk into a buffer,
        // then parse + persist when the stream ends. The shared_ptr
        // keeps the buffer alive across the move-from of res->next.
        auto buffer = std::make_shared<std::string>();
        auto inner_next = std::move(res->next);
        auto saved_body = std::make_shared<json>(std::move(req_body));
        res->next = [inner_next, buffer, cpc, saved_body](std::string & out) {
            const bool has_more = inner_next(out);
            buffer->append(out);
            if (!has_more) {
                persist_streaming(cpc, *saved_body, *buffer);
            }
            return has_more;
        };
        return res;
    };
}

} // namespace

int command_serve(const ServeOptions & opts) {
    if (opts.model.empty()) {
        fail(ExitCode::BadInput, "--model is required for `chimera serve`");
    }

    common_params params = build_common_params(opts);

    // llama_backend_init() / llama_backend_free() are handled by main();
    // we only initialize NUMA which depends on per-subcommand params.numa.
    llama_numa_init(params.numa);

    server_context      ctx_server;
    server_http_context ctx_http;

    if (!ctx_http.init(params)) {
        std::cerr << "failed to initialize HTTP server\n";
        return static_cast<int>(ExitCode::Runtime);
    }

    server_routes routes(params, ctx_server);

    // Phase 2: opt-in audio. The whisper context and its serializing mutex
    // live for as long as command_serve runs; the route handler captures
    // them by reference. If --enable-audio was not passed, the context
    // stays null and we don't bind the route.
    WhisperContextPtr whisper_ctx;
    std::mutex        whisper_mutex;
    if (!opts.audio_model.empty()) {
        std::cout << "chimera serve: loading audio model " << opts.audio_model << "...\n";
        whisper_ctx = chimera_whisper::load_model(opts.audio_model);
        if (!whisper_ctx) {
            std::cerr << "chimera serve: failed to load audio model: "
                      << opts.audio_model << "\n";
            ctx_http.stop();
            ctx_server.terminate();
            return static_cast<int>(ExitCode::Load);
        }
    }

    // Phase 3: opt-in image. vae_decode_only=false because /edits and
    // /variations need img2img (the VAE encode path). On txt2img-only
    // workloads this trades some memory for path uniformity.
    SdContextPtr sd_ctx;
    std::mutex   sd_mutex;
    if (!opts.sd_model.empty()) {
        std::cout << "chimera serve: loading image model " << opts.sd_model << "...\n";
        sd_ctx = chimera_sd::load_model(opts.sd_model, /*vae_decode_only=*/false);
        if (!sd_ctx) {
            std::cerr << "chimera serve: failed to load image model: "
                      << opts.sd_model << "\n";
            ctx_http.stop();
            ctx_server.terminate();
            return static_cast<int>(ExitCode::Load);
        }
    }

    // Phase 4: opt-in vector store / RAG. One embedding model per server
    // in this cut. The chimera SQLite DB is shared with the CLI (same
    // $CHIMERA_DB / platform default); ingest + search hit it via
    // per-request connections.
    RagContext rag_ctx;
    if (!opts.rag_embedding_model.empty()) {
        std::cout << "chimera serve: loading embedding model "
                  << opts.rag_embedding_model << "...\n";
        chimera_embed::Config ecfg;
        ecfg.model     = opts.rag_embedding_model;
        ecfg.normalize = true;
        rag_ctx.embedder     = std::make_unique<chimera_embed::Embedder>(ecfg);
        rag_ctx.loaded_model = opts.rag_embedding_model;
        rag_ctx.db_path      = opts.rag_db_path;
        // Touch the DB once at startup to surface migration failures
        // before the first request rather than mid-request. Also
        // creates the file at $CHIMERA_DB / the platform default if it
        // doesn't exist yet.
        (void) chimera_db::open_and_migrate(
            rag_ctx.db_path.empty() ? chimera_db::default_path() : rag_ctx.db_path);
    }

    // Route registration. LLM route handlers are pre-built lambdas on
    // server_routes — we just bind them. See the top-of-file comment for the
    // routes we are deliberately NOT exposing.
    ctx_http.get ("/health",              ex_wrapper(routes.get_health));
    ctx_http.get ("/v1/health",           ex_wrapper(routes.get_health));
    ctx_http.get ("/v1/models",           ex_wrapper(routes.get_models));
    ctx_http.get ("/metrics",             ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",               ex_wrapper(routes.get_props));

    // Phase 5: chat persistence shim. When --persist-chats is set we
    // wrap the upstream chat-completions handler so every successful
    // exchange is recorded in the chats + messages tables. The wrapper
    // handles both non-streaming responses and SSE streams (by mirroring
    // each chunk into a buffer and parsing on stream end). When the
    // flag is off, we bind the upstream handler unchanged.
    ChatPersistContext chat_persist_ctx;
    auto chat_handler = routes.post_chat_completions;  // capture by value
    if (opts.persist_chats) {
        chat_persist_ctx.db_path = opts.chat_db_path;
        // Touch the DB at startup to surface migration failures here
        // rather than mid-request.
        (void) chimera_db::open_and_migrate(
            chat_persist_ctx.db_path.empty()
                ? chimera_db::default_path() : chat_persist_ctx.db_path);
        chat_handler = make_persisting_chat_handler(chat_handler, &chat_persist_ctx);
    }

    // /chat/completions and /v1/chat/completions share one handler; the
    // legacy unprefixed variant is bound for compatibility with older
    // OpenAI clients and self-rolled tools that target llama-server's
    // historical path.
    ctx_http.post("/chat/completions",    ex_wrapper(chat_handler));
    ctx_http.post("/v1/chat/completions", ex_wrapper(chat_handler));
    ctx_http.post("/v1/completions",      ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/v1/embeddings",       ex_wrapper(routes.post_embeddings_oai));
    // Phase 5: bind /v1/responses (OpenAI Responses API). The upstream
    // handler holds conversation state in-process — it's *stateful within
    // one chimera serve invocation* but does not persist across
    // restarts. With --persist-chats on, the underlying chat-completions
    // path will still write to the chats table; the Responses API itself
    // is layered on top of that and inherits the same persistence.
    ctx_http.post("/v1/responses",        ex_wrapper(routes.post_responses_oai));

    // Anthropic Messages API compat — lets the Anthropic Python SDK and
    // claude-code-shaped clients point at chimera serve unchanged.
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages));
    ctx_http.post("/v1/messages/count_tokens", ex_wrapper(routes.post_anthropic_count_tokens));

    // Fill-in-the-middle for code models (continue.dev, llama.vim, ...).
    ctx_http.post("/infill",          ex_wrapper(routes.post_infill));

    // Tokenize / detokenize / apply-template — small but useful for
    // clients that don't bundle a tokenizer (token counting before send)
    // or want to debug chat-template behavior.
    ctx_http.post("/tokenize",        ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",      ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",  ex_wrapper(routes.post_apply_template));
    if (whisper_ctx) {
        ctx_http.post("/v1/audio/transcriptions",
                      ex_wrapper(make_audio_transcribe_handler(
                          whisper_ctx.get(), whisper_mutex)));
    }
    if (sd_ctx) {
        ctx_http.post("/v1/images/generations",
                      ex_wrapper(make_image_generations_handler(sd_ctx.get(), sd_mutex)));
        ctx_http.post("/v1/images/edits",
                      ex_wrapper(make_image_edits_handler(sd_ctx.get(), sd_mutex)));
        ctx_http.post("/v1/images/variations",
                      ex_wrapper(make_image_variations_handler(sd_ctx.get(), sd_mutex)));
    }
    if (rag_ctx.embedder) {
        ctx_http.get ("/v1/vector_stores",              ex_wrapper(make_vs_list_handler(&rag_ctx)));
        ctx_http.post("/v1/vector_stores",              ex_wrapper(make_vs_create_handler(&rag_ctx)));
        ctx_http.get ("/v1/vector_stores/:name",        ex_wrapper(make_vs_get_handler(&rag_ctx)));
        ctx_http.post("/v1/vector_stores/:name/delete", ex_wrapper(make_vs_delete_handler(&rag_ctx)));
        ctx_http.post("/v1/vector_stores/:name/files",  ex_wrapper(make_vs_ingest_handler(&rag_ctx)));
        ctx_http.post("/v1/vector_stores/:name/search", ex_wrapper(make_vs_search_handler(&rag_ctx)));
    }

    auto clean_up = [&]() {
        ctx_http.stop();
        ctx_server.terminate();
    };

    // Start HTTP before loading the model so /health responds early.
    if (!ctx_http.start()) {
        clean_up();
        return static_cast<int>(ExitCode::Runtime);
    }

    std::cout << "chimera serve: loading model...\n";
    if (!ctx_server.load_model(params)) {
        std::cerr << "chimera serve: failed to load model: " << opts.model << "\n";
        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        return static_cast<int>(ExitCode::Load);
    }

    routes.update_meta(ctx_server);
    ctx_http.is_ready.store(true);

    g_shutdown_handler = [&](int) { ctx_server.terminate(); };
    install_signal_handlers();

    std::cout << "chimera serve: listening on " << ctx_http.listening_address << "\n"
              << "  LLM:   /v1/chat/completions  /v1/completions  /v1/embeddings\n"
              << "  meta:  /v1/models  /health  /metrics  /props\n"
              << "  tools: /infill  /tokenize  /detokenize  /apply-template\n"
              << "  anthropic: /v1/messages  /v1/messages/count_tokens\n";
    if (whisper_ctx) std::cout << "  audio: /v1/audio/transcriptions\n";
    if (sd_ctx)      std::cout << "  image: /v1/images/generations  /v1/images/edits  /v1/images/variations\n";
    if (rag_ctx.embedder) std::cout << "  rag:   /v1/vector_stores  /v1/vector_stores/:name{,/delete,/files,/search}\n";
    if (opts.persist_chats) {
        std::cout << "  persistence: --persist-chats ON (DB: "
                  << (chat_persist_ctx.db_path.empty()
                        ? chimera_db::default_path() : chat_persist_ctx.db_path)
                  << ")\n";
    }

    // Blocks on the main thread until the task queue is terminated by the
    // signal handler. Worker tasks run on threads owned by server_context;
    // HTTP requests run on threads owned by cpp-httplib inside ctx_http.
    ctx_server.start_loop();

    clean_up();
    if (ctx_http.thread.joinable()) {
        ctx_http.thread.join();
    }
    return 0;
}
