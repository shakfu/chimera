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
//   GET  /slots                         per-slot status (id, state, prompt, ...)
//   POST /slots/:id_slot                save / restore / erase KV-cache snapshots
//                                       (write actions require --slot-save-path)
//   GET  /lora-adapters                 list LoRAs loaded via --lora
//   POST /lora-adapters                 hot-swap which LoRAs are active and
//                                       at what scale, without a model reload
//
// Chat history (only when --persist-chats):
//   GET  /v1/chats                      list persisted chats, sorted by
//                                       updated_at desc; ?limit=N (default 50)
//   GET  /v1/chats/:id                  chat metadata + ordered messages[]
//   GET  /v1/chats/search?q=...         FTS5 search over message content,
//                                       with [word]-highlighted snippets;
//                                       ?limit=N (default 20)
//
// Audio (only when --enable-audio):
//   POST /v1/audio/transcriptions       transcribe in source language (whisper.cpp)
//   POST /v1/audio/translations         translate to English regardless of input
//                                       language (whisper's built-in translate mode)
//                                       — both currently WAV-only; see handler comment
//
// Image (only when --enable-image):
//   POST /v1/images/generations         txt2img (stable-diffusion.cpp)
//   POST /v1/images/edits               img2img + optional mask
//   POST /v1/images/variations          img2img with no prompt
//
// Dedicated embedding model (only when --enable-embeddings):
//   POST /v1/embeddings                 (routes here instead of the main LLM's
//                                       embedding handler when this flag is set)
//
// Cross-encoder reranker (only when --reranking):
//   POST /v1/rerank, /rerank            documents reranked against a query;
//                                       request shape:
//                                       {"model": "...", "query": "...",
//                                        "documents": ["..."], "top_n": N}
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
// llama-server parity vs chimera-owned surface
// ----------------------------------------------------------------------------
// On the next llama.cpp bump, the question "which of these routes might
// silently change shape" has different answers depending on who owns the
// handler. Three buckets, ordered by upstream-drift risk:
//
// (A) UPSTREAM-OWNED — handler bound from `server_routes` (a typed lambda
//     field on libserver-context). `chimera_pin_check.cpp` static_asserts
//     each handler_t field so a type change fails to compile here.
//     Response JSON shape is whatever upstream emits; chimera does NOT
//     own it. A llama.cpp bump can change a field name / add a key / drop
//     a key without breaking the pin-check. These are the routes where
//     `make test-golden` matters.
//
//       GET  /health, /v1/health                 server_routes.get_health
//       GET  /v1/models                          server_routes.get_models
//       GET  /metrics                            server_routes.get_metrics
//       GET  /props                              server_routes.get_props
//       POST /chat/completions
//            + /v1/chat/completions              server_routes.post_chat_completions
//                                                (wrapped by chimera's persistence
//                                                shim when --persist-chats — see
//                                                make_persisting_chat_handler)
//       POST /v1/completions                     server_routes.post_completions_oai
//       POST /v1/embeddings                      server_routes.post_embeddings_oai
//                                                (or the dedicated emb_ctx variant
//                                                when --enable-embeddings is set)
//       POST /v1/messages                        server_routes.post_anthropic_messages
//       POST /v1/messages/count_tokens           server_routes.post_anthropic_count_tokens
//       POST /v1/responses                       server_routes.post_responses_oai
//       POST /infill                             server_routes.post_infill
//       POST /tokenize, /detokenize              server_routes.post_{tokenize,detokenize}
//       POST /apply-template                     server_routes.post_apply_template
//       POST /v1/rerank, /rerank                 rrk_ctx->routes->post_rerank
//       GET  /slots                              server_routes.get_slots
//       POST /slots/:id_slot                     server_routes.post_slots
//       GET  /lora-adapters                      server_routes.get_lora_adapters
//       POST /lora-adapters                      server_routes.post_lora_adapters
//
//     The `--public-path <dir>` CLI flag is also upstream-parity: it
//     drives `common_params.public_path`, which upstream's
//     `server_http_context::init` already handles via
//     cpp-httplib's `set_mount_point`. No chimera-side route binding.
//
// (B) CHIMERA-OWNED HANDLER, EXTERNAL-PROTOCOL SHAPE — chimera wrote the
//     handler, but the request/response JSON is defined by an external
//     spec (OpenAI's Audio / Images / Vector Stores). Upstream llama.cpp
//     either doesn't implement the protocol or implements it through a
//     different pipeline that chimera doesn't want. A llama.cpp bump
//     cannot change these shapes; OpenAI changing their spec can. Drift
//     surface = the external spec, not the vendored library.
//
//       POST /v1/audio/transcriptions            chimera_whisper, via
//       POST /v1/audio/translations              make_audio_transcribe_handler
//                                                (upstream routes audio through
//                                                mtmd's audio mmproj — different
//                                                pipeline, intentionally unused)
//       POST /v1/images/generations              chimera_sd, via
//       POST /v1/images/edits                    make_image_{generations,edits,
//       POST /v1/images/variations               variations}_handler
//                                                (no upstream equivalent — llama.cpp
//                                                has no SD)
//       GET  /v1/vector_stores                   chimera_vector_store, via
//       POST /v1/vector_stores                   make_vs_{list,create,get,delete,
//       GET  /v1/vector_stores/:name             ingest,search}_handler
//       POST /v1/vector_stores/:name/{delete,    (OpenAI Vector Stores API shape;
//            files,search}                       no upstream equivalent — llama.cpp
//                                                has no RAG)
//
// (C) CHIMERA-OWNED, CHIMERA-ONLY SHAPE — no external spec at all. Both
//     the route paths and the response shapes are chimera's invention.
//     Stable until *we* change them. The `object: "chimera.<thing>"`
//     namespace prefix is the marker.
//
//       GET  /v1/chats                           make_chats_list_handler
//       GET  /v1/chats/:id                       make_chats_get_handler
//       GET  /v1/chats/search?q=...              make_chats_search_handler
//
//     (Backed by chimera's SQLite chats + messages + messages_fts tables.
//     Upstream llama-server has no persistence at all — its `/v1/responses`
//     state lives in-process and is lost on restart. There is no equivalent
//     `/v1/chats*` surface upstream.)
//
// `X-Chimera-Chat-Id` (request + response header on /v1/chat/completions
// for multi-turn persistence consolidation) is similarly chimera-only;
// no upstream equivalent.
//
// ----------------------------------------------------------------------------
// Scope — what is DELIBERATELY NOT exposed
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
//   - Web chat UI (`public/{index.html, bundle.js, bundle.css}`). Experimental:
//     opt in at configure time with `-DCHIMERA_WEBUI_EMBED=ON` to xxd-bake
//     upstream's prebuilt bundle (~7 MB) into the chimera binary. When
//     enabled, GET / serves index.html and GET /bundle.{js,css} serve the
//     assets; the runtime route-binding lives in upstream's server-http.cpp
//     gated by the LLAMA_BUILD_WEBUI compile define + params.webui (chimera
//     defaults the latter true; pass `--no-webui` to disable per-run).
//   - Child-server / parent-process sleeping notification.
//   - SSL / TLS (no `--ssl-cert-file` / `--ssl-key-file`). Run behind a
//     reverse proxy for HTTPS.
//
// File organization. The per-modality handlers (audio / images / RAG /
// chat persistence / chat history read) live in sibling TUs to keep
// each file short:
//
//   chimera_serve_audio.cpp        — make_audio_transcribe_handler
//   chimera_serve_images.cpp       — make_image_{generations,edits,variations}_handler,
//                            plus the coerce_* JSON helpers (also linked
//                            from chimera_serve_rag.cpp)
//   chimera_serve_rag.cpp          — make_vs_{list,create,get,delete,ingest,search}_handler
//                            (RagContext)
//   chimera_serve_chat_persist.cpp — make_persisting_chat_handler (ChatPersistContext)
//   chimera_serve_chats_read.cpp   — make_chats_{list,get,search}_handler
//                            (ChatHistoryContext)
//
// chimera_serve_internal.h declares the seams (context structs + handler
// factories). This TU keeps the file-top docstring, ex_wrapper, signal
// handlers, build_common_params, the SecondaryServerCtx + bring_up_secondary
// helpers for --enable-embeddings / --reranking, and command_serve itself.
//
// ----------------------------------------------------------------------------

#include "chimera_serve_internal.h"

#include "chimera.h"
#include "chimera_chat_store.h"
#include "chimera_db.h"
#include "chimera_embed.h"
#include "chimera_embed_cache.h"
#include "chimera_vector_store.h"
#ifdef CHIMERA_HAS_WHISPER
#  include "chimera_whisper.h"
#endif
#ifdef CHIMERA_HAS_SD
#  include "chimera_sd.h"
#endif

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

#if defined(_WIN32)
// SetConsoleCtrlHandler + PHANDLER_ROUTINE live here. We include
// <windows.h> only on Windows because it's enormous (~10 k declarations
// on a typical SDK) and not needed by the POSIX signal path. Define the
// NOMINMAX / WIN32_LEAN_AND_MEAN guards so it doesn't fight nlohmann/json
// or anything else that exposes `min`/`max` as identifiers, and skip the
// Winsock + GDI surface we don't touch.
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

namespace chimera_serve {

// Mirrors llama-server's ex_wrapper (server.cpp:40). Ensures handlers never
// throw out of the HTTP layer; converts std::invalid_argument to 400 and
// every other exception to 500. Declared in chimera_serve_internal.h so the
// per-modality TUs can wrap their own handlers identically — we want
// 400/500 conversion at every boundary, not just on llama-server's routes.
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

// JSON-field coercion helpers. multipart text fields arrive as strings in
// the JSON body built by server-http.cpp (everything is `field.content`),
// while application/json bodies preserve their original numeric types.
// These helpers accept either form so the same field-reading code can
// drive both routes. Live here (always compiled) rather than in
// chimera_serve_images.cpp because chimera_serve_rag.cpp also uses them,
// and the RAG handler must keep working when CHIMERA_HAS_SD is undefined.
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

std::string coerce_string(const json & v, const std::string & dflt) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_null())   return dflt;
    return v.dump();
}

namespace {

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

    // Embedded webui. Only meaningful when the build was configured
    // with CHIMERA_WEBUI_EMBED=1 (see scripts/manage.py); the actual
    // route binding happens inside server_http_context::init based on
    // this field. In a webui-less build the routes don't exist either
    // way, so the flag is a harmless pass-through.
    params.webui = opts.webui;

    // External static UI directory. server_http_context::init mounts
    // this at GET / via cpp-httplib's set_mount_point when non-empty.
    // Independent of the xxd-baked variant above; when both are set,
    // the mount point is registered first and takes precedence over
    // the embedded GET / handler.
    params.public_path = opts.public_path;

    // KV-cache slot snapshots. Upstream gates POST /slots/:id on this
    // being non-empty (returns "not supported" otherwise). GET /slots
    // is independent and controlled by params.endpoint_slots, which
    // defaults to true. Upstream concatenates this path with the
    // user-supplied filename without a separator (see handle_slots_save
    // in server-context.cpp), so we have to ensure the trailing slash
    // ourselves — common/arg.cpp does it there, but chimera bypasses
    // common args.
    params.slot_save_path = opts.slot_save_path;
    if (!params.slot_save_path.empty() &&
        params.slot_save_path.back() != '/' &&
        params.slot_save_path.back() != '\\') {
        params.slot_save_path += '/';
    }

    // LoRA adapters: parse "path[:scale]" entries into the upstream
    // struct. Scale defaults to 1.0 when omitted. The HTTP routes
    // (GET/POST /lora-adapters) can re-weight or disable adapters at
    // request time but cannot add new ones — anything callable via
    // POST must be in this list at startup.
    for (const auto & spec : opts.lora_adapters) {
        common_adapter_lora_info lora{};
        lora.scale = 1.0f;
        lora.ptr   = nullptr;
        const auto colon = spec.find_last_of(':');
        if (colon != std::string::npos) {
            try {
                lora.scale = std::stof(spec.substr(colon + 1));
                lora.path  = spec.substr(0, colon);
            } catch (const std::exception &) {
                // No parseable scale suffix — treat the whole string
                // as a path (handles Windows "C:\..." with no scale).
                lora.path = spec;
            }
        } else {
            lora.path = spec;
        }
        params.lora_adapters.push_back(std::move(lora));
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

// Holds a secondary server_context — dedicated embedding model or
// cross-encoder reranker — that lives alongside the primary LLM. Each
// secondary owns its own task queue / scheduler and runs its
// `start_loop()` on a worker thread. Routes from the secondary's
// `server_routes` are bound onto the shared `ctx_http`. `params` must
// outlive `routes` (server_routes holds it by const reference), so this
// struct is non-movable and always heap-allocated.
struct SecondaryServerCtx {
    common_params                   params;
    std::unique_ptr<server_context> ctx;
    std::unique_ptr<server_routes>  routes;
    std::thread                     loop;

    SecondaryServerCtx() = default;
    SecondaryServerCtx(const SecondaryServerCtx &)             = delete;
    SecondaryServerCtx & operator=(const SecondaryServerCtx &) = delete;
    SecondaryServerCtx(SecondaryServerCtx &&)                  = delete;
    SecondaryServerCtx & operator=(SecondaryServerCtx &&)      = delete;
};

// Build a server_context tuned for embeddings (or, when `rank` is true,
// for cross-encoder reranking — same code path with pooling_type forced
// to RANK). Returns a heap-allocated instance with the model already
// loaded; caller spawns the loop thread. Returns nullptr on load failure
// so the caller can print a model-specific error message.
//
// Most settings track the primary ServeOptions so that --threads /
// --gpu-layers / --batch-size etc. apply uniformly. We deliberately keep
// `n_parallel = 1` on secondaries: the embed / rerank workloads are
// short and infrequent compared to LLM generation, and adding a slot
// pool there only multiplies KV memory for no throughput gain in the
// common case.
std::unique_ptr<SecondaryServerCtx> bring_up_secondary(
        const ServeOptions & opts,
        const std::string &  model_path,
        bool                 rank) {
    auto sec = std::make_unique<SecondaryServerCtx>();
    common_init();

    common_params & p = sec->params;
    p.model.path           = model_path;
    p.n_ctx                = 0;       // model's training default
    p.n_batch              = opts.n_batch;
    p.n_ubatch             = opts.n_ubatch;
    p.n_gpu_layers         = opts.gpu_layers;
    p.cpuparams.n_threads  = opts.threads;
    p.n_parallel           = 1;
    p.embedding            = true;
    p.endpoint_metrics     = false;   // metrics route is bound off primary

    if (rank) {
        // Same toggle llama-server uses for --reranking: embedding mode
        // plus the rank-head pooling type. The model must actually be a
        // reranker (typically pooling_type encoded in its GGUF metadata);
        // mismatched models will fail at decode time with a clear message.
        p.pooling_type = LLAMA_POOLING_TYPE_RANK;
    }

    // Embedding decode requires a single ubatch covers the full input;
    // clamp rather than fail. Matches the same guard on the primary.
    if (p.n_batch > p.n_ubatch) {
        p.n_batch = p.n_ubatch;
    }

    sec->ctx    = std::make_unique<server_context>();
    sec->routes = std::make_unique<server_routes>(sec->params, *sec->ctx);

    if (!sec->ctx->load_model(sec->params)) {
        return nullptr;
    }
    sec->routes->update_meta(*sec->ctx);
    return sec;
}

}  // namespace
}  // namespace chimera_serve

int command_serve(const ServeOptions & opts) {
    using namespace chimera_serve;

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

#ifdef CHIMERA_HAS_WHISPER
    // Opt-in audio. The whisper context and its serializing mutex live for
    // as long as command_serve runs; the route handler captures them by
    // reference. If --enable-audio was not passed, the context stays null
    // and we don't bind the route. With CHIMERA_HAS_WHISPER undefined the
    // whole block is compiled out, and --enable-audio doesn't even exist
    // on the CLI (see bind_serve_cmd).
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
#endif

#ifdef CHIMERA_HAS_SD
    // Opt-in image. vae_decode_only=false because /edits and /variations
    // need img2img (the VAE encode path). On txt2img-only workloads this
    // trades some memory for path uniformity. Compiled out entirely when
    // CHIMERA_HAS_SD is undefined.
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
#endif

    // Opt-in dedicated embedding model. When --enable-embeddings is set
    // a second server_context is brought up in embedding mode and bound
    // to /v1/embeddings, overriding the primary's binding. This lets the
    // main LLM stay in generative mode while still serving OpenAI-shaped
    // embedding requests.
    std::unique_ptr<SecondaryServerCtx> emb_ctx;
    if (!opts.embed_model.empty()) {
        std::cout << "chimera serve: loading dedicated embedding model "
                  << opts.embed_model << "...\n";
        emb_ctx = bring_up_secondary(opts, opts.embed_model, /*rank=*/false);
        if (!emb_ctx) {
            std::cerr << "chimera serve: failed to load embedding model: "
                      << opts.embed_model << "\n";
            ctx_http.stop();
            ctx_server.terminate();
            return static_cast<int>(ExitCode::Load);
        }
    }

    // Opt-in cross-encoder reranker. Same pattern, but the underlying
    // server_context runs with pooling_type=RANK. Bound on /v1/rerank.
    std::unique_ptr<SecondaryServerCtx> rrk_ctx;
    if (!opts.rerank_model.empty()) {
        std::cout << "chimera serve: loading rerank model "
                  << opts.rerank_model << "...\n";
        rrk_ctx = bring_up_secondary(opts, opts.rerank_model, /*rank=*/true);
        if (!rrk_ctx) {
            std::cerr << "chimera serve: failed to load rerank model: "
                      << opts.rerank_model << "\n";
            if (emb_ctx) { emb_ctx->ctx->terminate(); }
            ctx_http.stop();
            ctx_server.terminate();
            return static_cast<int>(ExitCode::Load);
        }
    }

    // Opt-in vector store / RAG (--enable-rag). One embedding model per
    // server in this cut. The chimera SQLite DB is shared with the CLI
    // (same $CHIMERA_DB / platform default); ingest + search hit it via
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

        // Optional embedding cache. Attaches to the same RAG Embedder
        // that powers /v1/vector_stores/:name/{files,search}, so both
        // ingest and search benefit. The cache owns its own SQLite
        // connection on the same DB.
        if (opts.cache_embeddings) {
            const std::string mid = chimera_embed_cache::compute_model_id(opts.rag_embedding_model);
            if (mid.empty()) {
                std::cerr << "chimera serve: --cache-embeddings: cannot fingerprint "
                          << opts.rag_embedding_model << " (unreadable); cache disabled.\n";
            } else {
                rag_ctx.embed_cache = std::make_unique<chimera_embed_cache::Cache>(
                    rag_ctx.db_path.empty() ? chimera_db::default_path() : rag_ctx.db_path,
                    mid);
                rag_ctx.embedder->set_cache(rag_ctx.embed_cache.get());
            }
        }
    }

    // Route registration. LLM route handlers are pre-built lambdas on
    // server_routes — we just bind them. See the top-of-file comment for the
    // routes we are deliberately NOT exposing.
    ctx_http.get ("/health",              ex_wrapper(routes.get_health));
    ctx_http.get ("/v1/health",           ex_wrapper(routes.get_health));
    ctx_http.get ("/v1/models",           ex_wrapper(routes.get_models));
    ctx_http.get ("/metrics",             ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",               ex_wrapper(routes.get_props));

    // KV-cache slot management. GET /slots returns slot status (gated
    // upstream by params.endpoint_slots, which we leave at its default
    // of true). POST /slots/:id_slot?action={save,restore,erase} drives
    // snapshot I/O — upstream returns "not supported" with a pointer to
    // --slot-save-path when that flag is unset, so the route is safe to
    // bind unconditionally.
    ctx_http.get ("/slots",               ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",      ex_wrapper(routes.post_slots));

    // LoRA hot-swap. GET lists the adapters loaded via --lora; POST
    // takes a JSON array of {"id": <int>, "scale": <float>} and applies
    // them to subsequent requests. Adapters must be loaded at startup
    // (the routes can only re-weight; they cannot register new files).
    // When --lora was not passed the POST handler still works against
    // the empty adapter list — sending [] is a valid no-op.
    ctx_http.get ("/lora-adapters",       ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",       ex_wrapper(routes.post_lora_adapters));

    // Chat persistence shim. When --persist-chats is set we wrap the
    // upstream chat-completions handler so every successful exchange is
    // recorded in the chats + messages tables. The wrapper handles both
    // non-streaming responses and SSE streams (by mirroring each chunk
    // into a buffer and parsing on stream end). When the
    // flag is off, we bind the upstream handler unchanged.
    ChatPersistContext chat_persist_ctx;
    ChatHistoryContext chat_hist_ctx;
    auto chat_handler = routes.post_chat_completions;  // capture by value
    if (opts.persist_chats) {
        chat_persist_ctx.db_path = opts.chat_db_path;
        chat_hist_ctx.db_path    = opts.chat_db_path;
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
    // When --enable-embeddings was passed, route /v1/embeddings to the
    // dedicated embedding context's handler instead of the primary LLM.
    // The primary's handler stays usable when --embeddings was set on
    // the LLM itself (single-model embed mode); the two flags are
    // mutually exclusive in practice — if both are set, the dedicated
    // model wins.
    ctx_http.post("/v1/embeddings",
                  ex_wrapper(emb_ctx ? emb_ctx->routes->post_embeddings_oai
                                     : routes.post_embeddings_oai));
    // /v1/responses (OpenAI Responses API). The upstream handler holds
    // conversation state in-process — it's *stateful within one chimera
    // serve invocation* but does not persist across restarts. With
    // --persist-chats on, the underlying chat-completions path will still
    // write to the chats table; the Responses API itself is layered on
    // top of that and inherits the same persistence.
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
#ifdef CHIMERA_HAS_WHISPER
    if (whisper_ctx) {
        ctx_http.post("/v1/audio/transcriptions",
                      ex_wrapper(make_audio_transcribe_handler(
                          whisper_ctx.get(), whisper_mutex, /*translate=*/false)));
        ctx_http.post("/v1/audio/translations",
                      ex_wrapper(make_audio_transcribe_handler(
                          whisper_ctx.get(), whisper_mutex, /*translate=*/true)));
    }
#endif
#ifdef CHIMERA_HAS_SD
    if (sd_ctx) {
        ctx_http.post("/v1/images/generations",
                      ex_wrapper(make_image_generations_handler(sd_ctx.get(), sd_mutex)));
        ctx_http.post("/v1/images/edits",
                      ex_wrapper(make_image_edits_handler(sd_ctx.get(), sd_mutex)));
        ctx_http.post("/v1/images/variations",
                      ex_wrapper(make_image_variations_handler(sd_ctx.get(), sd_mutex)));
    }
#endif
    // /v1/rerank takes {"query": "...", "documents": ["..."]} and returns
    // OpenAI-Cohere-shaped scored results. The handler is implemented by
    // upstream server_routes; we just bind it. Also bound on the legacy
    // /rerank for symmetry with other modalities.
    if (rrk_ctx) {
        ctx_http.post("/v1/rerank", ex_wrapper(rrk_ctx->routes->post_rerank));
        ctx_http.post("/rerank",    ex_wrapper(rrk_ctx->routes->post_rerank));
    }

    if (rag_ctx.embedder) {
        ctx_http.get ("/v1/vector_stores",              ex_wrapper(make_vs_list_handler(&rag_ctx)));
        ctx_http.post("/v1/vector_stores",              ex_wrapper(make_vs_create_handler(&rag_ctx)));
        ctx_http.get ("/v1/vector_stores/:name",        ex_wrapper(make_vs_get_handler(&rag_ctx)));
        ctx_http.post("/v1/vector_stores/:name/delete", ex_wrapper(make_vs_delete_handler(&rag_ctx)));
        ctx_http.post("/v1/vector_stores/:name/files",  ex_wrapper(make_vs_ingest_handler(&rag_ctx)));
        ctx_http.post("/v1/vector_stores/:name/search", ex_wrapper(make_vs_search_handler(&rag_ctx)));
    }

    // Chat history read endpoints. Paired with --persist-chats so we
    // only expose them when there's a write path producing the data.
    // Order matters: register the literal /v1/chats/search path BEFORE
    // /v1/chats/:id so cpp-httplib's first-match wins doesn't capture
    // "search" as an :id value.
    if (opts.persist_chats) {
        ctx_http.get("/v1/chats",         ex_wrapper(make_chats_list_handler  (&chat_hist_ctx)));
        ctx_http.get("/v1/chats/search",  ex_wrapper(make_chats_search_handler(&chat_hist_ctx)));
        ctx_http.get("/v1/chats/:id",     ex_wrapper(make_chats_get_handler   (&chat_hist_ctx)));
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

    // Secondaries must have their task loop running before the first
    // request lands. Spawning after is_ready=true means ctx_http will
    // start accepting traffic the moment the loops are up.
    if (emb_ctx) { emb_ctx->loop = std::thread([&]{ emb_ctx->ctx->start_loop(); }); }
    if (rrk_ctx) { rrk_ctx->loop = std::thread([&]{ rrk_ctx->ctx->start_loop(); }); }

    g_shutdown_handler = [&](int) {
        // Terminate the secondary loops first so they unblock and exit
        // before the primary's start_loop() returns and we begin joining.
        if (emb_ctx) emb_ctx->ctx->terminate();
        if (rrk_ctx) rrk_ctx->ctx->terminate();
        ctx_server.terminate();
    };
    install_signal_handlers();

    std::cout << "chimera serve: listening on " << ctx_http.listening_address << "\n"
              << "  LLM:   /v1/chat/completions  /v1/completions  /v1/embeddings\n"
              << "  meta:  /v1/models  /health  /metrics  /props  /slots  /lora-adapters\n"
              << "  tools: /infill  /tokenize  /detokenize  /apply-template\n"
#ifdef LLAMA_BUILD_WEBUI
              << (opts.webui
                    ? "  webui: GET /  /bundle.js  /bundle.css (built with CHIMERA_WEBUI_EMBED=ON)\n"
                    : "  webui: built in but disabled by --no-webui\n")
#endif
              << "  anthropic: /v1/messages  /v1/messages/count_tokens\n";
#ifdef CHIMERA_HAS_WHISPER
    if (whisper_ctx) std::cout << "  audio: /v1/audio/transcriptions  /v1/audio/translations\n";
#endif
#ifdef CHIMERA_HAS_SD
    if (sd_ctx)      std::cout << "  image: /v1/images/generations  /v1/images/edits  /v1/images/variations\n";
#endif
    if (rag_ctx.embedder) std::cout << "  rag:   /v1/vector_stores  /v1/vector_stores/:name{,/delete,/files,/search}\n";
    if (emb_ctx)     std::cout << "  embed: /v1/embeddings (dedicated model: " << opts.embed_model << ")\n";
    if (rrk_ctx)     std::cout << "  rerank: /v1/rerank (model: " << opts.rerank_model << ")\n";
    if (opts.persist_chats) {
        std::cout << "  persistence: --persist-chats ON (DB: "
                  << (chat_persist_ctx.db_path.empty()
                        ? chimera_db::default_path() : chat_persist_ctx.db_path)
                  << ")\n"
                  << "  chats: /v1/chats  /v1/chats/:id  /v1/chats/search\n";
    }
    if (!opts.public_path.empty()) {
        std::cout << "  webui: serving " << opts.public_path << " at GET /\n";
    }

    // Blocks on the main thread until the task queue is terminated by the
    // signal handler. Worker tasks run on threads owned by server_context;
    // HTTP requests run on threads owned by cpp-httplib inside ctx_http.
    ctx_server.start_loop();

    clean_up();
    if (ctx_http.thread.joinable()) {
        ctx_http.thread.join();
    }
    // Secondary loops were signaled to terminate in the shutdown handler;
    // wait for them to actually return before tearing down their owning
    // unique_ptrs.
    if (emb_ctx && emb_ctx->loop.joinable()) emb_ctx->loop.join();
    if (rrk_ctx && rrk_ctx->loop.joinable()) rrk_ctx->loop.join();
    return 0;
}
