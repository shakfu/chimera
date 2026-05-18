// chimera_serve_internal.h — shared types and factory declarations used by the
// chimera serve translation units.
//
// chimera_serve.cpp was once a single 2200-line file containing every
// /v1/* handler. The handlers naturally cluster by modality (audio,
// images, RAG, chat persistence, chat history read) and were split into
// per-modality TUs to keep each file under ~400 lines. This header
// declares the seams between the slim chimera_serve.cpp driver and the
// per-modality TUs. It is internal — never installed, never included
// outside src/chimera.

#pragma once

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

#include "server-context.h"
#include "server-http.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace chimera_serve {

using json = nlohmann::ordered_json;

// ----------------------------------------------------------------------------
// Shared helpers (defined in chimera_serve.cpp)
// ----------------------------------------------------------------------------

// Wrap a handler so it never throws out of the HTTP layer; converts
// std::invalid_argument to 400 and every other exception to 500. Mirrors
// llama-server's ex_wrapper.
server_http_context::handler_t ex_wrapper(server_http_context::handler_t func);

// ----------------------------------------------------------------------------
// JSON-field coercion helpers (defined in chimera_serve_images.cpp; used by both the
// /v1/images/* and /v1/vector_stores/:name/search handlers because multipart
// text fields arrive as strings while application/json bodies preserve
// numeric types)
// ----------------------------------------------------------------------------

int         coerce_int   (const json & v, int dflt);
int64_t     coerce_int64 (const json & v, int64_t dflt);
float       coerce_float (const json & v, float dflt);
std::string coerce_string(const json & v, const std::string & dflt = "");

// ----------------------------------------------------------------------------
// Audio — POST /v1/audio/{transcriptions,translations}. Compiled only when
// the build linked whisper.cpp; chimera_serve_audio.cpp is dropped from
// the source list when CHIMERA_HAS_WHISPER is undefined.
// ----------------------------------------------------------------------------

#ifdef CHIMERA_HAS_WHISPER
server_http_context::handler_t make_audio_transcribe_handler(
    whisper_context * ctx,
    std::mutex      & ctx_mutex,
    bool              translate);
#endif

// ----------------------------------------------------------------------------
// Images — POST /v1/images/{generations,edits,variations}. Compiled only
// when the build linked stable-diffusion.cpp; chimera_serve_images.cpp is
// dropped from the source list when CHIMERA_HAS_SD is undefined.
// ----------------------------------------------------------------------------

#ifdef CHIMERA_HAS_SD
server_http_context::handler_t make_image_generations_handler(
    sd_ctx_t * ctx, std::mutex & ctx_mutex);
server_http_context::handler_t make_image_edits_handler(
    sd_ctx_t * ctx, std::mutex & ctx_mutex);
server_http_context::handler_t make_image_variations_handler(
    sd_ctx_t * ctx, std::mutex & ctx_mutex);
#endif

// ----------------------------------------------------------------------------
// Vector store / RAG — /v1/vector_stores[/...]
// ----------------------------------------------------------------------------

// State shared by every /v1/vector_stores/* handler. Lifetime is the
// `command_serve` call; handlers capture this by pointer.
struct RagContext {
    std::string                                       db_path;
    std::unique_ptr<chimera_embed::Embedder>          embedder;
    std::mutex                                        embedder_mutex;
    std::string                                       loaded_model;  // basename or full path
    // Optional persistent embedding cache. Bound when serve was
    // invoked with --cache-embeddings; nullptr otherwise.
    std::unique_ptr<chimera_embed_cache::Cache>       embed_cache;
};

server_http_context::handler_t make_vs_list_handler  (RagContext * rag);
server_http_context::handler_t make_vs_create_handler(RagContext * rag);
server_http_context::handler_t make_vs_get_handler   (RagContext * rag);
server_http_context::handler_t make_vs_delete_handler(RagContext * rag);
server_http_context::handler_t make_vs_ingest_handler(RagContext * rag);
server_http_context::handler_t make_vs_search_handler(RagContext * rag);

// ----------------------------------------------------------------------------
// Chat persistence — wraps POST /v1/chat/completions when --persist-chats
// ----------------------------------------------------------------------------

// Per-server state for the /v1/chat/completions persistence wrapper.
// Captured by pointer in the wrapped handler.
struct ChatPersistContext {
    std::string db_path;
    std::mutex  mutex;   // serializes DB writes from concurrent HTTP workers
};

// Header name used to associate /v1/chat/completions requests with a
// persisted chat row across multiple HTTP calls.
constexpr const char * X_CHAT_ID_HEADER = "X-Chimera-Chat-Id";

server_http_context::handler_t make_persisting_chat_handler(
    server_http_context::handler_t inner,
    ChatPersistContext *           cpc);

// ----------------------------------------------------------------------------
// Chat history read — GET /v1/chats[/...]
// ----------------------------------------------------------------------------

struct ChatHistoryContext {
    std::string db_path;
};

server_http_context::handler_t make_chats_list_handler  (ChatHistoryContext * ctx);
server_http_context::handler_t make_chats_get_handler   (ChatHistoryContext * ctx);
server_http_context::handler_t make_chats_search_handler(ChatHistoryContext * ctx);

}  // namespace chimera_serve
