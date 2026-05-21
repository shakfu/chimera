// chimera_pin_check.cpp - compile-only assertions about the upstream
// surface chimera depends on. If a llama.cpp version bump renames a
// handler, changes a constructor, drops an enum, or alters a function
// signature we use, this file fails to compile with the offending line
// pointing directly at the broken contract -- instead of a cryptic
// instantiation error deep inside chimera_serve.cpp.
//
// Rule of thumb: any time you call an upstream symbol from chimera, a
// matching `static_assert` here is cheap insurance. The file generates
// no code at runtime (everything is `static_assert` or a discarded
// `void` cast of an unused function pointer); strip(1) discards it
// entirely.

// llama.cpp surface only. whisper.h and stable-diffusion.h each drag
// in their own ggml.h whose enums collide with llama.cpp's, so the
// per-modality TUs (chimera_whisper.cpp, chimera_sd.cpp) isolate them.
// Pin-asserts for whisper / sd live alongside their call sites in
// those TUs rather than here.
#include "common.h"
#include "llama.h"
#include "server-context.h"
#include "server-http.h"

#include <type_traits>

namespace {

// ---- server_routes lambdas -------------------------------------------
//
// Every `routes.X` ctx_http.post(...) call in chimera_serve.cpp depends
// on the matching `server_http_context::handler_t` field below. The
// upstream split that broke us last week was exactly a handler_t field
// moving around; a future rename would be the same shape.
//
// To extend: when you add a `ctx_http.{get,post}("...", routes.foo)`
// in chimera_serve.cpp, drop the matching assert here.

using H = server_http_context::handler_t;

#define CHIMERA_ASSERT_HANDLER(field) \
    static_assert(std::is_same_v<decltype(server_routes::field), H>, \
                  "server_routes::" #field " is no longer a handler_t " \
                  "(upstream renamed or retyped). Fix the binding in " \
                  "chimera_serve.cpp and update this assert.")

CHIMERA_ASSERT_HANDLER(get_health);
CHIMERA_ASSERT_HANDLER(get_metrics);
CHIMERA_ASSERT_HANDLER(get_models);
CHIMERA_ASSERT_HANDLER(get_props);
CHIMERA_ASSERT_HANDLER(post_chat_completions);
CHIMERA_ASSERT_HANDLER(post_completions_oai);
CHIMERA_ASSERT_HANDLER(post_embeddings_oai);
CHIMERA_ASSERT_HANDLER(post_rerank);
CHIMERA_ASSERT_HANDLER(post_responses_oai);
CHIMERA_ASSERT_HANDLER(post_infill);
CHIMERA_ASSERT_HANDLER(post_tokenize);
CHIMERA_ASSERT_HANDLER(post_detokenize);
CHIMERA_ASSERT_HANDLER(post_apply_template);
CHIMERA_ASSERT_HANDLER(post_anthropic_messages);
CHIMERA_ASSERT_HANDLER(post_anthropic_count_tokens);
CHIMERA_ASSERT_HANDLER(get_slots);
CHIMERA_ASSERT_HANDLER(post_slots);
CHIMERA_ASSERT_HANDLER(get_lora_adapters);
CHIMERA_ASSERT_HANDLER(post_lora_adapters);

#undef CHIMERA_ASSERT_HANDLER

// ---- server_routes / server_context constructibility -----------------
//
// command_serve constructs both. If upstream changes the constructor
// signatures we want this file (not chimera_serve.cpp) to flag it.

static_assert(std::is_constructible_v<server_routes,
                                      const common_params &, server_context &>,
              "server_routes constructor changed; update command_serve.");
static_assert(std::is_default_constructible_v<server_context>,
              "server_context is no longer default-constructible; "
              "update command_serve.");
static_assert(std::is_default_constructible_v<server_http_context>,
              "server_http_context is no longer default-constructible; "
              "update command_serve.");

// ---- common_params fields ---------------------------------------------
//
// build_common_params() in chimera_serve.cpp pokes these. If upstream
// renames `embedding` or changes its type the relevant call site fails
// here first.

static_assert(std::is_same_v<decltype(common_params::embedding),     bool>,
              "common_params::embedding changed type.");
static_assert(std::is_same_v<decltype(common_params::n_ctx),         int32_t>,
              "common_params::n_ctx changed type.");
static_assert(std::is_same_v<decltype(common_params::n_batch),       int32_t>,
              "common_params::n_batch changed type.");
static_assert(std::is_same_v<decltype(common_params::n_ubatch),      int32_t>,
              "common_params::n_ubatch changed type.");
static_assert(std::is_same_v<decltype(common_params::n_gpu_layers),  int32_t>,
              "common_params::n_gpu_layers changed type.");
static_assert(std::is_same_v<decltype(common_params::n_parallel),    int32_t>,
              "common_params::n_parallel changed type.");
static_assert(std::is_same_v<decltype(common_params::pooling_type),  enum llama_pooling_type>,
              "common_params::pooling_type changed type.");

// Fields touched by build_common_params() that were unpinned in earlier
// releases — each is a llama.cpp-bump blind spot if the type or name
// shifts upstream. Asserting them here makes the failure fail-fast at
// chimera-side compile time with a chimera-specific error message,
// instead of cascading into harder-to-read errors at the call sites
// in chimera_serve.cpp.
static_assert(std::is_same_v<decltype(common_params::api_keys),
                             std::vector<std::string>>,
              "common_params::api_keys changed type "
              "(chimera serve --api-key push_back relies on this).");
static_assert(std::is_same_v<decltype(common_params::webui),         bool>,
              "common_params::webui changed type "
              "(chimera serve --no-webui flips this).");
static_assert(std::is_same_v<decltype(common_params::public_path),   std::string>,
              "common_params::public_path changed type "
              "(chimera serve --public-path assigns this).");
static_assert(std::is_same_v<decltype(common_params::slot_save_path), std::string>,
              "common_params::slot_save_path changed type "
              "(chimera serve --slot-save-path assigns + trailing-slash-normalises this).");
static_assert(std::is_same_v<decltype(common_params::lora_adapters),
                             std::vector<common_adapter_lora_info>>,
              "common_params::lora_adapters changed type "
              "(chimera serve --lora parser push_backs this).");
static_assert(std::is_same_v<decltype(common_params::model_alias),
                             std::set<std::string>>,
              "common_params::model_alias changed type "
              "(chimera serve's fall-through name copy uses .insert()).");
static_assert(std::is_same_v<decltype(common_params::kv_unified),    bool>,
              "common_params::kv_unified changed type "
              "(chimera serve sets this true on --parallel<0 auto path).");

// ---- llama pooling enum -----------------------------------------------
//
// chimera_embed::parse_pooling and bring_up_secondary in
// chimera_serve.cpp use these enumerators by name. If upstream drops or
// renames LLAMA_POOLING_TYPE_RANK (used by --reranking), here's where
// it fails.

static_assert(static_cast<int>(LLAMA_POOLING_TYPE_NONE)  == 0, "");
static_assert(static_cast<int>(LLAMA_POOLING_TYPE_MEAN)  == 1, "");
static_assert(static_cast<int>(LLAMA_POOLING_TYPE_CLS)   == 2, "");
static_assert(static_cast<int>(LLAMA_POOLING_TYPE_LAST)  == 3, "");
static_assert(static_cast<int>(LLAMA_POOLING_TYPE_RANK)  == 4, "");

// ---- function-pointer-typed signatures --------------------------------
//
// Stronger than name-presence: pin the exact prototypes of upstream C
// functions we call. Assigning &fn to a typed function pointer fails to
// compile if the signature drifts. Wrapped in a function so the
// pointers aren't ODR-emitted globals.

void chimera_pin_check_signatures() {
    // llama
    [[maybe_unused]] int32_t (*p_llama_tokenize)(
        const struct llama_vocab *, const char *, int32_t,
        llama_token *, int32_t, bool, bool) = &llama_tokenize;
    [[maybe_unused]] int32_t (*p_llama_detokenize)(
        const struct llama_vocab *, const llama_token *, int32_t,
        char *, int32_t, bool, bool) = &llama_detokenize;
    [[maybe_unused]] const struct llama_vocab * (*p_llama_model_get_vocab)(
        const struct llama_model *) = &llama_model_get_vocab;
    [[maybe_unused]] int32_t (*p_llama_model_n_embd)(
        const struct llama_model *) = &llama_model_n_embd;

    // ---- persistent-handle dependencies (chimera::Llama) ------------
    //
    // The OOP layer in chimera.hpp holds a llama_context across many
    // generate() calls and relies on these symbols staying compatible:
    //
    //   * llama_get_memory + llama_memory_clear replaced the older
    //     llama_kv_self_clear API in late 2024. If they're renamed
    //     again, Llama::reset() and Llama::generate()'s pre-decode
    //     KV-clear break. Catch the signature drift here rather than
    //     deep inside chimera.hpp's template instantiation errors.
    //
    //   * llama_init_from_model is the backbone of new_llama_context;
    //     llama_model_load_from_file is the backbone of
    //     load_llama_model. Both are called by the OOP ctor.
    //
    //   * llama_decode + llama_batch_get_one + llama_vocab_is_eog are
    //     the core loop body of sample_loop, now publicly exposed in
    //     chimera_llama.h.
    [[maybe_unused]] llama_memory_t (*p_llama_get_memory)(
        const struct llama_context *) = &llama_get_memory;
    [[maybe_unused]] void (*p_llama_memory_clear)(
        llama_memory_t, bool) = &llama_memory_clear;
    [[maybe_unused]] struct llama_context * (*p_llama_init_from_model)(
        struct llama_model *, struct llama_context_params) = &llama_init_from_model;
    [[maybe_unused]] struct llama_model * (*p_llama_model_load_from_file)(
        const char *, struct llama_model_params) = &llama_model_load_from_file;
    [[maybe_unused]] void (*p_llama_model_free)(
        struct llama_model *) = &llama_model_free;
    [[maybe_unused]] void (*p_llama_free)(
        struct llama_context *) = &llama_free;
    [[maybe_unused]] int32_t (*p_llama_decode)(
        struct llama_context *, struct llama_batch) = &llama_decode;
    [[maybe_unused]] struct llama_batch (*p_llama_batch_get_one)(
        llama_token *, int32_t) = &llama_batch_get_one;
    [[maybe_unused]] bool (*p_llama_vocab_is_eog)(
        const struct llama_vocab *, llama_token) = &llama_vocab_is_eog;

    // Whisper / SD signature pins live in their per-TU isolated
    // files (see file-top comment). Add new ones there.
}

}  // namespace
