#pragma once

// Library-side llama.cpp glue. Hosts the model-loading, context-creation,
// sampler, grammar, LoRA, and generation helpers that used to live in the
// CLI translation unit (src/chimera_cli/chimera.cpp). Moving them here is
// what lets `command_prompt`, `command_embed`, and `command_tokenize` ship
// inside libchimera.a (and what the optional OOP layer in chimera.hpp wraps
// on top of). CLI shell code (`command_chat`, REPL, color, spinner) still
// lives in src/chimera_cli/ and calls back into these helpers.

#include "chimera.h"

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "sampling.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// --- smart-pointer wrappers --------------------------------------------------

struct LlamaModelDeleter {
    void operator()(llama_model * model) const;
};

struct LlamaContextDeleter {
    void operator()(llama_context * ctx) const;
};

using LlamaModelPtr   = std::unique_ptr<llama_model,   LlamaModelDeleter>;
using LlamaContextPtr = std::unique_ptr<llama_context, LlamaContextDeleter>;

// mtmd handles — exposed because `command_chat` (in chimera_cli/) keeps
// these alive across turns when an mmproj is loaded.
struct MtmdContextDeleter {
    void operator()(mtmd_context * c) const;
};
struct MtmdBitmapDeleter {
    void operator()(mtmd_bitmap * b) const;
};
struct MtmdInputChunksDeleter {
    void operator()(mtmd_input_chunks * c) const;
};
using MtmdContextPtr     = std::unique_ptr<mtmd_context, MtmdContextDeleter>;
using MtmdBitmapPtr      = std::unique_ptr<mtmd_bitmap, MtmdBitmapDeleter>;
using MtmdInputChunksPtr = std::unique_ptr<mtmd_input_chunks, MtmdInputChunksDeleter>;

// --- file I/O ---------------------------------------------------------------

std::string read_file(const std::string & path);

// Resolve a prompt from either --prompt or --prompt-file. Exactly one must
// be supplied; both empty is a usage error. "-" reads stdin.
std::string resolve_prompt(const std::string & prompt, const std::string & prompt_file);

// --- tokenization -----------------------------------------------------------

std::vector<llama_token> tokenize(const llama_vocab * vocab,
                                  const std::string & text,
                                  bool                add_special,
                                  bool                parse_special);

std::string token_to_piece(const llama_vocab * vocab, llama_token token);

// --- model + context loaders ------------------------------------------------

LlamaModelPtr   load_llama_model(const LlamaCommonOptions & opts);
LlamaContextPtr new_llama_context(llama_model *              model,
                                  const LlamaCommonOptions & opts,
                                  size_t                     min_prompt_tokens);

// --- LoRA, decoding, sampling -----------------------------------------------

struct LoraAdapters {
    std::vector<llama_adapter_lora_ptr> adapters;
};

LoraAdapters load_loras(llama_model *                    model,
                        llama_context *                  ctx,
                        const std::vector<std::string> & specs);

void decode_tokens(llama_context *                ctx,
                   const std::vector<llama_token> & tokens,
                   int32_t                          n_batch);

// Sentinels match upstream defaults so unset means "leave the chain alone".
struct ReasoningBudgetParams {
    const llama_vocab * vocab = nullptr;
    std::string         thinking_start_tag;
    std::string         thinking_end_tag;
    int                 budget = -1;
    std::string         budget_message;
};

common_sampler_ptr make_sampler(const llama_model *           model,
                                const LlamaCommonOptions &    opts,
                                const ReasoningBudgetParams & rbp = {});

// --- chat-history helper ----------------------------------------------------

common_chat_msg make_chat_msg(const std::string & role, const std::string & content);

// --- generation -------------------------------------------------------------

// Streaming hook for the generation loop. Invoked once per sampled token
// with the detokenized piece (UTF-8). An empty / default-constructed
// callback means "no streaming" - the loop still runs and the full
// generated text is returned, but no per-token side effect fires.
// Callers that want stdout streaming pass a lambda that writes to cout;
// callers that want to feed a UI / pipe / logger pass their own sink.
// Trailing newlines / flushing / buffering are the caller's choice.
using TokenCallback = std::function<void(std::string_view)>;

// Sample up to n_predict tokens from a context that already has the prompt
// decoded into its KV cache. Invokes `on_token` per sampled token when set.
// Optional out_tokens receives the generated token ids in order. Exposed
// so the OOP layer's persistent-context generate() can drive its own
// prompt-decode + sample cycle without calling run_generation (which
// builds and destroys a fresh ctx every call).
std::string sample_loop(llama_context *            ctx,
                        common_sampler *           sampler,
                        const llama_vocab *        vocab,
                        int                        n_predict,
                        const TokenCallback &      on_token = {},
                        std::vector<llama_token> * out_tokens = nullptr);

std::string run_generation(llama_model *              model,
                           const LlamaCommonOptions & opts,
                           const std::string &        prompt,
                           bool                       add_special,
                           const TokenCallback &      on_token = {});

std::string run_generation_mtmd(llama_model *              model,
                                const LlamaCommonOptions & opts,
                                const std::string &        user_prompt,
                                const TokenCallback &      on_token = {});
