#pragma once

// Optional C++ object-oriented layer over libchimera.
//
// libchimera exposes a procedural surface: build an Options struct, call
// `command_*(opts)`, get an exit code. This header wraps that surface in
// RAII classes so C++ consumers can treat chimera's capabilities as
// long-lived objects (load a model once, call .generate() many times)
// rather than as one-shot commands.
//
// Design notes:
//   * Header-only. Consumers link libchimera.a as before; no extra TU.
//   * Persistent-handle for Llama (load_model in ctor, reused across calls).
//   * Options-in-ctor for Whisper / SD / Server. Their underlying
//     `command_*` functions own the full load -> run -> unload lifecycle
//     today; persistent-handle would require library-side refactors that
//     are out of scope for this first cut. The wrappers therefore add no
//     state-reuse benefit over the C-style API, only API consistency.
//   * Each class hands out a `raw()` accessor to the underlying handle
//     for callers that need to drop down to the C API.
//
// Example:
//
//   chimera::Llama llm({.model = "Qwen3-1.7B-Q4_0.gguf", .n_predict = 128});
//   auto reply = llm.generate("What is the capital of France?");
//
//   chimera::Embedder emb({.model = "bge-small.gguf"});
//   auto vec = emb.embed("hello world");
//
//   chimera::Server srv({.model = "Qwen3-1.7B-Q4_0.gguf", .port = 8080});
//   return srv.run();
//
// What this header does NOT yet wrap:
//   * `command_chat` (interactive REPL) — still lives in src/chimera_cli/
//     because it owns terminal I/O, signal handling, and color streaming.
//     A future cut may carve out a stateless `LlamaChat` class from the
//     REPL shell; until then, callers wanting chat semantics should drive
//     `Llama::generate` against their own pre-rendered chat-templated
//     prompts (or talk to a `chimera::Server`).

#include "chimera.h"
#include "chimera_db.h"
#include "chimera_embed.h"
#include "chimera_embed_cache.h"
#include "chimera_llama.h"
#ifdef CHIMERA_HAS_SD
#include "chimera_sd.h"
#endif
#ifdef CHIMERA_HAS_WHISPER
#include "chimera_whisper.h"
#endif

#include "llama.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace chimera {

// -------------------------------------------------------------------------
// Llama -- persistent-handle wrapper over libchimera's text-generation glue.
//
// The model is loaded by the constructor and freed by the destructor; each
// `generate()` call builds a fresh `llama_context` against that model, so
// generation parameters in `options()` can be mutated between calls without
// reloading the (potentially multi-gigabyte) weights. Multimodal prompts
// (set `options().images`) automatically route through the mtmd pathway.
// -------------------------------------------------------------------------
class Llama {
public:
    explicit Llama(LlamaCommonOptions opts)
        : opts_(std::move(opts)),
          model_(load_llama_model(opts_)) {}

    // Run one generation against the persistent model. Returns the
    // generated text. When `stream` is true the tokens are also written
    // to stdout as they're sampled (matches the CLI's `gen` behavior).
    std::string generate(const std::string & prompt, bool stream = false) {
        if (!opts_.images.empty()) {
            if (opts_.mmproj.empty()) {
                fail(ExitCode::BadInput, "Llama::generate: images set but mmproj is empty");
            }
            return run_generation_mtmd(model_.get(), opts_, prompt, stream);
        }
        return run_generation(model_.get(), opts_, prompt, /*add_special=*/true, stream);
    }

    // Mutable accessor: tweak per-call parameters (n_predict, temp, ...)
    // between generate() calls without reloading the model.
    LlamaCommonOptions &       options()       { return opts_; }
    const LlamaCommonOptions & options() const { return opts_; }

    // Vocabulary-side helpers.
    std::vector<llama_token> tokenize(const std::string & text,
                                      bool add_special   = true,
                                      bool parse_special = true) const {
        return ::tokenize(llama_model_get_vocab(model_.get()),
                          text, add_special, parse_special);
    }

    std::string detokenize(llama_token token) const {
        return token_to_piece(llama_model_get_vocab(model_.get()), token);
    }

    std::string detokenize(const std::vector<llama_token> & tokens) const {
        const llama_vocab * vocab = llama_model_get_vocab(model_.get());
        std::string out;
        for (llama_token t : tokens) out += token_to_piece(vocab, t);
        return out;
    }

    // Drop down to the C handle for callers that need it (e.g. to feed
    // a tokens vector through a custom sampler).
    llama_model *       raw()       { return model_.get(); }
    const llama_model * raw() const { return model_.get(); }

private:
    LlamaCommonOptions opts_;
    LlamaModelPtr      model_;
};

// -------------------------------------------------------------------------
// Embedder -- wraps chimera_embed::Embedder against the chimera-CLI-style
// EmbedOptions surface. Persistent: the model loads once and the same
// instance can embed many strings.
// -------------------------------------------------------------------------
class Embedder {
public:
    explicit Embedder(EmbedOptions opts)
        : opts_(std::move(opts)), inner_(make_config(opts_)) {
        if (opts_.cache_embeddings) {
            const std::string mid = chimera_embed_cache::compute_model_id(opts_.model);
            if (mid.empty()) {
                fail(ExitCode::BadInput,
                     "Embedder: cannot fingerprint model file: " + opts_.model);
            }
            cache_ = std::make_unique<chimera_embed_cache::Cache>(
                opts_.cache_db.empty() ? chimera_db::default_path() : opts_.cache_db,
                mid);
            inner_.set_cache(cache_.get());
        }
    }

    std::vector<float> embed(const std::string & text) {
        return inner_.embed(text);
    }

    // Convenience: run a one-shot embed across many strings, returning
    // one vector per input in the same order.
    std::vector<std::vector<float>> embed_many(const std::vector<std::string> & texts) {
        std::vector<std::vector<float>> out;
        out.reserve(texts.size());
        for (const auto & t : texts) out.push_back(inner_.embed(t));
        return out;
    }

    const EmbedOptions & options() const { return opts_; }

private:
    static chimera_embed::Config make_config(const EmbedOptions & o) {
        chimera_embed::Config c;
        c.model            = o.model;
        c.pooling          = o.pooling;
        c.attention        = o.attention;
        c.threads          = o.threads;
        c.gpu_layers       = o.gpu_layers;
        c.n_ctx            = o.n_ctx;
        c.n_batch          = o.n_batch;
        c.n_ubatch         = o.n_ubatch;
        c.normalize        = o.normalize;
        c.use_mmap         = o.use_mmap;
        c.use_mlock        = o.use_mlock;
        c.flash_attn       = o.flash_attn;
        c.rope_freq_base   = o.rope_freq_base;
        c.rope_freq_scale  = o.rope_freq_scale;
        c.rope_scaling     = o.rope_scaling;
        c.yarn_orig_ctx    = o.yarn_orig_ctx;
        c.yarn_ext_factor  = o.yarn_ext_factor;
        c.yarn_attn_factor = o.yarn_attn_factor;
        c.yarn_beta_fast   = o.yarn_beta_fast;
        c.yarn_beta_slow   = o.yarn_beta_slow;
        c.main_gpu         = o.main_gpu;
        c.tensor_split     = o.tensor_split;
        c.split_mode       = o.split_mode;
        c.devices          = o.devices;
        return c;
    }

    EmbedOptions                                opts_;
    chimera_embed::Embedder                     inner_;
    std::unique_ptr<chimera_embed_cache::Cache> cache_;
};

#ifdef CHIMERA_HAS_WHISPER
// -------------------------------------------------------------------------
// Whisper -- options-in-ctor wrapper. `run()` performs the full load -> run
// -> unload cycle each call (mirrors the CLI subcommand). Persistent-handle
// would require carving a long-lived model holder out of chimera_whisper;
// not done in this cut.
// -------------------------------------------------------------------------
class Whisper {
public:
    explicit Whisper(WhisperOptions opts) : opts_(std::move(opts)) {}

    int run() { return command_whisper(opts_); }

    // Convenience: set the input path and run.
    int run(const std::string & input) {
        opts_.input = input;
        return command_whisper(opts_);
    }

    WhisperOptions &       options()       { return opts_; }
    const WhisperOptions & options() const { return opts_; }

private:
    WhisperOptions opts_;
};
#endif // CHIMERA_HAS_WHISPER

#ifdef CHIMERA_HAS_SD
// -------------------------------------------------------------------------
// SD -- stable-diffusion.cpp wrapper. Same caveat as Whisper: full lifecycle
// per `run()` call. Options-in-ctor.
// -------------------------------------------------------------------------
class SD {
public:
    explicit SD(SdOptions opts) : opts_(std::move(opts)) {}

    int run() { return command_sd(opts_); }

    // Convenience: set prompt and (optionally) output path, then run.
    int run(const std::string & prompt) {
        opts_.prompt = prompt;
        return command_sd(opts_);
    }

    SdOptions &       options()       { return opts_; }
    const SdOptions & options() const { return opts_; }

private:
    SdOptions opts_;
};
#endif // CHIMERA_HAS_SD

// -------------------------------------------------------------------------
// Server -- OpenAI-compatible HTTP server. `run()` blocks until the server
// shuts down (Ctrl-C or programmatic stop). Options-in-ctor; the server's
// internal lifecycle (model load + httplib listen + shutdown) is owned by
// command_serve, which makes the persistent-handle distinction moot here.
// -------------------------------------------------------------------------
class Server {
public:
    explicit Server(ServeOptions opts) : opts_(std::move(opts)) {}

    int run() { return command_serve(opts_); }

    ServeOptions &       options()       { return opts_; }
    const ServeOptions & options() const { return opts_; }

private:
    ServeOptions opts_;
};

// -------------------------------------------------------------------------
// Tokenizer -- minimal wrapper around `command_tokenize`'s primitives. Loads
// a model just for its vocab; useful when callers want token ids without
// pulling in the full Llama generation surface.
// -------------------------------------------------------------------------
class Tokenizer {
public:
    explicit Tokenizer(const std::string & model_path, bool use_mmap = true) {
        LlamaCommonOptions load_opts;
        load_opts.model      = model_path;
        load_opts.gpu_layers = 0;
        load_opts.use_mmap   = use_mmap;
        model_ = load_llama_model(load_opts);
    }

    std::vector<llama_token> encode(const std::string & text,
                                    bool add_special   = true,
                                    bool parse_special = true) const {
        return ::tokenize(llama_model_get_vocab(model_.get()),
                          text, add_special, parse_special);
    }

    std::string decode(llama_token token) const {
        return token_to_piece(llama_model_get_vocab(model_.get()), token);
    }

    std::string decode(const std::vector<llama_token> & tokens) const {
        const llama_vocab * vocab = llama_model_get_vocab(model_.get());
        std::string out;
        for (llama_token t : tokens) out += token_to_piece(vocab, t);
        return out;
    }

    const llama_vocab * vocab() const { return llama_model_get_vocab(model_.get()); }

private:
    LlamaModelPtr model_;
};

} // namespace chimera
