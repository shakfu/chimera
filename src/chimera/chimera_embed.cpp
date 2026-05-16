// chimera_embed.cpp — implementation of the embedding helper. See
// chimera_embed.h for the public contract.

#include "chimera_embed.h"
#include "chimera.h"   // ChimeraError, ExitCode

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

namespace chimera_embed {

namespace {

struct LlamaModelDeleter   { void operator()(llama_model   * m) const { if (m) llama_model_free(m); } };
struct LlamaContextDeleter { void operator()(llama_context * c) const { if (c) llama_free(c);       } };
using LlamaModelPtr   = std::unique_ptr<llama_model,   LlamaModelDeleter>;
using LlamaContextPtr = std::unique_ptr<llama_context, LlamaContextDeleter>;

enum llama_pooling_type parse_pooling(const std::string & name) {
    if (name == "mean") return LLAMA_POOLING_TYPE_MEAN;
    if (name == "cls")  return LLAMA_POOLING_TYPE_CLS;
    if (name == "last") return LLAMA_POOLING_TYPE_LAST;
    if (name == "none") return LLAMA_POOLING_TYPE_NONE;
    fail(ExitCode::BadInput, "unsupported pooling: " + name + " (use mean|cls|last|none)");
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text,
                                  bool add_special, bool parse_special) {
    std::vector<llama_token> tokens(text.size() + 8);
    int32_t n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                               tokens.data(), static_cast<int32_t>(tokens.size()),
                               add_special, parse_special);
    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()),
                           add_special, parse_special);
    }
    if (n < 0) fail(ExitCode::Generate, "failed to tokenize text for embedding");
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

}  // namespace

struct Embedder::Impl {
    LlamaModelPtr        model;
    LlamaContextPtr      ctx;
    enum llama_pooling_type ptype;
    int                  n_embd_ = 0;
    bool                 normalize = true;
};

Embedder::Embedder(const Config & cfg) : impl_(std::make_unique<Impl>()) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.gpu_layers;
    mparams.use_mmap     = cfg.use_mmap;
    impl_->model.reset(llama_model_load_from_file(cfg.model.c_str(), mparams));
    if (!impl_->model) {
        fail(ExitCode::Load, "failed to load embedding model: " + cfg.model);
    }

    const uint32_t n_train = llama_model_n_ctx_train(impl_->model.get());

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = cfg.n_ctx == 0 ? n_train : cfg.n_ctx;
    cparams.n_batch         = std::max<uint32_t>(cfg.n_batch, cparams.n_ctx);
    cparams.n_ubatch        = cparams.n_batch;
    cparams.n_threads       = cfg.threads;
    cparams.n_threads_batch = cfg.threads;
    cparams.embeddings      = true;
    cparams.pooling_type    = parse_pooling(cfg.pooling);
    cparams.no_perf         = true;

    impl_->ctx.reset(llama_init_from_model(impl_->model.get(), cparams));
    if (!impl_->ctx) {
        fail(ExitCode::Load, "failed to create llama context for embedding");
    }
    impl_->ptype     = cparams.pooling_type;
    impl_->n_embd_   = llama_model_n_embd(impl_->model.get());
    impl_->normalize = cfg.normalize;
}

Embedder::Embedder(Embedder &&) noexcept = default;
Embedder & Embedder::operator=(Embedder &&) noexcept = default;
Embedder::~Embedder() = default;

int Embedder::n_embd() const { return impl_->n_embd_; }

std::vector<float> Embedder::embed(const std::string & text) {
    if (text.empty()) return {};

    const llama_vocab * vocab = llama_model_get_vocab(impl_->model.get());
    auto tokens = tokenize(vocab, text, /*add_special=*/true, /*parse_special=*/true);
    if (tokens.empty()) {
        fail(ExitCode::BadInput, "input tokenized to zero tokens");
    }

    // Each call processes a single sequence; clear any leftover KV from
    // the previous call so the new tokens get a fresh sequence id.
    llama_memory_seq_rm(llama_get_memory(impl_->ctx.get()), 0, 0, -1);

    if (llama_decode(impl_->ctx.get(),
                     llama_batch_get_one(tokens.data(),
                                         static_cast<int32_t>(tokens.size()))) != 0) {
        fail(ExitCode::Generate, "failed to decode input for embedding");
    }

    const float * emb = nullptr;
    if (impl_->ptype == LLAMA_POOLING_TYPE_NONE) {
        emb = llama_get_embeddings(impl_->ctx.get());
    } else {
        emb = llama_get_embeddings_seq(impl_->ctx.get(), 0);
        if (emb == nullptr) {
            emb = llama_get_embeddings_ith(
                impl_->ctx.get(), static_cast<int32_t>(tokens.size()) - 1);
        }
    }
    if (emb == nullptr) {
        fail(ExitCode::Generate, "no embeddings produced (model may not support pooling)");
    }

    std::vector<float> vec(emb, emb + impl_->n_embd_);
    if (impl_->normalize) {
        double norm_sq = 0.0;
        for (float v : vec) norm_sq += static_cast<double>(v) * v;
        const float norm = static_cast<float>(std::sqrt(norm_sq));
        if (norm > 0.0f) {
            for (float & v : vec) v /= norm;
        }
    }
    return vec;
}

}  // namespace chimera_embed
