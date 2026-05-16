// chimera_embed.cpp — implementation of the embedding helper. See
// chimera_embed.h for the public contract.

#include "chimera_embed.h"
#include "chimera.h"   // ChimeraError, ExitCode
#include "chimera_embed_cache.h"

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
    int                  n_ctx_  = 0;
    bool                 normalize = true;
    chimera_embed_cache::Cache * cache = nullptr;  // not owned
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
    impl_->n_ctx_    = static_cast<int>(cparams.n_ctx);
    impl_->normalize = cfg.normalize;
}

Embedder::Embedder(Embedder &&) noexcept = default;
Embedder & Embedder::operator=(Embedder &&) noexcept = default;
Embedder::~Embedder() = default;

int Embedder::n_embd() const { return impl_->n_embd_; }
int Embedder::n_ctx () const { return impl_->n_ctx_;  }

std::vector<int> Embedder::tokenize(const std::string & text,
                                    bool add_special, bool parse_special) const {
    const llama_vocab * vocab = llama_model_get_vocab(impl_->model.get());
    auto toks = ::chimera_embed::tokenize(vocab, text, add_special, parse_special);
    return std::vector<int>(toks.begin(), toks.end());
}

std::string Embedder::detokenize(const std::vector<int> & tokens) const {
    if (tokens.empty()) return {};
    const llama_vocab * vocab = llama_model_get_vocab(impl_->model.get());

    // Two-call pattern: ask once for the required length, allocate,
    // then materialize. The negative return is the buffer size needed.
    std::string out(tokens.size() * 4 + 32, '\0');
    while (true) {
        const int32_t n = llama_detokenize(
            vocab, tokens.data(), static_cast<int32_t>(tokens.size()),
            out.data(), static_cast<int32_t>(out.size()),
            /*remove_special=*/true, /*unparse_special=*/false);
        if (n >= 0) {
            out.resize(static_cast<size_t>(n));
            return out;
        }
        // n is the negative of the required buffer length.
        out.resize(static_cast<size_t>(-n));
    }
}

void Embedder::set_cache(chimera_embed_cache::Cache * cache) {
    impl_->cache = cache;
}

std::vector<float> Embedder::embed(const std::string & text) {
    if (text.empty()) return {};

    if (impl_->cache) {
        std::vector<float> hit;
        if (impl_->cache->lookup(text, impl_->n_embd_, hit)) {
            return hit;
        }
    }

    const llama_vocab * vocab = llama_model_get_vocab(impl_->model.get());
    // Disambiguate from the public Embedder::tokenize member (3 args).
    auto tokens = ::chimera_embed::tokenize(
        vocab, text, /*add_special=*/true, /*parse_special=*/true);
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
    if (impl_->cache) {
        // Cache the final (normalized, if enabled) vector — that's what
        // future callers will receive on hit, so we want bit-identical
        // round-trips.
        impl_->cache->put(text, vec);
    }
    return vec;
}

std::vector<TokenChunk> chunk_by_tokens(const std::string & text,
                                        const Embedder &    embedder,
                                        int                 chunk_tokens,
                                        int                 overlap_tokens) {
    std::vector<TokenChunk> out;
    if (text.empty() || chunk_tokens <= 0) return out;
    if (overlap_tokens < 0 || overlap_tokens >= chunk_tokens) {
        fail(ExitCode::BadInput,
             "chunk overlap must be in [0, chunk_tokens) (got " +
             std::to_string(overlap_tokens) + " vs chunk_tokens=" +
             std::to_string(chunk_tokens) + ")");
    }

    // Clamp to what the embedding context can actually decode. Setting
    // chunk_tokens > n_ctx would silently truncate at embed time, so
    // refuse it here with a clear message.
    const int max_tokens = embedder.n_ctx();
    if (max_tokens > 0 && chunk_tokens > max_tokens) {
        chunk_tokens = max_tokens;
        if (overlap_tokens >= chunk_tokens) {
            overlap_tokens = chunk_tokens / 8;  // 12.5% overlap as a sane fallback
        }
    }

    // Tokenize the whole source once. add_special=false because we slice
    // mid-token-stream; BOS/EOS are model-dependent and the embed()
    // method re-adds them as needed when it tokenizes the chunk text
    // again. parse_special=false treats every byte as content.
    auto tokens = embedder.tokenize(text, /*add_special=*/false,
                                          /*parse_special=*/false);
    if (tokens.empty()) return out;

    const int n_total = static_cast<int>(tokens.size());
    const int step    = chunk_tokens - overlap_tokens;
    int idx           = 0;
    for (int start = 0; start < n_total; start += step) {
        const int end = std::min(start + chunk_tokens, n_total);
        std::vector<int> slice(tokens.begin() + start, tokens.begin() + end);

        std::string chunk_text = embedder.detokenize(slice);
        // Trim leading/trailing whitespace so KNN highlights don't show
        // ragged edges from a mid-token cut.
        auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
        chunk_text.erase(chunk_text.begin(),
                         std::find_if(chunk_text.begin(), chunk_text.end(), not_space));
        chunk_text.erase(std::find_if(chunk_text.rbegin(), chunk_text.rend(), not_space).base(),
                         chunk_text.end());
        if (!chunk_text.empty()) {
            out.push_back({std::move(chunk_text), idx++, end - start});
        }
        if (end >= n_total) break;
    }
    return out;
}

}  // namespace chimera_embed
