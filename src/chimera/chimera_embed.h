// chimera_embed.h — embedding helper shared by:
//   - command_embed             (CLI; emits one vector per call)
//   - chimera_vector_store      (ingestion; emits one vector per chunk,
//                                model loaded once and reused)
//
// The Embedder holds the loaded model + llama_context for its lifetime.
// Each embed() call tokenizes the input, decodes it through the context,
// and pulls the pooled vector. Reusing the same Embedder across many
// chunks is the entire point of this header — model load is the
// expensive step.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;

namespace chimera_embed_cache { class Cache; }

namespace chimera_embed {

struct Config {
    std::string model;                  // GGUF path
    std::string pooling = "mean";       // mean | cls | last | none
    int         threads = -1;
    int         gpu_layers = 0;
    uint32_t    n_ctx = 0;              // 0 = model's training context
    uint32_t    n_batch = 512;          // bumped to fit each chunk
    bool        normalize = true;       // L2-normalize the output vector
    bool        use_mmap = true;
};

class Embedder {
public:
    // Loads the GGUF model and creates the llama_context. Throws
    // ChimeraError(Load) on failure. The instance is move-only.
    explicit Embedder(const Config & cfg);

    Embedder(const Embedder &) = delete;
    Embedder & operator=(const Embedder &) = delete;
    Embedder(Embedder &&) noexcept;
    Embedder & operator=(Embedder &&) noexcept;
    ~Embedder();

    // Returns a single pooled vector for `text`. Length == n_embd().
    // Empty text returns an empty vector (caller decides whether to fail).
    //
    // When a cache is attached (`set_cache(...)`) and `text` is non-
    // empty, lookup happens before tokenize/decode; misses fall through
    // to the model and are written back on success.
    std::vector<float> embed(const std::string & text);

    // Attach an optional persistent cache. The Embedder does not own
    // the cache; the caller is responsible for keeping it alive at
    // least as long as this Embedder. Pass nullptr to detach.
    void set_cache(chimera_embed_cache::Cache * cache);

    // Embedding dimensionality. Stable for the lifetime of the Embedder.
    int n_embd() const;

    // Tokenize `text` with the model's vocab. `add_special` controls
    // BOS/EOS injection (whisper-style); `parse_special` controls
    // whether `<|...|>` strings in the input are interpreted as
    // special tokens. The default is what BGE / GTE / E5 expect.
    // Thread-safe: vocab is read-only post-load.
    std::vector<int> tokenize(const std::string & text,
                              bool add_special  = true,
                              bool parse_special = true) const;

    // Detokenize a token sequence back to UTF-8 text. Used by the
    // token-window chunker to materialize each chunk's text after
    // splitting in token-space.
    std::string detokenize(const std::vector<int> & tokens) const;

    // Maximum context size the embedding context was created with.
    // Token-window chunker uses this as the upper bound on chunk size.
    int n_ctx() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// One chunk produced by the token-window chunker. `text` is the
// detokenized slice; `token_count` is what the embedder will see at
// embed time (constant per chunk, equal to the window size except for
// the trailing chunk).
struct TokenChunk {
    std::string text;
    int         index;        // 0-based, monotonic per source
    int         token_count;
};

// Split `text` into overlapping chunks of approximately `chunk_tokens`
// tokens with `overlap_tokens` overlap between consecutive chunks.
// `embedder` provides the vocab used for tokenize/detokenize. Empty
// inputs return an empty vector; whitespace-only chunks are dropped.
//
// `chunk_tokens` is clamped to embedder.n_ctx() so the resulting chunks
// always fit through `embed(text)` without truncation. `overlap_tokens`
// must be in [0, chunk_tokens).
std::vector<TokenChunk> chunk_by_tokens(const std::string & text,
                                        const Embedder &    embedder,
                                        int                 chunk_tokens,
                                        int                 overlap_tokens);

// Sentence-aware chunker. Splits `text` on sentence-terminator
// punctuation (`.`, `?`, `!`) and paragraph breaks (blank lines), then
// greedily packs sentences into chunks of up to `chunk_tokens` tokens
// (measured against `embedder`'s vocab). Each subsequent chunk reuses
// the tail sentences of the previous chunk so the total overlap is at
// least `overlap_tokens` tokens (or zero if the previous chunk was
// itself shorter than the overlap).
//
// A single sentence longer than `chunk_tokens` is split mid-stream via
// `chunk_by_tokens`, which always succeeds — pathological inputs (one
// huge run-on, source code, base64 blobs) still produce usable chunks.
//
// Compared to the pure token-window splitter, this avoids cutting at
// arbitrary mid-sentence positions, which empirically improves
// retrieval quality on prose: the embedded text now corresponds to
// complete thoughts. For purely structured / non-prose input the
// behavior degenerates to the token-window splitter via the fallback.
//
// Constraints on `chunk_tokens` / `overlap_tokens` are the same as for
// `chunk_by_tokens` (overlap must be in [0, chunk_tokens); chunk_tokens
// is clamped to `embedder.n_ctx()`).
std::vector<TokenChunk> chunk_by_sentences(const std::string & text,
                                           const Embedder &    embedder,
                                           int                 chunk_tokens,
                                           int                 overlap_tokens);

}  // namespace chimera_embed
