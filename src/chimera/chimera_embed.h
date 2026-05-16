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
    std::vector<float> embed(const std::string & text);

    // Embedding dimensionality. Stable for the lifetime of the Embedder.
    int n_embd() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace chimera_embed
