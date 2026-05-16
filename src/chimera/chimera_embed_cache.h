// chimera_embed_cache.h — optional persistent memoization for
// `chimera_embed::Embedder::embed(text) -> vector`.
//
// Used by:
//   - command_embed (`chimera embed --cache-embeddings`)
//   - command_index_ingest (`chimera index ingest --cache-embeddings`)
//   - command_search (`chimera search --cache-embeddings`)
//   - command_serve (`chimera serve --cache-embeddings`), the RAG path
//
// Storage: the `embedding_cache` table (migration v2), keyed on
//   (model_id, sha256(text))
// and indexed for an `older-than` prune.
//
// The cache is intentionally optional and opt-in: it grows the DB at a
// rate of `dim*4 + sha + overhead` bytes per unique embedding. A 384-dim
// bge-small float32 vector is ~1.5 kB on disk; 100k cached entries =
// ~150 MB. Default OFF; users pass --cache-embeddings when they expect
// repeated work (re-ingesting partially-updated corpora, hammering the
// same query against /v1/vector_stores/:name/search with varying k).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace chimera_embed_cache {

// Compute a stable, fast fingerprint of a model file (typically GGUF).
// Hashes: file size || first 64 KB || last 64 KB. Catches re-quant /
// re-train / arch swaps because GGUF stores metadata in the header
// (architecture, vocab, tensor names, quant types); the tail-bytes
// capture protects against tail-only edits, which don't happen in
// practice (no tool re-writes a GGUF's tail without touching its
// header). Returns lowercase hex; empty on read failure.
//
// Cheap by design: a 4 GB model takes a few milliseconds to fingerprint
// because we never read the body. Cached per-process by the Cache
// constructor so repeated Embedders pointed at the same model only pay
// once.
std::string compute_model_id(const std::string & model_path);

// SHA-256 of a UTF-8 string, returned as 32 raw bytes. Used as the
// text key in embedding_cache. Exposed for tests; production callers
// go through `Cache::lookup` / `Cache::put`.
std::vector<unsigned char> sha256_bytes(const std::string & text);

class Cache {
public:
    // Open (or create + migrate) the SQLite database at `db_path` and
    // bind the cache to `model_id`. Thread-safe-ish: SQLite WAL mode
    // permits concurrent readers; writers serialize per-connection.
    // Throws ChimeraError(Runtime) on DB open / migration failure.
    Cache(const std::string & db_path, std::string model_id);

    Cache(const Cache &)             = delete;
    Cache & operator=(const Cache &) = delete;
    Cache(Cache &&) noexcept;
    Cache & operator=(Cache &&) noexcept;
    ~Cache();

    // Returns true on hit (and fills `vec`); false on miss. If a
    // cached entry's `dim` doesn't match `expected_dim` the row is
    // treated as a miss and silently overwritten on the next `put` —
    // that's the case where the user pointed --cache-embeddings at the
    // wrong DB / model_id collided / pooling changed.
    bool lookup(const std::string & text, int expected_dim,
                std::vector<float> & out);

    // Insert or replace the cache entry for `text`. No-op on empty vec.
    void put(const std::string & text, const std::vector<float> & vec);

    // Total cached rows for this model_id. For status output.
    int64_t count() const;

    const std::string & model_id() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace chimera_embed_cache
