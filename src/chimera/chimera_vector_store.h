// chimera_vector_store.h — vector-store operations on the SQLite +
// sqlite-vec backing store.
//
// Schema lives in chimera_db.cpp's v1 migration: `collections` +
// `documents` are static tables; per-collection vec0 tables (`vec_<id>`)
// are created on demand by `create_collection`. The dim is fixed at
// create time and stored in `collections.dim` so we can validate that
// re-ingested data matches.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct sqlite3;

namespace chimera_vector_store {

struct Collection {
    int64_t     id;
    std::string name;
    std::string embedding_model;   // e.g. "bge-small-en-v1.5-q8_0.gguf"
    int         dim;
    int64_t     created_at;
    int64_t     doc_count;          // populated by list()/stats()

    // v3 columns. `distance` is the sqlite-vec metric (`cosine | l2 |
    // l1`) used by this collection's vec0 table; pre-v3 rows backfill to
    // 'cosine'. `chunk_tokens` and `chunk_overlap` are the defaults
    // `chimera index ingest` uses when the user doesn't override on the
    // command line.
    std::string distance      = "cosine";
    int         chunk_tokens  = 512;
    int         chunk_overlap = 64;
};

// Allowed values for Collection::distance. Validated at create-time;
// rejection is the caller's responsibility.
bool is_valid_distance(const std::string & d);

struct Hit {
    int64_t     document_id;
    std::string source_uri;
    int         chunk_index;
    std::string text;
    double      distance;           // smaller is closer (cosine distance)

    // Populated by hybrid search; -1 means "this index did not surface
    // this hit". Semantic and lexical ranks are 0-based; the fused
    // score is reciprocal-rank-fusion (larger is better, in [0, 1]).
    int         semantic_rank = -1;
    int         lexical_rank  = -1;
    double      rrf_score     = 0.0;
};

// Search mode passed to `search()`. The default at the API surface is
// Hybrid: it is strictly better than Semantic on typical English text
// (it adds keyword recall without losing semantic recall) and the cost
// is one extra SELECT against the FTS5 index. Callers wanting the
// pre-hybrid behavior pass Semantic explicitly.
enum class SearchMode {
    Semantic,   // vec0 cosine/l2/l1 KNN only.
    Lexical,    // FTS5 over documents.text only.
    Hybrid,     // reciprocal-rank fusion of the two.
};

// String <-> enum conversions for the CLI/API surface.
std::optional<SearchMode> parse_search_mode(const std::string & s);
const char *              search_mode_name(SearchMode m);

// --- collection lifecycle ----------------------------------------------

// Create a collection, including the per-collection `vec_<id>` virtual
// table sized to `dim`. Throws ChimeraError(BadInput) if a collection
// with this name already exists.
struct CreateOptions {
    std::string distance      = "cosine";  // 'cosine' | 'l2' | 'l1'
    int         chunk_tokens  = 512;
    int         chunk_overlap = 64;
};

Collection create(sqlite3 *           db,
                  const std::string & name,
                  const std::string & embedding_model,
                  int                 dim,
                  const CreateOptions & opts = {});

// Drop a collection, its documents, and its per-collection vec0 table.
// Throws if no such collection exists.
void drop(sqlite3 * db, const std::string & name);

// Look up a collection by name. Returns std::nullopt if not found.
std::optional<Collection> find(sqlite3 * db, const std::string & name);

// List all collections, sorted by name. `doc_count` is populated per row.
std::vector<Collection> list(sqlite3 * db);

// --- ingestion ---------------------------------------------------------

struct DocumentInput {
    std::string        source_uri;   // optional, e.g. file path
    int                chunk_index;  // 0-indexed within source
    std::string        text;
    std::vector<float> embedding;    // length must equal collection.dim
    int                token_count = 0;  // optional, informational
};

// Insert one document + its embedding inside a transaction. The vec0
// row uses the same rowid as the `documents` row so KNN results join
// back trivially. For batch ingestion, wrap multiple calls in your own
// BEGIN/COMMIT for higher throughput.
int64_t insert_document(sqlite3 * db, const Collection & col, const DocumentInput & doc);

// --- search ------------------------------------------------------------

// Top-k nearest neighbors of `query_embedding` within `col`. Results are
// sorted by `distance` ascending (closest first).
//
// Kept as the no-mode overload for callers (and tests) that only need
// semantic KNN. Equivalent to `search(db, col, embedding, "", k,
// SearchMode::Semantic)`.
std::vector<Hit> search(sqlite3 *                   db,
                        const Collection &          col,
                        const std::vector<float> &  query_embedding,
                        int                         k);

// Mode-aware search. `query_text` is required for SearchMode::Lexical
// and SearchMode::Hybrid (used as the FTS5 MATCH expression);
// `query_embedding` is required for SearchMode::Semantic and
// SearchMode::Hybrid. Empty inputs for the unused side are tolerated.
//
// For Hybrid, both indexes are queried with `k_internal = max(k, 30)`
// and merged by reciprocal-rank fusion (RRF score
// `Σ 1 / (60 + rank_i)`). The returned vector is sorted by
// `rrf_score` descending and truncated to k.
//
// FTS5 syntax errors in `query_text` (rare for prose, common for
// user-typed query strings containing parens / quotes) are caught and
// fall back to a phrase-quoted form ("\"<text>\"") so search still
// returns results rather than HTTP 500.
std::vector<Hit> search(sqlite3 *                   db,
                        const Collection &          col,
                        const std::vector<float> &  query_embedding,
                        const std::string &         query_text,
                        int                         k,
                        SearchMode                  mode);

}  // namespace chimera_vector_store
