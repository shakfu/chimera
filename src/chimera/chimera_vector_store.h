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
};

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
std::vector<Hit> search(sqlite3 *                   db,
                        const Collection &          col,
                        const std::vector<float> &  query_embedding,
                        int                         k);

}  // namespace chimera_vector_store
