// chimera_vector_store.cpp — implementation of the vector-store API.
//
// Storage layout (see doc/dev/sqlite.md § 5.2):
//   - `collections`   metadata for one logical corpus.
//   - `documents`     one row per ingested chunk (text + provenance).
//   - `vec_<id>`      per-collection vec0 virtual table. Created on demand
//                     in `create()`; sized to the collection's `dim`. Rowids
//                     are kept in sync with `documents.id` so KNN results
//                     join trivially.
//
// sqlite-vec accepts vectors as either a JSON string ("[1.0, 0.0, ...]")
// or a raw little-endian float blob. We bind via blob: cheaper, no
// stringification, and the prepared-statement path can reuse the buffer.

#include "chimera_vector_store.h"
#include "chimera.h"      // ChimeraError, ExitCode
#include "chimera_db.h"

#include "sqlite3.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace chimera_vector_store {

namespace {

int64_t now_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

[[noreturn]] void sqlite_throw(sqlite3 * db, const std::string & ctx) {
    fail(ExitCode::Runtime, ctx + ": " + sqlite3_errmsg(db));
}

void exec(sqlite3 * db, const std::string & sql) {
    char * err = nullptr;
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
        const std::string msg = err ? err : sqlite3_errmsg(db);
        if (err) sqlite3_free(err);
        fail(ExitCode::Runtime, "sqlite exec failed: " + msg + " (SQL: " + sql + ")");
    }
}

// RAII for sqlite3_stmt so the inevitable error path always finalizes.
struct StmtGuard {
    sqlite3_stmt * stmt = nullptr;
    ~StmtGuard() { if (stmt) sqlite3_finalize(stmt); }
    sqlite3_stmt * get() { return stmt; }
    sqlite3_stmt ** out() { return &stmt; }
};

std::string vec_table_name(int64_t collection_id) {
    return "vec_" + std::to_string(collection_id);
}

void bind_vector(sqlite3_stmt * stmt, int index, const std::vector<float> & v) {
    // SQLITE_TRANSIENT: sqlite makes its own copy of the buffer.
    if (sqlite3_bind_blob(stmt, index, v.data(),
                          static_cast<int>(v.size() * sizeof(float)),
                          SQLITE_TRANSIENT) != SQLITE_OK) {
        fail(ExitCode::Runtime, "failed to bind vector blob");
    }
}

// Decode a `SELECT id, name, embedding_model, dim, created_at,
// distance, chunk_tokens, chunk_overlap[, doc_count]` row. The optional
// doc_count column is filled by the caller (list()) — row_to_collection
// itself doesn't read it.
Collection row_to_collection(sqlite3_stmt * stmt) {
    Collection c;
    c.id              = sqlite3_column_int64(stmt, 0);
    c.name            = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
    c.embedding_model = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2));
    c.dim             = sqlite3_column_int(stmt, 3);
    c.created_at      = sqlite3_column_int64(stmt, 4);
    c.doc_count       = 0;
    if (const auto * d = sqlite3_column_text(stmt, 5)) {
        c.distance = reinterpret_cast<const char *>(d);
    }
    c.chunk_tokens  = sqlite3_column_int(stmt, 6);
    c.chunk_overlap = sqlite3_column_int(stmt, 7);
    return c;
}

}  // namespace

bool is_valid_distance(const std::string & d) {
    return d == "cosine" || d == "l2" || d == "l1";
}

std::optional<SearchMode> parse_search_mode(const std::string & s) {
    if (s == "semantic") return SearchMode::Semantic;
    if (s == "lexical")  return SearchMode::Lexical;
    if (s == "hybrid")   return SearchMode::Hybrid;
    return std::nullopt;
}

const char * search_mode_name(SearchMode m) {
    switch (m) {
        case SearchMode::Semantic: return "semantic";
        case SearchMode::Lexical:  return "lexical";
        case SearchMode::Hybrid:   return "hybrid";
    }
    return "?";
}

// =======================================================================
// create / drop / find / list
// =======================================================================

Collection create(sqlite3 *           db,
                  const std::string & name,
                  const std::string & embedding_model,
                  int                 dim,
                  const CreateOptions & opts) {
    if (name.empty()) {
        fail(ExitCode::BadInput, "collection name must not be empty");
    }
    if (dim <= 0) {
        fail(ExitCode::BadInput, "collection dim must be positive (got " +
                                  std::to_string(dim) + ")");
    }
    if (!is_valid_distance(opts.distance)) {
        fail(ExitCode::BadInput,
             "invalid distance metric: '" + opts.distance +
             "' (expected one of: cosine, l2, l1)");
    }
    if (opts.chunk_tokens <= 0) {
        fail(ExitCode::BadInput,
             "chunk_tokens must be positive (got " +
             std::to_string(opts.chunk_tokens) + ")");
    }
    if (opts.chunk_overlap < 0 || opts.chunk_overlap >= opts.chunk_tokens) {
        fail(ExitCode::BadInput,
             "chunk_overlap must be in [0, chunk_tokens) (got " +
             std::to_string(opts.chunk_overlap) + " vs chunk_tokens=" +
             std::to_string(opts.chunk_tokens) + ")");
    }
    if (find(db, name)) {
        fail(ExitCode::BadInput, "collection already exists: " + name);
    }

    exec(db, "BEGIN");
    try {
        StmtGuard ins;
        const char * sql =
            "INSERT INTO collections "
            "  (name, embedding_model, dim, created_at, "
            "   distance, chunk_tokens, chunk_overlap) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)";
        if (sqlite3_prepare_v2(db, sql, -1, ins.out(), nullptr) != SQLITE_OK) {
            sqlite_throw(db, "prepare(create collection)");
        }
        sqlite3_bind_text (ins.get(), 1, name.c_str(),            -1, SQLITE_TRANSIENT);
        sqlite3_bind_text (ins.get(), 2, embedding_model.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int  (ins.get(), 3, dim);
        sqlite3_bind_int64(ins.get(), 4, now_seconds());
        sqlite3_bind_text (ins.get(), 5, opts.distance.c_str(),   -1, SQLITE_TRANSIENT);
        sqlite3_bind_int  (ins.get(), 6, opts.chunk_tokens);
        sqlite3_bind_int  (ins.get(), 7, opts.chunk_overlap);
        if (sqlite3_step(ins.get()) != SQLITE_DONE) {
            sqlite_throw(db, "step(create collection)");
        }
        const int64_t id = sqlite3_last_insert_rowid(db);

        // Per-collection vec0 table. `dim` and `distance` are
        // interpolated, not bound, because CREATE VIRTUAL TABLE doesn't
        // accept parameters. Both are server-controlled (dim validated
        // above; distance is one of cosine/l2/l1 by is_valid_distance)
        // so there's no injection concern.
        const std::string create_vec =
            "CREATE VIRTUAL TABLE " + vec_table_name(id) +
            " USING vec0(embedding FLOAT[" + std::to_string(dim) +
            "] distance_metric=" + opts.distance + ")";
        exec(db, create_vec);

        exec(db, "COMMIT");

        Collection c;
        c.id              = id;
        c.name            = name;
        c.embedding_model = embedding_model;
        c.dim             = dim;
        c.created_at      = now_seconds();
        c.doc_count       = 0;
        c.distance        = opts.distance;
        c.chunk_tokens    = opts.chunk_tokens;
        c.chunk_overlap   = opts.chunk_overlap;
        return c;
    } catch (...) {
        sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
        throw;
    }
}

void drop(sqlite3 * db, const std::string & name) {
    auto col = find(db, name);
    if (!col) {
        fail(ExitCode::BadInput, "no such collection: " + name);
    }
    exec(db, "BEGIN");
    try {
        // documents are FK-CASCADE-linked to collections, so deleting the
        // collection row also removes its documents. The vec0 table is
        // independent and must be dropped explicitly.
        exec(db, "DROP TABLE IF EXISTS " + vec_table_name(col->id));

        StmtGuard del;
        if (sqlite3_prepare_v2(db, "DELETE FROM collections WHERE id = ?",
                               -1, del.out(), nullptr) != SQLITE_OK) {
            sqlite_throw(db, "prepare(drop collection)");
        }
        sqlite3_bind_int64(del.get(), 1, col->id);
        if (sqlite3_step(del.get()) != SQLITE_DONE) {
            sqlite_throw(db, "step(drop collection)");
        }
        exec(db, "COMMIT");
    } catch (...) {
        sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
        throw;
    }
}

std::optional<Collection> find(sqlite3 * db, const std::string & name) {
    StmtGuard q;
    const char * sql =
        "SELECT id, name, embedding_model, dim, created_at, "
        "       distance, chunk_tokens, chunk_overlap "
        "FROM collections WHERE name = ?";
    if (sqlite3_prepare_v2(db, sql, -1, q.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(find collection)");
    }
    sqlite3_bind_text(q.get(), 1, name.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(q.get()) != SQLITE_ROW) {
        return std::nullopt;
    }
    return row_to_collection(q.get());
}

std::vector<Collection> list(sqlite3 * db) {
    std::vector<Collection> out;
    StmtGuard q;
    const char * sql =
        "SELECT c.id, c.name, c.embedding_model, c.dim, c.created_at, "
        "       c.distance, c.chunk_tokens, c.chunk_overlap, "
        "       COALESCE((SELECT COUNT(*) FROM documents d WHERE d.collection_id = c.id), 0) "
        "FROM collections c ORDER BY c.name";
    if (sqlite3_prepare_v2(db, sql, -1, q.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(list collections)");
    }
    while (sqlite3_step(q.get()) == SQLITE_ROW) {
        Collection c = row_to_collection(q.get());
        c.doc_count  = sqlite3_column_int64(q.get(), 8);
        out.push_back(std::move(c));
    }
    return out;
}

// =======================================================================
// insert_document
// =======================================================================

int64_t insert_document(sqlite3 * db, const Collection & col, const DocumentInput & doc) {
    if (static_cast<int>(doc.embedding.size()) != col.dim) {
        fail(ExitCode::BadInput,
             "embedding dim mismatch: collection '" + col.name + "' expects " +
             std::to_string(col.dim) + ", got " +
             std::to_string(doc.embedding.size()));
    }

    exec(db, "BEGIN");
    try {
        StmtGuard ins_doc;
        const char * sql_doc =
            "INSERT INTO documents "
            "  (collection_id, source_uri, chunk_index, text, token_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)";
        if (sqlite3_prepare_v2(db, sql_doc, -1, ins_doc.out(), nullptr) != SQLITE_OK) {
            sqlite_throw(db, "prepare(insert document)");
        }
        sqlite3_bind_int64(ins_doc.get(), 1, col.id);
        if (doc.source_uri.empty()) {
            sqlite3_bind_null(ins_doc.get(), 2);
        } else {
            sqlite3_bind_text(ins_doc.get(), 2, doc.source_uri.c_str(),
                              -1, SQLITE_TRANSIENT);
        }
        sqlite3_bind_int (ins_doc.get(), 3, doc.chunk_index);
        sqlite3_bind_text(ins_doc.get(), 4, doc.text.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int (ins_doc.get(), 5, doc.token_count);
        sqlite3_bind_int64(ins_doc.get(), 6, now_seconds());
        if (sqlite3_step(ins_doc.get()) != SQLITE_DONE) {
            sqlite_throw(db, "step(insert document)");
        }
        const int64_t doc_id = sqlite3_last_insert_rowid(db);

        StmtGuard ins_vec;
        const std::string sql_vec =
            "INSERT INTO " + vec_table_name(col.id) +
            "(rowid, embedding) VALUES (?, ?)";
        if (sqlite3_prepare_v2(db, sql_vec.c_str(), -1, ins_vec.out(), nullptr) != SQLITE_OK) {
            sqlite_throw(db, "prepare(insert vec row)");
        }
        sqlite3_bind_int64(ins_vec.get(), 1, doc_id);
        bind_vector(ins_vec.get(), 2, doc.embedding);
        if (sqlite3_step(ins_vec.get()) != SQLITE_DONE) {
            sqlite_throw(db, "step(insert vec row)");
        }
        exec(db, "COMMIT");
        return doc_id;
    } catch (...) {
        sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
        throw;
    }
}

// =======================================================================
// search
// =======================================================================

namespace {

// Semantic KNN, used both directly by search() and as one leg of
// hybrid_search(). Returns at most k hits, sorted by `distance` asc.
std::vector<Hit> search_semantic(sqlite3 *                  db,
                                 const Collection &         col,
                                 const std::vector<float> & query_embedding,
                                 int                        k) {
    if (static_cast<int>(query_embedding.size()) != col.dim) {
        fail(ExitCode::BadInput,
             "query embedding dim mismatch: collection '" + col.name + "' expects " +
             std::to_string(col.dim) + ", got " +
             std::to_string(query_embedding.size()));
    }
    // Join the vec0 KNN result back to `documents` for the user-visible
    // text + provenance. sqlite-vec requires the k-bound to come through
    // the MATCH operator's `k = ?` filter.
    StmtGuard q;
    const std::string sql =
        "SELECT d.id, COALESCE(d.source_uri, ''), d.chunk_index, d.text, v.distance "
        "FROM "   + vec_table_name(col.id) + " v "
        "JOIN documents d ON d.id = v.rowid "
        "WHERE v.embedding MATCH ? AND k = ? "
        "ORDER BY v.distance";
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, q.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(search_semantic)");
    }
    bind_vector(q.get(), 1, query_embedding);
    sqlite3_bind_int(q.get(), 2, k);

    std::vector<Hit> hits;
    hits.reserve(static_cast<size_t>(k));
    int rank = 0;
    while (sqlite3_step(q.get()) == SQLITE_ROW) {
        Hit h;
        h.document_id   = sqlite3_column_int64(q.get(), 0);
        h.source_uri    = reinterpret_cast<const char *>(sqlite3_column_text(q.get(), 1));
        h.chunk_index   = sqlite3_column_int  (q.get(), 2);
        h.text          = reinterpret_cast<const char *>(sqlite3_column_text(q.get(), 3));
        h.distance      = sqlite3_column_double(q.get(), 4);
        h.semantic_rank = rank++;
        hits.push_back(std::move(h));
    }
    return hits;
}

// Lexical (FTS5) leg. Uses bm25() ranking via the implicit `rank`
// virtual column. `query_text` is bound as a MATCH expression. FTS5
// syntax errors throw — `search()` catches the throw and falls back to
// a phrase-quoted form.
//
// Throws ChimeraError(Runtime) on FTS5 prepare/step failure; the
// non-error empty-result path is just an empty vector.
std::vector<Hit> search_lexical_raw(sqlite3 *           db,
                                    const Collection &  col,
                                    const std::string & query_text,
                                    int                 k) {
    StmtGuard q;
    const char * sql =
        "SELECT d.id, COALESCE(d.source_uri, ''), d.chunk_index, d.text, "
        "       documents_fts.rank "
        "FROM documents_fts "
        "JOIN documents d ON d.id = documents_fts.rowid "
        "WHERE documents_fts MATCH ? AND d.collection_id = ? "
        "ORDER BY documents_fts.rank "
        "LIMIT ?";
    if (sqlite3_prepare_v2(db, sql, -1, q.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(search_lexical)");
    }
    sqlite3_bind_text (q.get(), 1, query_text.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(q.get(), 2, col.id);
    sqlite3_bind_int  (q.get(), 3, k);

    std::vector<Hit> hits;
    hits.reserve(static_cast<size_t>(k));
    int rank_idx = 0;
    int rc;
    while ((rc = sqlite3_step(q.get())) == SQLITE_ROW) {
        Hit h;
        h.document_id  = sqlite3_column_int64(q.get(), 0);
        h.source_uri   = reinterpret_cast<const char *>(sqlite3_column_text(q.get(), 1));
        h.chunk_index  = sqlite3_column_int  (q.get(), 2);
        h.text         = reinterpret_cast<const char *>(sqlite3_column_text(q.get(), 3));
        // bm25() is reported as a negative double (lower = better) but
        // we keep the relative order via `lexical_rank`; surface the raw
        // value through `distance` for parity with the semantic leg so
        // CLI/JSON output always has *something* numeric to show.
        h.distance     = sqlite3_column_double(q.get(), 4);
        h.lexical_rank = rank_idx++;
        hits.push_back(std::move(h));
    }
    if (rc != SQLITE_DONE) {
        fail(ExitCode::Runtime,
             std::string("step(search_lexical): ") + sqlite3_errmsg(db));
    }
    return hits;
}

// FTS5 will throw on stray quotes / parens / special characters. Wrap
// the prepare+step pair so the caller can retry with a phrase-quoted
// form. Returns std::nullopt on syntax error; otherwise the hits.
std::optional<std::vector<Hit>> try_search_lexical(sqlite3 *           db,
                                                   const Collection &  col,
                                                   const std::string & query_text,
                                                   int                 k) {
    try {
        return search_lexical_raw(db, col, query_text, k);
    } catch (const std::exception & e) {
        const std::string msg = e.what();
        if (msg.find("fts5") != std::string::npos ||
            msg.find("syntax error") != std::string::npos ||
            msg.find("no such column") != std::string::npos) {
            return std::nullopt;
        }
        throw;
    }
}

// FTS5-safe quoting: escape any embedded `"` by doubling it and wrap
// the whole string in double quotes. This forces FTS5 to treat the
// input as a literal phrase — punctuation, parens, and reserved words
// inside are all neutralised.
std::string fts5_phrase_quote(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out.push_back('"');  // double up internal quotes
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

std::vector<Hit> search_lexical(sqlite3 *           db,
                                const Collection &  col,
                                const std::string & query_text,
                                int                 k) {
    if (query_text.empty()) return {};
    auto first = try_search_lexical(db, col, query_text, k);
    if (first) return std::move(*first);
    // FTS5 syntax error: fall back to phrase-quoted form. Don't catch
    // a second failure — that would be a real bug.
    return search_lexical_raw(db, col, fts5_phrase_quote(query_text), k);
}

// Reciprocal-rank fusion. Classic formulation:
//   rrf_score(d) = Σ_i  1 / (k_rrf + rank_i(d))
// with k_rrf = 60 (the value from the original RRF paper; values in
// 10..100 work in practice and the exact choice is not load-bearing).
//
// We feed both legs with `k_internal = max(k_target, 30)` so neither
// index starves the other before the merge.
std::vector<Hit> rrf_merge(std::vector<Hit> semantic,
                           std::vector<Hit> lexical,
                           int              k_target) {
    constexpr double K_RRF = 60.0;
    // Index by document_id so we can union the two result sets without
    // a quadratic scan. We keep the semantic-leg Hit as the canonical
    // copy (it carries the cosine distance) and fold in lexical_rank.
    std::vector<Hit> merged;
    merged.reserve(semantic.size() + lexical.size());
    auto find_by_id = [&](int64_t id) -> Hit * {
        for (auto & h : merged) {
            if (h.document_id == id) return &h;
        }
        return nullptr;
    };
    for (auto & s : semantic) {
        s.rrf_score = 1.0 / (K_RRF + (s.semantic_rank + 1));
        merged.push_back(std::move(s));
    }
    for (auto & l : lexical) {
        if (Hit * existing = find_by_id(l.document_id)) {
            existing->lexical_rank = l.lexical_rank;
            existing->rrf_score   += 1.0 / (K_RRF + (l.lexical_rank + 1));
        } else {
            l.rrf_score = 1.0 / (K_RRF + (l.lexical_rank + 1));
            merged.push_back(std::move(l));
        }
    }
    std::sort(merged.begin(), merged.end(),
              [](const Hit & a, const Hit & b) {
                  return a.rrf_score > b.rrf_score;
              });
    if (k_target > 0 && static_cast<int>(merged.size()) > k_target) {
        merged.resize(static_cast<size_t>(k_target));
    }
    return merged;
}

}  // namespace

std::vector<Hit> search(sqlite3 *                   db,
                        const Collection &          col,
                        const std::vector<float> &  query_embedding,
                        int                         k) {
    return search(db, col, query_embedding, /*query_text=*/"", k,
                  SearchMode::Semantic);
}

std::vector<Hit> search(sqlite3 *                   db,
                        const Collection &          col,
                        const std::vector<float> &  query_embedding,
                        const std::string &         query_text,
                        int                         k,
                        SearchMode                  mode) {
    if (k <= 0) k = 5;

    switch (mode) {
        case SearchMode::Semantic:
            return search_semantic(db, col, query_embedding, k);
        case SearchMode::Lexical:
            if (query_text.empty()) {
                fail(ExitCode::BadInput,
                     "lexical search requires non-empty query text");
            }
            return search_lexical(db, col, query_text, k);
        case SearchMode::Hybrid: {
            if (query_text.empty() || query_embedding.empty()) {
                fail(ExitCode::BadInput,
                     "hybrid search requires both query text and embedding");
            }
            const int k_internal = std::max(k, 30);
            auto sem = search_semantic(db, col, query_embedding, k_internal);
            auto lex = search_lexical (db, col, query_text,      k_internal);
            return rrf_merge(std::move(sem), std::move(lex), k);
        }
    }
    // Unreachable; switch above is exhaustive over SearchMode.
    return {};
}

}  // namespace chimera_vector_store
