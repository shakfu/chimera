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

Collection row_to_collection(sqlite3_stmt * stmt) {
    Collection c;
    c.id              = sqlite3_column_int64(stmt, 0);
    c.name            = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
    c.embedding_model = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2));
    c.dim             = sqlite3_column_int(stmt, 3);
    c.created_at      = sqlite3_column_int64(stmt, 4);
    c.doc_count       = 0;
    return c;
}

}  // namespace

// =======================================================================
// create / drop / find / list
// =======================================================================

Collection create(sqlite3 * db,
                  const std::string & name,
                  const std::string & embedding_model,
                  int                 dim) {
    if (name.empty()) {
        fail(ExitCode::BadInput, "collection name must not be empty");
    }
    if (dim <= 0) {
        fail(ExitCode::BadInput, "collection dim must be positive (got " +
                                  std::to_string(dim) + ")");
    }
    if (find(db, name)) {
        fail(ExitCode::BadInput, "collection already exists: " + name);
    }

    exec(db, "BEGIN");
    try {
        StmtGuard ins;
        const char * sql =
            "INSERT INTO collections (name, embedding_model, dim, created_at) "
            "VALUES (?, ?, ?, ?)";
        if (sqlite3_prepare_v2(db, sql, -1, ins.out(), nullptr) != SQLITE_OK) {
            sqlite_throw(db, "prepare(create collection)");
        }
        sqlite3_bind_text (ins.get(), 1, name.c_str(),            -1, SQLITE_TRANSIENT);
        sqlite3_bind_text (ins.get(), 2, embedding_model.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int  (ins.get(), 3, dim);
        sqlite3_bind_int64(ins.get(), 4, now_seconds());
        if (sqlite3_step(ins.get()) != SQLITE_DONE) {
            sqlite_throw(db, "step(create collection)");
        }
        const int64_t id = sqlite3_last_insert_rowid(db);

        // Per-collection vec0 table. dim is interpolated, not bound,
        // because CREATE VIRTUAL TABLE doesn't take parameters. Both
        // values are server-controlled (we validated dim above), so no
        // injection concern.
        const std::string create_vec =
            "CREATE VIRTUAL TABLE " + vec_table_name(id) +
            " USING vec0(embedding FLOAT[" + std::to_string(dim) + "])";
        exec(db, create_vec);

        exec(db, "COMMIT");

        Collection c;
        c.id              = id;
        c.name            = name;
        c.embedding_model = embedding_model;
        c.dim             = dim;
        c.created_at      = now_seconds();
        c.doc_count       = 0;
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
        "SELECT id, name, embedding_model, dim, created_at "
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
        "       COALESCE((SELECT COUNT(*) FROM documents d WHERE d.collection_id = c.id), 0) "
        "FROM collections c ORDER BY c.name";
    if (sqlite3_prepare_v2(db, sql, -1, q.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(list collections)");
    }
    while (sqlite3_step(q.get()) == SQLITE_ROW) {
        Collection c = row_to_collection(q.get());
        c.doc_count  = sqlite3_column_int64(q.get(), 5);
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

std::vector<Hit> search(sqlite3 *                   db,
                        const Collection &          col,
                        const std::vector<float> &  query_embedding,
                        int                         k) {
    if (static_cast<int>(query_embedding.size()) != col.dim) {
        fail(ExitCode::BadInput,
             "query embedding dim mismatch: collection '" + col.name + "' expects " +
             std::to_string(col.dim) + ", got " +
             std::to_string(query_embedding.size()));
    }
    if (k <= 0) k = 5;

    // Join the vec0 KNN result back to `documents` for the user-visible
    // text + provenance. sqlite-vec requires the k-bound to be a literal
    // or come through the MATCH operator alongside the vector; we use a
    // LIMIT clause which it interprets the same way.
    StmtGuard q;
    const std::string sql =
        "SELECT d.id, COALESCE(d.source_uri, ''), d.chunk_index, d.text, v.distance "
        "FROM "   + vec_table_name(col.id) + " v "
        "JOIN documents d ON d.id = v.rowid "
        "WHERE v.embedding MATCH ? AND k = ? "
        "ORDER BY v.distance";
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, q.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(search)");
    }
    bind_vector(q.get(), 1, query_embedding);
    sqlite3_bind_int(q.get(), 2, k);

    std::vector<Hit> hits;
    hits.reserve(static_cast<size_t>(k));
    while (sqlite3_step(q.get()) == SQLITE_ROW) {
        Hit h;
        h.document_id  = sqlite3_column_int64(q.get(), 0);
        h.source_uri   = reinterpret_cast<const char *>(sqlite3_column_text(q.get(), 1));
        h.chunk_index  = sqlite3_column_int(q.get(), 2);
        h.text         = reinterpret_cast<const char *>(sqlite3_column_text(q.get(), 3));
        h.distance     = sqlite3_column_double(q.get(), 4);
        hits.push_back(std::move(h));
    }
    return hits;
}

}  // namespace chimera_vector_store
