// chimera_db.cpp — embedded SQLite + sqlite-vec for chimera.
//
// Public API in chimera_db.h. See doc/dev/sqlite.md for the design.

#include "chimera_db.h"
#include "chimera.h"  // ChimeraError + ExitCode

#include "sqlite3.h"
#include "sqlite-vec.h"

#include <cstdlib>     // std::getenv
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace chimera_db {

// ===========================================================================
// Connection
// ===========================================================================

void Connection::close() {
    if (db_ != nullptr) {
        // sqlite3_close_v2 deferred-closes if there are still active
        // statements; safe to use on shutdown.
        sqlite3_close_v2(db_);
        db_ = nullptr;
    }
}

// ===========================================================================
// Default-path resolution (XDG-style)
// ===========================================================================

namespace {

std::string env_or_empty(const char * key) {
    const char * v = std::getenv(key);
    return (v && *v) ? std::string(v) : std::string();
}

fs::path platform_default_data_dir() {
    // Prefer XDG_DATA_HOME if explicitly set (Linux convention, also
    // honored by many cross-platform tools).
    const std::string xdg = env_or_empty("XDG_DATA_HOME");
    if (!xdg.empty()) {
        return fs::path(xdg) / "chimera";
    }

#if defined(__APPLE__)
    const std::string home = env_or_empty("HOME");
    if (!home.empty()) {
        return fs::path(home) / "Library" / "Application Support" / "chimera";
    }
#elif defined(_WIN32)
    const std::string local = env_or_empty("LOCALAPPDATA");
    if (!local.empty()) {
        return fs::path(local) / "chimera";
    }
#else
    const std::string home = env_or_empty("HOME");
    if (!home.empty()) {
        return fs::path(home) / ".local" / "share" / "chimera";
    }
#endif
    // Last-ditch fallback: a chimera/ subdir in the current working
    // directory. Better than throwing; the caller will still see a
    // valid path and the DB will be created where they ran the command.
    return fs::current_path() / ".chimera";
}

// Create `dir` and any missing parents. No-op if it already exists.
void ensure_dir(const fs::path & dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        fail(ExitCode::Runtime,
             "failed to create data directory '" + dir.string() + "': " + ec.message());
    }
}

// Throws on sqlite3_exec errors with the message sqlite produced.
void exec_or_throw(sqlite3 * db, const char * sql) {
    char * err = nullptr;
    const int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        const std::string msg = err ? err : sqlite3_errmsg(db);
        if (err) sqlite3_free(err);
        fail(ExitCode::Runtime,
             "sqlite exec failed (" + std::to_string(rc) + "): " + msg);
    }
}

}  // namespace

std::string default_path() {
    if (const std::string env_path = env_or_empty("CHIMERA_DB"); !env_path.empty()) {
        // Honor the override verbatim. Caller may point this at any
        // file path; we still ensure the parent dir exists.
        const fs::path p(env_path);
        ensure_dir(p.parent_path().empty() ? fs::current_path() : p.parent_path());
        return p.string();
    }
    const fs::path dir = platform_default_data_dir();
    ensure_dir(dir);
    return (dir / "chimera.db").string();
}

// ===========================================================================
// Migrations
// ===========================================================================

namespace {

// v1: initial schema. chats + messages + messages_fts (FTS5 mirror with
// triggers) + collections + documents. Per-collection vec0 tables are
// created on demand by chimera_vector_store, not in this migration.
//
// All DDL runs inside a single transaction (started by the runner).
//
// Note: SQLITE_OMIT_DECLTYPE is set at compile time, which removes the
// "type affinity from CREATE TABLE column types" feature *for
// sqlite3_column_decltype* — the storage class still works, so INTEGER
// PRIMARY KEY etc. continue to do the right thing.
constexpr const char * MIGRATION_V1 = R"SQL(
    CREATE TABLE chats (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at      INTEGER NOT NULL,
        updated_at      INTEGER NOT NULL,
        title           TEXT,
        model_path      TEXT    NOT NULL,
        model_alias     TEXT,
        system_prompt   TEXT,
        source          TEXT    NOT NULL CHECK (source IN ('chat', 'serve')),
        metadata_json   TEXT
    );
    CREATE INDEX idx_chats_updated ON chats(updated_at DESC);

    CREATE TABLE messages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id         INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
        seq             INTEGER NOT NULL,
        role            TEXT    NOT NULL,
        content         TEXT    NOT NULL,
        reasoning       TEXT,
        media_json      TEXT,
        tool_calls_json TEXT,
        tokens_in       INTEGER,
        tokens_out      INTEGER,
        created_at      INTEGER NOT NULL,
        UNIQUE (chat_id, seq)
    );
    CREATE INDEX idx_messages_chat ON messages(chat_id, seq);

    -- FTS5 mirror of messages.content. Keeps content out-of-table
    -- (content='messages') so we don't double the storage cost.
    CREATE VIRTUAL TABLE messages_fts USING fts5(
        content,
        role     UNINDEXED,
        chat_id  UNINDEXED,
        content=messages,
        content_rowid=id
    );

    -- Triggers keep messages_fts in sync with messages.
    CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, content, role, chat_id)
        VALUES (new.id, new.content, new.role, new.chat_id);
    END;
    CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content, role, chat_id)
        VALUES ('delete', old.id, old.content, old.role, old.chat_id);
    END;
    CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content, role, chat_id)
        VALUES ('delete', old.id, old.content, old.role, old.chat_id);
        INSERT INTO messages_fts(rowid, content, role, chat_id)
        VALUES (new.id, new.content, new.role, new.chat_id);
    END;

    CREATE TABLE collections (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        name            TEXT    UNIQUE NOT NULL,
        embedding_model TEXT    NOT NULL,
        dim             INTEGER NOT NULL,
        created_at      INTEGER NOT NULL,
        metadata_json   TEXT
    );

    CREATE TABLE documents (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_id   INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
        source_uri      TEXT,
        chunk_index     INTEGER NOT NULL,
        text            TEXT    NOT NULL,
        token_count     INTEGER,
        metadata_json   TEXT,
        created_at      INTEGER NOT NULL
    );
    CREATE INDEX idx_documents_collection ON documents(collection_id);
    CREATE INDEX idx_documents_source     ON documents(collection_id, source_uri, chunk_index);
)SQL";

struct Migration {
    int          to_version;
    const char * description;
    const char * sql;
};

// v2: add embedding_cache. Memoizes `embed(text) -> vector` so
// ingestion of partially-updated corpora and repeated query embedding
// skip the model. Keyed on (model_id, text_sha): `model_id` is a fast
// fingerprint of the embedding model file (see chimera_embed_cache.cpp),
// `text_sha` is the SHA-256 of the input string. Vectors are stored as
// raw little-endian float32 blobs (no JSON, no scaling) so the on-disk
// bytes round-trip exactly. The composite PRIMARY KEY doubles as the
// hot lookup index; the secondary index on created_at is for
// `db prune --older-than=<duration>` once that lands.
constexpr const char * MIGRATION_V2 = R"SQL(
    CREATE TABLE embedding_cache (
        model_id     TEXT    NOT NULL,
        text_sha     BLOB    NOT NULL,
        dim          INTEGER NOT NULL,
        vec          BLOB    NOT NULL,
        created_at   INTEGER NOT NULL,
        PRIMARY KEY (model_id, text_sha)
    );
    CREATE INDEX idx_embedding_cache_created ON embedding_cache(created_at);
)SQL";

// v3: per-collection knobs. `distance` is the sqlite-vec metric used by
// the per-collection vec0 table (validated at create-time against
// `cosine | l2 | l1`); `chunk_tokens` + `chunk_overlap` are the defaults
// that `chimera index ingest` (and the equivalent serve route) use when
// the caller doesn't override. Tokens, not characters: chunking moved
// from character-window to token-window so chunk sizes are accurate
// against the embedding model's vocab.
//
// ALTER TABLE ... ADD COLUMN ... NOT NULL DEFAULT <const> backfills
// existing rows with the default, so users upgrading from a v1 / v2 DB
// get cosine/512/64 for every collection they created before this
// migration.
constexpr const char * MIGRATION_V3 = R"SQL(
    ALTER TABLE collections ADD COLUMN distance      TEXT    NOT NULL DEFAULT 'cosine';
    ALTER TABLE collections ADD COLUMN chunk_tokens  INTEGER NOT NULL DEFAULT 512;
    ALTER TABLE collections ADD COLUMN chunk_overlap INTEGER NOT NULL DEFAULT 64;
)SQL";

constexpr Migration MIGRATIONS[] = {
    { 1, "initial chat + collection schema",       MIGRATION_V1 },
    { 2, "add embedding_cache table",              MIGRATION_V2 },
    { 3, "per-collection distance + chunk knobs",  MIGRATION_V3 },
};

constexpr int kLatest = sizeof(MIGRATIONS) / sizeof(MIGRATIONS[0]);

}  // namespace

int latest_schema_version() {
    return kLatest;
}

int current_schema_version(sqlite3 * db) {
    sqlite3_stmt * stmt = nullptr;
    if (sqlite3_prepare_v2(db, "PRAGMA user_version", -1, &stmt, nullptr) != SQLITE_OK) {
        fail(ExitCode::Runtime,
             std::string("sqlite prepare(PRAGMA user_version) failed: ") + sqlite3_errmsg(db));
    }
    int version = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        version = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return version;
}

void migrate(sqlite3 * db, int target) {
    const int from = current_schema_version(db);
    if (from >= target) return;

    for (const auto & m : MIGRATIONS) {
        if (m.to_version <= from || m.to_version > target) continue;
        exec_or_throw(db, "BEGIN");
        try {
            exec_or_throw(db, m.sql);
            // PRAGMA can't take a parameter, so build the statement.
            const std::string set =
                "PRAGMA user_version = " + std::to_string(m.to_version);
            exec_or_throw(db, set.c_str());
            exec_or_throw(db, "COMMIT");
        } catch (...) {
            sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
            throw;
        }
    }
}

// ===========================================================================
// open_and_migrate
// ===========================================================================

namespace {

// Register sqlite-vec as an auto-extension exactly once per process.
// All future connections will load it automatically on open.
void register_sqlite_vec_once() {
    static bool registered = false;
    if (registered) return;
    // The cast-to-void(*)() suppresses a -Wcast-function-type warning
    // on gcc; the underlying signature matches what sqlite3_auto_extension
    // expects (an extension entry point).
    const int rc = sqlite3_auto_extension(
        reinterpret_cast<void (*)()>(sqlite3_vec_init));
    if (rc != SQLITE_OK) {
        fail(ExitCode::Runtime,
             "sqlite3_auto_extension(sqlite3_vec_init) failed: rc=" +
                 std::to_string(rc));
    }
    registered = true;
}

}  // namespace

Connection open_and_migrate(const std::string & path) {
    register_sqlite_vec_once();

    sqlite3 * raw = nullptr;
    // OPEN_NOMUTEX: we built sqlite with SQLITE_THREADSAFE=2, so the
    // connection itself is single-threaded. Threading discipline is
    // managed at the chimera layer (one connection per worker thread).
    const int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX;
    const int rc = sqlite3_open_v2(path.c_str(), &raw, flags, nullptr);
    if (rc != SQLITE_OK) {
        const std::string msg = raw ? sqlite3_errmsg(raw) : "open failed";
        if (raw) sqlite3_close(raw);
        fail(ExitCode::Runtime,
             "sqlite open '" + path + "' failed (" + std::to_string(rc) + "): " + msg);
    }
    Connection conn(raw);

    // Pragmas applied before migrations so the DDL runs in WAL mode.
    exec_or_throw(conn.get(), "PRAGMA journal_mode = WAL");
    exec_or_throw(conn.get(), "PRAGMA synchronous = NORMAL");
    exec_or_throw(conn.get(), "PRAGMA foreign_keys = ON");

    migrate(conn.get(), latest_schema_version());
    return conn;
}

// ===========================================================================
// Introspection helpers
// ===========================================================================

std::vector<std::string> list_tables(sqlite3 * db) {
    std::vector<std::string> tables;
    const char * sql =
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name";
    sqlite3_stmt * stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        fail(ExitCode::Runtime,
             std::string("sqlite prepare(list_tables) failed: ") + sqlite3_errmsg(db));
    }
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        tables.emplace_back(reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0)));
    }
    sqlite3_finalize(stmt);
    return tables;
}

const char * sqlite_version()     { return SQLITE_VERSION; }
const char * sqlite_vec_version() { return SQLITE_VEC_VERSION; }

std::string sqlite_vec_loaded_version(sqlite3 * db) {
    sqlite3_stmt * stmt = nullptr;
    if (sqlite3_prepare_v2(db, "SELECT vec_version()", -1, &stmt, nullptr) != SQLITE_OK) {
        return {};
    }
    std::string out;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        if (const auto * txt = sqlite3_column_text(stmt, 0)) {
            out = reinterpret_cast<const char *>(txt);
        }
    }
    sqlite3_finalize(stmt);
    return out;
}

}  // namespace chimera_db
