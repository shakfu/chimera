// chimera_db.h — embedded SQLite + sqlite-vec for chimera.
//
// Public surface for every TU that needs the database (initially:
// chimera.cpp for `chimera db status` + `chimera --version`; later:
// chimera_chat_store.cpp + chimera_vector_store.cpp + the serve-side
// RAG handlers). Anything kept private to chimera_db.cpp stays in its
// own anonymous namespace.
//
// Covers: connection lifecycle, default-path resolution (XDG-compliant),
// migration runner, schema, and the maintenance helpers (backup_to,
// vacuum). Higher-level query helpers live in chimera_vector_store.* and
// chimera_chat_store.*.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct sqlite3;

namespace chimera_db {

// RAII wrapper around `sqlite3 *`. Closes on destruction. Move-only.
class Connection {
public:
    Connection() = default;
    Connection(sqlite3 * db) : db_(db) {}
    Connection(const Connection &) = delete;
    Connection & operator=(const Connection &) = delete;
    Connection(Connection && other) noexcept : db_(other.db_) { other.db_ = nullptr; }
    Connection & operator=(Connection && other) noexcept {
        if (this != &other) {
            close();
            db_ = other.db_;
            other.db_ = nullptr;
        }
        return *this;
    }
    ~Connection() { close(); }

    sqlite3 * get() const { return db_; }
    bool      ok()  const { return db_ != nullptr; }

    // Release ownership; caller is responsible for sqlite3_close().
    sqlite3 * release() {
        sqlite3 * p = db_;
        db_ = nullptr;
        return p;
    }

    void close();

private:
    sqlite3 * db_ = nullptr;
};

// --- path resolution ----------------------------------------------------

// Resolve the chimera DB path, in priority order:
//   1. $CHIMERA_DB
//   2. $XDG_DATA_HOME/chimera/chimera.db
//   3. macOS:   $HOME/Library/Application Support/chimera/chimera.db
//      Linux:   $HOME/.local/share/chimera/chimera.db
//      Windows: %LOCALAPPDATA%/chimera/chimera.db
//
// The parent directory is created if needed. The DB file itself is not
// created until open() is called against it.
std::string default_path();

// --- connection ---------------------------------------------------------

// Open (or create) the SQLite database at `path`, configure pragmas
// (WAL, NORMAL synchronous, foreign keys ON), register sqlite-vec as an
// auto-loaded extension (so future connections also load it), and run
// the migration chain to bring the schema up to the latest version.
//
// Throws ChimeraError on any failure. The migration step is idempotent;
// calling open_and_migrate on an already-current DB is cheap.
Connection open_and_migrate(const std::string & path);

// --- migrations ---------------------------------------------------------

// Latest schema version this build knows how to apply. Stored in the
// DB's `PRAGMA user_version` after each successful migration step.
int latest_schema_version();

// The current `user_version` recorded in the DB. 0 for a fresh file.
int current_schema_version(sqlite3 * db);

// Walk the schema from its current version to `target` (typically
// `latest_schema_version()`). No-op if already at or above target.
// Throws on migration failure; rolls back the partial transaction so
// the DB stays at the last successful version.
void migrate(sqlite3 * db, int target);

// --- introspection -------------------------------------------------------

// Names of user tables (excludes SQLite internal tables). For `chimera
// db status` output and tests.
std::vector<std::string> list_tables(sqlite3 * db);

// Compile-time SQLite + sqlite-vec version strings. These are pulled
// from the vendored headers, so they always match what is actually
// linked into the chimera binary.
const char * sqlite_version();
const char * sqlite_vec_version();

// Run-time check that sqlite-vec is actually loaded into `db`. Executes
// `SELECT vec_version()` and returns whatever string the extension
// reports. Returns an empty string if the function does not exist
// (i.e. sqlite_auto_extension never fired or it fired but failed).
std::string sqlite_vec_loaded_version(sqlite3 * db);

// --- maintenance ---------------------------------------------------------

// Snapshot the DB to `dst_path` via `VACUUM INTO '<dst_path>'`. The
// destination file is created (must not exist) and the snapshot is a
// fully-defragmented copy with the same schema + content. SQLite issues
// a brief shared lock on the source for the duration; concurrent
// readers continue, concurrent writers serialize. WAL + SHM siblings
// are not produced — `VACUUM INTO` rolls them into the destination
// itself, so a single file is enough to restore from.
//
// Throws ChimeraError on any failure (destination already exists,
// permission denied, source corruption, ...).
void backup_to(sqlite3 * db, const std::string & dst_path);

// Run plain `VACUUM` on the open DB. Defragments + reclaims free pages
// in place. Requires no other connections to the DB; SQLite errors
// otherwise. Throws ChimeraError on failure.
void vacuum(sqlite3 * db);

}  // namespace chimera_db
