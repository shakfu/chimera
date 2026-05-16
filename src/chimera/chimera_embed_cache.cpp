// chimera_embed_cache.cpp — implementation of the optional persistent
// embedding cache. See chimera_embed_cache.h for the contract.

#include "chimera_embed_cache.h"
#include "chimera.h"           // ChimeraError, ExitCode
#include "chimera_db.h"        // open_and_migrate, Connection

#include "sqlite3.h"
#include <openssl/evp.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

namespace chimera_embed_cache {

namespace {

constexpr size_t kFingerprintEdgeBytes = 64 * 1024;  // 64 KB head + 64 KB tail

int64_t now_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string hex_lower(const unsigned char * bytes, size_t n) {
    std::ostringstream os;
    os << std::hex << std::setfill('0');
    for (size_t i = 0; i < n; ++i) {
        os << std::setw(2) << static_cast<unsigned>(bytes[i]);
    }
    return os.str();
}

// One-shot SHA-256 over an in-memory buffer.
void sha256(const void * data, size_t len, unsigned char out[32]) {
    EVP_MD_CTX * ctx = EVP_MD_CTX_new();
    if (!ctx) fail(ExitCode::Runtime, "EVP_MD_CTX_new failed");
    unsigned int outlen = 0;
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data, len) != 1 ||
        EVP_DigestFinal_ex(ctx, out, &outlen) != 1) {
        EVP_MD_CTX_free(ctx);
        fail(ExitCode::Runtime, "SHA-256 digest failed");
    }
    EVP_MD_CTX_free(ctx);
}

// Sha-256 over a sequence of (ptr, len) chunks. Used by compute_model_id
// to fingerprint (size || head || tail) without an intermediate copy.
void sha256_chunks(const std::vector<std::pair<const void *, size_t>> & chunks,
                   unsigned char out[32]) {
    EVP_MD_CTX * ctx = EVP_MD_CTX_new();
    if (!ctx) fail(ExitCode::Runtime, "EVP_MD_CTX_new failed");
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        fail(ExitCode::Runtime, "SHA-256 init failed");
    }
    for (const auto & c : chunks) {
        if (EVP_DigestUpdate(ctx, c.first, c.second) != 1) {
            EVP_MD_CTX_free(ctx);
            fail(ExitCode::Runtime, "SHA-256 update failed");
        }
    }
    unsigned int outlen = 0;
    if (EVP_DigestFinal_ex(ctx, out, &outlen) != 1) {
        EVP_MD_CTX_free(ctx);
        fail(ExitCode::Runtime, "SHA-256 final failed");
    }
    EVP_MD_CTX_free(ctx);
}

[[noreturn]] void sqlite_throw(sqlite3 * db, const std::string & ctx) {
    fail(ExitCode::Runtime, ctx + ": " + sqlite3_errmsg(db));
}

struct StmtGuard {
    sqlite3_stmt * stmt = nullptr;
    ~StmtGuard() { if (stmt) sqlite3_finalize(stmt); }
};

}  // namespace

std::vector<unsigned char> sha256_bytes(const std::string & text) {
    std::vector<unsigned char> out(32);
    sha256(text.data(), text.size(), out.data());
    return out;
}

std::string compute_model_id(const std::string & model_path) {
    std::ifstream f(model_path, std::ios::binary);
    if (!f) return {};

    f.seekg(0, std::ios::end);
    const std::streamoff total = f.tellg();
    if (total < 0) return {};
    const uint64_t size = static_cast<uint64_t>(total);

    // Files smaller than 2 * edge bytes: just hash the whole thing.
    const size_t edge = static_cast<size_t>(
        std::min<uint64_t>(kFingerprintEdgeBytes, size));

    std::vector<char> head(edge);
    f.seekg(0, std::ios::beg);
    f.read(head.data(), static_cast<std::streamsize>(edge));
    if (f.gcount() != static_cast<std::streamsize>(edge)) return {};

    std::vector<char> tail;
    if (size > 2 * kFingerprintEdgeBytes) {
        tail.resize(kFingerprintEdgeBytes);
        f.seekg(static_cast<std::streamoff>(size - kFingerprintEdgeBytes),
                std::ios::beg);
        f.read(tail.data(), kFingerprintEdgeBytes);
        if (f.gcount() != static_cast<std::streamsize>(kFingerprintEdgeBytes)) return {};
    }

    // Hash the size as little-endian u64 so the on-disk value is
    // architecture-independent (compute_model_id called on the same
    // file on x86_64 and arm64 produces the same id).
    unsigned char size_le[8];
    for (int i = 0; i < 8; ++i) size_le[i] = static_cast<unsigned char>((size >> (8 * i)) & 0xFF);

    std::vector<std::pair<const void *, size_t>> chunks{
        { size_le,    sizeof(size_le) },
        { head.data(), head.size() },
    };
    if (!tail.empty()) chunks.push_back({ tail.data(), tail.size() });

    unsigned char digest[32];
    sha256_chunks(chunks, digest);
    return hex_lower(digest, sizeof(digest));
}

struct Cache::Impl {
    chimera_db::Connection conn;
    std::string            model_id;
    sqlite3_stmt *         stmt_lookup = nullptr;
    sqlite3_stmt *         stmt_put    = nullptr;
    sqlite3_stmt *         stmt_count  = nullptr;

    ~Impl() {
        if (stmt_lookup) sqlite3_finalize(stmt_lookup);
        if (stmt_put)    sqlite3_finalize(stmt_put);
        if (stmt_count)  sqlite3_finalize(stmt_count);
    }
};

Cache::Cache(const std::string & db_path, std::string model_id)
    : impl_(std::make_unique<Impl>()) {
    if (model_id.empty()) {
        fail(ExitCode::BadInput,
             "embedding cache requires a non-empty model_id "
             "(model file may be unreadable)");
    }
    impl_->conn     = chimera_db::open_and_migrate(db_path);
    impl_->model_id = std::move(model_id);

    sqlite3 * db = impl_->conn.get();

    const char * sql_lookup =
        "SELECT dim, vec FROM embedding_cache "
        "WHERE model_id = ?1 AND text_sha = ?2";
    if (sqlite3_prepare_v2(db, sql_lookup, -1, &impl_->stmt_lookup, nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(embedding_cache lookup)");
    }

    // INSERT OR REPLACE so dim/pooling drift (rare; should only happen
    // if model_id collisions occur) overwrites cleanly rather than
    // aborting. The primary key keeps the row count bounded.
    const char * sql_put =
        "INSERT OR REPLACE INTO embedding_cache "
        "(model_id, text_sha, dim, vec, created_at) VALUES (?1, ?2, ?3, ?4, ?5)";
    if (sqlite3_prepare_v2(db, sql_put, -1, &impl_->stmt_put, nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(embedding_cache put)");
    }

    const char * sql_count =
        "SELECT COUNT(*) FROM embedding_cache WHERE model_id = ?1";
    if (sqlite3_prepare_v2(db, sql_count, -1, &impl_->stmt_count, nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(embedding_cache count)");
    }
}

Cache::Cache(Cache &&) noexcept            = default;
Cache & Cache::operator=(Cache &&) noexcept = default;
Cache::~Cache()                             = default;

const std::string & Cache::model_id() const { return impl_->model_id; }

bool Cache::lookup(const std::string & text, int expected_dim,
                   std::vector<float> & out) {
    if (text.empty()) return false;

    const auto sha = sha256_bytes(text);
    sqlite3 * db   = impl_->conn.get();
    sqlite3_stmt * st = impl_->stmt_lookup;
    sqlite3_reset(st);
    sqlite3_bind_text(st, 1, impl_->model_id.c_str(),
                      static_cast<int>(impl_->model_id.size()), SQLITE_STATIC);
    sqlite3_bind_blob(st, 2, sha.data(),
                      static_cast<int>(sha.size()), SQLITE_TRANSIENT);

    const int rc = sqlite3_step(st);
    if (rc == SQLITE_DONE) return false;
    if (rc != SQLITE_ROW) sqlite_throw(db, "embedding_cache lookup step");

    const int dim = sqlite3_column_int(st, 0);
    if (dim != expected_dim) {
        // Treat dim mismatch as miss. Recompute + put will overwrite
        // the stale row (INSERT OR REPLACE on the same primary key).
        return false;
    }
    const void * blob = sqlite3_column_blob(st, 1);
    const int    nb   = sqlite3_column_bytes(st, 1);
    if (nb != static_cast<int>(dim * sizeof(float))) {
        // Corrupt row; treat as miss.
        return false;
    }
    out.resize(static_cast<size_t>(dim));
    std::memcpy(out.data(), blob, static_cast<size_t>(nb));
    return true;
}

void Cache::put(const std::string & text, const std::vector<float> & vec) {
    if (text.empty() || vec.empty()) return;

    const auto sha = sha256_bytes(text);
    sqlite3 * db   = impl_->conn.get();
    sqlite3_stmt * st = impl_->stmt_put;
    sqlite3_reset(st);
    sqlite3_bind_text (st, 1, impl_->model_id.c_str(),
                       static_cast<int>(impl_->model_id.size()), SQLITE_STATIC);
    sqlite3_bind_blob (st, 2, sha.data(),
                       static_cast<int>(sha.size()), SQLITE_TRANSIENT);
    sqlite3_bind_int  (st, 3, static_cast<int>(vec.size()));
    sqlite3_bind_blob (st, 4, vec.data(),
                       static_cast<int>(vec.size() * sizeof(float)),
                       SQLITE_TRANSIENT);
    sqlite3_bind_int64(st, 5, now_seconds());

    if (sqlite3_step(st) != SQLITE_DONE) {
        sqlite_throw(db, "embedding_cache put step");
    }
}

int64_t Cache::count() const {
    sqlite3 * db = impl_->conn.get();
    sqlite3_stmt * st = impl_->stmt_count;
    sqlite3_reset(st);
    sqlite3_bind_text(st, 1, impl_->model_id.c_str(),
                      static_cast<int>(impl_->model_id.size()), SQLITE_STATIC);
    if (sqlite3_step(st) != SQLITE_ROW) sqlite_throw(db, "embedding_cache count step");
    return sqlite3_column_int64(st, 0);
}

}  // namespace chimera_embed_cache
