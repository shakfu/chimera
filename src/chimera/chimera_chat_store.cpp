// chimera_chat_store.cpp — implementation of the chat persistence API.
//
// Schema is the v1 migration in chimera_db.cpp; we just talk to the
// already-created tables. FTS5 sync is handled by triggers, not by code
// here.

#include "chimera_chat_store.h"
#include "chimera.h"

#include "sqlite3.h"

#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>

namespace chimera_chat_store {

namespace {

int64_t now_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

[[noreturn]] void sqlite_throw(sqlite3 * db, const std::string & ctx) {
    fail(ExitCode::Runtime, ctx + ": " + sqlite3_errmsg(db));
}

struct StmtGuard {
    sqlite3_stmt * stmt = nullptr;
    ~StmtGuard() { if (stmt) sqlite3_finalize(stmt); }
    sqlite3_stmt * get() { return stmt; }
    sqlite3_stmt ** out() { return &stmt; }
};

// Read a TEXT column as std::string, tolerating NULL.
std::string col_text(sqlite3_stmt * stmt, int col) {
    const auto * txt = sqlite3_column_text(stmt, col);
    return txt ? std::string(reinterpret_cast<const char *>(txt)) : std::string();
}

Chat row_to_chat(sqlite3_stmt * stmt) {
    Chat c;
    c.id            = sqlite3_column_int64(stmt, 0);
    c.created_at    = sqlite3_column_int64(stmt, 1);
    c.updated_at    = sqlite3_column_int64(stmt, 2);
    c.title         = col_text(stmt, 3);
    c.model_path    = col_text(stmt, 4);
    c.model_alias   = col_text(stmt, 5);
    c.system_prompt = col_text(stmt, 6);
    c.source        = col_text(stmt, 7);
    c.metadata_json = col_text(stmt, 8);
    c.message_count = 0;  // filled by list_chats() via correlated subquery
    c.partial_count = 0;  // ditto
    return c;
}

}  // namespace

// ===========================================================================
// chats
// ===========================================================================

int64_t create_chat(sqlite3 *           db,
                    const std::string & model_path,
                    const std::string & model_alias,
                    const std::string & system_prompt,
                    const std::string & source,
                    const std::string & metadata_json) {
    StmtGuard ins;
    const char * sql =
        "INSERT INTO chats (created_at, updated_at, model_path, model_alias, "
        "                   system_prompt, source, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)";
    if (sqlite3_prepare_v2(db, sql, -1, ins.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(create_chat)");
    }
    const int64_t now = now_seconds();
    sqlite3_bind_int64(ins.get(), 1, now);
    sqlite3_bind_int64(ins.get(), 2, now);
    sqlite3_bind_text (ins.get(), 3, model_path.c_str(),  -1, SQLITE_TRANSIENT);
    if (model_alias.empty()) sqlite3_bind_null(ins.get(), 4);
    else sqlite3_bind_text  (ins.get(), 4, model_alias.c_str(), -1, SQLITE_TRANSIENT);
    if (system_prompt.empty()) sqlite3_bind_null(ins.get(), 5);
    else sqlite3_bind_text  (ins.get(), 5, system_prompt.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text (ins.get(), 6, source.c_str(),      -1, SQLITE_TRANSIENT);
    if (metadata_json.empty()) sqlite3_bind_null(ins.get(), 7);
    else sqlite3_bind_text  (ins.get(), 7, metadata_json.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(ins.get()) != SQLITE_DONE) {
        sqlite_throw(db, "step(create_chat)");
    }
    return sqlite3_last_insert_rowid(db);
}

std::optional<Chat> load_chat(sqlite3 * db, int64_t chat_id) {
    StmtGuard q;
    const char * sql =
        "SELECT id, created_at, updated_at, COALESCE(title,''), model_path, "
        "       COALESCE(model_alias,''), COALESCE(system_prompt,''), source, "
        "       COALESCE(metadata_json,'') "
        "FROM chats WHERE id = ?";
    if (sqlite3_prepare_v2(db, sql, -1, q.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(load_chat)");
    }
    sqlite3_bind_int64(q.get(), 1, chat_id);
    if (sqlite3_step(q.get()) != SQLITE_ROW) return std::nullopt;
    return row_to_chat(q.get());
}

std::optional<Chat> latest_chat(sqlite3 * db) {
    StmtGuard q;
    const char * sql =
        "SELECT id, created_at, updated_at, COALESCE(title,''), model_path, "
        "       COALESCE(model_alias,''), COALESCE(system_prompt,''), source, "
        "       COALESCE(metadata_json,'') "
        "FROM chats ORDER BY updated_at DESC LIMIT 1";
    if (sqlite3_prepare_v2(db, sql, -1, q.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(latest_chat)");
    }
    if (sqlite3_step(q.get()) != SQLITE_ROW) return std::nullopt;
    return row_to_chat(q.get());
}

std::vector<Chat> list_chats(sqlite3 * db, int limit) {
    std::vector<Chat> out;
    StmtGuard q;
    const char * sql =
        "SELECT c.id, c.created_at, c.updated_at, COALESCE(c.title,''), "
        "       c.model_path, COALESCE(c.model_alias,''), "
        "       COALESCE(c.system_prompt,''), c.source, "
        "       COALESCE(c.metadata_json,''), "
        "       (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id), "
        "       (SELECT COUNT(*) FROM messages m "
        "         WHERE m.chat_id = c.id AND m.partial = 1) "
        "FROM chats c "
        "ORDER BY c.updated_at DESC "
        "LIMIT ?";
    if (sqlite3_prepare_v2(db, sql, -1, q.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(list_chats)");
    }
    sqlite3_bind_int(q.get(), 1, limit);
    while (sqlite3_step(q.get()) == SQLITE_ROW) {
        Chat c = row_to_chat(q.get());
        c.message_count = sqlite3_column_int64(q.get(), 9);
        c.partial_count = sqlite3_column_int64(q.get(), 10);
        out.push_back(std::move(c));
    }
    return out;
}

void touch_chat(sqlite3 * db, int64_t chat_id) {
    StmtGuard upd;
    if (sqlite3_prepare_v2(db, "UPDATE chats SET updated_at = ? WHERE id = ?",
                           -1, upd.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(touch_chat)");
    }
    sqlite3_bind_int64(upd.get(), 1, now_seconds());
    sqlite3_bind_int64(upd.get(), 2, chat_id);
    if (sqlite3_step(upd.get()) != SQLITE_DONE) {
        sqlite_throw(db, "step(touch_chat)");
    }
}

void set_chat_title(sqlite3 * db, int64_t chat_id, const std::string & title) {
    StmtGuard upd;
    if (sqlite3_prepare_v2(db, "UPDATE chats SET title = ? WHERE id = ?",
                           -1, upd.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(set_chat_title)");
    }
    if (title.empty()) sqlite3_bind_null(upd.get(), 1);
    else sqlite3_bind_text(upd.get(), 1, title.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(upd.get(), 2, chat_id);
    if (sqlite3_step(upd.get()) != SQLITE_DONE) {
        sqlite_throw(db, "step(set_chat_title)");
    }
}

// ===========================================================================
// messages
// ===========================================================================

int64_t append_message(sqlite3 *           db,
                       int64_t             chat_id,
                       const std::string & role,
                       const std::string & content,
                       const std::string & reasoning,
                       const std::string & media_json,
                       int                 tokens_in,
                       int                 tokens_out,
                       bool                partial) {
    // Wrap in a transaction so the (compute-next-seq, insert, touch_chat)
    // triple is atomic. Concurrent writers (we don't have any today) would
    // otherwise race on seq.
    char * err = nullptr;
    if (sqlite3_exec(db, "BEGIN", nullptr, nullptr, &err) != SQLITE_OK) {
        const std::string m = err ? err : "begin failed";
        if (err) sqlite3_free(err);
        fail(ExitCode::Runtime, "append_message begin: " + m);
    }
    try {
        // Compute next seq within the chat.
        int next_seq = 0;
        {
            StmtGuard q;
            if (sqlite3_prepare_v2(db,
                    "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE chat_id = ?",
                    -1, q.out(), nullptr) != SQLITE_OK) {
                sqlite_throw(db, "prepare(next_seq)");
            }
            sqlite3_bind_int64(q.get(), 1, chat_id);
            if (sqlite3_step(q.get()) == SQLITE_ROW) {
                next_seq = sqlite3_column_int(q.get(), 0);
            }
        }

        StmtGuard ins;
        const char * sql =
            "INSERT INTO messages (chat_id, seq, role, content, reasoning, "
            "                      media_json, tokens_in, tokens_out, created_at, partial) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
        if (sqlite3_prepare_v2(db, sql, -1, ins.out(), nullptr) != SQLITE_OK) {
            sqlite_throw(db, "prepare(append_message)");
        }
        sqlite3_bind_int64(ins.get(), 1, chat_id);
        sqlite3_bind_int  (ins.get(), 2, next_seq);
        sqlite3_bind_text (ins.get(), 3, role.c_str(),    -1, SQLITE_TRANSIENT);
        sqlite3_bind_text (ins.get(), 4, content.c_str(), -1, SQLITE_TRANSIENT);
        if (reasoning.empty())  sqlite3_bind_null(ins.get(), 5);
        else sqlite3_bind_text (ins.get(), 5, reasoning.c_str(), -1, SQLITE_TRANSIENT);
        if (media_json.empty()) sqlite3_bind_null(ins.get(), 6);
        else sqlite3_bind_text (ins.get(), 6, media_json.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int  (ins.get(), 7, tokens_in);
        sqlite3_bind_int  (ins.get(), 8, tokens_out);
        sqlite3_bind_int64(ins.get(), 9, now_seconds());
        sqlite3_bind_int  (ins.get(), 10, partial ? 1 : 0);
        if (sqlite3_step(ins.get()) != SQLITE_DONE) {
            sqlite_throw(db, "step(append_message)");
        }
        const int64_t msg_id = sqlite3_last_insert_rowid(db);

        // Bump updated_at without another prepare; reuse touch_chat.
        touch_chat(db, chat_id);

        if (sqlite3_exec(db, "COMMIT", nullptr, nullptr, nullptr) != SQLITE_OK) {
            sqlite_throw(db, "commit(append_message)");
        }
        return msg_id;
    } catch (...) {
        sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
        throw;
    }
}

void delete_last_message(sqlite3 * db, int64_t chat_id) {
    StmtGuard del;
    // DELETE...WHERE seq = (SELECT MAX(seq) ...). The FTS5 trigger
    // catches the delete and removes the FTS row automatically.
    if (sqlite3_prepare_v2(db,
            "DELETE FROM messages "
            "WHERE chat_id = ? AND seq = (SELECT MAX(seq) FROM messages WHERE chat_id = ?)",
            -1, del.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(delete_last_message)");
    }
    sqlite3_bind_int64(del.get(), 1, chat_id);
    sqlite3_bind_int64(del.get(), 2, chat_id);
    if (sqlite3_step(del.get()) != SQLITE_DONE) {
        sqlite_throw(db, "step(delete_last_message)");
    }
}

std::vector<StoredMessage> load_messages(sqlite3 * db, int64_t chat_id) {
    std::vector<StoredMessage> out;
    StmtGuard q;
    const char * sql =
        "SELECT id, chat_id, seq, role, content, "
        "       COALESCE(reasoning,''), COALESCE(media_json,''), "
        "       COALESCE(tokens_in,0), COALESCE(tokens_out,0), created_at, "
        "       COALESCE(partial,0) "
        "FROM messages WHERE chat_id = ? ORDER BY seq";
    if (sqlite3_prepare_v2(db, sql, -1, q.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(load_messages)");
    }
    sqlite3_bind_int64(q.get(), 1, chat_id);
    while (sqlite3_step(q.get()) == SQLITE_ROW) {
        StoredMessage m;
        m.id          = sqlite3_column_int64(q.get(), 0);
        m.chat_id     = sqlite3_column_int64(q.get(), 1);
        m.seq         = sqlite3_column_int  (q.get(), 2);
        m.role        = col_text(q.get(), 3);
        m.content     = col_text(q.get(), 4);
        m.reasoning   = col_text(q.get(), 5);
        m.media_json  = col_text(q.get(), 6);
        m.tokens_in   = sqlite3_column_int  (q.get(), 7);
        m.tokens_out  = sqlite3_column_int  (q.get(), 8);
        m.created_at  = sqlite3_column_int64(q.get(), 9);
        m.partial     = sqlite3_column_int  (q.get(), 10) != 0;
        out.push_back(std::move(m));
    }
    return out;
}

// ===========================================================================
// search
// ===========================================================================

namespace {

// Run the FTS5 query as-is. Throws on prepare/step failure. Caller
// (`search_messages` below) catches FTS5 syntax errors and retries with
// the phrase-quoted form so user queries containing `:`, parens, or
// reserved words (AND/OR/NOT/NEAR) get matched literally instead of
// returning 500 / empty.
std::vector<SearchHit> search_messages_raw(sqlite3 *           db,
                                           const std::string & query,
                                           int                 limit) {
    std::vector<SearchHit> out;
    StmtGuard q;
    // snippet(table, col_index, prefix, suffix, ellipsis, token_count)
    // col_index=0 means the `content` column of messages_fts.
    const char * sql =
        "SELECT m.chat_id, m.id, m.seq, m.role, "
        "       snippet(messages_fts, 0, '[', ']', '...', 16) "
        "FROM messages_fts "
        "JOIN messages m ON m.id = messages_fts.rowid "
        "WHERE messages_fts MATCH ? "
        "ORDER BY rank "
        "LIMIT ?";
    if (sqlite3_prepare_v2(db, sql, -1, q.out(), nullptr) != SQLITE_OK) {
        sqlite_throw(db, "prepare(search_messages)");
    }
    sqlite3_bind_text(q.get(), 1, query.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int (q.get(), 2, limit);
    int rc;
    while ((rc = sqlite3_step(q.get())) == SQLITE_ROW) {
        SearchHit h;
        h.chat_id    = sqlite3_column_int64(q.get(), 0);
        h.message_id = sqlite3_column_int64(q.get(), 1);
        h.seq        = sqlite3_column_int  (q.get(), 2);
        h.role       = col_text(q.get(), 3);
        h.snippet    = col_text(q.get(), 4);
        out.push_back(std::move(h));
    }
    if (rc != SQLITE_DONE) {
        // FTS5 syntax errors surface here (e.g. unbalanced parens, stray
        // `:` outside a column filter). The caller distinguishes by
        // matching the message; everything else propagates as an error.
        throw ChimeraError(ExitCode::Runtime,
            std::string("step(search_messages): ") + sqlite3_errmsg(db));
    }
    return out;
}

std::optional<std::vector<SearchHit>> try_search_messages(sqlite3 *           db,
                                                          const std::string & query,
                                                          int                 limit) {
    try {
        return search_messages_raw(db, query, limit);
    } catch (const std::exception &) {
        // Any error on the raw query path falls back to phrase-quoted
        // retry. The retry input is well-formed by construction (any
        // input becomes a single FTS5 literal phrase), so a second
        // failure would have to come from something other than a query
        // syntax issue (corrupted FTS5 index, etc.) and is genuinely
        // worth propagating — which is what the caller does.
        //
        // Earlier versions narrowed this to errors mentioning "fts5",
        // "syntax error", or "no such column", but FTS5 also raises
        // "unterminated string" / "malformed MATCH" with different
        // wording per SQLite release, and the narrower set leaked
        // 500s through to the client.
        return std::nullopt;
    }
}

// FTS5-safe quoting: escape any embedded `"` by doubling it and wrap
// the whole string in double quotes. This forces FTS5 to treat the
// input as a literal phrase — punctuation, parens, and reserved words
// inside are all neutralised. Same shape as
// chimera_vector_store::fts5_phrase_quote.
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

}  // namespace

std::vector<SearchHit> search_messages(sqlite3 *           db,
                                       const std::string & query,
                                       int                 limit) {
    if (query.empty()) return {};
    auto first = try_search_messages(db, query, limit);
    if (first) return std::move(*first);
    // FTS5 syntax error: retry with the user's input wrapped as a literal
    // phrase. Mirrors the chimera_vector_store::search_lexical recovery
    // path so chat search and document search share the same UX guarantee
    // — a free-text query containing FTS5 metacharacters never errors,
    // it just matches the phrase. A second failure here is a real bug
    // and propagates.
    return search_messages_raw(db, fts5_phrase_quote(query), limit);
}

}  // namespace chimera_chat_store
