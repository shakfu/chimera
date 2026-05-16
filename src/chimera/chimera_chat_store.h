// chimera_chat_store.h — chat persistence on top of the v1 `chats` +
// `messages` + `messages_fts` tables (see chimera_db.cpp's MIGRATION_V1).
//
// Phase-3 scope: consumed only by `command_chat` (CLI). Phase 5 will
// add server-side use from chimera_serve.cpp.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct sqlite3;

namespace chimera_chat_store {

struct Chat {
    int64_t     id;
    int64_t     created_at;
    int64_t     updated_at;
    std::string title;
    std::string model_path;
    std::string model_alias;
    std::string system_prompt;
    std::string source;          // 'chat' | 'serve'
    std::string metadata_json;
    int64_t     message_count;   // populated by list_chats()
};

struct StoredMessage {
    int64_t     id;
    int64_t     chat_id;
    int         seq;
    std::string role;            // 'user', 'assistant', 'system', 'tool'
    std::string content;
    std::string reasoning;       // optional; <think>...</think> span
    std::string media_json;      // optional; serialized list of attached media paths
    int         tokens_in;
    int         tokens_out;
    int64_t     created_at;
};

struct SearchHit {
    int64_t     chat_id;
    int64_t     message_id;
    int         seq;
    std::string role;
    std::string snippet;         // FTS5-marked highlight
};

// ---- chats -------------------------------------------------------------

// Create a new chat row and return its id. system_prompt may be empty.
int64_t create_chat(sqlite3 *           db,
                    const std::string & model_path,
                    const std::string & model_alias,
                    const std::string & system_prompt,
                    const std::string & source,
                    const std::string & metadata_json = "");

// Load chat metadata by id. Returns nullopt if not found.
std::optional<Chat> load_chat(sqlite3 * db, int64_t chat_id);

// Most recent chat (largest updated_at). Useful for `--resume last`.
std::optional<Chat> latest_chat(sqlite3 * db);

// List chats sorted by updated_at DESC, limited to `limit`. Includes
// per-row message_count via a correlated subquery.
std::vector<Chat> list_chats(sqlite3 * db, int limit = 50);

// Touch updated_at on the chat. Called after each appended message so
// list ordering reflects activity.
void touch_chat(sqlite3 * db, int64_t chat_id);

// Set the chat title. Used to lazily record a user-friendly label
// (first user turn truncated, or a model-generated summary later).
void set_chat_title(sqlite3 * db, int64_t chat_id, const std::string & title);

// ---- messages ----------------------------------------------------------

// Append one message to a chat. Returns the message id. The `seq`
// column is auto-assigned (next available within the chat). Triggers
// keep messages_fts in sync.
int64_t append_message(sqlite3 *           db,
                       int64_t             chat_id,
                       const std::string & role,
                       const std::string & content,
                       const std::string & reasoning   = "",
                       const std::string & media_json  = "",
                       int                 tokens_in   = 0,
                       int                 tokens_out  = 0);

// Drop the last message in a chat (largest seq). Used by --resume +
// /regen on a persistent session, and for cleanup paths.
void delete_last_message(sqlite3 * db, int64_t chat_id);

// All messages for a chat, ordered by seq ascending.
std::vector<StoredMessage> load_messages(sqlite3 * db, int64_t chat_id);

// ---- search ------------------------------------------------------------

// Full-text search over messages_fts. Returns top `limit` hits with
// FTS5 `snippet()` highlights ([word] markers around matched terms).
std::vector<SearchHit> search_messages(sqlite3 *           db,
                                       const std::string & query,
                                       int                 limit = 20);

}  // namespace chimera_chat_store
