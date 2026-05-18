// chimera_serve_chats_read.cpp — GET /v1/chats[/...] read-side endpoints.
// chimera's persistence write path (--persist-chats wrapper around
// /v1/chat/completions) lands rows in the chats + messages tables.
// These three GET endpoints expose the read side over HTTP so a web
// UI can browse and search persisted conversations without going
// through the CLI. Bound only when --persist-chats is set so the
// db_path is known (and pairing the read side with the write side
// avoids the surprise of "I never asked you to persist anything, why
// are you exposing my chats?"). Extracted from chimera_serve.cpp.

#include "chimera_serve_internal.h"

#include <cstdint>
#include <memory>
#include <string>

namespace chimera_serve {

namespace {

chimera_db::Connection chat_hist_open_db(ChatHistoryContext * ctx) {
    return chimera_db::open_and_migrate(
        ctx->db_path.empty() ? chimera_db::default_path() : ctx->db_path);
}

server_http_res_ptr chat_hist_err(int code, const std::string & msg) {
    auto r = std::make_unique<server_http_res>();
    r->status = code;
    r->data = json{{ "error",
        { { "message", msg }, { "code", code }, { "type", "invalid_request_error" } }
    }}.dump();
    return r;
}

json chat_to_json(const chimera_chat_store::Chat & c) {
    return {
        { "id",             c.id },
        { "object",         "chimera.chat" },
        { "created_at",     c.created_at },
        { "updated_at",     c.updated_at },
        { "title",          c.title },
        { "model_path",     c.model_path },
        { "model_alias",    c.model_alias },
        { "system_prompt",  c.system_prompt },
        { "source",         c.source },
        { "message_count",  c.message_count },
        { "partial_count",  c.partial_count },
    };
}

json stored_message_to_json(const chimera_chat_store::StoredMessage & m) {
    return {
        { "id",          m.id },
        { "chat_id",     m.chat_id },
        { "seq",         m.seq },
        { "role",        m.role },
        { "content",     m.content },
        { "reasoning",   m.reasoning },
        { "media_json",  m.media_json },
        { "tokens_in",   m.tokens_in },
        { "tokens_out",  m.tokens_out },
        { "created_at",  m.created_at },
        { "partial",     m.partial },
    };
}

}  // namespace

// GET /v1/chats?limit=N  → { "object": "list", "data": [chat, ...] }
// Sorted by updated_at desc (most recently active first). Default limit
// 50, capped at 500 to keep the response bounded.
server_http_context::handler_t make_chats_list_handler(ChatHistoryContext * ctx) {
    return [ctx](const server_http_req & req) -> server_http_res_ptr {
        int limit = 50;
        const std::string lim = req.get_param("limit");
        if (!lim.empty()) {
            try { limit = std::stoi(lim); }
            catch (const std::exception &) { return chat_hist_err(400, "invalid limit"); }
            if (limit < 1 || limit > 500) {
                return chat_hist_err(400, "limit must be in [1, 500]");
            }
        }
        auto conn = chat_hist_open_db(ctx);
        const auto chats = chimera_chat_store::list_chats(conn.get(), limit);
        json data = json::array();
        for (const auto & c : chats) data.push_back(chat_to_json(c));
        auto res = std::make_unique<server_http_res>();
        res->data = json{{ "object", "list" }, { "data", std::move(data) }}.dump();
        return res;
    };
}

// GET /v1/chats/:id  → { chat fields ..., "messages": [...] }
// Messages ordered by seq ascending; includes partial-turn rows.
server_http_context::handler_t make_chats_get_handler(ChatHistoryContext * ctx) {
    return [ctx](const server_http_req & req) -> server_http_res_ptr {
        const std::string id_str = req.get_param("id");
        if (id_str.empty()) return chat_hist_err(400, "missing :id path param");
        int64_t id;
        try { id = std::stoll(id_str); }
        catch (const std::exception &) { return chat_hist_err(400, "invalid chat id"); }
        auto conn = chat_hist_open_db(ctx);
        const auto chat = chimera_chat_store::load_chat(conn.get(), id);
        if (!chat) {
            // 400 (not 404): cpp-httplib's default error handler overwrites
            // 404 bodies with "File Not Found". Same workaround we use in
            // the vector store handlers.
            return chat_hist_err(400, "no such chat: " + std::to_string(id));
        }
        const auto messages = chimera_chat_store::load_messages(conn.get(), id);
        json mjson = json::array();
        for (const auto & m : messages) mjson.push_back(stored_message_to_json(m));
        json body = chat_to_json(*chat);
        body["messages"] = std::move(mjson);
        auto res = std::make_unique<server_http_res>();
        res->data = body.dump();
        return res;
    };
}

// GET /v1/chats/search?q=...&limit=N
//   → { "query": "...", "hits": [{chat_id, message_id, seq, role, snippet}, ...] }
// Hits include FTS5 [word]-highlighted snippets. Default limit 20,
// capped at 200.
server_http_context::handler_t make_chats_search_handler(ChatHistoryContext * ctx) {
    return [ctx](const server_http_req & req) -> server_http_res_ptr {
        const std::string q = req.get_param("q");
        if (q.empty()) return chat_hist_err(400, "missing required query param 'q'");
        int limit = 20;
        const std::string lim = req.get_param("limit");
        if (!lim.empty()) {
            try { limit = std::stoi(lim); }
            catch (const std::exception &) { return chat_hist_err(400, "invalid limit"); }
            if (limit < 1 || limit > 200) {
                return chat_hist_err(400, "limit must be in [1, 200]");
            }
        }
        auto conn = chat_hist_open_db(ctx);
        const auto hits = chimera_chat_store::search_messages(conn.get(), q, limit);
        json data = json::array();
        for (const auto & h : hits) {
            data.push_back({
                { "chat_id",    h.chat_id },
                { "message_id", h.message_id },
                { "seq",        h.seq },
                { "role",       h.role },
                { "snippet",    h.snippet },
            });
        }
        auto res = std::make_unique<server_http_res>();
        res->data = json{{ "query", q }, { "hits", std::move(data) }}.dump();
        return res;
    };
}

}  // namespace chimera_serve
