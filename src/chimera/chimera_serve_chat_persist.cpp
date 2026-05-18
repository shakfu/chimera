// chimera_serve_chat_persist.cpp — /v1/chat/completions persistence wrapper.
// When --persist-chats is set, chimera_serve.cpp wraps the upstream
// chat-completions handler with make_persisting_chat_handler so each
// successful exchange is saved to the chats + messages tables.
// Extracted from chimera_serve.cpp.

#include "chimera_serve_internal.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace chimera_serve {

namespace {

// Extract a single message's content to a string. OpenAI permits
// `content` to be either a string, an array of content parts (image_url,
// text), or absent. We stringify non-string content as JSON so the row
// remains lossless.
std::string extract_message_content(const json & m) {
    if (!m.contains("content")) return {};
    const auto & c = m["content"];
    if (c.is_string()) return c.get<std::string>();
    return c.dump();
}

// Pull the system prompt (first system-role message with string content)
// out of a chat-completions body for the chats.system_prompt column.
std::string extract_system_prompt(const json & req_body) {
    if (!req_body.contains("messages") || !req_body["messages"].is_array()) return {};
    for (const auto & m : req_body["messages"]) {
        if (m.contains("role") && m["role"] == "system" &&
            m.contains("content") && m["content"].is_string()) {
            return m["content"].get<std::string>();
        }
    }
    return {};
}

// Create a new chats row up-front so its id can be set in the
// X-Chimera-Chat-Id response header before streaming begins. Returns 0
// on failure (logged); the caller treats 0 as "do not persist messages,
// do not echo header".
int64_t create_chat_row_for_request(ChatPersistContext * cpc,
                                    const json &         req_body) {
    try {
        chimera_db::Connection conn = chimera_db::open_and_migrate(
            cpc->db_path.empty() ? chimera_db::default_path() : cpc->db_path);

        std::string model_alias = "any";
        if (req_body.contains("model") && req_body["model"].is_string()) {
            model_alias = req_body["model"].get<std::string>();
        }
        const std::string system_prompt = extract_system_prompt(req_body);

        std::lock_guard<std::mutex> lk(cpc->mutex);
        return chimera_chat_store::create_chat(
            conn.get(),
            /*model_path=*/model_alias, model_alias,
            system_prompt, /*source=*/"serve");
    } catch (const std::exception & e) {
        std::fprintf(stderr,
                     "chimera serve: create_chat_row_for_request failed: %s\n",
                     e.what());
        return 0;
    }
}

// Verify a client-supplied X-Chimera-Chat-Id refers to an existing row.
// Returns true on success.
bool chat_id_exists(ChatPersistContext * cpc, int64_t chat_id) {
    try {
        chimera_db::Connection conn = chimera_db::open_and_migrate(
            cpc->db_path.empty() ? chimera_db::default_path() : cpc->db_path);
        std::lock_guard<std::mutex> lk(cpc->mutex);
        return chimera_chat_store::load_chat(conn.get(), chat_id).has_value();
    } catch (...) {
        return false;
    }
}

// Write a completed exchange to the DB. `chat_id` MUST already exist
// (created by `create_chat_row_for_request` or supplied by the client
// via X-Chimera-Chat-Id). When `append_full_history` is true, every
// message in the request body is inserted; when false, only the last
// message (intended to be the new user turn the client just appended)
// is inserted. The new assistant reply is always appended.
//
// The split exists because clients that echo X-Chimera-Chat-Id back
// resend the full conversation each request, but the prior turns are
// already on disk from earlier requests against this chat_id — only
// the new user message + assistant reply should be added.
void persist_completed_chat(ChatPersistContext * cpc,
                            int64_t             chat_id,
                            bool                append_full_history,
                            const json &        req_body,
                            const std::string & assistant_content,
                            const std::string & assistant_reasoning,
                            int                 tokens_in,
                            int                 tokens_out) {
    if (chat_id == 0) return;  // create failed earlier; nothing to do.
    try {
        if (!req_body.contains("messages") || !req_body["messages"].is_array()) return;

        chimera_db::Connection conn = chimera_db::open_and_migrate(
            cpc->db_path.empty() ? chimera_db::default_path() : cpc->db_path);

        std::lock_guard<std::mutex> lk(cpc->mutex);

        const auto & msgs = req_body["messages"];
        if (append_full_history) {
            for (const auto & m : msgs) {
                const std::string role = m.value("role", std::string("user"));
                chimera_chat_store::append_message(
                    conn.get(), chat_id, role, extract_message_content(m));
            }
        } else if (!msgs.empty()) {
            // Echo path: append only the new user turn. We don't try to
            // diff against what's already in the DB — clients that
            // misuse the header (e.g. resend a full history under an
            // existing chat_id) will get duplicate rows. That's their
            // contract violation.
            const auto & last = msgs.at(msgs.size() - 1);
            const std::string role = last.value("role", std::string("user"));
            chimera_chat_store::append_message(
                conn.get(), chat_id, role, extract_message_content(last));
        }
        chimera_chat_store::append_message(
            conn.get(), chat_id, "assistant", assistant_content,
            assistant_reasoning, /*media_json=*/"", tokens_in, tokens_out);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "chimera serve: persist_completed_chat failed: %s\n",
                     e.what());
    }
}

// Parse a non-streaming /v1/chat/completions response body and pull out
// the assistant content + token counts.
void persist_non_streaming(ChatPersistContext * cpc,
                            int64_t             chat_id,
                            bool                append_full_history,
                            const json &        req_body,
                            const std::string & res_data) {
    try {
        json r = json::parse(res_data);
        if (!r.contains("choices") || !r["choices"].is_array() || r["choices"].empty()) {
            return;
        }
        const auto & ch = r["choices"][0];
        std::string content, reasoning;
        if (ch.contains("message") && ch["message"].is_object()) {
            const auto & m = ch["message"];
            if (m.contains("content") && m["content"].is_string()) {
                content = m["content"].get<std::string>();
            }
            if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
                reasoning = m["reasoning_content"].get<std::string>();
            }
        }
        int tokens_in = 0, tokens_out = 0;
        if (r.contains("usage") && r["usage"].is_object()) {
            tokens_in  = r["usage"].value("prompt_tokens",     0);
            tokens_out = r["usage"].value("completion_tokens", 0);
        }
        persist_completed_chat(cpc, chat_id, append_full_history, req_body,
                                content, reasoning, tokens_in, tokens_out);
    } catch (...) {
        // Don't propagate.
    }
}

// Parse a streaming response (buffered SSE chunks already collected) and
// persist the result. SSE shape: `data: {json}\n\n` per chunk, plus a
// `data: [DONE]` trailer.
void persist_streaming(ChatPersistContext * cpc,
                        int64_t             chat_id,
                        bool                append_full_history,
                        const json &        req_body,
                        const std::string & sse_buffer) {
    std::string content, reasoning;
    int tokens_in = 0, tokens_out = 0;
    size_t pos = 0;
    while (pos < sse_buffer.size()) {
        // Each SSE event ends with "\n\n". Find the next boundary.
        const size_t end = sse_buffer.find("\n\n", pos);
        if (end == std::string::npos) break;
        const std::string event = sse_buffer.substr(pos, end - pos);
        pos = end + 2;

        // Strip the "data: " prefix on each line of the event. (Multi-line
        // data: events get concatenated; rare here but cheap to handle.)
        std::string payload;
        size_t line_start = 0;
        while (line_start < event.size()) {
            const size_t lend = event.find('\n', line_start);
            const std::string line = event.substr(line_start, lend - line_start);
            if (line.rfind("data: ", 0) == 0) {
                payload += line.substr(6);
            }
            if (lend == std::string::npos) break;
            line_start = lend + 1;
        }
        if (payload.empty() || payload == "[DONE]") continue;
        try {
            json j = json::parse(payload);
            if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
                const auto & delta = j["choices"][0].value("delta", json::object());
                if (delta.contains("content") && delta["content"].is_string()) {
                    content += delta["content"].get<std::string>();
                }
                if (delta.contains("reasoning_content") && delta["reasoning_content"].is_string()) {
                    reasoning += delta["reasoning_content"].get<std::string>();
                }
            }
            // Usage may appear in the final chunk (OpenAI sends it when
            // stream_options.include_usage = true).
            if (j.contains("usage") && j["usage"].is_object()) {
                tokens_in  = j["usage"].value("prompt_tokens",     tokens_in);
                tokens_out = j["usage"].value("completion_tokens", tokens_out);
            }
        } catch (...) {
            // Skip malformed event; keep parsing.
        }
    }
    if (!content.empty() || !reasoning.empty()) {
        persist_completed_chat(cpc, chat_id, append_full_history, req_body,
                                content, reasoning, tokens_in, tokens_out);
    }
}

// Case-insensitive HTTP header lookup. HTTP header names are
// case-insensitive; cpp-httplib preserves the client's casing in the
// std::map, so we walk and compare. Hand-rolled to keep this TU
// independent of cpp-httplib's internal helpers.
std::string lookup_header_ci(const std::map<std::string, std::string> & headers,
                             const std::string & name) {
    for (const auto & kv : headers) {
        if (kv.first.size() != name.size()) continue;
        bool eq = true;
        for (size_t i = 0; i < name.size(); ++i) {
            const char a = static_cast<char>(std::tolower(
                static_cast<unsigned char>(kv.first[i])));
            const char b = static_cast<char>(std::tolower(
                static_cast<unsigned char>(name[i])));
            if (a != b) { eq = false; break; }
        }
        if (eq) return kv.second;
    }
    return {};
}

}  // namespace

// Wraps server_routes::post_chat_completions so each successful exchange
// is saved to the chats + messages tables. Behaviour:
//   - No X-Chimera-Chat-Id on the request: create a new chats row before
//     calling inner, echo its id back as X-Chimera-Chat-Id on the
//     response, persist *all* request messages plus the assistant reply.
//   - X-Chimera-Chat-Id present and refers to an existing chat: reuse
//     it; persist only the last message in the body plus the assistant
//     reply (the prior turns are already on disk from earlier calls).
//   - X-Chimera-Chat-Id present but malformed or unknown: HTTP 404 with
//     a JSON error body; inner is not invoked.
//
// Streaming responses are passed through chunk-by-chunk and assembled in
// a buffer for parsing at stream end. Persistence runs *after* the chunk
// is sent to the client; any error in persistence is logged but never
// visible to the caller. The X-Chimera-Chat-Id response header is set
// before returning so it reaches the client even on streaming responses
// (where the body comes through `next`).
server_http_context::handler_t make_persisting_chat_handler(
    server_http_context::handler_t inner,
    ChatPersistContext *           cpc) {
    return [inner, cpc](const server_http_req & req) -> server_http_res_ptr {
        json req_body;
        try { req_body = json::parse(req.body); }
        catch (...) {
            // Malformed JSON — let the inner handler emit its own 400.
            return inner(req);
        }

        const std::string id_hdr = lookup_header_ci(req.headers, X_CHAT_ID_HEADER);
        int64_t  client_chat_id    = 0;
        bool     append_full_hist  = true;
        if (!id_hdr.empty()) {
            try {
                client_chat_id = std::stoll(id_hdr);
            } catch (...) {
                auto res = std::make_unique<server_http_res>();
                res->status = 400;
                res->data = json{
                    { "error", { { "message",
                        std::string("invalid ") + X_CHAT_ID_HEADER +
                        " header: '" + id_hdr + "' (expected integer)" },
                                 { "type", "invalid_request_error" } } }
                }.dump();
                return res;
            }
            if (!chat_id_exists(cpc, client_chat_id)) {
                auto res = std::make_unique<server_http_res>();
                res->status = 404;
                res->data = json{
                    { "error", { { "message",
                        std::string("no such chat id: ") + std::to_string(client_chat_id) },
                                 { "type", "not_found" } } }
                }.dump();
                return res;
            }
            append_full_hist = false;
        } else {
            // Create the row up-front so we can advertise the id on the
            // response header before streaming starts. If creation fails
            // we still serve the response — the header is just omitted.
            client_chat_id = create_chat_row_for_request(cpc, req_body);
        }

        auto res = inner(req);
        if (!res || res->status >= 300) return res;

        if (client_chat_id != 0) {
            res->headers[X_CHAT_ID_HEADER] = std::to_string(client_chat_id);
        }

        if (!res->is_stream()) {
            // Non-streaming: we have the full body in res->data.
            persist_non_streaming(cpc, client_chat_id, append_full_hist,
                                   req_body, res->data);
            return res;
        }

        // Streaming: wrap `next` to mirror each chunk into a buffer,
        // then parse + persist when the stream ends. The shared_ptr
        // keeps the buffer alive across the move-from of res->next.
        auto buffer     = std::make_shared<std::string>();
        auto inner_next = std::move(res->next);
        auto saved_body = std::make_shared<json>(std::move(req_body));
        res->next = [inner_next, buffer, cpc, saved_body,
                     client_chat_id, append_full_hist](std::string & out) {
            const bool has_more = inner_next(out);
            buffer->append(out);
            if (!has_more) {
                persist_streaming(cpc, client_chat_id, append_full_hist,
                                   *saved_body, *buffer);
            }
            return has_more;
        };
        return res;
    };
}

}  // namespace chimera_serve
