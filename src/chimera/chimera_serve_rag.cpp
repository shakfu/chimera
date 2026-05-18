// chimera_serve_rag.cpp — /v1/vector_stores/* handlers (chimera RAG: SQLite +
// sqlite-vec behind an OpenAI-shaped vector-store surface). Bound from
// chimera_serve.cpp when --enable-rag loaded an embedding model.
// Extracted from chimera_serve.cpp.

#include "chimera_serve_internal.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace chimera_serve {

namespace {

// Open one DB connection for a single handler call. SQLite open is
// microseconds in WAL mode; cheaper than running a per-thread pool
// with all the lifetime ceremony that brings.
chimera_db::Connection rag_open_db(RagContext * rag) {
    return chimera_db::open_and_migrate(
        rag->db_path.empty() ? chimera_db::default_path() : rag->db_path);
}

server_http_res_ptr rag_err(int code, const std::string & msg) {
    auto r = std::make_unique<server_http_res>();
    r->status = code;
    r->data = json{{ "error",
        { { "message", msg }, { "code", code }, { "type", "invalid_request_error" } }
    }}.dump();
    return r;
}

// Serialize a Collection into the OpenAI-ish shape we return on
// GET/POST. Fields outside OpenAI's `vector_store` schema (dim,
// embedding_model) are namespaced under `meta` rather than
// silently spliced into the OpenAI shape.
json collection_to_json(const chimera_vector_store::Collection & c) {
    return {
        { "id",        c.name },               // OpenAI keys vector stores by id
        { "object",    "vector_store" },
        { "name",      c.name },
        { "created_at", c.created_at },
        { "file_counts", { { "completed", c.doc_count }, { "total", c.doc_count } } },
        { "meta", {
            { "embedding_model", c.embedding_model },
            { "dim",             c.dim },
            { "distance",        c.distance },
            { "chunk_tokens",    c.chunk_tokens },
            { "chunk_overlap",   c.chunk_overlap },
        }},
    };
}

}  // namespace

// GET /v1/vector_stores → { "object": "list", "data": [...] }
server_http_context::handler_t make_vs_list_handler(RagContext * rag) {
    return [rag](const server_http_req & /*req*/) -> server_http_res_ptr {
        auto conn = rag_open_db(rag);
        const auto cols = chimera_vector_store::list(conn.get());
        json data = json::array();
        for (const auto & c : cols) data.push_back(collection_to_json(c));
        auto res = std::make_unique<server_http_res>();
        res->data = json{{ "object", "list" }, { "data", std::move(data) }}.dump();
        return res;
    };
}

// POST /v1/vector_stores → create. Body: { "name": "...", "embedding_model": "..."? }.
// embedding_model defaults to the model loaded at server start. If a
// different value is supplied we reject — single-model server in this cut.
server_http_context::handler_t make_vs_create_handler(RagContext * rag) {
    return [rag](const server_http_req & req) -> server_http_res_ptr {
        json body = json::object();
        if (!req.body.empty()) {
            try { body = json::parse(req.body); }
            catch (const std::exception & e) {
                return rag_err(400, std::string("invalid JSON body: ") + e.what());
            }
        }
        const std::string name = body.value("name", std::string());
        if (name.empty()) return rag_err(400, "field 'name' is required");
        const std::string model = body.value("embedding_model", rag->loaded_model);
        if (model != rag->loaded_model) {
            return rag_err(400,
                "embedding_model mismatch: server has '" + rag->loaded_model +
                "' loaded; collection requested '" + model + "'. "
                "Restart chimera serve with --enable-rag pointing at that model.");
        }
        auto conn = rag_open_db(rag);
        const int dim = rag->embedder->n_embd();
        // Optional per-collection knobs. Defaults are the same as CLI's
        // `chimera index create`. distance is validated server-side
        // (returning 400 on a bad value); chunk_* are bounds-checked by
        // chimera_vector_store::create itself.
        chimera_vector_store::CreateOptions cop;
        cop.distance      = body.value("distance",      cop.distance);
        cop.chunk_tokens  = body.value("chunk_tokens",  cop.chunk_tokens);
        cop.chunk_overlap = body.value("chunk_overlap", cop.chunk_overlap);
        if (!chimera_vector_store::is_valid_distance(cop.distance)) {
            return rag_err(400,
                "invalid distance: '" + cop.distance +
                "' (expected one of: cosine, l2, l1)");
        }
        auto col = chimera_vector_store::create(conn.get(), name, model, dim, cop);
        auto res = std::make_unique<server_http_res>();
        res->status = 201;
        res->data = collection_to_json(col).dump();
        return res;
    };
}

// GET /v1/vector_stores/:name → stats for one collection.
server_http_context::handler_t make_vs_get_handler(RagContext * rag) {
    return [rag](const server_http_req & req) -> server_http_res_ptr {
        const std::string name = req.get_param("name");
        if (name.empty()) return rag_err(400, "missing :name path param");
        auto conn = rag_open_db(rag);
        // Pull from list() so we get the doc_count via correlated subquery.
        for (const auto & c : chimera_vector_store::list(conn.get())) {
            if (c.name == name) {
                auto res = std::make_unique<server_http_res>();
                res->data = collection_to_json(c).dump();
                return res;
            }
        }
        // Returning 400 (not 404) keeps our error message visible —
        // server-http's set_error_handler unconditionally overwrites
        // any 404 body with the upstream-generic "File Not Found".
        return rag_err(400, "no such collection: '" + name + "'");
    };
}

// POST /v1/vector_stores/:name/delete → drop. (Not DELETE; see comment
// near the route registration.)
server_http_context::handler_t make_vs_delete_handler(RagContext * rag) {
    return [rag](const server_http_req & req) -> server_http_res_ptr {
        const std::string name = req.get_param("name");
        if (name.empty()) return rag_err(400, "missing :name path param");
        auto conn = rag_open_db(rag);
        try {
            chimera_vector_store::drop(conn.get(), name);
        } catch (const ChimeraError & e) {
            return rag_err(400, e.what());
        }
        auto res = std::make_unique<server_http_res>();
        res->data = json{{ "id", name }, { "deleted", true },
                          { "object", "vector_store.deleted" }}.dump();
        return res;
    };
}

// POST /v1/vector_stores/:name/files → chunk + embed + insert.
// Body forms accepted:
//   multipart/form-data with a `file` upload (and optional `source_uri`)
//   application/json: { "text": "...", "source_uri": "..." }
server_http_context::handler_t make_vs_ingest_handler(RagContext * rag) {
    return [rag](const server_http_req & req) -> server_http_res_ptr {
        const std::string name = req.get_param("name");
        if (name.empty()) return rag_err(400, "missing :name path param");

        // Resolve text input + source_uri. server-http folds multipart
        // text fields into req.body as JSON; for application/json the
        // body is whatever the client sent.
        std::string text, source_uri;
        if (!req.files.empty()) {
            auto it = req.files.find("file");
            if (it == req.files.end() || it->second.data.empty()) {
                return rag_err(400, "missing 'file' field in multipart form");
            }
            text.assign(reinterpret_cast<const char *>(it->second.data.data()),
                        it->second.data.size());
            source_uri = it->second.filename;
        }
        if (!req.body.empty()) {
            json b;
            try { b = json::parse(req.body); }
            catch (const std::exception &) { b = json::object(); }
            if (text.empty() && b.contains("text") && b["text"].is_string()) {
                text = b["text"].get<std::string>();
            }
            if (source_uri.empty() && b.contains("source_uri") && b["source_uri"].is_string()) {
                source_uri = b["source_uri"].get<std::string>();
            }
        }
        if (text.empty()) {
            return rag_err(400,
                "request must include either a multipart 'file' upload or a JSON 'text' field");
        }

        auto conn = rag_open_db(rag);
        auto col = chimera_vector_store::find(conn.get(), name);
        if (!col) return rag_err(400, "no such collection: '" + name + "'");
        if (col->embedding_model != rag->loaded_model) {
            return rag_err(400,
                "collection '" + name + "' was indexed with '" + col->embedding_model +
                "'; server has '" + rag->loaded_model + "' loaded.");
        }

        // Token-based chunking matching the CLI: window + overlap come
        // from the collection row. Tokenize / detokenize on the shared
        // Embedder; serialize behind embedder_mutex for the whole chunk
        // pass (the vocab is read-only but we also lock for the
        // subsequent embed() call below, so one lock per chunk loop is
        // simpler than scoping it tighter).
        std::vector<chimera_embed::TokenChunk> chunks;
        {
            std::lock_guard<std::mutex> lk(rag->embedder_mutex);
            chunks = chimera_embed::chunk_by_sentences(
                text, *rag->embedder, col->chunk_tokens, col->chunk_overlap);
        }
        if (chunks.empty()) return rag_err(400, "no non-empty chunks produced");

        int64_t inserted = 0;
        for (const auto & c : chunks) {
            std::vector<float> vec;
            {
                std::lock_guard<std::mutex> lk(rag->embedder_mutex);
                vec = rag->embedder->embed(c.text);
            }
            chimera_vector_store::DocumentInput doc;
            doc.source_uri  = source_uri;
            doc.chunk_index = c.index;
            doc.text        = c.text;
            doc.token_count = c.token_count;
            doc.embedding   = std::move(vec);
            chimera_vector_store::insert_document(conn.get(), *col, doc);
            ++inserted;
        }

        auto res = std::make_unique<server_http_res>();
        res->data = json{
            { "object",      "vector_store.file" },
            { "vector_store_id", name },
            { "source_uri",  source_uri },
            { "chunks_inserted", inserted },
        }.dump();
        return res;
    };
}

// POST /v1/vector_stores/:name/search → KNN.
// Body: { "query": "...", "k": 5? }
server_http_context::handler_t make_vs_search_handler(RagContext * rag) {
    return [rag](const server_http_req & req) -> server_http_res_ptr {
        const std::string name = req.get_param("name");
        if (name.empty()) return rag_err(400, "missing :name path param");

        json body = json::object();
        if (!req.body.empty()) {
            try { body = json::parse(req.body); }
            catch (const std::exception & e) {
                return rag_err(400, std::string("invalid JSON body: ") + e.what());
            }
        }
        const std::string query = body.value("query", std::string());
        if (query.empty()) return rag_err(400, "field 'query' is required");
        const int k = coerce_int(body.value("k", json(5)), 5);

        // Body "mode" defaults to hybrid — strictly better than the
        // pre-hybrid semantic-only behaviour on prose, and clients that
        // need the old behaviour can pass "semantic" explicitly.
        const std::string mode_str = body.value("mode", std::string("hybrid"));
        const auto mode_opt = chimera_vector_store::parse_search_mode(mode_str);
        if (!mode_opt) {
            return rag_err(400,
                "invalid mode '" + mode_str +
                "' (expected: semantic | lexical | hybrid)");
        }
        const auto mode = *mode_opt;

        auto conn = rag_open_db(rag);
        auto col = chimera_vector_store::find(conn.get(), name);
        if (!col) return rag_err(400, "no such collection: '" + name + "'");
        if (mode != chimera_vector_store::SearchMode::Lexical &&
            col->embedding_model != rag->loaded_model) {
            return rag_err(400,
                "collection '" + name + "' was indexed with '" + col->embedding_model +
                "'; server has '" + rag->loaded_model + "' loaded.");
        }

        std::vector<float> qvec;
        if (mode != chimera_vector_store::SearchMode::Lexical) {
            std::lock_guard<std::mutex> lk(rag->embedder_mutex);
            qvec = rag->embedder->embed(query);
        }
        const auto hits = chimera_vector_store::search(
            conn.get(), *col, qvec, query, k, mode);

        json data = json::array();
        for (const auto & h : hits) {
            json item = {
                { "document_id", h.document_id },
                { "source_uri",  h.source_uri },
                { "chunk_index", h.chunk_index },
                { "text",        h.text },
                { "distance",    h.distance },
            };
            if (mode == chimera_vector_store::SearchMode::Hybrid) {
                item["rrf_score"]     = h.rrf_score;
                item["semantic_rank"] = h.semantic_rank;
                item["lexical_rank"]  = h.lexical_rank;
            }
            data.push_back(std::move(item));
        }
        auto res = std::make_unique<server_http_res>();
        res->data = json{
            { "object", "list" },
            { "mode",   chimera_vector_store::search_mode_name(mode) },
            { "data",   std::move(data) }
        }.dump();
        return res;
    };
}

}  // namespace chimera_serve
