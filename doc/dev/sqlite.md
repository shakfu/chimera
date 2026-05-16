# Embedding SQLite + sqlite-vec in chimera

This is a planning document, not a built feature. It captures the
analysis behind the decision, the two driving use cases, and a phased
plan for landing the work.

---

## 1. Why

### Primary driver: RAG / vector store

chimera already loads embedding models (`chimera embed`,
`POST /v1/embeddings`). What it does not have is any way to *store*
those embeddings, index them, or query them by similarity. Building a
RAG-shaped feature on top of chimera means we need:

- a place to keep `(document_id, text, embedding[float32])` rows;
- approximate-nearest-neighbor search over the embedding column;
- some kind of metadata schema so documents have titles, source URIs,
  chunk indices, ingestion timestamps.

The clean, well-tested answer for an embedded application is
**SQLite + [sqlite-vec][vec]**. sqlite-vec is the actively maintained
successor to the older sqlite-vss; it ships as a single 50 KB
amalgamation; it exposes `vec0` virtual tables that store float32 (or
quantized int8/binary) vectors with KNN search via `MATCH`. It runs
in-process, has no compile-time dependency on a BLAS/FAISS-shaped
library, and the storage format is just rows in the same DB file as
everything else chimera persists.

The alternative answers — bolt on a flat-file index, depend on FAISS,
host a separate vector database — all involve more code, more
operational surface, or both, for chimera's "embed in a single static
binary" deployment story.

[vec]: https://github.com/asg017/sqlite-vec

### Secondary driver: persistent chat history across sessions

`chimera chat` currently writes line-readline history to
`~/.chimera_chat_history`. That captures *what the user typed*, not
*the assistant's replies, the system prompt, the model used, the
sampling params, attached media, or the conversation timestamps*. It
also doesn't compose with `chimera serve`: an HTTP client and an
interactive CLI session can't share a notion of "this is conversation
#42."

A small `chats` + `messages` schema in the same SQLite database makes
both consumers point at one persistent log. The CLI can resume past
sessions; the server (eventually) can implement `POST /v1/responses`
(OpenAI's stateful API) on top of the same store; users can grep,
diff, and back up their chat history with off-the-shelf tools.

### Nice-to-have follow-ons (not motivating the work)

- **Embedding cache** — memoize `embed(text)` results when the same
  text is embedded repeatedly during corpus ingestion.
- **Audit log** for `chimera serve` — request/response metadata
  including which model, which slot, how many tokens.
- **Per-API-key rate-limit state**, if `chimera serve` ever grows
  multi-tenancy.
- **`/slots/*` persistence** for KV-cache snapshots.

None of those individually justifies bringing SQLite in. With it in
the build for RAG + chat, they become cheap.

---

## 2. Build pipeline

### Vendoring

Both libraries are designed for single-translation-unit embedding.
`scripts/manage.py` already knows how to fetch + amalgamate dependencies
into `thirdparty/<name>/{include,lib,src-aux}/`; SQLite and sqlite-vec
follow the same shape but ship pre-amalgamated, so the fetch step is
simpler than for llama.cpp.

```
thirdparty/sqlite/
    include/sqlite3.h          # public API header
    src-aux/sqlite3.c          # amalgamation, compiled into chimera target

thirdparty/sqlite-vec/
    include/sqlite-vec.h
    src-aux/sqlite-vec.c       # amalgamation
```

`manage.py` adds an `SQLiteBuilder` that:

1. Downloads the SQLite amalgamation tarball (pinned version) from
   `sqlite.org`.
2. Unpacks `sqlite3.c` + `sqlite3.h` into `thirdparty/sqlite/`.
3. Downloads sqlite-vec's release tarball (pinned tag) from GitHub.
4. Unpacks `sqlite-vec.c` + `sqlite-vec.h` into `thirdparty/sqlite-vec/`.

No CMake configure step is needed for either — they're plain C with no
external dependencies. We don't build a static library; we compile the
`.c` files directly into the chimera target, the same way `server-http.cpp`
is handled today.

### CMake wiring

In `src/chimera/CMakeLists.txt`:

```cmake
target_sources(chimera PRIVATE
    "${CMAKE_SOURCE_DIR}/thirdparty/sqlite/src-aux/sqlite3.c"
    "${CMAKE_SOURCE_DIR}/thirdparty/sqlite-vec/src-aux/sqlite-vec.c"
)
target_include_directories(chimera PRIVATE
    "${CMAKE_SOURCE_DIR}/thirdparty/sqlite/include"
    "${CMAKE_SOURCE_DIR}/thirdparty/sqlite-vec/include"
)
```

SQLite compile-time flags worth setting via `target_compile_definitions`:

| Flag | Why |
|------|-----|
| `SQLITE_DQS=0` | Disable double-quoted-string-as-identifier hack. Catches buggy queries at compile time. |
| `SQLITE_DEFAULT_MEMSTATUS=0` | Skip the per-allocation accounting we don't read. |
| `SQLITE_DEFAULT_WAL_SYNCHRONOUS=1` | NORMAL is the right default in WAL mode. |
| `SQLITE_LIKE_DOESNT_MATCH_BLOBS` | We don't store blobs we'd LIKE against. |
| `SQLITE_MAX_EXPR_DEPTH=0` | Trivial query-planner speedup. |
| `SQLITE_OMIT_DECLTYPE`, `SQLITE_OMIT_DEPRECATED`, `SQLITE_OMIT_PROGRESS_CALLBACK`, `SQLITE_OMIT_SHARED_CACHE` | Trim ~50 KB of code we don't need. |
| `SQLITE_THREADSAFE=2` | "Multi-thread" mode: one connection per thread, no internal serialization. We'll manage one connection per worker. |
| `SQLITE_ENABLE_FTS5` | Full-text search for chat-content queries. |
| `SQLITE_ENABLE_MATH_FUNCTIONS` | sqlite-vec uses some of these. |

Approximate binary cost: **~1.5 MB** for sqlite3.c, **~60 KB** for
sqlite-vec.c, both stripped & release-built.

### `manage.py` changes

```python
class SQLiteBuilder(Builder):
    name = "sqlite"
    version = SQLITE_VERSION                  # pinned in __init__
    repo_url = "https://sqlite.org/2025/sqlite-amalgamation-...zip"
    libs = []                                 # no .a — header + source only

    def build(self):
        # fetch amalgamation, copy sqlite3.{c,h}
        # nothing to configure or compile here

class SqliteVecBuilder(Builder):
    name = "sqlite-vec"
    version = SQLITE_VEC_VERSION
    repo_url = "https://github.com/asg017/sqlite-vec.git"

    def build(self):
        # checkout tag, copy sqlite-vec.{c,h} from the release dir
```

Both are added to `--all`/`--deps-only` so the existing `make deps`
flow picks them up.

### Top-level `CMakeLists.txt`

```cmake
set(SQLITE_DIR     "${CMAKE_SOURCE_DIR}/thirdparty/sqlite")
set(SQLITE_VEC_DIR "${CMAKE_SOURCE_DIR}/thirdparty/sqlite-vec")
```

No `static_lib(...)` call — the `.c` files are pulled into the chimera
target directly, like `server-http.cpp` already is.

---

## 3. Module layout

| File | Responsibility |
|------|---------------|
| `src/chimera/chimera_db.h` | Public types: `DbConnection`, `Migration`, error types. Shared by every TU that talks to the DB. |
| `src/chimera/chimera_db.cpp` | Connection lifecycle, schema migration runner, prepared-statement helpers, sqlite-vec extension loader. |
| `src/chimera/chimera_chat_store.cpp` | High-level operations on the chat schema: `create_chat`, `append_message`, `list_chats`, `search_messages`. Consumed by `command_chat` and (later) the serve-side session routes. |
| `src/chimera/chimera_vector_store.cpp` | High-level operations on the vector schema: `create_collection`, `insert_document`, `search_similar`. Consumed by future `chimera index` / `chimera search` and `/v1/vector_stores/*` routes. |

`chimera_db.h` exposes a `DbConnection` RAII wrapper (`sqlite3 *` +
deleter), a `prepared_statement` helper, and a small `migrate(conn,
target_version)` function that walks `user_version` forward through a
list of migration steps. The connection wrapper opens with
`SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX`
(we're using threadmode 2 — one connection per thread), pragmas the
file into WAL mode, and loads sqlite-vec via
`sqlite3_vec_init(db, &errmsg, &api)`.

---

## 4. Storage location

We follow XDG. In priority order:

1. `$CHIMERA_DB` environment variable, if set.
2. `$XDG_DATA_HOME/chimera/chimera.db`, if `XDG_DATA_HOME` is set.
3. `$HOME/.local/share/chimera/chimera.db` (Linux default per spec).
4. `$HOME/Library/Application Support/chimera/chimera.db` (macOS).
5. `%LOCALAPPDATA%\chimera\chimera.db` (Windows).

`chimera_db::default_path()` resolves the platform default and creates
the parent directory if needed. A single DB file holds every table —
chats, messages, vector collections, embeddings. Keeping one file
simplifies backup (`cp chimera.db backup.db`), keeps WAL/SHM files
local to one directory, and lets the user trivially `rm` everything
chimera persists.

The existing `CHIMERA_HISTORY` env var (linenoise text history) stays
working unchanged; once the new chat store lands the line-history file
becomes a UI-only convenience (up-arrow recall) and the structured
chat data goes to SQLite.

---

## 5. Schema

### 5.1 Conversation history

```sql
-- One row per conversation.
CREATE TABLE chats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      INTEGER NOT NULL,            -- unix seconds
    updated_at      INTEGER NOT NULL,
    title           TEXT,                        -- model-generated or user-set
    model_path      TEXT NOT NULL,               -- the GGUF that was loaded
    model_alias     TEXT,                        -- the friendly name (Llama-3.2-1B)
    system_prompt   TEXT,                        -- nullable
    source          TEXT NOT NULL,               -- 'chat' | 'serve'
    metadata_json   TEXT                         -- free-form for sampling params, etc.
);
CREATE INDEX idx_chats_updated ON chats(updated_at DESC);

-- One row per turn (user / assistant / system / tool / etc.).
CREATE TABLE messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id         INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,            -- 0-indexed within chat
    role            TEXT    NOT NULL,            -- 'user', 'assistant', 'system', 'tool'
    content         TEXT    NOT NULL,            -- the text the model saw / produced
    reasoning       TEXT,                        -- the <think>...</think> span, if any
    media_json      TEXT,                        -- attached media descriptors
    tool_calls_json TEXT,                        -- structured tool-call payloads
    tokens_in       INTEGER,
    tokens_out      INTEGER,
    created_at      INTEGER NOT NULL,
    UNIQUE (chat_id, seq)
);
CREATE INDEX idx_messages_chat ON messages(chat_id, seq);

-- FTS index over message content so users can search past chats.
CREATE VIRTUAL TABLE messages_fts USING fts5(
    content,
    role UNINDEXED,
    chat_id UNINDEXED,
    content=messages,
    content_rowid=id
);
-- Triggers keep messages_fts in sync with messages (omitted here).
```

This is small and sufficient for both:
- `command_chat` resuming a prior conversation by `chat_id`.
- A future server-side `/v1/responses`-style API that needs to retrieve
  conversation state by external ID.

### 5.2 Vector store

```sql
-- A "collection" is one logical corpus (user's notes, a documentation
-- set, a codebase, etc.). Multiple collections can coexist.
CREATE TABLE collections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    UNIQUE NOT NULL,
    embedding_model TEXT  NOT NULL,             -- e.g. 'bge-small-en-v1.5'
    dim           INTEGER NOT NULL,             -- embedding dimensionality
    created_at    INTEGER NOT NULL,
    metadata_json TEXT
);

-- One row per ingested chunk. Vector lives in the companion vec0 table.
CREATE TABLE documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    source_uri    TEXT,                          -- file path, URL, etc.
    chunk_index   INTEGER NOT NULL,              -- 0-indexed within source
    text          TEXT    NOT NULL,
    token_count   INTEGER,
    metadata_json TEXT,
    created_at    INTEGER NOT NULL
);
CREATE INDEX idx_documents_collection ON documents(collection_id);
CREATE INDEX idx_documents_source     ON documents(collection_id, source_uri, chunk_index);

-- sqlite-vec virtual table: ANN search over the embedding column.
-- One vec0 table per collection (so each can have its own dim).
-- Created dynamically at `chimera index create` time:
--   CREATE VIRTUAL TABLE vec_<collection_id> USING vec0(
--       document_id INTEGER PRIMARY KEY,
--       embedding   FLOAT[<dim>]
--   );
```

Why one vec0 table per collection rather than a single shared one:

- sqlite-vec's `vec0` virtual table fixes the dimensionality at
  creation time. Different embedding models produce different
  dimensions (`bge-small` = 384, `bge-large` = 1024, OpenAI
  `text-embedding-3-small` = 1536). One table per collection lets us
  mix models cleanly.
- KNN search inside one collection is the common case; cross-collection
  search would have to handle dim mismatch anyway.

### 5.3 Schema versioning

`PRAGMA user_version` tracks the migration state. Migrations are
plain functions taking a connection and producing version N+1:

```cpp
struct Migration {
    int          to_version;     // 1, 2, 3, ...
    const char * description;
    int (*apply)(sqlite3 * db);  // runs inside an exclusive transaction
};

static const Migration MIGRATIONS[] = {
    { 1, "initial chat + collection schema",      &migrate_v1 },
    { 2, "add reasoning column to messages",      &migrate_v2 },
    { 3, "add tool_calls_json to messages",       &migrate_v3 },
    // ...
};
```

The runner is the conventional pattern:

1. `PRAGMA journal_mode = WAL`
2. `PRAGMA user_version` → current
3. For each migration with `to_version > current`: run `apply` inside
   a transaction, then `PRAGMA user_version = to_version`.
4. Done.

We never decrement `user_version`. If a future chimera build needs to
break the schema, it should be a numbered migration that adds new
tables (and possibly leaves old data alone) rather than dropping
columns. The promise we make to users: **a newer chimera will always
be able to open an older chimera's DB file.** The reverse is not
promised.

---

## 6. CLI surface area

### 6.1 New subcommands

```sh
# Vector store / RAG
chimera index create  -n notes -e bge-small-en-v1.5-q8_0.gguf
chimera index ingest  -n notes -f path/to/doc.md
chimera index ingest  -n notes -g 'docs/**/*.md'      # glob
chimera index list                                     # collections + counts
chimera index stats   -n notes
chimera index drop    -n notes
chimera search        -n notes -q "how does X work" -k 5
```

These run against the local SQLite file directly. No server needed.
The `ingest` subcommand chunks input text (initially: simple
fixed-window with overlap; smarter chunking is a follow-up), runs the
configured embedding model against each chunk, and writes both
`documents` + the vec0 row.

### 6.2 `chimera chat` integration

```sh
chimera chat -m model.gguf                         # ephemeral, no DB write
chimera chat -m model.gguf --persist               # opt-in; saves to chats table
chimera chat -m model.gguf --resume 42             # resume chat id 42
chimera chat -m model.gguf --resume last
chimera chat --list                                # list saved chats
chimera chat --search "secret password"            # FTS over message content
```

Persistence is **opt-in** for the first cut so existing users see no
behavior change. Once stable we can flip the default — but only after
deciding how to handle privacy (chat content includes user data).

### 6.3 `chimera serve` integration

The server can write to the same DB:

```sh
chimera serve -m model.gguf --persist-chats        # write every /v1/chat/completions to chats
chimera serve -m model.gguf --enable-rag           # bind /v1/vector_stores/* routes
```

The two flags are independent. `--persist-chats` is a logging /
history feature; `--enable-rag` enables a small set of new HTTP
routes (see §7).

---

## 7. Server-side HTTP routes (with `--enable-rag`)

OpenAI-shaped vector store API:

| Method | Path | Notes |
|--------|------|-------|
| `GET`  | `/v1/vector_stores` | List collections. |
| `POST` | `/v1/vector_stores` | Create a collection. Body: `{"name": "...", "embedding_model": "..."}`. |
| `GET`  | `/v1/vector_stores/{name}` | Collection details + doc count. |
| `DELETE` | `/v1/vector_stores/{name}` | Drop. |
| `POST` | `/v1/vector_stores/{name}/files` | Ingest text (multipart upload) or a JSON `{"text": "..."}` body. Chunks + embeds in one request. |
| `POST` | `/v1/vector_stores/{name}/search` | KNN search. Body: `{"query": "...", "k": 5}`. Returns top-k chunks. |

These are chimera-owned handlers, not bound `server_routes`
lambdas. They sit alongside the existing chimera-owned audio and image
routes on the same `server_http_context`.

A future `--persist-chats`-driven feature could also bind
`POST /v1/responses` (OpenAI Responses API) on top of the chats table.
Not in the first cut.

---

## 8. Concurrency

SQLite in WAL mode permits **one writer + many concurrent readers**.
That maps well onto chimera's threading shape:

```
chimera CLI (chat, index, search):  one process, one connection.

chimera serve:
  main thread          ──── ctx_server.start_loop()
  http worker thread 1 ──── handler ──── per-thread DbConnection
  http worker thread 2 ──── handler ──── per-thread DbConnection
  http worker thread N ──── handler ──── per-thread DbConnection
```

Each handler that needs the DB takes one connection from a small pool
(or opens a new connection — SQLite open is cheap). Connections are
not shared across threads; we built with `SQLITE_THREADSAFE=2` for that
exact reason.

Writes serialize on SQLite's WAL writer lock. For RAG ingestion this
is fine — ingestion is bursty and not in a hot HTTP path. For chat
persistence on `chimera serve` we batch a turn's worth of writes
(user message + assistant message + reasoning + token counts) into one
transaction.

`sqlite_vec` adds no additional locking concerns; it stores data in a
vec0 virtual table backed by regular SQLite pages.

---

## 9. Phased rollout

Each phase is shippable on its own.

### Phase 1 — vendor + connection + schema

- `manage.py` fetches both amalgamations into `thirdparty/`.
- `src/chimera/chimera_db.{h,cpp}` lands with the connection wrapper,
  migration runner, and v1 schema (chats + messages + collections +
  documents).
- A `chimera --version` line prints the SQLite version and confirms
  sqlite-vec loaded.
- No CLI / HTTP behavior change yet — only the build picks up the new
  deps and the DB file is created on first use.
- Risk: ~1.5 MB binary growth, OpenSSL- and OpenMP-style configure
  drift across platforms.

### Phase 2 — vector store CLI **[shipped]**

- `chimera index create / ingest / list / stats / drop`
- `chimera search`
- `scripts/test.sh` includes a vector-store smoke test that ingests a
  three-passage corpus and verifies the top-1 hit on a targeted query.
- New `src/chimera/chimera_embed.{h,cpp}` extracts the embedding loop
  out of `command_embed`. The `Embedder` is reused across all chunks
  of an ingest run (model load once, embed many).
- New `src/chimera/chimera_vector_store.{h,cpp}` holds the SQL: create
  (inserts collection row + creates per-collection `vec_<id>` virtual
  table), drop (cascading + drops vec0), find/list, insert_document
  (one row in `documents` + one row in `vec_<id>` per chunk), search
  (KNN via `WHERE embedding MATCH ? AND k = ?`).
- Chunker: character-window with 2048/256 default + sentence-boundary
  nudge. Token-based chunking remains a phase 6+ follow-up.

### Phase 3 — chat persistence **[shipped]**

- `chimera chat --persist`, `--resume <id|last>`, `--list`, `--search`.
- New module `src/chimera/chimera_chat_store.{h,cpp}` wraps the
  `chats` / `messages` / `messages_fts` tables created by the v1
  migration in phase 1: `create_chat`, `append_message`,
  `delete_last_message`, `load_messages`, `list_chats`, `latest_chat`,
  `search_messages`. FTS5 sync happens via triggers, not application
  code.
- Persistence is opt-in (`--persist`) so existing chat users see no
  behavior change. The linenoise text-history file remains the
  readline / up-arrow buffer; the structured saves are separate.
- `chimera chat --list` and `chimera chat --search QUERY` are
  print-and-exit; they never load a model and don't require `-m`.
  `chimera chat --resume <id|last>` picks the model from the saved
  chat row if `-m` is omitted, so resume is a single-flag command.
- `chat_sample_loop` got an optional `std::string * out_reasoning`
  parameter so the assistant's `<think>...</think>` span is captured
  into `messages.reasoning` rather than discarded.
- `/clear` in persistent mode starts a fresh chat row rather than
  wiping the existing one; old chats remain in the DB.
- `/regen` in persistent mode also drops the corresponding message(s)
  from the DB (last assistant turn — possibly several in a row if the
  user had regenerated multiple times).
- Open question §11.5 (save partial responses on Ctrl-C) is **not**
  addressed in this phase. Interrupted streams aren't saved.
- Open question §11.3 (auto-reattach media on resume) is **not**
  addressed. Media paths are serialized to `media_json` for later
  use; resume replays the conversation text but doesn't load the
  attached images/audio. Flagged as a future follow-up.

### Phase 4 — server-side RAG routes **[shipped]**

- `chimera serve --enable-rag <embedding.gguf>` loads the named
  embedding model at startup and binds six routes on the same
  `server_http_context`:
  - `GET  /v1/vector_stores`              — list collections
  - `POST /v1/vector_stores`              — create
  - `GET  /v1/vector_stores/:name`        — stats
  - `POST /v1/vector_stores/:name/delete` — drop (see note below on
    why this is POST and not DELETE)
  - `POST /v1/vector_stores/:name/files`  — ingest (multipart upload
    of `file`, or JSON `{"text": "..."}` body)
  - `POST /v1/vector_stores/:name/search` — KNN search; body
    `{"query": "...", "k": N}`
- Connection model: **open-per-request** rather than a pool. SQLite
  open is microseconds in WAL mode; the cost vs. a pool is well below
  the noise floor of any RAG operation, and the lifetime ceremony of
  a pool isn't worth it for the read patterns we have.
- Embedder serialization: one `Embedder` is loaded at startup; calls
  are serialized on a per-server `std::mutex`. Same pattern as the
  whisper and SD contexts.
- DELETE caveat: `server_http_context` exposes only `get()` and
  `post()` (the wrapped subset of cpp-httplib). Adding DELETE would
  mean patching our vendored `server-http.cpp`, which is a per-llama.cpp-
  version maintenance cost. Instead, drop is `POST :name/delete`.
  OpenAI SDK clients that send `DELETE /v1/vector_stores/{id}` won't
  work as-is; they need to be reconfigured.
- `set_error_handler` interaction: upstream's `server-http.cpp:140`
  unconditionally overwrites response bodies on status 404 with a
  generic `"File Not Found"` payload. To keep our specific error
  messages visible, "no such collection" cases return **400**
  (`invalid_request_error`) rather than 404. Semantically defensible
  for a name lookup inside a known route; pragmatically the only way
  to preserve the message without forking `server-http.cpp`.
- One embedding model per server in this cut. If a request targets a
  collection whose recorded `embedding_model` doesn't match what's
  loaded, the server returns 400 with a clear error pointing at
  `--enable-rag`. Multi-model would be Phase 4.1.
- `chimera_serve.cpp` got a `RagContext` struct (db_path, Embedder,
  mutex, loaded_model) plus six handler factories and a local
  `serve_chunk_text` (a copy of the CLI's chunker; keeping a copy is
  cheaper than promoting it to a shared header for one extra caller).

### Phase 5 — server-side chat persistence **[shipped]**

- `chimera serve --persist-chats` wraps `routes.post_chat_completions`
  with `make_persisting_chat_handler`. The wrapper handles both
  response shapes:
  - **Non-streaming**: parse the response JSON, pull out
    `choices[0].message.content` (+ `reasoning_content`), token
    counts from `usage`, then save.
  - **Streaming**: replace `res->next` with a wrapper that mirrors
    each chunk into a `std::shared_ptr<std::string>` buffer while
    still returning it to the client. When the stream ends
    (`next` returns false) the buffered SSE is parsed event-by-event
    and the concatenated content is saved.
  Errors in persistence are caught + logged but never break the
  client's HTTP response.
- Each request creates **one new chat row**. The OpenAI API does not
  carry a chat id, so multi-turn conversations from the same client
  produce multiple rows that share content. Phase 6+ might add
  `X-Chimera-Chat-Id` header support; for now the duplication is the
  cost of staying API-compatible.
- DB write throughput is serialized on `ChatPersistContext::mutex`.
  All HTTP worker threads contend on it; for chat-completions this
  is a non-issue because writes are once-per-request and small.
- `POST /v1/responses` is now bound (`routes.post_responses_oai`).
  It's stateful within a single chimera serve invocation; state is
  held by server-context in-process and lost on restart. The
  persistence layer doesn't fix that — server-context owns Responses
  state, not us — but with `--persist-chats` on, the underlying
  chat-completions traffic still hits the chats table, so audit-log
  use cases work.

### Phase 6+ — nice-to-haves

- Embedding cache (`embed(text) -> vector` memoized to a small KV
  table keyed by `sha256(text) || model_id`).
- Smarter chunking (sentence-aware, semantic boundaries).
- Hybrid search (FTS5 + sqlite-vec combined ranking).
- `chimera serve --enable-rag` audit table for ingest/search calls.
- Backup helpers (`chimera db backup`, `chimera db vacuum`).

---

## 10. Risks and things to watch out for

**Schema migration discipline forever.** Once a user has a DB file,
we own forward-compat indefinitely. The release process needs a check
that every new chimera version has run its DB through the migration
chain. A `make test-db-migrate` target that fixtures an old-version
DB and asserts the migration to current succeeds is the right
guardrail.

**Privacy of stored chat content.** Chat persistence captures
everything the user typed plus everything the model said. The DB
location follows XDG (user-private), the file is readable only by
the user's account, but we should document this clearly and make
`--persist` opt-in until we have a real policy.

**Embedding model drift.** If a user re-ingests a corpus with a new
embedding model, the vec0 dim differs and the table can't accept new
rows. The `collections` table records `embedding_model + dim`; on
ingest we verify they match, and on mismatch we surface a clear error
("collection 'notes' was indexed with bge-small@384, current model
emits 1024 dim — drop and re-create").

**sqlite-vec API stability.** sqlite-vec is at early-stable; the
public surface we use (`CREATE VIRTUAL TABLE vec_x USING vec0(...)`,
`SELECT ... WHERE embedding MATCH ? ORDER BY distance LIMIT k`) has
been stable across recent releases, but we should pin a specific tag
in `manage.py` and bump it deliberately.

**Binary size budget.** sqlite3.c + sqlite-vec.c add ~1.5–1.6 MB to
the chimera binary. Combined with the server-context work that
already pushed binary size up, this needs to be visible in the
project's "shipped size" expectations.

**WAL files.** WAL mode produces `chimera.db-wal` and `chimera.db-shm`
alongside the main file. Backups must include all three (or use
`VACUUM INTO` for a clean snapshot). Document this.

**FTS5 + UPDATE/DELETE triggers.** Keeping `messages_fts` in sync
with `messages` requires three triggers (after insert, update,
delete). They're standard but easy to forget — they're part of the
v1 migration, not application code.

**`--enable-rag` ingestion blocks the HTTP worker.** A single
`POST /v1/vector_stores/{name}/files` with a 10 MB text file can take
many seconds to embed and write. cpp-httplib's worker pool absorbs
this (other requests still progress), but the caller will see a slow
response. Streaming progress back over SSE is a Phase 6+ project; for
now ingestion is synchronous.

**Cross-process contention.** If the user has a `chimera chat` session
open and also runs `chimera index ingest`, both processes hold
DbConnections to the same file. WAL mode handles this correctly. But:
opening the DB with `--immutable` or pinning the journal mode wrong
in either side would break the other. We open identically everywhere
(WAL, normal synchronous, NOMUTEX threading).

---

## 11. Open design questions

1. **Default chunk size.** Fixed 512 tokens with 64-token overlap is
   a reasonable starting point. Per-collection override seems easy
   enough to add when it's needed; not in v1.
2. **Distance metric.** sqlite-vec supports L2, cosine, and inner
   product. Cosine is the right default for normalized embeddings
   (which is what `chimera embed --pooling mean` produces). We'll
   default to that and let `chimera index create --distance L2`
   override.
3. **Where does HTTP file ingestion accept input?** multipart upload
   only, or also raw text in the body? OpenAI's `files` API is
   multipart. Following that is easier for clients.
4. **Tokenizer for chunking.** The embedding model has a tokenizer
   reachable via the llama vocab; chunking by tokens (rather than
   characters) gives more accurate per-chunk sizes. Costs a tokenize
   call per chunk. Probably worth it.
5. **Should `chimera chat --persist` save mid-stream interruptions?**
   If the user hits `Ctrl-C` while the model is generating, do we
   save the partial response? Argument for: it might be useful.
   Against: it's a partial state nobody asked for. Default: yes,
   save it with a `partial=1` column; let `--resume` show it as
   incomplete.

---

## 12. References

- SQLite amalgamation: <https://sqlite.org/amalgamation.html>
- sqlite-vec docs: <https://alexgarcia.xyz/sqlite-vec/>
- sqlite-vec repo (releases pin point): <https://github.com/asg017/sqlite-vec>
- SQLite compile-time options: <https://sqlite.org/compile.html>
- XDG base-directory spec: <https://specifications.freedesktop.org/basedir-spec/>
- The analysis that fed into this plan: see the conversation logs
  preceding this document; the executive summary lives in section 1.
