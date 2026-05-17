# `chimera serve`

An OpenAI-compatible HTTP server backed by llama.cpp, with optional audio
transcription via whisper.cpp and image generation via stable-diffusion.cpp
running in the same process.

```
chimera serve -m model.gguf [--port 8080] [--host 127.0.0.1]
              [--embeddings]
              [--enable-audio whisper.gguf]
              [--enable-image sd.gguf]
              [--api-key TOKEN]
```

---

## Quickstart

Start a server with a small instruct model:

```
chimera serve -m models/Llama-3.2-1B-Instruct-Q8_0.gguf
```

Default bind: `127.0.0.1:8080`. The startup line tells you what is
listening and which endpoints are bound.

Talk to it the way you would talk to OpenAI:

```
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "any",
        "messages": [{"role": "user", "content": "Say hello in one word."}]
      }'
```

Or with the OpenAI Python SDK pointed at chimera:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-used")
r = client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": "Say hello in one word."}],
)
print(r.choices[0].message.content)
```

The `model` field is ignored — chimera serve loads one model at startup.
Pass any string.

---

## Opt-in features

By default, only the LLM text endpoints are bound. Three flags add
additional capabilities, each loading another model alongside the LLM.
Memory cost is roughly additive, so enable only what you need.

### Embeddings

```
chimera serve -m models/bge-small-en-v1.5-q8_0.gguf --embeddings
```

`--embeddings` switches the model to embedding mode (a model can only
serve one of `chat completions` *or* `embeddings` per load — that is a
constraint of how the weights are used, not of chimera). Without this
flag, `POST /v1/embeddings` returns HTTP 501 with a hint to set it.

```
curl -s http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "any", "input": "hello world"}'
```

### Audio transcription

```
chimera serve -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
              --enable-audio models/ggml-base.en.bin
```

This loads a whisper.cpp model alongside the LLM and binds
`POST /v1/audio/transcriptions`. Upload a WAV file as multipart form
data:

```
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav \
  -F model=any \
  -F response_format=json
```

Supported `response_format` values: `json` (default), `text`,
`verbose_json` (includes `segments[]`), `srt`, `vtt`.

### Image generation

```
chimera serve -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
              --enable-image models/sd_xl_turbo_1.0.q8_0.gguf
```

### Persistent chats

```
chimera serve -m model.gguf --persist-chats
```

When set, every `/v1/chat/completions` request (streaming and
non-streaming alike) is saved to the same SQLite DB the CLI uses
(`$CHIMERA_DB` or platform default; override with `--chat-db`).

By default, each request without an `X-Chimera-Chat-Id` header
creates a new chat row, and the response echoes the new row's id
back in `X-Chimera-Chat-Id`. Multi-turn clients should capture
that id from the first response and send it back on subsequent
requests in the same conversation:

```
$ curl -i -d '{"messages":[{"role":"user","content":"hi"}]}' \
       http://127.0.0.1:8080/v1/chat/completions
HTTP/1.1 200 OK
X-Chimera-Chat-Id: 17
...

$ curl -i -H 'X-Chimera-Chat-Id: 17' \
       -d '{"messages":[..., {"role":"user","content":"and another"}]}' \
       http://127.0.0.1:8080/v1/chat/completions
HTTP/1.1 200 OK
X-Chimera-Chat-Id: 17
```

When the header is present and points to an existing chat, only
the *last* message in the request body is appended (along with the
new assistant reply); the prior turns are already on disk from
earlier calls. This avoids the duplicate-row problem that arises
when clients resend the full conversation.

| Request shape                              | DB effect                                                                  | Response header                  |
|--------------------------------------------|----------------------------------------------------------------------------|----------------------------------|
| No `X-Chimera-Chat-Id`                     | New chats row; **all** request messages + assistant reply appended.        | `X-Chimera-Chat-Id: <new-id>`    |
| `X-Chimera-Chat-Id: <existing-id>`         | **Only** the last request message + assistant reply appended.              | `X-Chimera-Chat-Id: <same-id>`   |
| `X-Chimera-Chat-Id: <unknown-id>`          | None.                                                                       | HTTP 404; `inner` not invoked.   |
| `X-Chimera-Chat-Id: <non-integer>`         | None.                                                                       | HTTP 400; `inner` not invoked.   |

Use `chimera chat --list` and `chimera chat --search QUERY` from
the CLI to browse them.

See [Privacy / data on disk](#privacy--data-on-disk) for exactly
what is recorded.

### Vector store / RAG

```
chimera serve -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
              --enable-rag models/bge-small-en-v1.5-q8_0.gguf
```

This loads the named embedding model alongside the LLM and binds the
OpenAI-shaped `/v1/vector_stores/*` routes. The SQLite database it uses
is the same one the CLI sees (`$CHIMERA_DB` or the XDG default;
override with `--rag-db`).

```
# Create a collection
curl -s http://127.0.0.1:8080/v1/vector_stores \
  -X POST -H 'Content-Type: application/json' \
  -d '{"name": "notes"}'

# Ingest text (JSON body)
curl -s http://127.0.0.1:8080/v1/vector_stores/notes/files \
  -X POST -H 'Content-Type: application/json' \
  -d '{"text": "...", "source_uri": "label"}'

# Ingest text (multipart upload)
curl -s http://127.0.0.1:8080/v1/vector_stores/notes/files \
  -F file=@notes.md

# Search (default mode = hybrid; pass "mode" to override)
curl -s http://127.0.0.1:8080/v1/vector_stores/notes/search \
  -X POST -H 'Content-Type: application/json' \
  -d '{"query": "how does X work?", "k": 5, "mode": "hybrid"}'

# List and inspect
curl -s http://127.0.0.1:8080/v1/vector_stores
curl -s http://127.0.0.1:8080/v1/vector_stores/notes

# Drop (POST, not DELETE — see below)
curl -s -X POST http://127.0.0.1:8080/v1/vector_stores/notes/delete
```

Note the **POST :name/delete** path. server-http only exposes GET/POST,
so OpenAI SDK clients that send `DELETE /v1/vector_stores/{id}` will
need to be reconfigured. One embedding model per server in this cut.

This loads a stable-diffusion.cpp model alongside the LLM and binds
`POST /v1/images/{generations,edits,variations}`. Generated images come
back as base64-encoded PNGs in OpenAI's standard envelope:

```
curl -s http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a watercolor of a sleeping cat",
       "n": 1,
       "size": "512x512"}'
```

For `/v1/images/edits` and `/v1/images/variations`, send the source
image as multipart form data:

```
curl -s http://127.0.0.1:8080/v1/images/edits \
  -F image=@input.png \
  -F prompt="now make it sunset" \
  -F size=512x512 \
  -F strength=0.6
```

---

## Endpoints

Always bound:

| Method | Path | Body | Notes |
|--------|------|------|-------|
| GET | `/health`, `/v1/health` | — | Returns 200 once the model finishes loading. |
| GET | `/v1/models` | — | Lists the loaded model. |
| GET | `/metrics` | — | Prometheus-style server metrics. |
| GET | `/props` | — | Read-only: which chat template, mmproj caps, default sampling params. |
| POST | `/chat/completions`, `/v1/chat/completions` | JSON | Streaming + non-streaming. Pass `"stream": true` for SSE. |
| POST | `/v1/completions` | JSON | Legacy OpenAI text-completion. |
| POST | `/v1/embeddings` | JSON | Requires `--embeddings`. |
| POST | `/v1/messages` | JSON | Anthropic Messages API compat. |
| POST | `/v1/messages/count_tokens` | JSON | Anthropic token counting. |
| POST | `/infill` | JSON | Fill-in-the-middle for code models. 501 on models without FIM tokens. |
| POST | `/tokenize`, `/detokenize` | JSON | Vocab helpers — token-id ↔ text. |
| POST | `/apply-template` | JSON | Render the chat template against a `messages[]` array without generating. |
| POST | `/v1/responses` | JSON | OpenAI Responses API. Stateful within a single chimera serve invocation; state is lost on restart. |

Bound when `--enable-audio` is set:

| Method | Path | Body | Notes |
|--------|------|------|-------|
| POST | `/v1/audio/transcriptions` | multipart | `file` field required. Supports WAV; other formats return 415. |

Bound when `--enable-image` is set:

| Method | Path | Body | Notes |
|--------|------|------|-------|
| POST | `/v1/images/generations` | JSON | txt2img. |
| POST | `/v1/images/edits` | multipart | img2img / inpaint. `image` field required, optional `mask`. |
| POST | `/v1/images/variations` | multipart | img2img with no prompt. `image` field required. |

Bound when `--enable-rag` is set:

| Method | Path | Body | Notes |
|--------|------|------|-------|
| GET  | `/v1/vector_stores` | — | List collections. |
| POST | `/v1/vector_stores` | JSON | Create. Body: `{"name": "..."}`. |
| GET  | `/v1/vector_stores/:name` | — | Stats for one collection. |
| POST | `/v1/vector_stores/:name/delete` | — | Drop. POST because server-http doesn't expose DELETE. |
| POST | `/v1/vector_stores/:name/files` | multipart / JSON | Ingest. multipart with `file=@...`, or JSON `{"text": "...", "source_uri": "..."}`. |
| POST | `/v1/vector_stores/:name/search` | JSON | Retrieve. Body: `{"query": "...", "k": 5, "mode": "hybrid"}`. `mode` is one of `semantic` (vec0 KNN), `lexical` (FTS5 BM25), or `hybrid` (RRF-merge of the two; default). Hybrid hits include `rrf_score`, `semantic_rank`, `lexical_rank`. |

### Request fields

For LLM routes chimera serve accepts the full set that llama.cpp's
own server understands — see the [llama-server docs][1] for the
authoritative list. Common fields:

- `model` — ignored (single loaded model). Any string works.
- `messages` / `prompt` — your input. Same shape as OpenAI.
- `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `seed` — standard sampling controls.
- `stream` — boolean. Returns SSE when true.
- `tools`, `tool_choice` — function calling, when the model supports it.
- `logprobs`, `top_logprobs` — token probabilities.

[1]: https://github.com/ggml-org/llama.cpp/tree/master/tools/server

For audio (`/v1/audio/transcriptions`):

- `file` — required, the audio. **WAV / RIFF only by design** — chimera deliberately doesn't bundle audio codecs. Transcode other formats client-side: `ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`.
- `language` — ISO-639-1 code like `en`, or `auto` for autodetect.
- `prompt` — priming text (whisper's `initial_prompt`).
- `response_format` — `json` (default), `text`, `verbose_json`, `srt`, `vtt`.

For image routes:

- `prompt` — required for `generations` and `edits`.
- `n` — number of images (mapped to SD's batch count).
- `size` — `"<W>x<H>"`. Defaults to `512x512` for generations; to the
  source image's dimensions for edits/variations.
- `response_format` — `b64_json` only. `url` returns 400.
- `negative_prompt`, `steps`, `cfg_scale`, `seed`, `sample_method`,
  `scheduler` — SD-specific extensions.
- `strength` — img2img denoising strength, 0 (preserve input) to 1
  (full noise). Only meaningful for edits/variations.

---

## Authentication

Pass `--api-key <token>` to require a bearer token on `/v1/*` requests:

```
chimera serve -m model.gguf --api-key sk-chimera-local-dev
```

Clients then send `Authorization: Bearer sk-chimera-local-dev`. The
OpenAI SDK does this automatically when you set `api_key=`. `/health`
and `/v1/health` remain open (they're liveness probes).

Without `--api-key`, the server accepts any request — fine for a local
session, dangerous on a public network. Run behind a reverse proxy or
VPN for anything non-trivial.

---

## Streaming

For `POST /v1/chat/completions` with `"stream": true`, the response is
Server-Sent Events with `data: {...}` lines per token chunk:

```
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "any",
       "stream": true,
       "messages": [{"role": "user", "content": "count 1 to 3"}]}'
```

Other endpoints do not stream. `/v1/audio/transcriptions` and
`/v1/images/*` return the full body when generation finishes.

---

## Common errors

**`{"error":{"message":"This server does not support embeddings. Start it with \`--embeddings\`"}}`**
You called `/v1/embeddings` on a server started without `--embeddings`.
Restart with the flag (and likely a different model — most chat models
won't also serve as embedding models).

**`{"error":{"message":"missing 'file' field in multipart form"}}`**
You called `/v1/audio/transcriptions` or `/v1/images/{edits,variations}`
without uploading the audio/image field. Use `-F file=@path.wav` or
`-F image=@path.png` with curl.

**`{"error":{"message":"unsupported audio: ... (chimera accepts WAV / RIFF only by design; ...)"}}`**
chimera deliberately doesn't bundle audio codecs; transcode client-side.
`ffmpeg -i input.mp3 -ar 16000 -ac 1 input.wav` produces what whisper
wants.

**`{"error":{"message":"response_format=url is not supported (chimera serve has no static-file backend); use 'b64_json'"}}`**
Use `"response_format": "b64_json"` (which is also the default) and
decode the base64 yourself.

**`{"error":{"message":"image generation failed: ..."}}`**
The most common cause is an incompatible
`(sample_method, cfg_scale, model)` combination. Try omitting
`sample_method` and `cfg_scale` to use SD's defaults; if that succeeds,
add them back one at a time. Some combinations crash inside
stable-diffusion.cpp's VAE encode path; this is upstream behavior.

**HTTP 401 / "missing API key"**
You set `--api-key` on the server but the client didn't send
`Authorization: Bearer <key>`. With the OpenAI SDK, set
`api_key="<key>"`; with curl add `-H 'Authorization: Bearer <key>'`.

---

## Tips

**Pointing OpenAI clients at chimera.** Set the base URL to your
chimera serve URL plus `/v1`, and set any string as the API key (or
the one matching `--api-key` if set):

```python
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-used")
```

Same idea for the JS SDK, langchain, llama-index, OpenWebUI, etc.

**Concurrent requests.** chimera serve handles concurrent HTTP requests
on multiple worker threads. The LLM scheduler shares one model across
requests via KV-cache slots — pass `--parallel <N>` (default 1) to
allow up to N requests in flight simultaneously.

```
chimera serve -m model.gguf --parallel 4
```

Audio and image requests are serialized per modality (one whisper
transcription at a time, one SD generation at a time), because the
underlying contexts are not thread-safe.

**Production deployments.** chimera serve does not terminate TLS
itself. Run it behind nginx, caddy, or a cloud load balancer for
HTTPS. The `--host 0.0.0.0` flag lets it listen on all interfaces;
combine with `--api-key` for the minimum reasonable exposure on a
private network.

**Stopping the server.** `Ctrl-C` (SIGINT) shuts down cleanly. The
second `Ctrl-C` force-exits — useful if a request is hung.

---

## What's not supported

- Non-WAV audio formats (mp3, m4a, mp4, webm). **By design** — chimera
  deliberately doesn't bundle audio codecs (every viable option drags
  in either a multi-megabyte FFmpeg dep or a partial set of single-
  header decoders that still wouldn't cover m4a/webm). Transcode
  client-side with `ffmpeg -i in.<ext> -ar 16000 -ac 1 out.wav`.
- KV-cache slot save/load (`/slots`). Planned.
- LoRA hot-swap (`/lora-adapters`). Planned.
- HTTPS direct serving (use a reverse proxy).
- The web chat UI shipped with llama-server.
- Multi-model routing — chimera serve loads one LLM (plus optionally
  one whisper + one SD) per process.

See the changelog for the full per-phase list.

---

## Privacy / data on disk

chimera is local-only — nothing leaves the machine — but it does
write to disk when persistence features are enabled, and it's worth
knowing exactly what.

### What `--persist-chats` records

When `chimera serve --persist-chats` is set, every
`/v1/chat/completions` request triggers a write to the SQLite DB:

| Stored                                     | Notes                                                              |
|--------------------------------------------|--------------------------------------------------------------------|
| Full message content (all roles)           | The verbatim `messages[]` array sent by the client, including any system prompt, plus the assistant's reply. |
| `reasoning_content` (when the model emits it) | The `<think>...</think>` span, stored in a separate column.        |
| Model path + alias                         | The local path passed via `-m` and its basename.                   |
| Token counts (prompt + generated)          | Integers; no token IDs.                                            |
| Timestamps                                 | `created_at` / `updated_at` (unix seconds, machine local).         |
| `source = 'serve'`                         | So you can tell CLI-saved chats apart from server-saved ones.      |

What is **not** stored: client IP, request headers, API key, raw HTTP
body, image bytes (only paths are recorded by the CLI path when you
attach media; `serve` does not record image_url parts as files). FTS5
content lives in the `messages_fts` virtual table — a searchable
mirror of the same text, not an additional copy.

### What `chat --persist` records

The CLI variant stores the same shape as `--persist-chats`, plus:

- For `/image` and `/audio` attachments: the **filesystem paths** of
  attached media are serialized into `media_json`. The bytes themselves
  are not copied into the DB; if you later move or delete the file,
  resume won't be able to reconstruct it (and today's `--resume` does
  not re-attach media regardless).
- `source = 'chat'`.

If `chat --persist` is interrupted with Ctrl-C mid-generation, the
partial assistant turn is saved with `partial = 1`. `chat --list`
shows a `(N interrupted)` count next to affected chats; `chat --resume`
prints how many interrupted turns it found.

### Where the DB lives

Resolution order:

1. Explicit `--db / --chat-db / --rag-db` flag (per-subcommand).
2. `$CHIMERA_DB` environment variable.
3. Platform default:
   - macOS: `~/Library/Application Support/chimera/chimera.db`
   - Linux: `$XDG_DATA_HOME/chimera/chimera.db` or `~/.local/share/chimera/chimera.db`
   - Windows: `%LOCALAPPDATA%\chimera\chimera.db`

The file is created on first use with default user-owned permissions
(typically `0644` on Unix). WAL mode is enabled, so siblings
`chimera.db-wal` and `chimera.db-shm` live next to it.

### Turning it off / wiping it

- All persistence is **opt-in**. `chimera serve` without `--persist-chats`
  writes nothing chat-related; `chimera chat` without `--persist` runs
  fully in memory.
- The vector store (`--enable-rag`, `chimera index`) uses the same DB
  file but a disjoint set of tables — disabling RAG flags is enough to
  stop further writes to those tables.
- To wipe persisted chats only:
  `sqlite3 <path-to>/chimera.db "DELETE FROM chats; DELETE FROM messages;"`
  (FTS5 triggers cascade the delete.) Or just remove the whole `.db`,
  `.db-wal`, and `.db-shm` triple to reset.
- The embedding cache (`embed --cache-embeddings`) is also stored in
  the same DB; clear it with `DELETE FROM embedding_cache;`.

### Linenoise history

Independent of the SQLite DB, `chimera chat` writes a plain-text
readline history to `$CHIMERA_HISTORY` (default
`~/.chimera_chat_history`). This file contains every line you've typed
at the prompt across every chat session. Set `CHIMERA_HISTORY=/dev/null`
to disable, or just delete the file to clear it.
