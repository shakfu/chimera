# chimera cheatsheet

Copy-pasteable one-liners. For prose, see [`serve.md`](serve.md).

```
chimera <subcommand> [flags]
```

| Subcommand | Purpose |
|------------|---------|
| `gen`      | One-shot text generation. |
| `chat`     | Interactive REPL. |
| `tokenize` | Print token ids for a prompt. |
| `embed`    | Print an embedding vector. |
| `whisper`  | Transcribe a WAV file. |
| `sd`       | Text-to-image / img2img. |
| `serve`    | OpenAI-compatible HTTP server. |
| `index`    | Vector-store collections (create / ingest / list / stats / drop). |
| `search`   | KNN search over a vector-store collection. |
| `db`       | Embedded SQLite management (status, future: backup/vacuum). |
| `info`     | Print versions + ggml backends/devices + CPU features. |

Global: `-v` (verbose backend logs), `-V` (version), `-h` (help).
Every subcommand also accepts `-h` for its own flags.

---

## gen — one-shot text

```sh
chimera gen -m model.gguf -p "Tell me a joke"
chimera gen -m model.gguf -f prompt.txt
chimera gen -m model.gguf -f -                    # stdin
chimera gen -m model.gguf -p "..." -n 128 \
    --temp 0.8 --top-k 40 --top-p 0.95 --seed 42
chimera gen -m vl.gguf -p "describe this" \
    --mmproj mmproj.gguf --image img.png
```

---

## chat — interactive REPL

```sh
chimera chat -m model.gguf
chimera chat -m model.gguf --system "You are a curt assistant."
chimera chat -m model.gguf --system-prompt-file system.txt
chimera chat -m vl.gguf --mmproj mmproj.gguf
chimera chat -m model.gguf --color {auto|always|never}

# Persistent chats (off by default; data goes to the SQLite DB)
chimera chat -m model.gguf --persist             # save every turn
chimera chat --resume 42                         # resume saved chat #42
chimera chat --resume last                       # most recent
chimera chat --list                              # list saved chats, no model load
chimera chat --search "secret password"          # FTS5 over messages, no model load
```

Slash commands inside the REPL:

```
/help                 list commands
/exit, /quit          quit
/regen                drop last assistant turn, re-sample
/clear                reset history + KV
/read <file>          attach a text file to the next message
/glob <pattern>       attach text files matching a glob
/image <file>         attach an image      (requires --mmproj with vision)
/audio <file>         attach an audio file (requires --mmproj with audio)
```

History persists at `$CHIMERA_HISTORY` or `~/.chimera_chat_history`.

---

## tokenize — vocab debugging

```sh
chimera tokenize -m model.gguf -p "Hello world"
chimera tokenize -m model.gguf -f prompt.txt
chimera tokenize -m model.gguf -p "..." --pieces      # id<TAB>piece
chimera tokenize -m model.gguf -p "..." --no-bos --no-special
```

---

## embed — embedding vectors

```sh
chimera embed -m bge.gguf -p "hello world"
chimera embed -m bge.gguf -f prompt.txt -o vec.txt
chimera embed -m bge.gguf -p "..." --pooling mean     # mean|cls|last|none
chimera embed -m bge.gguf -p "..." --no-normalize
chimera embed -m bge.gguf -p "..." --cache-embeddings   # memoize to SQLite
chimera embed -m bge.gguf -p "..." --cache-embeddings --cache-db /tmp/cache.db
```

---

## whisper — transcribe a WAV file

```sh
chimera whisper -m ggml-base.en.bin -i speech.wav
chimera whisper -m ggml-base.en.bin -i speech.wav -o out.txt
chimera whisper -m ggml-base.en.bin -i speech.wav --timestamps
chimera whisper -m ggml-base.en.bin -i speech.wav --language auto
chimera whisper -m ggml-base.en.bin -i speech.wav --translate
```

Non-WAV input? Convert first:

```sh
ffmpeg -i input.mp3 -ar 16000 -ac 1 input.wav
```

---

## sd — text-to-image / img2img

```sh
# txt2img
chimera sd -m sd.gguf -p "a red cube" -o out.png \
    -W 512 -H 512 -s 20 --cfg-scale 7

# img2img (round-trip an existing image)
chimera sd -m sd.gguf -p "make it blue" -o out.png \
    --init-image in.png -W 512 -H 512 --strength 0.6

# inpaint (init + single-channel mask, both matching -W,-H)
chimera sd -m sd.gguf -p "in the masked area" -o out.png \
    --init-image in.png --mask-image mask.png -W 512 -H 512

# batch of N images: out.png becomes out_001.png ... out_NNN.png
chimera sd -m sd.gguf -p "..." -o out.png -b 4
```

---

## serve — OpenAI-compatible HTTP server

```sh
chimera serve -m model.gguf                                    # text-only
chimera serve -m embed.gguf --embeddings                       # /v1/embeddings (single-model embed mode)
chimera serve -m model.gguf --enable-embeddings embed.gguf     # +/v1/embeddings (dedicated model, LLM stays generative)
chimera serve -m model.gguf --reranking rerank.gguf            # +/v1/rerank (cross-encoder)
chimera serve -m model.gguf --enable-audio whisper.gguf        # +/v1/audio/{transcriptions,translations}
chimera serve -m model.gguf --enable-image sd.gguf             # +/v1/images/*
chimera serve -m model.gguf --enable-rag embed.gguf            # +/v1/vector_stores/*
chimera serve -m model.gguf --persist-chats                    # log every chat to DB
chimera serve -m model.gguf --enable-rag embed.gguf --cache-embeddings  # memoize embed(text)
chimera serve -m model.gguf --enable-rag embed.gguf --rag-db /path/to.db
chimera serve -m model.gguf --persist-chats     --chat-db /path/to.db
chimera serve -m model.gguf --host 0.0.0.0 --port 8080
chimera serve -m model.gguf --api-key sk-local
chimera serve -m model.gguf --parallel 4                       # 4 concurrent slots
```

Default bind: `127.0.0.1:8080`. Stop with Ctrl-C.

### Hitting it with curl

```sh
# Chat (non-streaming)
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"any","messages":[{"role":"user","content":"Hi"}]}'

# Chat (streaming SSE)
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"any","stream":true,"messages":[{"role":"user","content":"Hi"}]}'

# Legacy text completion
curl -s http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"any","prompt":"The capital of France is","max_tokens":4}'

# Embeddings (server must be started with --embeddings or --enable-embeddings)
curl -s http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"any","input":"hello world"}'
# Array input is also accepted: "input":["a","b","c"] -> one vector per item

# Rerank (requires --reranking <model.gguf>)
curl -s http://127.0.0.1:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"any","query":"capital of France","documents":["Paris is the capital.","Berlin is in Germany.","Bananas are yellow."],"top_n":3}'

# Models / health / metrics / props
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/metrics             # Prometheus format
curl -s http://127.0.0.1:8080/props               # template + defaults

# Anthropic Messages (point Anthropic SDK at chimera)
curl -s http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"any","max_tokens":64,"messages":[{"role":"user","content":"Hi"}]}'
curl -s http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"any","messages":[{"role":"user","content":"Hi"}]}'

# Fill-in-the-middle (code models only; 501 on chat models)
curl -s http://127.0.0.1:8080/infill \
  -H 'Content-Type: application/json' \
  -d '{"input_prefix":"def add(a, b):\n    return ","input_suffix":"\n","n_predict":4}'

# Vocab helpers
curl -s http://127.0.0.1:8080/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"content":"hello world"}'
curl -s http://127.0.0.1:8080/detokenize \
  -H 'Content-Type: application/json' \
  -d '{"tokens":[15339,1917]}'

# Render the chat template against messages without generating
curl -s http://127.0.0.1:8080/apply-template \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}]}'

# OpenAI Responses API (stateful within one chimera serve invocation)
curl -s http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"any","input":"Say hi"}'

# Audio transcription (WAV / RIFF only — transcode other formats with
#   ffmpeg -i in.<ext> -ar 16000 -ac 1 out.wav)
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F response_format=json
# response_format: json | text | verbose_json | srt | vtt

# Word-level timestamps in verbose_json (adds a top-level `words[]` array)
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F response_format=verbose_json \
  -F 'timestamp_granularities[]=word'

# Audio translation (any source language -> English)
curl -s http://127.0.0.1:8080/v1/audio/translations \
  -F file=@speech.wav -F response_format=json

# Image generation
curl -s http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a watercolor cat","n":1,"size":"512x512"}'

# Image edit (img2img / inpaint; optional `mask` field)
curl -s http://127.0.0.1:8080/v1/images/edits \
  -F image=@in.png -F prompt="now at sunset" -F strength=0.6

# Image variation (img2img, no prompt)
curl -s http://127.0.0.1:8080/v1/images/variations \
  -F image=@in.png

# Vector store (requires --enable-rag)
curl -s -X POST http://127.0.0.1:8080/v1/vector_stores \
  -H 'Content-Type: application/json' -d '{"name":"notes"}'
curl -s -X POST http://127.0.0.1:8080/v1/vector_stores/notes/files \
  -H 'Content-Type: application/json' -d '{"text":"..."}'
curl -s -X POST http://127.0.0.1:8080/v1/vector_stores/notes/files \
  -F file=@doc.md
curl -s -X POST http://127.0.0.1:8080/v1/vector_stores/notes/search \
  -H 'Content-Type: application/json' -d '{"query":"...","k":5}'
curl -s http://127.0.0.1:8080/v1/vector_stores
curl -s http://127.0.0.1:8080/v1/vector_stores/notes
curl -s -X POST http://127.0.0.1:8080/v1/vector_stores/notes/delete
```

### Pointing the OpenAI SDK at chimera

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-used")
```

---

## index, search — vector store / RAG

```sh
chimera index create -n notes -e bge-small.gguf       # discover dim, create
chimera index create -n notes -e bge-small.gguf --distance l2 \
                     --chunk-tokens 256 --chunk-overlap 32  # custom per-collection defaults
chimera index ingest -n notes -f path/to/doc.md       # one file
chimera index ingest -n notes -g 'docs/**/*.md'       # glob, one model load
chimera index ingest -n notes -f doc.md --chunk-tokens 384 --chunk-overlap 48  # per-call override
chimera index list                                    # collections + counts
chimera index stats  -n notes
chimera index drop   -n notes

chimera search -n notes -q "how does X work" -k 5

# Cache per-chunk + per-query embeddings in --db so re-runs skip the model:
chimera index ingest -n notes -f doc.md --cache-embeddings
chimera search       -n notes -q "..." --cache-embeddings
```

Override the DB location: `--db <path>` on any subcommand, or set
`$CHIMERA_DB` once. Tuning: `--chunk-chars` (default 2048), `--chunk-overlap`
(default 256). The pooling defaults to `mean`; use `--pooling cls` for
BERT-style models.

---

## db — embedded SQLite management

```sh
chimera db status                                     # path + version + schema
chimera db status --db /tmp/scratch.db                # arbitrary file
```

---

## info — versions + ggml backends + CPU features

```sh
chimera info
```

Prints chimera version, the platform tag, llama / whisper / sd
version + ggml view + backend registries + enumerated devices, plus
sqlite + sqlite-vec versions. Useful for bug reports.

---

## Exit codes

| Code  | Meaning                              |
|-------|--------------------------------------|
| 0     | Success                              |
| 1     | Runtime error                        |
| 2     | Bad input                            |
| 3     | Model-load failure                   |
| 4     | Generation / inference failure       |
| ≥ 100 | CLI11 argument parse error           |

---

## Environment

| Variable           | Effect |
|--------------------|--------|
| `CHIMERA_HISTORY`  | Where `chat` reads/writes its readline history. Defaults to `~/.chimera_chat_history`. |

---

## Building from source

```sh
make deps                          # fetch + build llama.cpp / whisper.cpp / sd.cpp / linenoise
make build                         # configure + build chimera
make rebuild                       # incremental chimera build (skips deps)
make test                          # smoke + end-to-end suite
make install PREFIX=/usr/local     # ./build/chimera -> $PREFIX/bin/
make uninstall PREFIX=/usr/local
make clean                         # remove build/
make reset                         # clean + remove fetched thirdparty sources
```
