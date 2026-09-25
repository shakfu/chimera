# Upstream PR brief: `llama.cpp-stream-cancel-on-shutdown.patch`

Source material for a llama.cpp PR. It is not a PR description.

llama.cpp prohibits AI-written PR descriptions, commit messages and reviewer
replies, and closes such PRs at once (`CONTRIBUTING.md`, "AI Usage Policy";
`AGENTS.md`, "Prohibited AI Usage"). Write the description yourself from the
facts below. The patch and this brief were produced with AI assistance, so the
template's "AI usage disclosure" must say YES and describe that use. You are
responsible for every line and must be able to explain it without AI help.

## Bug

`llama-server` does not exit on SIGTERM/SIGINT while a request with
`X-Conversation-Id` is streaming. A second signal is needed.

- Reproduced on unpatched `llama-server` b11146 (`7fe450e`), CPU,
  Llama-3.2-1B-Instruct-Q8_0: 3/3 runs hung past 15 s, and 1 run past 120 s.
  The generation itself needs about 110 s, so the hang does not end on its own.
- `stop_gc()` is unchanged on master `1ab7e5a` (2026-09-25), and the patch
  applies there with `git apply --check`. The bug was not run on master.

## Root cause

Line numbers are at b11146.

1. The first signal calls `ctx_server.terminate()` (`server.cpp:487`), so
   `start_loop()` returns (`server.cpp:542`).
2. `clean_up()` calls `server_stream_session_manager_stop()`, then
   `ctx_http.stop()` (`server.cpp:453-454`). `ctx_http.join()` follows (`:545`).
3. httplib stops writing the response. Its destructor runs
   `server_res_spipe::on_complete()`, which drains the rest of the generation
   into the session: `while (!spipe->is_cancelled())` (`server-stream.cpp:643`).
4. Each loop turn blocks in `server_response_reader::next()`
   (`server-queue.cpp:550`). That call returns only on a result or when
   `should_stop()` is true. For a tagged request, `should_stop()` is
   `spipe->is_cancelled()` (`server-stream.cpp:622`).
5. The task loop has exited, so no result arrives. `stop_gc()` only finalizes
   sessions (`server-stream.cpp:341-352`) and never cancels them. The drain loops
   forever, and `ctx_http.join()` waits on that worker.

gdb on the hung process (chimera, which embeds the same server code): the main
thread was in `server_http_context::join()` -> `httplib::ThreadPool::shutdown()`.
One worker was in `httplib::Response::~Response()` -> `server_res_spipe::on_complete()`
-> `server_response_reader::next()` -> `server_response::recv_with_timeout()`.

## Fix

`stop_gc()` calls `s->cancel()` before `s->finalize()` on each live session.
The diff adds one line and rewords one comment.

- `cancel()` stores an atomic flag (`server-stream.cpp:217`). It takes no lock, so
  the order relative to the snapshot's `map_mu` section does not matter.
- `finalize()` is idempotent. The producer's destructor still calls it later.
- Only the shutdown path changes. The GC loop, `evict()` and `DELETE /v1/stream`
  behave as before.
- Readers blocked in `GET /v1/stream` already poll `should_stop` and wake on
  finalize, so cancel changes nothing for them.

## Evidence

The client keeps the stream open and SIGTERM is sent after the first 200 bytes.

| Build | Tagged stream (`X-Conversation-Id`) |
|-|-|
| upstream b11146 | 3/3 hung > 15 s; 1/1 hung > 120 s |
| upstream b11146 + patch | 5/5 exited in 1.0-2.1 s |

Chimera carries the same patch. Its regression test fails without the patch
(10 s timeout) and passes with it: `scripts/test.py`, "SIGTERM exits serve
during an X-Conversation-Id stream".

Repro (Python stdlib; `$BIN` = `llama-server`, `$MODEL` = any GGUF that is slow
enough to still be generating at SIGTERM):

```python
import http.client, json, subprocess, time, urllib.request
p = subprocess.Popen([BIN, "--port", "18970", "-ngl", "0", "-m", MODEL])
while True:
    try: urllib.request.urlopen("http://127.0.0.1:18970/health"); break
    except Exception: time.sleep(0.5)
c = http.client.HTTPConnection("127.0.0.1", 18970)
c.request("POST", "/v1/chat/completions", json.dumps({
    "messages": [{"role": "user", "content": "Count."}],
    "max_tokens": 4000, "stream": True, "ignore_eos": True}),
    {"Content-Type": "application/json", "X-Conversation-Id": "x"})
c.getresponse().read(200)
p.terminate()
p.wait(15)  # TimeoutExpired without the patch
```

## Test gap

No upstream test is included. The server test presets use tiny models
(`stories260K` in `ServerPreset.tinyllama2()`), which finish before the SIGTERM
arrives. A test would need a stream still in flight at shutdown, for example a
slower preset or a way to hold generation. Expect a reviewer to ask about this.

## Related

- [#27482](https://github.com/ggml-org/llama.cpp/pull/27482) (open, fixes #27481):
  lets `DELETE /v1/stream` cancel before the first token. Same subsystem,
  different path, no overlap with `stop_gc()`.
- Out of scope, and it has an upper bound: an untagged stream whose client stays
  connected can delay exit by up to `--sse-ping-interval` (30 s). httplib checks
  for shutdown only between chunks, so it waits for the next token or ping.
  Upstream b11146 exited at about 31 s in 5 of 6 runs and at 1.0 s in 1.

## Before submitting

- As of 2026-09-25, no open or closed PR mentions `stop_gc`. Search again.
- Read `CONTRIBUTING.md` and `AGENTS.md`. Use ASCII only; `AGENTS.md` forbids
  em dashes and arrows.
- Fill in the PR template (`.github/pull_request_template.md`), including the
  AI disclosure.
