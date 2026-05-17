# Router mode — decision record

llama.cpp's `llama-server` has a **router mode** (aka multi-model mode,
`is_router_server`) where one process acts as an HTTP proxy in front of N
child server processes, one per loaded model. chimera deliberately does
not implement this. This document records *why*, with enough detail that
the decision doesn't have to be re-litigated next time the question
comes up.

Status: **not implemented; not on the roadmap.** Re-open only if one of
the triggers in [§ 5](#5-when-to-revisit) materializes.

For the related discussion of `POST /models/load` / `POST /models/unload`
as polite-error stubs (the "make the webui's model panel stop looking
broken" variant), see the open thread in [`doc/dev/webui.md`](webui.md)
§ 5.5 and § 6 — that's a much smaller piece of work and orthogonal to
this decision.

---

## 1. What router mode actually is upstream

The headline framing — "one llama-server, many models" — undersells the
architectural cost. Router mode is **multi-process**, not
multi-model-in-one-process:

- Triggered by starting `llama-server` with **no** `--model` flag
  (`is_router_server = params.model.path.empty()` in
  `tools/server/server.cpp` line 130).
- The router itself holds **zero model weights**. It's a thin HTTP
  proxy.
- Each loaded model runs as an **OS subprocess** — the router shells
  out via `sheredom/subprocess.h` to spawn another `llama-server`
  binary with `--model <path>` and the right port.
- A `server_models` class (~1,565 LOC in `tools/server/server-models.cpp`)
  manages a `map<name, instance_t>` where each `instance_t` holds:
  `{subprocess handle, monitoring thread, FILE* to child stdin, metadata}`.
- HTTP requests for model-serving routes are forwarded by
  `server_http_proxy` (using cpp-httplib's **client** API — not just the
  response API) to the right child port.
- Children detect their role via env var
  (`server_models::is_child_server()`), then call `setup_child_server()`
  to register with the parent and propagate state (sleep, ready, exit
  code) over a separate notification channel.
- LRU eviction kicks in when `--models-max` is hit; optional
  `--models-autoload` loads on first request rather than requiring an
  explicit `POST /models/load`.

When router mode is on, the proxy intercepts **every** model-serving
route — chat completions, embeddings, rerank, tokenize, slots,
lora-adapters, infill, all of it. Only `/health`, `/metrics`, the webui,
and the router-specific `/models/{load,unload}` + `/models` are handled
directly by the router process.

---

## 2. What porting it to chimera would entail

| Step | Work |
|---|---|
| Vendor `sheredom/subprocess.h` in `manage.py` (not currently a dep) | half day |
| Compile `server-models.cpp` (~1,565 LOC) into the chimera target via the same `src-aux/` pattern we use for `server-http.cpp` | half day |
| Restructure `command_serve()` for the bimodal layout — model-mode (today) vs. router-mode (proxy) | 1–2 days |
| Add CLI: `--router` (or accept empty `--model`), `--models-dir`, `--models-preset`, `--models-max`, `--models-autoload` | half day |
| Design the **child-process surface** — does the spawned child chimera support `--enable-audio`, `--enable-image`, `--enable-rag`, `--persist-chats`, `--reranking`, `--enable-embeddings`? If yes, the preset config has to express those flags and you've designed a config language. If no, you've shipped a degraded child mode that breaks "one binary handles everything." | 2–5 days |
| Resolve **persistence interactions** — with N child chimeras, who owns the SQLite DB? Shared (WAL handles concurrency, but per-chat-id semantics get weird across child boundaries), per-child (defeats the unified-store property the CLI relies on), or router-only with children RPC'ing in (new RPC channel) | 1–3 days |
| Same question for the **RAG / vector store DB** | rolled into above |
| Pin-checks for ~6 new handler-field asserts plus the static functions (`is_child_server`, `setup_child_server`, `notify_router_sleeping_state`) | half day |
| Multi-process orchestration in tests — today's harness spawns one process; router mode needs to spawn parent + ≥2 children, validate proxy round-trips, child crash recovery, LRU eviction | 3–5 days |
| Doc rewrite — "one process, one model" framing is in `doc/serve.md`, `doc/dev/server.md`, `README.md`, and the `chimera info` output. All become lies under router mode; needs a dedicated router-mode doc | 1–2 days |
| **Total** | **~2–3 weeks** of focused work |

**Plus a permanent maintenance tax.** Each llama.cpp bump now also has
to track:

- The child-process notification protocol (env var name, handshake
  shape, sleep/wake messages).
- The proxy's client-side cpp-httplib API usage (separate from the
  server-side API we already pin against).
- The preset config format (whatever `--models-preset` parses).
- LRU and reload semantics.

Today our drift surface is small: four handler-typedef asserts in
`chimera_pin_check.cpp` and a verbatim copy of `server-http.cpp` we
don't modify. Router mode would expand that surface by roughly an order
of magnitude.

---

## 3. What router mode actually buys (the concurrency question)

This is the part that gets the most magical thinking. On a single-GPU
machine, router mode buys you very little throughput, because the GPU
remains a serial resource. Concrete breakdown:

| Dimension | Router gain on single GPU |
|-----------|---------------------------|
| Multiple models *resident* in VRAM at once | Yes, until VRAM fills |
| Concurrent HTTP requests against *different* models | Architecturally yes; on the GPU they interleave at kernel granularity |
| Concurrent HTTP requests against the *same* model | Worse than `chimera serve --parallel N` (see § 4) |
| Aggregate wall-clock tokens/sec across N concurrent requests | ~Unchanged — the GPU is the bottleneck either way |
| Latency-to-first-byte when switching models | Better (warm-load a second child vs. process restart) |

The mechanism: a single GPU executes one kernel at a time per device.
Two child server processes hitting one CUDA device don't run "in
parallel" in any meaningful compute sense — the driver round-robins
kernel submissions, and aggregate throughput is approximately equal to
running them sequentially. The optics of "two requests in flight" looks
like parallelism; the wall-clock time-to-completion of the *batch* is
unchanged.

### 3.1. Where router mode does give real compute concurrency

1. **Multi-GPU machines.** Router with N children, each pinned to its
   own GPU via per-child `CUDA_VISIBLE_DEVICES`. Now you actually have
   N independent compute units. **This is the real router use case** —
   and the one that justifies the architectural cost. It requires the
   preset config to express per-child GPU pinning.
2. **CPU + GPU split.** E.g. a small embedding model running CPU-only
   as one child, a big LLM on GPU as another. They genuinely don't
   contend on the GPU. Niche but valid; chimera already approximates
   this in one process via `--enable-embeddings <embed.gguf>`.
3. **Mixed-modality on one GPU.** LLM child on GPU, SD child on the
   same GPU. Even on one GPU the SD diffusion steps and LLM generation
   alternate naturally because you'd never run them simultaneously
   anyway. chimera already does this in one process via
   `--enable-image` plus per-modality mutexes — no router needed.

### 3.2. Where router mode is strictly worse than the current design

Two children of the **same** model is worse than one chimera with
`--parallel 2`:

- The single-process slot scheduler batches prefills across slots and
  shares one KV cache memory pool. Cross-slot batching is a measurable
  throughput win on the same GPU.
- Two separate processes can't batch across each other and each pays
  the model-weight memory footprint independently. So you spend 2× VRAM
  for the same throughput-on-paper, and the realized throughput is
  worse because no batching.

If your concurrency need is "10 simultaneous users chatting with the
same model," `chimera serve --parallel 10` is the right answer.

---

## 4. The summary that matters

> Router mode buys **model-residency parallelism** (multiple distinct
> models hot at once), **not compute parallelism on a single GPU**.
> Compute parallelism only materializes on multi-GPU.

| Workload | Right tool today |
|----------|------------------|
| Many users, one model, one GPU | `chimera serve --parallel N` (already shipping) |
| Many users, one model, many GPUs | `chimera serve --parallel N` with `CUDA_VISIBLE_DEVICES` set; or one chimera per GPU behind a reverse proxy |
| Few users, many *distinct* models, one GPU | Run one chimera per model behind nginx/caddy/Cloudflare with host- or path-based routing |
| Few users, many distinct models, many GPUs | Same — one chimera per model, each pinned to a GPU, behind a reverse proxy |
| Many users, many distinct models, many GPUs, with **autoload + LRU eviction** | The narrow router-mode use case (see § 5) |

The pattern: every realistic workload except the last one has a
deployment shape that today's chimera plus a reverse proxy solves at
least as well as router mode would, often better (no maintenance tax,
no architectural rewrite, well-trodden ops territory).

---

## 5. When to revisit

Re-open this decision **only** if one of these specific signals shows
up. Vague "wouldn't it be nice if chimera could do multi-model" is not
a trigger — the reverse-proxy answer covers it.

- **Concrete multi-GPU deployment request** where the user can't or
  won't run a reverse proxy. ("Can't" usually means a constrained
  enterprise environment with no LB layer available; "won't" usually
  means operational simplicity matters more than throughput, in which
  case running one chimera per model on different ports is still
  simpler than chimera in router mode.) Bar: the user has the hardware
  and is currently hitting the limitation, not "might in the future."
- **Concrete demand for autoload + LRU eviction** ("load model X on
  first request, evict if idle for Y minutes") that can't be served by
  scripting around `chimera serve` lifecycle (systemd socket
  activation, k8s scale-to-zero, etc.). The autoload semantics are the
  one thing router mode does that the reverse-proxy pattern doesn't
  trivially replicate.
- **Upstream collapses router mode into the static library.** If
  llama.cpp ever lifts the router machinery out of `llama-server`'s
  executable and into `libserver-context.a`, the porting cost drops
  from weeks to days. Worth re-evaluating if that shift happens.

None of those three is currently true. Today's decision: **document
and defer.**

For the smaller question — "make the webui's model-picker panel show a
clean error instead of a 404 against chimera" — see
[`doc/dev/webui.md`](webui.md) § 5.5. That's option A in the earlier
discussion: bind `POST /models/load` and `POST /models/unload` to
chimera-side stubs that return 501 "not supported in single-model
mode." Total cost ~15 lines. Independent of this decision; can land
any time the webui ergonomics complaint shows up.

---

## 6. Cross-references

- [`TODO.md`](../../TODO.md) — "Out of scope (wontfix)" section,
  "Multi-tenancy / `is_router_server`" bullet.
- [`doc/dev/server.md`](server.md) — the broader `chimera serve`
  developer guide; the "one process, one model" framing throughout is
  the design property this document protects.
- [`doc/dev/webui.md`](webui.md) § 5.5 — the polite-stub option A for
  `/models/{load,unload}`, which is the small piece of this problem
  that *is* worth doing on demand.
- `build/llama.cpp/tools/server/server-models.{h,cpp}` — the upstream
  router implementation referenced throughout § 1 and § 2.
- `build/llama.cpp/tools/server/server.cpp` lines 130–170 — the
  `is_router_server` switch and the routes bound only in router mode.
