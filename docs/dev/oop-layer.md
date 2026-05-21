# Optional OOP layer (`chimera.hpp`)

`libchimera.a` exposes a procedural surface: build an `Options` struct,
call `command_*(opts)`, get an exit code. That is what `chimera_cli/`
drives and what an embedding host (e.g. a custom server, a notebook
binding, a sidecar daemon) can drive directly.

`src/chimera/chimera.hpp` is an **optional**, **header-only** C++ veneer
over that surface. It exists to make the "load a model once, call it many
times" pattern less awkward to express in C++ than juggling raw
`LlamaModelPtr` + `LlamaCommonOptions` by hand.

It is not compiled into `libchimera.a`. It compiles at the consumer's
call site when they `#include "chimera.hpp"`. The procedural API stays
the source of truth; the wrappers are inline glue.

## Classes

| Class | Style | Wraps | Persistent? |
|-------|-------|-------|-------------|
| `chimera::Llama` | persistent-handle | `load_llama_model` + `new_llama_context` + `sample_loop` (text path); `run_generation_mtmd` (vision path) | yes - model AND context loaded once, ctx reused per `generate()` (text path only) |
| `chimera::Embedder` | persistent-handle | `chimera_embed::Embedder` (+ optional `chimera_embed_cache::Cache`) | yes |
| `chimera::Tokenizer` | persistent-handle | `load_llama_model` + `tokenize` / `token_to_piece` | yes |
| `chimera::Whisper` | persistent-handle | `chimera_whisper::load_model` + `chimera_whisper::transcribe` (structured-API) or `run_whisper` (CLI-shaped) | yes - whisper_context loaded once, reused across `transcribe()` / `run()` |
| `chimera::SD` | persistent-handle | `chimera_sd::load_model` + `chimera_sd::generate` (structured-API) or `run_sd` (CLI-shaped) | yes - sd_ctx_t loaded once, reused across `generate()` / `run()` |
| `chimera::Server` | options-in-ctor | `command_serve` | n/a - server owns its own lifecycle internally |

`chimera::Server` uses options-in-ctor because the server owns its own
lifecycle internally; `run()` blocks until the server shuts down and
there's no per-call work to amortize a persistent handle against.

`chimera::Whisper` and `chimera::SD` are persistent-handle: the ctor
calls the lower-level `chimera_whisper::load_model` /
`chimera_sd::load_model` and caches the handle. Both expose two run
flavors: a structured-API path (`transcribe()` / `generate()`) that
returns the raw `TranscribeResult` / `vector<PixelImage>` for library
consumers, and a CLI-shaped `run()` that calls into the post-load
pipeline helpers `run_whisper(ctx, opts)` / `run_sd(ctx, opts)` -
the same helpers `command_whisper` / `command_sd` use after loading,
so the OOP and CLI paths share one body.

### `chimera::Whisper` / `chimera::SD` persistence model

The library-side refactor that made this possible:
`command_whisper` and `command_sd` were split into a load shim that
builds the context and a post-load helper (`run_whisper` / `run_sd`)
that owns everything after that - WAV loading, resampling, diarize,
grammar, format-file writes for whisper; cache validation, prompt
parsing, control-net wiring, PNG writes for SD. Both CLI driver and
OOP wrapper invoke the same post-load helper, so behavior is
identical between `chimera whisper -m foo.bin -i x.wav` and
`chimera::Whisper(opts).run()`.

**Dirty-options policy.** Load-time fields silently no-op after the
ctor runs:

- `Whisper`: `model`, `no_gpu`, `flash_attn`, `gpu_device`.
- `SD`: `model`, `diffusion_model`, all the split-checkpoint paths
  (`vae`, `clip_l/g`, `t5xxl`, `llm`, `taesd`, `clip_vision`,
  `llm_vision`, `tensor_type_rules`, `photo_maker`, `embd_dir`,
  `high_noise_diffusion_model`, `control_net`), the `wtype` /
  `prediction` / `lora_apply_mode` enums, all the `*_on_cpu` /
  `*_conv_direct` / `*_flash_attn` / `*_mmap` / `max_vram` /
  `force_sdxl_vae_conv_scale` / `rng` / `sampler_rng` /
  `offload_to_cpu` knobs, and crucially `init_image` (which decides
  whether vae_decode_only is set at load time -- an instance built
  with `init_image` empty cannot do img2img later even if the field
  is set afterwards; reconstruct or call `reset(/*reload=*/true)`).

Call `reset(/*reload=*/true)` to drop the cached ctx and have the
next call honor mutated load-time fields. `reset()` without `reload`
is a no-op for whisper / SD since neither has a per-call cache
analogous to llama's KV.

Per-call fields (everything else in `WhisperOptions` / `SdOptions`)
take effect immediately -- they're consumed by `run_whisper` /
`run_sd` on every call, not at load time.

### `chimera::Llama` persistence model

The model loads in the constructor. The `llama_context` is **lazy** -
the first `generate()` call calls `new_llama_context` and caches the
handle. Subsequent `generate()` calls reuse it, saving the per-call
allocation cost (context, backend buffers, KV-cache buffers).

**Semantics are still single-shot**: the KV cache is cleared at the
start of each `generate()` via `llama_memory_clear`. So the persistent
ctx is an *internal optimization*, not a behavior change - you don't
have to think about prompt accumulation. (Conversation-style
append-mode is what `command_chat` does in the CLI; an analogous
`LlamaChat` carve-out from the REPL is a plausible future addition.)

LoRA adapters and the sampler are rebuilt on every `generate()` call.
This is cheap and means mutations to `options().lora_adapters` and to
sampler knobs (temp, top_k, seed, n_predict, samplers chain, logit
bias, grammar, ...) take effect immediately.

**Dirty-options policy.** Mutating *context-creation* fields after the
first `generate()` silently no-ops because the ctx is cached. Those
fields are: `n_ctx`, `n_batch`, `n_ubatch`, `cache_type_k/v`,
`flash_attn`, `rope_*`, `yarn_*`, `swa_full`, `control_vector*`. Call
`reset(/*rebuild=*/true)` to drop the cached ctx; the next
`generate()` will honor the new values. `reset()` without `rebuild`
just clears the KV (cheap; useful if you want to release memory used
by cached compute without paying re-allocation cost on the next call).

The vision path (`options().images` non-empty) bypasses the persistent
ctx entirely and falls back to `run_generation_mtmd`, which builds a
fresh ctx and `mtmd_context` per call. The dominant cost there is the
model load (already amortized), so the per-call ctx churn is not worth
the complexity of caching the vision-encoder bundle.

`chimera::Llama::ctx()` exposes the cached handle for callers who want
to drop down to the C API. Returns `nullptr` until the first
`generate()` (or after `reset(/*rebuild=*/true)`).

### Streaming

`Llama::generate` has two overloads:

```cpp
std::string generate(const std::string & prompt, bool stream = false);
std::string generate(const std::string & prompt, const chimera::TokenCallback & on_token);
```

The `bool` overload is the CLI-shaped convenience: `stream=false`
collects the full generated text and returns it; `stream=true` writes
each token to `std::cout` as it's sampled and prints a trailing newline
when the generation completes. It exists so toy programs and ports of
the CLI's gen subcommand have a one-line call site.

The callback overload is the library-friendly form. `TokenCallback` is
`std::function<void(std::string_view)>`; it's invoked once per sampled
token with the detokenized UTF-8 piece. The full text is still
returned. The caller owns where the bytes go - a Slack bot writes to
its WebSocket, a notebook binding appends to a cell buffer, a logger
writes to a file - and the caller owns trailing-newline / buffering /
flushing policy. Passing an empty `TokenCallback{}` disables streaming
without changing the return value.

Library-side, the procedural `sample_loop` / `run_generation` /
`run_generation_mtmd` all take a `const TokenCallback &` now (replaced
what used to be a `bool stream_output`). The CLI's `command_prompt`
passes a stdout-writing lambda and prints its own trailing newline.

### Vision streaming

The mtmd path (vision prompts) supports streaming via the same
callback because `run_generation_mtmd` accepts a `TokenCallback`.
This is independent of the persistent-ctx limitation - even though
the vision path builds a fresh ctx per call, the per-token streaming
hook works the same way.

`command_chat` (the interactive REPL) is **not** wrapped. It still lives
in `src/chimera_cli/` because it owns terminal I/O, signal handling,
linenoise, and color streaming. A future carve-out (a stateless
`LlamaChat` class) is plausible; until then, callers wanting chat
semantics should drive `Llama::generate` against their own
chat-templated prompts or talk to a `chimera::Server`.

## Example

```cpp
#include "chimera.hpp"

int main() {
    LlamaCommonOptions opts;
    opts.model     = "Qwen3-1.7B-Q4_0.gguf";
    opts.n_predict = 128;

    chimera::Llama llm(opts);            // model loads here
    auto reply  = llm.generate("What is the capital of France?");
    llm.options().n_predict = 32;        // tweak between calls
    auto reply2 = llm.generate("And the capital of Spain?");

    chimera::Embedder emb({.model = "bge-small.gguf"});
    auto vec = emb.embed("hello world");
}
```

Every class also exposes `options()` (mutable + const) and most expose
`raw()` for callers that need to drop down to the underlying C handle.

## Including the header

The header is at `src/chimera/chimera.hpp`. `chimera_lib`'s PUBLIC
include directory already covers `src/chimera/`, so any target that
links `chimera_lib` (the CMake target, not just the `.a`) picks up the
header automatically:

```cmake
target_link_libraries(my_app PRIVATE chimera_lib)
#include "chimera.hpp"  // works
```

External consumers that link `libchimera.a` directly (i.e. not through
the CMake target) need to add `src/chimera/` to their include path
themselves. See `tests/external/CMakeLists.txt` for the exact recipe -
it also adds `thirdparty/llama.cpp/include` because the OOP header
transitively includes `chat.h` / `common.h` / `sampling.h` / `mtmd.h`
from the staged llama.cpp install root.

## Smoke test

`tests/external/hpp_smoke.cpp` is the canonical compile-and-link probe
for the header. It runs as part of `make test-external-smoke`:

- Instantiates every option struct and every wrapper class - proves
  the header parses and compiles without the consumer needing any
  CMake-only assumptions.
- (Optional, gated on `CHIMERA_SMOKE_MODEL=<path/to/model.gguf>`)
  Round-trips a string through `chimera::Tokenizer::encode` /
  `decode`, then drives a one-token generation through
  `chimera::Llama::generate`, then reuses the same `Llama` instance
  with a mutated `n_predict` to confirm the persistent-handle
  behavior end-to-end.

The procedural-API counterpart is `tests/external/smoke.cpp`. Both
build from the same `tests/external/CMakeLists.txt`.

## Relationship to the library refactor

Before the OOP work landed, several llama.cpp glue helpers
(`load_llama_model`, `new_llama_context`, `run_generation`,
`run_generation_mtmd`, `make_sampler`, `load_loras`, `decode_tokens`,
`command_prompt` / `command_embed` / `command_tokenize`) lived in
`src/chimera_cli/chimera.cpp` (the executable's TU). That meant
`libchimera.a` had no direct llama.cpp text-generation entrypoint at
all - only `command_serve` (HTTP) and the lower-level `Embedder`.

Those helpers moved into a new `src/chimera/chimera_llama.{h,cpp}` so
that the OOP layer (and any other external consumer of the archive)
can call them. The CLI shell now `#include`s `chimera_llama.h` and is
~1000 lines shorter.
