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
| `chimera::Llama` | persistent-handle | `load_llama_model` + `run_generation` / `run_generation_mtmd` | yes - model loaded once in ctor, reused per `generate()` |
| `chimera::Embedder` | persistent-handle | `chimera_embed::Embedder` (+ optional `chimera_embed_cache::Cache`) | yes |
| `chimera::Tokenizer` | persistent-handle | `load_llama_model` + `tokenize` / `token_to_piece` | yes |
| `chimera::Whisper` | options-in-ctor | `command_whisper` | no - full load/run/unload per `run()` |
| `chimera::SD` | options-in-ctor | `command_sd` | no - same |
| `chimera::Server` | options-in-ctor | `command_serve` | n/a - server owns its own lifecycle internally |

Whisper / SD / Server use options-in-ctor because their underlying
`command_*` functions own the full load-run-unload lifecycle today.
Persistent-handle for those would require library-side refactors that
are out of scope for the first cut.

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
