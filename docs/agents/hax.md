# Running the hax coding agent against chimera

[hax](https://github.com/shakfu/hax) is a terminal coding agent written in C.
`scripts/agent_hax.sh` and `make agent-hax` run it against a model hosted by
`chimera serve`.

hax is not vendored, linked, or version-pinned here. It stays an external
binary found on `PATH` or named by `HAX=<path>`. The two programs meet over
loopback HTTP.

## Why HTTP rather than a bundled build

Three properties of `chimera serve` make the connection work with no adapter:

- It serves the OpenAI Chat Completions API that hax's `llama.cpp` provider
  expects, and defaults to `127.0.0.1:8080`, which is that provider's own
  default endpoint.
- Jinja chat templating is on by default (`--no-jinja` disables it), so tool
  calling works. An agent is unusable without it.
- hax adopts the single served model from `GET /v1/models` and reads the
  context size from `GET /props`, so neither has to be configured.

Vendoring hax instead would add a fourth upstream pin to a project that already
carries three, pull in `libcurl` and `jansson` against chimera's single-static-
binary goal, and require porting a meson C project into chimera's CMake build.
hax's own `docs/embedding.md` states that composition through a subprocess is
its intended extension path.

## Quick start

Interactive:

```sh
CHIMERA_SERVE_ARGS="--gpu-layers 99 -c 32768" \
    scripts/agent_hax.sh models/Qwen3.5-9B-Q4_K_M.gguf
```

One-shot:

```sh
CHIMERA_SERVE_ARGS="--gpu-layers 99 -c 32768" \
    scripts/agent_hax.sh models/Qwen3.5-9B-Q4_K_M.gguf \
    -p "How many .cpp files are in src/chimera? Use a shell command, then answer with just the number."
```

```
hax: llama.cpp - Qwen3.5-9B-Q4_K_M - session 2dc8e424-...

19

32s - 2.4k / 33k (7%)
resume with: hax --resume=2dc8e424-86e3-4d96-82d1-953f6d80da84
```

No arguments at all:

```sh
make agent-hax-default
```

`agent-hax-default` runs `agent-hax` with `AGENT_MODEL` and `AGENT_SERVE_ARGS`
applied as target-specific variables, so it needs no duplicate recipe. Change
either default without editing the target:

```sh
make agent-hax-default AGENT_MODEL=models/Qwen3.5-9B-Q4_K_M.gguf
make agent-hax-default AGENT_SERVE_ARGS="--gpu-layers 99 -c 16384"
```

Through make, naming the model explicitly:

```sh
make agent-hax MODEL=models/Qwen3.5-9B-Q4_K_M.gguf \
    CHIMERA_SERVE_ARGS="--gpu-layers 99 -c 32768"
```

`HAX_ARGS` passes arguments to hax from the make target:

```sh
make agent-hax MODEL=models/Qwen3.5-9B-Q4_K_M.gguf HAX_ARGS='-p "list TODOs"'
```

## Serve flags worth setting

- `--gpu-layers 99` offloads all layers. Without it a 9B model runs on CPU and
  a turn takes minutes.
- `-c 32768` overrides chimera's `n_ctx = 0` default. That default requests the
  model's full training context (262144 tokens for Qwen3.5-9B), whose KV cache
  does not fit alongside 5.2 GB of weights. Raise the value if compaction
  triggers early; lower it on allocation failure at load.

## Model sizing

Tool calling requires a model that emits the protocol reliably. Measured on
this repository:

| Model | Result |
| --- | --- |
| `Llama-3.2-1B-Instruct-Q8_0` | Unusable. Loops until `max turns (100) exceeded`, or emits output hax cannot parse. |
| `Qwen3-4B-Q8_0` | Handles plain turns. Multi-step tool use untested. |
| `Qwen3.5-9B-Q4_K_M` | Correct tool call and answer in 32s with full GPU offload. |
| `LFM2.5-8B-A1B-Q4_K_M` | Same tool call and answer in 8s. The default for `agent-hax-default`. |

Treat 4B as the floor and 8B-9B as the working size. `LFM2.5-8B-A1B` is a
mixture-of-experts model activating roughly 1B parameters per token, which is
why it answers four times faster than the dense 9B on the same task.

## Environment

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHIMERA` | `<repo>/build/chimera` | chimera binary; falls back to `PATH`. |
| `HAX` | `hax` on `PATH` | hax binary. |
| `CHIMERA_AGENT_PORT` | `8080` | First port tried. |
| `CHIMERA_SERVE_ARGS` | empty | Extra `chimera serve` flags. Word-split deliberately. |
| `CHIMERA_AGENT_TIMEOUT` | `180` | Seconds to wait for model load. |

Make variables, for the `agent-hax` and `agent-hax-default` targets:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL` | none, required | Model for `agent-hax`. |
| `HAX_ARGS` | empty | Arguments passed through to hax. |
| `AGENT_MODEL` | `models/LFM2.5-8B-A1B-Q4_K_M.gguf` | Model for `agent-hax-default`. |
| `AGENT_SERVE_ARGS` | `--gpu-layers 99 -c 32768` | Serve flags for `agent-hax-default`. |

Serving with `--api-key K` additionally needs `HAX_LLAMACPP_API_KEY=K`.

The script sets three hax variables. `HAX_PROVIDER=llama.cpp` and
`HAX_LLAMACPP_BASE_URL` bind hax to the port actually chosen.
`HAX_CATALOG_URL` is emptied to skip the models.dev metadata fetch, which has
no entry for a local GGUF file; an explicit value in the environment wins.

## What the script handles

Four behaviours that a bare `chimera serve & hax` does not provide:

1. **Port selection.** Probes upward from 8080 until a port is free, detected
   by curl exit code 7 (failed to connect). A server already running on 8080
   is left alone.
2. **Signal routing.** The server starts as `( trap '' INT; exec chimera serve
   ... ) &`. A background child in the same process group would otherwise
   receive terminal SIGINT, so Ctrl-C in hax (which interrupts one turn) would
   kill the model server. `SIG_IGN` survives `exec`; an installed handler would
   not. The wrapper itself uses `trap : INT` rather than `trap '' INT`, because
   children inherit `SIG_IGN` across `exec` and hax would lose its own Ctrl-C.
   SIGTERM stays at its default so teardown still works.
3. **Readiness gate.** Polls `/health` and aborts if the server exits first, so
   a bad model path reports chimera's error rather than a hax connection error.
4. **Teardown.** An `EXIT`/`TERM` trap kills the server. The script deliberately
   does not `exec` hax, since `exec` would discard the trap and leak the server.

## Troubleshooting

- `hax not found on PATH` - install hax, or set `HAX=<path/to/hax>`. In the hax
  repository, `make symlink` links the built binary into `~/.local/bin`.
- `chimera serve exited before becoming ready` - the server failed during model
  load. Run the same `chimera serve` command directly to read its error.
- `server not ready after 180s` - a large model on a cold page cache. Raise
  `CHIMERA_AGENT_TIMEOUT`.
- `max turns exceeded` or a provider parse error - the model is too small. See
  the sizing table.
