# `--gpu-layers` defaults: why `serve` auto-fits but `chat`/`gen` don't

`chimera serve` defaults `--gpu-layers` to `-1` (auto) while `chat`, `gen`,
and `embed` default it to `0` (pure CPU). This is not an oversight or mere
conservatism: the two groups load models through different code paths, and
only the `serve` path has access to llama.cpp's VRAM-aware auto-fit. `-1`
on the `chat`/`gen` path would be actively unsafe (naive "all layers,
OOM-if-too-big"), so `0` is the only safe default there today.

This note records the trace and sketches what it would take to give the
`chat`/`gen`/`embed` paths the same auto behavior.

## What `-1` means to llama.cpp

`llama_model::n_gpu_layers()` resolves a negative request to "everything":

```cpp
// src/llama-model.cpp:8073
uint32_t llama_model::n_gpu_layers() const {
    return params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer + 1;
}
```

So on its own, `-1` is *not* "auto-detect what fits" — it's "all layers
(+ output)". The VRAM-awareness comes from a separate, earlier step:
`llama_params_fit` (`src/llama.cpp:162`), gated behind
`common_params::fit_params` (default **`true`**, `common/common.h:427`).
`llama_params_fit` measures free device memory, projects usage starting
from the requested layer count, and:

- **fits with margin** → leaves the request untouched (so `-1` stays `-1`
  → all layers);
- **doesn't fit** → shrinks context first (down to `fit_params_min_ctx`,
  default 4096), then reduces offloaded layers, targeting a per-device
  free-memory margin (`fit_params_target`, default 1 GiB).

Crucially, `llama_params_fit` only runs inside `common_init_result`'s
constructor:

```cpp
// common/common.cpp:1149
if (params.fit_params) {
    llama_params_fit(params.model.path.c_str(), &mparams, &cparams, ...);
}
```

Reach that constructor and `-1` becomes genuine "offload as much as fits,
leaving ~1 GiB headroom." Skip it and `-1` is a blind all-layers load.

## The `serve` path reaches the fit step

```text
ServeOptions.gpu_layers = -1                         chimera.h:491 (default)
  → build_common_params: params.n_gpu_layers = -1    chimera_serve.cpp:416
    (fit_params left at upstream default = TRUE       common.h:427)
  → server_context::load_model(params)               chimera_serve.cpp:1084
    → common_init_from_params(params_base)            server-context.cpp:643
      → new common_init_result(params)                common.cpp:1269
        → if (params.fit_params) llama_params_fit()   common.cpp:1149   ✅ runs
```

`serve` builds a `common_params`, leaves `fit_params` at its upstream
default of `true`, and loads through llama.cpp's own server engine — which
routes through `common_init_from_params`. The auto-fit runs. The
`-1 = auto` comment on `ServeOptions::gpu_layers` is therefore accurate.

## The `chat`/`gen`/`embed` paths skip it

```cpp
// src/chimera/chimera_llama.cpp:457
llama_model_params params = llama_model_default_params();
params.n_gpu_layers = opts.gpu_layers;
...
llama_model_load_from_file(opts.model.c_str(), params);   // bare llama API
```

These build a raw `llama_model_params` and call `llama_model_load_from_file`
directly. There is no `common_params`, no `common_init_from_params`, and so
no `fit_params` / `llama_params_fit` step. `embed` does the same
(`chimera_embed.cpp:76` sets `mparams.n_gpu_layers` directly).

On these paths `-1` would hit the `n_gpu_layers() → n_layer + 1` fallback
with no VRAM check and no margin: load every layer, OOM if the model is
bigger than the card. That is why `0` is the default here — it is the safe
choice given the path has no fit safety net, not a value anyone would want
for a GPU build.

## What it would take to make `chat`/`gen` auto-fit

Flipping the default to `-1` is *not* enough — it would just turn on the
unsafe all-layers behavior. The loader has to be routed through the fit
step. Two options:

1. **Route the load through `common_init_from_params`.** Convert the
   `chat`/`gen`/`embed` loaders to populate a `common_params` (as `serve`
   already does in `build_common_params`) and load via
   `common_init_from_params` instead of `llama_model_load_from_file`. This
   inherits `fit_params` for free and keeps behavior consistent with
   `serve`. Largest change; pulls in the full `common_params` surface.

2. **Call `llama_params_fit` directly before the bare load.** Keep the
   `llama_model_params` path but, when `gpu_layers < 0`, build the minimal
   `cparams`/margins and call `llama_params_fit(model_path, &mparams,
   &cparams, ...)` ourselves before `llama_model_load_from_file`. Smaller
   blast radius, but duplicates a slice of `common_init_result`'s logic
   and has to be kept in sync with upstream.

Either way the default for `chat`/`gen`/`embed` can then move to `-1`, and
the behavior matches `serve`: full offload when it fits, graceful
degradation when it doesn't.

### Caveats to verify before doing this

- `llama_params_fit` reads model metadata and probes device memory before
  the real load — confirm the added startup cost is acceptable for the
  one-shot `gen` path (it runs once per invocation, not per token).
- `--device` / `--tensor-split` / `--main-gpu` interactions: `serve`
  already passes these through `common_params`; the direct-call option (2)
  must forward them into the fit call or the projection will be wrong.
- `embed` runs with reduced batch sizes; make sure the fit projection uses
  the same `cparams` the model is actually loaded with.

## See also

- `docs/dev/server.md` — the `serve` engine and its `common_params` mapping.
- Memory: `project_chimera_gpu_layers` — the user-facing "`chat`/`gen`
  default to 0, pass `--gpu-layers 99`" note.
