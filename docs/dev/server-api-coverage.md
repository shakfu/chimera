# Chimera Serve — CLI/HTTP Parity Inventory

Report date: 2026-05-20
Companion to [cli-api-coverage.md](cli-api-coverage.md), which audits the
CLI subcommand surface against upstream. This document audits the
**HTTP server surface** (`chimera serve`) against the matching CLI
subcommands.

## TL;DR

After the 2026-05-20 audit cycle, every CLI subcommand (`gen`, `chat`,
`embed`, `whisper`, `sd`) reached zero unresolved `❌` rows against
upstream. The HTTP server (`chimera serve --enable-audio …`,
`--enable-image …`) shares the same engine — both modalities route
through `chimera_whisper::transcribe()` and `chimera_sd::generate()`
respectively — but the **JSON request shapes accept only the OpenAI
subset**. None of the long-tail flags that landed on the CLI
(`--diarize`, `--vad`, `--grammar`, `--guidance`, `--hires`,
`--cache-mode`, `--ref-image`, control vectors, etc.) are surfaced
per-request on the server.

Two parity gaps:

1. **Per-request gap.** Many CLI flags that already exist on
   `TranscribeRequest` / `GenerateRequest` aren't read from the HTTP
   body. Adding them is purely handler-side — the engine doesn't need to
   change.
2. **Server-init gap.** The serve path calls the **simple overload** of
   `chimera_whisper::load_model(path)` and
   `chimera_sd::load_model(path, vae_decode_only)`. The richer
   `LoadParams` overloads — which the CLI uses for `--flash-attn`,
   `--no-mmap`, `--max-vram`, `--photo-maker`, `--control-net`,
   `--lora`, `--prediction`, `--lora-apply-mode`, `--clip-on-cpu` /
   `--vae-on-cpu`, `--vad-model`, etc. — are **never invoked from the
   serve path today**. Closing this requires routing new
   `--enable-audio-*` / `--enable-image-*` CLI flags through to a
   `LoadParams` build, which is a one-time refactor that unlocks every
   server-init feature at once.

Steps 1 and 2 of the
[recommended order](#recommended-order) below close ~80% of the
practical parity gap.

---

## Currently accepted

### `POST /v1/audio/transcriptions` and `/v1/audio/translations`

Handlers in `src/chimera/chimera_serve_audio.cpp`.

| JSON / multipart field | Effect |
|---|---|
| `file` (multipart, required) | Audio bytes; decoded via `chimera_whisper::load_wav_bytes`. |
| `model` | Ignored — the server picks the loaded whisper model. |
| `language` | Pass-through to `treq.language`; `"auto"` triggers detection. |
| `prompt` | Pass-through to `treq.initial_prompt`. |
| `response_format` | `json` / `text` / `verbose_json` / `srt` / `vtt`. |
| `timestamp_granularities[]` | `"word"` sets `treq.word_timestamps = true`. |
| `temperature` | **Parsed but inert today** — see handler comment. |

### `POST /v1/images/generations` / `/v1/images/edits` / `/v1/images/variations`

Handlers in `src/chimera/chimera_serve_images.cpp`.

| JSON / multipart field | Effect |
|---|---|
| `prompt` | Required (generations / edits); ignored for variations. |
| `negative_prompt` | Pass-through. |
| `n` | → `req.batch_count`. |
| `size` | `<W>x<H>` → `req.width` / `req.height`. |
| `steps`, `cfg_scale`, `seed`, `sample_method`, `scheduler`, `strength` | Direct pass-through. |
| `response_format` | `b64_json` / `url` (default `b64_json`). |
| `image` (multipart, /edits, /variations) | → `req.init`. |
| `mask` (multipart, /edits) | → `req.mask` (inpaint). |

### Chat / completions / embeddings

Already mature — the LLM-side surface exposes the full common_params
field set via the OpenAI body. No parity work needed here.

---

## Audio — inventory

### Top picks (per-request, mechanical handler change)

All map to existing `TranscribeRequest` fields — engine is unchanged.

| Field (JSON) | Maps to | Why it matters |
|---|---|---|
| `temperature` | `treq.temperature` | **Already accepted but discarded.** Cheapest win: wire the existing field through. |
| `beam_size` | `treq.beam_size` | Per-request quality/speed trade; common ask. |
| `best_of` | `treq.best_of` | Pair with `temperature=0` greedy; common. |
| `no_fallback` | `treq.no_fallback` | Disable the temperature-fallback ladder. |
| `temperature_inc`, `entropy_thold`, `logprob_thold`, `no_speech_thold` | same | Decoder-fail thresholds; useful when debugging garbled audio. |
| `vad` (+ tuning knobs) | `treq.vad` etc. | VAD is becoming the de-facto default. Per-request opt-in. **Requires server-init load of the VAD model** — see Tier 3. |
| `offset_ms`, `duration_ms` | same | **High value for long uploads** — slice a 1-hour podcast without re-encoding the source. |
| `diarize` | `treq.diarize` | Stereo input → `(speaker N)` labels in the response. Pairs cleanly with `verbose_json` / `srt` / `vtt`. |
| `max_len`, `split_on_word` | same | Pair naturally with `response_format=srt`/`vtt`. |

### Tier 2 — per-request, more thought

- `grammar` / `grammar_rule` / `grammar_penalty` — per-request constrained
  decoding. Lifetime is fine (`parse_state` lives on the handler scope).
  Open question: JSON shape — pass GBNF as a string field, or as a
  separate file part. Probably string field with a size cap.
- `suppress_nst`, `suppress_regex` — useful but rarely the right API
  knob; usually belongs in `prompt`.
- `audio_ctx`, `tinydiarize` — niche; expose if/when asked.
- `processors` — careful. The server already runs handlers on its
  thread pool, so `processors > 1` would multiply thread pressure. If
  exposed, cap server-side.
- `detect_language` — useful as a probe, but I'd argue it deserves its
  own endpoint (`POST /v1/audio/detect_language`) rather than a query
  parameter that silently suppresses transcription. Whisper's
  OpenAI-shape doesn't have this concept.

### Tier 3 — server-init only (needs `LoadParams` routing first)

| CLI flag | Why server-init | Prerequisite |
|---|---|---|
| `--flash-attn` | Compiled into the whisper context. | `--enable-audio-flash-attn` |
| `--no-gpu`, `--device` | Backend choice at context init. | `--enable-audio-no-gpu`, `--enable-audio-device` |
| `--threads` | Thread budget at context init. | already partly covered (server has its own thread pool) |
| `--vad-model` | The VAD model file is loaded by whisper at context init when `vad=true` arrives on a request. | `--enable-audio-vad-model` |

Today's serve path calls `chimera_whisper::load_model(path)` — the
simple overload that takes only the model path. To enable any of these,
add `--enable-audio-*` family CLI flags and route them through to the
`chimera_whisper::LoadParams` overload.

### Skip

- All `--output-*` flags — subsumed by `response_format`.
- `--output-file` — N/A on HTTP (no filesystem destination).

---

## Image — inventory

### Top picks (per-request, mechanical handler change)

All map to existing `GenerateRequest` fields.

| Field (JSON) | Maps to | Why it matters |
|---|---|---|
| `clip_skip` | `req.clip_skip` | Common per-request override. |
| `guidance` | `req.guidance` | **Flux/SD3 essential** — without this, those models can't be steered correctly. |
| `flow_shift` | `req.flow_shift` | Same; Flux/SD3 sampling. |
| `img_cfg_scale` | `req.img_cfg_scale` | Separate img-cond CFG (Flux). |
| `eta`, `timestep_shift`, `sigmas` | same | Per-request sampler control. |
| `vae_tiling` (+ tile-size / overlap / relative-size) | same | **VRAM safety** for >1024² outputs — a different OOM / no-OOM outcome, not just a quality knob. |
| `skip_layers`, `slg_scale`, `skip_layer_start`, `skip_layer_end` | same | SLG values commonly shift per prompt. |
| `hires` + `hires_scale` / `hires_steps` / `hires_denoising_strength` / `hires_upscaler` | `req.hires_*` | Popular two-pass workflow. `Latent*` upscalers need no extra model; `Model` upscaler needs server-init. |
| `cache_mode` + `cache_option` | parsed via `chimera_sd::parse_cache_options` | Inference cache toggle per request. CPU and GPU users have very different cache needs. |
| `control_image` (multipart file) + `control_strength` | `req.control` / `req.control_strength` | The ControlNet *model* is server-init; the conditioning image is per-request. |

### Tier 2 — per-request, more thought

- `ref_image` (repeatable multipart) + `increase_ref_index` +
  `no_auto_resize_ref_image` — IP-adapter-style style/identity
  conditioning. JSON-shape decision: OpenAI's body has no precedent
  for repeatable image uploads; chimera would invent the field name.
- PhotoMaker bundle (`pm_id_images_dir` → multipart files, `pm_id_embed_path`,
  `pm_style_strength`) — requires `--photo-maker` model loaded at
  startup. Same multipart-plural shape question as ref-images.
- `lora` — per-request LoRA selection. **Design question:** should the
  server load a fixed set of LoRAs at startup and let requests pick by
  name, or should requests reference filesystem paths the server
  resolves at request time? The former is safer; the latter more
  flexible. Mirrors the choice llama-server made with its
  `--lora-adapters` config.

### Tier 3 — server-init only (needs `LoadParams` routing first)

Nearly everything else that's a CLI flag: model component paths
(`--diffusion-model`, `--vae`, `--clip-l`, `--clip-g`, `--t5xxl`, `--llm`,
`--clip-vision`, `--llm-vision`, `--taesd`, `--tensor-type-rules`,
`--high-noise-diffusion-model`, `--control-net`, `--photo-maker`,
`--embd-dir`), weights (`--type`, `--lora-model-dir`), RNG
(`--rng`, `--sampler-rng`), perf/offload (`--offload-to-cpu`,
`--diffusion-fa`, `--fa`, `--diffusion-conv-direct`, `--vae-conv-direct`,
`--no-mmap`, `--max-vram`, `--clip-on-cpu`, `--vae-on-cpu`,
`--control-net-cpu`, `--force-sdxl-vae-conv-scale`), enums
(`--prediction`, `--lora-apply-mode`).

Today's serve path calls
`chimera_sd::load_model(path, vae_decode_only=false)` — the simple
overload. To unlock any of these, add `--enable-image-*` flag families
and route through `chimera_sd::LoadParams`. Once routed, every flag in
this paragraph becomes available at server start.

### Skip

- `--disable-image-metadata` — already 🚫 in `cli-api-coverage.md`.
  Chimera writes no PNG metadata.
- Video-only flags (`--end-img`, full `--high-noise-*` family,
  `--moe-boundary`, `--control-video`, `--video-frames`, `--fps`,
  `--vace-strength`) — chimera-sd is img_gen-only.
- Standalone-mode flags (`--upscale-repeats`, `--mode`) — chimera-sd
  doesn't switch modes.

---

## Recommended order

Pareto-shaped roadmap:

1. **Audio wave 1** (1 PR, ~40 LOC). Wire the existing-but-inert
   `temperature` field, plus add `beam_size`, `best_of`, `no_fallback`,
   `offset_ms`, `duration_ms`, `diarize`, `max_len`, `split_on_word`. All
   purely handler-side; the `TranscribeRequest` fields already exist.
   No serve-init flags needed.
2. **Image wave 1** (1 PR, ~80 LOC). Add `clip_skip`, `guidance`,
   `flow_shift`, `img_cfg_scale`, `eta`, `timestep_shift`, `vae_tiling`
   (+ knobs), `skip_layers` / `slg_scale` / `skip_layer_start`/`_end`.
   All `GenerateRequest` fields already exist; same shape as wave 1.
3. **Hires-fix** (1 PR). Bundle `hires` + scale/steps/upscaler/tile
   knobs. `Latent*` upscalers work immediately; the `Model` upscaler
   waits on step 4.
4. **Cache + SCM** (1 PR). `cache_mode` + `cache_option` per request.
   The chimera-side parser (`parse_cache_options`) already validates
   per-mode kv-key combinations.
5. **Server-init routing** (1 PR, structural). Plumb
   `chimera_whisper::LoadParams` and `chimera_sd::LoadParams` through
   the serve path. Adds `--enable-audio-*` / `--enable-image-*` flag
   families to `chimera serve`. **This is the gating step** for every
   server-init item in Tier 3 of both modalities — once it lands, each
   subsequent flag is a one-line addition.
6. **VAD on serve** (after 5). Per-request `vad=true` opt-in.
7. **ControlNet on serve** (after 5). Per-request `control_image` +
   `control_strength`.
8. **PhotoMaker on serve** (after 5). Multipart `pm_id_images[]` +
   per-request `pm_style_strength`.
9. **Per-request LoRA selection** (after 5). Choose the "named adapters
   loaded at startup" vs. "paths resolved at request time" model first.

Steps 1 and 2 alone close ~80% of the practical CLI/serve parity gap.
Step 5 is the structural unblock for everything else.
