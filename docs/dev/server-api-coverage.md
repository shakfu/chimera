# Chimera Serve — CLI/HTTP Parity Inventory

Report date: 2026-05-20
Last status update: 2026-05-20 (audio wave 1 + image wave 1 landed)
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
   change. **Audio wave 1 closed 9 fields and image wave 1 closed 14
   fields on 2026-05-20.** Remaining per-request work for both
   modalities is now wave-2 / Tier-2 territory; Tier-3 items need the
   server-init plumbing (gap 2) first.
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
practical parity gap; both have shipped.

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
| `temperature` | ✅ Landed 2026-05-20 (audio wave 1) — pass-through to `treq.temperature`. |
| `beam_size`, `best_of`, `no_fallback` | ✅ Landed 2026-05-20 (audio wave 1). |
| `offset_ms`, `duration_ms` | ✅ Landed 2026-05-20 (audio wave 1) — slice-the-upload, no re-encoding. |
| `max_len`, `split_on_word` | ✅ Landed 2026-05-20 (audio wave 1) — pairs with `response_format=srt`/`vtt`. |
| `diarize` | ✅ Landed 2026-05-20 (audio wave 1) — mono uploads return 400; stereo uploads get `(speaker N)` labels stamped onto each segment's text (plus the joined transcript). |

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
| `clip_skip`, `guidance`, `flow_shift`, `img_cfg_scale`, `eta`, `timestep_shift` | ✅ Landed 2026-05-20 (image wave 1) — all direct pass-through; sentinel defaults on `GenerateRequest` keep omitted keys at upstream defaults. |
| `vae_tiling` (+ `vae_tile_size`, `vae_relative_tile_size`, `vae_tile_overlap`) | ✅ Landed 2026-05-20 (image wave 1) — VRAM-safety knob for large outputs. |
| `skip_layers`, `slg_scale`, `skip_layer_start`, `skip_layer_end` | ✅ Landed 2026-05-20 (image wave 1) — SLG bundle. `skip_layers` accepts JSON array OR comma-separated string; non-integer entries return 400 with a helpful message. |
| `sigmas` | ✅ Landed 2026-05-20 (image wave 1) — custom sigma schedule, same array-or-CSV shape as `skip_layers`. |

### Chat / completions / embeddings

Already mature — the LLM-side surface exposes the full common_params
field set via the OpenAI body. No parity work needed here.

---

## Audio — inventory

### Top picks (per-request, mechanical handler change)

All map to existing `TranscribeRequest` fields — engine is unchanged. **Wave 1 (9 fields) landed 2026-05-20.**

| Field (JSON) | Maps to | Status | Notes |
|---|---|---|---|
| `temperature` | `treq.temperature` | ✅ Wave 1 | Was previously parsed but discarded — wired through. |
| `beam_size` | `treq.beam_size` | ✅ Wave 1 | |
| `best_of` | `treq.best_of` | ✅ Wave 1 | |
| `no_fallback` | `treq.no_fallback` | ✅ Wave 1 | |
| `offset_ms`, `duration_ms` | same | ✅ Wave 1 | Slice verified — 4–8s window on the JFK sample returns the matching transcript fragment. |
| `max_len`, `split_on_word` | same | ✅ Wave 1 | |
| `diarize` | `treq.diarize` | ✅ Wave 1 | Energy-ratio classifier (1.1× threshold) duplicated from `command_whisper` into the handler — if a third caller appears, promote to `chimera_whisper::estimate_diarization_speaker`. Mono → 400; stereo → labels stamped on `Segment.speaker` + prefixed onto `Segment.text` + `result.text` rebuilt so `response_format=text` also reflects speakers. |
| `temperature_inc`, `entropy_thold`, `logprob_thold`, `no_speech_thold` | same | Wave 2 (planned) | Decoder-fail thresholds; useful when debugging garbled audio. |
| `vad` (+ tuning knobs) | `treq.vad` etc. | Wave 2 (planned) | VAD is becoming the de-facto default. **Requires server-init load of the VAD model** — see Tier 3. |

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

All map to existing `GenerateRequest` fields. **Wave 1 (14 fields) landed 2026-05-20.**

| Field (JSON) | Maps to | Status | Notes |
|---|---|---|---|
| `clip_skip` | `req.clip_skip` | ✅ Wave 1 | |
| `guidance` | `req.guidance` | ✅ Wave 1 | Flux/SD3 distilled guidance. |
| `flow_shift` | `req.flow_shift` | ✅ Wave 1 | Flux/SD3 timestep shift. |
| `img_cfg_scale` | `req.img_cfg_scale` | ✅ Wave 1 | |
| `eta`, `timestep_shift`, `sigmas` | same | ✅ Wave 1 | `sigmas` accepts JSON array `[14.6, 10.0, ...]` or CSV string `"14.6,10.0,..."`. Non-numeric tokens → 400. |
| `vae_tiling` (+ knobs) | same | ✅ Wave 1 | **VRAM safety** for large outputs. `vae_tiling` bool uses the shared `coerce_bool` helper. |
| `skip_layers`, `slg_scale`, `skip_layer_start`, `skip_layer_end` | same | ✅ Wave 1 | `skip_layers` accepts array or CSV; non-int → 400. Wrong-type values (number instead of array/string) → 400 with explicit message. |
| `cache_mode` + `cache_option` | parsed via `chimera_sd::parse_cache_options` | Wave 2 (planned) | Inference cache toggle. CPU/GPU users have different cache needs. |
| `hires` + `hires_scale` / `hires_steps` / `hires_denoising_strength` / `hires_upscaler` | `req.hires_*` | Wave 2 (planned) | Popular two-pass workflow. `Latent*` upscalers need no extra model; `Model` upscaler needs server-init. |
| `control_image` (multipart file) + `control_strength` | `req.control` / `req.control_strength` | Wave 2 (planned) | The ControlNet *model* is server-init; the conditioning image is per-request. |

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

1. ~~**Audio wave 1** — `temperature` (was inert), `beam_size`, `best_of`,
   `no_fallback`, `offset_ms`, `duration_ms`, `diarize`, `max_len`,
   `split_on_word`. Purely handler-side.~~ ✅ **Landed 2026-05-20** (~70 LOC
   in `chimera_serve_audio.cpp`; ~50 LOC of the total was the diarize
   energy-ratio classifier + post-transcribe segment stamping).
2. ~~**Image wave 1** — `clip_skip`, `guidance`, `flow_shift`,
   `img_cfg_scale`, `eta`, `timestep_shift`, `sigmas`, `vae_tiling`
   (+ knobs), `skip_layers` / `slg_scale` / `skip_layer_start`/`_end`.
   14 fields, all into the shared `fill_common_image_fields()` helper
   so `/generations`, `/edits`, `/variations` pick them up at once.~~
   ✅ **Landed 2026-05-20.** Same PR promoted `coerce_bool` from an
   inlined lambda in the audio handler to the shared
   `chimera_serve_internal.h` (two callers now). Array-or-CSV parsing
   pattern is duplicated for `skip_layers` and `sigmas` (~25 LOC each)
   — if a third caller wants the same shape, worth extracting
   `parse_int_csv_or_array()` / `parse_float_csv_or_array()` helpers.
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
