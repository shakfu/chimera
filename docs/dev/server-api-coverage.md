# Chimera Serve — CLI/HTTP Parity Inventory

Report date: 2026-05-21
Last status update: 2026-05-21 (waves 1–4 + steps 5a–5e landed; full sd LoadParams surface exposed on serve, 30 --sd-* flags + per-request PhotoMaker. Only step 6 (per-request LoRA) remains.)
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
   change. **Closed across four waves on 2026-05-20** (audio wave 1: 9
   fields, audio wave 2: 4 fields, image wave 1: 14 fields, image wave
   2: 13 fields — total 40 fields). **The per-request gap is now
   effectively closed.** Every remaining unsurfaced flag is blocked on
   step 5 (server-init plumbing) because it needs a model loaded at
   startup or compile-time linkage decisions — VAD needs the VAD model,
   ControlNet needs the ControlNet model, PhotoMaker needs the PM
   model, per-request LoRA needs the adapter pool, the `Model` hires
   upscaler needs an upscale model file, and the per-component CPU
   offload / mmap / max-vram knobs are all context-init decisions.
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
| `temperature_inc`, `entropy_thold`, `logprob_thold`, `no_speech_thold` | ✅ Landed 2026-05-20 (audio wave 2) — decoder-fail thresholds; NaN sentinels on `TranscribeRequest` preserve whisper defaults when the field is omitted. `logprob_thold`'s own upstream default is negative, which is why NaN-not-negative is the sentinel scheme. `no_fallback=true` still wins at the engine layer. |

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
| `hires` (+ `hires_upscaler`, `hires_upscale_model`, `hires_width`, `hires_height`, `hires_scale`, `hires_steps`, `hires_denoising_strength`, `hires_upscale_tile_size`) | ✅ Landed 2026-05-20 (image wave 2) — hires-fix two-pass upscale. All upscaler types work; the `Model` upscaler's model file loads per-request from the `hires_upscale_model` path (`sd_hires_params_t.model_path` is per-request, not context-init — the earlier note saying "Model upscaler waits on step 5" was overstated). |
| `control_image` (multipart file) + `control_strength` | ✅ Landed 2026-05-20 (step 5b) — per-request ControlNet conditioning. Gated on the server having been started with `--sd-control-net <path>`; a `control_image` upload without a ControlNet loaded returns HTTP 400. Available on all three image endpoints (`/generations`, `/edits`, `/variations`). |
| `cache_mode`, `cache_option`, `scm_mask`, `scm_policy` | ✅ Landed 2026-05-20 (image wave 2) — inference cache + sampler-cached-memory. Validated via the shared `chimera_sd::parse_cache_options()` so HTTP errors are identical to CLI errors (modulo the `--cache-mode` wording — see design notes). |
| `pm_id_images`, `pm_id_image_set`, `pm_style_strength` | ✅ Landed 2026-05-21 (step 5e) — per-request PhotoMaker. `pm_id_images` is a JSON base64 array (raw or `data:<mime>;base64,` URI); `pm_id_image_set` references a subdirectory of `--sd-pm-id-dir` scanned at server start. Explicit `pm_id_images` wins if both are given. Gated on `--sd-photo-maker <path>` at server start; PM fields without a PM model return HTTP 400. Available on all three image endpoints. |

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
| `temperature_inc`, `entropy_thold`, `logprob_thold`, `no_speech_thold` | same | ✅ Wave 2 | Decoder-fail thresholds. NaN-on-`TranscribeRequest` sentinel scheme because `logprob_thold`'s upstream default is itself negative; `no_fallback=true` still wins. ~8 LOC. |
| `vad` (+ tuning knobs) | `treq.vad` etc. | Blocked on step 5 | VAD is becoming the de-facto default but requires the VAD model loaded at server start (chimera serve doesn't accept `--enable-audio-vad-model` today — needs `LoadParams` routing). |

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
| `cache_mode` + `cache_option` + `scm_mask` + `scm_policy` | parsed via `chimera_sd::parse_cache_options` | ✅ Wave 2 | Validated up-front so a bad mode / unknown key / non-numeric value returns 400 with the parser's own message rather than failing inside generate(). Shared with the CLI. |
| `hires` + `hires_upscaler` + `hires_upscale_model` + `hires_width` / `hires_height` / `hires_scale` / `hires_steps` / `hires_denoising_strength` / `hires_upscale_tile_size` | `req.hires_*` | ✅ Wave 2 | `Latent*` upscalers work out of the box; `hires_upscaler: "Model"` accepts the field but generate() fails downstream until step 5 plumbs an upscale model through `LoadParams` at server start. |
| `control_image` (multipart file) + `control_strength` | `req.control` / `req.control_strength` | ✅ Step 5b | ControlNet model path is server-init (`--sd-control-net`); conditioning image is per-request. The `maybe_attach_control()` helper in `chimera_serve_images.cpp` decodes the multipart file to RGB, returns 400 with a precise message when the server has no ControlNet loaded. |

### Tier 2 — per-request, more thought

- `ref_image` (repeatable multipart) + `increase_ref_index` +
  `no_auto_resize_ref_image` — IP-adapter-style style/identity
  conditioning. JSON-shape decision: OpenAI's body has no precedent
  for repeatable image uploads; chimera would invent the field name.
- ~~PhotoMaker bundle — requires `--photo-maker` model loaded at
  startup. Same multipart-plural shape question as ref-images.~~
  ✅ Landed 2026-05-21 as step 5e. Sidestepped the multipart-plural
  shape question by accepting a **JSON base64 array** (`pm_id_images`)
  for explicit per-request identity, and a **named identity-set**
  shape (`pm_id_image_set: "<name>"`) that references a subdirectory
  of `--sd-pm-id-dir` scanned eagerly at server start. The
  explicit-base64 form wins over the named-set form when both are
  given. See step 5e in the [Recommended order](#recommended-order).
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
3. ~~**Hires-fix** + **Cache/SCM**~~ (~50 LOC). Both landed together as
   **image wave 2** on 2026-05-20. Hires-fix accepts the full 9-field
   bundle; the `Model` upscaler still waits on step 5. Cache/SCM
   reuses the CLI's `parse_cache_options` so HTTP errors are
   byte-identical to CLI errors. The same `fill_common_image_fields()`
   helper picks both bundles up for all three image endpoints.
4. ~~**Audio wave 2** — `temperature_inc`, `entropy_thold`,
   `logprob_thold`, `no_speech_thold`. Four decoder-fail thresholds,
   all NaN-sentineled on `TranscribeRequest`. Mechanical.~~
   ✅ **Landed 2026-05-20** (~8 LOC). VAD knobs were originally
   slotted into wave 2 but split out — they need the VAD model loaded
   at server start (step 5).
5. **Server-init `LoadParams` routing** — structural. The simple
   `load_model(path)` overloads in both `chimera_whisper` and
   `chimera_sd` are bypassed in favor of the `LoadParams` forms, and
   `chimera serve` grows `--audio-*` / `--sd-*` flag families that
   populate them. **The gating step for every Tier-3 item** — once a
   modality's `LoadParams` is plumbed, each subsequent flag is a
   one-line addition. Phased rollout:
   - **5a — Audio `LoadParams` + VAD on serve.** ✅ Landed 2026-05-20.
     Four flags: `--audio-flash-attn`, `--audio-no-gpu`, `--audio-device`,
     `--audio-vad-model`. Per-request `vad=true` (+ six tuning knobs)
     wired in the audio handler with a precise 400 when no VAD model is
     loaded. ~50 LOC across `chimera.h`, `chimera_serve.cpp`,
     `chimera_serve_internal.h`, `chimera_serve_audio.cpp`,
     `chimera_cli/chimera.cpp`.
   - **5b — SD `LoadParams` + ControlNet on serve.** ✅ Landed
     2026-05-20. One flag exposed (`--sd-control-net`) but the full
     `chimera_sd::LoadParams` is now built in `command_serve`, so
     phases 5c/5d become one-line-per-field additions. Per-request
     `control_image` (multipart) + `control_strength` (JSON) wired in
     all three image handlers via the new `maybe_attach_control()`
     helper. Gated 400 when no ControlNet is loaded. ~70 LOC.
   - **5c — SD split-checkpoint flags.** ✅ Landed 2026-05-20. 13
     flags: `--sd-diffusion-model`, `--sd-vae`, `--sd-clip-l`,
     `--sd-clip-g`, `--sd-t5xxl`, `--sd-llm`, `--sd-llm-vision`,
     `--sd-clip-vision`, `--sd-taesd`, `--sd-embd-dir`, `--sd-type`,
     `--sd-tensor-type-rules`, `--sd-high-noise-diffusion-model`.
     `chimera serve --enable-image <path>` is now optional when
     `--sd-diffusion-model <path>` is set — same combined-or-split
     allowance as `chimera sd`. Mechanical landing (~50 LOC) thanks to
     the 5b LoadParams plumbing — each flag was one line in three
     places (`ServeOptions`, the LoadParams build, the CLI binding).
     Unlocks serving Flux, SD3, Z-Image, and Qwen-Image; the existing
     per-request fields from image waves 1+2 (`guidance`, `flow_shift`,
     `img_cfg_scale`, etc.) Just Work for Flux/SD3-class models without
     further wiring.
   - **5d — SD perf/offload long-tail.** ✅ Landed 2026-05-20. 16
     flags: `--sd-fa`, `--sd-diffusion-fa`, `--sd-diffusion-conv-direct`,
     `--sd-vae-conv-direct`, `--sd-no-mmap`, `--sd-max-vram`,
     `--sd-offload-to-cpu`, `--sd-clip-on-cpu`, `--sd-vae-on-cpu`,
     `--sd-control-net-cpu`, `--sd-force-sdxl-vae-conv-scale`,
     `--sd-rng`, `--sd-sampler-rng`, `--sd-prediction`,
     `--sd-lora-apply-mode`, `--sd-threads`. Mechanical landing again
     (~50 LOC); the four enum-string flags (`--sd-rng`,
     `--sd-sampler-rng`, `--sd-prediction`, `--sd-lora-apply-mode`) get
     `CLI::IsMember` validators that exit before model load with the
     accepted-set listed in the error. `--sd-no-mmap` is inverted
     polarity (chimera defaults mmap=on; the flag flips it off) — same
     semantic as the CLI side. After 5d the full `chimera_sd::LoadParams`
     surface is exposed on serve — 30 `--sd-*` flags total across 5b/5c/5d.
   - **5e — PhotoMaker on serve.** ✅ Landed 2026-05-21. Three new
     server-init flags (`--sd-photo-maker <path>`, `--sd-pm-id-dir
     <dir>`, `--sd-pm-id-embed-path <file>`) and a per-request JSON
     bundle (`pm_id_images`, `pm_id_image_set`, `pm_style_strength`).
     The multipart-files-plural design question is sidestepped:
     identity images travel as a **JSON base64 array** rather than
     repeatable multipart parts. Two complementary shapes accepted —
     **(C)** explicit `pm_id_images: ["<base64 or data-URI>", ...]`
     for per-request identity images, and **(E)** `pm_id_image_set:
     "<name>"` referencing a named subdirectory of `--sd-pm-id-dir`
     scanned eagerly at startup. (C) wins if both are supplied —
     explicit per-request beats admin-curated default, same precedence
     as elsewhere in chimera. `--sd-pm-id-embed-path` is the
     server-init default ID-embedding applied to every PM request.
     Gating: any PM field against a server without `--sd-photo-maker`
     returns 400 with the missing-flag hint; `pm_id_image_set` against
     a server without `--sd-pm-id-dir` returns 400; an unknown set
     name returns 400 listing the known names; a malformed base64
     element returns 400 naming the offending index. Available on
     all three image endpoints. ~250 LOC (the bulk is the
     self-contained `base64_decode()` for the JSON array path and the
     `PmIdSetCache` eager-scan at server start).
6. **Per-request LoRA selection.** Pending. Choose between "named
   adapters loaded at startup" (safer) vs. "paths resolved at request
   time" (more flexible, security implications). Mirrors the choice
   llama-server made with its `--lora-adapters` config.

**Honest correction from the original roadmap:** `--sd-upscale-model`
was originally listed in the "unblocking trio" — it isn't actually
blocking anything. `sd_hires_params_t.model_path` is per-request and
loads fresh each `generate_image()` call, so wave 2's
`hires_upscale_model` per-request field already works without
server-init plumbing. Removed from the roadmap.

Steps 1–4 closed the entire per-request gap (40 fields). Steps 5a–5e
closed the server-init gap: audio LoadParams + VAD (5a), sd LoadParams
+ ControlNet (5b), sd split-checkpoint flags (5c, 13 flags), sd
perf/offload long-tail (5d, 16 flags), PhotoMaker (5e, 3 server-init
flags + 3 per-request fields). The full `chimera_sd::LoadParams`
surface is exposed on serve. Remaining work — step 6 (per-request
LoRA) — needs API-shape design (named-adapters-at-startup vs.
paths-resolved-at-request) rather than more mechanical wiring.
