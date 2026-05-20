# Chimera CLI / Upstream CLI Coverage Audit

Report date: 2026-05-20
Upstream versions compared against:
- llama.cpp `b9119` (flag definitions in `common/arg.cpp`, CLI binary in `tools/cli/cli.cpp`)
- whisper.cpp `v1.8.4` (`examples/cli/cli.cpp`)
- stable-diffusion.cpp `master-596-90e87bc` (CLI shell in `examples/cli/main.cpp`, model/gen flags in `examples/common/common.cpp`)

What triggered this audit: `chimera sd` had silently shipped without the entire split-checkpoint flag family (`--diffusion-model`, `--vae`, `--clip-l`, `--t5xxl`, `--llm`, `--offload-to-cpu`, `--diffusion-fa`), which made it impossible to run any Z-Image / Flux / SD3-class model. The hole was invisible because no one had cross-referenced sd's flag surface against ours; this audit closes that blind spot for the four CLI subcommands (`gen`/`chat`, `embed`, `whisper`, `sd`). `chimera serve` (wraps llama-server) is out of scope.

Status legend: ✅ exposed · 🔀 renamed · 🟡 partial · ❌ missing · 🚫 out-of-scope.

> Note: chimera's CLI definitions live in `src/chimera_cli/chimera.cpp` (`bind_*_cmd` helpers) and the option structs in `src/chimera/chimera.h`. Both files are referenced throughout this report.

> **Status update 2026-05-20:** the 20 flag groups identified as Tier 1–4 priorities in the original llama section have all landed on `gen` / `chat` / `embed`. The tables and "Notable gaps" sections below have been edited in-place to reflect this; see the CHANGELOG entry under [Unreleased] for the full list.
>
> **Whisper coverage closer 2026-05-20:** the remaining whisper gaps flagged below — VAD bundle, offset/duration, segment shaping, decoder-fail thresholds, audio-ctx, tinydiarize, token suppression, context-params (`--flash-attn` / `--no-gpu` / `--device`), and `--processors` — all landed on `whisper`. The whisper table rows are flipped to ✅ in-place; the "Notable gaps worth filing" list is now empty save for the documented out-of-scope items.
>
> **sd partial:** skip-layer guidance + the `--high-noise-diffusion-model` model-loading slot landed; the rest of the `--high-noise-*` family stays out of scope (video-only).
>
> **sd coverage closer 2026-05-20 (Rounds 1–8):** 38 additional flags landed — perf/offload (`--fa`, `--no-mmap`, `--max-vram`, per-component CPU offload, SDXL VAE fix), sampler/generation (`--img-cfg-scale`, `--eta`, `--timestep-shift`, `--sigmas`, `--prediction`, `--lora-apply-mode`), model-loading (`--taesd`, `--clip-vision`, `--llm-vision`, `--tensor-type-rules`, `--photo-maker`, `--embd-dir`), PhotoMaker bundle (`--pm-id-images-dir`/`--pm-id-embed-path`/`--pm-style-strength`), reference images (`--ref-image` + supporting flags), the full hires-fix bundle, and the cache/SCM bundle (`--cache-mode`, `--cache-option`, `--scm-mask`, `--scm-policy`). Tables below are flipped to ✅ in-place. **All sd ❌ rows are now resolved**: `--disable-image-metadata` is reclassified 🚫 (moot — chimera's stock `stb_image_write` writes no text chunks, so there's nothing to disable; the inverse "embed metadata" feature is a separate item not yet on the roadmap).

---

## Executive summary

| Subcommand | Upstream flags considered | Exposed | Renamed | Missing (real gap) | Deliberately out of scope |
|---|---:|---:|---:|---:|---:|
| `gen` (llama-cli) | ~80 CLI-relevant | 32 | 1 | ~2 | ~50 |
| `chat` (llama-cli, interactive) | ~85 | 37 | 0 | ~2 | ~55 |
| `embed` (llama-embedding) | ~14 | 12 | 1 | 1 | 1 |
| `whisper` (whisper-cli) | 58 | 41 | 1 | ~3 | ~13 |
| `sd` (sd cli) | 107 | 60 | 6 | 0 | ~51 |

"Real gaps" are flags whose absence we'd consider filing an issue for. "Deliberately out of scope" covers things like llama-cli's REPL plumbing (chimera replaces it with `chat`), perplexity/imatrix/training knobs, anything tied solely to llama-server, and obscure research/debug flags. The next two columns of the per-subcommand tables make each call individually.

The headline finding from the original audit — "**the sd surface is by far the largest source of meaningful gaps**" — no longer applies. After the 2026-05-20 sd closer (Rounds 1–8, 38 additional flags on top of the earlier Tier 1–2 work), **the sd surface has zero unresolved ❌ rows**. The lone remaining item, `--disable-image-metadata`, is reclassified 🚫 because chimera's stock `stb_image_write` writes no text chunks, so there is no metadata to disable; embedding generation params (the reverse direction, for parity with sd-cli's default behaviour) is a separate feature not yet on the roadmap. Everything else is documented out-of-scope (video, standalone modes, shell features, chroma/qwen tuning). The llama coverage is intentionally minimal — chimera leans on its own DSL (`chat` REPL, `serve` HTTP) and the wrapped subcommands are deliberately thin. The whisper surface, previously the most "leaky" relative to size, is now ~73% covered after the 2026-05-20 closer (Batches 1–3 + VAD + offset/duration); residual gaps are constrained-decoding (`--grammar` family) and wrapper-logic features (`--detect-language` exit-after-detect, stereo `--diarize`).

---

## `gen` and `chat` (llama-cli, llama.cpp b9119)

Upstream llama-cli inherits ~330 `common_arg` declarations from `common/arg.cpp`. Roughly 80 are tagged for the CLI context (the rest are server-only, training-only, perplexity-only, etc.). Chimera deliberately exposes only a thin generation slice and trusts upstream defaults for the rest; this is consistent with the project's framing as a thin C++ shell, so the size of the "missing" column below is *expected* — what matters is whether the missing ones meaningfully constrain users.

### Coverage table — generation core (applies to both `gen` and `chat`)

| Upstream flag | Chimera equivalent | Status | Notes |
|---|---|---|---|
| `--model, -m` | `-m,--model` | ✅ | Required for `gen`, soft-required for `chat`. |
| `--prompt, -p` | `-p,--prompt` (gen only) | ✅ | `chat` uses interactive input instead. |
| `--prompt-file, -f` / `--file` | `-f,--prompt-file` | ✅ | Stdin via `-` supported. |
| `--predict, -n / --n-predict` | `-n,--n-predict` | ✅ | |
| `--ctx-size, -c` | `-c,--ctx-size` | ✅ | |
| `--batch-size, -b` | `-b,--batch-size` | ✅ | |
| `--ubatch-size` | `--ubatch-size` | ✅ | Landed 2026-05-20. |
| `--threads, -t` | `-t,--threads` | ✅ | |
| `--threads-batch` | — | ❌ | Rare in practice. |
| `--seed` | `--seed` | ✅ | |
| `--temp` | `--temp` | ✅ | |
| `--top-k` | `--top-k` | ✅ | |
| `--top-p` | `--top-p` | ✅ | |
| `--min-p` | `--min-p` | ✅ | |
| `--repeat-penalty` | `--repeat-penalty` | ✅ | |
| `--repeat-last-n` | `--repeat-last-n` | ✅ | Landed 2026-05-20. |
| `--presence-penalty` / `--frequency-penalty` | `--presence-penalty` / `--frequency-penalty` | ✅ | Landed 2026-05-20. |
| `--typical` | — | ❌ | |
| `--top-nsigma` | — | ❌ | New-ish; nice-to-have. |
| `--xtc-probability` / `--xtc-threshold` | — | ❌ | Newer sampler; low priority. |
| `--dry-*` (multiplier/base/allowed-length/penalty-last-n/sequence-breaker) | `--dry-multiplier` / `--dry-base` / `--dry-allowed-length` / `--dry-penalty-last-n` / `--dry-sequence-breaker` | ✅ | Landed 2026-05-20. Sequence-breaker is repeatable. |
| `--mirostat` / `--mirostat-ent` / `--mirostat-lr` | same | ✅ | Landed 2026-05-20. |
| `--samplers` / `--sampler-seq` | — | ❌ | Sampler-chain ordering. Power-user, low priority. |
| `--dynatemp-range` / `--dynatemp-exp` | — | ❌ | |
| `--logit-bias` | `--logit-bias` | ✅ | Landed 2026-05-20. Repeatable, format `"<id>(+|-|=)<bias>"`. |
| `--ignore-eos` | `--ignore-eos` | ✅ | Landed 2026-05-20. |
| `--grammar` / `--grammar-file` / `--json-schema` / `--json-schema-file` | same | ✅ | Landed 2026-05-20. JSON schema converted via `json_schema_to_grammar`. Mutually exclusive group. End-to-end smoke verified. |
| `--flash-attn` | `--flash-attn` | ✅ | Landed 2026-05-20. Available on gen/chat/embed. |
| `--mmap` / `--mlock` | `--no-mmap` / `--mlock` | ✅ | Landed 2026-05-20. `use_mmap` default stays true; `--no-mmap` to opt out. |
| `--gpu-layers` | `--gpu-layers` | ✅ | |
| `--main-gpu` / `--tensor-split` / `--split-mode` | same | ✅ | Landed 2026-05-20. `--split-mode` accepts none/layer/row/tensor; `--tensor-split` parses comma-separated floats. |
| `--device` / `--list-devices` | `--device` only | 🟡 | `--device` landed 2026-05-20 (comma-separated device list). `--list-devices` skipped — better fit as a `chimera info` extension. |
| `--n-cpu-moe` / `--cpu-moe` | — | ❌ | MoE-specific offload. |
| `--override-tensor` / `--override-kv` | — | ❌ | Power-user. |
| `--cache-type-k` / `--cache-type-v` | same | ✅ | Landed 2026-05-20. Accepts f32/f16/bf16/q8_0/q5_0/q5_1/q4_0/q4_1/iq4_nl. End-to-end smoke verified. |
| `--rope-freq-base` / `--rope-freq-scale` / `--rope-scaling` / `--rope-scale` | same | ✅ | Landed 2026-05-20. `--rope-scaling` accepts none/linear/yarn/longrope. |
| `--yarn-*` (orig-ctx, ext-factor, attn-factor, beta-fast, beta-slow) | same | ✅ | Landed 2026-05-20. |
| `--lora` / `--lora-scaled` | `--lora <path[:scale]>` | ✅ | Landed 2026-05-20. Repeatable. Reuses the `serve`-side `path[:scale]` parser. Closes the asymmetry. |
| `--mmproj` | `--mmproj` | ✅ | |
| `--mmproj-offload` / `--mmproj-auto` / `--mmproj-url` | `--no-mmproj-offload` | 🟡 | `--no-mmproj-offload` landed 2026-05-20 (maps to `mtmd_context_params.use_gpu`). `--mmproj-auto` not modeled by upstream at b9119; `--mmproj-url` is network-fetch (out of scope). |
| `--image` | `--image` (gen; repeatable) | ✅ | `chat` injects images via `/image` REPL command. |
| `--image-min-tokens` / `--image-max-tokens` | — | ❌ | Vision-token budget. |
| `--system-prompt` | `--system` (chat) | 🔀 | Renamed; `gen` lacks it (it's an interactive concept). |
| `--system-prompt-file` | `--system-prompt-file` (chat) | ✅ | |
| `--chat-template` | `--chat-template` (chat) | ✅ | |
| `--chat-template-file` / `--chat-template-kwargs` | same | ✅ | Landed 2026-05-20 (`chat` only). `--chat-template-file` is mutually exclusive with `--chat-template`. `--chat-template-kwargs` is repeatable. |
| `--jinja` | `--no-jinja` | ✅ | Landed 2026-05-20 (`chat` only). Jinja defaults ON; `--no-jinja` opts out. |
| `--reasoning` / `--reasoning-budget` / `--reasoning-format` / `--reasoning-budget-message` | same | ✅ | Landed 2026-05-20 (`chat` only). `--reasoning-budget` enforcement landed in a follow-up the same day — `command_chat` probes the template via a dummy `common_chat_templates_apply` to read `thinking_{start,end}_tag`, tokenizes via `common_tokenize(parse_special=true)`, populates `sampling.reasoning_budget_{tokens,start,end,forced}`, and `common_sampler_init` chains the budget sampler into the chain. `--reasoning-budget-message` is tokenized into the forced-termination sequence as `<message> + <end_tag>` (mirrors `llama-cli`). When the active template has no thinking tags, a warning fires and the budget is silently ignored. |
| `--keep` | — | ❌ | Token retention across context overflow. |
| `--color` | `--color` (chat) | ✅ | `gen` is non-interactive so this is fine. |
| `--verbose-prompt` / `--special` / `--escape` / `--no-context-shift` | — | 🚫 | Debug/edge; out-of-scope. |
| `--prompt-cache` / `--prompt-cache-all` / `--prompt-cache-ro` | — | 🚫 | Tied to llama-cli's prompt-cache on-disk format; nicer to layer above. |
| `--ctx-checkpoints` / `--checkpoint-every-n-tokens` | — | ❌ | New mid-2025 feature; low priority. |
| `--swa-full` | — | ❌ | Sliding-window attention. |
| `--cache-ram` | — | ❌ | |
| `--n-predict` shorthand `-n` | covered | ✅ | |
| `--single-turn` / `--interactive` / `--interactive-first` / `--in-prefix` / `--in-prefix-bos` / `--in-suffix` / `--reverse-prompt` / `--multiline-input` / `--conversation` / `--display-prompt` / `--simple-io` / `--print-token-count` | — | 🚫 | Upstream's interactive REPL; chimera replaces with its own `chat` + linenoise. Do not port. |
| `--no-warmup` | — | 🚫 | |
| `--hf-repo` / `--hf-file` / `--hf-token` / `--model-url` / `--offline` / `--docker-repo` | — | 🚫 | Network model fetch; chimera assumes the user supplies a local path. |
| `--cpu-mask*` / `--cpu-range*` / `--cpu-strict*` / `--prio*` / `--poll*` | — | 🚫 | Thread-affinity knobs; specialist usage. |
| `--draft*` / `--spec-*` (~30 flags) | — | 🚫 | Speculative decoding. Out of scope until chimera grows a draft-model story. |
| `--control-vector*` | — | ❌ | Activation steering. Cool but low demand. |
| `--diffusion-*` (algorithm/steps/eps/etc.) | — | 🚫 | llama.cpp diffusion LM support; not the same thing as `chimera sd`. |
| `--hellaswag*` / `--winogrande*` / `--multiple-choice*` / `--ppl*` / `--kl-divergence` / `--perplexity*` | — | 🚫 | Eval-only. |
| `--logits-output-dir` / `--save-logits` / `--save-all-logits` | — | 🚫 | |
| `--epochs` / `--learning-rate*` / `--optimizer` / `--weight-decay` / `--method` / `--pca-*` | — | 🚫 | Training/fine-tune. |
| `--license` / `--version` / `--help` / `--completion-bash` / `--list-devices` | partial via top-level chimera | ✅/❌ | `--version` via `chimera -V`; `--list-devices` could be a nice `chimera info` follow-up. |
| `--log-file` / `--log-disable` / `--log-colors` / `--log-prefix` / `--log-timestamps` / `--verbosity` | partial via top-level `-v` | 🟡 | Chimera has a single `-v/--verbose`; finer-grained log control isn't exposed. |
| `--no-host` / `--api-key*` / `--api-prefix` | — | 🚫 | Server-mode only. |

### `chat`-only persistence / DB surface (chimera-specific, no upstream equivalent)

| Chimera flag | Status | Notes |
|---|---|---|
| `--persist`, `--resume`, `--list`, `--search`, `--list-limit`, `--db` | ✅ chimera-native | Backed by the embedded SQLite tables. No equivalent in llama-cli. |

### Notable gaps worth filing

All five priorities from the original audit landed on 2026-05-20 (`--flash-attn`, grammar/json-schema, DRY + repeat-last-n, `--lora` in gen/chat, reasoning family). Residual items:

1. ~~**`--reasoning-budget` enforcement**.~~ ✅ Landed 2026-05-20. The earlier "needs `chat_sample_loop` restructure" comment turned out to be wrong on closer reading: `common_sampler_init` itself chains `common_reasoning_budget_init` into the sampler whenever the `sampling.reasoning_budget_{tokens,start,end,forced}` fields are populated, so the integration is entirely upstream of `common_sampler_init` — no sample-loop changes needed. Implementation: `command_chat` probes the active chat template once at startup via a dummy `common_chat_templates_apply`, reads `thinking_{start,end}_tag`, tokenizes with `parse_special=true`, and stuffs the result into the sampling params before `make_sampler`. Forced-termination sequence = `--reasoning-budget-message + thinking_end_tag`. Templates without thinking tags warn and ignore the budget.
2. ~~**`--list-devices`**~~ ✅ Landed 2026-05-20 as `chimera info --list-devices`.
3. **`--mmproj-auto`** — not modeled by `mtmd_context_params` at llama.cpp `b9119`. Revisit on next pin bump.

### Deliberately omitted (do not re-flag)

- Anything under "interactive REPL" or "prompt cache on disk" — chimera owns its own REPL via `chat` + linenoise and persists via SQLite.
- HuggingFace/docker/network model fetch — chimera takes local paths.
- Speculative-decoding and draft-model flags — out of scope until chimera adds a draft-model wrapper.
- All training / perplexity / hellaswag / imatrix / cvector / pca / optimizer flags.
- CPU mask / affinity / strict / poll / prio knobs (specialist usage).
- llama.cpp's `--diffusion-*` flags — refer to diffusion-LMs, not stable-diffusion.cpp.

---

## `embed` (llama-embedding family, llama.cpp b9119)

`llama-embedding` was retired as a standalone binary in current llama.cpp — the same flag set is now available via `llama-cli` with `--embedding`. Coverage here is excellent because the surface is small.

| Upstream flag | Chimera equivalent | Status | Notes |
|---|---|---|---|
| `--model, -m` | `-m,--model` | ✅ | |
| `--prompt, -p` / file via `-f` | `-p,--prompt` / `-f,--prompt-file` | ✅ | Stdin via `-`. |
| `--embedding` / `--embeddings` | implicit (subcommand intent) | ✅ | Chimera dispatches embed mode automatically. |
| `--pooling {none,mean,cls,last,rank}` | `--pooling` | ✅ | `rank` (reranker) landed 2026-05-20 — `LLAMA_POOLING_TYPE_RANK` is now accepted alongside `mean\|cls\|last\|none`. |
| `--embd-normalize N` (-1 / 0 / 1 / 2 / >2) | `--no-normalize` flag | 🔀 | Chimera reduces to a boolean (L2 or off). Loses access to taxicab/p-norm. Acceptable simplification; document the choice. |
| `--embd-output-format` | `--embd-output-format` | ✅ | Landed 2026-05-20. Values: `''` (default; space-separated, preserves prior output), `array`, `json` (OpenAI envelope), `raw`. `json+` (cosine-similarity matrix add-on) not implemented. |
| `--embd-separator` | `--embd-separator` | ✅ | Landed 2026-05-20. Literal-string splitter (no regex); emits one vector per piece. |
| `--ctx-size, -c` | `-c,--ctx-size` | ✅ | |
| `--batch-size, -b` | `-b,--batch-size` | ✅ | |
| `--threads, -t` | `-t,--threads` | ✅ | |
| `--gpu-layers` | `--gpu-layers` | ✅ | |
| `--attention {causal,non-causal}` | `--attention` | ✅ | Landed 2026-05-20. Pins `llama_context_params.attention_type`; empty leaves the model default. |
| `--flash-attn` | `--flash-attn` | ✅ | Landed 2026-05-20. |
| `--cls-separator` | — | 🚫 | Eval/retrieval-specific. |
| `--chunk` / `--chunks` / `--chunk-size` / `--chunk-separator` | — | 🚫 | Belongs to chimera's own `index`/`search` layer, not the model invocation. |
| `--output-format` (general) | — | 🚫 | See `--embd-output-format`. |

### Chimera-specific extensions (no upstream)

- `--cache-embeddings` / `--cache-db` — SQLite memoization layer. No upstream analogue.
- `-o,--output` — chimera writes to a file/stdout instead of `embedding.txt` style upstream behavior. Cleaner.

### Notable gaps worth filing

1. ~~`--embd-output-format`~~ ✅ Landed 2026-05-20.
2. ~~`--embd-separator`~~ ✅ Landed 2026-05-20.
3. ~~`--attention causal|non-causal`~~ ✅ Landed 2026-05-20.
4. ~~Pooling `rank` value~~ ✅ Landed 2026-05-20.

### Also landed 2026-05-20 (carried over from the llama-shared option set)

`embed` picked up `--flash-attn`, `--ubatch-size`, `--no-mmap`, `--mlock`, `--main-gpu`, `--tensor-split`, `--split-mode`, `--device`, and the full RoPE / YaRN family (`--rope-freq-base`, `--rope-freq-scale`, `--rope-scale`, `--rope-scaling`, `--yarn-orig-ctx`, `--yarn-ext-factor`, `--yarn-attn-factor`, `--yarn-beta-fast`, `--yarn-beta-slow`). These aren't part of `llama-embedding`'s historic surface but are useful for embedding models on long-context fine-tunes / multi-GPU.

### Deliberately omitted

- All chunking flags (`--chunk*`) — chimera handles chunking at the `index`/`search` layer.
- `--cls-separator` and other retrieval-helper flags — same reasoning.

---

## `whisper` (whisper.cpp v1.8.4)

whisper-cli has a flat ~58-flag surface. Chimera exposes 5 of them. The result is a deliberately minimal wrapper, but several gaps are unforced — particularly around output formats and VAD.

| Upstream flag (short) | Chimera | Status | Notes |
|---|---|---|---|
| `-m / --model` | `-m,--model` | ✅ | |
| `-f / --file` | `-i,--input` | 🔀 | Renamed; upstream supports repeating; chimera takes one. |
| `-t / --threads` | `-t,--threads` | ✅ | |
| `-p / --processors` | `--processors` | ✅ | Landed 2026-05-20. >1 routes through `whisper_full_parallel`; default 1 keeps the serial path. |
| `-l / --language` | `-l,--language` | ✅ | |
| `-dl / --detect-language` | — | ❌ | Useful as exit-after-detect mode. |
| `-tr / --translate` | `--translate` | ✅ | |
| `--prompt` | `--prompt` | ✅ | Landed 2026-05-20. Initial-prompt biasing (`whisper_full_params.initial_prompt`). |
| `--carry-initial-prompt` | `--carry-initial-prompt` | ✅ | Landed 2026-05-20. |
| `-bs / --beam-size` | `--beam-size` | ✅ | Landed 2026-05-20. Sets `WHISPER_SAMPLING_BEAM_SEARCH` when N>0. |
| `-bo / --best-of` | `--best-of` | ✅ | Landed 2026-05-20. |
| `-tp / --temperature` | `--temperature` | ✅ | Landed 2026-05-20. |
| `-tpi / --temperature-inc` | `--temperature-inc` | ✅ | Landed 2026-05-20. NaN sentinel (not negative) because the field's upstream default is positive but `logprob_thold`'s isn't; same scheme across the four fallback knobs. `--no-fallback` still wins. |
| `-nf / --no-fallback` | `--no-fallback` | ✅ | Landed 2026-05-20. Sets `temperature_inc<0`. |
| `-mc / --max-context` | — | ❌ | |
| `-ml / --max-len` | `--max-len` | ✅ | Landed 2026-05-20. 0 = unlimited (whisper default). Pairs with `--output-srt` / `--output-vtt`. |
| `-sow / --split-on-word` | `--split-on-word` | ✅ | Landed 2026-05-20. Only takes effect when `--max-len > 0`. |
| `-wt / --word-thold` | — | ❌ | |
| `-et / --entropy-thold` / `-lpt / --logprob-thold` / `-nth / --no-speech-thold` | `--entropy-thold` / `--logprob-thold` / `--no-speech-thold` | ✅ | Landed 2026-05-20. NaN sentinel leaves the upstream default (necessary because `logprob_thold` defaults to a negative value). |
| `-ot / --offset-t` / `-on / --offset-n` / `-d / --duration` | `--offset` / `--duration` | 🟡 | Landed 2026-05-20 for the ms-based pair (`-ot` / `-d`). `-on` (sample-offset) is not exposed by `whisper_full_params` — it's internal to whisper-cli's WAV reader, so deliberately skipped. |
| `-ac / --audio-ctx` | `--audio-ctx` | ✅ | Landed 2026-05-20. 0 = model default; common tweak for tiny.en. |
| `-fa / --flash-attn` / `-nfa / --no-flash-attn` | `--flash-attn` | 🟡 | Landed 2026-05-20 as `--flash-attn`. `--no-flash-attn` is redundant (default is off) so not added. |
| `-ng / --no-gpu` | `--no-gpu` | ✅ | Landed 2026-05-20. Inverts whisper's default `use_gpu=true`. |
| `-dev / --device` | `--device` | ✅ | Landed 2026-05-20. Single CUDA device index (whisper's `gpu_device` field). Not the comma-separated list shape used by llama-side `--device`. |
| `-di / --diarize` | — | ❌ | Stereo diarization — implemented in whisper-cli as wrapper logic, not a `whisper_full_params` field. Out of scope for now. |
| `-tdrz / --tinydiarize` | `--tinydiarize` | ✅ | Landed 2026-05-20. Requires a tdrz-trained model; silently ignored on others. |
| `-otxt / -ovtt / -osrt / -ocsv / -olrc / -oj / -ojf` | `--output-txt` / `--output-vtt` / `--output-srt` / `--output-csv` / `--output-lrc` / `--output-json` / `--output-json-full` | ✅ | Landed 2026-05-20. CLI11 rejects multi-char short flags, so long-only here (no `-osrt` aliases). All combinable; segment-level timestamps auto-enabled when any format is requested. |
| `-owts` | — | 🚫 | Karaoke video script; depends on font/ffmpeg toolchain. |
| `-of / --output-file` | `--output-file` | ✅ | Landed 2026-05-20. Base name; defaults to input WAV's stem. Each enabled format writes `<base>.<ext>`. |
| `-fp / --font-path` | — | 🚫 | Karaoke-only. |
| `--timestamps` (chimera) ↔ `-nt / --no-timestamps` | `--timestamps` flag | 🔀 | Inverted polarity vs upstream default. Document this; don't change. |
| `--no-context` | `--no-context` | ✅ | |
| `--vad` | `--vad` | ✅ | Landed 2026-05-20. Requires `--vad-model`; chimera fails with `BadInput` if the toggle is set without the model path. |
| `--vad-model` / `--vad-threshold` / `--vad-min-speech-duration-ms` / `--vad-min-silence-duration-ms` / `--vad-max-speech-duration-s` / `--vad-speech-pad-ms` / `--vad-samples-overlap` | same | ✅ | Landed 2026-05-20. Numeric knobs inherit `whisper_vad_default_params()` when unset (negative-one sentinels). |
| `-sns / --suppress-nst` / `--suppress-regex` | `--suppress-nst` / `--suppress-regex` | ✅ | Landed 2026-05-20. Regex is matched against token strings; empty string leaves the default. |
| `--grammar` / `--grammar-rule` / `--grammar-penalty` | — | ❌ | Constrained decoding. |
| `-dtw / --dtw` | — | ❌ | Token-level timestamps. |
| `-oved / --ov-e-device` | — | 🚫 | OpenVINO-only. |
| `-debug / --debug-mode` / `-np / --no-prints` / `-ps / --print-special` / `-pc / --print-colors` / `--print-confidence` / `-pp / --print-progress` / `-ls / --log-score` | — | 🚫 | Debug / logging cosmetics; chimera owns its own logging. |

### Notable gaps worth filing

1. ~~**Output-format family** (`-osrt/-ovtt/-oj/-ojf/-ocsv/-olrc`).~~ ✅ Landed 2026-05-20.
2. ~~**VAD bundle** (`--vad` + the seven knobs).~~ ✅ Landed 2026-05-20. `--vad` requires `--vad-model`; tuning knobs use `-1` sentinels to inherit `whisper_vad_default_params()`.
3. ~~**`--prompt` / `--carry-initial-prompt`**.~~ ✅ Landed 2026-05-20.
4. ~~**Decoding strategy** (`--beam-size`, `--best-of`, `--temperature`, `--no-fallback`).~~ ✅ Landed 2026-05-20.
5. ~~**Offset/duration** (`-ot`, `-d`).~~ ✅ Landed 2026-05-20 as `--offset` / `--duration` (ms-based). `-on` is internal to whisper-cli's WAV reader and not exposed by `whisper_full_params`, so deliberately skipped.
6. ~~**Segment shaping + decoder thresholds + audio-ctx + tinydiarize + suppression + flash-attn/no-gpu/device + processors.**~~ ✅ Landed 2026-05-20 as Batches 1–3 of the whisper closer (see CHANGELOG).

Remaining out-of-scope or deferred (do not re-flag): `--grammar` family (needs the same grammar parser as `gen`/`chat` — bigger lift, low demand for audio), `--detect-language` exit-after-detect mode (whisper-cli wrapper logic, not a `whisper_full_params` field), stereo `--diarize` (same — wrapper logic over two-channel WAVs), `--dtw` token-level DTW (niche), `-wt / --word-thold` (we already emit per-word timing in `--output-json-full`), OpenVINO device selection.

### Deliberately omitted

- Karaoke / `--font-path` plumbing.
- OpenVINO device selection (`-oved`).
- All debug-print toggles — chimera has its own log control.
- `-dtw` (token-level DTW) — niche.

---

## `sd` (stable-diffusion.cpp master-596-90e87bc)

Even after closing the Z-Image/Flux/SD3 model-loading gap, sd remains the largest source of meaningful drift. `examples/common/common.cpp` declares 107 unique long flags across model loading, perf, sampler, generation, and hires/video extensions.

### Coverage table — model loading

| Upstream flag | Chimera | Status | Notes |
|---|---|---|---|
| `--model, -m` | `-m,--model` | ✅ | |
| `--diffusion-model` | `--diffusion-model` | ✅ | Landed in the audit that prompted this report. |
| `--high-noise-diffusion-model` | `--high-noise-diffusion-model` | ✅ | Landed 2026-05-20. Model-loading slot only; the full `--high-noise-*` sampler family is video-only and stays out of scope (chimera-sd is img_gen-only). |
| `--vae` | `--vae` | ✅ | |
| `--taesd` / `--tae` | `--taesd` | ✅ | Landed 2026-05-20. TAESD fast preview decode. Single `--taesd` (no `--tae` alias). |
| `--clip_l` | `--clip-l` | 🔀 | **Naming drift.** Upstream uses underscore; chimera uses kebab. Stay with kebab in chimera (project convention) but document. |
| `--clip_g` | `--clip-g` | 🔀 | Landed 2026-05-20. Naming drift (kebab vs underscore) tracked above. |
| `--clip_vision` | `--clip-vision` | 🔀 | Landed 2026-05-20. Kebab-cased per chimera convention. |
| `--t5xxl` | `--t5xxl` | ✅ | |
| `--llm` | `--llm` | ✅ | Z-Image text encoder. |
| `--llm_vision` / `--qwen2vl` / `--qwen2vl_vision` | `--llm-vision` (others 🚫) | 🟡 | `--llm-vision` landed 2026-05-20 (kebab). `--qwen2vl` is a deprecated alias of `--llm`; safe to skip. `--qwen2vl_vision` not modeled here. |
| `--control-net` | `--control-net` | ✅ | Landed 2026-05-20. Wired into `sd_ctx_params_t.control_net_path`. `--control-image` requires this. |
| `--embd-dir` | `--embd-dir` | ✅ | Landed 2026-05-20. Non-recursive scan for `.gguf`/`.safetensors`/`.pt`; filename stem becomes the prompt token. Validated before `new_sd_ctx` (non-directory exits with `BadInput`). Pointer-lifetime detail: the kv vector owns the strings, the `sd_embedding_t` vector borrows from it and is built only after the kv vector is fully sized to avoid realloc-induced pointer dangle. |
| `--lora-model-dir` | `--lora-model-dir` | ✅ | Landed 2026-05-20. Base directory used to resolve relative `--lora` paths (chimera-side; sd.cpp's C API takes resolved paths in `sd_lora_t`). |
| `--photo-maker` | `--photo-maker` | ✅ | Landed 2026-05-20. Model path only; paired with the PhotoMaker generation bundle below. |
| `--upscale-model` / `--hires-upscalers-dir` | `--upscale-model` (hires-upscalers-dir 🚫) | 🟡 | `--upscale-model` landed 2026-05-20 (sd_hires_params_t.model_path, used with `--hires-upscaler Model`). `--hires-upscalers-dir` is sd-cli-shell-only directory scan — out of scope. |
| `--tensor-type-rules` | `--tensor-type-rules` | ✅ | Landed 2026-05-20. Per-tensor wtype override. |
| `--type` | `--type` | ✅ | Landed 2026-05-20. Maps to `sd_ctx_params_t.wtype` via `str_to_sd_type`; unknown values exit with `BadInput`. |

### Coverage table — perf / offload

| Upstream flag | Chimera | Status | Notes |
|---|---|---|---|
| `--threads` | `-t,--threads` | ✅ | |
| `--offload-to-cpu` | `--offload-to-cpu` | ✅ | Landed in audit. |
| `--max-vram` | `--max-vram` | ✅ | Landed 2026-05-20. Soft VRAM cap in GiB; `0` leaves the upstream default. |
| `--mmap` | `--no-mmap` | 🔀 | Landed 2026-05-20 with inverted polarity. Chimera defaults `enable_mmap=true` (sd's upstream default is off), so `--no-mmap` is the opt-out — mirrors the llama-side flag. |
| `--fa` | `--fa` | ✅ | Landed 2026-05-20. Global flash-attn (sd_ctx_params_t.flash_attn); distinct from `--diffusion-fa` which only flips the diffusion path. |
| `--diffusion-fa` | `--diffusion-fa` | ✅ | Landed in audit. |
| `--diffusion-conv-direct` / `--vae-conv-direct` | same | ✅ | Landed 2026-05-20. Map directly to `sd_ctx_params_t.{diffusion,vae}_conv_direct`. |
| `--clip-on-cpu` / `--vae-on-cpu` / `--control-net-cpu` | same | ✅ | Landed 2026-05-20. Per-component CPU offload — more surgical than `--offload-to-cpu`. |
| `--force-sdxl-vae-conv-scale` | `--force-sdxl-vae-conv-scale` | ✅ | Landed 2026-05-20. SDXL VAE conv-scale numerics fix. |

### Coverage table — sampler / scheduler / generation core

| Upstream flag | Chimera | Status | Notes |
|---|---|---|---|
| `--prompt, -p` | `-p,--prompt` | ✅ | |
| `--negative-prompt` | `--negative-prompt` | ✅ | |
| `--width / -W` | `-W,--width` | ✅ | |
| `--height / -H` | `-H,--height` | ✅ | |
| `--steps` | `-s,--steps` | ✅ | |
| `--batch-count` | `-b,--batch-count` | ✅ | |
| `--seed` | `--seed` | ✅ | |
| `--cfg-scale` | `--cfg-scale` | ✅ | |
| `--img-cfg-scale` | `--img-cfg-scale` | ✅ | Landed 2026-05-20. Sentinel `-1` leaves the upstream INFINITY default so sd falls back to `--cfg-scale`. |
| `--guidance` | `--guidance` | ✅ | Landed 2026-05-20. Maps to `sd_sample_params_t.guidance.distilled_guidance`; `-1` sentinel leaves upstream default. |
| `--clip-skip` | `--clip-skip` | ✅ | |
| `--sampling-method` | `--sample-method` | 🔀 | Naming drift (`sampling` vs `sample`). Document. |
| `--scheduler` | `--scheduler` | ✅ | |
| `--sigmas` | `--sigmas` | ✅ | Landed 2026-05-20. Comma-separated float list (e.g. `"14.6,10.0,5.0,1.0"`); non-float entries exit with `BadInput`; the parsed `std::vector<float>` is borrowed into `sd_sample_params_t.custom_sigmas` for the duration of `generate_image`. |
| `--rng` / `--sampler-rng` | same | ✅ | Landed 2026-05-20. Resolved via `str_to_rng_type`; `--sampler-rng cpu` matches ComfyUI seeds. |
| `--prediction` | `--prediction` | ✅ | Landed 2026-05-20. Enum string resolved via `str_to_prediction`: `eps`/`v`/`edm_v`/`flow`/`flux_flow`/`flux2_flow`. CLI11-validated. |
| `--eta` | `--eta` | ✅ | Landed 2026-05-20. DDIM-style stochasticity in `[0,1]`; sentinel `-1` leaves the upstream INFINITY default. |
| `--flow-shift` | `--flow-shift` | ✅ | Landed 2026-05-20. Maps to `sd_sample_params_t.flow_shift`. |
| `--timestep-shift` | `--timestep-shift` | ✅ | Landed 2026-05-20. Maps to `sd_sample_params_t.shifted_timestep`; `0` = no shift (upstream default). |
| `--moe-boundary` | — | ❌ | High-noise/low-noise MoE boundary. |
| `--slg-scale` / `--skip-layer-start` / `--skip-layer-end` / `--skip-layers` | same | ✅ | Landed 2026-05-20. `--skip-layers` parses a comma-separated int list into `sd_slg_params_t.layers`; empty disables SLG regardless of the other knobs; non-integer tokens fail with `BadInput`. Scalars use `-1.0f` sentinels. |
| `--high-noise-*` (cfg-scale, img-cfg-scale, guidance, slg-scale, skip-layer-start/end, eta, sampling-method, skip-layers, steps) | — | ❌ | Entire high-noise group missing (pairs with `--high-noise-diffusion-model`). |

### Coverage table — img2img / inpaint / control

| Upstream flag | Chimera | Status | Notes |
|---|---|---|---|
| `--init-img` | `--init-image` | 🔀 | Naming. |
| `--end-img` | — | ❌ | End-frame for img-to-img blending / video. |
| `--mask` | `--mask-image` | 🔀 | Naming. |
| `--control-image` | `--control-image` | ✅ | Landed 2026-05-20. Requires `--control-net`. Dimensions must match `-W`/`-H`. |
| `--control-strength` | `--control-strength` | ✅ | Landed 2026-05-20. Default 0.9; only used with `--control-image`. |
| `--control-video` | — | 🚫 | Video-only; chimera-sd is image-only today. |
| `--strength` | `--strength` | ✅ | |
| `--ref-image` | `--ref-image` | ✅ | Landed 2026-05-20. Repeatable; each entry is decoded to RGB and borrowed into `sd_img_gen_params_t.ref_images`. Companion flags `--increase-ref-index` and `--no-auto-resize-ref-image` also landed (chimera inverts sd's auto-resize default-on into an opt-out). |
| `--pm-id-images-dir` / `--pm-id-embed-path` / `--pm-style-strength` | same | ✅ | Landed 2026-05-20. `--pm-id-images-dir` scans the directory non-recursively in alphabetical order; non-image entries are skipped, an empty result is `BadInput`. Decoded images are borrowed into `sd_pm_params_t.id_images`. |

### Coverage table — hires fix / VAE tiling

| Upstream flag | Chimera | Status | Notes |
|---|---|---|---|
| `--hires` | `--hires` | ✅ | Landed 2026-05-20. Toggles `sd_hires_params_t.enabled`. |
| `--hires-upscaler` / `--hires-width` / `--hires-height` / `--hires-steps` / `--hires-scale` / `--hires-denoising-strength` / `--hires-upscale-tile-size` | same | ✅ | Landed 2026-05-20. `--hires-upscaler` is the enum-string match against `hires_upscaler_to_str` (`None`/`Latent`/`Latent (nearest)`/`Latent (nearest-exact)`/`Latent (antialiased)`/`Latent (bicubic)`/`Latent (bicubic antialiased)`/`Lanczos`/`Nearest`/`Model`); values with spaces must be quoted at the shell. Scalar sentinels (`0` for ints, `-1` for floats) leave `sd_hires_params_init`'s defaults (scale=2.0, denoising=0.7, tile=128) untouched. `--upscale-model` (table above) provides the file path for `--hires-upscaler Model`. |
| `--vae-tiling` | `--vae-tiling` | ✅ | Landed 2026-05-20. Enables `sd_img_gen_params_t.vae_tiling_params.enabled`. |
| `--vae-tile-size` / `--vae-relative-tile-size` / `--vae-tile-overlap` | same | ✅ | Landed 2026-05-20. Sentinels (`-1`) leave the upstream default; otherwise applied symmetrically to both axes. |
| `--upscale-repeats` / `--upscale-tile-size` | — | ❌ | Standalone upscale mode. |

### Coverage table — video / advanced / output

| Upstream flag | Chimera | Status | Notes |
|---|---|---|---|
| `--video-frames` / `--fps` | — | 🚫 | Video mode out of scope for chimera-sd today (sd-cli has `vid_gen` mode). |
| `--vace-strength` / `--increase-ref-index` / `--disable-auto-resize-ref-image` | — | 🚫 | Video / VACE. |
| `--cache-mode` / `--cache-option` | same | ✅ | Landed 2026-05-20. Mirrors sd-cli's exact surface — `--cache-mode` picks the algorithm (disabled/easycache/ucache/dbcache/taylorseer/cache-dit/spectrum), `--cache-option` overrides per-mode tunables via `key=value,...` (15 keys with per-mode branching: threshold/start/end/decay/relative/reset/Fn/Bn/warmup/w/m/lam/window/flex/stop). Validated in `command_sd` before `load_model` via the chimera-side `parse_cache_options()` helper so typos exit fast. |
| `--scm-mask` / `--scm-policy` | same | ✅ | Landed 2026-05-20. `--scm-mask` borrows into `sd_cache_params_t.scm_mask` for the duration of generate; `--scm-policy` is `static` or `dynamic` (empty = sd's default dynamic). |
| `--lora-apply-mode` | `--lora-apply-mode` | ✅ | Landed 2026-05-20. Enum string via `str_to_lora_apply_mode`: `auto`/`immediately`/`at_runtime`. CLI11-validated. |
| `--circular` / `--circularx` / `--circulary` | — | 🚫 | Seamless-tile output; niche. |
| `--chroma-t5-mask-pad` / `--chroma-disable-dit-mask` / `--chroma-enable-t5-mask` / `--qwen-image-zero-cond-t` | — | 🚫 | Model-specific tuning; advanced. |
| `--disable-image-metadata` | — | 🚫 | Moot in chimera. sd-cli's flag disables a Civitai/A1111-style `parameters` tEXt chunk written by a **patched** `stbi_write_png` overload in sd's vendored fork of `stb_image_write.h`. Chimera uses stock `stb_image_write`, which writes no text chunks at all — so chimera's PNGs are already metadata-free and there is nothing to "disable". The reverse direction (embedding generation params for parity with sd-cli's default) is a separate feature, not yet on the roadmap. |
| `-o,--output` | `-o,--output` | ✅ | |
| `--mode -M {img_gen,vid_gen,upscale,convert,metadata}` | implicit | 🚫 | Chimera's `sd` subcommand is img_gen-only by design; other modes are out of scope today. |
| `--preview*` / `--metadata-*` | — | 🚫 | CLI-only sd-shell features; not portable into chimera. |

### Notable gaps worth filing

1. ~~**`--guidance` and `--flow-shift`**.~~ ✅ Landed 2026-05-20.
2. ~~**`--clip_g` (alongside `--clip-l`)**.~~ ✅ Landed 2026-05-20 as `--clip-g`.
3. ~~**`--control-image` + `--control-strength` + `--control-net`**.~~ ✅ Landed 2026-05-20 (ControlNet bundle).
4. ~~**`--vae-tiling` family**.~~ ✅ Landed 2026-05-20.
5. ~~**`--diffusion-conv-direct` / `--vae-conv-direct`**~~ ✅ Landed 2026-05-20.
6. ~~**Sampler-RNG / `--rng`**~~ ✅ Landed 2026-05-20.
7. ~~**`--lora-model-dir`**~~ ✅ Landed 2026-05-20 alongside `--lora <path[:scale]>` (repeatable). Note: prompt-side `<lora:foo:0.8>` extraction is **not** wired yet — `--lora` takes explicit paths. Follow-up.
8. ~~**`--type`**~~ ✅ Landed 2026-05-20.
9. ~~**Perf/offload bundle** (`--fa`, `--no-mmap`, `--max-vram`, `--clip-on-cpu`, `--vae-on-cpu`, `--control-net-cpu`, `--force-sdxl-vae-conv-scale`).~~ ✅ Landed 2026-05-20 (Round 1 of the closer).
10. ~~**Sampler/generation core** (`--img-cfg-scale`, `--eta`, `--timestep-shift`, `--sigmas`, `--prediction`, `--lora-apply-mode`).~~ ✅ Landed 2026-05-20 (Round 2).
11. ~~**Model-loading completers** (`--taesd`, `--clip-vision`, `--llm-vision`, `--tensor-type-rules`, `--photo-maker`).~~ ✅ Landed 2026-05-20 (Round 3).
12. ~~**PhotoMaker bundle** (`--pm-id-images-dir`, `--pm-id-embed-path`, `--pm-style-strength`).~~ ✅ Landed 2026-05-20 (Round 4).
13. ~~**Reference images** (`--ref-image`, `--increase-ref-index`, `--no-auto-resize-ref-image`).~~ ✅ Landed 2026-05-20 (Round 5).
14. ~~**Hires-fix bundle** (`--hires`, `--hires-upscaler`, `--upscale-model`, `--hires-width/height/scale/steps/denoising-strength/upscale-tile-size`).~~ ✅ Landed 2026-05-20 (Round 6).
15. ~~**Cache / SCM bundle** (`--cache-mode`, `--cache-option`, `--scm-mask`, `--scm-policy`).~~ ✅ Landed 2026-05-20 (Round 7). Mirrors sd-cli's 4-flag surface; the 15-key `--cache-option` kv-parser branches on the active mode just like sd-cli does.
16. ~~**`--embd-dir`** (textual-inversion directory).~~ ✅ Landed 2026-05-20 (Round 8). Non-recursive scan for `.gguf`/`.safetensors`/`.pt`; filename stem becomes the prompt token; validated before `new_sd_ctx`.

All sd items in this list are now resolved. `--disable-image-metadata` (the prior residual) was reclassified 🚫 in the table above — chimera's stock `stb_image_write` doesn't embed any metadata to begin with, so the flag has nothing to disable. A future "embed metadata" feature would be net-new functionality, not a port.

### Deliberately omitted

- Video mode (`vid_gen`, `--video-frames`, `--fps`, `--vace-strength`, `--end-img`, `--control-video`).
- Upscale-only / convert-only / metadata-only sd modes (chimera-sd is img_gen-scoped).
- Seamless-tile (`--circular*`).
- sd-cli shell features: `--preview*`, `--metadata-*`, `--canny`, `--mode`.
- Chroma-specific advanced flags (`--chroma-*`) unless we land Chroma support.

---

## Cross-cutting observations

### 1. Naming drift between chimera and upstreams

- **Kebab vs underscore.** sd.cpp's text-encoder flags are underscored (`--clip_l`, `--clip_g`, `--llm_vision`, `--qwen2vl`); chimera normalizes everything to kebab (`--clip-l`). This is a defensible house style but should be called out in `--help` text so users porting sd command lines don't get a "no such option" surprise.
- **`--sample-method` vs `--sampling-method`.** Minor drift, but the kind of thing that breaks copy-pasting from sd-cpp docs. Same for `--init-image` vs `--init-img`, `--mask-image` vs `--mask`, `--input` vs `--file` (whisper).
- **whisper `--timestamps` flips polarity** vs upstream's `--no-timestamps` (chimera defaults to off, upstream to on). Document loudly; do not change.

### 2. Flags chimera handles inconsistently across the three subcommands

- ~~**`--flash-attn`** — exists in upstream llama-cli, whisper-cli, *and* sd-cpp; not exposed in any of chimera's subcommands.~~ ✅ Landed everywhere (2026-05-20): `--flash-attn` on `gen`/`chat`/`embed` and `whisper`; on `sd`, both `--diffusion-fa` (sd-internal) and the generic global `--fa` are now exposed.
- **`--lora`** — exposed in `serve` but not in `gen`/`chat`/`embed`/`sd`. The asymmetry is a footgun.
- **Output formatting** — `embed` lacks `--embd-output-format`, `whisper` lacks `-oj/-osrt/-ovtt`. Both subcommands' output stories are unevenly developed compared to upstream.

### 3. Environment-variable fallbacks chimera doesn't honor

llama.cpp's `common_arg` machinery wires several flags to env vars (`LLAMA_ARG_CTX_CHECKPOINTS`, `LLAMA_ARG_CACHE_RAM`, `LLAMA_ARG_KV_UNIFIED`, `LLAMA_ARG_CONTEXT_SHIFT`, `LLAMA_ARG_CACHE_IDLE_SLOTS`, …). Chimera honors none of these. For *server* use this can matter (containerized deploys); for the four CLI subcommands the omission is fine. Flag for follow-up only if `chimera serve` users start asking.

### 4. The "scope cuts make sense" footnote

Three big slabs of upstream surface area are correctly out of scope and should stay that way:

- llama-cli's interactive REPL (`-i`, `--in-prefix`, `--reverse-prompt`, `--multiline-input`, etc.) — chimera replaces it with `chat` + linenoise + SQLite persistence.
- Speculative decoding (`--draft*`, `--spec-*`) — none of the chimera subcommands wrap a draft-model code path yet.
- Training / perplexity / hellaswag / cvector-generator / imatrix flags — those upstream binaries don't have chimera analogs.

### 5. Top issues to file from this audit

In priority order (highest user impact first). Items struck through landed on 2026-05-20.

1. ~~**sd: Flux/SD3 guidance pair** (`--guidance`, `--flow-shift`).~~ ✅ Landed 2026-05-20.
2. ~~**sd: ControlNet bundle** (`--control-net`, `--control-image`, `--control-strength`).~~ ✅ Landed 2026-05-20.
3. ~~**whisper: output-format family** (`-osrt`, `-oj`, `-ovtt`, `-ojf`, `-ocsv`, `-olrc`).~~ ✅
4. ~~**sd: VAE-tiling bundle** (`--vae-tiling` + tile-size/overlap).~~ ✅ Landed 2026-05-20.
5. ~~**llama: `--grammar` / `--json-schema` / `--json-schema-file`** in `gen`.~~ ✅
6. ~~**All three: `--flash-attn`**.~~ ✅
7. ~~**llama: `--lora` in `gen`/`chat`**.~~ ✅
8. ~~**whisper: `--prompt` + decoding-strategy basics** (`--beam-size`, `--best-of`, `--temperature`, `--no-fallback`).~~ ✅ Landed 2026-05-20.
9. ~~**sd: `--lora`, `--lora-model-dir`, `--clip_g`, `--type`**~~ ✅ Landed 2026-05-20.
10. ~~**embed: `--embd-output-format` + `--embd-separator` + `--attention`**~~ ✅ Landed 2026-05-20 (also `--pooling rank`).
11. ~~**sd coverage closer — Rounds 1–8 (38 flags).**~~ ✅ Landed 2026-05-20. Perf/offload (Round 1), sampler/generation (Round 2), model-loading completers (Round 3), PhotoMaker bundle (Round 4), reference images (Round 5), hires-fix bundle (Round 6), cache/SCM bundle (Round 7), `--embd-dir` (Round 8). See the per-section tables above and the CHANGELOG entry for the full enumeration.
12. ~~**whisper coverage closer — Batches 1–3 + VAD + offset/duration (22 flags).**~~ ✅ Landed 2026-05-20.

Residual open items at the close of this audit cycle: whisper `--grammar` family, whisper `--detect-language` + stereo `--diarize` (both wrapper logic). The prior `chat --reasoning-budget` enforcement gap was closed the same day — the integration turned out to be entirely upstream of `common_sampler_init`, not inside the sample loop. The prior sd `--disable-image-metadata` residual was reclassified 🚫 — chimera's stock `stb_image_write` doesn't embed any text chunks, so there is nothing to disable; a future "embed metadata" feature for parity with sd-cli's default is tracked as net-new functionality, not a port. None of the remaining open items are high-impact; all are documented in their per-section tables with a sentence explaining why they were deferred or are out of scope.
