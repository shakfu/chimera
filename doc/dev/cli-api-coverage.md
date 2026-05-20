# Chimera CLI / Upstream CLI Coverage Audit

Report date: 2026-05-20
Upstream versions compared against:
- llama.cpp `b9119` (flag definitions in `common/arg.cpp`, CLI binary in `tools/cli/cli.cpp`)
- whisper.cpp `v1.8.4` (`examples/cli/cli.cpp`)
- stable-diffusion.cpp `master-596-90e87bc` (CLI shell in `examples/cli/main.cpp`, model/gen flags in `examples/common/common.cpp`)

What triggered this audit: `chimera sd` had silently shipped without the entire split-checkpoint flag family (`--diffusion-model`, `--vae`, `--clip-l`, `--t5xxl`, `--llm`, `--offload-to-cpu`, `--diffusion-fa`), which made it impossible to run any Z-Image / Flux / SD3-class model. The hole was invisible because no one had cross-referenced sd's flag surface against ours; this audit closes that blind spot for the four CLI subcommands (`gen`/`chat`, `embed`, `whisper`, `sd`). `chimera serve` (wraps llama-server) is out of scope.

Status legend: ✅ exposed · 🔀 renamed · 🟡 partial · ❌ missing · 🚫 out-of-scope.

> Note: chimera's CLI definitions live in `src/chimera_cli/chimera.cpp` (`bind_*_cmd` helpers) and the option structs in `src/chimera/chimera.h`. Both files are referenced throughout this report.

---

## Executive summary

| Subcommand | Upstream flags considered | Exposed | Renamed | Missing (real gap) | Deliberately out of scope |
|---|---:|---:|---:|---:|---:|
| `gen` (llama-cli) | ~80 CLI-relevant | 13 | 1 | ~14 | ~50 |
| `chat` (llama-cli, interactive) | ~85 | 17 | 0 | ~10 | ~55 |
| `embed` (llama-embedding) | ~14 | 8 | 1 | 4 | 1 |
| `whisper` (whisper-cli) | 58 | 5 | 1 | ~18 | ~33 |
| `sd` (sd cli) | 107 | 22 | 6 | ~30 | ~50 |

"Real gaps" are flags whose absence we'd consider filing an issue for. "Deliberately out of scope" covers things like llama-cli's REPL plumbing (chimera replaces it with `chat`), perplexity/imatrix/training knobs, anything tied solely to llama-server, and obscure research/debug flags. The next two columns of the per-subcommand tables make each call individually.

The headline finding is that **the sd surface is still by far the largest source of meaningful gaps** — even after landing the model-loading family, ~30 generation-side flags (CFG variants, sampler-RNG, VAE tiling/conv-direct, LoRA dir, ControlNet, PhotoMaker, hires-fix, video) remain unexposed. The llama coverage is intentionally minimal — chimera leans on its own DSL (`chat` REPL, `serve` HTTP) and the wrapped subcommands are deliberately thin. The whisper surface is the most "leaky" relative to size: classic output-format and VAD flags are absent and most of them are one-liners to add.

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
| `--ubatch-size` | — | ❌ | Common knob; exposed in `serve`. Cheap to add. |
| `--threads, -t` | `-t,--threads` | ✅ | |
| `--threads-batch` | — | ❌ | Rare in practice. |
| `--seed` | `--seed` | ✅ | |
| `--temp` | `--temp` | ✅ | |
| `--top-k` | `--top-k` | ✅ | |
| `--top-p` | `--top-p` | ✅ | |
| `--min-p` | `--min-p` | ✅ | |
| `--repeat-penalty` | `--repeat-penalty` | ✅ | |
| `--repeat-last-n` | — | ❌ | Pairs with `--repeat-penalty`; should ship together. |
| `--presence-penalty` / `--frequency-penalty` | — | ❌ | Standard sampler knobs. |
| `--typical` | — | ❌ | |
| `--top-nsigma` | — | ❌ | New-ish; nice-to-have. |
| `--xtc-probability` / `--xtc-threshold` | — | ❌ | Newer sampler; low priority. |
| `--dry-*` (multiplier/base/allowed-length/penalty-last-n/sequence-breaker) | — | ❌ | DRY sampler; popular for repetition control. Worth one combined flag pass. |
| `--mirostat` / `--mirostat-ent` / `--mirostat-lr` | — | ❌ | Legacy but still used. |
| `--samplers` / `--sampler-seq` | — | ❌ | Sampler-chain ordering. Power-user, low priority. |
| `--dynatemp-range` / `--dynatemp-exp` | — | ❌ | |
| `--logit-bias` | — | ❌ | Common feature; modest cost. |
| `--ignore-eos` | — | ❌ | Cheap, occasionally useful. |
| `--grammar` / `--grammar-file` / `--json-schema` / `--json-schema-file` | — | ❌ | Structured-output gap. **Worth filing** — meaningful loss in CLI-driven scripting. |
| `--flash-attn` | — | ❌ | Universally useful perf knob. **Worth filing.** |
| `--mmap` / `--mlock` | hard-coded `use_mmap=true` | 🟡 | No way to disable; uncommon need but trivial flag. |
| `--gpu-layers` | `--gpu-layers` | ✅ | |
| `--main-gpu` / `--tensor-split` / `--split-mode` | — | ❌ | Multi-GPU users will notice. |
| `--device` / `--list-devices` | — | ❌ | Backend/device selection. |
| `--n-cpu-moe` / `--cpu-moe` | — | ❌ | MoE-specific offload. |
| `--override-tensor` / `--override-kv` | — | ❌ | Power-user. |
| `--cache-type-k` / `--cache-type-v` | — | ❌ | KV-quant. Asked for often in support threads. |
| `--rope-freq-base` / `--rope-freq-scale` / `--rope-scaling` / `--rope-scale` | — | ❌ | RoPE override; matters for long-context fine-tunes. |
| `--yarn-*` (orig-ctx, ext-factor, attn-factor, beta-fast, beta-slow) | — | ❌ | Pairs with `--rope-scaling yarn`. |
| `--lora` / `--lora-scaled` | — | ❌ | Exposed in `serve` but not `gen`/`chat`. **Worth filing** (inconsistent surface). |
| `--mmproj` | `--mmproj` | ✅ | |
| `--mmproj-offload` / `--mmproj-auto` / `--mmproj-url` | — | ❌ | Niche. |
| `--image` | `--image` (gen; repeatable) | ✅ | `chat` injects images via `/image` REPL command. |
| `--image-min-tokens` / `--image-max-tokens` | — | ❌ | Vision-token budget. |
| `--system-prompt` | `--system` (chat) | 🔀 | Renamed; `gen` lacks it (it's an interactive concept). |
| `--system-prompt-file` | `--system-prompt-file` (chat) | ✅ | |
| `--chat-template` | `--chat-template` (chat) | ✅ | |
| `--chat-template-file` / `--chat-template-kwargs` | — | ❌ | Templates from disk; reasonable miss. |
| `--jinja` | — | ❌ | Required if chat-template uses jinja; **worth filing** as a follow-up to template support. |
| `--reasoning` / `--reasoning-budget` / `--reasoning-format` / `--reasoning-budget-message` | — | ❌ | Reasoning-model controls; **worth filing** as one issue. |
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

1. **`--flash-attn`** — universally applicable perf win; not exposed in any llama subcommand. One line each in `gen`/`chat`/`embed`.
2. **`--grammar` / `--json-schema` family** — meaningful loss for scripting use cases (the main thing `gen` is for). Upstream offers both inline and file forms.
3. **DRY + repeat-last-n sampler family** — repetition-control flags ship in pairs and we only expose half of the pair (`--repeat-penalty` with no `--repeat-last-n`).
4. **`--lora` in `gen`/`chat`** — already exposed in `serve`; the asymmetry is a footgun.
5. **`--reasoning*` family** — required to drive reasoning models (DeepSeek-R1, o1-style) cleanly. Bundle as one issue.

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
| `--pooling {none,mean,cls,last,rank}` | `--pooling` | 🟡 | Chimera doc-string lists `mean\|cls\|last\|none`; `rank` (reranker) is **missing**. |
| `--embd-normalize N` (-1 / 0 / 1 / 2 / >2) | `--no-normalize` flag | 🔀 | Chimera reduces to a boolean (L2 or off). Loses access to taxicab/p-norm. Acceptable simplification; document the choice. |
| `--embd-output-format` | — | ❌ | No way to ask for OpenAI-style JSON / `array` / `raw`. **Worth filing.** |
| `--embd-separator` | — | ❌ | Batch separator; needed if anyone passes multi-doc prompts. **Worth filing.** |
| `--ctx-size, -c` | `-c,--ctx-size` | ✅ | |
| `--batch-size, -b` | `-b,--batch-size` | ✅ | |
| `--threads, -t` | `-t,--threads` | ✅ | |
| `--gpu-layers` | `--gpu-layers` | ✅ | |
| `--attention {causal,non-causal}` | — | ❌ | Some embed models need this overridden; **worth filing.** |
| `--flash-attn` | — | ❌ | Same gap as gen/chat. |
| `--cls-separator` | — | 🚫 | Eval/retrieval-specific. |
| `--chunk` / `--chunks` / `--chunk-size` / `--chunk-separator` | — | 🚫 | Belongs to chimera's own `index`/`search` layer, not the model invocation. |
| `--output-format` (general) | — | 🚫 | See `--embd-output-format`. |

### Chimera-specific extensions (no upstream)

- `--cache-embeddings` / `--cache-db` — SQLite memoization layer. No upstream analogue.
- `-o,--output` — chimera writes to a file/stdout instead of `embedding.txt` style upstream behavior. Cleaner.

### Notable gaps worth filing

1. `--embd-output-format` — without this, scripting against `chimera embed` for OpenAI-compatible output requires post-processing.
2. `--embd-separator` — needed for multi-prompt batching.
3. `--attention causal|non-causal` — some encoder models need this.
4. Pooling `rank` value — required to use reranker checkpoints via `embed`.

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
| `-p / --processors` | — | ❌ | Splits decode across N processors. Cheap to add. |
| `-l / --language` | `-l,--language` | ✅ | |
| `-dl / --detect-language` | — | ❌ | Useful as exit-after-detect mode. |
| `-tr / --translate` | `--translate` | ✅ | |
| `--prompt` | — | ❌ | Initial-prompt biasing; commonly used. **Worth filing.** |
| `--carry-initial-prompt` | — | ❌ | Pairs with `--prompt`. |
| `-bs / --beam-size` | — | ❌ | Decoding-strategy basics. **Worth filing.** |
| `-bo / --best-of` | — | ❌ | Same group. |
| `-tp / --temperature` | — | ❌ | |
| `-tpi / --temperature-inc` | — | ❌ | Temperature fallback ladder. |
| `-nf / --no-fallback` | — | ❌ | |
| `-mc / --max-context` | — | ❌ | |
| `-ml / --max-len` | — | ❌ | |
| `-sow / --split-on-word` | — | ❌ | |
| `-wt / --word-thold` | — | ❌ | |
| `-et / --entropy-thold` / `-lpt / --logprob-thold` / `-nth / --no-speech-thold` | — | ❌ | Decoder-fail thresholds. |
| `-ot / --offset-t` / `-on / --offset-n` / `-d / --duration` | — | ❌ | Region-of-audio selection. **Worth filing.** |
| `-ac / --audio-ctx` | — | ❌ | Halve context for tiny.en speedups. |
| `-fa / --flash-attn` / `-nfa / --no-flash-attn` | — | ❌ | Perf knob. |
| `-ng / --no-gpu` | — | ❌ | No way to force CPU; chimera builds with GPU support. |
| `-dev / --device` | — | ❌ | Specific GPU. |
| `-di / --diarize` | — | ❌ | Stereo diarization. |
| `-tdrz / --tinydiarize` | — | ❌ | tdrz-model diarization. |
| `-otxt / -ovtt / -osrt / -ocsv / -olrc / -oj / -ojf / -owts` | `-o,--output` (txt-ish only) | 🟡 | Chimera only emits one text file; **none of the structured formats** (vtt/srt/json/csv/lrc) are exposed. **Worth filing** — these are routine in transcription pipelines. |
| `-of / --output-file` | `-o,--output` | 🔀 | Single file, single format. |
| `-fp / --font-path` | — | 🚫 | Karaoke-only. |
| `--timestamps` (chimera) ↔ `-nt / --no-timestamps` | `--timestamps` flag | 🔀 | Inverted polarity vs upstream default. Document this; don't change. |
| `--no-context` | `--no-context` | ✅ | |
| `--vad` | — | ❌ | Enables built-in VAD. **Worth filing** (post-v1.8 push from upstream). |
| `--vad-model` / `--vad-threshold` / `--vad-min-speech-duration-ms` / `--vad-min-silence-duration-ms` / `--vad-max-speech-duration-s` / `--vad-speech-pad-ms` / `--vad-samples-overlap` | — | ❌ | All-or-nothing with `--vad`. Bundle as one issue. |
| `-sns / --suppress-nst` / `--suppress-regex` | — | ❌ | Token suppression. |
| `--grammar` / `--grammar-rule` / `--grammar-penalty` | — | ❌ | Constrained decoding. |
| `-dtw / --dtw` | — | ❌ | Token-level timestamps. |
| `-oved / --ov-e-device` | — | 🚫 | OpenVINO-only. |
| `-debug / --debug-mode` / `-np / --no-prints` / `-ps / --print-special` / `-pc / --print-colors` / `--print-confidence` / `-pp / --print-progress` / `-ls / --log-score` | — | 🚫 | Debug / logging cosmetics; chimera owns its own logging. |

### Notable gaps worth filing

1. **Output-format family** (`-osrt/-ovtt/-oj/-ojf/-ocsv/-olrc`) — single biggest UX gap. Transcription pipelines almost always want SRT or JSON.
2. **VAD bundle** (`--vad` + the seven knobs) — current whisper.cpp default mode is becoming "vad on"; not having it is increasingly anomalous.
3. **`--prompt` / `--carry-initial-prompt`** — required for vocabulary/style biasing; trivial to expose.
4. **Decoding strategy** (`--beam-size`, `--best-of`, `--temperature`, `--no-fallback`) — current chimera always uses upstream defaults with no escape hatch.
5. **Offset/duration** (`-ot`, `-on`, `-d`) — slice-the-audio is a common ask.

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
| `--high-noise-diffusion-model` | — | ❌ | Two-model "high noise" workflows (recent feature). |
| `--vae` | `--vae` | ✅ | |
| `--taesd` / `--tae` | — | ❌ | Tiny-AutoEncoder fast decode. **Worth filing**, low cost. |
| `--clip_l` | `--clip-l` | 🔀 | **Naming drift.** Upstream uses underscore; chimera uses kebab. Stay with kebab in chimera (project convention) but document. |
| `--clip_g` | — | ❌ | Required for SDXL split layouts. **Worth filing.** |
| `--clip_vision` | — | ❌ | |
| `--t5xxl` | `--t5xxl` | ✅ | |
| `--llm` | `--llm` | ✅ | Z-Image text encoder. |
| `--llm_vision` / `--qwen2vl` / `--qwen2vl_vision` | — | ❌/🚫 | `--qwen2vl` is a deprecated alias of `--llm`; safe to skip. `--llm_vision` may matter for vision-conditioning Qwen-Image. |
| `--control-net` | — | ❌ | ControlNet model path. **Worth filing.** |
| `--embd-dir` | — | ❌ | Textual-inversion / embedding directory. |
| `--lora-model-dir` | — | ❌ | LoRA discovery directory. **Worth filing** — pairs with prompt-syntax LoRA. |
| `--photo-maker` | — | ❌ | PhotoMaker checkpoint. |
| `--upscale-model` / `--hires-upscalers-dir` | — | ❌ | Upscaler integration. |
| `--tensor-type-rules` | — | ❌ | |
| `--type` | — | ❌ | Weights type / precision override; **worth filing**. |

### Coverage table — perf / offload

| Upstream flag | Chimera | Status | Notes |
|---|---|---|---|
| `--threads` | `-t,--threads` | ✅ | |
| `--offload-to-cpu` | `--offload-to-cpu` | ✅ | Landed in audit. |
| `--max-vram` | — | ❌ | VRAM cap. |
| `--mmap` | — | ❌ | |
| `--fa` | — | ❌ | Global flash-attn (not just diffusion). |
| `--diffusion-fa` | `--diffusion-fa` | ✅ | Landed in audit. |
| `--diffusion-conv-direct` / `--vae-conv-direct` | — | ❌ | conv-direct perf path. **Worth filing**. |
| `--clip-on-cpu` / `--vae-on-cpu` / `--control-net-cpu` | — | ❌ | Selective CPU offload (more surgical than `--offload-to-cpu`). |
| `--force-sdxl-vae-conv-scale` | — | ❌ | SDXL-VAE numerics fix. |

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
| `--img-cfg-scale` | — | ❌ | Separate img-cond CFG (Flux). |
| `--guidance` | — | ❌ | Flux/SD3 guidance scale (distinct from cfg). **Worth filing** — needed for Flux. |
| `--clip-skip` | `--clip-skip` | ✅ | |
| `--sampling-method` | `--sample-method` | 🔀 | Naming drift (`sampling` vs `sample`). Document. |
| `--scheduler` | `--scheduler` | ✅ | |
| `--sigmas` | — | ❌ | Custom sigma schedule. |
| `--rng` / `--sampler-rng` | — | ❌ | Reproducibility/cpu-vs-cuda RNG choice. **Worth filing.** |
| `--prediction` | — | ❌ | epsilon / v-prediction override. |
| `--eta` | — | ❌ | DDIM-style stochasticity. |
| `--flow-shift` | — | ❌ | Flow-matching shift; **needed for SD3 / Flux**. **Worth filing.** |
| `--timestep-shift` | — | ❌ | |
| `--moe-boundary` | — | ❌ | High-noise/low-noise MoE boundary. |
| `--slg-scale` / `--skip-layer-start` / `--skip-layer-end` / `--skip-layers` | — | ❌ | Skip-layer guidance. |
| `--high-noise-*` (cfg-scale, img-cfg-scale, guidance, slg-scale, skip-layer-start/end, eta, sampling-method, skip-layers, steps) | — | ❌ | Entire high-noise group missing (pairs with `--high-noise-diffusion-model`). |

### Coverage table — img2img / inpaint / control

| Upstream flag | Chimera | Status | Notes |
|---|---|---|---|
| `--init-img` | `--init-image` | 🔀 | Naming. |
| `--end-img` | — | ❌ | End-frame for img-to-img blending / video. |
| `--mask` | `--mask-image` | 🔀 | Naming. |
| `--control-image` | — | ❌ | ControlNet conditioning image. **Worth filing.** |
| `--control-strength` | — | ❌ | |
| `--control-video` | — | 🚫 | Video-only; chimera-sd is image-only today. |
| `--strength` | `--strength` | ✅ | |
| `--ref-image` | — | ❌ | Reference image (style/identity). |
| `--pm-id-images-dir` / `--pm-id-embed-path` / `--pm-style-strength` | — | ❌ | PhotoMaker — bundle with `--photo-maker`. |

### Coverage table — hires fix / VAE tiling

| Upstream flag | Chimera | Status | Notes |
|---|---|---|---|
| `--hires` | — | ❌ | Hires-fix toggle. **Worth filing** — popular feature. |
| `--hires-upscaler` / `--hires-width` / `--hires-height` / `--hires-steps` / `--hires-scale` / `--hires-denoising-strength` / `--hires-upscale-tile-size` | — | ❌ | Whole hires-fix family. Bundle with `--hires`. |
| `--vae-tiling` | — | ❌ | Reduces VRAM for large outputs. **Worth filing.** |
| `--vae-tile-size` / `--vae-relative-tile-size` / `--vae-tile-overlap` | — | ❌ | Bundle with `--vae-tiling`. |
| `--upscale-repeats` / `--upscale-tile-size` | — | ❌ | Standalone upscale mode. |

### Coverage table — video / advanced / output

| Upstream flag | Chimera | Status | Notes |
|---|---|---|---|
| `--video-frames` / `--fps` | — | 🚫 | Video mode out of scope for chimera-sd today (sd-cli has `vid_gen` mode). |
| `--vace-strength` / `--increase-ref-index` / `--disable-auto-resize-ref-image` | — | 🚫 | Video / VACE. |
| `--cache-mode` / `--cache-option` | — | ❌ | Intermediate-tensor cache (perf). |
| `--scm-mask` / `--scm-policy` | — | ❌ | Sampler-cached-memory. |
| `--lora-apply-mode` | — | ❌ | |
| `--circular` / `--circularx` / `--circulary` | — | 🚫 | Seamless-tile output; niche. |
| `--chroma-t5-mask-pad` / `--chroma-disable-dit-mask` / `--chroma-enable-t5-mask` / `--qwen-image-zero-cond-t` | — | 🚫 | Model-specific tuning; advanced. |
| `--disable-image-metadata` | — | ❌ | Strip metadata from PNG; reasonable to add. |
| `-o,--output` | `-o,--output` | ✅ | |
| `--mode -M {img_gen,vid_gen,upscale,convert,metadata}` | implicit | 🚫 | Chimera's `sd` subcommand is img_gen-only by design; other modes are out of scope today. |
| `--preview*` / `--metadata-*` | — | 🚫 | CLI-only sd-shell features; not portable into chimera. |

### Notable gaps worth filing

1. **`--guidance` and `--flow-shift`** — without these, Flux and SD3 work but with default-only guidance / shift. Same shape as the Z-Image fix that triggered this audit.
2. **`--clip_g` (alongside `--clip-l`)** — required for SDXL split-checkpoint layouts. Surprising omission given we ship `--clip-l`.
3. **`--control-image` + `--control-strength`** — ControlNet is one of the most-asked-for sd features; we ship `--control-net` model loading? Actually we *don't* (see above). Both halves are missing — bundle into a single "ControlNet support" issue.
4. **`--vae-tiling` family** — the no-VRAM-shame way to render large images. One toggle, three knobs.
5. **`--diffusion-conv-direct` / `--vae-conv-direct`** — measurable perf win on modern dGPUs; bundle as one issue.
6. **Sampler-RNG / `--rng`** — required for cross-implementation reproducibility (matching ComfyUI seeds, etc.).
7. **`--lora-model-dir`** — needed for prompt-side `<lora:foo:0.8>` syntax to resolve files.
8. **`--type`** — explicit weight-precision override; useful for low-VRAM users.

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

- **`--flash-attn`** — exists in upstream llama-cli, whisper-cli, *and* sd-cpp; not exposed in any of chimera's subcommands. (`--diffusion-fa` exists but is sd-internal, not the generic flag.) If we land it, do all three at once.
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

In priority order (highest user impact first):

1. **sd: Flux/SD3 guidance pair** (`--guidance`, `--flow-shift`) — direct analog of the Z-Image fix.
2. **sd: ControlNet bundle** (`--control-net`, `--control-image`, `--control-strength`).
3. **whisper: output-format family** (`-osrt`, `-oj`, `-ovtt`, `-ojf`, `-ocsv`, `-olrc`).
4. **sd: VAE-tiling bundle** (`--vae-tiling` + tile-size/overlap).
5. **llama: `--grammar` / `--json-schema` / `--json-schema-file`** in `gen`.
6. **All three: `--flash-attn`** — one shared flag, three subcommands.
7. **llama: `--lora` in `gen`/`chat`** — surface-consistency with `serve`.
8. **whisper: `--prompt` + decoding-strategy basics** (`--beam-size`, `--best-of`, `--temperature`).
9. **sd: `--lora-model-dir`, `--clip_g`, `--type`** — finishing the model-loading story.
10. **embed: `--embd-output-format` + `--embd-separator` + `--attention`**.
