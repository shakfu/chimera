#pragma once

// Optional C++ object-oriented layer over libchimera.
//
// libchimera exposes a procedural surface: build an Options struct, call
// `command_*(opts)`, get an exit code. This header wraps that surface in
// RAII classes so C++ consumers can treat chimera's capabilities as
// long-lived objects (load a model once, call .generate() many times)
// rather than as one-shot commands.
//
// Design notes:
//   * Header-only. Consumers link libchimera.a as before; no extra TU.
//   * Persistent-handle for Llama (load_model in ctor, reused across calls).
//   * Options-in-ctor for Whisper / SD / Server. Their underlying
//     `command_*` functions own the full load -> run -> unload lifecycle
//     today; persistent-handle would require library-side refactors that
//     are out of scope for this first cut. The wrappers therefore add no
//     state-reuse benefit over the C-style API, only API consistency.
//   * Each class hands out a `raw()` accessor to the underlying handle
//     for callers that need to drop down to the C API.
//
// Example:
//
//   chimera::Llama llm({.model = "Qwen3-1.7B-Q4_0.gguf", .n_predict = 128});
//   auto reply = llm.generate("What is the capital of France?");
//
//   chimera::Embedder emb({.model = "bge-small.gguf"});
//   auto vec = emb.embed("hello world");
//
//   chimera::Server srv({.model = "Qwen3-1.7B-Q4_0.gguf", .port = 8080});
//   return srv.run();
//
// What this header does NOT yet wrap:
//   * `command_chat` (interactive REPL) — still lives in src/chimera_cli/
//     because it owns terminal I/O, signal handling, and color streaming.
//     A future cut may carve out a stateless `LlamaChat` class from the
//     REPL shell; until then, callers wanting chat semantics should drive
//     `Llama::generate` against their own pre-rendered chat-templated
//     prompts (or talk to a `chimera::Server`).

#include "chimera.h"
#include "chimera_db.h"
#include "chimera_embed.h"
#include "chimera_embed_cache.h"
#include "chimera_llama.h"
#ifdef CHIMERA_HAS_SD
#include "chimera_sd.h"
#endif
#ifdef CHIMERA_HAS_WHISPER
#include "chimera_whisper.h"
#endif

#include "llama.h"

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace chimera {

// Streaming hook re-exported from chimera_llama.h so the OOP API surface
// is self-contained -- callers can write `chimera::TokenCallback` without
// reaching into the underlying C-style header.
using TokenCallback = ::TokenCallback;

// -------------------------------------------------------------------------
// Llama -- persistent-handle wrapper over libchimera's text-generation glue.
//
// Persistence model:
//   * Model: loaded by the ctor, freed by the dtor.
//   * Context (text path): lazy-initialized on the first generate() call
//     and reused thereafter. Sized to `options().n_ctx` (default 4096).
//     The KV cache is cleared before each generate() so semantics match
//     a one-shot call -- the persistence is an internal optimization,
//     not a behavior change. Call reset() to clear the cache manually
//     (e.g. between unrelated requests if you want to forget cached
//     compute artifacts -- usually unnecessary).
//   * LoRA adapters and the sampler: rebuilt per generate() call so
//     mutations to `options().lora_adapters` and sampler knobs (temp,
//     top_k, seed, n_predict, ...) take effect immediately.
//   * Vision path (set `options().images`): falls back to the per-call
//     fresh-ctx path inside run_generation_mtmd. The vision encoder and
//     mtmd_context have their own lifecycle that isn't worth piping
//     through here -- the dominant cost is still the model load.
//
// Dirty-options policy: mutating context-creation fields (n_ctx,
// cache_type_k/v, flash_attn, rope_*, yarn_*, swa_full, control_vector*)
// after the first generate() silently no-ops. Call reset(/*rebuild=*/true)
// to drop the cached ctx and have the next generate() honor the new
// values.
// -------------------------------------------------------------------------
class Llama {
public:
    explicit Llama(LlamaCommonOptions opts)
        : opts_(std::move(opts)),
          model_(load_llama_model(opts_)) {}

    // Run one generation against the persistent model. Returns the
    // generated text. With `stream=true` the tokens are also written to
    // stdout as they're sampled (matches the CLI's `gen` behavior; a
    // trailing newline is printed when the generation completes). For
    // anything other than stdout streaming, use the callback overload.
    std::string generate(const std::string & prompt, bool stream = false) {
        if (!stream) {
            return generate(prompt, TokenCallback{});
        }
        const TokenCallback on_token = [](std::string_view piece) {
            std::cout << piece << std::flush;
        };
        std::string text = generate(prompt, on_token);
        std::cout << '\n';
        return text;
    }

    // Callback-driven generate. `on_token` is invoked once per sampled
    // token with the detokenized piece (UTF-8). Pass an empty
    // TokenCallback (the default) to disable streaming and just collect
    // the returned text. The caller chooses where the bytes go and owns
    // trailing-newline / buffering / flushing policy.
    std::string generate(const std::string &  prompt,
                         const TokenCallback & on_token) {
        if (!opts_.images.empty()) {
            if (opts_.mmproj.empty()) {
                fail(ExitCode::BadInput, "Llama::generate: images set but mmproj is empty");
            }
            // Vision path -- fresh ctx per call (see class header).
            return run_generation_mtmd(model_.get(), opts_, prompt, on_token);
        }

        ensure_ctx_();

        // Single-shot semantics: clear KV before decoding the new prompt.
        // The persistent ctx still saves the per-call allocation cost; the
        // KV-cache content itself is not carried over.
        llama_memory_clear(llama_get_memory(ctx_.get()), /*data=*/true);

        const llama_vocab * vocab = llama_model_get_vocab(model_.get());
        const auto prompt_tokens  = ::tokenize(vocab, prompt, /*add_special=*/true, /*parse_special=*/true);

        // LoRA + sampler are cheap to rebuild and reflect the latest opts_
        // (samplers especially are stateful -- a stale sampler would
        // carry penalty history from the previous call).
        auto loras   = load_loras(model_.get(), ctx_.get(), opts_.lora_adapters);
        auto sampler = make_sampler(model_.get(), opts_);

        decode_tokens(ctx_.get(), prompt_tokens, static_cast<int32_t>(opts_.n_batch));
        for (llama_token tok : prompt_tokens) {
            common_sampler_accept(sampler.get(), tok, false);
        }
        return sample_loop(ctx_.get(), sampler.get(), vocab, opts_.n_predict, on_token);
    }

    // Clear the cached KV. `rebuild` also drops the cached context so the
    // next generate() builds a fresh one honoring any changes to
    // context-creation fields. Without rebuild, only the KV memory is
    // wiped -- cheap, but new n_ctx / cache_type / flash_attn values
    // won't take effect.
    void reset(bool rebuild = false) {
        if (rebuild) {
            ctx_.reset();
        } else if (ctx_) {
            llama_memory_clear(llama_get_memory(ctx_.get()), /*data=*/true);
        }
    }

    // Mutable accessor: tweak per-call parameters (n_predict, temp, ...)
    // between generate() calls without reloading the model.
    LlamaCommonOptions &       options()       { return opts_; }
    const LlamaCommonOptions & options() const { return opts_; }

    // Vocabulary-side helpers.
    std::vector<llama_token> tokenize(const std::string & text,
                                      bool add_special   = true,
                                      bool parse_special = true) const {
        return ::tokenize(llama_model_get_vocab(model_.get()),
                          text, add_special, parse_special);
    }

    std::string detokenize(llama_token token) const {
        return token_to_piece(llama_model_get_vocab(model_.get()), token);
    }

    std::string detokenize(const std::vector<llama_token> & tokens) const {
        const llama_vocab * vocab = llama_model_get_vocab(model_.get());
        std::string out;
        for (llama_token t : tokens) out += token_to_piece(vocab, t);
        return out;
    }

    // Drop down to the C handles for callers that need it. ctx() returns
    // nullptr until the first generate() (or after reset(/*rebuild=*/true)).
    llama_model *       raw()       { return model_.get(); }
    const llama_model * raw() const { return model_.get(); }
    llama_context *     ctx()       { return ctx_.get(); }

private:
    void ensure_ctx_() {
        if (ctx_) return;
        // min_prompt_tokens=0 lets new_llama_context fall back to opts_.n_ctx
        // (or its built-in floor of n_predict+32 when n_ctx is 0). Callers
        // who push prompts larger than opts_.n_ctx should set opts_.n_ctx
        // before the first generate().
        ctx_ = new_llama_context(model_.get(), opts_, /*min_prompt_tokens=*/0);
    }

    LlamaCommonOptions opts_;
    LlamaModelPtr      model_;
    LlamaContextPtr    ctx_;
};

// -------------------------------------------------------------------------
// Embedder -- wraps chimera_embed::Embedder against the chimera-CLI-style
// EmbedOptions surface. Persistent: the model loads once and the same
// instance can embed many strings.
// -------------------------------------------------------------------------
class Embedder {
public:
    explicit Embedder(EmbedOptions opts)
        : opts_(std::move(opts)), inner_(make_config(opts_)) {
        if (opts_.cache_embeddings) {
            const std::string mid = chimera_embed_cache::compute_model_id(opts_.model);
            if (mid.empty()) {
                fail(ExitCode::BadInput,
                     "Embedder: cannot fingerprint model file: " + opts_.model);
            }
            cache_ = std::make_unique<chimera_embed_cache::Cache>(
                opts_.cache_db.empty() ? chimera_db::default_path() : opts_.cache_db,
                mid);
            inner_.set_cache(cache_.get());
        }
    }

    std::vector<float> embed(const std::string & text) {
        return inner_.embed(text);
    }

    // Convenience: run a one-shot embed across many strings, returning
    // one vector per input in the same order.
    std::vector<std::vector<float>> embed_many(const std::vector<std::string> & texts) {
        std::vector<std::vector<float>> out;
        out.reserve(texts.size());
        for (const auto & t : texts) out.push_back(inner_.embed(t));
        return out;
    }

    const EmbedOptions & options() const { return opts_; }

private:
    static chimera_embed::Config make_config(const EmbedOptions & o) {
        chimera_embed::Config c;
        c.model            = o.model;
        c.pooling          = o.pooling;
        c.attention        = o.attention;
        c.threads          = o.threads;
        c.gpu_layers       = o.gpu_layers;
        c.n_ctx            = o.n_ctx;
        c.n_batch          = o.n_batch;
        c.n_ubatch         = o.n_ubatch;
        c.normalize        = o.normalize;
        c.use_mmap         = o.use_mmap;
        c.use_mlock        = o.use_mlock;
        c.flash_attn       = o.flash_attn;
        c.rope_freq_base   = o.rope_freq_base;
        c.rope_freq_scale  = o.rope_freq_scale;
        c.rope_scaling     = o.rope_scaling;
        c.yarn_orig_ctx    = o.yarn_orig_ctx;
        c.yarn_ext_factor  = o.yarn_ext_factor;
        c.yarn_attn_factor = o.yarn_attn_factor;
        c.yarn_beta_fast   = o.yarn_beta_fast;
        c.yarn_beta_slow   = o.yarn_beta_slow;
        c.main_gpu         = o.main_gpu;
        c.tensor_split     = o.tensor_split;
        c.split_mode       = o.split_mode;
        c.devices          = o.devices;
        return c;
    }

    EmbedOptions                                opts_;
    chimera_embed::Embedder                     inner_;
    std::unique_ptr<chimera_embed_cache::Cache> cache_;
};

#ifdef CHIMERA_HAS_WHISPER
// -------------------------------------------------------------------------
// Whisper -- persistent-handle wrapper. The ctor loads the whisper_context
// from `options().model` and holds it for the lifetime of the instance.
// `run(input)` and `transcribe(input)` both reuse this ctx -- one for the
// CLI-shaped pipeline (resample + diarize + format-file writes), the
// other for the structured-API path that returns the raw
// chimera_whisper::TranscribeResult.
//
// Dirty-options policy: load-time fields (`model`, `no_gpu`, `flash_attn`,
// `gpu_device`) silently no-op after construction. Call reset(/*reload=*/true)
// to drop the cached ctx and have the next call honor the new values. The
// `vad_model` field is also load-adjacent in whisper.cpp (loaded lazily on
// first VAD use), but mutating it between calls works because whisper.cpp
// re-checks the path each time.
// -------------------------------------------------------------------------
class Whisper {
public:
    explicit Whisper(WhisperOptions opts)
        : opts_(std::move(opts)) {
        load_();
    }

    // CLI-shaped flow: full pipeline including format-file writes
    // (txt/srt/vtt/json/csv/lrc) and stdout streaming. Reuses the
    // persistent ctx -- no model reload between calls.
    int run() { return ::run_whisper(ctx_.get(), opts_); }

    // Convenience: set the input path and run.
    int run(const std::string & input) {
        opts_.input = input;
        return ::run_whisper(ctx_.get(), opts_);
    }

    // Structured-API flow: build a chimera_whisper::TranscribeRequest
    // from opts_ + the given WAV (loaded + resampled here), run
    // chimera_whisper::transcribe against the persistent ctx, and
    // return the structured result. No stdout side effects; no format
    // files written. Intended for library consumers that want to drive
    // their own output (UI / JSON envelope / database).
    chimera_whisper::TranscribeResult transcribe(const std::string & wav_path) {
        opts_.input = wav_path;
        return transcribe();
    }

    chimera_whisper::TranscribeResult transcribe() {
        if (opts_.input.empty()) {
            fail(ExitCode::BadInput, "Whisper::transcribe: options().input is empty");
        }
        auto wav = chimera_whisper::load_wav_file(opts_.input);
        chimera_whisper::TranscribeRequest req;
        req.audio_16k_mono = chimera_whisper::resample_linear(
            wav.samples, wav.sample_rate, /*WHISPER_SAMPLE_RATE=*/16000);
        req.language        = opts_.language;
        req.translate       = opts_.translate;
        req.no_context      = opts_.no_context;
        req.emit_timestamps = opts_.timestamps;
        req.threads         = opts_.threads;
        req.initial_prompt        = opts_.initial_prompt;
        req.carry_initial_prompt  = opts_.carry_initial_prompt;
        req.beam_size             = opts_.beam_size;
        req.best_of               = opts_.best_of;
        req.temperature           = opts_.temperature;
        req.no_fallback           = opts_.no_fallback;
        req.offset_ms             = opts_.offset_ms;
        req.duration_ms           = opts_.duration_ms;
        req.vad                       = opts_.vad;
        req.vad_model_path            = opts_.vad_model;
        req.vad_threshold             = opts_.vad_threshold;
        req.vad_min_speech_duration_ms  = opts_.vad_min_speech_duration_ms;
        req.vad_min_silence_duration_ms = opts_.vad_min_silence_duration_ms;
        req.vad_max_speech_duration_s   = opts_.vad_max_speech_duration_s;
        req.vad_speech_pad_ms           = opts_.vad_speech_pad_ms;
        req.vad_samples_overlap         = opts_.vad_samples_overlap;
        req.max_len            = opts_.max_len;
        req.max_tokens         = opts_.max_tokens;
        req.split_on_word      = opts_.split_on_word;
        req.temperature_inc    = opts_.temperature_inc;
        req.entropy_thold      = opts_.entropy_thold;
        req.logprob_thold      = opts_.logprob_thold;
        req.no_speech_thold    = opts_.no_speech_thold;
        req.audio_ctx          = opts_.audio_ctx;
        req.tinydiarize        = opts_.tinydiarize;
        req.suppress_regex     = opts_.suppress_regex;
        req.suppress_nst       = opts_.suppress_nst;
        req.processors         = opts_.processors;
        req.detect_language    = opts_.detect_language;
        return chimera_whisper::transcribe(ctx_.get(), req);
    }

    void reset(bool reload = false) {
        if (reload) {
            ctx_.reset();
            load_();
        }
        // Without reload there's nothing to clear -- whisper_full owns
        // its own per-call state and doesn't carry KV between transcribes.
    }

    WhisperOptions &       options()       { return opts_; }
    const WhisperOptions & options() const { return opts_; }
    whisper_context *      raw()           { return ctx_.get(); }

private:
    void load_() {
        chimera_whisper::LoadParams lp;
        lp.model      = opts_.model;
        lp.use_gpu    = !opts_.no_gpu;
        lp.flash_attn = opts_.flash_attn;
        lp.gpu_device = opts_.gpu_device;
        ctx_ = chimera_whisper::load_model(lp);
        if (!ctx_) {
            fail(ExitCode::Load, "Whisper: failed to load model: " + opts_.model);
        }
    }

    WhisperOptions    opts_;
    WhisperContextPtr ctx_;
};
#endif // CHIMERA_HAS_WHISPER

#ifdef CHIMERA_HAS_SD
// -------------------------------------------------------------------------
// SD -- persistent-handle wrapper. The ctor builds chimera_sd::LoadParams
// from options() and constructs the sd_ctx_t; subsequent run() calls reuse
// it. The structured-API path is generate(prompt) which returns a vector
// of PixelImage without writing files.
//
// Dirty-options policy: load-time fields silently no-op after construction
// (`model`, `diffusion_model`, `vae`, `clip_l/g`, `t5xxl`, `llm`,
// `high_noise_diffusion_model`, `control_net`, `wtype`, `taesd`,
// `clip_vision`, `llm_vision`, `tensor_type_rules`, `photo_maker`,
// `embd_dir`, `init_image` -- which decides vae_decode_only at load time,
// `offload_to_cpu`, `diffusion_fa`, `diffusion_conv_direct`,
// `vae_conv_direct`, `rng`, `sampler_rng`, `threads`, `flash_attn_global`,
// `no_mmap`, `max_vram`, `keep_*_on_cpu`, `force_sdxl_vae_conv_scale`,
// `prediction`, `lora_apply_mode`). Call reset(/*reload=*/true) to honor
// new values on the next call.
//
// Notable corner case: an SD instance constructed with options.init_image
// empty was loaded with vae_decode_only=true and cannot do img2img later
// even if you set init_image afterwards. Construct with init_image set (or
// reset(reload=true) after setting it).
// -------------------------------------------------------------------------
class SD {
public:
    explicit SD(SdOptions opts) : opts_(std::move(opts)) {
        load_();
    }

    // CLI-shaped flow: full pipeline including PNG write to options().output.
    int run() { return ::run_sd(ctx_.get(), opts_); }

    int run(const std::string & prompt) {
        opts_.prompt = prompt;
        return ::run_sd(ctx_.get(), opts_);
    }

    // Structured-API flow: build a chimera_sd::GenerateRequest from opts_,
    // run chimera_sd::generate against the persistent ctx, and return the
    // raw pixel buffers. No file is written. Subset-only wrapper -- only
    // the most commonly used GenerateRequest fields are mapped here; for
    // full control reach into chimera_sd::generate(raw(), req) yourself.
    std::vector<chimera_sd::PixelImage> generate() {
        if (opts_.prompt.empty()) {
            fail(ExitCode::BadInput, "SD::generate: options().prompt is empty");
        }
        chimera_sd::GenerateRequest req;
        req.prompt           = opts_.prompt;
        req.negative_prompt  = opts_.negative_prompt;
        req.width            = opts_.width;
        req.height           = opts_.height;
        req.steps            = opts_.steps;
        req.batch_count      = opts_.batch_count;
        req.clip_skip        = opts_.clip_skip;
        req.threads          = opts_.threads;
        req.seed             = opts_.seed;
        req.cfg_scale        = opts_.cfg_scale;
        req.strength         = opts_.strength;
        req.sample_method    = opts_.sample_method;
        req.scheduler        = opts_.scheduler;
        req.guidance         = opts_.guidance;
        req.flow_shift       = opts_.flow_shift;
        req.img_cfg_scale    = opts_.img_cfg_scale;
        req.eta              = opts_.eta;
        req.shifted_timestep = opts_.shifted_timestep;
        return chimera_sd::generate(ctx_.get(), req);
    }

    std::vector<chimera_sd::PixelImage> generate(const std::string & prompt) {
        opts_.prompt = prompt;
        return generate();
    }

    void reset(bool reload = false) {
        if (reload) {
            ctx_.reset();
            load_();
        }
        // sd_ctx_t has no equivalent of llama_memory_clear -- internal
        // state is per-generate and doesn't carry between calls.
    }

    SdOptions &       options()       { return opts_; }
    const SdOptions & options() const { return opts_; }
    sd_ctx_t *        raw()           { return ctx_.get(); }

private:
    void load_() {
        if (opts_.model.empty() && opts_.diffusion_model.empty()) {
            fail(ExitCode::BadInput,
                 "SD: options().model or options().diffusion_model required");
        }
        const bool need_encode = !opts_.init_image.empty();
        chimera_sd::LoadParams lp;
        lp.model                = opts_.model;
        lp.diffusion_model      = opts_.diffusion_model;
        lp.vae                  = opts_.vae;
        lp.clip_l               = opts_.clip_l;
        lp.clip_g               = opts_.clip_g;
        lp.t5xxl                = opts_.t5xxl;
        lp.llm                  = opts_.llm;
        lp.high_noise_diffusion_model = opts_.high_noise_diffusion_model;
        lp.control_net          = opts_.control_net;
        lp.wtype                = opts_.wtype;
        lp.taesd                = opts_.taesd;
        lp.clip_vision          = opts_.clip_vision;
        lp.llm_vision           = opts_.llm_vision;
        lp.tensor_type_rules    = opts_.tensor_type_rules;
        lp.photo_maker          = opts_.photo_maker;
        lp.embd_dir             = opts_.embd_dir;
        lp.vae_decode_only      = !need_encode;
        lp.offload_to_cpu        = opts_.offload_to_cpu;
        lp.diffusion_flash_attn  = opts_.diffusion_fa;
        lp.diffusion_conv_direct = opts_.diffusion_conv_direct;
        lp.vae_conv_direct       = opts_.vae_conv_direct;
        lp.rng_type              = opts_.rng;
        lp.sampler_rng_type      = opts_.sampler_rng;
        lp.threads               = opts_.threads;
        lp.flash_attn                = opts_.flash_attn_global;
        lp.enable_mmap               = !opts_.no_mmap;
        lp.max_vram                  = opts_.max_vram;
        lp.keep_clip_on_cpu          = opts_.keep_clip_on_cpu;
        lp.keep_vae_on_cpu           = opts_.keep_vae_on_cpu;
        lp.keep_control_net_on_cpu   = opts_.keep_control_net_on_cpu;
        lp.force_sdxl_vae_conv_scale = opts_.force_sdxl_vae_conv_scale;
        lp.prediction                = opts_.prediction;
        lp.lora_apply_mode           = opts_.lora_apply_mode;
        ctx_ = chimera_sd::load_model(lp);
        if (!ctx_) {
            const std::string & shown = opts_.model.empty()
                ? opts_.diffusion_model : opts_.model;
            fail(ExitCode::Load, "SD: failed to load model: " + shown);
        }
    }

    SdOptions    opts_;
    SdContextPtr ctx_;
};
#endif // CHIMERA_HAS_SD

// -------------------------------------------------------------------------
// Server -- OpenAI-compatible HTTP server. `run()` blocks until the server
// shuts down (Ctrl-C or programmatic stop). Options-in-ctor; the server's
// internal lifecycle (model load + httplib listen + shutdown) is owned by
// command_serve, which makes the persistent-handle distinction moot here.
// -------------------------------------------------------------------------
class Server {
public:
    explicit Server(ServeOptions opts) : opts_(std::move(opts)) {}

    int run() { return command_serve(opts_); }

    ServeOptions &       options()       { return opts_; }
    const ServeOptions & options() const { return opts_; }

private:
    ServeOptions opts_;
};

// -------------------------------------------------------------------------
// Tokenizer -- minimal wrapper around `command_tokenize`'s primitives. Loads
// a model just for its vocab; useful when callers want token ids without
// pulling in the full Llama generation surface.
// -------------------------------------------------------------------------
class Tokenizer {
public:
    explicit Tokenizer(const std::string & model_path, bool use_mmap = true) {
        LlamaCommonOptions load_opts;
        load_opts.model      = model_path;
        load_opts.gpu_layers = 0;
        load_opts.use_mmap   = use_mmap;
        model_ = load_llama_model(load_opts);
    }

    std::vector<llama_token> encode(const std::string & text,
                                    bool add_special   = true,
                                    bool parse_special = true) const {
        return ::tokenize(llama_model_get_vocab(model_.get()),
                          text, add_special, parse_special);
    }

    std::string decode(llama_token token) const {
        return token_to_piece(llama_model_get_vocab(model_.get()), token);
    }

    std::string decode(const std::vector<llama_token> & tokens) const {
        const llama_vocab * vocab = llama_model_get_vocab(model_.get());
        std::string out;
        for (llama_token t : tokens) out += token_to_piece(vocab, t);
        return out;
    }

    const llama_vocab * vocab() const { return llama_model_get_vocab(model_.get()); }

private:
    LlamaModelPtr model_;
};

} // namespace chimera
