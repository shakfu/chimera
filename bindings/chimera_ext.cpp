// chimera_ext.cpp - nanobind wrapper over the chimera.hpp OOP layer.
//
// Builds a Python extension module `chimera` exposing the persistent-handle
// classes from src/chimera/chimera.hpp: Llama, Embedder, Tokenizer, Server,
// and (when libchimera.a was built with the modality) SD / Whisper.
//
// Design notes mirror the C++ layer:
//   * Errors: chimera::fail() throws ChimeraError(ExitCode, msg); translated
//     to a Python `chimera.ChimeraError` carrying the message. The
//     interpreter never sees a process exit().
//   * GIL: generate()/embed()/run()/transcribe() release the GIL for the
//     (long) compute. The streaming token callback re-acquires it.
//   * Options: the POD option structs are bound with FULL field coverage.
//     Note: chimera's option structs use std::string (not enums) for every
//     user-facing "choice" field -- sample_method, scheduler, rng,
//     prediction, lora_apply_mode, wtype, pooling, rope_scaling, split_mode,
//     etc. The CLI/engine converts those strings to the underlying engine
//     enums internally (str_to_sample_method, ...). So `ExitCode` is the
//     only nb::enum_ on this surface; everything else is string/scalar/list.
//     Valid string values are noted in comments where non-obvious.
//
// THREAD SAFETY: generate() releases the GIL and mutates one shared
// llama_context. Two Python threads calling generate() on the SAME object is
// a data race. Use one object per thread, or guard with a lock. Distinct
// objects are independent.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>

#include <string>
#include <string_view>
#include <vector>

#include "chimera.hpp"  // the OOP layer (src/chimera on the include path)

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(chimera, m) {
    m.doc() = "Python bindings for libchimera's chimera.hpp OOP layer";

    // ---- exception surface ---------------------------------------------
    nb::enum_<ExitCode>(m, "ExitCode")
        .value("Runtime",  ExitCode::Runtime)
        .value("BadInput", ExitCode::BadInput)
        .value("Load",     ExitCode::Load)
        .value("Generate", ExitCode::Generate);
    // Translator: ChimeraError -> chimera.ChimeraError (message = e.what()).
    nb::exception<ChimeraError>(m, "ChimeraError");

    // ====================================================================
    // LlamaCommonOptions
    // ====================================================================
    nb::class_<LlamaCommonOptions>(m, "LlamaOptions")
        .def(nb::init<>())
        // model + inputs
        .def_rw("model",              &LlamaCommonOptions::model)
        .def_rw("mmproj",             &LlamaCommonOptions::mmproj)
        .def_rw("images",             &LlamaCommonOptions::images)
        // context / batching
        .def_rw("n_ctx",              &LlamaCommonOptions::n_ctx)
        .def_rw("n_batch",            &LlamaCommonOptions::n_batch)
        .def_rw("n_ubatch",           &LlamaCommonOptions::n_ubatch)
        .def_rw("threads",            &LlamaCommonOptions::threads)
        .def_rw("threads_batch",      &LlamaCommonOptions::threads_batch)
        .def_rw("gpu_layers",         &LlamaCommonOptions::gpu_layers)
        .def_rw("n_predict",          &LlamaCommonOptions::n_predict)
        .def_rw("seed",               &LlamaCommonOptions::seed)
        // sampling
        .def_rw("temp",               &LlamaCommonOptions::temp)
        .def_rw("top_k",              &LlamaCommonOptions::top_k)
        .def_rw("top_p",              &LlamaCommonOptions::top_p)
        .def_rw("min_p",              &LlamaCommonOptions::min_p)
        .def_rw("typ_p",              &LlamaCommonOptions::typ_p)
        .def_rw("top_n_sigma",        &LlamaCommonOptions::top_n_sigma)
        .def_rw("repeat_penalty",     &LlamaCommonOptions::repeat_penalty)
        .def_rw("penalty_last_n",     &LlamaCommonOptions::penalty_last_n)
        .def_rw("penalty_present",    &LlamaCommonOptions::penalty_present)
        .def_rw("penalty_freq",       &LlamaCommonOptions::penalty_freq)
        .def_rw("mirostat",           &LlamaCommonOptions::mirostat)
        .def_rw("mirostat_tau",       &LlamaCommonOptions::mirostat_tau)
        .def_rw("mirostat_eta",       &LlamaCommonOptions::mirostat_eta)
        .def_rw("dry_multiplier",     &LlamaCommonOptions::dry_multiplier)
        .def_rw("dry_base",           &LlamaCommonOptions::dry_base)
        .def_rw("dry_allowed_length", &LlamaCommonOptions::dry_allowed_length)
        .def_rw("dry_penalty_last_n", &LlamaCommonOptions::dry_penalty_last_n)
        .def_rw("dry_sequence_breakers", &LlamaCommonOptions::dry_sequence_breakers)
        .def_rw("xtc_probability",    &LlamaCommonOptions::xtc_probability)
        .def_rw("xtc_threshold",      &LlamaCommonOptions::xtc_threshold)
        .def_rw("dynatemp_range",     &LlamaCommonOptions::dynatemp_range)
        .def_rw("dynatemp_exp",       &LlamaCommonOptions::dynatemp_exp)
        .def_rw("samplers",           &LlamaCommonOptions::samplers)   // ";"-joined order
        .def_rw("logit_bias",         &LlamaCommonOptions::logit_bias) // "<id>(+|-|=)<bias>"
        .def_rw("ignore_eos",         &LlamaCommonOptions::ignore_eos)
        // constrained decoding
        .def_rw("grammar",            &LlamaCommonOptions::grammar)
        .def_rw("grammar_file",       &LlamaCommonOptions::grammar_file)
        .def_rw("json_schema",        &LlamaCommonOptions::json_schema)
        .def_rw("json_schema_file",   &LlamaCommonOptions::json_schema_file)
        // model load / memory
        .def_rw("use_mmap",           &LlamaCommonOptions::use_mmap)
        .def_rw("use_mlock",          &LlamaCommonOptions::use_mlock)
        .def_rw("flash_attn",         &LlamaCommonOptions::flash_attn)
        .def_rw("swa_full",           &LlamaCommonOptions::swa_full)
        .def_rw("cache_type_k",       &LlamaCommonOptions::cache_type_k)  // f16|q8_0|...
        .def_rw("cache_type_v",       &LlamaCommonOptions::cache_type_v)
        // LoRA + control vectors
        .def_rw("lora_adapters",      &LlamaCommonOptions::lora_adapters)
        .def_rw("control_vector",        &LlamaCommonOptions::control_vector)
        .def_rw("control_vector_scaled", &LlamaCommonOptions::control_vector_scaled)
        .def_rw("control_vector_layer_start", &LlamaCommonOptions::control_vector_layer_start)
        .def_rw("control_vector_layer_end",   &LlamaCommonOptions::control_vector_layer_end)
        // rope / yarn
        .def_rw("rope_freq_base",     &LlamaCommonOptions::rope_freq_base)
        .def_rw("rope_freq_scale",    &LlamaCommonOptions::rope_freq_scale)
        .def_rw("rope_scaling",       &LlamaCommonOptions::rope_scaling)  // none|linear|yarn|longrope
        .def_rw("yarn_orig_ctx",      &LlamaCommonOptions::yarn_orig_ctx)
        .def_rw("yarn_ext_factor",    &LlamaCommonOptions::yarn_ext_factor)
        .def_rw("yarn_attn_factor",   &LlamaCommonOptions::yarn_attn_factor)
        .def_rw("yarn_beta_fast",     &LlamaCommonOptions::yarn_beta_fast)
        .def_rw("yarn_beta_slow",     &LlamaCommonOptions::yarn_beta_slow)
        // multi-gpu / device placement
        .def_rw("main_gpu",           &LlamaCommonOptions::main_gpu)
        .def_rw("tensor_split",       &LlamaCommonOptions::tensor_split)  // "0.5,0.5"
        .def_rw("split_mode",         &LlamaCommonOptions::split_mode)    // none|layer|row
        .def_rw("devices",            &LlamaCommonOptions::devices)
        .def_rw("override_tensor",    &LlamaCommonOptions::override_tensor)
        .def_rw("override_kv",        &LlamaCommonOptions::override_kv)
        .def_rw("cpu_moe",            &LlamaCommonOptions::cpu_moe)
        .def_rw("n_cpu_moe",          &LlamaCommonOptions::n_cpu_moe)
        // vision (mtmd)
        .def_rw("mmproj_use_gpu",     &LlamaCommonOptions::mmproj_use_gpu)
        .def_rw("image_min_tokens",   &LlamaCommonOptions::image_min_tokens)
        .def_rw("image_max_tokens",   &LlamaCommonOptions::image_max_tokens)
        // chat templating
        .def_rw("chat_template_file",   &LlamaCommonOptions::chat_template_file)
        .def_rw("chat_template_kwargs", &LlamaCommonOptions::chat_template_kwargs)  // dict[str,str]
        .def_rw("use_jinja",          &LlamaCommonOptions::use_jinja)
        // reasoning
        .def_rw("reasoning",          &LlamaCommonOptions::reasoning)
        .def_rw("reasoning_budget",   &LlamaCommonOptions::reasoning_budget)
        .def_rw("reasoning_format",   &LlamaCommonOptions::reasoning_format)
        .def_rw("reasoning_budget_message", &LlamaCommonOptions::reasoning_budget_message);

    // ---- Llama ----------------------------------------------------------
    nb::class_<chimera::Llama>(m, "Llama")
        .def(nb::init<std::string>(), "model_path"_a,
             "Load a model with all other options defaulted.")
        .def(nb::init<LlamaCommonOptions>(), "options"_a)
        .def_prop_ro("options",
            [](chimera::Llama &self) -> LlamaCommonOptions & { return self.options(); },
            nb::rv_policy::reference_internal)
        .def("generate",
            [](chimera::Llama &self, const std::string &prompt, nb::object on_token) {
                chimera::TokenCallback cb;
                if (!on_token.is_none()) {
                    nb::callable fn = nb::cast<nb::callable>(on_token);
                    cb = [fn](std::string_view piece) {
                        nb::gil_scoped_acquire gil;
                        fn(nb::str(piece.data(), piece.size()));
                    };
                }
                std::string out;
                {
                    nb::gil_scoped_release rel;
                    out = self.generate(prompt, cb);
                }
                return out;
            },
            "prompt"_a, "on_token"_a = nb::none(),
            "Generate text. Pass on_token=callable for per-token streaming.")
        .def("reset", &chimera::Llama::reset, "rebuild"_a = false,
             "Clear the KV cache. rebuild=True also drops the context so the "
             "next generate() honors changed context-creation options.")
        .def("tokenize",
            [](const chimera::Llama &self, const std::string &text,
               bool add_special, bool parse_special) {
                return self.tokenize(text, add_special, parse_special);
            },
            "text"_a, "add_special"_a = true, "parse_special"_a = true)
        .def("detokenize",
            [](const chimera::Llama &self, const std::vector<int32_t> &tokens) {
                return self.detokenize(tokens);
            },
            "tokens"_a);

    // ====================================================================
    // EmbedOptions + Embedder
    // ====================================================================
    nb::class_<EmbedOptions>(m, "EmbedOptions")
        .def(nb::init<>())
        .def_rw("model",             &EmbedOptions::model)
        .def_rw("input",             &EmbedOptions::input)
        .def_rw("input_file",        &EmbedOptions::input_file)
        .def_rw("output",            &EmbedOptions::output)
        .def_rw("pooling",           &EmbedOptions::pooling)   // mean|cls|last|none|rank
        .def_rw("embd_output_format",&EmbedOptions::embd_output_format)
        .def_rw("embd_separator",    &EmbedOptions::embd_separator)
        .def_rw("attention",         &EmbedOptions::attention) // causal|non-causal
        .def_rw("threads",           &EmbedOptions::threads)
        .def_rw("gpu_layers",        &EmbedOptions::gpu_layers)
        .def_rw("n_ctx",             &EmbedOptions::n_ctx)
        .def_rw("n_batch",           &EmbedOptions::n_batch)
        .def_rw("n_ubatch",          &EmbedOptions::n_ubatch)
        .def_rw("normalize",         &EmbedOptions::normalize)
        .def_rw("use_mmap",          &EmbedOptions::use_mmap)
        .def_rw("use_mlock",         &EmbedOptions::use_mlock)
        .def_rw("flash_attn",        &EmbedOptions::flash_attn)
        .def_rw("rope_freq_base",    &EmbedOptions::rope_freq_base)
        .def_rw("rope_freq_scale",   &EmbedOptions::rope_freq_scale)
        .def_rw("rope_scaling",      &EmbedOptions::rope_scaling)
        .def_rw("yarn_orig_ctx",     &EmbedOptions::yarn_orig_ctx)
        .def_rw("yarn_ext_factor",   &EmbedOptions::yarn_ext_factor)
        .def_rw("yarn_attn_factor",  &EmbedOptions::yarn_attn_factor)
        .def_rw("yarn_beta_fast",    &EmbedOptions::yarn_beta_fast)
        .def_rw("yarn_beta_slow",    &EmbedOptions::yarn_beta_slow)
        .def_rw("main_gpu",          &EmbedOptions::main_gpu)
        .def_rw("tensor_split",      &EmbedOptions::tensor_split)
        .def_rw("split_mode",        &EmbedOptions::split_mode)
        .def_rw("devices",           &EmbedOptions::devices)
        .def_rw("cache_embeddings",  &EmbedOptions::cache_embeddings)
        .def_rw("cache_db",          &EmbedOptions::cache_db);

    nb::class_<chimera::Embedder>(m, "Embedder")
        .def(nb::init<std::string>(),  "model_path"_a)
        .def(nb::init<EmbedOptions>(), "options"_a)
        // -> list[float] (np.asarray for numpy; or switch to nb::ndarray here)
        .def("embed", &chimera::Embedder::embed,
             nb::call_guard<nb::gil_scoped_release>(), "text"_a)
        .def("embed_many", &chimera::Embedder::embed_many,
             nb::call_guard<nb::gil_scoped_release>(), "texts"_a);

    // ---- Tokenizer ------------------------------------------------------
    nb::class_<chimera::Tokenizer>(m, "Tokenizer")
        .def(nb::init<std::string, bool>(), "model_path"_a, "use_mmap"_a = true)
        .def("encode",
            [](const chimera::Tokenizer &t, const std::string &s,
               bool add_special, bool parse_special) {
                return t.encode(s, add_special, parse_special);
            },
            "text"_a, "add_special"_a = true, "parse_special"_a = true)
        .def("decode",
            [](const chimera::Tokenizer &t, const std::vector<int32_t> &tokens) {
                return t.decode(tokens);
            },
            "tokens"_a);

    // ====================================================================
    // ServeOptions + Server
    // ====================================================================
    nb::class_<ServeOptions>(m, "ServeOptions")
        .def(nb::init<>())
        .def_rw("model",            &ServeOptions::model)
        .def_rw("mmproj",           &ServeOptions::mmproj)
        .def_rw("host",             &ServeOptions::host)
        .def_rw("port",             &ServeOptions::port)
        .def_rw("n_ctx",            &ServeOptions::n_ctx)
        .def_rw("n_batch",          &ServeOptions::n_batch)
        .def_rw("n_ubatch",         &ServeOptions::n_ubatch)
        .def_rw("threads",          &ServeOptions::threads)
        .def_rw("gpu_layers",       &ServeOptions::gpu_layers)
        .def_rw("parallel",         &ServeOptions::parallel)
        .def_rw("api_key",          &ServeOptions::api_key)
        .def_rw("embedding",        &ServeOptions::embedding)
        // audio (whisper) sidecar
        .def_rw("audio_model",      &ServeOptions::audio_model)
        .def_rw("audio_flash_attn", &ServeOptions::audio_flash_attn)
        .def_rw("audio_no_gpu",     &ServeOptions::audio_no_gpu)
        .def_rw("audio_gpu_device", &ServeOptions::audio_gpu_device)
        .def_rw("audio_vad_model",  &ServeOptions::audio_vad_model)
        // image (sd) sidecar
        .def_rw("sd_model",                   &ServeOptions::sd_model)
        .def_rw("sd_control_net",             &ServeOptions::sd_control_net)
        .def_rw("sd_diffusion_model",         &ServeOptions::sd_diffusion_model)
        .def_rw("sd_vae",                     &ServeOptions::sd_vae)
        .def_rw("sd_clip_l",                  &ServeOptions::sd_clip_l)
        .def_rw("sd_clip_g",                  &ServeOptions::sd_clip_g)
        .def_rw("sd_t5xxl",                   &ServeOptions::sd_t5xxl)
        .def_rw("sd_llm",                     &ServeOptions::sd_llm)
        .def_rw("sd_llm_vision",              &ServeOptions::sd_llm_vision)
        .def_rw("sd_clip_vision",             &ServeOptions::sd_clip_vision)
        .def_rw("sd_taesd",                   &ServeOptions::sd_taesd)
        .def_rw("sd_embd_dir",                &ServeOptions::sd_embd_dir)
        .def_rw("sd_type",                    &ServeOptions::sd_type)
        .def_rw("sd_tensor_type_rules",       &ServeOptions::sd_tensor_type_rules)
        .def_rw("sd_high_noise_diffusion_model", &ServeOptions::sd_high_noise_diffusion_model)
        .def_rw("sd_flash_attn",              &ServeOptions::sd_flash_attn)
        .def_rw("sd_diffusion_flash_attn",    &ServeOptions::sd_diffusion_flash_attn)
        .def_rw("sd_diffusion_conv_direct",   &ServeOptions::sd_diffusion_conv_direct)
        .def_rw("sd_vae_conv_direct",         &ServeOptions::sd_vae_conv_direct)
        .def_rw("sd_no_mmap",                 &ServeOptions::sd_no_mmap)
        .def_rw("sd_max_vram",                &ServeOptions::sd_max_vram)
        .def_rw("sd_offload_to_cpu",          &ServeOptions::sd_offload_to_cpu)
        .def_rw("sd_keep_clip_on_cpu",        &ServeOptions::sd_keep_clip_on_cpu)
        .def_rw("sd_keep_vae_on_cpu",         &ServeOptions::sd_keep_vae_on_cpu)
        .def_rw("sd_keep_control_net_on_cpu", &ServeOptions::sd_keep_control_net_on_cpu)
        .def_rw("sd_force_sdxl_vae_conv_scale", &ServeOptions::sd_force_sdxl_vae_conv_scale)
        .def_rw("sd_rng",                     &ServeOptions::sd_rng)
        .def_rw("sd_sampler_rng",             &ServeOptions::sd_sampler_rng)
        .def_rw("sd_prediction",              &ServeOptions::sd_prediction)
        .def_rw("sd_lora_apply_mode",         &ServeOptions::sd_lora_apply_mode)
        .def_rw("sd_threads",                 &ServeOptions::sd_threads)
        .def_rw("sd_photo_maker",             &ServeOptions::sd_photo_maker)
        .def_rw("sd_pm_id_dir",               &ServeOptions::sd_pm_id_dir)
        .def_rw("sd_pm_id_embed_path",        &ServeOptions::sd_pm_id_embed_path)
        .def_rw("sd_loras",                   &ServeOptions::sd_loras)
        // rag / embeddings / rerank
        .def_rw("rag_embedding_model", &ServeOptions::rag_embedding_model)
        .def_rw("rag_db_path",         &ServeOptions::rag_db_path)
        .def_rw("embed_model",         &ServeOptions::embed_model)
        .def_rw("rerank_model",        &ServeOptions::rerank_model)
        .def_rw("cache_embeddings",    &ServeOptions::cache_embeddings)
        // persistence / slots / lora / webui
        .def_rw("persist_chats",  &ServeOptions::persist_chats)
        .def_rw("chat_db_path",   &ServeOptions::chat_db_path)
        .def_rw("slot_save_path", &ServeOptions::slot_save_path)
        .def_rw("lora_adapters",  &ServeOptions::lora_adapters)
        .def_rw("webui",          &ServeOptions::webui)
        .def_rw("public_path",    &ServeOptions::public_path);

    nb::class_<chimera::Server>(m, "Server")
        .def(nb::init<ServeOptions>(), "options"_a)
        .def_prop_ro("options",
            [](chimera::Server &self) -> ServeOptions & { return self.options(); },
            nb::rv_policy::reference_internal)
        // run() blocks until SIGINT / programmatic stop -> release the GIL.
        .def("run", &chimera::Server::run, nb::call_guard<nb::gil_scoped_release>());

#ifdef CHIMERA_HAS_SD
    // ====================================================================
    // SD
    // ====================================================================
    nb::class_<chimera_sd::PixelImage>(m, "PixelImage")
        .def_ro("width",    &chimera_sd::PixelImage::width)
        .def_ro("height",   &chimera_sd::PixelImage::height)
        .def_ro("channels", &chimera_sd::PixelImage::channels)
        // bytes view (size == width*height*channels). For numpy, switch to
        // nb::ndarray<uint8_t> shaped (height, width, channels).
        .def_prop_ro("pixels", [](const chimera_sd::PixelImage &img) {
            return nb::bytes(reinterpret_cast<const char *>(img.pixels.data()),
                             img.pixels.size());
        });

    nb::class_<SdOptions>(m, "SdOptions")
        .def(nb::init<>())
        // model files
        .def_rw("model",            &SdOptions::model)
        .def_rw("diffusion_model",  &SdOptions::diffusion_model)
        .def_rw("vae",              &SdOptions::vae)
        .def_rw("clip_l",           &SdOptions::clip_l)
        .def_rw("clip_g",           &SdOptions::clip_g)
        .def_rw("t5xxl",            &SdOptions::t5xxl)
        .def_rw("llm",              &SdOptions::llm)
        .def_rw("high_noise_diffusion_model", &SdOptions::high_noise_diffusion_model)
        .def_rw("control_net",      &SdOptions::control_net)
        .def_rw("wtype",            &SdOptions::wtype)   // f16|q8_0|...
        .def_rw("taesd",            &SdOptions::taesd)
        .def_rw("clip_vision",      &SdOptions::clip_vision)
        .def_rw("llm_vision",       &SdOptions::llm_vision)
        .def_rw("tensor_type_rules",&SdOptions::tensor_type_rules)
        .def_rw("photo_maker",      &SdOptions::photo_maker)
        .def_rw("embd_dir",         &SdOptions::embd_dir)
        // prompt / io
        .def_rw("prompt",           &SdOptions::prompt)
        .def_rw("negative_prompt",  &SdOptions::negative_prompt)
        .def_rw("output",           &SdOptions::output)
        .def_rw("init_image",       &SdOptions::init_image)
        .def_rw("mask_image",       &SdOptions::mask_image)
        .def_rw("control_image",    &SdOptions::control_image)
        // sampling
        .def_rw("sample_method",    &SdOptions::sample_method)  // string; see sd's str_to_sample_method
        .def_rw("scheduler",        &SdOptions::scheduler)      // string; see str_to_scheduler
        .def_rw("width",            &SdOptions::width)
        .def_rw("height",           &SdOptions::height)
        .def_rw("steps",            &SdOptions::steps)
        .def_rw("batch_count",      &SdOptions::batch_count)
        .def_rw("clip_skip",        &SdOptions::clip_skip)
        .def_rw("threads",          &SdOptions::threads)
        .def_rw("seed",             &SdOptions::seed)
        .def_rw("cfg_scale",        &SdOptions::cfg_scale)
        .def_rw("strength",         &SdOptions::strength)
        .def_rw("guidance",         &SdOptions::guidance)
        .def_rw("flow_shift",       &SdOptions::flow_shift)
        .def_rw("control_strength", &SdOptions::control_strength)
        .def_rw("img_cfg_scale",    &SdOptions::img_cfg_scale)
        .def_rw("eta",              &SdOptions::eta)
        .def_rw("shifted_timestep", &SdOptions::shifted_timestep)
        .def_rw("sigmas",           &SdOptions::sigmas)
        // skip-layer guidance (SLG)
        .def_rw("skip_layers",      &SdOptions::skip_layers)
        .def_rw("slg_scale",        &SdOptions::slg_scale)
        .def_rw("skip_layer_start", &SdOptions::skip_layer_start)
        .def_rw("skip_layer_end",   &SdOptions::skip_layer_end)
        // lora
        .def_rw("lora_model_dir",   &SdOptions::lora_model_dir)
        .def_rw("lora_adapters",    &SdOptions::lora_adapters)
        // backend / memory
        .def_rw("offload_to_cpu",        &SdOptions::offload_to_cpu)
        .def_rw("diffusion_fa",          &SdOptions::diffusion_fa)
        .def_rw("diffusion_conv_direct", &SdOptions::diffusion_conv_direct)
        .def_rw("vae_conv_direct",       &SdOptions::vae_conv_direct)
        .def_rw("rng",                   &SdOptions::rng)          // string
        .def_rw("sampler_rng",           &SdOptions::sampler_rng)  // string
        .def_rw("flash_attn_global",     &SdOptions::flash_attn_global)
        .def_rw("no_mmap",               &SdOptions::no_mmap)
        .def_rw("max_vram",              &SdOptions::max_vram)
        .def_rw("keep_clip_on_cpu",          &SdOptions::keep_clip_on_cpu)
        .def_rw("keep_vae_on_cpu",           &SdOptions::keep_vae_on_cpu)
        .def_rw("keep_control_net_on_cpu",   &SdOptions::keep_control_net_on_cpu)
        .def_rw("force_sdxl_vae_conv_scale", &SdOptions::force_sdxl_vae_conv_scale)
        .def_rw("prediction",            &SdOptions::prediction)       // string
        .def_rw("lora_apply_mode",       &SdOptions::lora_apply_mode)  // string
        // vae tiling
        .def_rw("vae_tiling",             &SdOptions::vae_tiling)
        .def_rw("vae_tile_size",          &SdOptions::vae_tile_size)
        .def_rw("vae_relative_tile_size", &SdOptions::vae_relative_tile_size)
        .def_rw("vae_tile_overlap",       &SdOptions::vae_tile_overlap)
        // photo-maker
        .def_rw("pm_id_images_dir",  &SdOptions::pm_id_images_dir)
        .def_rw("pm_id_embed_path",  &SdOptions::pm_id_embed_path)
        .def_rw("pm_style_strength", &SdOptions::pm_style_strength)
        // reference images
        .def_rw("ref_images",               &SdOptions::ref_images)
        .def_rw("increase_ref_index",       &SdOptions::increase_ref_index)
        .def_rw("no_auto_resize_ref_image", &SdOptions::no_auto_resize_ref_image)
        // hires fix
        .def_rw("hires_fix",                &SdOptions::hires_fix)
        .def_rw("hires_upscaler",           &SdOptions::hires_upscaler)  // Latent|Lanczos|Model
        .def_rw("hires_upscale_model",      &SdOptions::hires_upscale_model)
        .def_rw("hires_width",              &SdOptions::hires_width)
        .def_rw("hires_height",             &SdOptions::hires_height)
        .def_rw("hires_scale",              &SdOptions::hires_scale)
        .def_rw("hires_steps",              &SdOptions::hires_steps)
        .def_rw("hires_denoising_strength", &SdOptions::hires_denoising_strength)
        .def_rw("hires_upscale_tile_size",  &SdOptions::hires_upscale_tile_size)
        // cache (sample/spectrum cache)
        .def_rw("cache_mode",   &SdOptions::cache_mode)
        .def_rw("cache_option", &SdOptions::cache_option)
        .def_rw("scm_mask",     &SdOptions::scm_mask)
        .def_rw("scm_policy",   &SdOptions::scm_policy);

    nb::class_<chimera::SD>(m, "SD")
        .def(nb::init<std::string>(), "model_path"_a)
        .def(nb::init<SdOptions>(),   "options"_a)
        .def_prop_ro("options",
            [](chimera::SD &self) -> SdOptions & { return self.options(); },
            nb::rv_policy::reference_internal)
        // -> list[PixelImage]
        .def("generate",
            [](chimera::SD &self, const std::string &prompt) {
                return self.generate(prompt);
            },
            nb::call_guard<nb::gil_scoped_release>(), "prompt"_a)
        .def("reset", &chimera::SD::reset, "reload"_a = false);
#endif // CHIMERA_HAS_SD

#ifdef CHIMERA_HAS_WHISPER
    // ====================================================================
    // Whisper
    // ====================================================================
    nb::class_<chimera_whisper::Segment>(m, "Segment")
        .def_ro("t0",      &chimera_whisper::Segment::t0)
        .def_ro("t1",      &chimera_whisper::Segment::t1)
        .def_ro("text",    &chimera_whisper::Segment::text)
        .def_ro("speaker", &chimera_whisper::Segment::speaker);

    nb::class_<chimera_whisper::TranscribeResult>(m, "TranscribeResult")
        .def_ro("text",              &chimera_whisper::TranscribeResult::text)
        .def_ro("segments",          &chimera_whisper::TranscribeResult::segments)
        .def_ro("detected_language", &chimera_whisper::TranscribeResult::detected_language)
        .def_ro("audio_duration_s",  &chimera_whisper::TranscribeResult::audio_duration_s);

    nb::class_<WhisperOptions>(m, "WhisperOptions")
        .def(nb::init<>())
        .def_rw("model",          &WhisperOptions::model)
        .def_rw("input",          &WhisperOptions::input)
        .def_rw("output",         &WhisperOptions::output)
        .def_rw("language",       &WhisperOptions::language)
        .def_rw("threads",        &WhisperOptions::threads)
        .def_rw("translate",      &WhisperOptions::translate)
        .def_rw("timestamps",     &WhisperOptions::timestamps)
        .def_rw("no_context",     &WhisperOptions::no_context)
        .def_rw("initial_prompt", &WhisperOptions::initial_prompt)
        .def_rw("carry_initial_prompt", &WhisperOptions::carry_initial_prompt)
        .def_rw("beam_size",      &WhisperOptions::beam_size)
        .def_rw("best_of",        &WhisperOptions::best_of)
        .def_rw("temperature",    &WhisperOptions::temperature)
        .def_rw("no_fallback",    &WhisperOptions::no_fallback)
        // output-file formats
        .def_rw("output_file_base", &WhisperOptions::output_file_base)
        .def_rw("out_txt",        &WhisperOptions::out_txt)
        .def_rw("out_srt",        &WhisperOptions::out_srt)
        .def_rw("out_vtt",        &WhisperOptions::out_vtt)
        .def_rw("out_json",       &WhisperOptions::out_json)
        .def_rw("out_json_full",  &WhisperOptions::out_json_full)
        .def_rw("out_csv",        &WhisperOptions::out_csv)
        .def_rw("out_lrc",        &WhisperOptions::out_lrc)
        // segmentation / offsets
        .def_rw("offset_ms",      &WhisperOptions::offset_ms)
        .def_rw("duration_ms",    &WhisperOptions::duration_ms)
        .def_rw("max_len",        &WhisperOptions::max_len)
        .def_rw("max_tokens",     &WhisperOptions::max_tokens)
        .def_rw("split_on_word",  &WhisperOptions::split_on_word)
        // decoding thresholds
        .def_rw("temperature_inc",&WhisperOptions::temperature_inc)
        .def_rw("entropy_thold",  &WhisperOptions::entropy_thold)
        .def_rw("logprob_thold",  &WhisperOptions::logprob_thold)
        .def_rw("no_speech_thold",&WhisperOptions::no_speech_thold)
        .def_rw("audio_ctx",      &WhisperOptions::audio_ctx)
        // VAD
        .def_rw("vad",            &WhisperOptions::vad)
        .def_rw("vad_model",      &WhisperOptions::vad_model)
        .def_rw("vad_threshold",  &WhisperOptions::vad_threshold)
        .def_rw("vad_min_speech_duration_ms",  &WhisperOptions::vad_min_speech_duration_ms)
        .def_rw("vad_min_silence_duration_ms", &WhisperOptions::vad_min_silence_duration_ms)
        .def_rw("vad_max_speech_duration_s",   &WhisperOptions::vad_max_speech_duration_s)
        .def_rw("vad_speech_pad_ms",           &WhisperOptions::vad_speech_pad_ms)
        .def_rw("vad_samples_overlap",         &WhisperOptions::vad_samples_overlap)
        // misc
        .def_rw("tinydiarize",    &WhisperOptions::tinydiarize)
        .def_rw("suppress_regex", &WhisperOptions::suppress_regex)
        .def_rw("suppress_nst",   &WhisperOptions::suppress_nst)
        .def_rw("flash_attn",     &WhisperOptions::flash_attn)
        .def_rw("no_gpu",         &WhisperOptions::no_gpu)
        .def_rw("gpu_device",     &WhisperOptions::gpu_device)
        .def_rw("processors",     &WhisperOptions::processors)
        // grammar
        .def_rw("grammar",        &WhisperOptions::grammar)
        .def_rw("grammar_file",   &WhisperOptions::grammar_file)
        .def_rw("grammar_rule",   &WhisperOptions::grammar_rule)
        .def_rw("grammar_penalty",&WhisperOptions::grammar_penalty)
        .def_rw("diarize",        &WhisperOptions::diarize)
        .def_rw("detect_language",&WhisperOptions::detect_language);

    nb::class_<chimera::Whisper>(m, "Whisper")
        .def(nb::init<std::string>(),    "model_path"_a)
        .def(nb::init<WhisperOptions>(), "options"_a)
        .def_prop_ro("options",
            [](chimera::Whisper &self) -> WhisperOptions & { return self.options(); },
            nb::rv_policy::reference_internal)
        .def("transcribe",
            [](chimera::Whisper &self, const std::string &wav_path) {
                return self.transcribe(wav_path);
            },
            nb::call_guard<nb::gil_scoped_release>(), "wav_path"_a)
        .def("reset", &chimera::Whisper::reset, "reload"_a = false);
#endif // CHIMERA_HAS_WHISPER
}
