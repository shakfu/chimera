// Library implementation of the llama.cpp glue declared in chimera_llama.h.
// Holds the model loader, context builder, sampler, grammar / LoRA / decode
// helpers, the two run_generation flavors (text-only and mtmd-vision), and
// the three CLI entrypoints that depend only on this glue (command_prompt,
// command_embed, command_tokenize). The chat REPL and its terminal-streaming
// pieces remain in src/chimera_cli/chimera.cpp.

#include "chimera_llama.h"

#include "chimera_embed.h"
#include "chimera_embed_cache.h"
#include "chimera_db.h"
#include "chimera_llama_load_mode.h"
#include "chimera.h"

#include "common.h"
#include "chat.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// ---- smart-pointer deleters -------------------------------------------------

void LlamaModelDeleter::operator()(llama_model * model) const {
    if (model != nullptr) {
        llama_model_free(model);
    }
}

void LlamaContextDeleter::operator()(llama_context * ctx) const {
    if (ctx != nullptr) {
        llama_free(ctx);
    }
}

// ---- file I/O --------------------------------------------------------------

std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail(ExitCode::BadInput, "failed to open file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string resolve_prompt(const std::string & prompt, const std::string & prompt_file) {
    const bool have_prompt = !prompt.empty();
    const bool have_file   = !prompt_file.empty();
    if (have_prompt && have_file) {
        fail(ExitCode::BadInput, "use only one of --prompt / --prompt-file");
    }
    if (have_prompt) {
        return prompt;
    }
    if (have_file) {
        if (prompt_file == "-") {
            std::ostringstream ss;
            ss << std::cin.rdbuf();
            return ss.str();
        }
        return read_file(prompt_file);
    }
    fail(ExitCode::BadInput, "either --prompt or --prompt-file is required");
}

// ---- tokenization ----------------------------------------------------------

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special, bool parse_special) {
    std::vector<llama_token> tokens(text.size() + 8);
    int32_t n = llama_tokenize(
        vocab,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_special,
        parse_special);
    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(
            vocab,
            text.c_str(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            add_special,
            parse_special);
    }
    if (n < 0) {
        fail("failed to tokenize prompt");
    }
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

std::string token_to_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buf(32);
    int32_t n = llama_token_to_piece(vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
    if (n < 0) {
        buf.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
    }
    if (n < 0) {
        fail("failed to convert token to piece");
    }
    return std::string(buf.data(), static_cast<size_t>(n));
}

// ---- internal helpers (anonymous namespace) --------------------------------

namespace {

// Parse a "path[:scale]" LoRA spec. Scale defaults to 1.0 when omitted.
// Mirrors the parser in chimera_serve.cpp so both code paths accept the
// same syntax (and tolerate Windows "C:\..." paths with no trailing
// numeric suffix).
struct LoraSpec { std::string path; float scale; };
LoraSpec parse_lora_spec(const std::string & spec) {
    LoraSpec out{spec, 1.0f};
    const auto colon = spec.find_last_of(':');
    if (colon != std::string::npos) {
        try {
            out.scale = std::stof(spec.substr(colon + 1));
            out.path  = spec.substr(0, colon);
        } catch (const std::exception &) {
            out.path = spec;
        }
    }
    return out;
}

// Map a string like "f16" / "q8_0" to the matching ggml type enum.
// Returns false on unknown values; caller decides whether to fail.
bool parse_cache_type(const std::string & name, ggml_type & out) {
    static const std::vector<std::pair<std::string, ggml_type>> table = {
        {"f32",    GGML_TYPE_F32},
        {"f16",    GGML_TYPE_F16},
        {"bf16",   GGML_TYPE_BF16},
        {"q8_0",   GGML_TYPE_Q8_0},
        {"q5_0",   GGML_TYPE_Q5_0},
        {"q5_1",   GGML_TYPE_Q5_1},
        {"q4_0",   GGML_TYPE_Q4_0},
        {"q4_1",   GGML_TYPE_Q4_1},
        {"iq4_nl", GGML_TYPE_IQ4_NL},
    };
    for (const auto & [k, v] : table) {
        if (k == name) { out = v; return true; }
    }
    return false;
}

llama_rope_scaling_type parse_rope_scaling(const std::string & name) {
    if (name.empty() || name == "unspecified") return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
    if (name == "none")     return LLAMA_ROPE_SCALING_TYPE_NONE;
    if (name == "linear")   return LLAMA_ROPE_SCALING_TYPE_LINEAR;
    if (name == "yarn")     return LLAMA_ROPE_SCALING_TYPE_YARN;
    if (name == "longrope") return LLAMA_ROPE_SCALING_TYPE_LONGROPE;
    fail(ExitCode::BadInput, "unknown --rope-scaling: " + name);
}

llama_split_mode parse_split_mode(const std::string & name) {
    if (name.empty() || name == "layer") return LLAMA_SPLIT_MODE_LAYER;
    if (name == "none")   return LLAMA_SPLIT_MODE_NONE;
    if (name == "row")    return LLAMA_SPLIT_MODE_ROW;
    if (name == "tensor") return LLAMA_SPLIT_MODE_TENSOR;
    fail(ExitCode::BadInput, "unknown --split-mode: " + name);
}

std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Parse a comma-separated float list (--tensor-split). Returns empty when
// the input is empty. Throws on malformed values.
std::vector<float> parse_float_csv(const std::string & s) {
    std::vector<float> out;
    for (const auto & tok : split_csv(s)) {
        try { out.push_back(std::stof(tok)); }
        catch (const std::exception &) {
            fail(ExitCode::BadInput, "invalid float in --tensor-split: '" + tok + "'");
        }
    }
    return out;
}

// Resolve a comma-separated device-name list to a NULL-terminated
// ggml_backend_dev_t array. Returns empty when input is empty.
std::vector<ggml_backend_dev_t> resolve_devices(const std::string & devices_csv) {
    std::vector<ggml_backend_dev_t> out;
    if (devices_csv.empty()) return out;
    for (const auto & name : split_csv(devices_csv)) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
        if (dev == nullptr) {
            fail(ExitCode::BadInput, "unknown device: '" + name + "'");
        }
        out.push_back(dev);
    }
    out.push_back(nullptr);
    return out;
}

struct ModelExtras {
    std::vector<ggml_backend_dev_t> devices_storage;
    std::vector<float>              tensor_split_storage;
    std::vector<std::string>                             buft_pattern_storage;
    std::vector<llama_model_tensor_buft_override>        buft_overrides_storage;
    std::vector<llama_model_kv_override>                 kv_overrides_storage;
};

std::map<std::string, ggml_backend_buffer_type_t> backend_buft_table() {
    ggml_backend_load_all();
    std::map<std::string, ggml_backend_buffer_type_t> out;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev  = ggml_backend_dev_get(i);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        if (buft) out[ggml_backend_buft_name(buft)] = buft;
    }
    return out;
}

void append_tensor_buft_override(ModelExtras & extras,
                                 const std::string & spec) {
    const auto eq = spec.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 == spec.size()) {
        fail(ExitCode::BadInput,
             "--override-tensor expects '<pattern>=<buft>', got: '" + spec + "'");
    }
    static thread_local std::map<std::string, ggml_backend_buffer_type_t> bufts;
    if (bufts.empty()) bufts = backend_buft_table();
    const std::string pat   = spec.substr(0, eq);
    const std::string bname = spec.substr(eq + 1);
    auto it = bufts.find(bname);
    if (it == bufts.end()) {
        std::string known;
        for (const auto & kv : bufts) {
            if (!known.empty()) known += ", ";
            known += kv.first;
        }
        fail(ExitCode::BadInput,
             "--override-tensor: unknown buffer type '" + bname +
             "' (available: " + known + ")");
    }
    extras.buft_pattern_storage.push_back(pat);
    llama_model_tensor_buft_override o{};
    o.pattern = extras.buft_pattern_storage.back().c_str();
    o.buft    = it->second;
    extras.buft_overrides_storage.push_back(o);
}

void apply_model_common(llama_model_params & params,
                        ModelExtras & extras,
                        bool use_mmap,
                        bool use_mlock,
                        const std::string & load_mode_name,
                        int main_gpu,
                        const std::string & tensor_split_csv,
                        const std::string & split_mode_name,
                        const std::string & devices_csv,
                        bool cpu_moe,
                        int  n_cpu_moe,
                        const std::vector<std::string> & override_tensor,
                        const std::vector<std::string> & override_kv) {
    params.load_mode = chimera_llama_load_mode(load_mode_name, use_mmap, use_mlock);
    params.main_gpu  = main_gpu;
    if (!split_mode_name.empty()) {
        params.split_mode = parse_split_mode(split_mode_name);
    }
    if (!tensor_split_csv.empty()) {
        extras.tensor_split_storage = parse_float_csv(tensor_split_csv);
        if (!extras.tensor_split_storage.empty()) {
            params.tensor_split = extras.tensor_split_storage.data();
        }
    }
    if (!devices_csv.empty()) {
        extras.devices_storage = resolve_devices(devices_csv);
        if (!extras.devices_storage.empty()) {
            params.devices = extras.devices_storage.data();
        }
    }

    if (cpu_moe) {
        auto o = llm_ffn_exps_cpu_override();
        extras.buft_overrides_storage.push_back(o);
    }
    for (int i = 0; i < n_cpu_moe; ++i) {
        extras.buft_pattern_storage.push_back(llm_ffn_exps_block_regex(i));
        llama_model_tensor_buft_override o{};
        o.pattern = extras.buft_pattern_storage.back().c_str();
        o.buft    = ggml_backend_cpu_buffer_type();
        extras.buft_overrides_storage.push_back(o);
    }
    for (const auto & spec : override_tensor) {
        for (const auto & item : split_csv(spec)) {
            append_tensor_buft_override(extras, item);
        }
    }
    if (!extras.buft_overrides_storage.empty()) {
        llama_model_tensor_buft_override sentinel{};
        sentinel.pattern = nullptr;
        extras.buft_overrides_storage.push_back(sentinel);
        params.tensor_buft_overrides = extras.buft_overrides_storage.data();
    }

    for (const auto & spec : override_kv) {
        for (const auto & item : split_csv(spec)) {
            if (!string_parse_kv_override(item.c_str(), extras.kv_overrides_storage)) {
                fail(ExitCode::BadInput,
                     "--override-kv: invalid entry '" + item +
                     "' (expected KEY=TYPE:VALUE with TYPE in int/float/bool/str)");
            }
        }
    }
    if (!extras.kv_overrides_storage.empty()) {
        llama_model_kv_override term{};
        term.key[0] = 0;
        extras.kv_overrides_storage.push_back(term);
        params.kv_overrides = extras.kv_overrides_storage.data();
    }
}

bool parse_logit_bias(const std::string & spec, llama_logit_bias & out) {
    size_t i = 0;
    if (i < spec.size() && (spec[i] == '+' || spec[i] == '-')) ++i;
    while (i < spec.size() && std::isdigit(static_cast<unsigned char>(spec[i]))) ++i;
    if (i == 0 || i >= spec.size()) return false;
    const char op = spec[i];
    if (op != '+' && op != '-' && op != '=') return false;
    try {
        out.token = static_cast<llama_token>(std::stoi(spec.substr(0, i)));
        const std::string rest = spec.substr(i + 1);
        const float v = rest.empty() ? 0.0f : std::stof(rest);
        if (op == '+') out.bias = +v;
        else if (op == '-') out.bias = -v;
        else                out.bias = v;
    } catch (const std::exception &) {
        return false;
    }
    return true;
}

// Read grammar from file or string. Mutually exclusive — caller checked.
std::string resolve_grammar(const LlamaCommonOptions & opts) {
    int set = 0;
    set += !opts.grammar.empty();
    set += !opts.grammar_file.empty();
    set += !opts.json_schema.empty();
    set += !opts.json_schema_file.empty();
    if (set > 1) {
        fail(ExitCode::BadInput,
             "use only one of --grammar / --grammar-file / "
             "--json-schema / --json-schema-file");
    }
    if (!opts.grammar.empty()) return opts.grammar;
    if (!opts.grammar_file.empty()) return read_file(opts.grammar_file);
    std::string schema_text;
    if (!opts.json_schema.empty()) schema_text = opts.json_schema;
    else if (!opts.json_schema_file.empty()) schema_text = read_file(opts.json_schema_file);
    if (schema_text.empty()) return {};
    try {
        return json_schema_to_grammar(common_json::parse(schema_text));
    } catch (const std::exception & e) {
        fail(ExitCode::BadInput,
             std::string("failed to convert --json-schema to grammar: ") + e.what());
    }
    return {};
}

} // anonymous namespace

// Sample up to n_predict tokens from a pre-filled context. Returns the
// generated text and (out) the generated token sequence (caller-owned).
// Public because chimera::Llama's persistent-context generate() drives
// its own prompt-decode + sample cycle and needs this helper.
std::string sample_loop(
    llama_context * ctx,
    common_sampler * sampler,
    const llama_vocab * vocab,
    int n_predict,
    const TokenCallback & on_token,
    std::vector<llama_token> * out_tokens) {

    std::string text;
    for (int i = 0; i < n_predict; ++i) {
        const llama_token token = common_sampler_sample(sampler, ctx, -1, false);
        if (token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, token)) {
            break;
        }
        common_sampler_accept(sampler, token, true);
        if (out_tokens) {
            out_tokens->push_back(token);
        }

        const std::string piece = token_to_piece(vocab, token);
        text += piece;
        if (on_token) {
            on_token(std::string_view(piece));
        }

        llama_token token_copy = token;
        if (llama_decode(ctx, llama_batch_get_one(&token_copy, 1)) != 0) {
            fail(ExitCode::Generate, "failed to decode generated token");
        }
    }
    // Trailing newline / flush is the caller's responsibility now -- a
    // library consumer driving a UI or pipe shouldn't have stdout state
    // changed under them.
    return text;
}

// ---- public model + context loaders ---------------------------------------

LlamaModelPtr load_llama_model(const LlamaCommonOptions & opts) {
    if (!opts.model.empty()) {
        std::error_code ec;
        const auto status = std::filesystem::status(opts.model, ec);
        if (ec || !std::filesystem::exists(status)) {
            fail(ExitCode::Load,
                 "model file not found: " + opts.model +
                 " (check the path; chimera does not auto-download models)");
        }
        if (std::filesystem::is_directory(status)) {
            fail(ExitCode::Load,
                 "model path is a directory, not a file: " + opts.model +
                 " (point -m at a single .gguf file)");
        }
    }

    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = opts.gpu_layers;
    ModelExtras extras;
    apply_model_common(params, extras,
                       opts.use_mmap, opts.use_mlock, opts.load_mode,
                       opts.main_gpu, opts.tensor_split,
                       opts.split_mode, opts.devices,
                       opts.cpu_moe, opts.n_cpu_moe,
                       opts.override_tensor, opts.override_kv);

    LlamaModelPtr model(llama_model_load_from_file(opts.model.c_str(), params));
    if (!model) {
        fail(ExitCode::Load,
             "failed to load llama model: " + opts.model +
             " (file exists but llama.cpp could not parse it - "
             "wrong format, corrupt GGUF, or version mismatch)");
    }
    return model;
}

LlamaContextPtr new_llama_context(llama_model * model, const LlamaCommonOptions & opts, size_t min_prompt_tokens) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = std::max<uint32_t>(opts.n_ctx, static_cast<uint32_t>(min_prompt_tokens + opts.n_predict + 32));
    params.n_batch = std::max<uint32_t>(1, opts.n_batch);
    params.n_ubatch = opts.n_ubatch > 0 ? opts.n_ubatch : params.n_batch;
    params.n_threads = opts.threads;
    params.n_threads_batch = opts.threads_batch > 0 ? opts.threads_batch : opts.threads;
    params.no_perf = true;
    if (opts.swa_full) {
        params.swa_full = true;
    }

    if (opts.flash_attn) {
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    if (!opts.cache_type_k.empty()) {
        ggml_type t;
        if (!parse_cache_type(opts.cache_type_k, t)) {
            fail(ExitCode::BadInput, "unknown --cache-type-k: " + opts.cache_type_k);
        }
        params.type_k = t;
    }
    if (!opts.cache_type_v.empty()) {
        ggml_type t;
        if (!parse_cache_type(opts.cache_type_v, t)) {
            fail(ExitCode::BadInput, "unknown --cache-type-v: " + opts.cache_type_v);
        }
        params.type_v = t;
    }
    if (opts.rope_freq_base  > 0.0f) params.rope_freq_base  = opts.rope_freq_base;
    if (opts.rope_freq_scale > 0.0f) params.rope_freq_scale = opts.rope_freq_scale;
    if (!opts.rope_scaling.empty()) {
        params.rope_scaling_type = parse_rope_scaling(opts.rope_scaling);
    }
    if (opts.yarn_orig_ctx > 0) params.yarn_orig_ctx = opts.yarn_orig_ctx;
    params.yarn_ext_factor  = opts.yarn_ext_factor;
    params.yarn_attn_factor = opts.yarn_attn_factor;
    params.yarn_beta_fast   = opts.yarn_beta_fast;
    params.yarn_beta_slow   = opts.yarn_beta_slow;

    LlamaContextPtr ctx(llama_init_from_model(model, params));
    if (!ctx) {
        fail(ExitCode::Load, "failed to create llama context");
    }

    // Activation steering. --control-vector takes plain paths (scale=1.0);
    // --control-vector-scaled takes "path:scale" entries.
    if (!opts.control_vector.empty() || !opts.control_vector_scaled.empty()) {
        std::vector<common_control_vector_load_info> load_infos;
        for (const auto & raw : opts.control_vector) {
            for (const auto & item : split_csv(raw)) {
                if (!item.empty()) load_infos.push_back({1.0f, item});
            }
        }
        for (const auto & raw : opts.control_vector_scaled) {
            for (const auto & item : split_csv(raw)) {
                const auto colon = item.rfind(':');
                if (colon == std::string::npos) {
                    fail(ExitCode::BadInput,
                         "--control-vector-scaled expects 'path:scale', got: '" + item + "'");
                }
                try {
                    const float scale = std::stof(item.substr(colon + 1));
                    load_infos.push_back({scale, item.substr(0, colon)});
                } catch (const std::exception &) {
                    fail(ExitCode::BadInput,
                         "--control-vector-scaled: invalid scale in '" + item + "'");
                }
            }
        }
        const auto cvec = common_control_vector_load(load_infos);
        if (cvec.n_embd == -1) {
            fail(ExitCode::Load, "failed to load control vectors");
        }
        const int layer_start = opts.control_vector_layer_start > 0
            ? opts.control_vector_layer_start : 1;
        const int layer_end   = opts.control_vector_layer_end   > 0
            ? opts.control_vector_layer_end   : llama_model_n_layer(model);
        if (llama_set_adapter_cvec(ctx.get(),
                                    cvec.data.data(), cvec.data.size(),
                                    cvec.n_embd, layer_start, layer_end) != 0) {
            fail(ExitCode::Load, "llama_set_adapter_cvec failed");
        }
    }

    return ctx;
}

// ---- public LoRA, decoding, sampling --------------------------------------

LoraAdapters load_loras(llama_model * model, llama_context * ctx,
                        const std::vector<std::string> & specs) {
    LoraAdapters out;
    if (specs.empty()) return out;
    std::vector<llama_adapter_lora *> raw_ptrs;
    std::vector<float>                scales;
    raw_ptrs.reserve(specs.size());
    scales.reserve(specs.size());
    for (const auto & raw : specs) {
        const auto parsed = parse_lora_spec(raw);
        llama_adapter_lora * a = llama_adapter_lora_init(model, parsed.path.c_str());
        if (a == nullptr) {
            fail(ExitCode::Load, "failed to load LoRA adapter: " + parsed.path);
        }
        raw_ptrs.push_back(a);
        scales.push_back(parsed.scale);
        out.adapters.emplace_back(a);
    }
    if (llama_set_adapters_lora(ctx, raw_ptrs.data(), raw_ptrs.size(), scales.data()) != 0) {
        fail(ExitCode::Load, "failed to attach LoRA adapters to context");
    }
    return out;
}

void decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, int32_t n_batch) {
    if (tokens.empty()) {
        return;
    }
    const int32_t total = static_cast<int32_t>(tokens.size());
    const int32_t chunk = std::max<int32_t>(1, n_batch);
    for (int32_t off = 0; off < total; off += chunk) {
        const int32_t n = std::min<int32_t>(chunk, total - off);
        if (llama_decode(ctx,
                         llama_batch_get_one(const_cast<llama_token *>(tokens.data() + off), n)) != 0) {
            fail(ExitCode::Generate, "failed to decode prompt");
        }
    }
}

// llama.cpp v0.3.0 removed the "-1 = context size" sentinel from both
// llama_sampler_init_penalties() and llama_sampler_init_dry(): a negative
// window is now clamped with std::max(n, 0), and 0 is each sampler's
// *disabled* value. Upstream used to resolve -1 one layer up and changed
// both common_params_sampling defaults to 64 when it dropped the sentinel;
// chimera kept -1, which silently turned the DRY sampler off entirely.
// chimera still accepts -1 and resolves it here, so --repeat-last-n and
// --dry-penalty-last-n keep meaning what their help text says. The window
// resolves to the context chimera is about to create (opts.n_ctx), falling
// back to the model's training context when that is left at 0.
static int32_t resolve_penalty_window(int n, const llama_model * model, uint32_t n_ctx) {
    if (n >= 0) {
        return static_cast<int32_t>(n);
    }
    if (n_ctx > 0) {
        return static_cast<int32_t>(n_ctx);
    }
    return llama_model_n_ctx_train(model);
}

common_sampler_ptr make_sampler(const llama_model *           model,
                                const LlamaCommonOptions &    opts,
                                const ReasoningBudgetParams & rbp) {
    common_params_sampling sampling;
    sampling.seed = opts.seed;
    sampling.top_k = opts.top_k;
    sampling.top_p = opts.top_p;
    sampling.min_p = opts.min_p;
    sampling.temp = opts.temp;
    sampling.penalty_repeat = opts.repeat_penalty;
    sampling.penalty_last_n  = resolve_penalty_window(opts.penalty_last_n, model, opts.n_ctx);
    sampling.penalty_present = opts.penalty_present;
    sampling.penalty_freq    = opts.penalty_freq;
    sampling.mirostat        = opts.mirostat;
    sampling.mirostat_tau    = opts.mirostat_tau;
    sampling.mirostat_eta    = opts.mirostat_eta;
    sampling.dry_multiplier     = opts.dry_multiplier;
    sampling.dry_base           = opts.dry_base;
    sampling.dry_allowed_length = opts.dry_allowed_length;
    sampling.dry_penalty_last_n = resolve_penalty_window(opts.dry_penalty_last_n, model, opts.n_ctx);
    if (!opts.dry_sequence_breakers.empty()) {
        sampling.dry_sequence_breakers = opts.dry_sequence_breakers;
    }
    sampling.ignore_eos = opts.ignore_eos;
    sampling.no_perf = true;
    sampling.typ_p             = opts.typ_p;
    sampling.top_n_sigma       = opts.top_n_sigma;
    sampling.xtc_probability   = opts.xtc_probability;
    sampling.xtc_threshold     = opts.xtc_threshold;
    sampling.dynatemp_range    = opts.dynatemp_range;
    sampling.dynatemp_exponent = opts.dynatemp_exp;
    if (!opts.samplers.empty()) {
        std::vector<std::string> names;
        size_t pos = 0;
        while (pos < opts.samplers.size()) {
            size_t sc = opts.samplers.find(';', pos);
            names.push_back(opts.samplers.substr(pos, sc == std::string::npos ? std::string::npos : sc - pos));
            if (sc == std::string::npos) break;
            pos = sc + 1;
        }
        sampling.samplers = common_sampler_types_from_names(names);
        if (sampling.samplers.empty()) {
            fail(ExitCode::BadInput,
                 "--samplers parsed to an empty chain from value: '" + opts.samplers + "'");
        }
    }

    for (const auto & spec : opts.logit_bias) {
        llama_logit_bias lb{};
        if (!parse_logit_bias(spec, lb)) {
            fail(ExitCode::BadInput, "invalid --logit-bias: '" + spec + "'");
        }
        sampling.logit_bias.push_back(lb);
    }

    const std::string grammar_str = resolve_grammar(opts);
    if (!grammar_str.empty()) {
        const bool from_schema = !opts.json_schema.empty() || !opts.json_schema_file.empty();
        sampling.grammar = common_grammar(
            from_schema ? COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT
                        : COMMON_GRAMMAR_TYPE_USER,
            grammar_str);
    }

    // A fixed token budget (>= 0) and runtime control (Ctrl-C ends thinking)
    // share the same reasoning-budget sampler; either one arms it. Upstream
    // requires the start+end token sequences to be present for the sampler
    // to be created at all (see common/sampling.cpp), so populate them
    // whenever either path is active. With control-only and no token budget,
    // reasoning_budget_tokens stays -1 (upstream treats that as INT_MAX), so
    // the sampler never auto-fires — only the runtime force ends thinking.
    if ((rbp.budget >= 0 || rbp.control) && rbp.vocab != nullptr &&
        !rbp.thinking_start_tag.empty() && !rbp.thinking_end_tags.empty()) {
        sampling.reasoning_budget_tokens = rbp.budget;
        sampling.reasoning_control       = rbp.control;
        sampling.reasoning_budget_start =
            common_tokenize(rbp.vocab, rbp.thinking_start_tag, /*add_special=*/false, /*parse_special=*/true);
        // Upstream stops on any advertised end tag; the first also forms the
        // forced sequence.
        sampling.reasoning_budget_end.clear();
        for (const std::string & tag : rbp.thinking_end_tags) {
            sampling.reasoning_budget_end.push_back(
                common_tokenize(rbp.vocab, tag, false, true));
        }
        sampling.reasoning_budget_forced =
            common_tokenize(rbp.vocab,
                            rbp.budget_message + rbp.thinking_end_tags.front(),
                            false, true);
    }

    common_sampler * sampler = common_sampler_init(model, sampling);
    if (sampler == nullptr) {
        fail(ExitCode::Load, "failed to create sampler");
    }
    return common_sampler_ptr(sampler);
}

// ---- chat-history helper --------------------------------------------------

common_chat_msg make_chat_msg(const std::string & role, const std::string & content) {
    common_chat_msg msg;
    msg.role = role;
    msg.content = content;
    return msg;
}

// ---- generation -----------------------------------------------------------

void MtmdContextDeleter::operator()(mtmd_context * c) const {
    if (c) mtmd_free(c);
}
void MtmdBitmapDeleter::operator()(mtmd_bitmap * b) const {
    if (b) mtmd_bitmap_free(b);
}
void MtmdInputChunksDeleter::operator()(mtmd_input_chunks * c) const {
    if (c) mtmd_input_chunks_free(c);
}
void MtmdVideoDeleter::operator()(mtmd_helper_video * v) const {
    if (v) mtmd_helper_video_free(v);
}

ChimeraVideoBitmap load_video_lazy_bitmap(
    mtmd_context * mctx, const std::string & path,
    float fps_target, int64_t timestamp_ms, const std::string & ffmpeg_dir) {
    mtmd_helper_video_init_params params = mtmd_helper_video_init_params_default();
    params.fps_target            = fps_target;
    params.timestamp_interval_ms = timestamp_ms;
    // ffmpeg_bin_dir is read only during _video_init (it resolves the binaries
    // into owned strings there), so a pointer into `ffmpeg_dir` is safe.
    if (!ffmpeg_dir.empty()) {
        params.ffmpeg_bin_dir = ffmpeg_dir.c_str();
    }
    mtmd_helper_video * vctx = mtmd_helper_video_init(mctx, path.c_str(), params);
    if (!vctx) {
        return {};
    }
    mtmd_bitmap * bmp = mtmd_bitmap_init_lazy(
        mctx, /*id=*/nullptr, /*user_data=*/vctx,
        [](size_t, void * user_data, mtmd_bitmap ** out_bitmap, char ** out_text) -> int {
            auto * v = static_cast<mtmd_helper_video *>(user_data);
            char * text = nullptr;
            int ret = mtmd_helper_video_read_next(v, out_bitmap, &text);
            *out_text = text;  // heap-allocated by read_next; freed by mtmd
            return ret;
        });
    if (!bmp) {
        mtmd_helper_video_free(vctx);
        return {};
    }
    return { bmp, vctx };
}

std::string run_generation_mtmd(
    llama_model * model,
    const LlamaCommonOptions & opts,
    const std::string & user_prompt,
    const TokenCallback & on_token) {

    if (opts.mmproj.empty() || (opts.images.empty() && opts.videos.empty())) {
        fail(ExitCode::Runtime, "run_generation_mtmd called without mmproj/media");
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu       = opts.mmproj_use_gpu;
    mparams.n_threads     = opts.threads;
    mparams.print_timings = false;
    if (opts.flash_attn) {
        mparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    if (opts.image_min_tokens > 0) mparams.image_min_tokens = opts.image_min_tokens;
    if (opts.image_max_tokens > 0) mparams.image_max_tokens = opts.image_max_tokens;

    MtmdContextPtr mctx(mtmd_init_from_file(opts.mmproj.c_str(), model, mparams));
    if (!mctx) {
        fail(ExitCode::Load, "failed to load mmproj: " + opts.mmproj);
    }
    if (!mtmd_support_vision(mctx.get())) {
        fail(ExitCode::Load, "mmproj does not support vision input");
    }
    if (!opts.videos.empty() && !mtmd_helper_support_video(mctx.get())) {
        fail(ExitCode::BadInput,
             "--video requires a build with video support (MTMD_VIDEO) and a "
             "vision-capable mmproj");
    }

    std::vector<MtmdBitmapPtr> bitmaps_owned;
    std::vector<const mtmd_bitmap *> bitmaps_c;
    // Video inputs decode to a lazy bitmap backed by a video_ctx; the frame
    // callback reads from that ctx during mtmd_tokenize() below, so the ctx must
    // stay alive until at least then. RAII-own it here (freed at function scope).
    std::vector<MtmdVideoPtr> videos_owned;
    const size_t n_media = opts.images.size() + opts.videos.size();
    bitmaps_owned.reserve(n_media);
    bitmaps_c.reserve(n_media);
    for (const std::string & path : opts.images) {
        // Upstream returns a wrapper {bitmap, video_ctx}; placeholder=false so the
        // bitmap holds real data. video_ctx is non-null only for video files.
        mtmd_helper_bitmap_wrapper w =
            mtmd_helper_bitmap_init_from_file(mctx.get(), path.c_str(), /*placeholder=*/false);
        MtmdBitmapPtr bmp(w.bitmap);
        if (!bmp) {
            fail(ExitCode::BadInput, "failed to load image: " + path);
        }
        if (w.video_ctx) {
            videos_owned.emplace_back(w.video_ctx);
        }
        bitmaps_c.push_back(bmp.get());
        bitmaps_owned.push_back(std::move(bmp));
    }
    // Explicit --video inputs: always the video decoder, honoring --video-* params.
    for (const std::string & path : opts.videos) {
        ChimeraVideoBitmap v = load_video_lazy_bitmap(
            mctx.get(), path, opts.video_fps, opts.video_timestamp_ms, opts.ffmpeg_dir);
        MtmdBitmapPtr bmp(v.bitmap);
        if (!bmp) {
            fail(ExitCode::BadInput, "failed to load video: " + path);
        }
        videos_owned.emplace_back(v.video);
        bitmaps_c.push_back(bmp.get());
        bitmaps_owned.push_back(std::move(bmp));
    }

    const char * marker = mtmd_default_marker();
    std::string augmented_prompt;
    if (user_prompt.find(marker) == std::string::npos) {
        for (size_t i = 0; i < n_media; ++i) {
            augmented_prompt += marker;
            augmented_prompt += '\n';
        }
    }
    augmented_prompt += user_prompt;

    std::string final_prompt = augmented_prompt;
    bool prompt_is_templated = false;
    common_chat_templates_ptr templates = common_chat_templates_init(model, "", "", "");
    if (templates) {
        common_chat_msg msg;
        msg.role = "user";
        msg.content = augmented_prompt;
        common_chat_templates_inputs inputs;
        inputs.messages = { msg };
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;
        common_chat_params cp = common_chat_templates_apply(templates.get(), inputs);
        if (!cp.prompt.empty()) {
            final_prompt = cp.prompt;
            prompt_is_templated = true;
        }
    }

    // text_len is load-bearing: newer mtmd bounds its media-marker scan by
    // this length rather than strlen(text). Leaving it unset makes
    // mtmd_tokenize see zero markers and fail rc=2 ("markers != bitmaps").
    mtmd_input_text input_text{};
    input_text.text = final_prompt.c_str();
    input_text.text_len = final_prompt.size();
    input_text.add_special = !prompt_is_templated;
    input_text.parse_special = true;

    MtmdInputChunksPtr chunks(mtmd_input_chunks_init());
    if (!chunks) {
        fail(ExitCode::Runtime, "failed to init mtmd input chunks");
    }
    const int32_t tok_rc = mtmd_tokenize(mctx.get(), chunks.get(), &input_text,
                                         bitmaps_c.data(), bitmaps_c.size());
    if (tok_rc != 0) {
        fail(ExitCode::BadInput,
             "mtmd_tokenize failed (rc=" + std::to_string(tok_rc) + ")");
    }

    const size_t mm_tokens = mtmd_helper_get_n_tokens(chunks.get());
    auto ctx = new_llama_context(model, opts, mm_tokens);
    auto loras = load_loras(model, ctx.get(), opts.lora_adapters);
    auto sampler = make_sampler(model, opts);

    llama_pos new_n_past = 0;
    const int32_t eval_rc = mtmd_helper_eval_chunks(
        mctx.get(), ctx.get(), chunks.get(),
        /*n_past=*/0, /*seq_id=*/0,
        static_cast<int32_t>(opts.n_batch),
        /*logits_last=*/true, &new_n_past);
    if (eval_rc != 0) {
        fail(ExitCode::Generate,
             "mtmd_helper_eval_chunks failed (rc=" + std::to_string(eval_rc) + ")");
    }

    return sample_loop(ctx.get(), sampler.get(),
                       llama_model_get_vocab(model),
                       opts.n_predict, on_token);
}

std::string run_generation(
    llama_model * model,
    const LlamaCommonOptions & opts,
    const std::string & prompt,
    bool add_special,
    const TokenCallback & on_token) {

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto prompt_tokens = tokenize(vocab, prompt, add_special, true);
    auto ctx = new_llama_context(model, opts, prompt_tokens.size());
    auto loras = load_loras(model, ctx.get(), opts.lora_adapters);
    auto sampler = make_sampler(model, opts);

    decode_tokens(ctx.get(), prompt_tokens, static_cast<int32_t>(opts.n_batch));
    for (llama_token token : prompt_tokens) {
        common_sampler_accept(sampler.get(), token, false);
    }
    return sample_loop(ctx.get(), sampler.get(), vocab, opts.n_predict, on_token);
}

// ---- command entrypoints --------------------------------------------------

int command_prompt(const LlamaCommonOptions & opts, const std::string & prompt) {
    if ((!opts.images.empty() || !opts.videos.empty()) && opts.mmproj.empty()) {
        fail(ExitCode::BadInput, "--image/--video requires --mmproj");
    }
    // The CLI's gen subcommand streams tokens to stdout. sample_loop no
    // longer manages the trailing newline (it stopped writing to cout
    // entirely when the bool stream flag became a callback); command_prompt
    // takes that responsibility instead so the prompt line ends cleanly.
    const TokenCallback stream_to_cout = [](std::string_view piece) {
        std::cout << piece << std::flush;
    };
    auto model = load_llama_model(opts);
    std::string text;
    if (!opts.images.empty() || !opts.videos.empty()) {
        text = run_generation_mtmd(model.get(), opts, prompt, stream_to_cout);
    } else {
        text = run_generation(model.get(), opts, prompt, /*add_special=*/true, stream_to_cout);
    }
    std::cout << '\n';
    return text.empty() ? static_cast<int>(ExitCode::Generate) : 0;
}

int command_tokenize(const TokenizeOptions & opts) {
    const std::string text = resolve_prompt(opts.input, opts.input_file);

    LlamaCommonOptions load_opts;
    load_opts.model = opts.model;
    load_opts.gpu_layers = 0;
    load_opts.use_mmap = opts.use_mmap;
    auto model = load_llama_model(load_opts);
    const llama_vocab * vocab = llama_model_get_vocab(model.get());

    const auto tokens = tokenize(vocab, text, opts.add_special, opts.parse_special);
    for (llama_token tok : tokens) {
        if (opts.show_pieces) {
            std::string piece = token_to_piece(vocab, tok);
            std::string escaped;
            escaped.reserve(piece.size());
            for (char c : piece) {
                switch (c) {
                    case '\n': escaped += "\\n"; break;
                    case '\r': escaped += "\\r"; break;
                    case '\t': escaped += "\\t"; break;
                    default:   escaped += c;     break;
                }
            }
            std::cout << tok << '\t' << escaped << '\n';
        } else {
            std::cout << tok << '\n';
        }
    }
    return 0;
}

int command_embed(const EmbedOptions & opts) {
    const std::string text = resolve_prompt(opts.input, opts.input_file);

    const std::string & fmt = opts.embd_output_format;
    if (!fmt.empty() && fmt != "array" && fmt != "json" && fmt != "raw") {
        fail(ExitCode::BadInput,
             "unknown --embd-output-format: '" + fmt +
             "' (expected '', 'array', 'json', or 'raw')");
    }

    chimera_embed::Config cfg;
    cfg.model      = opts.model;
    cfg.pooling    = opts.pooling;
    cfg.attention  = opts.attention;
    cfg.threads    = opts.threads;
    cfg.gpu_layers = opts.gpu_layers;
    cfg.n_ctx      = opts.n_ctx;
    cfg.n_batch    = opts.n_batch;
    cfg.n_ubatch   = opts.n_ubatch;
    cfg.normalize  = opts.normalize;
    cfg.use_mmap   = opts.use_mmap;
    cfg.use_mlock  = opts.use_mlock;
    cfg.load_mode  = opts.load_mode;
    cfg.flash_attn = opts.flash_attn;
    cfg.rope_freq_base   = opts.rope_freq_base;
    cfg.rope_freq_scale  = opts.rope_freq_scale;
    cfg.rope_scaling     = opts.rope_scaling;
    cfg.yarn_orig_ctx    = opts.yarn_orig_ctx;
    cfg.yarn_ext_factor  = opts.yarn_ext_factor;
    cfg.yarn_attn_factor = opts.yarn_attn_factor;
    cfg.yarn_beta_fast   = opts.yarn_beta_fast;
    cfg.yarn_beta_slow   = opts.yarn_beta_slow;
    cfg.main_gpu         = opts.main_gpu;
    cfg.tensor_split     = opts.tensor_split;
    cfg.split_mode       = opts.split_mode;
    cfg.devices          = opts.devices;
    chimera_embed::Embedder embedder(cfg);
    std::unique_ptr<chimera_embed_cache::Cache> ecache;
    if (opts.cache_embeddings) {
        const std::string mid = chimera_embed_cache::compute_model_id(opts.model);
        if (mid.empty()) {
            fail(ExitCode::BadInput,
                 "--cache-embeddings: cannot fingerprint embedding model "
                 "(unreadable file: " + opts.model + ")");
        }
        ecache = std::make_unique<chimera_embed_cache::Cache>(
            opts.cache_db.empty() ? chimera_db::default_path() : opts.cache_db, mid);
        embedder.set_cache(ecache.get());
    }

    std::vector<std::string> pieces;
    if (opts.embd_separator.empty()) {
        pieces.push_back(text);
    } else {
        const std::string & sep = opts.embd_separator;
        size_t start = 0;
        while (true) {
            const size_t hit = text.find(sep, start);
            if (hit == std::string::npos) {
                pieces.emplace_back(text.substr(start));
                break;
            }
            pieces.emplace_back(text.substr(start, hit - start));
            start = hit + sep.size();
        }
    }

    std::vector<std::vector<float>> vecs;
    vecs.reserve(pieces.size());
    for (const auto & p : pieces) {
        vecs.push_back(embedder.embed(p));
    }

    std::ofstream out_file;
    std::ostream * out = &std::cout;
    if (!opts.output.empty()) {
        out_file.open(opts.output);
        if (!out_file) {
            fail(ExitCode::BadInput, "failed to open output file: " + opts.output);
        }
        out = &out_file;
    }
    *out << std::setprecision(8);

    auto write_default = [&](const std::vector<float> & vec) {
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) *out << ' ';
            *out << vec[i];
        }
        *out << '\n';
    };
    auto write_array = [&](std::ostream & os, const std::vector<float> & vec) {
        os << '[';
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) os << ',';
            os << vec[i];
        }
        os << ']';
    };

    if (fmt.empty()) {
        for (const auto & v : vecs) write_default(v);
    } else if (fmt == "raw") {
        for (size_t i = 0; i < vecs.size(); ++i) {
            if (i > 0) *out << '\n';
            for (float v : vecs[i]) *out << v << '\n';
        }
    } else if (fmt == "array") {
        if (vecs.size() == 1) {
            write_array(*out, vecs[0]);
        } else {
            *out << '[';
            for (size_t i = 0; i < vecs.size(); ++i) {
                if (i > 0) *out << ',';
                write_array(*out, vecs[i]);
            }
            *out << ']';
        }
        *out << '\n';
    } else if (fmt == "json") {
        *out << "{\"object\":\"list\",\"data\":[";
        for (size_t i = 0; i < vecs.size(); ++i) {
            if (i > 0) *out << ',';
            *out << "{\"object\":\"embedding\",\"index\":" << i << ",\"embedding\":";
            write_array(*out, vecs[i]);
            *out << '}';
        }
        *out << "],\"model\":\"" << opts.model << "\"}\n";
    }
    return 0;
}
