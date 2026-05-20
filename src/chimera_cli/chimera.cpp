#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <csignal>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>  // isatty
#include <sys/ioctl.h>  // TIOCGWINSZ for terminal width in the chat banner
#else
#include <io.h>
// windows.h defines min/max as macros, which then collide with
// std::numeric_limits<T>::max() and std::min/max usages reached through
// llama.cpp's common.h and our own templates. Define NOMINMAX before the
// include so the macros are suppressed; WIN32_LEAN_AND_MEAN trims headers
// we don't need (winsock, GDI, ...) and shaves the TU's parse time.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>  // GetConsoleScreenBufferInfo for terminal width
#define isatty _isatty
#define STDIN_FILENO _fileno(stdin)
#define STDOUT_FILENO _fileno(stdout)
#define STDERR_FILENO _fileno(stderr)
#endif

#include "CLI11.hpp"
#include "rang.hpp"

#include "chat.h"
#include "chimera.h"
#include "json-schema-to-grammar.h"
#include "nlohmann/json.hpp"
#include "chimera_chat_store.h"
#include "chimera_db.h"
#include "chimera_embed.h"
#include "chimera_embed_cache.h"
#include "chimera_sd.h"
#include "chimera_vector_store.h"
#include "chimera_whisper.h"
#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"

#ifdef CHIMERA_HAS_LINENOISE
#include "linenoise.h"
#endif

namespace {

struct LlamaModelDeleter {
    void operator()(llama_model * model) const {
        if (model != nullptr) {
            llama_model_free(model);
        }
    }
};

struct LlamaContextDeleter {
    void operator()(llama_context * ctx) const {
        if (ctx != nullptr) {
            llama_free(ctx);
        }
    }
};

using LlamaModelPtr = std::unique_ptr<llama_model, LlamaModelDeleter>;
using LlamaContextPtr = std::unique_ptr<llama_context, LlamaContextDeleter>;

// Semantic color tags. Stream a Sem value to wash subsequent output in the
// color rang assigns to that role. All concrete color choices live in the
// one switch below, so re-skinning chat is a single-site edit.
enum class Sem {
    Reset,   // clear any active SGR
    User,    // the '> ' prompt + user-typed input (bold white)
    Title,   // "chimera" in the chat banner (bold cyan)
    Cmd,     // slash-command names in banner / help
    Think,   // model reasoning_content (between <think>...</think>)
    Stats,   // per-turn 'Prompt: X t/s | Generation: Y t/s' line
    Info,    // dim info notices ("attached text from ...", "history cleared")
    Err,     // errors
};

inline std::ostream & operator<<(std::ostream & os, Sem s) {
    switch (s) {
        case Sem::Reset: return os << rang::style::reset;
        // User input renders in bold white. We can't compose rang's
        // style::bold + fg::reset cleanly into a single ostream insertion
        // here (each emits its own SGR sequence and rang::fg::reset on
        // some terminals clears the bold attribute), so emit the SGR
        // directly: ESC [1;37m (bold + white foreground).
        case Sem::User:  return os << "\x1b[1;37m";
        // "chimera" in the banner: bold cyan.
        case Sem::Title: return os << "\x1b[1;36m";
        case Sem::Cmd:   return os << rang::fg::cyan;
        case Sem::Think: return os << rang::fg::gray;
        case Sem::Stats: return os << rang::fg::magenta;
        case Sem::Info:  return os << rang::style::dim;
        case Sem::Err:   return os << rang::fgB::red;
    }
    return os;
}

std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail(ExitCode::BadInput, "failed to open file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Resolve a prompt from either --prompt or --prompt-file. Exactly one must
// be supplied; both empty is a usage error. "-" reads stdin.
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

// CLI11's default formatter pads section breaks with double blank lines.
// This trims make_usage's trailing "\n\n" down to "\n", since the next
// section (OPTIONS / SUBCOMMANDS) already prepends its own '\n'.
struct CompactFormatter : public CLI::Formatter {
    std::string make_usage(const CLI::App * app, std::string name) const override {
        std::string s = CLI::Formatter::make_usage(app, name);
        if (s.size() >= 2 && s.compare(s.size() - 2, 2, "\n\n") == 0) {
            s.pop_back();
        }
        return s;
    }
};

void silent_ggml_log(enum ggml_log_level, const char *, void *) {}

void silence_all_logging() {
    llama_log_set(silent_ggml_log, nullptr);
    ggml_log_set(silent_ggml_log, nullptr);
    common_log_set_verbosity_thold(-1);
    // mtmd-helper.log_set also forwards to mtmd_log_set internally.
    mtmd_helper_log_set(silent_ggml_log, nullptr);
#ifdef CHIMERA_HAS_WHISPER
    chimera_silence_whisper_log();
#endif
#ifdef CHIMERA_HAS_SD
    chimera_silence_sd_log();
#endif
}

void restore_default_logging() {
    llama_log_set(nullptr, nullptr);
    ggml_log_set(nullptr, nullptr);
    mtmd_helper_log_set(nullptr, nullptr);
#ifdef CHIMERA_HAS_WHISPER
    chimera_restore_whisper_log();
#endif
#ifdef CHIMERA_HAS_SD
    chimera_restore_sd_log();
#endif
}

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
    out.push_back(nullptr);  // NULL-terminator expected by llama_model_params.devices
    return out;
}

// Apply model-load options that are shared across all subcommands (gen,
// chat, embed). Caller has already constructed `params` from
// llama_model_default_params(). The caller owns the lifetime of the
// std::string source members; this helper writes raw pointers into
// `params` that must stay valid until the model load completes.
struct ModelExtras {
    std::vector<ggml_backend_dev_t> devices_storage;
    std::vector<float> tensor_split_storage;
};

void apply_model_common(llama_model_params & params,
                        ModelExtras & extras,
                        bool use_mmap,
                        bool use_mlock,
                        int main_gpu,
                        const std::string & tensor_split_csv,
                        const std::string & split_mode_name,
                        const std::string & devices_csv) {
    params.use_mmap  = use_mmap;
    params.use_mlock = use_mlock;
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
}

// Parse a single --logit-bias entry. Accepts "<id>(+|-|=)<bias>" with the
// "=" form requiring an explicit sign or unsigned float. Returns false on
// malformed input; caller fails with a useful message.
bool parse_logit_bias(const std::string & spec, llama_logit_bias & out) {
    // Find the first operator after a run of digits.
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

LlamaModelPtr load_llama_model(const LlamaCommonOptions & opts) {
    // Bail out with a precise message when the path doesn't resolve to a
    // readable file before we even try llama_model_load_from_file. The
    // upstream error ("failed to load model") doesn't distinguish "wrong
    // path" from "valid path, wrong format" — a real foot-gun when users
    // mistype a directory or supply a safetensors path.
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
                       opts.use_mmap, opts.use_mlock,
                       opts.main_gpu, opts.tensor_split,
                       opts.split_mode, opts.devices);

    LlamaModelPtr model(llama_model_load_from_file(opts.model.c_str(), params));
    if (!model) {
        fail(ExitCode::Load,
             "failed to load llama model: " + opts.model +
             " (file exists but llama.cpp could not parse it — "
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
    params.n_threads_batch = opts.threads;
    params.no_perf = true;

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
    return ctx;
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
        auto schema = nlohmann::ordered_json::parse(schema_text);
        return json_schema_to_grammar(schema);
    } catch (const std::exception & e) {
        fail(ExitCode::BadInput,
             std::string("failed to convert --json-schema to grammar: ") + e.what());
    }
    return {};
}

// Load each --lora spec into the model and attach (with the parsed scale)
// to `ctx`. The returned vector keeps the adapter handles alive for the
// duration of the call site — drop it after generation to free them.
struct LoraAdapters {
    std::vector<llama_adapter_lora_ptr> adapters;
};
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

// Decode a batch of tokens into the context's KV cache, batched by n_batch
// so a prompt larger than the configured batch size still fits.
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

// Set by the SIGINT handler installed around chat_sample_loop. Polled
// each token so Ctrl-C aborts generation promptly; the caller persists
// whatever content streamed before the break with partial=1.
std::atomic<bool> g_chat_interrupt_requested{false};

extern "C" void chat_sigint_handler(int) {
    g_chat_interrupt_requested.store(true, std::memory_order_relaxed);
}

// RAII guard: install chat_sigint_handler for SIGINT on construction,
// restore the previous disposition on destruction. Also clears the
// interrupt flag on construction so a stale signal doesn't carry over.
struct ChatSigintGuard {
#ifndef _WIN32
    struct sigaction prev{};
    bool installed = false;
    ChatSigintGuard() {
        g_chat_interrupt_requested.store(false, std::memory_order_relaxed);
        struct sigaction sa{};
        sa.sa_handler = chat_sigint_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;  // no SA_RESTART; we want syscalls to EINTR
        installed = (sigaction(SIGINT, &sa, &prev) == 0);
    }
    ~ChatSigintGuard() {
        if (installed) sigaction(SIGINT, &prev, nullptr);
    }
#else
    void (*prev)(int) = nullptr;
    bool installed = false;
    ChatSigintGuard() {
        g_chat_interrupt_requested.store(false, std::memory_order_relaxed);
        prev = std::signal(SIGINT, chat_sigint_handler);
        installed = (prev != SIG_ERR);
    }
    ~ChatSigintGuard() {
        if (installed) std::signal(SIGINT, prev);
    }
#endif
};

// Sample a chat reply, routing reasoning content (e.g. text inside
// <think>...</think>) through Sem::Think. Each new token is appended to
// the raw accumulator and the running text is re-parsed with
// common_chat_parse (is_partial=true); the resulting message is diffed
// against the previous parse to produce streaming content / reasoning
// deltas. Returns the *content* portion of the reply (without reasoning),
// which is what we want to store in chat history — the next turn's
// templating shouldn't reinject the model's prior thinking.
//
// If *out_interrupted is non-null and SIGINT fires during the loop, the
// loop exits early and *out_interrupted is set to true. Whatever content
// was streamed up to that point is still returned (caller persists it
// with partial=1).
std::string chat_sample_loop(
    llama_context * ctx,
    common_sampler * sampler,
    const llama_vocab * vocab,
    int n_predict,
    const common_chat_parser_params & parser_params,
    std::vector<llama_token> * out_tokens,
    std::string * out_reasoning = nullptr,
    bool * out_interrupted = nullptr) {

    std::string raw;
    std::string content;
    // Seed `prev` from a parse of the empty string. For chat formats that
    // carry a fixed assistant-turn preamble (Llama-3 stamps
    // "<|start_header_id|>assistant<|end_header_id|>\n\n" into the parsed
    // content even when raw is empty), this anchors the diff at the
    // preamble so the first real delta is only the model's actual output
    // — without it, the preamble leaks into the displayed stream.
    common_chat_msg prev = common_chat_parse("", /*is_partial=*/true, parser_params);

    for (int i = 0; i < n_predict; ++i) {
        if (g_chat_interrupt_requested.load(std::memory_order_relaxed)) {
            if (out_interrupted) *out_interrupted = true;
            break;
        }
        const llama_token token = common_sampler_sample(sampler, ctx, -1, false);
        if (token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, token)) {
            break;
        }
        common_sampler_accept(sampler, token, true);
        if (out_tokens) {
            out_tokens->push_back(token);
        }

        raw += token_to_piece(vocab, token);

        common_chat_msg cur = common_chat_parse(raw, /*is_partial=*/true, parser_params);
        for (const auto & d : common_chat_msg_diff::compute_diffs(prev, cur)) {
            if (!d.reasoning_content_delta.empty()) {
                std::cout << Sem::Think << d.reasoning_content_delta
                          << Sem::Reset << std::flush;
            }
            if (!d.content_delta.empty()) {
                std::cout << d.content_delta << std::flush;
                content += d.content_delta;
            }
        }
        prev = std::move(cur);

        llama_token token_copy = token;
        if (llama_decode(ctx, llama_batch_get_one(&token_copy, 1)) != 0) {
            fail(ExitCode::Generate, "failed to decode generated token");
        }
    }
    std::cout << '\n';
    if (out_reasoning) {
        *out_reasoning = prev.reasoning_content;
    }
    return content;
}

// Sample up to n_predict tokens from a pre-filled context. Returns the
// generated text and (out) the generated token sequence (caller-owned).
std::string sample_loop(
    llama_context * ctx,
    common_sampler * sampler,
    const llama_vocab * vocab,
    int n_predict,
    bool stream_output,
    std::vector<llama_token> * out_tokens = nullptr) {

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
        if (stream_output) {
            std::cout << piece << std::flush;
        }

        llama_token token_copy = token;
        if (llama_decode(ctx, llama_batch_get_one(&token_copy, 1)) != 0) {
            fail(ExitCode::Generate, "failed to decode generated token");
        }
    }
    if (stream_output) {
        std::cout << '\n';
    }
    return text;
}

common_sampler_ptr make_sampler(const llama_model * model, const LlamaCommonOptions & opts) {
    common_params_sampling sampling;
    sampling.seed = opts.seed;
    sampling.top_k = opts.top_k;
    sampling.top_p = opts.top_p;
    sampling.min_p = opts.min_p;
    sampling.temp = opts.temp;
    sampling.penalty_repeat = opts.repeat_penalty;
    sampling.penalty_last_n  = opts.penalty_last_n;
    sampling.penalty_present = opts.penalty_present;
    sampling.penalty_freq    = opts.penalty_freq;
    sampling.mirostat        = opts.mirostat;
    sampling.mirostat_tau    = opts.mirostat_tau;
    sampling.mirostat_eta    = opts.mirostat_eta;
    sampling.dry_multiplier     = opts.dry_multiplier;
    sampling.dry_base           = opts.dry_base;
    sampling.dry_allowed_length = opts.dry_allowed_length;
    sampling.dry_penalty_last_n = opts.dry_penalty_last_n;
    if (!opts.dry_sequence_breakers.empty()) {
        sampling.dry_sequence_breakers = opts.dry_sequence_breakers;
    }
    sampling.ignore_eos = opts.ignore_eos;
    sampling.no_perf = true;

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

    common_sampler * sampler = common_sampler_init(model, sampling);
    if (sampler == nullptr) {
        fail(ExitCode::Load, "failed to create sampler");
    }
    return common_sampler_ptr(sampler);
}

// Deleters for the mtmd C handles so we can use unique_ptr.
struct MtmdContextDeleter {
    void operator()(mtmd_context * c) const { if (c) mtmd_free(c); }
};
struct MtmdBitmapDeleter {
    void operator()(mtmd_bitmap * b) const { if (b) mtmd_bitmap_free(b); }
};
struct MtmdInputChunksDeleter {
    void operator()(mtmd_input_chunks * c) const { if (c) mtmd_input_chunks_free(c); }
};
using MtmdContextPtr     = std::unique_ptr<mtmd_context, MtmdContextDeleter>;
using MtmdBitmapPtr      = std::unique_ptr<mtmd_bitmap, MtmdBitmapDeleter>;
using MtmdInputChunksPtr = std::unique_ptr<mtmd_input_chunks, MtmdInputChunksDeleter>;

// Multimodal generation: prompt + one or more images via mmproj. Loads the
// mtmd vision projector, builds a (text + image) chunk list against the
// prompt (auto-prepending the media marker once per image if the user did
// not place markers themselves), evaluates the chunks into the context,
// and runs the existing sample loop.
std::string run_generation_mtmd(
    llama_model * model,
    const LlamaCommonOptions & opts,
    const std::string & user_prompt,
    bool stream_output) {

    if (opts.mmproj.empty() || opts.images.empty()) {
        fail(ExitCode::Runtime, "run_generation_mtmd called without mmproj/images");
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    // use_gpu controls vision encoder offload, independent of --gpu-layers
    // (which only governs LLM layers). --mmproj-offload toggles it.
    mparams.use_gpu       = opts.mmproj_use_gpu;
    mparams.n_threads     = opts.threads;
    mparams.print_timings = false;
    if (opts.flash_attn) {
        mparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    MtmdContextPtr mctx(mtmd_init_from_file(opts.mmproj.c_str(), model, mparams));
    if (!mctx) {
        fail(ExitCode::Load, "failed to load mmproj: " + opts.mmproj);
    }
    if (!mtmd_support_vision(mctx.get())) {
        fail(ExitCode::Load, "mmproj does not support vision input");
    }

    // Load each --image path into an mtmd_bitmap (stb_image internally).
    std::vector<MtmdBitmapPtr> bitmaps_owned;
    std::vector<const mtmd_bitmap *> bitmaps_c;
    bitmaps_owned.reserve(opts.images.size());
    bitmaps_c.reserve(opts.images.size());
    for (const std::string & path : opts.images) {
        MtmdBitmapPtr bmp(mtmd_helper_bitmap_init_from_file(mctx.get(), path.c_str()));
        if (!bmp) {
            fail(ExitCode::BadInput, "failed to load image: " + path);
        }
        bitmaps_c.push_back(bmp.get());
        bitmaps_owned.push_back(std::move(bmp));
    }

    // If the prompt doesn't already contain the media marker, prepend one
    // marker per image so they're inserted before the text. Users who need
    // interleaved images can place the marker themselves (no rewrite).
    const char * marker = mtmd_default_marker();
    std::string augmented_prompt;
    if (user_prompt.find(marker) == std::string::npos) {
        for (size_t i = 0; i < opts.images.size(); ++i) {
            augmented_prompt += marker;
            augmented_prompt += '\n';
        }
    }
    augmented_prompt += user_prompt;

    // Wrap the prompt in the model's chat template. Vision models are
    // almost always instruct-tuned; without the user/assistant scaffolding
    // they tend to emit EOG immediately. Falling back to raw text on
    // template-init failure keeps base-model use working.
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

    mtmd_input_text input_text;
    input_text.text = final_prompt.c_str();
    // When the chat template fired, the prompt already embeds <|begin_of_text|>
    // (or equivalent BOS); add_special=true would prepend a second BOS and
    // confuse the model into emitting stray <|start_header_id|>...<|end_header_id|>
    // tokens. Raw-prompt fallback (no template) still needs the BOS added.
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

    // Size the context to hold the multimodal prompt + room to generate.
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
                       opts.n_predict, stream_output);
}

// One-shot generation: build a fresh ctx + sampler, decode the prompt,
// then sample n_predict tokens. Used by `gen`.
std::string run_generation(
    llama_model * model,
    const LlamaCommonOptions & opts,
    const std::string & prompt,
    bool add_special,
    bool stream_output) {

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto prompt_tokens = tokenize(vocab, prompt, add_special, true);
    auto ctx = new_llama_context(model, opts, prompt_tokens.size());
    auto loras = load_loras(model, ctx.get(), opts.lora_adapters);
    auto sampler = make_sampler(model, opts);

    decode_tokens(ctx.get(), prompt_tokens, static_cast<int32_t>(opts.n_batch));
    for (llama_token token : prompt_tokens) {
        common_sampler_accept(sampler.get(), token, false);
    }
    return sample_loop(ctx.get(), sampler.get(), vocab, opts.n_predict, stream_output);
}

common_chat_msg make_chat_msg(const std::string & role, const std::string & content) {
    common_chat_msg msg;
    msg.role = role;
    msg.content = content;
    return msg;
}

int command_prompt(const LlamaCommonOptions & opts, const std::string & prompt) {
    if (!opts.images.empty() && opts.mmproj.empty()) {
        fail(ExitCode::BadInput, "--image requires --mmproj");
    }
    auto model = load_llama_model(opts);
    std::string text;
    if (!opts.images.empty()) {
        text = run_generation_mtmd(model.get(), opts, prompt, /*stream=*/true);
    } else {
        text = run_generation(model.get(), opts, prompt, /*add_special=*/true, /*stream=*/true);
    }
    return text.empty() ? static_cast<int>(ExitCode::Generate) : 0;
}

// Length of the longest common token prefix between two sequences.
size_t common_prefix(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

// ---- chat-mode helpers --------------------------------------------------

enum class ColorMode { Auto, Always, Never };

ColorMode parse_color_mode(const std::string & s) {
    if (s == "auto")   return ColorMode::Auto;
    if (s == "always") return ColorMode::Always;
    if (s == "never")  return ColorMode::Never;
    fail(ExitCode::BadInput, "--color must be one of: auto, always, never");
}

// Apply --color {auto|always|never} to rang's global control mode. Auto is
// rang's default and uses isatty() on the relevant stream; Always forces
// codes even when piped (useful for `less -R`); Never suppresses them.
// Sem manipulators no-op when control is Off, so we can stream them
// unconditionally throughout command_chat.
//
// Returns whether color will actually render on stdout (computed here
// rather than queried from rang because rang only exposes its mode via an
// internal namespace; mirroring the decision keeps us off the private API).
bool apply_color_mode(ColorMode m) {
    using rang::control;
    switch (m) {
        case ColorMode::Auto:
            rang::setControlMode(control::Auto);
            return isatty(STDOUT_FILENO) != 0;
        case ColorMode::Always:
            rang::setControlMode(control::Force);
            return true;
        case ColorMode::Never:
            rang::setControlMode(control::Off);
            return false;
    }
    return false;
}

// Background spinner on stderr while a slow op runs (model / mmproj load).
// Auto-disables when stderr is not a TTY so piped logs stay clean.
class Spinner {
    std::atomic<bool> running_{false};
    std::thread       thread_;
    std::string       label_;
    bool              tty_;

public:
    Spinner() : tty_(isatty(STDERR_FILENO) != 0) {}
    ~Spinner() { stop(); }

    void start(std::string label) {
        if (!tty_ || running_.load()) return;
        label_ = std::move(label);
        running_.store(true);
        thread_ = std::thread([this]() {
            static const char frames[] = "|/-\\";
            size_t i = 0;
            while (running_.load()) {
                std::fprintf(stderr, "\r\x1b[2m%c %s\x1b[0m",
                             frames[i++ % 4], label_.c_str());
                std::fflush(stderr);
                std::this_thread::sleep_for(std::chrono::milliseconds(120));
            }
            std::fprintf(stderr, "\r\x1b[2K");
            std::fflush(stderr);
        });
    }

    void stop() {
        if (!running_.exchange(false)) return;
        if (thread_.joinable()) thread_.join();
    }
};

bool starts_with_sv(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

#ifdef CHIMERA_HAS_LINENOISE
// Linenoise's completion callback is a plain C function pointer with no
// user-data slot, so we pass the per-session command list via a thread_local.
struct ChatCompletionState {
    std::vector<std::string> cmds;
};
thread_local ChatCompletionState * tls_completion_state = nullptr;

void chat_completion_cb(const char * buf, linenoise_completions_t * lc) {
    if (!buf || !tls_completion_state) return;
    std::string_view line(buf);
    if (line.empty() || line.front() != '/') return;

    const size_t sp = line.find(' ');
    if (sp == std::string_view::npos) {
        for (const std::string & cmd : tls_completion_state->cmds) {
            if (starts_with_sv(cmd, line)) {
                linenoise_add_completion(lc, cmd.c_str());
            }
        }
        return;
    }

    // Path-completion for /read, /glob, /image, /audio arguments.
    std::string head_str(line.substr(0, sp + 1));
    std::string_view head = head_str;
    const bool path_cmd =
        starts_with_sv(head, "/read ")  ||
        starts_with_sv(head, "/glob ")  ||
        starts_with_sv(head, "/image ") ||
        starts_with_sv(head, "/audio ");
    if (!path_cmd) return;

    namespace fs = std::filesystem;
    std::string arg(line.substr(sp + 1));
    fs::path arg_path(arg);
    fs::path dir = arg_path.has_parent_path() ? arg_path.parent_path() : fs::path(".");
    const std::string stem = arg_path.filename().string();
    std::error_code ec;
    if (!fs::exists(dir, ec)) return;
    for (const auto & e : fs::directory_iterator(dir, ec)) {
        const std::string name = e.path().filename().string();
        if (!stem.empty() && !starts_with_sv(name, stem)) continue;
        std::string suggestion = head_str;
        if (arg_path.has_parent_path()) {
            suggestion += arg_path.parent_path().string();
            suggestion += '/';
        }
        suggestion += name;
        if (e.is_directory(ec)) suggestion += '/';
        linenoise_add_completion(lc, suggestion.c_str());
    }
}
#endif  // CHIMERA_HAS_LINENOISE

// Minimal recursive-glob: supports '*' and '?'. Anchored at the path prefix
// up to the first wildcard ('foo/bar/*.txt' -> root='foo/bar', pattern='*.txt').
// Caps output at 256 matches as a runaway-safety.
std::vector<std::string> expand_glob(const std::string & pattern_in) {
    namespace fs = std::filesystem;
    std::string pattern = pattern_in;
    if (!pattern.empty() && pattern.front() == '~') {
        if (const char * home = std::getenv("HOME")) {
            pattern = home + pattern.substr(1);
        }
    }
    const size_t wild  = pattern.find_first_of("*?");
    const size_t slash = (wild == std::string::npos) ? std::string::npos
                                                     : pattern.find_last_of('/', wild);
    fs::path    root;
    std::string pat;
    if (wild == std::string::npos) {
        std::vector<std::string> out;
        if (fs::exists(pattern)) out.push_back(pattern);
        return out;
    }
    if (slash == std::string::npos) {
        root = ".";
        pat  = pattern;
    } else {
        root = pattern.substr(0, slash);
        pat  = pattern.substr(slash + 1);
    }

    auto match = [](const std::string & p, const std::string & s) {
        size_t pi = 0, si = 0, star = std::string::npos, ssi = 0;
        while (si < s.size()) {
            if (pi < p.size() && (p[pi] == '?' || p[pi] == s[si])) { ++pi; ++si; }
            else if (pi < p.size() && p[pi] == '*') { star = pi++; ssi = si; }
            else if (star != std::string::npos) { pi = star + 1; si = ++ssi; }
            else return false;
        }
        while (pi < p.size() && p[pi] == '*') ++pi;
        return pi == p.size();
    };

    std::vector<std::string> out;
    std::error_code ec;
    if (!fs::exists(root, ec)) return out;
    for (const auto & e : fs::recursive_directory_iterator(root,
            fs::directory_options::skip_permission_denied, ec)) {
        if (!e.is_regular_file()) continue;
        const std::string rel = fs::relative(e.path(), root, ec).string();
        if (ec) { ec.clear(); continue; }
        if (match(pat, rel) || match(pat, fs::path(rel).filename().string())) {
            out.push_back((root / rel).string());
        }
        if (out.size() >= 256) break;
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Best-effort terminal width. Used by the chat banner to right-align
// the model filename against the left-aligned title. Falls back to 80
// columns when we can't query the terminal (pipe, redirect, weird TTY).
int terminal_width() {
#ifndef _WIN32
    struct winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        return ws.ws_col;
    }
#else
    CONSOLE_SCREEN_BUFFER_INFO csbi{};
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(h, &csbi)) {
        int w = csbi.srWindow.Right - csbi.srWindow.Left + 1;
        if (w > 0) return w;
    }
#endif
    if (const char * env = std::getenv("COLUMNS")) {
        try { int w = std::stoi(env); if (w > 0) return w; } catch (...) {}
    }
    return 80;
}

// Best-effort identification of the "modalities" line on the banner.
std::string describe_modalities(mtmd_context * mctx) {
    std::string s = "text";
    if (mctx && mtmd_support_vision(mctx)) s += ", vision";
    if (mctx && mtmd_support_audio(mctx))  s += ", audio";
    return s;
}

// Persistent-chat configuration. Empty defaults are "ephemeral mode" —
// no DB connection, no chat row, identical to pre-phase-3 behavior.
struct ChatPersistence {
    bool        persist     = false;   // --persist: opt-in save per turn
    std::string resume;                // --resume <id|last>; empty = new chat
    std::string db_path;               // --db override; empty = default_path()
};

int command_chat(const LlamaCommonOptions & opts,
                 const std::string & system_prompt,
                 const std::string & template_override,
                 ColorMode color_mode,
                 const ChatPersistence & persist_cfg = {}) {
    const bool color_on = apply_color_mode(color_mode);

    Spinner spinner;
    spinner.start("loading model...");
    auto model = load_llama_model(opts);
    spinner.stop();

    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    common_chat_templates_ptr templates =
        common_chat_templates_init(model.get(), template_override, "", "");
    if (!templates) {
        fail(ExitCode::Load, "failed to initialize chat template");
    }

    // Optional multimodal projector. When loaded, /image and /audio become
    // available; once the user attaches any media, the chat switches to a
    // "rebuild every turn" decode path because mtmd image/audio tokens are
    // not comparable to llama text tokens for prefix reuse.
    MtmdContextPtr mctx;
    if (!opts.mmproj.empty()) {
        spinner.start("loading mmproj...");
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu       = opts.mmproj_use_gpu;
        mparams.n_threads     = opts.threads;
        mparams.print_timings = false;
        if (opts.flash_attn) {
            mparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        mctx.reset(mtmd_init_from_file(opts.mmproj.c_str(), model.get(), mparams));
        spinner.stop();
        if (!mctx) {
            fail(ExitCode::Load, "failed to load mmproj: " + opts.mmproj);
        }
    }

    auto ctx     = new_llama_context(model.get(), opts, /*min_prompt_tokens=*/0);
    auto loras   = load_loras(model.get(), ctx.get(), opts.lora_adapters);
    auto sampler = make_sampler(model.get(), opts);
    llama_memory_t mem = llama_get_memory(ctx.get());

    struct Media {
        std::string   path;
        MtmdBitmapPtr bitmap;
    };

    std::vector<common_chat_msg> history;
    if (!system_prompt.empty()) {
        history.push_back(make_chat_msg("system", system_prompt));
    }

    // ---- chat persistence wiring (phase 3) ----------------------------
    // chat_conn stays alive for the whole session. chat_id == 0 means
    // "ephemeral mode" — no DB writes happen. We open the DB lazily so
    // ephemeral sessions don't touch the filesystem.
    chimera_db::Connection chat_conn;
    int64_t chat_id = 0;
    // The basename of opts.model is recorded on the chats row as model_alias.
    // Same string the banner uses at line ~870 (declared again below).
    const std::string chat_model_alias =
        std::filesystem::path(opts.model).filename().string();
    if (persist_cfg.persist || !persist_cfg.resume.empty()) {
        chat_conn = chimera_db::open_and_migrate(
            persist_cfg.db_path.empty()
                ? chimera_db::default_path() : persist_cfg.db_path);
    }
    if (!persist_cfg.resume.empty()) {
        std::optional<chimera_chat_store::Chat> existing;
        if (persist_cfg.resume == "last" || persist_cfg.resume == "latest") {
            existing = chimera_chat_store::latest_chat(chat_conn.get());
            if (!existing) {
                fail(ExitCode::BadInput, "no chats to resume");
            }
        } else {
            try {
                const int64_t id = std::stoll(persist_cfg.resume);
                existing = chimera_chat_store::load_chat(chat_conn.get(), id);
            } catch (const std::exception &) {
                fail(ExitCode::BadInput,
                     "invalid --resume value: '" + persist_cfg.resume +
                     "' (expected an integer chat id or 'last')");
            }
            if (!existing) {
                fail(ExitCode::BadInput,
                     "no such chat id: " + persist_cfg.resume);
            }
        }
        if (existing->model_alias != chat_model_alias && !existing->model_alias.empty()) {
            std::cerr << Sem::Info
                      << "note: chat #" << existing->id << " was started with model '"
                      << existing->model_alias << "', resuming under '" << chat_model_alias
                      << "'." << Sem::Reset << "\n";
        }
        chat_id = existing->id;
        const auto stored = chimera_chat_store::load_messages(chat_conn.get(), chat_id);
        // Replace any system-prompt we seeded above with whatever the
        // resumed chat actually carries. Then append the rest in order.
        history.clear();
        size_t partial_count = 0;
        for (const auto & m : stored) {
            history.push_back(make_chat_msg(m.role, m.content));
            if (m.partial) ++partial_count;
        }
        std::cout << Sem::Info << "resumed chat #" << chat_id
                  << " (" << stored.size() << " messages, model "
                  << existing->model_alias;
        if (partial_count > 0) {
            std::cout << ", " << partial_count << " interrupted";
        }
        std::cout << ")" << Sem::Reset << "\n";
    } else if (persist_cfg.persist) {
        chat_id = chimera_chat_store::create_chat(
            chat_conn.get(), opts.model, chat_model_alias, system_prompt,
            /*source=*/"chat");
        std::cout << Sem::Info << "persistent chat #" << chat_id
                  << " (DB: " << (persist_cfg.db_path.empty()
                                  ? chimera_db::default_path()
                                  : persist_cfg.db_path)
                  << ")" << Sem::Reset << "\n";
    }

    std::string         cur_text_prefix;   // /read /glob accumulator
    std::vector<Media>  pending_media;     // /image /audio for the next turn
    std::vector<Media>  conv_media;        // all media attached so far, in order

    std::vector<llama_token> kv_tokens;
    constexpr llama_seq_id seq_id = 0;
    bool multimodal_active = false;        // sticky: set on first attach

    const bool can_image = mctx && mtmd_support_vision(mctx.get());
    const bool can_audio = mctx && mtmd_support_audio(mctx.get());

    // ---- banner / startup help -----------------------------------------
    // Clear the screen on TTY entry so the chat session starts on a clean
    // canvas. Gated on color_on (which already implies isatty + a sane
    // terminal) to avoid emitting ANSI to a pipe or a dumb terminal.
    if (color_on) std::cout << "\x1b[2J\x1b[H";

    const std::string model_name = std::filesystem::path(opts.model).filename().string();
    // Left half of the banner: "chimera vX.Y.Z chat". The width
    // calculation has to use the *plain* (un-styled) string length so the
    // padding lands correctly when SGR escapes are present.
    const std::string title_plain = std::string("chimera ") + CHIMERA_VERSION + " chat";
    const int cols = terminal_width();
    int pad = cols - static_cast<int>(title_plain.size()) - static_cast<int>(model_name.size());
    if (pad < 1) pad = 1;
    std::cout << Sem::Title << "chimera" << Sem::Reset
              << " " << CHIMERA_VERSION << " chat"
              << std::string(static_cast<size_t>(pad), ' ')
              << model_name << "\n\n";
    std::cout << "type " << Sem::Cmd << "/help" << Sem::Reset
              << " to list available commands, or "
              << Sem::Cmd << "/exit" << Sem::Reset << " to quit\n\n";
    if (!system_prompt.empty()) {
        std::cout << Sem::Info << "using custom system prompt" << Sem::Reset << "\n\n";
    }

    auto cmd_line = [](const char * name, const char * desc) {
        std::cout << "  " << Sem::Cmd << name << Sem::Reset << desc << "\n";
    };
    auto print_help = [&]() {
        std::cout << "available commands:\n";
        cmd_line("/help               ", "list commands");
        cmd_line("/exit, /quit        ", "exit");
        cmd_line("/regen              ", "regenerate the last response");
        cmd_line("/clear              ", "clear chat history");
        cmd_line("/read <file>        ", "attach a text file to the next message");
        cmd_line("/glob <pattern>     ", "attach text files matching a glob");
        if (can_image) cmd_line("/image <file>       ", "attach an image to the next message");
        if (can_audio) cmd_line("/audio <file>       ", "attach an audio file to the next message");
    };

    // ---- linenoise wiring ----------------------------------------------
#ifdef CHIMERA_HAS_LINENOISE
    const bool use_linenoise = isatty(STDIN_FILENO);
    std::unique_ptr<linenoise_context_t, void(*)(linenoise_context_t *)> ln_ctx(
        use_linenoise ? linenoise_context_create() : nullptr,
        [](linenoise_context_t * c) { if (c) linenoise_context_destroy(c); });

    ChatCompletionState completion_state;
    completion_state.cmds = {"/clear", "/exit", "/glob", "/help",
                             "/quit", "/read", "/regen"};
    if (can_image) completion_state.cmds.push_back("/image");
    if (can_audio) completion_state.cmds.push_back("/audio");
    std::sort(completion_state.cmds.begin(), completion_state.cmds.end());

    std::string history_path;
    if (use_linenoise && ln_ctx) {
        tls_completion_state = &completion_state;
        linenoise_set_completion_callback(ln_ctx.get(), chat_completion_cb);
        if (const char * env = std::getenv("CHIMERA_HISTORY")) {
            history_path = env;
        } else if (const char * home = std::getenv("HOME")) {
            history_path = std::string(home) + "/.chimera_chat_history";
        }
        if (!history_path.empty()) {
            linenoise_history_load(ln_ctx.get(), history_path.c_str());
        }
    }
#endif

    // Plain prompt for linenoise: ANSI escapes inside the prompt string
    // confuse linenoise's width math (it uses utf8_str_width, which counts
    // ESC bytes as visible columns). Instead we emit the bold-white SGR
    // escape to stdout right before calling linenoise_read; the SGR state
    // persists across linenoise's cursor moves so both the prompt
    // characters and the user's typed input render bold white. We reset
    // the SGR after the call.
    const char * const prompt_str = "> ";

    auto attach_text_file = [&](const std::string & path) -> bool {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::cerr << Sem::Err << "cannot open file: " << path
                      << Sem::Reset << "\n";
            return false;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        cur_text_prefix += "--- File: ";
        cur_text_prefix += path;
        cur_text_prefix += " ---\n";
        cur_text_prefix += ss.str();
        if (cur_text_prefix.empty() || cur_text_prefix.back() != '\n') {
            cur_text_prefix += '\n';
        }
        std::cout << Sem::Info << "attached text from '" << path << "'"
                  << Sem::Reset << "\n";
        return true;
    };

    auto attach_media = [&](const std::string & path, bool need_vision, bool need_audio) -> bool {
        if (!mctx) {
            std::cerr << Sem::Err << "multimodal not loaded: pass --mmproj <gguf>"
                      << Sem::Reset << "\n";
            return false;
        }
        if (need_vision && !can_image) {
            std::cerr << Sem::Err << "this mmproj does not support vision"
                      << Sem::Reset << "\n";
            return false;
        }
        if (need_audio && !can_audio) {
            std::cerr << Sem::Err << "this mmproj does not support audio"
                      << Sem::Reset << "\n";
            return false;
        }
        MtmdBitmapPtr bmp(mtmd_helper_bitmap_init_from_file(mctx.get(), path.c_str()));
        if (!bmp) {
            std::cerr << Sem::Err << "failed to load media: " << path
                      << Sem::Reset << "\n";
            return false;
        }
        pending_media.push_back({path, std::move(bmp)});
        std::cout << Sem::Info << "attached " << (need_vision ? "image" : "audio")
                  << " from '" << path << "'" << Sem::Reset << "\n";
        return true;
    };

    while (true) {
        bool should_generate = false;

        std::string line;
        bool got_line = false;
        // Emit the bold-white SGR raw (rang would emit the same bytes; we
        // use the string form so it's identical to what we reset with
        // below, and so we can flush without a manipulator dance). The
        // SGR state persists across linenoise's cursor moves so both the
        // "> " prompt and the user's typed input render bold white.
        if (color_on) std::cout << "\x1b[1;37m" << std::flush;
#ifdef CHIMERA_HAS_LINENOISE
        if (use_linenoise && ln_ctx) {
            char * raw = linenoise_read(ln_ctx.get(), prompt_str);
            if (raw == nullptr) { if (color_on) std::cout << "\x1b[0m"; break; }
            line.assign(raw);
            linenoise_free(raw);
            got_line = true;
        }
#endif
        if (!got_line) {
            std::cout << prompt_str << std::flush;
            if (!std::getline(std::cin, line)) { if (color_on) std::cout << "\x1b[0m"; break; }
        }
        if (color_on) std::cout << "\x1b[0m" << std::flush;
        line = trim(line);
        if (line.empty()) continue;

        // ---- slash commands -----------------------------------------------
        if (line == "/exit" || line == "/quit") {
            break;
        } else if (line == "/help") {
            print_help();
            continue;
        } else if (line == "/clear") {
            history.clear();
            if (!system_prompt.empty()) history.push_back(make_chat_msg("system", system_prompt));
            llama_memory_seq_rm(mem, seq_id, 0, -1);
            kv_tokens.clear();
            pending_media.clear();
            conv_media.clear();
            cur_text_prefix.clear();
            multimodal_active = false;
            // Persistent mode: /clear starts a fresh chat row rather than
            // wiping the existing one. The old chat is still in the DB,
            // just no longer the active session.
            if (chat_conn.ok() && persist_cfg.persist) {
                chat_id = chimera_chat_store::create_chat(
                    chat_conn.get(), opts.model, chat_model_alias, system_prompt, "chat");
                std::cout << Sem::Info
                          << "chat history cleared; started new chat #"
                          << chat_id << "." << Sem::Reset << "\n";
            } else {
                std::cout << Sem::Info << "chat history cleared."
                          << Sem::Reset << "\n";
            }
            continue;
        } else if (line == "/regen") {
            bool dropped = false;
            while (!history.empty() && history.back().role == "assistant") {
                history.pop_back();
                if (chat_id) {
                    chimera_chat_store::delete_last_message(chat_conn.get(), chat_id);
                }
                dropped = true;
            }
            if (!dropped) {
                std::cerr << Sem::Err << "nothing to regenerate."
                          << Sem::Reset << "\n";
                continue;
            }
            should_generate = true;
        } else if (starts_with_sv(line, "/read ")) {
            attach_text_file(trim(line.substr(6)));
            continue;
        } else if (starts_with_sv(line, "/glob ")) {
            const std::string pat = trim(line.substr(6));
            const auto matches = expand_glob(pat);
            if (matches.empty()) {
                std::cerr << Sem::Err << "no files match '" << pat << "'"
                          << Sem::Reset << "\n";
                continue;
            }
            for (const auto & p : matches) attach_text_file(p);
            continue;
        } else if (starts_with_sv(line, "/image ")) {
            if (attach_media(trim(line.substr(7)), true, false)) {
                multimodal_active = true;
            }
            continue;
        } else if (starts_with_sv(line, "/audio ")) {
            if (attach_media(trim(line.substr(7)), false, true)) {
                multimodal_active = true;
            }
            continue;
        } else {
            // Plain user message: assemble content from any buffered /read text
            // and pending media markers, then commit to history.
            std::string content;
            content += cur_text_prefix;
            cur_text_prefix.clear();
            const char * marker = mtmd_default_marker();
            for (size_t i = 0; i < pending_media.size(); ++i) {
                content += marker;
                content += '\n';
            }
            content += line;
            history.push_back(make_chat_msg("user", content));
            // Persistent mode: serialize the just-attached media paths
            // into media_json so a future --resume could in principle
            // re-attach them. Current --resume does not auto-reattach;
            // the column is informational. See TODO.md.
            std::string media_json;
            if (!pending_media.empty()) {
                media_json = "[";
                bool first = true;
                for (const auto & m : pending_media) {
                    if (!first) media_json += ",";
                    media_json += "\"" + m.path + "\"";
                    first = false;
                }
                media_json += "]";
            }
            if (chat_id) {
                chimera_chat_store::append_message(
                    chat_conn.get(), chat_id, "user", content,
                    /*reasoning=*/"", media_json);
            }
            for (auto & m : pending_media) conv_media.push_back(std::move(m));
            pending_media.clear();
            should_generate = true;

#ifdef CHIMERA_HAS_LINENOISE
            if (use_linenoise && ln_ctx) {
                linenoise_history_add(ln_ctx.get(), line.c_str());
                if (!history_path.empty()) {
                    linenoise_history_save(ln_ctx.get(), history_path.c_str());
                }
            }
#endif
        }

        if (!should_generate) continue;

        // ---- generate -----------------------------------------------------
        common_chat_templates_inputs inputs;
        inputs.messages = history;
        inputs.add_generation_prompt = true;
        inputs.use_jinja = opts.use_jinja;
        inputs.chat_template_kwargs = opts.chat_template_kwargs;
        // Resolve reasoning format: --reasoning-format wins; falls back
        // to --reasoning; default keeps current behavior (DEEPSEEK in the
        // parser, NONE in template inputs).
        common_reasoning_format rf_inputs = COMMON_REASONING_FORMAT_NONE;
        const std::string & rf_name =
            !opts.reasoning_format.empty() ? opts.reasoning_format : opts.reasoning;
        if (!rf_name.empty()) {
            rf_inputs = common_reasoning_format_from_name(rf_name);
            inputs.reasoning_format = rf_inputs;
        }
        common_chat_params params = common_chat_templates_apply(templates.get(), inputs);

        // Parser config for streaming chat output. DEEPSEEK reasoning
        // format covers <think>...</think> spans (the de-facto standard
        // across DeepSeek, Qwen3-thinking, and most open reasoning models).
        common_chat_parser_params parser_params(params);
        parser_params.reasoning_format = rf_name.empty()
            ? COMMON_REASONING_FORMAT_DEEPSEEK
            : rf_inputs;

        std::vector<llama_token> generated;
        std::string reply;
        std::string reply_reasoning;   // populated when the model emits <think>...</think>
        bool reply_interrupted = false;
        size_t n_prompt = 0;
        double t_prompt = 0.0;
        double t_gen    = 0.0;
        using clock = std::chrono::steady_clock;
        const auto secs = [](clock::duration d) {
            return std::chrono::duration<double>(d).count();
        };

        if (multimodal_active) {
            // Re-evaluate the entire templated prompt as mtmd chunks each
            // turn. Correct but O(history) per turn.
            llama_memory_seq_rm(mem, seq_id, 0, -1);
            kv_tokens.clear();

            std::vector<const mtmd_bitmap *> bitmaps_c;
            bitmaps_c.reserve(conv_media.size());
            for (const auto & m : conv_media) bitmaps_c.push_back(m.bitmap.get());

            mtmd_input_text input_text;
            input_text.text = params.prompt.c_str();
            // See the text-only branch below for the rationale: the chat
            // template already includes BOS, so add_special=true would
            // double-BOS the prompt and the model emits a stray
            // <|start_header_id|>assistant<|end_header_id|> at reply start.
            input_text.add_special  = false;
            input_text.parse_special = true;
            MtmdInputChunksPtr chunks(mtmd_input_chunks_init());
            if (!chunks) {
                std::cerr << Sem::Err << "failed to init mtmd input chunks"
                          << Sem::Reset << "\n";
                continue;
            }
            const int32_t tok_rc = mtmd_tokenize(mctx.get(), chunks.get(), &input_text,
                                                 bitmaps_c.data(), bitmaps_c.size());
            if (tok_rc != 0) {
                std::cerr << Sem::Err << "mtmd_tokenize failed (rc=" << tok_rc << ")"
                          << Sem::Reset << "\n";
                continue;
            }
            llama_pos new_n_past = 0;
            const auto t0 = clock::now();
            const int32_t eval_rc = mtmd_helper_eval_chunks(
                mctx.get(), ctx.get(), chunks.get(),
                /*n_past=*/0, seq_id,
                static_cast<int32_t>(opts.n_batch),
                /*logits_last=*/true, &new_n_past);
            t_prompt = secs(clock::now() - t0);
            n_prompt = static_cast<size_t>(new_n_past);
            if (eval_rc != 0) {
                std::cerr << Sem::Err << "mtmd_helper_eval_chunks failed (rc=" << eval_rc << ")"
                          << Sem::Reset << "\n";
                continue;
            }
            const auto t1 = clock::now();
            {
                ChatSigintGuard sigint_guard;
                reply = chat_sample_loop(ctx.get(), sampler.get(), vocab,
                                         opts.n_predict, parser_params, &generated,
                                         &reply_reasoning, &reply_interrupted);
            }
            t_gen = secs(clock::now() - t1);
        } else {
            // Text-only fast path: KV-prefix reuse via token comparison.
            // Chat templates (Jinja or built-in) emit <|begin_of_text|>
            // themselves; asking llama_tokenize to also add_special would
            // prepend a second BOS. Llama-3 reacts to the double-BOS by
            // emitting a stray "<|start_header_id|>assistant<|end_header_id|>"
            // at the start of its reply and losing track of the user
            // message. parse_special=true is enough — it maps the
            // template's BOS literal back to the right token.
            const auto full_tokens = tokenize(vocab, params.prompt,
                                              /*add_special=*/true,
                                              /*parse_special=*/true);

            const size_t shared = common_prefix(kv_tokens, full_tokens);
            if (shared < kv_tokens.size()) {
                llama_memory_seq_rm(mem, seq_id,
                                    static_cast<llama_pos>(shared),
                                    static_cast<llama_pos>(kv_tokens.size()));
            }
            std::vector<llama_token> tail(full_tokens.begin() + static_cast<ptrdiff_t>(shared),
                                          full_tokens.end());
            if (tail.empty()) {
                tail.push_back(full_tokens.back());
                llama_memory_seq_rm(mem, seq_id,
                                    static_cast<llama_pos>(full_tokens.size() - 1),
                                    static_cast<llama_pos>(full_tokens.size()));
            }
            const auto t0 = clock::now();
            decode_tokens(ctx.get(), tail, static_cast<int32_t>(opts.n_batch));
            t_prompt = secs(clock::now() - t0);
            n_prompt = tail.size();
            for (llama_token tok : tail) common_sampler_accept(sampler.get(), tok, false);
            const auto t1 = clock::now();
            {
                ChatSigintGuard sigint_guard;
                reply = chat_sample_loop(ctx.get(), sampler.get(), vocab,
                                         opts.n_predict, parser_params, &generated,
                                         &reply_reasoning, &reply_interrupted);
            }
            t_gen = secs(clock::now() - t1);
            kv_tokens = full_tokens;
            kv_tokens.insert(kv_tokens.end(), generated.begin(), generated.end());
        }

        if (n_prompt > 0 && t_prompt > 0.0 && !generated.empty() && t_gen > 0.0) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "[ Prompt: %.1f t/s | Generation: %.1f t/s ]",
                          static_cast<double>(n_prompt) / t_prompt,
                          static_cast<double>(generated.size()) / t_gen);
            std::cout << Sem::Stats << buf << Sem::Reset << "\n";
        }

        if (reply_interrupted) {
            // chat_sample_loop returns with the SGR state mid-stream; emit
            // a newline and a reset so the [interrupted] notice renders on
            // its own line in default colors.
            std::cout << "\n" << Sem::Info
                      << "[interrupted — partial response "
                      << (chat_id ? "saved" : "kept in-memory only")
                      << "]" << Sem::Reset << "\n";
        }
        history.push_back(make_chat_msg("assistant", reply));
        if (chat_id) {
            chimera_chat_store::append_message(
                chat_conn.get(), chat_id, "assistant", reply,
                reply_reasoning, /*media_json=*/"",
                static_cast<int>(n_prompt),
                static_cast<int>(generated.size()),
                /*partial=*/reply_interrupted);
        }
    }

#ifdef CHIMERA_HAS_LINENOISE
    tls_completion_state = nullptr;
#endif
    return 0;
}

int command_tokenize(const TokenizeOptions & opts) {
    const std::string text = resolve_prompt(opts.input, opts.input_file);

    LlamaCommonOptions load_opts;
    load_opts.model = opts.model;
    load_opts.gpu_layers = 0;  // tokenization is CPU-bound; no GPU needed
    load_opts.use_mmap = opts.use_mmap;
    auto model = load_llama_model(load_opts);
    const llama_vocab * vocab = llama_model_get_vocab(model.get());

    const auto tokens = tokenize(vocab, text, opts.add_special, opts.parse_special);
    for (llama_token tok : tokens) {
        if (opts.show_pieces) {
            std::string piece = token_to_piece(vocab, tok);
            // Escape newlines/tabs in the piece for one-line output.
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

enum llama_pooling_type parse_pooling(const std::string & name) {
    if (name == "none") return LLAMA_POOLING_TYPE_NONE;
    if (name == "mean") return LLAMA_POOLING_TYPE_MEAN;
    if (name == "cls")  return LLAMA_POOLING_TYPE_CLS;
    if (name == "last") return LLAMA_POOLING_TYPE_LAST;
    fail(ExitCode::BadInput, "unknown pooling type: " + name +
        " (expected: mean | cls | last | none)");
}

int command_embed(const EmbedOptions & opts) {
    const std::string text = resolve_prompt(opts.input, opts.input_file);

    chimera_embed::Config cfg;
    cfg.model      = opts.model;
    cfg.pooling    = opts.pooling;
    cfg.threads    = opts.threads;
    cfg.gpu_layers = opts.gpu_layers;
    cfg.n_ctx      = opts.n_ctx;
    cfg.n_batch    = opts.n_batch;
    cfg.n_ubatch   = opts.n_ubatch;
    cfg.normalize  = opts.normalize;
    cfg.use_mmap   = opts.use_mmap;
    cfg.use_mlock  = opts.use_mlock;
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
    const auto vec = embedder.embed(text);

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
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) *out << ' ';
        *out << vec[i];
    }
    *out << '\n';
    return 0;
}

// ---- chunking ----------------------------------------------------------

// Token-based chunking lives in chimera_embed::chunk_by_tokens — it
// uses the loaded embedding model's vocab, so chunks are sized in the
// tokens the model will actually see at embed time. The previous
// character-window+sentence-nudge chunker is gone; ingest and the
// /v1/vector_stores ingest route both call chunk_by_tokens with the
// collection's recorded chunk_tokens / chunk_overlap.

// ---- index / search subcommand implementations ------------------------

// `chimera index create` — load the model long enough to discover its
// embedding dim, then record the collection metadata. We don't ingest
// anything here; that's a separate operation. Done as one upfront step
// rather than on-demand so subsequent `ingest` and `search` calls fail
// fast if the model is missing.
int command_index_create(const std::string & db_path,
                         const std::string & name,
                         const std::string & embedding_model,
                         int                 ctx_size,
                         int                 threads,
                         int                 gpu_layers,
                         const std::string & pooling,
                         const std::string & distance,
                         int                 chunk_tokens,
                         int                 chunk_overlap) {
    if (embedding_model.empty()) {
        fail(ExitCode::BadInput, "index create requires --embedding-model");
    }

    chimera_embed::Config cfg;
    cfg.model      = embedding_model;
    cfg.pooling    = pooling;
    cfg.threads    = threads;
    cfg.gpu_layers = gpu_layers;
    cfg.n_ctx      = static_cast<uint32_t>(ctx_size);
    cfg.normalize  = true;
    chimera_embed::Embedder embedder(cfg);
    const int dim = embedder.n_embd();

    auto conn = chimera_db::open_and_migrate(
        db_path.empty() ? chimera_db::default_path() : db_path);
    chimera_vector_store::CreateOptions cop;
    cop.distance      = distance;
    cop.chunk_tokens  = chunk_tokens;
    cop.chunk_overlap = chunk_overlap;
    auto col = chimera_vector_store::create(conn.get(), name, embedding_model, dim, cop);

    std::cout << "created collection '" << col.name << "'\n"
              << "  embedding model: " << col.embedding_model << "\n"
              << "  dim:             " << col.dim << "\n"
              << "  distance:        " << col.distance << "\n"
              << "  chunk_tokens:    " << col.chunk_tokens << "\n"
              << "  chunk_overlap:   " << col.chunk_overlap << "\n";
    return 0;
}

// `chimera index ingest` — chunk + embed + insert. Optionally accepts a
// glob pattern, in which case the same Embedder is reused across files
// (the costly part is model load, not per-chunk inference).
int command_index_ingest(const std::string &              db_path,
                         const std::string &              name,
                         const std::vector<std::string> & files,
                         const std::string &              glob_pattern,
                         int                              ctx_size,
                         int                              threads,
                         int                              gpu_layers,
                         const std::string &              pooling,
                         int                              chunk_tokens_override,
                         int                              chunk_overlap_override,
                         bool                             cache_embeddings) {
    auto conn = chimera_db::open_and_migrate(
        db_path.empty() ? chimera_db::default_path() : db_path);
    auto col = chimera_vector_store::find(conn.get(), name);
    if (!col) {
        fail(ExitCode::BadInput,
             "no such collection: '" + name + "'. Create it with `chimera index create`.");
    }

    // Resolve the input set: explicit `--file` args + any glob matches.
    std::vector<std::string> sources = files;
    if (!glob_pattern.empty()) {
        namespace fs = std::filesystem;
        const auto last_sep = glob_pattern.find_last_of('/');
        const fs::path root = (last_sep == std::string::npos)
            ? fs::path(".") : fs::path(glob_pattern.substr(0, last_sep));
        const std::string pat = (last_sep == std::string::npos)
            ? glob_pattern : glob_pattern.substr(last_sep + 1);
        // Tiny shell-glob: '*' matches anything but '/', '?' matches one char.
        auto match = [](const std::string & p, const std::string & s) {
            size_t pi = 0, si = 0, star = std::string::npos, ssi = 0;
            while (si < s.size()) {
                if (pi < p.size() && (p[pi] == '?' || p[pi] == s[si])) { ++pi; ++si; }
                else if (pi < p.size() && p[pi] == '*')   { star = pi++; ssi = si; }
                else if (star != std::string::npos)        { pi = star + 1; si = ++ssi; }
                else return false;
            }
            while (pi < p.size() && p[pi] == '*') ++pi;
            return pi == p.size();
        };
        std::error_code ec;
        if (fs::exists(root, ec)) {
            for (const auto & e : fs::recursive_directory_iterator(
                     root, fs::directory_options::skip_permission_denied, ec)) {
                if (!e.is_regular_file()) continue;
                if (match(pat, e.path().filename().string())) {
                    sources.push_back(e.path().string());
                }
            }
        }
        std::sort(sources.begin(), sources.end());
        sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
    }
    if (sources.empty()) {
        fail(ExitCode::BadInput, "no input files (pass -f or -g)");
    }

    chimera_embed::Config cfg;
    cfg.model      = col->embedding_model;
    cfg.pooling    = pooling;
    cfg.threads    = threads;
    cfg.gpu_layers = gpu_layers;
    cfg.n_ctx      = static_cast<uint32_t>(ctx_size);
    cfg.normalize  = true;
    chimera_embed::Embedder embedder(cfg);

    // Optional embedding cache: reuses --db (where the collection
    // already lives). Cache key is the model fingerprint of the
    // collection's recorded embedding_model file, so a rename or
    // re-quantization invalidates the cache automatically.
    std::unique_ptr<chimera_embed_cache::Cache> ecache;
    if (cache_embeddings) {
        const std::string mid = chimera_embed_cache::compute_model_id(col->embedding_model);
        if (mid.empty()) {
            fail(ExitCode::BadInput,
                 "--cache-embeddings: cannot fingerprint embedding model "
                 "(unreadable file: " + col->embedding_model + ")");
        }
        ecache = std::make_unique<chimera_embed_cache::Cache>(
            db_path.empty() ? chimera_db::default_path() : db_path, mid);
        embedder.set_cache(ecache.get());
    }

    // Per-collection chunk defaults; CLI overrides win when > 0. Token-
    // based, not character-based: a chunk is `chunk_tokens` tokens of
    // the collection's embedding-model vocab with `chunk_overlap`
    // tokens of overlap between neighbors. This makes per-chunk sizes
    // accurate against the embedding model's input limit, eliminating
    // the character-window proxy and its 400-800-token variance.
    const int eff_chunk_tokens   = chunk_tokens_override   > 0
        ? chunk_tokens_override   : col->chunk_tokens;
    const int eff_chunk_overlap  = chunk_overlap_override  >= 0
        ? chunk_overlap_override  : col->chunk_overlap;

    size_t total_chunks = 0;
    for (const auto & path : sources) {
        const std::string text = read_file(path);
        const auto chunks = chimera_embed::chunk_by_sentences(
            text, embedder, eff_chunk_tokens, eff_chunk_overlap);
        for (const auto & c : chunks) {
            auto vec = embedder.embed(c.text);
            if (static_cast<int>(vec.size()) != col->dim) {
                fail(ExitCode::Runtime,
                     "embedding dim drift: collection expects " +
                     std::to_string(col->dim) + ", model produced " +
                     std::to_string(vec.size()));
            }
            chimera_vector_store::DocumentInput doc;
            doc.source_uri  = path;
            doc.chunk_index = c.index;
            doc.text        = c.text;
            doc.token_count = c.token_count;
            doc.embedding   = std::move(vec);
            chimera_vector_store::insert_document(conn.get(), *col, doc);
            ++total_chunks;
        }
        std::cout << "  ingested " << chunks.size() << " chunk(s) from " << path << "\n";
    }
    std::cout << "done: " << total_chunks << " chunk(s) into '" << name << "'\n";
    return 0;
}

int command_index_list(const std::string & db_path) {
    auto conn = chimera_db::open_and_migrate(
        db_path.empty() ? chimera_db::default_path() : db_path);
    const auto cols = chimera_vector_store::list(conn.get());
    if (cols.empty()) {
        std::cout << "(no collections)\n";
        return 0;
    }
    std::cout << "collections:\n";
    for (const auto & c : cols) {
        std::cout << "  " << c.name
                  << "  (dim=" << c.dim
                  << ", model=" << c.embedding_model
                  << ", docs=" << c.doc_count << ")\n";
    }
    return 0;
}

int command_index_stats(const std::string & db_path, const std::string & name) {
    auto conn = chimera_db::open_and_migrate(
        db_path.empty() ? chimera_db::default_path() : db_path);
    auto col = chimera_vector_store::find(conn.get(), name);
    if (!col) {
        fail(ExitCode::BadInput, "no such collection: '" + name + "'");
    }
    const auto cols_with_counts = chimera_vector_store::list(conn.get());
    int64_t docs = 0;
    for (const auto & c : cols_with_counts) {
        if (c.id == col->id) { docs = c.doc_count; break; }
    }
    std::cout << "collection: " << col->name << "\n"
              << "  id:              " << col->id << "\n"
              << "  embedding model: " << col->embedding_model << "\n"
              << "  dim:             " << col->dim << "\n"
              << "  distance:        " << col->distance << "\n"
              << "  chunk_tokens:    " << col->chunk_tokens << "\n"
              << "  chunk_overlap:   " << col->chunk_overlap << "\n"
              << "  created_at:      " << col->created_at << "\n"
              << "  documents:       " << docs << "\n";
    return 0;
}

int command_index_drop(const std::string & db_path, const std::string & name) {
    auto conn = chimera_db::open_and_migrate(
        db_path.empty() ? chimera_db::default_path() : db_path);
    chimera_vector_store::drop(conn.get(), name);
    std::cout << "dropped collection '" << name << "'\n";
    return 0;
}

// `chimera search` — KNN over one collection. Loads the embedding model
// recorded on the collection, embeds the query, runs the vec0 query.
int command_search(const std::string & db_path,
                   const std::string & name,
                   const std::string & query,
                   int                 k,
                   int                 ctx_size,
                   int                 threads,
                   int                 gpu_layers,
                   const std::string & pooling,
                   bool                cache_embeddings,
                   const std::string & mode_str) {
    if (query.empty()) {
        fail(ExitCode::BadInput, "search requires -q/--query");
    }
    auto mode_opt = chimera_vector_store::parse_search_mode(mode_str);
    if (!mode_opt) {
        fail(ExitCode::BadInput,
             "invalid --mode '" + mode_str +
             "' (expected: semantic | lexical | hybrid)");
    }
    const auto mode = *mode_opt;

    auto conn = chimera_db::open_and_migrate(
        db_path.empty() ? chimera_db::default_path() : db_path);
    auto col = chimera_vector_store::find(conn.get(), name);
    if (!col) {
        fail(ExitCode::BadInput, "no such collection: '" + name + "'");
    }

    // Lexical-only mode skips the embedding model load entirely — FTS5
    // has no use for vectors, and a multi-100 MB GGUF load just to do a
    // BM25 lookup would be perverse.
    std::vector<float> qvec;
    if (mode != chimera_vector_store::SearchMode::Lexical) {
        chimera_embed::Config cfg;
        cfg.model      = col->embedding_model;
        cfg.pooling    = pooling;
        cfg.threads    = threads;
        cfg.gpu_layers = gpu_layers;
        cfg.n_ctx      = static_cast<uint32_t>(ctx_size);
        cfg.normalize  = true;
        chimera_embed::Embedder embedder(cfg);
        std::unique_ptr<chimera_embed_cache::Cache> ecache;
        if (cache_embeddings) {
            const std::string mid = chimera_embed_cache::compute_model_id(col->embedding_model);
            if (mid.empty()) {
                fail(ExitCode::BadInput,
                     "--cache-embeddings: cannot fingerprint embedding model "
                     "(unreadable file: " + col->embedding_model + ")");
            }
            ecache = std::make_unique<chimera_embed_cache::Cache>(
                db_path.empty() ? chimera_db::default_path() : db_path, mid);
            embedder.set_cache(ecache.get());
        }
        qvec = embedder.embed(query);
        if (static_cast<int>(qvec.size()) != col->dim) {
            fail(ExitCode::Runtime,
                 "query embedding dim mismatch: collection expects " +
                 std::to_string(col->dim) + ", model produced " +
                 std::to_string(qvec.size()));
        }
    }

    const auto hits = chimera_vector_store::search(
        conn.get(), *col, qvec, query, k, mode);
    if (hits.empty()) {
        std::cout << "(no hits)\n";
        return 0;
    }
    for (size_t i = 0; i < hits.size(); ++i) {
        const auto & h = hits[i];
        std::cout << "#" << (i + 1);
        if (mode == chimera_vector_store::SearchMode::Hybrid) {
            std::cout << "  rrf=" << h.rrf_score
                      << "  sem=" << (h.semantic_rank >= 0
                                      ? std::to_string(h.semantic_rank + 1) : "-")
                      << "  lex=" << (h.lexical_rank  >= 0
                                      ? std::to_string(h.lexical_rank + 1)  : "-");
        } else {
            std::cout << "  distance=" << h.distance;
        }
        std::cout << "  " << h.source_uri
                  << "  chunk=" << h.chunk_index
                  << "\n----\n" << h.text << "\n----\n\n";
    }
    return 0;
}

// `chimera chat --list` — short, recently-active-first index of stored
// chats. Print-and-exit; no model load.
int command_chat_list(const std::string & db_path, int limit) {
    auto conn = chimera_db::open_and_migrate(
        db_path.empty() ? chimera_db::default_path() : db_path);
    const auto chats = chimera_chat_store::list_chats(conn.get(), limit);
    if (chats.empty()) {
        std::cout << "(no saved chats)\n";
        return 0;
    }
    std::cout << "saved chats:\n";
    for (const auto & c : chats) {
        std::cout << "  #" << c.id
                  << "  " << c.message_count << " msgs";
        if (c.partial_count > 0) {
            std::cout << " (" << c.partial_count << " interrupted)";
        }
        std::cout << "  model=" << c.model_alias
                  << "  updated_at=" << c.updated_at;
        if (!c.title.empty()) std::cout << "  title=\"" << c.title << "\"";
        std::cout << "\n";
    }
    return 0;
}

// `chimera chat --search QUERY` — FTS5 query over messages_fts. Prints
// top hits with `[word]`-style snippet highlights. Print-and-exit.
int command_chat_search(const std::string & db_path,
                         const std::string & query,
                         int                 limit) {
    auto conn = chimera_db::open_and_migrate(
        db_path.empty() ? chimera_db::default_path() : db_path);
    const auto hits = chimera_chat_store::search_messages(conn.get(), query, limit);
    if (hits.empty()) {
        std::cout << "(no hits)\n";
        return 0;
    }
    for (const auto & h : hits) {
        std::cout << "#" << h.chat_id
                  << " seq=" << h.seq
                  << " role=" << h.role
                  << "\n  " << h.snippet << "\n";
    }
    return 0;
}

// ---- `chimera info` --------------------------------------------------
//
// Print a structured summary of every component baked into the binary:
// chimera version, platform, the three bundled inference libraries
// (llama, whisper, sd) with their ggml views, registered ggml backends
// + enumerated devices, and the embedded SQLite stack. Mirrors cyllama's
// `info` subcommand so users switching between native and Python sides
// see the same shape.

namespace {

// `whisper_print_system_info()` and `sd_get_system_info()` produce a
// stream of `NAME = 0|1` pairs separated by ` | `. We split into the
// backend names we recognize vs. everything else (CPU feature flags).
const std::vector<std::string> & known_backend_names() {
    static const std::vector<std::string> v = {
        "COREML",  "OPENVINO", "METAL",     "MTL",     "BLAS",
        "SYCL",    "VULKAN",   "KOMPUTE",   "OPENCL",  "CUDA",
        "CANN",    "MUSA",     "ROCBLAS",   "RPC",     "BLIS",
        "ACCELERATE", "HIP",   "WEBGPU",    "ZENDNN",  "VIRTGPU",
    };
    return v;
}

struct ParsedSysInfo {
    std::vector<std::string> backends;
    std::vector<std::string> cpu_features;
};

bool is_backend(const std::string & name) {
    for (const auto & b : known_backend_names()) {
        if (b == name) return true;
    }
    return false;
}

ParsedSysInfo parse_sys_info(const std::string & info) {
    ParsedSysInfo out;
    size_t pos = 0;
    while (pos < info.size()) {
        const size_t eq = info.find("= ", pos);
        if (eq == std::string::npos) break;
        size_t name_end = eq;
        while (name_end > 0 && info[name_end - 1] == ' ') --name_end;
        size_t name_start = name_end;
        while (name_start > 0) {
            const char c = info[name_start - 1];
            if (c == ' ' || c == '|' || c == ':') break;
            --name_start;
        }
        const std::string name(info.data() + name_start, name_end - name_start);
        const size_t val_pos = eq + 2;
        const bool enabled = (val_pos < info.size() && info[val_pos] == '1');
        pos = val_pos + 1;
        if (!enabled) continue;
        if (name.empty() || name == "WHISPER") continue;
        if (is_backend(name)) out.backends.push_back(name);
        else                  out.cpu_features.push_back(name);
    }
    return out;
}

std::string join_csv(const std::vector<std::string> & items) {
    std::string out;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) out += ", ";
        out += items[i];
    }
    return out;
}

const char * ggml_dev_type_label(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU  ";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU  ";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU ";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "META ";
    }
    return "?    ";
}

std::string platform_label() {
#if defined(__APPLE__)
    const char * os = "macOS";
#elif defined(__linux__)
    const char * os = "Linux";
#elif defined(_WIN32)
    const char * os = "Windows";
#else
    const char * os = "unknown";
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    const char * arch = "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    const char * arch = "x86_64";
#elif defined(__riscv)
    const char * arch = "riscv";
#else
    const char * arch = "unknown";
#endif
    return std::string(os) + "-" + arch;
}

// Render a `key=value` line in the build-flags block, but only if the
// value is non-empty. The macros come through as empty strings when the
// corresponding CMake/env var wasn't set.
void emit_build_flag(std::ostream & out, const char * key, const char * value) {
    if (value && *value) {
        out << "  " << std::left << std::setw(14) << key << " " << value << "\n";
    }
}

// ggml registry names are short ("MTL", "CUDA", "HIP", ...). Map them to
// the friendly labels users expect to see (and that cyllama prints).
std::string friendly_backend_name(const std::string & s) {
    if (s == "MTL")     return "Metal";
    if (s == "BLAS")    return "BLAS";
    if (s == "CUDA")    return "CUDA";
    if (s == "VULKAN")  return "Vulkan";
    if (s == "HIP")     return "HIP";
    if (s == "SYCL")    return "SYCL";
    if (s == "OPENCL")  return "OpenCL";
    if (s == "KOMPUTE") return "Kompute";
    return s;
}

// First non-CPU, non-Accelerate backend registry is the "primary" GPU/
// accelerator backend chimera was built with. Matches cyllama's
// `built:` line.
std::string primary_backend_label() {
    const size_t n = ggml_backend_reg_count();
    for (size_t i = 0; i < n; ++i) {
        const char * name = ggml_backend_reg_name(ggml_backend_reg_get(i));
        if (!name) continue;
        const std::string s(name);
        if (s == "CPU" || s == "BLAS" || s == "Accelerate") continue;
        return friendly_backend_name(s);
    }
    return "CPU";
}

std::vector<std::string> backend_registry_names() {
    std::vector<std::string> out;
    const size_t n = ggml_backend_reg_count();
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (const char * name = ggml_backend_reg_name(ggml_backend_reg_get(i))) {
            out.emplace_back(name);
        }
    }
    return out;
}

}  // namespace

int command_info() {
    std::cout << "chimera " << CHIMERA_VERSION << "\n"
              << platform_label() << "\n\n";

    // ---- llama.cpp --------------------------------------------------
    //
    // `built:`  comes from CHIMERA_BUILT_BACKENDS — the GGML_* flags that
    //           were ON when chimera was compiled. Stable across runs of
    //           the same binary.
    // `loaded:` comes from the ggml backend registry at runtime. Reflects
    //           what actually initialized successfully on this host (e.g.
    //           a CUDA-built binary on a box with no CUDA driver would
    //           print `built: CUDA` but `loaded: CPU`).
    std::cout << "llama.cpp:\n"
              << "  version:       " << CHIMERA_LLAMACPP_VERSION   << "\n"
              << "  ggml version:  " << ggml_version()             << "\n"
              << "  ggml commit:   " << ggml_commit()              << "\n"
              << "  built:         " << CHIMERA_BUILT_BACKENDS     << "\n"
              << "  loaded:        " << primary_backend_label()    << "\n"
              << "  registries:    " << join_csv(backend_registry_names()) << "\n"
              << "  devices:\n";
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        auto * d = ggml_backend_dev_get(i);
        const char * dn = ggml_backend_dev_name(d);
        const char * dd = ggml_backend_dev_description(d);
        std::cout << "    " << std::left << std::setw(20) << (dn ? dn : "?")
                  << " [" << ggml_dev_type_label(ggml_backend_dev_type(d)) << "]  "
                  << (dd ? dd : "")
                  << "\n";
    }
    std::cout << "  GPU offload:   " << (llama_supports_gpu_offload() ? "True" : "False") << "\n"
              << "  MMAP support:  " << (llama_supports_mmap()         ? "True" : "False") << "\n"
              << "  MLOCK support: " << (llama_supports_mlock()        ? "True" : "False") << "\n"
              << "  RPC support:   " << (llama_supports_rpc()          ? "True" : "False") << "\n";

    // ---- whisper.cpp ------------------------------------------------
    // For backends, chimera shares one ggml registry set across llama,
    // whisper, and sd, so the backends line mirrors the registry list
    // above (minus the duplicate-printing-as-CPU). The CPU features
    // come from each library's own probe of its (linked) ggml.
#ifdef CHIMERA_HAS_WHISPER
    const auto whisper_parsed = parse_sys_info(chimera_whisper::whisper_system_info_raw());
    std::cout << "\nwhisper.cpp:\n"
              << "  version:       " << CHIMERA_WHISPERCPP_VERSION                  << "\n"
              << "  ggml version:  " << chimera_whisper::whisper_ggml_version()     << "\n"
              << "  built:         " << CHIMERA_BUILT_BACKENDS                      << "\n"
              << "  loaded:        " << primary_backend_label()                     << "\n"
              << "  backends:      " << join_csv(backend_registry_names())          << "\n"
              << "  CPU features:  " << join_csv(whisper_parsed.cpu_features)       << "\n";
#else
    std::cout << "\nwhisper.cpp:    not linked (built with CHIMERA_WITH_WHISPER=OFF)\n";
#endif

    // ---- stable-diffusion.cpp ---------------------------------------
#ifdef CHIMERA_HAS_SD
    const auto sd_parsed = parse_sys_info(chimera_sd::sd_system_info_raw());
    std::cout << "\nstable-diffusion.cpp:\n"
              << "  version:       " << CHIMERA_SDCPP_VERSION                  << "\n"
              << "  ggml version:  " << chimera_sd::sd_ggml_version()          << "\n"
              << "  built:         " << CHIMERA_BUILT_BACKENDS                 << "\n"
              << "  loaded:        " << primary_backend_label()                << "\n"
              << "  backends:      " << join_csv(backend_registry_names())     << "\n"
              << "  CPU features:  " << join_csv(sd_parsed.cpu_features)       << "\n";
#else
    std::cout << "\nstable-diffusion.cpp: not linked (built with CHIMERA_WITH_SD=OFF)\n";
#endif

    // ---- SQLite + sqlite-vec ----------------------------------------
    std::cout << "\nsqlite:\n"
              << "  version:       " << chimera_db::sqlite_version()     << "\n"
              << "  sqlite-vec:    " << chimera_db::sqlite_vec_version() << "\n";

    // ---- build flags (only knobs that were actually set) ------------
    //
    // We surface the tuning knobs that affect runtime behavior so bug
    // reports can include them. Each macro is an empty string when its
    // GGML_*/CMAKE_* source wasn't set, so the helper skips it silently.
    std::ostringstream flags_block;
    emit_build_flag(flags_block, "CUDA_ARCH",      CHIMERA_CUDA_ARCHITECTURES);
    emit_build_flag(flags_block, "HIP_ARCH",       CHIMERA_HIP_ARCHITECTURES);
    emit_build_flag(flags_block, "BLAS_VENDOR",    CHIMERA_BLAS_VENDOR);
    emit_build_flag(flags_block, "CUDA_FORCE_MMQ",     CHIMERA_CUDA_FORCE_MMQ);
    emit_build_flag(flags_block, "CUDA_FORCE_CUBLAS",  CHIMERA_CUDA_FORCE_CUBLAS);
    emit_build_flag(flags_block, "HIP_ROCWMMA_FATTN",  CHIMERA_HIP_ROCWMMA_FATTN);
    const std::string flags_str = flags_block.str();
    if (!flags_str.empty()) {
        std::cout << "\nbuild flags:\n" << flags_str;
    }

    return 0;
}

// `chimera db status` — open (or create) the configured DB, run any
// pending migrations, and print a human-readable summary. The smallest
// smoke-testable surface for the phase-1 SQLite vendoring: confirms
// that sqlite3.c + sqlite-vec.c linked, the file path resolves, the
// migration runner works end-to-end, and the schema lands.
int command_db_backup(const std::string & path_override,
                      const std::string & dst) {
    const std::string path = path_override.empty()
        ? chimera_db::default_path()
        : path_override;
    auto conn = chimera_db::open_and_migrate(path);
    chimera_db::backup_to(conn.get(), dst);
    std::cout << "chimera db backup\n"
              << "  src: " << path << "\n"
              << "  dst: " << dst  << "\n"
              << "  ok\n";
    return 0;
}

int command_db_vacuum(const std::string & path_override) {
    const std::string path = path_override.empty()
        ? chimera_db::default_path()
        : path_override;
    auto conn = chimera_db::open_and_migrate(path);
    chimera_db::vacuum(conn.get());
    std::cout << "chimera db vacuum\n"
              << "  path: " << path << "\n"
              << "  ok\n";
    return 0;
}

int command_db_status(const std::string & path_override) {
    const std::string path = path_override.empty()
        ? chimera_db::default_path()
        : path_override;

    auto conn = chimera_db::open_and_migrate(path);
    const int v = chimera_db::current_schema_version(conn.get());
    const auto tables = chimera_db::list_tables(conn.get());

    std::cout << "chimera db status\n"
              << "  path:           " << path << "\n"
              << "  sqlite:         " << chimera_db::sqlite_version()     << "\n"
              << "  sqlite-vec:     " << chimera_db::sqlite_vec_version()
              << " (runtime: "        << chimera_db::sqlite_vec_loaded_version(conn.get())
              << ")\n"
              << "  schema version: " << v << " / "
                                       << chimera_db::latest_schema_version() << "\n"
              << "  tables (" << tables.size() << "):\n";
    for (const auto & t : tables) {
        std::cout << "    - " << t << "\n";
    }
    return 0;
}

std::string version_string() {
    return std::string("chimera ") + CHIMERA_VERSION + "\n"
        + "  llama.cpp:            " + CHIMERA_LLAMACPP_VERSION + "\n"
        + "  whisper.cpp:          " + CHIMERA_WHISPERCPP_VERSION + "\n"
        + "  stable-diffusion.cpp: " + CHIMERA_SDCPP_VERSION + "\n"
        + "  sqlite:               " + chimera_db::sqlite_version() + "\n"
        + "  sqlite-vec:           " + chimera_db::sqlite_vec_version();
}

// ----------------------------------------------------------------------------
// CLI wiring
// ----------------------------------------------------------------------------
//
// CLI11 binding lived inline in main() until the try-block grew past 470
// lines. The blocks below extract each subcommand's option wiring into a
// small `bind_*_cmd` helper so the file reads as "one subcommand per
// helper" instead of one giant call list. ParsedCli holds the CLI11
// subcommand pointers + every option struct + every local string/int
// that's filled by parsing. dispatch_cli() runs the matched subcommand
// and returns its exit code.
//
// The split is mechanical — no behaviour change. Each helper still owns
// the same options the inline block owned, in the same order, with the
// same defaults and help strings.

struct ParsedCli {
    // Subcommand pointers, set by the bind_*_cmd helpers and read back
    // in dispatch_cli to decide which subcommand was activated.
    CLI::App * prompt_cmd       = nullptr;
    CLI::App * chat_cmd         = nullptr;
    CLI::App * tokenize_cmd     = nullptr;
    CLI::App * embed_cmd        = nullptr;
#ifdef CHIMERA_HAS_WHISPER
    CLI::App * whisper_cmd      = nullptr;
#endif
#ifdef CHIMERA_HAS_SD
    CLI::App * sd_cmd           = nullptr;
#endif
    CLI::App * serve_cmd        = nullptr;
    CLI::App * db_status_cmd    = nullptr;
    CLI::App * db_backup_cmd    = nullptr;
    CLI::App * db_vacuum_cmd    = nullptr;
    CLI::App * info_cmd         = nullptr;
    CLI::App * index_create_cmd = nullptr;
    CLI::App * index_ingest_cmd = nullptr;
    CLI::App * index_list_cmd   = nullptr;
    CLI::App * index_stats_cmd  = nullptr;
    CLI::App * index_drop_cmd   = nullptr;
    CLI::App * search_cmd       = nullptr;

    // `gen`
    LlamaCommonOptions prompt_opts;
    std::string        prompt_text;
    std::string        prompt_file;

    // `chat`
    LlamaCommonOptions chat_opts;
    std::string        system_prompt;
    std::string        system_prompt_file;
    std::string        template_override;
    std::string        color_arg          = "auto";
    bool               chat_persist       = false;
    bool               chat_list          = false;
    std::string        chat_search;
    std::string        chat_resume;
    std::string        chat_db_path;
    int                chat_list_limit    = 20;

    // `tokenize`, `embed`, `whisper`, `sd`, `serve`
    TokenizeOptions tokenize_opts;
    EmbedOptions    embed_opts;
#ifdef CHIMERA_HAS_WHISPER
    WhisperOptions  whisper_opts;
#endif
#ifdef CHIMERA_HAS_SD
    SdOptions       sd_opts;
#endif
    ServeOptions    serve_opts;

    // `db`
    std::string db_path_override;
    std::string db_backup_dst;

    // `index`
    std::string              idx_db_path;
    std::string              idx_name;
    std::string              idx_embedding_model;
    int                      idx_ctx_size              = 0;
    int                      idx_threads               = -1;
    int                      idx_gpu_layers            = 0;
    std::string              idx_pooling               = "mean";
    std::vector<std::string> idx_files;
    std::string              idx_glob;
    std::string              idx_distance              = "cosine";
    int                      idx_chunk_tokens          = 512;
    int                      idx_chunk_overlap         = 64;
    int                      idx_chunk_tokens_override  = 0;
    int                      idx_chunk_overlap_override = -1;
    bool                     idx_cache_embeddings      = false;

    // `search`
    std::string srch_db_path;
    std::string srch_name;
    std::string srch_query;
    int         srch_k                = 5;
    int         srch_ctx_size         = 0;
    int         srch_threads          = -1;
    int         srch_gpu_layers       = 0;
    std::string srch_pooling          = "mean";
    bool        srch_cache_embeddings = false;
    std::string srch_mode             = "hybrid";
};

// Bind flags shared by `gen` and `chat`. These all map to fields on
// LlamaCommonOptions; the helper keeps the two subcommand bindings in
// sync without copy-paste drift.
void bind_llama_common_opts(CLI::App * cmd, LlamaCommonOptions & o) {
    // Performance / cache
    cmd->add_flag("--flash-attn", o.flash_attn, "Enable Flash Attention");
    cmd->add_option("--cache-type-k", o.cache_type_k,
        "KV cache K type: f32 | f16 | bf16 | q8_0 | q5_0 | q5_1 | q4_0 | q4_1 | iq4_nl")
        ->check(CLI::IsMember({"f32","f16","bf16","q8_0","q5_0","q5_1","q4_0","q4_1","iq4_nl"}));
    cmd->add_option("--cache-type-v", o.cache_type_v,
        "KV cache V type (same values as --cache-type-k)")
        ->check(CLI::IsMember({"f32","f16","bf16","q8_0","q5_0","q5_1","q4_0","q4_1","iq4_nl"}));
    cmd->add_option("--ubatch-size", o.n_ubatch,
        "Physical batch size (0 = follow --batch-size)");

    // Penalties + sampling
    cmd->add_option("--repeat-last-n", o.penalty_last_n,
        "Tokens scanned for repeat penalty (0 = disable, -1 = ctx-size)");
    cmd->add_option("--presence-penalty", o.penalty_present, "Presence penalty");
    cmd->add_option("--frequency-penalty", o.penalty_freq, "Frequency penalty");
    cmd->add_option("--mirostat", o.mirostat, "Mirostat: 0 disabled, 1 v1, 2 v2");
    cmd->add_option("--mirostat-ent", o.mirostat_tau, "Mirostat target entropy (tau)");
    cmd->add_option("--mirostat-lr",  o.mirostat_eta, "Mirostat learning rate (eta)");
    cmd->add_option("--dry-multiplier", o.dry_multiplier, "DRY repetition penalty multiplier");
    cmd->add_option("--dry-base", o.dry_base, "DRY repetition penalty base");
    cmd->add_option("--dry-allowed-length", o.dry_allowed_length,
        "DRY: tokens extending repetition beyond this receive penalty");
    cmd->add_option("--dry-penalty-last-n", o.dry_penalty_last_n,
        "DRY: tokens scanned for repetitions (-1 = ctx-size)");
    cmd->add_option("--dry-sequence-breaker", o.dry_sequence_breakers,
        "DRY sequence breaker (repeatable)");
    cmd->add_option("--logit-bias", o.logit_bias,
        "Bias a token (\"<id>(+|-|=)<bias>\"; repeatable)");
    cmd->add_flag("--ignore-eos", o.ignore_eos, "Ignore end-of-stream tokens");

    // Grammar / JSON schema (resolved later; mutually exclusive)
    cmd->add_option("--grammar", o.grammar, "GBNF grammar");
    cmd->add_option("--grammar-file", o.grammar_file, "Read GBNF grammar from file");
    cmd->add_option("--json-schema", o.json_schema,
        "Constrain output to a JSON schema (converted to grammar)");
    cmd->add_option("--json-schema-file", o.json_schema_file,
        "Read JSON schema from file");

    // LoRA
    cmd->add_option("--lora", o.lora_adapters,
        "LoRA adapter as path[:scale] (scale defaults to 1.0; repeatable)");

    // RoPE / YaRN
    cmd->add_option("--rope-freq-base", o.rope_freq_base, "RoPE base frequency (0 = from model)");
    cmd->add_option("--rope-freq-scale", o.rope_freq_scale, "RoPE frequency scale (0 = from model)");
    cmd->add_option("--rope-scale", o.rope_freq_scale,
        "Inverse of --rope-freq-scale; alias for compatibility");
    cmd->add_option("--rope-scaling", o.rope_scaling,
        "RoPE scaling: none | linear | yarn | longrope")
        ->check(CLI::IsMember({"none","linear","yarn","longrope"}));
    cmd->add_option("--yarn-orig-ctx",   o.yarn_orig_ctx,   "YaRN original context size");
    cmd->add_option("--yarn-ext-factor", o.yarn_ext_factor, "YaRN extrapolation mix factor (negative = from model)");
    cmd->add_option("--yarn-attn-factor",o.yarn_attn_factor,"YaRN magnitude scaling factor");
    cmd->add_option("--yarn-beta-fast",  o.yarn_beta_fast,  "YaRN low correction dim");
    cmd->add_option("--yarn-beta-slow",  o.yarn_beta_slow,  "YaRN high correction dim");

    // Multi-GPU
    cmd->add_option("--main-gpu", o.main_gpu,
        "Index of the GPU used when --split-mode=none");
    cmd->add_option("--tensor-split", o.tensor_split,
        "Per-GPU offload proportions (comma-separated floats)");
    cmd->add_option("--split-mode", o.split_mode,
        "Multi-GPU split mode: none | layer | row | tensor")
        ->check(CLI::IsMember({"none","layer","row","tensor"}));
    cmd->add_option("--device", o.devices,
        "Comma-separated device list (names from `ggml_backend_dev_by_name`)");

    // Mmap / mlock
    cmd->add_flag("!--no-mmap", o.use_mmap, "Disable mmap'ing model weights");
    cmd->add_flag("--mlock", o.use_mlock, "Force the system to keep model in RAM");

    // mmproj offload
    cmd->add_flag("!--no-mmproj-offload", o.mmproj_use_gpu,
        "Run the multimodal projector on CPU (defaults to GPU)");
}

void bind_gen_cmd(CLI::App & app, ParsedCli & p) {
    auto * cmd = app.add_subcommand("gen", "One-shot llama text generation");
    cmd->add_option("-m,--model", p.prompt_opts.model, "GGUF model")->required();
    cmd->add_option("-p,--prompt", p.prompt_text, "Prompt text");
    cmd->add_option("-f,--prompt-file", p.prompt_file,
        "Read prompt from file (use - for stdin)");
    cmd->add_option("-n,--n-predict", p.prompt_opts.n_predict, "Tokens to generate");
    cmd->add_option("-c,--ctx-size", p.prompt_opts.n_ctx, "Context size");
    cmd->add_option("-b,--batch-size", p.prompt_opts.n_batch, "Prompt batch size");
    cmd->add_option("-t,--threads", p.prompt_opts.threads, "CPU threads");
    cmd->add_option("--gpu-layers", p.prompt_opts.gpu_layers, "Layers to offload");
    cmd->add_option("--seed", p.prompt_opts.seed, "Sampler seed");
    cmd->add_option("--temp", p.prompt_opts.temp, "Temperature");
    cmd->add_option("--top-k", p.prompt_opts.top_k, "Top-k");
    cmd->add_option("--top-p", p.prompt_opts.top_p, "Top-p");
    cmd->add_option("--min-p", p.prompt_opts.min_p, "Min-p");
    cmd->add_option("--repeat-penalty", p.prompt_opts.repeat_penalty, "Repeat penalty");
    cmd->add_option("--mmproj", p.prompt_opts.mmproj,
        "Multimodal projector (mmproj GGUF) for vision/audio input");
    cmd->add_option("--image", p.prompt_opts.images,
        "Image to feed alongside the prompt (repeatable; requires --mmproj)");
    bind_llama_common_opts(cmd, p.prompt_opts);
    p.prompt_cmd = cmd;
}

void bind_chat_cmd(CLI::App & app, ParsedCli & p) {
    auto * cmd = app.add_subcommand("chat", "Minimal interactive llama chat");
    // --model is required for an interactive session but NOT for
    // --list / --search / --resume (which can read the model name
    // from the saved chat row). We enforce it after parse instead
    // of via CLI11's ->required() so the print-and-exit paths work
    // without a model argument.
    cmd->add_option("-m,--model", p.chat_opts.model, "GGUF model");
    cmd->add_option("-n,--n-predict", p.chat_opts.n_predict, "Tokens to generate per turn");
    cmd->add_option("-c,--ctx-size", p.chat_opts.n_ctx, "Context size");
    cmd->add_option("-b,--batch-size", p.chat_opts.n_batch, "Prompt batch size");
    cmd->add_option("-t,--threads", p.chat_opts.threads, "CPU threads");
    cmd->add_option("--gpu-layers", p.chat_opts.gpu_layers, "Layers to offload");
    cmd->add_option("--seed", p.chat_opts.seed, "Sampler seed");
    cmd->add_option("--temp", p.chat_opts.temp, "Temperature");
    cmd->add_option("--top-k", p.chat_opts.top_k, "Top-k");
    cmd->add_option("--top-p", p.chat_opts.top_p, "Top-p");
    cmd->add_option("--min-p", p.chat_opts.min_p, "Min-p");
    cmd->add_option("--repeat-penalty", p.chat_opts.repeat_penalty, "Repeat penalty");
    cmd->add_option("--system", p.system_prompt, "System prompt");
    cmd->add_option("--system-prompt-file", p.system_prompt_file,
        "Read system prompt from file");
    cmd->add_option("--chat-template", p.template_override, "Chat template override");

    // Persistence flags. `--list` and `--search` are print-and-exit;
    // they don't load a model. `--persist` opts a live session into
    // per-turn DB writes; `--resume <id|last>` loads a saved
    // conversation and continues from where it ended.
    cmd->add_flag("--persist",  p.chat_persist,
        "Save this chat to the embedded SQLite DB (off by default)");
    cmd->add_option("--resume",  p.chat_resume,
        "Resume a saved chat by id, or 'last' for the most recent");
    cmd->add_flag("--list",     p.chat_list,
        "List saved chats and exit (no model load)");
    cmd->add_option("--search",  p.chat_search,
        "Full-text-search saved messages and exit (no model load)");
    cmd->add_option("--list-limit", p.chat_list_limit,
        "Cap for --list / --search results");
    cmd->add_option("--db",      p.chat_db_path,
        "Path to the DB file (default: $CHIMERA_DB or platform default)");
    cmd->add_option("--mmproj", p.chat_opts.mmproj,
        "Multimodal projector (mmproj GGUF) enabling /image and /audio");
    cmd->add_option("--color", p.color_arg,
        "Colorize input/output: auto | always | never")
        ->check(CLI::IsMember({"auto", "always", "never"}));

    // Chat template + jinja + reasoning
    cmd->add_option("--chat-template-file", p.chat_opts.chat_template_file,
        "Read the chat template override from a file");
    cmd->add_option("--chat-template-kwargs", p.chat_opts.chat_template_kwargs,
        "Extra key=value pairs passed to the jinja template (repeatable)");
    cmd->add_flag("!--no-jinja", p.chat_opts.use_jinja,
        "Disable the jinja chat template renderer (default: enabled)");
    cmd->add_option("--reasoning", p.chat_opts.reasoning,
        "Reasoning format: none | deepseek | granite | ...");
    cmd->add_option("--reasoning-format", p.chat_opts.reasoning_format,
        "Alias of --reasoning");
    cmd->add_option("--reasoning-budget", p.chat_opts.reasoning_budget,
        "Token budget for reasoning content (-1 = disabled)");
    cmd->add_option("--reasoning-budget-message", p.chat_opts.reasoning_budget_message,
        "Message injected before the closing thought tag when the budget is exhausted");

    bind_llama_common_opts(cmd, p.chat_opts);
    p.chat_cmd = cmd;
}

void bind_tokenize_cmd(CLI::App & app, ParsedCli & p) {
    auto * cmd = app.add_subcommand("tokenize", "Tokenize text via a GGUF vocab");
    cmd->add_option("-m,--model", p.tokenize_opts.model, "GGUF model")->required();
    cmd->add_option("-p,--prompt", p.tokenize_opts.input, "Text to tokenize");
    cmd->add_option("-f,--prompt-file", p.tokenize_opts.input_file,
        "Read text from file (use - for stdin)");
    cmd->add_flag("!--no-bos", p.tokenize_opts.add_special,
        "Do not prepend BOS / model-special tokens");
    cmd->add_flag("!--no-special", p.tokenize_opts.parse_special,
        "Do not parse <|special|> tokens in the input");
    cmd->add_flag("--pieces", p.tokenize_opts.show_pieces,
        "Also print the decoded piece next to each id");
    p.tokenize_cmd = cmd;
}

void bind_embed_cmd(CLI::App & app, ParsedCli & p) {
    auto * cmd = app.add_subcommand("embed", "Emit an embedding vector for text");
    cmd->add_option("-m,--model", p.embed_opts.model, "GGUF embedding model")->required();
    cmd->add_option("-p,--prompt", p.embed_opts.input, "Text to embed");
    cmd->add_option("-f,--prompt-file", p.embed_opts.input_file,
        "Read text from file (use - for stdin)");
    cmd->add_option("-o,--output", p.embed_opts.output,
        "Output file (default: stdout)");
    cmd->add_option("--pooling", p.embed_opts.pooling,
        "Pooling: mean | cls | last | none");
    cmd->add_option("-c,--ctx-size", p.embed_opts.n_ctx,
        "Context size (0 = model's training context)");
    cmd->add_option("-b,--batch-size", p.embed_opts.n_batch, "Batch size");
    cmd->add_option("-t,--threads", p.embed_opts.threads, "CPU threads");
    cmd->add_option("--gpu-layers", p.embed_opts.gpu_layers, "Layers to offload");
    cmd->add_flag("--cache-embeddings", p.embed_opts.cache_embeddings,
        "Memoize embed(text) -> vector in SQLite so repeats skip the model");
    cmd->add_option("--cache-db", p.embed_opts.cache_db,
        "Path to the SQLite DB used by --cache-embeddings "
        "(default: $CHIMERA_DB or platform default)");
    cmd->add_flag("!--no-normalize", p.embed_opts.normalize,
        "Do not L2-normalize the output vector");
    cmd->add_option("--ubatch-size", p.embed_opts.n_ubatch,
        "Physical batch size (0 = follow --batch-size)");
    cmd->add_flag("--flash-attn", p.embed_opts.flash_attn, "Enable Flash Attention");
    cmd->add_flag("!--no-mmap", p.embed_opts.use_mmap, "Disable mmap'ing model weights");
    cmd->add_flag("--mlock", p.embed_opts.use_mlock, "Force the system to keep model in RAM");
    cmd->add_option("--rope-freq-base",  p.embed_opts.rope_freq_base,  "RoPE base frequency (0 = from model)");
    cmd->add_option("--rope-freq-scale", p.embed_opts.rope_freq_scale, "RoPE frequency scale (0 = from model)");
    cmd->add_option("--rope-scale",      p.embed_opts.rope_freq_scale, "Alias of --rope-freq-scale");
    cmd->add_option("--rope-scaling",    p.embed_opts.rope_scaling,
        "RoPE scaling: none | linear | yarn | longrope")
        ->check(CLI::IsMember({"none","linear","yarn","longrope"}));
    cmd->add_option("--yarn-orig-ctx",    p.embed_opts.yarn_orig_ctx,    "YaRN original context size");
    cmd->add_option("--yarn-ext-factor",  p.embed_opts.yarn_ext_factor,  "YaRN extrapolation mix factor");
    cmd->add_option("--yarn-attn-factor", p.embed_opts.yarn_attn_factor, "YaRN magnitude scaling factor");
    cmd->add_option("--yarn-beta-fast",   p.embed_opts.yarn_beta_fast,   "YaRN low correction dim");
    cmd->add_option("--yarn-beta-slow",   p.embed_opts.yarn_beta_slow,   "YaRN high correction dim");
    cmd->add_option("--main-gpu",     p.embed_opts.main_gpu,     "Index of the GPU used when --split-mode=none");
    cmd->add_option("--tensor-split", p.embed_opts.tensor_split, "Per-GPU offload proportions (comma-separated)");
    cmd->add_option("--split-mode",   p.embed_opts.split_mode,
        "Multi-GPU split mode: none | layer | row | tensor")
        ->check(CLI::IsMember({"none","layer","row","tensor"}));
    cmd->add_option("--device", p.embed_opts.devices,
        "Comma-separated device list");
    p.embed_cmd = cmd;
}

#ifdef CHIMERA_HAS_WHISPER
void bind_whisper_cmd(CLI::App & app, ParsedCli & p) {
    auto * cmd = app.add_subcommand("whisper", "Minimal whisper transcription");
    cmd->add_option("-m,--model", p.whisper_opts.model, "Whisper model")->required();
    cmd->add_option("-i,--input", p.whisper_opts.input, "Input WAV file")->required();
    cmd->add_option("-o,--output", p.whisper_opts.output, "Output text file");
    cmd->add_option("-t,--threads", p.whisper_opts.threads, "CPU threads");
    cmd->add_option("-l,--language", p.whisper_opts.language, "Language or auto");
    cmd->add_flag("--translate", p.whisper_opts.translate, "Translate to English");
    cmd->add_flag("--timestamps", p.whisper_opts.timestamps, "Print segment timestamps");
    cmd->add_flag("--no-context", p.whisper_opts.no_context, "Disable previous-text conditioning");
    // Note: CLI11 forbids multi-char short flags, so the upstream
    // whisper-cli aliases (-of/-otxt/-osrt/...) are long-only here.
    cmd->add_option("--output-file", p.whisper_opts.output_file_base,
        "Base name for format files (default: input WAV stem). "
        "Each --output-* flag writes <base>.<ext>.");
    cmd->add_flag("--output-txt",  p.whisper_opts.out_txt,  "Write .txt (one line per segment)");
    cmd->add_flag("--output-srt",  p.whisper_opts.out_srt,  "Write .srt subtitles");
    cmd->add_flag("--output-vtt",  p.whisper_opts.out_vtt,  "Write .vtt (WebVTT) subtitles");
    cmd->add_flag("--output-json", p.whisper_opts.out_json, "Write .json (segments + text)");
    cmd->add_flag("--output-json-full", p.whisper_opts.out_json_full,
        "Write .json with per-word timestamps (implies word-level timing)");
    cmd->add_flag("--output-csv",  p.whisper_opts.out_csv,  "Write .csv (start_ms,end_ms,text)");
    cmd->add_flag("--output-lrc",  p.whisper_opts.out_lrc,  "Write .lrc karaoke");
    p.whisper_cmd = cmd;
}
#endif

#ifdef CHIMERA_HAS_SD
void bind_sd_cmd(CLI::App & app, ParsedCli & p) {
    auto * cmd = app.add_subcommand("sd", "Minimal stable-diffusion text-to-image");
    cmd->add_option("-m,--model", p.sd_opts.model,
                    "Combined single-file checkpoint (classic SD/SDXL). "
                    "For split layouts (Z-Image, Flux, SD3) use --diffusion-model.");
    cmd->add_option("--diffusion-model", p.sd_opts.diffusion_model,
                    "Diffusion UNet/DiT file (split layout)");
    cmd->add_option("--vae", p.sd_opts.vae, "Separate VAE file");
    cmd->add_option("--clip-l", p.sd_opts.clip_l, "CLIP-L text encoder");
    cmd->add_option("--t5xxl", p.sd_opts.t5xxl, "T5-XXL text encoder");
    cmd->add_option("--llm", p.sd_opts.llm,
                    "LLM text encoder (e.g. Qwen3 for Z-Image)");
    cmd->add_flag("--offload-to-cpu", p.sd_opts.offload_to_cpu,
                  "Offload model params to CPU (saves VRAM)");
    cmd->add_flag("--diffusion-fa", p.sd_opts.diffusion_fa,
                  "Enable flash-attention in the diffusion model");
    cmd->add_option("-p,--prompt", p.sd_opts.prompt, "Prompt")->required();
    cmd->add_option("-o,--output", p.sd_opts.output, "Output PNG path");
    cmd->add_option("--negative-prompt", p.sd_opts.negative_prompt, "Negative prompt");
    cmd->add_option("-W,--width", p.sd_opts.width, "Image width");
    cmd->add_option("-H,--height", p.sd_opts.height, "Image height");
    cmd->add_option("-s,--steps", p.sd_opts.steps, "Sampling steps");
    cmd->add_option("-b,--batch-count", p.sd_opts.batch_count, "Image count");
    cmd->add_option("-t,--threads", p.sd_opts.threads, "CPU threads");
    cmd->add_option("--seed", p.sd_opts.seed, "Seed");
    cmd->add_option("--cfg-scale", p.sd_opts.cfg_scale, "CFG scale");
    cmd->add_option("--clip-skip", p.sd_opts.clip_skip, "CLIP skip");
    cmd->add_option("--sample-method", p.sd_opts.sample_method, "Sampling method");
    cmd->add_option("--scheduler", p.sd_opts.scheduler, "Scheduler");
    cmd->add_option("--init-image", p.sd_opts.init_image,
        "Initial image for img2img / inpaint (must match -W,-H)");
    cmd->add_option("--mask-image", p.sd_opts.mask_image,
        "Inpaint mask (single-channel; requires --init-image)");
    cmd->add_option("--strength", p.sd_opts.strength,
        "img2img denoising strength (0=preserve init, 1=full noise)");
    p.sd_cmd = cmd;
}
#endif

void bind_serve_cmd(CLI::App & app, ParsedCli & p) {
    auto * cmd = app.add_subcommand("serve",
        "OpenAI-compatible HTTP server (LLM only for now)");
    cmd->add_option("-m,--model", p.serve_opts.model, "GGUF model")->required();
    cmd->add_option("--mmproj", p.serve_opts.mmproj,
        "Multimodal projector for vision/audio inputs in chat completions");
    cmd->add_option("--host", p.serve_opts.host, "Bind address");
    cmd->add_option("--port", p.serve_opts.port, "Bind port");
    cmd->add_option("-c,--ctx-size", p.serve_opts.n_ctx,
        "Context size (0 = model's training context)");
    cmd->add_option("-b,--batch-size", p.serve_opts.n_batch,
        "Logical batch size for prompt processing");
    cmd->add_option("--ubatch-size", p.serve_opts.n_ubatch,
        "Physical batch size (auto-clamped to batch when --embeddings)");
    cmd->add_option("-t,--threads", p.serve_opts.threads, "CPU threads");
    cmd->add_option("--gpu-layers", p.serve_opts.gpu_layers, "Layers to offload");
    cmd->add_option("--parallel", p.serve_opts.parallel,
        "Number of concurrent request slots");
    cmd->add_option("--api-key", p.serve_opts.api_key,
        "Bearer token required on /v1/* requests (empty = no auth)");
    cmd->add_flag("--embeddings", p.serve_opts.embedding,
        "Load the model in embedding mode (enables /v1/embeddings)");
#ifdef CHIMERA_HAS_WHISPER
    cmd->add_option("--enable-audio", p.serve_opts.audio_model,
        "Whisper GGUF to load alongside the LLM (enables /v1/audio/transcriptions)");
#endif
#ifdef CHIMERA_HAS_SD
    cmd->add_option("--enable-image", p.serve_opts.sd_model,
        "Stable-diffusion GGUF to load alongside the LLM (enables /v1/images/*)");
#endif
    cmd->add_option("--enable-rag", p.serve_opts.rag_embedding_model,
        "Embedding GGUF to load alongside the LLM (enables /v1/vector_stores/*)");
    cmd->add_option("--enable-embeddings", p.serve_opts.embed_model,
        "Embedding GGUF to load alongside the LLM (routes /v1/embeddings to it)");
    cmd->add_option("--reranking", p.serve_opts.rerank_model,
        "Cross-encoder reranker GGUF to load alongside the LLM (enables /v1/rerank)");
    cmd->add_flag("--cache-embeddings", p.serve_opts.cache_embeddings,
        "Memoize RAG embeddings in --rag-db (no-op unless --enable-rag is set)");
    cmd->add_option("--rag-db", p.serve_opts.rag_db_path,
        "Path to the SQLite DB used by /v1/vector_stores/* "
        "(default: $CHIMERA_DB or platform default)");
    cmd->add_flag("--persist-chats", p.serve_opts.persist_chats,
        "Save every /v1/chat/completions exchange to the chats table");
    cmd->add_option("--chat-db", p.serve_opts.chat_db_path,
        "Path to the SQLite DB used by --persist-chats "
        "(default: $CHIMERA_DB or platform default)");
    cmd->add_option("--slot-save-path", p.serve_opts.slot_save_path,
        "Directory for KV-cache snapshots written/read by "
        "POST /slots/:id?action={save,restore} (GET /slots works regardless)");
    cmd->add_option("--lora", p.serve_opts.lora_adapters,
        "LoRA adapter to load alongside the base model as path[:scale] "
        "(scale defaults to 1.0; repeatable). Enables POST /lora-adapters "
        "to hot-swap which adapters are active without reloading.");
    cmd->add_flag("!--no-webui", p.serve_opts.webui,
        "Disable the embedded web chat UI at GET / (only meaningful in "
        "builds compiled with CHIMERA_WEBUI_EMBED=1; a no-op otherwise)");
    cmd->add_option("--public-path", p.serve_opts.public_path,
        "Directory to serve as static files at GET / (chimera-specific UI). "
        "Independent of CHIMERA_WEBUI_EMBED; when both apply, --public-path wins");
    p.serve_cmd = cmd;
}

void bind_db_cmd(CLI::App & app, ParsedCli & p) {
    auto * db_cmd = app.add_subcommand("db", "Embedded SQLite database management");
    db_cmd->add_option("--db", p.db_path_override,
        "Path to the DB file (default: $CHIMERA_DB or platform default)");
    db_cmd->require_subcommand(1);

    auto * db_status_cmd = db_cmd->add_subcommand("status",
        "Open the DB, run pending migrations, print path + version + schema info");
    p.db_status_cmd = db_status_cmd;

    auto * db_backup_cmd = db_cmd->add_subcommand("backup",
        "Snapshot the DB to another path via `VACUUM INTO`. The destination "
        "must not already exist. Single self-contained file — no WAL/SHM "
        "siblings to copy.");
    db_backup_cmd->add_option("--to", p.db_backup_dst,
        "Destination path for the snapshot")->required();
    p.db_backup_cmd = db_backup_cmd;

    auto * db_vacuum_cmd = db_cmd->add_subcommand("vacuum",
        "Defragment + reclaim free pages in place. No other chimera "
        "processes may have this DB open (SQLite serializes on the file).");
    p.db_vacuum_cmd = db_vacuum_cmd;
}

void bind_info_cmd(CLI::App & app, ParsedCli & p) {
    p.info_cmd = app.add_subcommand("info", "Print versions of bundled component");
}

void bind_index_cmd(CLI::App & app, ParsedCli & p) {
    auto * index_cmd = app.add_subcommand("index", "Vector store management");
    index_cmd->require_subcommand(1);
    index_cmd->add_option("--db", p.idx_db_path,
        "Path to the DB file (default: $CHIMERA_DB or platform default)");

    auto * create_cmd = index_cmd->add_subcommand("create",
        "Create a collection (sized to the embedding model's dim)");
    create_cmd->add_option("-n,--name", p.idx_name, "Collection name")->required();
    create_cmd->add_option("-e,--embedding-model", p.idx_embedding_model,
        "GGUF embedding model (recorded on the collection)")->required();
    create_cmd->add_option("-c,--ctx-size", p.idx_ctx_size, "Context size");
    create_cmd->add_option("-t,--threads", p.idx_threads, "CPU threads");
    create_cmd->add_option("--gpu-layers", p.idx_gpu_layers, "Layers to offload");
    create_cmd->add_option("--pooling", p.idx_pooling,
        "Pooling: mean | cls | last | none");
    create_cmd->add_option("--distance", p.idx_distance,
        "Distance metric on the vec0 table: cosine | l2 | l1 "
        "(default cosine; right for L2-normalized embeddings)");
    create_cmd->add_option("--chunk-tokens", p.idx_chunk_tokens,
        "Default tokens per chunk for this collection (default 512). "
        "Token units of the embedding model's vocab; not characters.");
    create_cmd->add_option("--chunk-overlap", p.idx_chunk_overlap,
        "Default token overlap between chunks (default 64)");
    p.index_create_cmd = create_cmd;

    auto * ingest_cmd = index_cmd->add_subcommand("ingest",
        "Chunk + embed + insert one or more text files into a collection");
    ingest_cmd->add_option("-n,--name", p.idx_name, "Collection name")->required();
    ingest_cmd->add_option("-f,--file", p.idx_files,
        "File to ingest (repeatable)");
    ingest_cmd->add_option("-g,--glob", p.idx_glob,
        "Glob pattern relative to cwd (e.g. 'docs/**/*.md')");
    ingest_cmd->add_option("-c,--ctx-size", p.idx_ctx_size, "Context size");
    ingest_cmd->add_option("-t,--threads", p.idx_threads, "CPU threads");
    ingest_cmd->add_option("--gpu-layers", p.idx_gpu_layers, "Layers to offload");
    ingest_cmd->add_option("--pooling", p.idx_pooling,
        "Pooling: mean | cls | last | none");
    ingest_cmd->add_option("--chunk-tokens", p.idx_chunk_tokens_override,
        "Tokens per chunk for this ingest call (overrides the collection's "
        "recorded chunk_tokens; 0 = use the collection default)");
    ingest_cmd->add_option("--chunk-overlap", p.idx_chunk_overlap_override,
        "Token overlap between chunks for this ingest call (overrides the "
        "collection's recorded chunk_overlap; -1 = use the collection default)");
    ingest_cmd->add_flag("--cache-embeddings", p.idx_cache_embeddings,
        "Memoize per-chunk embed(text) -> vector in --db so re-ingesting "
        "the same content skips the model");
    p.index_ingest_cmd = ingest_cmd;

    p.index_list_cmd  = index_cmd->add_subcommand("list",
        "List collections and their document counts");

    auto * stats_cmd = index_cmd->add_subcommand("stats",
        "Show details for one collection");
    stats_cmd->add_option("-n,--name", p.idx_name, "Collection name")->required();
    p.index_stats_cmd = stats_cmd;

    auto * drop_cmd  = index_cmd->add_subcommand("drop",
        "Drop a collection and all its documents");
    drop_cmd->add_option("-n,--name", p.idx_name, "Collection name")->required();
    p.index_drop_cmd = drop_cmd;
}

void bind_search_cmd(CLI::App & app, ParsedCli & p) {
    auto * cmd = app.add_subcommand("search",
        "Search a vector-store collection by similarity");
    cmd->add_option("--db", p.srch_db_path,
        "Path to the DB file (default: $CHIMERA_DB or platform default)");
    cmd->add_option("-n,--name", p.srch_name, "Collection name")->required();
    cmd->add_option("-q,--query", p.srch_query, "Search query text")->required();
    cmd->add_option("-k,--top-k", p.srch_k, "Number of hits to return");
    cmd->add_option("-c,--ctx-size", p.srch_ctx_size, "Context size");
    cmd->add_option("-t,--threads", p.srch_threads, "CPU threads");
    cmd->add_option("--gpu-layers", p.srch_gpu_layers, "Layers to offload");
    cmd->add_option("--pooling", p.srch_pooling,
        "Pooling: mean | cls | last | none");
    cmd->add_flag("--cache-embeddings", p.srch_cache_embeddings,
        "Memoize embed(query) -> vector in --db so repeated searches "
        "with the same query skip the model");
    cmd->add_option("--mode", p.srch_mode,
        "Retrieval mode: semantic | lexical | hybrid (default: hybrid)");
    p.search_cmd = cmd;
}

void bind_subcommands(CLI::App & app, ParsedCli & p) {
    bind_gen_cmd     (app, p);
    bind_chat_cmd    (app, p);
    bind_tokenize_cmd(app, p);
    bind_embed_cmd   (app, p);
#ifdef CHIMERA_HAS_WHISPER
    bind_whisper_cmd (app, p);
#endif
#ifdef CHIMERA_HAS_SD
    bind_sd_cmd      (app, p);
#endif
    bind_serve_cmd   (app, p);
    bind_db_cmd      (app, p);
    bind_info_cmd    (app, p);
    bind_index_cmd   (app, p);
    bind_search_cmd  (app, p);
}

// Resolve the activated subcommand and invoke its command_* entry
// point. Mutating chat_opts.model for `--resume` and resolving the
// prompt for `gen` happen here rather than inside command_chat /
// command_prompt to keep the entry points free of CLI11 dependencies.
int dispatch_cli(ParsedCli & p) {
    if (*p.prompt_cmd) {
        const std::string resolved = resolve_prompt(p.prompt_text, p.prompt_file);
        return command_prompt(p.prompt_opts, resolved);
    }
    if (*p.chat_cmd) {
        // Print-and-exit branches first: --list and --search never
        // load the model and don't need --model on the command line.
        if (p.chat_list) {
            return command_chat_list(p.chat_db_path, p.chat_list_limit);
        }
        if (!p.chat_search.empty()) {
            return command_chat_search(p.chat_db_path, p.chat_search, p.chat_list_limit);
        }
        if (p.chat_opts.model.empty() && p.chat_resume.empty()) {
            fail(ExitCode::BadInput,
                 "chat: -m/--model is required (or --resume <id|last>)");
        }
        // --resume without --model: pick up the model path
        // recorded on the saved chat. The user can override by
        // passing -m explicitly.
        if (!p.chat_resume.empty() && p.chat_opts.model.empty()) {
            auto conn = chimera_db::open_and_migrate(
                p.chat_db_path.empty()
                    ? chimera_db::default_path() : p.chat_db_path);
            std::optional<chimera_chat_store::Chat> existing;
            if (p.chat_resume == "last" || p.chat_resume == "latest") {
                existing = chimera_chat_store::latest_chat(conn.get());
            } else {
                try {
                    existing = chimera_chat_store::load_chat(
                        conn.get(), std::stoll(p.chat_resume));
                } catch (const std::exception &) {
                    fail(ExitCode::BadInput,
                         "invalid --resume value: '" + p.chat_resume + "'");
                }
            }
            if (!existing) {
                fail(ExitCode::BadInput,
                     "no such chat: '" + p.chat_resume + "'");
            }
            p.chat_opts.model = existing->model_path;
        }
        std::string resolved_system = p.system_prompt;
        if (!p.system_prompt_file.empty()) {
            if (!p.system_prompt.empty()) {
                fail(ExitCode::BadInput,
                    "use only one of --system / --system-prompt-file");
            }
            resolved_system = read_file(p.system_prompt_file);
        }
        ChatPersistence persist_cfg;
        persist_cfg.persist  = p.chat_persist;
        persist_cfg.resume   = p.chat_resume;
        persist_cfg.db_path  = p.chat_db_path;
        std::string resolved_template = p.template_override;
        if (!p.chat_opts.chat_template_file.empty()) {
            if (!resolved_template.empty()) {
                fail(ExitCode::BadInput,
                    "use only one of --chat-template / --chat-template-file");
            }
            resolved_template = read_file(p.chat_opts.chat_template_file);
        }
        return command_chat(p.chat_opts, resolved_system, resolved_template,
                            parse_color_mode(p.color_arg), persist_cfg);
    }
    if (*p.tokenize_cmd)     return command_tokenize    (p.tokenize_opts);
    if (*p.embed_cmd)        return command_embed       (p.embed_opts);
#ifdef CHIMERA_HAS_WHISPER
    if (*p.whisper_cmd)      return command_whisper     (p.whisper_opts);
#endif
#ifdef CHIMERA_HAS_SD
    if (*p.sd_cmd)           return command_sd          (p.sd_opts);
#endif
    if (*p.serve_cmd)        return command_serve       (p.serve_opts);
    if (*p.db_status_cmd)    return command_db_status   (p.db_path_override);
    if (*p.db_backup_cmd)    return command_db_backup   (p.db_path_override, p.db_backup_dst);
    if (*p.db_vacuum_cmd)    return command_db_vacuum   (p.db_path_override);
    if (*p.info_cmd)         return command_info        ();
    if (*p.index_create_cmd) {
        return command_index_create(p.idx_db_path, p.idx_name, p.idx_embedding_model,
                                     p.idx_ctx_size, p.idx_threads, p.idx_gpu_layers,
                                     p.idx_pooling, p.idx_distance,
                                     p.idx_chunk_tokens, p.idx_chunk_overlap);
    }
    if (*p.index_ingest_cmd) {
        return command_index_ingest(p.idx_db_path, p.idx_name, p.idx_files, p.idx_glob,
                                     p.idx_ctx_size, p.idx_threads, p.idx_gpu_layers,
                                     p.idx_pooling,
                                     p.idx_chunk_tokens_override,
                                     p.idx_chunk_overlap_override,
                                     p.idx_cache_embeddings);
    }
    if (*p.index_list_cmd)   return command_index_list  (p.idx_db_path);
    if (*p.index_stats_cmd)  return command_index_stats (p.idx_db_path, p.idx_name);
    if (*p.index_drop_cmd)   return command_index_drop  (p.idx_db_path, p.idx_name);
    if (*p.search_cmd) {
        return command_search(p.srch_db_path, p.srch_name, p.srch_query, p.srch_k,
                               p.srch_ctx_size, p.srch_threads, p.srch_gpu_layers,
                               p.srch_pooling, p.srch_cache_embeddings, p.srch_mode);
    }
    // app.require_subcommand(1) guarantees one of the above matched.
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    silence_all_logging();

    CLI::App app("chimera - {llama,whisper,stable-diffusion}.cpp multitool");
    auto fmt = std::make_shared<CompactFormatter>();
    // Pack short + long flags together in --help instead of padding short
    // flags to ~1/3 of the option column (CLI11 default 30 * 1/3 = 10 chars).
    fmt->long_option_alignment_ratio(0.0f);
    app.formatter(fmt);
    // CLI11's default make_usage() prepends an extra '\n', producing a
    // double blank line between description and usage. Setting an explicit
    // usage string skips that branch.
    app.usage("Usage: " + std::string(argv[0]) + " [OPTIONS] SUBCOMMAND");
    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "Enable native-backend logging");
    app.set_version_flag("-V,--version", &version_string,
        "Show version and bundled library versions");
    app.require_subcommand(1);

    ParsedCli parsed;
    bind_subcommands(app, parsed);

    bool backend_initialized = false;
    try {
        app.parse(argc, argv);
        if (verbose) {
            restore_default_logging();
        }
        llama_backend_init();
        backend_initialized = true;
        const int rc = dispatch_cli(parsed);
        llama_backend_free();
        return rc;
    } catch (const CLI::ParseError & e) {
        if (backend_initialized) llama_backend_free();
        return app.exit(e);
    } catch (const ChimeraError & e) {
        std::cerr << "error: " << e.what() << '\n';
        if (backend_initialized) llama_backend_free();
        return static_cast<int>(e.code());
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << '\n';
        if (backend_initialized) llama_backend_free();
        return static_cast<int>(ExitCode::Runtime);
    }
}

