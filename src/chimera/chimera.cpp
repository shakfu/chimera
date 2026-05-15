#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "CLI11.hpp"

#include "chat.h"
#include "chimera.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

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
    chimera_silence_whisper_log();
    chimera_silence_sd_log();
}

void restore_default_logging() {
    llama_log_set(nullptr, nullptr);
    ggml_log_set(nullptr, nullptr);
    chimera_restore_whisper_log();
    chimera_restore_sd_log();
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

LlamaModelPtr load_llama_model(const LlamaCommonOptions & opts) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = opts.gpu_layers;
    params.use_mmap = opts.use_mmap;

    LlamaModelPtr model(llama_model_load_from_file(opts.model.c_str(), params));
    if (!model) {
        fail(ExitCode::Load, "failed to load llama model: " + opts.model);
    }
    return model;
}

LlamaContextPtr new_llama_context(llama_model * model, const LlamaCommonOptions & opts, size_t min_prompt_tokens) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = std::max<uint32_t>(opts.n_ctx, static_cast<uint32_t>(min_prompt_tokens + opts.n_predict + 32));
    params.n_batch = std::max<uint32_t>(1, opts.n_batch);
    params.n_ubatch = params.n_batch;
    params.n_threads = opts.threads;
    params.n_threads_batch = opts.threads;
    params.no_perf = true;

    LlamaContextPtr ctx(llama_init_from_model(model, params));
    if (!ctx) {
        fail(ExitCode::Load, "failed to create llama context");
    }
    return ctx;
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
    sampling.no_perf = true;

    common_sampler * sampler = common_sampler_init(model, sampling);
    if (sampler == nullptr) {
        fail(ExitCode::Load, "failed to create sampler");
    }
    return common_sampler_ptr(sampler);
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
    auto model = load_llama_model(opts);
    std::string text = run_generation(model.get(), opts, prompt, true, true);
    return text.empty() ? static_cast<int>(ExitCode::Generate) : 0;
}

// Length of the longest common token prefix between two sequences.
size_t common_prefix(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

int command_chat(const LlamaCommonOptions & opts, const std::string & system_prompt, const std::string & template_override) {
    auto model = load_llama_model(opts);
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    common_chat_templates_ptr templates = common_chat_templates_init(model.get(), template_override, "", "");
    if (!templates) {
        fail(ExitCode::Load, "failed to initialize chat template");
    }

    // Build one context big enough for the whole session. min_prompt_tokens
    // is a starting hint; we let the user grow it via -c.
    auto ctx = new_llama_context(model.get(), opts, /*min_prompt_tokens=*/0);
    auto sampler = make_sampler(model.get(), opts);
    llama_memory_t mem = llama_get_memory(ctx.get());

    std::vector<common_chat_msg> history;
    if (!system_prompt.empty()) {
        history.push_back(make_chat_msg("system", system_prompt));
    }

    // Tokens already resident in the KV cache (prompt + generated assistant
    // tokens). On each turn we re-tokenize the templated prompt, find the
    // longest common prefix with this, rewind the KV to that boundary, and
    // decode only the tail. This keeps the bulk of prior context warm.
    std::vector<llama_token> kv_tokens;
    constexpr llama_seq_id seq_id = 0;

    std::string line;
    while (true) {
        std::cout << "> " << std::flush;
        if (!std::getline(std::cin, line)) {
            break;
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        if (line == "/exit" || line == "/quit") {
            break;
        }

        history.push_back(make_chat_msg("user", line));

        common_chat_templates_inputs inputs;
        inputs.messages = history;
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;
        common_chat_params params = common_chat_templates_apply(templates.get(), inputs);

        // First turn: tokenize with BOS / model-special tokens. Subsequent
        // turns: skip BOS so it doesn't reappear mid-stream (the template's
        // own delimiters fence the new turn).
        const bool add_special = kv_tokens.empty();
        const auto full_tokens = tokenize(vocab, params.prompt, add_special, true);

        // Find the boundary at which the new templated prompt diverges
        // from what the KV already holds, and roll the KV back to it.
        const size_t shared = common_prefix(kv_tokens, full_tokens);
        if (shared < kv_tokens.size()) {
            llama_memory_seq_rm(mem, seq_id,
                                static_cast<llama_pos>(shared),
                                static_cast<llama_pos>(kv_tokens.size()));
        }
        // Decode only the new tail. Always at least one token: even with
        // an unchanged prompt, the generation prompt suffix differs.
        std::vector<llama_token> tail(full_tokens.begin() + static_cast<ptrdiff_t>(shared),
                                      full_tokens.end());
        if (tail.empty()) {
            // Same prompt as last turn somehow; force re-decoding the last
            // token so the model has fresh logits to sample from.
            tail.push_back(full_tokens.back());
            llama_memory_seq_rm(mem, seq_id,
                                static_cast<llama_pos>(full_tokens.size() - 1),
                                static_cast<llama_pos>(full_tokens.size()));
        }
        decode_tokens(ctx.get(), tail, static_cast<int32_t>(opts.n_batch));
        for (llama_token tok : tail) {
            common_sampler_accept(sampler.get(), tok, false);
        }

        // Sample the assistant turn. Each generated token is decoded as
        // it arrives (inside sample_loop), so it lives in the KV cache by
        // the time we record it in kv_tokens.
        std::vector<llama_token> generated;
        std::string reply = sample_loop(ctx.get(), sampler.get(), vocab,
                                        opts.n_predict, true, &generated);

        // Update bookkeeping for the next turn.
        kv_tokens = std::move(full_tokens);
        kv_tokens.insert(kv_tokens.end(), generated.begin(), generated.end());
        history.push_back(make_chat_msg("assistant", reply));
    }

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

    LlamaCommonOptions load_opts;
    load_opts.model = opts.model;
    load_opts.gpu_layers = opts.gpu_layers;
    load_opts.use_mmap = opts.use_mmap;
    auto model = load_llama_model(load_opts);
    const llama_vocab * vocab = llama_model_get_vocab(model.get());

    const auto tokens = tokenize(vocab, text, true, true);
    if (tokens.empty()) {
        fail(ExitCode::BadInput, "input tokenized to zero tokens");
    }

    const int32_t n_embd = llama_model_n_embd(model.get());
    const uint32_t n_train = llama_model_n_ctx_train(model.get());
    const uint32_t requested_ctx = opts.n_ctx == 0 ? n_train : opts.n_ctx;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = std::max<uint32_t>(requested_ctx, static_cast<uint32_t>(tokens.size()));
    cparams.n_batch = std::max<uint32_t>(opts.n_batch, static_cast<uint32_t>(tokens.size()));
    cparams.n_ubatch = cparams.n_batch;
    cparams.n_threads = opts.threads;
    cparams.n_threads_batch = opts.threads;
    cparams.embeddings = true;
    cparams.pooling_type = parse_pooling(opts.pooling);
    cparams.no_perf = true;

    LlamaContextPtr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx) {
        fail(ExitCode::Load, "failed to create llama context");
    }

    if (llama_decode(ctx.get(),
                     llama_batch_get_one(const_cast<llama_token *>(tokens.data()),
                                         static_cast<int32_t>(tokens.size()))) != 0) {
        fail(ExitCode::Generate, "failed to decode input for embedding");
    }

    const enum llama_pooling_type ptype = cparams.pooling_type;
    const float * emb = nullptr;
    if (ptype == LLAMA_POOLING_TYPE_NONE) {
        emb = llama_get_embeddings(ctx.get());
    } else {
        emb = llama_get_embeddings_seq(ctx.get(), 0);
        if (emb == nullptr) {
            emb = llama_get_embeddings_ith(ctx.get(),
                                           static_cast<int32_t>(tokens.size()) - 1);
        }
    }
    if (emb == nullptr) {
        fail(ExitCode::Generate, "no embeddings produced (model may not support pooling)");
    }

    std::vector<float> vec(emb, emb + n_embd);
    if (opts.normalize) {
        double norm_sq = 0.0;
        for (float v : vec) norm_sq += static_cast<double>(v) * v;
        const float norm = static_cast<float>(std::sqrt(norm_sq));
        if (norm > 0.0f) {
            for (float & v : vec) v /= norm;
        }
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
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) *out << ' ';
        *out << vec[i];
    }
    *out << '\n';
    return 0;
}

std::string version_string() {
    return std::string("chimera ") + CHIMERA_VERSION + "\n"
        + "  llama.cpp:            " + CHIMERA_LLAMACPP_VERSION + "\n"
        + "  whisper.cpp:          " + CHIMERA_WHISPERCPP_VERSION + "\n"
        + "  stable-diffusion.cpp: " + CHIMERA_SDCPP_VERSION;
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

    bool backend_initialized = false;
    try {
        app.require_subcommand(1);

        LlamaCommonOptions prompt_opts;
        std::string prompt_text;
        std::string prompt_file;
        auto * prompt_cmd = app.add_subcommand("gen", "One-shot llama text generation");
        prompt_cmd->add_option("-m,--model", prompt_opts.model, "GGUF model")->required();
        prompt_cmd->add_option("-p,--prompt", prompt_text, "Prompt text");
        prompt_cmd->add_option("-f,--prompt-file", prompt_file,
            "Read prompt from file (use - for stdin)");
        prompt_cmd->add_option("-n,--n-predict", prompt_opts.n_predict, "Tokens to generate");
        prompt_cmd->add_option("-c,--ctx-size", prompt_opts.n_ctx, "Context size");
        prompt_cmd->add_option("-b,--batch-size", prompt_opts.n_batch, "Prompt batch size");
        prompt_cmd->add_option("-t,--threads", prompt_opts.threads, "CPU threads");
        prompt_cmd->add_option("--gpu-layers", prompt_opts.gpu_layers, "Layers to offload");
        prompt_cmd->add_option("--seed", prompt_opts.seed, "Sampler seed");
        prompt_cmd->add_option("--temp", prompt_opts.temp, "Temperature");
        prompt_cmd->add_option("--top-k", prompt_opts.top_k, "Top-k");
        prompt_cmd->add_option("--top-p", prompt_opts.top_p, "Top-p");
        prompt_cmd->add_option("--min-p", prompt_opts.min_p, "Min-p");
        prompt_cmd->add_option("--repeat-penalty", prompt_opts.repeat_penalty, "Repeat penalty");

        LlamaCommonOptions chat_opts;
        std::string system_prompt;
        std::string system_prompt_file;
        std::string template_override;
        auto * chat_cmd = app.add_subcommand("chat", "Minimal interactive llama chat");
        chat_cmd->add_option("-m,--model", chat_opts.model, "GGUF model")->required();
        chat_cmd->add_option("-n,--n-predict", chat_opts.n_predict, "Tokens to generate per turn");
        chat_cmd->add_option("-c,--ctx-size", chat_opts.n_ctx, "Context size");
        chat_cmd->add_option("-b,--batch-size", chat_opts.n_batch, "Prompt batch size");
        chat_cmd->add_option("-t,--threads", chat_opts.threads, "CPU threads");
        chat_cmd->add_option("--gpu-layers", chat_opts.gpu_layers, "Layers to offload");
        chat_cmd->add_option("--seed", chat_opts.seed, "Sampler seed");
        chat_cmd->add_option("--temp", chat_opts.temp, "Temperature");
        chat_cmd->add_option("--top-k", chat_opts.top_k, "Top-k");
        chat_cmd->add_option("--top-p", chat_opts.top_p, "Top-p");
        chat_cmd->add_option("--min-p", chat_opts.min_p, "Min-p");
        chat_cmd->add_option("--repeat-penalty", chat_opts.repeat_penalty, "Repeat penalty");
        chat_cmd->add_option("--system", system_prompt, "System prompt");
        chat_cmd->add_option("--system-prompt-file", system_prompt_file,
            "Read system prompt from file");
        chat_cmd->add_option("--chat-template", template_override, "Chat template override");

        TokenizeOptions tokenize_opts;
        auto * tokenize_cmd = app.add_subcommand("tokenize", "Tokenize text via a GGUF vocab");
        tokenize_cmd->add_option("-m,--model", tokenize_opts.model, "GGUF model")->required();
        tokenize_cmd->add_option("-p,--prompt", tokenize_opts.input, "Text to tokenize");
        tokenize_cmd->add_option("-f,--prompt-file", tokenize_opts.input_file,
            "Read text from file (use - for stdin)");
        tokenize_cmd->add_flag("!--no-bos", tokenize_opts.add_special,
            "Do not prepend BOS / model-special tokens");
        tokenize_cmd->add_flag("!--no-special", tokenize_opts.parse_special,
            "Do not parse <|special|> tokens in the input");
        tokenize_cmd->add_flag("--pieces", tokenize_opts.show_pieces,
            "Also print the decoded piece next to each id");

        EmbedOptions embed_opts;
        auto * embed_cmd = app.add_subcommand("embed", "Emit an embedding vector for text");
        embed_cmd->add_option("-m,--model", embed_opts.model, "GGUF embedding model")->required();
        embed_cmd->add_option("-p,--prompt", embed_opts.input, "Text to embed");
        embed_cmd->add_option("-f,--prompt-file", embed_opts.input_file,
            "Read text from file (use - for stdin)");
        embed_cmd->add_option("-o,--output", embed_opts.output,
            "Output file (default: stdout)");
        embed_cmd->add_option("--pooling", embed_opts.pooling,
            "Pooling: mean | cls | last | none");
        embed_cmd->add_option("-c,--ctx-size", embed_opts.n_ctx,
            "Context size (0 = model's training context)");
        embed_cmd->add_option("-b,--batch-size", embed_opts.n_batch, "Batch size");
        embed_cmd->add_option("-t,--threads", embed_opts.threads, "CPU threads");
        embed_cmd->add_option("--gpu-layers", embed_opts.gpu_layers, "Layers to offload");
        embed_cmd->add_flag("!--no-normalize", embed_opts.normalize,
            "Do not L2-normalize the output vector");

        WhisperOptions whisper_opts;
        auto * whisper_cmd = app.add_subcommand("whisper", "Minimal whisper transcription");
        whisper_cmd->add_option("-m,--model", whisper_opts.model, "Whisper model")->required();
        whisper_cmd->add_option("-i,--input", whisper_opts.input, "Input WAV file")->required();
        whisper_cmd->add_option("-o,--output", whisper_opts.output, "Output text file");
        whisper_cmd->add_option("-t,--threads", whisper_opts.threads, "CPU threads");
        whisper_cmd->add_option("-l,--language", whisper_opts.language, "Language or auto");
        whisper_cmd->add_flag("--translate", whisper_opts.translate, "Translate to English");
        whisper_cmd->add_flag("--timestamps", whisper_opts.timestamps, "Print segment timestamps");
        whisper_cmd->add_flag("--no-context", whisper_opts.no_context, "Disable previous-text conditioning");

        SdOptions sd_opts;
        auto * sd_cmd = app.add_subcommand("sd", "Minimal stable-diffusion text-to-image");
        sd_cmd->add_option("-m,--model", sd_opts.model, "Diffusion model")->required();
        sd_cmd->add_option("-p,--prompt", sd_opts.prompt, "Prompt")->required();
        sd_cmd->add_option("-o,--output", sd_opts.output, "Output PNG path");
        sd_cmd->add_option("--negative-prompt", sd_opts.negative_prompt, "Negative prompt");
        sd_cmd->add_option("-W,--width", sd_opts.width, "Image width");
        sd_cmd->add_option("-H,--height", sd_opts.height, "Image height");
        sd_cmd->add_option("-s,--steps", sd_opts.steps, "Sampling steps");
        sd_cmd->add_option("-b,--batch-count", sd_opts.batch_count, "Image count");
        sd_cmd->add_option("-t,--threads", sd_opts.threads, "CPU threads");
        sd_cmd->add_option("--seed", sd_opts.seed, "Seed");
        sd_cmd->add_option("--cfg-scale", sd_opts.cfg_scale, "CFG scale");
        sd_cmd->add_option("--clip-skip", sd_opts.clip_skip, "CLIP skip");
        sd_cmd->add_option("--sample-method", sd_opts.sample_method, "Sampling method");
        sd_cmd->add_option("--scheduler", sd_opts.scheduler, "Scheduler");
        sd_cmd->add_option("--init-image", sd_opts.init_image,
            "Initial image for img2img / inpaint (must match -W,-H)");
        sd_cmd->add_option("--mask-image", sd_opts.mask_image,
            "Inpaint mask (single-channel; requires --init-image)");
        sd_cmd->add_option("--strength", sd_opts.strength,
            "img2img denoising strength (0=preserve init, 1=full noise)");

        app.parse(argc, argv);

        if (verbose) {
            restore_default_logging();
        }

        llama_backend_init();
        backend_initialized = true;

        int rc = 0;
        if (*prompt_cmd) {
            const std::string resolved = resolve_prompt(prompt_text, prompt_file);
            rc = command_prompt(prompt_opts, resolved);
        } else if (*chat_cmd) {
            std::string resolved_system = system_prompt;
            if (!system_prompt_file.empty()) {
                if (!system_prompt.empty()) {
                    fail(ExitCode::BadInput,
                        "use only one of --system / --system-prompt-file");
                }
                resolved_system = read_file(system_prompt_file);
            }
            rc = command_chat(chat_opts, resolved_system, template_override);
        } else if (*tokenize_cmd) {
            rc = command_tokenize(tokenize_opts);
        } else if (*embed_cmd) {
            rc = command_embed(embed_opts);
        } else if (*whisper_cmd) {
            rc = command_whisper(whisper_opts);
        } else if (*sd_cmd) {
            rc = command_sd(sd_opts);
        }

        llama_backend_free();
        return rc;
    } catch (const CLI::ParseError & e) {
        if (backend_initialized) {
            llama_backend_free();
        }
        return app.exit(e);
    } catch (const ChimeraError & e) {
        std::cerr << "error: " << e.what() << '\n';
        if (backend_initialized) {
            llama_backend_free();
        }
        return static_cast<int>(e.code());
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << '\n';
        if (backend_initialized) {
            llama_backend_free();
        }
        return static_cast<int>(ExitCode::Runtime);
    }
}
