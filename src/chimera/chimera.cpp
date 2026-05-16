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
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>  // isatty
#else
#include <io.h>
#define isatty _isatty
#define STDIN_FILENO _fileno(stdin)
#define STDOUT_FILENO _fileno(stdout)
#define STDERR_FILENO _fileno(stderr)
#endif

#include "CLI11.hpp"
#include "rang.hpp"

#include "chat.h"
#include "chimera.h"
#include "common.h"
#include "ggml.h"
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
    User,    // the '> ' prompt + user-typed input
    Cmd,     // slash-command names in banner / help
    Think,   // model reasoning_content (between <think>...</think>)
    Stats,   // per-turn 'Prompt: X t/s | Generation: Y t/s' line
    Info,    // dim info notices ("attached text from ...", "history cleared")
    Err,     // errors
};

inline std::ostream & operator<<(std::ostream & os, Sem s) {
    switch (s) {
        case Sem::Reset: return os << rang::style::reset;
        case Sem::User:  return os << rang::fg::green;
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
    chimera_silence_whisper_log();
    chimera_silence_sd_log();
}

void restore_default_logging() {
    llama_log_set(nullptr, nullptr);
    ggml_log_set(nullptr, nullptr);
    mtmd_helper_log_set(nullptr, nullptr);
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

// Sample a chat reply, routing reasoning content (e.g. text inside
// <think>...</think>) through Sem::Think. Each new token is appended to
// the raw accumulator and the running text is re-parsed with
// common_chat_parse (is_partial=true); the resulting message is diffed
// against the previous parse to produce streaming content / reasoning
// deltas. Returns the *content* portion of the reply (without reasoning),
// which is what we want to store in chat history — the next turn's
// templating shouldn't reinject the model's prior thinking.
std::string chat_sample_loop(
    llama_context * ctx,
    common_sampler * sampler,
    const llama_vocab * vocab,
    int n_predict,
    const common_chat_parser_params & parser_params,
    std::vector<llama_token> * out_tokens) {

    std::string raw;
    std::string content;
    common_chat_msg prev;

    for (int i = 0; i < n_predict; ++i) {
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
    sampling.no_perf = true;

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
    // Leave use_gpu at its default (true): the vision encoder is a separate
    // backend choice from --gpu-layers, which only controls LLM offload.
    mparams.n_threads = opts.threads;
    mparams.print_timings = false;

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
        }
    }

    mtmd_input_text input_text;
    input_text.text = final_prompt.c_str();
    input_text.add_special = true;
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

// Best-effort identification of the "modalities" line on the banner.
std::string describe_modalities(mtmd_context * mctx) {
    std::string s = "text";
    if (mctx && mtmd_support_vision(mctx)) s += ", vision";
    if (mctx && mtmd_support_audio(mctx))  s += ", audio";
    return s;
}

int command_chat(const LlamaCommonOptions & opts,
                 const std::string & system_prompt,
                 const std::string & template_override,
                 ColorMode color_mode) {
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
        mparams.n_threads = opts.threads;
        mparams.print_timings = false;
        mctx.reset(mtmd_init_from_file(opts.mmproj.c_str(), model.get(), mparams));
        spinner.stop();
        if (!mctx) {
            fail(ExitCode::Load, "failed to load mmproj: " + opts.mmproj);
        }
    }

    auto ctx     = new_llama_context(model.get(), opts, /*min_prompt_tokens=*/0);
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

    std::string         cur_text_prefix;   // /read /glob accumulator
    std::vector<Media>  pending_media;     // /image /audio for the next turn
    std::vector<Media>  conv_media;        // all media attached so far, in order

    std::vector<llama_token> kv_tokens;
    constexpr llama_seq_id seq_id = 0;
    bool multimodal_active = false;        // sticky: set on first attach

    const bool can_image = mctx && mtmd_support_vision(mctx.get());
    const bool can_audio = mctx && mtmd_support_audio(mctx.get());

    // ---- banner / startup help -----------------------------------------
    const std::string modalities = describe_modalities(mctx.get());
    const std::string model_name = std::filesystem::path(opts.model).filename().string();
    std::cout << "build      : " << CHIMERA_LLAMACPP_VERSION << "\n"
              << "model      : " << model_name << "\n"
              << "modalities : " << modalities << "\n";
    if (!system_prompt.empty()) {
        std::cout << "using custom system prompt\n";
    }
    std::cout << "\ntype " << Sem::Cmd << "/help" << Sem::Reset
              << " for commands; " << Sem::Cmd << "/exit" << Sem::Reset
              << " or Ctrl-D to quit.\n\n";

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
    // ESC bytes as visible columns). Instead we emit the green SGR escape
    // to stdout right before calling linenoise_read; the SGR state persists
    // across linenoise's cursor moves so both the prompt characters and the
    // user's typed input render green. We reset the SGR after the call.
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
        // Emit the green SGR raw (rang would emit the same bytes; we use the
        // string form so it's identical to what we reset with below, and so
        // we can flush without a manipulator dance).
        if (color_on) std::cout << "\x1b[32m" << std::flush;
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
            std::cout << Sem::Info << "chat history cleared."
                      << Sem::Reset << "\n";
            continue;
        } else if (line == "/regen") {
            bool dropped = false;
            while (!history.empty() && history.back().role == "assistant") {
                history.pop_back();
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
        inputs.use_jinja = true;
        common_chat_params params = common_chat_templates_apply(templates.get(), inputs);

        // Parser config for streaming chat output. DEEPSEEK reasoning
        // format covers <think>...</think> spans (the de-facto standard
        // across DeepSeek, Qwen3-thinking, and most open reasoning models).
        common_chat_parser_params parser_params(params);
        parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

        std::vector<llama_token> generated;
        std::string reply;
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
            input_text.add_special  = true;
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
            reply = chat_sample_loop(ctx.get(), sampler.get(), vocab,
                                     opts.n_predict, parser_params, &generated);
            t_gen = secs(clock::now() - t1);
        } else {
            // Text-only fast path: KV-prefix reuse via token comparison.
            const bool add_special = kv_tokens.empty();
            const auto full_tokens = tokenize(vocab, params.prompt, add_special, true);

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
            reply = chat_sample_loop(ctx.get(), sampler.get(), vocab,
                                     opts.n_predict, parser_params, &generated);
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

        history.push_back(make_chat_msg("assistant", reply));
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
        prompt_cmd->add_option("--mmproj", prompt_opts.mmproj,
            "Multimodal projector (mmproj GGUF) for vision/audio input");
        prompt_cmd->add_option("--image", prompt_opts.images,
            "Image to feed alongside the prompt (repeatable; requires --mmproj)");

        LlamaCommonOptions chat_opts;
        std::string system_prompt;
        std::string system_prompt_file;
        std::string template_override;
        std::string color_arg = "auto";
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
        chat_cmd->add_option("--mmproj", chat_opts.mmproj,
            "Multimodal projector (mmproj GGUF) enabling /image and /audio");
        chat_cmd->add_option("--color", color_arg,
            "Colorize input/output: auto | always | never")
            ->check(CLI::IsMember({"auto", "always", "never"}));

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
            rc = command_chat(chat_opts, resolved_system, template_override,
                              parse_color_mode(color_arg));
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
