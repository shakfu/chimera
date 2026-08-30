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
#include "chimera_llama.h"
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


// Resolve a prompt from either --prompt or --prompt-file. Exactly one must
// be supplied; both empty is a usage error. "-" reads stdin.

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
    chimera_set_sd_log_verbose(false);
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
    // sd.cpp installs no default printer, so restoring its callback to null
    // is not enough -- load_model() reinstalls chimera's, and this is what
    // makes it echo INFO/DEBUG rather than warnings only.
    chimera_set_sd_log_verbose(true);
#endif
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
    bool * out_interrupted = nullptr,
    bool reasoning_control = false) {

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
            // With --reasoning-control, the first Ctrl-C while the model is
            // actively thinking ends the reasoning block (the model then
            // answers) instead of aborting. force() transitions only from
            // the COUNTING state and returns false otherwise (not yet
            // thinking, already ending, or already answering), in which
            // case we fall through to the normal abort. Clearing the flag
            // lets the loop continue; a later Ctrl-C then aborts as usual.
            if (reasoning_control &&
                common_sampler_reasoning_budget_force(sampler)) {
                g_chat_interrupt_requested.store(false, std::memory_order_relaxed);
                std::cout << Sem::Think << "\n[reasoning ended]\n"
                          << Sem::Reset << std::flush;
            } else {
                if (out_interrupted) *out_interrupted = true;
                break;
            }
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
        starts_with_sv(head, "/audio ") ||
        starts_with_sv(head, "/video ");
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
        if (opts.image_min_tokens > 0) mparams.image_min_tokens = opts.image_min_tokens;
        if (opts.image_max_tokens > 0) mparams.image_max_tokens = opts.image_max_tokens;
        mctx.reset(mtmd_init_from_file(opts.mmproj.c_str(), model.get(), mparams));
        spinner.stop();
        if (!mctx) {
            fail(ExitCode::Load, "failed to load mmproj: " + opts.mmproj);
        }
    }

    auto ctx     = new_llama_context(model.get(), opts, /*min_prompt_tokens=*/0);
    auto loras   = load_loras(model.get(), ctx.get(), opts.lora_adapters);

    // Resolve the active chat template's thinking tags once, up-front,
    // so we can install the reasoning-budget sampler inside the
    // long-lived `sampler` below. The tags are template-fixed (not
    // message-dependent) — common_chat_templates_apply populates
    // chat_params.thinking_start_tag/thinking_end_tags from the format, so a
    // dummy single-user-message apply is sufficient to read them out.
    // If the template doesn't advertise thinking tags (most non-
    // reasoning models), the tags come back empty and make_sampler
    // skips the budget wiring — matching the "no budget" path.
    // Both --reasoning-budget (fixed token cap) and --reasoning-control
    // (Ctrl-C ends thinking at runtime) drive the same reasoning-budget
    // sampler and need the template's thinking tags resolved here.
    ReasoningBudgetParams rbp;
    if (opts.reasoning_budget >= 0 || opts.reasoning_control) {
        common_chat_templates_inputs tpl_probe;
        common_chat_msg probe_msg;
        probe_msg.role    = "user";
        probe_msg.content = "probe";
        tpl_probe.messages.push_back(probe_msg);
        tpl_probe.add_generation_prompt = true;
        tpl_probe.use_jinja             = opts.use_jinja;
        common_chat_params probe = common_chat_templates_apply(templates.get(), tpl_probe);
        rbp.vocab              = vocab;
        rbp.thinking_start_tag = probe.thinking_start_tag;
        rbp.thinking_end_tags  = probe.thinking_end_tags;
        rbp.budget             = opts.reasoning_budget;
        rbp.budget_message     = opts.reasoning_budget_message;
        rbp.control            = opts.reasoning_control;
        if (probe.thinking_start_tag.empty() || probe.thinking_end_tags.empty()) {
            const char * which = opts.reasoning_budget >= 0
                ? "--reasoning-budget" : "--reasoning-control";
            std::cerr << "chimera: warning: " << which << " set but the "
                         "active chat template advertises no thinking tags; "
                         "it will be ignored (the budget sampler activates "
                         "on the template-provided start tag).\n";
        }
    }
    auto sampler = make_sampler(model.get(), opts, rbp);
    llama_memory_t mem = llama_get_memory(ctx.get());

    struct Media {
        std::string   path;
        MtmdBitmapPtr bitmap;
        // Non-null only for video files: the lazy bitmap's frame callback reads
        // from this ctx during mtmd_tokenize(), so it must live as long as bitmap.
        MtmdVideoPtr  video;
        // Video lazy bitmaps are single-use (frames are consumed at tokenize).
        // Because this REPL re-tokenizes the whole history every turn, video
        // entries defer decoding and are rebuilt from `path` before each turn;
        // `bitmap`/`video` are left null until then. Images decode once at attach.
        bool          is_video = false;
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
    const bool can_video = mctx && mtmd_helper_support_video(mctx.get());

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
        if (can_video) cmd_line("/video <file>       ", "attach a video to the next message (requires ffmpeg)");
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
    if (can_video) completion_state.cmds.push_back("/video");
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
        // Upstream returns a wrapper {bitmap, video_ctx}; placeholder=false so the
        // bitmap holds real data. video_ctx is non-null only when the file decoded
        // as video.
        mtmd_helper_bitmap_wrapper w =
            mtmd_helper_bitmap_init_from_file(mctx.get(), path.c_str(), /*placeholder=*/false);
        if (!w.bitmap) {
            std::cerr << Sem::Err << "failed to load media: " << path
                      << Sem::Reset << "\n";
            return false;
        }
        if (w.video_ctx) {
            // Auto-dispatched to video. Single-use lazy bitmaps can't survive the
            // per-turn re-tokenize, so drop this probe decode and defer to the
            // per-turn rebuild (which re-decodes from `path`).
            mtmd_bitmap_free(w.bitmap);
            mtmd_helper_video_free(w.video_ctx);
            pending_media.push_back({path, nullptr, nullptr, /*is_video=*/true});
            std::cout << Sem::Info << "attached video from '" << path << "'"
                      << Sem::Reset << "\n";
            return true;
        }
        pending_media.push_back({path, MtmdBitmapPtr(w.bitmap), nullptr, /*is_video=*/false});
        std::cout << Sem::Info << "attached " << (need_vision ? "image" : "audio")
                  << " from '" << path << "'" << Sem::Reset << "\n";
        return true;
    };

    // Explicit video attach: records the path for the per-turn rebuild, which
    // decodes via the video decoder honoring the session's --video-fps /
    // --video-timestamp-ms / --ffmpeg-dir params. Decoding is deferred (not done
    // here) because the lazy bitmap is single-use and would be consumed by the
    // first turn's tokenize; the rebuild re-decodes fresh from `path` each turn.
    auto attach_video = [&](const std::string & path) -> bool {
        if (!mctx) {
            std::cerr << Sem::Err << "multimodal not loaded: pass --mmproj <gguf>"
                      << Sem::Reset << "\n";
            return false;
        }
        if (!can_video) {
            std::cerr << Sem::Err << "video requires a build with video support "
                         "(MTMD_VIDEO) and a vision-capable mmproj"
                      << Sem::Reset << "\n";
            return false;
        }
        if (!std::filesystem::exists(path)) {
            std::cerr << Sem::Err << "no such file: " << path << Sem::Reset << "\n";
            return false;
        }
        pending_media.push_back({path, nullptr, nullptr, /*is_video=*/true});
        std::cout << Sem::Info << "attached video from '" << path << "'"
                  << Sem::Reset << "\n";
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
        } else if (starts_with_sv(line, "/video ")) {
            if (attach_video(trim(line.substr(7)))) {
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

            // Video entries carry single-use lazy bitmaps that the previous turn's
            // tokenize already consumed, so re-decode each one fresh from its path
            // before re-tokenizing. Images keep their bitmap across turns.
            bool video_rebuild_ok = true;
            for (auto & m : conv_media) {
                if (!m.is_video) continue;
                ChimeraVideoBitmap v = load_video_lazy_bitmap(
                    mctx.get(), m.path, opts.video_fps, opts.video_timestamp_ms,
                    opts.ffmpeg_dir);
                if (!v.bitmap) {
                    std::cerr << Sem::Err << "failed to decode video (is ffmpeg "
                                 "installed?): " << m.path << Sem::Reset << "\n";
                    video_rebuild_ok = false;
                    break;
                }
                m.bitmap.reset(v.bitmap);
                m.video.reset(v.video);
            }
            if (!video_rebuild_ok) continue;

            std::vector<const mtmd_bitmap *> bitmaps_c;
            bitmaps_c.reserve(conv_media.size());
            for (const auto & m : conv_media) bitmaps_c.push_back(m.bitmap.get());

            // text_len is load-bearing: newer mtmd bounds its media-marker
            // scan by this length rather than strlen(text). Leaving it unset
            // makes mtmd_tokenize see zero markers and fail rc=2.
            mtmd_input_text input_text{};
            input_text.text = params.prompt.c_str();
            input_text.text_len = params.prompt.size();
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
                                         &reply_reasoning, &reply_interrupted,
                                         opts.reasoning_control);
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
                                         &reply_reasoning, &reply_interrupted,
                                         opts.reasoning_control);
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

int command_info(bool list_devices_only = false) {
    if (list_devices_only) {
        const size_t n_dev = ggml_backend_dev_count();
        for (size_t i = 0; i < n_dev; ++i) {
            auto * d = ggml_backend_dev_get(i);
            const char * dn = ggml_backend_dev_name(d);
            std::cout << (dn ? dn : "?") << '\n';
        }
        return 0;
    }
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
    bool       info_list_devices = false;
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
        "DRY: tokens scanned for repetitions (0 = disable, -1 = ctx-size)");
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
    cmd->add_option("--load-mode",o.load_mode,
        "Model loading mode, overriding --no-mmap/--mlock: "
        "auto | none | mmap | mlock | mmap+mlock | dio (direct I/O where supported)")
        ->check(CLI::IsMember({"auto","none","mmap","mlock","mmap+mlock","dio"}));

    // mmproj offload
    cmd->add_flag("!--no-mmproj-offload", o.mmproj_use_gpu,
        "Run the multimodal projector on CPU (defaults to GPU)");

    // Extra samplers (Wave 1: typical / top-nsigma / xtc / dynatemp).
    cmd->add_option("--typical", o.typ_p,
        "Locally typical-p sampling (1.0 = disabled)");
    cmd->add_option("--top-nsigma", o.top_n_sigma,
        "top-n-sigma sampling (-1 = disabled)");
    cmd->add_option("--xtc-probability", o.xtc_probability,
        "XTC sampler: probability of removing a top token (0 = disabled)");
    cmd->add_option("--xtc-threshold", o.xtc_threshold,
        "XTC sampler: probability threshold (values > 0.5 also disable XTC)");
    cmd->add_option("--dynatemp-range", o.dynatemp_range,
        "Dynamic temperature range (0 = disabled)");
    cmd->add_option("--dynatemp-exp", o.dynatemp_exp,
        "Dynamic temperature exponent (entropy→temperature mapping)");
    cmd->add_option("--samplers", o.samplers,
        "Sampler chain order, ';'-separated. Example: "
        "\"dry;top_k;typ_p;top_p;min_p;xtc;temperature\". Empty = upstream default.");

    // Perf
    cmd->add_option("--threads-batch", o.threads_batch,
        "Threads for batch (prompt-prefill) processing; -1 = mirror --threads");
    cmd->add_flag("--swa-full", o.swa_full,
        "Use the full-size SWA cache (slower but exposes the full window)");

    // Vision-token budget (mtmd_context_params; only meaningful with --mmproj).
    cmd->add_option("--image-min-tokens", o.image_min_tokens,
        "Minimum tokens per image (-1 = leave model default)");
    cmd->add_option("--image-max-tokens", o.image_max_tokens,
        "Maximum tokens per image (-1 = leave model default)");

    // MoE expert offload (manipulates tensor_buft_overrides under the hood).
    cmd->add_flag("--cpu-moe", o.cpu_moe,
        "Keep ALL Mixture-of-Experts weights on CPU");
    cmd->add_option("--n-cpu-moe", o.n_cpu_moe,
        "Keep MoE weights of the first N layers on CPU (0 = unused)");

    // Manual tensor / KV overrides.
    cmd->add_option("--override-tensor", o.override_tensor,
        "Override a tensor's buffer type: '<pattern>=<buft_name>' (repeatable; "
        "each entry may itself be comma-separated). Use `chimera info --list-devices` "
        "to discover the buffer type names sd ggml knows.");
    cmd->add_option("--override-kv", o.override_kv,
        "Override model KV metadata: 'KEY=TYPE:VALUE' with TYPE in int/float/bool/str "
        "(repeatable; comma-separated)");

    // Activation steering.
    cmd->add_option("--control-vector", o.control_vector,
        "Path to a control-vector file (scale=1.0; repeatable; comma-separated)");
    cmd->add_option("--control-vector-scaled", o.control_vector_scaled,
        "Path:scale pair for a control vector (e.g. 'cvec.gguf:0.8'; repeatable; comma-separated)");
    cmd->add_option("--control-vector-layer-start", o.control_vector_layer_start,
        "First layer index to apply control vectors at (default: 1)");
    cmd->add_option("--control-vector-layer-end", o.control_vector_layer_end,
        "Last layer index to apply control vectors at (default: n_layer)");
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
        "Image to feed alongside the prompt (repeatable; requires --mmproj). "
        "Auto-detects image/audio/video; video uses default decode params");
    cmd->add_option("--video", p.prompt_opts.videos,
        "Video to feed alongside the prompt (repeatable; requires --mmproj and "
        "ffmpeg). Honors --video-fps / --video-timestamp-ms / --ffmpeg-dir");
    cmd->add_option("--video-fps", p.prompt_opts.video_fps,
        "Frames per second to sample from --video (<=0 = native fps) [default 4]");
    cmd->add_option("--video-timestamp-ms", p.prompt_opts.video_timestamp_ms,
        "Interval in ms for timestamp text chunks in --video (0 = disabled) [default 5000]");
    cmd->add_option("--ffmpeg-dir", p.prompt_opts.ffmpeg_dir,
        "Directory containing ffmpeg/ffprobe binaries (default: search PATH)");
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
        "Multimodal projector (mmproj GGUF) enabling /image, /audio and /video");
    cmd->add_option("--video-fps", p.chat_opts.video_fps,
        "Frames per second to sample from /video (<=0 = native fps) [default 4]");
    cmd->add_option("--video-timestamp-ms", p.chat_opts.video_timestamp_ms,
        "Interval in ms for timestamp text chunks in /video (0 = disabled) [default 5000]");
    cmd->add_option("--ffmpeg-dir", p.chat_opts.ffmpeg_dir,
        "Directory containing ffmpeg/ffprobe binaries (default: search PATH)");
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
    cmd->add_flag("--reasoning-control", p.chat_opts.reasoning_control,
        "Let Ctrl-C end the model's thinking mid-reply (it then answers) instead of "
        "aborting; second Ctrl-C, or Ctrl-C while answering, aborts. No-op for "
        "non-reasoning templates.");

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
        "Pooling: mean | cls | last | none | rank "
        "(use 'rank' for cross-encoder reranker checkpoints)");
    cmd->add_option("--embd-output-format", p.embed_opts.embd_output_format,
        "Output format: '' (default; space-separated floats), 'array' (JSON array), "
        "'json' (OpenAI-shaped envelope), 'raw' (one float per line)");
    cmd->add_option("--embd-separator", p.embed_opts.embd_separator,
        "If set, split the input on this literal string and emit one vector per piece");
    cmd->add_option("--attention", p.embed_opts.attention,
        "Attention type override for the embedding pass: causal | non-causal "
        "(empty leaves the model's training-time default in place)")
        ->check(CLI::IsMember({"", "causal", "non-causal"}));
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
    cmd->add_option("--load-mode",p.embed_opts.load_mode,
        "Model loading mode, overriding --no-mmap/--mlock: "
        "auto | none | mmap | mlock | mmap+mlock | dio (direct I/O where supported)")
        ->check(CLI::IsMember({"auto","none","mmap","mlock","mmap+mlock","dio"}));
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
    cmd->add_option("--prompt", p.whisper_opts.initial_prompt,
        "Initial prompt for vocabulary/style biasing (params.initial_prompt)");
    cmd->add_flag("--carry-initial-prompt", p.whisper_opts.carry_initial_prompt,
        "Prepend --prompt to every decode window (heavier conditioning)");
    cmd->add_option("--beam-size", p.whisper_opts.beam_size,
        "Beam width; >0 switches to beam-search sampling");
    cmd->add_option("--best-of", p.whisper_opts.best_of,
        "Number of candidates considered by greedy sampling");
    cmd->add_option("--temperature", p.whisper_opts.temperature,
        "Initial decoding temperature");
    cmd->add_flag("--no-fallback", p.whisper_opts.no_fallback,
        "Disable temperature-fallback ladder (sets temperature_inc<0)");
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
    // Region-of-audio (whisper-cli -ot/-d). -on (sample-offset) is not
    // exposed by whisper.cpp's whisper_full_params (it's internal to the
    // CLI's WAV reader), so only the ms-based forms are wired here.
    cmd->add_option("--offset",   p.whisper_opts.offset_ms,
        "Start offset into the audio in milliseconds (0 = from the beginning)");
    cmd->add_option("--duration", p.whisper_opts.duration_ms,
        "Audio duration to process in milliseconds (0 = until the end)");
    // Voice Activity Detection bundle.
    cmd->add_flag("--vad", p.whisper_opts.vad,
        "Enable VAD preprocessing (requires --vad-model)");
    cmd->add_option("--vad-model", p.whisper_opts.vad_model,
        "Path to a whisper VAD model file (e.g. silero v5)");
    cmd->add_option("--vad-threshold", p.whisper_opts.vad_threshold,
        "Probability threshold to consider a frame as speech");
    cmd->add_option("--vad-min-speech-duration-ms",
        p.whisper_opts.vad_min_speech_duration_ms,
        "Minimum duration for a valid speech segment (ms)");
    cmd->add_option("--vad-min-silence-duration-ms",
        p.whisper_opts.vad_min_silence_duration_ms,
        "Minimum silence duration to consider speech as ended (ms)");
    cmd->add_option("--vad-max-speech-duration-s",
        p.whisper_opts.vad_max_speech_duration_s,
        "Maximum speech-segment duration before forcing a new segment (s)");
    cmd->add_option("--vad-speech-pad-ms",
        p.whisper_opts.vad_speech_pad_ms,
        "Padding added before and after speech segments (ms)");
    cmd->add_option("--vad-samples-overlap",
        p.whisper_opts.vad_samples_overlap,
        "Overlap in seconds when copying audio samples from speech segments");
    // Segment shaping. --max-len pairs naturally with --output-srt/vtt.
    cmd->add_option("--max-len", p.whisper_opts.max_len,
        "Maximum segment length in characters (0 = unlimited)");
    cmd->add_option("--max-tokens", p.whisper_opts.max_tokens,
        "Maximum tokens per segment (0 = unlimited)");
    cmd->add_flag("--split-on-word", p.whisper_opts.split_on_word,
        "When --max-len>0, split segments on word boundaries instead of mid-token");
    // Decoder fallback thresholds.
    cmd->add_option("--temperature-inc", p.whisper_opts.temperature_inc,
        "Temperature increment for the fallback ladder (NaN = whisper default; "
        "--no-fallback still wins and forces it negative)");
    cmd->add_option("--entropy-thold", p.whisper_opts.entropy_thold,
        "Entropy threshold for decoder-fail fallback (compression-ratio analogue)");
    cmd->add_option("--logprob-thold", p.whisper_opts.logprob_thold,
        "Average log-probability threshold for decoder-fail fallback");
    cmd->add_option("--no-speech-thold", p.whisper_opts.no_speech_thold,
        "Probability threshold above which a segment is treated as no-speech");
    // Audio context / diarization / suppression.
    cmd->add_option("--audio-ctx", p.whisper_opts.audio_ctx,
        "Override the audio context size (0 = model default; halve for tiny.en perf)");
    cmd->add_flag("--tinydiarize", p.whisper_opts.tinydiarize,
        "Enable tinydiarize speaker-turn detection (requires a tdrz-trained model)");
    cmd->add_option("--suppress-regex", p.whisper_opts.suppress_regex,
        "Regex applied to token strings to suppress matching tokens");
    cmd->add_flag("--suppress-nst", p.whisper_opts.suppress_nst,
        "Suppress non-speech tokens during decoding");
    // Context-params (applied at whisper_init_from_file_with_params).
    cmd->add_flag("--flash-attn", p.whisper_opts.flash_attn,
        "Enable flash-attention in the whisper context (perf, GPU builds)");
    cmd->add_flag("--no-gpu", p.whisper_opts.no_gpu,
        "Force CPU-only inference (whisper defaults to GPU when built with one)");
    cmd->add_option("--device", p.whisper_opts.gpu_device,
        "CUDA device index (whisper_context_params.gpu_device); ignored under --no-gpu");
    cmd->add_option("--processors", p.whisper_opts.processors,
        "Split decode across N processors via whisper_full_parallel (1 = serial; "
        ">1 may degrade accuracy near chunk boundaries)");
    // Grammar (constrained decoding). --grammar and --grammar-file are
    // mutually exclusive; checked in command_whisper before model load.
    cmd->add_option("--grammar", p.whisper_opts.grammar,
        "GBNF grammar literal that constrains decoding (mutually exclusive with --grammar-file)");
    cmd->add_option("--grammar-file", p.whisper_opts.grammar_file,
        "Read the GBNF grammar from FILE (mutually exclusive with --grammar)");
    cmd->add_option("--grammar-rule", p.whisper_opts.grammar_rule,
        "Top-level GBNF rule name (default: 'root')");
    cmd->add_option("--grammar-penalty", p.whisper_opts.grammar_penalty,
        "Logit penalty for tokens that don't extend the grammar (default: 100.0)");
    cmd->add_flag("--diarize", p.whisper_opts.diarize,
        "Stereo speaker diarization. Requires a 2-channel WAV; each segment "
        "is prefixed with '(speaker 0)' / '(speaker 1)' / '(speaker ?)' based "
        "on the per-channel energy ratio over the segment's time range.");
    cmd->add_flag("--detect-language", p.whisper_opts.detect_language,
        "Detect the spoken language and exit (prints e.g. 'en'). No "
        "transcription is produced; format-file flags are ignored.");
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
    cmd->add_option("--clip-g", p.sd_opts.clip_g,
                    "CLIP-G text encoder (SDXL split-checkpoint layouts)");
    cmd->add_option("--t5xxl", p.sd_opts.t5xxl, "T5-XXL text encoder");
    cmd->add_option("--llm", p.sd_opts.llm,
                    "LLM text encoder (e.g. Qwen3 for Z-Image)");
    cmd->add_option("--high-noise-diffusion-model", p.sd_opts.high_noise_diffusion_model,
                    "Optional second diffusion model for two-stage 'high noise' workflows. "
                    "The full --high-noise-* sampler family is video-only (sd vid_gen) and "
                    "is intentionally not exposed here; chimera-sd is img_gen-only.");
    cmd->add_option("--control-net", p.sd_opts.control_net,
                    "ControlNet model file (pair with --control-image)");
    cmd->add_option("--type", p.sd_opts.wtype,
                    "Weights type override (f16, f32, bf16, q8_0, q5_1, "
                    "q5_0, q4_1, q4_0, q4_k, q3_k, q2_k, iq4_nl, ...)");
    cmd->add_flag("--offload-to-cpu", p.sd_opts.offload_to_cpu,
                  "Offload model params to CPU (saves VRAM)");
    cmd->add_flag("--diffusion-fa", p.sd_opts.diffusion_fa,
                  "Enable flash-attention in the diffusion model");
    cmd->add_flag("--diffusion-conv-direct", p.sd_opts.diffusion_conv_direct,
                  "Use conv-direct kernels in the diffusion model (faster on modern dGPUs)");
    cmd->add_flag("--vae-conv-direct", p.sd_opts.vae_conv_direct,
                  "Use conv-direct kernels in the VAE (pairs with --diffusion-conv-direct)");
    cmd->add_option("--rng", p.sd_opts.rng,
                  "RNG kind: std_default | cuda | cpu (leave unset for upstream default)");
    cmd->add_option("--sampler-rng", p.sd_opts.sampler_rng,
                  "Per-sampling RNG: std_default | cuda | cpu "
                  "(use cpu to match ComfyUI seeds across implementations)");
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
    cmd->add_option("--guidance", p.sd_opts.guidance,
        "Distilled guidance scale (Flux / SD3); leave unset for upstream default");
    cmd->add_option("--flow-shift", p.sd_opts.flow_shift,
        "Flux/SD3 timestep shift; leave unset for upstream default");
    cmd->add_option("--control-image", p.sd_opts.control_image,
        "ControlNet conditioning image (requires --control-net)");
    cmd->add_option("--control-strength", p.sd_opts.control_strength,
        "ControlNet conditioning strength (default 0.9, only used with --control-image)");
    cmd->add_flag("--vae-tiling", p.sd_opts.vae_tiling,
        "Tile the VAE decode to lower peak VRAM for large outputs");
    cmd->add_option("--vae-tile-size", p.sd_opts.vae_tile_size,
        "Absolute VAE tile size in pixels (applied to both axes)");
    cmd->add_option("--vae-relative-tile-size", p.sd_opts.vae_relative_tile_size,
        "VAE tile size as a fraction of the canvas (e.g. 0.5)");
    cmd->add_option("--vae-tile-overlap", p.sd_opts.vae_tile_overlap,
        "Fractional overlap between VAE tiles (e.g. 0.25)");
    cmd->add_option("--lora", p.sd_opts.lora_adapters,
        "LoRA adapter as path[:scale] (scale defaults to 1.0; repeatable). "
        "Relative paths are joined against --lora-model-dir when set.");
    cmd->add_option("--lora-model-dir", p.sd_opts.lora_model_dir,
        "Base directory used to resolve relative --lora paths");
    // Skip-layer guidance (SLG). All four flags are required together to
    // have effect; --skip-layers alone enables SLG with the upstream
    // default scale/start/end. Empty --skip-layers disables SLG.
    cmd->add_option("--skip-layers", p.sd_opts.skip_layers,
        "Comma-separated list of layer indices for skip-layer guidance (e.g. '7,8,9'). "
        "Empty disables SLG regardless of the other --skip-layer-* flags.");
    cmd->add_option("--slg-scale", p.sd_opts.slg_scale,
        "Skip-layer guidance scale (leave unset for upstream default)");
    cmd->add_option("--skip-layer-start", p.sd_opts.skip_layer_start,
        "Fractional step (0..1) at which to start applying SLG");
    cmd->add_option("--skip-layer-end", p.sd_opts.skip_layer_end,
        "Fractional step (0..1) at which to stop applying SLG");
    // Round 1 perf / offload (sd_ctx_params_t one-liners).
    cmd->add_flag("--fa", p.sd_opts.flash_attn_global,
        "Enable global flash-attention across all sd graphs (distinct from --diffusion-fa which only flips the diffusion model)");
    cmd->add_flag("--no-mmap", p.sd_opts.no_mmap,
        "Disable mmap for model loading (chimera defaults to mmap=on; this flips it off)");
    cmd->add_option("--max-vram", p.sd_opts.max_vram,
        "Soft cap on VRAM use in GiB (0 = leave the upstream default; sd.cpp may swap to CPU above the cap)");
    cmd->add_flag("--stream-layers", p.sd_opts.stream_layers,
        "Stream diffusion weights from CPU during generation (only engages with --max-vram > 0; ignored otherwise)");
    cmd->add_flag("--eager-load", p.sd_opts.eager_load,
        "Pre-load all weights into the params backend at model-load time instead of lazily on first use (slower load, no first-generation warmup)");
    cmd->add_flag("--clip-on-cpu", p.sd_opts.keep_clip_on_cpu,
        "Keep the CLIP / text-encoder pass on CPU even when a GPU backend is available");
    cmd->add_flag("--vae-on-cpu", p.sd_opts.keep_vae_on_cpu,
        "Keep the VAE encode/decode on CPU (more surgical than --offload-to-cpu)");
    cmd->add_flag("--control-net-cpu", p.sd_opts.keep_control_net_on_cpu,
        "Keep the ControlNet on CPU even when a GPU backend is available");
    cmd->add_option("--backend", p.sd_opts.backend,
        "Per-module compute placement: a bare target (\"cuda0\", \"cpu\") or "
        "comma-separated assignments (\"diffusion=cuda0,te=cpu\"). Appended after "
        "--clip-on-cpu/--vae-on-cpu/--control-net-cpu, which sd resolves last-wins");
    cmd->add_option("--params-backend", p.sd_opts.params_backend,
        "Per-module weight residency, same grammar as --backend and additionally "
        "accepting cpu / disk. Appended after --offload-to-cpu. On a single GPU that "
        "cannot hold every module, \"te=cpu\" is usually the fastest placement");
    cmd->add_flag("--auto-fit", p.sd_opts.auto_fit,
        "Let sd derive --backend and --params-backend from the model sizes and the "
        "available VRAM (overriding both). Fits where an explicit placement would "
        "not, but on a single-GPU box tends to resolve to all-CPU and is far slower");
    cmd->add_flag("--force-sdxl-vae-conv-scale", p.sd_opts.force_sdxl_vae_conv_scale,
        "Apply the SDXL VAE conv-scale numerics fix (workaround for some SDXL VAE checkpoints)");
    // Round 2 sampler / generation core.
    cmd->add_option("--img-cfg-scale", p.sd_opts.img_cfg_scale,
        "Separate img-cond CFG scale (Flux/SD3); -1 leaves the default INFINITY so sd falls back to --cfg-scale");
    cmd->add_option("--eta", p.sd_opts.eta,
        "DDIM/ancestral-style stochasticity in [0,1]; -1 leaves the upstream default");
    cmd->add_option("--timestep-shift", p.sd_opts.shifted_timestep,
        "Shift the sampling timestep schedule by N (0 = no shift)");
    cmd->add_option("--sigmas", p.sd_opts.sigmas,
        "Custom sigma schedule as a comma-separated float list (e.g. '14.6,10.0,5.0,1.0'); "
        "empty disables the override. Non-float entries exit with BadInput.");
    cmd->add_option("--prediction", p.sd_opts.prediction,
        "Prediction-type override: eps | v | edm_v | flow | flux_flow | flux2_flow (empty = model default)")
        ->check(CLI::IsMember({"eps","v","edm_v","flow","flux_flow","flux2_flow"}));
    cmd->add_option("--lora-apply-mode", p.sd_opts.lora_apply_mode,
        "LoRA application mode: auto | immediately | at_runtime (empty = upstream default)")
        ->check(CLI::IsMember({"auto","immediately","at_runtime"}));
    cmd->add_option("--vae-format", p.sd_opts.vae_format,
        "VAE latent format override: auto | flux | sd3 | flux2 (empty = auto)")
        ->check(CLI::IsMember({"auto","flux","sd3","flux2"}));
    // Round 3 model-loading completers.
    cmd->add_option("--taesd", p.sd_opts.taesd,
        "Tiny AutoEncoder (TAESD) model for fast preview decode");
    cmd->add_option("--clip-vision", p.sd_opts.clip_vision,
        "CLIP-Vision encoder (image-conditioning models)");
    cmd->add_option("--llm-vision", p.sd_opts.llm_vision,
        "LLM-Vision encoder (e.g. Qwen-Image vision-conditioning)");
    cmd->add_option("--tensor-type-rules", p.sd_opts.tensor_type_rules,
        "Per-tensor wtype override rules (sd.cpp's --tensor-type-rules syntax)");
    cmd->add_option("--photo-maker", p.sd_opts.photo_maker,
        "PhotoMaker model file (pair with --pm-id-images-dir at generate time)");
    cmd->add_option("--embd-dir", p.sd_opts.embd_dir,
        "Directory of textual-inversion embeddings to register at context init. "
        "Non-recursive; only .gguf/.safetensors/.pt files are accepted; "
        "the filename stem becomes the prompt token.");
    // Round 4 PhotoMaker generation bundle.
    cmd->add_option("--pm-id-images-dir", p.sd_opts.pm_id_images_dir,
        "Directory of reference identity images for PhotoMaker (non-recursive, alphabetical order). "
        "Empty entries / unreadable files are skipped; an empty result is an error.");
    cmd->add_option("--pm-id-embed-path", p.sd_opts.pm_id_embed_path,
        "Path to a precomputed PhotoMaker ID embedding (.bin)");
    cmd->add_option("--pm-style-strength", p.sd_opts.pm_style_strength,
        "PhotoMaker style strength (-1 = upstream default)");
    // Round 5 reference images.
    cmd->add_option("--ref-image", p.sd_opts.ref_images,
        "Reference image for style/identity conditioning (repeatable; each becomes one entry "
        "in sd_img_gen_params_t.ref_images)");
    cmd->add_flag("--increase-ref-index", p.sd_opts.increase_ref_index,
        "Increment the ref-image index across batch positions");
    cmd->add_flag("--no-auto-resize-ref-image", p.sd_opts.no_auto_resize_ref_image,
        "Disable sd's automatic resize of --ref-image inputs to match --width/--height");
    cmd->add_option("--ref-image-args", p.sd_opts.ref_image_args,
        "Reference-image processing args as a comma-separated key=value list, layered "
        "on the preset sd picks from the model architecture. Keys: preset, pass_to_vlm, "
        "pass_to_dit, ref_index_mode (fixed|increase|decrease), force_ref_timestep_zero, "
        "resize_before_vae, vae_input_max_pixels, vlm_resize_mode (longest_side|area|none), "
        "vlm_min_size, vlm_max_size, vlm_size. --increase-ref-index and "
        "--no-auto-resize-ref-image are applied after this string and override it");
    // Round 6 hires-fix.
    cmd->add_flag("--hires", p.sd_opts.hires_fix,
        "Enable the hires-fix second-pass upscale (sd_hires_params_t.enabled)");
    cmd->add_option("--hires-upscaler", p.sd_opts.hires_upscaler,
        "Hires upscaler name (sd's enum). Values: None, Latent, "
        "'Latent (nearest)', 'Latent (nearest-exact)', 'Latent (antialiased)', "
        "'Latent (bicubic)', 'Latent (bicubic antialiased)', Lanczos, Nearest, Model. "
        "Empty leaves sd's default (Latent).");
    cmd->add_option("--upscale-model", p.sd_opts.hires_upscale_model,
        "Upscaler model file (pair with --hires-upscaler Model)");
    cmd->add_option("--hires-width", p.sd_opts.hires_width,
        "Target width for the hires pass (0 = derive from --hires-scale)");
    cmd->add_option("--hires-height", p.sd_opts.hires_height,
        "Target height for the hires pass (0 = derive from --hires-scale)");
    cmd->add_option("--hires-scale", p.sd_opts.hires_scale,
        "Scale factor for the hires pass; -1 leaves sd's default (2.0). "
        "Ignored when --hires-width / --hires-height are set.");
    cmd->add_option("--hires-steps", p.sd_opts.hires_steps,
        "Sample steps for the hires pass (0 = use the main --steps)");
    cmd->add_option("--hires-denoising-strength", p.sd_opts.hires_denoising_strength,
        "Denoising strength for the hires pass in [0,1]; -1 leaves sd's default (0.7)");
    cmd->add_option("--hires-upscale-tile-size", p.sd_opts.hires_upscale_tile_size,
        "Tile size used when running the upscaler (0 leaves sd's default of 128)");
    // Cache / SCM bundle (sd_cache_params_t).
    cmd->add_option("--cache-mode", p.sd_opts.cache_mode,
        "Inference cache algorithm: disabled | easycache | ucache | dbcache | "
        "taylorseer | cache-dit | spectrum (empty = sd's default disabled)")
        ->check(CLI::IsMember({"disabled","easycache","ucache","dbcache","taylorseer","cache-dit","spectrum"}));
    cmd->add_option("--cache-option", p.sd_opts.cache_option,
        "Comma-separated 'key=value' overrides for the active --cache-mode. Keys: "
        "threshold/start/end/decay/relative/reset (easycache/ucache), "
        "threshold/Fn/Bn/warmup (dbcache/taylorseer/cache-dit), "
        "w/m/lam/window/flex/warmup/stop (spectrum). Unknown keys / non-numeric "
        "values exit with BadInput before model load.");
    cmd->add_option("--scm-mask", p.sd_opts.scm_mask,
        "Path to a sampler-cached-memory mask file (borrowed by sd_cache_params_t.scm_mask)");
    cmd->add_option("--scm-policy", p.sd_opts.scm_policy,
        "SCM policy: 'static' or 'dynamic' (empty = sd's default 'dynamic')")
        ->check(CLI::IsMember({"static","dynamic"}));
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
    cmd->add_option("--http-timeout", p.serve_opts.http_timeout,
        "HTTP read+write timeout in seconds (0 = upstream default, 3600)");
    cmd->add_option("--sse-ping-interval", p.serve_opts.sse_ping_interval,
        "SSE keep-alive ping interval in seconds (0 = upstream default, 30)");
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
    cmd->add_flag("--audio-flash-attn", p.serve_opts.audio_flash_attn,
        "Enable flash-attention in the whisper context (GPU builds)");
    cmd->add_flag("--audio-no-gpu", p.serve_opts.audio_no_gpu,
        "Force whisper to run on CPU even when GPU support is built in");
    cmd->add_option("--audio-device", p.serve_opts.audio_gpu_device,
        "CUDA device index for the whisper context (ignored under --audio-no-gpu)");
    cmd->add_option("--audio-vad-model", p.serve_opts.audio_vad_model,
        "Path to a whisper VAD model (e.g. silero v5). Required to enable "
        "per-request vad=true on /v1/audio/transcriptions — without it, "
        "requests asking for VAD return HTTP 400.");
#endif
#ifdef CHIMERA_HAS_SD
    cmd->add_option("--enable-image", p.serve_opts.sd_model,
        "Stable-diffusion GGUF to load alongside the LLM (enables /v1/images/*)");
    cmd->add_option("--sd-control-net", p.serve_opts.sd_control_net,
        "Path to a ControlNet model. Required to enable per-request "
        "control_image (multipart file) + control_strength on /v1/images/* — "
        "without it, requests with control_image return HTTP 400.");
    // Split-checkpoint layouts (Flux / SD3 / Z-Image / Qwen-Image).
    // Mirror the `chimera sd` CLI flag names with a --sd- prefix.
    // --enable-image becomes optional when --sd-diffusion-model is set.
    cmd->add_option("--sd-diffusion-model", p.serve_opts.sd_diffusion_model,
        "Separate UNet/DiT file (split-checkpoint layouts: Flux, Z-Image, SD3, ...)");
    cmd->add_option("--sd-vae", p.serve_opts.sd_vae,
        "Separate VAE for split-checkpoint layouts");
    cmd->add_option("--sd-clip-l", p.serve_opts.sd_clip_l,
        "CLIP-L text encoder (SDXL / SD3 split layouts)");
    cmd->add_option("--sd-clip-g", p.serve_opts.sd_clip_g,
        "CLIP-G text encoder (SDXL split layouts)");
    cmd->add_option("--sd-t5xxl", p.serve_opts.sd_t5xxl,
        "T5-XXL text encoder (Flux / SD3)");
    cmd->add_option("--sd-llm", p.serve_opts.sd_llm,
        "LLM text encoder (e.g. Qwen3 for Z-Image)");
    cmd->add_option("--sd-llm-vision", p.serve_opts.sd_llm_vision,
        "LLM-Vision encoder (e.g. Qwen-Image vision-conditioning)");
    cmd->add_option("--sd-clip-vision", p.serve_opts.sd_clip_vision,
        "CLIP-Vision encoder (image-conditioning models)");
    cmd->add_option("--sd-taesd", p.serve_opts.sd_taesd,
        "Tiny AutoEncoder (TAESD) model for fast preview decode");
    cmd->add_option("--sd-embd-dir", p.serve_opts.sd_embd_dir,
        "Directory of textual-inversion embeddings to register at context init "
        "(non-recursive; .gguf/.safetensors/.pt)");
    cmd->add_option("--sd-type", p.serve_opts.sd_type,
        "Weights type override (f16/f32/bf16/q8_0/q5_1/q5_0/q4_1/q4_0/q4_k/q3_k/q2_k/iq4_nl/...)");
    cmd->add_option("--sd-tensor-type-rules", p.serve_opts.sd_tensor_type_rules,
        "Per-tensor wtype override rules (sd.cpp's --tensor-type-rules syntax)");
    cmd->add_option("--sd-high-noise-diffusion-model", p.serve_opts.sd_high_noise_diffusion_model,
        "Optional second diffusion model for two-stage 'high noise' workflows");
    // 5d perf / offload long-tail.
    cmd->add_flag("--sd-fa", p.serve_opts.sd_flash_attn,
        "Enable global flash-attention across all sd graphs "
        "(distinct from --sd-diffusion-fa which only flips the diffusion model)");
    cmd->add_flag("--sd-diffusion-fa", p.serve_opts.sd_diffusion_flash_attn,
        "Enable flash-attention in the diffusion model only");
    cmd->add_flag("--sd-diffusion-conv-direct", p.serve_opts.sd_diffusion_conv_direct,
        "Conv-direct kernels in the diffusion model");
    cmd->add_flag("--sd-vae-conv-direct", p.serve_opts.sd_vae_conv_direct,
        "Conv-direct kernels in the VAE");
    cmd->add_flag("--sd-no-mmap", p.serve_opts.sd_no_mmap,
        "Disable mmap for sd model loading (chimera defaults to mmap=on)");
    cmd->add_option("--sd-max-vram", p.serve_opts.sd_max_vram,
        "Soft cap on VRAM use in GiB (0 = leave the upstream default; sd.cpp may swap to CPU above the cap)");
    cmd->add_flag("--sd-stream-layers", p.serve_opts.sd_stream_layers,
        "Stream diffusion weights from CPU during generation (only engages with --sd-max-vram > 0)");
    cmd->add_flag("--sd-eager-load", p.serve_opts.sd_eager_load,
        "Pre-load all sd weights into the params backend at model-load time instead of lazily on first use (slower load, no first-request warmup)");
    cmd->add_flag("--sd-offload-to-cpu", p.serve_opts.sd_offload_to_cpu,
        "Offload sd model parameters to CPU memory");
    cmd->add_flag("--sd-clip-on-cpu", p.serve_opts.sd_keep_clip_on_cpu,
        "Keep the CLIP / text-encoder pass on CPU even when a GPU backend is available");
    cmd->add_flag("--sd-vae-on-cpu", p.serve_opts.sd_keep_vae_on_cpu,
        "Keep the VAE encode/decode on CPU");
    cmd->add_flag("--sd-control-net-cpu", p.serve_opts.sd_keep_control_net_on_cpu,
        "Keep the ControlNet on CPU even when a GPU backend is available");
    cmd->add_option("--sd-backend", p.serve_opts.sd_backend,
        "Per-module compute placement: a bare target (\"cuda0\", \"cpu\") or "
        "comma-separated assignments (\"diffusion=cuda0,te=cpu\"); resolved last-wins "
        "after the --sd-*-on-cpu flags");
    cmd->add_option("--sd-params-backend", p.serve_opts.sd_params_backend,
        "Per-module weight residency, same grammar as --sd-backend and additionally "
        "accepting cpu / disk; resolved last-wins after --sd-offload-to-cpu");
    cmd->add_flag("--sd-auto-fit", p.serve_opts.sd_auto_fit,
        "Let sd derive --sd-backend and --sd-params-backend from the model sizes and "
        "the available VRAM (overriding both)");
    cmd->add_flag("--sd-force-sdxl-vae-conv-scale", p.serve_opts.sd_force_sdxl_vae_conv_scale,
        "Apply the SDXL VAE conv-scale numerics fix");
    cmd->add_option("--sd-rng", p.serve_opts.sd_rng,
        "RNG type: std_default | cuda | cpu (empty = upstream default)")
        ->check(CLI::IsMember({"std_default","cuda","cpu"}));
    cmd->add_option("--sd-sampler-rng", p.serve_opts.sd_sampler_rng,
        "Sampler RNG type: std_default | cuda | cpu (empty = upstream default; "
        "cpu matches ComfyUI seeds across implementations)")
        ->check(CLI::IsMember({"std_default","cuda","cpu"}));
    cmd->add_option("--sd-prediction", p.serve_opts.sd_prediction,
        "Prediction-type override: eps | v | edm_v | flow | flux_flow | flux2_flow "
        "(empty = model default)")
        ->check(CLI::IsMember({"eps","v","edm_v","flow","flux_flow","flux2_flow"}));
    cmd->add_option("--sd-lora-apply-mode", p.serve_opts.sd_lora_apply_mode,
        "LoRA application mode: auto | immediately | at_runtime (empty = upstream default)")
        ->check(CLI::IsMember({"auto","immediately","at_runtime"}));
    cmd->add_option("--sd-vae-format", p.serve_opts.sd_vae_format,
        "VAE latent format override: auto | flux | sd3 | flux2 (empty = auto)")
        ->check(CLI::IsMember({"auto","flux","sd3","flux2"}));
    cmd->add_option("--sd-threads", p.serve_opts.sd_threads,
        "Threads for the sd context (-1 = leave sd.cpp default)");
    // 5e PhotoMaker on serve.
    cmd->add_option("--sd-photo-maker", p.serve_opts.sd_photo_maker,
        "Path to a PhotoMaker model. Required to enable per-request PhotoMaker "
        "(pm_id_images / pm_id_image_set / pm_style_strength) on /v1/images/*.");
    cmd->add_option("--sd-pm-id-dir", p.serve_opts.sd_pm_id_dir,
        "Root directory of named PhotoMaker identity sets. Each subdirectory "
        "becomes one set, addressable per-request as pm_id_image_set: \"<name>\". "
        "Scanned eagerly at startup; non-image files inside subdirs are skipped.");
    cmd->add_option("--sd-pm-id-embed-path", p.serve_opts.sd_pm_id_embed_path,
        "Path to a precomputed PhotoMaker ID embedding (.bin). Applied to every "
        "PM request unless the request later supplies its own override.");
    // Step 6: per-request LoRA selection via named aliases. Each
    // --sd-lora flag registers one entry in the allowlist; requests
    // reference adapters by name. Repeatable.
    cmd->add_option("--sd-lora", p.serve_opts.sd_loras,
        "Register a named LoRA adapter, format <name>=<path>. Repeatable. "
        "Requests select adapters via JSON `loras: [{\"name\":\"<name>\","
        "\"scale\":<float>}, ...]` on /v1/images/* endpoints. Names only — "
        "request bodies cannot reference arbitrary filesystem paths.");
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
    p.info_cmd->add_flag("--list-devices", p.info_list_devices,
        "Print only the available ggml device names, one per line "
        "(suitable for piping into `--device` on gen/chat/embed)");
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
    if (*p.info_cmd)         return command_info        (p.info_list_devices);
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
    // Let the global flags below bind after the subcommand as well, so both
    // `chimera -v sd ...` and `chimera sd -v ...` work; every subcommand
    // defines its own options first, so fallthrough only catches leftovers.
    app.fallthrough();
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

