#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
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
        fail("failed to load llama model: " + opts.model);
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
        fail("failed to create llama context");
    }
    return ctx;
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
        fail("failed to create sampler");
    }
    return common_sampler_ptr(sampler);
}

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

    if (llama_decode(ctx.get(), llama_batch_get_one(const_cast<llama_token *>(prompt_tokens.data()), static_cast<int32_t>(prompt_tokens.size()))) != 0) {
        fail("failed to decode prompt");
    }

    for (llama_token token : prompt_tokens) {
        common_sampler_accept(sampler.get(), token, false);
    }

    std::string text;
    for (int i = 0; i < opts.n_predict; ++i) {
        const llama_token token = common_sampler_sample(sampler.get(), ctx.get(), -1, false);
        if (token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, token)) {
            break;
        }

        common_sampler_accept(sampler.get(), token, true);

        const std::string piece = token_to_piece(vocab, token);
        text += piece;
        if (stream_output) {
            std::cout << piece << std::flush;
        }

        llama_token token_copy = token;
        if (llama_decode(ctx.get(), llama_batch_get_one(&token_copy, 1)) != 0) {
            fail("failed to decode generated token");
        }
    }

    if (stream_output) {
        std::cout << '\n';
    }

    return text;
}

common_chat_msg make_chat_msg(const std::string & role, const std::string & content) {
    common_chat_msg msg;
    msg.role = role;
    msg.content = content;
    return msg;
}

int command_prompt(const LlamaCommonOptions & opts, const std::string & prompt) {
    if (prompt.empty()) {
        fail("prompt text is required");
    }

    auto model = load_llama_model(opts);
    std::string text = run_generation(model.get(), opts, prompt, true, true);
    return text.empty() ? 1 : 0;
}

int command_chat(const LlamaCommonOptions & opts, const std::string & system_prompt, const std::string & template_override) {
    auto model = load_llama_model(opts);
    common_chat_templates_ptr templates = common_chat_templates_init(model.get(), template_override, "", "");
    if (!templates) {
        fail("failed to initialize chat template");
    }

    std::vector<common_chat_msg> history;
    if (!system_prompt.empty()) {
        history.push_back(make_chat_msg("system", system_prompt));
    }

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
        std::string reply = run_generation(model.get(), opts, params.prompt, false, true);
        history.push_back(make_chat_msg("assistant", reply));
    }

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
        auto * prompt_cmd = app.add_subcommand("gen", "One-shot llama text generation");
        prompt_cmd->add_option("-m,--model", prompt_opts.model, "GGUF model")->required();
        prompt_cmd->add_option("-p,--prompt", prompt_text, "Prompt text")->required();
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
        chat_cmd->add_option("--chat-template", template_override, "Chat template override");

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

        app.parse(argc, argv);

        if (verbose) {
            restore_default_logging();
        }

        llama_backend_init();
        backend_initialized = true;

        int rc = 0;
        if (*prompt_cmd) {
            rc = command_prompt(prompt_opts, prompt_text);
        } else if (*chat_cmd) {
            rc = command_chat(chat_opts, system_prompt, template_override);
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
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << '\n';
        if (backend_initialized) {
            llama_backend_free();
        }
        return 1;
    }
}
