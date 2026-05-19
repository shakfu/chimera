#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Version strings. Overridden at compile time via -DCHIMERA_*_VERSION="..."
// from CMake; fall back to "unknown" when built outside the project's CMake.
#ifndef CHIMERA_VERSION
#define CHIMERA_VERSION "unknown"
#endif
#ifndef CHIMERA_LLAMACPP_VERSION
#define CHIMERA_LLAMACPP_VERSION "unknown"
#endif
#ifndef CHIMERA_WHISPERCPP_VERSION
#define CHIMERA_WHISPERCPP_VERSION "unknown"
#endif
#ifndef CHIMERA_SDCPP_VERSION
#define CHIMERA_SDCPP_VERSION "unknown"
#endif

struct LlamaCommonOptions {
    std::string model;
    std::string mmproj;             // empty = text-only; otherwise mtmd vision projector
    std::vector<std::string> images;  // images to feed alongside the prompt (gen only)
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 512;
    int threads = -1;
    int gpu_layers = 0;
    int n_predict = 256;
    uint32_t seed = 0xFFFFFFFFu;
    float temp = 0.8f;
    int top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.05f;
    float repeat_penalty = 1.1f;
    bool use_mmap = true;
};

struct EmbedOptions {
    std::string model;
    std::string input;
    std::string input_file;
    std::string output;            // empty = stdout
    std::string pooling = "mean";  // mean | cls | last | none
    int threads = -1;
    int gpu_layers = 0;
    uint32_t n_ctx = 0;            // 0 = use model's default training context
    uint32_t n_batch = 512;
    bool normalize = true;
    bool use_mmap = true;

    // Opt-in persistent embedding cache. When true, embed(text) -> vector
    // is memoized in `cache_db` (or the platform default if empty) so
    // repeats skip the model. See doc/dev/sqlite.md and
    // chimera_embed_cache.h.
    bool        cache_embeddings = false;
    std::string cache_db;
};

struct TokenizeOptions {
    std::string model;
    std::string input;
    std::string input_file;
    bool add_special = true;
    bool parse_special = true;
    bool show_pieces = false;
    bool use_mmap = true;
};

#ifdef CHIMERA_HAS_WHISPER
struct WhisperOptions {
    std::string model;
    std::string input;
    std::string output;
    std::string language;
    int threads = -1;
    bool translate = false;
    bool timestamps = false;
    bool no_context = false;
};
int command_whisper(const WhisperOptions & opts);
#endif

#ifdef CHIMERA_HAS_SD
struct SdOptions {
    std::string model;            // combined checkpoint (legacy single-file)
    std::string diffusion_model;  // separate UNet/DiT (e.g. Z-Image, Flux)
    std::string vae;              // separate VAE
    std::string clip_l;           // CLIP-L text encoder
    std::string t5xxl;            // T5-XXL text encoder
    std::string llm;              // LLM text encoder (e.g. Qwen3 for Z-Image)
    std::string prompt;
    std::string negative_prompt;
    std::string output = "output.png";
    std::string sample_method;
    std::string scheduler;
    std::string init_image;  // empty = text-to-image
    std::string mask_image;  // empty = no inpainting mask
    int width = 512;
    int height = 512;
    int steps = 20;
    int batch_count = 1;
    int clip_skip = -1;
    int threads = -1;
    int64_t seed = -1;
    float cfg_scale = 7.0f;
    float strength = 0.75f;  // img2img denoising strength (0 = preserve, 1 = full noise)
    bool offload_to_cpu = false;
    bool diffusion_fa = false;   // flash-attention in the diffusion model
};
int command_sd(const SdOptions & opts);
#endif

// OpenAI-compatible HTTP server backed by llama.cpp's server-context engine.
// First-cut scope is text generation only; audio/image endpoints are planned
// follow-ups. Field names mirror common_params so the translation in
// command_serve is one-for-one. Defaults are picked to match llama-server's
// behavior when the corresponding flag is omitted.
struct ServeOptions {
    std::string model;             // required (--model)
    std::string mmproj;             // optional (--mmproj), enables vision/audio inputs
    std::string host = "127.0.0.1"; // --host
    int         port = 8080;        // --port
    int         n_ctx = 0;          // --ctx-size; 0 = the model's training context
    int         n_batch = 2048;     // --batch-size (logical)
    int         n_ubatch = 512;     // --ubatch-size (physical); reduced for embeddings
    int         threads = -1;       // --threads; -1 = let common_params auto-pick
    int         gpu_layers = -1;    // --gpu-layers; -1 = auto
    int         parallel = 1;       // --parallel; number of concurrent slots
    std::string api_key;            // --api-key; empty disables auth
    bool        embedding = false;  // --embeddings; switches model to embed mode

    // Opt-in audio. When non-empty, a whisper.cpp model is loaded
    // alongside the LLM and POST /v1/audio/transcriptions becomes available.
    std::string audio_model;        // --enable-audio <whisper.gguf>

    // Opt-in image. When non-empty, a stable-diffusion.cpp model is
    // loaded alongside the LLM and POST /v1/images/{generations,edits,
    // variations} become available. The context is built with
    // vae_decode_only=false so the encode path is available for img2img
    // (/edits and /variations).
    std::string sd_model;           // --enable-image <sd.gguf>

    // Opt-in vector store / RAG. When non-empty, the named GGUF embedding
    // model is loaded alongside the LLM and the POST/GET /v1/vector_stores/*
    // routes are bound. Ingest and search requests targeting a collection
    // whose recorded embedding_model does not match this one are rejected
    // with a 400 (single-model server in this cut).
    std::string rag_embedding_model;  // --enable-rag <embedding.gguf>
    std::string rag_db_path;          // --rag-db (default: $CHIMERA_DB or platform default)

    // Opt-in dedicated embedding model. When non-empty a second
    // server_context is spun up with embedding mode enabled and
    // /v1/embeddings is routed to it (instead of the main LLM). Use this
    // when the main LLM is a generative model and you also want OpenAI-
    // compatible embeddings without launching a second process.
    std::string embed_model;          // --enable-embeddings <embedding.gguf>

    // Opt-in cross-encoder reranker. When non-empty a third server_context
    // is spun up with embedding=true + pooling=rank and /v1/rerank is
    // bound. Natural follow-up to the RAG vector search: feed the top-N
    // hits + the user query through this to get a refined ordering.
    std::string rerank_model;         // --reranking <model.gguf>

    // Opt-in persistent embedding cache for the RAG path. Reuses the
    // same SQLite DB as --enable-rag (and --rag-db override). No effect
    // unless --enable-rag is also set (the cache attaches to the RAG
    // Embedder; the dedicated --enable-embeddings model goes through
    // server-context and isn't cacheable from here).
    bool cache_embeddings = false;    // --cache-embeddings

    // Chat persistence: when true, every /v1/chat/completions exchange is
    // saved to the chats + messages tables. Multi-turn requests that echo
    // back the X-Chimera-Chat-Id response header get folded into a single
    // chats row; requests without the header produce one chats row each
    // (the OpenAI API has no chat-id concept). Uses the same SQLite DB
    // as the CLI's `chat --persist`.
    bool        persist_chats = false; // --persist-chats
    std::string chat_db_path;          // --chat-db (default: $CHIMERA_DB or platform default)

    // KV-cache slot snapshots. When non-empty, POST /slots/:id?action=...
    // (save / restore / erase) becomes functional and writes/reads
    // snapshot files under this directory. GET /slots (slot status
    // metadata) is always available regardless of this setting — the
    // path only gates the write/restore side. Trailing slash is added
    // by upstream if missing.
    std::string slot_save_path;         // --slot-save-path <dir>

    // LoRA adapters loaded alongside the base model. Each entry is
    // "path[:scale]" (default scale 1.0). All listed adapters are
    // loaded at startup; POST /lora-adapters lets clients change which
    // are *active* (and at what scale) per-request without a reload.
    // GET /lora-adapters lists the loaded set. Repeatable on the CLI.
    std::vector<std::string> lora_adapters; // --lora <path[:scale]>

    // Embedded web chat UI. Only takes effect when the build was
    // configured with CHIMERA_WEBUI_EMBED=1 — the bundle is xxd-baked
    // into libserver-context.a at compile time and the routes (GET /,
    // /bundle.js, /bundle.css) are bound by upstream's
    // server_http_context::init when this flag is true. With a webui-
    // less build, this flag is a no-op (no /, no /bundle.* routes
    // exist either way). Defaults to true so the UI is served whenever
    // it was compiled in; pass --no-webui to suppress at runtime
    // (e.g. on production deployments behind a reverse proxy that
    // serves a different UI).
    bool webui = true;                  // --no-webui (only meaningful in webui-embedded builds)

    // External static-file directory for a web UI. When non-empty,
    // chimera mounts the directory at GET / via cpp-httplib's
    // set_mount_point — independent of CHIMERA_WEBUI_EMBED. Intended
    // for a chimera-specific UI shipped as a separate tarball that
    // surfaces chimera-only routes (/v1/messages, /v1/vector_stores/*,
    // X-Chimera-Chat-Id, etc.) that the upstream llama.cpp webui has
    // no UI for. When both are set, --public-path wins (mount_point is
    // registered before the xxd-baked GET / handler).
    std::string public_path;            // --public-path <dir>
};

int command_serve(const ServeOptions & opts);

#ifdef CHIMERA_HAS_WHISPER
void chimera_silence_whisper_log();
void chimera_restore_whisper_log();
#endif
#ifdef CHIMERA_HAS_SD
void chimera_silence_sd_log();
void chimera_restore_sd_log();
#endif

// Shared utility helpers, inlined here so all three TUs (chimera.cpp,
// chimera_whisper.cpp, chimera_sd.cpp) get one definition without needing
// a separate .cpp -- and without re-introducing the ggml.h collision.

// Exit codes. CLI11 parse errors come from CLI11 itself (codes >= 100 by
// default); use values <= 10 here so they remain visually distinct.
//   1  generic runtime error (fallback)
//   2  bad input (missing/invalid file, malformed argument)
//   3  model-load failure
//   4  generation / inference failure
enum class ExitCode : int {
    Runtime  = 1,
    BadInput = 2,
    Load     = 3,
    Generate = 4,
};

// Runtime exception that carries a structured exit code. Caught in main()
// to map to the process exit status. Use fail(message) for generic errors
// (Runtime) or fail(code, message) for a specific category.
class ChimeraError : public std::runtime_error {
public:
    ChimeraError(ExitCode code, const std::string & msg)
        : std::runtime_error(msg), code_(code) {}
    ExitCode code() const noexcept { return code_; }

private:
    ExitCode code_;
};

[[noreturn]] inline void fail(const std::string & message) {
    throw ChimeraError(ExitCode::Runtime, message);
}

[[noreturn]] inline void fail(ExitCode code, const std::string & message) {
    throw ChimeraError(code, message);
}

inline std::string trim(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}
