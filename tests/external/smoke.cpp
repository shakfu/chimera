// smoke.cpp - external-consumer smoke test for the three-archive link
// contract (libchimera.a + libchimera_thirdparty.a + libchimera_ggml.a).
//
// What this proves end-to-end:
//   1. All three archives are link-complete (no undefined references).
//   2. The platform whole-archive wrapper on libchimera_ggml.a is
//      doing its job: ggml backends self-register at static init time,
//      so ggml_backend_dev_count() must return >= 1 (CPU) and typically
//      more (Metal on macOS, CUDA on Linux, ...).
//   3. chimera's own symbols resolve: we call a no-op chimera helper
//      to force a reference into libchimera.a.
//   4. (Optional, gated on CHIMERA_SMOKE_MODEL=<path/to/model.gguf>)
//      Actually load a model and run llama_decode() once. This proves
//      the linked library can produce compute, not just symbols and
//      backend registrations.
//
// Build via tests/external/CMakeLists.txt. This file is intentionally
// trivial - the link line is the test, not the code.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

#include "chimera.h"

int main() {
    // (1) ggml side: count registered backend devices. If the whole-
    // archive wrapper was forgotten the CPU backend constructor would
    // never run and this returns 0, which is the canonical "silent
    // backend disappearance" failure mode the doc warns about.
    const size_t devs = ggml_backend_dev_count();
    std::printf("ggml_backend_dev_count = %zu\n", devs);
    if (devs == 0) {
        std::fprintf(stderr, "FAIL: no ggml backends registered - "
                             "whole-archive wrapper missing?\n");
        return 1;
    }

    // (2) llama side: a cheap symbol-resolution probe. Returns a
    // pointer to a static string describing CPU features the build
    // was compiled with. Forces a reference into libllama.a.
    std::printf("llama_print_system_info:\n  %s\n", llama_print_system_info());

    // (3) chimera side: pull in a chimera symbol so the link includes
    // libchimera.a. CHIMERA_VERSION is a PUBLIC compile define on
    // chimera_lib, but a macro alone doesn't force a link reference.
    // Calling a real function does. These two are cheap no-ops.
#ifdef CHIMERA_HAS_WHISPER
    chimera_silence_whisper_log();
    chimera_restore_whisper_log();
#endif
#ifdef CHIMERA_HAS_SD
    chimera_silence_sd_log();
    chimera_restore_sd_log();
#endif

    std::printf("chimera version: %s\n", CHIMERA_VERSION);

    // (4) Optional inference probe. Gated on CHIMERA_SMOKE_MODEL so the
    // test stays runnable on machines without a model file. The point
    // here is to exercise a real compute path through the linked
    // library, not to validate inference quality.
    const char * model_path = std::getenv("CHIMERA_SMOKE_MODEL");
    if (model_path == nullptr || std::strlen(model_path) == 0) {
        std::printf("inference probe: SKIP (set CHIMERA_SMOKE_MODEL to enable)\n");
        std::printf("PASS\n");
        return 0;
    }

    std::printf("inference probe: loading %s\n", model_path);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    // n_gpu_layers=0 keeps the test deterministic across hardware - we
    // are proving the link works, not benchmarking the GPU path. The
    // CPU backend is registered via the same whole-archive contract.
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "FAIL: llama_model_load_from_file returned null\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 256;
    cparams.n_batch = 256;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL: llama_init_from_model returned null\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * prompt = "Hello";
    // Two-call tokenize: first to size, then to fill.
    int32_t n_tokens = -llama_tokenize(vocab, prompt, (int32_t)std::strlen(prompt),
                                       nullptr, 0, /*add_special=*/true,
                                       /*parse_special=*/false);
    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(vocab, prompt, (int32_t)std::strlen(prompt),
                              tokens.data(), (int32_t)tokens.size(),
                              /*add_special=*/true, /*parse_special=*/false);
    if (n_tokens <= 0) {
        std::fprintf(stderr, "FAIL: llama_tokenize returned %d\n", (int)n_tokens);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    std::printf("inference probe: tokenized to %d tokens\n", (int)n_tokens);

    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "FAIL: llama_decode returned non-zero\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Validate output. A generative model populates logits, an
    // embedding model run with embeddings=true populates embeddings.
    // We do not know which kind the caller passed, so try logits
    // first and fall through to embeddings if logits are empty (an
    // embedding model with default cparams returns a zero-filled
    // logits buffer, not null - we have to detect by content).
    auto scan = [](const float * buf, int n, int & finite, float & max_abs) {
        finite = 0;
        max_abs = 0.0f;
        if (buf == nullptr) return;
        for (int i = 0; i < n; ++i) {
            if (std::isfinite(buf[i])) {
                ++finite;
                if (std::fabs(buf[i]) > max_abs) max_abs = std::fabs(buf[i]);
            }
        }
    };

    const float * out = llama_get_logits_ith(ctx, n_tokens - 1);
    const char * out_kind = "logits";
    int out_n = llama_vocab_n_tokens(vocab);
    int n_finite = 0;
    float max_abs = 0.0f;
    scan(out, out_n, n_finite, max_abs);

    // Empty logits buffer? Try the embedding side.
    if (out == nullptr || max_abs == 0.0f) {
        const float * emb = llama_get_embeddings_seq(ctx, 0);
        if (emb == nullptr) {
            emb = llama_get_embeddings_ith(ctx, n_tokens - 1);
        }
        if (emb != nullptr) {
            out = emb;
            out_kind = "embeddings";
            out_n = llama_model_n_embd(model);
            scan(out, out_n, n_finite, max_abs);
        }
    }

    if (out == nullptr) {
        std::fprintf(stderr, "FAIL: no logits or embeddings produced\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    std::printf("inference probe: produced %s, n=%d, finite=%d/%d, max|x|=%g\n",
                out_kind, out_n, n_finite, out_n, (double)max_abs);
    if (n_finite != out_n) {
        std::fprintf(stderr, "FAIL: non-finite values in output - broken backend kernel?\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    if (max_abs == 0.0f) {
        std::fprintf(stderr, "FAIL: output is all zeros - backend did not actually compute. "
                             "If this is an embedding model, the consumer must run it with "
                             "cparams.embeddings = true; this smoke test uses defaults.\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    std::printf("PASS\n");
    return 0;
}
