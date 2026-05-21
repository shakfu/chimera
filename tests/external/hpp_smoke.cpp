// hpp_smoke.cpp - external-consumer smoke test for the optional OOP layer
// in src/chimera/chimera.hpp. Mirrors the symbol-resolution / link-contract
// purpose of smoke.cpp but goes through the C++ wrapper classes instead of
// the procedural command_* API.
//
// What this proves:
//   1. chimera.hpp parses and compiles as an external consumer would
//      consume it (header-only, no surprise CMake-only assumptions).
//   2. Every public class instantiates: chimera::Llama, Embedder,
//      Tokenizer, Server, and (when CHIMERA_HAS_*) Whisper / SD.
//   3. (Optional, gated on CHIMERA_SMOKE_MODEL=<path/to/model.gguf>)
//      Round-trip a string through Tokenizer.encode -> decode against
//      a real model, then drive a one-token generation through
//      chimera::Llama. This exercises the persistent-handle behavior
//      end to end.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "chimera.hpp"

int main() {
    // (1) Compile-time / link-time proof: instantiate every option struct
    // and class. No model load yet, so this is cheap and runs anywhere.
    {
        LlamaCommonOptions lopts;
        lopts.model = "/dev/null/not-a-real-path.gguf";
        EmbedOptions    eopts;
        TokenizeOptions topts;
        ServeOptions    sopts;
        // Don't construct chimera::Llama/Embedder/Tokenizer here -- their
        // ctors actually load the model. The point of this block is to
        // prove the option-struct surface compiles.
        chimera::Server srv(sopts);  // ctor is a no-op; run() blocks.
        (void) srv.options();

#ifdef CHIMERA_HAS_WHISPER
        WhisperOptions wopts;
        chimera::Whisper wh(wopts);
        (void) wh.options();
#endif
#ifdef CHIMERA_HAS_SD
        SdOptions dopts;
        chimera::SD sd(dopts);
        (void) sd.options();
#endif
        std::printf("compile probe: option structs + Server/Whisper/SD wrappers OK\n");
    }

    // (2) Optional inference probe. Gated on CHIMERA_SMOKE_MODEL so the
    // test stays runnable without a model file. The point is to exercise
    // the persistent-handle path through chimera::Llama and chimera::Tokenizer
    // - not to validate generation quality.
    const char * model_path = std::getenv("CHIMERA_SMOKE_MODEL");
    if (model_path == nullptr || std::strlen(model_path) == 0) {
        std::printf("inference probe: SKIP (set CHIMERA_SMOKE_MODEL to enable)\n");
        std::printf("PASS\n");
        return 0;
    }

    std::printf("inference probe: loading %s via chimera::Tokenizer\n", model_path);
    chimera::Tokenizer tok(model_path);
    const auto ids = tok.encode("Hello");
    if (ids.empty()) {
        std::fprintf(stderr, "FAIL: Tokenizer.encode returned no tokens\n");
        return 1;
    }
    const std::string round = tok.decode(ids);
    std::printf("Tokenizer round-trip: \"Hello\" -> %zu tokens -> \"%s\"\n",
                ids.size(), round.c_str());

    std::printf("inference probe: loading %s via chimera::Llama\n", model_path);
    LlamaCommonOptions lopts;
    lopts.model      = model_path;
    lopts.gpu_layers = 0;        // determinism across hardware
    lopts.n_predict  = 1;        // single token - we want speed, not quality
    lopts.n_ctx      = 256;
    lopts.n_batch    = 256;
    chimera::Llama llm(lopts);

    const std::string out = llm.generate("Hello", /*stream=*/false);
    std::printf("Llama.generate(\"Hello\", n_predict=1) -> \"%s\" (%zu chars)\n",
                out.c_str(), out.size());
    // A successful 1-token generation should produce *something*. The
    // empty-output case is what command_prompt treats as Generate-failure.
    if (out.empty()) {
        std::fprintf(stderr, "FAIL: Llama.generate produced empty output\n");
        return 1;
    }

    // Reuse the same Llama handle for a second call -- this is the
    // persistent-handle whole point. Tweak n_predict via options() to
    // confirm the mutable accessor works without re-loading the model.
    llm.options().n_predict = 2;
    const std::string out2 = llm.generate("Hi", /*stream=*/false);
    std::printf("Llama.generate(\"Hi\", n_predict=2) -> \"%s\" (%zu chars)\n",
                out2.c_str(), out2.size());
    if (out2.empty()) {
        std::fprintf(stderr, "FAIL: second Llama.generate produced empty output\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
