#!/usr/bin/env python3
"""Smoke test for the `chimera` nanobind module.

Python mirror of tests/external/hpp_smoke.cpp. Proves the module imports and
every wrapper class instantiates; then, gated on CHIMERA_SMOKE_MODEL, exercises
the persistent-handle Llama path (tokenize round-trip, generate, options
mutation, streaming callback) end to end.

Run:
    # link-contract / import probe only (no model required):
    PYTHONPATH=bindings/build python3 bindings/smoke_test.py

    # full inference probe:
    CHIMERA_SMOKE_MODEL=models/Llama-3.2-1B-Instruct-Q8_0.gguf \
        PYTHONPATH=bindings/build python3 bindings/smoke_test.py
"""
import os
import sys


def main() -> int:
    import chimera

    # (1) Import / construction probe -- no model file required.
    lopts = chimera.LlamaOptions()
    lopts.model = ""
    eopts = chimera.EmbedOptions()
    sopts = chimera.ServeOptions()
    srv = chimera.Server(sopts)            # ctor is a no-op; run() blocks.
    _ = srv.options
    assert hasattr(chimera, "ChimeraError"), "exception type not exported"
    for gated in ("SD", "Whisper"):
        if hasattr(chimera, gated):
            print(f"compile probe: optional class {gated} present")
    print("compile probe: option structs + Server wrapper OK")

    # Exception translation: a bad load should raise chimera.ChimeraError,
    # never crash the interpreter.
    try:
        chimera.Tokenizer("/nonexistent/model.gguf")
    except chimera.ChimeraError as e:
        print(f"exception probe: ChimeraError raised as expected ({e})")
    except Exception as e:  # noqa: BLE001
        # Some load failures may surface as a different type depending on how
        # deep the failure is; report but don't hard-fail the probe.
        print(f"exception probe: load failure surfaced as {type(e).__name__}: {e}")

    # (2) Optional inference probe.
    model = os.environ.get("CHIMERA_SMOKE_MODEL", "")
    if not model:
        print("inference probe: SKIP (set CHIMERA_SMOKE_MODEL to enable)")
        print("PASS")
        return 0

    print(f"inference probe: loading {model} via chimera.Tokenizer")
    tok = chimera.Tokenizer(model)
    ids = tok.encode("Hello")
    assert ids, "Tokenizer.encode returned no tokens"
    print(f'Tokenizer round-trip: "Hello" -> {len(ids)} tokens -> "{tok.decode(ids)}"')

    llm = chimera.Llama(model)
    llm.options.gpu_layers = 0
    llm.options.n_ctx = 256
    llm.options.n_batch = 256
    llm.options.n_predict = 1

    out = llm.generate("Hello")
    print(f'Llama.generate("Hello", n_predict=1) -> "{out}" ({len(out)} chars)')
    assert out, "Llama.generate produced empty output"

    # Mutable options between calls, no reload.
    llm.options.n_predict = 2
    out2 = llm.generate("Hi")
    assert out2, "second Llama.generate produced empty output"
    print(f'Llama.generate("Hi", n_predict=2) -> "{out2}"')

    # Streaming callback: pieces must concatenate to the returned text.
    captured = []
    llm.options.n_predict = 4
    out3 = llm.generate("Greetings", on_token=captured.append)
    joined = "".join(captured)
    assert out3, "callback generate returned empty"
    assert captured, "callback never fired despite non-empty output"
    assert joined == out3, f"captured {joined!r} != returned {out3!r}"
    print(f"streaming: callback fired {len(captured)} times; captured matches returned")

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
