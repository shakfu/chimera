"""Import / construction / option-coverage probes for the chimera module.

These require no model files -- they exercise the link contract (module
imports, every wrapper class and option struct constructs), option-field
read/write round-trips, and exception translation. This is the pytest mirror
of the first half of ``bindings/smoke_test.py`` and ``tests/external/hpp_smoke.cpp``.
"""

from __future__ import annotations

import pytest

# Classes that are always present regardless of which modalities the underlying
# archive was built with.
CORE_CLASSES = [
    "ExitCode",
    "ChimeraError",
    "LlamaOptions",
    "Llama",
    "EmbedOptions",
    "Embedder",
    "Tokenizer",
    "ServeOptions",
    "Server",
]

# Present only when libchimera.a was built with the modality (CHIMERA_HAS_SD /
# CHIMERA_HAS_WHISPER). Absence is not a failure.
OPTIONAL_CLASSES = ["SD", "SdOptions", "PixelImage", "Whisper", "WhisperOptions",
                    "Segment", "TranscribeResult"]


def test_module_imports(chimera_mod):
    assert chimera_mod.__doc__


@pytest.mark.parametrize("name", CORE_CLASSES)
def test_core_class_exported(chimera_mod, name):
    assert hasattr(chimera_mod, name), f"core symbol {name} missing from module"


def test_exit_code_members(chimera_mod):
    members = {m for m in dir(chimera_mod.ExitCode) if not m.startswith("_")}
    assert {"Runtime", "BadInput", "Load", "Generate"} <= members


def test_chimera_error_is_exception(chimera_mod):
    assert issubclass(chimera_mod.ChimeraError, Exception)


def test_option_structs_default_construct(chimera_mod):
    # Every bound option struct must be default-constructible.
    chimera_mod.LlamaOptions()
    chimera_mod.EmbedOptions()
    chimera_mod.ServeOptions()
    for name in ("SdOptions", "WhisperOptions"):
        cls = getattr(chimera_mod, name, None)
        if cls is not None:
            cls()


def test_server_constructs_without_running(chimera_mod):
    # The Server ctor is a no-op (run() is what blocks), so this must not hang.
    srv = chimera_mod.Server(chimera_mod.ServeOptions())
    assert srv.options is not None


def test_llama_options_roundtrip(chimera_mod):
    o = chimera_mod.LlamaOptions()
    o.model = "/some/path.gguf"
    o.n_ctx = 1234
    o.n_predict = 7
    o.temp = 0.25
    o.gpu_layers = 0
    o.ignore_eos = True
    o.samplers = "top_k;top_p;temperature"
    assert o.model == "/some/path.gguf"
    assert o.n_ctx == 1234
    assert o.n_predict == 7
    assert o.temp == pytest.approx(0.25)
    assert o.gpu_layers == 0
    assert o.ignore_eos is True
    assert o.samplers == "top_k;top_p;temperature"


def test_embed_options_roundtrip(chimera_mod):
    o = chimera_mod.EmbedOptions()
    o.model = "/emb.gguf"
    o.pooling = "mean"
    o.normalize = True
    o.n_batch = 64
    assert (o.model, o.pooling, o.normalize, o.n_batch) == ("/emb.gguf", "mean", True, 64)


def test_serve_options_roundtrip(chimera_mod):
    o = chimera_mod.ServeOptions()
    o.host = "127.0.0.1"
    o.port = 8123
    o.n_ctx = 2048
    o.embedding = True
    o.api_key = "secret"
    assert (o.host, o.port, o.n_ctx, o.embedding, o.api_key) == (
        "127.0.0.1", 8123, 2048, True, "secret",
    )


def test_options_handle_is_live_reference(chimera_mod):
    # `.options` returns a reference into the C++ object (reference_internal),
    # so a mutation through the handle is observable on re-fetch.
    srv = chimera_mod.Server(chimera_mod.ServeOptions())
    srv.options.port = 9999
    assert srv.options.port == 9999


def test_chat_template_kwargs_is_dict(chimera_mod):
    o = chimera_mod.LlamaOptions()
    o.chat_template_kwargs = {"enable_thinking": "true"}
    assert o.chat_template_kwargs == {"enable_thinking": "true"}


def test_lora_adapters_is_list(chimera_mod):
    o = chimera_mod.LlamaOptions()
    o.lora_adapters = ["/a.gguf", "/b.gguf"]
    assert list(o.lora_adapters) == ["/a.gguf", "/b.gguf"]


def test_bad_model_path_raises_chimera_error(chimera_mod):
    # A load failure must translate to a Python exception, never crash the
    # interpreter or call exit(). Tokenizer loads eagerly in its ctor.
    with pytest.raises(chimera_mod.ChimeraError):
        chimera_mod.Tokenizer("/nonexistent/model-does-not-exist.gguf")
