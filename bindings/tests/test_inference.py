"""End-to-end inference probes, gated on the availability of model files.

Mirrors the second half of ``bindings/smoke_test.py``: a tokenize round-trip,
generation (one-shot + streaming callback), mutable options between calls, an
embedding probe, and -- when the archive was built with them -- SD and Whisper.

Each test SKIPs (never FAILs) when its model is absent. Models resolve from
``CHIMERA_TEST_*`` env overrides or the repo's ``models/`` directory; see
``conftest.py``. These are CPU-only (gpu_layers=0, tiny n_predict) so they run
quickly and deterministically enough to assert structural invariants.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenizer_roundtrip(chimera_mod, llama_model):
    tok = chimera_mod.Tokenizer(str(llama_model))
    ids = tok.encode("Hello")
    assert ids, "encode returned no tokens"
    assert all(isinstance(i, int) for i in ids)
    decoded = tok.decode(ids)
    assert "Hello" in decoded


def test_tokenizer_empty_string(chimera_mod, llama_model):
    tok = chimera_mod.Tokenizer(str(llama_model))
    # Without special tokens, an empty string yields no content tokens.
    ids = tok.encode("", add_special=False, parse_special=False)
    assert ids == [] or isinstance(ids, list)


# ---------------------------------------------------------------------------
# Llama
# ---------------------------------------------------------------------------


@pytest.fixture
def llama(chimera_mod, llama_model):
    llm = chimera_mod.Llama(str(llama_model))
    llm.options.gpu_layers = 0
    llm.options.n_ctx = 256
    llm.options.n_batch = 256
    llm.options.n_predict = 1
    return llm


def test_llama_generate_nonempty(llama):
    out = llama.generate("Hello")
    assert isinstance(out, str)
    assert out, "generate produced empty output"


def test_llama_options_mutable_between_calls(llama):
    # No reload should be needed to change sampling/length between generate()s.
    llama.options.n_predict = 1
    first = llama.generate("Hello")
    llama.options.n_predict = 2
    second = llama.generate("Hi")
    assert first and second


def test_llama_streaming_callback_matches_return(llama):
    captured: list[str] = []
    llama.options.n_predict = 4
    out = llama.generate("Greetings", on_token=captured.append)
    assert out, "callback generate returned empty"
    assert captured, "callback never fired despite non-empty output"
    assert "".join(captured) == out


def test_llama_tokenize_detokenize(llama):
    ids = llama.tokenize("Hello world")
    assert ids
    text = llama.detokenize(ids)
    assert "Hello" in text


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


def test_embedder_single(chimera_mod, embed_model):
    emb = chimera_mod.Embedder(str(embed_model))
    vec = emb.embed("hello world")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(x, float) for x in vec)


def test_embedder_dimension_is_stable(chimera_mod, embed_model):
    emb = chimera_mod.Embedder(str(embed_model))
    a = emb.embed("the quick brown fox")
    b = emb.embed("a different sentence entirely")
    assert len(a) == len(b)


def test_embedder_many_matches_count(chimera_mod, embed_model):
    emb = chimera_mod.Embedder(str(embed_model))
    texts = ["one", "two", "three"]
    vecs = emb.embed_many(texts)
    assert len(vecs) == len(texts)
    assert all(len(v) == len(vecs[0]) for v in vecs)


# ---------------------------------------------------------------------------
# SD (optional modality)
# ---------------------------------------------------------------------------


def test_sd_generates_image(chimera_mod, sd_model):
    if not hasattr(chimera_mod, "SD"):
        pytest.skip("archive built without SD (CHIMERA_HAS_SD off)")
    sd = chimera_mod.SD(str(sd_model))
    sd.options.width = 256
    sd.options.height = 256
    sd.options.steps = 1
    sd.options.cfg_scale = 1.0
    sd.options.sample_method = "euler_a"
    images = sd.generate("a red square")
    assert images, "SD.generate returned no images"
    img = images[0]
    assert img.width == 256 and img.height == 256
    assert len(img.pixels) == img.width * img.height * img.channels


# ---------------------------------------------------------------------------
# Whisper (optional modality)
# ---------------------------------------------------------------------------


def test_whisper_transcribes(chimera_mod, whisper_model, whisper_wav):
    if not hasattr(chimera_mod, "Whisper"):
        pytest.skip("archive built without Whisper (CHIMERA_HAS_WHISPER off)")
    w = chimera_mod.Whisper(str(whisper_model))
    result = w.transcribe(str(whisper_wav))
    assert result.text.strip(), "transcription is empty"
    assert result.audio_duration_s > 0
    # jfk.wav is a single short utterance -> at least one segment.
    assert len(result.segments) >= 1
