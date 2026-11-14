"""Tests for whisper.cpp compiling against llama.cpp's ggml (WhisperCppBuilder)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import manage  # noqa: E402


@pytest.fixture
def builder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # Project() lays out build/ under the cwd
    b = manage.WhisperCppBuilder()
    (b.llama_ggml / "include").mkdir(parents=True)
    b.include.mkdir(parents=True)
    return b


def headers(directory: Path, **files: str) -> None:
    for name, text in files.items():
        (directory / f"{name}.h").write_text(text)


def test_verify_accepts_llama_headers(builder):
    headers(builder.llama_ggml / "include", ggml="enum ggml_op { A };", gguf="x")
    headers(builder.include, ggml="enum ggml_op { A };", gguf="x", whisper="w")
    builder._verify_shared_ggml_headers()


def test_verify_ignores_llama_headers_ggml_does_not_install(builder):
    headers(builder.llama_ggml / "include", ggml="x", **{"ggml-et": "y"})
    headers(builder.include, ggml="x")
    builder._verify_shared_ggml_headers()


def test_verify_rejects_header_llama_does_not_have(builder, caplog):
    headers(builder.llama_ggml / "include", ggml="x")
    headers(builder.include, ggml="x", **{"ggml-kompute": "old"})
    with pytest.raises(SystemExit):
        builder._verify_shared_ggml_headers()
    assert "['ggml-kompute.h']" in caplog.text


def test_verify_rejects_whispers_own_headers(builder, caplog):
    headers(builder.llama_ggml / "include", ggml="enum ggml_op { A };", gguf="x")
    headers(builder.include, ggml="enum ggml_op { ADD1, A };", gguf="x")
    with pytest.raises(SystemExit):
        builder._verify_shared_ggml_headers()
    assert "['ggml.h']" in caplog.text


def test_verify_rejects_missing_headers(builder, caplog):
    headers(builder.llama_ggml / "include", ggml="x")
    with pytest.raises(SystemExit):
        builder._verify_shared_ggml_headers()
    assert "ggml.h (missing)" in caplog.text


def test_parent_project_adds_llama_ggml_before_whisper(builder, tmp_path):
    builder._write_parent_project(tmp_path / "parent")
    lines = (tmp_path / "parent" / "CMakeLists.txt").read_text().splitlines()
    subdirs = [line for line in lines if line.startswith("add_subdirectory")]
    assert subdirs == [
        f'add_subdirectory("{builder.llama_ggml.as_posix()}" ggml)',
        f'add_subdirectory("{builder.src_dir.as_posix()}" whisper)',
    ]


def test_whisper_skips_ggml_patches(builder):
    assert not any(p.name.startswith("ggml-") for p in builder.source_patches())


def test_max_name_flags_keep_platform_defaults(builder, monkeypatch):
    # A plain CMAKE_CXX_FLAGS replaces MSVC's /EHsc /GR; the _INIT form keeps them.
    monkeypatch.setattr(manage.StableDiffusionCppBuilder, "uses_shared_ggml", staticmethod(lambda: True))
    monkeypatch.setattr(manage.StableDiffusionCppBuilder, "ggml_max_name", classmethod(lambda cls, project=None: 160))
    assert builder.ggml_max_name_flags() == {
        f"CMAKE_{lang}_FLAGS_INIT": "-DGGML_MAX_NAME=160" for lang in ("C", "CXX", "CUDA", "HIP")
    }
