"""Pytest fixtures and import wiring for the chimera nanobind bindings.

The extension is a host build: the compiled module lives in ``bindings/build``
(produced by ``make bindings`` / ``uv pip install -e ./bindings``) rather than
on ``sys.path``. To let ``pytest`` find it without an install step, we prepend
the build dir to ``sys.path`` here if a plain ``import chimera`` would fail.

Model-dependent tests are gated: each model fixture resolves a path from an
environment override, falling back to the repo's ``models/`` directory (the
same fixtures ``scripts/test.py`` uses). A missing model SKIPs the test rather
than failing it, mirroring the rest of the suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# bindings/tests/conftest.py -> bindings/ -> repo root
BINDINGS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BINDINGS_DIR.parent
MODELS = REPO_ROOT / "models"


def _ensure_importable() -> None:
    """Make ``import chimera`` work from an uninstalled build tree.

    Tries a plain import first (covers a working installed/editable wheel). If
    that fails -- including the case where a stale scikit-build editable hook
    intercepts the import and aborts on a failed rebuild -- we drop that hook
    and fall back to the standalone CMake output dir (``bindings/build``),
    where ``make bindings`` drops the prebuilt ``chimera.*.so``.
    """
    try:
        import chimera  # noqa: F401
        return
    except Exception:
        # A broken editable rebuild raises CalledProcessError (not ImportError),
        # so catch broadly. Remove the scikit-build editable meta-path finder so
        # it stops shadowing the prebuilt extension below.
        sys.meta_path[:] = [
            f for f in sys.meta_path
            if "_chimera_editable" not in getattr(type(f), "__module__", "")
        ]
    build_dir = BINDINGS_DIR / "build"
    if build_dir.is_dir():
        sys.path.insert(0, str(build_dir))


_ensure_importable()


def _resolve_model(env_var: str, *candidates: str) -> Path | None:
    """Return a model path from ``$env_var``, else the first existing candidate
    under ``models/``, else None."""
    override = os.environ.get(env_var)
    if override:
        p = Path(override)
        return p if p.is_file() else None
    for name in candidates:
        p = MODELS / name
        if p.is_file():
            return p
    return None


@pytest.fixture(scope="session")
def chimera_mod():
    """The imported extension module, or skip the whole test if unbuilt."""
    try:
        import chimera
    except ImportError as e:  # pragma: no cover - environment guard
        pytest.skip(f"chimera extension not importable (build it first): {e}")
    return chimera


@pytest.fixture(scope="session")
def llama_model() -> Path:
    p = _resolve_model("CHIMERA_TEST_LLAMA_MODEL", "Llama-3.2-1B-Instruct-Q8_0.gguf")
    if p is None:
        pytest.skip("no llama model (set CHIMERA_TEST_LLAMA_MODEL or populate models/)")
    return p


@pytest.fixture(scope="session")
def embed_model() -> Path:
    p = _resolve_model(
        "CHIMERA_TEST_EMBED_MODEL", "bge-small-en-v1.5-q8_0.gguf", "gte-small-q8_0.gguf"
    )
    if p is None:
        pytest.skip("no embedding model (set CHIMERA_TEST_EMBED_MODEL or populate models/)")
    return p


@pytest.fixture(scope="session")
def sd_model() -> Path:
    p = _resolve_model(
        "CHIMERA_TEST_SD_MODEL",
        "sd_xl_turbo_1.0.q8_0.gguf",
        "v1-5-pruned-emaonly.q8_0.gguf",
        "z_image_turbo-Q6_K.gguf",
    )
    if p is None:
        pytest.skip("no SD model (set CHIMERA_TEST_SD_MODEL or populate models/)")
    return p


@pytest.fixture(scope="session")
def whisper_model() -> Path:
    p = _resolve_model("CHIMERA_TEST_WHISPER_MODEL", "ggml-base.en.bin")
    if p is None:
        pytest.skip("no whisper model (set CHIMERA_TEST_WHISPER_MODEL or populate models/)")
    return p


@pytest.fixture(scope="session")
def whisper_wav() -> Path:
    override = os.environ.get("CHIMERA_TEST_WHISPER_WAV")
    wav = Path(override) if override else REPO_ROOT / "build" / "whisper.cpp" / "samples" / "jfk.wav"
    if not wav.is_file():
        pytest.skip(f"no sample wav at {wav} (set CHIMERA_TEST_WHISPER_WAV)")
    return wav
