"""Tests for manage.py's source-patch application (GgmlBuilder._apply_patch)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import manage  # noqa: E402

PATCH = """--- a/src.c
+++ b/src.c
@@ -1,3 +1,3 @@
 int a;
-int b;
+int b = 1;
 int c;
"""


class FakeBuilder(manage.GgmlBuilder):
    name = "fake.cpp"
    version = "v1"
    repo_url = ""


@pytest.fixture
def builder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # Project() lays out build/ under the cwd
    b = FakeBuilder()
    b.src_dir.mkdir(parents=True)
    return b


def write(builder, source: str, patch: str = PATCH) -> Path:
    (builder.src_dir / "src.c").write_text(source)
    p = builder.src_dir.parent / "fix.patch"
    p.write_text(patch)
    return p


def test_applies_matching_patch(builder):
    builder._apply_patch(write(builder, "int a;\nint b;\nint c;\n"))
    assert (builder.src_dir / "src.c").read_text() == "int a;\nint b = 1;\nint c;\n"


def test_skips_already_applied_patch(builder):
    builder._apply_patch(write(builder, "int a;\nint b = 1;\nint c;\n"))
    assert (builder.src_dir / "src.c").read_text() == "int a;\nint b = 1;\nint c;\n"


def test_non_matching_patch_fails_the_build(builder, caplog):
    patch = write(builder, "int a;\nint moved;\nint c;\n")
    with pytest.raises(SystemExit) as exc:
        builder._apply_patch(patch)
    assert exc.value.code == 1
    assert "fix.patch no longer applies to fake.cpp v1" in caplog.text
    assert "patch failed" in caplog.text  # git's own reason is included
    assert (builder.src_dir / "src.c").read_text() == "int a;\nint moved;\nint c;\n"


@pytest.mark.parametrize("vendored, expect_ggml", [("0", False), ("1", True)])
def test_sd_gets_ggml_patches_only_when_compiling_its_own_ggml(
    tmp_path, monkeypatch, vendored, expect_ggml
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SD_USE_VENDORED_GGML", vendored)
    seen: list[str] = []
    monkeypatch.setattr(manage.GgmlBuilder, "_apply_patch", lambda self, p: seen.append(p.name))
    manage.StableDiffusionCppBuilder()._apply_source_patches()
    assert any(n.startswith("ggml-") for n in seen) is expect_ggml
    assert all(n.startswith(("ggml-", "stable-diffusion.cpp-")) for n in seen)


def test_other_trees_always_get_ggml_patches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SD_USE_VENDORED_GGML", "0")
    for cls in (manage.LlamaCppBuilder, manage.WhisperCppBuilder):
        seen: list[str] = []
        monkeypatch.setattr(manage.GgmlBuilder, "_apply_patch", lambda self, p: seen.append(p.name))
        cls()._apply_source_patches()
        assert any(n.startswith("ggml-") for n in seen), cls.__name__
