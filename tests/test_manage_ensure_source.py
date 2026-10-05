"""Tests for manage.py re-cloning a source tree left at a stale ref (AbstractBuilder.ensure_source)."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import manage  # noqa: E402

PATCH = """--- a/src.c
+++ b/src.c
@@ -1 +1 @@
-int v = 2;
+int v = 3;
"""


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def upstream(tmp_path) -> Path:
    """A repo with tags v1 and v2, each changing src.c."""
    repo = tmp_path / "remote" / "fake.cpp"  # git_clone names the clone after the url
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    for tag in ("v1", "v2"):
        (repo / "src.c").write_text(f"int v = {tag[1]};\n")
        git(repo, "add", "src.c")
        git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", tag)
        git(repo, "tag", tag)
    return repo


@pytest.fixture
def make_builder(tmp_path, monkeypatch, upstream):
    monkeypatch.chdir(tmp_path)  # Project() lays out build/ under the cwd

    def make(version: str, patches: list[Path] = []) -> manage.Builder:
        class FakeBuilder(manage.Builder):
            name = "fake.cpp"
            repo_url = upstream.as_uri()

            def source_patches(self) -> list[Path]:
                return patches

        return FakeBuilder(version=version)

    return make


def test_clones_when_missing(make_builder):
    b = make_builder("v2")
    b.ensure_source()
    assert (b.src_dir / "src.c").read_text() == "int v = 2;\n"


def test_reclones_a_stale_checkout(make_builder):
    make_builder("v1").ensure_source()
    b = make_builder("v2")
    b.ensure_source()
    assert (b.src_dir / "src.c").read_text() == "int v = 2;\n"


def test_keeps_a_checkout_at_the_pin(make_builder):
    b = make_builder("v2")
    b.ensure_source()
    (b.src_dir / "build").mkdir()  # stands in for a configured build tree
    b.ensure_source()
    assert (b.src_dir / "build").is_dir()


def test_refuses_to_delete_local_edits(make_builder, caplog):
    make_builder("v1").ensure_source()
    b = make_builder("v2")
    (b.src_dir / "src.c").write_text("int mine;\n")
    with pytest.raises(SystemExit):
        b.ensure_source()
    assert "has local edits outside scripts/patches/: ['src.c']" in caplog.text
    assert (b.src_dir / "src.c").read_text() == "int mine;\n"


def test_edits_from_own_patches_do_not_block_reclone(make_builder, tmp_path):
    patch = tmp_path / "fix.patch"
    patch.write_text(PATCH)
    make_builder("v1").ensure_source()
    b = make_builder("v2", [patch])
    (b.src_dir / "src.c").write_text("int v = 3;\n")  # what applying the patch leaves
    b.ensure_source()
    assert (b.src_dir / "src.c").read_text() == "int v = 2;\n"
