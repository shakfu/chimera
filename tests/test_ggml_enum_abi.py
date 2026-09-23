"""Tests for manage.py's ggml enum ABI check (ggml_enum_mismatches)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import manage  # noqa: E402

GGML_H = """
// comment with enum fake { NOPE };
enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    /* 2 and 3 removed */
    GGML_TYPE_Q4_0 = 2u,
    GGML_TYPE_COUNT,
};
enum ggml_op {
    GGML_OP_NONE = 0,
    GGML_OP_DUP,
    GGML_OP_ADD,
    GGML_OP_COUNT,
};
enum ggml_scale_flag {
    GGML_SCALE_FLAG_ALIGN_CORNERS = (1 << 8),
    GGML_SCALE_FLAG_BOTH = GGML_SCALE_FLAG_ALIGN_CORNERS | 1,
};
"""


def write_tree(root: Path, ggml_h: str, **extra: str) -> Path:
    root.mkdir(parents=True)
    (root / "ggml.h").write_text(ggml_h)
    for name, text in extra.items():
        (root / name).write_text(text)
    return root


def test_parse_values(tmp_path):
    enums = manage.parse_c_enums(write_tree(tmp_path / "a", GGML_H) / "ggml.h")
    assert set(enums) == {"ggml_type", "ggml_op", "ggml_scale_flag"}
    assert enums["ggml_type"] == {
        "GGML_TYPE_F32": 0, "GGML_TYPE_F16": 1, "GGML_TYPE_Q4_0": 2, "GGML_TYPE_COUNT": 3,
    }
    assert enums["ggml_op"]["GGML_OP_ADD"] == 2
    assert enums["ggml_scale_flag"] == {
        "GGML_SCALE_FLAG_ALIGN_CORNERS": 256, "GGML_SCALE_FLAG_BOTH": 257,
    }


def test_identical_trees_match(tmp_path):
    a = write_tree(tmp_path / "a", GGML_H)
    b = write_tree(tmp_path / "b", GGML_H)
    assert manage.ggml_enum_mismatches(a, b) == []


def test_appended_enumerator_is_compatible(tmp_path):
    consumer = write_tree(tmp_path / "a", GGML_H)
    provider = write_tree(
        tmp_path / "b", GGML_H.replace("GGML_OP_ADD,\n", "GGML_OP_ADD,\n    GGML_OP_SUB,\n")
    )
    assert manage.ggml_enum_mismatches(consumer, provider) == []


def test_inserted_enumerator_is_reported(tmp_path):
    consumer = write_tree(tmp_path / "a", GGML_H)
    provider = write_tree(
        tmp_path / "b", GGML_H.replace("GGML_OP_DUP,\n", "GGML_OP_NEW,\n    GGML_OP_DUP,\n")
    )
    problems = manage.ggml_enum_mismatches(consumer, provider)
    assert problems == [
        "ggml.h: ggml_op::GGML_OP_DUP = 1, linked ggml has 2",
        "ggml.h: ggml_op::GGML_OP_ADD = 2, linked ggml has 3",
    ]


def test_removed_enumerator_and_enum_are_reported(tmp_path):
    consumer = write_tree(tmp_path / "a", GGML_H)
    provider_h = GGML_H.replace("    GGML_OP_ADD,\n", "")
    provider_h = provider_h[: provider_h.index("enum ggml_scale_flag")]
    provider = write_tree(tmp_path / "b", provider_h)
    problems = manage.ggml_enum_mismatches(consumer, provider)
    assert "ggml.h: ggml_op::GGML_OP_ADD = 2, linked ggml has None" in problems
    assert any("enum ggml_scale_flag missing" in p for p in problems)


def test_headers_absent_from_provider_are_skipped(tmp_path):
    consumer = write_tree(tmp_path / "a", GGML_H, **{"ggml-foo.h": "enum foo { FOO_A };"})
    provider = write_tree(tmp_path / "b", GGML_H)
    assert manage.ggml_enum_mismatches(consumer, provider) == []


def test_unevaluable_initializer_raises(tmp_path):
    header = write_tree(tmp_path / "a", "enum e { E_A = sizeof(int) };") / "ggml.h"
    with pytest.raises(ValueError, match="cannot evaluate"):
        manage.parse_c_enums(header)


def test_pinned_whisper_matches_pinned_llama():
    whisper = ROOT / "build" / "whisper.cpp" / "ggml" / "include"
    llama = ROOT / "build" / "llama.cpp" / "ggml" / "include"
    if not (whisper.exists() and llama.exists()):
        pytest.skip("run `make deps` first")
    assert manage.ggml_enum_mismatches(whisper, llama) == []
