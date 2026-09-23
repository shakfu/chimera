"""Every field of a bound option struct in chimera.h has a def_rw in chimera_ext.cpp.

Text-level check: nanobind has no reflection over C++ members, so a field added
to chimera.h and forgotten here is otherwise silent drift. Needs no built module.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADER = REPO_ROOT / "src" / "chimera" / "chimera.h"
BINDINGS = REPO_ROOT / "bindings" / "chimera_ext.cpp"

# Intentionally unbound: the Tokenizer class takes a path + use_mmap directly.
UNBOUND_STRUCTS = {"TokenizeOptions"}

# One declaration per line: `type name;`, `type name = value;` or `type name{...};`.
# A method never matches: `(` follows its name.
FIELD_RE = re.compile(r"^\s*(?!return\b|using\b|typedef\b)[\w:<>,\s*&]+?\s(\w+)\s*(?:=[^;]*|\{[^;]*\})?;", re.M)


def struct_fields(text: str, struct: str) -> list[str]:
    start = text.index(f"struct {struct} {{")
    body = text[start : text.index("\n};", start)]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    return FIELD_RE.findall(body)


def option_structs(text: str) -> list[str]:
    return [s for s in re.findall(r"^struct (\w+Options) \{", text, re.M) if s not in UNBOUND_STRUCTS]


def test_parser_sees_known_fields():
    text = HEADER.read_text()
    fields = struct_fields(text, "SdOptions")
    for name in ("model", "auto_fit", "ref_images", "max_vram", "tokenizer"):
        assert name in fields
    assert "LlamaCommonOptions" in option_structs(text)


@pytest.mark.parametrize("struct", option_structs(HEADER.read_text()))
def test_every_field_is_bound(struct):
    bindings = BINDINGS.read_text()
    assert f"nb::class_<{struct}>" in bindings, f"{struct} is not bound; add it or list it in UNBOUND_STRUCTS"
    bound = set(re.findall(rf"&{struct}::(\w+)", bindings))
    missing = [f for f in struct_fields(HEADER.read_text(), struct) if f not in bound]
    assert missing == [], f"{struct} fields with no def_rw in chimera_ext.cpp: {missing}"
