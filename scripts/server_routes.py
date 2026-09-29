"""Route parity between llama-server and `chimera serve`.

Used by `manage.py bump_check` (against an upstream ref) and by
`scripts/test.py` (against the pinned clone under build/llama.cpp).
"""

from __future__ import annotations

import re
from pathlib import Path

# Matches `ctx_http.get ("/path", ...)`; upstream pads the method with spaces.
_ROUTE_RE = re.compile(r'ctx_http\.(get|post|put|patch|del)\s*\(\s*"(/[^"]*)"')

# llama-server routes chimera deliberately does not bind, with the reason.
# See docs/dev/server.md section 4.4.
UNBOUND = {
    ("post", "/props"): "CLI is the config; GET /props is bound",
    ("get", "/models/sse"): "router mode (docs/dev/server-router-mode.md)",
    ("post", "/models"): "router mode",
    ("post", "/models/load"): "router mode",
    ("post", "/models/unload"): "router mode",
    ("del", "/models"): "router mode",
}


def extract(text: str) -> set[tuple[str, str]]:
    """Return the (method, path) pairs registered in C++ source `text`."""
    return set(_ROUTE_RE.findall(text))


def chimera_routes(repo: Path) -> set[tuple[str, str]]:
    """Return the routes registered across src/chimera/*.cpp."""
    out: set[tuple[str, str]] = set()
    for f in sorted((repo / "src" / "chimera").glob("*.cpp")):
        out |= extract(f.read_text(encoding="utf-8"))
    return out


def missing(upstream_server_cpp: str, repo: Path) -> list[tuple[str, str]]:
    """Return upstream routes chimera neither binds nor lists in UNBOUND."""
    return sorted(extract(upstream_server_cpp) - chimera_routes(repo) - UNBOUND.keys())
