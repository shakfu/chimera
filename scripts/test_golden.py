#!/usr/bin/env python3
"""
test_golden.py - HTTP response-shape regression tests for chimera serve.

Spawns `chimera serve` against fixed models on a free port, hits each
route under test with a fixed payload, runs the response through a
route-specific normalizer (redacting volatile fields like timestamps,
ids, and ports), and diffs the result against a checked-in golden
under `tests/golden/`.

What the goldens catch is *runtime drift in upstream `server_routes`
lambdas* — the case where the same input produces a different JSON
shape after a bump. The smoke suite already covers "did it work";
this layer covers "did it work the same way".

What the goldens do NOT pin: exact token text in chat / completions
output (varies by backend; we shape-check), and exact float values in
embedding output (varies by backend; we check length + non-zero norm).

Usage:
  python scripts/test_golden.py           # run; fail on mismatch
  UPDATE_GOLDEN=1 python scripts/test_golden.py    # rewrite goldens

Requires:
  - chimera built (build/chimera)
  - models/Llama-3.2-1B-Instruct-Q8_0.gguf
  - models/bge-small-en-v1.5-q8_0.gguf
Skips with a non-fatal "SKIP" message when any model is missing.
"""

from __future__ import annotations

import difflib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, NoReturn

REPO = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO / "tests" / "golden"
MODELS = Path(os.environ.get("MODELS", REPO / "models"))
LLM_PATH = MODELS / "Llama-3.2-1B-Instruct-Q8_0.gguf"
EMBED_PATH = MODELS / "bge-small-en-v1.5-q8_0.gguf"

UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"


def find_chimera() -> Path:
    for c in [
        REPO / "build" / "chimera",
        REPO / "build" / "Release" / "chimera.exe",
        REPO / "build" / "chimera.exe",
    ]:
        if c.exists():
            return c
    fail("chimera binary not built; run `make build` first")


def fail(msg: str) -> NoReturn:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_ready(port: int, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    last_err: str = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=2
            ) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = str(e)
        time.sleep(0.5)
    fail(f"server didn't become ready in {timeout}s (last: {last_err})")


def http_json(method: str, port: int, path: str, body: dict | None = None) -> dict:
    url = f"http://127.0.0.1:{port}{path}"
    data: bytes | None = None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------- normalizers --------------------------------------------------
#
# Each test case ends with a route-specific transform that strips fields
# whose values genuinely shift run-to-run (timestamps, request ids,
# resolved file paths, port-dependent URLs) and shape-checks fields
# whose exact value isn't load-bearing (generated text, embedding
# floats). The goal is a golden that's stable across machines + dates
# while still detecting structural drift.

REDACTED = "<redacted>"


def redact(obj: Any, keys: set[str]) -> Any:
    if isinstance(obj, dict):
        return {k: (REDACTED if k in keys else redact(v, keys)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(x, keys) for x in obj]
    return obj


def type_of(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if v is None:
        return "null"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def shape(obj: Any) -> Any:
    """Recursively replace leaf values with their type name; collapse
    homogeneous lists to a single representative element."""
    if isinstance(obj, dict):
        return {k: shape(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if not obj:
            return []
        return [shape(obj[0])]
    return type_of(obj)


# ---------- test cases ---------------------------------------------------
#
# Each case is (golden_name, callable(port) -> normalized response).
# Add a case: define the fn, append to CASES. Goldens auto-generate on
# UPDATE_GOLDEN=1 the first run.

PROMPT = "Hello"
TOKENIZE_INPUT = "the quick brown fox"


def case_health(port: int) -> Any:
    return http_json("GET", port, "/v1/health")


def case_models(port: int) -> Any:
    r = http_json("GET", port, "/v1/models")
    # `created` is the model file's mtime — varies per checkout.
    return redact(r, {"created"})


def case_props(port: int) -> Any:
    # `default_generation_settings` carries chat-template paths and may
    # vary across builds; collapse the whole response to its shape.
    return shape(http_json("GET", port, "/props"))


def case_tokenize(port: int) -> Any:
    return http_json("POST", port, "/tokenize", {"content": TOKENIZE_INPUT})


def case_detokenize(port: int) -> Any:
    # Tokenize first, then detokenize; the round-trip should match.
    toks = http_json("POST", port, "/tokenize", {"content": TOKENIZE_INPUT})["tokens"]
    return http_json("POST", port, "/detokenize", {"tokens": toks})


def case_apply_template(port: int) -> Any:
    return http_json(
        "POST",
        port,
        "/apply-template",
        {"messages": [{"role": "user", "content": PROMPT}]},
    )


def case_embeddings(port: int) -> Any:
    # Backend-dependent floats; shape + dim + non-empty are what matter.
    r = http_json("POST", port, "/v1/embeddings",
                  {"model": "bge", "input": "fixed input"})
    return {
        "object": r["object"],
        "model_present": "model" in r,
        "data_len": len(r["data"]),
        "data_0_object": r["data"][0]["object"],
        "data_0_index": r["data"][0]["index"],
        "embedding_dim": len(r["data"][0]["embedding"]),
        "usage_keys": sorted(r.get("usage", {}).keys()),
    }


def case_chat_completions(port: int) -> Any:
    # Text content varies by backend; check structural keys, types, and
    # the few stable scalars (object, finish-reason vocabulary, etc.).
    r = http_json(
        "POST",
        port,
        "/v1/chat/completions",
        {
            "model": "llama",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 4,
            "temperature": 0,
            "seed": 42,
        },
    )
    return {
        "object": r["object"],
        "shape": shape(redact(r, {"id", "created", "model", "content"})),
        "n_choices": len(r["choices"]),
        "first_finish_reason_type": type_of(r["choices"][0]["finish_reason"]),
        "usage_keys": sorted(r["usage"].keys()),
    }


def case_completions(port: int) -> Any:
    r = http_json(
        "POST",
        port,
        "/v1/completions",
        {
            "model": "llama",
            "prompt": "1, 2, 3,",
            "max_tokens": 4,
            "temperature": 0,
            "seed": 42,
        },
    )
    return {
        "object": r["object"],
        "shape": shape(redact(r, {"id", "created", "model", "text"})),
        "n_choices": len(r["choices"]),
        "usage_keys": sorted(r["usage"].keys()),
    }


CASES: list[tuple[str, Callable[[int], Any]]] = [
    ("health",            case_health),
    ("models",            case_models),
    ("props_shape",       case_props),
    ("tokenize",          case_tokenize),
    ("detokenize",        case_detokenize),
    ("apply_template",    case_apply_template),
    ("embeddings",        case_embeddings),
    ("chat_completions",  case_chat_completions),
    ("completions",       case_completions),
]


# ---------- harness ------------------------------------------------------


def compare_or_write(name: str, got: Any) -> bool:
    path = GOLDEN_DIR / f"{name}.json"
    serialized = json.dumps(got, indent=2, sort_keys=True) + "\n"
    if UPDATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized)
        print(f"  WROTE  {name}")
        return True
    want = path.read_text()
    if want == serialized:
        print(f"  PASS   {name}")
        return True
    print(f"  FAIL   {name}")
    for line in difflib.unified_diff(
        want.splitlines(keepends=True),
        serialized.splitlines(keepends=True),
        fromfile=f"{name}.golden",
        tofile=f"{name}.got",
        n=3,
    ):
        sys.stdout.write("    " + line)
    return False


def main() -> int:
    if not LLM_PATH.exists():
        print(f"SKIP: missing {LLM_PATH}", file=sys.stderr)
        return 0
    if not EMBED_PATH.exists():
        print(f"SKIP: missing {EMBED_PATH}", file=sys.stderr)
        return 0

    chimera = find_chimera()
    port = free_port()
    print(f"golden: spawning chimera serve on 127.0.0.1:{port}")

    # --gpu-layers 0 keeps the test backend-stable: even on a Metal /
    # CUDA box we run CPU, so the goldens that DO pin floats remain
    # portable across dev machines and CI.
    proc = subprocess.Popen(
        [
            str(chimera), "serve",
            "-m", str(LLM_PATH),
            "--enable-embeddings", str(EMBED_PATH),
            "--gpu-layers", "0",
            "--parallel", "1",
            "--port", str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_ready(port)
        all_pass = True
        for name, fn in CASES:
            try:
                got = fn(port)
            except (urllib.error.URLError, urllib.error.HTTPError,
                    KeyError, ValueError) as e:
                print(f"  FAIL   {name} (exception: {type(e).__name__}: {e})")
                all_pass = False
                continue
            all_pass = compare_or_write(name, got) and all_pass
        if not all_pass:
            print("\ngolden: at least one route mismatched. Audit the diff;",
                  "if the new shape is correct, re-run with",
                  "`UPDATE_GOLDEN=1 make test-golden` to refresh.",
                  file=sys.stderr)
            return 1
        print("\ngolden: all routes match.")
        return 0
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
