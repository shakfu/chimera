#!/usr/bin/env python3
"""
Smoke + end-to-end tests for chimera. Replaces the earlier scripts/test.sh.

Why Python instead of Bash: cross-platform (Windows included), real argparse,
per-test timing, easier multipart/JSON handling, and a single context manager
collapses the half-dozen spawn/poll/teardown blocks in the previous bash.

Smoke tests always run (no model files needed): exercise --version, --help,
and per-subcommand --help. They catch link-time regressions and CLI11
wiring breakage.

End-to-end tests run only when the matching model file is present under
models/. Missing models are reported as SKIP, not FAIL, so the test target
stays green on a fresh checkout.

A second tier of opt-in tests fires when adapter / aux-model fixtures are
pointed at via env vars (CHIMERA_TEST_LORA, CHIMERA_TEST_CONTROLNET +
CHIMERA_TEST_CONTROL_IMAGE, CHIMERA_TEST_PHOTOMAKER + CHIMERA_TEST_PM_ID_DIR).
See docs/dev/maintenance.md for the full table.

Usage:
  scripts/test.py                # smoke + e2e (where models available)
  scripts/test.py --smoke        # smoke only
  scripts/test.py --filter PAT   # run only tests whose name matches the regex
  scripts/test.py --verbose      # stream subprocess output as it happens
  scripts/test.py --no-color     # disable ANSI escapes
  scripts/test.py --no-timing    # don't print the slowest-first table

Set CHIMERA=/path/to/chimera to test a non-default binary.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional


# ----------------------------------------------------------------------------
# Configuration: paths, fixtures, env vars.
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS    = REPO_ROOT / "models"

# Model fixtures bundled under models/. We tolerate missing files: any test
# that depends on a missing model is SKIPped rather than FAILed.
GEN_MODEL         = MODELS / "Llama-3.2-1B-Instruct-Q8_0.gguf"
EMBED_CANDIDATES  = ["bge-small-en-v1.5-q8_0.gguf", "gte-small-q8_0.gguf"]
WHISPER_MODEL     = MODELS / "ggml-base.en.bin"
WHISPER_WAV       = REPO_ROOT / "build" / "whisper.cpp" / "samples" / "jfk.wav"
SD_CANDIDATES     = ["sd_xl_turbo_1.0.q8_0.gguf",
                     "v1-5-pruned-emaonly.q8_0.gguf",
                     "z_image_turbo-Q6_K.gguf"]
MTMD_TEXT_MODEL   = MODELS / "gemma-4-E4B-it-Q4_K_M.gguf"
MTMD_MMPROJ       = MODELS / "mmproj-gemma-4-E4B-it-BF16.gguf"


def first_existing(model_dir: Path, names: list[str]) -> Optional[Path]:
    """Return the first candidate filename that exists, or None."""
    for name in names:
        p = model_dir / name
        if p.is_file():
            return p
    return None


EMBED_MODEL: Optional[Path] = first_existing(MODELS, EMBED_CANDIDATES)
SD_MODEL:    Optional[Path] = first_existing(MODELS, SD_CANDIDATES)


# ----------------------------------------------------------------------------
# Output + result tracking.
# ----------------------------------------------------------------------------

class Status:
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class Outcome:
    name: str
    status: str
    duration: float           # wall-clock seconds
    detail: str = ""          # extra context (FAIL diagnostics, SKIP reason)


_color_enabled = False  # set by main() once argparse has decided


def _ansi(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _color_enabled else s


def tag(status: str) -> str:
    if status == Status.PASS: return _ansi("32;1", "PASS")
    if status == Status.FAIL: return _ansi("31;1", "FAIL")
    if status == Status.SKIP: return _ansi("33;1", "SKIP")
    return status


class Recorder:
    """Collects Outcome records, prints them live, and reports the summary.

    Keeping the printing inside the recorder (rather than in each test
    function) means every test reaches the terminal through the same
    formatting path, which makes --no-color / --verbose toggles one switch.
    """

    def __init__(self, *, name_filter: Optional[re.Pattern] = None) -> None:
        self.outcomes: list[Outcome] = []
        self._filter = name_filter

    def matches(self, name: str) -> bool:
        return self._filter is None or self._filter.search(name) is not None

    def record(self, name: str, status: str, duration: float, detail: str = "") -> Outcome:
        outcome = Outcome(name=name, status=status, duration=duration, detail=detail)
        self.outcomes.append(outcome)
        self._print(outcome)
        return outcome

    def _print(self, o: Outcome) -> None:
        # Two-space indent matches the bash output convention so existing
        # log scrapers (and human muscle memory) keep working.
        print(f"  {tag(o.status)}  {o.name}")
        if o.detail:
            for line in o.detail.splitlines():
                print(f"        {line}")

    # Convenience wrappers — they let test functions write
    # `rec.pass_("...")` instead of `rec.record(..., Status.PASS, ...)`.
    def pass_(self, name: str, duration: float, detail: str = "") -> None:
        self.record(name, Status.PASS, duration, detail)

    def fail(self, name: str, duration: float, detail: str = "") -> None:
        self.record(name, Status.FAIL, duration, detail)

    def skip(self, name: str, reason: str) -> None:
        # Honor --filter for SKIPs too — otherwise `--filter X` floods the
        # output with SKIPs for unrelated unavailable fixtures.
        if not self.matches(name):
            return
        self.record(name, Status.SKIP, 0.0, reason)

    def summary(self, show_timing: bool) -> int:
        passed = sum(1 for o in self.outcomes if o.status == Status.PASS)
        failed = sum(1 for o in self.outcomes if o.status == Status.FAIL)
        skipped = sum(1 for o in self.outcomes if o.status == Status.SKIP)

        print()
        pass_color = "32" if passed else "32"
        fail_color = "32" if not failed else "31"
        print(
            f"summary: {_ansi(pass_color, f'pass={passed}')} "
            f"{_ansi(fail_color, f'fail={failed}')} "
            f"{_ansi('33', f'skip={skipped}')}"
        )

        if failed:
            failed_names = [o.name for o in self.outcomes if o.status == Status.FAIL]
            print(_ansi("31", "failed:"), "; ".join(failed_names))

        if show_timing:
            ran = [o for o in self.outcomes if o.status != Status.SKIP]
            ran.sort(key=lambda o: o.duration, reverse=True)
            if ran:
                print()
                print("timing (slowest first):")
                # Pad name column to the longest name for tidy alignment.
                width = max(len(o.name) for o in ran)
                for o in ran:
                    print(f"  {o.duration:7.2f}s  {o.name:<{width}}  [{o.status}]")

        return failed


# ----------------------------------------------------------------------------
# Subprocess helpers.
# ----------------------------------------------------------------------------

_verbose = False  # toggled by main()


# Sentinel returncode for "subprocess hit the timeout." Picked far outside
# any plausible real exit code (Unix exit-codes are 0–255 unsigned, signals
# fall in -1..-64 when reported by subprocess). Distinguishing this from
# "process actually exited -1" matters because the latter is a real
# program signal (SIGHUP) and the former is "we gave up waiting" — which
# usually means the timeout was too tight, not that the binary misbehaved.
TIMED_OUT = -1000


def run_silent(cmd: list, *, timeout: float = 600, stdin: Optional[bytes] = None,
               env: Optional[dict] = None) -> int:
    """Run `cmd`, suppress output (unless --verbose), return exit code.

    Returns TIMED_OUT (a sentinel int that no real process produces) when
    the timeout fires, so callers can surface a clearer failure message
    than the previous "got -1" — that diagnostic conflated timeouts with
    SIGHUP-killed processes.

    Used for the simple "exit-code-only" tests like `chimera --version` and
    `chimera gen ...`. If `--verbose` was passed we mirror stdout/stderr to
    the terminal so failures are easier to diagnose without re-running.
    """
    stdout = None if _verbose else subprocess.DEVNULL
    stderr = None if _verbose else subprocess.DEVNULL
    try:
        proc = subprocess.run(
            cmd, stdout=stdout, stderr=stderr, input=stdin,
            timeout=timeout, env=env,
        )
        return proc.returncode
    except subprocess.TimeoutExpired:
        return TIMED_OUT


def _check_rc(t: "TimedTest", rc: int, want: int, timeout: float) -> None:
    """Compare a returncode against the expected value, surfacing the
    timeout-vs-mismatch distinction in the failure message. Used by the
    smoke tests where conflating "took too long" with "wrong exit code"
    sends maintainers down the wrong diagnostic path."""
    if rc == TIMED_OUT:
        t.fail(f"timed out after {timeout:g}s (expected exit code {want})")
    elif rc != want:
        t.fail(f"want {want}, got {rc}")


def run_capture(cmd: list, *, timeout: float = 600, stdin: Optional[bytes] = None,
                env: Optional[dict] = None,
                combined: bool = False) -> tuple[int, str, str]:
    """Run `cmd`, capture stdout + stderr, return (rc, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if combined else subprocess.PIPE,
            input=stdin, timeout=timeout, env=env,
        )
        out = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
        err = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def free_port() -> int:
    """Bind to an ephemeral port to discover a free one; release it before
    handing the number back. Same trick the bash version used via Python."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ----------------------------------------------------------------------------
# HTTP helpers (stdlib only — no `requests` dependency).
# ----------------------------------------------------------------------------

@dataclass
class HttpResponse:
    status: int
    body:   str
    headers: dict      # case-insensitive; built from the response

    def json(self) -> object:
        return json.loads(self.body)


def _do_request(req: urllib.request.Request, timeout: float) -> HttpResponse:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
            headers = {k.lower(): v for k, v in r.headers.items()}
            return HttpResponse(status=r.status, body=body, headers=headers)
    except urllib.error.HTTPError as e:
        # 4xx / 5xx — still return the response so the test can inspect it.
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        headers = {k.lower(): v for k, v in (e.headers.items() if e.headers else [])}
        return HttpResponse(status=e.code, body=body, headers=headers)
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        return HttpResponse(status=0, body=str(e), headers={})


def http_get(url: str, *, timeout: float = 10) -> HttpResponse:
    return _do_request(urllib.request.Request(url, method="GET"), timeout)


def http_post_json(url: str, body: object, *, headers: Optional[dict] = None,
                   timeout: float = 30) -> HttpResponse:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    return _do_request(req, timeout)


def http_post_multipart(url: str, fields: Optional[dict] = None,
                        files: Optional[dict] = None, *,
                        timeout: float = 120) -> HttpResponse:
    """Build a multipart/form-data body by hand.

    `fields` is name -> str. `files` is name -> (filename, bytes) tuple or
    a Path (in which case we read the bytes and derive the filename).
    """
    boundary = "----chimeraTest" + uuid.uuid4().hex
    buf = io.BytesIO()
    nl = b"\r\n"

    def write_field(name: str, value: str) -> None:
        buf.write(f"--{boundary}".encode()); buf.write(nl)
        buf.write(f'Content-Disposition: form-data; name="{name}"'.encode()); buf.write(nl)
        buf.write(nl)
        buf.write(value.encode("utf-8")); buf.write(nl)

    def write_file(name: str, filename: str, data: bytes) -> None:
        buf.write(f"--{boundary}".encode()); buf.write(nl)
        buf.write(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'
            .encode()
        ); buf.write(nl)
        buf.write(b"Content-Type: application/octet-stream"); buf.write(nl)
        buf.write(nl)
        buf.write(data); buf.write(nl)

    for k, v in (fields or {}).items():
        write_field(k, str(v))
    for k, f in (files or {}).items():
        if isinstance(f, (str, Path)):
            p = Path(f)
            write_file(k, p.name, p.read_bytes())
        else:
            filename, data = f
            write_file(k, filename, data)
    buf.write(f"--{boundary}--".encode()); buf.write(nl)

    req = urllib.request.Request(url, data=buf.getvalue(), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    return _do_request(req, timeout)


# ----------------------------------------------------------------------------
# Chimera serve context manager — collapses the 5+ spawn-poll-teardown
# blocks the bash version had into one helper.
# ----------------------------------------------------------------------------

@dataclass
class ServeHandle:
    base_url: str
    port:     int
    log_path: Path


@contextlib.contextmanager
def chimera_serve(chimera: Path, extra_args: list, *,
                  wait_for_sd: bool = False,
                  readiness_path: str = "/health",
                  startup_timeout: float = 120.0) -> Iterator[ServeHandle]:
    """Spawn chimera serve, wait until it answers requests, yield the URL.

    `wait_for_sd=True` polls /v1/images/generations instead of /health,
    because the SD context loads in-band after /health flips to 200 and
    image POSTs return 503 ("Loading model") until the SD context is ready.

    Teardown is graceful (terminate + 5s) then forceful (kill). The
    server's log file is removed unless the test failed and --verbose was
    set (in which case we print the tail before cleanup).
    """
    port = free_port()
    log_fd, log_path_str = tempfile.mkstemp(prefix="chimera-test-serve-", suffix=".log")
    os.close(log_fd)
    log_path = Path(log_path_str)
    cmd = [str(chimera), "serve",
           "--host", "127.0.0.1", "--port", str(port),
           "--gpu-layers", "0", *extra_args]

    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"

    try:
        # Two readiness strategies:
        #   - /health: LLM is loaded. Used by every test that doesn't need SD.
        #   - POST /v1/images/generations with `{}`: SD is loaded. Returns 503
        #     "Loading model" while SD is still initializing; flips to 4xx
        #     (e.g. 400 "prompt is required") once the handler is live.
        deadline = time.monotonic() + startup_timeout
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"chimera serve exited before readiness "
                    f"(rc={proc.returncode}, log tail: {_tail(log_path, 5)!r})"
                )
            try:
                if wait_for_sd:
                    r = http_post_json(f"{base}/v1/images/generations", {}, timeout=2)
                    if r.status in (200, 400, 405, 415, 422):
                        ready = True
                        break
                else:
                    r = http_get(f"{base}{readiness_path}", timeout=2)
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(0.5)

        if not ready:
            raise RuntimeError(
                f"chimera serve didn't become ready within {startup_timeout}s "
                f"(log tail: {_tail(log_path, 10)!r})"
            )

        yield ServeHandle(base_url=base, port=port, log_path=log_path)

    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                with contextlib.suppress(Exception):
                    proc.wait(timeout=5.0)
        try:
            log_path.unlink(missing_ok=True)
        except Exception:
            pass


def _tail(path: Path, n_lines: int) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read().decode("utf-8", errors="replace")
        return "\n".join(data.splitlines()[-n_lines:])
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# A tiny tempfile helper so test blocks can request a scratch path without
# leaving files behind on early returns.
# ----------------------------------------------------------------------------

@contextlib.contextmanager
def scratch_dir(prefix: str = "chimera-test-") -> Iterator[Path]:
    d = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@contextlib.contextmanager
def scratch_file(prefix: str = "chimera-test-", suffix: str = "") -> Iterator[Path]:
    fd, path_str = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    path = Path(path_str)
    try:
        yield path
    finally:
        with contextlib.suppress(Exception):
            path.unlink()
        # Also clean SQLite WAL/SHM sidecar files if they were created.
        for sidecar in (path.with_name(path.name + "-wal"),
                        path.with_name(path.name + "-shm")):
            with contextlib.suppress(Exception):
                sidecar.unlink()


# ----------------------------------------------------------------------------
# Test convenience: a decorator-like `timed` context manager that captures
# wall-clock and emits the right PASS/FAIL record.
# ----------------------------------------------------------------------------

class TimedTest:
    """Bracket a test block; on `.ok()` records PASS, on `.fail(...)`
    records FAIL with diagnostic detail. If the block falls through
    without calling either, treat that as PASS (some tests are pure
    "expected to not throw")."""

    def __init__(self, rec: Recorder, name: str) -> None:
        self.rec = rec
        self.name = name
        self.t0: float = 0.0
        self._resolved = False

    def __enter__(self) -> "TimedTest":
        self.t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.fail(f"unexpected exception: {exc_type.__name__}: {exc}")
            return True  # swallow — one bad test shouldn't sink the runner
        if not self._resolved:
            self.ok()
        return False

    def ok(self, detail: str = "") -> None:
        self._resolved = True
        self.rec.pass_(self.name, time.monotonic() - self.t0, detail)

    def fail(self, detail: str) -> None:
        self._resolved = True
        self.rec.fail(self.name, time.monotonic() - self.t0, detail)


def maybe(rec: Recorder, name: str) -> contextlib.AbstractContextManager:
    """Filter-aware TimedTest. If the test name doesn't match --filter we
    return a no-op context manager so the test body is skipped silently."""
    if not rec.matches(name):
        # Nothing recorded — filtered-out tests are invisible by design.
        return contextlib.nullcontext(_FilteredTest())
    return TimedTest(rec, name)


class _FilteredTest:
    """Stand-in returned when --filter excludes a test. Calls to ok()/fail()
    are no-ops so test bodies can keep their structure unchanged."""

    def ok(self, detail: str = "") -> None: pass
    def fail(self, detail: str) -> None:    pass


def skip_if_unavailable(rec: Recorder, name: str, condition: bool, reason: str) -> bool:
    """Helper for the `if model_present else skip` pattern. Returns True if
    the test should proceed; False if it was SKIPped (and recorded)."""
    if not rec.matches(name):
        return False
    if condition:
        return True
    rec.skip(name, reason)
    return False


# ----------------------------------------------------------------------------
# Binary location.
# ----------------------------------------------------------------------------

def locate_chimera() -> Optional[Path]:
    """Find the built chimera binary. Same candidate list as the bash
    locate_chimera() so existing setups keep working."""
    candidates = [
        REPO_ROOT / "build" / "chimera",
        REPO_ROOT / "build" / "chimera.exe",
        REPO_ROOT / "build" / "Release" / "chimera.exe",
        REPO_ROOT / "build" / "Release" / "chimera",
        REPO_ROOT / "build" / "Debug" / "chimera.exe",
        REPO_ROOT / "build" / "Debug" / "chimera",
    ]
    env_override = os.environ.get("CHIMERA")
    if env_override:
        candidates.insert(0, Path(env_override))
    for c in candidates:
        if c.is_file():
            return c
    return None


# ----------------------------------------------------------------------------
# Test sections. Each section is a function that takes the recorder and
# whatever context it needs (chimera path, model paths, opts).
#
# The section structure mirrors the bash file so anyone diffing the two
# during the migration can find each block by name.
# ----------------------------------------------------------------------------

# ============================================================================
# Smoke tests
# ============================================================================

def smoke_tests(rec: Recorder, chimera: Path) -> None:
    """--version + --help variants. No model files needed; runs in <1s total."""
    for label, args in [
        ("chimera --version", ["--version"]),
        ("chimera --help",    ["--help"]),
        ("gen --help",        ["gen", "--help"]),
        ("chat --help",       ["chat", "--help"]),
        ("tokenize --help",   ["tokenize", "--help"]),
        ("embed --help",      ["embed", "--help"]),
        ("whisper --help",    ["whisper", "--help"]),
        ("sd --help",         ["sd", "--help"]),
    ]:
        with maybe(rec, label) as t:
            rc = run_silent([str(chimera), *args], timeout=15)
            if rc != 0:
                t.fail(f"exit code {rc}")

    # `gen` without -m must fail at CLI parse. The bash version asserted
    # "non-zero exit" without pinning the exact code; the same loose check
    # here protects against the CLI11 wiring silently accepting the call.
    # CLI11 short-circuits before backend init, so 15s is plenty.
    with maybe(rec, "gen without -m exits non-zero") as t:
        rc = run_silent([str(chimera), "gen", "-p", "hi"], timeout=15)
        if rc == TIMED_OUT:
            t.fail("timed out after 15s (expected non-zero exit)")
        elif rc == 0:
            t.fail("expected non-zero exit, got 0")

    # Structured exit codes: 2 = BadInput. Only runs when the gen model is
    # actually present — without -m the call would fail earlier (above).
    # 60s timeout: this path runs backend init before the no-prompt check
    # fires, and CI runners without real GPU access can take several
    # seconds to time out the Metal/CUDA probe before falling back to CPU.
    if GEN_MODEL.is_file():
        with maybe(rec, "gen without prompt exits 2 (BadInput)") as t:
            rc = run_silent([str(chimera), "gen", "-m", str(GEN_MODEL)], timeout=60)
            _check_rc(t, rc, want=2, timeout=60)

    # Structured exit codes: 3 = Load (model file not found). The binary
    # initializes the backend BEFORE attempting to open the model, so on
    # CI macos-metal runners (no real GPU; Metal probes time out
    # ~5-10s before CPU fallback) this needs more than the CLI-parse
    # budget. 60s is comfortably above the observed worst case.
    with maybe(rec, "gen with missing model exits 3 (Load)") as t:
        rc = run_silent(
            [str(chimera), "gen", "-m", "/no/such/model.gguf", "-p", "hi"],
            timeout=60,
        )
        _check_rc(t, rc, want=3, timeout=60)


# ============================================================================
# End-to-end: gen + tokenize
# ============================================================================

def e2e_gen_tests(rec: Recorder, chimera: Path) -> None:
    if not GEN_MODEL.is_file():
        rec.skip("gen Llama-3.2-1B", f"missing {GEN_MODEL}")
        rec.skip("tokenize Llama-3.2-1B", f"missing {GEN_MODEL}")
        rec.skip("gen --prompt-file - (stdin)", f"missing {GEN_MODEL}")
        return

    with maybe(rec, "gen Llama-3.2-1B") as t:
        rc = run_silent(
            [str(chimera), "gen", "-m", str(GEN_MODEL), "-p", "Hello", "-n", "8"],
            timeout=120,
        )
        if rc != 0:
            t.fail(f"exit code {rc}")

    with maybe(rec, "tokenize Llama-3.2-1B") as t:
        rc = run_silent(
            [str(chimera), "tokenize", "-m", str(GEN_MODEL), "-p", "hello world"],
            timeout=30,
        )
        if rc != 0:
            t.fail(f"exit code {rc}")

    with maybe(rec, "gen --prompt-file - (stdin)") as t:
        rc = run_silent(
            [str(chimera), "gen", "-m", str(GEN_MODEL), "-f", "-", "-n", "4"],
            timeout=60, stdin=b"Hi\n",
        )
        if rc != 0:
            t.fail(f"exit code {rc}")


# ============================================================================
# End-to-end: embedding + vector store + search
# ============================================================================

def e2e_embed_tests(rec: Recorder, chimera: Path) -> None:
    if EMBED_MODEL is None:
        rec.skip("embed",                            f"no embedding model under {MODELS}/")
        rec.skip("embedding cache (bit-identical round trip)", f"no embedding model under {MODELS}/")
        rec.skip("vector store",                     f"no embedding model under {MODELS}/")
        rec.skip("search --mode lexical",            f"no embedding model under {MODELS}/")
        rec.skip("search --mode hybrid",             f"no embedding model under {MODELS}/")
        rec.skip("search default mode is hybrid",    f"no embedding model under {MODELS}/")
        rec.skip("search --mode <bogus> exits with BadInput", f"no embedding model under {MODELS}/")
        return

    embed_label = f"embed {EMBED_MODEL.name}"
    with maybe(rec, embed_label) as t:
        rc = run_silent(
            [str(chimera), "embed", "-m", str(EMBED_MODEL), "-p", "a quick brown fox"],
            timeout=60,
        )
        if rc != 0:
            t.fail(f"exit code {rc}")

    # Embedding cache: two calls with the same input must produce byte-
    # identical output. The cache flag enables read-through caching; the
    # second call should be a hit, but we can't time-assert that without
    # extra plumbing — bit-equality is the load-bearing invariant.
    with maybe(rec, "embedding cache (bit-identical round trip)") as t:
        with scratch_file(suffix=".db") as cache_db, \
             scratch_file() as out1, \
             scratch_file() as out2:
            cmd = [str(chimera), "embed", "-m", str(EMBED_MODEL),
                   "-p", "cache me",
                   "--cache-embeddings", "--cache-db", str(cache_db)]
            with open(out1, "wb") as fh:
                p1 = subprocess.run(cmd, stdout=fh, stderr=subprocess.DEVNULL, timeout=60)
            with open(out2, "wb") as fh:
                p2 = subprocess.run(cmd, stdout=fh, stderr=subprocess.DEVNULL, timeout=60)
            if p1.returncode != 0 or p2.returncode != 0:
                t.fail(f"embed exit codes: {p1.returncode}, {p2.returncode}")
            elif out1.read_bytes() != out2.read_bytes():
                t.fail("cached output differs from first call")

    # Vector store smoke: create, ingest, top-1 hit on a query that should
    # unambiguously prefer the WAV passage.
    with maybe(rec, "vector store (rag_test top hit on 'audio file formats')") as t:
        with scratch_file(suffix=".db") as vec_db, \
             scratch_file(suffix=".txt") as vec_doc:
            vec_doc.write_text(
                "The chimera serve subcommand exposes an OpenAI-compatible HTTP server.\n\n"
                "The whisper.cpp wrapper accepts WAV audio files only in the first cut.\n\n"
                "SQLite plus sqlite-vec is embedded for RAG and persistent chat history.\n"
            )
            env = {**os.environ, "CHIMERA_DB": str(vec_db)}
            rc_c = run_silent([str(chimera), "index", "create", "-n", "rag_test",
                               "-e", str(EMBED_MODEL),
                               "--chunk-tokens", "128", "--chunk-overlap", "16"],
                              env=env, timeout=60)
            rc_i = run_silent([str(chimera), "index", "ingest", "-n", "rag_test",
                               "-f", str(vec_doc)], env=env, timeout=60)
            if rc_c != 0 or rc_i != 0:
                t.fail(f"create/ingest exit codes: {rc_c}, {rc_i}")
            else:
                rc, hit, _ = run_capture(
                    [str(chimera), "search", "-n", "rag_test",
                     "-q", "audio file formats", "-k", "1"],
                    env=env, timeout=60,
                )
                if "wav audio files" not in hit.lower():
                    t.fail(f"top hit didn't mention wav: {hit!r}")

    # Hybrid / lexical search: deliberately rare proper-noun-like token so
    # the lexical leg has something semantic search wouldn't surface.
    with scratch_file(suffix=".db") as hyb_db, \
         scratch_file(suffix=".txt") as hyb_doc:
        hyb_doc.write_text(
            "Audio file formats commonly used in machine learning include WAV, "
            "FLAC, and MP3. WAV is uncompressed and preferred for training.\n\n"
            "Image file formats vary widely. PNG is lossless. JPEG is lossy "
            "and not recommended for masks.\n\n"
            "The xyzzyfrobnoc protocol is a fictitious wire format invented "
            "for this regression test. It carries no real semantic content "
            "but is keyword-unique within this corpus.\n\n"
            "Photosynthesis converts light energy into chemical energy in plants.\n"
        )
        env = {**os.environ, "CHIMERA_DB": str(hyb_db)}
        rc_c = run_silent([str(chimera), "index", "create", "-n", "hyb_test",
                           "-e", str(EMBED_MODEL),
                           "--chunk-tokens", "96", "--chunk-overlap", "12"],
                          env=env, timeout=60)
        rc_i = run_silent([str(chimera), "index", "ingest", "-n", "hyb_test",
                           "-f", str(hyb_doc)], env=env, timeout=60)
        setup_ok = (rc_c == 0 and rc_i == 0)

        with maybe(rec, "search --mode lexical (BM25 hits rare token)") as t:
            if not setup_ok:
                t.fail(f"hybrid setup failed (create={rc_c}, ingest={rc_i})")
            else:
                _, lex, _ = run_capture(
                    [str(chimera), "search", "-n", "hyb_test",
                     "-q", "xyzzyfrobnoc", "-k", "1", "--mode", "lexical"],
                    env=env, timeout=60,
                )
                if "xyzzyfrobnoc" not in lex.lower():
                    t.fail(f"lexical leg missed token: {lex!r}")

        with maybe(rec, "search --mode hybrid (RRF merge + rare-token recall)") as t:
            if not setup_ok:
                t.fail("hybrid setup failed")
            else:
                _, hyb, _ = run_capture(
                    [str(chimera), "search", "-n", "hyb_test",
                     "-q", "xyzzyfrobnoc wire format", "-k", "3", "--mode", "hybrid"],
                    env=env, timeout=60,
                )
                if "rrf=" not in hyb or "xyzzyfrobnoc" not in hyb.lower():
                    t.fail(f"hybrid output missing rrf=/token: {hyb!r}")

        with maybe(rec, "search default mode is hybrid") as t:
            if not setup_ok:
                t.fail("hybrid setup failed")
            else:
                _, default, _ = run_capture(
                    [str(chimera), "search", "-n", "hyb_test",
                     "-q", "xyzzyfrobnoc wire format", "-k", "3"],
                    env=env, timeout=60,
                )
                if "rrf=" not in default:
                    t.fail(f"default-mode output missing rrf=: {default!r}")

        with maybe(rec, "search --mode <bogus> exits with BadInput") as t:
            if not setup_ok:
                t.fail("hybrid setup failed")
            else:
                # combined=True merges stderr into stdout (subprocess.STDOUT),
                # so the relevant string lives in `out`, not `err`. The CLI11
                # validation message can go to either stream depending on
                # where the parser decided to print it, so we want the union.
                rc, out, _ = run_capture(
                    [str(chimera), "search", "-n", "hyb_test",
                     "-q", "anything", "--mode", "nonsense"],
                    env=env, timeout=30, combined=True,
                )
                if "invalid --mode" not in out.lower():
                    t.fail(f"validation failed (rc={rc}, output: {out!r})")


# ============================================================================
# End-to-end: whisper CLI
# ============================================================================

def e2e_whisper_cli_test(rec: Recorder, chimera: Path) -> None:
    if not WHISPER_MODEL.is_file():
        rec.skip("whisper jfk.wav", f"missing {WHISPER_MODEL}")
        return
    if not WHISPER_WAV.is_file():
        rec.skip("whisper jfk.wav", f"missing {WHISPER_WAV} (run 'make deps' to fetch)")
        return
    with maybe(rec, "whisper jfk.wav") as t:
        rc = run_silent(
            [str(chimera), "whisper", "-m", str(WHISPER_MODEL), "-i", str(WHISPER_WAV)],
            timeout=120,
        )
        if rc != 0:
            t.fail(f"exit code {rc}")


# ============================================================================
# End-to-end: SD CLI (txt2img, img2img, plus fixture-gated --lora /
# --control-net / --photo-maker).
# ============================================================================

def e2e_sd_cli_tests(rec: Recorder, chimera: Path) -> None:
    sd_label = f"sd {SD_MODEL.name}" if SD_MODEL else "sd"
    if SD_MODEL is None:
        rec.skip(sd_label, f"no diffusion model under {MODELS}/")
        rec.skip("sd img2img round-trip", f"no diffusion model under {MODELS}/")
        # The fixture-gated tests also need an SD base model; SKIP them
        # all with the same reason so the absent-base case is uniform.
        for label in ("sd --lora", "sd --control-net", "sd --photo-maker"):
            rec.skip(label, f"no diffusion model under {MODELS}/")
        return

    # We run SD twice (txt2img + img2img round-trip) instead of reusing the
    # first image as the img2img init. Costs ~10s more than the bash version
    # but keeps the two tests independent — a failure in one no longer
    # cascades into a SKIP of the other.
    with maybe(rec, sd_label) as t:
        with scratch_file(suffix=".png") as out:
            rc = run_silent(
                [str(chimera), "sd", "-m", str(SD_MODEL), "-p", "a red cube",
                 "-o", str(out), "-W", "256", "-H", "256", "-s", "2"],
                timeout=300,
            )
            if rc != 0 or not out.is_file() or out.stat().st_size == 0:
                t.fail(f"rc={rc}, size={out.stat().st_size if out.exists() else 0}")

    with maybe(rec, "sd img2img round-trip") as t:
        with scratch_file(suffix=".png") as src, \
             scratch_file(suffix=".png") as dst:
            rc1 = run_silent(
                [str(chimera), "sd", "-m", str(SD_MODEL), "-p", "a red cube",
                 "-o", str(src), "-W", "256", "-H", "256", "-s", "2"],
                timeout=300,
            )
            if rc1 != 0 or not src.is_file() or src.stat().st_size == 0:
                t.fail("could not synthesize input image for img2img")
            else:
                rc2 = run_silent(
                    [str(chimera), "sd", "-m", str(SD_MODEL), "-p", "a blue cube",
                     "--init-image", str(src), "--strength", "0.6",
                     "-o", str(dst), "-W", "256", "-H", "256", "-s", "2"],
                    timeout=300,
                )
                if rc2 != 0 or not dst.is_file() or dst.stat().st_size == 0:
                    t.fail(f"img2img rc={rc2}, size={dst.stat().st_size if dst.exists() else 0}")

    # ---- Fixture-gated: --lora ----
    lora = os.environ.get("CHIMERA_TEST_LORA")
    if not lora:
        rec.skip("sd --lora", "set CHIMERA_TEST_LORA=<path/to/lora.safetensors>")
    elif not Path(lora).is_file():
        rec.fail("sd --lora", 0.0, f"CHIMERA_TEST_LORA={lora} not found")
    else:
        with maybe(rec, "sd --lora (CHIMERA_TEST_LORA fixture)") as t:
            with scratch_file(suffix=".png") as out:
                rc = run_silent(
                    [str(chimera), "sd", "-m", str(SD_MODEL), "-p", "a red cube",
                     "--lora", lora,
                     "-W", "256", "-H", "256", "-s", "2", "-o", str(out)],
                    timeout=600,
                )
                if rc != 0 or out.stat().st_size == 0:
                    t.fail(f"rc={rc}")

    # ---- Fixture-gated: --control-net + --control-image ----
    cn       = os.environ.get("CHIMERA_TEST_CONTROLNET")
    cn_image = os.environ.get("CHIMERA_TEST_CONTROL_IMAGE")
    if not cn and not cn_image:
        rec.skip("sd --control-net",
                 "set CHIMERA_TEST_CONTROLNET=<path> + CHIMERA_TEST_CONTROL_IMAGE=<path>")
    elif not cn or not cn_image:
        rec.fail("sd --control-net", 0.0,
                 "need BOTH CHIMERA_TEST_CONTROLNET + CHIMERA_TEST_CONTROL_IMAGE")
    elif not Path(cn).is_file():
        rec.fail("sd --control-net", 0.0, f"CHIMERA_TEST_CONTROLNET={cn} not found")
    elif not Path(cn_image).is_file():
        rec.fail("sd --control-net", 0.0, f"CHIMERA_TEST_CONTROL_IMAGE={cn_image} not found")
    else:
        with maybe(rec, "sd --control-net (CHIMERA_TEST_CONTROLNET fixture)") as t:
            with scratch_file(suffix=".png") as out:
                rc = run_silent(
                    [str(chimera), "sd", "-m", str(SD_MODEL), "-p", "a cyberpunk skyline",
                     "--control-net", cn, "--control-image", cn_image,
                     "-W", "256", "-H", "256", "-s", "2", "-o", str(out)],
                    timeout=600,
                )
                if rc != 0 or out.stat().st_size == 0:
                    t.fail(f"rc={rc}")

    # ---- Fixture-gated: --photo-maker + --pm-id-images-dir ----
    pm     = os.environ.get("CHIMERA_TEST_PHOTOMAKER")
    pm_dir = os.environ.get("CHIMERA_TEST_PM_ID_DIR")
    if not pm and not pm_dir:
        rec.skip("sd --photo-maker",
                 "set CHIMERA_TEST_PHOTOMAKER=<path> + CHIMERA_TEST_PM_ID_DIR=<dir>")
    elif not pm or not pm_dir:
        rec.fail("sd --photo-maker", 0.0,
                 "need BOTH CHIMERA_TEST_PHOTOMAKER + CHIMERA_TEST_PM_ID_DIR")
    elif not Path(pm).is_file():
        rec.fail("sd --photo-maker", 0.0, f"CHIMERA_TEST_PHOTOMAKER={pm} not found")
    elif not Path(pm_dir).is_dir():
        rec.fail("sd --photo-maker", 0.0, f"CHIMERA_TEST_PM_ID_DIR={pm_dir} not a dir")
    else:
        with maybe(rec, "sd --photo-maker (CHIMERA_TEST_PHOTOMAKER fixture)") as t:
            with scratch_file(suffix=".png") as out:
                rc = run_silent(
                    [str(chimera), "sd", "-m", str(SD_MODEL), "-p", "a portrait of img",
                     "--photo-maker", pm, "--pm-id-images-dir", pm_dir,
                     "-W", "256", "-H", "256", "-s", "2", "-o", str(out)],
                    timeout=600,
                )
                if rc != 0 or out.stat().st_size == 0:
                    t.fail(f"rc={rc}")


# ============================================================================
# End-to-end: MTMD vision pipeline
# ============================================================================

def e2e_mtmd_test(rec: Recorder, chimera: Path) -> None:
    label = "gen --mmproj --image (vision pipeline)"
    if not (MTMD_TEXT_MODEL.is_file() and MTMD_MMPROJ.is_file() and SD_MODEL is not None):
        rec.skip(label, "missing gemma model, mmproj, or sd model")
        return
    with maybe(rec, label) as t:
        with scratch_file(suffix=".png") as img:
            rc_sd = run_silent(
                [str(chimera), "sd", "-m", str(SD_MODEL),
                 "-p", "a single red apple, centered, white background",
                 "-W", "512", "-H", "512", "-s", "4", "-o", str(img)],
                timeout=300,
            )
            if rc_sd != 0 or img.stat().st_size == 0:
                t.fail("failed to synthesize input image")
                return
            rc, out, _ = run_capture(
                [str(chimera), "gen",
                 "-m", str(MTMD_TEXT_MODEL), "--mmproj", str(MTMD_MMPROJ),
                 "--image", str(img),
                 "-p", "Describe this image briefly.", "-n", "32"],
                timeout=300,
            )
            stripped = "".join(out.split())
            if not stripped:
                t.fail("empty output")


# ============================================================================
# End-to-end: chat (persistent KV cache + --persist round-trip)
# ============================================================================

def e2e_chat_cli_tests(rec: Recorder, chimera: Path) -> None:
    if not GEN_MODEL.is_file():
        rec.skip("chat persistent KV cache (recalls 'zephyrine')", f"missing {GEN_MODEL}")
        rec.skip("chat --persist + --list + --search round-trip", f"missing {GEN_MODEL}")
        return

    with maybe(rec, "chat persistent KV cache (recalls 'zephyrine')") as t:
        # -n 64: the 1B model often pads the recall reply with preamble.
        rc, out, _ = run_capture(
            [str(chimera), "chat", "-m", str(GEN_MODEL), "-n", "64"],
            timeout=120,
            stdin=b"my secret password is zephyrine.\nrepeat my secret password exactly.\n/exit\n",
        )
        normalized = "".join(out.split()).lower()
        if "zephyrine" not in normalized:
            t.fail(f"reply: {out!r}")

    with maybe(rec, "chat --persist + --list + --search round-trip") as t:
        with scratch_file(suffix=".db") as chat_db:
            run_capture(
                [str(chimera), "chat", "-m", str(GEN_MODEL),
                 "--db", str(chat_db), "--persist", "-n", "4", "--color", "never"],
                timeout=120,
                stdin=b"my secret token is xyzzy42.\n/exit\n",
            )
            _, list_out,   _ = run_capture(
                [str(chimera), "chat", "--db", str(chat_db), "--list"], timeout=30)
            _, search_out, _ = run_capture(
                [str(chimera), "chat", "--db", str(chat_db), "--search", "xyzzy42"],
                timeout=30,
            )
            if "#1" not in list_out or "[xyzzy42]" not in search_out:
                t.fail(f"list={list_out!r} search={search_out!r}")


# ============================================================================
# End-to-end: X-Chimera-Chat-Id header (server-side persistence)
# ============================================================================

def e2e_chat_id_header_tests(rec: Recorder, chimera: Path) -> None:
    names = [
        "X-Chimera-Chat-Id new chat → header echoed",
        "X-Chimera-Chat-Id echo → same id reused",
        "X-Chimera-Chat-Id unknown id → 404",
        "X-Chimera-Chat-Id malformed → 400",
        "X-Chimera-Chat-Id DB state (1 chat, 4 messages)",
    ]
    if not GEN_MODEL.is_file():
        for n in names:
            rec.skip(n, f"missing {GEN_MODEL}")
        return
    if not any(rec.matches(n) for n in names):
        return

    with scratch_file(suffix=".db") as chat_db:
        try:
            with chimera_serve(
                chimera,
                ["-m", str(GEN_MODEL), "--persist-chats", "--chat-db", str(chat_db)],
            ) as srv:
                first_id: Optional[str] = None
                # Case 1: no incoming header → server assigns + echoes a new id.
                with maybe(rec, names[0]) as t:
                    r = http_post_json(f"{srv.base_url}/v1/chat/completions", {
                        "model": "any",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 4, "stream": False,
                    }, timeout=120)
                    first_id = r.headers.get("x-chimera-chat-id", "").strip()
                    if not first_id:
                        t.fail(f"no X-Chimera-Chat-Id in response headers: {r.headers}")

                # Case 2: re-send with the captured id; same id must come back.
                with maybe(rec, names[1]) as t:
                    if not first_id:
                        t.fail("no first_id from case 1")
                    else:
                        r = http_post_json(f"{srv.base_url}/v1/chat/completions", {
                            "model": "any",
                            "messages": [
                                {"role": "user", "content": "hi"},
                                {"role": "assistant", "content": "hello"},
                                {"role": "user", "content": "again"},
                            ],
                            "max_tokens": 4, "stream": False,
                        }, headers={"X-Chimera-Chat-Id": first_id}, timeout=120)
                        echo = r.headers.get("x-chimera-chat-id", "").strip()
                        if echo != first_id:
                            t.fail(f"expected {first_id}, got {echo!r}")

                # Case 3: unknown id → 404.
                with maybe(rec, names[2]) as t:
                    r = http_post_json(f"{srv.base_url}/v1/chat/completions", {
                        "model": "any",
                        "messages": [{"role": "user", "content": "x"}],
                        "max_tokens": 4, "stream": False,
                    }, headers={"X-Chimera-Chat-Id": "9999999"}, timeout=30)
                    if r.status != 404:
                        t.fail(f"got HTTP {r.status}")

                # Case 4: malformed id → 400.
                with maybe(rec, names[3]) as t:
                    r = http_post_json(f"{srv.base_url}/v1/chat/completions", {
                        "model": "any",
                        "messages": [{"role": "user", "content": "x"}],
                        "max_tokens": 4, "stream": False,
                    }, headers={"X-Chimera-Chat-Id": "notanint"}, timeout=30)
                    if r.status != 400:
                        t.fail(f"got HTTP {r.status}")

        except RuntimeError as e:
            for n in names[:-1]:  # all four interactive cases failed; DB state below
                rec.fail(n, 0.0, str(e))

        # DB state: 1 chat + 4 messages (2 from case 1, 2 from case 2's
        # incremental save). Cases 3+4 are negative tests that must NOT
        # have created rows; this assertion catches it.
        with maybe(rec, names[4]) as t:
            try:
                conn = sqlite3.connect(str(chat_db))
                try:
                    chats    = conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
                    messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                finally:
                    conn.close()
            except Exception as e:
                t.fail(f"sqlite query failed: {e}")
                return
            if (chats, messages) != (1, 4):
                t.fail(f"got chats={chats}, messages={messages} (expected 1, 4)")


# ============================================================================
# End-to-end: /v1/chats* read endpoints + --public-path
# ============================================================================

def e2e_chats_endpoints_tests(rec: Recorder, chimera: Path) -> None:
    names = [
        "GET /v1/chats lists persisted chats",
        "GET /v1/chats/:id returns chat + ordered messages",
        "GET /v1/chats/search returns FTS5-highlighted hit",
        "GET /v1/chats/99999 → 400 (not 404)",
        "GET /v1/chats/search without q → 400",
        "GET / with --public-path serves index.html",
    ]
    if not GEN_MODEL.is_file():
        for n in names:
            rec.skip(n, f"missing {GEN_MODEL}")
        return
    if not any(rec.matches(n) for n in names):
        return

    with scratch_file(suffix=".db") as chat_db, \
         scratch_dir("chimera-pub-") as pub_dir:
        (pub_dir / "index.html").write_text(
            "<!DOCTYPE html><html><body>chimera-pub-marker</body></html>\n"
        )
        try:
            with chimera_serve(
                chimera,
                ["-m", str(GEN_MODEL),
                 "--persist-chats", "--chat-db", str(chat_db),
                 "--public-path", str(pub_dir)],
            ) as srv:
                # Seed two chats. The deliberately rare token guarantees a
                # unique FTS5 hit for the search assertion below.
                for prompt in ("tell me about quibblefrond", "explain photosynthesis briefly"):
                    http_post_json(f"{srv.base_url}/v1/chat/completions", {
                        "model": "any",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 4, "stream": False,
                    }, timeout=120)

                with maybe(rec, names[0]) as t:
                    r = http_get(f"{srv.base_url}/v1/chats")
                    try:
                        d = r.json()
                    except Exception as e:
                        t.fail(f"non-JSON response: {r.body!r} ({e})")
                        d = {}
                    if d.get("object") != "list" or len(d.get("data", [])) < 2:
                        t.fail(f"got: {r.body!r}")

                with maybe(rec, names[1]) as t:
                    r = http_get(f"{srv.base_url}/v1/chats/1")
                    try:
                        d = r.json()
                    except Exception as e:
                        t.fail(f"non-JSON: {r.body!r} ({e})")
                        d = {}
                    ms = d.get("messages", [])
                    if not (d.get("id") == 1 and ms and ms[0].get("role") == "user"
                            and "quibblefrond" in ms[0].get("content", "")):
                        t.fail(f"got: {r.body!r}")

                with maybe(rec, names[2]) as t:
                    r = http_get(f"{srv.base_url}/v1/chats/search?q=quibblefrond")
                    try:
                        d = r.json()
                    except Exception as e:
                        t.fail(f"non-JSON: {r.body!r} ({e})")
                        d = {}
                    hits = d.get("hits", [])
                    if not any("[quibblefrond]" in h.get("snippet", "").lower() for h in hits):
                        t.fail(f"got: {r.body!r}")

                with maybe(rec, names[3]) as t:
                    r = http_get(f"{srv.base_url}/v1/chats/99999")
                    if r.status != 400:
                        t.fail(f"got HTTP {r.status}")

                with maybe(rec, names[4]) as t:
                    r = http_get(f"{srv.base_url}/v1/chats/search")
                    if r.status != 400:
                        t.fail(f"got HTTP {r.status}")

                with maybe(rec, names[5]) as t:
                    r = http_get(f"{srv.base_url}/")
                    ct = r.headers.get("content-type", "")
                    if r.status != 200 or "text/html" not in ct \
                            or "chimera-pub-marker" not in r.body:
                        t.fail(f"status={r.status} content-type={ct} body={r.body[:120]!r}")

        except RuntimeError as e:
            for n in names:
                rec.fail(n, 0.0, str(e))


# ============================================================================
# End-to-end: /slots + /lora-adapters (two server passes — one without
# --slot-save-path and one with it, mirroring the bash structure).
# ============================================================================

def e2e_slots_lora_tests(rec: Recorder, chimera: Path) -> None:
    pass1_names = [
        "GET /slots returns JSON array of slot status",
        "POST /slots/0?action=save without --slot-save-path → 501",
        "GET /lora-adapters (no --lora) returns []",
        "POST /lora-adapters [] → 200",
    ]
    pass2_names = [
        "POST /slots/0?action=save with --slot-save-path → 200 + file written",
        "POST /slots/0?action=restore → 200",
    ]
    if not GEN_MODEL.is_file():
        for n in pass1_names + pass2_names:
            rec.skip(n, f"missing {GEN_MODEL}")
        return

    # Pass 1: no --slot-save-path, no --lora. Verifies the routes are bound
    # and report their unloaded shapes.
    if any(rec.matches(n) for n in pass1_names):
        try:
            with chimera_serve(chimera, ["-m", str(GEN_MODEL)]) as srv:
                with maybe(rec, pass1_names[0]) as t:
                    r = http_get(f"{srv.base_url}/slots")
                    try:
                        d = r.json()
                    except Exception as e:
                        t.fail(f"non-JSON: {r.body!r} ({e})")
                        d = []
                    if not (isinstance(d, list) and d and "id" in d[0]):
                        t.fail(f"got: {r.body!r}")

                with maybe(rec, pass1_names[1]) as t:
                    r = http_post_json(
                        f"{srv.base_url}/slots/0?action=save",
                        {"filename": "x.bin"}, timeout=30,
                    )
                    if r.status != 501:
                        t.fail(f"got HTTP {r.status}")

                with maybe(rec, pass1_names[2]) as t:
                    r = http_get(f"{srv.base_url}/lora-adapters")
                    try:
                        d = r.json()
                    except Exception as e:
                        t.fail(f"non-JSON: {r.body!r} ({e})")
                        d = None
                    if d != []:
                        t.fail(f"got: {r.body!r}")

                with maybe(rec, pass1_names[3]) as t:
                    r = http_post_json(f"{srv.base_url}/lora-adapters", [], timeout=30)
                    if r.status != 200:
                        t.fail(f"got HTTP {r.status}")

        except RuntimeError as e:
            for n in pass1_names:
                rec.fail(n, 0.0, str(e))

    # Pass 2: real --slot-save-path. Drive a small completion so the slot
    # has KV state to serialize, then save -> restore.
    if not any(rec.matches(n) for n in pass2_names):
        return
    with scratch_dir("chimera-slot-") as slot_dir:
        try:
            with chimera_serve(chimera, [
                "-m", str(GEN_MODEL),
                "--slot-save-path", str(slot_dir),
            ]) as srv:
                http_post_json(f"{srv.base_url}/v1/chat/completions", {
                    "model": "any",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4, "stream": False,
                }, timeout=120)

                with maybe(rec, pass2_names[0]) as t:
                    r = http_post_json(
                        f"{srv.base_url}/slots/0?action=save",
                        {"filename": "snap.bin"}, timeout=60,
                    )
                    snap = slot_dir / "snap.bin"
                    if r.status != 200 or not (snap.is_file() and snap.stat().st_size > 0):
                        t.fail(f"HTTP {r.status}, file_exists={snap.is_file()}")

                with maybe(rec, pass2_names[1]) as t:
                    r = http_post_json(
                        f"{srv.base_url}/slots/0?action=restore",
                        {"filename": "snap.bin"}, timeout=60,
                    )
                    if r.status != 200:
                        t.fail(f"got HTTP {r.status}")

        except RuntimeError as e:
            for n in pass2_names:
                rec.fail(n, 0.0, str(e))


# ============================================================================
# End-to-end: POST /v1/audio/detect-language
# ============================================================================

def e2e_detect_language_test(rec: Recorder, chimera: Path) -> None:
    label = "POST /v1/audio/detect-language (well-formed {language, duration})"
    if not (GEN_MODEL.is_file() and WHISPER_MODEL.is_file() and WHISPER_WAV.is_file()):
        rec.skip(label, f"needs {GEN_MODEL.name} + {WHISPER_MODEL.name} + {WHISPER_WAV.name}")
        return
    if not rec.matches(label):
        return
    try:
        with chimera_serve(chimera,
                           ["-m", str(GEN_MODEL), "--enable-audio", str(WHISPER_MODEL)]) as srv:
            with maybe(rec, label) as t:
                r = http_post_multipart(
                    f"{srv.base_url}/v1/audio/detect-language",
                    files={"file": WHISPER_WAV}, timeout=60,
                )
                try:
                    d = r.json()
                except Exception as e:
                    t.fail(f"HTTP {r.status} non-JSON: {r.body!r} ({e})")
                    return
                lang = d.get("language", "")
                dur  = d.get("duration", 0)
                # The .en model gives noise for non-English lang tokens, so
                # we don't assert "en" — only that the wire format is sane:
                # a 2-letter alphabetic code + positive duration.
                ok = (isinstance(lang, str) and len(lang) == 2 and lang.isalpha()
                      and isinstance(dur, (int, float)) and dur > 0)
                if not ok:
                    t.fail(f"got body: {r.body!r}")
    except RuntimeError as e:
        rec.fail(label, 0.0, str(e))


# ============================================================================
# Image-serve gating: CLI enum validators (step 5d, no serve spawn)
# ============================================================================

def e2e_image_serve_enum_validators(rec: Recorder, chimera: Path) -> None:
    # The validators fire before model load; we can use any path for -m
    # because CLI11's IsMember check kicks in first. We point at GEN_MODEL
    # (a known-good file) just so the negative result is unambiguous.
    if not GEN_MODEL.is_file():
        for label in (
            "serve --sd-rng bogus_rng exits non-zero (CLI11 enum)",
            "serve --sd-sampler-rng bogus_srng exits non-zero (CLI11 enum)",
            "serve --sd-prediction bogus_pred exits non-zero (CLI11 enum)",
            "serve --sd-lora-apply-mode bogus_lam exits non-zero (CLI11 enum)",
        ):
            rec.skip(label, f"missing {GEN_MODEL}")
        return

    for flag, val in [
        ("--sd-rng",             "bogus_rng"),
        ("--sd-sampler-rng",     "bogus_srng"),
        ("--sd-prediction",      "bogus_pred"),
        ("--sd-lora-apply-mode", "bogus_lam"),
    ]:
        label = f"serve {flag} {val} exits non-zero (CLI11 enum)"
        with maybe(rec, label) as t:
            # Port=1 is privileged; this would fail-to-bind even if the
            # validator were silently stripped. So we use a free port to
            # make sure the only reason for non-zero exit is the enum check.
            rc = run_silent(
                [str(chimera), "serve", "-m", str(GEN_MODEL), flag, val,
                 "--host", "127.0.0.1", "--port", str(free_port()),
                 "--gpu-layers", "0"],
                timeout=10,
            )
            if rc == 0:
                t.fail("expected non-zero exit (validator absent?)")


# ============================================================================
# Image-serve gating: per-request 400s for ControlNet / PhotoMaker / LoRA
# (steps 5b, 5e, 6). One server spawn, no `--sd-control-net /
# --sd-photo-maker / --sd-lora` — every opt-in path must 400 with the
# missing-flag hint.
# ============================================================================

# 1x1 PNG used as a placeholder image where multipart fields require one
# but the test only cares about gate behavior. Base64-decoded inline so
# the test runner stays self-contained.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAA"
    "C0lEQVR42mNkAAIAAAUAAeImBZsAAAAASUVORK5CYII="
)

def _tiny_png_bytes() -> bytes:
    import base64
    return base64.b64decode(_TINY_PNG_B64)


def e2e_image_serve_gating(rec: Recorder, chimera: Path) -> None:
    names = [
        "5e pm_id_images without --sd-photo-maker → 400 + named-flag hint",
        "5e pm_id_image_set without --sd-photo-maker → 400 + named-flag hint",
        "5b control_image without --sd-control-net → 400 + named-flag hint",
        "6 loras without --sd-lora → 400 + named-flag hint",
        "6 GET /v1/images/lora-adapters (no --sd-lora) returns []",
    ]
    if not (GEN_MODEL.is_file() and SD_MODEL is not None):
        for n in names:
            rec.skip(n, f"needs {GEN_MODEL.name} + an SD model")
        return
    if not any(rec.matches(n) for n in names):
        return
    try:
        with chimera_serve(
            chimera,
            ["-m", str(GEN_MODEL), "--enable-image", str(SD_MODEL)],
            wait_for_sd=True, startup_timeout=240.0,
        ) as srv:
            base = srv.base_url

            with maybe(rec, names[0]) as t:
                r = http_post_json(f"{base}/v1/images/generations", {
                    "prompt": "x", "pm_id_images": ["AAAA"],
                }, timeout=30)
                if r.status != 400 or "--sd-photo-maker" not in r.body:
                    t.fail(f"HTTP {r.status}, body={r.body!r}")

            with maybe(rec, names[1]) as t:
                r = http_post_json(f"{base}/v1/images/generations", {
                    "prompt": "x", "pm_id_image_set": "foo",
                }, timeout=30)
                if r.status != 400 or "--sd-photo-maker" not in r.body:
                    t.fail(f"HTTP {r.status}, body={r.body!r}")

            # ControlNet gating runs via /v1/images/edits (the natural
            # multipart endpoint). The handler decodes the init `image`
            # before calling maybe_attach_control, so we must supply a
            # valid PNG for both `image` and `control_image`.
            with maybe(rec, names[2]) as t:
                png = _tiny_png_bytes()
                r = http_post_multipart(
                    f"{base}/v1/images/edits",
                    fields={"prompt": "x"},
                    files={"image": ("tiny.png", png),
                           "control_image": ("tiny.png", png)},
                    timeout=60,
                )
                if r.status != 400 or "--sd-control-net" not in r.body:
                    t.fail(f"HTTP {r.status}, body={r.body!r}")

            with maybe(rec, names[3]) as t:
                r = http_post_json(f"{base}/v1/images/generations", {
                    "prompt": "x",
                    "loras": [{"name": "foo", "scale": 0.7}],
                }, timeout=30)
                if r.status != 400 or "--sd-lora" not in r.body:
                    t.fail(f"HTTP {r.status}, body={r.body!r}")

            with maybe(rec, names[4]) as t:
                r = http_get(f"{base}/v1/images/lora-adapters")
                try:
                    d = r.json()
                except Exception as e:
                    t.fail(f"non-JSON: {r.body!r} ({e})")
                    d = None
                if d != []:
                    t.fail(f"got: {r.body!r}")

    except RuntimeError as e:
        for n in names:
            rec.fail(n, 0.0, str(e))


def e2e_image_serve_aliases(rec: Recorder, chimera: Path) -> None:
    """Second server pass with two --sd-lora aliases registered. Tests the
    listing + unknown-name path. Adapter files don't have to exist — SD
    only opens them at generate() time, and these tests fire before that."""
    names = [
        "6 GET /v1/images/lora-adapters lists registered aliases (names only)",
        "6 loras unknown name → 400 listing known aliases",
    ]
    if not (GEN_MODEL.is_file() and SD_MODEL is not None):
        for n in names:
            rec.skip(n, f"needs {GEN_MODEL.name} + an SD model")
        return
    if not any(rec.matches(n) for n in names):
        return
    try:
        with chimera_serve(
            chimera,
            ["-m", str(GEN_MODEL), "--enable-image", str(SD_MODEL),
             "--sd-lora", "alpha=/nonexistent/alpha.safetensors",
             "--sd-lora", "beta=/nonexistent/beta.safetensors"],
            wait_for_sd=True, startup_timeout=240.0,
        ) as srv:
            base = srv.base_url

            with maybe(rec, names[0]) as t:
                r = http_get(f"{base}/v1/images/lora-adapters")
                try:
                    d = r.json()
                except Exception as e:
                    t.fail(f"non-JSON: {r.body!r} ({e})")
                    d = []
                names_present = {item.get("name") for item in d if isinstance(item, dict)}
                has_path = any("path" in item for item in d if isinstance(item, dict))
                if names_present != {"alpha", "beta"} or has_path:
                    t.fail(f"got: {r.body!r}")

            with maybe(rec, names[1]) as t:
                r = http_post_json(f"{base}/v1/images/generations", {
                    "prompt": "x",
                    "loras": [{"name": "gamma", "scale": 0.5}],
                }, timeout=30)
                ok = (r.status == 400
                      and "not registered" in r.body
                      and "alpha" in r.body and "beta" in r.body)
                if not ok:
                    t.fail(f"HTTP {r.status}, body={r.body!r}")

    except RuntimeError as e:
        for n in names:
            rec.fail(n, 0.0, str(e))


# ============================================================================
# Image-serve success paths (fixture-gated; each spawns its own server).
# Each block fires independently so a developer with only one fixture set
# still gets that test run.
# ============================================================================

def _check_b64_image(body: str) -> bool:
    """Return True iff `body` parses as JSON {"data":[{"b64_json":"..."}]}
    with at least one ~100-byte image. Any real PNG clears that easily."""
    import base64
    try:
        d = json.loads(body)
    except Exception:
        return False
    data = d.get("data") or []
    if not data:
        return False
    b64 = (data[0] or {}).get("b64_json", "")
    if not b64:
        return False
    try:
        return len(base64.b64decode(b64)) > 100
    except Exception:
        return False


def e2e_image_serve_success_paths(rec: Recorder, chimera: Path) -> None:
    # ---- LoRA ----
    label = "serve loras success path"
    lora = os.environ.get("CHIMERA_TEST_LORA")
    if not lora:
        rec.skip(label, "set CHIMERA_TEST_LORA=<path/to/lora.safetensors>")
    elif not Path(lora).is_file():
        rec.fail(label, 0.0, f"CHIMERA_TEST_LORA={lora} not found")
    elif rec.matches(label):
        try:
            with chimera_serve(
                chimera,
                ["-m", str(GEN_MODEL), "--enable-image", str(SD_MODEL),
                 "--sd-lora", f"test={lora}"],
                wait_for_sd=True, startup_timeout=240.0,
            ) as srv:
                pass_label = "serve loras success (CHIMERA_TEST_LORA fixture, 200 + b64 image)"
                with maybe(rec, pass_label) as t:
                    r = http_post_json(f"{srv.base_url}/v1/images/generations", {
                        "prompt": "a red cube",
                        "loras": [{"name": "test", "scale": 0.7}],
                        "size": "256x256", "steps": 2,
                    }, timeout=600)
                    if not (r.status == 200 and _check_b64_image(r.body)):
                        t.fail(f"HTTP {r.status}, first 200 chars: {r.body[:200]!r}")
        except RuntimeError as e:
            rec.fail(label, 0.0, str(e))

    # ---- ControlNet ----
    label = "serve control_image success path"
    cn       = os.environ.get("CHIMERA_TEST_CONTROLNET")
    cn_image = os.environ.get("CHIMERA_TEST_CONTROL_IMAGE")
    if not cn and not cn_image:
        rec.skip(label, "set CHIMERA_TEST_CONTROLNET=<path> + CHIMERA_TEST_CONTROL_IMAGE=<path>")
    elif not cn or not cn_image:
        rec.fail(label, 0.0, "need BOTH CHIMERA_TEST_CONTROLNET + CHIMERA_TEST_CONTROL_IMAGE")
    elif not Path(cn).is_file():
        rec.fail(label, 0.0, f"CHIMERA_TEST_CONTROLNET={cn} not found")
    elif not Path(cn_image).is_file():
        rec.fail(label, 0.0, f"CHIMERA_TEST_CONTROL_IMAGE={cn_image} not found")
    elif rec.matches(label):
        try:
            with chimera_serve(
                chimera,
                ["-m", str(GEN_MODEL), "--enable-image", str(SD_MODEL),
                 "--sd-control-net", cn],
                wait_for_sd=True, startup_timeout=240.0,
            ) as srv:
                pass_label = "serve control_image success (CHIMERA_TEST_CONTROLNET fixture, 200 + b64 image)"
                with maybe(rec, pass_label) as t:
                    r = http_post_multipart(
                        f"{srv.base_url}/v1/images/edits",
                        fields={"prompt": "a cyberpunk skyline",
                                "control_strength": "0.7",
                                "size": "256x256", "steps": "2"},
                        files={"image": cn_image, "control_image": cn_image},
                        timeout=600,
                    )
                    if not (r.status == 200 and _check_b64_image(r.body)):
                        t.fail(f"HTTP {r.status}, first 200 chars: {r.body[:200]!r}")
        except RuntimeError as e:
            rec.fail(label, 0.0, str(e))

    # ---- PhotoMaker ----
    label = "serve pm_id_image_set success path"
    pm     = os.environ.get("CHIMERA_TEST_PHOTOMAKER")
    pm_dir = os.environ.get("CHIMERA_TEST_PM_ID_DIR")
    if not pm and not pm_dir:
        rec.skip(label, "set CHIMERA_TEST_PHOTOMAKER=<path> + CHIMERA_TEST_PM_ID_DIR=<dir>")
    elif not pm or not pm_dir:
        rec.fail(label, 0.0, "need BOTH CHIMERA_TEST_PHOTOMAKER + CHIMERA_TEST_PM_ID_DIR")
    elif not Path(pm).is_file():
        rec.fail(label, 0.0, f"CHIMERA_TEST_PHOTOMAKER={pm} not found")
    elif not Path(pm_dir).is_dir():
        rec.fail(label, 0.0, f"CHIMERA_TEST_PM_ID_DIR={pm_dir} not a dir")
    elif rec.matches(label):
        subdirs = sorted(p.name for p in Path(pm_dir).iterdir() if p.is_dir())
        if not subdirs:
            rec.fail(label, 0.0, "no subdirectories under CHIMERA_TEST_PM_ID_DIR")
        else:
            pm_set = subdirs[0]
            try:
                with chimera_serve(
                    chimera,
                    ["-m", str(GEN_MODEL), "--enable-image", str(SD_MODEL),
                     "--sd-photo-maker", pm, "--sd-pm-id-dir", pm_dir],
                    wait_for_sd=True, startup_timeout=240.0,
                ) as srv:
                    pass_label = (
                        f"serve pm_id_image_set success "
                        f"(CHIMERA_TEST_PHOTOMAKER fixture, set='{pm_set}', 200 + b64 image)"
                    )
                    with maybe(rec, pass_label) as t:
                        r = http_post_json(f"{srv.base_url}/v1/images/generations", {
                            "prompt": "a portrait of img",
                            "pm_id_image_set": pm_set,
                            "size": "256x256", "steps": 2,
                        }, timeout=600)
                        if not (r.status == 200 and _check_b64_image(r.body)):
                            t.fail(f"HTTP {r.status}, first 200 chars: {r.body[:200]!r}")
            except RuntimeError as e:
                rec.fail(label, 0.0, str(e))


# ============================================================================
# Main
# ============================================================================

# Section table — the order here is the order tests run in. Keeping the
# order deterministic matters because some tests share state (e.g. all
# the GEN_MODEL-driven blocks share the gen model's warmup cost on disk
# cache).
_SECTIONS = [
    ("smoke",                       smoke_tests),
    ("gen",                         e2e_gen_tests),
    ("embed",                       e2e_embed_tests),
    ("whisper",                     e2e_whisper_cli_test),
    ("sd",                          e2e_sd_cli_tests),
    ("mtmd",                        e2e_mtmd_test),
    ("chat",                        e2e_chat_cli_tests),
    ("chat_id_header",              e2e_chat_id_header_tests),
    ("chats_endpoints",             e2e_chats_endpoints_tests),
    ("slots_lora",                  e2e_slots_lora_tests),
    ("detect_language",             e2e_detect_language_test),
    ("image_serve_enum_validators", e2e_image_serve_enum_validators),
    ("image_serve_gating",          e2e_image_serve_gating),
    ("image_serve_aliases",         e2e_image_serve_aliases),
    ("image_serve_success_paths",   e2e_image_serve_success_paths),
]


def parse_args(argv: list) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run chimera's smoke + end-to-end test suite.",
    )
    p.add_argument("--smoke", action="store_true",
                   help="Run smoke tests only (skip end-to-end).")
    p.add_argument("--filter", metavar="REGEX",
                   help="Run only tests whose name matches REGEX.")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI color output. Forced off if stdout isn't a TTY.")
    p.add_argument("--verbose", action="store_true",
                   help="Stream chimera subprocess stdout/stderr to the terminal.")
    p.add_argument("--no-timing", action="store_true",
                   help="Don't print the timing table at the end of the summary.")
    return p.parse_args(argv)


def main(argv: list) -> int:
    args = parse_args(argv)

    global _color_enabled, _verbose
    _color_enabled = (
        not args.no_color
        and sys.stdout.isatty()
        and not os.environ.get("NO_COLOR")
    )
    _verbose = args.verbose

    chimera = locate_chimera()
    if chimera is None:
        print(f"FAIL: chimera binary not found under {REPO_ROOT}/build/", file=sys.stderr)
        print("      searched: build/chimera, build/chimera.exe, "
              "build/{Release,Debug}/chimera[.exe]", file=sys.stderr)
        print("      (run 'make build' first, or set CHIMERA=...)", file=sys.stderr)
        return 1

    name_filter: Optional[re.Pattern] = None
    if args.filter:
        try:
            name_filter = re.compile(args.filter)
        except re.error as e:
            print(f"FAIL: invalid --filter regex: {e}", file=sys.stderr)
            return 1

    rec = Recorder(name_filter=name_filter)

    print("== smoke ==")
    _SECTIONS[0][1](rec, chimera)

    if not args.smoke:
        print()
        print("== end-to-end ==")
        for _, fn in _SECTIONS[1:]:
            fn(rec, chimera)

    failed = rec.summary(show_timing=not args.no_timing)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
