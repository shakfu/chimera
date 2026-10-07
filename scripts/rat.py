#!/usr/bin/env python3
"""Run artifact tests: a self-contained smoke-test runner for chimera build
artifacts, meant to check a binary before it is released.

chimera ships as one static executable, so there is nothing to `pip install`
and no virtualenv to reason about: ``install`` unpacks an archive (local, by
URL, or from a GitHub release) into ``build/rat/``, and every test target runs
that binary's CLI. ``--bin`` points at a bare executable instead (a downloaded
CI artifact or a local ``build/chimera``, say).

``--cuda`` (and ``--cpu`` / ``--metal`` / ``--vulkan`` / ``--rocm`` /
``--sycl``) names the backend, which selects the release asset for this
platform -- ``--cuda`` on Linux is ``chimera-<version>-linux-x86_64-cuda.tar.gz``.
Without one the backend is read back out of the installed binary's own
``chimera info`` (`built:`), so a test run never has to be told what it is
testing. ``--cpu`` and ``--metal`` name the same macOS asset: CI builds the
macos-arm64 artifact with Metal on and there is no CPU-only macOS build, so a
plain macos-arm64 binary is reported as ``metal``.

``install`` is the only subcommand that writes to ``build/rat/``: ``--asset`` says
what to put there -- a local archive, a full URL, or a bare release-asset
filename, told apart by shape -- and ``--version`` picks the release when it
does not (default: whatever ``/releases/latest`` resolves to). Every test
target expects a binary that is already in place.

``test`` takes one target -- ``test-all``, ``test-gen-all``, ``test-sd-3`` --
named identically to the generated Makefile rules; ``list tests`` prints them.

``run`` is ``install``, ``test`` and ``clean`` in one invocation, stopping at
the first step that fails and taking the options of all three. It is the whole
cycle for one backend, so a release can be checked on a machine that has
nothing installed yet without three commands that must agree on which binary
they mean. ``--fast`` swaps ``test-all`` for ``test-embed-1``, ``test-gen-1``
and ``test-sd-3`` -- the same shape of coverage without the image cases that
dominate the wall clock.

Models are never fetched by a test. ``download all`` is the separate step that
puts them (and jfk.wav) in the models dir; ``test`` uses what
is on disk and reports a case whose model is missing as SKIP, so one absent
model costs one case rather than the run. Downloading multiple GB in the
middle of a timed test run is neither a test result nor a thing to wait for.

The cases are the shell scripts that used to live in ``scripts/case/``,
inlined here so they share one model registry, one timeout, one summary and
one place that knows a GPU build needs ``--gpu-layers``.

The script is organised as a handful of collaborating objects rather than
module state: :class:`Paths` resolves the directory layout, :class:`Release`
knows how release assets are named and fetched, :class:`Env` owns the binary
under test, :class:`ModelRegistry` knows where models and data assets come
from, :class:`TestSuite` holds the test cases, and :class:`Cli` wires them to
argparse.

Examples:
    # download the latest linux-x86_64-cuda release into ./bin and test it
    python3 scripts/rat.py install --cuda
    python3 scripts/rat.py test --cuda test-all

    # a specific release, or a local artifact / explicit URL
    python3 scripts/rat.py install --cuda --version 0.2.16
    python3 scripts/rat.py install --asset dist/chimera-0.2.16-linux-x86_64-cuda.tar.gz
    python3 scripts/rat.py install --asset https://github.com/shakfu/chimera/releases/download/0.2.16/chimera-0.2.16-linux-x86_64-cuda.tar.gz

    # install, test everything, then remove the binary again -- one command
    python3 scripts/rat.py run --cuda
    python3 scripts/rat.py run --cuda --fast    # a short cycle instead of everything
    python3 scripts/rat.py run --vulkan test-sd-all --timeout 900

    # run everything, one family, or one case
    python3 scripts/rat.py test test-all
    python3 scripts/rat.py test test-rag-all
    python3 scripts/rat.py test test-sd-3 --timeout 600

    # against a binary that was not installed from a release; the backend is
    # detected from `chimera info`, so no --cuda/--vulkan/... is needed
    python3 scripts/rat.py test --bin build/chimera test-all

    # show the matrix without downloading or running anything
    python3 scripts/rat.py test --cuda test-all --dry-run

    # environment, registry and target listings
    python3 scripts/rat.py info
    python3 scripts/rat.py list
    python3 scripts/rat.py download all --models-dir models
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import json
import math
import os
import platform
import re
import shutil
import sqlite3
import statistics
import struct
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
import zipfile
import zlib
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_NAME = Path(__file__).name
# The `project` column in the shared run history; see RunLog.
PROJECT = "chimera"


# ---------------------------------------------------------------------------
# exceptions
# ---------------------------------------------------------------------------


class ModelSourceUnavailable(RuntimeError):
    """Raised when a model cannot be obtained.

    Either it has no configured source and isn't on disk, or the download for
    it failed. A test that needs such a model is skipped rather than failed:
    an unreachable model says nothing about the binary under test.
    """


class AssetUnavailable(RuntimeError):
    """Raised when no release asset exists for a backend/platform pair."""


# ---------------------------------------------------------------------------
# image checks
# ---------------------------------------------------------------------------


def read_png(path: Path) -> tuple[int, int, int, bytes]:
    """Decode an 8-bit, non-interlaced PNG to (width, height, channels, pixels).

    Stdlib only: the script must run standalone, with no image library. That
    covers what stb_image_write produces.

    Raises:
        OSError: the file cannot be read.
        ValueError, zlib.error: the file is not a PNG this can decode.
    """
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    header: tuple[int, ...] | None = None
    idat = bytearray()
    pos = 8
    while pos + 8 <= len(data):
        length, ctype = struct.unpack(">I4s", data[pos : pos + 8])
        body = data[pos + 8 : pos + 8 + length]
        if ctype == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            break
        pos += 12 + length
    if header is None:
        raise ValueError("no IHDR chunk")
    width, height, depth, color, _, _, interlace = header
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color)
    if depth != 8 or channels is None or interlace:
        raise ValueError(f"unsupported PNG (bit depth {depth}, color type {color}, interlace {interlace})")

    raw = zlib.decompress(idat)
    stride = width * channels
    if len(raw) != height * (stride + 1):
        raise ValueError(f"image data is {len(raw)} bytes, expected {height * (stride + 1)}")
    pixels = bytearray()
    prev = bytearray(stride)
    for y in range(height):
        start = y * (stride + 1) + 1
        ftype = raw[start - 1]
        row = bytearray(raw[start : start + stride])
        if ftype == 1:  # Sub
            for i in range(channels, stride):
                row[i] = (row[i] + row[i - channels]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                left = row[i - channels] if i >= channels else 0
                row[i] = (row[i] + (left + prev[i]) // 2) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = row[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                row[i] = (row[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 0xFF
        elif ftype != 0:
            raise ValueError(f"row {y} has unknown filter type {ftype}")
        pixels += row
        prev = row
    return width, height, channels, bytes(pixels)


def channel_stddevs(pixels: bytes, channels: int) -> list[float]:
    """Population standard deviation of each channel of interleaved 8-bit pixels."""
    result = []
    for ch in range(channels):
        hist = Counter(pixels[ch::channels])
        n = sum(hist.values())
        mean = sum(v * k for v, k in hist.items()) / n
        result.append(math.sqrt(sum(k * (v - mean) ** 2 for v, k in hist.items()) / n))
    return result


# ---------------------------------------------------------------------------
# run history (keep identical across cyllama/inferna rwt.py and chimera rat.py)
# ---------------------------------------------------------------------------


class RunLog:
    """Run history in one SQLite database shared by rwt.py and rat.py.

    Every project writes to the same file, so `runs diff` compares any two runs:
    two versions, two backends, or two projects. Rows are written as each case
    ends; a run with no `finished_at` was interrupted. A database error disables
    recording with a warning and never fails the test run.
    """

    SCHEMA_VERSION = 1
    SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY,
    session         TEXT NOT NULL,  -- shared by the test steps of one `run`
    project         TEXT NOT NULL,
    target          TEXT NOT NULL,
    backend         TEXT NOT NULL,
    version         TEXT,
    artifact        TEXT,           -- distribution name, or binary path
    artifact_sha256 TEXT,           -- wheel RECORD, or binary
    git_commit      TEXT,
    git_dirty       INTEGER,
    host            TEXT NOT NULL,
    platform        TEXT NOT NULL,
    argv            TEXT NOT NULL,  -- JSON
    extra           TEXT,           -- JSON, project-specific
    started_at      TEXT NOT NULL,  -- UTC ISO 8601
    finished_at     TEXT,
    seconds         REAL,
    rc              INTEGER
);
CREATE TABLE IF NOT EXISTS cases (
    id      INTEGER PRIMARY KEY,
    run_id  INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    family  TEXT NOT NULL,
    n       TEXT NOT NULL,
    status  TEXT NOT NULL,  -- pass | fail | timeout | skip
    rc      INTEGER,
    seconds REAL NOT NULL,
    detail  TEXT
);
CREATE TABLE IF NOT EXISTS outputs (
    id      INTEGER PRIMARY KEY,
    case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    name    TEXT NOT NULL,
    bytes   INTEGER NOT NULL,
    sha256  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    id      INTEGER PRIMARY KEY,
    case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    name    TEXT NOT NULL,  -- e.g. tokens_per_second
    value   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_by_key ON runs(project, backend, target, id);
CREATE INDEX IF NOT EXISTS cases_by_run ON cases(run_id);
CREATE INDEX IF NOT EXISTS outputs_by_case ON outputs(case_id);
CREATE INDEX IF NOT EXISTS metrics_by_case ON metrics(case_id);
"""

    def __init__(self, project: str, path: Path | None = None) -> None:
        self.project = project
        self.path = path or self.default_path()
        self.enabled = True
        self.session = uuid.uuid4().hex[:12]
        self.run_id: int | None = None
        self._db: sqlite3.Connection | None = None
        self._started = 0.0

    @staticmethod
    def default_path() -> Path:
        """``$RUNS_DB``, else ``~/config/runs/db.sqlite``.

        Not under ``~/.config``: snap-packaged browsers cannot read hidden
        directories, so a report written beside the database would not open.
        """
        return Path(os.environ.get("RUNS_DB") or "~/config/runs/db.sqlite").expanduser()

    # -- plumbing -----------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        if self._db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path, timeout=30)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            # WAL lets a run in one project write while another project's run reads.
            db.execute("PRAGMA journal_mode = WAL")
            found = db.execute("PRAGMA user_version").fetchone()[0]
            if found > self.SCHEMA_VERSION:
                db.close()
                raise sqlite3.DatabaseError(f"{self.path} has schema {found}; this script knows {self.SCHEMA_VERSION}")
            db.executescript(self.SCHEMA)
            db.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
            self._db = db
        return self._db

    def _tx(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        """Run `fn` in one transaction; on any database error, stop recording."""
        if not self.enabled:
            return None
        try:
            db = self.connect()
            with db:
                return fn(db)
        except (sqlite3.Error, OSError) as e:
            print(f"warning: run history disabled ({self.path}): {e}", file=sys.stderr)
            self.enabled = False
            return None

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def git_state(root: Path) -> tuple[str | None, bool | None]:
        """(HEAD commit, has uncommitted changes) of `root`; (None, None) outside git."""
        try:
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            )
            if head.returncode != 0:
                return None, None
            status = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None, None
        return head.stdout.strip(), bool(status.stdout.strip())

    # -- recording ----------------------------------------------------------

    def start(
        self,
        target: str,
        backend: str,
        root: Path,
        version: str | None = None,
        artifact: str | None = None,
        artifact_sha256: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        commit, dirty = self.git_state(root)
        self._started = time.monotonic()
        row = (
            self.session,
            self.project,
            target,
            backend,
            version,
            artifact,
            artifact_sha256,
            commit,
            None if dirty is None else int(dirty),
            platform.node(),
            f"{sys.platform}-{platform.machine()}",
            json.dumps(sys.argv[1:]),
            json.dumps(extra or {}, sort_keys=True),
            self.now(),
        )
        self.run_id = self._tx(
            lambda db: (
                db.execute(
                    "INSERT INTO runs (session, project, target, backend, version, artifact, artifact_sha256,"
                    " git_commit, git_dirty, host, platform, argv, extra, started_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    row,
                ).lastrowid
            )
        )

    @staticmethod
    def status(rc: int, skipped: str | None) -> str:
        if skipped is not None:
            return "skip"
        if rc == 124:  # Env.run's timeout code
            return "timeout"
        return "pass" if rc == 0 else "fail"

    def case(
        self,
        family: str,
        n: str,
        rc: int,
        seconds: float,
        skipped: str | None = None,
        outputs: Sequence[Path] = (),
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Record one case, its `metrics`, and the size and sha256 of each output it left on disk."""
        if self.run_id is None:
            return
        run_id = self.run_id
        files = [(p.name, p.stat().st_size, self.sha256(p)) for p in outputs if p.is_file()]

        def write(db: sqlite3.Connection) -> None:
            case_id = db.execute(
                "INSERT INTO cases (run_id, family, n, status, rc, seconds, detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, family, n, self.status(rc, skipped), None if skipped else rc, seconds, skipped),
            ).lastrowid
            db.executemany(
                "INSERT INTO outputs (case_id, name, bytes, sha256) VALUES (?, ?, ?, ?)",
                [(case_id, *f) for f in files],
            )
            db.executemany(
                "INSERT INTO metrics (case_id, name, value) VALUES (?, ?, ?)",
                [(case_id, k, v) for k, v in sorted((metrics or {}).items())],
            )

        self._tx(write)

    def finish(self, rc: int) -> None:
        if self.run_id is None:
            return
        row = (self.now(), time.monotonic() - self._started, rc, self.run_id)
        self._tx(lambda db: db.execute("UPDATE runs SET finished_at = ?, seconds = ?, rc = ? WHERE id = ?", row))
        self.run_id = None

    # -- reporting ----------------------------------------------------------

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        if not self.path.exists():
            return []
        return self.connect().execute(sql, params).fetchall()

    def print_list(self, limit: int, backend: str | None = None, all_projects: bool = False) -> int:
        rows = self._query(
            "SELECT r.*, COUNT(c.id) AS ran, COALESCE(SUM(c.status = 'pass'), 0) AS passed"
            " FROM runs r LEFT JOIN cases c ON c.run_id = r.id"
            " WHERE (? OR r.project = ?) AND (? IS NULL OR r.backend = ?)"
            " GROUP BY r.id ORDER BY r.id DESC LIMIT ?",
            (all_projects, self.project, backend, backend, limit),
        )
        if not rows:
            print(f"no runs recorded in {self.path}")
            return 0
        print(
            f"{'id':>5}  {'started (UTC)':<19}  {'project':<8}  {'backend':<7}  {'version':<12}  "
            f"{'target':<16}  {'passed':>6}  {'secs':>7}  rc"
        )
        for r in reversed(rows):
            secs = f"{r['seconds']:.1f}" if r["seconds"] is not None else "-"
            rc = "-" if r["rc"] is None else str(r["rc"])
            print(
                f"{r['id']:>5}  {r['started_at'][:19]:<19}  {r['project']:<8}  {r['backend']:<7}  "
                f"{(r['version'] or '?'):<12}  {r['target']:<16}  {r['passed']:>3}/{r['ran']:<2}  {secs:>7}  {rc}"
            )
        return 0

    def _resolve_pair(self, a: int | None, b: int | None, backend: str | None) -> tuple[sqlite3.Row, sqlite3.Row]:
        """Runs `a` and `b`. Missing `b` is this project's latest finished run;
        missing `a` is the finished run before `b` with the same project, backend
        and target."""

        def one(sql: str, params: Sequence[Any], what: str) -> sqlite3.Row:
            rows = self._query(sql, params)
            if not rows:
                raise LookupError(f"no {what} in {self.path}")
            return rows[0]

        if b is None:
            run_b = one(
                "SELECT * FROM runs WHERE project = ? AND (? IS NULL OR backend = ?) AND finished_at IS NOT NULL"
                " ORDER BY id DESC LIMIT 1",
                (self.project, backend, backend),
                f"finished {self.project} run",
            )
        else:
            run_b = one("SELECT * FROM runs WHERE id = ?", (b,), f"run {b}")
        if a is None:
            run_a = one(
                "SELECT * FROM runs WHERE project = ? AND backend = ? AND target = ? AND id < ?"
                " AND finished_at IS NOT NULL ORDER BY id DESC LIMIT 1",
                (run_b["project"], run_b["backend"], run_b["target"], run_b["id"]),
                f"earlier {run_b['project']} {run_b['backend']} {run_b['target']} run to compare run {run_b['id']} with",
            )
        else:
            run_a = one("SELECT * FROM runs WHERE id = ?", (a,), f"run {a}")
        return run_a, run_b

    def _cases(
        self, run_id: int
    ) -> dict[tuple[str, str], tuple[sqlite3.Row, dict[str, sqlite3.Row], dict[str, float]]]:
        cases = self._query("SELECT * FROM cases WHERE run_id = ? ORDER BY id", (run_id,))
        result = {}
        for c in cases:
            outs = self._query("SELECT * FROM outputs WHERE case_id = ?", (c["id"],))
            mets = self._query("SELECT name, value FROM metrics WHERE case_id = ?", (c["id"],))
            result[(c["family"], c["n"])] = (c, {o["name"]: o for o in outs}, {m["name"]: m["value"] for m in mets})
        return result

    def print_diff(self, a: int | None = None, b: int | None = None, backend: str | None = None) -> int:
        try:
            run_a, run_b = self._resolve_pair(a, b, backend)
        except LookupError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

        def short(value: Any, n: int = 12) -> str:
            return "-" if value is None else str(value)[:n]

        def secs(value: float | None) -> str:
            return "-" if value is None else f"{value:.1f}"

        def commit(r: sqlite3.Row) -> str:
            return short(r["git_commit"], 10) + ("+dirty" if r["git_dirty"] else "")

        fields: list[tuple[str, Callable[[sqlite3.Row], str]]] = [
            ("run", lambda r: str(r["id"])),
            ("started", lambda r: r["started_at"][:19]),
            ("project", lambda r: r["project"]),
            ("backend", lambda r: r["backend"]),
            ("target", lambda r: r["target"]),
            ("version", lambda r: short(r["version"], 30)),
            ("artifact", lambda r: short(r["artifact_sha256"])),
            ("commit", commit),
            ("host", lambda r: r["host"]),
            ("seconds", lambda r: secs(r["seconds"])),
            ("rc", lambda r: short(r["rc"])),
        ]
        for label, get in fields:
            va, vb = get(run_a), get(run_b)
            mark = "*" if va != vb and label not in ("run", "started") else " "
            print(f"{mark} {label:<9}{va:<32}{vb}")

        print(f"\n  {'case':<14}{'A':<9}{'B':<9}{'secs A':>8}{'secs B':>8}{'delta':>9}  tok/s, outputs")
        for row in self._diff_rows(run_a["id"], run_b["id"]):
            mark = " " if row["status_a"] == row["status_b"] else "*"
            print(
                f"{mark} {row['case']:<14}{row['status_a']:<9}{row['status_b']:<9}"
                f"{secs(row['secs_a']):>8}{secs(row['secs_b']):>8}{row['delta']:>9}  {', '.join(row['notes'])}".rstrip()
            )
        return 0

    # rwt.py's tokens/s is end to end; rat.py's generation rate excludes prompt
    # time. Different names keep the two from being compared.
    RATES: tuple[tuple[str, str], ...] = (
        ("tokens_per_second", "tok/s"),
        ("generation_tokens_per_second", "gen tok/s"),
    )

    def _diff_rows(self, id_a: int, id_b: int) -> list[dict[str, Any]]:
        """One row per case of runs `id_a` and `id_b`: statuses, seconds, notes."""

        def secs(value: float | None) -> str:
            return "-" if value is None else f"{value:.1f}"

        cases_a, cases_b = self._cases(id_a), self._cases(id_b)
        rows = []
        for key in [*cases_a, *(k for k in cases_b if k not in cases_a)]:
            ca, outs_a, mets_a = cases_a.get(key, (None, {}, {}))
            cb, outs_b, mets_b = cases_b.get(key, (None, {}, {}))
            sa = ca["seconds"] if ca is not None else None
            sb = cb["seconds"] if cb is not None else None
            notes = []
            for metric, label in self.RATES:
                ta, tb = mets_a.get(metric), mets_b.get(metric)
                if ta is not None or tb is not None:
                    change = f" ({100 * (tb - ta) / ta:+.1f}%)" if ta and tb is not None else ""
                    notes.append(f"{label} {secs(ta)} -> {secs(tb)}{change}")
            for name in sorted({*outs_a, *outs_b}):
                oa, ob = outs_a.get(name), outs_b.get(name)
                if oa is None or ob is None:
                    notes.append(f"{name} {'new' if oa is None else 'missing'}")
                elif oa["sha256"] != ob["sha256"]:
                    notes.append(f"{name} changed ({oa['bytes']} -> {ob['bytes']} bytes)")
                else:
                    notes.append(f"{name} identical")
            rows.append(
                {
                    "case": " ".join(key),
                    "status_a": ca["status"] if ca is not None else "-",
                    "status_b": cb["status"] if cb is not None else "-",
                    "secs_a": sa,
                    "secs_b": sb,
                    "delta": f"{100 * (sb - sa) / sa:+.1f}%" if sa and sb is not None else "",
                    "notes": notes,
                }
            )
        return rows

    # -- html report --------------------------------------------------------

    REPORT_CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb; --surface-2: #f3f2ef; --border: #e2e1dc;
  --text: #0b0b0b; --text-2: #52514e; --text-3: #7a7974;
  --series: #2a78d6; --grid: #e8e7e3;
  --good: #006300; --critical: #b52f2f;
  --b-cuda: #2a78d6; --b-vulkan: #eb6834; --b-cpu: #1baf7a; --b-metal: #eda100; --b-rocm: #e87ba4; --b-sycl: #008300;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --surface-2: #232322; --border: #3a3a37;
    --text: #ffffff; --text-2: #c3c2b7; --text-3: #8f8e86;
    --series: #3987e5; --grid: #2e2e2c;
    --good: #4fbf4f; --critical: #e66767;
    --b-cuda: #3987e5; --b-vulkan: #d95926; --b-cpu: #199e70; --b-metal: #c98500; --b-rocm: #d55181; --b-sycl: #008300;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --surface-2: #232322; --border: #3a3a37;
  --text: #ffffff; --text-2: #c3c2b7; --text-3: #8f8e86;
  --series: #3987e5; --grid: #2e2e2c;
  --good: #4fbf4f; --critical: #e66767;
  --b-cuda: #3987e5; --b-vulkan: #d95926; --b-cpu: #199e70; --b-metal: #c98500; --b-rocm: #d55181; --b-sycl: #008300;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 48px; background: var(--surface); color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1200px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 40px 0 4px; padding-top: 16px; border-top: 1px solid var(--border); }
h3 { font-size: 14px; margin: 20px 0 8px; color: var(--text-2); font-weight: 600; }
.meta { color: var(--text-2); margin: 0 0 16px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
th, td { padding: 4px 10px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--text-2); font-weight: 600; }
td.num, th.num { text-align: right; }
td.notes { white-space: normal; min-width: 240px; color: var(--text-2); }
.pass { color: var(--good); } .fail, .timeout { color: var(--critical); font-weight: 600; }
.skip, .none { color: var(--text-3); }
.changed { background: var(--surface-2); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }
figure { margin: 0; padding: 10px 12px 6px; background: var(--surface-2); border-radius: 8px; }
figcaption { font-size: 13px; font-weight: 600; }
figcaption span { color: var(--text-2); font-weight: 400; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 6px 0 10px; font-size: 12px; color: var(--text-2); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.cmp-row { display: grid; grid-template-columns: 90px 1fr; gap: 10px; padding: 5px 0; border-top: 1px solid var(--border); }
.cmp-case { font-size: 13px; padding-top: 1px; }
.cmp-bar { display: flex; align-items: center; gap: 6px; height: 16px; font-size: 11px;
  color: var(--text-2); font-variant-numeric: tabular-nums; }
.cmp-bar + .cmp-bar { margin-top: 2px; }
.regression, .now-failing { color: var(--critical); font-weight: 600; }
.improvement, .now-passing { color: var(--good); font-weight: 600; }
.within-noise { color: var(--text-3); }
.scroll + .meta { margin-top: 12px; }
tr.flag td { background: var(--surface-2); }
.cmp-bar .fill { height: 10px; border-radius: 0 3px 3px 0; min-width: 2px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg .axis { fill: var(--text-3); font-size: 10px; }
svg .gridline { stroke: var(--grid); stroke-width: 1; }
svg .release { stroke: var(--text-3); stroke-width: 1; stroke-dasharray: 3 3; }
svg .line { fill: none; stroke: var(--series); stroke-width: 2; stroke-linejoin: round; }
svg .dot { fill: var(--series); stroke: var(--surface-2); stroke-width: 2; }
svg .hit { fill: transparent; cursor: default; }
svg .hit:hover + .dot, svg .dot.on { r: 6; }
#tip { position: fixed; pointer-events: none; display: none; z-index: 10; padding: 6px 8px;
  background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 6px;
  font-size: 12px; white-space: pre; box-shadow: 0 2px 8px rgb(0 0 0 / 0.15); }
"""

    REPORT_JS = """
const tip = document.getElementById("tip");
document.addEventListener("mouseover", (e) => {
  const t = e.target.closest(".hit");
  if (!t) return;
  tip.textContent = t.dataset.tip;
  tip.style.display = "block";
});
document.addEventListener("mousemove", (e) => {
  if (tip.style.display !== "block") return;
  const x = Math.min(e.clientX + 12, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + "px";
  tip.style.top = (e.clientY + 14) + "px";
});
document.addEventListener("mouseout", (e) => {
  if (e.target.closest(".hit")) tip.style.display = "none";
});
"""

    @staticmethod
    def _svg_trend(points: list[tuple[str, float | None, str, str]], unit: str) -> str:
        """Line chart of one measure over runs; each point is (x label, value, tooltip,
        version). A None value (a failed or skipped case) breaks the line rather than
        plotting a time that measures nothing. A dashed line marks each version change."""
        w, h, left, right, top, bottom = 300, 140, 40, 8, 18, 20
        values = [p[1] for p in points if p[1] is not None]
        # Two gridline steps, each 1, 2, 2.5 or 5 times a power of ten.
        raw = max(max(values), 1e-9) / 2
        mag = 10 ** math.floor(math.log10(raw))
        step = next(f * mag for f in (1, 2, 2.5, 5, 10) if f * mag >= raw)
        top_value = 2 * step
        n = len(points)

        def x(i: int) -> float:
            return left + (w - left - right) * (i / (n - 1) if n > 1 else 0.5)

        def y(v: float) -> float:
            return top + (h - top - bottom) * (1 - v / top_value)

        parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(unit)} per run">']
        for v in (0, step, top_value):
            parts.append(f'<line class="gridline" x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
            parts.append(f'<text class="axis" x="{left - 6}" y="{y(v) + 3:.1f}" text-anchor="end">{v:g}</text>')
        for i in range(1, n):
            if points[i][3] != points[i - 1][3]:
                xv = (x(i - 1) + x(i)) / 2
                parts.append(f'<line class="release" x1="{xv:.1f}" x2="{xv:.1f}" y1="{top - 4}" y2="{h - bottom}"/>')
                parts.append(
                    f'<text class="axis" x="{xv + 3:.1f}" y="{top - 6}">{html.escape(points[i][3][:16])}</text>'
                )
        for i in {0, n - 1}:
            parts.append(
                f'<text class="axis" x="{x(i):.1f}" y="{h - 4}" text-anchor="middle">{html.escape(points[i][0])}</text>'
            )
        segment: list[str] = []
        for i, (_, v, _, _) in enumerate([*points, ("", None, "", "")]):
            if v is not None:
                segment.append(f"{x(i):.1f},{y(v):.1f}")
            elif segment:
                if len(segment) > 1:
                    parts.append(f'<polyline class="line" points="{" ".join(segment)}"/>')
                segment = []
        for i, (_, v, tip, _) in enumerate(points):
            if v is not None:
                parts.append(
                    f'<circle class="hit" cx="{x(i):.1f}" cy="{y(v):.1f}" r="11" data-tip="{html.escape(tip)}"/>'
                )
                parts.append(f'<circle class="dot" cx="{x(i):.1f}" cy="{y(v):.1f}" r="4"/>')
        parts.append("</svg>")
        return "".join(parts)

    # A version is a regression (or improvement) on a measure when its median moves
    # by at least this much AND lands outside the range of the previous version's
    # runs. The second condition keeps one noisy run from deciding the verdict.
    REGRESSION_PCT = 10.0

    @staticmethod
    def _version_key(version: str) -> tuple[int, ...]:
        """Numeric parts of `version`, for ordering: "0.10.1" sorts above "0.9.3".
        Pre-release suffixes are not understood ("0.6.0rc1" sorts above "0.6.0")."""
        return tuple(int(part) for part in re.findall(r"\d+", version))

    def _version_rows(self, group: list[sqlite3.Row]) -> tuple[str, str, int, int, list[dict[str, Any]]] | None:
        """Compare the newest version in `group` (finished runs of one project,
        backend and target) with the next version below it.

        Versions are ordered by number, not by when they were tested, so a
        baseline recorded after the release it precedes still compares the
        right way round. Returns (previous, current, runs of previous, runs of
        current, rows), or None when the group has one version. Each row is one
        case and measure.
        """
        versions = sorted({r["version"] or "?" for r in group}, key=self._version_key)
        if len(versions) < 2:
            return None
        previous, current = versions[-2], versions[-1]
        runs_prev = [r for r in group if (r["version"] or "?") == previous]
        runs_cur = [r for r in group if (r["version"] or "?") == current]
        cases_prev = [self._cases(r["id"]) for r in runs_prev]
        cases_cur = [self._cases(r["id"]) for r in runs_cur]
        keys = list(dict.fromkeys(k for c in [*cases_cur, *cases_prev] for k in c))

        # (label, metric or None for seconds, higher is better)
        measures: list[tuple[str, str | None, bool]] = [
            ("seconds", None, False),
            *((label, metric, True) for metric, label in self.RATES),
        ]
        rows: list[dict[str, Any]] = []
        for key in keys:
            entries_prev = [c[key] for c in cases_prev if key in c]
            entries_cur = [c[key] for c in cases_cur if key in c]
            passed_prev = any(e[0]["status"] == "pass" for e in entries_prev)
            passed_cur = any(e[0]["status"] == "pass" for e in entries_cur)
            hashes_prev = {o["sha256"] for e in entries_prev for o in e[1].values()}
            hashes_cur = {o["sha256"] for e in entries_cur for o in e[1].values()}
            note = "image changed" if hashes_prev and hashes_cur and hashes_prev != hashes_cur else ""
            if entries_prev and entries_cur and passed_prev != passed_cur:
                rows.append(
                    {
                        "case": " ".join(key),
                        "measure": "status",
                        "prev": None,
                        "cur": None,
                        "n_prev": len(entries_prev),
                        "n_cur": len(entries_cur),
                        "delta": None,
                        "verdict": "now failing" if passed_prev else "now passing",
                        "note": note,
                    }
                )
                continue
            for label, metric, higher_better in measures:

                def values(entries: list[Any], metric: str | None = metric) -> list[float]:
                    out = []
                    for c, _, m in entries:
                        v = c["seconds"] if metric is None else m.get(metric)
                        if c["status"] == "pass" and v is not None:
                            out.append(v)
                    return out

                vp, vc = values(entries_prev), values(entries_cur)
                if not vp or not vc:
                    continue
                a, b = statistics.median(vp), statistics.median(vc)
                delta = 100 * (b - a) / a if a else 0.0
                worse = delta < 0 if higher_better else delta > 0
                outside = b < min(vp) or b > max(vp)
                if abs(delta) >= self.REGRESSION_PCT and outside:
                    verdict = "regression" if worse else "improvement"
                else:
                    verdict = "within noise"
                rows.append(
                    {
                        "case": " ".join(key),
                        "measure": label,
                        "prev": a,
                        "cur": b,
                        "n_prev": len(vp),
                        "n_cur": len(vc),
                        "delta": delta,
                        "verdict": verdict,
                        "note": note if metric is None else "",
                    }
                )
        return previous, current, len(runs_prev), len(runs_cur), rows

    # Backend -> CSS colour token. Colour follows the backend, never its position
    # among the backends a chart happens to show.
    BACKEND_ORDER: tuple[str, ...] = ("cuda", "vulkan", "cpu", "metal", "rocm", "sycl")

    @classmethod
    def _html_backends(
        cls,
        title: str,
        note: str,
        runs: dict[str, sqlite3.Row],
        rows: list[tuple[str, dict[str, tuple[float | None, str]]]],
    ) -> str:
        """Grouped horizontal bars: one row per case, one bar per backend. Each row
        is scaled to its own longest bar; the value label carries the magnitude."""
        esc = html.escape
        backends = sorted(runs, key=lambda b: cls.BACKEND_ORDER.index(b) if b in cls.BACKEND_ORDER else 99)
        parts = [f"<figure><figcaption>{esc(title)} <span>{esc(note)}</span></figcaption>", '<div class="legend">']
        for b in backends:
            r = runs[b]
            parts.append(
                f'<span><i style="background: var(--b-{esc(b)}, var(--text-3))"></i>{esc(b)} '
                f"(run {r['id']}, {esc(r['version'] or '?')})</span>"
            )
        parts.append("</div>")
        for case, values in rows:
            top = max((v for v, _ in values.values() if v is not None), default=0.0) or 1.0
            parts.append(f'<div class="cmp-row"><div class="cmp-case">{esc(case)}</div><div>')
            for b in backends:
                value, status = values.get(b, (None, "not run"))
                if value is None:
                    tip = f"{case}  {b}\n{status}"
                    parts.append(f'<div class="cmp-bar hit" data-tip="{esc(tip)}">{esc(status)}</div>')
                    continue
                tip = f"{case}  {b}\nrun {runs[b]['id']}  {runs[b]['version'] or '?'}\n{value:.2f}"
                parts.append(
                    f'<div class="cmp-bar hit" data-tip="{esc(tip)}"><span class="fill" '
                    f'style="width: calc((100% - 48px) * {value / top:.4f}); background: var(--b-{esc(b)}, var(--text-3))">'
                    f"</span>{value:.1f}</div>"
                )
            parts.append("</div></div>")
        parts.append("</figure>")
        return "".join(parts)

    def _html_versions(self, runs: list[sqlite3.Row]) -> list[str]:
        """The "version over version" section: per project, backend and target,
        the latest version against the one tested before it."""
        esc = html.escape
        groups: dict[tuple[str, str, str], list[sqlite3.Row]] = {}
        for r in reversed(runs):  # oldest first
            if r["finished_at"] is not None:
                groups.setdefault((r["project"], r["backend"], r["target"]), []).append(r)
        compared, single = [], []
        for (project, backend, target), group in groups.items():
            result = self._version_rows(group)
            name = f"{project} / {backend} / {target}"
            if result is None:
                single.append(f"{name} ({group[-1]['version'] or '?'}, {len(group)} run{'s' * (len(group) != 1)})")
            else:
                compared.append((name, result))

        out = [
            "<h2>Version over version</h2>",
            f'<p class="meta">Same project, backend and target; the newest version against the next one below '
            f"it, by version number. Medians over passing runs. A regression or improvement moves the median by at least "
            f"{self.REGRESSION_PCT:g}% and leaves the range of the previous version's runs. Builds that share a "
            f"version string count as one version.</p>",
        ]
        if compared:
            flagged = [
                f"{name}: {sum(r['verdict'] in ('regression', 'now failing') for r in rows)} regression(s)"
                for name, (_, _, _, _, rows) in compared
                if any(r["verdict"] in ("regression", "now failing") for r in rows)
            ]
            out.append(
                '<p class="regression">' + esc("; ".join(flagged)) + "</p>"
                if flagged
                else '<p class="improvement">No regressions.</p>'
            )
        for name, (previous, current, n_prev, n_cur, rows) in compared:
            out.append(
                f"<h3>{esc(name)}: {esc(previous)} ({n_prev} run{'s' * (n_prev != 1)}) &rarr; "
                f"{esc(current)} ({n_cur} run{'s' * (n_cur != 1)})</h3>"
            )
            out.append(
                '<div class="scroll"><table><tr><th>case</th><th>measure</th>'
                f'<th class="num">{esc(previous)}</th><th class="num">{esc(current)}</th>'
                '<th class="num">change</th><th>verdict</th><th>note</th></tr>'
            )
            for r in rows:
                cls = r["verdict"].replace(" ", "-")
                flag = ' class="flag"' if r["verdict"] != "within noise" else ""
                prev = "-" if r["prev"] is None else f"{r['prev']:.2f} <small>(n={r['n_prev']})</small>"
                cur = "-" if r["cur"] is None else f"{r['cur']:.2f} <small>(n={r['n_cur']})</small>"
                delta = "" if r["delta"] is None else f"{r['delta']:+.1f}%"
                out.append(
                    f"<tr{flag}><td>{esc(r['case'])}</td><td>{esc(r['measure'])}</td>"
                    f'<td class="num">{prev}</td><td class="num">{cur}</td><td class="num">{delta}</td>'
                    f'<td class="{cls}">{esc(r["verdict"])}</td><td class="notes">{esc(r["note"])}</td></tr>'
                )
            out.append("</table></div>")
        if single:
            out.append(
                '<p class="meta">One version recorded so far, so nothing to compare: '
                + esc("; ".join(single))
                + ". Test the next version on the same backend and target to compare.</p>"
            )
        return out

    def write_report(self, out: Path, limit: int, backend: str | None = None, all_projects: bool = False) -> bool:
        """Write a self-contained HTML report: recent runs, then per project,
        backend and target the latest-vs-previous diff and a trend per case.
        Returns False, writing nothing, when there are no runs to report."""
        esc = html.escape
        runs = self._query(
            "SELECT * FROM runs WHERE (? OR project = ?) AND (? IS NULL OR backend = ?) ORDER BY id DESC",
            (all_projects, self.project, backend, backend),
        )
        if not runs:
            print(f"no runs recorded in {self.path}")
            return False

        def cls(status: str) -> str:
            return status if status in ("pass", "fail", "timeout", "skip") else "none"

        def secs(value: float | None) -> str:
            return "-" if value is None else f"{value:.1f}"

        body = [
            "<h1>Run history</h1>",
            f'<p class="meta">{esc(str(self.path))} &middot; generated {esc(self.now())} &middot; '
            f"{len(runs)} runs{'' if all_projects else ' of ' + esc(self.project)}</p>",
            *self._html_versions(runs),
            "<h3>Recent runs</h3>",
            '<div class="scroll"><table><tr><th class="num">id</th><th>started (UTC)</th><th>project</th>'
            '<th>backend</th><th>version</th><th>target</th><th>commit</th><th class="num">passed</th>'
            '<th class="num">secs</th><th>result</th></tr>',
        ]
        counts = {
            r["run_id"]: (r["ran"], r["passed"])
            for r in self._query(
                "SELECT run_id, COUNT(*) AS ran, SUM(status = 'pass') AS passed FROM cases GROUP BY run_id"
            )
        }
        for r in runs[:limit]:
            ran, passed = counts.get(r["id"], (0, 0))
            result = (
                "running or interrupted" if r["rc"] is None else ("pass" if r["rc"] == 0 else f"fail (rc={r['rc']})")
            )
            commit = (r["git_commit"] or "-")[:10] + ("+dirty" if r["git_dirty"] else "")
            body.append(
                f'<tr><td class="num">{r["id"]}</td><td>{esc(r["started_at"][:19])}</td><td>{esc(r["project"])}</td>'
                f"<td>{esc(r['backend'])}</td><td>{esc(r['version'] or '?')}</td><td>{esc(r['target'])}</td>"
                f'<td>{esc(commit)}</td><td class="num">{passed}/{ran}</td><td class="num">{secs(r["seconds"])}</td>'
                f'<td class="{cls("pass" if r["rc"] == 0 else "none" if r["rc"] is None else "fail")}">{esc(result)}</td></tr>'
            )
        body.append("</table></div>")

        # Latest finished run of each backend, per project and target.
        latest: dict[tuple[str, str], dict[str, sqlite3.Row]] = {}
        for r in runs:  # newest first, so the first run seen per backend is its latest
            if r["finished_at"] is not None:
                latest.setdefault((r["project"], r["target"]), {}).setdefault(r["backend"], r)
        for (project, target), by_backend in latest.items():
            if len(by_backend) < 2:
                continue
            cases = {b: self._cases(r["id"]) for b, r in by_backend.items()}
            keys = list(dict.fromkeys(k for c in cases.values() for k in c))
            body.append(f"<h2>{esc(project)} &middot; {esc(target)} &middot; backends side by side</h2>")
            body.append('<div class="grid">')
            measures: list[tuple[str, str, str | None]] = [
                ("seconds", "lower is faster", None),
                *((label, "higher is faster", metric) for metric, label in self.RATES),
            ]
            for title, direction, metric in measures:
                rows = []
                for key in keys:
                    values: dict[str, tuple[float | None, str]] = {}
                    for b, c in cases.items():
                        entry = c.get(key)
                        if entry is None:
                            values[b] = (None, "not run")
                        elif entry[0]["status"] != "pass":
                            values[b] = (None, entry[0]["status"])
                        else:
                            value = entry[0]["seconds"] if metric is None else entry[2].get(metric)
                            values[b] = (value, "pass" if value is not None else "no value")
                    if metric is not None and all(v is None for v, _ in values.values()):
                        continue  # no backend recorded this metric for the case
                    rows.append((" ".join(key), values))
                if rows:
                    body.append(
                        self._html_backends(
                            f"{title}", f"latest run per backend; {direction}; bars scaled per case", by_backend, rows
                        )
                    )
            body.append("</div>")

        groups: dict[tuple[str, str, str], list[sqlite3.Row]] = {}
        for r in runs:
            if r["finished_at"] is not None:
                groups.setdefault((r["project"], r["backend"], r["target"]), []).append(r)
        for (project, run_backend, target), group in groups.items():
            group = list(reversed(group[:limit]))  # oldest first, for the trend
            latest = group[-1]
            body.append(f"<h2>{esc(project)} &middot; {esc(run_backend)} &middot; {esc(target)}</h2>")
            body.append(
                f'<p class="meta">{len(group)} finished runs shown &middot; latest {esc(latest["version"] or "?")} '
                f"on {esc(latest['host'])}, {esc(latest['started_at'][:19])} UTC</p>"
            )
            if len(group) < 2:
                body.append(
                    '<p class="meta">One finished run. The diff and trend charts appear after the next run '
                    "of this project, backend and target.</p>"
                )
            else:
                prev = group[-2]
                body.append(f"<h3>Run {latest['id']} vs run {prev['id']}</h3>")
                body.append(
                    '<div class="scroll"><table><tr><th>case</th>'
                    f'<th>run {prev["id"]}</th><th>run {latest["id"]}</th><th class="num">secs {prev["id"]}</th>'
                    f'<th class="num">secs {latest["id"]}</th><th class="num">delta</th><th>tok/s, outputs</th></tr>'
                )
                for row in self._diff_rows(prev["id"], latest["id"]):
                    changed = ' class="changed"' if row["status_a"] != row["status_b"] else ""
                    body.append(
                        f"<tr{changed}><td>{esc(row['case'])}</td>"
                        f'<td class="{cls(row["status_a"])}">{esc(row["status_a"])}</td>'
                        f'<td class="{cls(row["status_b"])}">{esc(row["status_b"])}</td>'
                        f'<td class="num">{secs(row["secs_a"])}</td><td class="num">{secs(row["secs_b"])}</td>'
                        f'<td class="num">{esc(row["delta"])}</td><td class="notes">{esc(", ".join(row["notes"]))}</td></tr>'
                    )
                body.append("</table></div>")

            per_run = [(r, self._cases(r["id"])) for r in group]
            keys = list(per_run[-1][1])
            figures = []
            for key in keys:
                series: list[tuple[str, str, Callable[[Any, dict[str, float]], float | None]]] = [
                    ("seconds", "s", lambda c, m: c["seconds"] if c["status"] == "pass" else None),
                    *(
                        (label, label, lambda c, m, metric=metric: m.get(metric) if c["status"] == "pass" else None)
                        for metric, label in self.RATES
                    ),
                ]
                for title, unit, get in series:
                    points: list[tuple[str, float | None, str, str]] = []
                    for r, cases in per_run:
                        entry = cases.get(key)
                        value = get(entry[0], entry[2]) if entry is not None else None
                        status = entry[0]["status"] if entry is not None else "not run"
                        hashes = ", ".join(f"{n} {o['sha256'][:8]}" for n, o in entry[1].items()) if entry else ""
                        tip = (
                            f"run {r['id']}  {r['version'] or '?'}\n{r['started_at'][:19]} UTC\n"
                            + (f"{value:.2f} {unit}" if value is not None else status)
                            + (f"\n{hashes}" if hashes else "")
                        )
                        points.append((str(r["id"]), value, tip, r["version"] or "?"))
                    if sum(p[1] is not None for p in points) < 2:
                        continue
                    figures.append(
                        f"<figure><figcaption>{esc(' '.join(key))} <span>{esc(title)}</span></figcaption>"
                        f"{self._svg_trend(points, unit)}</figure>"
                    )
            if figures:
                body.append(
                    "<h3>Trend per case (x: run id; dashed line: new version; failed and skipped runs leave a gap)</h3>"
                )
                body.append(f'<div class="grid">{"".join(figures)}</div>')

        page = (
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Run history</title><style>{self.REPORT_CSS}</style></head>"
            f'<body><main>{"".join(body)}</main><div id="tip" role="tooltip"></div>'
            f"<script>{self.REPORT_JS}</script></body></html>\n"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page, encoding="utf-8")
        print(f"wrote {out}")
        return True


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


@dataclass
class Paths:
    """The directory layout every other object resolves against."""

    root: Path
    # `build/` rather than the project root as the base for everything this
    # script creates: it is already gitignored and already the directory
    # `make clean` sweeps -- a smoke run should not leave anything for the next
    # `git status` to report, and throwing away a tested binary along with the
    # build tree is the right default. `models/` is the deliberate exception:
    # it stays at the root, shared with `make test`, because re-downloading
    # tens of GiB of weights after every `make clean` is not.
    build_dir: Path
    # One directory under it holds this script's whole footprint -- the
    # installed binary at `<rat_dir>/chimera`, everything a run produces in
    # `<rat_dir>/out` -- so it is obvious at a glance what rat.py owns inside a
    # build tree it shares with cmake.
    rat_dir: Path
    models_dir: Path
    data_dir: Path
    bin_dir: Path
    out_dir: Path

    @staticmethod
    def find_root() -> Path:
        """Locate the project root: the cwd for subprocesses, the parent of
        ``models/``, and what ``build/`` is resolved against.

        This file is checked in as ``<repo>/scripts/rat.py`` but is also meant
        to be copied out standalone (as ``./rat.py``) into a bare directory
        that holds nothing but ``models/`` and a ``build/rat/``. Walking up to the
        nearest project marker handles both layouts; using ``__file__``'s own
        directory would resolve to ``<repo>/scripts`` in-repo and download
        models to ``scripts/models``.
        """
        here = Path(__file__).resolve().parent
        for candidate in (here, *here.parents):
            if (candidate / "CMakeLists.txt").exists() or (candidate / ".git").exists():
                return candidate
        return here

    @staticmethod
    def resolve_build_dir(root: Path) -> Path:
        """Where `build/` is. Honours ``BUILD_DIR`` the way the Makefile does
        (``BUILD_DIR ?= build``), so an out-of-tree build directory is named
        once and both agree on it."""
        raw = os.environ.get("CHIMERA_BUILD_DIR") or os.environ.get("BUILD_DIR") or "build"
        build = Path(raw).expanduser()
        return build if build.is_absolute() else root / build

    @classmethod
    def from_environ(cls) -> Paths:
        root = cls.find_root()
        build = cls.resolve_build_dir(root)
        rat = Path(os.environ.get("CHIMERA_RAT_DIR", build / "rat"))
        return cls(
            root=root,
            build_dir=build,
            rat_dir=rat,
            models_dir=Path(os.environ.get("CHIMERA_MODELS_DIR", root / "models")),
            data_dir=Path(os.environ.get("CHIMERA_DATA_DIR", build / "whisper.cpp" / "samples")),
            bin_dir=Path(os.environ.get("CHIMERA_BIN_DIR", rat)),
            out_dir=Path(os.environ.get("CHIMERA_RAT_OUT", rat / "out")),
        )

    def rebase(self, build: Path) -> None:
        """Re-point every build-relative default at `build`.

        Called when --build-dir moves it after construction. Only the defaults
        move: a directory the caller named outright with --bin-dir / --out-dir /
        --data-dir is applied afterwards and wins.
        """
        old = self.build_dir
        self.build_dir = build
        for attr in ("rat_dir", "data_dir", "bin_dir", "out_dir"):
            current = getattr(self, attr)
            try:
                setattr(self, attr, build / current.relative_to(old))
            except ValueError:
                pass  # explicitly set elsewhere (env var); leave it alone

    @property
    def cache_dir(self) -> Path:
        """Where downloaded release archives are kept.

        A sibling of `out_dir`, not a child: an archive is the *input* to a run
        -- the artifact under test -- while `out/` is what a run produced.
        """
        return self.rat_dir / "downloads"

    @property
    def data_dirs(self) -> list[Path]:
        # jfk.wav is not checked in: it arrives under <build>/whisper.cpp/samples
        # when `make deps` fetches the vendored whisper.cpp tree. Standalone
        # there is no such tree, and ModelRegistry downloads it instead.
        return [
            self.data_dir,
            self.build_dir / "whisper.cpp" / "samples",
            self.root / "tests" / "media",
        ]

    def find_data_asset(self, name: str) -> Path | None:
        """First existing copy of `name` in --data-dir or the checkout's data dirs."""
        for d in self.data_dirs:
            candidate = d / name
            if candidate.exists():
                return candidate
        return None


# ---------------------------------------------------------------------------
# release assets
# ---------------------------------------------------------------------------


class Release:
    """How chimera's release assets are named, resolved and unpacked.

    The names come straight from the two release workflows: `release.yml`
    stages ``chimera-<version>-<target>.{tar.gz,zip}`` for the CPU/Metal
    matrix, and `release-gpu.yml` appends the backend to the target for each
    GPU leg (``linux-x86_64-cuda``, ``windows-x86_64-vulkan``, ...). Each
    archive contains exactly one member: the bare ``chimera`` binary.
    """

    DEFAULT_REPO = "shakfu/chimera"
    API = "https://api.github.com/repos/{repo}/releases/{ref}"
    DOWNLOAD = "https://github.com/{repo}/releases/download/{tag}/{asset}"

    # (backend, os, arch) -> release target. Only the pairs CI actually
    # publishes are here; anything else is an AssetUnavailable rather than a
    # 404 halfway through a download. `cpu` and `metal` share the macOS entry
    # because there is no CPU-only macOS build to tell them apart.
    TARGETS: dict[tuple[str, str, str], str] = {
        ("cpu", "linux", "x86_64"): "linux-x86_64",
        ("cpu", "windows", "x86_64"): "windows-x86_64",
        ("cpu", "macos", "arm64"): "macos-arm64",
        ("metal", "macos", "arm64"): "macos-arm64",
        ("cuda", "linux", "x86_64"): "linux-x86_64-cuda",
        ("cuda", "windows", "x86_64"): "windows-x86_64-cuda",
        ("vulkan", "linux", "x86_64"): "linux-x86_64-vulkan",
        ("vulkan", "windows", "x86_64"): "windows-x86_64-vulkan",
        ("rocm", "linux", "x86_64"): "linux-x86_64-rocm",
        ("sycl", "linux", "x86_64"): "linux-x86_64-sycl",
    }

    def __init__(self, repo: str | None = None) -> None:
        self.repo = repo or os.environ.get("CHIMERA_RAT_REPO", self.DEFAULT_REPO)

    # -- naming -------------------------------------------------------------

    @staticmethod
    def host() -> tuple[str, str]:
        """(os, arch) in the spelling :attr:`TARGETS` uses."""
        system = {"Darwin": "macos", "Windows": "windows", "Linux": "linux"}.get(platform.system(), "linux")
        machine = platform.machine().lower()
        arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
        return system, arch

    @classmethod
    def target_for(cls, backend: str) -> str:
        system, arch = cls.host()
        try:
            return cls.TARGETS[(backend, system, arch)]
        except KeyError:
            published = sorted({b for b, o, a in cls.TARGETS if (o, a) == (system, arch)})
            raise AssetUnavailable(
                f"no '{backend}' release is published for {system}-{arch} "
                f"(available here: {', '.join(published) or 'none'})"
            ) from None

    @classmethod
    def asset_name(cls, backend: str, version: str) -> str:
        target = cls.target_for(backend)
        # Windows gets a .zip (what Windows users expect); everyone else a
        # .tar.gz -- matching how the workflows package each leg.
        ext = "zip" if target.startswith("windows") else "tar.gz"
        return f"chimera-{version}-{target}.{ext}"

    # -- resolution ---------------------------------------------------------

    def _api(self, ref: str) -> dict[str, Any]:
        url = self.API.format(repo=self.repo, ref=ref)
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        # Anonymous API calls are rate-limited to 60/hour per IP, which a CI
        # matrix can exhaust. Use a token when the environment offers one.
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req) as r:
            return json.load(r)

    def latest_tag(self) -> str:
        try:
            return str(self._api("latest")["tag_name"])
        except (urllib.error.URLError, KeyError, ValueError) as e:
            raise AssetUnavailable(f"could not resolve the latest release of {self.repo}: {e}") from None

    def url_for(self, backend: str, version: str, tag: str | None = None) -> str:
        # Tags have been pushed both as `0.2.16` and `v0.2.16`; the workflows
        # strip a leading `v` for the filename, so the tag and the version in
        # the asset name are not interchangeable. Keep them separate.
        return self.DOWNLOAD.format(repo=self.repo, tag=tag or version, asset=self.asset_name(backend, version))

    # -- fetch + unpack -----------------------------------------------------

    @staticmethod
    def download(url: str, dest: Path) -> Path:
        print(f"downloading {url} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1024 * 1024)
        tmp.replace(dest)
        return dest

    @staticmethod
    def extract(archive: Path, bin_dir: Path) -> Path:
        """Unpack the single ``chimera`` member of `archive` into `bin_dir`.

        The member is looked up by basename rather than by index: the archives
        are flat today, but a future one that adds a README should still put
        the binary in the right place instead of unpacking whatever came first.
        """
        wanted = {"chimera", "chimera.exe"}
        bin_dir.mkdir(parents=True, exist_ok=True)

        if archive.suffix == ".zip" or zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                names = [n for n in zf.namelist() if Path(n).name in wanted]
                if not names:
                    raise AssetUnavailable(f"{archive.name} contains no chimera binary")
                name = names[0]
                dest = bin_dir / Path(name).name
                with zf.open(name) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
        else:
            with tarfile.open(archive) as tf:
                members = [m for m in tf.getmembers() if m.isfile() and Path(m.name).name in wanted]
                if not members:
                    raise AssetUnavailable(f"{archive.name} contains no chimera binary")
                member = members[0]
                dest = bin_dir / Path(member.name).name
                src = tf.extractfile(member)
                if src is None:
                    raise AssetUnavailable(f"could not read {member.name} from {archive.name}")
                with src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)

        # tar/zip modes are not carried over by the streaming copy above, and
        # an archive built on Windows has no POSIX mode to carry anyway.
        dest.chmod(dest.stat().st_mode | 0o755)
        print(f"installed {dest}")
        return dest


# ---------------------------------------------------------------------------
# the environment under test
# ---------------------------------------------------------------------------


class Env:
    """The chimera binary under test: where it lives, what it was built with,
    and how subprocesses are run against it.

    There is no virtualenv and no interpreter indirection -- the thing under
    test is one static executable, so `chimera` is invoked directly and the
    only question is *which* executable. ``bin_path`` answers that: an explicit
    ``--bin``, or ``<bin_dir>/chimera`` where ``install`` puts it.
    """

    BACKENDS: tuple[str, ...] = ("cpu", "metal", "cuda", "vulkan", "rocm", "sycl")

    # `chimera info` prints CHIMERA_BUILT_BACKENDS on its `built:` line, using
    # the labels CMakeLists.txt assigns to each GGML_* option. Map them back to
    # this script's backend names. BLAS is a CPU accelerator, not a backend a
    # release is cut for, so it reads as `cpu`.
    BUILT_LABELS: dict[str, str] = {
        "Metal": "metal",
        "CUDA": "cuda",
        "Vulkan": "vulkan",
        "HIP": "rocm",
        "SYCL": "sycl",
        "CPU": "cpu",
        "BLAS": "cpu",
    }

    # chimera's llama-side subcommands default to --gpu-layers 0, i.e. CPU,
    # even in a GPU build -- unlike `sd`, which picks up the GPU on its own.
    # A GPU release tested without this would pass every case while measuring
    # nothing but the CPU path, so the backend chooses the default.
    GPU_LAYERS_DEFAULT = 99

    def __init__(self, paths: Paths, bin_path: Path | None = None, gpu_layers: int | None = None) -> None:
        self.paths = paths
        # Explicit --bin; None means <bin_dir>/chimera.
        self._bin_path = bin_path
        # Explicit --gpu-layers; None means "derive from the backend".
        self.gpu_layers_override = gpu_layers
        # While set, `run` copies each child's stderr here as well as to ours.
        self.stderr_sink: bytearray | None = None
        # Binary path -> whether its `gen` accepts --stats.
        self._has_stats: dict[Path, bool] = {}

    # -- the binary ---------------------------------------------------------

    @property
    def exe_name(self) -> str:
        return "chimera.exe" if os.name == "nt" else "chimera"

    @property
    def bin_override(self) -> Path | None:
        """The `--bin` path, or None when the binary is the installed one.

        The distinction matters to `clean`, which may only delete a binary this
        script put there.
        """
        return self._bin_path

    @property
    def bin_path(self) -> Path:
        return self._bin_path if self._bin_path is not None else self.paths.bin_dir / self.exe_name

    @bin_path.setter
    def bin_path(self, value: Path | None) -> None:
        self._bin_path = value

    def gpu_layers(self, backend: str) -> int:
        if self.gpu_layers_override is not None:
            return self.gpu_layers_override
        return 0 if backend == "cpu" else self.GPU_LAYERS_DEFAULT

    def stats_args(self) -> list[str]:
        """``--stats`` if this binary's `gen` has it; releases before it reject the flag."""
        if self.bin_path not in self._has_stats:
            self._has_stats[self.bin_path] = "--stats" in self.capture(["gen", "--help"]).stdout
        return ["--stats"] if self._has_stats[self.bin_path] else []

    def gpu_args(self, backend: str) -> list[str]:
        """``--gpu-layers N`` for the subcommands that take it."""
        return ["--gpu-layers", str(self.gpu_layers(backend))]

    # -- subprocesses -------------------------------------------------------

    @staticmethod
    def _kill_tree(proc: "subprocess.Popen[bytes]") -> None:
        """Kill `proc` and every process it spawned.

        chimera is a single process rather than a re-execing launcher, so this
        is usually the same as ``proc.kill()`` -- but a timed-out image run
        that leaves anything behind holds several GiB of VRAM, and every later
        test in the matrix then OOMs or crawls, which silently invalidates the
        whole run's timings. Take the entire tree down instead.
        """
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        else:
            import signal

            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"warning: could not fully reap pid {proc.pid}", file=sys.stderr)

    def run(
        self,
        cmd: list[str],
        env: dict[str, str] | None = None,
        check: bool = False,
        timeout: float | None = None,
    ) -> int:
        """Run a subprocess; return the exit code.

        `check=False` is the default so callers can accumulate failures across
        a smoke-test matrix. Pass ``check=True`` for fail-fast behaviour.
        """
        print(f"$ {' '.join(cmd)}", flush=True)
        full_env = os.environ.copy()
        # Redirected stdout on Windows defaults to the ANSI codepage, and the
        # sd log callback emits byte-level BPE markers (U+0120, U+010A) that
        # cp1252 cannot encode. Force UTF-8 so a logged run matches a console one.
        full_env.setdefault("PYTHONIOENCODING", "utf-8")
        if env:
            full_env.update(env)
        sink = self.stderr_sink
        proc = subprocess.Popen(
            cmd,
            cwd=self.paths.root,
            env=full_env,
            start_new_session=os.name != "nt",
            stderr=subprocess.PIPE if sink is not None else None,
        )
        reader = None
        if sink is not None:
            reader = threading.Thread(target=self._tee, args=(proc.stderr, sink), daemon=True)
            reader.start()
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"error: command timed out after {timeout}s", file=sys.stderr)
            self._kill_tree(proc)
            rc = 124  # conventional timeout exit code
        if reader is not None:
            reader.join(timeout=10)
        if check and rc != 0:
            sys.exit(rc)
        return rc

    @staticmethod
    def _tee(pipe: Any, sink: bytearray) -> None:
        """Copy `pipe` to our stderr as it arrives, appending it to `sink`."""
        out = sys.stderr.buffer
        for chunk in iter(lambda: pipe.read1(1 << 16), b""):
            out.write(chunk)
            out.flush()
            sink += chunk

    def chimera(self, argv: list[str], env: dict[str, str] | None = None, timeout: float | None = None) -> int:
        return self.run([str(self.bin_path), *argv], env=env, timeout=timeout)

    def capture(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        """Run chimera and capture its output; for probes, not for test cases."""
        return subprocess.run(
            [str(self.bin_path), *argv],
            cwd=self.paths.root,
            capture_output=True,
            text=True,
            errors="replace",
        )

    # -- backend detection --------------------------------------------------

    def info_text(self) -> str | None:
        """``chimera info`` output, or None if the binary will not run."""
        if not self.bin_path.exists():
            return None
        proc = self.capture(["info"])
        return proc.stdout if proc.returncode == 0 else None

    @classmethod
    def parse_info(cls, text: str) -> tuple[str | None, str | None, str | None]:
        """(version, built backend, loaded backend) out of `chimera info`.

        `built:` is a compile-time constant baked into the binary and may be a
        comma-separated list (``Vulkan,BLAS``); the first entry that names a
        real backend wins. `loaded:` is what the ggml registry actually brought
        up on this host, and the two disagreeing is the interesting case -- a
        CUDA build on a box with no driver reports `built: CUDA, loaded: CPU`.
        """
        version = None
        if m := re.match(r"chimera\s+(\S+)", text):
            version = m.group(1)

        built = None
        if m := re.search(r"^\s*built:\s*(.+)$", text, re.M):
            for label in (s.strip() for s in m.group(1).split(",")):
                mapped = cls.BUILT_LABELS.get(label)
                if mapped and (built is None or built == "cpu"):
                    built = mapped

        loaded = None
        if m := re.search(r"^\s*loaded:\s*(.+)$", text, re.M):
            loaded = cls.BUILT_LABELS.get(m.group(1).strip())

        return version, built, loaded

    def detect(self) -> tuple[str | None, str | None, str | None]:
        text = self.info_text()
        return self.parse_info(text) if text else (None, None, None)

    def detect_backend(self) -> str | None:
        return self.detect()[1]

    def require_backend(self, requested: str | None) -> str:
        detected = self.detect_backend()
        if requested and detected and requested != detected:
            print(
                f"warning: requested backend '{requested}' but {self.bin_path} was built for '{detected}'",
                file=sys.stderr,
            )
        backend = requested or detected
        if not backend:
            flags = ",".join("--" + b for b in self.BACKENDS)
            print(
                f"error: no chimera binary at {self.bin_path}."
                f"\n  Install from a release: {SCRIPT_NAME} install {{{flags}}}"
                f"\n  ...or a local archive:  {SCRIPT_NAME} install --asset <path-or-url>"
                f"\n  ...or test a binary you already have: {SCRIPT_NAME} test --bin <path> ...",
                file=sys.stderr,
            )
            sys.exit(2)
        return backend

    def preflight(self, backend: str) -> str | None:
        """Run the binary once up front; return an error message, or None.

        A release archive built for the wrong glibc, or missing a runtime the
        host does not have, fails identically on every case in the matrix.
        Finding that out once, with the loader's own message attached, beats
        watching twelve cases die with the same opaque exit code.
        """
        if not self.bin_path.exists():
            return f"no chimera binary at {self.bin_path}"
        proc = self.capture(["info"])
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            tail = detail[-1] if detail else f"exit code {proc.returncode}"
            return f"cannot run {self.bin_path}: {tail}"

        _version, built, loaded = self.parse_info(proc.stdout)
        if built and built != "cpu" and loaded == "cpu":
            # Not fatal: the suite still runs, just on the CPU. Say so loudly,
            # because otherwise a green matrix looks like the GPU release works.
            print(
                f"warning: {self.bin_path} was built for '{built}' but the ggml registry loaded only CPU"
                f"\n  every case below will run on the CPU; check the driver / runtime for '{built}'",
                file=sys.stderr,
            )
        elif backend != "cpu" and loaded == "cpu":
            print(f"warning: '{backend}' requested but chimera info reports loaded: CPU", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------


@dataclass
class ModelSource:
    """Where to fetch a model from.

    One of repo_id (HF Hub) or url (direct http) must be set.
    """

    filename: str
    repo_id: str | None = None
    hf_filename: str | None = None  # defaults to filename
    url: str | None = None
    notes: str = ""

    def hub_filename(self) -> str:
        return self.hf_filename or self.filename


class ModelRegistry:
    """Known models and data assets, and how to get them onto disk."""

    JFK_WAV_URL = "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav"

    # Per-socket-operation timeout, not a deadline for the whole transfer: a
    # multi-GB model is minutes of 1 MiB reads, each of which must make
    # progress within this.
    NET_TIMEOUT = 60.0

    # Which tests need which models.
    SD_REQUIREMENTS: list[str] = ["z-image-turbo", "ae", "qwen3-4b"]
    RAG_REQUIREMENTS: list[str] = ["bge-small-en"]

    # One text per line -- the format `chimera index ingest -f` and the
    # per-line embed case both read. Deliberately includes a cluster about
    # mortality and one proper noun (Kilimanjaro), so the semantic and the
    # lexical retrieval legs each have something to rank.
    GENERATED_CORPUS: list[str] = [
        "The old man knew that he was dying, and he felt no fear of it.",
        "Death comes for everyone eventually, and grief is the price of having loved.",
        "Mourners gathered at the graveside in the cold morning air.",
        "He had spent his last years writing about mortality and the end of life.",
        "The hospice nurse spoke gently about what the final days would be like.",
        "Photosynthesis converts light energy into chemical energy stored in glucose.",
        "The compiler performs constant folding before emitting machine code.",
        "Mount Kilimanjaro is the highest free-standing mountain in the world.",
        "She sold the bakery and moved to a small town near the coast.",
        "Quicksort has an average time complexity of O(n log n).",
        "The bridge was rebuilt after the flood washed away its central span.",
        "A leopard was found frozen near the western summit of the mountain.",
        "Offloading model weights to the CPU trades throughput for VRAM headroom.",
    ]

    def __init__(self, paths: Paths, allow_download: bool = False) -> None:
        self.paths = paths
        self.allow_download = allow_download
        self.sources = self.default_sources()
        self.apply_env_overrides()

    @staticmethod
    def default_sources() -> dict[str, ModelSource]:
        """Best-effort defaults -- overridable via CHIMERA_MODEL_<KEY>=repo_id:file
        or by placing files in the models dir yourself. Use `list models` to inspect.
        """
        return {
            "llama-3.2-1b": ModelSource(
                filename="Llama-3.2-1B-Instruct-Q8_0.gguf",
                repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
                url="https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/resolve/main/llama-3.2-1b-instruct-q8_0.gguf",
            ),
            "qwen3-4b": ModelSource(
                filename="Qwen3-4B-Q8_0.gguf",
                repo_id="Qwen/Qwen3-4B-GGUF",
                url="https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
            ),
            "gemma-e4b": ModelSource(
                filename="gemma-4-E4B-it-Q5_K_M.gguf",
                repo_id="",  # override via env if/when available
                notes="set CHIMERA_MODEL_GEMMA_E4B=<repo_id>:<hf_filename> to enable download",
                url="https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q5_K_M.gguf",
            ),
            "z-image-turbo": ModelSource(
                filename="z_image_turbo-Q6_K.gguf",
                repo_id="",
                notes="set CHIMERA_MODEL_Z_IMAGE_TURBO=<repo_id>:<hf_filename> to enable download",
                url="https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/resolve/main/z-image-turbo-Q6_K.gguf",
            ),
            "ae": ModelSource(
                filename="ae.safetensors",
                repo_id="black-forest-labs/FLUX.1-schnell",
                hf_filename="ae.safetensors",
                url="https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
            ),
            "bge-small-en": ModelSource(
                filename="bge-small-en-v1.5-q8_0.gguf",
                repo_id="CompendiumLabs/bge-small-en-v1.5-gguf",
                url="https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf",
            ),
            "whisper-base-en": ModelSource(
                filename="ggml-base.en.bin",
                repo_id="ggerganov/whisper.cpp",
                url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
            ),
        }

    def apply_env_overrides(self) -> None:
        """Allow overriding repo ids via env vars (CHIMERA_MODEL_<KEY>=repo:file)."""
        for key, src in self.sources.items():
            env_key = "CHIMERA_MODEL_" + key.upper().replace("-", "_")
            val = os.environ.get(env_key)
            if not val:
                continue
            if ":" in val:
                repo, fname = val.split(":", 1)
                src.repo_id = repo
                src.hf_filename = fname
            else:
                src.repo_id = val

    # -- downloads ----------------------------------------------------------

    @staticmethod
    @contextlib.contextmanager
    def unavailable_on_failure(what: str) -> "Iterator[None]":
        """Re-raise any fetch failure as ModelSourceUnavailable, tagged with `what`.

        KeyboardInterrupt is a BaseException and passes through: ^C means stop
        the run, not skip this case.
        """
        try:
            yield
        except ModelSourceUnavailable:
            raise
        except urllib.error.HTTPError as e:
            raise ModelSourceUnavailable(f"{what}: HTTP {e.code} fetching {e.url}") from e
        except (urllib.error.URLError, OSError) as e:
            raise ModelSourceUnavailable(f"{what}: download failed: {e}") from e
        except Exception as e:  # huggingface_hub raises its own exception hierarchy
            raise ModelSourceUnavailable(f"{what}: download failed: {e}") from e

    @staticmethod
    def download_urllib(url: str, dest: Path) -> None:
        print(f"downloading {url} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        last_report = time.monotonic()
        bytes_read = 0
        chunk = 1024 * 1024  # 1 MiB
        try:
            with urllib.request.urlopen(url, timeout=ModelRegistry.NET_TIMEOUT) as r, open(tmp, "wb") as f:
                total_hdr = r.headers.get("Content-Length")
                total = int(total_hdr) if total_hdr and total_hdr.isdigit() else None
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    bytes_read += len(buf)
                    now = time.monotonic()
                    if now - last_report >= 2.0:
                        if total:
                            pct = 100.0 * bytes_read / total
                            print(
                                f"  {bytes_read / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)",
                                flush=True,
                            )
                        else:
                            print(f"  {bytes_read / 1e6:.1f} MB", flush=True)
                        last_report = now
        except BaseException:
            # A partial file must not be left where `ensure_model` would take it
            # for a finished download on the next run. Includes KeyboardInterrupt:
            # ^C during a multi-GB fetch is the common way this ends.
            tmp.unlink(missing_ok=True)
            raise
        tmp.rename(dest)

    @staticmethod
    def download_hf(repo_id: str, filename: str, dest: Path) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ModelSourceUnavailable(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            ) from e
        print(f"downloading {repo_id}:{filename} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Land the file directly in the models dir rather than copying from the
        # HF cache. Newer huggingface_hub uses `local_dir_use_symlinks=False`
        # and places the file at `<local_dir>/<filename>`; older releases
        # fall back to the cache path which we then copy.
        try:
            out = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(dest.parent),
                local_dir_use_symlinks=False,
            )
        except TypeError:
            # Older huggingface_hub without local_dir kwarg.
            out = hf_hub_download(repo_id=repo_id, filename=filename)
        out_path = Path(out)
        if out_path != dest:
            shutil.copyfile(out_path, dest)

    # -- lookups ------------------------------------------------------------

    def path_for(self, key: str) -> Path:
        return self.paths.models_dir / self.sources[key].filename

    def ensure_model(self, key: str) -> Path:
        """Path to a model already in the models dir, fetching it only if allowed.

        Raises ModelSourceUnavailable when the file is absent and this registry
        may not download (every path but the `download` subcommand), or when a
        permitted download fails.
        """
        src = self.sources[key]
        dest = self.path_for(key)
        if dest.exists():
            return dest
        if not self.allow_download:
            raise ModelSourceUnavailable(
                f"{src.filename} not in {self.paths.models_dir}; fetch it first with"
                f" `{SCRIPT_NAME} download {key}`"
            )
        with self.unavailable_on_failure(f"{key} ({src.filename})"):
            if src.url:
                self.download_urllib(src.url, dest)
            elif src.repo_id:
                self.download_hf(src.repo_id, src.hub_filename(), dest)
            else:
                raise ModelSourceUnavailable(f"no source configured for model '{key}' ({src.filename}). {src.notes}")
        return dest

    def ensure_models(self, keys: list[str]) -> dict[str, Path]:
        return {k: self.ensure_model(k) for k in keys}

    # -- data assets --------------------------------------------------------
    #
    # These are inputs rather than models. In a checkout that has run
    # `make deps`, jfk.wav already exists under build/whisper.cpp/samples;
    # standalone it does not, and `download all` fetches it from whisper.cpp
    # alongside the models. The corpus is synthesised locally rather than
    # downloaded -- the suite has to have something to index even when the
    # checkout's own docs are not there, and it needs no network for it. Being
    # free to regenerate, it is run output under `out_dir`, not a resident of
    # `models/`: that dir is shared with `make test` and may not be ours to write.

    def ensure_corpus(self) -> Path:
        """Path to a line-per-text corpus, preferring the checkout's own."""
        repo_copy = self.paths.find_data_asset("corpus1.txt")
        if repo_copy is not None:
            return repo_copy
        generated = self.paths.out_dir / "corpus_generated.txt"
        if not generated.exists():
            print(f"writing generated corpus -> {generated}")
            generated.parent.mkdir(parents=True, exist_ok=True)
            generated.write_text("\n".join(self.GENERATED_CORPUS) + "\n", encoding="utf-8")
        return generated

    def ensure_audio(self) -> Path:
        """Path to the jfk.wav sample, from the checkout or the models dir."""
        repo_copy = self.paths.find_data_asset("jfk.wav")
        if repo_copy is not None:
            return repo_copy
        dest = self.paths.models_dir / "jfk.wav"
        if dest.exists():
            return dest
        if not self.allow_download:
            raise ModelSourceUnavailable(
                f"jfk.wav not in {self.paths.models_dir}; fetch it first with `{SCRIPT_NAME} download all`"
            )
        with self.unavailable_on_failure("jfk.wav"):
            self.download_urllib(self.JFK_WAV_URL, dest)
        return dest

    def ingest_files(self) -> list[Path]:
        """What the rag cases ingest.

        The `scripts/case` originals indexed the checkout's own README.md and
        CHANGELOG.md, which is the more realistic corpus and the one the
        canned query ("how do I offload weights to the CPU?") was written for.
        Standalone those files are not there, so fall back to the generated
        corpus -- which carries a line about CPU offload for exactly this reason.
        """
        docs = [self.paths.root / name for name in ("README.md", "CHANGELOG.md")]
        present = [d for d in docs if d.exists()]
        return present or [self.ensure_corpus()]


# ---------------------------------------------------------------------------
# tests (inlined from the shell scripts that were in scripts/case)
# ---------------------------------------------------------------------------

TestFn = Callable[[str, "float | None"], int]


class TestSuite:
    """The smoke-test cases, grouped into families.

    Every case has the same signature -- ``(backend, timeout) -> exit code`` --
    and its docstring is the one-line description `list tests` and the generated
    Makefile print, so keep them short.
    """

    # Every test family, in the order `test-all` runs them: cheap and
    # fast-failing first, the multi-minute image cases last.
    FAMILY_ORDER: tuple[str, ...] = ("embed", "transcribe", "gen", "rag", "sd")

    # What `run --fast` runs in place of `test-all`. The sd cases dominate the
    # wall clock and mostly re-exercise the same three modules, so the third --
    # cpu-offload plus flash-attn, the cheatsheet recipe for an 8 GiB card --
    # stands in for all of them. gen-3 is left out rather than the family being
    # named as a whole: `gemma-e4b` is the largest model in the registry and the
    # one least likely to be on a given machine.
    FAST_TARGETS: tuple[str, ...] = ("test-embed-1", "test-gen-1", "test-sd-3")

    # Human-readable section headings for the generated Makefile's help text.
    # Rows of the `gen --stats` table -> metric names in the run history.
    # Times are seconds. The rates are per phase: chimera's generation rate
    # excludes prompt time, unlike the `tokens_per_second` rwt.py records.
    STATS_ROWS: dict[str, str] = {
        "Prompt tokens": "prompt_tokens",
        "Generated tokens": "generated_tokens",
        "Prompt eval time": "prompt_seconds",
        "Generation time": "generation_seconds",
        "Prompt tokens/second": "prompt_tokens_per_second",
        "Generation tokens/second": "generation_tokens_per_second",
    }

    @classmethod
    def parse_stats(cls, text: str) -> dict[str, float]:
        """Metrics from the last `--stats` table in `text`; empty if there is none."""
        metrics: dict[str, float] = {}
        for m in re.finditer(r"^\s+([A-Za-z/ ]+?)\s+\|\s+([0-9.]+)", text, re.MULTILINE):
            name = cls.STATS_ROWS.get(m.group(1))
            if name:
                metrics[name] = float(m.group(2))
        return metrics

    FAMILY_TITLES: dict[str, str] = {
        "embed": "Embedding",
        "transcribe": "Transcription",
        "gen": "Generation",
        "rag": "Vector store / RAG",
        "sd": "Stable Diffusion",
    }

    def __init__(self, env: Env, models: ModelRegistry) -> None:
        self.env = env
        self.models = models
        self.families: dict[str, dict[str, TestFn]] = {
            "embed": {"1": self.embed_1, "2": self.embed_2},
            "transcribe": {"1": self.transcribe_1, "2": self.transcribe_2},
            "gen": {"1": self.gen_1, "2": self.gen_2, "3": self.gen_3},
            "rag": {"1": self.rag_1, "2": self.rag_2},
            "sd": {"1": self.sd_1, "2": self.sd_2, "3": self.sd_3},
        }
        # Declared separately from FAMILY_ORDER so a family added to one and not
        # the other is caught here rather than silently skipped by `test-all`.
        assert tuple(self.families) == self.FAMILY_ORDER, "families must match FAMILY_ORDER"
        # Same reasoning: a renamed case would otherwise turn `run --fast` into an
        # argparse KeyError deep in the sequence, after the install step has run.
        unknown = [t for t in self.FAST_TARGETS if t not in self.targets()]
        assert not unknown, f"FAST_TARGETS names no such target: {unknown}"

    # -- output bookkeeping -------------------------------------------------

    @property
    def out_dir(self) -> Path:
        """Everything the suite writes goes here, so `clean` sweeps exactly
        what a run produced instead of globbing the project root."""
        self.env.paths.out_dir.mkdir(parents=True, exist_ok=True)
        return self.env.paths.out_dir

    def rag_db(self, name: str) -> Path:
        """Vector store for one rag case.

        Always passed explicitly. Without ``--db`` chimera opens ``$CHIMERA_DB``
        or the platform default -- i.e. the user's real database -- and a smoke
        test has no business creating collections in it.
        """
        return self.out_dir / f"{name}.db"

    # -- stable diffusion ---------------------------------------------------
    #
    # Three cases: te-on-cpu + vae-tiling, cpu-offload + vae-on-cpu, and
    # offload + flash-attn. Z-Image Turbo is a split-checkpoint model, so all
    # three use the component flags (--diffusion-model / --vae / --llm) rather
    # than -m, and none pass --gpu-layers: `sd` picks up the GPU on its own.

    # Z-Image Turbo is distilled for 8 steps without guidance. chimera's
    # defaults (20 steps, cfg 7.0, random seed) cost ~5x the passes, and the
    # fixed seed makes a backend's images comparable from one release to the next.
    SD_SAMPLING: tuple[str, ...] = ("--steps", "8", "--cfg-scale", "1.0", "--seed", "42")
    SD_WIDTH, SD_HEIGHT = 512, 1024
    SD_PROMPT = "a lovely plump cat"
    # Below this in every channel an image is blank: black from a NaN render, or
    # one flat colour. A real render is in the tens.
    SD_MIN_STDDEV = 2.0

    def sd_output(self, n: str) -> Path:
        return self.out_dir / f"z_turbo_{n}.png"

    def sd_case(self, n: str, extra: list[str], timeout: float | None) -> int:
        paths = self.models.ensure_models(ModelRegistry.SD_REQUIREMENTS)
        out = self.sd_output(n)
        out.unlink(missing_ok=True)  # a stale image would otherwise pass the check
        rc = self.env.chimera(
            [
                "sd",
                "--diffusion-model",
                str(paths["z-image-turbo"]),
                "--vae",
                str(paths["ae"]),
                "--llm",
                str(paths["qwen3-4b"]),
                *self.SD_SAMPLING,
                *extra,
                "-H",
                str(self.SD_HEIGHT),
                "-W",
                str(self.SD_WIDTH),
                "-o",
                str(out),
                "-p",
                self.SD_PROMPT,
            ],
            timeout=timeout,
        )
        return rc or self.check_image(out)

    def check_image(self, path: Path) -> int:
        """Fail an image of the wrong size, or one with no variation in any channel.

        Exit 0 from the CLI only means an image was written; a NaN render still
        writes one, all black.
        """
        try:
            width, height, channels, pixels = read_png(path)
        except (OSError, ValueError, zlib.error) as e:
            print(f"error: {path.name}: {e}", file=sys.stderr)
            return 1
        if (width, height) != (self.SD_WIDTH, self.SD_HEIGHT):
            print(
                f"error: {path.name} is {width}x{height}, expected {self.SD_WIDTH}x{self.SD_HEIGHT}",
                file=sys.stderr,
            )
            return 1
        spread = max(channel_stddevs(pixels, channels))
        print(f"-- {path.name}: {width}x{height}, max channel stddev {spread:.1f}")
        if spread < self.SD_MIN_STDDEV:
            print(
                f"error: {path.name} is blank (max channel stddev {spread:.2f} < {self.SD_MIN_STDDEV})", file=sys.stderr
            )
            return 1
        return 0

    # The three cases mirror cyllama's scripts/rwt.py so the two projects'
    # results compare directly. Each fits an 8 GiB card: the unqualified
    # all-on-GPU run needs ~9.4 GiB of weights (3.9 text encoder + 5.5
    # diffusion) and OOMs there, so no case runs it.

    def sd_1(self, _backend: str, timeout: float | None) -> int:
        """z_turbo te-on-cpu + vae-tiling."""
        # Parks only the text encoder's weights in RAM; every module still
        # computes on the GPU. --vae-tiling is not optional: without it the VAE
        # decode wants a ~3.3 GiB compute buffer while the diffusion weights are
        # still resident, and no --params-backend spelling helps because that is
        # a compute buffer, not weights.
        return self.sd_case("1", ["--params-backend", "te=cpu", "--vae-tiling"], timeout)

    def sd_2(self, _backend: str, timeout: float | None) -> int:
        """z_turbo cpu-offload + vae-on-cpu."""
        # Moves all the weights to RAM and the VAE's compute to the CPU as well;
        # expect it to be the slowest of the three.
        return self.sd_case("2", ["--offload-to-cpu", "--vae-on-cpu"], timeout)

    def sd_3(self, _backend: str, timeout: float | None) -> int:
        """z_turbo cpu-offload + flash-attn."""
        # The recipe docs/cheatsheet.md gives for Z-Image Turbo: weights stream
        # from RAM while compute stays on the GPU.
        return self.sd_case("3", ["--offload-to-cpu", "--diffusion-fa"], timeout)

    # -- generation ---------------------------------------------------------

    def gen_1(self, backend: str, timeout: float | None) -> int:
        """Llama-3.2-1B short prompt."""
        model = self.models.ensure_model("llama-3.2-1b")
        return self.env.chimera(
            [
                "gen",
                "-m",
                str(model),
                "-p",
                "Explain quantum entanglement in one paragraph.",
                "-n",
                "256",
                *self.env.gpu_args(backend),
                *self.env.stats_args(),
            ],
            timeout=timeout,
        )

    def gen_2(self, backend: str, timeout: float | None) -> int:
        """Qwen3-4B, same shape as gen-1."""
        # Output streams to stdout as it is produced; there is no --stream flag.
        model = self.models.ensure_model("qwen3-4b")
        return self.env.chimera(
            [
                "gen",
                "-m",
                str(model),
                "-p",
                "Write a haiku about GPUs.",
                "-n",
                "256",
                *self.env.gpu_args(backend),
                *self.env.stats_args(),
            ],
            timeout=timeout,
        )

    def gen_3(self, backend: str, timeout: float | None) -> int:
        """Gemma-4-E4B with sampler knobs."""
        # The temperature flag is --temp (the sd-cli spelling), not --temperature.
        model = self.models.ensure_model("gemma-e4b")
        return self.env.chimera(
            [
                "gen",
                "-m",
                str(model),
                "-p",
                "List three interesting facts about octopuses.",
                "-n",
                "512",
                "--temp",
                "0.7",
                "--top-p",
                "0.95",
                *self.env.gpu_args(backend),
                *self.env.stats_args(),
            ],
            timeout=timeout,
        )

    # -- embedding ----------------------------------------------------------

    def embed_1(self, backend: str, timeout: float | None) -> int:
        """multi-text vectors via --embd-separator."""
        # `embed` has no --similarity/--threshold: it emits vectors, and ranking
        # a corpus by similarity is what the rag family does. --embd-separator
        # splits one -p into several texts and prints one vector each, which is
        # the closest thing to a batch the subcommand offers.
        model = self.models.ensure_model("bge-small-en")
        return self.env.chimera(
            [
                "embed",
                "-m",
                str(model),
                "-p",
                "death and dying;grief and mourning;a cheerful summer picnic",
                "--embd-separator",
                ";",
                "--embd-output-format",
                "array",
                "--pooling",
                "mean",
                *self.env.gpu_args(backend),
            ],
            timeout=timeout,
        )

    def embed_2(self, backend: str, timeout: float | None) -> int:
        """corpus file -> one vector per line, memoized."""
        model = self.models.ensure_model("bge-small-en")
        corpus = self.models.ensure_corpus()
        out = self.out_dir / "corpus_vectors.txt"
        # --cache-embeddings writes into --cache-db, which defaults to the
        # user's real database; point it at the scratch dir like the rag cases do.
        rc = self.env.chimera(
            [
                "embed",
                "-m",
                str(model),
                "-f",
                str(corpus),
                "--embd-separator",
                "\n",
                "--embd-output-format",
                "raw",
                "-o",
                str(out),
                "--cache-embeddings",
                "--cache-db",
                str(self.rag_db("embed_cache")),
                *self.env.gpu_args(backend),
            ],
            timeout=timeout,
        )
        if rc != 0:
            return rc
        if not out.exists() or out.stat().st_size == 0:
            print(f"error: -o was given but no vectors were written to {out}", file=sys.stderr)
            return 1
        return 0

    # -- transcription ------------------------------------------------------

    def transcribe_1(self, _backend: str, timeout: float | None) -> int:
        """jfk.wav speech-to-text."""
        # The subcommand is `whisper`, not `transcribe`, and the audio flag is
        # -i/--input, not -f. whisper takes no --gpu-layers: it offloads whole
        # encoders, and picks the device itself (--no-gpu / --device opt out).
        model = self.models.ensure_model("whisper-base-en")
        audio = self.models.ensure_audio()
        return self.env.chimera(
            ["whisper", "-m", str(model), "-i", str(audio), "--timestamps"],
            timeout=timeout,
        )

    def transcribe_2(self, _backend: str, timeout: float | None) -> int:
        """jfk.wav -> srt / vtt / json files."""
        model = self.models.ensure_model("whisper-base-en")
        audio = self.models.ensure_audio()
        stem = self.out_dir / "jfk"
        expected = [stem.with_suffix(ext) for ext in (".srt", ".vtt", ".json")]
        for path in expected:
            path.unlink(missing_ok=True)  # so an old run cannot pass this for us
        rc = self.env.chimera(
            [
                "whisper",
                "-m",
                str(model),
                "-i",
                str(audio),
                "--output-file",
                str(stem),
                "--output-srt",
                "--output-vtt",
                "--output-json",
            ],
            timeout=timeout,
        )
        if rc != 0:
            return rc
        missing = [p.name for p in expected if not p.exists() or p.stat().st_size == 0]
        if missing:
            print(f"error: whisper reported success but wrote no {', '.join(missing)}", file=sys.stderr)
            return 1
        return 0

    # -- rag ----------------------------------------------------------------

    def rag_1(self, backend: str, timeout: float | None) -> int:
        """index create + ingest + search via $CHIMERA_DB."""
        # The no---db spelling: chimera falls back to $CHIMERA_DB, so this case
        # covers that path while still keeping its collections out of the user's
        # real database. There is no `chimera rag` subcommand -- retrieval is
        # `index create` -> `index ingest` -> `search`, and the collection
        # records the embedding model, so `search` does not take -e.
        model = self.models.ensure_model("bge-small-en")
        db = self.rag_db("rag_env")
        db.unlink(missing_ok=True)  # start from nothing so the create path is covered
        env = {"CHIMERA_DB": str(db)}
        gpu = self.env.gpu_args(backend)

        rc = self.env.chimera(
            ["index", "create", "-n", "docs", "-e", str(model), *gpu],
            env=env,
            timeout=timeout,
        )
        if rc != 0:
            return rc
        rc = self.env.chimera(
            ["index", "ingest", "-n", "docs", *sum((["-f", str(f)] for f in self.models.ingest_files()), []), *gpu],
            env=env,
            timeout=timeout,
        )
        if rc != 0:
            return rc
        if not db.exists():
            print(f"error: $CHIMERA_DB was set but no store was created at {db}", file=sys.stderr)
            return 1
        return self.env.chimera(
            ["search", "-n", "docs", "-q", "how do I offload weights to the CPU?", "-k", "5", "--mode", "hybrid", *gpu],
            env=env,
            timeout=timeout,
        )

    def rag_2(self, backend: str, timeout: float | None) -> int:
        """explicit --db: ingest, all three retrieval modes, stats, drop."""
        # --db is a per-subcommand flag, not a global one, so every step below
        # repeats it. The three modes are the point of the case: `lexical` never
        # loads the embedding model at all, `semantic` is vec0 KNN, and `hybrid`
        # fuses them -- three different code paths over one store.
        model = self.models.ensure_model("bge-small-en")
        db = self.rag_db("rag_explicit")
        db.unlink(missing_ok=True)
        gpu = self.env.gpu_args(backend)
        dbf = ["--db", str(db)]

        steps: list[list[str]] = [
            ["index", "create", "-n", "docs", "-e", str(model), *dbf, *gpu],
            ["index", "ingest", "-n", "docs", *sum((["-f", str(f)] for f in self.models.ingest_files()), []), *dbf, *gpu],
            ["index", "list", *dbf],
            ["index", "stats", "-n", "docs", *dbf],
            ["search", "-n", "docs", "-q", "how do I offload weights to the CPU?", "-k", "5", "--mode", "semantic", *dbf, *gpu],
            ["search", "-n", "docs", "-q", "Kilimanjaro", "-k", "5", "--mode", "lexical", *dbf],
            ["search", "-n", "docs", "-q", "how do I offload weights to the CPU?", "-k", "5", "--mode", "hybrid", *dbf, *gpu],
            # `db status` runs pending migrations and prints the schema version;
            # cheap, and it is the only thing in the suite that touches the
            # migration path on a store this run just created.
            ["db", "status", *dbf],
            ["index", "drop", "-n", "docs", *dbf],
        ]
        for argv in steps:
            rc = self.env.chimera(argv, timeout=timeout)
            if rc != 0:
                return rc
        return 0

    # -- target bookkeeping -------------------------------------------------

    def targets(self) -> dict[str, tuple[str, str]]:
        """Map each ``test-*`` target name to the (family, case) it runs.

        One token per test -- ``test-all``, ``test-gen-all``, ``test-sd-3`` -- so
        the CLI and the generated Makefile name the same things.
        """
        targets: dict[str, tuple[str, str]] = {"test-all": ("all", "all")}
        for fam, mapping in self.families.items():
            for n in sorted(mapping):
                targets[f"test-{fam}-{n}"] = (fam, n)
            targets[f"test-{fam}-all"] = (fam, "all")
        return targets

    def describe(self, kind: str, n: str) -> str:
        """One-line description of a target, from the case's docstring."""
        if kind == "all":
            return "every test in every family"
        if n == "all":
            return f"all {kind} tests"
        return (self.families[kind][n].__doc__ or "").strip()

    def collect_runs(self, kind: str, n: str) -> list[tuple[str, str]]:
        """Expand ('all'|<family>, 'all'|'1'|...) into concrete (kind, n) pairs."""
        kinds = list(self.families) if kind == "all" else [kind]
        runs: list[tuple[str, str]] = []
        for k in kinds:
            mapping = self.families[k]
            if n == "all":
                runs.extend((k, nk) for nk in sorted(mapping))
            elif n in mapping:
                runs.append((k, n))
            elif kind != "all":
                # An explicit `test embed 3` is a mistake worth reporting; the same
                # number under `test all 3` just means "the families that have a 3".
                print(
                    f"error: no test '{n}' in family '{k}' (have: {', '.join(sorted(mapping))})",
                    file=sys.stderr,
                )
                sys.exit(2)
        if not runs:
            print(f"error: no tests matched kind={kind} n={n}", file=sys.stderr)
            sys.exit(2)
        return runs

    def run_case(self, kind: str, n: str, backend: str, timeout: float | None) -> int:
        return self.families[kind][n](backend, timeout)


# ---------------------------------------------------------------------------
# generated Makefile
# ---------------------------------------------------------------------------


class MakefileRenderer:
    """Renders the Makefile whose rules mirror this script's own targets.

    Written to a *separate* file (`-o rat.mk`, included from the main Makefile
    if wanted) rather than to ./Makefile: chimera's Makefile is the build
    system, and this one is a frontend for a script that tests binaries the
    build system has already produced.
    """

    PY_VAR = "python3 scripts/rat.py"

    def __init__(self, env: Env, suite: TestSuite) -> None:
        self.env = env
        self.suite = suite
        self.lines: list[str] = []

    def render(self) -> str:
        self.lines = []
        backends = list(self.env.BACKENDS)

        family_targets: dict[str, list[str]] = {
            fam: [f"test-{fam}-{n}" for n in sorted(mapping)] + [f"test-{fam}-all"]
            for fam, mapping in self.suite.families.items()
        }
        width = max(len(t) for ts in family_targets.values() for t in ts) + 2

        # Group .PHONY into readable lines
        groups = [
            ["help", "info", "clean"],
            [f"install-{b}" for b in backends],
            [f"run-{b}" for b in backends],
            [f"run-{b}-fast" for b in backends],
            ["list-models", "list-tests", "download", "runs", "runs-diff", "report"],
            *family_targets.values(),
            ["test-all"],
        ]
        phony_lines = " \\\n\t\t".join(" ".join(g) for g in groups if g)

        add = self.lines.append
        add("")
        add(f"PY := {self.PY_VAR}")
        add("")
        add(f".PHONY: {phony_lines}")
        add("")
        add("help:")
        add('\t@echo "Available targets (frontend for $(PY)):"')
        add('\t@echo ""')
        add('\t@echo "  Setup:"')
        add('\t@echo "    info         - show the binary under test and its backends"')
        add('\t@echo "    clean        - remove the installed binary and any test output"')
        for b in backends:
            add(f'\t@echo "    install-{b:<7} - download the latest {b} release into build/rat/"')
        add('\t@echo ""')
        add('\t@echo "  Models:"')
        add('\t@echo "    list-models  - list known models and whether they are on disk"')
        add('\t@echo "    download     - fetch all models + data assets (use $(PY) download <key> for one)"')
        add('\t@echo "                   tests never download; a missing model makes its case SKIP"')

        for fam, mapping in self.suite.families.items():
            title = self.suite.FAMILY_TITLES.get(fam, fam)
            add('\t@echo ""')
            add(f'\t@echo "  {title} tests (backend auto-detected):"')
            for n in sorted(mapping):
                doc = (mapping[n].__doc__ or "").strip().rstrip(".")
                label = f"test-{fam}-{n}"
                add(f'\t@echo "    {label:<{width}}- {doc}"')
            label = f"test-{fam}-all"
            add(f'\t@echo "    {label:<{width}}- run all {fam} tests"')

        add('\t@echo ""')
        add('\t@echo "  Full cycle (install + test-all + clean):"')
        for b in backends:
            add(f'\t@echo "    run-{b:<8} - install, test and clean the {b} backend"')
        fast = ", ".join(self.suite.FAST_TARGETS)
        add(f'\t@echo "    run-<backend>-fast - as above, but {fast} in place of test-all"')
        add('\t@echo ""')
        add('\t@echo "    list         - list test targets and models"')
        add('\t@echo "    test-all     - run every test in every family"')
        add('\t@echo "    runs         - list recorded runs"')
        add('\t@echo "    runs-diff    - compare the latest run with the one before it"')
        add('\t@echo "    report       - write an HTML report of the run history and open it"')

        self.rule("info", "info")
        self.rule("clean", "clean")
        for b in backends:
            self.rule(f"install-{b}", f"install --{b}")
        for b in backends:
            self.rule(f"run-{b}", f"run --{b}")
            self.rule(f"run-{b}-fast", f"run --{b} --fast")
        self.rule("list-models", "list models")
        self.rule("list-tests", "list tests")
        self.rule("download", "download all")
        self.rule("runs", "runs list")
        self.rule("runs-diff", "runs diff")
        self.rule("report", "report")
        for target in self.suite.targets():
            if target != "test-all":
                self.rule(target, f"test {target}")
        self.rule("test-all", "test test-all")
        add("")
        return "\n".join(self.lines)

    def rule(self, target: str, args: str) -> None:
        self.lines.append("")
        self.lines.append(f"{target}:")
        self.lines.append(f"\t@$(PY) {args}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class Cli:
    """Argparse wiring and the subcommand implementations.

    The parser is built against the *defaults* (so --help can quote them), then
    :meth:`configure` rebuilds the collaborators from what was actually parsed.
    """

    def __init__(self) -> None:
        self.paths = Paths.from_environ()
        self.release = Release()
        self.env = Env(self.paths)
        self.models = ModelRegistry(self.paths)
        self.suite = TestSuite(self.env, self.models)
        self.runlog = RunLog(PROJECT)

    # -- configuration ------------------------------------------------------

    def configure(self, args: argparse.Namespace) -> None:
        """Apply parsed options; every collaborator is rebuilt from them."""
        # First, so the directories derived from it move too; the explicit
        # --bin-dir / --out-dir / --data-dir below then override what they name.
        if getattr(args, "build_dir", None):
            self.paths.rebase(Path(args.build_dir).expanduser().resolve())
        if getattr(args, "models_dir", None):
            self.paths.models_dir = Path(args.models_dir).expanduser().resolve()
        if getattr(args, "data_dir", None):
            self.paths.data_dir = Path(args.data_dir).expanduser().resolve()
        if getattr(args, "bin_dir", None):
            self.paths.bin_dir = Path(args.bin_dir).expanduser().resolve()
        if getattr(args, "out_dir", None):
            self.paths.out_dir = Path(args.out_dir).expanduser().resolve()
        if getattr(args, "repo", None):
            self.release.repo = args.repo
        # --bin wins over --bin-dir: it names the executable outright, which is
        # how a build/chimera or a system install is tested without an `install`.
        self.env.bin_path = Path(args.bin).expanduser().resolve() if getattr(args, "bin", None) else None
        if self.env.bin_override is not None:
            self._ensure_executable(self.env.bin_override)
        self.env.gpu_layers_override = getattr(args, "gpu_layers", None)

    @staticmethod
    def _ensure_executable(path: Path) -> None:
        """Set the exec bits on a `--bin` that lacks them.

        A binary downloaded as a GitHub Actions artifact arrives zipped without
        its POSIX mode, so it cannot be run until it is chmod +x'd.
        """
        if os.name == "nt" or not path.is_file() or os.access(path, os.X_OK):
            return
        try:
            path.chmod(path.stat().st_mode | 0o755)
            print(f"made {path} executable")
        except OSError as e:
            print(f"warning: {path} is not executable and chmod failed: {e}", file=sys.stderr)

    # -- simple commands ----------------------------------------------------

    def cmd_info(self, _args: argparse.Namespace) -> int:
        version, built, loaded = self.env.detect()
        print(f"{'binary:':<10}{self.env.bin_path}{'' if self.env.bin_path.exists() else '  (not installed)'}")
        print(f"{'version:':<10}{version or '(unknown)'}")
        print(f"{'built:':<10}{built or '(unknown)'}")
        print(f"{'loaded:':<10}{loaded or '(unknown)'}")
        print(f"{'models:':<10}{self.paths.models_dir}")
        print(f"{'build:':<10}{self.paths.build_dir}")
        print(f"{'rat:':<10}{self.paths.rat_dir}")
        print(f"{'output:':<10}{self.paths.out_dir}")
        print(f"{'host:':<10}{'-'.join(Release.host())}")
        if version:
            print()
            self.env.chimera(["info"])
        return 0

    def cmd_clean(self, args: argparse.Namespace) -> int:
        # Only ever removes the binary this script installed. An explicit --bin
        # points at something the caller owns -- a build/chimera, a system
        # install -- and deleting that would be a nasty surprise.
        installed = self.paths.bin_dir / self.env.exe_name
        if self.env.bin_override is not None:
            print(f"keeping {self.env.bin_path} (named by --bin, not installed here)")
        elif installed.exists():
            print(f"removing {installed}")
            installed.unlink()
        keep_output = getattr(args, "keep_output", False)
        keep_images = getattr(args, "keep_images", False)
        for path in (self.paths.out_dir, self.paths.cache_dir):
            if not path.exists():
                continue
            if path == self.paths.out_dir and keep_output:
                print(f"keeping {path}")
            elif path == self.paths.out_dir and keep_images:
                self._remove_except_images(path)
            else:
                print(f"removing {path}")
                shutil.rmtree(path)
        # All of the above normally live in <build>/rat, so once they are gone
        # the directory itself is this script's last trace; drop it too. Guarded
        # on emptiness rather than removed outright, since --bin-dir / --out-dir
        # can point elsewhere and something else may own what is left.
        if self.paths.rat_dir.is_dir() and not any(self.paths.rat_dir.iterdir()):
            print(f"removing {self.paths.rat_dir}")
            self.paths.rat_dir.rmdir()
        return 0

    @staticmethod
    def _remove_except_images(out_dir: Path) -> None:
        """Empty `out_dir` of everything but its PNGs; only the sd cases write those."""
        for child in sorted(out_dir.iterdir()):
            if child.suffix == ".png" and child.is_file():
                print(f"keeping {child}")
            elif child.is_dir() and not child.is_symlink():
                print(f"removing {child}")
                shutil.rmtree(child)
            else:
                print(f"removing {child}")
                child.unlink()
        if not any(out_dir.iterdir()):
            print(f"removing {out_dir}")
            out_dir.rmdir()

    # -- install ------------------------------------------------------------

    def resolve_asset(self, args: argparse.Namespace) -> tuple[str | None, Path | None]:
        """What ``--asset`` asks to install, as (url, local path).

        The value is a URL, a local archive, or a bare release-asset filename,
        told apart by shape rather than by a second flag: anything with a scheme
        is a URL, anything carrying a path separator or naming a file that
        exists is local, and everything else is treated as an asset name to
        fetch from the release ``--version`` selects.
        """
        value = args.asset
        if not value:
            return None, None

        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
            return value, None

        path = Path(value).expanduser()
        looks_local = path.exists() or "/" in value or "\\" in value
        if looks_local:
            resolved = path.resolve()
            if not resolved.exists():
                print(f"error: archive not found: {resolved}", file=sys.stderr)
                sys.exit(2)
            return None, resolved

        tag = args.version or self.release.latest_tag()
        return Release.DOWNLOAD.format(repo=self.release.repo, tag=tag, asset=value), None

    def cmd_install(self, args: argparse.Namespace) -> int:
        if self.env.bin_override is not None:
            print("error: --bin names the binary under test; there is nothing to install", file=sys.stderr)
            return 2
        target = self.paths.bin_dir / self.env.exe_name
        if target.exists() and not args.force:
            # Re-installing over a binary the caller may be mid-investigation on
            # is worth asking for explicitly. It is an error, not a no-op, so
            # `run` does not go on to test a binary other than the one asked for.
            print(f"error: {target} already exists; pass --force to replace it", file=sys.stderr)
            return 2

        try:
            url, local = self.resolve_asset(args)
            if url is None and local is None:
                backend = getattr(args, "backend", None)
                if not backend:
                    flags = "/".join("--" + b for b in self.env.BACKENDS)
                    print(f"error: give a backend ({flags}), or --asset <path-or-url>", file=sys.stderr)
                    return 2
                tag = args.version or self.release.latest_tag()
                # The workflows strip a leading `v` from the tag for the
                # filename, so the asset is named for the version, not the tag.
                url = self.release.url_for(backend, tag.lstrip("v"), tag=tag)

            if local is None:
                assert url is not None
                local = Release.download(url, self.paths.cache_dir / Path(url).name)
            Release.extract(local, self.paths.bin_dir)
        except AssetUnavailable as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except urllib.error.HTTPError as e:
            print(f"error: {e.code} fetching {e.url}", file=sys.stderr)
            return 2
        except (urllib.error.URLError, OSError, tarfile.TarError, zipfile.BadZipFile) as e:
            print(f"error: install failed: {e}", file=sys.stderr)
            return 2

        # Report what actually landed, since the asset name is the only thing
        # that has been checked so far and it is chosen, not verified.
        version, built, loaded = self.env.detect()
        print(f"chimera {version or '?'} (built: {built or '?'}, loaded: {loaded or '?'}) at {self.env.bin_path}")
        return 0

    # -- registries ---------------------------------------------------------

    def cmd_download(self, args: argparse.Namespace) -> int:
        """Fetch models into the models dir. The only subcommand that may.

        Test cases read the models dir and skip what is missing, so this is the
        step that decides how much of the suite can run.
        """
        self.models.allow_download = True
        keys = list(self.models.sources) if args.key == "all" else [args.key]
        failures = 0
        for k in keys:
            try:
                path = self.models.ensure_model(k)
                print(f"ok: {k} -> {path}")
            except ModelSourceUnavailable as e:
                print(f"error: {k}: {e}", file=sys.stderr)
                failures += 1
        if args.key == "all":
            # The transcribe cases need jfk.wav, which is not a model but must be
            # on disk before `test` runs. (The rag corpus is generated by `test`.)
            for what, get in (("jfk.wav", self.models.ensure_audio),):
                try:
                    print(f"ok: {what} -> {get()}")
                except ModelSourceUnavailable as e:
                    print(f"error: {e}", file=sys.stderr)
                    failures += 1
        return 1 if failures else 0

    def cmd_list_models(self, _args: argparse.Namespace) -> int:
        for key, src in self.models.sources.items():
            source = f"hf:{src.repo_id}:{src.hub_filename()}" if src.repo_id else (src.url or "(no source configured)")
            on_disk = "YES" if (self.paths.models_dir / src.filename).exists() else "no"
            print(f"{key:<16} file={src.filename:<40} on_disk={on_disk:<3} source={source}")
            if src.notes and not src.repo_id and not src.url:
                print(f"{'':<16} note: {src.notes}")
        return 0

    def cmd_list_tests(self, _args: argparse.Namespace) -> int:
        targets = self.suite.targets()
        width = max(len(t) for t in targets)
        for target, (kind, n) in targets.items():
            print(f"{target:<{width}}  {self.suite.describe(kind, n)}")
        return 0

    def cmd_list_assets(self, _args: argparse.Namespace) -> int:
        """The release assets this script knows how to install, per backend."""
        system, arch = Release.host()
        print(f"host: {system}-{arch}")
        for backend in self.env.BACKENDS:
            try:
                name = Release.asset_name(backend, "<version>")
            except AssetUnavailable as e:
                print(f"{backend:<8} -- {e}")
                continue
            print(f"{backend:<8} {name}")
        return 0

    def cmd_list(self, args: argparse.Namespace) -> int:
        """`list` with no argument shows every registry; `list tests|models|assets` narrows."""
        what = getattr(args, "what", "all")
        rc = 0
        sections: list[tuple[str, Callable[[argparse.Namespace], int]]] = [
            ("tests", self.cmd_list_tests),
            ("models", self.cmd_list_models),
            ("assets", self.cmd_list_assets),
        ]
        for name, fn in sections:
            if what in (name, "all"):
                if what == "all":
                    print(f"{name}:" if name == "tests" else f"\n{name}:")
                rc |= fn(args)
        return rc

    def cmd_gen_makefile(self, args: argparse.Namespace) -> int:
        content = MakefileRenderer(self.env, self.suite).render()
        if args.output:
            Path(args.output).write_text(content)
            print(f"wrote {args.output}")
        else:
            sys.stdout.write(content)
        return 0

    # -- test ---------------------------------------------------------------

    @staticmethod
    def _use_color(no_color: bool) -> bool:
        if no_color or os.environ.get("NO_COLOR"):
            return False
        return sys.stdout.isatty()

    def cmd_test(self, args: argparse.Namespace) -> int:
        kind, n = self.suite.targets()[args.target]

        # --dry-run promises to touch nothing, so it precedes every other step.
        if args.dry_run:
            backend = getattr(args, "backend", None) or self.env.detect_backend() or "?"
            for k, case in self.suite.collect_runs(kind, n):
                print(f"would run: {k} {case} (backend={backend})")
            return 0

        backend = self.env.require_backend(getattr(args, "backend", None))
        runs = self.suite.collect_runs(kind, n)

        problem = self.env.preflight(backend)
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 1

        color = self._use_color(args.no_color)
        green = "\033[32m" if color else ""
        red = "\033[31m" if color else ""
        yellow = "\033[33m" if color else ""
        reset = "\033[0m" if color else ""

        if not args.no_record:
            version, built, loaded = self.env.detect()
            binary = self.env.bin_path
            self.runlog.start(
                target=args.target,
                backend=backend,
                root=self.paths.root,
                version=version,
                artifact=str(binary),
                artifact_sha256=RunLog.sha256(binary),
                extra={"built": built, "loaded": loaded, "gpu_layers": self.env.gpu_layers(backend)},
            )

        # rc, plus the skip reason when the case never ran. A skip is held
        # apart from an rc rather than encoded as one: `chimera` itself exits 2
        # for its own reasons, and a missing model must not be read as one.
        results: list[tuple[str, str, int, float, str | None]] = []
        for k, case in runs:
            print(f"\n=== {k} test {case} (backend={backend}) ===", flush=True)
            started = time.monotonic()
            skipped: str | None = None
            rc = 0
            # Only the gen cases print `--stats`; every other case keeps a
            # terminal stderr.
            self.env.stderr_sink = bytearray() if k == "gen" else None
            try:
                rc = self.suite.run_case(k, case, backend, args.timeout)
            except ModelSourceUnavailable as e:
                print(f"skip: {e}", file=sys.stderr)
                skipped = str(e)
            finally:
                captured, self.env.stderr_sink = self.env.stderr_sink, None
            secs = time.monotonic() - started
            results.append((k, case, rc, secs, skipped))
            outputs = [self.suite.sd_output(case)] if k == "sd" else []
            metrics = self.suite.parse_stats(captured.decode("utf-8", "replace")) if captured else {}
            self.runlog.case(k, case, rc, secs, skipped, outputs, metrics)
            if rc != 0 and skipped is None and args.fail_fast:
                break

        # Summary
        print("\n=== summary ===")
        worst = 0
        for k, case, rc, secs, skipped in results:
            if skipped is not None:
                status = f"{yellow}SKIP{reset}"
            else:
                status = f"{green}PASS{reset}" if rc == 0 else f"{red}FAIL (rc={rc}){reset}"
                worst = max(worst, rc)
            print(f"  {k} {case}: {status}  ({secs:.1f}s)")
        skips = sum(1 for r in results if r[4] is not None)
        passed = sum(1 for r in results if r[4] is None and r[2] == 0)
        total = sum(r[3] for r in results)
        ran = len(results) - skips
        tail = f", {skips} skipped" if skips else ""
        print(f"{passed}/{ran} passed{tail} in {total:.1f}s")
        for k, case, _rc, _secs, skipped in results:
            if skipped is not None:
                print(f"  skipped {k} {case}: {skipped}")
        self.runlog.finish(worst)
        return worst

    def cmd_runs(self, args: argparse.Namespace) -> int:
        backend = getattr(args, "backend", None)
        if args.action == "list":
            if args.ids:
                print("error: `runs list` takes no ids", file=sys.stderr)
                return 2
            return self.runlog.print_list(args.limit, backend, args.all_projects)
        if len(args.ids) > 2:
            print("error: `runs diff` takes at most two ids", file=sys.stderr)
            return 2
        ids: list[int | None] = [None] * (2 - len(args.ids)) + list(args.ids)
        return self.runlog.print_diff(ids[0], ids[1], backend)

    def cmd_report(self, args: argparse.Namespace) -> int:
        out = Path(args.output).expanduser() if args.output else self.runlog.path.with_name("report.html")
        written = self.runlog.write_report(out, args.limit, getattr(args, "backend", None), args.all_projects)
        if written and not args.no_open:
            webbrowser.open(out.resolve().as_uri())
        return 0

    # -- run ----------------------------------------------------------------

    def run_targets(self, args: argparse.Namespace) -> list[str]:
        """The test targets one `run` invocation covers, in order."""
        if not args.fast:
            return [args.target or "test-all"]
        if args.target is not None:
            print(
                f"error: --fast already names its targets ({', '.join(self.suite.FAST_TARGETS)});"
                f" drop it to run '{args.target}' alone",
                file=sys.stderr,
            )
            sys.exit(2)
        return list(self.suite.FAST_TARGETS)

    def cmd_run(self, args: argparse.Namespace) -> int:
        """install -> test... -> clean, stopping at the first step that fails.

        A failure leaves the binary in place rather than cleaning up after it:
        the thing worth inspecting when a release fails is the binary it failed
        with, and `clean` is one command away once it has been looked at.
        """

        def test_step(target: str) -> Callable[[argparse.Namespace], int]:
            def step(a: argparse.Namespace) -> int:
                a.target = target
                return self.cmd_test(a)

            return step

        targets = self.run_targets(args)
        # --bin names the binary under test, so there is nothing to install.
        installs = [("install", self.cmd_install)] if self.env.bin_override is None else []
        steps: list[tuple[str, Callable[[argparse.Namespace], int]]] = [
            *installs,
            *((f"test {t}", test_step(t)) for t in targets),
            ("clean", self.cmd_clean),
        ]

        if args.dry_run:
            # `test --dry-run` promises to touch nothing, and `run` inherits that
            # promise for the whole sequence: print the steps, run none of them.
            where = f" --bin {self.env.bin_path}" if self.env.bin_override is not None else ""
            for name, _ in steps:
                verb, _, target = name.partition(" ")
                if verb == "clean" and args.keep_output:
                    target = "--keep-output"
                elif verb == "clean" and args.keep_images:
                    target = "--keep-images"
                print(f"would run: {SCRIPT_NAME} {verb}{where}{' ' + target if target else ''}")
            print()
            for _, step in steps[len(installs) : len(installs) + len(targets)]:
                step(args)
            return 0

        for i, (name, step) in enumerate(steps):
            print(f"\n=== {name} ===")
            rc = step(args)
            if rc != 0:
                skipped = ", ".join(n for n, _ in steps[i + 1 :])
                print(f"\nerror: {name} failed (rc={rc}); skipping {skipped}", file=sys.stderr)
                return rc
        return 0

    # -- argparse -----------------------------------------------------------

    def common_parser(self) -> argparse.ArgumentParser:
        """Options accepted both before and after the subcommand."""
        c = argparse.ArgumentParser(add_help=False)
        c.add_argument(
            "--bin",
            metavar="PATH",
            default=argparse.SUPPRESS,
            help="chimera executable to test, instead of the one `install` puts in "
            f"{self.paths.bin_dir}. Use it to point the suite at build/chimera or a "
            "system install; `clean` never deletes a binary named this way.",
        )
        c.add_argument(
            "--build-dir",
            "--build_dir",
            metavar="PATH",
            dest="build_dir",
            default=argparse.SUPPRESS,
            help=f"build tree this script works inside; it puts everything it creates -- the "
            f"installed binary, downloaded archives, test output -- in <build-dir>/rat "
            f"(default: {self.paths.build_dir}, following the Makefile's BUILD_DIR). Models are "
            "not under it and are not affected.",
        )
        c.add_argument(
            "--bin-dir",
            "--bin_dir",
            metavar="PATH",
            dest="bin_dir",
            default=argparse.SUPPRESS,
            help=f"directory `install` unpacks the release binary into (default: {self.paths.bin_dir})",
        )
        c.add_argument(
            "--models-dir",
            "--models_dir",
            metavar="PATH",
            dest="models_dir",
            default=argparse.SUPPRESS,
            help=f"directory holding the GGUF/safetensors models (default: {self.paths.models_dir})",
        )
        shorthand = c.add_mutually_exclusive_group()
        for backend in self.env.BACKENDS:
            shorthand.add_argument(
                f"--{backend}",
                dest="backend",
                action="store_const",
                const=backend,
                default=argparse.SUPPRESS,
                help=f"the {backend} backend: selects that release asset for `install`, "
                "and asserts it for `test`",
            )
        c.add_argument(
            "--data-dir",
            "--data_dir",
            metavar="PATH",
            dest="data_dir",
            default=argparse.SUPPRESS,
            help=f"directory holding jfk.wav / corpus1.txt (default: {self.paths.data_dir})",
        )
        c.add_argument(
            "--out-dir",
            "--out_dir",
            metavar="PATH",
            dest="out_dir",
            default=argparse.SUPPRESS,
            help=f"where images, transcripts and scratch DBs are written (default: {self.paths.out_dir}); "
            "`clean` removes it wholesale",
        )
        c.add_argument(
            "--repo",
            metavar="OWNER/NAME",
            default=argparse.SUPPRESS,
            help=f"GitHub repository releases are fetched from (default: {self.release.repo})",
        )
        return c

    @staticmethod
    def install_parser() -> argparse.ArgumentParser:
        """Options that only mean something while writing to build/rat/."""
        i = argparse.ArgumentParser(add_help=False)
        i.add_argument(
            "--version",
            metavar="TAG",
            default=None,
            help="release to install (e.g. 0.2.16). Default: whatever /releases/latest resolves to.",
        )
        i.add_argument(
            "--asset",
            metavar="PATH|URL|NAME",
            default=None,
            help="override what to install: a local archive "
            "(dist/chimera-0.2.16-linux-x86_64-cuda.tar.gz), a full URL, or a bare asset "
            "name to fetch from the --version release. Usually unnecessary -- without it "
            "the backend and the host pick the asset (--cuda on Linux -> "
            "chimera-<version>-linux-x86_64-cuda.tar.gz).",
        )
        i.add_argument(
            "--force",
            action="store_true",
            help="replace an existing binary in --bin-dir instead of leaving it alone",
        )
        return i

    @staticmethod
    def clean_parser() -> argparse.ArgumentParser:
        """Options for what `clean` removes; shared by `clean` and `run`."""
        c = argparse.ArgumentParser(add_help=False)
        c.add_argument(
            "--keep-output",
            action="store_true",
            help="leave --out-dir (sd images, transcripts, scratch DBs) in place for inspection",
        )
        c.add_argument(
            "--keep-images",
            action="store_true",
            help="leave the sd cases' PNGs in --out-dir and remove the rest of it; "
            "--keep-output already keeps them",
        )
        return c

    def test_parser(self) -> argparse.ArgumentParser:
        """Options that shape a test run; shared by `test` and `run`."""
        t = argparse.ArgumentParser(add_help=False)
        t.add_argument(
            "--timeout",
            type=float,
            default=None,
            help="per-test timeout in seconds (default: no timeout)",
        )
        t.add_argument(
            "--gpu-layers",
            "--gpu_layers",
            dest="gpu_layers",
            type=int,
            default=None,
            help="layers the llama-side cases offload (default: "
            f"{Env.GPU_LAYERS_DEFAULT} on a GPU backend, 0 on cpu). chimera itself defaults "
            "to 0, so a GPU release tested without this would only measure the CPU path.",
        )
        t.add_argument(
            "--fail-fast",
            action="store_true",
            help="stop at the first failing test instead of running the full matrix",
        )
        t.add_argument(
            "--dry-run",
            action="store_true",
            help="print the test matrix without downloading or invoking anything",
        )
        t.add_argument(
            "--no-color",
            action="store_true",
            help="disable colored PASS/FAIL output in the summary",
        )
        t.add_argument(
            "--no-record",
            action="store_true",
            help=f"do not record the run in the run history ({RunLog.default_path()})",
        )
        return t

    def build_parser(self) -> argparse.ArgumentParser:
        common = self.common_parser()
        p = argparse.ArgumentParser(
            description="chimera release tester",
            parents=[common],
            epilog="example: rat.py install --cuda && rat.py test --cuda test-all --models-dir models",
        )
        _sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

        class sub:  # noqa: N801 - thin shim so add_parser always inherits `common`
            @staticmethod
            def add_parser(
                name: str,
                parents: Sequence[argparse.ArgumentParser] = (),
                **kw: Any,
            ) -> argparse.ArgumentParser:
                return _sub.add_parser(name, parents=[common, *parents], **kw)

        sub.add_parser("info", help="show the binary under test, its backends and the models dir").set_defaults(
            func=self.cmd_info
        )
        sub.add_parser(
            "clean",
            parents=[self.clean_parser()],
            help="remove the installed binary and everything the suite wrote",
        ).set_defaults(
            func=self.cmd_clean
        )

        inst = sub.add_parser(
            "install",
            parents=[self.install_parser()],
            help="download a chimera release and unpack it into --bin-dir",
        )
        inst.set_defaults(func=self.cmd_install)

        dl = sub.add_parser(
            "download",
            help="download a model (or 'all') into --models-dir; the only command that fetches models",
        )
        dl.add_argument("key", choices=[*self.models.sources.keys(), "all"])
        dl.set_defaults(func=self.cmd_download)

        lst = sub.add_parser("list", help="list test targets, models and release assets (or one of them)")
        lst.add_argument(
            "what",
            nargs="?",
            choices=["tests", "models", "assets", "all"],
            default="all",
            help="which registry to show (default: all of them)",
        )
        lst.set_defaults(func=self.cmd_list)

        # The flat names, kept working but out of --help so `list` is the one
        # obvious spelling.
        sub.add_parser("list-models").set_defaults(func=self.cmd_list_models)
        sub.add_parser("list-tests").set_defaults(func=self.cmd_list_tests)

        gm = sub.add_parser("gen-makefile", help="generate a Makefile from this script's registries")
        gm.add_argument("-o", "--output", help="write to file instead of stdout (e.g. -o rat.mk)")
        gm.set_defaults(func=self.cmd_gen_makefile)

        # `test` takes one target name -- `test-sd-3` rather than `test sd 3`, so a
        # target is a single token and matches the Makefile rule of the same name.
        t = sub.add_parser("test", parents=[self.test_parser()], help="run a test target (see `list tests`)")
        t.add_argument(
            "target",
            choices=list(self.suite.targets()),
            metavar="TARGET",
            help="one of the targets `list tests` prints, e.g. test-all, test-gen-1",
        )
        t.set_defaults(func=self.cmd_test)

        # `run` is the whole cycle in one command, so a release can be checked on
        # a clean machine without three invocations that must agree on the backend.
        r = sub.add_parser(
            "run",
            parents=[self.install_parser(), self.clean_parser(), self.test_parser()],
            help="install, test, then clean -- stopping at the first failure",
        )
        r.add_argument(
            "target",
            nargs="?",
            default=None,
            choices=list(self.suite.targets()),
            metavar="[TARGET]",
            help="the target to run (default: test-all)",
        )
        r.add_argument(
            "--fast",
            action="store_true",
            help="the short cycle: run "
            + ", ".join(self.suite.FAST_TARGETS)
            + " in place of test-all, skipping the image cases that dominate the wall clock",
        )
        r.set_defaults(func=self.cmd_run)

        rs = sub.add_parser(
            "runs",
            help=f"list recorded runs, or diff two of them (history in {RunLog.default_path()})",
        )
        rs.add_argument("action", nargs="?", choices=["list", "diff"], default="list")
        rs.add_argument(
            "ids",
            nargs="*",
            type=int,
            metavar="ID",
            help="diff: `B` compares B with the run before it; `A B` compares the two; "
            "none compares the latest run with the one before it",
        )
        rs.add_argument(
            "-n",
            "--limit",
            type=int,
            default=20,
            help="list: how many runs (default: 20)",
        )
        rs.add_argument(
            "--all-projects",
            action="store_true",
            help="list: include every project's runs, not only " + PROJECT + "'s",
        )
        rs.set_defaults(func=self.cmd_runs)

        rep = sub.add_parser("report", help="write an HTML report of the run history and open it in the browser")
        rep.add_argument(
            "-o",
            "--output",
            metavar="FILE",
            help=f"where to write it (default: {RunLog.default_path().with_name('report.html')})",
        )
        rep.add_argument(
            "-n",
            "--limit",
            type=int,
            default=20,
            help="recent runs listed, and runs per project/backend/target trend (default: 20)",
        )
        rep.add_argument(
            "--all-projects",
            action="store_true",
            help="include every project's runs, not only " + PROJECT + "'s",
        )
        rep.add_argument("--no-open", action="store_true", help="write the report without opening it")
        rep.set_defaults(func=self.cmd_report)

        return p

    def main(self, argv: list[str] | None = None) -> int:
        args = self.build_parser().parse_args(argv)
        self.configure(args)
        return int(args.func(args) or 0)


def main() -> None:
    sys.exit(Cli().main())


if __name__ == "__main__":
    main()
