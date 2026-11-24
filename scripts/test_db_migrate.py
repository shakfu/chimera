#!/usr/bin/env python3
"""
test_db_migrate.py - assert chimera's schema migration is backward-compatible.

Creates a v1-schema chimera.db (the shape every existing user has, since v1
was the first schema we shipped), seeds it with one row per table, runs
`chimera db status` against it to trigger `open_and_migrate`, and then
verifies:

  1. PRAGMA user_version advanced to the current latest.
  2. Pre-existing rows survived.
  3. The v2 / v3 additions are in place:
     - embedding_cache table exists (v2)
     - collections gained `distance` defaulting to 'cosine' (v3)
     - collections gained `chunk_tokens`  defaulting to 512   (v3)
     - collections gained `chunk_overlap` defaulting to  64   (v3)

The v1 fixture below is a hand-copy of the v1 migration in
src/chimera/chimera_db.cpp, **minus** the FTS5 virtual table + its three
triggers. The Python stdlib's sqlite3 module isn't guaranteed to ship with
FTS5 (depends on how libsqlite3 was built), and the v2 / v3 migrations
don't touch messages_fts anyway, so skipping it keeps the test portable
without weakening what it covers.

Exits 0 on pass, non-zero (with a clear message) on any assertion failure.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent

# Probe for the chimera binary. CI builds to ./build/chimera (or
# build/Release/chimera.exe on MSVC); local installs may live elsewhere.
CANDIDATES = [
    REPO_ROOT / "build" / "chimera",
    REPO_ROOT / "build" / "Release" / "chimera.exe",
    REPO_ROOT / "build" / "chimera.exe",
]
CHIMERA = next((p for p in CANDIDATES if p.exists()), None)


# v1 schema, copied from src/chimera/chimera_db.cpp::MIGRATION_V1.
# Trimmed: no messages_fts / FTS5 triggers (see file docstring).
V1_DDL = """
    CREATE TABLE chats (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at      INTEGER NOT NULL,
        updated_at      INTEGER NOT NULL,
        title           TEXT,
        model_path      TEXT    NOT NULL,
        model_alias     TEXT,
        system_prompt   TEXT,
        source          TEXT    NOT NULL CHECK (source IN ('chat', 'serve')),
        metadata_json   TEXT
    );
    CREATE INDEX idx_chats_updated ON chats(updated_at DESC);

    CREATE TABLE messages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id         INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
        seq             INTEGER NOT NULL,
        role            TEXT    NOT NULL,
        content         TEXT    NOT NULL,
        reasoning       TEXT,
        media_json      TEXT,
        tool_calls_json TEXT,
        tokens_in       INTEGER,
        tokens_out      INTEGER,
        created_at      INTEGER NOT NULL,
        UNIQUE (chat_id, seq)
    );
    CREATE INDEX idx_messages_chat ON messages(chat_id, seq);

    CREATE TABLE collections (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        name            TEXT    UNIQUE NOT NULL,
        embedding_model TEXT    NOT NULL,
        dim             INTEGER NOT NULL,
        created_at      INTEGER NOT NULL,
        metadata_json   TEXT
    );

    CREATE TABLE documents (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_id   INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
        source_uri      TEXT,
        chunk_index     INTEGER NOT NULL,
        text            TEXT    NOT NULL,
        token_count     INTEGER,
        metadata_json   TEXT,
        created_at      INTEGER NOT NULL
    );
    CREATE INDEX idx_documents_collection ON documents(collection_id);
    CREATE INDEX idx_documents_source     ON documents(collection_id, source_uri, chunk_index);
"""


def fail(msg: str) -> NoReturn:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def seed_v1(db_path: Path) -> None:
    """Create a v1-shaped DB at db_path and insert one row per table."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(V1_DDL)
        now = int(time.time())
        conn.execute(
            "INSERT INTO chats "
            "  (created_at, updated_at, title, model_path, source) "
            "VALUES (?, ?, ?, ?, ?)",
            (now, now, "pre-migrate chat", "/no/such/model.gguf", "chat"),
        )
        chat_id = conn.execute("SELECT id FROM chats").fetchone()[0]
        conn.execute(
            "INSERT INTO messages "
            "  (chat_id, seq, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (chat_id, 0, "user", "hello from v1", now),
        )
        conn.execute(
            "INSERT INTO collections "
            "  (name, embedding_model, dim, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("legacy", "bge-small.gguf", 384, now),
        )
        coll_id = conn.execute("SELECT id FROM collections").fetchone()[0]
        conn.execute(
            "INSERT INTO documents "
            "  (collection_id, source_uri, chunk_index, text, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (coll_id, "doc.txt", 0, "v1 doc body", now),
        )
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    finally:
        conn.close()


def trigger_migration(db_path: Path) -> None:
    """Drive `chimera db status` against db_path; it calls open_and_migrate."""
    env = os.environ.copy()
    env["CHIMERA_DB"] = str(db_path)
    res = subprocess.run(
        [str(CHIMERA), "db", "status"],
        env=env,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(res.stdout, file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        fail(f"`chimera db status` exited {res.returncode}")


def assert_eq(label: str, got, want) -> None:
    if got != want:
        fail(f"{label}: got {got!r}, want {want!r}")


def column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def verify(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        # 1. Schema version advanced.
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        if v < 3:
            fail(f"user_version did not advance: got {v}, expected >= 3")

        # 2. Pre-existing rows still there.
        assert_eq(
            "chats row count",
            conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0],
            1,
        )
        assert_eq(
            "messages row count",
            conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
            1,
        )
        assert_eq(
            "collections row count",
            conn.execute("SELECT COUNT(*) FROM collections").fetchone()[0],
            1,
        )
        assert_eq(
            "documents row count",
            conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            1,
        )

        title = conn.execute("SELECT title FROM chats").fetchone()[0]
        assert_eq("chats.title preserved", title, "pre-migrate chat")
        body = conn.execute("SELECT content FROM messages").fetchone()[0]
        assert_eq("messages.content preserved", body, "hello from v1")

        # 3a. v2: embedding_cache present.
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "embedding_cache" not in tables:
            fail(
                f"embedding_cache table missing after migrate (have: {sorted(tables)})"
            )

        # 3b. v3: collections gained 3 new columns with default backfill.
        cols = column_names(conn, "collections")
        for needed in ("distance", "chunk_tokens", "chunk_overlap"):
            if needed not in cols:
                fail(f"collections.{needed} missing after migrate (have: {cols})")

        row = conn.execute(
            "SELECT distance, chunk_tokens, chunk_overlap FROM collections"
        ).fetchone()
        assert_eq("collections.distance default", row[0], "cosine")
        assert_eq("collections.chunk_tokens default", row[1], 512)
        assert_eq("collections.chunk_overlap default", row[2], 64)
    finally:
        conn.close()


def db_status(db_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CHIMERA_DB"] = str(db_path)
    return subprocess.Popen(
        [str(CHIMERA), "db", "status"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def check_concurrent_fresh(db_path: Path, n: int = 8) -> None:
    """n processes migrate one fresh DB at once; all must succeed."""
    env = os.environ.copy()
    env["CHIMERA_DB"] = str(db_path)
    # Output goes to files, not pipes, so waiting on one process never
    # blocks another on a full pipe.
    logs = [db_path.with_name(f"status{i}.err") for i in range(n)]
    procs = []
    for log in logs:
        with open(log, "w") as f:
            procs.append(subprocess.Popen([str(CHIMERA), "db", "status"], env=env,
                                          stdout=subprocess.DEVNULL, stderr=f))
    deadline = time.monotonic() + 60
    hung = []
    for i, p in enumerate(procs):
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            hung.append(i)
    if hung:
        for i in hung:
            if sys.platform == "darwin":
                s = subprocess.run(["sample", str(procs[i].pid), "1"],
                                   capture_output=True, text=True)
                print(f"--- sample of process {i} (pid {procs[i].pid}) ---\n"
                      f"{s.stdout}{s.stderr}", file=sys.stderr)
            procs[i].kill()
            procs[i].wait()
        for i, log in enumerate(logs):
            state = "hung" if i in hung else f"exit {procs[i].returncode}"
            print(f"--- process {i}: {state}; stderr ---\n{log.read_text()}",
                  file=sys.stderr)
        fail(f"concurrent migrate: processes {hung} still running after 60 s")
    for i, p in enumerate(procs):
        if p.returncode != 0:
            fail(f"concurrent migrate: process {i} exited {p.returncode}: "
                 f"{logs[i].read_text().strip()}")
    v = sqlite3.connect(db_path).execute("PRAGMA user_version").fetchone()[0]
    if v < 5:
        fail(f"concurrent migrate: user_version {v}, expected >= 5")


def check_newer_schema_refused(db_path: Path) -> None:
    """A DB from a newer chimera must be refused, not used as-is."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA user_version = 9999")
    conn.commit()
    conn.close()
    p = db_status(db_path)
    _, err = p.communicate(timeout=60)
    if p.returncode == 0:
        fail("newer schema: `chimera db status` succeeded on a v9999 DB")
    if "v9999" not in err:
        fail(f"newer schema: error does not name the version: {err.strip()}")


def main() -> int:
    if CHIMERA is None:
        fail("chimera binary not built; run `make build` first")

    tmp = Path(tempfile.mkdtemp(prefix="chimera-migrate-"))
    db = tmp / "chimera.db"
    try:
        print(f"db-migrate: seeding v1 DB at {db}")
        seed_v1(db)

        v = sqlite3.connect(db).execute("PRAGMA user_version").fetchone()[0]
        assert_eq("seed user_version", v, 1)

        print("db-migrate: invoking `chimera db status` to drive open_and_migrate")
        trigger_migration(db)

        print("db-migrate: verifying post-migrate state")
        verify(db)
        print("PASS: v1 -> latest migration is data-preserving and schema-correct")

        print("db-migrate: 8 concurrent migrations of one fresh DB")
        check_concurrent_fresh(tmp / "fresh.db")
        print("PASS: concurrent fresh-DB migration")

        print("db-migrate: DB with a newer schema version")
        check_newer_schema_refused(db)
        print("PASS: newer schema refused")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
