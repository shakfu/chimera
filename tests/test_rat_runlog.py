"""Tests for the run history in scripts/rat.py (RunLog and its cmd_test hook)."""

import hashlib
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

RAT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rat.py"


@pytest.fixture(scope="module")
def rat():
    spec = importlib.util.spec_from_file_location("rat", RAT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rat"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("rat", None)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "runs" / "db.sqlite"


def record(rat, db_path, tmp_path, *, version, image_bytes, sd_rc=0, backend="cuda", target="test-all"):
    """One complete run: a passing gen case, an sd case with an image, a skip."""
    log = rat.RunLog("chimera", db_path)
    log.start(target=target, backend=backend, root=tmp_path, version=version, artifact="build/rat/chimera")
    image = tmp_path / "z_turbo_3.png"
    image.write_bytes(image_bytes)
    log.case("gen", "1", 0, 2.0)
    log.case("sd", "3", sd_rc, 40.0, outputs=[image])
    log.case("gen", "3", 2, 0.1, skipped="no source")
    log.finish(max(0, sd_rc))
    return log


def test_records_run_cases_and_outputs(rat, db_path, tmp_path):
    record(rat, db_path, tmp_path, version="0.4.3", image_bytes=b"png-a")
    db = sqlite3.connect(db_path)
    run = db.execute("SELECT project, backend, version, rc, finished_at FROM runs").fetchone()
    assert run[:4] == ("chimera", "cuda", "0.4.3", 0)
    assert run[4] is not None
    cases = db.execute("SELECT family, n, status, rc, detail FROM cases ORDER BY id").fetchall()
    assert cases == [
        ("gen", "1", "pass", 0, None),
        ("sd", "3", "pass", 0, None),
        ("gen", "3", "skip", None, "no source"),
    ]
    out = db.execute("SELECT name, bytes, sha256 FROM outputs").fetchall()
    assert out == [("z_turbo_3.png", 5, hashlib.sha256(b"png-a").hexdigest())]
    assert db.execute("PRAGMA user_version").fetchone()[0] == rat.RunLog.SCHEMA_VERSION


@pytest.mark.parametrize(
    ("rc", "skipped", "status"),
    [(0, None, "pass"), (1, None, "fail"), (124, None, "timeout"), (2, "why", "skip")],
)
def test_status(rat, rc, skipped, status):
    assert rat.RunLog.status(rc, skipped) == status


def test_missing_output_is_not_recorded(rat, db_path, tmp_path):
    log = rat.RunLog("chimera", db_path)
    log.start(target="test-sd-3", backend="cuda", root=tmp_path)
    log.case("sd", "3", 1, 5.0, outputs=[tmp_path / "absent.png"])
    log.finish(1)
    assert sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM outputs").fetchone()[0] == 0


def test_unfinished_run_has_no_finished_at(rat, db_path, tmp_path):
    log = rat.RunLog("chimera", db_path)
    log.start(target="test-all", backend="cpu", root=tmp_path)
    log.case("gen", "1", 0, 1.0)
    row = sqlite3.connect(db_path).execute("SELECT finished_at, rc FROM runs").fetchone()
    assert row == (None, None)


def test_newer_schema_disables_recording(rat, db_path, tmp_path, capsys):
    db_path.parent.mkdir(parents=True)
    db = sqlite3.connect(db_path)
    db.execute(f"PRAGMA user_version = {rat.RunLog.SCHEMA_VERSION + 1}")
    db.close()
    log = rat.RunLog("chimera", db_path)
    log.start(target="test-all", backend="cpu", root=tmp_path)
    log.case("gen", "1", 0, 1.0)
    log.finish(0)
    assert not log.enabled
    assert log.run_id is None
    assert "run history disabled" in capsys.readouterr().err


def test_diff_defaults_to_previous_run_with_same_key(rat, db_path, tmp_path, capsys):
    record(rat, db_path, tmp_path, version="0.4.2", image_bytes=b"png-a")
    record(rat, db_path, tmp_path, version="0.4.2", image_bytes=b"png-a", backend="vulkan")
    record(rat, db_path, tmp_path, version="0.4.3", image_bytes=b"png-bb", sd_rc=1)
    capsys.readouterr()
    log = rat.RunLog("chimera", db_path)
    assert log.print_diff() == 0
    out = capsys.readouterr().out
    # Run 3 (cuda) compares with run 1 (cuda), not run 2 (vulkan).
    assert out.split("  run      1")[1].split()[0] == "3"
    assert "* version  0.4.2" in out
    assert "  backend  cuda" in out
    assert "* sd 3          pass     fail" in out
    assert "z_turbo_3.png changed (5 -> 6 bytes)" in out
    assert "  gen 1         pass     pass" in out


def test_diff_identical_outputs(rat, db_path, tmp_path, capsys):
    record(rat, db_path, tmp_path, version="0.4.3", image_bytes=b"same")
    record(rat, db_path, tmp_path, version="0.4.3", image_bytes=b"same")
    capsys.readouterr()
    assert rat.RunLog("chimera", db_path).print_diff(1, 2) == 0
    assert "z_turbo_3.png identical" in capsys.readouterr().out


def test_diff_without_history(rat, db_path, tmp_path, capsys):
    log = rat.RunLog("chimera", db_path)
    assert log.print_diff() == 2
    record(rat, db_path, tmp_path, version="0.4.3", image_bytes=b"x")
    assert log.print_diff() == 2
    assert "no earlier chimera cuda test-all run" in capsys.readouterr().err


def test_list_filters_by_project(rat, db_path, tmp_path, capsys):
    record(rat, db_path, tmp_path, version="0.4.3", image_bytes=b"x")
    other = rat.RunLog("cyllama", db_path)
    other.start(target="test-all", backend="cuda", root=tmp_path, version="0.2.16")
    other.finish(0)
    capsys.readouterr()
    rat.RunLog("chimera", db_path).print_list(20)
    out = capsys.readouterr().out
    assert "chimera" in out and "cyllama" not in out
    assert " 2/3 " in out
    rat.RunLog("chimera", db_path).print_list(20, all_projects=True)
    assert "cyllama" in capsys.readouterr().out


def test_list_does_not_create_db(rat, db_path, capsys):
    rat.RunLog("chimera", db_path).print_list(20)
    assert not db_path.exists()
    assert "no runs recorded" in capsys.readouterr().out


def test_cmd_test_records(rat, db_path, tmp_path, monkeypatch):
    """`test` writes one run with a row per case; --no-record and --dry-run write nothing."""
    cli = rat.Cli()
    cli.runlog = rat.RunLog("chimera", db_path)
    cli.paths.root = tmp_path
    cli.paths.out_dir = tmp_path / "out"
    binary = tmp_path / "chimera"
    binary.write_bytes(b"ELF")
    cli.env.bin_path = binary
    monkeypatch.setattr(cli, "configure", lambda args: None)
    monkeypatch.setattr(cli.env, "require_backend", lambda requested: "cpu")
    monkeypatch.setattr(cli.env, "preflight", lambda backend: None)
    monkeypatch.setattr(cli.env, "detect", lambda: ("0.2.16", "cpu", "cpu"))

    def run_case(kind, n, backend, timeout):
        if kind == "sd":
            cli.suite.sd_output(n).write_bytes(b"img")
        return 0 if n != "2" else 1

    monkeypatch.setattr(cli.suite, "run_case", run_case)

    assert cli.main(["test", "--no-color", "--no-record", "test-sd-all"]) == 1
    assert cli.main(["test", "--dry-run", "test-sd-all"]) == 0
    assert not db_path.exists()

    assert cli.main(["test", "--no-color", "test-sd-all"]) == 1
    db = sqlite3.connect(db_path)
    run = db.execute("SELECT target, backend, version, artifact, artifact_sha256, rc FROM runs").fetchall()
    assert run == [("test-sd-all", "cpu", "0.2.16", str(binary), hashlib.sha256(b"ELF").hexdigest(), 1)]
    statuses = db.execute("SELECT n, status FROM cases ORDER BY id").fetchall()
    assert statuses == [("1", "pass"), ("2", "fail"), ("3", "pass")]
    assert db.execute("SELECT COUNT(*) FROM outputs").fetchone()[0] == 3


# The table `chimera gen --stats` prints; scripts/test.py pins the same rows
# against the real binary.
STATS_TABLE = """-----------------------------------
  Prompt tokens            |      4
  Generated tokens         |     32
  Prompt eval time         | 0.03 s
  Generation time          | 0.95 s
  Prompt tokens/second     | 116.34
  Generation tokens/second |  33.82
-----------------------------------
"""


def test_parse_stats(rat):
    assert rat.TestSuite.parse_stats("log | line\n" + STATS_TABLE) == {
        "prompt_tokens": 4,
        "generated_tokens": 32,
        "prompt_seconds": 0.03,
        "generation_seconds": 0.95,
        "prompt_tokens_per_second": 116.34,
        "generation_tokens_per_second": 33.82,
    }
    assert rat.TestSuite.parse_stats("no table") == {}


def fake_binary(path, help_text):
    path.write_text(f"#!/bin/sh\ncat <<'EOF'\n{help_text}\nEOF\n")
    path.chmod(0o755)
    return path


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script as a fake binary")
def test_stats_args_probes_binary(rat, tmp_path):
    env = rat.Env(rat.Paths.from_environ())
    env.bin_path = fake_binary(tmp_path / "new", "--stats   Print token counts")
    assert env.stats_args() == ["--stats"]
    env.bin_path = fake_binary(tmp_path / "old", "--temp  Temperature")
    assert env.stats_args() == []


def test_env_run_tees_stderr(rat, tmp_path, capfd):
    env = rat.Env(rat.Paths.from_environ())
    script = "import sys; print('out'); sys.stderr.write('err-line\\n'); sys.exit(3)"
    env.stderr_sink = bytearray()
    assert env.run([sys.executable, "-c", script]) == 3
    assert bytes(env.stderr_sink) == b"err-line\n"
    seen = capfd.readouterr()
    assert "err-line" in seen.err and "out" in seen.out


def test_cmd_test_records_gen_stats(rat, db_path, tmp_path, monkeypatch, capsys):
    cli = rat.Cli()
    cli.runlog = rat.RunLog("chimera", db_path)
    cli.paths.root = tmp_path
    binary = tmp_path / "chimera"
    binary.write_bytes(b"ELF")
    cli.env.bin_path = binary
    monkeypatch.setattr(cli, "configure", lambda args: None)
    monkeypatch.setattr(cli.env, "require_backend", lambda requested: "cpu")
    monkeypatch.setattr(cli.env, "preflight", lambda backend: None)
    monkeypatch.setattr(cli.env, "detect", lambda: ("0.5.1", "cpu", "cpu"))
    script = f"import sys; sys.stderr.write({STATS_TABLE!r})"
    monkeypatch.setattr(
        cli.suite, "run_case", lambda kind, n, backend, timeout: cli.env.run([sys.executable, "-c", script])
    )
    for _ in range(2):
        assert cli.main(["test", "--no-color", "test-gen-1"]) == 0
    assert cli.env.stderr_sink is None
    rows = sqlite3.connect(db_path).execute(
        "SELECT value FROM metrics WHERE name = 'generation_tokens_per_second'"
    ).fetchall()
    assert rows == [(33.82,), (33.82,)]
    capsys.readouterr()
    cli.runlog.print_diff()
    assert "gen tok/s 33.8 -> 33.8 (+0.0%)" in capsys.readouterr().out


def test_report(rat, db_path, tmp_path, capsys):
    """Recent-runs table, a diff per group, a trend per case and measure, a
    version marker, a gap for a failed case, and escaped text."""
    record(rat, db_path, tmp_path, version="0.4.2", image_bytes=b"a")
    record(rat, db_path, tmp_path, version="0.4.2", image_bytes=b"a", sd_rc=1)
    record(rat, db_path, tmp_path, version="0.4.3<x>", image_bytes=b"b")
    out = tmp_path / "r" / "report.html"
    assert rat.RunLog("chimera", db_path).write_report(out, 20) is True
    page = out.read_text()
    assert page.startswith("<!doctype html>")
    assert "chimera &middot; cuda &middot; test-all" in page
    assert "Run 3 vs run 2" in page
    assert 'class="fail">fail<' in page  # run 2's sd case in the diff
    assert "0.4.3&lt;x&gt;" in page and "0.4.3<x>" not in page
    assert page.count("<figure>") == 2  # gen 1 seconds, sd 3 seconds; gen 3 never passes
    assert page.count('class="release"') == 2  # one version change per chart
    # sd 3 failed in run 2, so its line breaks: two one-point segments, no polyline.
    sd = page[page.index("sd 3 <span>") :]
    assert "<polyline" not in sd.split("</figure>")[0]


def test_report_empty(rat, db_path, tmp_path, capsys):
    out = tmp_path / "report.html"
    assert rat.RunLog("chimera", db_path).write_report(out, 20) is False
    assert not out.exists()
    assert "no runs recorded" in capsys.readouterr().out


def test_svg_trend_ticks(rat):
    svg = rat.RunLog._svg_trend([("1", 3.7, "t", "a"), ("2", None, "t", "a"), ("3", 4.1, "t", "b")], "s")
    # Half the max is 2.05; the next nice step is 2.5, so the axis tops out at 5.
    assert [t for t in (">0<", ">2.5<", ">5<") if t in svg] == [">0<", ">2.5<", ">5<"]
    assert svg.count("<polyline") == 0  # the None splits the only two points


def test_cli_report_writes_and_opens(rat, db_path, tmp_path, monkeypatch):
    """`report` writes beside the database and opens it; --no-open and -o behave."""
    opened = []
    monkeypatch.setattr(rat.webbrowser, "open", opened.append)
    cli = rat.Cli()
    cli.runlog = rat.RunLog("chimera", db_path)
    if hasattr(cli, "configure"):
        monkeypatch.setattr(cli, "configure", lambda args: None)

    assert cli.main(["report"]) == 0  # empty history: nothing written, nothing opened
    assert opened == []

    record(rat, db_path, tmp_path, version="0.4.3", image_bytes=b"x")
    assert cli.main(["report"]) == 0
    default = db_path.parent / "report.html"
    assert default.exists()
    assert opened == [default.resolve().as_uri()]

    custom = tmp_path / "custom.html"
    assert cli.main(["report", "--no-open", "-o", str(custom), "-n", "5", "--all-projects"]) == 0
    assert custom.exists()
    assert len(opened) == 1


def test_default_path_is_not_hidden(rat, monkeypatch):
    """Snap browsers cannot read hidden directories, so the report would not open."""
    monkeypatch.delenv("RUNS_DB", raising=False)
    path = rat.RunLog.default_path()
    assert path == Path("~/config/runs/db.sqlite").expanduser()
    assert not any(part.startswith(".") for part in path.relative_to(Path.home()).parts)


def test_report_explains_single_run_group(rat, db_path, tmp_path):
    record(rat, db_path, tmp_path, version="0.4.3", image_bytes=b"x")
    out = tmp_path / "report.html"
    assert rat.RunLog("chimera", db_path).write_report(out, 20) is True
    page = out.read_text()
    assert "One finished run" in page
    assert "<figure>" not in page


def test_report_compares_backends(rat, db_path, tmp_path):
    """Latest run per backend side by side; colour follows the backend; a failed
    case shows its status; a case with no metric on any backend is left out."""
    record(rat, db_path, tmp_path, version="0.4.2", image_bytes=b"a", backend="vulkan")  # superseded below
    log = rat.RunLog("chimera", db_path)
    log.start(target="test-all", backend="vulkan", root=tmp_path, version="0.4.3")
    log.case("gen", "1", 0, 2.5, metrics={"tokens_per_second": 80.0})
    log.case("sd", "3", 1, 30.0)
    log.finish(1)
    log.start(target="test-all", backend="cuda", root=tmp_path, version="0.4.3")
    log.case("gen", "1", 0, 1.5, metrics={"tokens_per_second": 120.0})
    log.case("sd", "3", 0, 20.0)
    log.finish(0)
    out = tmp_path / "report.html"
    assert rat.RunLog("chimera", db_path).write_report(out, 20) is True
    page = out.read_text()
    section = page[page.index("backends side by side") :]
    assert "cuda (run 3, 0.4.3)" in section and "vulkan (run 2, 0.4.3)" in section  # latest per backend
    assert section.index("cuda (run") < section.index("vulkan (run")  # fixed backend order
    assert "var(--b-cuda" in section and "var(--b-vulkan" in section
    assert ">fail</div>" in section  # vulkan sd 3
    tok = section[section.index("tok/s <span>") :]
    tok = tok[: tok.index("</figure>")]
    assert "gen 1" in tok and "sd 3" not in tok and "120.0" in tok and "80.0" in tok


def test_report_no_comparison_for_one_backend(rat, db_path, tmp_path):
    record(rat, db_path, tmp_path, version="0.4.3", image_bytes=b"x")
    record(rat, db_path, tmp_path, version="0.4.3", image_bytes=b"x")
    out = tmp_path / "report.html"
    rat.RunLog("chimera", db_path).write_report(out, 20)
    assert "backends side by side" not in out.read_text()


def version_history(rat, db_path, tmp_path, runs):
    """Record `runs`: (version, {case: (rc, seconds, tok/s or None, image bytes or None)})."""
    log = rat.RunLog("chimera", db_path)
    image = tmp_path / "z_turbo_3.png"
    for version, cases in runs:
        log.start(target="test-all", backend="cuda", root=tmp_path, version=version)
        for (family, n), (rc, secs, tps, img) in cases.items():
            outputs = []
            if img is not None:
                image.write_bytes(img)
                outputs = [image]
            metrics = {"tokens_per_second": tps} if tps is not None else None
            log.case(family, n, rc, secs, outputs=outputs, metrics=metrics)
        log.finish(0)
    return log


def verdicts(rows):
    return {(r["case"], r["measure"]): r["verdict"] for r in rows}


def test_version_rows_verdicts(rat, db_path, tmp_path):
    old = {
        ("gen", "1"): (0, 10.0, 100.0, None),
        ("gen", "2"): (0, 10.0, 50.0, None),
        ("rag", "1"): (0, 5.0, None, None),
        ("sd", "3"): (0, 40.0, None, b"a"),
        ("embed", "1"): (0, 2.0, None, None),
    }
    log = version_history(
        rat,
        db_path,
        tmp_path,
        [
            ("1.0", old),
            ("1.0", {**old, ("rag", "1"): (0, 7.0, None, None)}),  # rag 1 ranges 5..7 in 1.0
            (
                "1.1",
                {
                    ("gen", "1"): (0, 12.0, 80.0, None),  # +20% seconds, -20% tok/s
                    ("gen", "2"): (0, 10.5, 51.0, None),  # +5%: under the threshold
                    ("rag", "1"): (0, 6.6, None, None),  # +10% on the median but inside 5..7
                    ("sd", "3"): (0, 30.0, None, b"b"),  # -25%, new image
                    ("embed", "1"): (1, 2.0, None, None),  # passed in 1.0, fails in 1.1
                },
            ),
        ],
    )
    group = log._query("SELECT * FROM runs ORDER BY id")
    previous, current, n_prev, n_cur, rows = log._version_rows(group)
    assert (previous, current, n_prev, n_cur) == ("1.0", "1.1", 2, 1)
    assert verdicts(rows) == {
        ("gen 1", "seconds"): "regression",
        ("gen 1", "tok/s"): "regression",
        ("gen 2", "seconds"): "within noise",
        ("gen 2", "tok/s"): "within noise",
        ("rag 1", "seconds"): "within noise",
        ("sd 3", "seconds"): "improvement",
        ("embed 1", "status"): "now failing",
    }
    sd = next(r for r in rows if r["case"] == "sd 3")
    assert sd["note"] == "image changed"
    assert next(r for r in rows if r["case"] == "rag 1")["prev"] == 6.0  # median of 5 and 7


def test_version_rows_compares_with_last_different_version(rat, db_path, tmp_path):
    case = {("gen", "1"): (0, 10.0, None, None)}
    log = version_history(rat, db_path, tmp_path, [("1.0", case), ("1.1", case), ("1.2", case)])
    group = log._query("SELECT * FROM runs ORDER BY id")
    assert log._version_rows(group)[:2] == ("1.1", "1.2")
    assert log._version_rows(group[:1]) is None


def test_version_rows_orders_by_number_not_test_order(rat, db_path, tmp_path):
    """A baseline tested after the newer release still compares old -> new."""
    case = {("gen", "1"): (0, 10.0, None, None)}
    log = version_history(rat, db_path, tmp_path, [("0.10.0", case), ("0.9.3", case), ("0.6.0", case)])
    group = log._query("SELECT * FROM runs ORDER BY id")
    assert log._version_rows(group)[:4] == ("0.9.3", "0.10.0", 1, 1)


def test_report_version_section(rat, db_path, tmp_path):
    slow = {("gen", "1"): (0, 15.0, None, None)}
    version_history(rat, db_path, tmp_path, [("1.0", {("gen", "1"): (0, 10.0, None, None)}), ("1.1", slow)])
    out = tmp_path / "report.html"
    rat.RunLog("chimera", db_path).write_report(out, 20)
    page = out.read_text()
    section = page[page.index("Version over version") : page.index("Recent runs")]
    assert "chimera / cuda / test-all: 1 regression(s)" in section
    assert "1.0 (1 run) &rarr; 1.1 (1 run)" in section
    assert 'class="regression">regression<' in section


def test_report_version_section_single_version(rat, db_path, tmp_path):
    record(rat, db_path, tmp_path, version="0.4.3", image_bytes=b"x")
    out = tmp_path / "report.html"
    rat.RunLog("chimera", db_path).write_report(out, 20)
    page = out.read_text()
    assert "nothing to compare: chimera / cuda / test-all (0.4.3, 1 run)" in page
    assert "No regressions" not in page

