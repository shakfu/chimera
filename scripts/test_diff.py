#!/usr/bin/env python3
"""
Diff two timings JSON files produced by `scripts/test.py --timings-out`.

Typical use during a regression hunt — capture a baseline on a known-good
ref, capture a "current" timings file on the suspect ref, then diff:

    git checkout main
    make test-bench BENCH=baseline.json

    git checkout dev
    make test-bench BENCH=current.json

    scripts/test_diff.py baseline.json current.json

The output is a single sorted table. Default sort is by absolute wall-clock
delta (largest regressions first). Pass --by-rel for relative delta or
--by-name for stable alphabetical ordering useful for `diff`ing two runs
of this tool itself.

The exit code is 0 if no test regressed past both thresholds, 1 otherwise,
so this is usable as a CI gate ("regression budget: 0.5s and 20%").
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Match the thresholds used by test.py's --timings-baseline inline
# annotation. A row is highlighted as a regression / speedup if BOTH
# the absolute and relative deltas exceed the respective threshold —
# stops a 0.02s -> 0.04s test from spamming the regression list while
# still catching genuine multi-second slowdowns.
REGRESSION_ABS_S = 0.5
REGRESSION_REL   = 0.20


# ----------------------------------------------------------------------------
# ANSI helpers — duplicated from test.py rather than imported so this script
# remains standalone-runnable from anywhere.

def _ansi(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _color_enabled else s


def _red(s: str) -> str:   return _ansi("31", s)
def _green(s: str) -> str: return _ansi("32", s)
def _gray(s: str) -> str:  return _ansi("90", s)


_color_enabled = False


# ----------------------------------------------------------------------------

@dataclass
class Row:
    name:        str
    baseline_s:  Optional[float]
    current_s:   Optional[float]

    @property
    def delta_s(self) -> Optional[float]:
        if self.baseline_s is None or self.current_s is None:
            return None
        return self.current_s - self.baseline_s

    @property
    def delta_rel(self) -> Optional[float]:
        if self.baseline_s is None or self.current_s is None or self.baseline_s <= 0:
            return None
        return (self.current_s - self.baseline_s) / self.baseline_s

    @property
    def is_regression(self) -> bool:
        d, r = self.delta_s, self.delta_rel
        return d is not None and r is not None and d > REGRESSION_ABS_S and r > REGRESSION_REL

    @property
    def is_speedup(self) -> bool:
        d, r = self.delta_s, self.delta_rel
        return d is not None and r is not None and d < -REGRESSION_ABS_S and r < -REGRESSION_REL


def load_run(path: Path) -> tuple[dict, dict]:
    """Returns (metadata_without_tests, {test_name -> duration_s for PASS rows})."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "tests" not in doc:
        raise SystemExit(f"FAIL: {path}: not a timings-JSON document")
    times: dict = {}
    for row in doc["tests"]:
        if row.get("status") != "PASS":
            continue
        n, d = row.get("name"), row.get("duration_s")
        if isinstance(n, str) and isinstance(d, (int, float)):
            times[n] = float(d)
    meta = {k: v for k, v in doc.items() if k != "tests"}
    return meta, times


def build_rows(base: dict, curr: dict) -> list[Row]:
    names = sorted(set(base) | set(curr))
    return [Row(n, base.get(n), curr.get(n)) for n in names]


def fmt_seconds(s: Optional[float]) -> str:
    return "—" if s is None else f"{s:7.2f}s"


def fmt_delta(row: Row) -> str:
    d = row.delta_s
    if d is None:
        return _gray("—")
    sign = "+" if d >= 0 else ""
    text = f"{sign}{d:6.2f}s"
    if row.delta_rel is not None:
        text += f" ({sign}{row.delta_rel * 100:.0f}%)"
    if row.is_regression: return _red(text)
    if row.is_speedup:    return _green(text)
    return text


def main(argv: list) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[3:]))
    p.add_argument("baseline", type=Path, help="Path to the baseline timings JSON.")
    p.add_argument("current",  type=Path, help="Path to the current timings JSON.")
    p.add_argument("--by-rel",  action="store_true",
                   help="Sort rows by relative delta (largest %% first) instead "
                        "of absolute wall-clock delta.")
    p.add_argument("--by-name", action="store_true",
                   help="Sort rows alphabetically by test name (stable order "
                        "for diff-comparing two `test_diff.py` runs).")
    p.add_argument("--only", choices=("regressions", "speedups", "changed", "all"),
                   default="all",
                   help="Filter rows. 'changed' includes both regressions and "
                        "speedups; 'all' includes neutral rows too (default).")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI color output. Forced off if stdout isn't a TTY.")
    args = p.parse_args(argv)

    global _color_enabled
    _color_enabled = (
        not args.no_color and sys.stdout.isatty() and not __import__("os").environ.get("NO_COLOR")
    )

    try:
        base_meta, base = load_run(args.baseline)
        curr_meta, curr = load_run(args.current)
    except (FileNotFoundError, OSError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 2

    rows = build_rows(base, curr)

    # Sort
    if args.by_name:
        rows.sort(key=lambda r: r.name)
    elif args.by_rel:
        rows.sort(key=lambda r: (r.delta_rel if r.delta_rel is not None else 0),
                  reverse=True)
    else:
        # Default: largest absolute delta first. None deltas (only in one
        # side) sink to the bottom — those are reported in a separate
        # section below.
        rows.sort(key=lambda r: (r.delta_s if r.delta_s is not None else float("-inf")),
                  reverse=True)

    # Filter
    if args.only == "regressions":
        rows = [r for r in rows if r.is_regression]
    elif args.only == "speedups":
        rows = [r for r in rows if r.is_speedup]
    elif args.only == "changed":
        rows = [r for r in rows if r.is_regression or r.is_speedup]

    # ---- header ----------------------------------------------------------

    def meta_line(label: str, m: dict) -> str:
        parts = [label]
        for key in ("chimera_version", "llamacpp_version", "whispercpp_version",
                    "sdcpp_version", "generated_at"):
            if m.get(key):
                parts.append(f"{key.replace('_version', '')}={m[key]}")
        return "  " + " · ".join(parts)

    print("test-diff:")
    print(meta_line("baseline", base_meta))
    print(meta_line("current ", curr_meta))
    print()

    in_both = sum(1 for r in rows if r.baseline_s is not None and r.current_s is not None)
    n_regressions = sum(1 for r in rows if r.is_regression)
    n_speedups    = sum(1 for r in rows if r.is_speedup)
    only_base = [r.name for r in build_rows(base, curr) if r.baseline_s is not None and r.current_s is None]
    only_curr = [r.name for r in build_rows(base, curr) if r.baseline_s is None and r.current_s is not None]

    print(f"  {in_both} test(s) in both runs · "
          f"{_red(f'regressions={n_regressions}')} · "
          f"{_green(f'speedups={n_speedups}')}")
    if only_base:
        print(f"  only in baseline ({len(only_base)}): {', '.join(only_base)}")
    if only_curr:
        print(f"  only in current  ({len(only_curr)}): {', '.join(only_curr)}")
    print()

    # ---- table -----------------------------------------------------------

    if not rows:
        print("  (no rows match the filter)")
        return 1 if n_regressions > 0 else 0

    width = max(len(r.name) for r in rows)
    header = f"  {'baseline':>9}  {'current':>9}  {'delta':>20}  test"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(f"  {fmt_seconds(r.baseline_s):>9}  {fmt_seconds(r.current_s):>9}  "
              f"{fmt_delta(r):>20}  {r.name:<{width}}")

    return 1 if n_regressions > 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
