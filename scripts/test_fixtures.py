#!/usr/bin/env python3
"""
Convenience runner for the env-var-gated fixture tests.

Six tests in scripts/test.py are SKIPped by default because they need
adapter / aux-model fixtures chimera doesn't ship (LoRA, ControlNet,
PhotoMaker — large, shape-specific, and in some cases license-encumbered).
This script wraps test.py with the right --filter regex and reports
up-front which fixture sets are configured, so a partial config doesn't
silently SKIP everything.

Discovery is via env vars (see docs/dev/maintenance.md for the full
table); paths are validated before invoking the test runner so a broken
path produces a clear message rather than a deep-in-the-stack FAIL.

Usage:
  CHIMERA_TEST_LORA=/path/lora.safetensors \\
      python3 scripts/test_fixtures.py

  # All three sets at once:
  CHIMERA_TEST_LORA=... \\
  CHIMERA_TEST_CONTROLNET=... CHIMERA_TEST_CONTROL_IMAGE=... \\
  CHIMERA_TEST_PHOTOMAKER=... CHIMERA_TEST_PM_ID_DIR=... \\
      python3 scripts/test_fixtures.py

  # No env vars: prints what would run and exits 0.
  python3 scripts/test_fixtures.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# (label, [(env_var, expected_kind), ...]) where expected_kind is "file" or "dir".
# Order matches the columns in docs/dev/maintenance.md's fixture table.
FIXTURES = [
    (
        "LoRA",
        [
            ("CHIMERA_TEST_LORA", "file"),
        ],
    ),
    (
        "ControlNet",
        [
            ("CHIMERA_TEST_CONTROLNET", "file"),
            ("CHIMERA_TEST_CONTROL_IMAGE", "file"),
        ],
    ),
    (
        "PhotoMaker",
        [
            ("CHIMERA_TEST_PHOTOMAKER", "file"),
            ("CHIMERA_TEST_PM_ID_DIR", "dir"),
        ],
    ),
]

# The same regex the deleted CI workflow used. Matches both SKIP placeholders
# (e.g. "sd --lora") and the PASS variants ("sd --lora (CHIMERA_TEST_LORA
# fixture)"), so an operator with a partial fixture set still sees which
# slots were exercised vs skipped.
FILTER = (
    "(sd --lora|sd --control-net|sd --photo-maker"
    "|serve loras success|serve control_image success"
    "|serve pm_id_image_set success)"
)


def probe_one(label: str, vars_and_kinds: list[tuple[str, str]]) -> str:
    """Return one of: 'present', 'partial', 'invalid', 'absent'."""
    values = [(v, os.environ.get(v, "")) for v, _ in vars_and_kinds]
    set_count = sum(1 for _, val in values if val)
    if set_count == 0:
        return "absent"
    if set_count != len(values):
        return "partial"
    for (name, kind), (_, val) in zip(vars_and_kinds, values):
        path = Path(val)
        if kind == "file" and not path.is_file():
            return "invalid"
        if kind == "dir" and not path.is_dir():
            return "invalid"
    return "present"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe the fixtures and print the command, but don't invoke test.py.",
    )
    p.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to scripts/test.py (e.g. --no-color, --verbose).",
    )
    args = p.parse_args(argv)

    print("Fixture probe:")
    any_present = False
    any_invalid = False
    any_partial = False
    for label, vars_and_kinds in FIXTURES:
        status = probe_one(label, vars_and_kinds)
        names = [v for v, _ in vars_and_kinds]
        if status == "present":
            print(f"  [+] {label:<11} ready ({', '.join(names)})")
            any_present = True
        elif status == "partial":
            print(f"  [!] {label:<11} PARTIAL config; need ALL of: {', '.join(names)}")
            any_partial = True
        elif status == "invalid":
            print(
                f"  [!] {label:<11} env set but path(s) missing on disk: {', '.join(names)}"
            )
            any_invalid = True
        else:
            print(f"  [-] {label:<11} skipped (set {', '.join(names)} to enable)")

    if any_partial or any_invalid:
        # A partial set is a misuse, not a SKIP — same contract as test.py.
        # Mirror that here so a typo in the env var name fails fast.
        print(
            "\nError: partial or invalid fixture configuration. "
            "Set ALL env vars for a fixture set, and point each at an existing path."
        )
        return 2

    if not any_present:
        print(
            "\nNo fixtures configured — nothing to do.\n"
            "See docs/dev/maintenance.md for the env-var table."
        )
        return 0

    repo_root = Path(__file__).resolve().parent.parent
    test_py = repo_root / "scripts" / "test.py"
    cmd = [
        sys.executable,
        str(test_py),
        "--filter",
        FILTER,
        *args.passthrough,
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
