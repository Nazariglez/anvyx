#!/usr/bin/env python3
"""Rust backend regression floor check.

Runs all integration tests with --backend rust and asserts the pass count
never drops below a known floor.

    just rust-coverage        # uses default floor (795)
    just rust-coverage 810    # after landing new coverage
"""

import argparse
import re
import subprocess
import sys


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def run_tests() -> str:
    proc = subprocess.run(
        [
            "cargo", "run", "--quiet", "--package", "test-runner",
            "--", "tests", "--quiet", "--backend", "rust",
        ],
        capture_output=True,
        text=True,
    )
    return proc.stdout + proc.stderr


def parse_count(output: str, label: str) -> int | None:
    clean = strip_ansi(output)
    m = re.search(rf"(\d+) {label}", clean)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(description="Rust backend regression floor check")
    parser.add_argument("--floor", type=int, required=True)
    args = parser.parse_args()

    output = run_tests()
    passed = parse_count(output, "passed")
    failed = parse_count(output, "failed")
    if passed is None or failed is None:
        print("\nERROR: could not parse pass/fail counts from test runner output", file=sys.stderr)
        sys.exit(1)

    if passed < args.floor:
        print(f"\nREGRESSION: {passed} passed < {args.floor} floor", file=sys.stderr)
        sys.exit(1)

    total = passed + failed
    percent = 100.0 if total == 0 else (passed / total) * 100.0
    print(f"\nRust backend: {passed} passed, {failed} failed ({percent:.1f}% pass rate, floor: {args.floor})")


if __name__ == "__main__":
    main()
