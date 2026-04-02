#!/usr/bin/env python3
"""Scan .anv test files for duplicate or near-duplicate bodies."""

import argparse
import difflib
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

DIRECTIVE_PREFIX = "// @"


def parse_test(path: Path) -> tuple[list[str], list[str]] | None:
    """Split a test file into (directives, body). Returns None for helpers."""
    lines = path.read_text().splitlines()
    directives = []
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(DIRECTIVE_PREFIX):
            if stripped.startswith("// @helper"):
                return None
            directives.append(stripped)
            body_start = i + 1
        elif stripped == "" or stripped == "//":
            body_start = i + 1
        else:
            break
    body = lines[body_start:]
    return directives, body


def normalize(body: list[str]) -> list[str]:
    """Normalize body lines: strip trailing whitespace, collapse blank runs."""
    out = []
    prev_blank = False
    for line in body:
        stripped = line.rstrip()
        if stripped == "":
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(stripped)
            prev_blank = False
    # strip leading/trailing blank lines
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return out


def similarity(a: list[str], b: list[str]) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def unified_diff(path_a: Path, body_a: list[str], path_b: Path, body_b: list[str]) -> str:
    diff = difflib.unified_diff(
        body_a, body_b,
        fromfile=str(path_a), tofile=str(path_b),
        lineterm="",
    )
    return "\n".join(diff)


def group_key(path: Path, root: Path) -> str:
    """Group by the first two path components relative to root (e.g. run/arrays)."""
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0] if parts else ""


def collect_tests(root: Path) -> dict[str, list[tuple[Path, list[str], list[str]]]]:
    """Collect and group all non-helper test files."""
    groups: dict[str, list[tuple[Path, list[str], list[str]]]] = defaultdict(list)
    for path in sorted(root.rglob("*.anv")):
        result = parse_test(path)
        if result is None:
            continue
        directives, body = result
        normed = normalize(body)
        if len(normed) < 2:
            continue  # skip trivially small files
        key = group_key(path, root)
        groups[key].append((path, directives, normed))
    return groups


def find_pairs(root: Path, threshold: float, top: int, cross_group: bool):
    """Compute similar pairs and return (groups, pairs)."""
    groups = collect_tests(root)
    pairs = []

    if cross_group:
        all_files = [f for files in groups.values() for f in files]
        for (pa, _da, ba), (pb, _db, bb) in combinations(all_files, 2):
            ratio = similarity(ba, bb)
            if ratio >= threshold:
                pairs.append((ratio, pa, ba, pb, bb))
    else:
        for _, files in sorted(groups.items()):
            if len(files) < 2:
                continue
            for (pa, _da, ba), (pb, _db, bb) in combinations(files, 2):
                ratio = similarity(ba, bb)
                if ratio >= threshold:
                    pairs.append((ratio, pa, ba, pb, bb))

    pairs.sort(key=lambda x: x[0], reverse=True)
    if top > 0:
        pairs = pairs[:top]

    return groups, pairs


def print_simple(root: Path, groups, pairs, threshold: float):
    """Compact one-line-per-pair listing, sorted by similarity descending."""
    total_files = sum(len(v) for v in groups.values())
    print(f"Scanned {total_files} files in {len(groups)} groups")

    if not pairs:
        print(f"No pairs found above {threshold:.0%} similarity.")
        return

    print(f"Found {len(pairs)} pair(s) above {threshold:.0%} similarity:\n")

    for ratio, pa, ba, pb, bb in pairs:
        rel_a = pa.relative_to(root)
        rel_b = pb.relative_to(root)
        print(f"  {ratio:.0%}  {rel_a}  <->  {rel_b}")


def print_full(root: Path, groups, pairs, threshold: float):
    """Detailed output with unified diffs."""
    total_files = sum(len(v) for v in groups.values())
    print(f"Scanned {total_files} files in {len(groups)} groups")

    if not pairs:
        print(f"No pairs found above {threshold:.0%} similarity.")
        return

    print(f"Found {len(pairs)} pair(s) above {threshold:.0%} similarity:\n")
    print("=" * 72)

    for ratio, pa, ba, pb, bb in pairs:
        rel_a = pa.relative_to(root)
        rel_b = pb.relative_to(root)
        print(f"\n{ratio:.0%} similar:")
        print(f"  {rel_a}  ({len(ba)} lines)")
        print(f"  {rel_b}  ({len(bb)} lines)")
        diff = unified_diff(rel_a, ba, rel_b, bb)
        if diff:
            print()
            for line in diff.splitlines():
                print(f"    {line}")
        print()
        print("-" * 72)


def main():
    parser = argparse.ArgumentParser(description="Scan .anv tests for duplicates")
    parser.add_argument(
        "root",
        nargs="?",
        default="tests",
        help="Root test directory (default: tests)",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=int,
        default=70,
        help="Minimum similarity percentage (default: 70)",
    )
    parser.add_argument(
        "-n", "--top",
        type=int,
        default=0,
        help="Show only top N pairs (default: all)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show detailed output with unified diffs",
    )
    parser.add_argument(
        "--cross-group",
        action="store_true",
        help="Compare across groups (slower, O(n^2) on all files)",
    )

    args = parser.parse_args()
    root = Path(args.root).resolve()

    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    threshold = args.threshold / 100
    groups, pairs = find_pairs(root, threshold, args.top, args.cross_group)

    if args.full:
        print_full(root, groups, pairs, threshold)
    else:
        print_simple(root, groups, pairs, threshold)


if __name__ == "__main__":
    main()
