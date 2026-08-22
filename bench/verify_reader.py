#!/usr/bin/env python3
"""Check our MPS reader against two independent references.

Primary oracle is HiGHS: it prints the model dimensions it read, and those must
match ours exactly. Secondary is the Netlib index table, which counts the
objective row and its entries, so a model with 27 constraints and 83 constraint
nonzeros is listed there as 28 and 88.

Where the two references disagree, HiGHS wins and the difference is recorded as a
known exception below rather than being silently tolerated.
"""

import argparse
import csv
import json
import pathlib
import re
import subprocess
import sys

# Instances where the Netlib index itself is wrong. Verified by hand: for
# boeing1 the file holds 1 objective row plus 351 constraints, HiGHS reads 351
# constraints, and the index's nonzero count follows the usual convention
# (3485 + 380 objective entries = 3865). Only its row count does not.
INDEX_EXCEPTIONS = {
    "boeing1": "index row count is 351; file and HiGHS both give 351 constraints, so the table is short by the objective row",
}

HIGHS_DIMS = re.compile(r"has\s+(\d+)\s+rows;\s+(\d+)\s+cols;\s+(\d+)\s+nonzeros")


def highs_dimensions(binary: str, path: pathlib.Path):
    proc = subprocess.run([binary, str(path)], capture_output=True, text=True)
    match = HIGHS_DIMS.search(proc.stdout)
    if not match:
        return None
    return tuple(int(g) for g in match.groups())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build/sankhya")
    parser.add_argument("--highs", default="highs")
    parser.add_argument("--instances", default="data/netlib")
    parser.add_argument("--reference", default="data/reference/netlib.csv")
    parser.add_argument("--skip-highs", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    reference = {
        row["name"]: row for row in csv.DictReader(open(args.reference, newline=""))
    }
    inst_dir = pathlib.Path(args.instances)

    checked = 0
    highs_checked = 0
    highs_mismatch = []
    table_mismatch = []
    unreadable = []
    warned = []
    fixed_format = []

    for name in sorted(reference):
        path = inst_dir / f"{name}.mps"
        if not path.exists():
            continue
        proc = subprocess.run(
            [args.binary, "read", str(path), "--format=json", "--quiet"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()
            unreadable.append((name, tail[-1] if tail else ""))
            continue

        got = json.loads(proc.stdout)
        want = reference[name]
        checked += 1
        if got.get("fixed_format"):
            fixed_format.append(name)
        if got["warnings"]:
            warned.append((name, got["warnings"]))

        if not args.skip_highs:
            dims = highs_dimensions(args.highs, path)
            if dims is not None:
                highs_checked += 1
                mine = (got["rows"], got["cols"], got["nnz"])
                if mine != dims:
                    highs_mismatch.append((name, mine, dims))

        problems = []
        for key, column in (
            ("netlib_rows", "table_rows"),
            ("netlib_cols", "table_cols"),
            ("netlib_nnz", "table_nonzeros"),
        ):
            if got[key] != int(want[column]):
                problems.append(f"{column}: got {got[key]}, table says {want[column]}")
        if problems and name not in INDEX_EXCEPTIONS:
            table_mismatch.append((name, problems))

        if args.verbose:
            flag = "ok"
            if problems:
                flag = "index?" if name in INDEX_EXCEPTIONS else "MISMATCH"
            print(
                f"{flag:9s} {name:12s} {got['rows']:6d} x {got['cols']:6d}"
                f"  nnz {got['nnz']:8d}"
                f"{'  [fixed format]' if got.get('fixed_format') else ''}"
            )

    print(f"\nread            {checked} instances from {inst_dir}")
    print(f"  vs HiGHS      {highs_checked} compared, {len(highs_mismatch)} mismatched")
    print(f"  vs index      {len(table_mismatch)} mismatched "
          f"({len(INDEX_EXCEPTIONS)} known index errors excluded)")
    print(f"  unreadable    {len(unreadable)}")
    print(f"  with warnings {len(warned)}")
    print(f"  fixed format  {len(fixed_format)}"
          + (f" ({', '.join(fixed_format)})" if fixed_format else ""))

    for name, mine, theirs in highs_mismatch:
        print(f"  HIGHS MISMATCH {name}: ours {mine} vs HiGHS {theirs}")
    for name, problems in table_mismatch:
        print(f"  INDEX MISMATCH {name}: " + "; ".join(problems))
    for name, err in unreadable:
        print(f"  UNREADABLE {name}: {err}")
    for name, count in warned:
        print(f"  warnings {name}: {count}")
    for name, why in INDEX_EXCEPTIONS.items():
        print(f"  known index error: {name} - {why}")

    return 1 if (highs_mismatch or table_mismatch or unreadable) else 0


if __name__ == "__main__":
    sys.exit(main())
