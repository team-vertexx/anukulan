#!/usr/bin/env python3
"""Fetch the Netlib LP test set and build the reference table.

The files on netlib are stored in a compressed form that has to be expanded with
the `emps` program that lives in the same directory, so this compiles emps first
and pipes every instance through it.

The README in that directory also carries a table of rows / columns / nonzeros /
optimal objective value for each instance. That table is what we check solver
output against, so it is parsed into data/reference/netlib.csv here rather than
being copied by hand.

Note on the counts in that table: "Rows" includes the objective row and
"Nonzeros" includes the objective row's entries, so a model with 27 constraints
and 83 constraint nonzeros is listed as 28 and 88. The CSV keeps the table's own
convention and the harness compares like with like.

The QAP instances and stocfor3 are listed in the README without a byte count and
are not provided as MPS files in that directory, so they do not end up in the
table. That is intentional, not a parsing failure.
"""

import argparse
import csv
import pathlib
import re
import subprocess
import sys
import urllib.request

BASE = "https://netlib.org/lp/data/"

# Entries in the README that are notes rather than instances.
SKIP = {"ascii", "changes", "readme", "emps.c", "emps.f", "emps.exe.gz", "kennington"}

ROW_RE = re.compile(
    r"^([A-Z0-9_\-]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([A-Z ]{0,4}?)\s*"
    r"(-?[\d.]+E[+\-]\d+|-?[\d.]+)\s*(\*\*)?\s*$"
)


def fetch(url: str, dest: pathlib.Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response:
        dest.write_bytes(response.read())


def parse_readme(text: str):
    """Yields (name, rows, cols, nonzeros, optimal, note) from the index table."""
    for line in text.splitlines():
        match = ROW_RE.match(line.strip())
        if not match:
            continue
        name = match.group(1).lower()
        if name in SKIP:
            continue
        yield (
            name,
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            float(match.group(7)),
            "approximate" if match.group(8) else "",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", help="where to put the data")
    parser.add_argument(
        "--only",
        default="",
        help="comma separated instance names; default is the whole set",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="just rebuild the reference table, do not download instances",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    inst_dir = root / "netlib"
    ref_dir = root / "reference"
    work = root / "_work"
    for d in (inst_dir, ref_dir, work):
        d.mkdir(parents=True, exist_ok=True)

    readme_path = work / "netlib_readme.txt"
    if not readme_path.exists():
        print("fetching README")
        fetch(BASE + "readme", readme_path)
    entries = [list(e) for e in parse_readme(readme_path.read_text(errors="ignore"))]
    if not entries:
        print("could not parse any rows out of the README", file=sys.stderr)
        return 1

    # The README's own optimal values are wrong on eight instances - it handles
    # the objective row's constant differently from every solver that reads the
    # file. A full-set verification made this solver look wrong on eleven Netlib
    # instances at once, which is usually the reference and not the code, and on
    # eight of them it was: HiGHS returns what this solver returns. e226 is the
    # clearest, off by exactly 7.113, which is that instance's objective
    # constant.
    #
    # The corrections live in their own file so re-fetching cannot silently undo
    # them, which is exactly what happened the first time CI ran this.
    corrections_path = ref_dir / "netlib_corrections.csv"
    corrections = {}
    if corrections_path.exists():
        with corrections_path.open() as handle:
            for row in csv.DictReader(handle):
                corrections[row["name"]] = (row["optimal"], row["note"])
    applied = 0
    for entry in entries:
        fix = corrections.get(entry[0])
        if fix is None:
            continue
        entry[4], entry[5] = fix[0], f"corrected against {fix[1]}"
        applied += 1
    if corrections and applied != len(corrections):
        missing = set(corrections) - {e[0] for e in entries}
        print(f"warning: corrections for {sorted(missing)} matched no instance",
              file=sys.stderr)

    ref_path = ref_dir / "netlib.csv"
    with ref_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["name", "table_rows", "table_cols", "table_nonzeros", "optimal", "note"]
        )
        writer.writerows(entries)
    print(f"reference table: {len(entries)} instances "
          f"({applied} corrected) -> {ref_path}")
    if args.table_only:
        return 0

    emps = work / "emps"
    if not emps.exists():
        print("building emps")
        source = work / "emps.c"
        fetch(BASE + "emps.c", source)
        subprocess.run(
            ["cc", "-O2", "-w", "-o", str(emps), str(source)],
            check=True,
        )

    wanted = {n.strip().lower() for n in args.only.split(",") if n.strip()}
    todo = [e for e in entries if not wanted or e[0] in wanted]
    if wanted:
        missing = wanted - {e[0] for e in todo}
        if missing:
            print(f"not in the index: {sorted(missing)}", file=sys.stderr)

    failures = []
    for name, *_ in todo:
        target = inst_dir / f"{name}.mps"
        if target.exists() and target.stat().st_size > 0:
            continue
        packed = work / name
        try:
            if not packed.exists():
                fetch(BASE + name, packed)
            with target.open("wb") as out:
                subprocess.run([str(emps), str(packed)], stdout=out, check=True)
            if target.stat().st_size == 0:
                raise RuntimeError("emps produced an empty file")
            print(f"  {name}")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures.append((name, str(exc)))
            target.unlink(missing_ok=True)

    print(f"expanded {len(todo) - len(failures)}/{len(todo)} instances into {inst_dir}")
    for name, why in failures:
        print(f"  failed: {name}: {why}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
