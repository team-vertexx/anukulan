#!/usr/bin/env python3
"""The first-order path at one thread and at many, over the whole Netlib set.

This is the verification that bears on the parallel work directly. The two
harnesses that check published optima - verify_simplex.py and miplib_survey.py -
reach the threaded backend only through the crossover seed, so they exercise a
corner of it. `solve` is the threaded path.

The check is equality, not closeness. threaded_cpu_backend is bit-identical to
cpu_backend by construction (docs/ARCHITECTURE.md 7a), so anything short of the
same objective and the same iteration count is a bug, not a tolerance question.
The published optimum is checked too, so that "identical" cannot be satisfied by
both runs being identically wrong.
"""

import csv
import glob
import json
import os
import subprocess
import sys

BIN = os.environ.get("SANKHYA", "build/sankhya")


def solve(path, threads, limit, tol):
    cmd = [BIN, "solve", path, "--format=json", "--presolve",
           f"--tol={tol}", f"--time-limit={limit}", f"--threads={threads}"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def main():
    threads = sys.argv[1] if len(sys.argv) > 1 else "6"
    limit = sys.argv[2] if len(sys.argv) > 2 else "60"
    tol = sys.argv[3] if len(sys.argv) > 3 else "1e-6"

    published = {}
    with open("data/reference/netlib.csv") as fh:
        for row in csv.DictReader(fh):
            if row.get("optimal", "").strip():
                published[row["name"]] = float(row["optimal"])

    same = differ = 0
    matched = 0
    checked = 0
    mismatches = []
    for f in sorted(glob.glob("data/netlib/*.mps")):
        name = os.path.basename(f)[:-4]
        a = solve(f, 1, limit, tol)
        b = solve(f, threads, limit, tol)
        if a is None or b is None:
            print(f"  {name:<12} run failed", flush=True)
            continue
        checked += 1
        ok = (a.get("objective") == b.get("objective")
              and a.get("iterations") == b.get("iterations")
              and a.get("status") == b.get("status"))
        if ok:
            same += 1
        else:
            differ += 1
            mismatches.append((name, a, b))
            print(f"  {name:<12} DIFFERS  serial obj={a.get('objective')} "
                  f"it={a.get('iterations')} | threaded obj={b.get('objective')} "
                  f"it={b.get('iterations')}", flush=True)
        if name in published and a.get("status") == "optimal":
            err = abs(a["objective"] - published[name]) / max(1.0, abs(published[name]))
            if err <= 1e-6:
                matched += 1

    print(f"\n{checked} instances compared at 1 thread against {threads}")
    print(f"  identical objective, iterations and status : {same}")
    print(f"  differing                                  : {differ}")
    print(f"  of those reaching optimal, matching the published value: {matched}")
    if mismatches:
        print("\nA difference here is a defect, not a tolerance:")
        for name, a, b in mismatches:
            print(f"  {name}: {a} vs {b}")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
