#!/usr/bin/env python3
"""Does the solve that follows presolve actually get faster?

Reduction counts are the easy half. A presolve that removes more and solves
slower is not an improvement, so this measures the thing that decides it:
simplex iterations on the LP side and branch-and-bound nodes on the MILP side,
each with presolve on and off, from the same binary.

Iterations and nodes are reported rather than seconds because they do not move
with machine load - this repository has twice recorded a regression that was
only a busy CPU. Seconds are collected too, and should be read only when the
run was taken quiet.

Pairs are compared instance by instance and reported as a distribution, not
just a geometric mean. A reduction that halves one instance and multiplies
another by sixteen has not "improved things by 2x on average" in any sense a
user would recognise, and the spread is the finding.
"""

import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(binary, mode, model, presolve, limit, extra):
    cmd = [binary, mode, model, "--quiet", "--format=json", f"--time-limit={limit}"]
    if presolve:
        cmd.append("--presolve")
        cmd.extend(extra)
    done = subprocess.run(cmd, capture_output=True, text=True)
    for line in reversed(done.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.path.join(ROOT, "build-work", "sankhya"))
    ap.add_argument("--mode", default="simplex", choices=["simplex", "milp"])
    ap.add_argument("--set", default="netlib")
    ap.add_argument("--limit", default="60")
    ap.add_argument("--extra", default="",
                    help="flags for the presolved side only, so one reduction "
                         "can be switched off and the two sides compared")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    extra = args.extra.split() if args.extra else []

    if args.set == "refinery":
        models = [os.path.join(ROOT, "data", "refinery", "refinery.mps")]
    else:
        directory = os.path.join(ROOT, "data", args.set)
        names = args.instances
        models = ([os.path.join(directory, f"{n}.mps") for n in names] if names
                  else sorted(os.path.join(directory, f)
                              for f in os.listdir(directory) if f.endswith(".mps")))

    # "work" is iterations for the simplex and nodes for the tree.
    key = "iterations" if args.mode == "simplex" else "nodes"
    print(f"{'instance':<26} {'off':>12} {'on':>12} {'ratio':>8}   {'off s':>7} {'on s':>7}")
    ratios, rows = [], []
    both_solved = off_only = on_only = neither = 0
    for model in models:
        name = os.path.basename(model)[:-4]
        off = run(args.binary, args.mode, model, False, args.limit, extra)
        on = run(args.binary, args.mode, model, True, args.limit, extra)
        if off is None or on is None:
            print(f"{name:<26} no result")
            continue

        def solved(d):
            # For the tree, an objective with no incumbent behind it is not an
            # answer. Counting incumbents rather than reading the objective is
            # the only way to tell the difference on an instance whose optimum
            # happens to be zero.
            if args.mode == "milp":
                return d.get("incumbents", 0) > 0
            return d.get("status") == "optimal"

        so, sn = solved(off), solved(on)
        if so and sn: both_solved += 1
        elif so: off_only += 1
        elif sn: on_only += 1
        else: neither += 1

        wo, wn = off.get(key, 0), on.get(key, 0)
        ratio = (wo / wn) if wn else float("nan")
        # Only pairs where both sides finished say anything about work done.
        if so and sn and wo > 0 and wn > 0:
            ratios.append(ratio)
        mark = ""
        if so and not sn: mark = "  <-- lost it"
        elif sn and not so: mark = "  <-- gained it"
        elif so and sn and ratio < 0.8: mark = "  <-- slower"
        print(f"{name:<26} {wo:12d} {wn:12d} {ratio:8.2f}   "
              f"{off.get('seconds', 0):7.2f} {on.get('seconds', 0):7.2f}{mark}")
        rows.append(dict(instance=name, off_work=wo, on_work=wn, ratio=ratio,
                         off_seconds=off.get("seconds"), on_seconds=on.get("seconds"),
                         off_solved=so, on_solved=sn,
                         off_objective=off.get("objective"),
                         on_objective=on.get("objective")))

    print(f"\nboth finished {both_solved}, only without presolve {off_only}, "
          f"only with {on_only}, neither {neither}")
    if ratios:
        geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        faster = sum(1 for r in ratios if r > 1.05)
        slower = sum(1 for r in ratios if r < 0.95)
        same = len(ratios) - faster - slower
        print(f"{key}: geomean {geo:.2f}x less work over {len(ratios)} pairs")
        print(f"  fewer {faster}, more {slower}, unchanged {same}")
        worst = min(ratios)
        best = max(ratios)
        print(f"  best {best:.2f}x, worst {worst:.2f}x")
    if args.csv and rows:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
