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


# Integrality is declared two different ways in MPS and both have to be looked
# for. A MARKER/INTORG block in COLUMNS is the one everybody remembers; BV, LI
# and UI in BOUNDS is the other, and it is not a rare corner - of the eight
# MIPLIB instances whose LP relaxation moves under presolve, five have no INTORG
# anywhere and declare 1,250 binaries apiece with BV. Checking only for INTORG
# reports those five as presolve corrupting the answer.
INTEGER_BOUND_TYPES = ("BV", "LI", "UI")


def has_integers(path):
    """True when the MPS declares any integer column, by either route."""
    try:
        with open(path, errors="ignore") as fh:
            in_bounds = False
            for line in fh:
                if not line.strip():
                    continue
                if not line[0].isspace():
                    in_bounds = line.split()[0].upper() == "BOUNDS"
                    continue
                if "INTORG" in line:
                    return True
                if in_bounds:
                    fields = line.split()
                    if fields and fields[0].upper() in INTEGER_BOUND_TYPES:
                        return True
    except OSError:
        pass
    return False


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
    # Presolve must not change the answer. Comparing the two runs against each
    # other is the weak form of that check - two runs sharing a bug agree - but
    # it needs no reference value, so it covers every instance rather than the
    # ones somebody has published an optimum for. Where a reference exists the
    # stronger check belongs in verify_presolve.py.
    #
    # It does NOT apply to an LP relaxation of a model with integer columns, and
    # that is not a detail. Presolve is entitled to use integrality - it rounds
    # integer bounds and rounds the implied bounds tightening derives - so the
    # relaxation of the presolved model is legitimately tighter than the
    # relaxation of the original. On beasleyC3 that is 40.43 against 152.05, and
    # the larger number is the better bound rather than a wrong answer. Checking
    # it anyway reports four MIPLIB instances as corrupted by a presolve that is
    # working exactly as intended, which is a test failing for the wrong reason.
    disagreed = []
    skipped_integral = 0
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
        # An LP relaxation of an integer model is not a comparison; see above.
        integral = args.mode == "simplex" and has_integers(model)
        if integral:
            skipped_integral += 1
        oo, on_obj = off.get("objective"), on.get("objective")
        if so and sn and not integral and oo is not None and on_obj is not None:
            scale = max(1.0, abs(oo))
            if abs(oo - on_obj) / scale > 1e-6:
                disagreed.append((name, oo, on_obj, abs(oo - on_obj) / scale))

        mark = ""
        if so and sn and any(d[0] == name for d in disagreed):
            mark = "  <-- ANSWER CHANGED"
        elif so and not sn: mark = "  <-- lost it"
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
    # Printed before the speed summary and never folded into it, because a
    # presolve that changed an answer has no speed to report.
    if disagreed:
        print(f"\nANSWER CHANGED on {len(disagreed)} of {both_solved} instances "
              f"both sides finished:")
        for name, a, b, rel in disagreed:
            print(f"  {name:<26} off {a:.10g}  on {b:.10g}  rel {rel:.3e}")
    else:
        print(f"answers agree on all {both_solved} instances both sides finished")
    if skipped_integral:
        print(f"  ({skipped_integral} skipped: an LP relaxation of an integer "
              f"model, where presolve may legitimately tighten the bound)")
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
