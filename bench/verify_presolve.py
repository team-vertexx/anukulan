#!/usr/bin/env python3
"""Check that presolve does not change the answer, and measure what it buys.

Presolve is the reduction most able to be confidently wrong: it deletes parts of
a model and then claims values for the parts it deleted. Two things therefore
have to be checked separately, and a speedup is not one of them.

  1. Correctness.  Solve each instance twice, plain and presolved, and compare
     both objectives against the published optimum - not against each other. Two
     runs sharing a presolve bug would agree with each other perfectly.
  2. Feasibility.  Hand the postsolved point to check_feasibility.py, which
     parses the original MPS in Python and shares no code with the solver, so a
     column restored to the wrong value has nowhere to hide.

Only then is the timing worth reading.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def published_optima(path):
    out = {}
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            value = (row.get("optimal") or "").strip()
            if value:
                try:
                    out[row["name"].strip()] = float(value)
                except ValueError:
                    pass
    return out


def solve(binary, model, presolve, tol, time_limit, max_iter, solution=None):
    cmd = [binary, "solve", model, "--quiet", "--format=json",
           f"--tol={tol}", f"--time-limit={time_limit}", f"--max-iter={max_iter}"]
    if presolve:
        cmd.append("--presolve")
    if solution:
        cmd.append(f"--solution={solution}")
    done = subprocess.run(cmd, capture_output=True, text=True)
    for line in reversed(done.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def feasibility(model, solution):
    done = subprocess.run(
        [sys.executable, os.path.join(HERE, "check_feasibility.py"), model, solution],
        capture_output=True, text=True)
    worst = None
    for line in done.stdout.splitlines():
        parts = line.split()
        if line.lower().startswith("worst") and parts:
            for token in parts:
                try:
                    worst = max(worst or 0.0, abs(float(token)))
                except ValueError:
                    continue
    return worst, done.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.path.join(ROOT, "build", "sankhya"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "netlib"))
    ap.add_argument("--reference",
                    default=os.path.join(ROOT, "data", "reference", "netlib.csv"))
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--tol", default="1e-6")
    ap.add_argument("--time-limit", default="300")
    ap.add_argument("--max-iter", default="1000000")
    ap.add_argument("--tolerance", type=float, default=1e-5,
                    help="relative objective error allowed against the published optimum")
    ap.add_argument("--check-feasibility", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    optima = published_optima(args.reference)
    names = args.instances
    if not names:
        names = sorted(n for n in optima
                       if os.path.exists(os.path.join(args.data, f"{n}.mps")))

    print(f"{'instance':12s} {'published':>15s} {'plain err':>10s} {'pre err':>10s} "
          f"{'plain s':>9s} {'pre s':>9s} {'iters plain':>11s} {'iters pre':>10s}")
    rows, speedups = [], []
    agreed = compared = regressed = 0
    for name in names:
        model = os.path.join(args.data, f"{name}.mps")
        target = optima.get(name)
        if target is None or not os.path.exists(model):
            continue
        plain = solve(args.binary, model, False, args.tol, args.time_limit, args.max_iter)
        sol = os.path.join("/tmp", f"presolved_{name}.sol") if args.check_feasibility else None
        pre = solve(args.binary, model, True, args.tol, args.time_limit, args.max_iter, sol)
        if plain is None or pre is None:
            print(f"{name:12s} did not produce a result")
            continue

        scale = max(1.0, abs(target))
        err_plain = abs(plain["objective"] - target) / scale
        err_pre = abs(pre["objective"] - target) / scale
        compared += 1
        good = err_pre < args.tolerance and pre["status"] == "optimal"
        agreed += 1 if good else 0
        # A presolve that is correct but slower is still a finding worth naming.
        worse = plain["status"] == "optimal" and pre["status"] != "optimal"
        regressed += 1 if worse else 0

        note = ""
        if not good:
            note = "  <-- wrong" if err_pre >= args.tolerance else "  <-- did not converge"
        elif pre["seconds"] > 1.5 * plain["seconds"] and plain["seconds"] > 0.05:
            note = "  <-- slower"

        print(f"{name:12s} {target:15.6f} {err_plain:10.2e} {err_pre:10.2e} "
              f"{plain['seconds']:9.2f} {pre['seconds']:9.2f} "
              f"{plain['iterations']:11d} {pre['iterations']:10d}{note}")

        if plain["seconds"] > 0.05 and pre["seconds"] > 0:
            speedups.append(plain["seconds"] / pre["seconds"])

        worst = None
        if args.check_feasibility and sol and os.path.exists(sol):
            worst, _ = feasibility(model, sol)
            if worst is not None and worst > 1e-5:
                print(f"{'':12s} postsolved point violates the original model by {worst:.3e}")
        rows.append(dict(instance=name, published=target,
                         plain_objective=plain["objective"], plain_error=err_plain,
                         plain_status=plain["status"], plain_seconds=plain["seconds"],
                         plain_iterations=plain["iterations"],
                         presolved_objective=pre["objective"], presolved_error=err_pre,
                         presolved_status=pre["status"], presolved_seconds=pre["seconds"],
                         presolved_iterations=pre["iterations"],
                         worst_violation="" if worst is None else worst))

    print(f"\nreached the published optimum with presolve on: {agreed}/{compared}")
    if regressed:
        print(f"stopped converging once presolved: {regressed}")
    if speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"geomean wall-clock speedup: {geo:.2f}x over {len(speedups)} instances "
              f"that take more than 50 ms")
        print(f"best {max(speedups):.2f}x, worst {min(speedups):.2f}x")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    return 0 if agreed == compared else 1


if __name__ == "__main__":
    sys.exit(main())
