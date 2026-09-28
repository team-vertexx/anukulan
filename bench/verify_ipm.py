#!/usr/bin/env python3
"""Every Netlib instance against its published optimum, through the interior
point method: alone, and finished by crossover and the simplex.

Same discipline as verify_simplex.py: status alone is not the test, the
objective is, and an "optimal" that disagrees with the published value is
counted as WRONG rather than folded into the failures that just did not
finish.
"""
import csv, glob, json, os, statistics, subprocess, sys

BIN = os.environ.get("ANUKULAN", "build/anukulan")
limit = sys.argv[1] if len(sys.argv) > 1 else "60"
extra = sys.argv[2:]

opt = {}
with open("data/reference/netlib.csv") as fh:
    for row in csv.DictReader(fh):
        if row.get("optimal", "").strip():
            opt[row["name"]] = float(row["optimal"])


def run(name, path, args):
    # A generous margin over the requested limit: reading and standardising a
    # large instance is not itself timed, so the process needs more wall time
    # than the solve budget alone, and this is the sweep's own backstop against
    # a single instance hanging the whole run if the binary's time limit is
    # ever wrong rather than the thing this script is trying to measure.
    try:
        p = subprocess.run([BIN, "ipm", path, "--format=json", f"--time-limit={limit}"] + args + extra,
                           capture_output=True, text=True, timeout=float(limit) + 60.0)
    except subprocess.TimeoutExpired:
        return {"status": "no answer, subprocess timed out past its own time limit"}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"status": f"no answer, exit code {p.returncode}"}


def check(d, name, correct_iters):
    status = d.get("status", "?")
    if status != "optimal":
        return status, None
    err = abs(d["objective"] - opt[name]) / max(1.0, abs(opt[name]))
    if err > 1e-6:
        return "WRONG", err
    correct_iters.append(d)
    return "optimal", err


alone_solved = alone_wrong = alone_unfinished = 0
cross_solved = cross_wrong = cross_unfinished = 0
alone_iters = []
cross_iters = []
alone_failures = []
cross_failures = []

instances = sorted(f for f in glob.glob("data/netlib/*.mps")
                   if os.path.basename(f)[:-4] in opt)
print(f"{len(instances)} instances with a published optimum\n")

for f in instances:
    name = os.path.basename(f)[:-4]

    alone = run(name, f, ["--no-crossover"])
    status, err = check(alone, name, alone_iters)
    if status == "optimal":
        alone_solved += 1
    elif status == "WRONG":
        alone_wrong += 1
        alone_failures.append((name, status, alone.get("objective"), opt[name]))
        print(f"  ALONE     {name:<12} WRONG  got {alone['objective']:.10g} "
             f"want {opt[name]:.10g}", flush=True)
    else:
        alone_unfinished += 1
        alone_failures.append((name, status, alone.get("message", ""), None))
        print(f"  ALONE     {name:<12} {status}  {alone.get('message', '')}", flush=True)

    crossed = run(name, f, [])
    status, err = check(crossed, name, cross_iters)
    if status == "optimal":
        cross_solved += 1
    elif status == "WRONG":
        cross_wrong += 1
        cross_failures.append((name, status, crossed.get("objective"), opt[name]))
        print(f"  CROSSOVER {name:<12} WRONG  got {crossed['objective']:.10g} "
             f"want {opt[name]:.10g}", flush=True)
    else:
        cross_unfinished += 1
        cross_failures.append((name, status, crossed.get("message", ""), None))
        print(f"  CROSSOVER {name:<12} {status}  {crossed.get('message', '')}", flush=True)

print(f"\nIPM alone, tol 1e-8, {limit}s limit:")
print(f"  {alone_solved} reach the published optimum (1e-6 relative), "
     f"{alone_wrong} WRONG, {alone_unfinished} did not finish")
if alone_iters:
    its = [d["ipm_iterations"] for d in alone_iters]
    print(f"  iterations: min {min(its)}, median {statistics.median(its):.0f}, "
         f"max {max(its)}, mean {statistics.mean(its):.1f}")
for name, status, a, b in alone_failures:
    print(f"    {name:<12} {status}  {a}" + (f" (published {b})" if b is not None else ""))

print(f"\nIPM plus crossover and the simplex:")
print(f"  {cross_solved} at the optimum, {cross_wrong} WRONG, {cross_unfinished} did not finish")
if cross_iters:
    its = [d["iterations"] for d in cross_iters]
    seeds = [d["ipm_iterations"] for d in cross_iters]
    print(f"  finishing simplex iterations: min {min(its)}, median {statistics.median(its):.0f}, "
         f"max {max(its)}, mean {statistics.mean(its):.1f}")
    print(f"  ipm seed iterations: min {min(seeds)}, median {statistics.median(seeds):.0f}, "
         f"max {max(seeds)}, mean {statistics.mean(seeds):.1f}")
for name, status, a, b in cross_failures:
    print(f"    {name:<12} {status}  {a}" + (f" (published {b})" if b is not None else ""))

sys.exit(1 if (alone_wrong or cross_wrong) else 0)
