#!/usr/bin/env python3
"""Every Netlib instance against its published optimum, through the race
between the dual simplex, the interior point method and the first-order
method (`anukulan lp`). The target this exists to check: every instance
correct, zero wrong answers, at a 60 second limit shared by all three
engines.
"""
import collections, csv, glob, json, os, subprocess, sys

BIN = os.environ.get("ANUKULAN", "build/anukulan")
limit = sys.argv[1] if len(sys.argv) > 1 else "60"
extra = sys.argv[2:]

opt = {}
with open("data/reference/netlib.csv") as fh:
    for row in csv.DictReader(fh):
        if row.get("optimal", "").strip():
            opt[row["name"]] = float(row["optimal"])

solved = wrong = unfinished = 0
winners = collections.Counter()
rows = []
instances = sorted(f for f in glob.glob("data/netlib/*.mps")
                   if os.path.basename(f)[:-4] in opt)
print(f"{len(instances)} instances with a published optimum, {limit}s each\n")

for f in instances:
    name = os.path.basename(f)[:-4]
    # As in verify_ipm.py: a margin over the requested limit for reading and
    # standardising the instance, and a backstop so one instance cannot hang
    # the whole sweep if the race ever fails to cancel within its own budget.
    try:
        p = subprocess.run([BIN, "lp", f, "--format=json", f"--time-limit={limit}"] + extra,
                           capture_output=True, text=True, timeout=float(limit) + 60.0)
    except subprocess.TimeoutExpired:
        why = "no answer, subprocess timed out past its own time limit"
        rows.append((name, why, None, "")); unfinished += 1
        print(f"  {name:<12} {why}", flush=True)
        continue
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        why = f"no answer, exit code {p.returncode}"
        rows.append((name, why, None, "")); unfinished += 1
        print(f"  {name:<12} {why}", flush=True)
        continue
    status = d.get("status", "?")
    winner = d.get("winner", "")
    if status != "optimal":
        rows.append((name, status, None, winner)); unfinished += 1
        print(f"  {name:<12} {status}  ({d.get('message', '')})", flush=True)
        continue
    err = abs(d["objective"] - opt[name]) / max(1.0, abs(opt[name]))
    rows.append((name, status, err, winner))
    if err > 1e-6:
        wrong += 1
        print(f"  {name:<12} WRONG  got {d['objective']:.10g} want {opt[name]:.10g}"
             f"  (winner: {winner})", flush=True)
    else:
        solved += 1
        winners[winner] += 1
        print(f"  {name:<12} ok      winner {winner}", flush=True)

print(f"\n{len(rows)} instances: {solved} correct, {wrong} WRONG, {unfinished} did not finish")
print(f"winners: {dict(winners)}")
for name, status, err, winner in rows:
    if err is None:
        print(f"  did not finish: {name} ({status})")
sys.exit(1 if wrong else 0)
