#!/usr/bin/env python3
"""Every Netlib instance against its published optimum, through the simplex.

Status alone is not the test. An instance that reports optimal and returns the
wrong number is the failure this exists to catch, and it has happened here more
than once - fit1p reported 33,609 against a true 9,146.38 and looked healthy,
modszk1 reported a bounded model unbounded. So the check is the objective.
"""
import csv, glob, json, os, subprocess, sys

BIN = os.environ.get("SANKHYA", "build/sankhya")
limit = sys.argv[1] if len(sys.argv) > 1 else "45"
extra = sys.argv[2:]

opt = {}
with open("data/reference/netlib.csv") as fh:
    for row in csv.DictReader(fh):
        if row.get("optimal", "").strip():
            opt[row["name"]] = float(row["optimal"])

solved = wrong = unfinished = 0
rows = []
for f in sorted(glob.glob("data/netlib/*.mps")):
    name = os.path.basename(f)[:-4]
    if name not in opt: continue
    p = subprocess.run([BIN, "simplex", f, "--format=json", f"--time-limit={limit}"] + extra,
                       capture_output=True, text=True)
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        rows.append((name, "read failed", None)); unfinished += 1; continue
    status = d.get("status", "?")
    if status != "optimal":
        rows.append((name, status, None)); unfinished += 1
        print(f"  {name:<12} {status}", flush=True)
        continue
    err = abs(d["objective"] - opt[name]) / max(1.0, abs(opt[name]))
    rows.append((name, status, err))
    if err > 1e-6:
        wrong += 1
        print(f"  {name:<12} WRONG  got {d['objective']:.10g} want {opt[name]:.10g}", flush=True)
    else:
        solved += 1

print(f"\n{len(rows)} instances: {solved} correct, {wrong} WRONG, {unfinished} did not finish")
for name, status, err in rows:
    if err is None and status != "read failed":
        print(f"  did not finish: {name} ({status})")
sys.exit(1 if wrong else 0)
