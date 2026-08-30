#!/usr/bin/env python3
"""This solver and HiGHS on the same MIPLIB instances at the same limit.

MIPLIB 2017 at a short limit is a hard way to be judged and an honest one, but
an absolute count of "solved" from it says very little on its own - most of the
set is out of reach for any solver in fifteen seconds. What says something is
the same number for an established solver on the same machine with the same
budget, which is what this produces.

HiGHS is run single-threaded because this solver is, and a comparison against
eight threads would be measuring the machine rather than the code.
"""
import csv, glob, json, os, re, subprocess, sys, time

ours = os.environ.get("SANKHYA", "build/sankhya")
highs = os.environ.get("HIGHS", "highs")
limit = sys.argv[1] if len(sys.argv) > 1 else "15"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 999

opt = {}
with open("data/reference/miplib.csv") as fh:
    for row in csv.DictReader(fh):
        opt[row["name"]] = float(row["optimal"])

def run_ours(path):
    p = subprocess.run([ours, "milp", path, "--format=json", f"--time-limit={limit}"],
                       capture_output=True, text=True)
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return ("read failed", None)
    if d.get("incumbents", 0) == 0 or d.get("objective") is None:
        return (d.get("status", "?"), None)
    return (d.get("status", "?"), d["objective"])

def run_highs(path):
    p = subprocess.run([highs, "--time_limit", limit, "--threads", "1", path],
                       capture_output=True, text=True)
    text = p.stdout
    status = re.search(r"Status\s+(.+)", text)
    primal = re.search(r"Primal bound\s+(\S+)", text)
    status = status.group(1).strip() if status else "?"
    try:
        value = float(primal.group(1))
    except Exception:
        return (status, None)
    if not (abs(value) < 1e29):
        return (status, None)
    return (status, value)

def err(name, value):
    if value is None: return None
    return abs(value - opt[name]) / max(1.0, abs(opt[name])) * 100

files = [f for f in sorted(glob.glob("data/miplib/*.mps"))
         if os.path.basename(f)[:-4] in opt][:count]

print(f"{'instance':<26} {'sankhya':>26} {'HiGHS':>26}")
rows = []
for f in files:
    name = os.path.basename(f)[:-4]
    a_status, a_val = run_ours(f)
    b_status, b_val = run_highs(f)
    ae, be = err(name, a_val), err(name, b_val)
    def show(st, e):
        return f"{st[:14]:<15}{'   none' if e is None else f'{e:9.3f}%'}"
    rows.append((name, a_status, ae, b_status, be))
    print(f"{name[:25]:<26} {show(a_status, ae):>26} {show(b_status, be):>26}", flush=True)

ours_have = sum(1 for r in rows if r[2] is not None)
highs_have = sum(1 for r in rows if r[4] is not None)
ours_opt = sum(1 for r in rows if r[1] == "optimal")
highs_opt = sum(1 for r in rows if r[3].lower().startswith("optimal"))
both = [r for r in rows if r[2] is not None and r[4] is not None]
ours_better = sum(1 for r in both if r[2] < r[4] - 1e-9)
highs_better = sum(1 for r in both if r[4] < r[2] - 1e-9)
only_ours = [r[0] for r in rows if r[2] is not None and r[4] is None]

print(f"\n{len(rows)} instances at a {limit}s limit, single threaded")
print(f"  found any solution     sankhya {ours_have:>3}    HiGHS {highs_have:>3}")
print(f"  proved optimal         sankhya {ours_opt:>3}    HiGHS {highs_opt:>3}")
print(f"  where both found one   sankhya closer on {ours_better}, HiGHS closer on {highs_better}")
if only_ours:
    print("  found only by sankhya: " + ", ".join(only_ours))
