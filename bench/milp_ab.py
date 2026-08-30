#!/usr/bin/env python3
"""Two binaries, the same instances, the same limit - what changed.

The measure that matters first is not the gap but whether there is an incumbent
at all: a tree with nothing to prune against is exploring blind, and on the
wider MIPLIB set that was the common case rather than the exception.
"""
import csv, glob, json, os, subprocess, sys, time

old_bin = sys.argv[1]
new_bin = sys.argv[2]
limit = sys.argv[3] if len(sys.argv) > 3 else "15"
count = int(sys.argv[4]) if len(sys.argv) > 4 else 999

opt = {}
with open("data/reference/miplib.csv") as fh:
    for row in csv.DictReader(fh):
        opt[row["name"]] = float(row["optimal"])

def run(binary, path):
    p = subprocess.run([binary, "milp", path, "--format=json", f"--time-limit={limit}"],
                       capture_output=True, text=True)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return None

def summary(d, name):
    if d is None: return ("read failed", None)
    o = d.get("objective")
    if d.get("incumbents", 0) == 0 or o is None:
        return (d.get("status", "?"), None)
    return (d.get("status", "?"), abs(o - opt[name]) / max(1.0, abs(opt[name])) * 100)

files = [f for f in sorted(glob.glob("data/miplib/*.mps"))
         if os.path.basename(f)[:-4] in opt][:count]

print(f"{'instance':<26} {'before':>22} {'after':>22}   verdict")
gained = lost = better = worse = 0
rows = []
for f in files:
    name = os.path.basename(f)[:-4]
    a = summary(run(old_bin, f), name)
    b = summary(run(new_bin, f), name)
    def show(t):
        return f"{t[0][:11]:<12}{'  none' if t[1] is None else f'{t[1]:7.3f}%'}"
    verdict = ""
    if a[1] is None and b[1] is not None: verdict = "FOUND ONE"; gained += 1
    elif a[1] is not None and b[1] is None: verdict = "LOST IT"; lost += 1
    elif a[1] is not None and b[1] is not None:
        if b[1] < a[1] - 1e-9: verdict = "better"; better += 1
        elif b[1] > a[1] + 1e-9: verdict = "worse"; worse += 1
    rows.append((name, a, b, verdict))
    print(f"{name[:25]:<26} {show(a):>22} {show(b):>22}   {verdict}", flush=True)

have_a = sum(1 for r in rows if r[1][1] is not None)
have_b = sum(1 for r in rows if r[2][1] is not None)
print(f"\n{len(rows)} instances")
print(f"  with an incumbent: {have_a} -> {have_b}")
print(f"  newly found {gained}, lost {lost}, closer {better}, further {worse}")
