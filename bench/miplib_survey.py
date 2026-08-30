#!/usr/bin/env python3
"""Which MIPLIB instances this solver can actually finish, and how close it gets.

The seven instances the branch-and-bound defaults were tuned against are a small
sample, and every commit that set one of those defaults says so. This runs the
wider set fetched by scripts/fetch_miplib.py so those choices can be rechecked
against something harder to overfit.
"""
import csv, json, subprocess, sys, time, os, glob

limit = sys.argv[1] if len(sys.argv) > 1 else "15"
extra = sys.argv[2:]
opt = {}
with open("data/reference/miplib.csv") as fh:
    for row in csv.DictReader(fh):
        opt[row["name"]] = float(row["optimal"])

rows = []
for f in sorted(glob.glob("data/miplib/*.mps")):
    name = os.path.basename(f)[:-4]
    if name not in opt: continue
    t = time.perf_counter()
    p = subprocess.run(["build/sankhya", "milp", f, "--format=json",
                        f"--time-limit={limit}"] + extra,
                       capture_output=True, text=True)
    wall = time.perf_counter() - t
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        rows.append((name, "read/parse failed", None, None, wall, 0))
        print(f"  {len(rows):>3}/103 {name[:26]:<27} read/parse failed", flush=True); continue
    o = d.get("objective")
    err = abs(o - opt[name]) / max(1.0, abs(opt[name])) * 100 if o is not None else None
    rows.append((name, d.get("status", "?"), o, err, wall, d.get("nodes", 0)))
    print(f"  {len(rows):>3}/103 {name[:26]:<27} {d.get('status','?')[:12]:<13} "
          f"{'' if err is None else f'{err:8.3f}%'} {wall:6.1f}s", flush=True)

solved = [r for r in rows if r[1] == "optimal"]
close = [r for r in rows if r[1] != "optimal" and r[3] is not None and r[3] < 1.0]
print(f"{'instance':<28} {'status':<14} {'err%':>9} {'secs':>7} {'nodes':>9}")
for r in sorted(rows, key=lambda r: (r[1] != "optimal", r[3] if r[3] is not None else 1e9)):
    e = f"{r[3]:9.3f}" if r[3] is not None else "        -"
    print(f"{r[0][:27]:<28} {r[1][:13]:<14} {e} {r[4]:7.1f} {r[5]:>9}")
print(f"\n{len(rows)} instances: {len(solved)} proved optimal, "
      f"{len(close)} more within 1% of the published optimum")
wrong = [r for r in solved if r[3] is not None and r[3] > 1e-4]
if wrong:
    print("CLAIMED OPTIMAL BUT WRONG: " + ", ".join(f"{r[0]} ({r[3]:.4f}%)" for r in wrong))
else:
    print("every instance claimed optimal matches the published optimum")
