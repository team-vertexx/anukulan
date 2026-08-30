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
# Which binary to survey, and where to keep the per-instance records. Both exist
# so an A/B can run two builds without either overwriting the other's tree, and
# so a later run can be diffed against an earlier one instance by instance
# rather than on the totals - a change that gains two instances and loses two
# reads as no change on the totals alone.
binary = os.environ.get("SANKHYA_BIN", "build/sankhya")
record_to = os.environ.get("SANKHYA_SURVEY_OUT")
opt = {}
with open("data/reference/miplib.csv") as fh:
    for row in csv.DictReader(fh):
        opt[row["name"]] = float(row["optimal"])

rows = []
records = []
files = [f for f in sorted(glob.glob("data/miplib/*.mps"))
         if os.path.basename(f)[:-4] in opt]
total = len(files)
for f in files:
    name = os.path.basename(f)[:-4]
    t = time.perf_counter()
    p = subprocess.run([binary, "milp", f, "--format=json",
                        f"--time-limit={limit}"] + extra,
                       capture_output=True, text=True)
    wall = time.perf_counter() - t
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        rows.append((name, "read/parse failed", None, None, wall, 0))
        records.append({"name": name, "status": "read/parse failed", "wall": wall})
        print(f"  {len(rows):>3}/{total} {name[:26]:<27} read/parse failed", flush=True); continue
    o = d.get("objective")
    err = abs(o - opt[name]) / max(1.0, abs(opt[name])) * 100 if o is not None else None
    rows.append((name, d.get("status", "?"), o, err, wall, d.get("nodes", 0)))
    d["name"], d["wall"], d["err"] = name, wall, err
    records.append(d)
    # A solution in hand is the thing the tree has nothing to prune against
    # without, so it is worth seeing per instance and not only in the total.
    mark = "  " if d.get("incumbents", 0) else " -"
    print(f"  {len(rows):>3}/{total} {name[:26]:<27} {d.get('status','?')[:12]:<13} "
          f"{'' if err is None else f'{err:8.3f}%'} {wall:6.1f}s{mark}", flush=True)

solved = [r for r in rows if r[1] == "optimal"]
close = [r for r in rows if r[1] != "optimal" and r[3] is not None and r[3] < 1.0]
print(f"{'instance':<28} {'status':<14} {'err%':>9} {'secs':>7} {'nodes':>9}")
for r in sorted(rows, key=lambda r: (r[1] != "optimal", r[3] if r[3] is not None else 1e9)):
    e = f"{r[3]:9.3f}" if r[3] is not None else "        -"
    print(f"{r[0][:27]:<28} {r[1][:13]:<14} {e} {r[4]:7.1f} {r[5]:>9}")
# The count that matters first. A tree with no incumbent has nothing to prune
# against, so it explores blind and the bound never moves - which is a different
# and worse failure than a poor gap, and the totals used to hide it inside
# "not optimal".
# Read from the incumbent count, not from the objective. A build that leaves the
# objective at its default zero when it found nothing reports a number either
# way, and on acc-tight4 - whose optimum is 0 - that number scores as an exact
# answer. The count of incumbents is the thing that cannot be faked.
feasible = [r for r in records if r.get("incumbents", 0) > 0]
print(f"\n{len(rows)} instances: {len(feasible)} ended with a feasible solution, "
      f"{len(solved)} proved optimal, "
      f"{len(close)} more within 1% of the published optimum")
blank = [r["name"] for r in records if r.get("incumbents", 0) == 0]
print(f"no feasible solution at all: {len(blank)}"
      + ("" if not blank else "  (" + ", ".join(sorted(blank)) + ")"))
if record_to:
    with open(record_to, "w") as fh:
        json.dump(records, fh, indent=1)
    print(f"records written to {record_to}")
wrong = [r for r in solved if r[3] is not None and r[3] > 1e-4]
if wrong:
    print("CLAIMED OPTIMAL BUT WRONG: " + ", ".join(f"{r[0]} ({r[3]:.4f}%)" for r in wrong))
else:
    print("every instance claimed optimal matches the published optimum")
