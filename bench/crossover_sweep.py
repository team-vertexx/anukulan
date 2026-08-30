#!/usr/bin/env python3
"""Does seeding the simplex from a first-order point actually save pivots?

Iterations are reported rather than seconds, because the machine may be busy
and an iteration count is not. The seed solve's own cost is reported beside
them so the trade is visible: crossover is only worth it if the pivots saved
are worth the iterations spent getting the point.
"""
import glob, json, os, subprocess, sys

BIN = os.environ.get("SANKHYA", "build/sankhya")
tol = sys.argv[1] if len(sys.argv) > 1 else "1e-4"
only = sys.argv[2:] 

def run(path, extra):
    p = subprocess.run([BIN, "simplex", path, "--format=json"] + extra,
                       capture_output=True, text=True)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return None

files = sorted(glob.glob("data/netlib/*.mps"))
if only:
    files = [f for f in files if os.path.basename(f)[:-4] in only]

print(f"{'instance':<12} {'cold':>7} {'warm':>7} {'ratio':>6} | "
      f"{'pushed':>7} {'of':>6} {'left':>6} {'seed it':>8}  status")
cold_total = warm_total = 0
wins = losses = same = 0
bad = []
for f in files:
    name = os.path.basename(f)[:-4]
    a = run(f, [])
    b = run(f, ["--crossover", f"--crossover-tol={tol}"])
    if not a or not b:
        print(f"{name:<12} run failed"); continue
    if a["status"] != "optimal" or b["status"] != "optimal":
        print(f"{name:<12} {a['status']:>10} -> {b['status']:>10}   skipped")
        continue
    if abs(a["objective"] - b["objective"]) > 1e-6 * max(1.0, abs(a["objective"])):
        bad.append((name, a["objective"], b["objective"]))
    ci, wi = a["iterations"], b["iterations"]
    cold_total += ci; warm_total += wi
    if wi < ci: wins += 1
    elif wi > ci: losses += 1
    else: same += 1
    print(f"{name:<12} {ci:>7} {wi:>7} {wi/max(1,ci):>6.2f} | "
          f"{b['crossover_pushed']:>7} {b['crossover_candidates']:>6} "
          f"{b['crossover_logicals_left']:>6} {b['crossover_seed_iterations']:>8}  "
          f"{'warm' if b.get('started_warm') else 'COLD'}", flush=True)

print(f"\n{wins} fewer, {losses} more, {same} unchanged")
print(f"simplex iterations {cold_total} -> {warm_total}  "
      f"({warm_total/max(1,cold_total):.2f}x)")
if bad:
    print("DIFFERENT ANSWER: " + ", ".join(f"{n} {x} vs {y}" for n, x, y in bad))
else:
    print("every instance reached the same objective")
