#!/usr/bin/env python3
"""Sweep the PID coefficients on the primal weight.

cuPDLPx does not publish its Kp/Ki/Kd, so they are ours to find. Kp = 0.5 with
the other two at zero reproduces the exponential smoothing that came before, so
that row is the baseline and anything below it has to beat it.
"""
import json, subprocess, sys, time

SET = ["afiro","sc50a","adlittle","blend","share1b","stocfor1","sctap1","scfxm1",
       "bandm","degen2","fit1p","25fv47","woodw","degen3","stocfor2","maros-r7"]

def run(name, flags):
    cmd = (["build/sankhya","solve",f"data/netlib/{name}.mps","--format=json",
            "--presolve","--tol=1e-8"] + flags)
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    w = time.perf_counter() - t
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return None
    return (d["iterations"] + d.get("polish_iterations", 0), w,
            d["status"] == "optimal")

def total(flags):
    it = wall = 0.0; bad = []
    for n in SET:
        r = run(n, flags)
        if r is None:
            bad.append(n + "(failed)"); continue
        it += r[0]; wall += r[1]
        if not r[2]: bad.append(n)
    return it, wall, bad

import itertools
mode = sys.argv[1] if len(sys.argv) > 1 else "1d"

grid = [("baseline Kp=.5 Kd=0", ["--kp=0.5", "--kd=0.0"])]
if mode == "2d":
    # The one-at-a-time sweep moved Kd around a fixed Kp = 0.5. These two are
    # not independent - Kp sets how hard the controller pushes and Kd how much
    # of that push is damped - so the minimum found on one axis need not be the
    # minimum of the surface.
    for kp, kd in itertools.product((0.3, 0.4, 0.5, 0.6, 0.7), (0.2, 0.3, 0.4)):
        grid.append((f"Kp={kp} Kd={kd}", [f"--kp={kp}", f"--kd={kd}"]))
else:
    for kd in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
        grid.append((f"Kd={kd}", ["--kp=0.5", f"--kd={kd}"]))
    for ki in (0.02, 0.05, 0.1, 0.2):
        grid.append((f"Ki={ki}", ["--kp=0.5", f"--ki={ki}"]))

base = None
print(f"{'config':<18} {'iterations':>11} {'wall':>7} {'vs base':>8}  not-optimal")
for label, flags in grid:
    it, wall, bad = total(flags)
    if base is None: base = it
    print(f"{label:<18} {int(it):>11} {wall:>7.2f} {it/base:>8.3f}  "
          f"{', '.join(bad) if bad else '-'}", flush=True)
