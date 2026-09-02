#!/usr/bin/env python3
"""Speedup against thread count for the first-order method.

Reports the solver's own solve time, which excludes MPS parsing and presolve.
Those are serial and identical at every thread count, so including them would
measure Amdahl's law rather than the thing that changed - the same convention
docs/RESULTS.md already uses for the GPU numbers. End-to-end wall clock is
reported alongside so the dilution is visible rather than hidden.

Every configuration is run `--repeat` times and the minimum is kept. This box
is shared; a mean carries whatever else was running, and the minimum is the
closest thing to an uncontended run that a shared machine can report. Thread
counts are visited in a rotated order on each pass so that one burst of
interference cannot land entirely on one row.
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "build", "sankhya")


def load_average():
    return os.getloadavg()[0]


def run(path, threads, extra):
    cmd = [BIN, "solve", path, "--quiet", f"--threads={threads}"] + extra
    started = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    wall = time.perf_counter() - started
    # A non-zero exit means the solve did not reach optimality, which is what a
    # fixed iteration budget is *for* - every thread count then does exactly the
    # same work. So the output is parsed either way, and it is the parse that
    # decides whether the run counted.
    text = out.stdout
    solve = None
    # The last "time" line is the solve; the first is presolve.
    for line in text.splitlines():
        m = re.match(r"^time\s+([0-9.eE+-]+) s", line)
        if m:
            solve = float(m.group(1))
    obj = None
    m = re.search(r"^objective\s+(\S+)", text, re.M)
    if m:
        obj = m.group(1)
    iters = None
    m = re.search(r"^iterations\s+(\d+)", text, re.M)
    if m:
        iters = int(m.group(1))
    status = None
    m = re.search(r"^status\s+(.+)$", text, re.M)
    if m:
        status = m.group(1).strip()
    if solve is None:
        return None
    return {"solve": solve, "wall": wall, "objective": obj,
            "iterations": iters, "status": status}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("instances", nargs="+")
    p.add_argument("--threads", default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--extra", action="append", default=[])
    p.add_argument("--target-seconds", type=float, default=3.0,
                   help="calibrate each instance's iteration budget to about "
                        "this many seconds serially, so every instance "
                        "contributes a comparable amount of work")
    args = p.parse_args()

    counts = [int(t) for t in args.threads.split(",")]
    extra = list(args.extra)

    print(f"# binary   {BIN}")
    print(f"# extra    {' '.join(extra) if extra else '(none)'}")
    print(f"# repeats  {args.repeat}, minimum kept")
    print(f"# load average at start {load_average():.2f}")
    print()

    all_rows = {}
    for path in args.instances:
        name = os.path.basename(path).replace(".mps", "")

        # Calibrate: a short serial probe, then an iteration budget that makes
        # the serial run about --target-seconds. Without this an instance that
        # finishes 600 iterations in 0.19s contributes mostly process startup,
        # and one that takes 30s dominates the geometric mean.
        probe_iters = 200
        probe = run(path, 1, extra + [f"--max-iter={probe_iters}"])
        if probe is None or probe["solve"] <= 0:
            print(f"{name}: calibration failed, skipping\n")
            continue
        per_iter = probe["solve"] / probe["iterations"]
        budget = max(200, int(args.target_seconds / per_iter))
        run_extra = extra + [f"--max-iter={budget}"]
        print(f"# {name}: {budget} iterations "
              f"(~{budget * per_iter:.1f}s serial)")

        best = {t: None for t in counts}
        meta = {}
        for r in range(args.repeat):
            order = counts[r % len(counts):] + counts[: r % len(counts)]
            for t in order:
                got = run(path, t, run_extra)
                if got is None:
                    continue
                meta.setdefault(t, got)
                if best[t] is None or got["solve"] < best[t]["solve"]:
                    best[t] = got
        base = best.get(1)
        if base is None:
            print(f"{name}: serial run failed, skipping")
            continue

        # Every thread count must agree with the serial run, or the speedup is
        # measuring a different computation.
        disagreements = [
            t for t, v in best.items()
            if v is not None and (v["objective"] != base["objective"]
                                  or v["iterations"] != base["iterations"])
        ]

        print(f"## {name}   status {base['status']}, "
              f"{base['iterations']} iterations, objective {base['objective']}")
        if disagreements:
            print(f"   !! DIFFERENT ANSWER at threads {disagreements}")
        print(f"{'threads':>8} {'solve s':>10} {'speedup':>8} "
              f"{'wall s':>10} {'end-to-end':>11}")
        for t in counts:
            v = best[t]
            if v is None:
                print(f"{t:>8} {'failed':>10}")
                continue
            print(f"{t:>8} {v['solve']:>10.3f} {base['solve']/v['solve']:>8.2f} "
                  f"{v['wall']:>10.3f} {base['wall']/v['wall']:>11.2f}")
        print()
        all_rows[name] = (base, best)

    print(f"# load average at end {load_average():.2f}")

    # Geometric mean of the speedup across instances, per thread count.
    print("\n## geometric mean speedup over the set")
    print(f"{'threads':>8} {'solve':>8} {'end-to-end':>11}")
    for t in counts:
        s, w = [], []
        for base, best in all_rows.values():
            v = best.get(t)
            if v is None:
                continue
            s.append(base["solve"] / v["solve"])
            w.append(base["wall"] / v["wall"])
        if s:
            print(f"{t:>8} {statistics.geometric_mean(s):>8.2f} "
                  f"{statistics.geometric_mean(w):>11.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
