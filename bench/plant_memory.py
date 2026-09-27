#!/usr/bin/env python3
"""Plant memory, measured: a refinery plan re-solved day after day, each day
solved cold and from the basis the previous day ended on.

Generates the family with scripts/refinery_family.py, runs `anukulan family`
on it, and writes one line per family to bench/results/plant_memory.txt.

    python3 -u bench/plant_memory.py                 # 12, 24 and 52 periods
    python3 -u bench/plant_memory.py --periods 12 --days 30
"""

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def run_family(binary, periods, days, seed):
    out = ROOT / "data" / "refinery" / "family" / f"p{periods}"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "refinery_family.py"),
                    "--periods", str(periods), "--days", str(days),
                    "--seed", str(seed), "--out", str(out)], check=True)
    files = sorted(str(p) for p in out.glob("day_*.mps"))
    proc = subprocess.run([str(binary), "family", *files, "--format=json"],
                          capture_output=True, text=True)
    lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    summary = next(line for line in lines if line.get("summary"))
    first = next(line for line in lines if not line.get("summary"))
    return first, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "build" / "anukulan"))
    ap.add_argument("--periods", type=int, nargs="*", default=[12, 24, 52])
    ap.add_argument("--days", type=int, nargs="*", default=None,
                    help="days per family; default 30, 10 and 6 for 12, 24 and 52 periods")
    ap.add_argument("--seed", type=int, default=26119)
    args = ap.parse_args()

    default_days = {12: 30, 24: 10, 52: 6}
    days = args.days or [default_days.get(p, 10) for p in args.periods]
    rows = []
    for periods, n in zip(args.periods, days):
        first, s = run_family(args.binary, periods, n, args.seed)
        row = (f"periods {periods:>3}  rows {first['std_rows']:>6}  re-solves {s['members_compared']:>3}  "
               f"pivots {s['cold_pivots']:>7} -> {s['warm_pivots']:>6} ({s['pivot_ratio']:.3f}x)  "
               f"seconds {s['cold_seconds']:8.2f} -> {s['warm_seconds']:7.2f} ({s['time_ratio']:.3f}x)  "
               f"answers that differ {s['disagreements']}  warm failures {s['warm_failures']}")
        print(row, flush=True)
        rows.append(row)

    results = ROOT / "bench" / "results" / "plant_memory.txt"
    results.write_text("\n".join(rows) + "\n")
    print(f"wrote {results.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
