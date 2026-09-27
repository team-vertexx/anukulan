#!/usr/bin/env python3
"""A case stack, measured: the base refinery plan and sixty what-if cases
around it, each case solved cold and from the base's basis.

Generates the stack with scripts/refinery_cases.py, runs
`anukulan family --from-first` on it, and writes one line per plan size to
bench/results/case_stack.txt.

    python3 -u bench/case_stack.py                 # 12 and 24 periods
    python3 -u bench/case_stack.py --periods 12
"""

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def run_stack(binary, periods):
    out = ROOT / "data" / "refinery" / "cases" / f"p{periods}"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "refinery_cases.py"),
                    "--periods", str(periods), "--out", str(out)], check=True)
    files = sorted(str(p) for p in out.glob("case_*.mps"))
    proc = subprocess.run([str(binary), "family", "--from-first", *files, "--format=json"],
                          capture_output=True, text=True)
    lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    summary = next(line for line in lines if line.get("summary"))
    members = [line for line in lines if not line.get("summary")]
    return members, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "build" / "anukulan"))
    ap.add_argument("--periods", type=int, nargs="*", default=[12, 24])
    args = ap.parse_args()

    rows = []
    for periods in args.periods:
        members, s = run_stack(args.binary, periods)
        base = members[0]
        row = (f"periods {periods:>3}  rows {base['std_rows']:>6}  cases {s['members_compared']:>3}  "
               f"pivots {s['cold_pivots']:>7} -> {s['warm_pivots']:>6} ({s['pivot_ratio']:.3f}x)  "
               f"seconds {s['cold_seconds']:8.2f} -> {s['warm_seconds']:7.2f} ({s['time_ratio']:.3f}x)  "
               f"answers that differ {s['disagreements']}  warm failures {s['warm_failures']}")
        print(row, flush=True)
        rows.append(row)

    results = ROOT / "bench" / "results" / "case_stack.txt"
    results.write_text("\n".join(rows) + "\n")
    print(f"wrote {results.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
