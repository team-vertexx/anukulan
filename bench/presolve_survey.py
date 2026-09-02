#!/usr/bin/env python3
"""What presolve removes, per instance set.

Reduction counts are deterministic, so unlike every timing harness here this one
can be run on a loaded machine without lying. It reports rows, columns and
nonzeros removed, and which reduction did the removing, so a change can be
attributed rather than just observed.

Totals are over the whole set - sum of removed over sum of original - not a mean
of per-instance percentages, because the second lets a 27-row instance count as
much as a 15,000-row one.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REDUCTIONS = ["empty_rows", "singleton_rows", "redundant_rows", "forcing_rows",
              "duplicate_rows", "fixed_columns", "empty_columns",
              "free_column_singletons", "slack_column_singletons",
              "inequality_column_singletons", "slack_singletons_declined",
              "doubleton_equations", "dual_fixed_columns",
              "coefficients_tightened"]


def run(binary, model, extra):
    cmd = [binary, "presolve", model, "--quiet", "--format=json"] + extra
    done = subprocess.run(cmd, capture_output=True, text=True)
    for line in reversed(done.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.path.join(ROOT, "build-work", "sankhya"))
    ap.add_argument("--set", default="netlib",
                    help="netlib, miplib, refinery, or a directory of .mps files")
    ap.add_argument("--extra", default="", help="flags passed to presolve")
    ap.add_argument("--json", default=None, help="write per-instance records here")
    ap.add_argument("--quiet", action="store_true", help="totals only")
    args = ap.parse_args()

    if args.set == "refinery":
        models = [os.path.join(ROOT, "data", "refinery", "refinery.mps")]
    else:
        directory = args.set if os.path.isdir(args.set) \
            else os.path.join(ROOT, "data", args.set)
        models = sorted(os.path.join(directory, f)
                        for f in os.listdir(directory) if f.endswith(".mps"))

    extra = args.extra.split() if args.extra else []
    totals = dict.fromkeys(
        ["rows_before", "rows_after", "cols_before", "cols_after",
         "nnz_before", "nnz_after"], 0)
    fired = dict.fromkeys(REDUCTIONS, 0)
    records, failures = [], []

    if not args.quiet:
        print(f"{'instance':28s} {'rows':>16s} {'cols':>16s} {'nnz':>18s} {'s':>7s}")
    for model in models:
        name = os.path.basename(model)[:-4]
        r = run(args.binary, model, extra)
        if r is None:
            failures.append((name, "no output"))
            continue
        if r["status"] != "reduced":
            failures.append((name, r["status"]))
            continue
        for k in totals:
            totals[k] += r[k]
        for k in REDUCTIONS:
            fired[k] += r.get(k, 0)
        records.append(dict(instance=name, **{k: r[k] for k in r if k != "name"}))
        if not args.quiet:
            dr = 100.0 * (r["rows_before"] - r["rows_after"]) / max(1, r["rows_before"])
            dc = 100.0 * (r["cols_before"] - r["cols_after"]) / max(1, r["cols_before"])
            dn = 100.0 * (r["nnz_before"] - r["nnz_after"]) / max(1, r["nnz_before"])
            print(f"{name:28s} {r['rows_before']:7d}->{r['rows_after']:<6d}{dr:5.1f}% "
                  f"{r['cols_before']:7d}->{r['cols_after']:<6d}{dc:5.1f}% "
                  f"{r['nnz_before']:8d}->{r['nnz_after']:<7d}{dn:5.1f}% "
                  f"{r['seconds']:7.3f}")

    def pct(before, after):
        return 100.0 * (totals[before] - totals[after]) / max(1, totals[before])

    print(f"\n{len(records)} instances"
          + (f", {len(failures)} not reduced" if failures else ""))
    print(f"rows      {totals['rows_before']:8d} -> {totals['rows_after']:<8d} "
          f"{pct('rows_before', 'rows_after'):5.2f}% removed")
    print(f"columns   {totals['cols_before']:8d} -> {totals['cols_after']:<8d} "
          f"{pct('cols_before', 'cols_after'):5.2f}% removed")
    print(f"nonzeros  {totals['nnz_before']:8d} -> {totals['nnz_after']:<8d} "
          f"{pct('nnz_before', 'nnz_after'):5.2f}% removed")
    print("\nfired:")
    for k in REDUCTIONS:
        if fired[k]:
            print(f"  {k:24s} {fired[k]:8d}")
    for k in REDUCTIONS:
        if not fired[k]:
            print(f"  {k:24s} {0:8d}")
    if failures:
        print("\nnot reduced:")
        for name, why in failures:
            print(f"  {name:28s} {why}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dict(totals=totals, fired=fired, records=records,
                           failures=failures), fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
