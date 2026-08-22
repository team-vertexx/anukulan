#!/usr/bin/env python3
"""Benchmark our solver against HiGHS on a set of instances.

Three baselines are run, not one:

  highs-simplex  HiGHS's default dual simplex - the established solver the
                 problem statement asks to be compared against
  highs-pdlp     HiGHS running the same first-order algorithm we implement, so
                 that a difference in iteration count is a difference in our
                 implementation and nothing else
  sankhya        ours

Every run is scored on achieved accuracy against the published optimum, not just
on the tolerance it was asked for. Two solvers that both claim 1e-6 can land at
quite different distances from the true objective, and reporting only iteration
counts would hide that.
"""

import argparse
import csv
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time

HIGHS_ITERS = {
    "simplex": re.compile(r"Simplex\s+iterations:\s*(\d+)"),
    "pdlp": re.compile(r"PDLP\s+iterations:\s*(\d+)"),
    "ipm": re.compile(r"IPM\s+iterations:\s*(\d+)"),
}
HIGHS_OBJ = re.compile(r"Objective value\s*:\s*(\S+)")
HIGHS_STATUS = re.compile(r"Model status\s*:\s*(.+)")


def run_highs(binary, path, solver, tolerance, time_limit):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write(f"solver = {solver}\n")
        handle.write(f"time_limit = {time_limit}\n")
        if solver == "pdlp":
            handle.write(f"kkt_tolerance = {tolerance}\n")
        options = handle.name

    started = time.perf_counter()
    proc = subprocess.run(
        [binary, f"--options_file={options}", str(path)],
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - started
    pathlib.Path(options).unlink(missing_ok=True)

    out = proc.stdout
    status = HIGHS_STATUS.search(out)
    obj = HIGHS_OBJ.search(out)
    iters = None
    for pattern in HIGHS_ITERS.values():
        found = pattern.search(out)
        if found:
            iters = int(found.group(1))
            break
    return {
        "status": status.group(1).strip().lower() if status else "unknown",
        "objective": float(obj.group(1)) if obj else None,
        "iterations": iters,
        "seconds": wall,
    }


def run_sankhya(binary, path, tolerance, time_limit, max_iter):
    started = time.perf_counter()
    proc = subprocess.run(
        [
            binary, "solve", str(path),
            f"--tol={tolerance}",
            f"--time-limit={time_limit}",
            f"--max-iter={max_iter}",
            "--format=json", "--quiet",
        ],
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - started
    if not proc.stdout.strip():
        return {"status": "failed", "objective": None, "iterations": None,
                "seconds": wall}
    d = json.loads(proc.stdout)
    return {
        "status": d["status"],
        "objective": d["objective"],
        "iterations": d["iterations"],
        "restarts": d["restarts"],
        "seconds": wall,
        "solve_seconds": d["seconds"],
        "rel_gap": d["rel_gap"],
    }


def relative_error(objective, published):
    if objective is None or published is None:
        return None
    return abs(objective - published) / max(1.0, abs(published))


DEFAULT_SET = [
    "afiro", "sc50a", "adlittle", "blend", "share1b", "stocfor1", "sctap1",
    "scfxm1", "bandm", "degen2", "fit1p", "25fv47", "woodw", "degen3",
    "stocfor2", "greenbea", "pilot87", "maros-r7",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build/sankhya")
    parser.add_argument("--highs", default="highs")
    parser.add_argument("--instances", default="data/netlib")
    parser.add_argument("--reference", default="data/reference/netlib.csv")
    parser.add_argument("--out", default="bench/results")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--max-iter", type=int, default=1000000)
    parser.add_argument("--set", default=",".join(DEFAULT_SET))
    parser.add_argument(
        "--solvers", default="sankhya,highs-pdlp,highs-simplex",
    )
    args = parser.parse_args()

    reference = {
        row["name"]: float(row["optimal"])
        for row in csv.DictReader(open(args.reference, newline=""))
    }
    inst_dir = pathlib.Path(args.instances)
    names = [n.strip() for n in args.set.split(",") if n.strip()]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]

    rows = []
    for name in names:
        path = inst_dir / f"{name}.mps"
        if not path.exists():
            print(f"skip {name}: not downloaded", file=sys.stderr)
            continue
        published = reference.get(name)
        print(f"{name} ...", end="", flush=True)

        record = {"instance": name, "published": published}
        for solver in solvers:
            if solver == "sankhya":
                r = run_sankhya(args.binary, path, args.tolerance, args.time_limit,
                                args.max_iter)
            elif solver == "highs-pdlp":
                r = run_highs(args.highs, path, "pdlp", args.tolerance,
                              args.time_limit)
            elif solver == "highs-simplex":
                r = run_highs(args.highs, path, "simplex", args.tolerance,
                              args.time_limit)
            else:
                raise SystemExit(f"unknown solver {solver}")
            prefix = solver.replace("-", "_")
            record[f"{prefix}_status"] = r["status"]
            record[f"{prefix}_iterations"] = r["iterations"]
            record[f"{prefix}_seconds"] = round(r["seconds"], 4)
            record[f"{prefix}_objective"] = r["objective"]
            record[f"{prefix}_rel_error"] = relative_error(r["objective"], published)
            print(f" {solver}={r['iterations']}", end="", flush=True)
        rows.append(record)
        print()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"netlib_tol{args.tolerance:g}"
    csv_path = out_dir / f"{tag}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Markdown table, ready to paste into the report.
    md = [
        f"### Netlib LP, relative tolerance {args.tolerance:g}",
        "",
        "Accuracy is the relative distance from the objective published in the "
        "Netlib index, not the tolerance the solver was asked for.",
        "",
        "| instance | rows x cols | sankhya iters | HiGHS-PDLP iters | ratio | "
        "sankhya acc | HiGHS-PDLP acc | HiGHS simplex s | sankhya s |",
        "|---|---|---:|---:|---:|---|---|---:|---:|",
    ]
    ratios = []
    for r in rows:
        ours = r.get("sankhya_iterations")
        theirs = r.get("highs_pdlp_iterations")
        ratio = ""
        if ours and theirs:
            value = ours / theirs
            ratios.append(value)
            ratio = f"{value:.2f}x"
        def fmt(x, spec="{:.2e}"):
            return spec.format(x) if isinstance(x, float) else "-"
        md.append(
            f"| {r['instance']} | | {ours or '-'} | {theirs or '-'} | {ratio} | "
            f"{fmt(r.get('sankhya_rel_error'))} | {fmt(r.get('highs_pdlp_rel_error'))} | "
            f"{fmt(r.get('highs_simplex_seconds'), '{:.3f}')} | "
            f"{fmt(r.get('sankhya_seconds'), '{:.3f}')} |"
        )
    if ratios:
        geo = 1.0
        for value in ratios:
            geo *= value
        geo **= 1.0 / len(ratios)
        md += ["", f"Geometric mean iteration ratio against HiGHS-PDLP: **{geo:.2f}x** "
                   f"over {len(ratios)} instances."]
    md_path = out_dir / f"{tag}.md"
    md_path.write_text("\n".join(md) + "\n")

    print(f"\nwrote {csv_path}\nwrote {md_path}")
    if ratios:
        print(f"geometric mean iteration ratio vs HiGHS-PDLP: {geo:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
