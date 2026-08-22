#!/usr/bin/env python3
"""Measure what each piece of the solver actually contributes.

The solver is four separable pieces of machinery on top of plain PDHG:
preconditioning, the adaptive step size, restarts, and the primal weight. Each
is claimed in the literature to matter. This turns each one off in turn and
records the cost, so the claim is a measurement rather than a citation.
"""

import argparse
import csv
import json
import pathlib
import subprocess
import sys

CONFIGS = [
    ("everything", []),
    ("no restarts", ["--no-restarts"]),
    ("no adaptive step", ["--no-adaptive"]),
    ("no primal weight", ["--no-primal-weight"]),
    ("no scaling", ["--no-scaling"]),
    ("plain PDHG", ["--no-restarts", "--no-adaptive", "--no-primal-weight",
                    "--no-scaling"]),
]

DEFAULT_SET = ["afiro", "sc50a", "blend", "stocfor1", "sctap1", "degen2",
               "scfxm1", "bandm", "share1b", "25fv47"]


def run(binary, path, flags, tolerance, max_iter, time_limit):
    proc = subprocess.run(
        [binary, "solve", str(path), f"--tol={tolerance}", f"--max-iter={max_iter}",
         f"--time-limit={time_limit}", "--format=json", "--quiet"] + flags,
        capture_output=True, text=True,
    )
    if not proc.stdout.strip():
        return None
    return json.loads(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build/sankhya")
    parser.add_argument("--instances", default="data/netlib")
    parser.add_argument("--out", default="bench/results/ablation")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=2000000)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--set", default=",".join(DEFAULT_SET))
    args = parser.parse_args()

    inst_dir = pathlib.Path(args.instances)
    names = [n.strip() for n in args.set.split(",") if n.strip()]

    table = {}
    for name in names:
        path = inst_dir / f"{name}.mps"
        if not path.exists():
            print(f"skip {name}", file=sys.stderr)
            continue
        table[name] = {}
        print(f"{name}", end="", flush=True)
        for label, flags in CONFIGS:
            r = run(args.binary, path, flags, args.tolerance, args.max_iter,
                    args.time_limit)
            table[name][label] = r
            shown = "fail" if r is None else (
                r["iterations"] if r["status"] == "optimal" else f"{r['status'][:4]}")
            print(f"  {label}={shown}", end="", flush=True)
        print()

    out_dir = pathlib.Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pathlib.Path(str(args.out) + ".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance", "configuration", "status", "iterations",
                         "seconds", "objective"])
        for name, configs in table.items():
            for label, r in configs.items():
                if r is None:
                    writer.writerow([name, label, "failed", "", "", ""])
                else:
                    writer.writerow([name, label, r["status"], r["iterations"],
                                     f"{r['seconds']:.4f}", r["objective"]])

    labels = [label for label, _ in CONFIGS]
    md = ["### What each piece of the solver is worth", "",
          "Iterations to a relative tolerance of "
          f"{args.tolerance:g}. `limit` means the run hit "
          f"{args.max_iter:,} iterations or {args.time_limit:g}s without converging.",
          "",
          "| instance | " + " | ".join(labels) + " |",
          "|---" * (len(labels) + 1) + "|"]
    for name, configs in table.items():
        cells = []
        for label in labels:
            r = configs.get(label)
            if r is None:
                cells.append("fail")
            elif r["status"] != "optimal":
                cells.append("limit")
            else:
                cells.append(f"{r['iterations']:,}")
        md.append(f"| {name} | " + " | ".join(cells) + " |")

    # Cost of removing each piece, over the instances where both configurations
    # actually converged - a ratio against a run that never finished is not a
    # ratio, and averaging it in would flatter whichever piece failed hardest.
    md += ["", "Cost of removing each piece, geometric mean over the instances "
               "where both that configuration and the full solver converged:", ""]
    for label in labels[1:]:
        ratios = []
        for _, configs in table.items():
            base = configs.get("everything")
            other = configs.get(label)
            if (base and other and base["status"] == "optimal"
                    and other["status"] == "optimal" and base["iterations"] > 0):
                ratios.append(other["iterations"] / base["iterations"])
        solved = sum(
            1 for _, c in table.items()
            if c.get(label) and c[label]["status"] == "optimal"
        )
        if ratios:
            geo = 1.0
            for value in ratios:
                geo *= value
            geo **= 1.0 / len(ratios)
            md.append(f"- **{label}**: {geo:.2f}x more iterations "
                      f"(over {len(ratios)} comparable instances); "
                      f"solved {solved}/{len(table)}")
        else:
            md.append(f"- **{label}**: never converged on any instance "
                      f"(solved {solved}/{len(table)})")

    md_path = pathlib.Path(str(args.out) + ".md")
    md_path.write_text("\n".join(md) + "\n")
    print(f"\nwrote {csv_path}\nwrote {md_path}")
    print()
    print("\n".join(md[-len(labels):]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
