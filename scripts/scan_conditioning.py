#!/usr/bin/env python3
"""Find badly-scaled instances by measuring, not by guessing from the name.

Every large instance pulled so far turned out to be a 0/+-1 combinatorial LP -
coefficient ratio exactly 1, perfectly equilibrated before we touch it. Those say
nothing about how the solver behaves on the kind of model a refinery actually
runs, where cents per litre sit in the same matrix as millions of barrels.

This downloads a batch and reports the coefficient dynamic range of each, so the
badly-scaled ones can be picked on evidence.
"""

import argparse
import bz2
import os
import json
import pathlib
import subprocess
import sys
import urllib.request

BASE = "https://plato.asu.edu/ftp/lptestset/"

CANDIDATES = [
    # name, remote path, why it is worth looking at
    ("irish-electricity", "irish-electricity.mps.bz2", "energy system, mixed units"),
    ("cont1", "misc/cont1.bz2", "PDE constrained, known hard for first-order"),
    ("cont11", "misc/cont11.bz2", "PDE constrained, harder still"),
    ("watson_1", "misc/watson_1.bz2", "financial planning, mixed units"),
    ("sgpf5y6", "misc/sgpf5y6.bz2", "stochastic programming"),
    ("stormG2_1000", "misc/stormG2_1000.bz2", "stochastic, large"),
    ("woodlands09", "woodlands09.mps.bz2", "forestry planning"),
    ("rmine15", "rmine15.mps.bz2", "mine scheduling"),
    ("brazil3", "brazil3.mps.bz2", "small, mixed"),
    ("bdry2", "bdry2.bz2", "boundary value problem"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/lptestset")
    parser.add_argument("--binary", default="build/sankhya")
    parser.add_argument("--only", default="")
    parser.add_argument("--max-mb", type=float, default=60.0,
                        help="skip anything whose expanded file exceeds this")
    args = parser.parse_args()

    out = pathlib.Path(args.root)
    out.mkdir(parents=True, exist_ok=True)
    wanted = {n.strip() for n in args.only.split(",") if n.strip()}

    results = []
    for name, remote, why in CANDIDATES:
        if wanted and name not in wanted:
            continue
        target = out / f"{name}.mps"
        if not target.exists() or target.stat().st_size == 0:
            print(f"fetching {name} ({why}) ...", end="", flush=True)
            try:
                with urllib.request.urlopen(BASE + remote, timeout=600) as response:
                    data = bz2.decompress(response.read())
            except Exception as exc:  # noqa: BLE001
                print(f" FAILED: {exc}", file=sys.stderr)
                continue
            # Files whose remote name lacks ".mps" are stored in the packed
            # format that netlib's emps expands - the lptestset README says so,
            # and the giveaway is a binary-looking line 5. Same expander we
            # already build for the Netlib set.
            if ".mps" not in remote:
                emps = pathlib.Path("data/_work/emps")
                if not emps.exists():
                    print(" no emps binary; run scripts/fetch_netlib.py first",
                          file=sys.stderr)
                    continue
                packed = out / f"{name}.packed"
                packed.write_bytes(data)
                with target.open("wb") as handle:
                    subprocess.run([str(emps), str(packed)], stdout=handle, check=True)
                packed.unlink(missing_ok=True)
                if target.stat().st_size > args.max_mb * 1e6:
                    print(f" skipped, {target.stat().st_size/1e6:.0f} MB expanded")
                    target.unlink(missing_ok=True)
                    continue
                print(f" {target.stat().st_size/1e6:.1f} MB (via emps)")
            else:
                if len(data) > args.max_mb * 1e6:
                    print(f" skipped, {len(data)/1e6:.0f} MB expanded")
                    continue
                    print(f" {target.stat().st_size/1e6:.1f} MB")

        proc = subprocess.run(
            [args.binary, "read", str(target), "--format=json", "--quiet"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  {name}: unreadable - {proc.stderr.strip()[:120]}", file=sys.stderr)
            continue
        d = json.loads(proc.stdout)
        lo, hi = d["min_abs_coeff"], d["max_abs_coeff"]
        results.append((hi / lo if lo else 0.0, name, d, why))

    results.sort(reverse=True)
    print(f"\n{'instance':<20} {'rows':>9} {'cols':>9} {'nnz':>11} "
          f"{'|a| min':>10} {'|a| max':>10} {'ratio':>10}")
    for ratio, name, d, _ in results:
        print(f"{name:<20} {d['rows']:>9} {d['cols']:>9} {d['nnz']:>11} "
              f"{d['min_abs_coeff']:>10.3g} {d['max_abs_coeff']:>10.3g} {ratio:>10.3g}")
    print("\nRatio is the coefficient dynamic range. Anything near 1 is a 0/+-1 "
          "matrix and tells us nothing about scaling robustness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
