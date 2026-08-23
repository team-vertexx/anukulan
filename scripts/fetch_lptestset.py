#!/usr/bin/env python3
"""Fetch large sparse LPs from Mittelmann's lptestset.

These are the instances the GPU story rests on. The Netlib set tops out at a
few thousand rows, which is small enough that simplex wins on it no matter what
we do; these run from tens of thousands to millions of nonzeros, which is where
first-order methods start to pay.

Files are bzip2-compressed MPS. Reference runtimes for many of them are in
Mittelmann's LPfeas benchmark at plato.asu.edu/ftp/lpfeas.html.
"""

import argparse
import bz2
import pathlib
import sys
import urllib.request

BASE = "https://plato.asu.edu/ftp/lptestset/"

# A size ladder. The point is to find where our solver stops coping, so they are
# listed smallest first and pulled in that order.
LADDER = [
    ("qap15", "qap15.mps.bz2"),
    ("supportcase10", "supportcase10.mps.bz2"),
    ("datt256_lp", "datt256_lp.mps.bz2"),
    ("graph40-40", "graph40-40.mps.bz2"),
    ("s100", "s100.mps.bz2"),
    ("savsched1", "savsched1.mps.bz2"),
    ("woodlands09", "woodlands09.mps.bz2"),
    ("ex10", "ex10.mps.bz2"),
    ("rmine15", "rmine15.mps.bz2"),
    ("s250r10", "s250r10.mps.bz2"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/lptestset")
    parser.add_argument("--only", default="", help="comma separated names")
    parser.add_argument("--max", type=int, default=4, help="how many from the ladder")
    args = parser.parse_args()

    out = pathlib.Path(args.root)
    out.mkdir(parents=True, exist_ok=True)

    wanted = {n.strip() for n in args.only.split(",") if n.strip()}
    todo = [(n, f) for n, f in LADDER if not wanted or n in wanted]
    if not wanted:
        todo = todo[: args.max]

    for name, remote in todo:
        target = out / f"{name}.mps"
        if target.exists() and target.stat().st_size > 0:
            print(f"  have {name} ({target.stat().st_size/1e6:.1f} MB)")
            continue
        print(f"  fetching {name} ...", end="", flush=True)
        try:
            with urllib.request.urlopen(BASE + remote, timeout=300) as response:
                packed = response.read()
            target.write_bytes(bz2.decompress(packed))
            print(f" {target.stat().st_size/1e6:.1f} MB")
        except Exception as exc:  # noqa: BLE001
            print(f" FAILED: {exc}", file=sys.stderr)
            target.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
