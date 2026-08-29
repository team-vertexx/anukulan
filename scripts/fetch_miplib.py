#!/usr/bin/env python3
"""Fetch MIPLIB 2017 instances with known optima.

The repository ships with seven MIPLIB instances, and seven is not enough to
fit a threshold on. Several defaults in the branch-and-bound - the cut
usefulness cutoff, the propagation round count, the strong branching depth -
were chosen against those seven, and every commit that set one says so. This
widens the set so those choices can be rechecked against something harder to
overfit.

Instances come from miplib.zib.de. The .solu file lists the optimal objective
for each instance that has one, which is what makes them usable as a
correctness check rather than only a timing set.

Only small instances are kept by default: this is a measurement set that has to
run in a loop many times, not a stress test. --max-bytes controls that.
"""

import argparse
import gzip
import os
import pathlib
import shutil
import sys
import urllib.request

SOLU = "https://miplib.zib.de/downloads/miplib2017-v29.solu"
INSTANCE = "https://miplib.zib.de/WebData/instances/{}.mps.gz"


def known_optima(url):
    """Instance name -> optimal objective, for the ones proved optimal."""
    out = {}
    with urllib.request.urlopen(url, timeout=60) as fh:
        for line in fh.read().decode("utf-8", "replace").splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "=opt=":
                try:
                    out[parts[1]] = float(parts[2])
                except ValueError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/miplib")
    ap.add_argument("--reference", default="data/reference/miplib.csv")
    ap.add_argument("--max-bytes", type=int, default=120_000,
                    help="skip instances whose compressed file is larger")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    optima = known_optima(SOLU)
    print(f"{len(optima)} instances have a proved optimum", flush=True)

    # Stride through the sorted list rather than taking a prefix: the first N
    # alphabetically is a biased sample, and several instance families share a
    # name prefix, so a prefix would be a handful of families rather than a
    # spread.
    names = sorted(optima)
    stride = max(1, len(names) // max(1, args.limit * 3))
    names = names[::stride]

    kept = {}
    skipped = 0
    for name in names:
        if len(kept) >= args.limit:
            break
        target = out / f"{name}.mps"
        if target.exists():
            kept[name] = optima[name]
            continue
        try:
            req = urllib.request.Request(INSTANCE.format(name))
            with urllib.request.urlopen(req, timeout=90) as fh:
                blob = fh.read(args.max_bytes + 1)
                if len(blob) > args.max_bytes:
                    skipped += 1
                    continue
                # A short read means we have the whole file.
                text = gzip.decompress(blob)
        except Exception:
            skipped += 1
            continue
        target.write_bytes(text)
        kept[name] = optima[name]
        print(f"  {name:<28} {len(blob):>7} bytes gz", flush=True)

    ref = pathlib.Path(args.reference)
    ref.parent.mkdir(parents=True, exist_ok=True)
    with ref.open("w") as fh:
        fh.write("name,optimal\n")
        for n in sorted(kept):
            fh.write(f"{n},{kept[n]!r}\n")
    print(f"\nkept {len(kept)} instances, skipped {skipped}")
    print(f"optima written to {ref}")


main()
