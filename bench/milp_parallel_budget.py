#!/usr/bin/env python3
"""How much of a MILP solve could a thread possibly help with?

Before writing a parallel tree it is worth knowing what share of the work is in
the parts that are actually independent. Three are measured here, all from
counters the solver already reports, so nothing had to be instrumented:

  strong branching  up to eight candidates a node, two bounded LP solves each,
                    and every one of them independent of the others. Capped at
                    depth ten, so the question is how many nodes are shallow
                    enough to pay it.
  root cuts         separated once, per row, each row independent.
  node relaxations  the tree itself. Everything left over.

The share is reported as an upper bound: a strong branch probe runs under an
iteration cap that a real node solve does not, so counting it as a full
relaxation overstates it. An upper bound is what the decision needs - if the
most it could be is small, the answer is no.
"""

import glob
import json
import os
import subprocess
import sys

BIN = os.environ.get("SANKHYA", "build/sankhya")


def solve(path, limit, extra):
    cmd = [BIN, "milp", path, "--format=json", f"--time-limit={limit}"] + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def main():
    limit = sys.argv[1] if len(sys.argv) > 1 else "20"
    names = sys.argv[2:]
    if names:
        files = [f"data/miplib/{n}.mps" for n in names]
    else:
        files = sorted(glob.glob("data/miplib/*.mps"))[:40]

    print(f"# time limit {limit}s per instance")
    print(f"{'instance':<22} {'nodes':>8} {'relax':>8} {'probes':>8} "
          f"{'probe/relax':>12} {'cuts':>6} {'secs':>7}")
    tot_probe = tot_relax = 0
    for f in files:
        name = os.path.basename(f)[:-4]
        d = solve(f, limit, [])
        if d is None:
            print(f"{name[:21]:<22} {'failed':>8}")
            continue
        relax = d.get("relaxations", 0)
        probes = d.get("strong_branch_probes", 0)
        share = probes / relax if relax else 0.0
        tot_probe += probes
        tot_relax += relax
        print(f"{name[:21]:<22} {d.get('nodes',0):>8} {relax:>8} {probes:>8} "
              f"{share:>11.1%} {d.get('cuts_added',0):>6} "
              f"{d.get('seconds',0):>7.1f}")
    if tot_relax:
        print(f"\nprobes are at most {tot_probe/(tot_relax):.1%} of the "
              f"relaxations solved, over {len(files)} instances")
        print(f"({tot_probe} probes, {tot_relax} relaxations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
