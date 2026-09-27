#!/usr/bin/env python3
"""Generate a family of refinery plans: the same planning model re-solved day
after day while crude prices, product prices and demands move.

This is the workload plant memory is for. A refinery's planning model keeps its
structure from one day to the next; what changes is the data. So every member
written here has exactly the structure of scripts/refinery_model.py, and only
the numbers move:

    crude prices     a random walk of about 1% a day, held within +-8%
    product prices   a random walk of about 0.7% a day, held within +-6%
    product demand   each period's sales ceiling redrawn within +-4% a day

The walk is seeded, so the family is the same on every machine.

    python3 scripts/refinery_family.py --days 30
    build/anukulan family data/refinery/family/day_*.mps
"""

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import refinery_model as rm  # noqa: E402


def member_of(column, prefix):
    """BUY_ARAB_LIGHT_7 -> ARAB_LIGHT: the name between the prefix and the
    period, which may itself contain underscores."""
    return column[len(prefix):].rsplit("_", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--periods", type=int, default=12)
    ap.add_argument("--crudes", type=int, default=8)
    ap.add_argument("--capacity", type=float, default=1.5e6,
                    help="CDU throughput per period, in barrels")
    ap.add_argument("--seed", type=int, default=26119)
    ap.add_argument("--out", default="data/refinery/family")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    crude_level = {c[0]: 1.0 for c in rm.CRUDES[:args.crudes]}
    product_level = {p[0]: 1.0 for p in rm.PRODUCTS}

    for day in range(1, args.days + 1):
        if day > 1:
            for name in crude_level:
                step = 1.0 + rng.gauss(0.0, 0.010)
                crude_level[name] = min(1.08, max(0.92, crude_level[name] * step))
            for name in product_level:
                step = 1.0 + rng.gauss(0.0, 0.007)
                product_level[name] = min(1.06, max(0.94, product_level[name] * step))

        m = rm.build(args.periods, args.crudes, args.capacity, 1.0, False)

        # Prices: what a crude costs to buy and to hold, and what a product sells for.
        for column, coefficient in list(m.obj.items()):
            if column.startswith("BUY_"):
                m.obj[column] = coefficient * crude_level[member_of(column, "BUY_")]
            elif column.startswith("INVC_"):
                m.obj[column] = coefficient * crude_level[member_of(column, "INVC_")]
            elif column.startswith("SELL_"):
                m.obj[column] = coefficient * product_level[member_of(column, "SELL_")]

        # Demand: each period's sales ceiling moves a few percent.
        moved = []
        for kind, column, value in m.bounds:
            if kind == "UP" and column.startswith("SELL_") and value is not None:
                value = value * (1.0 + rng.uniform(-0.04, 0.04))
            moved.append((kind, column, value))
        m.bounds = moved

        path = out / f"day_{day:02d}.mps"
        with path.open("w") as handle:
            m.write(handle)

    print(f"wrote {args.days} days to {out}/  "
          f"(periods {args.periods}, crudes {args.crudes}, seed {args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
