#!/usr/bin/env python3
"""Generate a case stack of one refinery plan: the base plan, and the what-if
cases a planner weighs around it.

A refinery does not decide on one solve. Before buying a cargo or taking a unit
down, the planner solves the base plan and then a stack of cases around it:
what if this crude were cheaper, what if diesel demand fell, what if the FCC
lost half its capacity for a period. Every case keeps the base plan's structure
and differs from it in one piece of data. So every case here is built by
scripts/refinery_model.py exactly as the base is, and then changed in one way:

    crude price      one crude's cost moved by -6%, -3%, +3% or +6%     32 cases
    crude supply     one crude's purchase ceiling halved                  8 cases
    product demand   one product's sales ceiling moved by -10% or +10%    8 cases
    unit outage      the CDU at 80%, or the FCC or the hydrotreater at
                     50%, in one of the first four periods              12 cases

Sixty cases and the base. No case can be infeasible: each one only moves a
price or tightens a ceiling, and buying nothing is always a feasible plan.

The base is written first, so `anukulan family --from-first` starts every case
from the base's basis:

    python3 scripts/refinery_cases.py
    build/anukulan family --from-first data/refinery/cases/case_*.mps
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import refinery_model as rm  # noqa: E402


def cases(crudes):
    """(name, kind, subject, factor, period) for every case, base first."""
    out = [("base", "base", None, 1.0, None)]
    for crude in [c[0] for c in rm.CRUDES[:crudes]]:
        for pct in (-6, -3, 3, 6):
            out.append((f"price_{crude}_{pct:+d}", "price", crude, 1.0 + pct / 100.0, None))
    for crude in [c[0] for c in rm.CRUDES[:crudes]]:
        out.append((f"supply_{crude}_half", "supply", crude, 0.5, None))
    for product in [p[0] for p in rm.PRODUCTS]:
        for pct in (-10, 10):
            out.append((f"demand_{product}_{pct:+d}", "demand", product, 1.0 + pct / 100.0, None))
    for unit, factor in (("CDU", 0.8), ("FCC", 0.5), ("HDT", 0.5)):
        for period in (1, 2, 3, 4):
            out.append((f"outage_{unit}_p{period}", "outage", unit, factor, period))
    return out


def member_of(column, prefix):
    """BUY_ARAB_LIGHT_7 -> ARAB_LIGHT: the name between the prefix and the
    period, which may itself contain underscores."""
    return column[len(prefix):].rsplit("_", 1)[0]


def apply(m, kind, subject, factor, period):
    if kind == "price":
        for column in list(m.obj):
            for prefix in ("BUY_", "INVC_"):
                if column.startswith(prefix) and member_of(column, prefix) == subject:
                    m.obj[column] *= factor
    elif kind in ("supply", "demand"):
        prefix = "BUY_" if kind == "supply" else "SELL_"
        moved = []
        for bound_kind, column, value in m.bounds:
            if (bound_kind == "UP" and value is not None and column.startswith(prefix)
                    and member_of(column, prefix) == subject):
                value *= factor
            moved.append((bound_kind, column, value))
        m.bounds = moved
    elif kind == "outage":
        row = f"{subject}CAP_{period}"
        assert row in m.rhs, f"no capacity row {row}"
        m.rhs[row] *= factor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", type=int, default=12)
    ap.add_argument("--crudes", type=int, default=8)
    ap.add_argument("--capacity", type=float, default=1.5e6,
                    help="CDU throughput per period, in barrels")
    ap.add_argument("--out", default="data/refinery/cases")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("case_*.mps"):
        old.unlink()

    stack = cases(args.crudes)
    for index, (name, kind, subject, factor, period) in enumerate(stack):
        m = rm.build(args.periods, args.crudes, args.capacity, 1.0, False)
        if kind != "base":
            apply(m, kind, subject, factor, period)
        with (out / f"case_{index:03d}_{name}.mps").open("w") as handle:
            m.write(handle)

    print(f"wrote the base and {len(stack) - 1} cases to {out}/  "
          f"(periods {args.periods}, crudes {args.crudes})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
