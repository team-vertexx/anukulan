#!/usr/bin/env python3
"""Independently check a solution against the model file.

This shares no code with the solver. It parses the MPS itself, in Python, and
recomputes every row activity, every bound, and the objective from scratch. The
point is that a bug in our C++ reader or our residual arithmetic cannot hide
here - if both agreed because they share a mistake, this would still catch it.

Reports absolute and relative violations, and the largest offenders by name, so
a disagreement can be localised to specific rows instead of argued about.
"""

import argparse
import sys
from collections import defaultdict

INF = float("inf")


def parse_mps(path):
    """Minimal but complete MPS parse. Free format; falls back to fixed columns
    when a ROWS line does not split into exactly two tokens."""
    rows, row_type, order = {}, {}, []
    cols, obj = defaultdict(dict), defaultdict(float)
    rhs, rng = {}, {}
    lower, upper = defaultdict(float), defaultdict(lambda: INF)
    seen_cols, objective_name, offset = [], None, 0.0
    section, fixed, row_lines = None, False, []

    def fields(line):
        if not fixed:
            return line.split()
        spans = [(1, 3), (4, 12), (14, 22), (24, 36), (39, 47), (49, len(line))]
        out = []
        for a, b in spans:
            if a >= len(line):
                break
            piece = line[a:min(b, len(line))].strip()
            if piece:
                out.append(piece)
        return out

    raw = open(path, errors="ignore").read().splitlines()

    # Decide the format from the ROWS section, same rule the C++ reader uses.
    in_rows = False
    for line in raw:
        if not line or line[0] == "*":
            continue
        if not line[0].isspace():
            in_rows = line.split()[0].upper() == "ROWS"
            continue
        if in_rows and len(line.split()) != 2:
            fixed = True
            break

    for line in raw:
        if not line or line[0] == "*":
            continue
        if not line[0].isspace():
            head = line.split()
            section = head[0].upper()
            if section == "ENDATA":
                break
            continue
        t = fields(line)
        if not t:
            continue

        if section == "ROWS":
            kind, name = t[0].upper(), t[1]
            if kind == "N":
                if objective_name is None:
                    objective_name = name
                    rows[name] = "OBJ"
                else:
                    rows[name] = "FREE"
            else:
                rows[name] = kind
                row_type[name] = kind
                order.append(name)
        elif section == "COLUMNS":
            if any(x.upper().strip("'") == "MARKER" for x in t):
                continue
            col = t[0]
            if col not in cols:
                seen_cols.append(col)
                cols[col] = {}
            for k in range(1, len(t) - 1, 2):
                name, value = t[k], float(t[k + 1])
                if rows.get(name) == "OBJ":
                    obj[col] += value
                elif rows.get(name) == "FREE":
                    continue
                else:
                    cols[col][name] = cols[col].get(name, 0.0) + value
        elif section in ("RHS", "RANGES"):
            first = len(t) % 2
            for k in range(first, len(t) - 1, 2):
                name, value = t[k], float(t[k + 1])
                if rows.get(name) == "OBJ":
                    offset = -value
                elif rows.get(name) == "FREE":
                    continue
                elif section == "RHS":
                    rhs[name] = value
                else:
                    rng[name] = value
        elif section == "BOUNDS":
            kind = t[0].upper()
            takes_value = kind in ("UP", "LO", "FX", "LI", "UI", "SI", "SC")
            if takes_value:
                col, value = (t[1], float(t[2])) if len(t) == 3 else (t[2], float(t[3]))
            else:
                col = t[1] if (len(t) == 2 or t[1] in cols) else t[2]
                value = 0.0
            if col not in cols:
                seen_cols.append(col)
                cols[col] = {}
            if abs(value) >= 1e30:
                value = INF if value > 0 else -INF
            if kind == "UP":
                upper[col] = value
            elif kind == "LO":
                lower[col] = value
            elif kind == "FX":
                lower[col] = upper[col] = value
            elif kind == "FR":
                lower[col], upper[col] = -INF, INF
            elif kind == "MI":
                lower[col] = -INF
            elif kind == "PL":
                upper[col] = INF
            elif kind == "BV":
                lower[col], upper[col] = 0.0, 1.0
            elif kind in ("LI",):
                lower[col] = value
            elif kind in ("UI", "SI", "SC"):
                upper[col] = value

    bounds = {}
    for col in seen_cols:
        bounds[col] = (lower[col], upper[col])

    limits = {}
    for name in order:
        kind, b = row_type[name], rhs.get(name, 0.0)
        if kind == "E":
            lo = hi = b
        elif kind == "L":
            lo, hi = -INF, b
        else:
            lo, hi = b, INF
        if name in rng:
            r = rng[name]
            if (kind == "E" and r < 0) or kind == "L":
                lo = hi - abs(r)
            elif (kind == "E" and r > 0) or kind == "G":
                hi = lo + abs(r)
        limits[name] = (lo, hi)

    return {"cols": cols, "obj": obj, "offset": offset, "bounds": bounds,
            "limits": limits, "order": order, "col_order": seen_cols}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("solution")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    m = parse_mps(args.model)

    x, claimed = {}, None
    for line in open(args.solution):
        if line.startswith("# objective"):
            claimed = float(line.split()[2])
            continue
        if line.startswith("#") or not line.strip():
            continue
        name, value = line.split()
        x[name] = float(value)

    activity = defaultdict(float)
    for col, entries in m["cols"].items():
        v = x.get(col, 0.0)
        if v == 0.0:
            continue
        for row, coeff in entries.items():
            activity[row] += coeff * v

    row_bad, worst_rows = 0.0, []
    for name in m["order"]:
        lo, hi = m["limits"][name]
        a = activity.get(name, 0.0)
        viol = max(lo - a, a - hi, 0.0)
        if viol > 0:
            worst_rows.append((viol, name, a, lo, hi))
        row_bad = max(row_bad, viol)

    bound_bad, worst_bounds = 0.0, []
    for col, (lo, hi) in m["bounds"].items():
        v = x.get(col, 0.0)
        viol = max(lo - v, v - hi, 0.0)
        if viol > 0:
            worst_bounds.append((viol, col, v, lo, hi))
        bound_bad = max(bound_bad, viol)

    recomputed = m["offset"] + sum(c * x.get(col, 0.0) for col, c in m["obj"].items())

    print(f"model            {args.model}")
    print(f"rows {len(m['order'])}   cols {len(m['bounds'])}")
    print(f"objective claimed by solver   {claimed!r}")
    print(f"objective recomputed here     {recomputed:.12e}")
    if claimed is not None:
        d = abs(recomputed - claimed) / max(1.0, abs(claimed))
        print(f"  agreement                   {d:.3e}"
              f"   {'OK' if d < 1e-9 else 'MISMATCH'}")
    print(f"worst row violation           {row_bad:.6e}")
    print(f"worst bound violation         {bound_bad:.6e}")

    worst_rows.sort(reverse=True)
    for viol, name, a, lo, hi in worst_rows[: args.top]:
        print(f"    row {name:<20} activity {a:.6e} outside [{lo:.6e}, {hi:.6e}]"
              f"  by {viol:.3e}")
    worst_bounds.sort(reverse=True)
    for viol, col, v, lo, hi in worst_bounds[: args.top]:
        print(f"    col {col:<20} value {v:.6e} outside [{lo:.6e}, {hi:.6e}]"
              f"  by {viol:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
