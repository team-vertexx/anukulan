#!/usr/bin/env python3
"""Generate a multi-period crude blending and production planning LP as MPS.

Written because the problem statement names refinery scheduling, crude blending
and production planning as expected data, and because every public benchmark
instance we had was a 0/+-1 combinatorial LP - nothing that looks like the model
an actual refinery runs.

The structure follows the standard planning formulation: buy crudes, charge them
to the CDU, take fixed cut yields, blend cuts into finished products subject to
quality specifications, sell against demand, carry inventory between periods.
Reference formulations in Grossmann's group's refinery planning papers.

    maximise   product revenue - crude cost - operating cost - inventory cost

Two things are deliberate:

Units are left as a refinery would state them, not normalised. Volumes are in
barrels (1e3 to 1e6), prices in rupees per barrel (1e3 to 1e4), sulphur in ppm
(1e1 to 1e4). That is what makes a real planning model badly scaled, and it is
the property the public combinatorial instances do not have.

Blending quality is treated linearly. True blending is bilinear - the pooling
problem, and nonconvex - so this is the linear approximation the industry uses
for planning, with a piecewise-linear MILP version being the standard next step.
Said plainly rather than glossed over, because a refinery engineer will ask.
"""

import argparse
import sys

# Crude slate: cost in rupees per barrel, sulphur in ppm, cut yields by volume.
# Yields sum to 1. Heavier, more sour crudes are cheaper and yield more residue.
CRUDES = [
    # name        cost     sulphur   naphtha  kero   diesel   vgo  residue
    ("BOMBAY_HIGH", 6850.0,   900.0,   0.26, 0.18, 0.24, 0.20, 0.12),
    ("ARAB_LIGHT",  6420.0,  19000.0,  0.22, 0.17, 0.23, 0.22, 0.16),
    ("ARAB_HEAVY",  5980.0,  29000.0,  0.16, 0.14, 0.21, 0.25, 0.24),
    ("BASRAH",      6100.0,  24500.0,  0.18, 0.15, 0.22, 0.24, 0.21),
    ("URALS",       6260.0,  13800.0,  0.20, 0.16, 0.24, 0.23, 0.17),
    ("MURBAN",      6980.0,   7400.0,  0.28, 0.19, 0.23, 0.19, 0.11),
    ("ES_SIDER",    6540.0,   4100.0,  0.25, 0.18, 0.25, 0.20, 0.12),
    ("QUA_IBOE",    6890.0,   1300.0,  0.27, 0.20, 0.25, 0.18, 0.10),
]

CUTS = ["NAPHTHA", "KERO", "DIESEL", "VGO", "RESIDUE"]

# Conversion. Without these a planning model stops at about half of CDU capacity,
# because every extra barrel of crude also makes residue that has nowhere
# profitable to go. Real refineries upgrade it, which is why they run full.
FCC_YIELD = {"NAPHTHA": 0.55, "DIESEL": 0.25}   # rest is coke and gas, lost
FCC_CAPACITY_FRACTION = 0.30                    # of CDU throughput
FCC_COST = 310.0                                # rupees per barrel of feed
FCC_SULPHUR_FACTOR = 0.35                       # cracking removes some sulphur

# Hydrotreating is the decision that makes cheap sour crude usable: pay per
# barrel, get the sulphur down. Trading treating cost against crude price is the
# central economic question in refinery planning.
HDT_CAPACITY_FRACTION = 0.45
HDT_COST = 240.0
HDT_SULPHUR_FACTOR = 0.04

# Sulphur carried by each cut, as a multiple of the crude's sulphur. Light ends
# come off clean; sulphur concentrates in the bottom of the barrel.
CUT_SULPHUR_FACTOR = {"NAPHTHA": 0.02, "KERO": 0.15, "DIESEL": 0.55,
                      "VGO": 1.35, "RESIDUE": 2.60}

# Products: price per barrel, sulphur ceiling in ppm, which cuts may blend in.
PRODUCTS = [
    ("GASOLINE", 9250.0,   50.0, ["NAPHTHA", "KERO"]),
    ("JET",      8900.0,  2500.0, ["KERO", "NAPHTHA"]),
    ("DIESEL_P", 8600.0,   350.0, ["DIESEL", "KERO", "VGO"]),
    ("FUELOIL",  4900.0, 35000.0, ["RESIDUE", "VGO", "DIESEL"]),
]


class MpsWriter:
    """Builds an MPS file. Rows are declared first, then columns are emitted
    column-wise, which is what the format requires."""

    def __init__(self, name):
        self.name = name
        self.rows = []           # (type, name)
        self.row_seen = set()
        self.cols = {}           # col -> list of (row, coefficient)
        self.col_order = []
        self.obj = {}
        self.rhs = {}
        self.ranges = {}
        self.bounds = []         # (type, col, value)
        self.integer_cols = set()
        self.maximize = False

    def row(self, kind, name, rhs=None):
        assert name not in self.row_seen, f"duplicate row {name}"
        self.row_seen.add(name)
        self.rows.append((kind, name))
        if rhs is not None:
            self.rhs[name] = rhs
        return name

    def col(self, name):
        if name not in self.cols:
            self.cols[name] = []
            self.col_order.append(name)
        return name

    def integer(self, name):
        self.col(name)
        self.integer_cols.add(name)
        return name

    def put(self, col, row, coeff):
        if coeff == 0.0:
            return
        self.col(col)
        self.cols[col].append((row, coeff))

    def cost(self, col, coeff):
        self.col(col)
        self.obj[col] = self.obj.get(col, 0.0) + coeff

    def bound(self, kind, col, value=None):
        self.bounds.append((kind, self.col(col), value))

    def write(self, handle):
        w = handle.write
        w(f"NAME          {self.name}\n")
        if self.maximize:
            w("OBJSENSE\n    MAX\n")
        w("ROWS\n")
        w(" N  COST\n")
        for kind, name in self.rows:
            w(f" {kind}  {name}\n")
        w("COLUMNS\n")
        # Continuous columns first, then the integer block between markers, so
        # the file needs one INTORG/INTEND pair instead of one per variable.
        ordered = ([c for c in self.col_order if c not in self.integer_cols] +
                   [c for c in self.col_order if c in self.integer_cols])
        marker_open = False
        for col in ordered:
            if col in self.integer_cols and not marker_open:
                w("    MARKER                 'MARKER'                 'INTORG'\n")
                marker_open = True
            entries = []
            if col in self.obj:
                entries.append(("COST", self.obj[col]))
            entries.extend(self.cols[col])
            for i in range(0, len(entries), 2):
                pair = entries[i:i + 2]
                line = f"    {col:<24}"
                for row, coeff in pair:
                    line += f"  {row:<24} {coeff:.10g}"
                w(line + "\n")
        if marker_open:
            w("    MARKER                 'MARKER'                 'INTEND'\n")
        w("RHS\n")
        for name, value in self.rhs.items():
            if value != 0.0:
                w(f"    RHS       {name:<24} {value:.10g}\n")
        if self.ranges:
            w("RANGES\n")
            for name, value in self.ranges.items():
                w(f"    RNG       {name:<24} {value:.10g}\n")
        w("BOUNDS\n")
        for kind, col, value in self.bounds:
            if value is None:
                w(f" {kind} BND       {col}\n")
            else:
                w(f" {kind} BND       {col:<24} {value:.10g}\n")
        w("ENDATA\n")


# Discrete decisions. These are what make refinery scheduling a mixed-integer
# problem rather than a linear one, and both are physical rather than modelling
# conveniences:
#
#   Crude arrives in cargoes. You cannot buy 12,000 barrels of Arab Light; you
#   charter a vessel or you do not, and a parcel is on the order of a third of a
#   period's throughput.
#
#   A conversion unit either runs or is down. Below a minimum throughput the
#   catalyst circulation and heat balance do not hold, so a plan that runs the
#   FCC at 3% of capacity is not a plan.
CARGO_FRACTION = 0.30        # one parcel, as a fraction of CDU throughput
FCC_MIN_FRACTION = 0.40      # of FCC capacity, when running at all
HDT_MIN_FRACTION = 0.25


def build(periods, crudes, cdu_capacity, seed_scale, discrete=False):
    """Component-level blending.

    The first version of this generator pooled each cut across all crudes and
    gave the pool one average sulphur figure. That is wrong in a way that shows
    up immediately in the answer rather than in the objective: with a single
    average, the quality constraint stops depending on which crude was bought,
    so crude selection - the entire point of the model - carries no consequence.
    It also made the average naphtha 82 ppm against a 50 ppm gasoline
    specification, so gasoline became impossible to produce and the solver ran
    the CDU at a quarter of capacity making fuel oil.

    Keeping the streams separate down to the blender fixes it and stays linear:
    the sulphur of the naphtha cut from a specific crude is a known constant, so
    a blend of those streams has a linear sulphur balance. This is the
    no-pooling formulation. Pooling the streams first is what makes real
    blending bilinear and nonconvex, and the piecewise-linear MILP version of
    that is the standard next step.
    """
    slate = CRUDES[:crudes]
    m = MpsWriter("REFINERY")
    m.maximize = True

    def sulphur_of(crude_sulphur, cut):
        return crude_sulphur * CUT_SULPHUR_FACTOR[cut]

    def demand(product_index, t):
        # A seasonal wobble so the periods are not identical, which would make
        # the model separable and uninterestingly easy.
        base = [0.34, 0.14, 0.36, 0.16][product_index] * cdu_capacity
        wobble = 1.0 + 0.18 * ((t * 7 + product_index * 3) % 5 - 2) / 2.0
        return base * wobble

    for t in range(1, periods + 1):
        cap = m.row("L", f"CDUCAP_{t}", cdu_capacity)

        for cname, ccost, csulph, *yields in slate:
            buy = f"BUY_{cname}_{t}"
            chg = f"CHG_{cname}_{t}"
            inv = f"INVC_{cname}_{t}"

            m.cost(buy, -ccost)
            m.cost(inv, -0.012 * ccost)
            m.cost(chg, -145.0)          # CDU operating cost per barrel

            bal = m.row("E", f"CBAL_{cname}_{t}", 0.0)
            m.put(buy, bal, 1.0)
            m.put(chg, bal, -1.0)
            m.put(inv, bal, -1.0)
            if t > 1:
                m.put(f"INVC_{cname}_{t-1}", bal, 1.0)

            m.put(chg, cap, 1.0)
            m.bound("UP", inv, 0.55 * cdu_capacity)

            if discrete:
                # buy = cargo_size * (number of parcels chartered)
                parcels = f"NCARGO_{cname}_{t}"
                link = m.row("E", f"CARGO_{cname}_{t}", 0.0)
                m.put(buy, link, 1.0)
                m.put(parcels, link, -CARGO_FRACTION * cdu_capacity)
                m.integer(parcels)
                m.bound("UI", parcels, 2.0)
            else:
                m.bound("UP", buy, 0.42 * cdu_capacity)

            # Each (crude, cut) stream is produced by yield, then routed: blended
            # straight, hydrotreated first, cracked (VGO only), or left unused.
            for cut_index, cut in enumerate(CUTS):
                stream = m.row("E", f"STRM_{cname}_{cut}_{t}", 0.0)
                m.put(chg, stream, yields[cut_index])
                m.put(f"SLACK_{cname}_{cut}_{t}", stream, -1.0)
                m.cost(f"SLACK_{cname}_{cut}_{t}", -180.0)
                m.put(f"HDT_{cname}_{cut}_{t}", stream, -1.0)
                if cut == "VGO":
                    m.put(f"FCC_{cname}_{t}", stream, -1.0)
                for pname, _price, _spec, allowed in PRODUCTS:
                    if cut in allowed:
                        m.put(f"BL_{cname}_{cut}_{pname}_{t}", stream, -1.0)

                # Hydrotreated stream: same volume, much less sulphur, at a cost.
                m.cost(f"HDT_{cname}_{cut}_{t}", -HDT_COST)
                treated = m.row("E", f"TSTRM_{cname}_{cut}_{t}", 0.0)
                m.put(f"HDT_{cname}_{cut}_{t}", treated, 1.0)
                m.put(f"TSLACK_{cname}_{cut}_{t}", treated, -1.0)
                m.cost(f"TSLACK_{cname}_{cut}_{t}", -180.0)
                for pname, _price, _spec, allowed in PRODUCTS:
                    if cut in allowed:
                        m.put(f"TBL_{cname}_{cut}_{pname}_{t}", treated, -1.0)

            # Cracking VGO into lighter streams.
            m.cost(f"FCC_{cname}_{t}", -FCC_COST)
            for cut, y in FCC_YIELD.items():
                cracked = m.row("E", f"FSTRM_{cname}_{cut}_{t}", 0.0)
                m.put(f"FCC_{cname}_{t}", cracked, y)
                m.put(f"FSLACK_{cname}_{cut}_{t}", cracked, -1.0)
                m.cost(f"FSLACK_{cname}_{cut}_{t}", -180.0)
                for pname, _price, _spec, allowed in PRODUCTS:
                    if cut in allowed:
                        m.put(f"FBL_{cname}_{cut}_{pname}_{t}", cracked, -1.0)

        # Unit capacities, shared across the crude slate.
        fcc_cap = m.row("L", f"FCCCAP_{t}", 0.0 if discrete
                        else FCC_CAPACITY_FRACTION * cdu_capacity)
        hdt_cap = m.row("L", f"HDTCAP_{t}", 0.0 if discrete
                        else HDT_CAPACITY_FRACTION * cdu_capacity)
        for cname, _cc, _cs, *_y in slate:
            m.put(f"FCC_{cname}_{t}", fcc_cap, 1.0)
            for cut in CUTS:
                m.put(f"HDT_{cname}_{cut}_{t}", hdt_cap, 1.0)

        if discrete:
            # Running the unit is a decision, and running it at all means
            # running it above a minimum rate. Written as
            #   min_rate * on <= throughput <= capacity * on
            # so the on/off variable both caps and floors the unit.
            for tag, cap_row, cap_fraction, min_fraction, columns in (
                ("FCC", fcc_cap, FCC_CAPACITY_FRACTION, FCC_MIN_FRACTION,
                 [f"FCC_{c[0]}_{t}" for c in slate]),
                ("HDT", hdt_cap, HDT_CAPACITY_FRACTION, HDT_MIN_FRACTION,
                 [f"HDT_{c[0]}_{cut}_{t}" for c in slate for cut in CUTS]),
            ):
                on = f"ON_{tag}_{t}"
                m.integer(on)
                m.bound("UP", on, 1.0)
                m.put(on, cap_row, -cap_fraction * cdu_capacity)
                floor_row = m.row("G", f"{tag}MIN_{t}", 0.0)
                for col in columns:
                    m.put(col, floor_row, 1.0)
                m.put(on, floor_row, -min_fraction * cap_fraction * cdu_capacity)
                # A unit that is down still costs something to keep warm, so
                # switching it off is a real decision rather than free.
                m.cost(on, -0.004 * cap_fraction * cdu_capacity * 145.0)

        for pi, (pname, price, spec, allowed) in enumerate(PRODUCTS):
            sale = f"SELL_{pname}_{t}"
            m.cost(sale, price)

            pbal = m.row("E", f"PBAL_{pname}_{t}", 0.0)
            m.put(sale, pbal, -1.0)
            qual = m.row("L", f"QUAL_{pname}_{t}", 0.0)

            for cname, _ccost, csulph, *_y in slate:
                for cut in allowed:
                    # Three routes to the blender, each with its own sulphur:
                    # straight-run, hydrotreated, and cracked.
                    for prefix, factor in (("BL", 1.0),
                                           ("TBL", HDT_SULPHUR_FACTOR),
                                           ("FBL", FCC_SULPHUR_FACTOR)):
                        if prefix == "FBL" and cut not in FCC_YIELD:
                            continue
                        col = f"{prefix}_{cname}_{cut}_{pname}_{t}"
                        m.put(col, pbal, 1.0)
                        # sum (sulphur_stream - spec) * volume <= 0, which is the
                        # blended sulphur staying under the specification.
                        m.put(col, qual, sulphur_of(csulph, cut) * factor - spec)

            m.bound("UP", sale, demand(pi, t))

    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", type=int, default=12)
    ap.add_argument("--crudes", type=int, default=8)
    ap.add_argument("--capacity", type=float, default=1.5e6,
                    help="CDU throughput per period, in barrels")
    ap.add_argument("--out", default="data/refinery/refinery.mps")
    ap.add_argument("--milp", action="store_true",
                    help="add cargo sizing and unit on/off decisions")
    args = ap.parse_args()

    if args.crudes > len(CRUDES):
        print(f"only {len(CRUDES)} crudes are defined", file=sys.stderr)
        return 1

    import pathlib
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m = build(args.periods, args.crudes, args.capacity, 1.0, args.milp)
    with out.open("w") as handle:
        m.write(handle)

    coeffs = [abs(c) for entries in m.cols.values() for _r, c in entries if c]
    print(f"wrote {out}")
    print(f"  periods {args.periods}, crudes {args.crudes}, "
          f"CDU capacity {args.capacity:,.0f} bbl/period")
    if m.integer_cols:
        print(f"  integer columns {len(m.integer_cols)}")
    print(f"  rows {len(m.rows)}, cols {len(m.col_order)}, "
          f"nonzeros {sum(len(v) for v in m.cols.values()) + len(m.obj)}")
    print(f"  |a| in [{min(coeffs):.4g}, {max(coeffs):.4g}]  "
          f"ratio {max(coeffs)/min(coeffs):.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
