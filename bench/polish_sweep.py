#!/usr/bin/env python3
"""Feasibility polishing, measured against no polishing on the Netlib set.

Two regimes, because polishing is only meaningful in one of them:

  strict    one tolerance on everything, which is what a caller usually asks
            for. Polishing has to earn its keep without a relaxed gap.
  polish    the regime the technique is for: tight feasibility, a gap allowed
            to be 1%. This is the setting the PDLP paper reports.
"""
import csv, json, pathlib, subprocess, sys, time

BINARY = "build/sankhya"
SET = ["afiro", "sc50a", "adlittle", "blend", "share1b", "stocfor1", "sctap1",
       "scfxm1", "bandm", "degen2", "fit1p", "25fv47", "woodw", "degen3",
       "stocfor2", "greenbea", "pilot87", "maros-r7"]

def optima():
    out = {}
    with open("data/reference/netlib.csv") as fh:
        for row in csv.DictReader(fh):
            if row["optimal"]:
                out[row["name"]] = float(row["optimal"])
    return out

def run(name, args):
    path = f"data/netlib/{name}.mps"
    cmd = [BINARY, "solve", path, "--format=json", "--presolve"] + args
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - started
    try:
        d = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return None
    d["wall"] = wall
    # polishing iterations are additional to the main loop's, so a fair
    # comparison of work done has to add them.
    d["work"] = d["iterations"] + d.get("polish_iterations", 0)
    return d

def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "polish"
    if regime == "strict":
        base = ["--tol=1e-8"]
    else:
        base = ["--tol=1e-8", "--gap-tol=1e-2"]
    extra = sys.argv[2:]          # applied to the polished run only
    base_on = base + extra
    ref = optima()
    print(f"regime={regime}  base={' '.join(base)}  extra={' '.join(extra) or '(none)'}")
    print(f"{'instance':<11} | {'no polish':>28} | {'with polish':>34} | ratio")
    print(f"{'':<11} | {'iters':>8} {'secs':>6} {'obj err':>11} |"
          f" {'iters':>8} {'secs':>6} {'obj err':>11} {'feas':>8} | iters  secs")
    tot_off = tot_on = twall_off = twall_on = 0.0
    wins = 0
    adopted = []
    for name in SET:
        off = run(name, base + ["--no-polish"])
        on = run(name, base_on)
        if off is None or on is None:
            print(f"{name:<11} | run failed")
            continue
        def err(d):
            if name not in ref: return float("nan")
            o = ref[name]
            return abs(d["objective"] - o) / (1.0 + abs(o))
        feas = max(on.get("abs_primal", 0.0), on.get("original_row_violation", 0.0))
        ri = on["work"] / off["work"] if off["work"] else float("nan")
        rw = on["wall"] / off["wall"] if off["wall"] else float("nan")
        tot_off += off["work"]; tot_on += on["work"]
        twall_off += off["wall"]; twall_on += on["wall"]
        if on["work"] < off["work"]: wins += 1
        if on.get("polished"): adopted.append(name)
        def tag(d):
            return "ok" if d["status"] == "optimal" else d["status"].split(":")[0]
        flag = "" if (on["status"] == "optimal" and off["status"] == "optimal") \
               else f"  <- off:{tag(off)} on:{tag(on)}"
        print(f"{name:<11} | {off['work']:8d} {off['wall']:6.2f} {err(off):11.3e} |"
              f" {on['work']:8d} {on['wall']:6.2f} {err(on):11.3e} {feas:8.1e} |"
              f" {ri:5.2f} {rw:5.2f}{flag}")
    print(f"{'TOTAL':<11} | {int(tot_off):8d} {twall_off:6.2f} {'':>11} |"
          f" {int(tot_on):8d} {twall_on:6.2f} {'':>11} {'':>8} |"
          f" {tot_on/tot_off:5.2f} {twall_on/twall_off:5.2f}")
    print(f"polishing used fewer iterations on {wins} of {len(SET)}")
    print(f"polished point adopted on {len(adopted)}: {', '.join(adopted) or 'none'}")

main()
