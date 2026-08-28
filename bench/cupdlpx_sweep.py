#!/usr/bin/env python3
"""The four cuPDLPx additions, measured against the Halpern base they sit on.

Run with a config name to ablate one piece at a time; with no argument it
compares the full set against the baseline.
"""
import json, subprocess, sys, time

SET = ["afiro","sc50a","adlittle","blend","share1b","stocfor1","sctap1","scfxm1",
       "bandm","degen2","fit1p","25fv47","woodw","degen3","stocfor2","greenbea",
       "pilot87","maros-r7"]

CONFIGS = {
    "cupdlpx":     ["--cupdlpx"],
    "no-reflect":  ["--constant-step","--fixed-point-restart","--pid-weight"],
    "no-constant": ["--reflection=1.0","--fixed-point-restart","--pid-weight"],
    "no-fpr":      ["--reflection=1.0","--constant-step","--pid-weight"],
    "no-pid":      ["--reflection=1.0","--constant-step","--fixed-point-restart"],
    "step090":     ["--reflection=1.0","--step-scale=0.90","--fixed-point-restart","--pid-weight"],
    "step095":     ["--reflection=1.0","--step-scale=0.95","--fixed-point-restart","--pid-weight"],
}

def run(name, extra, tol):
    cmd = ["build/sankhya","solve",f"data/netlib/{name}.mps","--format=json",
           "--presolve",f"--tol={tol}"] + extra
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    w = time.perf_counter() - t
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return None
    d["wall"] = w
    d["work"] = d["iterations"] + d.get("polish_iterations", 0)
    return d

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cupdlpx"
    tol = sys.argv[2] if len(sys.argv) > 2 else "1e-8"
    extra = CONFIGS[which]
    print(f"config={which}  flags={' '.join(extra)}  tol={tol}")
    print(f"{'instance':<11} | {'base iters':>10} {'base s':>7} | {'new iters':>10} {'new s':>7} | {'iters':>6} {'secs':>6}")
    tb=tn=wb=wn=0.0; wins=0; bad=[]
    for n in SET:
        a = run(n, [], tol); b = run(n, extra, tol)
        if a is None or b is None:
            print(f"{n:<11} | run failed"); continue
        tb+=a["work"]; tn+=b["work"]; wb+=a["wall"]; wn+=b["wall"]
        if b["work"] < a["work"]: wins += 1
        flag = ""
        if a["status"]=="optimal" and b["status"]!="optimal":
            flag = f"  <- LOST ({b['status'].split(':')[0]})"; bad.append(n)
        elif b["status"]=="optimal" and a["status"]!="optimal":
            flag = "  <- gained"
        print(f"{n:<11} | {a['work']:10d} {a['wall']:7.2f} | {b['work']:10d} {b['wall']:7.2f} | "
              f"{b['work']/a['work']:6.2f} {b['wall']/a['wall']:6.2f}{flag}")
    print(f"{'TOTAL':<11} | {int(tb):10d} {wb:7.2f} | {int(tn):10d} {wn:7.2f} | "
          f"{tn/tb:6.2f} {wn/wb:6.2f}")
    print(f"fewer iterations on {wins} of {len(SET)}"
          + (f";  REGRESSED: {', '.join(bad)}" if bad else ";  no instance lost"))

main()
