import re, subprocess, sys
inst = sys.argv[1] if len(sys.argv)>1 else "data/lptestset/datt256_lp.mps"
iters = sys.argv[2] if len(sys.argv)>2 else "300"
counts = [2,3,4,5,6,7,8]
rows={}
for t in counts:
    out = subprocess.run(["build/sankhya","solve",inst,"--quiet","--no-polish",
                          f"--max-iter={iters}",f"--threads={t}","--profile"],
                         capture_output=True,text=True).stdout
    for line in out.splitlines():
        m = re.match(r"^(K x|K' y|primal_step|dual_step|dot \(serial\)|accumulate|blend|advance_kx|inf_norm)\s+([\d.]+)\s+([\d.]+)%\s+(\d+)\s+([\d.]+)", line)
        if m:
            rows.setdefault(m.group(1),{})[t] = (float(m.group(2)), float(m.group(5)))
print(f"per-call microseconds, and scaling against 2 threads\n")
hdr = "kernel".ljust(16) + "".join(f"{t:>10}" for t in counts) + "   sp(2->8)"
print(hdr); print("-"*len(hdr))
for name, d in sorted(rows.items(), key=lambda kv: -kv[1].get(counts[0],(0,0))[0]):
    line = name.ljust(16)
    for t in counts:
        line += f"{d.get(t,(0,0))[1]:>10.1f}" if t in d else " "*10
    if counts[0] in d and counts[-1] in d and d[counts[-1]][1]>0:
        line += f"{d[counts[0]][1]/d[counts[-1]][1]:>11.2f}"
    print(line)
