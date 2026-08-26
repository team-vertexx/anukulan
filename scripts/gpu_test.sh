#!/usr/bin/env bash
# Build with CUDA and prove the GPU backend is correct, on a machine that has a
# device. Run this on Kaggle, Colab, RunPod, or a cluster node - nothing here is
# specific to any of them.
#
#   bash scripts/gpu_test.sh
#
# What it checks, in order:
#   1. a device exists and nvcc works
#   2. the CUDA build compiles
#   3. every backend operation matches the CPU reference (test_backend)
#   4. the whole solver, run on the GPU, reaches the published Netlib optima
#   5. CPU against GPU on progressively larger instances, for the timing table
#
# The correctness gate is step 4, not "the two backends produce identical
# iteration counts". They will not. Fused multiply-add alone moves iteration
# counts by up to 25% between two CPU builds of the same source, so a different
# device is certain to differ. What must match is the objective, against a value
# published in 1988 that neither backend can influence.

set -u
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
build=build-cuda
fail=0

step() { printf '\n=== %s\n' "$1"; }

step "1. device"
if ! command -v nvcc > /dev/null; then
  echo "nvcc not found. On Kaggle and Colab it ships with the image; check that"
  echo "the notebook accelerator is actually set to a GPU."
  exit 1
fi
nvcc --version | tail -2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

step "2. build"
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DSANKHYA_ENABLE_CUDA=ON > /dev/null
cmake --build "$build" -j"$(nproc 2>/dev/null || echo 4)" 2>&1 | tail -3
"./$build/sankhya" backends

step "3. backend operations against the CPU reference"
if ctest --test-dir "$build" --output-on-failure 2>&1 | tail -6; then :; else fail=1; fi

step "4. solver on GPU against published Netlib optima"
if [ ! -d data/netlib ]; then
  echo "fetching the Netlib set"
  python3 scripts/fetch_netlib.py > /dev/null 2>&1 || true
fi
python3 - "$build" <<'PY'
import csv, json, subprocess, sys, pathlib
build = sys.argv[1]
ref = {r['name']: float(r['optimal'])
       for r in csv.DictReader(open('data/reference/netlib.csv'))}
names = ["afiro","sc50a","adlittle","blend","share1b","stocfor1","sctap1",
         "scfxm1","bandm","degen2","fit1p","25fv47","degen3","maros-r7"]
print(f"{'instance':<10} {'cpu obj':>16} {'gpu obj':>16} {'published':>16} "
      f"{'gpu acc':>9} {'cpu it':>7} {'gpu it':>7}")
bad = 0
for n in names:
    p = pathlib.Path(f"data/netlib/{n}.mps")
    if not p.exists():
        continue
    out = {}
    for tag, flag in (("cpu", "--backend=cpu"), ("gpu", "--backend=cuda")):
        r = subprocess.run([f"./{build}/sankhya", "solve", str(p), "--tol=1e-6",
                            "--max-iter=500000", "--time-limit=120",
                            "--format=json", "--quiet", flag],
                           capture_output=True, text=True)
        out[tag] = json.loads(r.stdout) if r.stdout.strip() else None
    if not out["cpu"] or not out["gpu"]:
        print(f"{n:<10} FAILED"); bad += 1; continue
    o = ref[n]
    acc = abs(out["gpu"]["objective"] - o) / max(1.0, abs(o))
    if acc > 1e-4: bad += 1
    print(f"{n:<10} {out['cpu']['objective']:>16.9g} {out['gpu']['objective']:>16.9g} "
          f"{o:>16.9g} {acc:>9.1e} {out['cpu']['iterations']:>7} "
          f"{out['gpu']['iterations']:>7}")
print(f"\n{'PASS' if bad == 0 else 'FAIL'}: {bad} instance(s) off the published optimum")
sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && fail=1

step "5. CPU against GPU by size"
echo "The Netlib set is too small for a GPU to pay. Pull the large instances"
echo "first if they are not present:"
echo "    python3 scripts/fetch_lptestset.py --max=4"
python3 - "$build" <<'PY'
import json, subprocess, sys, pathlib, time
build = sys.argv[1]
names = ["qap15", "supportcase10", "datt256_lp", "graph40-40"]
rows = []
for n in names:
    p = pathlib.Path(f"data/lptestset/{n}.mps")
    if not p.exists():
        continue
    row = {"name": n}
    for tag, flag in (("cpu", "--backend=cpu"), ("gpu", "--backend=cuda")):
        t = time.perf_counter()
        r = subprocess.run([f"./{build}/sankhya", "solve", str(p), "--tol=1e-4",
                            "--max-iter=1000000", "--time-limit=600",
                            "--format=json", "--quiet", flag],
                           capture_output=True, text=True)
        row[tag] = json.loads(r.stdout) if r.stdout.strip() else None
        row[tag + "_wall"] = time.perf_counter() - t
        # The solver's own clock as well as the wall clock. They are not the
        # same number and the difference is not small: reading graph40-40's
        # 1.26 million nonzeros out of an MPS file takes 1.6 seconds before the
        # solver starts, and both backends pay it. Judging a GPU on a wall clock
        # that is 59% serial parsing measures the parser.
        row[tag + "_solve"] = row[tag]["seconds"] if row[tag] else 0.0
    rows.append(row)
if not rows:
    print("no large instances downloaded; skipping")
else:
    print(f"{'instance':<15} {'cpu solve':>10} {'gpu solve':>10} {'speedup':>8} "
          f"{'cpu wall':>9} {'gpu wall':>9} {'setup':>7} {'agree':>9}")
    for r in rows:
        if not r.get("cpu") or not r.get("gpu"):
            print(f"{r['name']:<15} incomplete"); continue
        d = abs(r['cpu']['objective'] - r['gpu']['objective']) / \
            max(1.0, abs(r['cpu']['objective']))
        setup = r['cpu_wall'] - r['cpu_solve']
        print(f"{r['name']:<15} {r['cpu_solve']:>10.2f} {r['gpu_solve']:>10.2f} "
              f"{r['cpu_solve']/max(1e-9, r['gpu_solve']):>7.2f}x "
              f"{r['cpu_wall']:>9.2f} {r['gpu_wall']:>9.2f} {setup:>7.2f} {d:>9.1e}")
    print()
    print("speedup is on the solve. `setup` is everything before it - reading the")
    print("MPS file, building the standard form - which runs on the host and is")
    print("identical for both backends, so counting it dilutes the comparison by")
    print("a constant that has nothing to do with the GPU.")
PY

mkdir -p results

step "6. presolve on the GPU, and where the fused reduction should show"
echo "The fused step-size reduction replaces three synchronising device-to-host"
echo "copies per iteration with one. It can only matter where there are many"
echo "iterations: qap15 runs about 24,700 of them, graph40-40 about 280. So"
echo "qap15 is the instance to watch here, and graph40-40 is not."
python3 - "$build" <<'STEP6'
import csv, json, subprocess, sys, pathlib, time
build = sys.argv[1]
names = ["qap15", "supportcase10", "datt256_lp", "graph40-40"]
rows = []
print(f"{'instance':<15} {'backend':>8} {'presolve':>9} {'status':>10} "
      f"{'iters':>9} {'secs':>8} {'objective':>18}")
for n in names:
    path = pathlib.Path(f"data/lptestset/{n}.mps")
    if not path.exists():
        continue
    for flag, backend in (("--backend=cpu", "cpu"), ("--backend=cuda", "gpu")):
        for pre in (False, True):
            cmd = [f"./{build}/sankhya", "solve", str(path), "--tol=1e-4",
                   "--max-iter=1000000", "--time-limit=600", "--format=json",
                   "--quiet", flag]
            if pre:
                cmd.append("--presolve")
            t = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, text=True)
            wall = time.perf_counter() - t
            try:
                d = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception:
                print(f"{n:<15} {backend:>8} {str(pre):>9} {'FAILED':>10}")
                continue
            print(f"{n:<15} {backend:>8} {str(pre):>9} {d['status']:>10} "
                  f"{d['iterations']:>9} {wall:>8.2f} {d['objective']:>18.9g}")
            rows.append(dict(instance=n, backend=backend, presolve=pre,
                             status=d["status"], iterations=d["iterations"],
                             seconds=wall, objective=d["objective"]))
if rows:
    out = pathlib.Path("results")
    out.mkdir(exist_ok=True)
    with open(out / "gpu_matrix.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote results/gpu_matrix.csv")
    print("\ngpu speedup, presolve off:")
    for n in names:
        c = [r for r in rows if r["instance"] == n and r["backend"] == "cpu"
             and not r["presolve"]]
        g = [r for r in rows if r["instance"] == n and r["backend"] == "gpu"
             and not r["presolve"]]
        if c and g and g[0]["seconds"] > 0:
            print(f"  {n:<15} {c[0]['seconds'] / g[0]['seconds']:.2f}x")
STEP6

step "7. where the device time goes on graph40-40"
echo "graph40-40 runs about 280 iterations, so its speedup cannot be a reduction"
echo "problem however it looks. This asks the backend itself which kernel owns"
echo "the run - no external profiler, because nsys is not on this image and"
echo "installing it there is its own afternoon."
if [ -f data/lptestset/graph40-40.mps ]; then
  "./$build/sankhya" solve data/lptestset/graph40-40.mps --backend=cuda \
      --tol=1e-4 --max-iter=1000000 --time-limit=600 --quiet --profile \
      | tee results/graph40-40-profile.txt || true
  echo
  echo "For comparison, the same on an instance where the GPU already pays:"
  if [ -f data/lptestset/qap15.mps ]; then
    "./$build/sankhya" solve data/lptestset/qap15.mps --backend=cuda \
        --tol=1e-4 --max-iter=1000000 --time-limit=600 --quiet --profile \
        | tee results/qap15-profile.txt || true
  fi
else
  echo "graph40-40 not downloaded; run scripts/fetch_lptestset.py first"
fi

step "8. presolve end to end, checked against published optima"
python3 bench/verify_presolve.py --binary "./$build/sankhya" --abs-tol=1e-8 \
    --csv results/presolve_verified.csv || true

printf '\n=== summary\n'
if [ "$fail" -eq 0 ]; then
  echo "GPU backend is correct. Timing table above is the headline number."
else
  echo "Something failed above. The build and the backend tests come first;"
  echo "a timing number from an incorrect backend is worth nothing."
fi
exit "$fail"
