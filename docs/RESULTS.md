# Results

Every number here was measured on the machine described below. Next to each is
the command that reproduces it. Where a number is not verified, it says so.

**Machine.** Apple Silicon (arm64), macOS 26.5, Apple Clang 21.0, Release build.
GPU numbers are from a rented **NVIDIA Tesla T4** on Kaggle — see
[KAGGLE.md](KAGGLE.md).

**Benchmark sets.** Netlib LP (88 instances, `data/netlib/`), published optima in
`data/reference/netlib.csv`. MIPLIB subset (`data/miplib/`). Maros–Meszaros QP
(`data/maros/`). The refinery model (`data/refinery/refinery.mps`) is the MRPL
problem statement's own model.

---

## 1. Where this actually sits

Read this section before the tables, because the tables flatter without it.

Against **HiGHS**, the realistic open-source bar, this solver is roughly **half**
of one:

| component | vs production | why |
|---|---|---|
| MPS reader | ~95 | genuinely at parity, and faster |
| First-order LP | ~62 | strongest piece; missing multithreading and some cuPDLPx tuning |
| GPU | ~55 | real 2.7–7× measured, but not cuPDLP-C level |
| QP | ~50 | OSQP's core is here; no AMD ordering, thin regularisation strategy |
| Simplex | ~40 | correct but textbook. No Forrest–Tomlin, no bound-flipping ratio test, no hypersparsity |
| Presolve | ~38 | 10 reductions; HiGHS/PaPILO have roughly 25 |
| Infrastructure | ~70 | good tests, no CI, no packaging |
| **MILP** | **~22** | **the weak leg — see below** |

Weighted, that is about **50/100**.

**MILP is the honest gap.** What is missing: node presolve, symmetry detection,
feasibility pump, RINS, local branching, conflict analysis, restarts, clique
tables, strong branching. Five of seven exact is on the *smallest* MIPLIB
instances. Real refinery scheduling is MILP, so this is both the weakest
component and the one the problem statement cares most about. Do not let a
demo imply otherwise.

---

## 2. Reader

88/88 agreement with HiGHS on dimensions, nonzeros and objective, at 1.4–1.5×
the speed.

| instance | before | after |
|---|---|---|
| graph40-40 | 1.67 s | 1.19 s |
| datt256_lp | 1.07 s | 0.70 s |
| supportcase10 | 0.44 s | 0.30 s |

```bash
python3 bench/verify_reader.py
```

---

## 3. Presolve

Eighteen Netlib instances with published optima, doubleton equations on vs off:

| | off | on |
|---|---|---|
| rows removed | 9.1% | **14.0%** |
| columns removed | 13.4% | **15.4%** |
| nonzeros removed | 8.6% | **10.0%** |

758 doubleton equations. Nonzeros fall *despite* the fill the substitution
creates — 322,824 to 317,673.

Round trip over all 88 instances, presolved answer against plain, `--tol=1e-6`:

| | off | on |
|---|---|---|
| reached the published optimum | 77/88 | **78/88** |
| geomean wall-clock speedup | 1.52× | **1.63×** |

```bash
python3 bench/verify_presolve.py --tol=1e-6 --abs-tol=1e-6 --check-feasibility
python3 bench/verify_presolve.py --tol=1e-6 --abs-tol=1e-6 --extra=--presolve-no-doubletons
```

### Shadow prices

The dual survives presolve. On the refinery model the presolved and unpresolved
solves agree on the top rows in the same order to within 0.016%
(`TSTRM_BOMBAY_HIGH_NAPHTHA_4`: 9602.63 vs 9601.05), and the presolved one is
correctly labelled `# duals approximate` because bound tightening fired.

---

## 4. First-order LP

### Feasibility polishing

Eighteen Netlib instances, presolved, counting the polishing sub-solves'
iterations as work:

| regime | iterations | wall |
|---|---|---|
| one tolerance on everything (`--tol=1e-8`) | 1.00× | 1.00× |
| feasibility 1e-8, gap 1e-2 | **0.79×** | **0.88×** |

The first line is the point of the first: with no slack in the gap the trigger
almost never fires, so leaving it on costs nothing. The second is what the
technique is for. Gains: `fit1p` 0.06×, `bandm` 0.11×, `maros-r7` 0.12×,
`25fv47` 0.12×.

Objective errors get *worse* — `25fv47` from 1.5e-07 to 6.8e-03 — and that is
the trade, not a defect. A 1% gap was asked for.

**On the refinery model**, at the shipped defaults:

| | no polish | with polish |
|---|---|---|
| iterations | 160,720 | **12,800** + 1,720 polishing |
| capacity violation | 1.46e-02 | **1.28e-04** |
| duality gap | 1.1e-09 | 4.5e-03 |

Eleven times fewer iterations for a capacity violation 114 times smaller, paid
for with a 0.45% gap. A plan that overruns a unit by 0.015 is still a plan you
have to argue about; one that overruns by 0.0001 is not.

Before the cuPDLPx additions landed this was starker still — the unpolished
solve hit the iteration limit at 200,000 and left a violation of 0.85. The base
method getting faster is what turned that from *no answer* into *a worse
answer*, which is the better problem to have.

```bash
python3 bench/polish_sweep.py polish
python3 bench/polish_sweep.py strict
build/sankhya solve data/refinery/refinery.mps --tol=1e-8 --gap-tol=1e-2 --presolve
```

### The cuPDLPx additions

Eighteen instances, `--tol=1e-8`, presolved, against the Halpern base:

**0.79× iterations, 16 of 18 improved, no instance lost.** Excluding `greenbea`
and `pilot87` — which hit the iteration limit either way and dominate the clock
— wall time is **0.57×**, i.e. 1.75× faster.

Leave-one-out says all four are load-bearing and coupled:

| configuration | iterations vs base |
|---|---|
| all four | **0.79×** |
| without reflection | 1.06× |
| without constant step | 3.21×, `maros-r7` lost entirely |

Each piece *alone* is neutral or worse. Reflection with an adaptive step
diverges, because the adaptive safety rule was derived for plain PDHG rather
than for the reflected operator.

**The cost, stated plainly.** Across all 88 instances at `--tol=1e-8`:

| step scale | reached the published optimum | iterations (18-instance set) |
|---|---|---|
| cuPDLPx off | **76/88** | 1.27× |
| **0.90** (default) | 75/88 | 1.06× |
| 0.95 | 74/88 | 1.02× |
| 0.998 (the paper's) | 74/88 | **1.00×** |

So it buys a large speed gain and costs one instance at the tightest tolerance.
A gentler step is more robust and slower, smoothly; 0.90 buys back one of the
two for six per cent, and 0.95 buys nothing over 0.998. The default is 0.90 and
the sweep is recorded next to `constant_step_scale` in `pdhg.hpp`.

```bash
python3 bench/cupdlpx_sweep.py cupdlpx 1e-8
python3 bench/cupdlpx_sweep.py no-reflect 1e-8
python3 bench/verify_presolve.py --tol=1e-8 --abs-tol=1e-6 --both-extra="--step-scale=0.90"
```

---

## 5. Simplex

Sixteen Netlib instances with published optima:

| algorithm | correct | time |
|---|---|---|
| primal | **16/16** | 0.75 s total |
| dual | **16/16** | — |

```bash
build/sankhya simplex data/netlib/sctap1.mps
build/sankhya simplex data/netlib/fit1p.mps --dual
```

Measured and **not** adopted, recorded so they are not retried:

- Harris ratio test without EXPAND — 3 better, 6 worse, `blend` stopped solving
- dual steepest edge inside branch and bound — nodes re-optimise in ~3 pivots, the extra FTRAN never pays

---

## 6. MILP

| instance | gap | nodes |
|---|---|---|
| flugpl | 0.000% | 29,927 |
| gt2 | 0.000% | 7,961 |
| khb05250 | 0.000% | 3,561 |
| neos5 | 0.000% | 3,314 |
| p0201 | 0.000% | 2,853 |
| gen-ip054 | 0.330% | 304,799 |
| mas76 | 0.562% | 128,547 |

Warm starts on 18,772 of 18,775 relaxations, 2.9 simplex iterations per node.

Gomory cuts **off** by default — better root bounds on all seven, worse tree on
three:

| | no gomory | with gomory |
|---|---|---|
| gen-ip054 | 0.528% | 0.847% |
| gt2 | 5.556% | 11.112% |
| mas76 | 0.562% | 1.593% |

```bash
build/sankhya milp data/miplib/gt2.mps
```

---

## 7. QP

21 of the 24 smallest Maros–Meszaros instances. Direct sparse LDL' is 1.52×
over conjugate gradient. Polishing accepted on 16 of 24, error at least halved
on 15, worsened on none (`dualc5` 3.3e-05 → 7.6e-09; `hs118` primal
9.0e-05 → 5.0e-21).

Fill budget sweep (33/40 correct at every setting):

| budget | direct solves | total |
|---|---|---|
| 20× | 28 | 62.15 s |
| **50×** | 34 | **58.86 s** |
| 200× | 38 | 60.01 s |
| 1000× | 38 | 60.25 s |

Raising the budget puts more instances on the direct path and makes the set
*slower*, which is why AMD ordering is deprioritised: what it would fix is
exactly what gains nothing.

```bash
build/sankhya qp data/maros/HS21.QPS
```

---

## 8. GPU (Tesla T4)

Solve-time speedup, same algorithm, CPU against GPU:

| instance | speedup |
|---|---|
| supportcase10 | **7.09×** |
| graph40-40 | **3.00×** |
| datt256_lp | **2.82×** |
| qap15 | **2.70×** |

Accuracy 1.9e-08 to 0.0 against the CPU answer. `spmv K x` went 0.207 s → 0.042 s
from the adaptive vector width.

**Setup time is not in these numbers and is identical on both sides** — it is
MPS parsing, which is serial. Reporting end-to-end wall clock instead would be
Amdahl's law quietly eating the result.

```bash
bash scripts/gpu_test.sh          # on a CUDA box; see docs/KAGGLE.md
bash scripts/check_cuda_syntax.sh # type-check the kernels with no GPU
```

**Not yet verified on hardware:** the dual-iterate upload fix, sparse LDL',
direct QP solve, polishing, the dual simplex fix, and the cuPDLPx additions have
all landed since the last T4 run. Step 3 of `gpu_test.sh` (the backend contract
tests) is the gating check.

---

## 9. Tests

```bash
ctest --test-dir build --output-on-failure
```

Eleven suites, all passing: `sparse`, `mps`, `standard_form`, `scaling`, `pdhg`,
`backend`, `cuts`, `lu`, `simplex`, `presolve`, `ldl`.

Three of them exist because of a specific bug and are worth understanding:

- `test_cuts` separates at **simplex vertices** as well as random interior
  points. 426 random-point separations missed the cover-cut sign bug; the first
  vertex caught it.
- `test_ldl` builds K, picks x, forms b = Kx, solves and compares — it never
  checks a residual the factorisation computed about itself.
- `test_presolve` gives each column a **distinct** reduced cost when checking
  dual postsolve. The three older dual tests all passed an all-zero vector,
  which is the one input that cannot tell two different indexings apart.
