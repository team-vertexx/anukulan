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

Dual fixing, across all 88 Netlib instances (the eighteen with published optima
understate it — only 13 columns go there):

| | off | on |
|---|---|---|
| rows removed | 19.27% | **19.45%** |
| columns removed | 11.32% | **11.61%** |
| nonzeros removed | 11.84% | **11.99%** |

520 columns on 22 of the 88 — `80bau3b` 9,195 → 9,037, `finnis` 525 → 461,
`czprob` and `bnl2` 64 and 63 apiece. It costs one early-exiting pass over the
columns.

The reduction itself is modest; what it also does is recover an instance. At
`--tol=1e-8` the count goes **75/88 to 76/88**, and the instances that stop
converging once presolved go from 3 to 2. Fixing a column removes it from every
row it was in, and the rows that get shorter are then reachable by the other
reductions.

Worth recording how nearly this was written off: the first four instances it was
checked on all reported zero, which reads exactly like a reduction that does not
fire. Four is not a sample.

Round trip over all 88 instances, presolved answer against plain, `--tol=1e-6`:

| | off | on |
|---|---|---|
| reached the published optimum | 77/88 | **78/88** |
| geomean wall-clock speedup | 1.52× | **1.63×** |

```bash
python3 bench/verify_presolve.py --tol=1e-6 --abs-tol=1e-6 --check-feasibility
python3 bench/verify_presolve.py --tol=1e-6 --abs-tol=1e-6 --extra=--presolve-no-doubletons
build/sankhya presolve data/netlib/80bau3b.mps --no-dual-fixing
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

### The primal weight controller

cuPDLPx does not publish its PID coefficients, so they were swept here. Sixteen
Netlib instances with published optima, presolved, `--tol=1e-8`, in total
iterations against `Kp = 0.5, Ki = Kd = 0` — which is the exponential smoothing
the controller replaces:

| Ki | | Kd | | Kp (with Kd = 0.2) | |
|---|---|---|---|---|---|
| 0.02 | 1.039 | 0.1 | 0.968 | 0.3 | 1.031 |
| 0.05 | 1.138 | 0.2 | 0.943 | 0.4 | 0.945 |
| 0.1 | 1.077 | **0.3** | **0.893** | 0.5 | 1.000 |
| 0.2 | 1.321 | 0.4 | 0.910 | 0.6 | 0.945 |
| | | 0.5 | 1.043 | 0.7 | 1.010 |
| | | 0.7 | 1.616 | | |

**The integral term hurts monotonically** and is left at zero. **The derivative
term has a clean minimum at 0.3** and turns hard on both sides of it — 0.7 is
worse than having no controller at all.

That shape is what an integral term would be expected to struggle with here. The
primal weight is not tracking a fixed setpoint; it is chasing a ratio that
legitimately moves as the solve progresses, so accumulated error is mostly stale
information. The derivative term, which responds to how fast the imbalance is
changing, has something real to act on.

`Kd = 0.3` is the default and is worth **10.7% of the iteration count**.

Not done: the sweep moved one coefficient at a time around `Kp = 0.5`. A proper
2-D sweep of (Kp, Kd) has not been run, and there is no reason to think the two
are independent.

```bash
python3 bench/pid_sweep.py
```

---

## 5. Simplex

Sixteen Netlib instances with published optima, **presolved** — which the
commands below now say, because they did not and the claim does not hold
without it. Plain, `woodw` and `stocfor2` both run into the 200,000 iteration
limit; presolved they take 1,782 and 3,668 iterations.

| algorithm | correct |
|---|---|
| primal | **16/16** |
| dual | **16/16** |

```bash
build/sankhya simplex data/netlib/sctap1.mps --presolve
build/sankhya simplex data/netlib/fit1p.mps --presolve --dual
```

### Incremental pricing

The primal simplex used to make two full passes over the matrix every
iteration. `compute_duals` recomputed every reduced cost from scratch, and the
Devex weight update then made a pass of exactly the same shape for the pivot row
`α_rj = ρ'a_j` — which is precisely what an incremental reduced-cost update
needs. The two are now one pass:

| | recompute | incremental |
|---|---|---|
| iterations | 154,739 | **126,502** (0.818×) |
| time | 99.1 s | **79.4 s** (0.801×) |
| correct | 16/16 | 16/16 |

`degen3` carries most of it — 133,657 iterations and 85.2 s become 105,433 and
66.2 s.

It runs in phase two only: phase one rebuilds its cost vector from the basis
infeasibilities every iteration, and nothing incremental survives a cost vector
that moves underneath it. It needs Devex, since that is what computes the pivot
row. And it is thrown away and recomputed at every refactorisation, at the phase
transition, and whenever a pivot fails — that last set is the drift control, and
it is why the iteration counts move at all rather than being identical.

```bash
build/sankhya simplex data/netlib/degen3.mps --presolve --no-incremental-pricing
build/sankhya simplex data/netlib/degen3.mps --presolve
```

Measured and **not** adopted, recorded so they are not retried:

- Harris ratio test without EXPAND — 3 better, 6 worse, `blend` stopped solving
- dual steepest edge inside branch and bound — nodes re-optimise in ~3 pivots, the extra FTRAN never pays

---

## 6. MILP

At the shipped defaults — node limit 1,000,000, 60 s here:

| instance | status | nodes | gap | error vs published |
|---|---|---|---|---|
| flugpl | **optimal** | 28,917 | 0.000% | 0.000% |
| gt2 | **optimal** | 1,167 | 0.000% | 0.000% |
| khb05250 | **optimal** | 4,247 | 0.000% | 0.000% |
| p0201 | **optimal** | 2,569 | 0.000% | 0.000% |
| mas76 | time limit | 220,148 | 2.778% | **0.000%** |
| neos5 | time limit | 9,937 | 13.333% | **0.000%** |
| gen-ip054 | time limit | 617,387 | 1.231% | 0.140% |

**Four of seven proved optimal; on the other three the solution is already
optimal or within 0.14%.** What is missing there is the proof, not the answer —
`neos5` and `mas76` both hold the published optimum and cannot show it.

That distinction was invisible until the node limit was fixed. It used to be
20,000, which stopped `flugpl` half a second before it would have proved
optimality, and the table then reported a 3.2% error on an instance the solver
can finish. Time is the resource that matters; a node limit that binds first is
measuring the limit rather than the solver.

`neos5` is the interesting one. Its bound does not move at all — 13.333% at
20,000 nodes and 13.333% at 200,000 — so more search is not the answer there.
That is a dual-bound problem, which means cuts, not heuristics or nodes.

Warm starts on 18,772 of 18,775 relaxations, 2.9 simplex iterations per node.

### Reliability branching

Strong branch a candidate until enough is known about it, then trust its
pseudocost — Achterberg, Koch and Martin's rule. Pure pseudocost branching has
to guess at a variable it has never branched on, and the guess is the same
optimistic constant for all of them, so the decisions near the root that shape
the whole tree are made with no information.

Six instances, 45 s, threshold 2, varying only how deep the probes are allowed:

| instance | off | d=3 | d=6 | **d=10** | no limit |
|---|---|---|---|---|---|
| flugpl | 477 | 246 | 246 | **246** | 246 |
| **gt2** | 783 | 1,509 | 374 | **194** | 2,225 |
| khb05250 | 142 | 79 | 76 | **86** | 78 |
| p0201 | 2,282 | 804 | 873 | **920** | 776 |
| gen-ip054 | **1.700%** | 1.714% | 2.192% | 2.192% | 2.047% |
| mas76 | **2.778%** | 3.565% | 3.565% | 4.099% | 4.047% |

**The depth cap is not a refinement here, it is the whole thing.** Unlimited,
strong branching makes gt2 nearly three times *worse* — 783 nodes to 2,225.
Capped at ten it is 194. Across the four instances that solve, 3,684 nodes
become **1,446**.

Near the root a branching decision shapes the whole tree and is worth paying to
get right. Deep down it settles a subtree about to be pruned anyway, the probes
are pure cost, and the greedy one-level-ahead choice is not the one that makes
the smallest tree.

**What it costs:** `gen-ip054` and `mas76` both end with worse gaps — 1.700% to
2.192% and 2.778% to 4.099%. Neither finishes either way, and the probes take
time those two would otherwise spend on nodes. On `mas76` the *solution* is the
published optimum in both cases: what gets worse is the proof, not the answer.

One dead end worth recording. gt2's unlimited-depth blow-up looked like an
implementation bug — an unconverged probe being scored as a zero bound change,
which would turn a good candidate into a rejected one. A fallback to the
pseudocost for unfinished probes changed nothing, because the probes were
converging. The information was right and the greedy decision on it was wrong,
which is a known property of strong branching and not a defect.

```bash
build/sankhya milp data/miplib/gt2.mps --no-reliability
build/sankhya milp data/miplib/gt2.mps --strong-depth=-1   # the 2,225-node version
```

### Node propagation

Branching pins a variable to one side of its fractional value, which is exactly
the moment interval arithmetic on the rows has something new to say. Propagating
into each child as it is created tightens the other variables, and sometimes
proves the child infeasible before its relaxation is ever solved.

The machinery is the same `propagate_bounds` the fix-and-propagate heuristic
already used; this only points it at the children. Six instances, 45 s budget —
nodes where it solves, remaining gap where it does not:

| instance | off | 1 round | 2 rounds | **4 rounds** |
|---|---|---|---|---|
| **flugpl** | 28,917 | 2,283 | 979 | **477** |
| gt2 | 1,181 | 826 | 789 | **783** |
| khb05250 | 143 | 143 | 142 | 142 |
| **p0201** | **6.076% — does not finish** | 2,364 | 2,212 | 2,282 |
| gen-ip054 | 1.700% | 1.700% | 1.700% | 1.700% |
| mas76 | 4.192% | 4.192% | 4.192% | 4.192% |

**No instance is worse for it and two are transformed.** flugpl needs a
sixty-first of the tree; p0201 stops failing to finish inside the budget. More
rounds keeps helping where it helps at all, and the only cost anywhere is
p0201's seventy extra nodes against flugpl's five hundred saved. Default 4.

An earlier run of this reported `gen-ip054` as *worse* under propagation —
1.700% against 1.284%. Re-measured on a quiet machine it is 1.700% at every
setting including off. That is the second false regression this session produced
by measuring three sixty-second solves per instance back to back, and both times
the solver was fine and the harness was not.

### Reduced-cost fixing

Once there is an incumbent, a node's own bound and reduced costs say how far a
nonbasic variable can move before the subtree stops being worth exploring. On
the seven instances, against the same run with it off:

| instance | status | nodes off → on | verdict |
|---|---|---|---|
| **gt2** | optimal | 7,901 → **1,167** | **0.15×** |
| p0201 | optimal | 2,743 → 2,569 | 0.94× |
| khb05250 | optimal | 3,525 → 4,247 | 1.20×, worse |
| gen-ip054 | node limit | — | **better incumbent**, 1.637% → 1.569% error |
| flugpl, neos5, mas76 | node limit | — | unchanged |

No instance returns a different answer where optimality is proved. gt2 fixes
7,322 variables outright and gets six times smaller for it; khb05250 loses 20%,
which is the honest cost of changing the order in which the tree is explored.

It costs one transpose product per node, which is small against a node solve,
and it gets stronger as the tree deepens and the gap closes.

```bash
build/sankhya milp data/miplib/gt2.mps --no-reduced-cost-fixing
build/sankhya milp data/miplib/gt2.mps
```

### Cuts, decided per instance

Whether a cut family pays is a property of the instance, not of the family.
Gomory takes `khb05250` from 4,247 nodes to 143 and makes four other instances
worse — no single on-or-off answer expresses that.

What separates the two groups is how far the cuts move the root bound, measured
as the relative rise in the standard-form root objective. It needs nothing the
solver does not already have; in particular it does not need the optimum:

| instance | root bound rise | tree with cuts | decision |
|---|---|---|---|
| gt2 | 55.1% | 1.01× | keep |
| khb05250 | 10.2% | **0.03×** | keep |
| p0201 | 4.7% | 0.88× | keep |
| neos5 | 1.3% | worse | drop |
| flugpl | 0.41% | 1.31× | drop |
| mas76 | 0.17% | worse | drop |
| gen-ip054 | 0.02% | worse | drop |

The three that gain are the three largest rises; the four that lose are the four
smallest; there is a factor of three between the classes. Threshold 0.02.

Against both fixed alternatives:

| instance | gomory off | gomory always | **adaptive** |
|---|---|---|---|
| flugpl | 28,917 | 37,953 | **28,917** |
| gt2 | 1,167 | 1,181 | 1,181 |
| khb05250 | 4,247 | **143** | **143** |
| p0201 | 2,569 | **2,263** | **2,263** |
| neos5 | 13.33% gap | 15.05% | **13.33%** |
| gen-ip054 | 1.28% gap | 1.73% | **1.28%** |
| mas76 | 2.78% gap | 3.60% | **2.78%** |

**It picks the better of the two on all seven.** The cost when it decides to
discard is the wasted cut generation — about 1.5% of nodes on mas76, which is
what trying costs.

Two things worth stating against this. Seven instances is a small sample to fit
a threshold on, and it could be overfitted; what argues otherwise is that the
mechanism is not a correlation — cuts that do not move the bound have made every
node more expensive and bought nothing. And the first run of this comparison
reported mas76 as *worse* under adaptive; re-running it alone gave 2.778% both
ways, twice. That first run was doing three sixty-second solves per instance
back to back, and a measurement taken under load is not a measurement.

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
