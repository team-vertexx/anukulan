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
| First-order LP | ~64 | strongest piece; multithreading is no longer missing, though §10 measures the gain at 1.41x on five threads and shows the memory bus is what caps it. Some cuPDLPx tuning still missing |
| GPU | ~55 | real 2.7–7× measured, but not cuPDLP-C level |
| QP | ~50 | OSQP's core is here; no AMD ordering, thin regularisation strategy |
| Simplex | ~55 | 78/88 Netlib correct and none wrong, and it now takes a basis from the first-order method. Still no Forrest–Tomlin, no bound-flipping ratio test, no hypersparsity |
| Presolve | ~38 | 10 reductions; HiGHS/PaPILO have roughly 25 |
| Infrastructure | ~70 | good tests, no CI, no packaging |
| **MILP** | **~30** | **the weak leg — see below.** Time limits are now enforced, cuts get a slice of the budget rather than all of it, there is a feasibility pump, and the root relaxation gets a crossover basis |

Weighted, that is about **60/100** as of 2026-08-30, up from 50. The move is
mostly the simplex — six wrong answers is a worse problem than any slow one, and
fixing them is worth more than the speed that came with it.

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

**Coefficient tightening is implemented and off**, which is the opposite outcome
and worth recording for the same reason. Across the seven MIPLIB instances it
fires three times, all on `p0201`, and those three move its root LP bound from
6875.00000004 to 6875.00000002 — noise. A reduction that fires on one instance
in seven and moves no bound does not belong in the default path.

It is kept rather than deleted because it is correct and the shape is right;
what it lacks is coverage. As written it handles only a positive coefficient in
the `≤` reading, skipping the mirror case, and stops after one tightening per
row. Fixing both might double three hits, and six hits that move nothing is
still nothing — which is why that was not done.

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

**All 88 Netlib instances with published optima, cold, no presolve, 60 s each:**

| | count |
|---|---|
| reach the published optimum | **78** |
| return a wrong answer | **0** |
| do not finish (time, iteration limit, or numerical error) | 10 |

```bash
python3 -u bench/verify_simplex.py 60
```

This replaces a "16 of 16" that stood here until 2026-08-30. That number was
true of a sixteen-instance subset and it was hiding six wrong answers on the
rest — three reporting optimal at a point missing the rows by up to 1.8e+09,
two calling a bounded model unbounded, one simply wrong:

| instance | reported | true |
|---|---|---|
| grow15 | −205,842,493 | −106,870,941 |
| grow22 | −68,986,355,650 | −160,834,336 |
| maros | −102,064.67 | −58,063.74 |
| modszk1 | "unbounded" | 320.61972906 |
| scsd1 | "unbounded" | 8.6666666743 |
| cycle | −30.888 | −5.2263930249 |

One cause: the ratio test broke ties by row, and under Bland's rule by index,
but never by the size of the pivot — so a candidate winning the ratio by 1e-12
and losing the pivot by six orders of magnitude was taken, and the basis update
divided by it. Two guards went in behind the fix: the simplex recomputes `Ax`
from the matrix before reporting an optimum rather than trusting the basis that
produced `x`, and the CLI repeats that check against the original model after
postsolve. `cycle`, `modszk1` and `scsd8` now fail visibly rather than quietly.

The same run flagged eight further disagreements that turned out to be errors in
**our own reference table** — HiGHS returns what this solver returns on all
eight, and `e226`'s stored value was off by exactly 7.113, that instance's
objective constant. Those are corrected in `data/reference/netlib.csv` with a
note saying why.

### Crossover from a first-order point

Solve loosely with the first-order method, read off which columns that point
wants basic, and start the simplex from that basis instead of all logicals.

| | |
|---|---|
| simplex pivots over the set | **0.52×** |
| degen3 | 89,640 → 25,027 |
| d2q06c | 25,012 → 3,952 |
| czprob | 5,261 → 1,130 |
| handed its own optimum | 0 iterations |

The status column matters more than the ratio. Five instances that returned no
answer at all now return the right one: `degen3` and `stocfor2` (time limit),
`scsd8` (iteration limit), `wood1p` (numerical error), `modszk1` (a wrong
"unbounded").

```bash
build/sankhya simplex data/netlib/degen3.mps --crossover
python3 -u bench/crossover_sweep.py 1e-4
```

If the seeded solve does not reach an optimum, the cold solve is run and that is
the answer — crossover can save pivots or do nothing, but it cannot cost one.

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

(Sixteen here is the ablation's own set, not a claim about the whole of Netlib —
the number for that is at the top of this section.)

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

### Hyper-sparsity, surveyed

Hall and McKinnon call an instance hyper-sparse when more than 60% of FTRAN and
BTRAN results have a density under 10%, and report a mean 5.2× speedup on the
ones that are. The solver counts this, so it can be checked rather than assumed:

| | ftran <10% | btran <10% | combined |
|---|---|---|---|
| czprob | 99.6% | 43.2% | **75.6%** |
| 80bau3b | 80.2% | 65.0% | **72.9%** |
| stocfor2 | 50.9% | 85.7% | **68.6%** |
| fit1p | 44.5% | 72.6% | **61.8%** |
| 25fv47 | 10.5% | 17.1% | 14.1% |
| degen3 | 6.1% | 10.5% | 8.5% |
| pilot87 | 3.4% | 4.7% | 4.1% |
| **21 instances** | **13.1%** | **17.5%** | **15.4%** |

Four of twenty-one clear the threshold. Their 5.2× came from a test set chosen
for the property; this one is not that set, and the exploitation techniques are
not the right next work here. See [ROADMAP](ROADMAP.md#4-simplex--the-ceiling-is-lower-than-the-literature-suggests).

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

**35 of the 40 smallest Maros–Meszaros instances.** The five that fail are
`PRIMALC1`, `PRIMALC2`, `PRIMALC5`, `PRIMALC8` and `QPCBOEI2` — four of them one
family, whose `DUALC` counterparts all solve.

The adaptive rho update is both why the other 35 work and why those four do not:

| | optimal | time |
|---|---|---|
| adaptive rho | **35/40** | 33.8 s |
| rho held fixed | 28/40 | 87.3 s |

Fixed rho solves `PRIMALC1` and `PRIMALC8` exactly — to the published optima —
and loses nine others. On `PRIMALC1` the rule takes rho from 1e-01 to 1.9e-05
within five hundred iterations and leaves it near 2e-03; the primal residual is
still 74 at iteration 16,500, where a fixed rho converges in 7,000.

The mechanism is that the threshold gate which stops rho thrashing also stops it
recovering — once the primal residual has grown to match the dual one the ratio
sits near one and nothing fires. **Two fixes that follow from that were tried and
neither worked**: limiting rho's drift from its starting value made things
monotonically worse (35/40 unlimited, 33 at a factor of 100, 30 at 10), and
relaxing the gate after a long stretch without an update changed nothing at all.
Both are recorded next to `adaptive_rho` in `qp.hpp`. What is untried is OSQP's
own normalisation, which divides the residuals by the scale of the terms that
make them up rather than by the tolerances. Direct sparse LDL' is 1.52×
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

Thirteen suites, all passing: `sparse`, `mps`, `standard_form`, `scaling`,
`pdhg`, `backend`, `cuts`, `lu`, `simplex`, `crossover`, `presolve`, `ldl`,
`threading`.

Three of them exist because of a specific bug and are worth understanding:

- `test_cuts` separates at **simplex vertices** as well as random interior
  points. 426 random-point separations missed the cover-cut sign bug; the first
  vertex caught it.
- `test_ldl` builds K, picks x, forms b = Kx, solves and compares — it never
  checks a residual the factorisation computed about itself.
- `test_presolve` gives each column a **distinct** reduced cost when checking
  dual postsolve. The three older dual tests all passed an all-zero vector,
  which is the one input that cannot tell two different indexings apart.
- `test_threading` asserts **bit equality** between the threaded and serial
  backends, not closeness. A tolerance of 1e-13 would pass just as happily on a
  backend that summed in a different order, and a different sum changes the
  iteration count. It runs at thread counts past this machine's core count,
  because a pool that only behaves when it fits will misbehave elsewhere.

---

## 10. Threads

**What was threaded: the first-order method, and nothing else yet.** The
simplex and the branch and bound tree are still serial, and sections 10.7 and
10.8 are the measurements that say why rather than an apology for it.

Every number in this section was taken on the machine at the top of this
document — an M4 with **four performance cores and six efficiency ones**. That
split is not a footnote; it is most of what the numbers below are about, and a
reader on a homogeneous machine should expect a different shape.

**On the conditions.** This box was shared with other work for the whole
period these were measured, and a thread-scaling measurement is the one kind
that a second busy process does not merely add noise to but can invert — at ten
threads there is no spare core to absorb it. Three defences are in the harness
and none of them is a substitute for a quiet machine: every configuration is
run three times and the **minimum** is kept, the thread counts are visited in a
rotated order on each pass so one burst of interference cannot land entirely on
one row, and the load average is recorded at the start and end of every run and
printed with the results. Where the load was high enough to matter it is said
so against the number.

### 10.1 The answers do not change

This is the first claim because everything else is worthless without it.

`threaded_cpu_backend` returns **bit-identical** results to `cpu_backend()` at
every thread count. Not "reproducible at a fixed thread count" — which is the
shape of what production solvers offer, since Gurobi's guarantee is the same
results from the same *parameters* and the thread count is one of them — and not
"agrees to 1e-13". Identical bits. `docs/ARCHITECTURE.md` §7a has the argument
and §10.8 the comparison; this is the check.

Three checks, in increasing breadth.

**Byte-for-byte on the solution files.** Six Netlib instances solved at two,
four, six and eight threads, each compared against `--backend=cpu`:

```bash
build/sankhya solve data/netlib/25fv47.mps --tol=1e-8 --presolve \
    --backend=cpu --solution=/tmp/serial
build/sankhya solve data/netlib/25fv47.mps --tol=1e-8 --presolve \
    --threads=6 --solution=/tmp/threaded
cmp /tmp/serial /tmp/threaded
```

Twenty-four comparisons, all identical. Not close — the same bytes.

**The whole Netlib set through the simplex.** `verify_simplex.py` with
`--presolve --crossover`, which reaches the threaded backend through the
crossover seed, run serially and at six threads:

| | serial | 6 threads |
|---|---|---|
| correct against the published optimum | **83/88** | **83/88** |
| wrong | **0** | **0** |
| did not finish | 5 | 5 |

The two output files are identical, down to which five instances did not finish
and why — `cycle` on a numerical error, `d6cube`, `degen3`, `dfl001` and
`pilot87` on the time limit. That the same five stopped in the same way under a
*time* limit, on a shared machine, is a stronger result than it looks.

**The whole Netlib set through the first-order method**, which is the path that
is actually threaded rather than one that merely touches it.
`bench/verify_parallel_solve.py` compares objective, iteration count and status
at one thread against six, over all 88 instances:

| | |
|---|---|
| identical objective, iterations and status | **88 / 88** |
| differing | **0** |

It compares under an **iteration** cap rather than a time limit, and that is not
a detail. A time limit stops the solve wherever the wall clock happens to be, so
the two runs would stop at different iterations under any load at all, and the
harness would report differences that are the clock's doing rather than the
backend's.

```bash
python3 bench/verify_parallel_solve.py 6 20000 1e-6
```

**One caveat, stated rather than buried.** `miplib_survey.py` is run under a
*time* limit, and a time-limited branch and bound explores as many nodes as it
gets through — so on a shared box the node counts and the remaining gaps move
between runs whatever the thread count does. Differences there are the time
limit interacting with load, not the parallel path, which in `milp` reaches
only the root crossover seed. The invariant that has to hold, and does, is the
last line of both runs: *every instance claimed optimal matches the published
optimum*.

### 10.2 The speedup curve

Eight instances, the largest available here — six from `data/lptestset/` and the
two biggest Netlib ones. Every configuration runs a **fixed iteration budget**,
calibrated per instance to about three seconds serially, so every thread count
does identical work and the comparison is of speed rather than of luck. Three
repeats, minimum kept, thread counts visited in a rotated order.

Geometric mean over the set, against the serial baseline measured in the same
run:

| threads | solve | end-to-end |
|---|---|---|
| 1 | 1.00 | 1.00 |
| 2 | 1.28 | 1.24 |
| 3 | 1.35 | 1.29 |
| 4 | 1.37 | 1.30 |
| **5** | **1.41** | **1.33** |
| 6 | 1.39 | 1.31 |
| 7 | 1.29 | 1.23 |
| 8 | 0.96 | 0.95 |
| 9 | 0.83 | 0.82 |
| 10 | 0.78 | 0.78 |

**It peaks at 1.41× on five threads and is a loss from eight.** Per instance, at
each one's own best count:

| instance | best speedup | at threads |
|---|---|---|
| maros-r7 | **2.06×** | 6 |
| qap15 | 1.49× | 6 |
| supportcase10 | 1.48× | 4 |
| datt256_lp | 1.46× | 5 |
| graph40-40 | 1.38× | 5 |
| cont1 | 1.25× | 3 |
| sgpf5y6 | 1.24× | 5 |
| watson_1 | 1.20× | 5 |

`bench/results/thread_scaling.txt` has the full grid, and the load average at
each end of it: 3.57 rising to 7.77, so the eight-and-above rows carry
contention as well as oversubscription and should be read as "this is where it
stops helping" rather than as precise figures.

```bash
python3 bench/thread_scaling.py data/lptestset/*.mps --extra=--no-polish
```

### 10.3 Why the numbers are in this range: the loop saturates the bus

A streaming triad — `c[i] = a[i] + 3*b[i]` over 384 MB arrays, far past any
cache — measures what this machine can move. Best of sixty interleaved trials:

| threads | 1 | 2 | 3 | **4** | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| GB/s | 37.4 | 71.7 | 77.9 | **88.5** | 85.4 | 78.1 | 80.9 |

**The machine's memory bandwidth saturates at four threads, and the solver's
speedup peaks at five.** That coincidence is the argument. Both curves stop
improving at the same place, several threads short of the core count, which is
what a memory-bound loop looks like — and it does not depend on any single
number in either table being exact.

**What I could not measure, and will not pretend to.** An earlier draft of this
section claimed the ceiling was exactly 1.32× and that the solver's 1.32× sat
precisely on it. That was wrong twice over: the solver number improved to 1.41×
once the pool defects in 10.6 were fixed, and the single-thread bandwidth figure
turned out not to be stable enough to divide by. Across seven runs it came back
as 34.9, 35.6, 37.4, 53.2, 58.2, 59.4 and 60.0 GB/s — a factor of 1.7 — because
one thread lands on a performance core or an efficiency core and the scheduler
decides which. The multi-thread figures repeat to within a few per cent; the
single-thread one does not. So the honest form of the ceiling is a range, **very
roughly 1.5× to 2.4×**, and the solver's 1.41× is at or a little under the
bottom of it rather than exactly on a line.

The conclusion that survives all of that: this component is limited by memory
rather than by arithmetic, on a machine whose bus is saturated by about four of
its ten cores. That is a property of the hardware, and no amount of better
threading moves it.

It is also what makes sense of the GPU numbers in section 8. A T4 moves several
times what this machine does, the method is bandwidth-bound, and the measured
2.70×–7.09× is roughly that ratio appearing where it should. CPU threads are not
a substitute for the device on this component.

**The spread between instances, one candidate tested and eliminated.**
Per-instance speedups run from 1.20× to 2.06×. Two explanations have now been
tried and neither survived, and both are recorded so they are not tried again.

*Cache residency.* It fitted until datt256_lp went from 1.17× to 1.46× on a pool
fix and broke the correlation. maros-r7 and qap15 have the same working set and
differ by a factor of 1.4.

*Row length of the sparser product.* A product whose rows are very short
degenerates into a stream over the matrix, and streams are what the bus caps —
so the shorter-rowed of `K` and `K'` should be the one that limits the solve.
Across the eight instances that correlates at **r = 0.957**:

| instance | K nnz/row | K' nnz/row | shorter | speedup |
|---|---|---|---|---|
| cont1 | 2.2 | 10.9 | 2.2 | 1.25 |
| sgpf5y6 | 3.4 | 2.7 | 2.7 | 1.24 |
| watson_1 | 5.2 | 2.7 | 2.7 | 1.20 |
| supportcase10 | 3.4 | 37.6 | 3.4 | 1.48 |
| graph40-40 | 3.5 | 12.3 | 3.5 | 1.38 |
| qap15 | 15.0 | 4.3 | 4.3 | 1.49 |
| datt256_lp | 135.8 | 5.7 | 5.7 | 1.46 |
| maros-r7 | 46.2 | 15.4 | 15.4 | **2.06** |

That is a good-looking correlation on eight points and it is **wrong**, which is
why the per-kernel profiler in 10.4 was worth building: it turns the idea into a
prediction inside a single instance, where everything else is held fixed. On
datt256_lp the two products differ in row length by a factor of 24 and their
scaling does track it — `K x` improves 2.34× from two threads to eight, `K' y`
only 1.17×. On supportcase10 the row lengths are **reversed**, so `K x` should
be the poor one. It is not: the two come out at 1.47× and 1.49×, within noise of
each other.

The controlled test refutes it, and a correlation on eight points that fails its
own prediction is a coincidence, not a mechanism. **The spread remains
unexplained.**

```bash
build/micro_triad 10 60
```

### 10.4 Where the time actually goes, kernel by kernel

`--profile` reported nothing on the CPU, so the question this work most needed
answering could only be guessed at from outside the solver. The threaded backend
now carries the same per-kernel timing the CUDA backend has, with one extra
column: whether a call actually spread, or fell below the size threshold and ran
on the calling thread.

datt256_lp, six threads, 300 iterations:

| kernel | share | µs each | threaded |
|---|---|---|---|
| `K x` | 29.8% | 1080 | yes |
| `K' y` | 25.1% | 904 | yes |
| `primal_step` | 15.3% | 563 | yes |
| **`dot` (serial)** | **10.0%** | 185 | **no** |
| `accumulate` | 7.7% | 95 | yes |
| `blend` | 7.4% | 93 | yes |
| `dual_step` | 2.1% | 79 | yes |
| `advance_kx` | 2.1% | 77 | yes |

Two things follow, and the first one corrects an assumption I had been carrying.

**The serial reductions are not what caps this.** Leaving `dot` on the calling
thread to keep the determinism guarantee costs 10% of the loop. Amdahl on 10%
serial permits about 3.5× at six threads, and the measured figure is 1.46× — so
the guarantee is not what is being paid for. **The parallel parts simply do not
scale**, and that is worth stating plainly because the obvious suspicion about a
backend that deliberately serialises its reductions is exactly the wrong one.

**What does not scale is the streaming.** From two threads to eight, `K x`
improves **2.34×** while `K' y` improves **1.17×** and `primal_step` **1.23×** —
and those last two are 40% of the run between them. Both are dominated by moving
bytes rather than by chasing indices, which is the same conclusion 10.3 reaches
from the triad, arrived at from the other direction.

**And the smallest kernels get worse the more threads they are given.** Over
this instance's dual vector of eleven thousand entries, `advance_kx` and
`dual_step` both degrade with thread count: the barrier is a larger share of the
work than the work is.

That reads like an argument for raising the size threshold, and
`bench/micro/kernel_threshold.cpp` — timing the fused primal step back to back
with a hot pool — reads like an argument for *lowering* it, putting the crossover
nearer 4,096 entries. Swept end to end, four instances at five threads, best of
three, both are wrong:

| element threshold | 4,096 | **8,192** | 32,768 | 131,072 |
|---|---|---|---|---|
| geomean solve seconds | 1.116 | **1.087** | 1.244 | 1.275 |

The value does not move. It is a measurement now rather than the guess it was,
and it is the **third** time in this section that kernel-level evidence has
pointed the wrong way against an end-to-end number — after the microbenchmark's
3.26× in 10.5 and the row-length correlation in 10.3 that failed its own
controlled prediction. The pattern is consistent enough to be worth stating as a
rule: on this component, measure the solver.

A profiled run is slower than a real one by two clock reads per call and its wall
clock is not a benchmark; the proportions are the point.

```bash
build/sankhya solve data/lptestset/datt256_lp.mps --no-polish \
    --max-iter=300 --threads=6 --profile
python3 bench/micro/kernel_scaling.py data/lptestset/datt256_lp.mps 300
```

### 10.5 What the kernel microbenchmark got wrong, and why it is kept

`bench/micro/spmv_scaling.cpp` measures `K*x` alone, run back to back. On
datt256_lp it reaches **3.26×** on seven threads. The solver, on the same
instance, gets **1.46×**.

The microbenchmark is not wrong about the kernel; it is wrong about the problem.
Repeating one product leaves the matrix in cache so it never pays the traffic
the real loop pays, and the real loop also runs elementwise steps over vectors
larger than cache and serial inner products between them.

It is kept, with this paragraph, because the gap is the most useful thing either
measurement produced — and because 3.26× nearly became the headline of this
section. A kernel benchmark is evidence about a kernel.

### 10.6 Cores of two different speeds, and a barrier that must not spin

This is an M4: **four performance cores and six efficiency ones**. Three things
follow, all measured, all now designed around, and two of them were defects that
made the threaded path slower than the serial one.

**Blocks are taken, not dealt.** Under an even static split every barrier waits
for whichever chunk landed on an efficiency core. On datt256_lp's `K*x` at six
threads, a static split reaches 2.30× and dynamic block-stealing reaches
**3.17×**.

**But the block count has to follow the work, not the thread count.** A block
costs an atomic increment, and past a point that is all it is doing. On 25fv47,
which has 10,400 nonzeros, the same comparison at six threads runs the other
way: static **3.30×**, dynamic **1.53×**. So the count is chosen from the
nonzero count, floored at one block per thread and capped at eight.

**Idle workers must sleep, and busy ones must spin long enough.** Both halves
were wrong in the first version and both showed up as the threaded path losing
to the serial one:

- A worker that spins forever burns a core. Passing `--threads=6` to `milp`,
  where the node LPs are small enough that the backend correctly runs them
  serially, left five workers spinning against the one doing the work: gt2
  measured **1.92 s serial against 2.47 s at six threads**, same 3,230 nodes.
  With a condition variable it is 1.89 against 1.88.
- A worker that sleeps too eagerly has to be woken for every kernel. The spin
  has to cover the serial part of the iteration — the inner products this
  backend deliberately does not thread. maros-r7 at four threads: **1.465 s at a
  spin limit of 1,000, 0.941 s at 100,000**. The idle case is flat across that
  sweep, because the spin happens once and then the worker sleeps until woken.
  The full table is in `threading.cpp` beside the constant.

The barrier itself, once both were fixed:

| threads | 2 | 4 | 6 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| µs | 0.14 | 0.50 | 0.58 | 0.68 | 0.85 | **17.8** |

The cliff at ten is oversubscription: with every core occupied the scheduler
takes a spinner off its core and everyone waits out its quantum. It is why
`--threads=0` asks for the performance core count rather than for every core —
that default previously returned nine, which 10.2 shows is a **loss**.

### 10.7 The simplex, and why it is still serial

Two measurements decided this, and the second one contradicts a claim this
repository has been making.

**Pricing is not where the time is.** The obvious thing to thread in a simplex
is the pricing pass, since it walks every nonbasic column. On `degen3` — the
slowest Netlib instance here — `compute_duals` is **2.5%** of samples. Threading
2.5% perfectly and paying a barrier for it is a loss, not a win.

**The time is in the factorisation, which does not thread.** A sampling profile
of `degen3 --presolve` puts **65.7%** of samples in `LuFactor::factorize`,
66.8% counting both of the symbols it compiles to.
Markowitz-ordered sparse LU with threshold pivoting is a sequential algorithm:
each elimination step chooses its pivot from what the previous step left behind.
Parallelising it is a research problem, not an afternoon.

This matters beyond the threading question. `ARCHITECTURE.md` §16.3 rejects
Forrest–Tomlin updates on the grounds that a profile of `degen3` put
`LuFactor::factorize` at **11 samples out of about 4,500** — 0.2%. That profile
and this one cannot both describe the same code. The rejection of Forrest–Tomlin
rests on the old number, so it should be re-argued against the new one rather
than left standing — and re-argued in the direction of *doing* it, since a
refactorisation every fifty pivots is 3,991 of them here and two thirds of the
run. `bench/results/profile_degen3.txt` is the profile.

**Re-taken after the ratio-test fix, and the conclusion survives it.** The first
version of this measurement was made before that fix, where `degen3` reached the
200,000 iteration limit in 145.5 s and refactorised 3,991 times — a run that
disagreed with §5 and made the profile alongside it hard to trust. On the fixed
base the instance solves: **optimal, 105,433 iterations, 108.1 s, 2,089
refactorisations**, and 105,433 is exactly what §5 documents. So the
disagreement was the bug, and it is gone.

What matters here is that the profile did not move with it. `factorize` was
66.4% before and is 65.7% now; the fix changed how many iterations the instance
takes, not where the time inside one goes.

### 10.8 The branch and bound tree, measured and not attempted

The prompt for this work called node parallelism the most valuable and the
hardest, and both halves check out.

**Strong branching probes are independent, and they help the wrong instances.**
Up to eight candidates a node, two bounded LP solves each, none of them
depending on another — the easiest thing in the tree to thread, and the tree
would not change shape, since a probe is pure information and the deltas can be
reduced in candidate order. But probing is capped at depth ten, so the share it
holds depends entirely on whether an instance is shallow or deep:

| instance | relaxations | probes | probes as a share | finishes? |
|---|---|---|---|---|
| 22433 | 260 | 218 | **83.8%** | yes, 1.9 s |
| beasleyC1 | 336 | 176 | 52.4% | no |
| khb05250 | 154 | 62 | 40.3% | yes, 0.8 s |
| p0201 | 1,665 | 556 | 33.4% | yes, 3.4 s |
| gt2 | 3,663 | 512 | 14.0% | yes, 2.7 s |
| aflow30a | 1,825 | 176 | 9.6% | no |
| binkar10_1 | 2,710 | 176 | 6.5% | no |
| **mas76** | **21,502** | **84** | **0.4%** | no |
| **gen-ip054** | **78,915** | **98** | **0.1%** | no |

**The instances where probing is most of the work are the instances that already
finish, and the ones that run out of time spend essentially nothing on it.**
Threading the probes would make `khb05250` finish in half of the 0.8 s it
already takes and would do nothing at all for `gen-ip054`. That is the wrong end
of the problem, and it is why this was measured rather than built.

Do not read the aggregate here as the answer: over these twelve instances probes
are 2.3% of relaxations, but that single number is just the two enormous trees
outvoting everything else, and it hides the split that makes the decision.

**The tree itself is where the win would be, and it is not small.** `gen-ip054`
explores 78,811 nodes in ten seconds. Those nodes are independent subtrees, and
unlike the first-order path they would actually scale: a node relaxation here is
a few dozen rows, so it sits in cache and never touches the memory bus that
caps §10.3 at 1.32×. The component that cannot be helped by threads and the
component that could be are opposite ones, which is the reverse of where
intuition points.

**What it would be worth is published.** Para-B&B ([arXiv:2604.09556]
(https://arxiv.org/abs/2604.09556)) is a deterministic parallel branch and bound
built on HiGHS, and over 80 MIPLIB 2017 instances it reports a **geometric mean
speedup of 2.17× on eight threads** with determinism preserved, reaching 5.12×
on the node-heavy instances — and still averaging a 34.7% thread idle rate even
with a learned load balancer. So the honest comparison is 2.17× for the tree
against the 1.41× in §10.2 for the first-order path. The tree is worth more, on
the same machine class, and by roughly the factor the reasoning above predicts.

It was not built here, for two reasons, and neither is that it would not pay.

The first is determinism. A parallel tree changes the order nodes are explored,
which changes when the incumbent is found, which changes what is pruned, which
changes the node count — so the guarantee §10.1 opens with cannot be had the
same way. What the established codes offer instead is narrower than it sounds.
Gurobi's guarantee is the same results "from the same inputs (model and
parameters)" — and the thread count *is* a parameter, so it does not span thread
counts. The same page lists `TimeLimit` as undermining determinism outright,
because wall-clock timing varies with machine load, which is exactly the effect
that made `miplib_survey.py` differ between the two runs in §10.1 and is worth
knowing is a documented property of production solvers rather than a defect
here. Para-B&B gets its determinism by replicating full solver state per worker
and removing non-deterministic synchronisation.

**So no major solver promises what §10.1 promises**, and the reason is not
cleverness on this side: the first-order path happens to admit bit-identity
because its parallel work is elementwise and row-disjoint, and a search tree
does not.

The second reason is scope. Doing it properly means a deterministic clock,
synchronisation points, race-free work stealing against a shared incumbent, and
per-thread node pools — inside `branch_and_bound.cpp`, which another work stream
owns. Weeks, not days, and half-doing it produces exactly the solver this
section exists to avoid: one whose node counts move between runs.
