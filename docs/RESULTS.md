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
| Simplex | ~55 | 78/88 Netlib correct and none wrong, and it now takes a basis from the first-order method. Still no Forrest–Tomlin, no bound-flipping ratio test, no hypersparsity |
| Presolve | ~38 | 10 reductions; HiGHS/PaPILO have roughly 25 |
| Infrastructure | ~70 | good tests, no CI, no packaging |
| **MILP** | **~40** | **still the weak leg — see below.** Time limits are now actually enforced, a cut round has to pay for itself before it starts, node selection is best-estimate with plunging rather than depth first, and RINS is in. Missing: conflict analysis, restarts, clique tables, symmetry detection, node presolve, a real cut pool |

Weighted, that is about **62/100** as of 2026-09-01, up from 50. Most of that
move is the simplex — six wrong answers is a worse problem than any slow one, and
fixing them is worth more than the speed that came with it. The MILP part of it
is smaller than the ten points look: roughly half is a solver that had stopped
wasting its own budget rather than one that could do anything new.

**MILP is the honest gap.** What is missing: node presolve, symmetry detection,
local branching, conflict analysis, restarts, clique tables, and cut management
with a pool rather than one shared matrix. On the wider MIPLIB set at a fifteen
second limit, **twenty-three of seventy still end with no feasible solution at
all**, and several of those never get past the root because the root relaxation
itself is too slow — which is the simplex's problem, not the tree's. Real
refinery scheduling is MILP, so this is both the weakest component and the one
the problem statement cares most about. Do not let a demo imply otherwise.

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

### The wider set says something the seven do not

Everything above this line was measured on seven MIPLIB instances, and every
branch-and-bound constant in the code was fitted against them. The wider set —
70 instances from `data/miplib/` with a published optimum, fifteen second limit
— is not the same picture:

| | before | after |
|---|---|---|
| ended with a feasible solution | 41 | **45** |
| proved optimality | 5 | **7** |

Six instances newly found a solution, two lost it, nineteen ended closer to the
published optimum and six further. Every instance claimed optimal matches the
published value, on both sides.

That is one run, and one run is not the number. The same comparison was made five
times over the course of this work:

| | before | after |
|---|---|---|
| ended with a feasible solution | 38, 41, 41, 42, 43 | 45, 45, 46, 46, 47 |
| proved optimality | 5 every time | 6, 6, 6, 7, and 4 |

The before column moves because the build it measures does not enforce its own
time limit, so how much work it gets done in fifteen seconds depends on what else
the machine is doing. The after column moves less. Which instances sit on the
feasibility boundary is not stable between runs; the separation between the two
columns is.

The 4 in that last row is not noise and is worth keeping: it is the run taken
with root cut filtering switched off, which costs the proofs of `gt2` and `fiber`
outright. See the end of this section for why that switch went off and came back.

The final figure — 7 proved — is from the run after the propagation tolerance fix
below, which is the only configuration that is both correct on the refinery MILP
and current. Earlier runs in the table were taken before that fix. Their
claimed-optimal check passed on both sides every time, because MIPLIB's 0/±1
instances never reach the noise floor where the bug lives.

**Twenty-nine of the seventy ended with no feasible solution at all** — not a
poor gap, nothing. A tree with no incumbent has nothing to prune against, so it
explores blind and the bound never moves. Thirteen of those never got past the
root node.

Both binaries are run back to back on each instance, so load hits both sides of a
pair equally. That is not fastidiousness: this repository has twice recorded a
regression that did not exist because a measurement was taken under load, and the
machine these were taken on was shared with other work.

```bash
python3 bench/milp_ab.py old-binary new-binary 15
python3 bench/miplib_survey.py 15
```

### A run that found nothing reported an objective of zero

`BranchAndBoundResult::objective` was left at its default on every exit path but
the infeasible one, so a solve that hit the time limit with no incumbent came
back reporting zero.

On `acc-tight4`, whose published optimum is 0, the survey scored that as a
**0.000% error** — an exact answer, from a run that never found a feasible point.
On `10teams` it read as 100%, which is at least visibly wrong. Both are the same
missing line. It now reports infinity by sense, which the CLI prints as JSON
`null`, and the harnesses count incumbents rather than objectives.

### The time limit was read once and spent three times

`solve_node` computed its remaining budget at the top, then ran a first-order seed
that may take a fifth of what is left, then a warm simplex solve entitled to the
whole limit, then — when the seeded basis did not work out — a cold retry handed a
copy of the same stale number.

At a fifteen second limit: `acc-tight4` took 31.3 s, `cvs16r128-89` 32.1 s,
`10teams` 28.8 s. The budget is now read immediately before each solve. The
comment above the fix already said *"a solver ignoring a time limit"* — the defect
had survived one layer down.

### Cuts that cost more than they buy

Over the 29 instances that ended with nothing, turning root cuts off outright took
that from **1 finding something to 4**. Cutting the cut loop's time share to a
tenth changed nothing: still 1. So the cost was never the slice.

A cut round is not one LP. It separates, appends, and re-solves to check the bound
did not fall, and the loop ends with a third solve to decide whether any of it
paid. `newdano`'s root relaxation takes about a second; twenty cuts make the same
relaxation take five, and three of those are the entire budget. It explored **zero
nodes**, ran no heuristic, and ended with nothing — and finds a solution in eight
nodes with cuts off.

Three changes, none of which turn cuts off: a round is started only if two more
solves of what the last one cost still fit; the check that follows gets a few
times what the uncut root cost and the round is rolled back when it does not
finish; and cuts the root optimum does not sit on are dropped afterwards, which
leaves the root bound unchanged by construction because a slack constraint carries
a zero multiplier.

`neos-3046615-murg` is the sharpest case. 105 cuts take its root bound from 192 to
288, and then the relaxation carrying them does not converge, so the solver
reported "relaxation failed" after **0.104 seconds of a fifteen second budget** —
having spent all of it on a bound for a tree it then declined to search. A root
that will not solve with cuts now drops them and tries once more, and it goes from
nothing at all to **33 incumbents over 29,570 nodes**.

### Node selection was depth first, which ignores the bound

Depth first is right inside a plunge — a child differs from its parent in one
bound, so the parent's basis re-optimises it in a few pivots — and wrong between
plunges. The bound this solver reports is the smallest bound over its open nodes,
and under depth first the open list always contains the root's own second child,
sitting there until its sibling's entire subtree is finished. The reported bound
is therefore the root bound for almost the whole run, and optimality can only be
proved by exhausting the tree, never by the bound meeting the incumbent.

The replacement is Forrest, Hirst and Tomlin's estimate of the best integer
objective obtainable below a node,

$$e = z_{\text{node}} + \sum_j \min\left(P_j^-\,(x_j - \lfloor x_j \rfloor),\; P_j^+\,(\lceil x_j \rceil - x_j)\right)$$

over the columns still fractional there, priced by the pseudocosts the tree
already keeps. Plunge, and when the plunge ends jump to the best-estimated open
node. That is SCIP's default node selector. The estimate is a guess and never a
bound; nothing prunes on it.

| instance | depth first | best estimate |
|---|---|---|
| gt2 | 982 nodes | **307** |
| p0201 | 3,364 | **989** |
| khb05250 | 72 | **52** |
| fiber | 1,139 | **1,087** |
| flugpl | 246 | 262 |

And on the two that do not finish, where the point is the bound and not the tree:

| instance | gap, depth first | gap, best estimate | dual bound |
|---|---|---|---|
| mas76 | 4.470% | **2.418%** | 38,893.90 → 39,037.85 |
| gen-ip054 | 2.292% | **1.294%** | 6,765.21 → 6,772.53 |

```bash
build/sankhya milp data/miplib/gt2.mps --depth-first
build/sankhya milp data/miplib/gt2.mps
```

### RINS — the first heuristic here that improves a solution

Every other heuristic in this tree is a *start* heuristic: rounding, fix-and-
propagate, the feasibility pump. They find a first solution and have nothing to
say afterwards. That is why instances which found something early and then
wandered ended where they did.

RINS works from two points at once — the incumbent, integral and feasible, and the
node relaxation, neither but optimal for a problem that contains the answer. Where
those two agree on an integer column, both a good solution and the best available
bound say the value is right, so it is fixed; what remains is a small MIP over the
columns they disagree about, solved with a node limit as a problem in its own
right. Danna, Rothberg and Le Pape, *Math. Prog.* 102 (2005).

| instance | without RINS | with RINS | sub-MIPs, improved |
|---|---|---|---|
| aflow30a | 318.221% | **21.589%** | 3, 2 |
| mik-250-20-75-5 | 700.138% | **0.050%** | 4, 2 |
| r50x360 | 41.440% | **18.512%** | 5, 4 |
| ran13x13 | 9.041% | **4.643%** | 16, 3 |
| nexp-50-20-1-1 | 13.793% | **3.448%** | 2, 1 |
| beasleyC2 | 81.250% | 81.250% | 1, 0 |

Four of those end better than they did before any of this work, not merely better
than they had become. Rothberg's explanation is the one that fits: fixing
variables does not only make the problem smaller, it changes what the problem is,
and resolving a few key decisions can decompose the rest.

### Objective integrality

When the objective touches no continuous column and every coefficient it does
touch is a whole number, every feasible objective value is a whole number — so a
node whose bound is anywhere above `incumbent - 1` is dead, and the prune arrives
up to a full unit earlier. It is exact rather than a tolerance: there is no
version of it that discards a better answer.

What it needs is for a unit to be larger than the noise. A node bound is an LP
objective carrying about 1e-9 of relative rounding, so on a model whose objective
runs to 1e9 the slack required is itself about a unit, and the rule turns itself
off rather than claim what it cannot measure. Same lesson as the two absolute
tolerances in this file that each cost an answer.

Checked with the debug-solution tracker, which is now reachable from the command
line rather than only from C++. On `flugpl`, `gt2`, `khb05250`, `p0201` and
`fiber` — the instance that once returned a proved optimum 60.8% wrong — no prune
discards the known optimum, no kept cut is invalid, and all five reach the
published value.

```bash
build/sankhya milp data/miplib/fiber.mps --no-objective-integrality --solution=fiber.sol
build/sankhya milp data/miplib/fiber.mps --debug-solution=fiber.sol
```

### The refinery MILP, and the wrong answer it caught

Every MILP number above is against MIPLIB. The problem statement is about
refinery scheduling, and `data/refinery/refinery.mps` — the model named for it —
**has no integer columns at all**. The refinery MILP is a separate `--milp`
invocation of the same generator:

```bash
python3 scripts/refinery_model.py --periods=4 --crudes=4 --milp     --out=data/refinery/small_milp.mps
build/sankhya milp data/refinery/small_milp.mps --time-limit=60
```

276 rows, 760 columns, 24 integer and 8 binary. HiGHS 1.15.1 at a forced 0% gap
proves its optimum at 7,246,146,141.83 in **26 nodes**;
`data/reference/refinery.csv` records it.

This tree returned **7,156,892,194.84 and called it optimal** — 1.23% low on a
maximisation, with a matching dual bound and no violation anywhere.

The cause was an absolute tolerance. `propagate_bounds` tested row activities
against `q - 1e-9`, and this model's activities reach 1e9 because it is stated in
barrels and rupees: roughly 1e-7 of rounding per term in double precision, a few
hundred terms, and a row that is exactly tight computes as violated. The child
was declared infeasible by interval arithmetic and the subtree holding the
optimum was discarded without its relaxation ever being solved.

`provably_infeasible`, thirty lines below in the same file, already scaled its
tolerances, and its comment says *"the two tests have to agree about what an empty
box is"* and *"an absolute tolerance on a quantity whose scale the caller chooses
is always this bug waiting"*. That one was fixed after `fiber`. This one was not.

**MIPLIB could not have caught it.** Those instances are 0/±1 and their
activities never come near the noise floor. Only a model stated in real
engineering units does — which is the argument for having one.

After the fix: 7,246,146,141.826, matching the published value to six decimal
places of relative error, in 167 nodes. It does not prove optimality inside sixty
seconds where HiGHS needs 26, so the honest problem is now a slow answer rather
than a wrong one.

Found by the debug-solution tracker in one command: solve once with a prune
disabled to get a point, hand it back with `--debug-solution`, and every prune
reports whether it discards it. It printed *"propagation declared the child
infeasible"* on the first run. That is the second time in this repository it has
named a bug that manual elimination had not found.

### MILP presolve, re-measured

`command_milp` has presolve off by default, and the numbers that justified it —
`gt2` 500 nodes to 8,320 — were taken under depth-first search. Under
best-estimate node selection the blow-up does not happen:

| instance | presolve off | on |
|---|---|---|
| flugpl | 274 | 262 |
| gt2 | 307 | 399 |
| khb05250 | 55 | 158 |
| p0201 | 989 | **358** |
| small_milp | 167 | 157 |

Two better, two worse, one neutral. The honest statement is that MILP presolve
**was a net loss under depth-first search and is roughly neutral under
best-estimate**, not that the tree does not want it. It stays off by default
because roughly neutral is not a reason to turn something on, and because these
five instances are not a sample.

Depth-first on the same binary gives `gt2` 847 → 1,388, which is what says the
difference is the search order rather than the reduction.

### Two things that were built and do not pay

**LP-guided diving.** A dive that re-solves the relaxation after every decision
rather than reading all of them off one, which is what the fix-and-propagate dive
does. It is affordable — a dive step bounds one column, which is a branching
child, so the dual simplex re-optimises it in a handful of pivots — and on the 28
instances that end with nothing, which is the population it was built for, it
finds the *same five* whether it is on or off. Not five different ones. The dives
run and go deep before they die (22 probe solves per dive on `haprp`, 135 on
`neos2`), so what they lack is not budget: a dive that fixes forwards with a
single-level backtrack cannot recover from a decision made twenty steps earlier.
Off, kept, switchable. What would make it work is several dives with different
rules, which is how the family earns its place in SCIP; one rule is not the
family.

**Root cut filtering, nearly.** Dropping cuts the root optimum does not sit on is
free on the root bound — a slack constraint carries a zero multiplier — and costs
the tree below, where a cut slack at the root can bind once a variable is pinned.
Measured on the eleven instances a comparison had flagged as worse, it looked
clearly bad: six worse, three better. It was switched off on that evidence and
switched back on when the whole set was counted, because those eleven were the
instances *selected for having got worse* — a sample chosen by the outcome cannot
decide the outcome. On the whole set it is what proves `gt2` and `fiber`, and
without it `gt2` does not merely lose the proof, it returns a 5.556% answer
instead of the exact one.

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
`pdhg`, `backend`, `cuts`, `branch_and_bound`, `lu`, `simplex`, `crossover`,
`presolve`, `ldl`.

Four of them exist because of a specific bug or a specific silence, and are worth
understanding:

- `test_branch_and_bound` enumerates every integer point of small models and
  checks what the tree *proves* against that, under all sixteen combinations of
  the switches that discard nodes. The tree is the component this repository
  rates lowest and the one that has twice returned a proved optimum that was
  wrong, and it had no test at all. Checked that it can fail rather than assumed:
  changing the integral-objective cutoff from one unit to two makes nine cases
  fail and names the instance.

- `test_cuts` separates at **simplex vertices** as well as random interior
  points. 426 random-point separations missed the cover-cut sign bug; the first
  vertex caught it.
- `test_ldl` builds K, picks x, forms b = Kx, solves and compares — it never
  checks a residual the factorisation computed about itself.
- `test_presolve` gives each column a **distinct** reduced cost when checking
  dual postsolve. The three older dual tests all passed an all-zero vector,
  which is the one input that cannot tell two different indexings apart.
