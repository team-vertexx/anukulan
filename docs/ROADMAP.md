# Roadmap

Where this goes next, what each step is worth, and the paper behind it.

The scoring is against **production solvers** (HiGHS, OSQP, cuPDLP), not against
a rubric of our own. Today that composite is about **50/100**. See
[RESULTS.md §1](RESULTS.md#1-where-this-actually-sits) for the breakdown.

---

## The honest ceiling

Taking every component as far as one person realistically can:

| component | now | reachable | effort |
|---|---|---|---|
| Reader | 95 | 98 | — |
| First-order LP | 62 | **85** | 3–4 weeks |
| GPU | 55 | **75** | 3–4 weeks (overlaps) |
| QP | 50 | 70 | 3 weeks |
| Presolve | 38 | 65 | 4–6 weeks |
| Simplex | 40 | 65 | 6–8 weeks |
| MILP | 22 | 52 | 2–3 months |
| Infrastructure | 70 | 90 | 2 weeks |

All of it lands around **73**, over roughly 9–12 months of focused work.

**80+ across the full scope is not reachable solo, and no paper shortcuts it.**
Simplex and MILP are what stop it: HiGHS's simplex is years of work by people
who do only that, and SCIP is 400k+ lines. Anyone claiming otherwise has not
looked at the size of those codebases.

**80+ on a narrowed scope is reachable.** Dropping the general-purpose framing
for *"GPU-accelerated first-order LP solver"* and weighting accordingly —
first-order 85, GPU 75, presolve 65, reader 98, infrastructure 90, QP 70 — comes
out at **80**, in about 4–5 months. For a portfolio, a narrow world-class thing
beats a broad average one, so this is not a consolation prize.

---

## Priority order

### 1. Finish the cuPDLPx work — first-order 62 → 85, GPU 55 → 75

Mostly landed. [cuPDLPx (arXiv:2507.14051)](https://arxiv.org/abs/2507.14051)
sits on the same restarted Halpern base this solver already had, and adds four
things: reflection, constant step size, a fixed-point restart criterion, and a
PID-controlled primal weight. All four are in; see
[RESULTS.md §4](RESULTS.md#4-first-order-lp).

Remaining:

- ~~settle the step scale~~ — done, 0.90, sweep recorded next to the constant.
- ~~sweep the PID coefficients~~ — done. The integral term hurts monotonically
  and stays at zero; the derivative term has a clean minimum at `Kd = 0.3` and
  turns hard on both sides. 10.7% fewer iterations. Both sweeps are in
  [RESULTS.md §4](RESULTS.md#4-first-order-lp).
- **still open on the primal weight:** the sweep was one coefficient at a time
  around `Kp = 0.5`. A proper 2-D sweep of (Kp, Kd) has not been run, and the
  two are unlikely to be independent.
- **non-kernel time.** On graph40-40 the kernels are 0.1386 s of a 0.67 s solve,
  so 79% is outside them. Making the KKT error lazy removed two host-side
  matrix products per termination check. The larger remaining item is computing
  the convergence residuals **on the device**: the primal side is free (`k_x`
  already holds `K·x̃` and the unscaled residual is `D1⁻¹` times the scaled one),
  and the dual side needs one device matvec to replace one host matvec.
- **verify on hardware.** Nothing since the last T4 run has been GPU-tested.

### 2. Presolve 38 → 65

- [Achterberg et al., *Presolve Reductions in Mixed Integer Programming*](https://dl.acm.org/doi/10.1287/ijoc.2018.0857)
  (INFORMS JoC) — the full reduction catalogue
- [*Presolving for GPU-Accelerated First-Order LP Solvers*](https://arxiv.org/pdf/2604.23951)
  — directly this combination

Missing reductions: dominated columns, probing, clique merging,
implied-integer detection, sparsify, SimplifyIneq, Stuffing.

**Parallel columns was measured and not built**, and the numbers are worth
keeping because the headline is misleading. Across the 88 Netlib instances,
31,954 of 159,369 columns are parallel to another — 20%, and all of them
continuous, which is the safe case. But merging needs the objective to follow
the same ratio as the coefficients, and that cuts it to **1,471**: `standata`
goes from 606 parallel to 12 mergeable, `shell` from 104 to 5.

1,471 columns is 0.9% of the set. Presolve carries 8% of the composite, so the
whole reduction is worth about a quarter of a point — against needing a new
postsolve entry kind that writes two columns from one merged value, which is
the most dangerous part of this codebase to extend. The doubleton substitution
put `stocfor2` 189 units outside the original model before its second bug was
found.

Worth doing eventually. Not worth doing before QP and the simplex, which carry
12% and 16% and have twenty and fifteen points of headroom each.

One structural note: the doubleton pass runs **after** the main loop and does not
re-enter it, because it is the only reduction that rewrites A and re-entering
would mean rebuilding the matrix each round to keep activity bounds honest. What
that leaves on the table has not been measured.

### 3. QP 50 → 70

AMD ordering for the LDL', a proper regularisation strategy, warm starting,
infeasibility detection. Note the fill-budget sweep in RESULTS before starting:
raising the budget makes the set *slower*, so AMD needs to be justified by
measurement rather than assumed.

### 4. Simplex — the ceiling is lower than the literature suggests

~~Simplex 40 → 65~~. The incremental pricing work is done (RESULTS §5) and took
it to about 50. **The rest of the way is not available on this benchmark**, and
that is measured rather than assumed.

**Hypersparsity does not apply here.** Hall and McKinnon call an instance
hyper-sparse when more than 60% of FTRAN and BTRAN results have a density under
10%, and report a mean 5.2× on the ones that are. Over 21 Netlib instances,
presolved, the combined rate here is **15.4%**, and only four instances clear
the threshold at all:

| | ftran <10% | btran <10% | combined |
|---|---|---|---|
| czprob | 99.6% | 43.2% | **75.6%** |
| 80bau3b | 80.2% | 65.0% | **72.9%** |
| stocfor2 | 50.9% | 85.7% | **68.6%** |
| fit1p | 44.5% | 72.6% | **61.8%** |
| pilot87 | 3.4% | 4.7% | 4.1% |
| degen3 | 6.1% | 10.5% | 8.5% |
| *21 instances* | *13.1%* | *17.5%* | *15.4%* |

Their 5.2× was measured on a test set selected for the property — KEN-18,
PDS-20, STOCFOR3, large network-structured problems. Netlib is mostly not that.
Six to eight weeks of work to help four instances in twenty-one.

**Forrest–Tomlin is aimed at something that is not hot.** A sampling profile of
`degen3`, the slowest instance in the set at 66 s, puts 67% of samples in one
spot and `LuFactor::factorize` at 11 samples out of ~4,500. Basis updates and
refactorisation are not where the time goes; the single pivot-row pass that
Devex needs is.

**The bound-flipping ratio test needs boxed columns**, and most instances here
have none: `pilot87` 36.8%, `80bau3b` 35.6%, `fit1p` 23.8%, `greenbea` 7.3%, and
**0%** for afiro, sctap1, degen3, 25fv47, woodw, stocfor2 and maros-r7.

So the realistic ceiling on this benchmark is nearer **55 than 65**. The gap to
HiGHS is real and these are the techniques that close it — a general solver
should have them — but implementing them would move the capability without
moving these numbers, and this document scores against measurements.

What would actually move the profile is **partial pricing**: the dominant cost
is now scanning every nonbasic column for the pivot row, and scanning a rotating
subset instead trades iteration count against per-iteration cost. Untried, and
the trade has to be measured rather than assumed.

> **Re-measured 31 Aug 2026, and the two paragraphs above no longer hold.**
>
> A fresh sampling profile of the same command — `simplex degen3 --presolve` —
> puts **66.4% of samples in `LuFactor::factorize`** (67.7% counting both
> symbols it compiles to), `pivot_row` at **2.1%** and `compute_duals` at
> **2.0%**. That is the reverse of the reading above, which had factorisation at
> 11 samples in 4,500 and the pricing pass as the dominant cost.
>
> Both conclusions drawn from the old profile therefore need re-arguing, and in
> the opposite direction. **Forrest–Tomlin is aimed at the hottest thing in the
> component**, not a cold one: the run refactorises 3,991 times, once every
> fifty pivots, and spends two thirds of itself there. And **partial pricing is
> aimed at 2%**, so it cannot move the profile whatever the trade turns out to
> be.
>
> The run behind the new number also disagrees with §5 of RESULTS: it reaches
> the 200,000 iteration limit in 145.5 s where that section documents 105,433
> iterations and 66.2 s. So one of the two profiles is describing code that no
> longer exists, and it is worth settling which before either technique is
> costed again. Evidence: `bench/results/profile_degen3.txt`. Measured on the
> `parallel` branch, forked from master at f6ef9d6 and so *before* the
> ratio-test fix on the simplex branch — if that fix changes degen3's iteration
> count, this needs taking again rather than believing.

### 5. MILP 22 → 52

The largest single piece of work, and the one the problem statement cares most
about.

The strategic finding from [Achterberg and Wunderling, *Mixed Integer
Programming: Analyzing 12 Years of Progress*](https://link.springer.com/chapter/10.1007/978-3-642-38189-8_18):
**cutting planes and presolve dominate every other component**, and both are
already here. What is missing are the smaller multipliers, which is better news
than it sounds.

> Caveat: the exact per-component ablation factors from that paper could not be
> retrieved — ResearchGate returns 403 and the Gurobi slide deck 404. The
> qualitative ranking is confirmed by secondary sources; the specific numbers
> are **not verified** and should not be quoted.

**What the seven-instance set actually says is needed.** Diagnosed by raising
the node limit and watching what moved:

- `flugpl` was only ever stopped by the node limit. Fixed.
- `mas76` and `neos5` **already hold the published optimum** and cannot prove it.
  More search does not help; `neos5`'s bound is identical at 20,000 and 200,000
  nodes. These need a stronger relaxation — cuts — not heuristics and not nodes.
- `gen-ip054` improves slowly with search: 2.621% gap at 20k, 1.231% at 617k.

So for this set, **primal heuristics are the wrong next thing** — the incumbents
are already optimal on two of the three unsolved instances. The bound is what is
missing. That may not generalise past seven instances, but it is what these say.

**Cut usefulness varies enormously per instance.** Gomory takes khb05250 from
4,247 nodes to 143 and makes four others worse. A single on/off flag cannot
express that. Deciding per instance — enable a cut family, measure whether the
root bound actually moved, keep it only if it did — is what production solvers
do and is the highest-value cut work here.

In order:

- **per-instance cut selection** — see above; the 30× on khb05250 is sitting there
- **node presolve** — bound tightening at each node
- **reliability branching** — pseudocost plus strong branching, replacing plain pseudocost
- **primal heuristics** — feasibility pump, RINS. See Berthold, *A computational
  study of primal heuristics inside an MIP solver*
- **conflict analysis** — Achterberg, *Conflict analysis in mixed integer programming*
- **restarts**, **symmetry detection**, **clique tables**

### 6. Infrastructure 70 → 90

CI on push, a fuzzer for the MPS reader, packaging, an installable API rather
than only a CLI.

---

## The benchmark to be measured against

[Mittelmann's LPfeas benchmark](https://plato.asu.edu/ftp/lpfeas.html) is where
cuPDLPx and HPR-LP — the two papers this solver's first-order method comes from
— are actually scored. The reference table is copied into
[data/reference/lpfeas_mittelmann.md](../data/reference/lpfeas_mittelmann.md) so
our runs have something external to be checked against.

Eight of its forty public instances are already in this repository. Running them
at `--tol=1e-6` and reporting solved-or-not against that table is a comparison a
reader can check, and it is worth more than any number computed against a rubric
we wrote ourselves. Wall-clock will not compare — their GPU runs are on a B200
and their CPU runs on an i7-11700K — which is exactly why solved-or-not is the
line to report.

One number from it worth keeping in view: OR-Tools' PDLP, the same algorithm
family this implements, solves 50 of 65 there while HiGHS solves 55 and cuPDLPx
57. The method is not automatically good. The implementation is most of it.

## For SIH specifically

The competition timeline is not the same as the engineering one, and conflating
them is the main risk.

**Coding is ~93% of what was planned, and the remaining 7% is worth about two
points.** It will not decide the outcome. What will:

1. **A demo a planner can actually drive.** Currently at zero. This is the
   single largest lever on the result and it is not solver work.
2. **A head-to-head number on MRPL's own model** — against HiGHS, on their
   instance. That converts "is this any good?" into a figure.
3. **Scale evidence** — one refinery model is not a scale argument.

The claim that survives scrutiny is not a score out of 100 on a rubric we wrote.
It is: *"restarted Halpern PDHG with PDLP's feasibility polishing and cuPDLPx's
four additions, benchmarked against published Netlib optima and against the
current state of the art, and here are the numbers."* That is reproducible, and
a judge can run it.

The question to have a crisp answer for is **"why not just use Gurobi?"** —
indigenous requirement (it is in the problem statement), zero licence cost, GPU
acceleration, full control. Twenty seconds, not two minutes.

Simplex and MILP depth will not happen before the finale. That is the correct
call, not a failure of nerve.
