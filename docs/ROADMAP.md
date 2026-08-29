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

Missing reductions, roughly in value order: dominated columns, parallel columns,
probing, dual fixing, clique merging, implied-integer detection, sparsify.

One structural note: the doubleton pass runs **after** the main loop and does not
re-enter it, because it is the only reduction that rewrites A and re-entering
would mean rebuilding the matrix each round to keep activity bounds honest. What
that leaves on the table has not been measured.

### 3. QP 50 → 70

AMD ordering for the LDL', a proper regularisation strategy, warm starting,
infeasibility detection. Note the fill-budget sweep in RESULTS before starting:
raising the budget makes the set *slower*, so AMD needs to be justified by
measurement rather than assumed.

### 4. Simplex 40 → 65

**Start with this, before any of the papers below.** The primal simplex does
*two* full O(nnz(A)) passes over the matrix every iteration, and one of them
should not exist:

- `SimplexBasis::compute_duals` recomputes **every** reduced cost from scratch,
  `d_j = c_j − π'A_j` over all nonbasic columns. It is called at the top of the
  loop, every iteration.
- The Devex weight update then computes the full pivot row `α_rj = ρ'A_j`, again
  over every nonbasic column.

A revised simplex updates the reduced costs from the pivot row rather than
recomputing them:

```
θ_d      = d_q / α_rq
d_j     ← d_j − θ_d · α_rj      for nonbasic j
d_leaving = −θ_d
d_q       = 0
```

and `α_rj` is exactly what the Devex pass already has in hand. Fusing the two
turns two passes into one, and the pricing pass then only has to run at
refactorization, for drift.

Two things to be careful about:

- **Phase one changes the cost vector every iteration.** `phase_cost` is rebuilt
  from the basis infeasibilities, and the piecewise-linear phase one walks
  through breakpoints flipping several statuses in a single step. The
  incremental update is only valid where the costs are fixed, so phase two gets
  it and phase one keeps recomputing — at least until someone measures whether
  tracking the cost changes explicitly is worth it.
- **Drift.** Recompute at each refactorization and compare, the way the dual
  steepest edge weights are already checked.

Honest note on the payoff: the whole Netlib set solves in 0.75 s today, so the
absolute saving is small and this will not show up in a demo. It matters because
it is the difference between a revised simplex and one that recomputes what it
already knows, and because everything below gets measured against it.

Then, three papers in order of value:

- **Hall and McKinnon (2005), hypersparsity** — the single biggest jump for
  Netlib-scale problems. When the right-hand side of an FTRAN/BTRAN is sparse,
  a different solve algorithm wins. Conceptually simple.
- [Hall and Huangfu, *A High Performance Dual Revised Simplex Solver*](https://webhomes.maths.ed.ac.uk/hall/HaHu11/ERGO-11-007.pdf)
  — what HiGHS is built on
- [Koberstein's thesis](https://d-nb.info/978580478/34) — already used here for
  dual steepest edge; it also contains Forrest–Tomlin and the bound-flipping
  ratio test

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
