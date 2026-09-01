# Architecture

How the solver is put together, component by component. This describes the code
as it is, not as it was planned — for the plan, read [PLAN.md](../PLAN.md), and
for the survey the plan came from, [RESEARCH.md](../RESEARCH.md).

Read this alongside the headers. Most of the reasoning lives in
`src/sankhya/*.hpp` next to the thing it explains, and this document points at
those rather than repeating them.

---

## 1. Two forms of the problem

Everything hinges on there being two representations, and on knowing which one
you are looking at.

**`Model`** (`model.hpp`) is the problem as the file described it:

```
min or max  c'x + ½x'Qx
subject to  rl ≤ Ax ≤ ru
            cl ≤ x ≤ cu
            some x integral
```

Two-sided rows, either sense, integer columns, an optional Hessian. This is what
the reader produces, what presolve reduces, and what the answer is finally
reported against.

**`StandardLp`** (`standard_form.hpp`) is what the first-order method works in:

```
min   c'x
s.t.  K_eq   x  = q_eq        the first num_equalities rows
      K_ineq x ≥ q_ineq       the rest
      lower ≤ x ≤ upper
```

Equalities first, so the dual projection is one split: leave the first block
alone, clamp the rest at zero. A two-sided row becomes **two** inequality rows.
That costs a duplicated row and buys a trivial projection, which matters because
that projection runs every iteration on the GPU.

`row_origin` records where each standard row came from (`model_row` and a sign),
which is how row duals get mapped back. A two-sided row contributes twice and
the two are summed.

The simplex works on the `Model` form directly, not on `StandardLp` — it wants
two-sided rows and bounds, not a cone.

---

## 2. Sparse matrix (`sparse.hpp`)

CSR, built from triplets. Two properties the rest of the code leans on:

- `from_triplets` **sums duplicates and drops exact zeros**. The doubleton
  presolve reduction depends on this: it emits a coefficient for a column that
  may or may not already be in the row, and lets the builder merge or fill.
- `transpose()` gives the column view. Presolve keeps both and uses `at` to walk
  a column's rows.

---

## 3. Reader (`mps_reader.cpp`)

Free and fixed format MPS, plus QPS for the Maros–Meszaros quadratic set.
Matches HiGHS on all 88 Netlib instances and reads them 1.4–1.5× faster.

The speed came from three specific things, none of them clever:

- tokenisers fill one reused buffer instead of returning a fresh `vector` per line
- fixed-format trims over the original line rather than taking a substring of a substring
- `column_index` caches the last column name, because the COLUMNS section is grouped
  by column — on graph40-40 that is 1.26M hash lookups reduced to 102,600

`--neg-up-bound` exists because a negative `UP` bound with no lower bound is
genuinely ambiguous in the format: HiGHS keeps the lower bound at zero, CPLEX
sends it to −∞. Default matches HiGHS.

---

## 4. Presolve (`presolve.cpp`, `presolve.hpp`)

Ten reductions, applied in rounds until nothing fires:

| reduction | what it does |
|---|---|
| empty row | no entries — check it is satisfiable, drop it |
| singleton row | one entry — it is a bound on that column |
| redundant row | activity bounds already inside the row bounds |
| forcing row | activity can only be met at one end — fixes every column in the row |
| duplicate row | parallel rows merge into the tighter one |
| fixed column | `lower == upper` — substitute the value out |
| empty column | in no row, so the objective decides it |
| free column singleton | in one equality row, bounds not binding — solve for it |
| bound tightening | interval arithmetic on each row |
| **doubleton equation** | `a·xi + b·xj = c` — substitute one column out |

### Postsolve

`PostsolveStack` records what left and replays it in reverse.

The **primal** side needs two entry kinds: a column either takes a known value
(`kFixed`) or is read off one equality row (`kSingleton`,
`x_col = (rhs − Σ terms) / coefficient`). The doubleton substitution fits the
second form exactly, which is why it needed no new primal machinery.

The **dual** side is harder and is the part most solvers skip. Each removed row
records how its dual is recovered:

- `kZero` — empty and redundant rows carry no dual
- `kFreeSingleton` — the eliminated column was free, so its reduced cost is zero
  and `y_i = c_k / a_ik` exactly
- `kSingleton` — the row became a bound; that bound carries the column's reduced
  cost in the reduced model, so `y_i = d_j / a`
- `kDoubleton` — the eliminated column was implied free, so
  `y_r = (c_i − Σ_{k≠r} A_ki y_k) / a`, and the other rows' duals are unchanged
  by the substitution (proved in the header)
- `kUnrecoverable` — forcing and duplicate rows; the dual is set to zero and
  `dual_is_exact()` returns false

`dual_is_exact()` is the honesty valve. Bound tightening also trips it, because
a tightened bound that turns out active changes the dual. The CLI prints
`# duals exact` or `# duals approximate` accordingly.

### Why doubleton is special

It is the **only reduction that changes a coefficient of A**. Every other one is
a flag plus a bound adjustment, which is why the workspace can hold the matrix
`const`. Doubleton substitution rewrites every row the eliminated column
appeared in, so it runs *after* the others and does not re-enter them, and the
rewritten coefficients are applied when `build_reduced` assembles triplets.

It only fires when the eliminated column is **implied free** — the row and the
other column's bounds already confine it inside its own. That guarantees a zero
reduced cost, which is what makes the dual recoverable exactly. 827 of the 841
doubletons standing after the other reductions qualify, so the restriction costs
almost nothing.

Two bugs it had first, both worth knowing because both are easy to reintroduce:

1. The guard that skips rows holding an already-substituted column ran *after*
   the liveness check. A substituted column is also a dead one, so the guard
   never fired and rows were read with stale coefficients.
2. A column that an earlier substitution points at may not itself be eliminated
   later — the rows that traded a column for it carry terms aimed at it. Getting
   this wrong put `stocfor2` 189 units outside the original model.

---

## 5. Scaling (`scaling.cpp`)

Ruiz equilibration in the infinity norm (10 sweeps) then one Pock–Chambolle pass
in the 1-norm, following PDLP. Not a tuning knob: Netlib has instances spanning
ten orders of magnitude in their coefficients, and PDHG on those unscaled does
not converge in any useful number of iterations.

`x = D2 x̃`, `y = D1 ỹ`, and everything the solver reports is unscaled first,
because the problem the user asked about is the unscaled one.

---

## 6. First-order LP (`pdhg.cpp`, `pdhg.hpp`)

The largest and most developed component. The iteration is two sparse
matrix-vector products and some vector arithmetic — no factorisation, no basis,
which is exactly why it is the one that ports to a GPU.

### Base scheme

Primal-dual hybrid gradient on the saddle-point form, with PDLP's enhancements:

- **adaptive step size** — PDLP's rule, `η` below the local curvature limit
- **primal weight** — splits `η` between the two steps, `τ = η/ω`, `σ = ηω`
- **restarts** — cuPDLP's three conditions (sufficient, necessary + stalling, artificial)
- **Halpern iteration** — `z^{k+1} = (k+1)/(k+2)·T(z^k) + 1/(k+2)·z^anchor`,
  from Lu and Yang, *Restarted Halpern PDHG for Linear Programming*
  ([arXiv:2407.16144](https://arxiv.org/abs/2407.16144))

The Halpern anchor blend also blends `K·z`, using a stored `K·anchor`. K is
linear so this is exact, and recomputing the product instead was the whole
difference between Halpern winning and losing.

### Feasibility polishing

Algorithm 4 of the PDLP paper
([arXiv:2501.07018](https://arxiv.org/abs/2501.07018), §4). Two sub-problems,
both solved by the same routine, warm started from the iterate that triggered
them:

```
primal   c := 0                        started from (x, 0)
dual     q := 0, finite bounds := 0    started from (0, y)
```

A feasibility problem has nothing pulling against the constraints, so PDHG
converges on it far faster than on the LP it came from; and PDHG iterates are
non-increasing in distance to any optimal solution, so a warm start near a good
point stays near it.

The technique buys **tight feasibility and pays for it in the gap** — which is
why `gap_tolerance` exists separately from `tolerance`. Asking for 1e-9
feasibility with a 1% gap is a sensible thing to want from a production model; a
refinery plan that violates a capacity is not a plan, while one that leaves 1%
on the table is.

Each sub-solve starts the **other** coordinate at zero. Carrying the seed's dual
into a problem whose objective was just deleted sent the dual objective from
5969 to −11234 the first time this was written.

### The cuPDLPx additions

From [cuPDLPx (arXiv:2507.14051)](https://arxiv.org/abs/2507.14051), which is
built on the same restarted Halpern base:

1. **Reflection** — `R(z) = (1+γ)T(z) − γz`. Free: `R(z) = T(z) + γ(T(z) − z)`
   and `T(z) − z` is the `dx`/`dy` the step already produced, while
   `K·R(z) = K·T(z) + γ·K·dx` with `K·dx` already computed for the step-size rule.
2. **Constant step size** — `0.998/‖K‖` instead of the adaptive rule. Every
   rejected adaptive trial is a matrix product spent on a discarded step.
3. **Fixed-point restart** — the same three conditions measured on `‖z − T(z)‖`
   instead of the KKT error. That quantity is computed every iteration anyway,
   where the KKT error costs two matrix products per check.
4. **PID primal weight** — the existing exponential smoothing turns out to be
   this law with `Kp = 0.5, Ki = Kd = 0`, so it is a strict generalisation.

They are coupled: each alone is neutral or worse, and reflection with an
adaptive step diverges outright, because the adaptive safety rule was derived
for plain PDHG rather than the reflected operator.

### Infeasibility detection

A first-order method cannot otherwise tell an infeasible problem from a slowly
converging one. PDLP's observation is that while the iterates diverge, the
*difference* of consecutive dual iterates converges to a Farkas ray, so that is
what gets tested. Cheap, and it matters inside branch and bound where most nodes
are infeasible and a guess makes the dual bound worthless as a proof.

---

## 7. GPU backend (`cuda_backend.cu`, `backend.hpp`)

`LinAlgBackend` is the seam. The CPU implementation is the reference; CUDA
overrides it. The solver loop is written once.

The single most important design fact: **the working set stays on the device.**
The loop never moves a vector across the bus. Data comes back only at the
convergence check, every fortieth iteration.

The first version took host pointers and staged every argument across on every
call — roughly ten transfers per iteration. Measured on a T4 it came out *slower
than the CPU it was meant to accelerate.* The kernels were never the problem.

Other things that earned their place:

- **adaptive SpMV vector width** — `choose_vector_width` picks the smallest power
  of two ≥ the average row length, clamped to [2,32], following Bell and Garland.
  On `spmv K x` this took 0.207s to 0.042s.
- **the loop is over the block's first row**, not each vector's own row, so the
  loop condition is warp-uniform and `__shfl_down_sync`'s full-warp mask is not
  a lie.
- **fused step-size terms** — three inner products in one launch, finished on
  device, three doubles crossing the bus. Each reduction ends in a
  device-to-host copy and a copy is a synchronisation; three per iteration on
  qap15's 24,720 iterations is about a second of doing nothing.

`scripts/check_cuda_syntax.sh` stubs the CUDA runtime and type-checks
`cuda_backend.cu` with clang on a machine with no GPU. It has caught real errors
before a Kaggle round trip.

### One trap worth knowing

`prepare()` caches uploaded matrices **by address** and `release()` frees **all
of them**. Nested solves (which feasibility polishing creates) must therefore not
release while an enclosing solve is still running — `g_solve_depth` in
`pdhg.cpp` handles that. On the CPU both are no-ops, so this class of bug is
invisible here and only appears on the GPU.

The same asymmetry bit us once already: `cudaMalloc` does not zero, the CPU
allocator value-initialises, and the dual iterate was never uploaded to the
working vector. Harmless while everything started at zero — and silently fatal
for any warm start.

---

## 7a. Threads (`threading.hpp`, `threaded_backend.cpp`)

The same seam that carries the CUDA backend carries a threaded CPU one.
`LinAlgBackend` was written so the solver loop is expressed once and the
arithmetic under it can be replaced; `threaded_cpu_backend(n)` is a second
implementation of that interface and nothing in `pdhg.cpp` knows it exists.
`--threads=n` selects it. The default is 1, which is `cpu_backend()` itself.

### What is threaded, and what is deliberately not

| operation | threaded | why |
|---|---|---|
| `multiply`, `multiply_transpose` | yes | rows are independent and disjoint |
| `primal_step`, `dual_step`, `advance_kx` | yes | no cross-entry dependence |
| `accumulate`, `scale_into`, `blend`, `fill`, `copy` | yes | elementwise |
| `inf_norm` | yes | maximum is exactly associative |
| **`dot`, `two_norm`, `weighted_norm_squared`** | **no** | **summation is not** |

That last row is the whole design. Floating point addition is not associative,
so splitting a sum changes it in the last bits — and this method restarts on a
residual computed from these sums, so the last bits decide when an epoch ends
and therefore the iteration count. A solver whose iteration count moves with the
thread count cannot be demonstrated, benchmarked or debugged, which is exactly
the failure this component had to avoid.

Leaving them serial buys a guarantee that is stronger than what production codes
offer:

> The threaded backend returns **bit-identical** results to the serial one, at
> every thread count, on every run.

Not "reproducible for a fixed thread count", which is what CPLEX's deterministic
mode and Gurobi both promise, and not "close to". Identical. The matrix products
split by rows, so each output entry is still accumulated by the same serial inner
loop in the same order and no two blocks write the same entry; the elementwise
steps have no cross-entry dependence at all; and the one reduction that is
threaded is a maximum, which is associative and commutative in floating point.

The cost is bounded by Amdahl on whatever share the inner products hold, and
that share is small — see RESULTS.

`test_threading` asserts equality rather than closeness, at thread counts past
this machine's core count, because a pool that only behaves when it fits is one
that will misbehave on somebody else's laptop.

### Two things the machine decided

**Blocks are taken, not dealt.** This is an M4: four performance cores and six
efficiency ones. Under an even static split every barrier waits for whichever
chunk landed on an efficiency core, and that chunk takes about three times as
long. The clearest evidence is a streaming triad, which should be flat once
memory is saturated and instead falls by more than half when the efficiency
cores join. So the loop is cut into more blocks than there are threads and
workers take the next free one. Which thread runs which block does not affect
the answer, by the argument above — that is what makes dynamic scheduling
available here at all.

**The block count follows the work, not the thread count.** A block costs an
atomic increment, and past a point that is all it is doing: on a small matrix,
eight blocks per thread is markedly *slower* than one. So the count is chosen
from the nonzero count, floored at one block per thread and capped at eight.

**Workers spin briefly, then yield.** Pure spinning made a ten-thread barrier
cost milliseconds rather than microseconds: with every core occupied the
scheduler takes a spinner off its core and everyone waits out its quantum.

**`--threads=0` asks for the performance core count, not every core.** The
measured speedup peaks at four threads here and is a *loss* at nine and ten, so
a default that took the whole machine would ship a flag that slows the solver
down when it is used. Where the system will report how many performance cores
it has, that is the number; otherwise half the logical cores, which is the
closest guess available on a machine that does not distinguish them. It is a
starting point rather than a tuned value — the right number depends on the
machine's bandwidth per core, and `bench/thread_scaling.py` is how to find it.

### Why not OpenMP

Three reasons, in the order they decided it.

1. **It cannot promise a reduction order.** The combining order in a `reduction`
   clause is the runtime's business and may vary with the thread count. Every
   guarantee above rests on owning the partition.
2. **Apple Clang ships without libomp.** Requiring it means a Homebrew
   dependency on every machine that builds this, which breaks the one thing the
   build currently promises: clone it and run `cmake`.
3. **A thread pool is ninety lines.** The sparse LU is three hundred, and it is
   ours. The premise of the project is that the stack is its own.

### What is not threaded, and why

The simplex is serial, and so is the branch and bound tree. Both are
measurements rather than omissions, and RESULTS carries them.

---

## 8. Simplex (`simplex.cpp`, `lu.cpp`)

A revised simplex with both algorithms.

**Basis** — sparse LU with Markowitz pivoting and singleton peeling (Suhl and
Suhl), product-form updates between refactorizations, refactorization every 50
pivots. `ftran`/`btran` solve with the basis and its transpose; `pivot_row` is a
BTRAN of a unit vector and is used by both Devex and the dual.

**Primal** — Devex pricing (Harris), a piecewise-linear phase one, Bland's rule
as an anti-cycling fallback, and a bounded rollback of the last update when
numerics go wrong.

The piecewise-linear phase one is the single largest thing in this component.
Before it, `brandy` took 20,109 iterations with 26 switches to Bland; after, 839
iterations and zero switches. It also removed the refactorization cliff, which
meant every constant tuned above it had to be re-measured, and it flipped Devex
from losing to winning.

**Dual** — Koberstein's Algorithm 2: pricing, BTRAN, pivot row, dual ratio test,
FTRAN, update. Dual steepest edge with `β_i = ‖e_i'B⁻¹‖²`, updated from the
**old** `β_r` (Koberstein 3.47b — using the new one makes every update term too
small), and *without* recomputing all weights on a drift test (Koberstein 6.32
says explicitly not to).

**Bland's rule is not in the dual's ratio test**, and that is load-bearing. In
the primal, pricing and the ratio test are separate steps, so overriding pricing
is safe. In the dual the ratio test *is* the entering choice and the only thing
holding reduced costs on the right side of zero. Overriding it made the solver
report wrong answers as optimal — `fit1p` at 33,609 against a true 9,146.38,
with a row violation of 2.8e-14. Removing the override took the dual from 13/16
to **16/16**.

That bug was found by the dual's own optimality check, which was added on
principle — a solver must not assert what it has not established — and named the
cause on its first run.

---

## 9. MILP (`branch_and_bound.cpp`, `cuts.cpp`)

Branch and cut on the LP relaxation.

- **node solve** — warm-started dual simplex from the parent's basis. On the
  measured set this warm-starts 18,772 of 18,775 relaxations at 2.9 simplex
  iterations per node.
- **branching** — pseudocost
- **diving** — fix-and-propagate, on a short iteration leash
- **cuts** — knapsack cover, c-MIR, and Gomory mixed-integer, selected on
  efficacy and orthogonality so the families compete on one list

Gomory cuts are **off by default**, and that is measured: they give better root
bounds on all seven test instances and a worse tree on three. They are limited
to two rounds with a bound-monotonicity rollback when enabled.

The cover cut separator had a bug worth remembering: it inferred whether an item
was complemented from `slack == x[j]`, which is true for *everything* at
`x = 0.5`. The flag is now recorded explicitly. It was only visible at simplex
vertices — 426 random-point separations never caught it, which is why
`test_cuts` now separates at vertices too.

---

## 10. QP (`qp.cpp`, `ldl.cpp`)

OSQP-style ADMM. The KKT system is quasi-definite, so by Vanderbei's theorem any
symmetric permutation of it factorises — which is what lets an up-looking LDL'
(Davis, Algorithm 849) work without pivoting for stability.

`ldl.hpp` takes the **lower triangle stored row-wise**, and the header says so
loudly because the first version was handed the upper triangle and silently
became a diagonal factorisation. The diagonal test passed at 1e-14 and told us
nothing.

`analyse()` takes a nonzero budget and stops early when predicted fill exceeds
it, falling back to conjugate gradient. This is not premature: the fill on
random systems is about 3.4×, and on AUG2DC it is **2,772×**. Counting all the
way to 2772 before declining costs most of the price of the thing being
declined.

Solution polishing solves the reduced KKT system directly rather than
eliminating ν, which produced a condition number around 1/δ that CG could not
handle. It is accepted on 16 of 24 instances, halves the error on 15 of those,
and worsens none.

---

## 11. How they compose

```
      .mps ──► mps_reader ──► Model
                                │
                                ├──► presolve ──► Model (reduced) ──┐
                                │                                   │
                                ▼                                   ▼
                        to_standard_form                     to_standard_form
                                │                                   │
                                ▼                                   ▼
                          scale_lp                             scale_lp
                                │                                   │
                                ▼                                   ▼
                          solve_pdhg  ◄── LinAlgBackend (CPU | CUDA)
                                │
                                ▼
                    unscale ──► postsolve.apply ──► answer in the original model
```

The simplex and MILP paths take `Model` directly. The QP path takes `Model` with
its Hessian. Presolve switches its column-removing reductions off when a Hessian
is present, because substituting a column out of a quadratic rewrites Q's
neighbours as well as c.

The CLI (`app/sankhya_cli.cpp`) is one command per entry point and does the
mapping back: standard-form row duals fold onto model rows through `row_origin`,
then presolve's row recovery runs, then the solution file carries
`# dual <row name> <value>` and a `# duals exact|approximate` marker.
