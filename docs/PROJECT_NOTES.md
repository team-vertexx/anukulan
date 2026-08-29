---
title: "Sankhya — an optimization solver written from scratch"
subtitle: "What it is, how it was built, what went wrong, and what is next"
author: "Team Vertex"
date: "August 2026"
geometry: margin=2.5cm
fontsize: 11pt
colorlinks: true
---

# 1. What this is

Sankhya is a solver for linear programs, mixed-integer programs and convex
quadratic programs, written from scratch in C++17, with a CUDA backend for the
part of it that runs well on a GPU.

It was built for Smart India Hackathon 2026, problem statement **SIH26119** from
MRPL (Mangalore Refinery and Petrochemicals). The problem statement asks for an
indigenous, GPU-accelerated optimizer for refinery planning.

"From scratch" is meant literally. There is no HiGHS, no OR-Tools, no SciPy
underneath. The sparse matrix type, the LU factorisation, the simplex method,
the branch-and-bound tree, the ADMM loop and the CUDA kernels are all in `src/`.
HiGHS appears in the repository exactly once: as the thing the results are
benchmarked *against*.

Roughly 11,000 lines of solver and command-line tool, plus 3,700 lines of tests.

---

# 2. What an LP solver actually has to do

Skip this section if you already know it.

A **linear program** is: minimise `c'x` subject to `l <= Ax <= u` and
`lo <= x <= hi`. Everything is linear. The feasible region is a polyhedron and the
optimum sits at a corner of it.

A **mixed-integer program** is the same, except some variables must be whole
numbers. That single restriction makes the problem enormously harder — you can
no longer just walk to a corner, because the corner may not be an integer point.

A **quadratic program** adds a `½x'Qx` term to the objective.

For a refinery, the variables are things like how much of each crude to run,
how to route intermediate streams, and how much of each product to blend. The
constraints are unit capacities, material balances and product specifications.
The objective is margin.

There are three main ways to solve an LP, and this project implements two:

- **Simplex.** Walk from corner to corner, each step improving the objective.
  Exact, gives you a clean basis you can restart from, and it is what makes
  branch-and-bound practical. Poor fit for a GPU — each step depends on the
  last.
- **First-order methods (PDHG/PDLP).** Iterate with nothing but matrix-vector
  products. Much less accurate per step, but every step is embarrassingly
  parallel, which is why this is the family that ports to a GPU.
- **Interior point.** Not implemented, and that was a deliberate scoping call
  stated openly from the start.

---

# 3. The plan, and why it was in that order

The order was chosen so that each piece could be *checked* as soon as it
existed, rather than building the whole thing and hoping.

1. **Reader first.** Nothing can be tested until models can be read. MPS is an
   old, fiddly, column-oriented format with real ambiguities in it.
2. **Standard form and scaling.** The first-order method needs the problem in a
   particular shape, and needs the numbers equilibrated or it simply will not
   converge.
3. **First-order LP, then the GPU backend.** This is the piece the problem
   statement is really asking for.
4. **Simplex.** Needed for accuracy, and needed as the engine inside
   branch-and-bound.
5. **Presolve.** Shrinks the model before anything else touches it. Helps
   everything downstream.
6. **MILP.** Branch-and-bound on top of the simplex.
7. **QP.** ADMM, sharing the sparse linear algebra already written.

The governing rule throughout: **every claim has to be checked against something
external.** Netlib publishes optimal objective values for its LP instances;
MIPLIB publishes them for integer programs; Maros–Meszaros for QPs. A solver
that agrees with itself proves nothing.

---

# 4. What got built

## 4.1 MPS reader

Reads free-format and fixed-format MPS, plus the QPS variant for quadratic
problems. Checked against HiGHS on all 88 Netlib instances — same dimensions,
same nonzero counts, same objective — and it reads them about 1.4 to 1.5 times
faster.

The speed came from three unglamorous things: reusing one buffer instead of
allocating a fresh vector per line; trimming over the original line instead of
taking a substring of a substring; and caching the last column name, because MPS
groups entries by column, which turned 1.26 million hash lookups into 102,600 on
one instance.

There is a flag, `--neg-up-bound`, because a negative upper bound with no lower
bound is genuinely ambiguous in the format: HiGHS keeps the lower bound at zero,
CPLEX sends it to minus infinity. The default matches HiGHS, and the flag exists
so the choice is visible rather than hidden.

## 4.2 Presolve

Eleven reductions that shrink the model before solving: empty rows, singleton
rows, redundant rows, forcing rows, duplicate rows, fixed columns, empty
columns, free column singletons, bound tightening, doubleton equations, and dual
fixing.

Two of these are worth explaining.

**Doubleton equations.** A row with exactly two terms, `a*xi + b*xj = c`, lets
one variable be written in terms of the other and substituted out, taking a row
and a column with it. Across the Netlib set this removes 758 rows and columns.

**Dual fixing.** For each column, count the constraints that moving it in each
direction could break — its "locks". A column that can be pushed down without
breaking anything, and whose objective does not object, belongs at its lower
bound in some optimal solution. It fires on 22 of the 88 instances, removing 520
columns.

The hard part of presolve is not removing things. It is **putting the answer
back**. Every reduction has to record enough to reconstruct the full solution,
and — much harder — the dual values, which is the part most hobby solvers skip.
The solver reports whether the duals it hands back are exact or approximate,
because for some reductions the recovery genuinely is a guess, and saying so is
better than pretending.

## 4.3 First-order LP

The largest and most developed component. The iteration is two sparse
matrix-vector products and some vector arithmetic — no factorisation, no basis,
which is exactly why it ports to a GPU.

On top of the basic method sit:

- **Adaptive step size and primal weight**, following PDLP.
- **Restarts**, which is what makes the method practical at all.
- **Halpern iteration** (Lu and Yang, arXiv:2407.16144) instead of averaging.
- **Feasibility polishing** (PDLP paper, arXiv:2501.07018, section 4).
- **The four cuPDLPx additions** (arXiv:2507.14051): reflection, a constant step
  size, a restart criterion on the fixed-point residual, and PID control on the
  primal weight.

**Feasibility polishing deserves a paragraph** because it is the piece that
matters most for a refinery. The idea: a feasibility problem — the same
constraints with no objective — is far easier for this method than the LP it
came from, because nothing is pulling against the constraints. So when the
answer is nearly good enough, pause, solve two feasibility problems warm-started
from where you are, and you land on a point that is *almost exactly feasible*
with roughly the objective you already had.

The trade is explicit: it buys feasibility and pays for it in the duality gap.
That is why there is a separate `--gap-tol`. On the refinery model, asking for
"feasible to 1e-8, and I will accept a 1% gap" is the difference between a plan
you can run and one you cannot. A plan that overruns a unit capacity is not a
plan; a plan that leaves a fraction of a percent of margin on the table is.

## 4.4 GPU backend

A `LinAlgBackend` interface with two implementations: plain C++ and CUDA. The
solver loop is written once.

The single most important design decision: **the working set stays on the
device.** The loop never moves a vector across the bus. Data comes back only at
the convergence check, every fortieth iteration.

The first version did the opposite — it took host pointers and staged every
argument across on every call, roughly ten transfers per iteration. Measured on
a Tesla T4 it came out *slower than the CPU it was meant to accelerate*. The
kernels were never the problem. That rewrite is the single largest lesson in the
GPU work.

Measured on a T4, solve-time speedups over the same algorithm on CPU:
supportcase10 7.09×, graph40-40 3.00×, datt256_lp 2.82×, qap15 2.70×.

## 4.5 Simplex

A revised simplex with both algorithms — primal and dual — sparse LU with
Markowitz pivoting, product-form updates between refactorisations, Devex
pricing, dual steepest edge, a piecewise-linear phase one, and Bland's rule as
an anti-cycling fallback.

Both algorithms get all 16 Netlib instances with published optima correct.

## 4.6 MILP

Branch-and-cut. Nodes are solved by the dual simplex warm-started from the
parent's basis, which is what makes a child cheap — on the measured set it warm
starts 18,772 of 18,775 relaxations at about 3 simplex iterations per node.

On top: pseudocost branching upgraded to reliability branching, cover / c-MIR /
Gomory cuts with per-instance selection, reduced-cost fixing, node propagation,
and a fix-and-propagate diving heuristic.

## 4.7 QP

OSQP-style ADMM. The KKT system is quasi-definite, so by Vanderbei's theorem any
symmetric permutation of it factorises — which is what lets an up-looking LDL'
work without pivoting for stability. Solves 35 of the 40 smallest
Maros–Meszaros instances.

---

# 5. The problems, and how they were solved

This is the part worth reading. Every one of these cost real time, and every one
is recorded in the code next to the thing it explains, so it does not have to be
rediscovered.

## 5.1 Presolve declared a feasible model infeasible

Netlib's `maros` came back "infeasible". It is not.

The cause was my own tolerance. I was padding implied bounds outward by 1e-9 to
be safe, and the forcing-row rule also used 1e-9. So the padding manufactured
forcing rows: whole rows of variables got pinned onto bounds that only existed
because of the padding, and the errors accumulated across hundreds of rows.

Fix: stop padding, and use **two** tolerances — one to decide a reduction may
fire (1e-9), a much looser one to declare infeasibility (1e-7). Anything in
between means decline to reduce. A presolve that is unsure should do nothing,
not guess.

## 5.2 The dual simplex reported wrong answers as optimal

`fit1p` came back at 33,609 against a true 9,146.38 — a feasible point, wrong by
a factor of four, reported as optimal with a row violation of 2.8e-14.

I added a check on principle before knowing what was wrong: **before claiming
optimality, verify that every nonbasic column's reduced cost has the right
sign.** A solver must not assert what it has not established. The check named
the cause on its first run: dual feasibility held for 101 iterations and broke at
102, and the anti-cycling stall threshold was 100.

The cause: **Bland's rule was overriding the dual's ratio test.** In the primal
simplex, pricing and the ratio test are separate steps, so overriding pricing is
safe. In the dual, the ratio test *is* the entering choice and the only thing
holding reduced costs on the right side of zero. Overriding it breaks the
invariant the algorithm rests on.

Removing the override took the dual simplex from 13 of 16 correct to **16 of
16**, and `fit1p` from 67,323 iterations and wrong to 1,624 and right.

## 5.3 A cut that removed feasible solutions, invisible for weeks

The cover cut separator decides whether an item was "complemented" — whether
`x_j` was replaced by `1 - x_j`. It used to infer this by testing whether the
slack equalled `x_j`, which is true for a complemented item and false otherwise.

Except at `x_j = 0.5`, where `1 - x_j` is also 0.5 and the test says
"complemented" about everything. A simplex vertex can sit a binary variable at
exactly 0.5. When it did, the cut came out with `+1` where it needed `-1` and
removed feasible points.

426 separations at random interior points never landed on 0.5 and never saw it.
The fix was to record the flag instead of re-deriving it, and — more importantly
— the test now separates **at simplex vertices**, not only at random points.

## 5.4 The LDL' that factorised a diagonal matrix

The sparse LDL' factorisation was handed the upper triangle stored row-wise when
the algorithm needs the lower triangle. Every entry was skipped, the elimination
tree came out empty, and it silently became a diagonal factorisation.

The diagonal test passed at 1e-14 and told me nothing. The interface now says
"lower triangle, stored row-wise" in the header, loudly, and the test builds a
matrix, picks an `x`, forms `b = Kx`, solves and compares — it never checks a
residual the factorisation computed about itself.

## 5.5 The dual iterate that was never uploaded

On the GPU, the primal vector was uploaded to device memory at the start of the
solve and the dual vector was not.

This was harmless for as long as both started at zero — which they did, so it
sat there unnoticed. It became a real bug the moment anything needed a warm
start, because the warm start was silently discarded. And on CUDA the loop was
starting from whatever `cudaMalloc` happened to return, because `cudaMalloc`
does not zero memory while the CPU allocator does.

That asymmetry is the general lesson: **the CPU path hides a whole class of GPU
bug.** Anything that has not actually run on a GPU is untested there.

## 5.6 The wrong answer that a wider test set found immediately

For most of the project the MILP code was measured against seven MIPLIB
instances. Seven is not enough, and every tuned constant chosen against them
said so in its comment.

So I wrote a fetcher and pulled **103 instances** with published optima. The
wider set found, on its first run:

```
CLAIMED OPTIMAL BUT WRONG: fiber (60.8012%)
```

The solver reported 652,748.78 as a *proved* optimum, with a matching dual
bound, against a true 405,935.18. No warning, no numerical complaint — because
once a wrong prune happens, everything after it is consistent with the wrong
answer.

Finding it took eliminating layers one at a time: were the cuts valid? (yes) Was
the model with cuts appended valid? (yes) Were the node LPs solved correctly?
(yes — primal and dual simplex agreed exactly)

So I built the tool the problem actually needed: a **debug solution tracker**.
You hand the solver a solution you know is correct, and at every point where a
node could be discarded it first checks whether that solution is inside the
node, and reports the first prune that throws it away. SCIP carries the same
facility for the same reason. It found two separate bugs within minutes.

**Bug one — a bound crossing by 1.78e-15.**

```
BOUNDS CROSSED on column 1269: lower 6.0000000000000018 > upper 6
```

A variable's lower bound came out of bound propagation fractionally above its
upper bound — the last bit of a double, nothing more. The infeasibility check
compared them with **no tolerance at all** and declared the whole subtree
provably infeasible. The optimum was in that subtree.

Bound propagation, meanwhile, tolerates a crossing of up to 1e-7 before it
complains. **The two checks disagreed about what an empty box is**, and the
answer fell through the gap between them.

**Bug two — cover cuts derived from earlier cuts.**

The cut separator loops over every row of the working model. After the first
round, the working model contains the cuts from previous rounds. In round eight,
a cover cut was derived from a row that was itself a generated cut, and that cut
removed the optimum.

The reason this hid so well: **each cut family on its own was valid.** Cover
alone, MIR alone, Gomory alone — all fine. Only the combination broke, because
only in the combined run did the search reach the particular point that produced
the bad cut. Testing families one at a time said everything was healthy.

Fix: cover cuts are now separated only from rows of the original model.

After both fixes, `fiber` returns exactly 405,935.18.

## 5.7 Things that were tried, measured, and rejected

Not everything works, and the ones that did not are written down with their
numbers so nobody tries them again, including me.

- **Harris ratio test without EXPAND.** Three instances better, six worse, and
  `blend` stopped solving at all.
- **Coefficient tightening.** Implemented in full. Fires three times across
  seven instances, all on one, and moves the root bound from 6875.00000004 to
  6875.00000002 — which is noise. Left switched off.
- **Parallel column merging.** 20% of Netlib columns are parallel to another,
  which looks compelling. But merging also needs the objective in the same
  ratio, which cuts it from 31,954 columns to 1,471. Measured, not built.
- **Hypersparsity.** The literature reports a 5.2× mean speedup from exploiting
  it. Their test set was *selected* for the property. On ours, only 4 of 21
  instances qualify and the overall rate is 15.4% against a 60% threshold. Six
  to eight weeks of work to help four instances.
- **Two fixes for a QP convergence failure.** The diagnosis was clear — the
  adaptive step-size rule drives a parameter somewhere bad and the gate that
  stops it oscillating also stops it recovering. Both fixes that follow from
  that diagnosis were implemented and measured and neither worked. Recorded with
  what remains untried.

## 5.8 Traps in measuring, not in the code

Three of these cost more time than most of the real bugs.

- **`zsh` does not word-split unquoted variables.** A variable holding
  `--presolve --abs-tol=1e-8` arrives as *one* argument, the program rejects it,
  and the checker silently reads a stale result file. This one bit three times,
  including once after I had written it down in the onboarding notes.
- **A measurement taken under load is not a measurement.** Twice a benchmark
  reported a regression that did not exist, because it was running three
  60-second solves per instance back to back. Both times, re-running alone
  showed the two configurations were identical. Had I believed the first
  numbers, I would have discarded a change that was a pure improvement.
- **A test program linked against a static library does not relink when the
  library is rebuilt.** I spent twenty minutes reading correct code that was not
  the code being executed.

---

# 6. How to check any of this yourself

Nothing here has to be taken on trust.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build            # 11 test suites
```

```bash
python3 scripts/fetch_netlib.py   # 88 LP instances with published optima
python3 scripts/fetch_miplib.py   # 103 integer instances with published optima
```

```bash
# Does the reader agree with HiGHS?
python3 bench/verify_reader.py

# Does presolve change the answer? (it must not)
python3 bench/verify_presolve.py --tol=1e-6 --abs-tol=1e-6 --check-feasibility

# Does the solver match published optima on integer problems?
python3 bench/miplib_survey.py

# One model, end to end
build/sankhya solve data/refinery/refinery.mps --tol=1e-8 --gap-tol=1e-2 --presolve
```

Two conventions in the source worth knowing before reading it:

**Every tuned constant carries its measurement.** When you find a number in this
codebase, the comment above it says what else was tried and what happened. A
constant without a recorded sweep is a guess, and the comment says that too.

**Failures are recorded, not deleted.** The approaches that lost are written
down with their numbers next to the thing they would have replaced. `git log` is
the other half of this — the commit messages carry the reasoning, not just the
change.

---

# 7. Where it stands

Working and checked against published answers:

- Reader agrees with HiGHS on all 88 Netlib instances, and is faster.
- Primal and dual simplex both get all 16 Netlib instances with published
  optima correct.
- Presolve removes about 19% of rows and 12% of columns across Netlib without
  changing any answer, and recovers the dual values as well as the primal.
- The first-order method solves the large instances the GPU work targets, and
  the GPU is measurably faster than the CPU on the same algorithm.
- Branch-and-cut proves optimality on the smaller integer instances and gets
  close on the rest.
- The QP solves 35 of the 40 smallest Maros–Meszaros instances.

Known and open:

- Five QP instances fail, four of them one family. The cause is understood, two
  fixes have been tried and did not work, and what remains untried is written
  down.
- Everything built since the last GPU run is unverified on GPU hardware. That
  is the next thing to close.

---

# 8. What is next

**Immediately.** Verify the current code on a GPU. Nothing since the last run
has been tested there, and the CPU genuinely does hide a class of bug that only
appears on the device. This is not an improvement, it is closing a risk.

**Then, in rough order of what the measurements say is worth it:**

1. **Move the convergence check onto the device.** On one instance, 79% of the
   solve time is outside the kernels — the host is computing things the device
   already has. Part of this is done; the larger half is not.
2. **More integer instances, and re-check the tuned constants against them.**
   The wider MIPLIB set found a wrong answer within minutes of existing. Several
   branch-and-bound constants were chosen against seven instances and should be
   rechecked against a hundred.
3. **Partial pricing in the simplex.** Profiling says the dominant cost is now
   scanning every column for the pivot row. Scanning a rotating subset trades
   iteration count against per-iteration cost — worth measuring, not assuming.
4. **The remaining presolve reductions.** Dominated columns, probing,
   sparsification. Each is documented in the literature; each needs measuring
   before being kept.
5. **Continuous integration.** The test suite is good and nothing runs it
   automatically.

**Not next, and deliberately so.** Interior point methods, hypersparsity, and
Forrest–Tomlin updates are all standard things a mature solver has. Each was
looked at and each was measured against this instance set rather than against
the literature, and none of them is where the next real gain is.

---

# 9. The one thing to take away

The parts of this project that took the longest were not the algorithms. The
algorithms are in papers and they work.

What took the time was **finding out when the solver was quietly wrong** — and
building the things that make that visible: a dual-feasibility check the simplex
runs on itself before claiming optimality, cut tests that enumerate every
feasible point of small problems, a survey against a hundred published optima, a
debug-solution tracker that names the exact step where a correct answer gets
thrown away.

A solver that is slow tells you it is slow. A solver that is wrong tells you
nothing at all — it hands you a confident number with a matching bound and no
complaint. On `fiber` it was wrong by 60% and looked completely healthy. Every
one of those checking tools exists because something got through.
