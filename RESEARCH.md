# SIH26119 — Phase 0 research

Everything below is what I found before writing any solver code. Sources are linked
inline. Where I am stating something from memory rather than from a source I checked,
I say so.

Date of research: 22 Aug 2026.

---

## 0. What the problem statement actually says

I pulled the official text rather than working from a summary. Full copy is in
[refs/SIH26119-problem-statement.md](refs/SIH26119-problem-statement.md), scraped from
sih.gov.in/sih2026PS on 21 Aug 2026.

Title: **Indigenous GPU-Accelerated Optimization Solver (Sovereign Alternative to
Express / CEPLEX)**
Org: Mangalore Refinery and Petrochemicals Limited (MRPL). Category: Software.
Theme: Miscellaneous. Idea submission deadline: **20 September 2026**. Cap: 500 ideas.

Four things in the official text change how I plan this, versus the working brief I
started from:

1. **GPU is in the title.** Not optional flavour. The description softens it
   ("GPU acceleration considered where it provides measurable benefits"), but the
   title is the thing a jury reads first. A working GPU path is the headline and I
   should not treat it as a stretch goal.

2. **The from-scratch rule is explicit and strict:** "It shall not be built upon any
   existing open source solver library but shall be built from scratch from
   mathematical foundation." See section 7 for how I read that.

3. **The dataset field names the benchmark sets for us:** "MIPLIB, Netlib LP,
   Mittelmann benchmark instances, QPLIB (for quadratic programming where applicable),
   along with representative refinery scheduling, crude blending, production planning
   and supply chain optimization case studies from open literature."
   That last clause is important and my working brief did not have it. **A refinery
   model is part of the expected deliverable, not a nice-to-have.** MRPL is the
   proposing org; a solver that only ever prints `AFIRO solved` will read as an
   academic exercise to them.

4. **The Expected Solution paragraph is the real contract**, and it is much softer
   than the Description paragraph:
   - basic API or CLI is enough, no GUI;
   - solve standard benchmarks from MIPLIB / Netlib / Mittelmann;
   - compare quality and performance against **at least one** established commercial
     or open-source solver;
   - demonstrate numerical robustness on degeneracy, weak LP relaxations,
     ill-conditioned constraint matrices;
   - be a transparent, extensible, sovereign foundation.

   The Description paragraph's "thousands to millions of variables, beats weaker
   implementations on hard degenerate MILP" is aspiration, not an acceptance test. I
   am building to the Expected Solution and putting the rest on a roadmap slide.

One more thing the Description asks for that my brief dropped: "revised simplex **and
interior-point methods** for continuous optimization." I am not building an IPM. I
will say so explicitly in the deck and put it on the roadmap — an honest omission
beats a broken one, and PDHG covers the same "handles huge sparse LPs" ground that
IPM would.

---

## 1. LP by first-order method: PDHG / PDLP / cuPDLP

This is the headline. It is also, for a hackathon, by far the best value per line of
code: the whole solver is sparse matrix-vector products, vector arithmetic and
projections onto boxes. No factorization, no basis, no pivoting. That is exactly what
a GPU wants, and it is why cuPDLP-class solvers now beat simplex on large sparse LPs.

Papers I used:

- Applegate, Díaz, Hinder, Lu, Lubin, O'Donoghue, Schudy, *Practical Large-Scale
  Linear Programming using Primal-Dual Hybrid Gradient*, NeurIPS 2021 —
  [arXiv:2106.04756](https://arxiv.org/abs/2106.04756)
- *PDLP: A Practical First-Order Method for Large-Scale Linear Programming* (the long
  journal version, 2025) — [arXiv:2501.07018](https://arxiv.org/abs/2501.07018)
- Lu, Yang, *cuPDLP.jl: A GPU Implementation of Restarted PDHG for LP in Julia* —
  [arXiv:2311.12180](https://arxiv.org/abs/2311.12180), published in Operations
  Research
- *cuPDLP-C* (the C rewrite) — [arXiv:2312.14832](https://arxiv.org/html/2312.14832)
- *cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for LP*, 2025 —
  [arXiv:2507.14051](https://arxiv.org/abs/2507.14051)

### 1.1 Problem form

I will normalise every LP into:

```
min  c^T x
s.t. A x  = b        (m1 equality rows)
     G x >= h        (m2 inequality rows)
     l <= x <= u     (bounds, either side may be infinite)
```

Stack `K = [A; G]` (m = m1 + m2 rows), `q = [b; h]`. Then

```
X = { x : l <= x <= u }
Y = { y in R^m : y_i >= 0 for i in the inequality block }
```

MPS `L` rows are negated into `G` rows; `RANGES` rows become two `G` rows (costs a
duplicated row, keeps the code simple). The newer PDLP paper uses a two-sided form
`l_c <= Ax <= u_c` with a support-function term `p(y; l, u) = u^T y^+ - l^T y^-`; that
avoids the row duplication but complicates the dual projection. I am taking the
simpler one-sided form first and treating two-sided as an optimisation later.

### 1.2 Saddle point and the PDHG iteration

```
min_{x in X} max_{y in Y}  L(x, y) = c^T x - y^T K x + q^T y
```

The iteration (this is the exact form in cuPDLP.jl):

```
x^{k+1} = proj_X ( x^k - tau * (c - K^T y^k) )
y^{k+1} = proj_Y ( y^k + sigma * (q - K (2 x^{k+1} - x^k)) )
```

`proj_X` is a componentwise clamp to `[l, u]`. `proj_Y` is `max(y_i, 0)` on the
inequality block and identity on the equality block. Both are trivially parallel.

Per iteration cost: one `K^T y` (SpMV with the transpose) and one `K x` (SpMV), plus
O(n+m) vector work. Nothing else.

Reparameterise the two step sizes as a single step size `eta` and a **primal weight**
`omega`:

```
tau = eta / omega        sigma = eta * omega
```

Convergence needs `eta < 1 / ||K||_2` in the weighted norm

```
||z||_omega = sqrt( omega * ||x||_2^2 + ||y||_2^2 / omega ),   z = (x, y)
```

Splitting one step size into (eta, omega) is the trick that makes the method work in
practice: `eta` handles the scale of `K`, `omega` handles the relative scale of the
primal and dual iterates, and they are tuned by two different rules.

### 1.3 Adaptive step size

PDLP's Algorithm 2. Take a trial step with the current `eta`, get `(x', y')`, then:

```
eta_bar = ||z' - z||_omega^2  /  ( 2 * (y' - y)^T K (x' - x) )

eta_next = min( (1 - (k+1)^-0.3) * eta_bar ,  (1 + (k+1)^-0.6) * eta )
```

Accept the trial step if `eta <= eta_bar`, otherwise discard it and retry. Either way
set `eta <- eta_next`. The two exponents mean the step shrinks aggressively when the
local curvature says so, but can only grow slowly, and both effects damp out as `k`
grows so the sequence settles.

Note for the CUDA port: this line search is a sequential accept/reject with a
reduction inside it. cuPDLP.jl keeps it (the reductions are cheap on GPU); **cuPDLPx
throws it away** and uses a fixed `eta = 0.998 / ||K||_2` with `||K||_2` estimated by
power iteration, precisely because the search does not suit GPU parallelism. Plan:
implement adaptive first (it is what makes the CPU version robust), then A/B it
against fixed-step on GPU.

### 1.4 Primal weight

Updated only at restarts, by exponential smoothing in log space (theta = 0.5):

```
Delta_x = ||x^{n,0} - x^{n-1,0}||_2
Delta_y = ||y^{n,0} - y^{n-1,0}||_2

omega^n = exp( theta * log(Delta_y / Delta_x) + (1 - theta) * log(omega^{n-1}) )
```

Initial value `omega^0 = ||c||_2 / ||q||_2` (guard both norms against zero and fall
back to 1).

### 1.5 Restarts — the part that actually buys the speed

Plain averaged PDHG on an LP converges, slowly. Restarting the average is what turns
it into a linearly convergent method. PDLP restarts on a **normalized duality gap**:

```
rho_r^n(z) = (1/r) * max_{ ||z_hat - z||_omega <= r } { L(x, y_hat) - L(x_hat, y) }
mu_n(z, z_ref) = rho_{ ||z - z_ref||_omega }^n (z)
```

Restart when any of:

- **sufficient decay:** `mu(z_c, z^{n,0}) <= 0.1 * mu(z^{n,0}, z^{n-1,0})`
- **necessary decay + no local progress:** `mu(z_c, z^{n,0}) <= 0.9 * mu(...)` and
  `mu(z_c^{t+1}, ...) > mu(z_c^{t}, ...)`
- **long inner loop:** `t >= 0.5 * k`

Evaluating `rho_r` means solving a trust-region subproblem, which is fine on CPU and
bad on GPU. **cuPDLP replaces it with a weighted KKT error** — this is the version I
will implement, since I want one code path that runs on both:

```
KKT_omega(z) = sqrt(
      omega^2 * || ( A x - b , [h - G x]^+ ) ||_2^2
    + (1/omega^2) * || c - K^T y - lambda ||_2^2
    + ( q^T y + l^T lambda^+ - u^T lambda^- - c^T x )^2
)
```

where `lambda` is the bound-multiplier / reduced-cost vector. The three terms are
primal infeasibility, dual infeasibility, and the duality gap. Restart conditions with
cuPDLP's constants:

- `KKT_omega(z_c) <= 0.2 * KKT_omega(z^{n,0})`                      (beta_sufficient)
- `KKT_omega(z_c) <= 0.8 * KKT_omega(z^{n,0})` and the KKT error went up since the
  previous check                                                    (beta_necessary)
- `t >= 0.36 * k`                                                   (beta_artificial)

`z_c` is the restart candidate: whichever of the current iterate and the running
average has the lower KKT error. On restart, both `x` and `y` are set to `z_c`, the
average is reset, and `omega` is updated per 1.4.

### 1.6 Preconditioning

Applied once, to the scaled instance, before any iteration. Both are diagonal, so they
are just row and column scalings of `K` plus matching scalings of `c`, `q`, `l`, `u`.

- **Ruiz**, ~10 iterations, infinity norm:
  `D1_jj = sqrt(||K_{j,.}||_inf)`, `D2_ii = sqrt(||K_{.,i}||_inf)`
- then **Pock–Chambolle** with alpha = 1, 1-norm:
  `D1_jj = sqrt(||K_{j,.}||_1)`, `D2_ii = sqrt(||K_{.,i}||_1)`

PDLP does Ruiz first, then Pock–Chambolle. This is not optional — on badly scaled
Netlib instances the unscaled method effectively does not converge.

### 1.7 Termination

Relative criteria on the **unscaled** problem, with a single tolerance `eps`
(`1e-4` for the moderate-accuracy story, `1e-8` for the high-accuracy story):

```
|| ( A x - b , [h - G x]^+ ) ||_2        <= eps * (1 + ||q||_2)
|| c - K^T y - lambda ||_2               <= eps * (1 + ||c||_2)
| primal_obj - dual_obj |                <= eps * (1 + |primal_obj| + |dual_obj|)
```

I am writing these from the PDLP formulation as I understand it; I will pin the exact
constants against the paper again when I implement, and I will validate against known
optimal objective values (section 5) rather than trusting my own residuals.

### 1.8 Infeasibility detection

PDLP checks three sequences for approximate infeasibility certificates: the difference
of consecutive iterates, the normalized iterates, and the normalized running average,
each measured from the last restart. Same idea as OSQP's `delta y` / `delta x`
certificates (section 3.6). Low priority for us — the benchmark sets are feasible —
but cheap to add and it is a good robustness slide.

### 1.9 cuPDLPx — the newer variant, and whether to chase it

cuPDLPx replaces restarted **averaged** PDHG with restarted **Halpern** PDHG plus
reflection:

```
z^{k+1} = (k+1)/(k+2) * T(z^k) + 1/(k+2) * z^0
T = (1 + gamma) * PDHG - gamma * Id,   gamma in [0, 1]
```

Fixed step size `eta = 0.998 / ||K||_2`. Restart on the fixed-point residual
`||z - PDHG(z)||_P`. Primal weight driven by a PID controller instead of exponential
smoothing. Reported: 2.5x–5x over cuPDLP on 383 MIPLIB 2017 LP relaxations, 3x–6.8x on
Mittelmann, on an H100 with CUDA 12.4.

My read: implement restarted-average first because it is the well-documented baseline
and I can check it against published behaviour. Halpern + reflection is a ~30-line
change on top of a working PDHG loop, and if it lands it is a genuinely strong slide
("we implemented the 2025 variant, here is the ablation"). Treat it as a stretch, not
a dependency.

---

## 2. LP by revised simplex

PDHG gives moderate accuracy fast on big problems. Simplex gives exact vertex
solutions, clean 1e-9 objectives on the classic Netlib set, and — critically — a
**basis**, which is what makes branch-and-bound cheap (warm-started dual simplex at
every node). I need both. Simplex is the correctness backbone; PDHG is the headline.

Sources:

- Huangfu, Hall, *Novel update techniques for the revised simplex method*, COAP 2015 —
  [preprint](https://optimization-online.org/wp-content/uploads/2013/02/3774.pdf),
  plus Hall's ERGO talk
  [slides](https://webhomes.maths.ed.ac.uk/hall/ERGO_301116/ERGO_301116.pdf) which I
  read in full — this is the HiGHS lineage
- Huangfu, Hall, *Parallelizing the dual revised simplex method*, MPC 2018 —
  [arXiv:1503.01889](https://arxiv.org/pdf/1503.01889)
- Koberstein, *The dual simplex method: techniques for a fast and stable
  implementation*, PhD thesis, Paderborn 2005 — the canonical practical reference for
  Harris ratio test + bound flipping + cost shifting integrated together

### 2.1 Computational form

```
min c^T x   s.t.  A x = b,  l <= x <= u
```

Every MPS row gets a **logical** (slack) variable with bounds derived from the row
type, so `A` becomes `[A_struct  -I]` and every row is an equality. Free variables get
infinite bounds. This is the standard bounded-variable form and it means the code
never special-cases row types after the reader.

Partition columns into basic `B` (m of them) and nonbasic `N`. Then:

```
x_B = B^-1 (b - N x_N)          b_hat = B^-1 b
c_hat_N = c_N - c_B^T B^-1 N    (reduced costs)
```

Primal feasible when `b_hat` is within bounds; dual feasible when reduced costs have
the right sign for which bound each nonbasic sits at.

### 2.2 The four kernels

Every simplex implementation is these four operations plus bookkeeping:

| Kernel | What it computes | Note |
|---|---|---|
| `BTRAN` | `B^T pi_p = e_p` | gives the pivotal row multipliers |
| `PRICE` | `a_hat_p^T = pi_p^T N` | the pivotal row; the expensive one |
| `FTRAN` | `B a_hat_q = a_q` | the pivotal column |
| `INVERT` | refactorize `B = LU` | periodically, or on accuracy failure |

Plus `UPDATE`: patch the factored form after a basis change instead of refactorizing.

### 2.3 Basis factorization and update

`INVERT`: sparse LU with **Markowitz pivoting** and a threshold stability test — pick
the pivot minimising `(r_i - 1)(c_j - 1)` (row and column counts) among candidates
satisfying `|a_ij| >= u_thresh * max_col`, with `u_thresh` around 0.1. Exploit the
triangular structure first (the standard singleton-row/singleton-column peeling), which
on most LP bases removes the large majority of the matrix before any real elimination
happens.

`UPDATE`, in increasing order of effort:

- **Product form (PFI)**, Dantzig & Orchard-Hays 1954. `B_bar = B E` where
  `E = I + (a_hat_q - e_p) e_p^T`. Solving with `B_bar` is a solve with `B` followed by
  one eta-vector sweep: `x_p := x_p / mu; x := x - x_p * eta`. Twenty lines of code.
  Fill grows every iteration, so refactorize often.
- **Forrest–Tomlin (1972)**: substitute `B = LU`, take `L` out on the left, leaving a
  spiked upper triangular `U' = U + (a_tilde_q - u_p) e_p^T`; eliminate the spike. Much
  better fill behaviour, and it is what HiGHS/Xpress-class codes use.

Plan: PFI first (get the thing solving Netlib), Forrest–Tomlin second if time allows.
Refactorize every 50–100 basis changes, or immediately when a residual check fails.

### 2.4 Which algorithm: dual, not primal

Hall's slides are blunt about it: the dual simplex is the preferred variant. Two
reasons that matter to me:

1. After a branch-and-bound bound change, the basis stays **dual** feasible but not
   primal feasible — so the dual simplex re-optimises in a handful of iterations while
   the primal would restart. This is the entire reason B&B is affordable.
2. On degenerate problems the dual with a good ratio test is far more stable.
   Netlib's own README quotes Maros on MODSZK1: a dual simplex "may require up to 10
   times" fewer iterations than primal on that instance.

I still need a primal simplex for phase 1 / fallback, but the dual is the main path.

Dual iteration: scan `b_hat_i` outside its bounds to pick the leaving row `p`; scan
`c_hat_j / a_hat_pj` over the pivotal row to pick the entering column `q`.

### 2.5 Pricing: dual steepest edge

Choosing the leaving row by largest infeasibility is bad. **Dual steepest edge (DSE)**
weights each candidate `b_hat_i` by `w_i`, an estimate of `||B^-T e_i||_2^2`, and picks
the largest `infeasibility_i^2 / w_i`. Costs one extra FTRAN per iteration
(FTRAN-DSE, solving with `pi_p`) and typically cuts iteration counts by a large factor.

Cheaper alternative: **dual Devex**, which costs nothing to initialise but is less
effective. Hall's own summary of the tradeoff: DSE is expensive to initialise when the
starting basis is not `I`, Devex is free but weaker. Plan: Devex first (simpler,
always works), DSE once the rest is stable.

### 2.6 Ratio test: Harris two-pass + bound flipping

This is where degeneracy is won or lost, and it is the single most important stability
decision in the whole simplex code.

- **Harris two-pass**: pass 1 computes the max step allowed if every variable may
  violate its bound by a small tolerance `delta`; pass 2 picks, among all candidates
  passing that relaxed limit, the one with the **largest pivot magnitude**. Trading a
  tiny bound violation for a big pivot is what stops the factorization from degrading.
- **Bound-flipping ratio test (BFRT)**: for boxed variables, a dual variable may pass
  through zero and change sign as long as the corresponding primal variable flips from
  one finite bound to the other. So instead of stopping at the first blocking
  candidate, keep walking, accumulating flips, while the dual objective still
  increases. Needs one extra FTRAN on the aggregated flip set (FTRAN-BFRT). Big
  iteration-count win on models with many boxed variables.
- **Cost shifting** (Koberstein): perturb costs to escape dual degeneracy, and remove
  the perturbation at the end.

Koberstein's thesis is exactly the "how do these three interact" reference; I will
work from it when I write the ratio test.

### 2.7 Dual phase 1

Getting a dual feasible start. Options, cheapest first: use a "big-M"/composite
objective; use the artificial-bound / subproblem approach; use Pan's method.
Koberstein's conclusion was that Pan's method combined with the subproblem approach beat
the alternatives in his tests. For our scope, the simplest workable choice is the
composite objective, and I will only go further if it actually blocks instances.

### 2.8 Hyper-sparsity

On many real LPs the vectors in BTRAN/FTRAN/PRICE are themselves extremely sparse —
Hall's 2005 result. Exploiting it means solving `Bx = r` with a graph-traversal that
touches only the nonzero pattern, instead of a dense sweep. This is a serious win on
network-like models (which is most refinery and supply-chain LPs). It is also a nice,
self-contained optimisation to demo. Stretch goal, not phase 1.

---

## 3. QP by ADMM (OSQP)

Source: Stellato, Banjac, Goulart, Bemporad, Boyd, *OSQP: An Operator Splitting Solver
for Quadratic Programs*, Mathematical Programming Computation 12(4), 2020 —
[arXiv:1711.08013](https://arxiv.org/pdf/1711.08013). I read the paper text directly,
formulas below are from it.

### 3.1 Problem form

```
min  (1/2) x^T P x + q^T x
s.t. l <= A x <= u
```

`P` positive semidefinite. Two-sided bounds cover equalities (`l_i = u_i`), one-sided
rows and box constraints uniformly, so the QPS reader maps straight onto it. No
requirement that `P` be positive definite or `A` full rank — that robustness is one of
the paper's selling points and a good line for the deck.

### 3.2 The iteration (Algorithm 1, verbatim structure)

Given `x^0, z^0, y^0` and parameters `rho > 0`, `sigma > 0`, `alpha` in `(0,2)`:

```
1.  solve  [ P + sigma*I    A^T     ] [ x_tilde^{k+1} ]   [ sigma*x^k - q      ]
           [ A            -rho^-1 I ] [ nu^{k+1}      ] = [ z^k - rho^-1 y^k   ]

2.  z_tilde^{k+1} = z^k + rho^-1 (nu^{k+1} - y^k)
3.  x^{k+1}       = alpha * x_tilde^{k+1} + (1 - alpha) * x^k
4.  z^{k+1}       = Pi_[l,u]( alpha * z_tilde^{k+1} + (1 - alpha) * z^k + rho^-1 y^k )
5.  y^{k+1}       = y^k + rho * ( alpha * z_tilde^{k+1} + (1 - alpha) * z^k - z^{k+1} )
```

Steps 2–5 are vector arithmetic and a clamp — trivially parallel, GPU-friendly. The
only real work is step 1.

The KKT matrix in step 1 is **quasi-definite** and — this is the key structural point —
**constant across iterations** as long as `rho` does not change. So: factorize once
(LDL^T, no pivoting needed for quasi-definite matrices), then every iteration is two
triangular solves. Refactorize only when `rho` is updated.

That gives me a clean reuse story with the simplex work: I need a sparse LDL^T with
AMD-style fill-reducing ordering here, and a sparse LU with Markowitz there. Same
sparse-matrix infrastructure, same graph code.

### 3.3 Parameters

From the paper's defaults:

```
sigma = 1e-6          (regularization, makes step 1 solvable even if P is singular)
alpha = 1.6           (over-relaxation; the paper says 1.5-1.8 works, 1.6 is default)
rho_bar_0 = 0.1
```

`rho` is a **diagonal matrix, not a scalar** — per-constraint:

```
rho_i = rho_bar          if l_i != u_i
rho_i = 1e3 * rho_bar    if l_i == u_i   (equalities are always active at the optimum)
```

Adaptive `rho` update, driven by the ratio of relatively-scaled residuals:

```
rho_bar^{k+1} = rho_bar^k * sqrt(
    ( ||r_prim||_inf / max{||A x||_inf, ||z||_inf} )
  / ( ||r_dual||_inf / max{||P x||_inf, ||A^T y||_inf, ||q||_inf} )
)
```

Because updating `rho` forces a new numerical factorization (the sparsity pattern is
unchanged, so the symbolic factorization is reused), OSQP only updates when it is worth
it: accumulated iteration time exceeds ~40% of factorization time **and** the new value
is 5x larger or smaller than the current one.

### 3.4 Termination

```
||r_prim||_inf <= eps_prim,     r_prim = A x - z
||r_dual||_inf <= eps_dual,     r_dual = P x + q + A^T y

eps_prim = eps_abs + eps_rel * max{ ||A x||_inf, ||z||_inf }
eps_dual = eps_abs + eps_rel * max{ ||P x||_inf, ||A^T y||_inf, ||q||_inf }
```

Defaults `eps_abs = eps_rel = 1e-3`, `eps_pinf = eps_dinf = 1e-4`. For benchmarking I
will tighten to 1e-6 / 1e-8 and report the tolerance alongside every number — quoting
a time without its tolerance is the classic way to look like you are cheating.

### 3.5 Scaling and polishing

**Modified Ruiz equilibration** (Algorithm 2 in the paper) on the symmetric matrix
`M = [[P, A^T], [A, 0]]`, with an extra cost-scaling step the original Ruiz does not
have:

```
repeat until ||1 - delta||_inf <= eps_equil:
    delta_i <- 1 / sqrt( ||M_i||_inf )        for i = 1..n+m
    scale P, q, A, l, u by diag(delta)
    gamma <- 1 / max{ mean_i(||P_i||_inf), ||q||_inf }      # cost scaling
    P <- gamma*P ;  q <- gamma*q
    S <- diag(delta) * S ;  c <- gamma * c
```

Termination criteria are then evaluated on the **unscaled** residuals, using the stored
`D`, `E`, `c` factors — the paper gives the unscaled forms explicitly and I will follow
them, because getting this wrong is a silent-wrong-answer bug.

**Polishing** is what turns a 1e-3 ADMM answer into a 1e-9 answer, and it is the reason
I can claim accuracy comparable to an active-set or IPM solver. Guess the active set
from the dual signs:

```
L = { i : y_i < 0 }   (active at lower)
U = { i : y_i > 0 }   (active at upper)
```

then solve the small regularized system

```
[ P + delta*I    A_L^T     A_U^T   ] [ x_hat  ]   [ -q  ]
[ A_L          -delta*I            ] [ y_hat_L ] = [  l_L ]
[ A_U                    -delta*I  ] [ y_hat_U ]   [  u_U ]
```

with `delta ~ 1e-6`, followed by ~3 steps of iterative refinement against the
unregularized system (forward/back solves only, no new factorization). Accept the
polished point only if it satisfies the optimality conditions; otherwise keep the ADMM
iterate. Never returns something worse — the guess is checked.

Paper's own numbers: polishing succeeds in ~83% of cases at high accuracy and costs
roughly 22–32% of solve time. On badly scaled degenerate problems it usually fails,
which is honest and is itself a data point for the robustness slide.

### 3.6 Infeasibility certificates

The iterate differences `(dx, dz, dy) = (x^k - x^{k-1}, ...)` converge even when the
iterates do not. `dy` certifies primal infeasibility, `dx` certifies dual
infeasibility, both up to tolerances `eps_pinf`, `eps_dinf`. Cheap to implement (it is
just the deltas I already have) and OSQP's paper makes a point of being the first
operator-splitting QP method to do it reliably. Good slide, low cost.

### 3.7 GPU note

Steps 2–5 are pure elementwise work. Step 1 is a factorize-once/solve-many, which does
**not** GPU-accelerate well at our scale. If I want a GPU QP story, the route is the
indirect variant: replace the direct KKT solve with conjugate gradient on
`(P + sigma I + rho A^T A) x = rhs`, which is SpMV-only and lands on exactly the same
CUDA kernels as the PDHG path. Worth noting on a slide; only build it if PDHG lands
early.

---

## 4. MILP: branch and bound, cuts, presolve, heuristics

Realistic framing first. Modern MILP performance is decades of engineering — presolve,
cut families, branching rules, node selection, conflict analysis, restarts, symmetry.
I am not going to be competitive on hard MIPLIB 2017 instances and I will not pretend
otherwise. What I *can* build is a correct, warm-started branch-and-cut that closes
easy-to-medium instances and proves optimality, and be explicit about where the wall
is. A jury that knows the field will respect a bounded honest claim far more than a
vague broad one.

### 4.1 Branch and bound skeleton

- Solve the LP relaxation with the dual simplex.
- If integral, done. Otherwise pick a fractional integer variable `x_j` with value
  `v`, create two children: `x_j <= floor(v)` and `x_j >= ceil(v)`.
- Each child re-solves with a **warm-started dual simplex** from the parent basis. A
  bound change keeps dual feasibility, so this is typically a few dozen iterations
  rather than a fresh solve. This is the whole game.
- Prune by bound (`node_lb >= incumbent - epsilon`), by infeasibility, by integrality.

**Branching rule.** Escalating in cost:
- most-fractional: bad, but 10 lines, use it to get the tree correct first;
- **pseudocost**: keep per-variable running averages of objective degradation per unit
  of fractionality, from actual past branchings. Cheap, much better.
- **reliability branching**: do real strong branching (trial dual-simplex solves on
  both children, limited iterations) at the top of the tree until a variable has been
  branched on enough times to trust its pseudocost, then fall back to pseudocost.
  This is the current standard and it is what I should land on.

**Node selection.** Hybrid, which is what everyone does: depth-first (LIFO) early to
find an incumbent fast so pruning can start, then best-bound to close the gap and to
make the reported dual bound meaningful. Best-bound alone explodes memory; depth-first
alone gives a useless bound for ages.

Reference for the survey of these: [arXiv:2111.06257](https://arxiv.org/pdf/2111.06257),
*Branch and Bound in MILP: A Survey of Techniques and Trends*.

### 4.2 Gomory mixed-integer cuts

The one cut family with the best value-per-line ratio, because it comes straight out of
the simplex tableau I already have. Take a tableau row whose basic variable `x_i` is an
integer variable sitting at a fractional value:

```
x_i + sum_{j in N} a_bar_ij * x_j = b_bar_i
f0 = b_bar_i - floor(b_bar_i)   in (0,1)
fj = a_bar_ij - floor(a_bar_ij)
```

The GMI cut (nonbasics at lower bound zero):

```
sum_{j int, fj <= f0}    fj * xj
+ sum_{j int, fj >  f0}  f0*(1-fj)/(1-f0) * xj
+ sum_{j cont, a_bar>0}  a_bar_ij * xj
+ sum_{j cont, a_bar<0} -f0/(1-f0) * a_bar_ij * xj
>= f0
```

Four coefficient cases: integer nonbasics split at `f0`, continuous nonbasics split by
sign. I am writing this from standard references (Cornuéjols' and Mitchell's
derivations; the [ojAlgo writeup](https://www.ojalgo.org/2022/04/gomory-mixed-integer-cuts/)
reproduces both forms). **I could not find the four-case formula in a citable
plaintext source I fully verified**, so before trusting it in code I will validate the
generated cuts numerically: every generated cut must (a) be violated by the current LP
point and (b) not cut off a known optimal integer solution on instances where I have
the reference solution from MIPLIB's `.solu` file.

Practical cautions that matter more than the formula:
- nonbasics at their **upper** bound need the complement transform first
  (`x_j -> u_j - x_j`), and slack/logical variables must be tracked back to the
  original rows. Getting this bookkeeping wrong is the standard way GMI
  implementations silently produce invalid cuts.
- GMI cuts get numerically dangerous fast when applied in rounds. Standard defence:
  drop cuts with dynamic range (max|coef| / min|coef|) above ~1e6, require a minimum
  violation, and only apply cuts at the root plus maybe a few levels.
- **Root-only cut loop** is the sane scope: a few rounds at the root, then pure B&B.

MIR cuts are the natural second family and generalise GMI; the same Cornell page gives
the base MIR form. Knapsack cover cuts are the third. Only if time allows.

### 4.3 Presolve

Cheap, high-yield, and easy to demo ("this model shrank 40% before we solved
anything"). The standard easy set:

- remove empty/singleton rows, fixed variables, free column singletons;
- tighten bounds from row activity (bound propagation), repeat to a fixed point;
- coefficient tightening on integer variables;
- detect and remove redundant / forcing constraints;
- for MILP: round bounds on integer variables to integers after propagation.

Every reduction must have a matching **postsolve** step to map the reduced solution
back. Building presolve and postsolve as a stack of undo records from day one is much
easier than retrofitting it.

### 4.4 Primal heuristics

Just enough to get an incumbent early so pruning works:

- **simple rounding / fixing**: round the LP relaxation, check feasibility;
- **diving**: repeatedly fix the least-fractional variable and re-solve the LP with
  dual simplex, bounded depth;
- **feasibility pump** if there is time — it is well documented and genuinely effective.

### 4.5 What makes MIPLIB instances hard

Not size. The three things that actually kill you: a **weak LP relaxation** (large
integrality gap, so bounding barely prunes), **symmetry** (the tree explores thousands
of equivalent solutions), and **numerical trouble** (wide coefficient ranges making
bounds untrustworthy). MIPLIB 2017's own paper notes that of 629 instances classed as
easy, bounded and feasible, 119 could not be solved within 24 hours on standard
hardware — so "easy" in MIPLIB terms is not easy for us. See
[MIPLIB 2017 paper](https://www.or.rwth-aachen.de/files/research/repORt/MIPLIB2017.pdf).

Consequence for instance selection: I pick MILP instances by *measured* solve time with
a reference solver, not by the MIPLIB difficulty tag. See section 5.3.

---

## 5. Benchmarks: the exact instances, and the plumbing to get them

I checked every source below is live and downloadable from this machine on 22 Aug 2026.

### 5.1 Netlib LP — correctness backbone

Source: <https://netlib.org/lp/data/>. **The files are not MPS.** They are stored in a
compressed format and must be expanded with `emps`. I verified the full pipeline
already:

```bash
curl -sS https://netlib.org/lp/data/emps.c -o emps.c && cc -O2 -w -o emps emps.c
curl -sS https://netlib.org/lp/data/afiro  -o afiro
./emps afiro > afiro.mps        # 83 lines of standard MPS, correct
```

The larger files are additionally `.Z`-compressed (`zcat file.Z | emps > file.mps`),
and the Kennington instances live in `lp/data/kennington/`.

The README at `lp/data/readme` carries a reference table of rows / cols / nonzeros /
**optimal objective value** for every instance. That table is my correctness oracle —
I check my objective against it, not against my own residual. Values I pulled:

| Instance | Rows | Cols | Nonzeros | Optimal value | Why it is in the set |
|---|---:|---:|---:|---|---|
| AFIRO | 28 | 32 | 88 | -4.6475314286E+02 | smallest thing that exists; first light |
| SC50A | 51 | 48 | 131 | -6.4575077059E+01 | tiny, second smoke test |
| ADLITTLE | 57 | 97 | 465 | 2.2549496316E+05 | tiny |
| BLEND | 75 | 83 | 521 | -3.0812149846E+01 | a blending model — on-theme for MRPL |
| SHARE1B | 118 | 225 | 1182 | -7.6589318579E+04 | classic bad conditioning |
| STOCFOR1 | 118 | 111 | 474 | -4.1131976219E+04 | stochastic, structured |
| ISRAEL | 175 | 142 | 2358 | -8.9664482186E+05 | dense-ish rows |
| BANDM | 306 | 472 | 2659 | -1.5862801845E+02 | mid |
| SCTAP1 | 301 | 480 | 2052 | 1.4122500000E+03 | traffic assignment, network-like |
| SCFXM1 | 331 | 457 | 2612 | 1.8416759028E+04 | mid |
| **DEGEN2** | 445 | 534 | 4449 | -1.4351780000E+03 | **degeneracy demo** |
| FIT1P | 628 | 1677 | 10894 | 9.1463780924E+03 | bounded, many boxed vars — BFRT demo |
| 25FV47 | 822 | 1571 | 11127 | 5.5018458883E+03 | the classic "is your simplex real" test |
| WOODW | 1099 | 8405 | 37478 | 1.3044763331E+00 | wide, hyper-sparse |
| **DEGEN3** | 1504 | 1818 | 26230 | -9.8729400000E+02 | **degeneracy demo, larger** |
| STOCFOR2 | 2158 | 2031 | 9492 | -3.9024408538E+04 | mid-large |
| GREENBEA | 2393 | 5405 | 31499 | -7.2462405908E+07 | large, nasty bounds |
| **PILOT87** | 2031 | 4883 | 73804 | 3.0171072827E+02 | **ill-conditioning demo** |
| MAROS-R7 | 3137 | 9408 | 151120 | 1.4971851665E+06 | largest classic Netlib; banded |
| DFL001 | 6072 | 12230 | 41873 | 1.12664E+07 | large, notoriously hard to nail |

The robustness claims in section 0 need evidence, and Netlib's README supplies it
directly. MINOS 5.3 degeneracy statistics quoted there:

```
DEGEN2   1075 steps,  610 degenerate  (56.7%)
DEGEN3   6283 steps, 3299 degenerate  (52.5%)
CYCLE    3156 steps, 1485 degenerate  (47.1%)
WOOD1P   1059 steps,  471 degenerate  (44.5%)
```

and on ill-conditioning, Irv Lustig via the same README: PILOT87 is considered harder
than PILOT "because of the bad scaling in the numerics." That is a citable, third-party
justification for exactly the instances I pick for the robustness slide — much stronger
than me asserting an instance is degenerate.

Also flagged there: DFL001's optimal value is footnoted as hard to reproduce (MINOS
only got there from a LOAD file). Good instance to *discuss*, bad instance to stake a
correctness claim on.

Two further notes for the deck: MODSZK1 is described in the README as "very degenerate"
with dual simplex needing up to 10x fewer iterations than primal — a ready-made
justification for the dual-first design. And BLEND being a blending model gives me a
Netlib instance that is literally on MRPL's topic.

### 5.2 Large sparse LPs — where GPU PDHG can actually win

This is the important find of this research phase. Mittelmann's
**[LPfeas benchmark](https://plato.asu.edu/ftp/lpfeas.html)** (updated 10 Aug 2026) is
subtitled "find PD feasible point; **also for GPUs**" and benchmarks GPU first-order
solvers head to head with simplex/barrier codes. Instances live at
<https://plato.asu.edu/ftp/lptestset/> as `.mps.bz2`.

Extract from the 10 Aug 2026 results (seconds; `t` = timeout, tolerance 1e-6), for
HiGHS 1.15.0 versus the GPU first-order codes cuPDLPx and cuOpt:

```
                HiGHS   cuPDLPx  cuOpt
Linf_520c        1063        2      2
s82               261      611     53
thk_48              t       38     37
dlr1              218       16     14
stat96v2          198        8      8
storm_1000        270        3      3
rail4284           96       27      4
```

and the scaled shifted geometric means over all 65 problems: HiGHS 16.9, cuPDLPx 2.08,
cuOpt 1.0. HiGHS solved 55/65, cuPDLPx 57/65, cuOpt 62/65.

What this means for us: **on large sparse LPs at moderate tolerance, a GPU PDHG solver
genuinely beats HiGHS, often by one to two orders of magnitude.** That is not a claim I
have to fudge — it is the published state of the field, and if my implementation gets
anywhere near the right shape, my own benchmark table will show the same direction. The
PS asks for a comparison against "at least one established solver". This is how the
comparison ends up favourable *without* lying about hard MILP.

Candidate large instances to target (all in `lptestset/`, chosen for size and for
appearing in the published table so I have a sanity reference):
`fome13`, `qap15`, `nug08-3rd`, `pds-100`, `rail02`, `rail4284`, `datt256_lp`,
`graph40-40`, `storm_1000`, `cont1`, `cont11`, `s100`, `savsched1`, `supportcase10`,
`woodlands09`, `stat96v2`, `L2CTA3D`, `degme`, `scpm1`.

`cont1` / `cont11` are PDE-constrained and are known bad cases for first-order methods
(PDLP takes 256s / 1988s there while HiGHS takes 6s / 14s) — I should include one of
them deliberately, report that I lose on it, and explain why. Reporting a loss you
understand is the strongest credibility move available in a benchmark talk.

### 5.3 MIPLIB 2017 — MILP

Source: <https://miplib.zib.de/>. Verified live:

- benchmark instance list (240 names): `https://miplib.zib.de/downloads/benchmark-v2.test`
- collection list: `https://miplib.zib.de/downloads/collection-v1.test`
- **reference solutions**: `https://miplib.zib.de/downloads/miplib2017-v29.solu`
  — lines look like `=opt=  50v-10  3311.1799841`. This is the MILP correctness oracle,
  same role the Netlib README table plays for LP.
- full benchmark archive is a 332 MB zip; individual instances are far cheaper to pull.

Selection method — and I want to be disciplined about this, because MIPLIB "easy" is
not easy (section 4.5): **do not pick by tag.** Instead, run HiGHS over the collection
with a 10-second limit, keep everything HiGHS closes in under ~5 seconds, then pick 8–12
of those spanning set-covering / scheduling / knapsack-ish / network structures. That
gives a defensible set ("these are instances a reference solver closes in seconds; here
is where we land relative to it") and it is reproducible by the jury.

I will publish the selection script and the resulting list, so the set does not look
cherry-picked. It *is* selected for difficulty — but stating the selection rule up front
is the difference between curation and cheating.

### 5.4 QP — Maros–Meszaros

Source: <http://www.doc.ic.ac.uk/~im/00README.QP> (Istvan Maros's page, verified live),
data in `QPDATA1/2/3` zips; mirror `ftp://ftp.sztaki.hu/pub/oplab/QPDATA`. 138 convex QP
problems in QPS format:

```
min c0 + c^T x + (1/2) x^T Q x,  Q symmetric PSD
s.t. A x = b,  l <= x <= u
```

The README's table gives M, N, NZ, number of quadratic variables, off-diagonal
nonzeros in the lower triangle of Q, and a reference optimal value from BPMPD. Same
oracle role again. Instances I picked:

| Instance | Rows | Cols | A nnz | Q offdiag nnz | Reference optimum | Role |
|---|---:|---:|---:|---:|---|---|
| QPTEST | 2 | 2 | 4 | 1 | 4.3718750e+00 | 2-variable first light |
| HS21 | 1 | 2 | 2 | 0 | -9.9960000e+01 | trivial, separable |
| HS76 | 3 | 4 | 10 | 2 | -4.6818182e+00 | trivial, non-separable |
| QAFIRO | 27 | 32 | 83 | 3 | -1.5907818e+00 | QP version of AFIRO |
| QSHARE1B | 117 | 225 | 1151 | 21 | 7.2007832e+05 | ill-conditioned lineage |
| DUAL1 | 1 | 85 | 85 | 3473 | 3.5012966e-02 | dense Q, single row |
| DUALC1 | 215 | 9 | 1935 | 36 | 6.1552508e+03 | many rows, few cols |
| PRIMAL1 | 85 | 325 | 5815 | 0 | -3.5012965e-02 | separable dual of DUAL1 |
| CVXQP1_S | 50 | 100 | 148 | 286 | 1.1590718e+04 | the CVXQP family, small |
| CVXQP1_M | 500 | 1000 | 1498 | 2984 | 1.0875116e+06 | medium; known hard for ADMM |
| MOSARQP1 | 700 | 2500 | 3422 | 45 | -9.5287544e+02 | sparse, mid-size |
| QSCTAP1 | 300 | 480 | 1692 | 117 | 1.4158611e+03 | QP over a network LP |
| AUG2DC | 10000 | 20200 | 40000 | 0 | 1.8183681e+06 | large separable, good scaling demo |
| CONT-050 | 2401 | 2597 | 12005 | 0 | -4.5638509e+00 | PDE-flavoured, ill-conditioned |
| LISWET1 | 10000 | 10002 | 30000 | 0 | 3.6122402e+01 | large, degenerate-ish |
| QSHIP08L | 778 | 4283 | 12802 | 34025 | 2.3760406e+06 | larger Q |

Note the PS names **QPLIB**, not Maros–Meszaros. QPLIB is mostly *nonconvex* QP/QCQP —
out of scope for an ADMM convex-QP solver. I will use Maros–Meszaros as the main set
(it is the standard convex-QP benchmark and is what OSQP itself is benchmarked on), and
say in one line why QPLIB is the wrong set for a convex QP core. Answering that
gracefully if a jury member asks is better than silently substituting.

### 5.5 The refinery case (the one my brief was missing)

The PS explicitly lists "refinery scheduling, crude blending, production planning and
supply chain optimization case studies from open literature" as expected data. MRPL is
the proposing organisation. I need at least one model in the deck that a refinery
engineer recognises.

Plan: build a **multi-period crude blending + production planning model** as MPS from
open-literature parameters — LP version first, MILP version with binary
crude-selection / mode-switching second. The relevant literature is well documented;
Grossmann's group at CMU publishes complete formulations, e.g.
[Modeling for Integrated Refinery Planning with Crude-oil Scheduling](https://egon.cheme.cmu.edu/Papers/Su_IntegratedRefineryPlanning.pdf).
Note that true blending is bilinear (the pooling problem) and therefore nonconvex —
the industry standard workaround is a linear or piecewise-linear approximation, and the
piecewise-linear version is a MILP, which is squarely in scope. See
[Computational Experience with Piecewise Linear Relaxations for Petroleum Refinery
Planning](https://www.mdpi.com/2227-9717/9/9/1624).

This is a scope addition versus my original brief and I think it is the single highest
-leverage one: it converts "a solver that solves academic benchmarks" into "a solver
that solves *your* problem".

(File names in the Maros–Meszaros archives are lowercase with a `.qps` extension:
`cvxqp1_s.qps` appears as `cvxqp1s.qps`, `qship08l.qps`, `aug2dc.qps`, and so on. The
files are DOS line-ended — unzip with `unzip -aa`.)

### 5.6 HiGHS already ships a PDLP — I need to handle this head on

Found while checking HiGHS's MPS reader: HiGHS has a `highs/pdlp/` directory containing
a **vendored copy of cuPDLP-C** (including a `cuda/` subdirectory) plus its own `hipdlp`
implementation. So HiGHS 1.15 can solve LPs with the same first-order algorithm I am
planning to write.

This cuts two ways and I would rather deal with it now than be surprised by it in a
Q&A.

**The good half — it hands me a debugging oracle I did not expect.** `highs` with the
PDLP solver selected runs the *same algorithm* on the *same instance*. If my iteration
counts are wildly different from theirs, I have a bug, and I will know which phase
introduced it. Nothing else I have gives me that. The harness will therefore run
**three** baselines, not one:

1. `highs` default (dual simplex) — the "established solver" the PS asks for;
2. `highs` with PDLP — same-algorithm reference, for correctness and iteration counts;
3. mine.

**The awkward half — "HiGHS already does this."** The honest answers, in order:

- The PS does not ask for a novel algorithm. It asks for a **sovereign core built from
  scratch from mathematical foundation**, explicitly not built on an existing solver
  library. HiGHS's PDLP is vendored third-party C code sitting inside a solver library —
  which is exactly the dependency the PS exists to remove.
- Benchmarking honesty: against HiGHS's **default** (dual simplex) on large sparse LPs I
  expect to win, and that is the comparison the PS asks for. Against HiGHS's **PDLP** I
  am competing with a mature implementation of my own algorithm and I will probably
  lose. **I am going to report both columns.** A table that quietly compares only
  against the baseline I beat is the kind of thing an engineering jury spots, and once
  they spot it nothing else in the deck is trusted.
- The differentiators that survive that comparison: the full stack is ours end to end
  (reader, scaling, kernels, simplex, QP, MILP), our GPU kernels are hand-written rather
  than cuSPARSE, and the refinery model is ours. "We wrote all of it" is the claim, not
  "we invented it".

---

## 6. GPU environment

The blocker: this machine is Apple Silicon. No CUDA, no NVIDIA GPU, ever. The GPU path
has to be developed and demoed somewhere else, and that "somewhere else" has to still
exist and be reachable during a 36-hour finale at a nodal centre — which is a different
requirement from "somewhere I can develop".

### 6.1 Options, with what I found

| Option | GPU | Cost | Notes |
|---|---|---|---|
| Kaggle Notebooks | P100 or 2x T4 | free | **~30 GPU-hours/week, published quota**, no card needed |
| Google Colab free | T4 16GB | free | quota unpublished, roughly 15–30 h/week, varies with demand; 12h session cap, ~90 min idle disconnect |
| Colab Pro | T4/L4 | subscription | more reliable allocation |
| Vast.ai | RTX 4090 etc. | ~$0.09–0.59/hr (spot from ~$0.35) | cheapest; peer-to-peer; **spot instances can be reclaimed on 15 s notice** |
| RunPod | RTX 4090 | ~$0.34–0.69/hr | Secure Cloud tier ~99% uptime, ~5 min setup, $10 signup credit |
| Lambda | A100 80GB / H100 | ~$2.06 / ~$2.99 per hr | overkill for us |
| College / IIT cluster | varies | free | best if it exists; needs a person and a queue |

Sources: [getdeploying GPU price index](https://getdeploying.com/gpus),
[Northflank cheapest cloud GPU providers 2026](https://northflank.com/blog/cheapest-cloud-gpu-providers),
[Thunder Compute on Colab alternatives, Aug 2026](https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025).

### 6.2 Recommendation

**Develop on Kaggle, demo on RunPod, keep a recorded fallback.**

- **Kaggle for development.** The quota is published and predictable (~30 h/week) rather
  than Colab's undisclosed and demand-dependent allocation, it hands out a P100 or two
  T4s without a credit card, and `nvcc` works. 30 h/week is plenty for kernel work —
  actual GPU runs are seconds, the time goes on compiles and uploads.
- **RunPod Secure Cloud for the finale demo.** Not Vast.ai: a spot instance being
  reclaimed on 15 seconds notice during a live demo is exactly the failure I cannot
  accept. RTX 4090 at ~$0.34–0.69/hr means a full 36-hour finale window costs under
  ₹2,500 even if I leave it running, and the $10 signup credit covers most testing.
- **Recorded fallback, always.** A pre-recorded terminal capture of the GPU run plus
  the committed logs and result CSVs. If the venue network dies, the demo does not.
  This is not optional; it is the single most likely thing to go wrong on the day.

### 6.3 Keeping the code portable

The architecture has to make the Mac a first-class dev target, not a degraded one:

- CPU core in plain C++20, no CUDA in any header the CPU build sees;
- one narrow interface — something like `LinAlgBackend` with `spmv`, `spmv_transpose`,
  `axpy`, `dot`, `nrm2`, `clamp_project`, `max_zero_project` — with a CPU implementation
  and a CUDA implementation;
- the PDHG loop is written once against that interface, so the *algorithm* is verified
  on the Mac and only the *kernels* need the NVIDIA box;
- CMake option `SOLVER_ENABLE_CUDA=OFF` by default; CI-equivalent script builds both.

The payoff: I can build, test and debug the entire solver on this machine, and the
GPU box is only needed for timing runs. That inverts the usual "I can't work without
the GPU" failure mode.

Data movement is a non-issue: upload the scaled instance once, download the solution
once. cuPDLP.jl does exactly this and keeps preconditioning and I/O on the host.

### 6.4 What runs on GPU

Following cuPDLP.jl: the whole PDHG main loop — SpMV in CSR, the projections, the norms
and reductions, the adaptive step size, the restart checks, the primal weight update.
Host keeps: file reading, presolve, preconditioning, infeasibility certificate checks,
final unscaling. Two host-device transfers total.

Kernels I have to write: CSR SpMV, CSC (or transposed-CSR) SpMV, fused axpy+clamp for
the primal update, fused axpy+project for the dual update, and reductions for dot/norm.
That is a small, tractable kernel set — five kernels, all textbook — which is exactly
why this is the right GPU target for a hackathon.

### 5.7 Two MPS conventions that silently produce wrong answers

Checked against the HiGHS free-format reader source (`highs/io/HMpsFF.cpp`) rather than
against documentation, because my numbers have to match HiGHS's numbers.

**RANGES.** With `b` from the RHS section and `r` from RANGES, HiGHS applies:

```
row type L, or type E with r < 0:   lower = upper - |r|
row type G, or type E with r > 0:   upper = lower + |r|
row type E with r == 0:             unchanged (stays an equality)
```

The sign of `r` matters **only** for E rows; for L and G rows it is ignored. This
matches the CPLEX/OSL documented convention.

**Negative UP bound.** This one genuinely differs between solvers. CPLEX documents that
an upper bound below zero with no lower bound given sets the lower bound to minus
infinity. **HiGHS does not do this** — I read `parseBounds` and there is no such case;
it simply assigns `col_upper = value` and leaves the lower bound at its default of zero.

Since HiGHS is my correctness oracle, my reader defaults to **HiGHS behaviour**, with a
`--mps-neg-up-bound=minus-inf` switch for the CPLEX convention, and a warning naming any
column that hits the case. Silently picking either convention is how you end up with an
objective that differs from the reference by a mysterious amount on one instance.

Also worth knowing for reader tests: the Netlib README contains a **BOUND-TYPE TABLE**
listing which bound types each instance uses. That tells me exactly which instances
exercise `FR` (CAPRI, CYCLE, GREENBEB), `FX` (CZPROB, BORE3D), `LO` (D6CUBE) and so on —
so reader coverage can be targeted instead of hoped for.

---

## 7. The "from scratch" rule

The PS text: *"It shall not be built upon any existing open source solver library but
shall be built from scratch from mathematical foundation."*

My reading, which I will state explicitly on a slide rather than leave implicit:

**Banned** — any optimization solver: HiGHS, SCIP, CBC, GLPK, Clp, OR-Tools, OSQP,
Gurobi, CPLEX, Xpress, and their wrappers. We do not link, vendor, or call them from
the solver. HiGHS appears in this project **only** as an external comparison baseline
invoked as a separate process by the benchmark harness — which the PS explicitly asks
for ("compared against at least one established... solver").

**Ours, written from scratch** — everything algorithmic: MPS/QPS readers, sparse
matrix types, LU with Markowitz, LDL^T, PFI/Forrest-Tomlin updates, the simplex, PDHG,
ADMM, branch and bound, cuts, presolve, heuristics, and the CUDA kernels.

**The grey zone** — BLAS/LAPACK, cuSPARSE, cuBLAS. The PS says the solver "should
exploit sparse matrix techniques, efficient numerical linear algebra and multi-core
parallelization", which reads as encouragement to use good numerical primitives, not a
ban on them. It bans building *on a solver library*, and cuSPARSE is not a solver.

**Decision: write our own anyway, for the LP path.** Reasons:

1. Our hot kernels are SpMV and vector ops. A CSR SpMV kernel is maybe 40 lines. The
   cost of writing it is small and the benefit is that the sovereignty claim has zero
   asterisk on it — which is the entire point of this PS.
2. The sparse factorizations (Markowitz LU, quasi-definite LDL^T) genuinely have to be
   ours regardless — no library gives us the *update* operations the simplex needs.
3. It removes an attack surface in Q&A. "Did you use cuSPARSE?" — "No" is a much better
   answer than a three-sentence justification.

I will benchmark our SpMV against cuSPARSE as a *sanity check* on the GPU box (if ours
is 3x slower, that is a bug worth knowing about), and report the comparison honestly on
a slide. Using it as a yardstick is not the same as depending on it.

Toolchain dependencies I will use without apology, because they are not solvers:
CMake, a C++ compiler, CUDA toolkit, a testing framework, and Python + matplotlib for
generating benchmark plots.

---

## 8. Competition read, and the bar

### 8.1 How crowded is this PS

The scrape I pulled (21 Aug 2026, one day after the PS list went live) shows
**0/500 submitted ideas** for SIH26119. That number is one day old and effectively
meaningless as a measure — everything was at zero. So I will not claim scarcity from
data I do not have. What I can reason about:

- The PS demands a numerical optimization core written from scratch. That is a
  materially different skill set from the web/ML/app work that most SIH teams bring.
  The natural failure mode for a team that picks this up is to wrap OR-Tools or PuLP
  and hope nobody reads the code — which is explicitly disqualifying here.
- SIH 2025 had problem statements where **no winner was declared** because no team
  cleared the bar. That cuts both ways: it means a working solver may win almost by
  default, and it means a non-working one loses even with no competition.

So the strategy is not "beat other teams", it is "clear the bar with room to spare, on
a PS where the bar is what eliminates people." I should re-check the idea count on
sih.gov.in in early September and again before submission — if it stays very low that
is worth knowing, but it changes nothing about what I build.

### 8.2 What the MRPL jury will actually check

They run a refinery. They already use CPLEX or Xpress behind a planning system. They
are technically literate but they are not simplex researchers. My guess at the
questions, in the order they will come:

1. *"Does it actually solve anything, right now, in front of me?"* — the live run. This
   is the whole game and everything else is secondary.
2. *"Is the answer correct?"* — reference objective values from Netlib README / MIPLIB
   `.solu`, side by side, with the gap printed. This is why the oracle tables in
   section 5 matter more than my own residual checks.
3. *"How does it compare to what we already pay for?"* — the HiGHS table. Honest,
   tolerance-labelled, with losses shown.
4. *"Did you really write this?"* — a walk through our own LU factorization or our own
   CUDA kernel, and a clean answer on section 7. Being able to open the file and
   explain the Markowitz pivot choice is the proof.
5. *"Would this work on our problem?"* — the refinery model from 5.5. Without it, the
   whole thing reads as a student exercise.
6. *"What breaks?"* — the degeneracy / ill-conditioning demos, and a straight answer
   about hard MILP being out of reach for now.

The failure mode I most want to avoid is the one where the deck claims "competitive
with Gurobi", someone asks for evidence, and there is none. Bounded claims backed by
reproducible numbers beat big claims every single time with an engineering audience.

---

## 9. Decisions and open questions for the team

Things I want your call on before I write PLAN.md, or that I am assuming unless you say
otherwise:

1. **Scope addition: the refinery model (5.5).** My brief did not have it; the PS does,
   and MRPL is the org. I want to build it. It costs maybe 3–4 days. Confirm.
2. **IPM.** The PS Description names interior-point methods. I am not building one and
   will say so on the roadmap slide. Confirm you are OK with that omission.
3. **QPLIB vs Maros–Meszaros (5.4).** I am substituting and explaining why. Confirm.
4. **GPU: Kaggle for dev, RunPod for the demo (6.2).** Needs ~₹2,500 budget and a card
   on RunPod. Also: does your college have an NVIDIA cluster? That changes the answer.
5. **Team.** Six members, at least one female, and I need one person who genuinely owns
   CUDA and one who owns the demo/story. Do you have the team, and who are they? The
   role split in PLAN.md depends on this and it is the biggest unknown in the schedule.
6. **Own kernels vs cuSPARSE (7).** I am writing our own. Slightly more work, cleaner
   claim. Confirm.
7. **Simplex depth.** Devex + PFI is enough to solve the Netlib set. DSE + Forrest–
   Tomlin + hyper-sparsity is what makes it fast. I am planning the first for the idea
   submission and the second only if the schedule holds. Flagging so it is not a
   surprise later.

## 10. What I still have not verified

Being explicit about the soft spots in this document:

- The **GMI cut formula** (4.2) is from memory plus secondary sources; the primary
  four-case derivation I could not pull in citable plaintext. Must be validated
  numerically before it goes anywhere near a claim.
- The **PDLP termination constants** (1.7) I reconstructed from the paper's structure;
  I will re-check them against the paper when implementing.
- **MIPLIB instance selection** (5.3) is a method, not a list — the list comes out of
  actually running HiGHS, which I have not done yet.
- **Nothing here has been run.** Everything in this document is reading. The first
  thing PLAN.md's phase 1 does is turn AFIRO into a solved LP.
