# SIH26119 — build plan

Written 22 Aug 2026, after [RESEARCH.md](RESEARCH.md).

Confirmed with myself before writing this:
- refinery model is **in** (LP first, piecewise-linear MILP version later);
- no interior-point method — roadmap slide, stated openly;
- Maros–Meszaros for QP, not QPLIB — stated openly, with the one-line reason;
- GPU box decision deferred, but with a hard date on it (section 6);
- **I am the only confirmed builder.** I have a team of 6 (see Phase A), but until I
  know what the other five can actually do, this plan assumes I write all of the code
  myself. That assumption drives the sequencing, and section 1.1 says exactly what gets
  handed off the moment I know otherwise.

---

## 0. Read this first: the thing that is not code

I am solo on the build. SIH is not solo. The rules, verified today:

- **exactly 6 student members**, all from the same institution, no inter-college teams;
- **minimum 1 female member, mandatory**;
- 1–2 faculty or industry mentors;
- your college's **SPOC** registers on sih.gov.in — the SPOC registration deadline was
  **31 July 2026** and has already passed;
- you must **win your college's internal hackathon (September 2026)** to be nominated;
- the SPOC then uploads the idea PPT to the national portal;
- national idea submission for SIH26119 closes **20 September 2026**;
- online evaluation Oct–Nov; **36-hour Grand Finale, December 2026**, at a nodal centre.

Sources: [Reskilll SIH 2026 guide](https://reskilll.com/blogs/smart-india-hackathon-2026-complete-guide-registration-themes-winning/),
[Where U Elevate rules writeup](https://whereuelevate.com/blogs/smart-india-hackathon-2026).
Everything here must still be confirmed against my own SPOC — editions change rules.

**Status, 22 Aug 2026: team found.** Six members including me, at least one female
member, all from the same institution — I have confirmed both. That takes the worst
risk in this project off the critical path.

What is left in Phase A is mechanical but still gating: SPOC contact, internal
hackathon date, registration, a mentor, and — the one that changes this plan — finding
out what the other five can actually build.

Their default job is the deck, the video, the documentation, the presentation and
showing up for 36 hours in December. That is a real and useful job and it is worth
saying out loud so nobody feels like a passenger. But if any of them can write C++ or
Python, that changes my schedule materially, so section 1.1 pre-scopes the work
packages I can hand over on day one.

---

## Phase A — get eligible to compete (22 Aug → 12 Sep, runs alongside the code)

| # | Action | By |
|---|---|---|
| A1 | Confirm my institute has a registered SIH 2026 SPOC, and get the name and contact. Start with the dean's office / student technical body / the SIH notice on the institute site. | 26 Aug |
| A2 | Get the **internal hackathon date, format and nomination quota** in writing from the SPOC. Everything downstream is scheduled off this date, not off 20 Sep. | 29 Aug |
| A3 | ~~Recruit 5 members including at least 1 female.~~ **Done 22 Aug.** Six members, at least one female, same institution. | done |
| A4 | Line up 1 faculty mentor. Anyone in optimization / numerical methods / OR / chemical engineering. A ChemE mentor is a genuine asset here — they will know refinery planning models better than I do. | 5 Sep |
| A7 | **Skills audit of the other five.** Who can write C++? Python? Has anyone touched CUDA? Who is the strongest presenter — that person owns the power round, not me. Ask directly, do not assume. | 26 Aug |
| A8 | Assign owners for deck, video, documentation, and logistics. Named person per artefact, not "the team". | 29 Aug |
| A5 | Register the team on sih.gov.in against SIH26119 via the SPOC. | per SPOC date |
| A6 | Confirm whether the institute allows a team to submit to more than one PS, and whether the SPOC quota is per-PS or overall. | 5 Sep |

**Definition of done:** a 6-member team registered against SIH26119, internal hackathon
date known and on my calendar, mentor named, and every deliverable in Phase 3 with a
named owner.

**Watch item:** a team that exists on a registration form and a team that turns up are
different things. The check for that is A8 — if the deck has a named owner by 29 Aug and
a draft outline by 12 Sep, the team is real. If those slip, I plan Phase 3 as if I am
building the deck myself too, and I would rather find that out on 29 Aug than on 18 Sep.

---

## 1. Scope, ruthlessly tiered

Six people could build the full LP + MILP + QP + GPU scope. I cannot, not while also
carrying everything else on my plate. So I am fixing the priority order **now**, in
writing, so that when time runs short in November I cut from the bottom instead of
panicking and half-finishing four things.

| Tier | What | Why it is at this level |
|---|---|---|
| **T0** | MPS/QPS reader, sparse types, benchmark harness with HiGHS baseline and reference-value checking | Nothing is demonstrable without it |
| **T0** | **PDHG LP solver on CPU** | The headline algorithm, and the best value-per-line in the whole project. ~600 lines gets a working from-scratch LP solver |
| **T0** | Refinery crude-blending + production-planning LP | MRPL is the jury. Without this the deck is a student exercise |
| **T0** | Robustness demos: degeneracy, ill-conditioning, scaling ablation | Directly named in the Expected Solution paragraph |
| **T1** | **CUDA port of PDHG** | GPU is in the PS title. One week once the CPU version is correct |
| **T2** | Bounded-variable dual simplex (Markowitz LU + PFI + Devex + Harris/BFRT) | Exact vertex solutions, 1e-9 objectives, and the warm starts that make B&B possible |
| **T3** | QP by ADMM, **indirect variant** | See 5.3 — the indirect solve reuses the SpMV I already have and saves me writing a sparse LDL^T with AMD ordering. That is a week of solo time saved |
| **T4** | MILP: branch and bound + root GMI cuts + light presolve | Needs T2. First thing to be cut if the schedule slips |

### 1.1 What I hand off the moment I know who can code (A7)

I am keeping the plan solo-safe, but I am not going to waste a capable teammate by
having nothing ready for them. These packages are genuinely separable — clean input,
clean output, testable on their own, and none of them need to understand PDHG:

| Package | Needs | Frees me | Hand to |
|---|---|---|---|
| **MPS/QPS reader** | careful C++, no numerics | ~4 days | anyone comfortable in C++ |
| **Instance fetcher + reference-value oracle** | Python, shell | ~2 days | any Python person |
| **Benchmark harness + plots** | Python, matplotlib | ~3 days | any Python person |
| **Refinery model generator** | Python + reading two papers | ~3 days | ideally the ChemE-adjacent member or the mentor's student |
| **Presolve + postsolve stack** | C++, self-contained, well-specified | ~5 days | the strongest C++ person |
| **CUDA kernels** (5 kernels, spec in Phase 4) | CUDA | ~1 week | anyone who has written CUDA |

If one person takes the reader and another takes the harness and the fetcher, I get
roughly **two weeks back before 20 Sep** — which is the difference between "PDHG works"
and "PDHG works and the GPU port has already started." If someone can do CUDA, the port
comes off my critical path entirely and can run parallel with the simplex through
October, which is what makes T4 (MILP) survivable.

Rule for delegation: every package gets a written spec with its definition-of-done and
a test they can run themselves. If I have to review it line by line it has not saved me
anything.

Cut rule: **if T2 slips past 20 Nov, T4 is dropped** and MILP goes on the roadmap slide
with a clear statement of why. Three things that work beat five things that half-work,
and a jury of engineers will read it the same way.

The thing I will *not* cut, no matter what: honest, tolerance-labelled benchmark numbers
with the losses shown. That is the credibility of the whole submission.

---

## 2. Timeline overview

```
22 Aug ─────────────────────────────────────────────────► 20 Sep  (idea submission)
        Phase A: team + SPOC  ══════════════════════
        P1 skeleton   P2 PDHG CPU        P3 refinery + demos + PPT
        ▔▔▔▔▔▔▔▔▔▔▔   ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔   ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
        22-28 Aug     29 Aug - 8 Sep     9-19 Sep
                                    ▲ internal hackathon lands somewhere in here

21 Sep ──────────────────────────────────────────────────► Dec  (36h finale)
        P4 GPU port + simplex start   P5 simplex     P6 QP + MILP   P7 freeze
        ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔   ▔▔▔▔▔▔▔▔▔▔▔    ▔▔▔▔▔▔▔▔▔▔▔▔   ▔▔▔▔▔▔▔
        21 Sep - 31 Oct              1-23 Nov       24 Nov-10 Dec  pre-finale
```

The internal hackathon date (A2) is the real near-term deadline and it may be earlier
than 20 Sep. If it lands before 15 Sep, Phase 3 compresses and the PPT gets built
against whatever Phase 2 has produced by then. Phase 2's output is what makes the PPT
credible, so Phase 2 is the one that must not slip.

---

## 3. Phase 1 — skeleton, reader, harness (22–28 Aug)

**Goal:** I can read any benchmark instance, and I can already produce a comparison
table — before I have written a single line of solver.

Building the harness *first* is deliberate. It means every later phase is measured from
day one, and it means that even in the worst case I turn up with real data about the
benchmark sets.

1. CMake C++20 project. `SOLVER_ENABLE_CUDA=OFF` by default; **no CUDA in any header
   the CPU build sees.**
2. Sparse matrix types: CSR and CSC, with a transpose that is just a view swap where
   possible. Triplet builder → compressed.
3. **MPS reader** — fixed and free format, `ROWS` / `COLUMNS` / `RHS` / `RANGES` /
   `BOUNDS`, all bound types (`UP` `LO` `FX` `FR` `MI` `PL` `BV` `LI` `UI`), negative RHS
   on the objective row as the objective constant, `OBJSENSE` where present.
   QPS extension (`QUADOBJ` / `QMATRIX`) for the QP set.
4. Standard-form builder producing the `(c, A, b, G, h, l, u)` form from RESEARCH 1.1,
   plus the logical-variable form from RESEARCH 2.1 for the simplex later.
5. **Instance fetcher**: script that compiles `emps.c` and pulls the Netlib set, pulls
   `lptestset` instances, MIPLIB instances, and the Maros–Meszaros zips. The emps
   pipeline is already verified working on this machine.
6. **Reference-value oracle**: parse the Netlib README table, the MIPLIB `.solu` file
   and the Maros–Meszaros table into one lookup. Every solve is checked against a
   published optimum, never against my own residual.
7. **Benchmark harness**: run instance → objective, iterations, wall time, residuals →
   CSV. Invoke `highs` as an external process. **Three baselines, not one** (RESEARCH
   5.6): HiGHS dual simplex, HiGHS PDLP, and mine. The PDLP column is my same-algorithm
   debugging oracle from Phase 2 onward, and reporting it is what keeps the final table
   honest. Emit a markdown table and a matplotlib performance-profile plot.
8. `brew install highs` (1.15.1 is bottled — the same 1.15.x line Mittelmann
   benchmarked, which is convenient for cross-checking).

Reader correctness is where silent wrong answers come from, so two conventions get
explicit treatment (RESEARCH 5.7): RANGES sign handling on E rows, and the negative-UP
bound case where HiGHS and CPLEX genuinely disagree. Default to HiGHS behaviour since
HiGHS is the oracle, with a switch for the other and a warning naming affected columns.

**Definition of done:** `bench --set netlib20 --solver highs` produces a CSV where all
20 objectives match the Netlib README to 8 significant figures. That is the harness
validating itself before I trust it with my own solver. Plus: the reader round-trips
every instance in the Netlib set without warnings I have not explained, checked against
the README's BOUND-TYPE TABLE for coverage of FR / FX / LO / MI / BV.

---

## 4. Phase 2 — PDHG LP on CPU (29 Aug – 8 Sep)

**Goal:** a from-scratch LP solver that actually solves things. This is the phase that
makes the submission real.

Build order, each step verifiable on its own:

1. **Scaling**: Ruiz (10 sweeps, inf-norm) then Pock–Chambolle (alpha=1, 1-norm).
   Verify by checking row/column norm spread before and after.
2. **Plain PDHG loop** with a fixed step `eta = 0.9 / ||K||_2` (power iteration), no
   restarts, no adaptivity. Verify on AFIRO — it will be slow and that is fine.
3. **Unscaling and termination** on the true problem, relative criteria at 1e-6.
   Verify: AFIRO objective = -4.6475314286E+02.
4. **Adaptive step size** (RESEARCH 1.3). Verify: iteration count drops on 25FV47.
5. **Primal weight** update (1.4). Verify on a badly balanced instance like FIT1P.
6. **Restarts** on the weighted KKT error with cuPDLP's 0.2 / 0.8 / 0.36 constants
   (1.5). This is the step that turns it from "converges eventually" into a solver.
   Verify: large iteration-count drop across the whole set.
7. Infeasibility certificates (1.8) — only if the above is done early.

**Definition of done:** at least 15 of the 20 Netlib instances in RESEARCH 5.1 solved to
1e-6 relative, with objectives matching the README table to 6 significant figures, and
AFIRO / SC50A / ADLITTLE / BLEND / SHARE1B / DEGEN2 / 25FV47 all correct.

**Ablation to record while building, because it is a slide:** run the whole set with
scaling off. It will mostly fail or crawl. "Here is the same solver without Ruiz +
Pock–Chambolle" is a far better answer to *"how do you handle ill-conditioning?"* than
any paragraph I could write.

---

## 5. Phase 3 — refinery model, robustness, PPT (9–19 Sep)

This phase is doing double duty: it has to win the **internal** hackathon and pass the
national screening.

### 5.1 Refinery model (9–12 Sep)

A multi-period crude blending + production planning LP, written as a generator that
emits MPS so it scales from toy to a few thousand variables:

- crudes with assay properties (sulphur, density, yields), purchase costs and
  availability;
- CDU cuts and downstream unit capacities;
- product blends with quality specs as linear bounds on blended properties;
- multi-period inventory balances;
- objective: maximise margin.

True blending is bilinear — the pooling problem, and nonconvex. The industry-standard
answer is a linear or piecewise-linear approximation, and the piecewise-linear version
is a MILP, which lands squarely in scope for the T4 work later. Both facts go in the
deck; knowing *why* the linear model is an approximation is exactly the kind of thing
an MRPL engineer will probe. References in RESEARCH 5.5.

Sanity check: solve it with HiGHS first, then with mine, and confirm they agree. A model
I wrote myself has no published optimum, so HiGHS **is** the oracle here.

### 5.2 Robustness runs (13–14 Sep)

- **Degeneracy**: DEGEN2, DEGEN3, CYCLE — quoting the Netlib README's own MINOS
  statistics (56.7% / 52.5% / 47.1% degenerate steps) as third-party evidence that these
  instances are what I claim they are.
- **Ill-conditioning**: PILOT87, SHARE1B, GREENBEA — with the Netlib README's own note
  that PILOT87 is harder than PILOT "because of the bad scaling in the numerics".
- **Scaling ablation** from Phase 2.
- Convergence plots: KKT error against iteration, restarts marked.

### 5.3 Benchmark table vs HiGHS (15 Sep)

Netlib set, both solvers, tolerance labelled on every column. I will lose on most small
Netlib instances — simplex is simply better there — and I am going to show that,
because the large-instance story in Phase 4 is what carries the argument, and a table
where I win everything is a table nobody believes.

### 5.4 Deck and video (16–19 Sep)

SIH's idea-submission template is normally: Proposed Solution / Technical Approach /
Feasibility and Viability / Impact and Benefits / Research and References. **Get the
actual template from the SPOC — do not rebuild it from memory.**

What goes in each, for this PS:

- **Proposed solution** — sovereign solver core, from scratch, GPU-accelerated LP path.
  Lead with the working demo, not the ambition. One screenshot of a real solve with the
  reference objective next to it beats a page of architecture.
- **Technical approach** — the four algorithm families, the from-scratch boundary
  (RESEARCH 7) stated explicitly, the CPU/GPU backend split.
- **Feasibility and viability** — this is where the honest scoping lives: what is built,
  what is planned, what is out of reach and why. Including "no interior-point method"
  and "hard MILP is a roadmap item". State the limits before the jury finds them.
- **Impact and benefits** — licence-cost sovereignty, the refinery model, the fact that
  the GPU path is where the field is actually moving (cite the Aug 2026 Mittelmann
  LPfeas results).
- **Research and references** — PDLP, cuPDLP, cuPDLPx, OSQP, Huangfu & Hall, Koberstein,
  MIPLIB 2017. A real bibliography signals that this is not a wrapper project.

Video: screen capture of an actual run. Reading a reference optimum out of the Netlib
README and then watching my solver print the same number is the most persuasive 30
seconds available.

**Definition of done:** deck + video submitted through the SPOC before the portal
deadline, containing at least one benchmark table and one screenshot from a real run.

---

## 6. Phase 4 — GPU port and simplex start (21 Sep – 31 Oct)

### 6.1 The GPU decision, with a date on it

Deferred for now, but not indefinitely. **Decision by 30 Sep**, because the port needs
October.

Default until then, costing nothing and requiring no commitment: **Kaggle Notebooks**.
Published ~30 GPU-hours/week, P100 or 2x T4, `nvcc` available, no card needed. That is
more than enough for kernel development — the runs are seconds, the time goes on
compiles and uploads.

The decision that actually matters is **which box the December demo runs on**, and the
answer is not Kaggle. Options in RESEARCH 6.1. My recommendation stands: RunPod Secure
Cloud RTX 4090, roughly ₹2,500 for the whole finale window, ~99% uptime. Not Vast.ai —
a spot instance reclaimed on 15 seconds notice during a live demo is not a risk worth
the ₹1,000 saved. If the institute has an NVIDIA cluster, that beats both, and finding
out is an A-phase question for the mentor.

**Recorded fallback is mandatory regardless.** Terminal capture of the GPU run plus
committed logs and CSVs. Venue networks fail; this is the single most likely thing to
go wrong on the day.

### 6.2 The port (Oct)

1. Introduce `LinAlgBackend`: `spmv`, `spmv_transpose`, `axpy`, `dot`, `nrm2`,
   `clamp_project`, `max_zero_project`. CPU implementation first, and **the CPU results
   must not change** — that is the regression test for the whole refactor.
2. CUDA implementation. Five kernels: CSR SpMV, transposed SpMV, fused axpy+clamp for
   the primal update, fused axpy+project for the dual update, and reductions.
   Written by hand, not cuSPARSE (RESEARCH 7).
3. Move the whole loop to device: adaptive step, restart checks, primal weight. Host
   keeps I/O, scaling, unscaling. Two transfers total, per cuPDLP.jl's design.
4. **Correctness gate:** GPU and CPU must agree to solver tolerance on every Netlib
   instance before any timing number is recorded.
5. Timing runs on the Mittelmann `lptestset` instances from RESEARCH 5.2, against HiGHS
   on CPU. Include `cont1` or `cont11`, where first-order methods lose badly, and report
   the loss.
6. Sanity-benchmark my SpMV against cuSPARSE once. If mine is 3x slower that is a bug I
   want to know about. Report the comparison; using it as a yardstick is not depending
   on it.

**Definition of done:** a table of large sparse LPs where the GPU build beats HiGHS at
1e-6, with at least one instance where it does not, and CPU/GPU agreement demonstrated
across the Netlib set.

This table is the headline of the finale. Everything else supports it.

### 6.3 Simplex, started in parallel (late Oct)

Begin the bounded-variable simplex while the GPU work is fresh, because it is the long
pole and it must not start in November.

---

## 7. Phase 5 — revised simplex (1–23 Nov)

Order, each step independently testable (RESEARCH section 2):

1. Logical-variable standard form; basis bookkeeping; `b_hat` and reduced costs from a
   dense LU. Correct before fast.
2. **Primal simplex**, Dantzig pricing, textbook ratio test. Gate: AFIRO, SC50A,
   ADLITTLE, BLEND exact.
3. Sparse **LU with Markowitz pivoting** and threshold stability (`u ~ 0.1`), with
   triangular peeling first. Gate: 25FV47 factorizes with sane fill.
4. **PFI update** + refactorization every 50–100 iterations and on residual failure.
5. **Dual simplex** with a composite-objective phase 1. Gate: DEGEN2, DEGEN3 solved.
6. **Harris two-pass ratio test** + **bound flipping**. Gate: DEGEN2/DEGEN3 iteration
   counts drop and no accuracy warnings fire.
7. **Devex pricing**. (DSE and Forrest–Tomlin are explicitly stretch — they make it
   fast, they do not make it correct.)

**Definition of done:** all 20 Netlib instances from RESEARCH 5.1 solved to 1e-9,
objectives matching the README table, and a warm-started re-solve after a bound change
taking on the order of tens of iterations rather than a fresh solve. That last property
is the one T4 depends on.

---

## 8. Phase 6 — QP and MILP (24 Nov – 10 Dec)

### 8.1 QP by ADMM, indirect (24 Nov – 1 Dec)

Using the **indirect** variant deliberately (RESEARCH 3.7): instead of factorizing the
quasi-definite KKT matrix, solve

```
(P + sigma*I + rho*A^T A) x = rhs
```

by conjugate gradient. That is SpMV-only, so it reuses the exact kernels the PDHG path
already has — CPU and CUDA both — and it saves me writing a sparse LDL^T with a
fill-reducing ordering, which is easily a week of solo work I do not have.

Then: modified Ruiz scaling with cost step, over-relaxation `alpha = 1.6`, per-constraint
`rho` with the `1e3` multiplier on equalities, the adaptive `rho` rule, unscaled
termination, and polishing (RESEARCH 3.5) to get from 1e-3 ADMM accuracy to something
worth quoting.

**Definition of done:** the 16 Maros–Meszaros instances in RESEARCH 5.4 solved, matching
the reference optima, with polishing on/off reported. The QPTEST/HS21/HS76/QAFIRO end of
the list should be exact; CVXQP1_M is the honest hard case.

### 8.2 MILP (2–10 Dec) — first thing cut if the schedule slips

1. B&B on the warm-started dual simplex. Most-fractional branching, depth-first, to get
   the tree provably correct against MIPLIB `.solu` values on tiny instances.
2. Light presolve with a postsolve undo stack (RESEARCH 4.3).
3. Pseudocost branching; hybrid depth-first-then-best-bound node selection.
4. Root-only **GMI cuts**, with the validation discipline from RESEARCH 4.2: every cut
   must be violated by the current LP point and must not cut off the known optimal
   integer solution on instances where I have the reference value. Reject cuts with
   dynamic range above ~1e6.
5. Simple rounding + diving heuristics for an early incumbent.

**Instance selection** by the method in RESEARCH 5.3: run HiGHS over the MIPLIB
collection with a 10-second limit, keep what it closes in under ~5 seconds, pick 8–12
spanning different structures. Publish the selection script so the set reads as curated,
not cherry-picked.

**Definition of done:** 8+ MIPLIB instances solved to proven optimality with objectives
matching `.solu`, plus a table of where I time out. Timeouts shown, not hidden.

---

## 9. Phase 7 — freeze and rehearse (11 Dec → finale)

- **Code freeze one week before the finale.** No new features. Only bug fixes.
- Regenerate every benchmark table and plot from scratch, from a clean clone, with a
  single command. If a number in the deck cannot be reproduced by one script, it does
  not go in the deck.
- Record the GPU demo. Commit the logs and CSVs.
- Rehearse the live demo end to end at least five times, including the failure paths:
  no network, GPU box unreachable, laptop only.
- Write the README so someone can build and run it in under five minutes. "Transparent
  and extensible" is in the PS text; a build that only I can do fails that.

### The finale demo, in order

1. `solve refinery_blend.mps` — the MRPL-shaped model. Objective, time, done. Open with
   *their* problem.
2. `solve degen3.mps` alongside the Netlib reference value. Correct on a
   52%-degenerate instance.
3. Same instance, scaling disabled. It struggles. Turn scaling back on. That is the
   numerical-robustness answer, demonstrated rather than asserted.
4. The GPU run on a large `lptestset` instance next to HiGHS on CPU. The headline number.
5. The full benchmark table, including the instances where I lose.
6. One file open on screen — the CUDA SpMV kernel, or the Markowitz pivot selection —
   as the answer to "did you actually write this?"

### Power-round pitch, roughly

> Every refinery in India runs its planning on CPLEX or Xpress. We wrote an optimization
> core from scratch — no solver library underneath it, our own LU factorization, our own
> CUDA kernels. It solves the standard Netlib and MIPLIB benchmarks to published optimal
> values, it beats HiGHS by [N]x on large sparse LPs on a GPU, and it solves a refinery
> crude-blending model. It is not CPLEX yet. Hard mixed-integer problems are still ahead
> of us and we will tell you exactly where the line is. But the foundation is Indian,
> it is open, and every line of it can be inspected.

Bounded claims, backed by numbers anyone can reproduce. With an engineering audience
that beats a big claim every time.

---

## 10. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | ~~No team of 6~~ — **resolved 22 Aug.** Remaining: no internal-hackathon win | Medium | Fatal | Team confirmed (6 members, ≥1 female, same institution). Internal round is now the gate — Phase 3's deliverable has to win it, which is why Phase 2 cannot slip |
| R2 | **Solo build capacity** — this is a 6-person scope and I am the only confirmed builder | High | Severe | T0–T4 tiering in section 1 with the cut rule written down *before* the pressure arrives. T0 alone is a defensible submission. Section 1.1 lists what gets handed off if A7 turns up anyone who can code |
| R11 | **Teammates registered but not delivering** — deck and video land back on me in mid-Sep | Medium | High | A8 assigns named owners by 29 Aug; draft deck outline due 12 Sep. If either slips I re-plan Phase 3 assuming I build the deck too, while there is still time to do it |
| R3 | **PDHG does not converge on the harder Netlib instances** | Medium | High | Scaling is non-optional and comes first; restarts are the known fix. Fallback: report moderate-accuracy results honestly and let the simplex carry high accuracy |
| R4 | **GPU box unavailable at the finale** | Medium | High | Kaggle for dev costs nothing; RunPod Secure Cloud (not spot) for the demo; **recorded fallback is mandatory** |
| R5 | **Simplex slips past 20 Nov** | Medium | Medium | Pre-agreed: drop T4 (MILP), ship LP + QP + GPU solidly, put MILP on the roadmap with an honest explanation |
| R6 | **GMI cuts silently invalid** (nonbasic-at-upper-bound and slack bookkeeping) | Medium | Medium | Validate every cut against known optimal integer solutions from `.solu` before trusting any of it. Never ship a cut generator that has not been checked this way |
| R7 | **Jury reads "from scratch" more strictly than I do** | Low | High | RESEARCH 7 is an explicit slide. Own kernels, own factorizations, HiGHS only as an external baseline process |
| R8 | **"No winner declared"** — happened on several SIH 2025 problem statements | Medium | Fatal-ish | Exactly why T0 is scoped to be genuinely finishable. A working solver clears a bar that eliminates most entrants |
| R9 | Benchmark numbers not reproducible under questioning | Low | High | One script regenerates every table and plot from a clean clone. Nothing in the deck that the script cannot produce |
| R10 | My other commitments eat September | High | High | Phase 2 is the only thing that truly cannot slip before 20 Sep. Phase 3 can compress to four days if it has to |

---

## 11. Weekly check-in

Every Sunday, three lines, committed to the repo:

1. Phase A status — team count, SPOC contact, internal hackathon date.
2. What passed its definition-of-done this week.
3. What is at risk, and which tier gets cut if it does not recover.

If Phase A is still red on 5 Sep, that week's work is Phase A and nothing else.

---

## 12. Immediate next actions

1. Find out who my institute's SIH 2026 SPOC is. Today or tomorrow. Everything else is
   downstream of this.
2. Start the recruiting ask — five people, deck/video/docs/presentation roles, at least
   one female member.
3. Start Phase 1: CMake skeleton, sparse types, MPS reader.

Phase 1 is ready to start as soon as this plan is approved.

---
---

# Part II — The plan revisited, 30 August 2026

Everything above is what I thought on 22 August, before writing a line of solver
code. I am leaving it exactly as it was, because a plan that gets quietly edited
to match what happened is not a record of anything.

This part is the honest review: what the plan got right, where it was thinking
in one direction only, and what I would write differently now.

## 13. Where the plan turned out to be right

- **Reader first.** Correct, and for the reason given: nothing could be checked
  until models could be read. Every later measurement rests on it.
- **Scaling before anything else in the first-order method.** Non-negotiable, as
  written. Netlib has instances spanning ten orders of magnitude and the method
  does not converge on them unscaled.
- **Checking against published optima rather than against itself.** This is the
  single most valuable rule in the document. Almost every bug found since was
  found because an external number disagreed.
- **R6 — "GMI cuts silently invalid, validate against known optimal integer
  solutions before trusting any of it."** This was the most prescient line in
  the plan. A cut generator did ship invalid, twice.

## 14. Where the plan was thinking in one direction

### 14.1 Every risk was about finishing. None was about being wrong.

Read the risk register again. R2 solo capacity, R5 simplex slips, R10 September
disappears — ten of eleven risks are variations of *will I get it built in
time*. The tiering, the cut rule, the "first thing cut if the schedule slips" —
all of it is scheduling.

The schedule risk did not materialise. Everything through T4 was built, and
built earlier than planned. **MILP, explicitly named as the first thing to cut,
is instead where the most recent and strongest work went.**

What did materialise, repeatedly, is the risk the register almost entirely
missed: **the solver being confidently wrong.**

- Presolve declared a feasible model infeasible, because my own 1e-9 bound
  padding manufactured forcing rows.
- The dual simplex reported a wrong answer as optimal — `fit1p` at 33,609
  against a true 9,146.38, feasible, with a row violation of 2.8e-14.
- A cover cut removed feasible integer points whenever a binary sat at exactly
  0.5, and 426 separations at random points never landed there.
- A sparse LDL' silently became a diagonal factorisation, and its test passed.
- The GPU never uploaded the dual iterate, which was invisible for as long as
  everything started at zero.
- `fiber` came back as a *proved* optimum 60.8% away from the published one,
  with a matching dual bound and no numerical complaint — two separate bugs,
  a bound crossing of 1.78e-15 treated as a proof of infeasibility, and a cover
  cut derived from an earlier cut.

R6 was right in spirit and far too narrow in scope. It named one cut family. The
correct version of that risk is: **any component that can discard part of the
search space can discard the answer, and none of them will tell you.** A slow
solver announces itself. A wrong one hands you a confident number.

If I were writing the register today, R0 would be that, and the mitigation would
be the tooling rather than the vigilance: cut validity by enumeration over small
programs, a dual-feasibility check the simplex runs on itself before claiming
optimality, a survey against a hundred published optima rather than seven, and a
debug-solution tracker that names the exact step at which a known-correct answer
is thrown away. Each of those exists now, and each exists because something got
through.

### 14.2 "Build it, then measure" should have been "measure, then decide"

The plan sequences work by tier and by week. It never asks, before a piece of
work, **how much is actually available on our instances**.

That question, asked late, has since saved weeks:

- **Coefficient tightening** — implemented in full, fires three times across
  seven instances, moves the root bound by nothing. Off.
- **Parallel column merging** — 20% of Netlib columns are parallel, which looks
  compelling until the objective condition cuts it to 1,471 columns out of
  159,369. Measured, not built.
- **Hypersparsity** — the literature reports a 5.2× mean speedup. Their test set
  was selected for the property. On ours only 4 of 21 instances qualify and the
  overall rate is 15.4% against a 60% threshold. Six to eight weeks of work to
  help four instances.
- **Forrest–Tomlin updates** — a sampling profile of the slowest instance puts
  `LuFactor::factorize` at 11 samples out of about 4,500. It is aimed at
  something that is not hot.

None of these would have been caught by the plan as written. All four are
standard, respectable, in every textbook. The only thing that separates the ones
worth building from the ones that are not is a measurement taken **first**.

### 14.3 T3's reasoning was overturned by measurement

The plan chose "QP by ADMM, **indirect variant**" and gave the reason: it reuses
the existing SpMV and "saves me writing a sparse LDL' with AMD ordering. That is
a week of solo time saved."

The direct sparse LDL' got written anyway, and it is **1.52× faster** than the
indirect solve. The week was worth spending.

But the more interesting half is what the measurement said next: raising the
fill budget, which puts *more* instances on the direct path, makes the whole set
**slower** — 20× budget gives 62.15 s, 50× gives 58.86 s, 200× gives 60.01 s.
Which means AMD ordering, the thing the plan was avoiding, would improve exactly
the instances that gain nothing from being on the direct path at all.

The plan's conclusion was wrong and its instinct about cost was right, and only
measurement separates those.

## 15. The interior-point decision, from every side

The plan disposes of this in eight words: *"no interior-point method — roadmap
slide, stated openly"*. Honest, but not reasoned. Here it is properly.

**The case against, as the plan implicitly had it.** An IPM needs a sparse
Cholesky of `A·D·Aᵀ` (or an LDL' of the augmented system) at every iteration,
with a fill-reducing ordering. That is a large piece of machinery that nothing
else in the project reuses.

**The case against that the plan did not state, and which is stronger.** It
serves neither of the two goals:

- **MILP needs warm starts.** The dual simplex restarts from the parent's basis
  and finishes a child in about three pivots — 18,772 of 18,775 relaxations warm
  start on the measured set. An IPM does not warm start usefully, so every node
  would solve from scratch. Branch-and-bound as built would not survive it.
- **The GPU needs matrix-vector products.** An IPM's time goes into a sparse
  factorisation, which is the part that parallelises worst.

**The case *for*, which the plan never considered at all.** First-order methods
converge linearly, and that is precisely where this solver is weakest — it is
why `--gap-tol` exists and why feasibility polishing had to be built. A
second-order method gives high accuracy natively. The recent literature makes
this argument explicitly: the linear rate of first-order methods restricts them
where fast convergence to tight tolerances is required, which is the motivation
given for revisiting second-order methods on GPUs.

**And the ground has moved since 22 August.** Three things I did not know then:

- NVIDIA shipped **cuDSS**, a GPU direct sparse solver with Cholesky, LDL' and
  LU — so the factorisation an IPM needs is now available on the device rather
  than being a research problem.
- **Condensed-space IPM** methods reshape the KKT system into a
  symmetric-positive-definite form that factorises far better on a GPU, and the
  reported result is that the structure of the condensed system offsets the
  ill-conditioning it introduces
  ([arXiv:2405.14236](https://arxiv.org/html/2405.14236v2),
  [Math Prog Comp](https://link.springer.com/article/10.1007/s12532-026-00335-0)).
- There is active 2025–26 work on exactly this
  ([GPU Implementation of Second-Order LP and NLP Solvers, arXiv:2508.16094](https://arxiv.org/pdf/2508.16094)).

**So does the decision still hold? Yes — but for a different reason than the
plan gave.**

It does not hold because "IPM cannot work on a GPU". That was never quite true
and is less true now. It holds because:

1. The accuracy gap an IPM would fill **is already filled by the simplex**,
   which is built, exact, and gets 16 of 16 Netlib instances with published
   optima on both algorithms.
2. Even with cuDSS and condensed-space methods, the reported gains for fully
   GPU-based interior-point LP solvers **remain modest** — the sparse
   factorisation is still the bottleneck and still the part GPUs are worst at.
3. It would still not warm start for branch-and-bound.

That is a decision I can defend under questioning. "We did not have time" is
not, and it is what the original line would have forced me to say.

**What would change it:** if the refinery model grew to a size where the simplex
becomes impractical *and* high accuracy is genuinely required, a condensed-space
IPM on cuDSS becomes the right third path. That is the trigger, written down, so
the decision can be revisited on evidence rather than on mood.

## 16. The risk register, rewritten

| # | Risk | Why it is here now |
|---|---|---|
| **R0** | **A component that prunes, reduces or cuts discards the answer, and reports success** | This is what actually happened, six times. Mitigation is tooling, not care: enumeration tests for cuts, a self-check before any optimality claim, a hundred published optima rather than seven, and a debug-solution tracker |
| **R0b** | **A tuned constant is fitted to too few instances** | Every branch-and-bound default was chosen against seven instances and every commit that set one says so. The wider set found a wrong answer within minutes of existing. Re-fit against the hundred |
| **R0c** | **A measurement is wrong and a good change gets discarded** | Twice a benchmark reported a regression that did not exist, because it ran three sixty-second solves per instance back to back. Both times the change was a pure improvement. Measure on a quiet machine, and re-run anything surprising alone before believing it |
| R4 | GPU box unavailable at the finale | Unchanged and still live. Recorded fallback stays mandatory |
| R7 | Jury reads "from scratch" more strictly than I do | Unchanged. Strengthened by the fact that the method now has a name: Proposition 3.1 of [arXiv:2509.23903](https://arxiv.org/abs/2509.23903) shows what is implemented here **is** a Halpern Peaceman–Rachford method, not an approximation of one |
| R9 | Benchmark numbers not reproducible under questioning | Stronger now than the plan hoped: every table has the command that regenerates it, and the external reference tables are copied into the repository so they cannot drift |
| ~~R2, R5, R10~~ | Solo capacity, simplex slipping, September disappearing | Did not materialise. Everything through T4 was built ahead of schedule |

## 17. What actually remains

Not a schedule this time. An ordered list, with the reason each one is in that
position — and the first item is not an improvement at all.

1. **Verify on a GPU.** Nothing built since the last run has been tested on
   hardware, and the CPU genuinely hides a class of bug that only appears on the
   device. This is closing a risk, not adding a feature.
2. **Re-fit the branch-and-bound constants against a hundred instances rather
   than seven.** The tooling now exists. The seven-instance fits are the largest
   remaining source of the kind of error that does not announce itself.
3. **Move the convergence check onto the device.** On one instance 79% of solve
   time is outside the kernels. Half of this is done.
4. **Report against Mittelmann's LPfeas benchmark.** Eight of its forty public
   instances are already here and the reference table is in
   `data/reference/`. Solved-or-not at 1e-6 against a published table is worth
   more than any number computed against a rubric of our own.
5. **Continuous integration.** The test suite is good and nothing runs it
   automatically.

What is deliberately *not* on this list, each for a measured reason given above:
interior point, hypersparsity, Forrest–Tomlin, parallel columns, coefficient
tightening.

## 18. The one line I would add to the top of the original plan

The plan opens by fixing scope so that under pressure I cut from the bottom
instead of half-finishing four things. That was right.

What it should also have said: **a feature that is missing is visible, and a
feature that is wrong is not.** Every hour spent on something that makes
wrongness visible — a test that enumerates, a check the solver runs on itself, a
wider instance set, a tool that names the step where an answer was lost — bought
more than an hour spent on the next feature. That is not a lesson I had on 22
August, and it is the one I would most want to have had.
