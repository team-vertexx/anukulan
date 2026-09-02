# Sankhya

An LP / MILP / QP solver written from scratch in C++17, with a CUDA backend for
the first-order method. Built for SIH 2026, problem statement **SIH26119**
(MRPL — indigenous GPU-accelerated optimization for refinery planning).

Nothing here wraps an existing solver. There is no HiGHS, no OR-Tools, no SciPy
underneath — the sparse matrix type, the LU factorisation, the simplex, the
branch and bound tree, the ADMM loop and the CUDA kernels are all in `src/`.
HiGHS appears in this repository exactly once: as the thing we benchmark
*against*.

```
13,651 lines of solver and CLI     13 test suites, all passing
88 Netlib instances verified against published optima — 0 wrong answers
```

---

## What it does

| | what | where |
|---|---|---|
| **Reader** | MPS free and fixed format, LP and QP (Maros–Meszaros QPS) | `src/sankhya/mps_reader.cpp` |
| **Presolve** | 12 reductions + postsolve, including the dual | `src/sankhya/presolve.cpp` |
| **First-order LP** | PDHG / PDLP with restarts, Halpern, feasibility polishing | `src/sankhya/pdhg.cpp` |
| **GPU** | CUDA backend for the first-order method | `src/sankhya/cuda_backend.cu` |
| **Simplex** | revised primal *and* dual, sparse LU, Devex, steepest edge | `src/sankhya/simplex.cpp` |
| **MILP** | branch and cut — cover, c-MIR, Gomory; reliability branching; best-estimate node selection; pump, diving, RINS | `src/sankhya/branch_and_bound.cpp` |
| **QP** | OSQP-style ADMM with a sparse LDL' of the KKT system | `src/sankhya/qp.cpp` |
| **Crossover** | turns a first-order point into a simplex basis | `src/sankhya/crossover.cpp` |

## Where it stands

Every number below was measured on this machine and can be reproduced with the
command next to it in [docs/RESULTS.md](docs/RESULTS.md). None of it is
estimated.

- **Reader** — matches HiGHS on all 88 Netlib instances, 1.4–1.5× faster
- **Simplex** — **82 of 88** Netlib instances reach the published optimum and
  **none returns a wrong answer**; the other six stop visibly with a time limit,
  iteration limit or numerical error
- **Crossover** — seeding the simplex from a first-order point cuts pivots
  substantially and turns six instances that returned no answer at all
  (`degen3`, `stocfor2`, `woodw`, `scsd8`, `wood1p`, `modszk1`) into correct ones
- **First-order** — 76/88 Netlib published optima at `--tol=1e-8`
- **Presolve** — removes 23.3% of Netlib rows and 20.9% of its columns, and
  29.7% of the refinery model's columns, where it previously removed none.
  1.36× fewer simplex iterations, without changing an answer, and it recovers
  duals as well as primals
- **MILP** — on 70 MIPLIB instances at a 15 s limit, **45 end with a feasible
  solution and 7 prove optimality**, against 41 and 5 before this work. Twenty-five
  still find nothing at all, which is where the remaining work is
- **QP** — 35 of the 40 smallest Maros–Meszaros instances
- **GPU (Tesla T4)** — **3.14×–12.07×** on solve time over the same algorithm on
  CPU, verified on hardware with the backend contract tests passing and CPU/GPU
  agreement at machine precision

Read [docs/RESULTS.md](docs/RESULTS.md) for the full tables and the honest
assessment of where this sits against a production solver — about **62% of
HiGHS**, with MILP still the weak leg at roughly 40. That is not modesty; it is in the
results document with a component-by-component breakdown and the reasoning for
each number.

**The correctness figure above replaced a "16 of 16" that stood until
2026-08-30.** That number was true of a sixteen-instance subset and it was
hiding six wrong answers on the rest — three reporting `optimal` at a point
missing the constraints by up to 1.8e+09. One cause: the ratio test never
compared pivot magnitudes when breaking ties. The full account is in
[docs/RESULTS.md §5](docs/RESULTS.md).

## Quick start

```bash
git clone https://github.com/team-vertexx/sankhya && cd sankhya
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
ctest --test-dir build
```

```bash
# LP, first-order method (this is the path the CUDA backend accelerates)
build/sankhya solve   data/netlib/25fv47.mps --tol=1e-8 --presolve

# LP, simplex — exact, with a certificate
build/sankhya simplex data/netlib/25fv47.mps --presolve

# LP, first-order seeding the simplex: fast to close, then exact
build/sankhya simplex data/netlib/degen3.mps --crossover

# MILP
build/sankhya milp    data/miplib/flugpl.mps --time-limit=30

# The refinery planning model the problem statement is about
build/sankhya solve   data/refinery/refinery.mps --presolve
```

Verify the whole thing against published optima:

```bash
python3 scripts/fetch_netlib.py
python3 -u bench/verify_simplex.py 60
```

Full onboarding — layout, conventions, how to run each benchmark — is in
[docs/ONBOARDING.md](docs/ONBOARDING.md).

## Documentation

| document | what it is for |
|---|---|
| [docs/ONBOARDING.md](docs/ONBOARDING.md) | build it, run it, find your way around |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how each component works and why it is built that way |
| [docs/RESULTS.md](docs/RESULTS.md) | every measured number, and the command that reproduces it |
| [docs/ROADMAP.md](docs/ROADMAP.md) | what is next, with the papers behind each item |
| [docs/KAGGLE.md](docs/KAGGLE.md) | running the GPU benchmarks on a rented T4 |
| [RESEARCH.md](RESEARCH.md) | the survey done before any code was written |
| [PLAN.md](PLAN.md) | the build plan written from that research |
| [docs/HANDOVER.md](docs/HANDOVER.md) | design decisions, the testing strategy, and what is next |

Two PDFs, both generated from the documents above:

| | pages | what |
|---|---|---|
| [Sankhya_Complete_Guide.pdf](Sankhya_Complete_Guide.pdf) | 41 | the mathematics of every method from first principles, every rejected option with its measurement, and every wrong answer this solver has produced |
| [Sankhya_Handover.pdf](Sankhya_Handover.pdf) | 23 | design decisions, how the thing is tested and what each layer catches, where it stands, and what is next |

`RESEARCH.md` and `PLAN.md` are dated 22 Aug 2026 and describe the project
*before* it was built. They are kept because the reasoning in them is still the
reasoning, not because they describe the current code. For the current code,
read `ARCHITECTURE.md`.

## A note on how this repository is written

Two conventions worth knowing before reading the source, because both are
unusual and both are deliberate.

**Every tuned constant carries its measurement.** When you find a number in this
codebase, the comment above it says what was tried and what happened. The
refactorization frequency, the dual stall window, the LDL' fill budget, the
polish trigger — each has its sweep recorded next to it. The rule is that a
constant without a measurement is a guess, and guesses get labelled as guesses.

**Failures are recorded, not deleted.** Approaches that were tried and lost are
written down with their numbers so they are not tried again: the Harris ratio
test without EXPAND, Gomory cuts on by default, Halpern's own restart criterion,
c-MIR restricted to original rows, dual steepest edge inside branch and bound.
Each cost real time to disprove. `git log` is the other half of this — the
commit messages carry the reasoning, not just the change.

## Team

| | |
|---|---|
| Abhishek Kumar | [@Abhishek-Kumar-2312](https://github.com/Abhishek-Kumar-2312) |

Built for Smart India Hackathon 2026, problem statement SIH26119.

## Reproducing any number in this repository

Every measurement has the command that produces it, in
[docs/RESULTS.md](docs/RESULTS.md). The benchmark harnesses live in `bench/`:

| script | what it checks |
|---|---|
| `bench/verify_simplex.py` | all 88 Netlib instances against published optima |
| `bench/miplib_survey.py` | 103 MIPLIB instances against published optima |
| `bench/verify_presolve.py` | presolve does not change any answer |
| `bench/verify_reader.py` | the reader agrees with HiGHS on all 88 models |
| `bench/crossover_sweep.py` | pivots saved by seeding the simplex |
| `bench/milp_vs_highs.py` | this solver and HiGHS, same instances, same limit |
| `bench/ablation.py` | every optional feature on and off |

Instance sets are fetched, not vendored: `scripts/fetch_netlib.py`,
`scripts/fetch_miplib.py`, `scripts/fetch_lptestset.py`.
