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
11,192 lines of solver and CLI     11 test suites, all passing
```

---

## What it does

| | what | where |
|---|---|---|
| **Reader** | MPS free and fixed format, LP and QP (Maros–Meszaros QPS) | `src/sankhya/mps_reader.cpp` |
| **Presolve** | 10 reductions + postsolve, including the dual | `src/sankhya/presolve.cpp` |
| **First-order LP** | PDHG / PDLP with restarts, Halpern, feasibility polishing | `src/sankhya/pdhg.cpp` |
| **GPU** | CUDA backend for the first-order method | `src/sankhya/cuda_backend.cu` |
| **Simplex** | revised primal *and* dual, sparse LU, Devex, steepest edge | `src/sankhya/simplex.cpp` |
| **MILP** | branch and cut — cover, c-MIR, Gomory; pseudocost; diving | `src/sankhya/branch_and_bound.cpp` |
| **QP** | OSQP-style ADMM with a sparse LDL' of the KKT system | `src/sankhya/qp.cpp` |

## Where it stands

Every number below was measured on this machine and can be reproduced with the
command next to it in [docs/RESULTS.md](docs/RESULTS.md). None of it is
estimated.

- **Reader** — matches HiGHS on all 88 Netlib instances, 1.4–1.5× faster
- **Simplex** — primal 16/16 and dual 16/16 against published Netlib optima
- **First-order** — 76/88 Netlib published optima at `--tol=1e-8`
- **Presolve** — removes 14.0% of rows, 15.4% of columns; 1.56× geomean speedup
- **MILP** — 5 of 7 MIPLIB instances exact; `gen-ip054` 0.330%, `mas76` 0.562%
- **QP** — 21 of the 24 smallest Maros–Meszaros instances
- **GPU (Tesla T4)** — 2.70×–7.09× on solve time over the same algorithm on CPU

Read [docs/RESULTS.md](docs/RESULTS.md) for the full tables and the honest
assessment of where this sits against a production solver, which is roughly
**half of HiGHS**. That number is not modesty — it is in the results doc with a
component-by-component breakdown.

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
ctest --test-dir build
```

```bash
build/sankhya solve data/netlib/25fv47.mps --tol=1e-8 --presolve
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
