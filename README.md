# Anukulan

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/team-vertexx/anukulan/blob/master/notebooks/anukulan_demo.ipynb)
**Run it yourself in about four minutes, no install:** the notebook builds the
solver from source, runs the tests, and reproduces plant memory and the 60-case
stack measured below on a free CPU runtime.

**अनुकूलन (anukūlan)** is Hindi for both *optimisation* and *adaptation*, which is
what this solver is built to become: an optimiser that adapts to the plant it
serves. Earlier versions used the working name Sankhya.

An LP / MILP / QP solver written from scratch in C++20, with a CUDA backend for
the first-order method. Built for SIH 2026, problem statement **SIH26119**
(MRPL: an indigenous GPU-accelerated optimization solver for refinery planning).

Nothing here wraps an existing solver. There is no HiGHS, no OR-Tools and no
SciPy underneath: the sparse matrix type, the LU factorisation, the simplex, the
interior point and its sparse Cholesky, the branch and bound tree, the ADMM loop
and the CUDA kernels are all in `src/`.
HiGHS appears in this repository exactly once: as the thing we benchmark
*against*.

```
19,436 lines of solver and CLI     16 test suites, all passing
88 Netlib instances verified against published optima: 0 wrong answers
```

---

## The workload it is built for: a refinery plan, re-solved every day

A refinery does not solve one model once. It re-solves the same plan every day
as crude prices, product prices and demands move, and it weighs each decision
over a stack of cases. Anukulan is being built around that workload:

- **Plant memory** (measured below): every certified solve is kept, and the next
  solve of the same model family starts from it.
- **The case stack** (measured below on the CPU; the GPU batch is next): every
  what-if case differs from the base case in one piece of data, so each starts
  from the base case's basis. On the GPU the cases share the matrix, so one
  first-order pass can bound all of them together.
- **Heuristics that evolve for the plant** (planned): a candidate heuristic is
  kept only if it is faster on held-out models of the same plant, with every
  answer still certified.

### Plant memory, measured

A refinery planning model re-solved day after day, with crude prices, product
prices and demands moving (`scripts/refinery_family.py`, seeded). Each day is
solved twice on the identical model: cold, the way every solve starts today,
and from the basis the previous day ended on. Dual simplex both times.

| plan | rows | re-solves | pivots, cold to warm | seconds, cold to warm | answers that differ |
|---|---|---|---|---|---|
| 12 periods | 1,380 | 29 | 127,303 to 5,701 (0.045x) | 15.23 to 1.19 (0.078x) | 0 |
| 24 periods | 2,760 | 9 | 154,635 to 3,770 (0.024x) | 38.16 to 1.88 (0.049x) | 0 |
| 52 periods | 5,980 | 5 | 115,957 to 5,148 (0.044x) | 81.10 to 8.07 (0.100x) | 0 |

```bash
python3 -u bench/plant_memory.py        # all three, writes bench/results/plant_memory.txt
```

### A case stack, measured

The base plan and sixty what-if cases around it (`scripts/refinery_cases.py`):
one crude's price moved by 3% or 6% either way, one crude's supply halved, one
product's demand moved by 10% either way, or the CDU, the FCC or the
hydrotreater cut back in one of the first four periods. Each case is solved
twice on the identical model: cold, and from the base case's basis. Dual
simplex both times.

| plan | rows | cases | pivots, cold to warm | seconds, cold to warm | answers that differ |
|---|---|---|---|---|---|
| 12 periods | 1,380 | 60 | 195,152 to 5,986 (0.031x) | 20.05 to 1.00 (0.050x) | 0 |
| 24 periods | 2,760 | 60 | 764,471 to 11,794 (0.015x) | 162.81 to 4.42 (0.027x) | 0 |

```bash
python3 -u bench/case_stack.py          # both, writes bench/results/case_stack.txt
```

Why the GPU batch comes second: on this model the first-order method takes
102,400 iterations to reach a relative tolerance of 1e-4 (with presolve), where
a cold simplex takes about 3,000 pivots and a warm one about 100. For a plan
this size the GPU earns its place by bounding every case of a large stack at
once; the cases that decide are finished exactly by the warm simplex.

## What it does

| | what | where |
|---|---|---|
| **Reader** | MPS free and fixed format, LP and QP (Maros–Meszaros QPS) | `src/anukulan/mps_reader.cpp` |
| **Presolve** | 12 reductions + postsolve, including the dual | `src/anukulan/presolve.cpp` |
| **First-order LP** | PDHG / PDLP with restarts, Halpern, feasibility polishing | `src/anukulan/pdhg.cpp` |
| **GPU** | CUDA backend for the first-order method | `src/anukulan/cuda_backend.cu` |
| **Simplex** | revised primal *and* dual, sparse LU, Devex, steepest edge | `src/anukulan/simplex.cpp` |
| **MILP** | branch and cut: cover, c-MIR, Gomory; reliability branching; best-estimate node selection; pump, diving, RINS | `src/anukulan/branch_and_bound.cpp` |
| **QP** | OSQP-style ADMM with a sparse LDL' of the KKT system | `src/anukulan/qp.cpp` |
| **Crossover** | turns a first-order point into a simplex basis | `src/anukulan/crossover.cpp` |

## Where it stands

Every number below was measured on this machine and can be reproduced with the
command next to it in [docs/RESULTS.md](docs/RESULTS.md). None of it is
estimated.

- **Reader**: matches HiGHS on all 88 Netlib instances, 1.4–1.5× faster
- **Simplex**: **82 of 88** Netlib instances reach the published optimum and
  **none returns a wrong answer**; the other six stop visibly with a time limit,
  iteration limit or numerical error
- **Crossover**: seeding the simplex from a first-order point takes 0.52x the
  pivots over Netlib, and turns instances that returned no answer at all, such as
  `degen3` and `stocfor2`, into correct ones
- **First-order**: 76/88 Netlib published optima at `--tol=1e-8`
- **Presolve**: removes 23.3% of Netlib rows and 20.9% of its columns, and
  29.7% of the refinery model's columns, where it previously removed none.
  1.36× fewer simplex iterations, without changing an answer, and it recovers
  duals as well as primals
- **MILP**: on 70 MIPLIB instances at a 15 s limit, **45 end with a feasible
  solution and 7 prove optimality** (45 to 47 feasible across five runs), against
  41 and 5 before this work. The rest still find nothing at all, which is where the
  remaining work is
- **QP**: 35 of the 40 smallest Maros–Meszaros instances
- **GPU (Tesla T4)**: **3.14× to 12.07×** on solve time over the same algorithm on
  CPU, verified on hardware with the backend contract tests passing and CPU/GPU
  agreement at machine precision
- **Plant memory**: re-solving a 30-day refinery family from the previous day's
  basis takes 0.045x the pivots and 0.078x the time, with identical answers
- **Case stack**: sixty what-if cases of the refinery plan, each started from
  the base case's basis, take 0.050x the time at 12 periods and 0.027x at 24,
  with identical answers

Read [docs/RESULTS.md](docs/RESULTS.md) for the full tables and the honest
assessment of where this sits against a production solver: about **62% of
HiGHS**, with MILP still the weak leg at roughly 40. That is not modesty; it is in the
results document with a component-by-component breakdown and the reasoning for
each number.

**The correctness figure above replaced a "16 of 16" that stood until
2026-08-30.** That number was true of a sixteen-instance subset and it was
hiding six wrong answers on the rest, three reporting `optimal` at a point
missing the constraints by up to 1.8e+09. One cause: the ratio test never
compared pivot magnitudes when breaking ties. The full account is in
[docs/RESULTS.md §5](docs/RESULTS.md).

## Quick start

```bash
git clone https://github.com/team-vertexx/anukulan && cd anukulan
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
ctest --test-dir build
```

```bash
# LP, first-order method (this is the path the CUDA backend accelerates)
build/anukulan solve   data/netlib/25fv47.mps --tol=1e-8 --presolve

# LP, simplex: exact, with a certificate
build/anukulan simplex data/netlib/25fv47.mps --presolve

# LP, first-order seeding the simplex: fast to close, then exact
build/anukulan simplex data/netlib/degen3.mps --crossover

# MILP
build/anukulan milp    data/miplib/flugpl.mps --time-limit=30

# Plant memory: a family of refinery plans, cold against warm from yesterday
python3 scripts/refinery_family.py --days 30
build/anukulan family data/refinery/family/day_*.mps

# A case stack: the base plan and sixty what-ifs, each from the base's basis
python3 scripts/refinery_cases.py
build/anukulan family --from-first data/refinery/cases/case_*.mps

# The refinery planning model the problem statement is about
build/anukulan solve   data/refinery/refinery.mps --presolve
```

Verify the whole thing against published optima:

```bash
python3 scripts/fetch_netlib.py
python3 -u bench/verify_simplex.py 60
```

Full onboarding (layout, conventions, how to run each benchmark) is in
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

Four PDFs, generated from the documents in `docs/` with pandoc:

| | pages | what |
|---|---|---|
| [Anukulan_Complete_Guide.pdf](Anukulan_Complete_Guide.pdf) | 54 | the mathematics of every method from first principles, every rejected option with its measurement, and every wrong answer this solver has produced |
| [Anukulan_Handover.pdf](Anukulan_Handover.pdf) | 23 | design decisions, how the thing is tested and what each layer catches, where it stands, and what is next |
| [Anukulan_Assessment.pdf](Anukulan_Assessment.pdf) | 9 | what works, what does not, and what would fix it |
| [Anukulan_GPU_Report.pdf](Anukulan_GPU_Report.pdf) | 7 | what runs on the device, what it costs, what it was measured at, and the bug the verification found |

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
polish trigger: each has its sweep recorded next to it. The rule is that a
constant without a measurement is a guess, and guesses get labelled as guesses.

**Failures are recorded, not deleted.** Approaches that were tried and lost are
written down with their numbers so they are not tried again: the Harris ratio
test without EXPAND, Gomory cuts on by default, Halpern's own restart criterion,
c-MIR restricted to original rows, dual steepest edge inside branch and bound.
Each cost real time to disprove. `git log` is the other half of this: the
commit messages carry the reasoning, not just the change.

## Author

The solver in `src/` is written by Mohit Prajapati
([@mohitt31](https://github.com/mohitt31)).

## Team

Team_Vertex_, Indian Institute of Technology Kharagpur:

- Abhishek Kumar
- Mohit Prajapati
- Manish Paul
- Kunjika Tripathi
- Reeck Mondal
- Ayush Saha

Built for Smart India Hackathon 2026, problem statement SIH26119.

## Licence

Apache License 2.0: see [LICENSE](LICENSE).

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
| `bench/plant_memory.py` | a refinery plan re-solved day after day, cold and from the previous day's basis |
| `bench/case_stack.py` | sixty what-if cases of a refinery plan, cold and from the base case's basis |

Instance sets are fetched, not vendored: `scripts/fetch_netlib.py`,
`scripts/fetch_miplib.py`, `scripts/fetch_lptestset.py`.
