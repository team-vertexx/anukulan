# Onboarding

Getting the thing built, run and understood. Written for someone joining who
knows the maths but not this codebase.

---

## 1. Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

Thirteen suites, all should pass. No dependencies beyond a C++17 compiler and
CMake — no Boost, no Eigen, nothing to install. That is deliberate: the problem
statement asks for an indigenous solver, and a dependency is a thing you have to
explain.

With CUDA:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DSANKHYA_ENABLE_CUDA=ON
cmake --build build-cuda -j8
build-cuda/sankhya backends      # reports which backends this build has
```

On a machine with no GPU you can still type-check the kernels:

```bash
bash scripts/check_cuda_syntax.sh
```

It stubs the CUDA runtime, strips the `<<<…>>>` launch syntax and compiles with
clang. It has caught real errors before a Kaggle round trip, which is worth
about forty minutes each time.

---

## 2. Getting the data

Benchmark sets are not in the repository (see `.gitignore`) because they are
large and freely available:

```bash
python3 scripts/fetch_netlib.py            # 88 Netlib LP instances
python3 scripts/fetch_lptestset.py --max=4 # the large instances the GPU needs
python3 scripts/refinery_model.py          # regenerates data/refinery/refinery.mps
```

`data/reference/netlib.csv` holds the published optima and **is** tracked. It is
what every correctness claim is checked against.

---

## 3. Running it

One command per entry point:

```bash
build/sankhya read      model.mps    # statistics only
build/sankhya standard  model.mps    # build the standard form, print its shape
build/sankhya presolve  model.mps    # reduce and report what went
build/sankhya solve     model.mps    # first-order LP
build/sankhya simplex   model.mps    # revised simplex (--dual for the dual)
build/sankhya milp      model.mps    # branch and cut
build/sankhya qp        model.qps    # convex QP by ADMM
build/sankhya backends                # which backends this build has
```

Useful flags, and the ones you will reach for first:

```bash
--tol=1e-8            relative tolerance on all three convergence measures
--gap-tol=1e-2        loosen only the duality gap (see below — this matters)
--abs-tol=1e-6        additionally cap the worst single violation
--presolve            reduce first, map the answer back
--format=json         machine-readable, which every harness in bench/ uses
--verbose             per-iteration logging
--backend=cpu|cuda    force one, which is how the CPU/GPU comparison is done
--solution=out.txt    write the primal, the duals, and an exactness marker
```

`--gap-tol` is the one worth understanding early. Feasibility polishing buys
tight feasibility and pays for it in the duality gap, so `--tol=1e-8
--gap-tol=1e-2` means *"feasible to 1e-8, and I will accept a 1% gap"*. On the
refinery model that is the difference between an answer and no answer. Asking
for both at 1e-8 is what the simplex is for.

Everything is deterministic. Same input, same answer, every time — there is a
test for it, because without it an ablation table means nothing.

---

## 4. Repository layout

```
src/sankhya/     the solver. One header per component, and the reasoning
                 lives in the headers.
app/             the CLI, one function per command
tests/           13 suites, no framework — just CHECK macros
bench/           measurement harnesses (Python), and their recorded results
scripts/         data fetching, CUDA syntax check, GPU test, packaging
data/            reference optima tracked; instance sets fetched
docs/            this, ARCHITECTURE, RESULTS, ROADMAP, KAGGLE
refs/            papers
```

Where to start reading, in this order:

1. `src/sankhya/standard_form.hpp` — the two forms, and why they differ
2. `src/sankhya/pdhg.hpp` — the largest component, and the most documented
3. `src/sankhya/presolve.hpp` — the reductions and the postsolve contract
4. `src/sankhya/simplex.hpp` — pricing, ratio tests, the basis
5. `docs/ARCHITECTURE.md` — how they compose

---

## 5. The benchmark harnesses

All under `bench/`, all Python, all writing to `bench/results/`.

```bash
python3 bench/verify_reader.py                    # reader vs HiGHS
python3 bench/verify_presolve.py --tol=1e-6 --abs-tol=1e-6 --check-feasibility
python3 bench/run_benchmark.py                    # vs HiGHS simplex and PDLP
python3 bench/polish_sweep.py polish              # feasibility polishing
python3 bench/cupdlpx_sweep.py cupdlpx 1e-8       # the cuPDLPx additions
python3 bench/ablation.py                         # each PDHG feature on and off
```

Two flags on `verify_presolve.py` exist for A/B work and are worth knowing:
`--extra` applies flags to the presolved run only, `--both-extra` applies them to
both — which is how a solver option gets A/B'd without breaking the
plain-versus-presolved comparison the harness is built on.

### Measurement traps, all of which have bitten

- **`zsh` does not word-split unquoted variables.** `$FLAGS` containing
  `--presolve --abs-tol=1e-8` arrives as *one* argument, the CLI rejects it, and
  the checker silently reads a stale solution file. This cost three separate
  debugging sessions. Use arrays, or pass flags literally.
- **Piping Python to `tail` buffers everything** until the process ends. Use
  `python3 -u` and redirect to a file.
- **Rebuilding during a background measurement corrupts it** — the harness
  spawns the binary fresh for each instance, so half the run uses the old binary
  and half the new.
- **A measurement taken under load is not a measurement.** One reader run said
  1.61 s → 1.95 s; three runs on a quiet machine said 1.67 s → 1.19 s. The first
  was taken while seven MILP solves were running.
- **`timeout` is not on macOS.**

---

## 6. Conventions

**Every tuned constant carries its measurement.** If you change one, re-run the
sweep and update the comment. If you add one, sweep it first. A constant with no
recorded sweep is a guess and the comment should say so — there are a few, and
they say so.

**Failures get written down, not deleted.** If an approach is tried and loses,
record it with its numbers next to the thing it would have replaced. There is a
list of these in `RESULTS.md` and more in the headers. They cost real time to
disprove and rediscovering them costs it again.

**When a layer gets faster, every tuned number above it is stale.** The
piecewise-linear phase one removed the refactorization cliff, and the
refactorization frequency, node limit, dive warm start, dual stall window and
fill budget all had to be re-measured. This is not optional bookkeeping.

**A solver must not assert what it has not established.** The dual simplex
checks its own optimality claim before making it. That check found a bug on its
first run that had been reporting wrong answers as optimal.

**Tests should not be able to pass for the wrong reason.** Three tests here exist
because an earlier version could: `test_ldl` never checks a residual the
factorisation computed about itself; `test_cuts` separates at simplex vertices,
not only random points; `test_presolve` gives each column a distinct reduced
cost, because an all-zero vector cannot tell two indexings apart.

**Commit messages carry the reasoning.** `git log` is half the documentation.
Read it before changing anything that looks arbitrary — it probably is not.

**No AI attribution anywhere.** Commits, PRs, comments, application materials,
public text. First person, always.

---

## 7. GPU work

The GPU is rented, not owned — see [KAGGLE.md](KAGGLE.md) for the full loop.
Short version:

```bash
bash scripts/package_for_kaggle.sh   # git archive HEAD -> sankhya-source.zip
```

Upload as a **new version** of the existing Kaggle dataset, not a new dataset,
then run `scripts/gpu_test.sh`. Its seven steps, in the order that matters:

1. device — what card did we get
2. build — does it compile with nvcc
3. **backend operations against the CPU reference** — the correctness gate
4. solver on GPU against published Netlib optima
5. CPU against GPU by size — the speedup table
6. presolve on GPU, and where the fused reduction should show
7. per-kernel profile on graph40-40

Step 3 first, always. Anything that has not run on a GPU since the last trip is
untested there, and the CPU hides a whole class of bug: `cudaMalloc` does not
zero where the CPU allocator does, and `release()` frees every cached matrix
where the CPU no-ops.
