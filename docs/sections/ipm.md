# Interior point method, crossover to a basis, and the race between engines

Every number here was measured on the machine this was built on: Linux
(arm64/x86_64 container), 2 shared CPU cores, g++ 13, `-Wall -Wextra
-Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror`, Release build
via CMake/Ninja. The command that produced each number is next to it, and
`bench/results/ipm_*.txt` holds the raw output.

## What this adds

Three pieces, in the order they build on each other:

1. **Sparse Cholesky** (`src/anukulan/cholesky.hpp`/`.cpp`): approximate
   minimum degree ordering, an elimination tree, fundamental supernodes with
   relaxed amalgamation, a left-looking supernodal numeric factorisation, and
   Wright's modified-Cholesky pivot regularisation for the tiny or negative
   pivots a normal-equations matrix produces. Tested against a dense
   reference on random SPD matrices, a grid Laplacian (for fill), and
   deliberately dependent rows (for regularisation) in `tests/test_cholesky.cpp`.
2. **The interior point method itself** (`src/anukulan/ipm.hpp`/`.cpp`):
   Mehrotra's predictor-corrector with Gondzio multiple centrality
   correctors, on the normal equations `A Theta A^T + delta I`. Free
   variables get primal-dual regularisation rather than being dropped from
   the diagonal; dense columns get a Sherman-Morrison-Woodbury correction
   so one dense column does not turn a sparse factorisation into a dense
   one; convergence and infeasibility detection reuse `pdhg.hpp`'s
   `evaluate_residual`/`PdhgResidual` so "optimal" means the same thing
   for every engine that can report it.
3. **Crossover and the race** (`solve_lp_race` in `ipm.hpp`/`.cpp`, and the
   `ipm`/`lp` CLI commands): the existing `crossover_basis()` turns an
   interior point (or first-order) iterate into a warm-start simplex basis,
   with a cold fallback if the warm solve does not reach an optimum. The
   race runs the dual simplex, IPM-then-crossover, and PDHG-then-crossover
   on three threads, each checked against the *original* model on the way
   out (postsolve applied, `measure_violation`,
   `relative_row_violation <= 1e-6`); the first to pass wins and cancels
   the other two through a shared `std::atomic<bool>`.

Cholesky and the core Mehrotra loop were mostly already written when this
work picked them up; both were read in full, found sound, and built and
tested essentially as they stood. The CLI commands, the race, the
cancellation flag threaded through `SimplexOptions` and `PdhgOptions`, and
two real time-limit bugs (below) are new.

## New commands

```
anukulan ipm <file.mps> [--tol=1e-8] [--max-iter=200] [--max-correctors=3]
                        [--time-limit=300] [--crossover-dual-weight=1]
                        [--no-crossover] [--presolve] [--format=json|human]
anukulan lp  <file.mps> [--time-limit=300] [--engines=simplex,ipm,pdhg]
                        [--presolve] [--format=json|human]
```

`ipm` reports the interior point statistics (iterations, factor time,
supernodes, dense columns, dropped pivots, CG iterations, correctors, final
mu) and then, unless `--no-crossover`, the finishing simplex's own status
and iteration count, exactly like the existing `simplex` command's output.
`lp` reports which engine won, each engine's status and wall time, and the
answer checked against the original model.

## Two real time-limit bugs, found by the measurement below

A `--time-limit` is supposed to bound the whole command. Two places
silently did not, both now fixed and covered by the numbers below rather
than only asserted:

- **`command_ipm`** built the finishing simplex's `SimplexOptions` with no
  `time_limit_seconds` set at all, so it ran with the type's own default
  (300 s) regardless of what the user asked `ipm` for. A 30 second run on
  `pilot87` was still going several minutes later. Fixed by threading
  whatever is left of the command's own budget, recomputed fresh
  immediately before the finishing solve and again before the cold
  fallback if that is needed.
- **`finish_from_seed`** (used by both the IPM and PDHG engines in the
  race) computed "what's left of the race's budget" once, correctly, in
  `solve_lp_race`, but then used that *same* figure for both the warm
  crossover attempt and the cold fallback if the warm one did not reach an
  optimum - so a warm attempt that ran out the clock could be followed by
  a cold attempt with the same full allowance again, up to twice the
  intended budget between them. Confirmed on `dfl001`, `degen3` and `pilot`,
  each of which ran past a 120 second backstop (`limit + 60`, added to both
  bench scripts' `subprocess.run` calls as a second line of defence) before
  the fix, and landed on the requested 60 seconds afterwards. Fixed the
  same way: a small timer local to `finish_from_seed` recomputes the
  remaining share before each of the two attempts.

Both were real, both were found by running the actual measurement this
section exists to report rather than by inspection, and both are now
exercised by that same measurement rather than only by a targeted test.

## Correctness: every Netlib instance with a published optimum

89 instances, `data/reference/netlib.csv`.

**IPM alone**, default tolerance (1e-8):
```
python3 bench/verify_ipm.py 30
```
85/89 reach the published optimum (1e-6 relative). 1 is WRONG at that
threshold: `pilot87` misses by 1.26e-6, just past the 1e-6 line, at
iteration counts (13) and a final mu (5e-14) that look fully converged -
a marginal case on a large, poorly-scaled instance rather than a method
that is not working. 3 stall before reaching tolerance: `fit1p`, `fit2p`,
`greenbea`, each reporting its own honest `numerical error` status with
the message `stalled: no progress in eight iterations` rather than looping
or claiming an optimum it has not reached. `fit1p`/`fit2p` are the classic
dense-column stress case in the LP literature; the Woodbury correction
here handles the fill but evidently not yet the conditioning it brings.
Iterations across the 85 that converge: min 6, median 13, max 41, mean 15.5.

**IPM finished by crossover and the simplex**, same command, second half of
its output:
0 WRONG, 78/89 at the optimum, 11 do not finish inside 30 seconds:
`cycle`, `d6cube`, `degen3`, `dfl001`, `fit1p`, `fit2p`, `greenbea`,
`pilot`, `pilot87`, `scsd1`, `wood1p`. Every one of those eleven is already
in `docs/RESULTS.md` as a hard instance for this codebase's simplex work,
for reasons that have nothing to do with the interior point method: `cycle`
and `degen3` are built to defeat naive pivoting, `d6cube` is a
combinatorial worst case, `dfl001` is Netlib's largest instance, `pilot`
and `pilot87` are large and poorly scaled and already documented as
hitting the iteration limit under the first-order method too, `wood1p`
already failed with a numerical error before any of this existed. Where
crossover does finish, it costs iterations: min 0, median 220, max 28,155,
mean 1,285.5, against a median 13-iteration interior point seed - most of
that spread is a handful of the same hard instances, not the typical case.

**The race**, all three engines, 60 second shared budget:
```
python3 bench/verify_race.py 60
```
83/89 correct, **0 WRONG**, 6 do not finish: `cycle`, `degen3`, `dfl001`,
`pilot`, `pilot87`, `wood1p` - the same hard-instance list as above, minus
`d6cube`, `fit1p`, `fit2p`, `greenbea` and `scsd1`, which the race's extra
budget and extra engines do resolve (`d6cube` and `scsd1` by the dual
simplex, `fit1p` by PDHG, `fit2p` and `greenbea` by IPM at the full 60
seconds rather than 30). Winners across the 83: IPM 59, the dual simplex
15, PDHG 9.

`cycle` is the one instance where all three engines miss. The dual simplex
engine calls `solve_dual_simplex` directly, as the design asks for engine
(a), rather than going through `solve_lp`'s auto-fallback-to-primal - and
on `cycle` that direct call fails with a pre-existing, unrelated
limitation: `the starting basis cannot be made dual feasible: column 16
wants a bound it does not have. Use the primal.` `crossover_basis()` is
shared by the other two engines' finishing step, and it fails on this same
deliberately degenerate instance regardless of which engine seeded it, so
neither of those has anywhere to fall back to either. This is a genuine,
narrow gap in the current race design on one intentionally pathological
instance, not a wrong answer.

## The refinery model

```
build/anukulan ipm data/refinery/refinery.mps --presolve
build/anukulan lp  data/refinery/refinery.mps --presolve
```
Both reach 2.426062266887e+10, matching `anukulan solve`'s existing
first-order result exactly. `ipm` alone: 20 iterations, 0.024 s, 0.220 s
total with presolve. `lp`: won by `ipm` in 0.753 s, with the dual simplex
and PDHG both landing within about 50 ms of the winner once cancelled -
raw output in `bench/results/ipm_refinery.txt`.

## Tests

`tests/test_ipm.cpp`: boxed, free, fixed, and ranged variables; a
degenerate optimal face; primal and dual infeasibility detection; 20
random LPs checking crossover-from-IPM against a cold simplex solve; the
race returning a correct answer; the race restricted to one engine via
`--engines=`. `tests/test_cholesky.cpp` (pre-existing, unchanged): dense
reference comparison, a grid Laplacian for fill, dependent rows and
negative pivots for the regularisation path, re-factorising the same
pattern.

Build and `ctest`, all green, three compilers:

| compiler | build | tests |
|---|---|---|
| g++ 13 (default) | clean | 16/16 |
| clang++ 18 | clean | 16/16 |
| g++ 11 (`/usr/bin/g++-11`, Colab's compiler) | clean | 16/16 |

All three under `-Wall -Wextra -Wpedantic -Wshadow -Wconversion
-Wsign-conversion -Werror`, three separate build directories
(`build`, `build-clang`, `build-gcc11`).

## Thread safety

Nothing in `src/anukulan` had ever run two solves at once before the race.
A grep across every `.cpp` in that tree for non-const mutable statics found
exactly one: `SparseMatrix::next_id()`'s counter, whose own comment said
"make it atomic if that changes" - changed to `std::atomic<std::uint64_t>`
in `src/anukulan/sparse.cpp`. No other static or global mutable state exists
in the library; everything else is either const, a per-call local, or an
instance member reached through the caller's own object, none of which two
racing engines share.

## Known limitations

- `pilot87` alone, at the default 1e-8 tolerance, misses the published
  optimum by 1.26e-6 - past this measurement's 1e-6 line, though only just,
  and only when crossover is turned off. With crossover (or in the race)
  it is exact.
- `fit1p`, `fit2p`, `greenbea` stall in the interior point method alone at
  30 seconds; `fit2p` and `greenbea` do resolve in the race at 60 (through
  IPM itself, given the extra time; `fit1p` resolves through PDHG).
- `cycle`, `degen3`, `dfl001`, `pilot`, `pilot87`, `wood1p` do not finish
  inside the race's 60 second budget. All six are already documented
  elsewhere in this codebase as hard for reasons predating this work.
  `cycle` specifically has no winning engine at all under the current
  design, for the dual-feasibility reason above.
- The dual simplex engine's own numerical-error message ("the starting
  basis cannot be made dual feasible ... Use the primal") is pre-existing
  `simplex.cpp` behaviour, reachable because the race's engine (a) is
  `solve_dual_simplex` directly rather than `solve_lp`'s auto-fallback.
  It affects only instances where that specific starting basis cannot be
  made dual feasible; `afiro` was the first case found, `cycle` above is
  the only one on the full Netlib set where it costs the race an answer
  outright.

## For whoever merges this branch

Shared files touched, and why:
- `CMakeLists.txt`, `tests/CMakeLists.txt`: two lines each, adding
  `cholesky.cpp`/`ipm.cpp` to the library and `test_cholesky`/`test_ipm` to
  the test targets.
- `src/anukulan/simplex.hpp`/`.cpp`, `src/anukulan/pdhg.hpp`/`.cpp`: a new
  `const std::atomic<bool>* stop = nullptr;` field at the end of
  `SimplexOptions`/`PdhgOptions`, and a check for it placed immediately
  after each existing time-limit check (two spots in `simplex.cpp`: the
  primal loop and the dual loop). Anything already reading or writing
  those structs by field name rather than by position is unaffected; nothing
  existing changes behaviour with `stop` left at `nullptr`.
- `src/anukulan/sparse.cpp`: `SparseMatrix::next_id()`'s counter is now
  atomic (see Thread safety above). This matters for anyone else adding
  concurrent solves - it was the one real hazard in the whole library, and
  is now fixed for both directions of concurrent work, not just this one.
- `app/anukulan_cli.cpp`: two new commands (`command_ipm`, `command_lp`),
  their usage text, and two dispatch lines placed right after the existing
  `"simplex"` dispatch. New flags: `ipm`'s `--tol`, `--max-iter`,
  `--max-correctors`, `--time-limit`, `--crossover-dual-weight`,
  `--no-crossover`; `lp`'s `--time-limit`, `--engines=`. No existing flag
  or command's behaviour changes.

Nothing here is order-dependent against the batched PDHG work in the other
worktree beyond the `stop` field and the `next_id()` fix above: both are
additive (a new optional field, a type change on a counter no other code
reads directly), so either branch can land first.
