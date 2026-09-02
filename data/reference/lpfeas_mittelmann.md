# Mittelmann LPfeas benchmark — reference times

Taken from <https://plato.asu.edu/ftp/lpfeas.html>, run of **10 August 2026**.
Copied here so our own runs have something external to be checked against, and
so the comparison does not quietly drift as the page is updated.

This is the benchmark the first-order and GPU work in this repository should be
measured against. It is where cuPDLPx and HPR-LP — the two papers this solver
takes its first-order method from — are actually scored.

## How the reference numbers were produced

- Tolerance **1e-6** for every code.
- CPU codes on a Linux i7-11700K, 3.6 GHz, 64 GB.
- **GPU codes on an NVIDIA B200 (192 GiB)**, 1 GPU × 31 CPUs on AMD EPYC 9575F.
- Time limit 15,000 s for CPU codes, **1,000 s for GPU codes**.
- Scoring is the shifted geometric mean (shift 10 s), not a performance profile.
- `t` = timed out, `f` = failed, `m` = out of memory. All counted as max time.

**Our numbers will not be comparable in wall-clock terms.** A T4 is not a B200
and an M-series laptop is not an i7-11700K. What *is* comparable, and is the
more meaningful comparison anyway, is **which instances get solved at 1e-6**.

## Summary over 65 problems

| code | shifted geomean (s) | scaled | solved |
|---|---|---|---|
| cuOpt 26.08 | 13.7 | 1.00 | 62 |
| COPTG | 16.2 | 1.11 | 64 |
| HPR-LP-C 0.1.2 | 20.6 | 1.51 | 58 |
| COPT 8.0.0 | 25.3 | 1.67 | **65** |
| **cuPDLPx 0.3.0** | **28.5** | **2.08** | **57** |
| MOSEK 11.1.11 | 79.8 | 5.28 | 57 |
| XOPT 0.0.8 | 146 | 9.63 | 59 |
| **HiGHS 1.15.0** | **256** | **16.9** | **55** |
| KNITRO 16.0.0 | 372 | 23.0 | 48 |
| **OR-Tools PDLP 9.10** | **421** | **27.8** | **50** |

Worth sitting with: on this set the GPU first-order codes beat HiGHS, and
OR-Tools' PDLP — the same algorithm family this solver implements, on CPU —
solves the fewest of any code except KNITRO. The method is not automatically
good; the implementation is most of it.

## Instances this repository already has

Eight of the forty public instances. Reference seconds:

| instance | HiGHS | PDLP | cuPDLPx | cuOpt |
|---|---|---|---|---|
| graph40-40 | 13 | 2 | 1 | 1 |
| qap15 | 7 | 3 | 1 | 1 |
| datt256 | 7 | 2 | 1 | 1 |
| support10 | 19 | 8 | 1 | 1 |
| cont1 | 6 | 256 | 22 | 1 |
| cont11 | 14 | 1988 | 60 | 1 |
| bdry2 | t | t | t | 8 |
| irish-e | 18 | t | 24 | 2 |

`cont11` is the interesting row: OR-Tools PDLP takes 1,988 s and cuPDLPx 60 s on
the same algorithm family, while HiGHS does it in 14 s. Neither the method nor
the hardware decides that on its own.

## The rest of the public set, not yet fetched

    L1_sixm  Linf_520c  a2864  dlr1  ex10  fhnw-bin1  fome13  neos  neos3
    neos3025225  neos5052403  neos5251015  ns1687037  ns1688926  nug08-3rd
    pds-100  psched3-3  rail02  rail4284  rmine15  s82  s100  s250r10
    savsched1  scpm1  shs1023  square41  stat96v2  storm_1000  stp3d
    tpl-tub-ws  woodlands09

Files are at <https://plato.asu.edu/ftp/lptestset/> and some need expanding with
`emps`. `scripts/fetch_lptestset.py` already knows how to pull from there.

Sixteen further instances in the benchmark are undisclosed and cannot be
reproduced.

## Where this solver's method sits

Chen, Sun, Yuan, Zhang and Zhao, [*On the Relationships among GPU-Accelerated
First-Order Methods for Solving Linear Programming*](https://arxiv.org/abs/2509.23903),
settles what these methods actually are relative to each other. Proposition 3.1:
**cuPDLPx's iteration with the reflection parameter at 1 is the Halpern
Peaceman–Rachford method**, with the semi-proximal term taken as
`T1 = lambda_A I - A A*`.

Our default reflection is 1.0, so this repository implements an HPR method — not
something adjacent to one. HPR-LP is the same algorithm with a freer choice of
that term and a more developed restart and penalty-parameter strategy.

The same paper measures the distance, at accuracy 1e-8:

| | HPR-LP | cuPDLPx |
|---|---|---|
| Mittelmann LP benchmark (49) | 82.1 s, 44 solved | 90.7 s, 44 solved |
| MIPLIB 2017 relaxations | +2 instances, 1.8× faster | — |

Their conclusion, in their words, is that both are effective realizations of the
same method and the difference is implementation rather than algorithm. That is
the useful part: **the remaining gap to the frontier is tuning and engineering,
not a method we do not have.**

## What to do with this

Run our solver on the eight we have at `--tol=1e-6`, and report solved/not-solved
against the table above, stating the hardware plainly. That is a comparison a
reader can check, and it is worth more than any score computed against a rubric
we wrote ourselves.


## Our run, 2026-09-02

Tolerance 1e-6, presolve on, 300 s limit, single threaded, on the machine at the
top of `docs/RESULTS.md` — an M-series laptop, not the i7-11700K the CPU
reference times were taken on, and certainly not the B200 the GPU ones were.
**The machine was shared with other work for this run, so the times are
pessimistic.** What is comparable is which instances get solved.

| instance | this solver | HiGHS | OR-Tools PDLP | cuPDLPx |
|---|---|---|---|---|
| graph40-40 | **1.5 s** | 13 | 2 | 1 |
| datt256_lp | **2.3 s** | 7 | 2 | 1 |
| qap15 | **4.8 s** | 7 | 3 | 1 |
| supportcase10 | **102 s** | 19 | 8 | 1 |
| cont1 | iteration limit | 6 | 256 | 22 |
| cont11 | iteration limit | 14 | 1988 | 60 |
| irish-electricity | iteration limit | 18 | t | 24 |
| bdry2 | time limit | t | t | t |
| brazil3 | **0.4 s** | — | — | — |
| sgpf5y6 | numerical error | — | — | — |
| watson_1 | time limit | — | — | — |

**Five of eleven solved; four of the eight that are in the published table.**

The useful comparison is not against HiGHS, which is a mature simplex and solves
`cont1` in six seconds. It is against **OR-Tools PDLP** — the same algorithm
family as this solver, implemented on CPU by the group that wrote the PDLP
paper. The failure pattern is the same one: PDLP takes 256 s on `cont1`, 1,988 s
on `cont11`, and times out on `irish-electricity`, which are exactly the three
this solver cannot finish. On three of the four both solve, this solver's time
is comparable or better; on `supportcase10` it is twelve times slower.

`bdry2` is solved by nothing on this table except cuOpt.

Two honest gaps this exposes. `sgpf5y6` ends in a **numerical error**, which is a
failure of this implementation and not of the method. And `supportcase10` at 102
seconds against PDLP's 8 says the constant factor is well behind even the CPU
reference for this family — which is consistent with arXiv:2509.23903's finding
that the remaining distance to the frontier is implementation rather than
algorithm.
