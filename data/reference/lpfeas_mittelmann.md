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

## What to do with this

Run our solver on the eight we have at `--tol=1e-6`, and report solved/not-solved
against the table above, stating the hardware plainly. That is a comparison a
reader can check, and it is worth more than any score computed against a rubric
we wrote ourselves.
