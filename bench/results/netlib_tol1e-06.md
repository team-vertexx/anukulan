### Netlib LP, relative tolerance 1e-06

Accuracy is the relative distance from the objective published in the Netlib index, not the tolerance the solver was asked for.

| instance | rows x cols | sankhya iters | HiGHS-PDLP iters | ratio | sankhya acc | HiGHS-PDLP acc | HiGHS simplex s | sankhya s |
|---|---|---:|---:|---:|---|---|---:|---:|
| afiro | | 360 | 320 | 1.12x | 2.60e-07 | 1.35e-07 | 0.007 | 0.007 |
| sc50a | | 1000 | 480 | 2.08x | 2.96e-06 | 1.75e-06 | 0.006 | 0.005 |
| adlittle | | 16560 | 7240 | 2.29x | 1.82e-06 | 9.20e-07 | 0.008 | 0.024 |
| blend | | 2320 | 1320 | 1.76x | 9.76e-08 | 5.25e-07 | 0.007 | 0.009 |
| share1b | | 60560 | 18480 | 3.28x | 2.49e-07 | 1.11e-06 | 0.010 | 0.172 |
| stocfor1 | | 13120 | 5360 | 2.45x | 9.91e-07 | 9.28e-08 | 0.007 | 0.025 |
| sctap1 | | 2520 | 2200 | 1.15x | 1.09e-06 | 1.20e-07 | 0.014 | 0.022 |
| scfxm1 | | 15440 | 12640 | 1.22x | 1.47e-06 | 1.21e-06 | 0.017 | 0.116 |
| bandm | | 32160 | 22400 | 1.44x | 4.67e-07 | 8.36e-07 | 0.018 | 0.234 |
| degen2 | | 3920 | 3040 | 1.29x | 1.13e-07 | 8.97e-08 | 0.024 | 0.051 |
| fit1p | | 30240 | 28680 | 1.05x | 4.14e-09 | 7.76e-09 | 0.060 | 0.972 |
| 25fv47 | | 73560 | 48840 | 1.51x | 2.00e-06 | 1.20e-06 | 0.174 | 2.070 |
| woodw | | 64760 | 11600 | 5.58x | 7.60e-10 | 2.46e-07 | 0.099 | 8.257 |
| degen3 | | 19600 | 20000 | 0.98x | 2.42e-07 | 5.86e-07 | 0.191 | 1.186 |
| stocfor2 | | 87280 | 29360 | 2.97x | 3.73e-08 | 3.28e-08 | 0.041 | 2.928 |
| greenbea | | 1000000 | 2329102 | 0.43x | 9.15e-05 | 1.00e+00 | 0.263 | 89.935 |
| pilot87 | | 271200 | 168360 | 1.61x | 4.95e-06 | 1.72e-05 | 3.885 | 47.751 |
| maros-r7 | | 5840 | 3720 | 1.57x | 1.27e-07 | 2.04e-07 | 0.674 | 1.851 |

Geometric mean iteration ratio against HiGHS-PDLP: **1.61x** over 18 instances.
