### Netlib LP, relative tolerance 1e-06

Accuracy is the relative distance from the objective published in the Netlib index, not the tolerance the solver was asked for.

| instance | rows x cols | sankhya iters | HiGHS-PDLP iters | ratio | sankhya acc | HiGHS-PDLP acc | HiGHS simplex s | sankhya s |
|---|---|---:|---:|---:|---|---|---:|---:|
| afiro | | 360 | 320 | 1.12x | 2.60e-07 | 1.35e-07 | 0.010 | 0.007 |
| sc50a | | 1000 | 480 | 2.08x | 2.96e-06 | 1.75e-06 | 0.009 | 0.007 |
| adlittle | | 16560 | 7240 | 2.29x | 1.82e-06 | 9.20e-07 | 0.015 | 0.029 |
| blend | | 2640 | 1320 | 2.00x | 8.01e-08 | 5.25e-07 | 0.013 | 0.014 |
| share1b | | 61040 | 18480 | 3.30x | 2.29e-08 | 1.11e-06 | 0.010 | 0.217 |
| stocfor1 | | 13200 | 5360 | 2.46x | 2.66e-07 | 9.28e-08 | 0.009 | 0.027 |
| sctap1 | | 2520 | 2200 | 1.15x | 1.09e-06 | 1.20e-07 | 0.013 | 0.022 |
| scfxm1 | | 17560 | 12640 | 1.39x | 1.61e-07 | 1.21e-06 | 0.018 | 0.137 |
| bandm | | 32200 | 22400 | 1.44x | 4.76e-07 | 8.36e-07 | 0.020 | 0.253 |
| degen2 | | 3920 | 3040 | 1.29x | 1.13e-07 | 8.97e-08 | 0.028 | 0.055 |
| fit1p | | 33120 | 28680 | 1.15x | 2.21e-10 | 7.76e-09 | 0.136 | 1.080 |
| 25fv47 | | 73560 | 48840 | 1.51x | 2.00e-06 | 1.20e-06 | 0.209 | 3.127 |
| woodw | | 64760 | 11600 | 5.58x | 7.60e-10 | 2.46e-07 | 0.111 | 8.735 |
| degen3 | | 19600 | 20000 | 0.98x | 2.42e-07 | 5.86e-07 | 0.250 | 1.326 |
| stocfor2 | | 93520 | 29360 | 3.19x | 2.62e-08 | 3.28e-08 | 0.043 | 3.859 |
| pilot87 | | 405760 | 168360 | 2.41x | 4.75e-06 | 1.72e-05 | 4.047 | 77.961 |
| maros-r7 | | 13760 | 3720 | 3.70x | 2.18e-11 | 2.04e-07 | 0.676 | 4.571 |

Geometric mean iteration ratio against HiGHS-PDLP: **1.93x** over 17 instances.
