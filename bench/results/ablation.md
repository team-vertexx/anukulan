### What each piece of the solver is worth

Iterations to a relative tolerance of 1e-06. `limit` means the run hit 2,000,000 iterations or 60s without converging.

| instance | everything | no restarts | no adaptive step | no primal weight | no scaling | plain PDHG |
|---|---|---|---|---|---|---|
| afiro | 360 | 480 | 360 | 4,600 | 560 | 19,800 |
| sc50a | 1,000 | 6,800 | 1,120 | 38,600 | 1,880 | 73,720 |
| blend | 2,320 | 33,360 | 2,600 | 5,280 | 195,440 | limit |
| stocfor1 | 13,120 | 288,480 | 27,440 | 12,960 | limit | limit |
| sctap1 | 2,520 | 95,880 | 2,880 | 16,680 | 98,520 | limit |
| degen2 | 3,920 | 16,800 | 4,880 | 6,000 | 4,840 | 122,040 |
| scfxm1 | 15,440 | 885,640 | 22,600 | 444,360 | limit | limit |
| bandm | 32,160 | 622,880 | 72,600 | 734,400 | limit | limit |
| share1b | 60,560 | limit | 57,840 | limit | limit | limit |
| 25fv47 | 73,560 | 448,800 | 155,280 | limit | limit | limit |

Cost of removing each piece, geometric mean over the instances where both that configuration and the full solver converged:

- **no restarts**: 11.37x more iterations (over 9 comparable instances); solved 9/10
- **no adaptive step**: 1.38x more iterations (over 10 comparable instances); solved 10/10
- **no primal weight**: 7.22x more iterations (over 8 comparable instances); solved 8/10
- **no scaling**: 6.53x more iterations (over 5 comparable instances); solved 5/10
- **plain PDHG**: 50.16x more iterations (over 3 comparable instances); solved 3/10
