# iris-ood — multi-seed self-editing ablation

| arm | seed | cycles | final best (kept) | keep rate | mean running best |
|---|---|---|---|---|---|
| self-editing ON | 7 | 4188 | 1.0000 | 15.57% | 0.9694 |
| self-editing ON | 42 | 3042 | 1.0000 | 9.76% | 0.8700 |
| self-editing ON | 123 | 3525 | 0.9778 | 6.10% | 0.8783 |
| self-editing OFF | 7 | 1263 | 0.9333 | 6.73% | 0.8774 |
| self-editing OFF | 42 | 5784 | 0.9778 | 16.25% | 0.7596 |
| self-editing OFF | 123 | 1430 | 1.0000 | 17.97% | 0.8095 |

![running best](figures/compare_seeds.pdf)

![final best per seed](figures/final_best.pdf)
