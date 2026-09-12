# cap 粒度修正扫描：完整 title_path vs 顶层节（样本 v1 250 题）

| config | R@8 | R@12 | R@16 | MRR@16 |
|---|---|---|---|---|
| baseline | 0.897 | 0.995 | 1.000 | 0.484 |
| toppath1 | 0.945 | 1.000 | 1.000 | 0.544 |
| fullpath1 | 0.923 | 1.000 | 1.000 | 0.506 |
| fullpath2 | 0.897 | 0.995 | 1.000 | 0.486 |

hard baseline: R@8=0.810 R@12=0.983 MRR=0.367
hard toppath1: R@8=0.867 R@12=1.000 MRR=0.421
hard fullpath1: R@8=0.855 R@12=1.000 MRR=0.380
hard fullpath2: R@8=0.810 R@12=0.983 MRR=0.367
