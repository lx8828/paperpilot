# S_merge A/B：只并 <900 超小块（base 其余不动）

> 论文 208 | 记录题 250（缺 0）
| 策略 | R@8 | R@12 | R@16 | MRR@16 | NDCG@12 | gold@1 | miss@12 |
|---|---|---|---|---|---|---|---|
| base | 0.864 | 0.960 | 0.972 | 0.458 | 0.576 | 0.264 | 0.000 |
| merge | 0.916 | 0.976 | 0.996 | 0.464 | 0.585 | 0.256 | 0.000 |

> merge 后块数 2377（base 3247）| 字符 中位 1928 | p90 3663 | <900 剩 75

> 逐题：变好 85 / 变差 49 / 持平 116

### merge 变差 Top15
  e28a6e3d8f [normal] base=#2 merge=#12
  4b8a0e99bf [hard] base=#2 merge=#8
  c2cbc26377 [normal] base=#8 merge=#13
  2ed02be0c1 [normal] base=#4 merge=#8
  046ff04d10 [normal] base=#3 merge=#6
  3b995a7358 [hard] base=#3 merge=#5
  14634943d9 [hard] base=#1 merge=#3
  8df89988ad [hard] base=#2 merge=#4
  3241f90a03 [hard] base=#9 merge=#11
  7aba5e4483 [hard] base=#3 merge=#5
  4d4b9ff2da [normal] base=#1 merge=#3
  a7cb4f8e29 [normal] base=#3 merge=#5
  74db8301d4 [normal] base=#7 merge=#9
  c8031c1629 [normal] base=#5 merge=#7
  1afd550cbe [normal] base=#1 merge=#3

### merge 变好 Top15
  1a678d081f [hard] base=#17 merge=#3
  d83304c70f [normal] base=#21 merge=#8
  29571867fe [hard] base=#25 merge=#15
  cc5d8e12f6 [hard] base=#26 merge=#16
  96526a1482 [hard] base=#17 merge=#8
  1763a029da [hard] base=#22 merge=#14
  05bb75a1e1 [normal] base=#9 merge=#3
  2869d19e54 [normal] base=#10 merge=#4
  dfbab3cd99 [hard] base=#10 merge=#5
  a69af5937c [hard] base=#12 merge=#7
  e2e31ab279 [normal] base=#6 merge=#1
  579941de28 [hard] base=#14 merge=#10
  dfd9302615 [normal] base=#7 merge=#3
  d3bb06d730 [normal] base=#16 merge=#12
  790ed4458a [hard] base=#6 merge=#3
