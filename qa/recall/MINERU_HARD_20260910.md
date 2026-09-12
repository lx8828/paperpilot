# MinerU 端到端 hard 自测（10 篇 × 3 题）

> 链路：graph.ask（v3 两级 + nol3j + 闸门）| 解析源：MinerU | 裁判：glm-4-flash ≥4=pass
> **总通过 29/30 = 97%** | 均分 4.77
| 论文 | 题 | 通过 | 均分 |
|---|---|---|---|
| 2608.27843v1 | 3 | 2 | 4.3 |
| 2608.28447v1 | 3 | 3 | 5.0 |
| 2608.29179v1 | 3 | 3 | 4.7 |
| 2608.29290v1 | 3 | 3 | 5.0 |
| 2608.30333v1 | 3 | 3 | 4.3 |
| 2608.30938v1 | 3 | 3 | 5.0 |
| 2608.31046v1 | 3 | 3 | 5.0 |
| 2608.31108v1 | 3 | 3 | 4.7 |
| 2609.01316v1 | 3 | 3 | 4.7 |
| 2609.01456v1 | 3 | 3 | 5.0 |

## 逐题
| qid | 论文 | 分 | level | 闸门 |
|---|---|---|---|---|
| mh-27843-q2 | 2608.27843v1 | 3 | L3 | pass |
| mh-29179-q2 | 2608.29179v1 | 4 | L3 | pass |
| mh-30333-q1 | 2608.30333v1 | 4 | L3 | pass |
| mh-30333-q2 | 2608.30333v1 | 4 | L3 | pass |
| mh-31108-q2 | 2608.31108v1 | 4 | L0 | repaired |
| mh-01316-q2 | 2609.01316v1 | 4 | L0 | pass |
| mh-27843-q1 | 2608.27843v1 | 5 | L0 | pass |
| mh-27843-q3 | 2608.27843v1 | 5 | L3 | pass |
| mh-28447-q1 | 2608.28447v1 | 5 | L0 | pass |
| mh-28447-q2 | 2608.28447v1 | 5 | L3 | pass |
| mh-28447-q3 | 2608.28447v1 | 5 | L3 | pass |
| mh-29179-q1 | 2608.29179v1 | 5 | L3 | pass |
| mh-29179-q3 | 2608.29179v1 | 5 | L3 | pass |
| mh-29290-q1 | 2608.29290v1 | 5 | L3 | pass |
| mh-29290-q2 | 2608.29290v1 | 5 | L3 | pass |
| mh-29290-q3 | 2608.29290v1 | 5 | L3 | pass |
| mh-30333-q3 | 2608.30333v1 | 5 | L3 | pass |
| mh-30938-q1 | 2608.30938v1 | 5 | L3 | pass |
| mh-30938-q2 | 2608.30938v1 | 5 | L3 | pass |
| mh-30938-q3 | 2608.30938v1 | 5 | L3 | pass |
| mh-31046-q1 | 2608.31046v1 | 5 | L3 | pass |
| mh-31046-q2 | 2608.31046v1 | 5 | L3 | pass |
| mh-31046-q3 | 2608.31046v1 | 5 | L3 | pass |
| mh-31108-q1 | 2608.31108v1 | 5 | L0 | pass |
| mh-31108-q3 | 2608.31108v1 | 5 | L3 | pass |
| mh-01316-q1 | 2609.01316v1 | 5 | L0 | repaired |
| mh-01316-q3 | 2609.01316v1 | 5 | L3 | pass |
| mh-01456-q1 | 2609.01456v1 | 5 | L3 | pass |
| mh-01456-q2 | 2609.01456v1 | 5 | L3 | pass |
| mh-01456-q3 | 2609.01456v1 | 5 | L3 | pass |

## 失败题详情（score<4）

### mh-27843-q2 (2608.27843v1) score=3 [L3/pass]
Q: Under a countertypical partner, what is the reported effect of the intact agent minus the personal-history ablation on Reference-RUL and on adult lifetime, in both the Formal and Independent Witness conditions?
gold: Formal: Reference-RUL +1.235 [1.065, 1.411] and adult lifetime +3.145 [2.760, 3.528] rounds; Independent Witness: +1.224 [1.059, 1.382] and +3.353 [3.005, 3.705]
A: 你问的“反典型伙伴（countertypical partner）下，intact 减去 personal-history ablation 的效应”对应论文中该消融实验的 Reference-RUL 与 adult lifetime 两项结果。在 Formal 条件下，Reference-RUL 增加 1.235 [1.065, 1.411]，adult lifetime 增加 3.145 [2.760, 3.528] 轮 [1]；在 Independent Witness 条件下，对应效应分别为 1.224 [1.059, 1.382] 和 3.353 [3.005, 3.705] [1]。这一修正使正确动作概率提高 0.356 至 0.377，Reference-RUL 提高 1.224 至 1.235 [2]。
judge: 系统回答正确地引用了论文中的数值，但未提及 'rounds' 和未引用 'Table 3' 或 'Figure 9' 的具体内容，这可能导致读者难以直接定位到原文。

