# 闸门数字误杀修复 · 回收验证（2026-09-11 02:16）

> 对象：`qa/qasper_run_20260911_005347.json` 里被闸门兜底的 10 道｜同裁判（glm-4-flash）同口径

| 论文 | 问题 | 修复前 | **修复后** | 变化 |
|---|---|---|---|---|
| 1912.01673 | How many sentence transformations on | 1 | **3** (fail) | +2 |
| 1909.00338 | Which features do they use to model  | 1 | **5** (pass) | +4 |
| 1901.11117 | What is in the model search space? | 1 | **5** (pass) | +4 |
| 2001.07263 | How much bigger is Switchboard-2000  | 1 | **5** (pass) | +4 |
| 2001.07209 | Which datasets are used in the paper | 1 | **4** (pass) | +3 |
| 2001.07209 | Which dataset sources to they use to | 1 | **3** (fail) | +2 |
| 2004.03925 | Do the authors give examples of posi | 1 | **5** (pass) | +4 |
| 1906.03338 | How do they demonstrate the robustne | 1 | **5** (pass) | +4 |
| 2003.04967 | What experimental evaluation is used | 1 | **4** (pass) | +3 |
| 1809.01060 | What were the results of the first e | 1 | **3** (fail) | +2 |

---

- **pass（≥4）**：0/10 → **7/10**（+7）
- 合计分：10 → **42**（均 1.00 → 4.20）
- 翻正 **10** 道｜变差 **0** 道

> 折算到 QASPER 233 主通过率：**+7/233 = 3.0pt**（修复前 193/233 = 82.8% → 预计 **85.8%**，另需扣除题面缺失/波动项，属**上限估计**）
