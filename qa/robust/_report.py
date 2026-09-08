"""生成稳定性报告 md（读最后一个 robust_run json，算一致性并分类）。"""
from __future__ import annotations
import glob
import json
import statistics

fs = sorted(glob.glob("qa/robust/robust_run_*.json"))
d = json.load(open(fs[-1], encoding="utf-8"))
n = len(d)
stable = sum(1 for r in d if r["stable_pass"])
allpass = sum(1 for r in d if r["pass_all"])
spreads = [r["spread"] for r in d]
sigmas = [r["sigma"] for r in d]
big = [r for r in d if r["spread"] >= 2]   # 真正的翻盘
border = [r for r in d if not r["stable_pass"] and r["spread"] == 1]

L = []
L.append("# 稳定性 / 同义改写测试报告（2026-09-07）")
L.append("")
L.append("> 目的：量 PaperPilot(B2) 对**问法变化**的稳定性——同义改写不改语义，成绩不应变。")
L.append("> 方法：30 题（20 篇 QASPER 论文内取样，含 4 道无答案题）× 每题 3 种问法 = 90 次；"
         "全部 B2 链路（deepseek-chat）+ glm-4-flash 按 QASPER gold 盲判，pass≥4。")
L.append("> 产物：`qa/robust/questions.json`、`qa/robust/robust_run_*.json`、runner `_run_robust.py`。")
L.append("")
L.append("## 总览")
L.append("")
L.append("| 指标 | 数值 |")
L.append("|---|---|")
L.append(f"| 3 变体 pass 全一致 | {stable}/{n} = **{stable/n:.1%}** |")
L.append(f"| 3 变体全 pass | {allpass}/{n} = {allpass/n:.1%} |")
L.append(f"| 分数跨度均值 / 最大 | {sum(spreads)/n:.2f} / {max(spreads)} |")
L.append(f"| 分数 std 均值 | {sum(sigmas)/n:.2f} |")
L.append(f"| 跨度≥2（真翻盘） | {len(big)} 题 |")
L.append(f"| 跨度=1（边界±1） | {len(border)} 题 |")
L.append("")
L.append("## 不稳定题逐题（scores 为 v0/v1/v2）")
L.append("")
L.append("| qid | scores | 判读 |")
L.append("|---|---|---|")
labels = {
    "1771a5523682": "真翻盘：v0/v2 走 answer_unknown 断言\"未提供召回对比值\"（实有），v1 答对(4)→ 缺缺失闸门对 L3/unknown 终答的覆盖",
    "c034f38a570d": "v0(4) vs v1/v2(3)：v1/v2 断言\"未给出 perplexity 具体数值\"（gold 有 19.795 对比）→ 同上，答案差异主导",
    "8602160e98e4": "v0/v1 列出三数据集(4)；v2(3) 声称\"未见 corpora 名称\"——检索到了但收口时误断言",
    "ffa4d4bfb226": "v1(3) 浅层(L0)答成\"关系抽取/SemEval\"，v0/v2(L2)答对 paraphrase identification → 浅层收口答错任务",
    "4d4b9ff2da51": "v1(3) 浅层收口缺细节 vs v0/v2(4)，±1 但属\"该下钻没下钻\"",
    "a1885f807753": "v2(3)：问\"contains\"，量词表述差异触发误断言→ 弱 ±1",
    "a1c4f9e8661d": "v2(3) vs v0/v1(4)：baselines 枚举 v2 漏一项 → 枚举类问法敏感",
    "a1e07c7563ad": "v0(3) vs v1/v2(4)：v0 用\"size\"措辞下检索略差 → ±1",
    "f63519bb5e11": "unanswerable(gold 空)但论文文本确有内容：v1(4)/v0v2(3) 波动源于 QASPER 标注噪音+内容题判分口径",
}
for r in d:
    if not r["stable_pass"]:
        tag = labels.get(r["qid"], "")
        L.append(f"| {r['qid']} | {r['scores']} | {tag} |")
L.append("")
L.append("## 解读")
L.append("")
L.append("1. **70% 严格一致、平均跨度 0.53**：大部分题换问法成绩稳定；不稳定主要是 ±1 的阈值抖动（3↔4 正好在 pass 线上），"
         "对外报数建议自带 ± 说明。")
L.append("2. **不稳定里藏着 2 类真问题（不是纯噪声）**：")
L.append("   - **\"缺失/给不出\"陷阱随问法出现**（1771a5523682 v0/v2、c034f38a570d v1/v2、8602160e98e4 v2、a1885f807753 v2）："
         "论文明明有数据，换个说法漏斗就判\"材料不足\"→ answer 断言\"未提供\"。已知缺失断言闸门只覆盖 L0–L2 浅层；"
         "这几条是 **L3/unknown 终答层**仍会误拒——闸门没兜到。")
L.append("   - **浅层收口答错/答偏任务**（ffa4d4bfb226 v1 答成关系抽取、4d4b9ff2da51 v1）：部分问法在 L0/L1 就被 judge 判\"够\"，"
         "直接收口到 overview 层，没下钻到实验节 → 换问法改变了下钻决策。")
L.append("3. 结论：稳定性缺口和既有残差画像（answer 误断言/浅层收口）**是同一件事**；修法候选——把\"缺失断言复核\"从仅 L0–L2 扩展到 "
         "\"L3 judge 判不足将转 answer_unknown 前也做一次全文确认\"，以及\"数值/对比类题 judge 的够字须含可核验数值\"。")
L.append("")
L.append("## 局限")
L.append("- n=30、改写为人工同义改写，含主观性；判分受 glm-4-flash 边界噪声影响（±1 常见）。")
L.append("- 未与 B0/B1 对照——本项定位是 B2 自身诊断；同口径基线对比留给后续深水区。")

out = "qa/robust/ROBUST_REPORT_20260907.md"
open(out, "w", encoding="utf-8").write("\n".join(L))
print("written", out, "| stable", stable, "/", n)
