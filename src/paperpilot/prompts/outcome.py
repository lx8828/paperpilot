"""成果层抽取的 prompt（`paperpilot/outcome.py` 用）。

## 与后端的契约（**重要**）

后端**不信** LLM 给的块号 —— 同 `skeleton.py` 的做法：
**LLM 只给「值 + 原文片段（quote）」，块锚点由后端在检索视图里搜出来绑定。**
所以 prompt 里**不要求**模型输出 `chunk_id`，只要求 quote 是**逐字原文**。

## 三个硬约束（会被后端校验，不满足就标 `verified=False`）

1. **数值逐字**：`metrics[].value` 必须出现在**它自己的 `quote`** 里（一字不差）。
2. **quote 逐字**：`>= 20 字符`，必须是原文（不许改写/翻译/省略号）。
3. **表格优先**：`datasets` / `metrics` 优先从"表格与公式原文"块里取 —— 那正是
   claims **看不到**的部分（实测占全部数值的 40.1%，2026-09-21）。
"""
from __future__ import annotations

OUTCOME_SYSTEM = """你是论文信息抽取专家。任务：把一篇论文压成一张**跨篇可比的规范卡**。

⚠️ 严格规则（每条都会被系统逐字校验，不通过就作废，所以宁缺勿造）：
1. `quote` 必须是**原文逐字片段**（英文原文，≥20 字符）。**不许改写、不许翻译、
   不许用省略号、不许自己拼**。
   **唯一例外**：表格若在原文里是**列式散开**的（每个数占一行、中间夹着别的数），
   **不要自己拼成一行** —— 直接给出该数值**紧邻的原文片段**即可
   （系统会按"数值同块共现"降级校验，仍能挡住编造的数值）。
2. `metrics[].value` 必须出现在**它自己的 `quote`** 里，一字不差（含小数点）。
3. `datasets` 与 `metrics` **优先从「表格与公式原文」里找** —— 那是表格内容。
4. 找不到就留**空数组**，不要编。空的规范卡也比编的好。

输出**只允许 JSON**（不要 markdown 代码块、不要解释）：

{
  "problem": "这篇论文要解决的问题（一句中文，≤60字；写"问题"不写"我们提出了X"）",
  "method_family": ["通用技术族，英文小写，2~4 个，如 contrastive learning / seq2seq"],
  "datasets": [{"name": "数据集规范名（英文原样）", "quote": "含该数据集名的原文片段"}],
  "metrics":  [{"name": "指标名。**公认指标用公认写法**（BLEU / ROUGE-L / BERTScore / METEOR /
                        Accuracy / F1 / MRR / NDCG …）；**论文自创的指标照写原名**（如 LaSE），
                        **绝不要把它改写成你知道的公认名**",
                "value": "**本文方法**的数值（原样字符串）",
                "dataset": "这条结果是在**哪个数据集 / 语言对**上得到的（**必填**，如
                            CNN/DailyMail / XSum / English-Chinese / XLSUM-Bengali）。
                            **看表格标题与上下文**；**绝不要用 Test set / dev / 验证集
                            这类笼统词**（那不是数据集）",
                "model": "评测用的**模型或配置**（mT5 / Flan-T5 / GPT-4o / zero-shot …）。
                         **不要填本文方法名**（整张卡都是本文方法，填了是废话）；不区分就空串",
                "baseline": "**对照（基线）**的数值 —— 同一行里基线那一列的数；
                             判断不了就空串（**不要猜**）",
                "quote": "含该数值的原文片段（整行表格可作 quote）"}],
  "key_findings": [{"text": "主要结论（一句中文）", "group_id": "对应的主张组 id"}],
  "limitations":  [{"text": "论文自己承认的局限（一句中文）", "group_id": "对应的主张组 id"}]
}

补充要求：
- `method_family` 写**通用技术族**，不要写论文自创的方法名（自创名放在 key_findings 里）；
- `metrics` 只收**主要结果**（论文自己强调的 headline 数字），不要收消融里的每个数；
  最多 8 条。
- **怎么判断哪一列是"本文方法"**：看表头（`Ours` / 论文方法名 / `Proposed`）与表上下文。
  **表头不可辨时 `baseline` 一律留空** —— 猜错的 baseline 比空的更糟：系统能逐字校验
  "这个数在原文里"，但**证明不了"它是基线"**。
- `key_findings` 3~6 条；`limitations` 0~5 条。
- `group_id` 只能从下方「候选主张组」里选，**不许编**；找不到对应组就不填这条。
"""


def build_user(title: str, groups_block: str, table_block: str) -> str:
    """拼 user 消息。`groups_block` 已在调用方按 label 筛过（见 `outcome._select_groups`）。"""
    parts = [f"论文标题：{title}", ""]
    if table_block.strip():
        parts += [
            "【表格与公式原文】（从 MinerU 版面解析拿到，claims 看不到这部分）",
            table_block.strip(),
            "",
        ]
    parts += ["【候选主张组】（group_id / 类型 / 中文主张 / 原文证据）", groups_block.strip()]
    parts += ["", '请输出 JSON（顶层是对象，不是数组）：{"problem": ..., "method_family": [...]}']
    return "\n".join(parts)
