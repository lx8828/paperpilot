"""**NLI judge 冒烟测试**：中文论断 × 英文证据，能否判定蕴含/矛盾/中立？

## 为什么需要 NLI（而不是继续用词面锚点）
1. **解耦**：现在的"判定器"（词面锚点）与**真值同源** → `cite_recall=0.33` 是被**高估**的，
   而且**结构上不可能**发现"论断说反了"（论文写"未来可做 X"被判成"当前没做 X"）。
2. **对标**：ALCE 的 citation recall/precision 定义就是 NLI（`TRUE`），不换就只能自称"代理上界"。
3. **新能力**：NLI 三分类 → `contradiction` 恰好抓"说反了"；`neutral` = 不支持（比"没出现那个词"准）。

## 候选（6.4GB 显存只能 base/small 级；T5-11B 的 TRUE 放不下）
| 模型 | 规模 | 特点 |
|---|---|---|
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | 279M | **100 语言**（MNLI+XNLI）→ 中文论断 × 英文证据 |
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | 279M | 20 万级多语言 NLI 对（26 语言）→ 多语言更强 |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | 184M | **FEVER 训练**（事实核查）→ 英文"支持/反驳"最对口 |

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_nli_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

CANDIDATES = [
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",                    # 跨语言（100 语言）
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",  # 跨语言（更多多语言数据）
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",               # 英文·FEVER 事实核查
    "cross-encoder/nli-deberta-v3-base",                          # 英文·通用
]

# (期望标签, 论断, 前提)  —— 前提取自真实论文句式（省略号处保留原文风格）
CASES: list[tuple[str, str, str]] = [
    ("entail", "该论文公开了代码。",
     "Our code and data are publicly available at https://github.com/example/paper."),
    ("contradiction", "该论文没有公开代码。",
     "Our code and data are publicly available at https://github.com/example/paper."),
    ("neutral", "该论文公开了代码。",
     "We conduct an ablation study by removing each component of our model."),
    ("entail", "该论文做了消融实验。",
     "To understand the contribution of each component, we ablate them one by one. "
     "Table 3 shows the results of this analysis."),
    ("contradiction", "该论文只在一个数据集上做了实验。",
     "We evaluate our method on eight datasets spanning question answering, summarization, and dialogue."),
    ("neutral", "该论文报告了推理延迟。",
     "Human annotators rated the fluency and coherence of the generated summaries on a 5-point scale."),
    ("entail", "该论文做了人工评估。",
     "Three graduate students annotated 200 examples, reaching a Cohen's kappa of 0.71."),
    ("contradiction", "该论文没有讨论局限性。",
     "We discuss the limitations of our approach in Section 7, including evaluation cost."),
    # 英文改写论断（**不含锚点原词**）→ 检验"语义支持"而非"词面命中"
    ("entail", "The paper conducted experiments in languages other than English.",
     "We evaluate on XNLI, which covers 15 languages including Swahili and Urdu."),
    ("entail", "The authors compared their approach against a strong baseline.",
     "Table 2 compares our method to BM25 and a dense retriever; both are outperformed."),
]


def main() -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name in CANDIDATES:
        print("=" * 100)
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float16)
            model.eval().to(dev)
        except Exception as e:  # noqa: BLE001
            print(f"【{name}】加载失败：{type(e).__name__} {str(e)[:140]}")
            continue
        lab = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
        print(f"【{name}】labels={lab} | 设备 {dev} | "
              f"显存 {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        ok = 0
        for expect, claim, prem in CASES:
            enc = tok(prem, claim, return_tensors="pt", truncation=True, max_length=512).to(dev)
            with torch.no_grad():
                p = torch.softmax(model(**enc).logits.float(), dim=-1)[0].cpu().tolist()
            top = lab[int(max(range(len(p)), key=lambda i: p[i]))]
            hit = top.replace("label_", "").startswith(expect[:5]) or (
                expect == "entail" and "entail" in top) or (expect == "contradiction" and "contra" in top)
            ok += bool(hit)
            prob = " ".join(f"{l[:6]}={v:.2f}" for l, v in zip(lab, p))
            print(f"  {'✓' if hit else '✗'} 期望 {expect:<13} 预测 {top:<20} {prob}")
            print(f"      论断：{claim}")
            print(f"      前提：{prem[:88]}…")
        print(f"  → 命中 {ok}/{len(CASES)}")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
