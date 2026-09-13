"""内置**演示模式**：不需要 API Key / GPU / 模型下载，也能跑通整条产品链路。

## 为什么要有

招聘方 / 评审**不会花两小时装环境**。演示模式让"clone → 一条命令 → 拖一篇 PDF →
看到结构化报告 + 带引用的问答"在几分钟内成立，且**内容固定、可复现**（不依赖任何外部服务）。

## 三个开关

| env | 作用 |
|---|---|
| `PAPERPILOT_MOCK_LLM=1` | LLM 走固定响应（claim / 去重 / 打标 / 骨架 / 概述 / 导读 / 判够 / 作答 / 质检） |
| `PAPERPILOT_MOCK_EMBED=1` | 向量走确定性"词袋哈希"（**不加载 bge-m3**，省下 2 GB 下载） |
| `PAPERPILOT_MINERU=0` | 跳过 MinerU（默认演示模式即为 0；有 GPU 想试真解析就设回 1） |

`uv run python web/app.py --mock` 会把前两个打开、第三个设成 0（若你已显式设过则尊重你的设置）。

## 诚实边界（别被演示骗了）

演示模式**不会真的读懂论文**：claim、概述、导读、答案都是**固定文案**（带"（演示内容）"前缀）。
唯一真实的是——**引用锚点**：答案里的 `[n]` 仍按真实检索到的块解析成 `cites`（页码/原文片段），
以及整条工程链路（摄取 → 索引 → 检索 → 闸门 → 溯源）。

固定文案里的证据句是**从输入正文里逐字摘的**（`_quote_from`），所以"证据回核"是真跑通的，
不是硬编码假命中。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

# 演示文案统一前缀，避免被误当成真实产出
_TAG = "（演示内容）"


# ───────────────────────── 开关 ─────────────────────────


def llm_enabled() -> bool:
    """LLM 是否走假实现（无需 key）。"""
    return os.environ.get("PAPERPILOT_MOCK_LLM") == "1"


def embed_enabled() -> bool:
    """向量是否走假实现（不加载 bge-m3）。"""
    return os.environ.get("PAPERPILOT_MOCK_EMBED") == "1"


def mineru_enabled() -> bool:
    """MinerU 是否启用（默认启用；`PAPERPILOT_MINERU=0` 关闭）。"""
    return os.environ.get("PAPERPILOT_MINERU", "1") != "0"


# ───────────────────────── 假 LLM ─────────────────────────

DEMO_METHOD = f"{_TAG}本文提出一种双塔检索结构，并用对比学习在多数据集上验证了有效性。"
DEMO_OVERVIEW = f"{_TAG}本文提出一种双塔检索结构，在公开数据集上取得 87.3% 的成绩。"
DEMO_GUIDE = f"{_TAG}一句话：两个编码器分别处理图文，再做对齐与重排。"
DEMO_ANSWER = f"{_TAG}本文提出一种双塔检索结构，并用对比学习在多个数据集上验证了有效性 [1]。"
DEMO_CORE = f"{_TAG}核心方法：双塔检索结构 + 对比学习对齐。"


def _quote_from(user: str, max_len: int = 120) -> str:
    """从输入正文里挑一句**逐字**原文当证据（→ 证据回核能判 hit，不是硬编码假命中）。"""
    txt = str(user or "").replace("[TABLE_CELL]", " ")
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9 ,\-–()/%]{25,220}[.!?]", txt):
        return m.group(0).strip()[:max_len]
    for seg in re.split(r"[。！？\n]", txt):
        s = seg.strip()
        if 12 <= len(s) <= max_len:
            return s
    return f"{_TAG}原文片段"


def response(system: str, user: str = "") -> str:
    """按 system prompt 返回固定响应（**永不联网**）。

    认不出的 prompt 一律回 `"[]"`：多数阶段对空数组都能安全降级
    （去重 → 回退单组、骨架 → 空、图表指南 → 空），不会把链路打断。
    """
    # 延迟导入：`agents.nodes.*` 会反向 import `tools.llm`，模块级导入会成环
    from paperpilot.agents.nodes import answer as A
    from paperpilot.agents.nodes import judge as J
    from paperpilot.components import validator as V
    from paperpilot.prompts import analyzer as Pa
    from paperpilot.prompts import figures as Pf
    from paperpilot.prompts import report as Pr
    from paperpilot.prompts import skeleton as Ps
    from paperpilot.prompts import viewer as Pv

    table: dict[str, Any] = {
        Pa.SYSTEM_PROMPT: [{"type": "method", "text": DEMO_METHOD,
                            "evidence_quote": _quote_from(user)}],
        Pv.DEDUPE_SYSTEM: [{"best_id": "c0001", "claim_ids": ["c0001"]}],
        Pv.LABEL_SYSTEM: [{"group_id": "g1", "label": "core_claim", "why": "演示：视为核心命题"}],
        Ps.SKELETON_SYSTEM: [],
        Pf.GUIDE_SYSTEM: [],
        Pr.OVERVIEW_SYSTEM: {"overview": DEMO_OVERVIEW},
        Pr.GUIDE_SYSTEM: {"guide": DEMO_GUIDE},
        J._SYS_L0: {"enough": True, "target_sections": [], "gap": ""},
        J._SYS_L3: {"enough": True, "target_sections": [], "gap": ""},
        V._SYS_UNCITED: {"uncited": []},
        V._SYS_VALIDATE: {"unsupported": [], "off_topic": False},
        A.SYSTEM: DEMO_ANSWER,
        A.SYSTEM_EXTRACT: DEMO_ANSWER,
        A.SYSTEM_EXTRACT_TABLE: f"{_TAG}（表格抽取略）",
        A.SYSTEM_UNKNOWN: f"{_TAG}论文中没有提到这一点。",
    }
    for key, val in table.items():
        if key == system:
            return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
    return "[]"


# ───────────────────────── 假向量 ─────────────────────────


def hash_vector(text: str, dim: int) -> "Any":
    """确定性"词袋哈希"向量（MD5 分桶 + L2 归一化）。

    为什么不用随机数：演示里"检索到正确块"要看得出来——词重叠越多、余弦越高。
    用 MD5（而不是 `hash()`）保证**跨进程/跨平台稳定**，向量缓存才有意义。
    """
    import numpy as np

    v = np.zeros(dim, dtype="float32")
    for tok in str(text).lower().replace("|", " ").split():
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
        v[h % dim] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n else v


def encode_texts(texts: list[str], dim: int) -> "Any":
    import numpy as np

    if not texts:
        return np.zeros((0, dim), dtype="float32")
    return np.stack([hash_vector(t, dim) for t in texts]).astype("float32")


def banner() -> str:
    """演示模式横幅文案（前端/CLI 共用）。"""
    on = [n for n, v in (("LLM", llm_enabled()), ("向量", embed_enabled()),
                         ("MinerU", not mineru_enabled())) if v]
    return f"演示模式已开启（{('、'.join(on)) or '无'} 使用内置假实现；内容为固定示例）"
