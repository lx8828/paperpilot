"""语料指代解析：从问题里解析用户**明确指明**的论文（2026-09-22）。

## 背景（多篇语料）

L0 判「够不够」、检索取料，都以"哪一篇"为前提。语料有 5 篇时：
  · 问题**明确指了篇** → 应把范围收到那一篇（L0 用它的总览、检索给它保底）；
  · 问题**没指**（跨篇问题）→ 保持全局。

## 只处理「明确指代」（与产品共识一致）

    ✅ 「篇 1」「第 2 篇」「paper 3」「篇A」        —— 编号/字母
    ✅ 「在 CrossSum 中」「MASkills 那篇」         —— 篇名子串命中
    ❌ 「**这篇**论文…」                          —— 5 篇语料下**不是目标用法**（对着 5 篇
        提问必然指明篇名或编号）→ 本模块**不解析**它，并且**不再注入任何"指代不明"提示**
        （2026-09-23 删：那句提示曾让 judge 在 5 篇下必然判不够、把 L0 直答废掉）。

## 注意：篇名类其实已经能工作

篇名是**专名** → `search_hybrid` 的 BM25 路能精确召回（实测「MASkills」那题主篇 12/12）。
**真正必须解析的是编号** —— 检索器完全无法理解「篇 2」是什么意思。
本模块两者都做，但编号才是不可替代的那部分。
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

# components/focus.py → paperpilot → src → 项目根（**3 层**；nodes/ 比这里深一层用 4）
ROOT = Path(__file__).resolve().parents[3]
VIEW_DIR = ROOT / "assets/artifacts/out_views"

# 「第 2 篇」「篇2」「paper 3」「篇B」「第 3 篇」
_ORD_RES = (
    re.compile(r"第\s*(\d+)\s*篇"),
    re.compile(r"篇\s*(\d+)"),
    re.compile(r"paper\s*(\d+)", re.I),
    re.compile(r"第\s*([A-Ea-e])\s*篇"),
    re.compile(r"篇\s*([A-Ea-e])"),
)
# 冒号前的主名（"MASkills: Continual..." → "MASkills"）—— 最像产品名、最不易误命中。
# ⚠️ **不要把 `-`/`–` 当分隔符**：`Skill-MAS: ...` 的 `-` 是复合词的一部分，
#    切了会得到 "Skill"（通用词）→ 单元自检实测误命中（2026-09-22）。
_SPLIT = re.compile(r"[:：]")
_NAME_MAX = 40          # 主名过长不适合被自然引用 → 不用它做指代


@lru_cache(maxsize=256)
def _raw_names(pdf: str) -> tuple[str, ...]:
    """单篇的候选短名（**未去歧义**）。"""
    names: list[str] = []
    rp = VIEW_DIR / f"{Path(pdf).stem}.report.json"
    if rp.exists():
        try:
            title = str(json.loads(rp.read_text(encoding="utf-8")).get("title") or "")
        except Exception:  # noqa: BLE001
            title = ""
        head = _SPLIT.split(title.strip(), 1)[0].strip()
        if 4 <= len(head) <= _NAME_MAX:
            names.append(head)
    stem = Path(pdf).stem
    if len(stem) >= 4:
        names.append(stem)
    return tuple(names)


@lru_cache(maxsize=64)
def _names(corpus: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """语料级的**去歧义**候选名。

    规则：若某候选名是**另一篇**候选名的子串 → 它不足以唯一指代 → **丢弃**。
    实测（2026-09-22）：`2606.18837` 的标题主名是 `Skill`，而语料里还有
    `MASkills` / `SkillLearnBench` → 不丢的话「MASkills」会被 `skill` 子串
    **误命中到 2606.18837**（单元自检抓到）。规则是数据驱动的，不需要词表。
    """
    raw = {p: list(_raw_names(p)) for p in corpus}
    out: dict[str, tuple[str, ...]] = {}
    for p, ns in raw.items():
        others = [n.lower() for q, ms in raw.items() if q != p for n in ms]
        out[p] = tuple(n for n in ns if not any(n.lower() in o for o in others))
    return out


def resolve_focus(question: str, corpus: list[str]) -> list[str]:
    """问题里**明确指明**的篇 → pdf 列表（`[]` = 未指明，保持全局/跨篇）。

    Args:
        question: 用户问题。
        corpus: 本次语料（有序，**顺序即「篇 1/篇 2…」的编号依据**）。

    Returns:
        被指明的 pdf（按 corpus 顺序、去重）；未指明返回 `[]`。
    """
    if not question:
        return []
    # 语料必然是多篇（单篇路径 2026-09-23 已删）；本模块只做"定位到哪一篇"。
    hits: list[str] = []

    for rx in _ORD_RES:
        for m in rx.finditer(question):
            tok = m.group(1)
            idx = (int(tok) - 1 if tok.isdigit()
                   else ord(tok.upper()) - ord("A"))     # 篇A → 第 1 篇
            if 0 <= idx < len(corpus) and corpus[idx] not in hits:
                hits.append(corpus[idx])

    low = question.lower()
    names = _names(tuple(corpus))
    for p in corpus:
        if p in hits:
            continue
        if any(n.lower() in low for n in names.get(p, ())):
            hits.append(p)

    return [p for p in corpus if p in hits]        # 保持 corpus 顺序
