"""**全上下文回答（Full-Context QA）**：把语料（如 5 篇）一次性塞进**单次** LLM 调用。

## 为什么要有这条路径（2026-09-27 实测结论）

跨篇题（M1）在"检索式"架构下已经见顶：失败分解显示 **50.5% 的失败是"篇进来了但选错块"**、
**篇内 top1 命中只有 20%**（`retrieval/tmp/_archive/_design_probe.py`）；而"只有 5 篇"这个约束
（≈180k 字符 / ≈48k tokens）**让"不检索、直接全读"成为可行**。小样实测
（`retrieval/tmp/_archive/_fullctx_trial.py`，17 题）：

    · 全上下文 13/17（人工核对 17/17）｜ 检索臂 11/17
    · 均耗时 **1.9 秒 vs 13.1 秒**（单次调用 vs 4.8 次调用 + 检索）
    · 均输入 21k~48k tokens（按需塞 1 篇 / 全 5 篇）

## 设计要点

1. **问题放在 prompt 最后**：system + 上下文构成**稳定前缀** → 同一批语料的连续提问
   可命中 **prompt 前缀缓存**（命中价通常只有未命中的 1/10）。
   这是"全塞"能成立的关键 —— 别把问题插到上下文前面，否则缓存全失。
2. **逐段编号 `[P·§·¶]`**：答案要求带编号 → 可**溯源回 chunk**（`_extract_cites`），
   让 `score_answer` 的"引用通道"也能用，且人能核对。
3. **单次调用**：不走 `judge_l0` / `judge_l3` / `validator` 的多轮门控
   —— 那些门控每轮都**重复携带同一份上下文**（实测 4.8 次调用/题），
   是全塞方案成本的主要来源。
"""
from __future__ import annotations

import re
import sys
from typing import Any

from paperpilot.agents.embedder import ChunkIndex
from paperpilot.tools import llm

# ⚠️ 约束必须在 system 里、且**不要在中间插入变量**（否则前缀缓存失效）
SYS = (
    "你是论文问答助手。下面按【篇】给出同一批论文的全文（每段带 [P·§·¶] 编号）。"
    "请**只依据给定文本**回答，并遵守：\n"
    "1) 逐篇标注：每个要点前写该篇编号，如【P1】；涉及多篇时逐个列出。\n"
    "2) 数字、指标名、方法名、数据集名、专有名词必须**逐字照抄原文**"
    "（原文是英文就保留英文原词，不要只给中文意译）。\n"
    "3) 问『哪些篇/哪几篇』时，先列全部符合的篇，再逐篇说明。\n"
    "4) 文本中没有的，明确写『给定文本中未提及』，不要凭常识补。\n"
    "5) 引用时在句末标出段号，如 [P2·§4.2·¶xx]。\n"
    "6) 直接给答案，不要复述题目、不要解释你的流程。"
)


def _sec_of(paths: list[str]) -> str:
    for p in reversed(paths or []):
        if " · " in p:
            return p.split(" · ", 1)[1].strip()
    return (paths or [""])[-1]


def build_context(pdfs: list[str]) -> tuple[str, list[dict[str, Any]]]:
    """语料 → (带编号的上下文, 段落索引表)。索引表用于把答案里的编号还原成 chunk 引用。"""
    parts: list[str] = []
    index: list[dict[str, Any]] = []
    for n, pdf in enumerate(pdfs, 1):
        idx = ChunkIndex(str(pdf))
        parts.append(f"\n\n{'=' * 88}\n【P{n}】{pdf}\n{'=' * 88}")
        for c in idx._doc_chunks():                      # noqa: SLF001 与检索同源
            sec = _sec_of(list(c.title_path))
            key = f"P{n}·§{sec}·¶{c.chunk_id}"
            parts.append(f"\n[{key}]\n{str(c.text).strip()}")
            # ★ `p`（第几篇）必须留住：`chunk_id` **只在篇内唯一**（每篇都从 `c1` 开始），
            #   少了它就无法区分"A 的 c1"和"B 的 c1" → 引用会串篇（见 `_extract_cites`）。
            index.append({"p": n, "key": key, "pdf": str(pdf),
                          "chunk_id": str(c.chunk_id),
                          "section": sec, "page": int(c.page_span[0])})
    return "\n".join(parts), index


# 答案里的锚点形态：`[P1·§3.2.2·¶c8]` → `("1", "3.2.2", "c8")`。
# ⚠️ 三段都要，**尤其是 P（第几篇）**；`§` 后非贪婪到第一个 `·¶`，
#    这样章节名里带 `·`（如「3.1 · 引言」）也不会切错。
_ANCHOR_RE = re.compile(r"\[P(\d+)\s*·\s*§([^\]]*?)·\s*¶\s*([^\]]+?)\s*\]")

# 索引行里的篇号：`key` 形如 `P1·§3.3·¶c9`，**一直带着"第几篇"**。
_ROW_P_RE = re.compile(r"^P(\d+)·")


def _row_p(row: dict[str, Any]) -> int:
    """取一行索引的「第几篇」。

    优先显式 `p`（`build_context` 2026-10-03 起会写）；没有就从 `key` 里刨。
    ★ 为什么要有这个兜底：只认显式 `p` 的话，**任何没带 `p` 的 index 会静默返回
    零条引用**（比给错的更隐蔽 —— 用户以为"这篇没引用"）。
    `key` 从一开始就带篇号，所以老 index 也能正确消歧。
    """
    if row.get("p"):
        return int(row["p"])
    m = _ROW_P_RE.match(str(row.get("key") or ""))
    return int(m.group(1)) if m else 0


def _extract_cites(answer: str, index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从答案里的 `[P·§·¶…]` 锚点还原出引用的 chunk（带 evidence=该块原文）。

    为什么带 evidence：与检索路径同口径 —— `score_answer` 的"引用通道"判定文本是
    「答案 + 答案自己引用的证据原文」。不带它，全上下文方案会被系统性低估。

    ⚠️ **必须按 `(P, chunk_id)` 成对匹配**（2026-10-03 修，评审指出，实测坐实两处）：
      · 旧判据 `cid in answer` 是**宽子串**匹配 → `c1` 会被 `c12` 命中（凭空多出引用）；
      · 旧去重 `seen` 只按 cid → 而**每篇都从 `c1` 开始**，于是"A 的 c1"会把
        "B 的 c1"挤掉 → 答案引的是 `[P2…c1]`，返回的却是 **A 篇的 c1**（串篇）。
    现在：只认**方括号锚点**（不接受裸子串），且 `(p, chunk_id)` 必须**同时**相等。
    """
    hits = {(int(p), cid.strip()) for p, _sec, cid in _ANCHOR_RE.findall(answer or "")}
    if not hits:
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for row in index:
        key = (_row_p(row), str(row.get("chunk_id") or ""))
        if key not in hits or key in seen:
            continue
        seen.add(key)
        cid = key[1]
        idx = ChunkIndex(row["pdf"])
        text = ""
        for c in idx._doc_chunks():                      # noqa: SLF001
            if str(c.chunk_id) == cid:
                text = str(c.text)
                break
        out.append({"p": key[0], "pdf": row["pdf"], "chunk_id": cid,
                    "page": row["page"], "section": row["section"],
                    "evidence": text})
    return out


# ⚠️ **逐篇覆盖**变体（2026-09-27 小样实测）：
#    M1 小样 10 题里 3 道失败，模式都是**"答了一半"** —— 模型只答了 1~2 篇，
#    漏掉某篇的关键点（如 `G3-M1-1` 漏 `topic documents`、`G3-M1-4` 漏 `fact revision`），
#    而**不是**输出被 max_tokens 截断（实测 completion 292~1481，无一触顶）。
#    → 对策是**强制逐篇穷举**：要求 P1..Pn 每篇都有结论，不涉及的必须显式写"未涉及"。
SYS_COVER = SYS.replace(
    "3) 问『哪些篇/哪几篇』时，先列全部符合的篇，再逐篇说明。",
    "3) **必须逐篇穷举**：按【P1】【P2】…顺序，**每一篇都要给出结论**"
    "（该篇涉及就写要点；不涉及就明确写『该篇未涉及』）—— 不许跳过任何一篇。\n"
    "   问『哪些篇/哪几篇』时，先列出全部符合的篇，再按上面的逐篇格式说明。\n"
    "   注意：题干若问『各自的 X 是什么』，**每一篇都要有 X**。")


def _fmt_history(history: list[dict[str, Any]] | None) -> str:
    """多轮追问：把上几轮对话拼成一块**放在问题之前**（前缀仍是 `sys+ctx` → 缓存不受影响）。

    与 `agents/nodes/answer._fmt_history` 同口径（最近 6 轮、每条截 500 字符），
    目的是让直读路径也能理解"这个方法/它"等指代。
    """
    if not history:
        return ""
    lines = ["", "", "对话历史（本轮问题可能指代其中的内容，回答时请结合理解）："]
    for m in history[-6:]:
        who = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"  {who}: {(m.get('content') or '')[:500]}")
    return "\n".join(lines)


# ── 上下文预算窗口（2026-10-03）──────────────────────────────────────────────
#
# 为什么需要：`build_context` 与 `llm.chat_text` **都没有截断保护** → 一旦超窗口，
# 用户拿到的是一次**难懂的 API 400**。这里在"上下文已构造、尚未调用"这个点上拦一道，
# 换成**说得清、可操作**的话术（少喂几篇 / 换 reader）。
#
# **单位用「字符」而不是 token**：`len(ctx)` 是我们运行时能**精确且免费**测到的量；
# token 只能按经验比折算（英文 ≈4 字符/token，表格/公式还会更差），估不准 ——
# 拿一个估出来的数当硬闸，只会得到一个会漂移的闸。
#
# 取值依据（实测/规格，不是拍的）：
#   · 模型 `deepseek-chat`（服务端实为 `deepseek-flash`）：**上下文 1M token**、
#     最大输出 384K（官方规格；实测 6M 字符重复英文仍被完整读到，与该规格相容）；
#   · 本批语料实测 **46,138 字符/篇**（38,060~60,606，**篇间差 1.6 倍**）；
#   · 1M token 按**保守** 4 字符/token ≈ 4.0M 字符；
#   · 再打 **30% 折扣** → 1.2M 字符：吸收 token 估算误差、history 累积与输出预留，
#     并且**不贴边跑**（"能塞进去" ≠ "答得好"）。
#   ⇒ 1.2M 字符 ≈ **300k token** ≈ **26 篇**（按均值），远宽于当前的产品上限。
#
# ⚠️ 它只是**安全闸**，不是"该喂几篇"的答案：该喂几篇要由**质量**实测决定
#    （窗口 1M 不代表喂 1M 还答得好）。`PAPERPILOT_CTX_BUDGET_CHARS` 可覆盖，`0` = 不限制。
DEFAULT_CTX_BUDGET_CHARS = 1_200_000
ENV_CTX_BUDGET = "PAPERPILOT_CTX_BUDGET_CHARS"

# 模型的上下文窗口（**token**）。**只作展示与推导依据，不当硬闸用**
# （硬闸用字符，见上；理由同样：token 只能估）。来源：官方规格 —— `deepseek-chat`
# 服务端实为 `deepseek-flash`：**1M token**、最大输出 384K。
CTX_WINDOW_TOKENS = 1_000_000

# 本批语料实测均值（字符/篇）：38,060 ~ 60,606 → **46,138**（2026-10-02 实测，5 篇）。
# 只用于把"字符预算"换算成**人话**（"预算内约可放 N 篇"），不参与任何判定。
CTX_PER_PAPER_CHARS = 46_138

_ctx_budget_warned = False


def ctx_budget_chars() -> int:
    """上下文预算（**字符**）。`0` = 不限制。

    **fail-safe**：负数 / 非数字 / 空串 → 回退 `DEFAULT_CTX_BUDGET_CHARS` 并提示一次
    （沿用 `web/app.py::_max_pdf_bytes` 的惯例：异常配置只能**收紧**、不许静默放开）。
    """
    global _ctx_budget_warned
    import os
    raw = os.environ.get(ENV_CTX_BUDGET)
    if raw is None or not str(raw).strip():
        return DEFAULT_CTX_BUDGET_CHARS
    try:
        n = int(str(raw).strip())
    except ValueError:
        n = -1
    if n < 0:
        if not _ctx_budget_warned:
            _ctx_budget_warned = True
            print(f"[warn] {ENV_CTX_BUDGET}={raw!r} 非法（需为 ≥0 的整数，0=不限制）"
                  f" → 回退默认 {DEFAULT_CTX_BUDGET_CHARS:,} 字符", file=sys.stderr)
        return DEFAULT_CTX_BUDGET_CHARS
    return n


def estimate_ctx_chars(pdfs: list[str]) -> int:
    """**不拼大字符串**地估算 `build_context` 会产出多少字符。

    给"调用前先判预算"用：`build_context` 要把几篇全文拼成一个 20 万字符量级的字符串，
    为了判预算先拼一遍再丢掉不划算。这里只累加各块文本 + 每块的编号开销，**不分配**大字符串。

    精度（2026-10-03 实测 1/2/3/5 篇）：**估算/实际 = 0.994 ~ 1.001**（误差 <1%），
    所以它不只是"给个话术"，判预算基本就是准的。
    ⚠️ 但真正的**硬闸**仍是 `answer()` 里对 `len(ctx)` 的精确比较（两层，后者兜底）。
    读不出某篇（产物缺失等）→ 返回 0 = **不拦**（交给下游自己的错误处理）。
    """
    total = 0
    for pdf in pdfs or []:
        try:
            chunks = ChunkIndex(str(pdf))._doc_chunks()      # noqa: SLF001 与检索同源
        except Exception:                                    # noqa: BLE001
            return 0
        total += 120                                         # 篇头「【Pn】pdf」那一行
        for c in chunks:
            total += len(str(c.text).strip()) + 48           # 正文 + `[P·§·¶cid]` 编号
    return total


def answer(pdfs: list[str], question: str, *, temperature: float = 0.0,
           max_tokens: int | None = 1600, style: str | None = None,
           history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """单次调用回答。返回 {answer, cites, ctx_chars, n_papers, style}。

    `style`：`"base"`（默认）或 `"cover"`（**强制逐篇覆盖**，见 `SYS_COVER`）。
             不传时读环境变量 `PAPERPILOT_FULLCTX_STYLE`。
    `history`：上几轮对话（`[{role, content}]`）—— 插在**问题之前**，
             故 `sys+ctx` 前缀不变 → **前缀缓存仍命中**（这是"全塞"能成立的关键）。
    """
    import os
    st = (style or os.environ.get("PAPERPILOT_FULLCTX_STYLE") or "base").strip().lower()
    sysmsg = SYS_COVER if st == "cover" else SYS
    ctx, index = build_context(pdfs)
    n_ctx = len(ctx)

    # ★ 上下文预算闸（2026-10-03）：**在调用之前**拦，给可操作的话术，
    #   而不是让用户去读一次 API 400。`0` = 不限制。
    budget = ctx_budget_chars()
    if budget and n_ctx > budget:
        per = n_ctx // max(len(pdfs), 1)
        return {
            "answer": (f"（未作答：本批 {len(pdfs)} 篇的全文共 **{n_ctx:,} 字符**，"
                       f"超过上下文预算 **{budget:,} 字符**（平均 {per:,} 字符/篇）。\n"
                       f"请**减少篇数**（预算内约可放 "
                       f"{max(budget // max(per, 1), 1)} 篇），或调大 "
                       f"`{ENV_CTX_BUDGET}`。）"),
            "cites": [], "ctx_chars": n_ctx, "n_papers": len(pdfs),
            "style": st, "over_budget": True,
        }

    hist = _fmt_history(history)
    text = llm.chat_text(sysmsg, f"{ctx}{hist}\n\n{'=' * 88}\n【问题】{question}",
                         temperature=temperature, max_tokens=max_tokens)
    return {"answer": text, "cites": _extract_cites(text, index),
            "ctx_chars": n_ctx, "n_papers": len(pdfs), "style": st}
