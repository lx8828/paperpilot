"""多答案开放域检索 · **把逐篇判定汇总成「回答」**（LLM）。

## ★ 为什么需要这个模块

`components/set_judge.py::render()` 只把判定结果**拼成清单**：

    在 50 篇语料里，判为「…」的有 7 篇：
    1. `C1D11`：报告了均值±标准差，误差棒显示标准差…（证据：片段 4，第 5 页 4.4）

读者拿到的是**原料**，还要自己归纳。本模块把 `papers`（每篇的 `why` + `extract` +
证据位置）交给 LLM，产出**直接回答问题的段落** —— 也就是「多答案展示页」的主输出；
逐篇清单**不丢**，降为可展开的「依据」（仍由 `render()` 产出）。

## 设计约定

* **绝不抛异常**：LLM 失败 → `ok=False`，调用方回退到 `render()` 的清单。
  ★ 展示页宁可显示"清单 + 一句失败提示"，也不能白屏。
* **只依据材料**：prompt 里明令禁止引入材料外的论文/数字（多答案场景最容易幻觉）。
* **命中 0 篇 → 不调 LLM**：结论是确定的（"没有找到"），省一次调用。
* 开关：环境变量 `PAPERPILOT_MA_SYNTH=0/false` 可关闭（回退旧行为）。
"""
from __future__ import annotations

import os
from typing import Any

# ★ 汇总用的 system prompt。改动前先读上面的「设计约定」——
#   「只依据材料」这条是防幻觉的底线，删掉它这份输出就不可信了。
SYS_SYNTH = """你要根据一份「逐篇判定」的结果，回答用户的问题。

硬规则：
1. **只依据给定材料**：不得引入材料里没有的论文、数字、结论；材料没说的就写"未报告"。
2. 材料分两组：YES = 该篇**做了**这件事（附具体内容），NO = 没做。
3. 结构：先用一句话给**结论**（共几篇、分别是哪几篇），再**逐篇**说明它具体说了什么
   （带上章节号 / 页 / 数字等可核对的细节）。
4. 用中文，篇幅 150~350 字；直接作答，**不要**复述题目、**不要**「根据提供的信息」这类套话。
5. 不要输出 Markdown 表格，用「1. 2. 3.」编号列表。"""


def enabled() -> bool:
    """汇总是否启用（默认启用；`PAPERPILOT_MA_SYNTH=0/false` 关闭）。"""
    return (os.environ.get("PAPERPILOT_MA_SYNTH") or "").strip().lower() not in (
        "0", "false", "no", "off")


def _fmt_yes(p: dict[str, Any], i: int) -> str:
    """把一篇「判 yes」的结果写成一行材料（缺 `extract` 就只用 `why`）。"""
    why = str(p.get("why") or "").strip()
    ex = str(p.get("extract") or "").strip()
    page, sec = p.get("page"), str(p.get("section") or "").strip()
    loc = f"第 {page} 页 " if page else ""
    loc += sec
    body = f"理由：{why}" if why else ""
    if ex:
        body += f" ｜ 具体内容：{ex}"
    line = f"{i}. `{p.get('pdf')}` ｜ {body}"
    if loc.strip():
        line += f" ｜ 位置：{loc.strip()}"
    return line


def build_user(question: str, res: dict[str, Any]) -> str:
    """构造汇总用的 user prompt（**可单独测**，不必真调 LLM）。"""
    papers = [p for p in (res.get("papers") or []) if isinstance(p, dict)]
    n_all = int(res.get("n_papers") or 0)
    unclear = [str(x) for x in (res.get("unclear") or [])]

    yes_lines = "\n".join(_fmt_yes(p, i) for i, p in enumerate(papers, 1)) or "（无）"
    hired = {str(p.get("pdf")) for p in papers}
    # ★ NO = 候选 − YES − unclear。**必须排除 unclear**：
    #   unclear 是"证据不足、没作判断"，若混进 NO，LLM 会把它当成
    #   "明确没做这件事" —— 那是**把"没判"说成"没做"**，是假结论。
    unset = {str(x) for x in (res.get("unclear") or [])}
    no_names = [str(c.get("pdf")) for c in (res.get("candidates") or [])
                if str(c.get("pdf")) not in hired and str(c.get("pdf")) not in unset]

    parts = [f"【问题】{question}", "",
             f"【逐篇判定结果】（语料共 {n_all} 篇）",
             f"YES —— {len(papers)} 篇：", yes_lines]
    if no_names:
        shown = "、".join(f"`{x}`" for x in no_names[:20])
        more = f" 等 {len(no_names)} 篇" if len(no_names) > 20 else ""
        parts += ["", f"NO —— {len(no_names)} 篇：{shown}{more}"]
    if unclear:
        parts += ["", f"（另有 {len(unclear)} 篇证据不足、未作判断："
                      f"{'、'.join(f'`{x}`' for x in unclear[:10])}）"]
    return "\n".join(parts)


def synthesize(question: str, res: dict[str, Any], *,
               temperature: float = 0.2, max_tokens: int = 900) -> dict[str, Any]:
    """把逐篇判定汇总成一段**回答**。

    Returns:
        `{"answer": str, "ok": bool, "error": str, "n_yes": int}`
        ★ `ok=False` 时 `answer` 里仍是**可读的兜底文本**（调用方也可改用 `render()`）。
    """
    papers = [p for p in (res.get("papers") or []) if isinstance(p, dict)]
    n_yes = len(papers)
    n_all = int(res.get("n_papers") or 0)

    if not enabled():
        return {"answer": "", "ok": False, "error": "已关闭（PAPERPILOT_MA_SYNTH=0）",
                "n_yes": n_yes}

    # ★ 命中 0 篇：结论确定，不必花一次 LLM 调用
    if not papers:
        unclear = len(res.get("unclear") or [])
        tail = f"（另有 {unclear} 篇证据不足、未作判断）" if unclear else ""
        return {"answer": f"在 {n_all} 篇语料里，**没有找到**符合该问题的论文。{tail}",
                "ok": True, "error": "", "n_yes": 0}

    from paperpilot.tools import llm

    try:
        txt = llm.chat_text(SYS_SYNTH, build_user(question, res),
                            temperature=temperature, max_tokens=max_tokens)
        txt = (txt or "").strip()
        if not txt:
            raise llm.LLMError("汇总返回空文本")
        return {"answer": txt, "ok": True, "error": "", "n_yes": n_yes}
    except Exception as e:                                        # noqa: BLE001
        # ★ 不抛：展示页要能降级成「清单 + 提示」，而不是白屏
        return {"answer": "", "ok": False,
                "error": f"{type(e).__name__}: {str(e)[:200]}", "n_yes": n_yes}
