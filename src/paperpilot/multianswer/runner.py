"""多答案开放域检索 · **跑一次** + **离线扫 N**。

## 两个函数的分工

    answer(question, idx, n=30)     跑一次：检索 → 逐篇判定 → 产出多答案
    sweep_n(res, ns=[10,20,30,50])  用**同一次跑批**离线扫多个 N（**不重跑判官**）

## ★ 为什么能"离线扫 N"（这是省时间的关键）

`set_judge` 的判果**只依赖该篇自己的片段**，与"候选有多少篇"无关；
所以一次 `top_papers=0`（判**全部**候选）的跑批，返回里带着：

    candidates = 候选篇的**检索序 + 篇内最高分**
    labels     = 每一篇的判果（yes / no / …）

于是"把 N 从 30 改成 50"只是**切个前缀**，不需要再调一次判官。

⚠️ **前提**：那次跑批必须 `n=None`（= 判全部候选）。
   若已经用某个 N 截断过，被截掉的篇**没有判果**，`sweep_n` 会**报错**而不是
   悄悄给出偏小的结果 —— 静默给错数比报错糟得多。
"""
from __future__ import annotations

from typing import Any, Iterable

from paperpilot.components import set_judge as SJ

# ★ `render` 直接用生产的（不重写判据/措辞）
render = SJ.render


def answer(question: str, idx: Any, *, n: int | None = None,
           b: int | None = None, workers: int | None = None,
           extract: bool | None = None, dual: bool | None = None,
           claim: str | None = None) -> dict[str, Any]:
    """**一次多答案开放域检索**：问题 → （检索 → 逐篇判定）→ 多答案。

    - `n`：候选篇上限（判几篇）。`None`/`0` = **不限**（判全部候选）。
      ★ 传 `None` 才能事后 `sweep_n` 扫任意 K。
    - `b`：每篇给判官的候选块数（生产默认 12）。
    - `workers`：判官并发（生产默认 6）。
    - `extract`：是否额外抽取逐篇内容（复合题需要）。

    返回 = 生产 `set_judge.run()` 的返回，**原样透传**，另加 `question` / `n_requested` /
    `warning`（如有）。

    ## ★ 判官未配置 → **直接报错**（不让它静默）
    实测（2026-10-05）：**直接跑脚本不会加载 `.env`**（`llm._load_dotenv` 只在
    `workflow.py` / `web/app.py` 里被调用）。缺 key 时判官**不报错**，只是把
    **每一篇**都判成 `unclear` → 输出看起来像"**没有论文符合**" ✗

        判定耗时 13.2s ｜ 候选 50 篇 ｜ 判 yes 0 篇 ｜ 另有 50 篇证据不足

    ★ 这种"看起来是业务结论、其实是配置坏了"的失效，比抛异常危险得多 → 这里**抛异常**。
    """
    from paperpilot.tools import llm

    if not llm.is_configured():
        raise RuntimeError(
            "判官主链路未配置（`PAPERPILOT_LLM_*` 读不到）—— 多答案检索**跑不了**。\n"
            "⚠️ 直接跑脚本**不会**自动加载 `.env`；请先：\n"
            "    from paperpilot.tools import llm; llm._load_dotenv(<仓库根>)\n"
            "★ 不拦的话，判官会把**每一篇**都判成 unclear，输出看起来像"
            "「没有论文符合」—— 那是假的。")

    res = SJ.run(question, idx, claim=claim or question, b=b, workers=workers,
                 extract=extract, dual=dual,
                 top_papers=(int(n) if n else 0))
    res["question"] = question
    res["n_requested"] = int(n or 0)

    # ★ 第二道防线：配置在读、但**全部候选都 unclear** —— 多半仍是链路问题
    #   （模型名错 / 网络不通 / 返回格式不合），而不是"真的没有论文做过这件事"。
    unclear = res.get("unclear") or []
    if res.get("n_papers") and res["n_yes"] == 0 and len(unclear) >= res["n_papers"]:
        res["warning"] = (
            f"⚠️ **全部 {res['n_papers']} 篇候选都判 unclear** —— 这通常意味着"
            f"判官链路有问题（模型名 / 网络 / 输出格式），而**不是**"
            f"「没有论文做过这件事」。请先核对 `PAPERPILOT_LLM_MODEL` 与调用是否报错。")
    return res


def sweep_n(res: dict[str, Any], ns: Iterable[int]) -> list[dict[str, Any]]:
    """在**同一次跑批**上扫多个 N，返回每个 N 的交付规模（不重跑判官）。

    ⚠️ 要求那次跑批是 `n=None`（判了**全部**候选）；否则报错 —— 见模块 docstring。
    """
    if int(res.get("top_papers") or 0) != 0:
        raise ValueError(
            "sweep_n 只能在「判全部候选」的跑批上做（那次 n 必须为 None/0），"
            f"但这次 top_papers={res.get('top_papers')} —— 被截掉的篇没有判果，"
            "扫出来的数会**偏小**。请用 n=None 重跑一次。")
    cand: list[str] = [str(c["pdf"]) for c in (res.get("candidates") or [])]
    labels: dict[str, str] = {str(k): str(v) for k, v in (res.get("labels") or {}).items()}
    out = []
    for n in sorted({int(x) for x in ns}):
        kept = cand[:n]
        yes = [p for p in kept if labels.get(p) == "yes"]
        out.append({"n": n, "n_candidates": len(kept), "n_yes": len(yes),
                    "papers": yes, "unjudged": max(0, n - len(kept))})
    return out


def deliver(res: dict[str, Any], n: int) -> list[dict[str, Any]]:
    """按 N 取**交付集合**（判 yes 的前 n 篇候选），每条带证据/理由。"""
    cand = {str(c["pdf"]): c for c in (res.get("candidates") or [])}
    order = [str(c["pdf"]) for c in (res.get("candidates") or [])][:n]
    by_pdf = {str(p["pdf"]): p for p in (res.get("papers") or [])}
    return [dict(by_pdf[p], rank=i + 1, score=cand[p].get("score"))
            for i, p in enumerate(order) if p in by_pdf]
