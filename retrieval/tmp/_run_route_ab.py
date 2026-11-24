# -*- coding: utf-8 -*-
"""跑批：**分层（现状） vs 不分层（每题都读全文）** —— 验证「先看摘要答简单题」的设计（2026-10-13）。

对照设计（**同题、同裁判、配对**）：
  · **臂1 = 现状**（`paperpilot.graph.ask`）：L0 判"够" → 只吃**报告层**直答；不够 → fullctx 读全文。
  · **臂2 = 不分层**：同一个生产图，但把 `paperpilot.agents.nodes.judge.judge_l0` 换成"恒判不够"
    （★ **运行时打补丁，不改仓库代码**）→ **每题都走 fullctx**。

记录每题 `level`（`L0` = 被拦住 / `FULLCTX` = 下沉读全文）→ 直接得到：
  · **A 题（只需标题+摘要）该停在 L0** —— 看有多少**被拦住**
  · **B 题（必须读全文）该下沉** —— 看有多少**真下沉**；停在 L0 的 = 「该下沉却没下沉」= 唯一的错
以及**集合级**：臂2 − 臂1 = 「这 N 题全放全文要多花多少」。

判分 = 外部裁判（`PAPERPILOT_JUDGE_*`，GLM），gold 用造题时落盘的 gold/evidence（**与出题模型不同族**）。
用法：`python retrieval/tmp/_run_route_ab.py [--limit N] [--arm1-only]`
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from paperpilot.graph import ask as graph_ask          # noqa: E402
from paperpilot.tools import llm                        # noqa: E402

HERE = Path(__file__).resolve().parent
SET = HERE / "_route_set.jsonl"
OUT = HERE / "_route_ab.jsonl"
PASS = 4

_JUDGE_SYS = (
    "你是严谨的论文问答裁判。你会看到：① 论文中的一道问题；② 人工标注的标准答案（可能含原文摘录）；"
    "③ 待评测系统回答与它引用的论文原文。请独立判断系统回答是否**准确**回答了问题（与标准答案一致、不编造、"
    '不遗漏关键信息）。只输出 JSON：{"score": 1-5, "reason": "一句话"}。5=完全正确且细节齐全；'
    "4=正确仅缺次要细节；3=部分正确有重要遗漏或轻微错误；2=明显错误或遗漏核心；1=答非所问或编造。")
_JUDGE_USER = """【问题】
{question}

【标准答案】
gold: {gold}
evidence: {evidence}

【系统回答】
{answer}

【引用原文】
{cites}

请打分(1-5)。引用必须真实支撑，否则视为编造降分。"""


def judge(q: str, gold: str, ev: str, answer: str, cites: list) -> int:
    ct = "\n".join(f"[{i + 1}] {str(c.get('evidence') or c)[:200]}"
                   for i, c in enumerate((cites or [])[:5])) or "（无引用）"
    try:
        raw = llm.judge_json(_JUDGE_SYS,
                             _JUDGE_USER.format(question=q, gold=gold, evidence=ev,
                                                answer=(answer or "")[:1500], cites=ct),
                             temperature=0.0, max_tokens=200)
        return int(float(raw.get("score", 0))) if isinstance(raw, dict) else 0
    except Exception:  # noqa: BLE001
        return 0


def run_arm(recs: list[dict], arm: str) -> None:
    """就地写入 rec[arm] = {level, pass, calls, prompt, sec, score}。

    ★ 语料 = **[目标篇, 干扰篇]**（系统要求 ≥2 篇，无单篇路径）。
    """
    t_all = time.time()
    for i, r in enumerate(recs, 1):
        if r.get(arm):
            continue
        pdfs = [r["pdf"], r["distractor"]]
        llm.reset_usage()
        t0 = time.time()
        try:
            out = graph_ask(r["question"], pdfs)
            answer = str(out.get("answer") or "")
            cites = list(out.get("cites") or [])
            lv = str(((out.get("debug") or {}).get("answer") or {}).get("level") or "")
        except Exception as e:  # noqa: BLE001
            answer, cites, lv = f"ERR {type(e).__name__}: {e}", [], "err"
        u = llm.usage_stats()
        sec = time.time() - t0
        llm.reset_usage()
        sc = judge(r["question"], r["gold"], r["evidence"], answer, cites)
        r[arm] = {"level": lv, "score": sc, "pass": sc >= PASS,
                  "calls": u["calls"], "prompt": u["prompt_tokens"], "sec": round(sec, 1),
                  "n_cites": len(cites), "ans": answer[:200]}
        print(f"  [{arm} {i}/{len(recs)}] {r['qid']:<18} {lv:<8} score={sc} "
              f"calls={u['calls']} tok={u['prompt_tokens']} {sec:.0f}s", flush=True)
        OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in recs), encoding="utf-8")
    print(f"  {arm} 臂完成 t={time.time() - t_all:.0f}s", flush=True)


_SYS_DS = ("你是严格的判据校验员。只依据给定【材料】判断：能否**完整、准确**回答【问题】？"
           "★ 材料里若已含同一事实（换说法/换单位也算），算「能答」。"
           "只输出 JSON：{\"enough\": true/false, \"why\": \"一句话\"}。")


def label_audit(recs: list[dict]) -> None:
    """★ 跨模型标签复核：用**裁判模型（GLM，`PAPERPILOT_JUDGE_*`）**看**语料实际的 L0 材料**够不够。
    ⚠️ **必须用 GLM**：Router 的 `judge_l0` 走的是 `llm.chat_json` = **主模型（DeepSeek）**
    （`judge.py:169`）→ 复核若也用主模型就成了**同源自证**。"""
    from paperpilot.components import router as RT
    for i, r in enumerate(recs, 1):
        if r.get("l0_enough_ds") is not None:
            continue
        try:
            st = RT.load_l0([r["pdf"], r["distractor"]])
            mat = "\n".join([str(st.get("title") or ""), str(st.get("overview") or "")]
                            + [str(x.get("text") if isinstance(x, dict) else x)
                               for x in (st.get("core_points") or []) + (st.get("limitations") or [])])
            raw = llm.judge_json(_SYS_DS, f"【材料】\n{mat}\n\n【问题】\n{r['question']}",
                                 temperature=0.0, max_tokens=200)
            r["l0_enough_ds"] = bool(raw.get("enough")) if isinstance(raw, dict) else None
            r["l0_chars_corpus"] = len(mat)
        except Exception:  # noqa: BLE001
            r["l0_enough_ds"] = None
        if i % 10 == 0:
            OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in recs),
                           encoding="utf-8")
    OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in recs), encoding="utf-8")
    ok = [r for r in recs if r.get("l0_enough_ds") is not None]
    if ok:
        agree = sum(1 for r in ok
                    if bool(r["l0_enough_ds"]) == (r["grp"] == "A"))
        print(f"  ★ 标签复核（**GLM 裁判模型**判「语料 L0 材料够不够」vs 构造标签）："
              f"一致 {agree}/{len(ok)} = {agree / len(ok) * 100:.0f}%", flush=True)


def report(recs: list[dict]) -> None:
    from collections import Counter, defaultdict
    groups = [("A", "只需标题+摘要（该停在 L0）"), ("B", "必须读全文（该下沉）")]

    def arm_of(name: str) -> str:
        return "分层（现状）" if name == "a1" else "不分层（全放全文）"

    print("\n" + "=" * 96)
    print("  ① 路由混淆矩阵（臂1 = 现状分层）")
    print(f"  {'组':<26}{'n':>4}{'停 L0（拦住）':>15}{'下沉 FULLCTX':>15}{'其它/err':>10}")
    for g, desc in groups:
        sub = [r for r in recs if r["grp"] == g and r.get("a1")]
        if not sub:
            continue
        c = Counter(str(r["a1"]["level"]) for r in sub)
        l0 = c.get("L0", 0)
        full = sum(v for k, v in c.items() if k in ("FULLCTX", "L3"))
        print(f"  {desc:<26}{len(sub):>4}{l0:>15}{full:>15}{len(sub) - l0 - full:>10}")

    print("\n  ② ★ 关键指标")
    for g, desc in groups:
        sub = [r for r in recs if r["grp"] == g and r.get("a1")]
        if not sub:
            continue
        c = Counter(str(r["a1"]["level"]) for r in sub)
        if g == "A":
            ok = c.get("L0", 0)
            print(f"  · A 组**被拦住**（进 L0 直答）= {ok}/{len(sub)} = {ok / len(sub) * 100:.0f}%"
                  f"  → 不分层（臂2）在这组的通过数对比见 ③")
        else:
            bad = c.get("L0", 0)
            print(f"  · B 组**该下沉却没下沉**（停在 L0）= {bad}/{len(sub)} = {bad / len(sub) * 100:.0f}%"
                  f"  ★ 这才是唯一的错（其余下沉题即便答错，也不是分层造成的）")

    print("\n  ③ 集合级：这 N 题「全放全文」要多花多少")
    print(f"  {'臂':<22}{'总calls':>9}{'总prompt':>12}{'总秒':>9}{'通过':>7}{'通过率':>8}")
    for name in ("a1", "a2"):
        sub = [r for r in recs if r.get(name)]
        if not sub:
            continue
        print(f"  {arm_of(name):<22}{sum(r[name]['calls'] for r in sub):>9}"
              f"{sum(r[name]['prompt'] for r in sub):>12,}"
              f"{sum(r[name]['sec'] for r in sub):>9.0f}"
              f"{sum(1 for r in sub if r[name]['pass']):>7}"
              f"{sum(1 for r in sub if r[name]['pass']) / len(sub) * 100:>7.1f}%")
    a1 = [r for r in recs if r.get("a1")]
    a2 = [r for r in recs if r.get("a2")]
    if a1 and a2 and sum(r["a1"]["prompt"] for r in a1):
        t1, t2 = sum(r["a1"]["prompt"] for r in a1), sum(r["a2"]["prompt"] for r in a2)
        c1, c2 = sum(r["a1"]["calls"] for r in a1), sum(r["a2"]["calls"] for r in a2)
        s1, s2 = sum(r["a1"]["sec"] for r in a1), sum(r["a2"]["sec"] for r in a2)
        print(f"\n  ★ 不分层 vs 分层：prompt **+{t2 - t1:,}（{(t2 / t1 - 1) * 100:+.0f}%）** ｜ "
              f"calls +{c2 - c1}（{(c2 / c1 - 1) * 100:+.0f}%） ｜ 时间 +{s2 - s1:.0f}s"
              f"（{(s2 / s1 - 1) * 100:+.0f}%）")
        print(f"    → 即：**分层省下了 {t2 - t1:,} prompt token（{t2 / t1:.2f}×）、{s2 - s1:.0f}s** —— "
              f"代价是 A/B 两组在 ② 里的路由错误率")

    print("\n  ④ 分组通过数（分层 vs 不分层）")
    for g, desc in groups:
        sub = [r for r in recs if r["grp"] == g]
        if not sub or not sub[0].get("a2"):
            continue
        p1 = sum(1 for r in sub if r["a1"]["pass"])
        p2 = sum(1 for r in sub if r["a2"]["pass"])
        print(f"  {desc:<26} 分层 {p1}/{len(sub)} ｜ 不分层 {p2}/{len(sub)} ｜ Δ={p2 - p1:+d}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--balance", action="store_true", help="两组各截到 min(nA,nB) 题，使混淆矩阵可比")
    ap.add_argument("--arm1-only", action="store_true")
    args = ap.parse_args()
    llm._load_dotenv(str(ROOT))

    recs = [json.loads(x) for x in SET.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.balance:
        na = [r for r in recs if r["grp"] == "A"]
        nb = [r for r in recs if r["grp"] == "B"]
        k = min(len(na), len(nb))
        recs = na[:k] + nb[:k]
        print(f"  已平衡：两组各 {k} 题")
    if args.limit:
        recs = recs[:args.limit]
    from collections import Counter
    print(f"  题数 = {len(recs)} ｜ 分组 {dict(Counter(r['grp'] for r in recs))}")
    if OUT.exists():                      # 断点续跑：合并已有结果
        old = {json.loads(x)["qid"]: json.loads(x)
               for x in OUT.read_text(encoding="utf-8").splitlines() if x.strip()}
        for r in recs:
            if r["qid"] in old:
                r.update({k: v for k, v in old[r["qid"]].items() if k in ("a1", "a2")})

    # ★ 语料 = [目标篇, 干扰篇]（系统要 ≥2 篇，无单篇路径）
    stems = sorted({r["stem"] for r in recs})
    for i, r in enumerate(recs):
        others = [s for s in stems if s != r["stem"]]
        r["distractor"] = f"{others[i % len(others)]}.pdf" if others else r["pdf"]

    label_audit(recs)
    print(f"  语料 L0 材料中位 = "
          f"{int(sorted(r.get('l0_chars_corpus') or 0 for r in recs)[len(recs) // 2]):,} 字符", flush=True)

    run_arm(recs, "a1")
    if not args.arm1_only:
        # ★ 臂2：把 judge_l0 换成"恒判不够" → 每题都走 fullctx（**运行时补丁**）
        from paperpilot.agents.nodes import judge as J
        orig = J.judge_l0

        def _never_enough(st):  # noqa: ANN001
            dbg = dict(st.get("debug") or {})
            dbg["judge_l0"] = {"enough": False, "gap": "[forced] 全放全文对照臂"}
            return {"verdict": {"enough": False, "gap": "[forced]", "target_sections": []},
                    "route": ["L0"], "debug": dbg}

        J.judge_l0 = _never_enough
        try:
            print("\n  ── 臂2：已把 judge_l0 打成「恒判不够」→ 全部走 fullctx ──")
            run_arm(recs, "a2")
        finally:
            J.judge_l0 = orig
    report(recs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
