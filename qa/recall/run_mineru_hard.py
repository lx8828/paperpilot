"""端到端自测：10 篇正常论文（MinerU 解析）× 3 hard 题，跑我们自己接的全链。

链路 = paperpilot.graph.ask（默认 v3 两级 + nol3j + 输出闸门），
解析源 = MinerU（PAPERPILOT_USE_MINERU=1，产物 assets/artifacts/out_mineru/<stem>/）。
裁判 = glm-4-flash 异源，对照人工金标准（mineru_hard_set_v1.json），≥4 = pass。
断点续跑：结果落盘 qa/recall/mineru_hard_result.json，重跑跳过已完成 qid。
"""
from __future__ import annotations
import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

# 分臂评测（PP_HARD_ARM 未设 = 旧行为：全链 MinerU + 旧路径，保持兼容）
#   plain  A0：pymupdf 正文，不注入表格/公式
#   inject A1：pymupdf 正文 + MinerU 表格/公式按页注入（**线上默认**）
#   mineru A2：全链 MinerU（MinerU 切分当主源）
ARM_ENV = {
    "plain": {"PAPERPILOT_USE_MINERU": "0", "PAPERPILOT_MINERU_INJECT": "0"},
    "inject": {"PAPERPILOT_USE_MINERU": "0", "PAPERPILOT_MINERU_INJECT": "1"},
    "mineru": {"PAPERPILOT_USE_MINERU": "1", "PAPERPILOT_MINERU_INJECT": "1"},
}
ARM = os.environ.get("PP_HARD_ARM", "")
RUN = os.environ.get("PP_HARD_RUN", "1")
if ARM:
    if ARM not in ARM_ENV:
        raise SystemExit(f"未知 arm: {ARM}（可选 {list(ARM_ENV)}）")
    os.environ.update(ARM_ENV[ARM])
    OUT = Path(f"qa/recall/ab_{ARM}_r{RUN}.json")
    REPORT = Path(f"qa/recall/MINERU_AB_{ARM}_r{RUN}.md")
else:
    os.environ["PAPERPILOT_USE_MINERU"] = "1"
    OUT = Path("qa/recall/mineru_hard_result.json")
    REPORT = Path("qa/recall/MINERU_HARD_20260910.md")
ARM_LABEL = {"plain": "A0 无 MinerU（pymupdf 正文，无表格注入）",
             "inject": "A1 注入（pymupdf 正文 + MinerU 表格/公式按页注入；**线上默认**）",
             "mineru": "A2 全链 MinerU（MinerU 切分当主源）"}.get(ARM, "legacy 全链 MinerU")

from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

SET = Path("qa/recall/mineru_hard_set_v1.json")
PASS = 4

_JUDGE_SYS = (
    "你是严谨的论文问答裁判。你会看到：① 面向某篇论文的一个问题；② 人工核对过的标准答案"
    "（含具体数值/成员）；③ 待评测系统回答与它引用的论文原文。请独立判断系统回答是否**准确**"
    "回答了问题（与标准答案一致、不编造、不遗漏关键信息）。只输出 JSON："
    '{"score": 1-5, "reason": "一句话"}。5=完全正确且细节齐全（数值/成员全对）；'
    "4=正确仅缺次要细节；3=部分正确有重要遗漏或轻微错误；2=明显错误或遗漏核心；1=答非所问或编造。")
_JUDGE_USER = """【问题】
{question}

【标准答案】
{gold}

【系统回答】
{answer}

【引用原文】
{cites}

请打分(1-5)。数值/成员必须与标准答案一致；引用必须真实支撑，否则视为编造降分。"""


def judge(question: str, gold: str, answer: str, cites: list[dict]) -> tuple[int, str]:
    ct = "\n".join(f"[{i+1}](p{c.get('page','?')}) {str(c.get('evidence',''))[:600]}"
                   for i, c in enumerate(cites[:5])) or "（无引用）"
    user = _JUDGE_USER.format(question=question, gold=gold,
                              answer=(answer or "")[:2000], cites=ct)
    try:
        raw = llm.judge_json(_JUDGE_SYS, user, temperature=0.0)
        if isinstance(raw, dict):
            return int(float(raw.get("score", 0))), str(raw.get("reason", ""))[:200]
    except Exception as e:  # noqa: BLE001
        return 0, f"judge_err:{type(e).__name__}"
    return 0, "judge_parse_fail"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    items = json.loads(SET.read_text(encoding="utf-8"))["items"]
    if args.limit:
        items = items[: args.limit]
    done = {}
    if OUT.exists():
        try:
            done = {r["qid"]: r for r in json.loads(OUT.read_text(encoding="utf-8"))}
        except Exception:
            done = {}
    recs = list(done.values())
    t0 = time.time()
    for i, it in enumerate(items, 1):
        qid = it["qid"]
        if qid in done:
            continue
        llm.reset_usage()
        pdf = f"{it['pid']}.pdf"
        try:
            r = graph_ask(it["question"], pdf)
            ans = r.get("answer") or ""
            cites = list(r.get("cites") or [])
            dbg = r.get("debug") or {}
            level = ((dbg.get("answer") or {}).get("level", "")) or ""
            v = r.get("validator") or {}
            err = ""
        except Exception as e:  # noqa: BLE001
            ans, cites, level, v = f"ERR {e}", [], "err", {}
            err = f"{type(e).__name__}: {e}"
        u = llm.usage_stats()
        llm.reset_usage()
        score, reason = judge(it["question"], it["gold"], ans, cites)
        rec = {
            "arm": ARM or "mineru",
            "qid": qid, "pid": it["pid"], "question": it["question"], "gold": it["gold"],
            "src": it.get("src", ""), "answer": ans, "n_cites": len(cites),
            "score": score, "pass": score >= PASS, "reason": reason,
            "level": level, "action": v.get("action"), "err": err,
            "calls": u["calls"], "prompt": u["prompt_tokens"],
            "completion": u["completion_tokens"],
        }
        recs = [r for r in recs if r["qid"] != qid] + [rec]
        OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/{len(items)}] {qid} {it['pid']} score={score} "
              f"({level}/{v.get('action')}) {reason[:70]}", flush=True)

    # 汇总
    n = len(recs)
    p = sum(1 for r in recs if r["pass"])
    L = ["# MinerU 端到端 hard 自测（10 篇 × 3 题）", "",
         f"> 链路：graph.ask（v3 两级 + nol3j + 闸门）| 解析臂：{ARM_LABEL} | 裁判：glm-4-flash ≥4=pass",
         f"> **总通过 {p}/{n} = {p/max(n,1)*100:.0f}%** | 均分 "
         f"{sum(r['score'] for r in recs)/max(n,1):.2f}",
         "| 论文 | 题 | 通过 | 均分 |", "|---|---|---|---|"]
    pids = []
    for r in recs:
        if r["pid"] not in pids:
            pids.append(r["pid"])
    for pid in pids:
        sub = [r for r in recs if r["pid"] == pid]
        L.append(f"| {pid} | {len(sub)} | {sum(1 for x in sub if x['pass'])} | "
                 f"{sum(x['score'] for x in sub)/len(sub):.1f} |")
    L += ["", "## 逐题", "| qid | 论文 | 分 | level | 闸门 |", "|---|---|---|---|---|"]
    for r in sorted(recs, key=lambda x: x["score"]):
        L.append(f"| {r['qid']} | {r['pid']} | {r['score']} | {r['level']} | "
                 f"{r['action']} |")
    L += ["", "## 失败题详情（score<4）", ""]
    for r in sorted(recs, key=lambda x: x["score"]):
        if r["pass"]:
            continue
        L += [f"### {r['qid']} ({r['pid']}) score={r['score']} [{r['level']}/{r['action']}]",
              f"Q: {r['question']}", f"gold: {r['gold']}",
              f"A: {(r['answer'] or '')[:700]}", f"judge: {r['reason']}", ""]
    txt = "\n".join(L)
    REPORT.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
