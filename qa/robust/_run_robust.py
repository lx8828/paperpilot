"""稳定性/同义改写 runner：每题 3 个问法跑 B2 + glm-4-flash 判分，看一致性。

用法：uv run python qa/robust/_run_robust.py [--q-limit N]
产物：qa/robust/robust_run_<ts>.json（逐变体 + 每题一致性汇总）
"""
from __future__ import annotations
import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from paperpilot.qasper_source import gold_answer, load_papers
from paperpilot.tools import llm

ROOT = Path(__file__).resolve().parents[2]
PASS = 4

_JUDGE_SYS = (
    "你是严谨的论文问答裁判。你会看到：① 论文中的一道问题；② 人工标注的标准答案"
    "（ground truth，可能包含论文原文摘录）；③ 待评测系统的回答与它引用的论文原文。"
    "请独立判断：系统的回答是否**准确**地回答了问题（与标准答案一致、不编造、不遗漏"
    '关键信息）。只输出 JSON：{"score": 1-5, "reason": "一句话理由"}。评分基准：5=完全正确且细节齐全；'
    "4=正确，仅缺次要细节或措辞差异；3=部分正确，有重要遗漏或轻微错误；"
    "2=明显错误或遗漏核心；1=答非所问或编造。"
)
_JUDGE_USER = """【问题】
{question}

【标准答案（人工 gold + 原文证据）】
gold: {gold}
evidence: {evidence}

【系统回答】
{answer}

【系统引用的论文原文】
{cites}

请打分（1-5）。注意：系统回答若声称有引用，引用原文必须真实支撑其说法，否则视为编造降分。"""


def judge(qorig: dict, question: str, answer: str, cites_text: str) -> tuple[int, str]:
    gold, ev = gold_answer(qorig)
    user = _JUDGE_USER.format(
        question=question,
        gold=gold or "（该题在论文中无答案，系统应诚实说明）",
        evidence=(ev or ""),
        answer=(answer or "")[:1200],
        cites=cites_text or "（无引用）",
    )
    try:
        raw = llm.judge_json(_JUDGE_SYS, user, temperature=0.0)
        score = int(float(raw.get("score", 0))) if isinstance(raw, dict) else 0
        reason = str(raw.get("reason", ""))[:160] if isinstance(raw, dict) else ""
    except (llm.LLMError, ValueError, TypeError) as e:
        return 0, f"裁判失败 {e}"
    return score, reason


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--q-limit", type=int, default=None)
    args = ap.parse_args()

    llm._load_dotenv(str(ROOT))
    data = json.load(open(ROOT / "qa/robust/questions.json", encoding="utf-8"))
    items = data["items"]
    if args.q_limit:
        items = items[: args.q_limit]
    papers = load_papers()
    # qid 前缀 → (pdf, 原始 q)
    index: dict[str, tuple[str, dict]] = {}
    for qid, qitem in enumerate(items):
        pre = qitem["qid"]
        for pid, p in papers.items():
            for q in p.get("qas") or []:
                if str(q.get("question_id", "")).startswith(pre):
                    index[pre] = (f"qasper_{pid}.qpdf", q)
                    break
            if pre in index:
                break
        if pre not in index:
            print(f"[warn] 未匹配 qid 前缀 {pre}")
    print(f"匹配 {len(index)}/{len(items)} 题")

    # 预热：LLM + embedder（丢弃计时/用量）
    try:
        llm.chat_text("你是连通性测试助手。", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    llm.reset_usage()

    out_rows = []
    t_start = time.time()
    for i, it in enumerate(items, 1):
        pdf, qorig = index[it["qid"]]
        gold, _ev = gold_answer(qorig)
        is_unans = not gold
        print(f"\n[{i}/{len(items)}] {it['qid']} unans={is_unans} {pdf}", flush=True)
        var_rows = []
        for key in ("v0", "v1", "v2"):
            llm.reset_usage()
            t0 = time.time()
            try:
                from paperpilot.graph import ask as graph_ask
                r = graph_ask(it[key], pdf)
                answer = (r.get("answer") or "").strip()
                cites = list(r.get("cites") or [])
                dbg = r.get("debug") or {}
                ans_dbg = dbg.get("answer") or {}
                level = ans_dbg.get("level", "")
                route = list(r.get("route") or [])
            except Exception as e:  # noqa: BLE001
                answer, cites, level, route = f"ERROR {type(e).__name__}: {e}", [], "error", []
            u_ans = llm.usage_stats()
            cite_text = "\n".join(
                f"[{j+1}] (p{c.get('page','?')}) {str(c.get('evidence',''))}"
                for j, c in enumerate(cites[:5]))
            score, reason = judge(qorig, it[key], answer, cite_text)
            jdg = llm.usage_stats()
            jdg = {k: jdg.get(k, 0) - u_ans.get(k, 0) for k in ("calls", "prompt_tokens", "completion_tokens")}
            vrow = {"variant": key, "question": it[key], "answer": answer[:400],
                    "score": score, "reason": reason, "pass": score >= PASS,
                    "level": level, "route": route,
                    "time_s": round(time.time() - t0, 1),
                    "gen": u_ans, "judge": jdg}
            var_rows.append(vrow)
            print(f"  {key}: score={score} {'PASS' if score>=PASS else 'fail'} "
                  f"level={level} t={vrow['time_s']}s", flush=True)
        scores = [v["score"] for v in var_rows]
        passes = [v["pass"] for v in var_rows]
        stable_pass = len(set(passes)) == 1
        spread = max(scores) - min(scores)
        out_rows.append({"qid": it["qid"], "unanswerable": is_unans,
                         "variants": var_rows, "scores": scores,
                         "pass_all": all(passes), "pass_any": any(passes),
                         "stable_pass": stable_pass, "spread": spread,
                         "sigma": round(__import__("statistics").pstdev(scores), 2)})
        # 每题落盘一次（可中断恢复认知）
        ts = "robust_run_{}.json".format(time.strftime("%Y%m%d_%H%M%S"))
        (ROOT / "qa/robust" / ts).write_text(json.dumps(out_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    n = len(out_rows)
    n_stable = sum(1 for r in out_rows if r["stable_pass"])
    n_allpass = sum(1 for r in out_rows if r["pass_all"])
    spreads = [r["spread"] for r in out_rows]
    print("\n" + "=" * 80)
    print(f"完成 {n} 题×3 变体：3变体 pass 一致率 = {n_stable}/{n} ({n_stable/n:.1%})")
    print(f"3变体全 pass 题 = {n_allpass}/{n}；分数跨度均值 = {sum(spreads)/n:.2f}")
    print(f"总耗时 {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
