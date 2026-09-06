"""QASPER 强制深潜实验：浅层 fail 强制走 L3-only 路径，测深潜回报率。

背景（QASPER_EVAL_LOG.md 实验 E1）：
    V1 有 40 条 fail 停在 L1/L2（judge 判够就收口）。若放宽下钻条件它们会进 L3。
    改漏斗前先量化：强制 L3（search_l3→judge_l3→answer/unknown）能救回几条？

用法：
    uv run python cli/run_qasper_deep.py --run qa/qasper_run_20260905_224146.json
输出：
    qa/qasper_deep_<ts>.json   每条（原 fail vs 深潜后）记录
    qa/qasper_deep_<ts>.md     对比汇总
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / "qa"
sys.path.insert(0, str(ROOT / "cli"))  # noqa: E402 复用 run_qasper_eval 的裁判

import run_qasper_eval as rqe  # noqa: E402

from paperpilot.agents.nodes.answer import answer_unknown, generate_answer  # noqa: E402
from paperpilot.agents.nodes.judge import judge_l3  # noqa: E402
from paperpilot.agents.nodes.pull_chunk import search_l3  # noqa: E402


def route_deepest(route: list) -> str:
    r = set(route or [])
    if "L3" in r or "answer_L3" in r:
        return "L3"
    if "L2" in r or "answer_L2" in r:
        return "L2"
    if "L1" in r or "answer_L1" in r:
        return "L1"
    return "L0"


def force_l3(question: str, pdf: str, title: str) -> dict:
    """跳过 L0/L1/L2，强制走 L3-only：search_l3 → judge_l3 → answer/unknown。"""
    t0 = time.time()
    state: dict = {"question": question, "pdf": pdf, "title": title}
    state.update(search_l3(state))
    state.update(judge_l3(state))
    v = state.get("verdict") or {}
    route = list(state.get("route") or [])
    if v.get("enough"):
        state.update(generate_answer(state))
    else:
        state.update(answer_unknown(state))
    dbg = state.get("debug") or {}
    ans_dbg = dbg.get("answer") or {}
    return {
        "answer": state.get("answer") or "",
        "cites": list(state.get("cites") or []),
        "route": list(state.get("route") or []),
        "level": ans_dbg.get("level", ""),
        "l3_n": len(state.get("l3_chunks") or []),
        "time_s": round(time.time() - t0, 2),
    }


def main() -> int:
    llm_cfg = rqe.llm
    llm_cfg._load_dotenv(str(ROOT))
    if not llm_cfg.is_configured():
        print("[错误] 主 LLM 未配置")
        return 1
    if not llm_cfg.judge_configured():
        print("[错误] 裁判 LLM 未配置（PAPERPILOT_JUDGE_*）")
        return 1

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="V1 run json（取其中的浅层 fail）")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条（调试）")
    args = ap.parse_args()

    recs = json.loads(Path(args.run).read_text(encoding="utf-8"))
    shallow_fail = [r for r in recs
                    if r.get("status") == "fail"
                    and route_deepest(r.get("route")) in ("L1", "L2")]
    if args.limit:
        shallow_fail = shallow_fail[: args.limit]
    print(f"V1 fail 中停在 L1/L2 的: {len(shallow_fail)} 条")

    papers = rqe.load_papers()
    # paper 名 → 题目列表
    qa_by_paper: dict[str, list[dict]] = {}
    for r in shallow_fail:
        name = r["paper"]  # qasper_<pid>.qpdf
        qa_by_paper.setdefault(name, []).append(r)

    out: list[dict] = []
    for name, qs in qa_by_paper.items():
        pid = name.replace(rqe._QASPER_PREFIX, "").replace(rqe._QASPER_SUFFIX, "")
        paper = papers.get(pid, {})
        title = paper.get("title", "")
        all_qas = {q["question_id"]: q for q in paper.get("qas", [])}
        print(f"\n===== {pid} {title[:40]}｜{len(qs)} 条浅层 fail =====", flush=True)
        for old in qs:
            q = all_qas.get(old["qid"], {})
            try:
                r = force_l3(old["question"], name, title)
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {old['qid'][:12]} 深潜异常: {type(e).__name__}: {e}",
                      flush=True)
                out.append({**old, "deep_route": [], "deep_level": "error",
                            "deep_score": 0, "deep_status": "error",
                            "deep_reason": f"{type(e).__name__}: {e}"})
                continue
            score, reason, is_pass = rqe._judge(q, r["answer"], r["cites"])
            status = ("pass" if is_pass else
                      ("unknown_ok" if r["level"] == "unknown" else "fail"))
            rec = {**old,
                   "deep_answer": r["answer"][:800],
                   "deep_cites_n": len(r["cites"]),
                   "deep_route": r["route"],
                   "deep_level": r["level"],
                   "deep_score": score,
                   "deep_reason": reason,
                   "deep_status": status,
                   "deep_time_s": r["time_s"]}
            out.append(rec)
            print(f"  [{status}] {old['qid'][:12]} 原fail({old['score']}) "
                  f"→ 深潜 score={score} {r['level']} L3top={r['l3_n']} "
                  f"{r['time_s']}s | {old['question'][:45]}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_file = QA_DIR / f"qasper_deep_{ts}.json"
    run_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总
    from collections import Counter
    changed = [r for r in out if r.get("deep_status") != "fail"]
    n_pass = sum(1 for r in out if r.get("deep_status") == "pass")
    lines = [f"# 强制深潜实验（{ts}）\n",
             f"- 样本：{len(out)} 条 V1 浅层 fail（停在 L1/L2）",
             f"- 深潜后: pass={n_pass}  fail={sum(1 for r in out if r.get('deep_status')=='fail')}  "
             f"unknown_ok={sum(1 for r in out if r.get('deep_status')=='unknown_ok')}  error={sum(1 for r in out if r.get('deep_status')=='error')}",
             f"- 非 fail 转化: {len(changed)} 条\n",
             "| paper | 问题 | 原score | 深潜score | 原route | 深潜route | 状态 |",
             "|---|---|---|---|---|---|---|"]
    for r in out:
        lines.append(f"| {r['paper'].replace(rqe._QASPER_SUFFIX,'')} | {r['question'][:42]} | "
                     f"{r['score']} | {r.get('deep_score','?')} | {'→'.join(r.get('route') or [])} | "
                     f"{'→'.join(r.get('deep_route') or [])} | {r.get('deep_status')} |")
    sum_file = QA_DIR / f"qasper_deep_{ts}.md"
    sum_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n记录: {run_file}\n汇总: {sum_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
