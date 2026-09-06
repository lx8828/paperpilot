"""V2-A 实验：41 条浅层 fail 用新 judge（题型完整性）跑完整漏斗。

观察：下钻率 / L2 扩窗到半径几 / 预算是否耗尽 / 翻 pass 数 / L3 增量。

用法：
    uv run python cli/run_qasper_judgeA.py
输出：
    qa/qasper_judgeA_<ts>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / "qa"
sys.path.insert(0, str(ROOT / "cli"))

import run_qasper_eval as rqe  # noqa: E402
from paperpilot.graph import ask as graph_ask  # noqa: E402


def route_deepest(route: list) -> str:
    r = set(route or [])
    if "L3" in r or "answer_L3" in r:
        return "L3"
    if "L2" in r or "answer_L2" in r:
        return "L2"
    if "L1" in r or "answer_L1" in r:
        return "L1"
    return "L0"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", default="",
                    help="读取某次实验 run json，重跑其中选中的题")
    ap.add_argument("--take", default="fail", choices=("fail", "all"),
                    help="fail=只重跑仍 fail 的题；all=重跑全部（用于回归：把已 pass 的题也带上）")
    args = ap.parse_args()

    rqe.llm._load_dotenv(str(ROOT))
    if not rqe.llm.is_configured() or not rqe.llm.judge_configured():
        print("[错误] LLM 未配置")
        return 1

    prev = json.loads((QA_DIR / "qasper_run_20260905_224146.json").read_text(encoding="utf-8"))
    if args.from_run:
        base = json.loads((QA_DIR / args.from_run).read_text(encoding="utf-8"))
        if args.take == "all":
            qids = {r["qid"] for r in base}
            tag = "全部"
        else:
            qids = {r["qid"] for r in base if r.get("new_status") == "fail"}
            tag = "仍 fail"
        target = [r for r in prev if r["qid"] in qids]
        print(f"定向: 从 {args.from_run} 取{tag} {len(target)} 条")
    else:
        target = [r for r in prev if r.get("status") == "fail"
                  and route_deepest(r.get("route")) in ("L1", "L2")]
        print(f"目标: {len(target)} 条浅层 fail")

    papers = rqe.load_papers()
    by_paper: dict[str, list[dict]] = {}
    for r in target:
        by_paper.setdefault(r["paper"], []).append(r)

    out: list[dict] = []
    for name, qs in by_paper.items():
        pid = name.replace(rqe._QASPER_PREFIX, "").replace(rqe._QASPER_SUFFIX, "")
        paper = papers.get(pid, {})
        title = paper.get("title", "")
        all_qas = {q["question_id"]: q for q in paper.get("qas", [])}
        print(f"\n===== {pid} {title[:40]}｜{len(qs)} 条 =====", flush=True)
        for old in qs:
            q = all_qas.get(old["qid"], {})
            t0 = time.time()
            try:
                r = graph_ask(old["question"], name)
            except Exception as e:  # noqa: BLE001
                out.append({**old, "new_route": [], "new_level": "error",
                            "new_score": 0, "new_status": "error",
                            "new_reason": f"{type(e).__name__}: {e}",
                            "l2_trace": []})
                print(f"  [error] {old['qid'][:12]} {type(e).__name__}", flush=True)
                continue
            answer = r.get("answer") or ""
            cites = list(r.get("cites") or [])
            dbg = r.get("debug") or {}
            ans_dbg = dbg.get("answer") or {}
            score, reason, is_pass = rqe._judge(q, answer, cites)
            new_status = ("pass" if is_pass else
                          ("unknown_ok" if ans_dbg.get("level") == "unknown" else "fail"))
            rec = {**old,
                   "new_answer": answer[:600],
                   "new_route": list(r.get("route") or []),
                   "new_level": ans_dbg.get("level", ""),
                   "new_score": score,
                   "new_reason": reason,
                   "new_status": new_status,
                   "l2_step": dbg.get("l2_step"),
                   "judge_l2": dbg.get("judge_l2"),
                   "judge_l1": dbg.get("judge_l1"),
                   "n_facts": ans_dbg.get("n_facts"),
                   "facts": ans_dbg.get("facts"),
                   "time_s": round(time.time() - t0, 2)}
            out.append(rec)
            changed = "★翻盘" if old["status"] != new_status else ""
            print(f"  [{new_status}{changed}] {old['qid'][:12]} 原fail({old['score']}) "
                  f"→ score={score} {ans_dbg.get('level','?')} "
                  f"{'/'.join(r.get('route') or [])} {rec['time_s']}s | {old['question'][:40]}",
                  flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_file = QA_DIR / f"qasper_judgeA_{ts}.json"
    run_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n记录: {run_file}")

    # 汇总
    from collections import Counter
    print("=== 汇总 ===")
    print("新状态:", dict(Counter(r.get("new_status") for r in out)))
    deep = Counter(route_deepest(r.get("new_route")) for r in out)
    print("新深潜分布:", dict(deep))
    flip = [r for r in out if r.get("new_status") == "pass"]
    print(f"翻 pass: {len(flip)} / {len(out)}")
    for r in flip:
        print(f"  ✓ {r['paper'].replace('.qpdf','')[:16]} | {r['question'][:45]} | {r['score']}→{r['new_score']} | {'→'.join(r.get('new_route') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
