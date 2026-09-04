"""RAG 评估：对 qa_set 每道题真实跑 LangGraph 问答图，收集诊断信息。

对照 ground truth（qa_set 里每题的 hint_groups / note）检查：
    召回   ground-truth 该命中的组是否出现在检索 topK
    闸门   judge 的 enough/need_chunk_ids 判得如何
    升级   什么时候拉 chunk、拉了几个
    答案   与 note 大意是否一致（人工判读）

用法：
    uv run python cli/run_rag_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.graph import ask
from paperpilot.tools import llm

ROOT = Path(__file__).resolve().parents[1]
QA_FILE = ROOT / "qa" / "qa_set.json"
OUT_FILE = ROOT / "qa" / "rag_eval_out.json"


def summarize(q: dict, r: dict) -> dict:
    retrieved = r.get("retrieved", [])
    hint = q.get("hint_groups") or []
    hit_groups = [g for g in hint if g in {h["gid"] for h in retrieved}]
    verdict = r.get("verdict") or {}
    return {
        "qid": q["qid"],
        "level": q["level"],
        "question": q["question"],
        "note": q.get("note", ""),
        "hint_groups": hint,
        "retrieved_gids": [h["gid"] for h in retrieved],
        "recall_hit": hit_groups,
        "verdict": verdict,
        "n_chunks": len(r.get("chunks", [])),
        "answer": (r.get("answer") or "")[:400],
        "cites": r.get("cites") or [],
    }


def main() -> int:
    llm._load_dotenv(str(ROOT))
    filter_pdf = sys.argv[1] if len(sys.argv) > 1 else None
    data = json.loads(QA_FILE.read_text(encoding="utf-8"))
    if not llm.is_configured():
        print("[错误] LLM 未配置")
        return 1

    questions = [q for q in data["questions"]
                 if not filter_pdf or q["pdf"] == filter_pdf]
    results = []
    for i, q in enumerate(questions, 1):
        print("=" * 80)
        print(f"[{i}/{len(data['questions'])}] {q['qid']} ({q['level']}) {q['question']}")
        t0 = time.time()
        try:
            r = ask(q["question"], q["pdf"])
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ 图执行失败: {type(e).__name__}: {e}")
            results.append({"qid": q["qid"], "error": f"{type(e).__name__}: {e}"})
            continue
        dt = time.time() - t0
        s = summarize(q, r)
        results.append(s)
        v = s["verdict"]
        print(f"  ({dt:.0f}s) 检索={len(s['retrieved_gids'])} | "
              f"hint命中={s['recall_hit']} | enough={v.get('enough')} | "
              f"升级chunk={s['n_chunks']}")
        if v.get("gap"):
            print(f"  gap: {str(v['gap'])[:100]}")
        print(f"  答: {' '.join(str(s['answer']).split())[:170]}")

    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n结果已保存: {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
