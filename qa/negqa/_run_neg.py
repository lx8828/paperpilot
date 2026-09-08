"""防幻觉负向组 runner（B2 单系统，20 题）。

判定（自动初判 + 人工复核）：
  contra 篡改数值题：
    HALLUC     答案把 fake_kw 当事实陈述（窗口内无否定、且未给真值）→ 编造 ❌
    OK         给出 real_kw（含否定回答或更正）→ ✅
    ABSENCE-WRONG  未给真值且"论文没给/未提及"口吻 → 误断言缺失 ❌
    ?          未命中以上，需人工看
  absent 不存在题：
    OK-REFUSE  答案含拒答/未提及口吻 → 诚实拒答 ✅
    ?          无明显拒答词 → 人工判断是否编造
"""
from __future__ import annotations
import io
import json
import re
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from paperpilot.graph import ask as graph_ask
from paperpilot.tools import llm

ROOT = Path(__file__).resolve().parents[2]
QA = ROOT / "qa" / "negqa"

ABSENCE_RE = re.compile(
    r"(未给出|没有给出|未提供|未提及|未提到|没有提到|没有提及|并未|未使用|没有使用|"
    r"未涉及|不包含|未包含|未找到|没有找到|未发现|论文[^，。；]{0,8}没有|"
    r"没有[^，。；]{0,6}提到|不确定|无法确认)", re.I)
NEG_WORDS = ("不", "并非", "不是", "没有", "未", "否", "没")
# 篡改值若出现，只看它前后窗口内是否有否定词，避免把"不是0.618，是0.429"误判成编造


def window_neg(ans: str, pos: int, span: int = 30) -> bool:
    seg = ans[max(0, pos - span): pos + span]
    return any(w in seg for w in NEG_WORDS)


def absence_toned(ans: str) -> bool:
    return bool(ABSENCE_RE.search(ans))


def verdict_contra(ans: str, fake_kws, real_kws):
    found_real = any(k in ans for k in real_kws)
    for fk in fake_kws:
        idx = ans.find(fk)
        while idx != -1:
            if not found_real and not window_neg(ans, idx):
                return "HALLUC(把假值当真)"
            idx = ans.find(fk, idx + 1)
    if found_real:
        return "OK(给出真值)"
    if absence_toned(ans):
        return "ABSENCE-WRONG(误断言论文缺失)"
    return "?(人工看)"


def verdict_absent(ans: str):
    if absence_toned(ans):
        return "OK-REFUSE(诚实拒答)"
    return "?(人工看是否编造)"


def main() -> int:
    llm._load_dotenv(str(ROOT))
    data = json.loads((QA / "questions.json").read_text(encoding="utf-8"))
    recs = []
    print("=" * 90)
    for i, q in enumerate(data["questions"], 1):
        pdf = q["pdf"]
        llm.reset_usage()
        t0 = time.time()
        try:
            r = graph_ask(q["question"], pdf)
            answer = (r.get("answer") or "").strip()
            dbg = r.get("debug") or {}
            ans_dbg = dbg.get("answer") or {}
            level = ans_dbg.get("level", "")
            route = list(r.get("route") or [])
            cites_n = len(r.get("cites") or [])
        except Exception as e:  # noqa: BLE001
            answer, level, route, cites_n = f"ERROR {type(e).__name__}: {e}", "error", [], 0
        u = llm.usage_stats()
        if q["type"] == "contra":
            vd = verdict_contra(answer, q.get("fake_kw", []), q.get("real_kw", []))
        else:
            vd = verdict_absent(answer)
        rec = {"pdf": pdf, "qid": q["qid"], "type": q["type"], "question": q["question"],
               "expect": q["expect"], "verdict": vd, "answer": answer[:600],
               "level": level, "route": route, "cites_n": cites_n,
               "time_s": round(time.time() - t0, 1),
               "gen": {k: u.get(k, 0) for k in ("calls", "prompt_tokens", "completion_tokens")}}
        recs.append(rec)
        print(f"\n[{i:02d}] {q['qid']} [{q['type']}] → {vd}  (level={level} route={'→'.join(route)})")
        print(f"    Q: {q['question'][:80]}")
        print(f"    A: {answer[:220].replace(chr(10), ' ')}")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = QA / f"neg_run_{ts}.json"
    out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    n_halluc = sum(1 for r in recs if r["verdict"].startswith("HALLUC"))
    n_ok = sum(1 for r in recs if r["verdict"].startswith("OK"))
    n_abswrong = sum(1 for r in recs if r["verdict"].startswith("ABSENCE"))
    n_q = sum(1 for r in recs if r["verdict"].startswith("?"))
    print("\n" + "=" * 90)
    print(f"总计 {len(recs)}：自动✅(OK*)={n_ok} ｜ 编造(HALLUC)={n_halluc} ｜ "
          f"误断言缺失(ABSENCE)={n_abswrong} ｜ 待人工(?)={n_q}")
    print(f"已落盘 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
