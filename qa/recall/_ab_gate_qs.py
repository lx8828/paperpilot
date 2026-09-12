"""修复回路题集实测：HyGRAIL 需检索题 × GATE on/off 双对照。

指标：action 分布（pass/repaired/fallback）、issues 类型、关键词判定 pass(on/off)、
每步 LLM 成本。关键词判定用题集自带 must_have/must_all/must_not（对称，不烧裁判）。
"""
from __future__ import annotations
import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

OUT = Path("qa/recall/ab_gate_qs_result.json")
PDF = "2609.02056v1.pdf"
QALL = json.loads(Path("qa/questions/2609.02056v1.json").read_text(encoding="utf-8"))
SEL = [q for q in QALL if "L3" in (q.get("expect") or []) or "L2" in (q.get("expect") or [])]


def must_kw(q) -> list[str]:
    out = list(q.get("must_all") or []) + list(q.get("must_have") or [])
    return [k for k in out if len(k) >= 2]


def pass_kw(q, answer: str) -> bool:
    a = answer or ""
    if not (q.get("must_all") or []):
        if not all(k in a for k in (q.get("must_have") or [])):
            return False
    else:
        if not all(k in a for k in q.get("must_all") or []):
            return False
    for k in q.get("must_not") or []:
        if k in a:
            return False
    return True


def run_one(q, gate: str) -> dict:
    os.environ["PAPERPILOT_VALIDATOR_GATE"] = gate
    llm.reset_usage()
    t0 = time.time()
    try:
        r = graph_ask(q["question"], PDF)
        dbg = r.get("debug") or {}
        level = ((dbg.get("answer") or {}).get("level", "")) or ""
    except Exception as e:  # noqa: BLE001
        return {"err": f"{type(e).__name__}: {e}"[:150], "sec": round(time.time() - t0)}
    u = llm.usage_stats()
    v = r.get("validator") or {}
    return {"sec": round(time.time() - t0, 1),
            "action": v.get("action", "(no gate)"),
            "pass": pass_kw(q, r.get("answer") or ""),
            "issues": [(i.get("sev"), i.get("type")) for i in (v.get("issues") or [])],
            "calls": u.get("calls"), "prompt": u.get("prompt_tokens"),
            "level": level, "n_cites": len(r.get("cites") or []),
            "answer": (r.get("answer") or "")[:120]}


def main() -> int:
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    recs = []
    for i, q in enumerate(SEL, 1):
        off = run_one(q, "0")
        on = run_one(q, "1")
        recs.append({"qid": q["qid"], "intent": q.get("intent"),
                     "off": off, "on": on,
                     "p_off": off.get("pass"), "p_on": on.get("pass")})
        print(f"[{i}/{len(SEL)}] {q['qid']} {q.get('intent'):14} "
              f"off pass={off.get('pass')} | on action={on.get('action')} "
              f"pass={on.get('pass')} issues={on.get('issues')}", flush=True)
        OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    # 汇总
    from collections import Counter
    acts = Counter(r["on"].get("action") for r in recs)
    print("=" * 60)
    print("action 分布:", dict(acts))
    print(f"关键词 pass: off {sum(r['p_off'] for r in recs)}/{len(recs)}  "
          f"on {sum(r['p_on'] for r in recs)}/{len(recs)}")
    print("prompt/题: off {:.0f}  on {:.0f}".format(
        sum(r["off"].get("prompt", 0) for r in recs) / len(recs),
        sum(r["on"].get("prompt", 0) for r in recs) / len(recs)))
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
