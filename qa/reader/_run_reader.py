"""小白帮读问答题 runner（A 模块）：逐题 B2 作答落盘，不做 LLM 判分（人工按帮读三问核）。"""
from __future__ import annotations
import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))
from paperpilot.graph import ask as graph_ask  # noqa: E402
from paperpilot.tools import llm  # noqa: E402


def main() -> int:
    llm._load_dotenv(str(ROOT))
    data = json.load(open(ROOT / "qa/reader/questions.json", encoding="utf-8"))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    recs = []
    for i, it in enumerate(data["items"], 1):
        llm.reset_usage()
        t0 = time.time()
        try:
            r = graph_ask(it["question"], it["pdf"])
            ans = (r.get("answer") or "").strip()
            cites = list(r.get("cites") or [])
            dbg = r.get("debug") or {}
            lvl = (dbg.get("answer") or {}).get("level", "")
            rt = list(r.get("route") or [])
        except Exception as e:  # noqa: BLE001
            ans, cites, lvl, rt = f"ERR {type(e).__name__}: {e}", [], "error", []
        u = llm.usage_stats()
        recs.append({"qid": it["qid"], "pdf": it["pdf"], "question": it["question"],
                     "answer": ans, "cites": cites[:5], "cites_n": len(cites),
                     "level": lvl, "route": rt, "time_s": round(time.time() - t0, 1),
                     "gen": {k: u[k] for k in ("calls", "prompt_tokens", "completion_tokens")}})
        print(f"[{i:02d}] {it['qid']} level={lvl} t={recs[-1]['time_s']}s calls={u['calls']}", flush=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / f"qa/reader/reader_run_{ts}.json"
    out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done →", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
