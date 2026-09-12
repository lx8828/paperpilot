"""全流程冒烟：Router 组件接入 v3 + 新 SYSTEM(硬引用) + gate on，一次跑通。"""
from __future__ import annotations
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.graph import ask  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

OUT = Path("qa/recall/_flow_smoke.txt")


def main() -> int:
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    os.environ["PAPERPILOT_VALIDATOR_GATE"] = "1"
    lines = []
    cases = [
        ("2609.02056v1.pdf", "这篇论文的核心方法是什么？"),
        ("qasper_1901.00570.qpdf", "What is the main contribution of this paper?"),
    ]
    for pdf, q in cases:
        try:
            r = ask(q, pdf)
            dbg = r.get("debug") or {}
            lv = ((dbg.get("answer") or {}).get("level", "")) or ""
            route = ",".join(r.get("route") or [])
            ans = (r.get("answer") or "")[:120].replace("\n", " ")
            v = r.get("validator") or {}
            lines.append(f"[{pdf}] q={q[:40]}")
            lines.append(f"  route={route} level={lv} n_cites={len(r.get('cites') or [])}")
            lines.append(f"  gate action={v.get('action')} issues={len(v.get('issues') or [])}")
            lines.append(f"  ans: {ans}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"[{pdf}] ERROR: {type(e).__name__}: {e}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
