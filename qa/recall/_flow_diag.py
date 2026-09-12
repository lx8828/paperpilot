"""诊断 case1：L0 概述答案被 gate 兜底的具体 issues。"""
from __future__ import annotations
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.components import validator as V  # noqa: E402
from paperpilot.graph import ask  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

OUT = Path("qa/recall/_flow_diag.txt")


def main() -> int:
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    os.environ["PAPERPILOT_VALIDATOR_GATE"] = "0"  # 先拿原始输出
    q = "这篇论文的核心方法是什么？"
    pdf = "2609.02056v1.pdf"
    r = ask(q, pdf)
    ans = r.get("answer") or ""
    cites = list(r.get("cites") or [])
    lines = [f"raw level={( (r.get('debug') or {}).get('answer') or {}).get('level')} n_cites={len(cites)}"]
    lines.append(f"ans: {ans[:220]}")
    g = V.gate(q, ans, cites)
    lines.append(f"gate action={g['action']}")
    for i in g["issues"]:
        lines.append(f"  [{i['sev']}] {i['type']}: {i['detail'][:160]} | {i.get('sentence','')[:60]}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
