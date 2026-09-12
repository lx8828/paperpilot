"""临时：真实 QASPER 题目 e2e 验证 supplement 链路（GATE=1）。

  - 545ff2（3600万 跨块，真数在全篇）→ 期望 supplement 触发、复核后保持原答案不误伤
  - cf63a4（7.5 天，若全文无此数）→ 期望走 high（真编数）→ repair/兜底
跑完把结果落盘供检查。
"""
from __future__ import annotations
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.graph import ask  # noqa: E402
from paperpilot.qasper_source import load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

OUT = Path("qa/recall/_supplement_e2e.txt")


def main() -> int:
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    os.environ["PAPERPILOT_VALIDATOR_GATE"] = "1"
    papers = load_papers()
    qmap = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qmap[str(q.get("question_id") or "")] = (pid, q)
    lines = []
    for pre in ("545ff2f769", "cf63a4f9fe"):
        try:
            qid = next(k for k in qmap if k.startswith(pre))
            pid, q = qmap[qid]
            pdf = f"qasper_{pid}.qpdf"
            qq = q.get("question", "")
            lines.append("=" * 78)
            lines.append(f"{pre[:10]} pid={pid} | Q: {qq[:100]}")
            r = ask(qq, pdf)
            v = r.get("validator") or {}
            dbg = r.get("debug") or {}
            lv = ((dbg.get("answer") or {}).get("level", "")) or ""
            lines.append(f"  route={','.join(r.get('route') or [])} level={lv} "
                         f"n_cites={len(r.get('cites') or [])}")
            lines.append(f"  gate action={v.get('action')} "
                         f"issues={len(v.get('issues') or [])} "
                         f"supplements={len(v.get('supplements') or [])}")
            for s in (v.get("supplements") or [])[:3]:
                lines.append(f"    supplement num={s.get('num_text')} "
                             f"chunks={[c.get('chunk_id') for c in (s.get('chunks') or [])]}")
            lines.append(f"  ans: {(r.get('answer') or '')[:300].replace(chr(10), ' ')}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"{pre[:10]} ERROR: {type(e).__name__}: {e}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
