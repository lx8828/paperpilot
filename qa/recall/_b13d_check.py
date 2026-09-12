"""异常题 b13d0e46 归因：为什么 L3 检索只拿到 1 块 / 248 字符。"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.document_cache import current_source, ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import ChunkIndex  # noqa: E402
from paperpilot.qasper_source import load_papers  # noqa: E402

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "b13d0e46"


def main() -> int:
    papers = load_papers()
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            if not str(q.get("question_id") or "").startswith(PREFIX):
                continue
            pdf = f"qasper_{pid}.qpdf"
            print(f"pid={pid}  qid={q.get('question_id')}")
            print(f"Q: {q.get('question')}")
            print(f"source={current_source(pdf)}")
            ch = ordered_chunks(pdf)
            print(f"n_chunks={len(ch)}  total_chars={sum(len(c.text) for c in ch)}")
            for c in ch:
                leaf = (c.title_path or [""])[-1]
                print(f"  {c.chunk_id} blocks={c.n_blocks} len={len(c.text)} "
                      f"part={c.part or '-'} sec={leaf!r}")
            ft = p.get("full_text") or []
            print(f"full_text sections={len(ft)}  "
                  f"paras={sum(len(s.get('paragraphs') or []) for s in ft)}")
            for s in ft[:12]:
                ps = s.get("paragraphs") or []
                print(f"   sec={(s.get('section_name') or '')[:60]!r} paras={len(ps)} "
                      f"chars={sum(len(x) for x in ps)}")
            v = ChunkIndex(pdf).vectors()
            print(f"vectors shape={getattr(v, 'shape', None)}")
            return 0
    print("未找到")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
