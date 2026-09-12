"""非判断题的 LLM 体检仍工作：注入编造对象应被抓 unsupported。"""
from __future__ import annotations
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.components.retriever import ChunkRetriever  # noqa: E402
from paperpilot.components.validator import check  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

OUT = Path("qa/recall/_valid_llm_smoke.txt")


def main() -> int:
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    pdf = "2609.03335v1.pdf"
    q = "这篇论文的调度系统相比现有工作流调度器有什么改进？"  # 非判断题
    hits = ChunkRetriever(pdf).search(q, top_k=6)
    entries = [{"kind": "chunk", "text": h.get("text", ""), "page": h.get("page", 0)} for h in hits]
    good = ("该系统相比现有工作流调度器显著降低端到端完成时间，且不需要人工干预 [1]。")
    bad = good + " 同时它基于全新的 GraphFormer 架构，在 NVIDIA H100 集群上训练了 900M 样本 [1]。"
    lines = []
    for tag, ans in (("good", good), ("bad(注入GraphFormer/900M编造)", bad)):
        r = check(q, ans, entries, use_llm=True)
        lines.append(f"{tag}: ok={r['ok']} high={r['high']} issues={len(r['issues'])}")
        for i in r["issues"]:
            lines.append(f"  [{i['sev']}] {i['type']}: {i['detail'][:130]}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
