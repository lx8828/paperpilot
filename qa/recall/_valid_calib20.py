"""Validator 误报校准：negqa 20 道好答案跑 check（机器+LLM 体检）。

期望：好答案 high≈0（误报会让 gate 把好答案变兜底 = 灾难）。
近似：neg_run 无 cites 存档，evidence 用 ChunkRetriever top12 重建（检索确定，编号近似一致）。
"""
from __future__ import annotations
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.components.retriever import ChunkRetriever  # noqa: E402
from paperpilot.components.validator import check  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

OUT = Path("qa/recall/_valid_calib20.txt")


def main() -> int:
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    neg = json.load(open("qa/negqa/neg_run_20260907_195322.json", encoding="utf-8"))
    rows = []
    n_high = 0
    type_cnt: Counter = Counter()
    lines = []
    for r in neg:
        pdf, q, ans = r["pdf"], r["question"], r["answer"]
        hits = ChunkRetriever(pdf).search(q, top_k=12)
        entries = [{"kind": "chunk", "text": h.get("text", ""), "page": h.get("page", 0)}
                   for h in hits]
        res = check(q, ans, entries, use_llm=True)
        # citation 越界是本脚本"检索近似重建 entries"的假象（真实 gate 用 cites 不会越界），不计入
        highs = [i for i in res["issues"] if i["sev"] == "high" and i["type"] != "citation"]
        mids = [i for i in res["issues"] if i["sev"] == "mid"]
        n_high += len(highs)
        for i in res["issues"]:
            if i["type"] != "citation":
                type_cnt[i["type"]] += 1
        rows.append({"qid": r["qid"], "type": r["type"], "n_issues": len(res["issues"]),
                     "n_high": len(highs), "n_mid": len(mids)})
        if res["issues"]:
            lines.append("-" * 60)
            lines.append(f"{r['qid']} [{r['type']}] issues={len(res['issues'])}")
            for i in res["issues"][:4]:
                lines.append(f"  [{i['sev']}] {i['type']}: {i['detail'][:110]}"
                             + (f" | {i['sentence'][:60]}" if i.get('sentence') else ""))
    lines.insert(0, f"negqa {len(rows)} 道好答案：high 总数 {n_high} | issue type 分布 {dict(type_cnt)}")
    lines.insert(1, "有 issue 的题（逐条）：")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("written", len(rows), "high", n_high)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
