"""Validator.check 冒烟：好答案应少误报；注入编造句/编数应被抓。"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.components.validator import check  # noqa: E402
from paperpilot.components.retriever import ChunkRetriever  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

OUT = Path("qa/recall/_valid_smoke.txt")


def main() -> int:
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    neg = json.load(open("qa/negqa/neg_run_20260907_195322.json", encoding="utf-8"))
    lines = []
    cases = [r for r in neg if r.get("type") == "contra"][:2]
    for r in cases:
        pdf = r["pdf"]
        q = r["question"]
        ans = r["answer"]
        hits = ChunkRetriever(pdf).search(q, top_k=12)
        entries = [{"kind": "chunk", "text": h.get("text", ""), "page": h.get("page", 0)}
                   for h in hits]
        good = check(q, ans, entries)
        lines.append("=" * 70)
        lines.append(f"Q: {q[:120]}")
        lines.append(f"good answer({r['verdict']}): {ans[:160]}…")
        lines.append(f"  check ok={good['ok']} high={good['high']} issues={len(good['issues'])}")
        for i in good["issues"]:
            lines.append(f"    [{i['sev']}] {i['type']}: {i['detail'][:120]}")
        bad_ans = ans + "\nThe model also achieved a 99.9% accuracy on the Gemini benchmark and surpassed human experts."
        bad = check(q, bad_ans, entries)
        lines.append(f"bad answer(注入编造句): {bad_ans[-90:]}")
        lines.append(f"  check ok={bad['ok']} high={bad['high']} issues={len(bad['issues'])}")
        for i in bad["issues"][:6]:
            lines.append(f"    [{i['sev']}] {i['type']}: {i['detail'][:120]} {('| ' + i['sentence'][:80]) if i['sentence'] else ''}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
