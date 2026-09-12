"""模型 A/B：unsupported 体检 glm-4-flash(JUDGE) vs deepseek-chat(LLM=供应商 v4-flash)。

样本：negqa 20 判断题好答案(good) + 6 条注入编造(bad)。直接调体检模板（不豁免、不看判断题门），
统计 unsupported 命中：good 上误报越低越好，bad 上召回越高越好。
注意：deepseek 与生成主链路同源（自评偏宽风险），本实验只诊断"是否模型问题"，不直接作默认依据。
"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.components import validator as V  # noqa: E402
from paperpilot.components.retriever import ChunkRetriever  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

OUT = Path("qa/recall/_valid_model_ab.txt")
PREFIX = {"glm4flash(JUDGE)": "PAPERPILOT_JUDGE", "deepseek(LLM)": "PAPERPILOT_LLM"}


def run_body(question: str, answer: str, entries, prefix: str) -> list[dict]:
    evidence = "\n".join(f"[{i+1}] {V._entry_text(e)[:600]}" for i, e in enumerate(entries[:10]))
    user = V._VALIDATE_TPL.format(question=question, answer=answer[:2000], evidence=evidence)
    try:
        raw = llm.chat_json(V._SYS_VALIDATE, user, temperature=0.0, prefix=prefix)
    except Exception:
        return []
    return raw.get("unsupported") or [] if isinstance(raw, dict) else []


def main() -> int:
    llm._load_dotenv(str(ROOT))
    try:
        llm.chat_text("连通", "OK", temperature=0.0, max_tokens=4)
    except Exception:
        pass
    neg = json.load(open("qa/negqa/neg_run_20260907_195322.json", encoding="utf-8"))
    lines = []
    counts = {k: {"good_mis": 0, "bad_catch": 0, "good_n": 0, "bad_n": 0} for k in PREFIX}
    detail = {k: [] for k in PREFIX}
    for i, r in enumerate(neg):
        pdf, q, ans = r["pdf"], r["question"], r["answer"]
        hits = ChunkRetriever(pdf).search(q, top_k=12)
        entries = [{"kind": "chunk", "text": h.get("text", ""), "page": h.get("page", 0)} for h in hits]
        bad = ans + "\n(The system is also built on a GraphFormer-like architecture that outperforms every baseline by a large margin on all benchmarks.)"
        for name, pfx in PREFIX.items():
            g = len(run_body(q, ans, entries, pfx))
            b = len(run_body(q, bad, entries, pfx))
            counts[name]["good_n"] += 1
            counts[name]["bad_n"] += 1
            counts[name]["good_mis"] += 1 if g > 0 else 0
            counts[name]["bad_catch"] += 1 if b > 0 else 0
            if g:
                detail[name].append(f"  good-mis {r['qid']}: {g} 句")
    for name in PREFIX:
        c = counts[name]
        lines.append(f"{name}: good误报 {c['good_mis']}/{c['good_n']}  | bad召回(抓到unsupported) {c['bad_catch']}/{c['bad_n']}")
        lines += detail[name]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
