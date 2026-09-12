"""两步法第一步（_extract_facts）失效规模与根因探针。

对 ab_rewrite off 臂题集：用 search_l3 复现同一份 context，再调一次
llm.chat_text(SYSTEM_EXTRACT, 同一 user) 抓**原始返回**，把结果分三类：
  EMPTY   raw ≤ 30 字符（模型真的返回空清单，如 {"facts": []}）
  PARSE   raw 有内容但 _parse_json 失败 → **事实被丢弃（真 bug）**
  OK      正常解析

用法：
    uv run python qa/recall/_facts_probe.py [N]      # 默认 40
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.nodes import answer as A  # noqa: E402
from paperpilot.agents.nodes.pull_chunk import search_l3  # noqa: E402
from paperpilot.qasper_source import load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    llm._load_dotenv(str(ROOT))
    papers = load_papers()
    qmap = {}
    for pid, p in papers.items():
        for q in p.get("qas") or []:
            qmap[str(q.get("question_id") or "")] = (pid, q)

    recs = json.load(open("qa/recall/ab_rewrite_result.json", encoding="utf-8"))
    sel = recs[:n]

    L = ["# 两步法第一步失效规模（_extract_facts）", "",
         f"> 样本 {len(sel)} 题（ab_rewrite off 臂前 N 题）｜每题 1 次 LLM", "",
         "| qid | 组 | ctx字符 | parsed facts | raw字符 | 判定 | 备注 |",
         "|---|---|---|---|---|---|---|"]
    cnt = {"EMPTY": 0, "PARSE": 0, "OK": 0, "ERR": 0}
    for i, r in enumerate(sel, 1):
        qid = str(r["qid"])
        got = qmap.get(qid)
        if not got:
            cnt["ERR"] += 1
            continue
        pid, q = got
        st = search_l3({"question": q["question"], "pdf": f"qasper_{pid}.qpdf"})
        chunks = list(st.get("l3_chunks") or [])
        parts = ["标题：\n\n=== 全文检索到的正文（独立检索） ==="]
        for j, c in enumerate(chunks, 1):
            parts.append(A._fmt_chunk_entry(j, c))
        context = "\n\n".join(parts)
        facts = A._extract_facts(q["question"], context)
        user = (f"{context}\n\n用户问题：{q['question']}\n\n"
                "请逐条通读上方全部条目，穷举提取与问题直接相关的事实，并只输出指定 JSON。")
        raw = ""
        try:
            raw = llm.chat_text(A.SYSTEM_EXTRACT, user, temperature=0.0)
        except Exception as e:  # noqa: BLE001
            cnt["ERR"] += 1
            L.append(f"| {qid[:8]} | {r['grp']} | {len(context)} | — | — | ERR | "
                     f"{type(e).__name__} |")
            print(f"[{i}/{len(sel)}] {qid[:8]} ERR", flush=True)
            continue
        note = ""
        if len(raw.strip()) <= 30:
            verdict = "EMPTY"
            note = raw.strip()[:30]
        else:
            try:
                obj = llm._parse_json(raw)
                verdict = "OK"
                if isinstance(obj, dict):
                    fc = obj.get("facts")
                    note = f"ok facts={len(fc) if isinstance(fc, list) else '?'}"
                else:
                    note = f"ok {type(obj).__name__}"
            except Exception:  # noqa: BLE001
                verdict = "PARSE"
                note = raw.strip()[-60:].replace("|", "｜")
        cnt[verdict] += 1
        L.append(f"| {qid[:8]} | {r['grp']} | {len(context)} | {len(facts)} | {len(raw)} | "
                 f"{verdict} | {note} |")
        print(f"[{i}/{len(sel)}] {qid[:8]} ctx={len(context)} facts={len(facts)} "
              f"raw={len(raw)} {verdict}", flush=True)

    tot = max(sum(cnt.values()), 1)
    head = ["", "## 汇总", "",
            f"- 样本 {tot} 题",
            f"- **EMPTY（模型返回空清单）**：{cnt['EMPTY']} = {cnt['EMPTY']/tot:.0%}",
            f"- **PARSE（提取到了但 JSON 解析失败 → 被静默丢弃）**：{cnt['PARSE']} = "
            f"{cnt['PARSE']/tot:.0%}",
            f"- OK：{cnt['OK']} = {cnt['OK']/tot:.0%}",
            f"- ERR（调用异常）：{cnt['ERR']}",
            "",
            "> 判读：`PARSE` = **信息明明提取到了却被丢掉**（真 bug，需宽松解析兜底）；",
            "> `EMPTY` = 模型判定该上下文无相关事实（可能是检索没送到 / 题目本身无答案）。"]
    txt = "\n".join(L[:5] + head + L[5:])
    Path("qa/recall/FACTS_PROBE_20260910.md").write_text(txt + "\n", encoding="utf-8")
    print("\n".join(head))
    print("已写 qa/recall/FACTS_PROBE_20260910.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
