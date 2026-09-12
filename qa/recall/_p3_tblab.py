"""表专项抽取：**相关性过滤** vs **宁多勿少全列**，在全部含表的题上对照。

背景：单例上"相关性过滤"返回空（模型判定表里没有直接答案），但正文那一轮的覆盖极不稳定
（同一调用两次：0 条 / 31 条表事实）。这里量化两种表专项策略的空手率与出数率。
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
os.environ["PAPERPILOT_QASPER_TABLES"] = "1"
os.environ["PAPERPILOT_EXT_QUOTA"] = "2"

from paperpilot.agents.embedder import ChunkIndex  # noqa: E402
from paperpilot.agents.nodes.answer import (_fmt_chunk_entry,  # noqa: E402
                                            _is_table_entry, _normalize_facts, _salvage_facts)
from paperpilot.agents.nodes.pull_chunk import _section_tail  # noqa: E402
from paperpilot.qasper_source import load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

SYS_FILTER = ("你是论文表格提取器。给你编号表格条目（markdown，首行为列名，复合列名用 / 表示层级）"
              "与一个问题。\n任务：针对问题提取**相关单元格的数值**，每条写成「列路径｜行标签：数值」。\n"
              "规则：必须给具体数字；同一列不同行分别列出；无关的表/行列可跳过；"
              "若确实没有相关数值，facts 输出空数组。\n"
              '输出 JSON：{"facts": [{"n": 13, "text": "列路径｜行标签：数值"}]}，只输出 JSON。')
SYS_DUMP = ("你是论文表格提取器。给你编号表格条目（markdown，首行为列名，复合列名用 / 表示层级）"
            "与一个问题。\n任务：把表里**所有数值型单元格**按「列路径｜行标签：数值」逐条列出，"
            "**宁多勿少、不做相关性筛选**（筛选留给后续作答步骤）。\n"
            "规则：必须给具体数字；同一列不同行的数值分别列出；行数很多时最多列 40 条。\n"
            '输出 JSON：{"facts": [{"n": 13, "text": "列路径｜行标签：数值"}]}，只输出 JSON。')

llm._load_dotenv(str(ROOT))
papers = load_papers()
qids = [x["qid"] for x in json.loads(
    Path("qa/recall/qasper_tbl_exp_20260911.json").read_text(encoding="utf-8"))]
NUM = re.compile(r"\d")


def facts_of(sys_prompt: str, question: str, body: str):
    try:
        raw = llm.chat_text(sys_prompt, f"用户问题：{question}\n\n{body}", temperature=0.0)
    except Exception:  # noqa: BLE001
        return None
    try:
        obj = llm._parse_json(raw)
    except Exception:  # noqa: BLE001
        return _salvage_facts(raw)
    return _normalize_facts(obj.get("facts") if isinstance(obj, dict) else obj)


rows = []
for qid in qids:
    rec = next((x for x in papers.items() if any(
        str(qq.get("question_id") or "") == qid for qq in x[1]["qas"])), None)
    if rec is None:
        continue
    pid, paper = rec
    q = next(qq for qq in paper["qas"] if str(qq.get("question_id") or "") == qid)
    hits = ChunkIndex(f"qasper_{pid}.qpdf").search_hybrid(q["question"], top_k=12)
    entries = [{"chunk_id": h.get("chunk_id", ""), "page": h.get("page", 0),
                "section": _section_tail(list(h.get("title_path") or [])),
                "text": h.get("text") or ""} for h in hits]
    pairs = [(i, e) for i, e in enumerate(entries, 1) if _is_table_entry(e)]
    if not pairs:
        continue
    body = "\n\n".join(_fmt_chunk_entry(n, e) for n, e in pairs)
    a = facts_of(SYS_FILTER, q["question"], body) or []
    b = facts_of(SYS_DUMP, q["question"], body) or []
    rows.append((qid[:8], pid, len(pairs), len(a), len(b),
                 sum(1 for f in a if NUM.search(f["text"])),
                 sum(1 for f in b if NUM.search(f["text"]))))
    print(f"  {qid[:8]} {pid:<14} 表{len(pairs)} | 过滤 {len(a):>3} 条(含数字 {rows[-1][5]:>2}) "
          f"| 全列 {len(b):>3} 条(含数字 {rows[-1][6]:>2})", flush=True)

n = len(rows)
print(f"\n含表题的题数：{n}")
print(f"过滤式**空手**（0 条）：{sum(1 for r in rows if r[3]==0)}/{n}")
print(f"全列式**空手**（0 条）：{sum(1 for r in rows if r[4]==0)}/{n}")
print(f"过滤式含数字条数中位：{sorted(r[5] for r in rows)[n//2] if n else 0}")
print(f"全列式含数字条数中位：{sorted(r[6] for r in rows)[n//2] if n else 0}")
Path("qa/recall/p3_tblab_20260912.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
