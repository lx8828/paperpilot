"""P3 验证：打印**改造后**实际喂给 LLM 的用户消息，并检查 facts 是否真去读表。

要点检查：
  1) 问题是否在**最前面**（改为前置）；
  2) 表块是否标成【表格】、正文是否仍是"正文段落"；
  3) 是否出现一次性"表格阅读说明"；
  4) facts 清单的锚点是否**落在表块条目上**，且是否含表里的原样数字（如 0.631/0.302）。
"""
import os
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
os.environ["PAPERPILOT_QASPER_TABLES"] = "1"
os.environ["PAPERPILOT_EXT_QUOTA"] = "2"

from paperpilot.agents.embedder import ChunkIndex  # noqa: E402
from paperpilot.agents.nodes.answer import (_extract_facts,  # noqa: E402
                                            _extract_table_facts, _fmt_chunk_entry,
                                            _fmt_facts_block, _has_table, _is_table_entry)
from paperpilot.agents.nodes.pull_chunk import _section_tail  # noqa: E402
from paperpilot.qasper_source import load_papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

PID, PREF = "2002.05058", "75b69eef"
llm._load_dotenv(str(ROOT))
papers = load_papers()
q = next(qq for qq in papers[PID]["qas"]
         if str(qq.get("question_id") or "").startswith(PREF))
idx = ChunkIndex(f"qasper_{PID}.qpdf")
hits = idx.search_hybrid(q["question"], top_k=12)
entries = [{"chunk_id": h.get("chunk_id", ""), "page": h.get("page", 0),
            "section": _section_tail(list(h.get("title_path") or [])),
            "text": h.get("text") or ""} for h in hits]

header = f"标题：{papers[PID]['title']}\n\n=== 全文检索到的正文（独立检索） ==="
context = "\n\n".join([header] + [_fmt_chunk_entry(i, e) for i, e in enumerate(entries, 1)])

print("=== 条目类型标记（应只有表块是【表格】）===")
for i, e in enumerate(entries, 1):
    print("  ", _fmt_chunk_entry(i, e)[:76].replace("\n", " "))

print(f"\n_has_table(entries) = {_has_table(entries)}")
print("\n=== 用户消息开头（问题前置 + 表格阅读说明）===")
head = f"用户问题：{q['question']}\n"
if _has_table(entries):
    head += ("\n本批证据含【表格】：表格为 markdown，**第一行是列名**，"
             "复合列名用 / 表示层级（如 Story Generation/Pearson）；"
             "核对数值时请先定位到正确的列，再横向比对同一列里各行的数值。\n")
print(head)

print("=== facts 装配：复现 _compose 的新逻辑（含条件触发的表专项）===")
tbl_pairs = [(i, e) for i, e in enumerate(entries, 1) if _is_table_entry(e)]
tbl_idx = {i for i, _ in tbl_pairs}
KEY = ("0.631", "0.302", "0.783", "0.553")
for trial in (1, 2):
    prose = _extract_facts(q["question"], context)
    anchors = {f["n"] for f in prose}
    triggered = bool(tbl_pairs) and not (anchors & tbl_idx)
    extra = _extract_table_facts(q["question"], tbl_pairs)[:40] if triggered else []
    facts = prose + extra
    has_key = sum(1 for f in facts if any(k in f["text"] for k in KEY))
    print(f"  第{trial}次：通读式 {len(prose)} 条（表锚点 {len(anchors & tbl_idx)} 条）"
          f" → 表专项{'触发' if triggered else '不触发'}（{len(extra)} 条）"
          f" | 合并 {len(facts)} 条 | 含关键数值 {has_key} 条")
    for f in [x for x in facts if any(k in x["text"] for k in KEY)][:3]:
        print(f"      [{f['n']}] {f['text'][:120]}")

print("\n=== 消息长度 ===")
print(f"  context {len(context)} 字 | facts 块 {len(_fmt_facts_block(facts))} 字 "
      f"| 表块 {len(tbl_idx)} 个")
