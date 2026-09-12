"""诊断：目标表已进上下文、但端到端仍失败的题，卡在哪一环？

对每道题输出：金标准 / 目标表块实际内容 / 两臂答案与裁判理由 / 是否引用了目标表块。
再给一个汇总：表在上下文里时，答案**引用它**的比例。
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
import os  # noqa: E402

os.environ["PAPERPILOT_QASPER_TABLES"] = "1"

from paperpilot.agents.document_cache import retrieval_chunks  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402

papers = load_papers()

R = Path("qa/recall")
q0 = {x["qid"]: x for x in json.loads((R / "union_ab_q0_20260911.json").read_text(encoding="utf-8"))}
q2 = {x["qid"]: x for x in json.loads((R / "union_ab_q2_20260911.json").read_text(encoding="utf-8"))}
exp = json.loads((R / "qasper_tbl_exp_20260911.json").read_text(encoding="utf-8"))
ev = json.loads((R / "union_prod_eval_20260911.json").read_text(encoding="utf-8"))


def ck(t: str) -> str:
    m = re.match(r"\s*(table|figure)\s*([A-Za-z]?\d+)", str(t), re.I)
    return f"{m.group(1).lower()}{m.group(2).lower()}" if m else ""


def cites_ids(rec):
    out = []
    for c in rec.get("cites") or []:
        ref = str(c.get("ref") or "")
        out.append(ref.split("#")[-1] if "#" in ref else ref)
    return out


L = ["# 为什么「表进了上下文却答不出」— 5 题逐条诊断（2026-09-12）", ""]
focus = [("9c4a4dfa", "1910.02339"), ("75b69eef", "2002.05058"),
         ("579941de", "1909.13375"), ("0fd678d2", "1804.08139"),
         ("c4c9c790", "2002.08307")]
def find(d, pref):
    """记录里的 qid 是完整 id，这里按前缀查。"""
    for k, v in d.items():
        if k.startswith(pref):
            return v
    return {}


for qid, pid in focus:
    a, b = find(q0, qid), find(q2, qid)
    chs = {c.chunk_id: c.text for c in retrieval_chunks(f"qasper_{pid}.qpdf")
           if str(c.chunk_id).startswith("xtbl")}
    # 目标表编号：必须用 **gold 证据**（含 "Table N"），而不是 gold 答案文本
    gold = (b or a).get("gold") or ""
    qobj = next((qq for qq in papers.get(pid, {}).get("qas") or []
                 if str(qq.get("question_id") or "").startswith(qid[:8])), None)
    want = set()
    if qobj is not None:
        _g, evs = gold_answer_full(qobj)
        for e in evs:
            for x in re.findall(r"(?:Table|Figure)\s*([A-Za-z]?\d+)", str(e), re.I):
                want.add(f"table{x}".lower())
    tgt = {cid for cid, t in chs.items() if ck(t) in want} if want else set()
    ids2 = list(b.get("entry_ids") or [])
    pos = next((i + 1 for i, c in enumerate(ids2) if c in tgt), None) if tgt else None
    L += [f"## {qid}（{pid}）score {a.get('score')} → {b.get('score')}", "",
          f"- **问题**：{(b or a).get('question')}",
          f"- **金标准**：{gold[:300].replace(chr(10), ' ')}",
          f"- **目标表块**：{sorted(tgt) or '（该篇未产出/未匹配到目标表）'} "
          f"| 在 q2 上下文里的位次：{pos or '不在候选'}（共 {len(ids2)} 块）"]
    for cid in sorted(tgt):
        L += [f"- 表块 `{cid}` 内容（前 700 字）：", "", "```", chs[cid][:700], "```"]
    for lab, r in (("q0（现状）", a), ("q2（并集）", b)):
        ci = cites_ids(r)
        hit = [c for c in ci if c in chs]
        L += [f"- **{lab}**：score={r.get('score')} cites={ci or '无'}"
              f"（其中表块：{hit or '无'}）",
              f"  - 答案：{(r.get('answer') or '')[:400].replace(chr(10), ' ')}",
              f"  - 裁判：{(r.get('reason') or '')[:300]}"]
    L += [""]

# 汇总：表在上下文里时，答案引用它的比例
in_ctx = cited = 0
for qid, r in q2.items():
    ids = list(r.get("entry_ids") or [])
    extc = [c for c in ids if str(c).startswith("xtbl")]
    if not extc:
        continue
    in_ctx += 1
    if any(c in extc for c in cites_ids(r)):
        cited += 1
L += ["## 汇总", "",
      f"- q2 臂里**上下文含表块**的题：{in_ctx}/{len(q2)}",
      f"- 其中答案**引用了表块**的：{cited}/{in_ctx} = {(cited/in_ctx if in_ctx else 0):.0%}"]

L += ["", "## 结论（2026-09-12，逐条人工核对后）", "",
      "「召回变好但回答没变好」不是一个原因，是**五个独立障碍**：", "",
      "1. **模型不会从表里读数值做对比（真瓶颈，铁证）** —— `75b69eef`：correlation 表含 gold 的 "
      "`0.631 vs 0.302`，它就在上下文中、**还被引用了** `xtbl-77`，模型仍写「**并未给出具体的提升数值**」。",
      "   同类措辞在 `579941de`（「证据中未给出具体数字」）、`c4c9c790`（「无法确认具体数值」）、"
      "`0fd678d2`（「没有明确说明」）反复出现。",
      "2. **误拒** —— `0fd678d2`：gold =「Accuracy on each dataset and the average accuracy」，"
      "而含 accuracy 的表（`xtbl-86`，16 tasks 性能表）**没进候选**、正文又没明说 → 系统答「论文没提评价指标」。",
      f"3. **表没被用** —— 汇总：**表块在上下文里时只有 {cited}/{in_ctx} = "
      f"{(cited/in_ctx if in_ctx else 0):.0%} 的答案引用了表块**，近一半白给。",
      "4. **评测天花板** —— `9c4a4dfa`：问题问 **speed**，gold 给 **accuracy**（表里也只有 accuracy）",
      "   → 系统诚实答「没有速度数据」反被判 3 分。**这题召回再好也答不对**，属 QASPER 标注与问题不匹配。",
      "5. **MinerU 表体质量** —— `5a0841cc`(1701.00185) 的 Table 6 出现 "
      "`| bi-LSTM (last)bi-LSTM (mean)` 这种**行结构错乱**；",
      "   另 `579941de` 确证 1 例**错表**（gold 证据指「Table 2. Performance ... EM and F1」，"
      "而该篇 MinerU 的 `xtbl-78` 是「Table 2: Example failure cases」→ 目标性能表根本没被产出）。", "",
      "**自我纠错**：本轮一度用「表块是否含 gold 数字」估出假阳性 47%，**该指标不可用** ——",
      "它把「gold 是聚合/推导值」（如 gold `over 104k documents` vs 表里 `83,077`）误判为「表错了」；",
      "抽查 2 例均是对的表。真错表**至少 1 例**（579941de），**不是**系统性问题。", "",
      "**指向**：瓶颈已从「召回」转移到「**读表**」——①表内容解析质量、②模型从表取数做对比的能力、"
      "③误拒倾向。与上下文压缩/重排无关。"]
out = Path("qa/recall/WHY_TABLE_FAILS_20260912.md")
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print("已写", out, f"({len(L)} 行)")
