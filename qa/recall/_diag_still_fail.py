"""诊断：补了表格通道后**仍然 ≤3 分的 22 题**到底卡在哪一段？

对每题判三段（离线，零 LLM）：
  S1 表解析：QASPER 标注的 evidence 若是 `FLOAT SELECTED: Table N ...`，
            该表在 MinerU 产物里**解析出来了吗**？（按 caption 匹配）
  S2 表检索：该表对应的 `xtbl-*` 块**进 top-12 了吗**？（与线上同款 search_hybrid）
  S3 答采用：答案里**有没有给出 gold 的数字**？（派生数看操作数是否出现）

分桶：
  A 表未解析        → MinerU 覆盖问题
  B 表解析了没进候选 → 检索/排序问题
  C 进候选了答案没用 → answer 层遗漏
  D 答案给了数仍≤3   → gold 要更多值 / 裁判口径 / 部分正确
  E 无表格 evidence  → 本题其实不是表格题（A 桶启发式误判）

用法：uv run python qa/recall/_diag_still_fail.py
"""
from __future__ import annotations

import collections
import io
import json
import os
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
os.environ["PAPERPILOT_QASPER_TABLES"] = "1"

from paperpilot.agents.embedder import ChunkIndex  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402
from paperpilot.tools.mineru_bridge import _find_content_list, element_text  # noqa: E402

NUM = re.compile(r"\d+(?:[.,]\d+)*")
TBL = Path("assets/artifacts/out_mineru")


def nums(text: str) -> set[str]:
    return {t.replace(",", "") for t in NUM.findall(text or "") if len(t.replace(",", "")) >= 2}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).strip()


def main() -> int:
    exp = json.loads(Path("qa/recall/qasper_tbl_exp_20260911.json").read_text(encoding="utf-8"))
    bad = [x for x in exp if not x["pass"]]
    papers = load_papers()
    print(f"诊断 {len(bad)} 道仍 ≤3 的题")

    detail, bucket = [], collections.Counter()
    for x in bad:
        pid, qid = x["pid"], x["qid"]
        paper = papers.get(pid) or {}
        q = next((qq for qq in paper.get("qas") or []
                  if str(qq.get("question_id") or "") == qid), None)
        if q is None:
            continue
        gold, evs = gold_answer_full(q)
        g_nums = nums(gold)
        ans_nums = nums(x["answer"])

        # QASPER 标注里指向的表（caption）
        cap_targets = [re.sub(r"^FLOAT SELECTED:\s*", "", str(e), flags=re.I)
                       for e in evs if str(e).upper().startswith("FLOAT")]
        cl = _find_content_list(TBL / f"{pid}v1") or []
        mn_tables = [(e, element_text(e) or "") for e in cl if e.get("type") == "table"]

        # S1：目标表是否被解析（按 caption 的 "table N" 号匹配）
        def cap_key(s: str) -> str:
            m = re.match(r"\s*(table|figure)\s*([A-Za-z]?\d+)", str(s), re.I)
            return f"{m.group(1).lower()}{m.group(2).lower()}" if m else ""

        want = {cap_key(c) for c in cap_targets if cap_key(c)}
        have = {cap_key(" ".join(e.get("table_caption") or [])) for e, _ in mn_tables}
        s1 = (not cap_targets) or bool(want & have)
        parsed_txt = "\n".join(t for _, t in mn_tables)
        gold_in_parsed = len(g_nums & nums(parsed_txt)) / max(len(g_nums), 1) if g_nums else None

        # S2：目标表的 xtbl 块是否进 top-12
        hits = ChunkIndex(f"qasper_{pid}.qpdf").search_hybrid(q["question"], top_k=12)
        hit_ids = [h.get("chunk_id", "") for h in hits]
        hit_txt = "\n".join(str(h.get("text") or "") for h in hits)
        xtbl_in_hits = [i for i in hit_ids if str(i).startswith("xtbl")]
        # 目标表的文本片段是否出现在候选里
        s2 = True
        if want:
            s2 = any(cap_key(" ".join(e.get("table_caption") or [])) in want
                     and (element_text(e) or "")[:120] in hit_txt for e, _ in mn_tables)

        # S3：答案里有没有 gold 的数
        s3 = bool(g_nums & ans_nums) if g_nums else None

        if not cap_targets:
            b = "E 非表格题（启发式误判）"
        elif not s1:
            b = "A 表未解析"
        elif not s2:
            b = "B 表没进 top-12"
        elif not s3:
            b = "C 进了候选答案没用"
        else:
            b = "D 给了数仍 ≤3"
        bucket[b] += 1
        detail.append({"qid": qid, "pid": pid, "score": x["score"], "bucket": b,
                       "cap_targets": cap_targets[:2], "n_mn_tables": len(mn_tables),
                       "want": sorted(want), "have_in_parsed": bool(want & have),
                       "gold_in_parsed": gold_in_parsed, "xtbl_in_hits": len(xtbl_in_hits),
                       "gold_n": len(g_nums), "hit_n": len(g_nums & ans_nums),
                       "question": q["question"][:80]})

    print("\n== 分桶 ==")
    for k, v in bucket.most_common():
        print(f"  {k:<22} {v}")
    print("\n== 逐题 ==")
    for d in sorted(detail, key=lambda z: z["bucket"]):
        print(f"  [{d['bucket'][:14]:<14}] {d['qid'][:8]} {d['pid']} score={d['score']} "
              f"解析表={d['n_mn_tables']} 目标表命中={d['have_in_parsed']} "
              f"gold在解析表={d['gold_in_parsed']} xtbl进候选={d['xtbl_in_hits']} "
              f"gold数 {d['hit_n']}/{d['gold_n']}")
        print(f"       Q: {d['question']}")
        if d["cap_targets"]:
            print(f"       目标表: {d['cap_targets'][0][:70]}")
    Path("qa/recall/still_fail_diag_20260911.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n已写 qa/recall/still_fail_diag_20260911.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
