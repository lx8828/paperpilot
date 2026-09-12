"""算清两套口径各自的**分母**（旧口径定位不到的题不该算进旧口径的分母）。

只跑 locate（纯文本正则，无 embedding），再与 union_prod_eval_v2 的布尔结果合并。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "qa" / "recall"))
os.environ["PAPERPILOT_QASPER_TABLES"] = "1"

import _tgt  # noqa: E402
from paperpilot.agents.document_cache import retrieval_chunks  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402


def main() -> int:
    papers = load_papers()
    exp = json.loads(Path("qa/recall/qasper_tbl_exp_20260911.json").read_text(encoding="utf-8"))
    d = json.loads(Path("qa/recall/union_prod_eval_v2_20260912.json").read_text(encoding="utf-8"))
    hits = {r["qid"]: r["hits"] for r in d["detail"]}
    rows = []
    for x in exp:
        pid, qid = x["pid"], x["qid"]
        q = next((qq for qq in papers[pid]["qas"]
                  if str(qq.get("question_id") or "") == qid), None)
        if q is None:
            continue
        gold, evs = gold_answer_full(q)
        cs = retrieval_chunks(f"qasper_{pid}.qpdf")
        tn, _ = _tgt.locate(cs, evs, gold, mode="num")
        to, _ = _tgt.locate(cs, evs, gold, mode="or")
        if qid not in hits:
            continue                        # 两口径都定位不到 → 全局弃题
        rows.append((qid, bool(tn), bool(to)))
    n_num = sum(1 for _, a, _ in rows if a)
    n_or = sum(1 for _, _, b in rows if b)
    print(f"参与评估 {len(rows)} 题 | 旧口径可定位 {n_num} | 新口径可定位 {n_or}")
    for mode, key, n in (("num", "num", n_num), ("or", "or", n_or)):
        print(f"\nmode={mode}（分母 {n}）")
        for k in ("0", "2", "3"):
            h = sum(1 for qid, a, b in rows
                    if (a if mode == "num" else b) and hits[qid][k][mode])
            print(f"   quota={k} -> {h}/{n} = {h/max(n,1):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
