"""生产链路下的并集候选评估（**口径 v2**：编号 ∪ 内容/数值）。

口径为什么改、怎么改：见 `_tgt.py` 顶部。要点：
- 旧口径 = 只用「证据里的 Table N」匹配「块文本首行的 Table N」→ 上游编号错位/caption
  缺失时会**量错表**（假阳性）或把题**从分母丢掉**（假阴性）；
- 新口径 = 编号命中 **或** 表体含答案的显著数值/关键词。
本脚本**两套口径都报**（`num` 旧 / `or` 新），并打印两口径不一致的题，便于判断
"并集收益"里有多少是口径造成的。

以生产路径为准（`ChunkIndex.search_hybrid` + 生产 ext 池：表格 + 公式），顺带预热 cvec。
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "qa" / "recall"))
os.environ["PAPERPILOT_QASPER_TABLES"] = "1"

import _tgt  # noqa: E402
from paperpilot.agents.document_cache import retrieval_chunks  # noqa: E402
from paperpilot.agents.embedder import ChunkIndex  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402

MODES = ("num", "or")
QUOTAS = [0, 2, 3]
OUT = Path(os.environ.get("PP_OUT") or "qa/recall/union_prod_eval_v2_20260912.json")
TOPK = int(os.environ.get("PP_TOPK", "12"))          # 命中判定**必须**用生产的 12
# 位次（灵敏度指标）单独用**更深的候选**测，且只在 quota=0 下测。
# ⚠️ 不要把 TOPK 调大来测位次：`_select` 在 quota>0 时是"正文 top_k + 追加表块"，
#    top_k=30 会把追加的表块挤到 31~32 位 → 并集效应被自己的测量方式吃掉（已踩）。
RANK_TOPK = int(os.environ.get("PP_RANK_TOPK", "30"))


def main() -> int:
    papers = load_papers()
    exp = json.loads(Path("qa/recall/qasper_tbl_exp_20260911.json").read_text(encoding="utf-8"))
    res = {m: {k: [] for k in QUOTAS} for m in MODES}
    detail: list[dict] = []
    t0 = time.time()
    n_paper = 0
    n_by_num = n_by_cnt = n_onlycnt = 0
    for i, x in enumerate(exp, 1):
        pid, qid = x["pid"], x["qid"]
        q = next((qq for qq in papers[pid]["qas"]
                  if str(qq.get("question_id") or "") == qid), None)
        if q is None:
            continue
        gold, evs = gold_answer_full(q)
        chunk_list = retrieval_chunks(f"qasper_{pid}.qpdf")
        tgts: dict[str, set[str]] = {}
        diags: dict[str, dict] = {}
        for mode in MODES:
            tgts[mode], diags[mode] = _tgt.locate(chunk_list, evs, gold, mode=mode)
        if not tgts["or"] and not tgts["num"]:
            continue                      # 该篇没产出任何候选目标 → 与检索无关，弃题
        n_paper += 1
        if diags["or"]["by_num"]:
            n_by_num += 1
        if diags["or"]["by_cnt"]:
            n_by_cnt += 1
        if diags["or"]["by_cnt"] and not diags["or"]["by_num"]:
            n_onlycnt += 1
        idx = ChunkIndex(f"qasper_{pid}.qpdf")
        row = {"qid": qid, "pid": pid, "keys": sorted(diags["or"]["keys"]),
               "nums": sorted(diags["or"]["nums"]), "hits": {}, "rank": None}
        for k in QUOTAS:
            os.environ["PAPERPILOT_EXT_QUOTA"] = str(k)
            ids = [h["chunk_id"] for h in idx.search_hybrid(q["question"], top_k=TOPK)]
            got = set(ids)
            for mode in MODES:
                res[mode][k].append(bool(got & tgts[mode]) if tgts[mode] else False)
            row["hits"][str(k)] = {m: bool(got & tgts[m]) if tgts[m] else False for m in MODES}
        # 位次：quota=0（纯混池）下用深候选看目标块的名次——比"是否进 top-12"敏感得多
        os.environ["PAPERPILOT_EXT_QUOTA"] = "0"
        deep = [h["chunk_id"] for h in idx.search_hybrid(q["question"], top_k=RANK_TOPK)]
        row["rank"] = next((j + 1 for j, cid in enumerate(deep) if cid in tgts["or"]), None)
        detail.append(row)
        if i % 6 == 0:
            print(f"  [{i}/{len(exp)}] t={time.time()-t0:.0f}s", flush=True)

    print(f"\n涉及 {n_paper} 题 | 耗时 {time.time()-t0:.0f}s")
    print(f"定位来源（新口径）：编号命中 {n_by_num} 题 | 内容命中 {n_by_cnt} 题 "
          f"| **仅靠内容救回** {n_onlycnt} 题")
    print()
    for mode, tag in (("num", "旧口径（只按编号）"), ("or", "新口径（编号 ∪ 内容）")):
        n = len(res[mode][0])
        print(f"### {tag}：n = {n}")
        print("| 候选方案 | 目标表进候选 |")
        print("|---|---|")
        for k in QUOTAS:
            h = sum(res[mode][k])
            lab = "（现状）" if k == 0 else f"（并集 +{k} 表块）"
            print(f"| quota={k}{lab} | {h}/{n} = {h/n:.0%} |")
        print()
    # 目标块位次（quota=0 混池，深候选；新口径目标集合里最早出现的那个）
    import statistics
    rr = [d["rank"] or (RANK_TOPK + 1) for d in detail]
    print(f"### 目标表位次（quota=0 混池，PP_RANK_TOPK={RANK_TOPK}，未进记 {RANK_TOPK + 1}）")
    print(f"  位次中位 {statistics.median(rr):.1f} | 进 top-12 "
          f"{sum(1 for x in rr if x <= 12)}/{len(rr)}")
    print()
    diff = [d for d in detail if d["hits"]["0"]["num"] != d["hits"]["0"]["or"]]
    if diff:
        print(f"### 两口径在 quota=0 上判定不同的题（{len(diff)} 道）")
        for d in diff:
            print(f"  {d['qid'][:8]} {d['pid']} keys={d['keys']} nums={d['nums'][:3]} "
                  f"num={d['hits']['0']['num']} or={d['hits']['0']['or']}")
    OUT.write_text(json.dumps({"res": {m: {str(k): res[m][k] for k in QUOTAS} for m in MODES},
                               "detail": detail}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
