"""「问题都用英文问」到底能解决多少？—— 按**词法路是否开火**分层量化。

背景（用户 2026-09-26 提问）：
  我此前说过"中文提问 → `_bm_or_none` 判定 BM25 无信号 → 词法路必死 → 调权重收益=0"。
  这句**过强**：`_bm_or_none` 的实际判据是 `BM25Index.score(q).max() > 0`，
  而中文问句常常**内嵌英文实体**（MAMuJoCo / VAE / w/o Expand / 表格数字）
  → 这些 token 在英文正文里有匹配 → 词法路**并不死**。

本脚本回答三个问题（零 LLM 增量：译文走 `_xling_cache.json`）：
  ① 120 道中文题里，词法路真正"死掉"的占多少？（量化上面那句的错）
  ② 分层后：中文原问 / 英文译式 / 并集 各漏多少？
  ③ "英文救回来"的题，是不是恰好集中在**词法死**的那层？（因果核对）

口径与 `_check_reachable --search` 一致：题命中 = 候选里 `must_all` 锚点**全中**。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "retrieval" / "scripts"))
sys.path.insert(0, str(ROOT / "retrieval" / "tmp"))
sys.path.insert(0, str(ROOT / "src"))

import _scan_quota as sq  # noqa: E402
import _xling_ab as xa  # noqa: E402
from _qa_groups import papers  # noqa: E402
from paperpilot.agents.embedder import BM25Index, _tokenize  # noqa: E402

TMP = ROOT / "retrieval" / "tmp"


def ascii_tokens(q: str) -> list[str]:
    """问题里**长度≥2 的 ASCII token**（= 唯一可能在英文正文里命中的东西）。"""
    return [t for t in _tokenize(q) if len(t) >= 2 and t.isascii()]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="group1,group2")
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()

    buckets = {  # 分层 → [题数, A0漏, A1漏, A2漏, A0救回, ...]
        "live": {"n": 0, "A0": 0, "A1": 0, "A2": 0},
        "dead": {"n": 0, "A0": 0, "A1": 0, "A2": 0},
    }
    saved_by_en: list[tuple] = []      # 英文救回的题
    lost_by_en: list[tuple] = []       # 英文弄丢的题（中文原本命中）
    details: list[tuple] = []
    n_all = 0

    for group in [g.strip() for g in args.groups.split(",") if g.strip()]:
        ps = sq.load_probes(group)
        corpus = [f"{s}.pdf" for s in papers(group)]
        idx = sq.MultiChunkIndex(corpus)
        texts = [c.text for c in idx._doc_chunks()]
        bm = BM25Index(texts)                       # 全语料 BM25（与 search_hybrid 同口径）
        print(f"\n########## {group} ｜ {len(ps)} 道需检索题 ｜ N={args.n}", flush=True)

        for i, q in enumerate(ps, 1):
            raw = q["question"]
            bm_max = float(bm.score(raw).max())     # ⚠️ 与 `embedder._bm_or_none` 同判据
            live = bm_max > 0
            en = xa.english_queries(q["qid"], raw)
            tr = str(en.get("translation") or "").strip()
            if not tr:                              # 没译文 → 英文臂无法测，只记中文
                print(f"    ⚠️ {q['qid']} 无英文译式，跳过英文臂")
                continue

            pool_zh = sq.build_pool(idx, q, TMP)
            pool_en = sq.build_pool(idx, {**q, "question": tr}, TMP)
            a0 = sq.sel_quota(pool_zh, n=args.n, floor=1)
            a1 = sq.sel_quota(pool_en, n=args.n, floor=1)
            a2 = xa.union([a0, a1], [args.n, 8])
            m0 = xa.stat(a0, q["must"], 0)[0]
            m1 = xa.stat(a1, q["must"], 0)[0]
            m2 = xa.stat(a2, q["must"], 0)[0]

            b = buckets["live" if live else "dead"]
            b["n"] += 1
            b["A0"] += m0
            b["A1"] += m1
            b["A2"] += m2
            n_all += 1
            if m0 and not m1:
                saved_by_en.append((group, q["qid"], bm_max, raw, tr, q["must"]))
            if m1 and not m0:
                lost_by_en.append((group, q["qid"], bm_max, raw, tr, q["must"]))
            details.append((group, q["qid"], live, bm_max, ascii_tokens(raw), m0, m1, m2))

            if i % 20 == 0 or i == len(ps):
                print(f"    {i}/{len(ps)}", flush=True)

    print(f"\n\n########## ① 中文题里词法路到底死不死（共 {n_all} 题）")
    lv, dd = buckets["live"], buckets["dead"]
    print(f"  词法路**开火**（问句含英文正文里命中的 token）：{lv['n']:>3} 题 "
          f"({lv['n'] / max(n_all, 1):.0%})")
    print(f"  词法路**死掉**（纯中文 token）：           {dd['n']:>3} 题 "
          f"({dd['n'] / max(n_all, 1):.0%})")
    print("  → 所以『中文提问 → 词法路必死』这句是不成立的：中文题的词法路开火与否，"
          "取决于它内嵌了多少**英文原文 token**")

    print(f"\n\n########## ② 分层后的未命中（N={args.n}，A0=中文原问 / A1=英文译式 / A2=并集）")
    print(f"  {'分层':<26}{'题数':>6}{'A0 中文':>10}{'A1 英文':>10}{'A2 并集':>10}")
    for k, nm in (("live", "词法路开火"), ("dead", "词法路死（纯中文）")):
        b = buckets[k]
        print(f"  {nm:<24}{b['n']:>6}{b['A0']:>10}{b['A1']:>10}{b['A2']:>10}")
    tot = {a: lv[a] + dd[a] for a in ("A0", "A1", "A2")}
    print(f"  {'合计':<24}{n_all:>6}{tot['A0']:>10}{tot['A1']:>10}{tot['A2']:>10}")

    print(f"\n\n########## ③ 因果核对：英文**救回**的题（中文漏→英文中）")
    print(f"  {'qid':<12}{'BM25max':>9}  {'题':<40}")
    for _g, qid, bmx, raw, _tr, _must in saved_by_en:
        print(f"  {qid:<12}{bmx:>9.3f}  {raw[:38]}")

    print("\n\n########## ④ 英文**弄丢**的题（中文中→英文漏）—— 这就是『不能全英文』的证据")
    print(f"  {'qid':<12}{'BM25max':>9}  {'题':<40}")
    for _g, qid, bmx, raw, _tr, _must in lost_by_en:
        print(f"  {qid:<12}{bmx:>9.3f}  {raw[:38]}")

    print(f"\n########## ⑤ 明细（供二次分析）")
    print(f"  {'qid':<12}{'词法':<6}{'BM25max':>9}{'A0':>4}{'A1':>4}{'A2':>4}  英文 token")
    for _g, qid, live, bmx, ats, m0, m1, m2 in details:
        if not (m0 or m1):
            continue                                # 只打有问题的那批
        print(f"  {qid:<12}{'开火' if live else '死':<6}{bmx:>9.3f}"
              f"{'❌' if m0 else '✅':>4}{'❌' if m1 else '✅':>4}{'❌' if m2 else '✅':>4}"
              f"  {','.join(ats[:5])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
