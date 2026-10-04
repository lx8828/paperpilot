"""跨语言查询改写 A/B：**中文问题 → 英文检索式**，看能否补上"词法路失效"。

为什么（2026-09-24）：产品是中文提问、语料是英文论文 → `_bm_or_none` 判定 BM25 全 0
→ `rrf_order` **整条词法路被跳过**，只剩向量路。锚点（英文专名/指标）排名靠后时
只能靠加大 top_k 硬捞（已实测：N=12→22/120、N=24→11/120、N=48→5/120）。
如果查询侧能给出**英文表述**，BM25 就有信号 → 理论上不需要那么大 N。

臂（检索侧零 LLM 成本，翻译结果落盘缓存）：
  A0  zh       中文原问 @ layered(N)
  A1  en1      英文译式 @ layered(N)
  A2  zh+en1   中文主路 + 英文译式**配额补充**（`quota_union` 语义：整块拼在尾部，
               **不交错**——交错会稀释主路排序，见 `query_optimizer.quota_union`）
  A3  zh+en3   中文主路 + 3 条英文变体各占配额（需 `--variants`）

⚠️ **A2 的口径与生产不一致，别引用它的数字**（2026-09-26 由 `_sandbox.py --union` 查出）：
   本脚本把英文臂算成 `sel_quota(pool_en, n=N)` 再**取前 xling_k 个**；
   而**生产**调的是 `search_layered(en, top_k=L3_XLING_K)` —— K=8 时 floor 会
   **从尾部替换 5 个槽位**（5 篇 × floor=1）→ "英文臂的前 8"根本不是英文排名的前 8。
   实测（120 题）：本脚本口径 5/120 ｜ **生产实际 7/120** ｜ 不截断 4/120。
   → 检索层的**权威数字请用 `_sandbox.py`**；本脚本只作"翻译质量"的快速观察。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "retrieval" / "scripts"))
sys.path.insert(0, str(ROOT / "retrieval" / "tmp"))
sys.path.insert(0, str(ROOT / "src"))

import _scan_quota as sq  # noqa: E402
from _qa_groups import papers  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

# ⚠️ 与本项目所有 CLI 一致：脚本必须显式加载 .env，否则 `llm.chat_json` 抛
# `LLMError: LLM 未配置`（踩过一次：三臂全 0 是因为翻译全静默失败了）。
llm._load_dotenv(str(ROOT))

CACHE = ROOT / "retrieval/tmp/_xling_cache.json"

_SYS = (
    "你是学术检索的**跨语言查询改写器**。目标语料是**英文论文全文**，用户的问题是**中文**。\n"
    "现状：只有中文查询 → 词法（BM25）路完全失效（中文 token 在英文正文里零匹配），"
    "只能靠向量路 → 英文专名/指标类证据捞不回来。你的任务是给出**英文检索式**，让词法路可用。\n"
    "规则：\n"
    "1. `translation`：把中文问题改写成**一条英文检索句**，≤25 词；\n"
    "   用**论文正文里可能出现的英文表述**（不要逐字直译），例：'分哪几步' → 'stages pipeline'；\n"
    "   **实体原名必须保留**（模型名/数据集名/指标/数字/公式符号，如 MAMuJoCo / VAE / w/o Expand）；\n"
    "2. `variants`：2~3 条**更短**的英文查询（每条 ≤12 词），各覆盖问题的不同侧面或同义表述；\n"
    "   问题里出现'哪几篇/哪些/分别'等比较时，variants 要覆盖各对比维度；\n"
    "3. 不要回答问题、不要自创新问题，只产出检索式。\n"
    '只输出 JSON：{"translation": "...", "variants": ["...", "..."]}'
)


def english_queries(qid: str, question: str, allow_llm: bool = True) -> dict:
    cache: dict = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    hit = cache.get(qid)
    # ⚠️ 只在**译文非空**时信任缓存：否则上一轮"LLM 未配置"写下的空值会**自我毒化**
    #    （命中空值 → 永远不再调 LLM → 三臂全 0，踩过一次）。
    if hit and hit.get("q") == question and str(hit.get("translation") or "").strip():
        return hit
    if not allow_llm:
        return {"q": question, "translation": "", "variants": []}
    try:
        obj = llm.chat_json(_SYS, f"问题：{question}", temperature=0.0)
    except Exception as e:  # noqa: BLE001
        print(f"    ⚠️ {qid} 翻译失败：{e}")
        obj = {}
    rec = {"q": question,
           "translation": str((obj or {}).get("translation") or "").strip(),
           "variants": [str(x).strip() for x in ((obj or {}).get("variants") or []) if str(x).strip()]}
    cache[qid] = rec
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def union(lists: list[list[tuple]], caps: list[int]) -> list[tuple]:
    """`quota_union` 语义：按路顺序**整块拼接** + 去重（**不交错**）。"""
    out: list[tuple] = []
    seen: set = set()
    for lst, cap in zip(lists, caps):
        for e in lst[: max(cap, 0)]:
            k = (e[0], e[1])
            if k not in seen:
                seen.add(k)
                out.append(e)
    return out


def stat(sel: list[tuple], must: list[str], nq: int) -> tuple[int, int, int, int, int]:
    if not sel:
        return 1, 0, 0, 0, 0
    full = (1 << len(must)) - 1
    am = qm = 0
    for e in sel:
        am |= e[3]
        qm |= e[4]
    return (0 if am == full else 1, 1 if qm else 0,
            len({e[0] for e in sel}), sum(e[5] for e in sel), 0)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="group2")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--variants", action="store_true", help="额外测 A3（需 3 个变体的池子，成本×3）")
    args = ap.parse_args()

    if not llm.is_configured():                       # 静默失败过一次，这里必须硬拦
        print("[错误] LLM 未配置（缺 .env 的 PAPERPILOT_LLM_*）→ 翻译会全空")
        return 1

    arms = ["A0 zh", "A1 en1", "A2 zh+en1"] + (["A3 zh+en3"] if args.variants else [])
    tot: dict[str, list[int]] = {a: [0, 0, 0, 0] for a in arms}

    for group in [g.strip() for g in args.groups.split(",") if g.strip()]:
        ps = sq.load_probes(group)
        corpus = [f"{s}.pdf" for s in papers(group)]
        idx = sq.MultiChunkIndex(corpus)
        print(f"\n########## {group} ｜ {len(ps)} 道需检索题 ｜ N={args.n}", flush=True)

        rows = []
        for i, q in enumerate(ps, 1):
            en = english_queries(q["qid"], q["question"])
            tr = en["translation"]
            if not tr:
                rows.append((q, None, None))
                continue
            pool_zh = sq.build_pool(idx, q, ROOT / "retrieval" / "tmp")
            pool_en = sq.build_pool(idx, {**q, "question": tr}, ROOT / "retrieval" / "tmp")
            pools_v = [sq.build_pool(idx, {**q, "question": v}, ROOT / "retrieval" / "tmp")
                       for v in en["variants"][:3]] if args.variants else []
            sels = {
                "A0 zh": sq.sel_quota(pool_zh, n=args.n, floor=1),
                "A1 en1": sq.sel_quota(pool_en, n=args.n, floor=1),
                "A2 zh+en1": union([sq.sel_quota(pool_zh, n=args.n, floor=1),
                                    sq.sel_quota(pool_en, n=args.n, floor=1)],
                                   [args.n, 8]),
            }
            if args.variants:
                vsel = [sq.sel_quota(p, n=args.n, floor=1) for p in pools_v]
                sels["A3 zh+en3"] = union([sels["A2 zh+en1"]] + vsel,
                                          [args.n + 8] + [6] * len(vsel))
            rows.append((q, en, sels))
            if i % 20 == 0 or i == len(ps):
                print(f"    {i}/{len(ps)}", flush=True)

        print(f"\n{'臂':<12}{'未命中':>8}{'有原文支撑':>11}{'平均篇数':>9}{'平均字符':>9}")
        for a in arms:
            m = s = p = c = 0
            for _q, _en, sels in rows:
                if not sels:
                    m += 1
                    continue
                dm, dq, dp, dc, _ = stat(sels[a], _q["must"], len(ps))
                m, s, p, c = m + dm, s + dq, p + dp, c + dc
            print(f"{a:<12}{m:>5}/{len(ps):<2}{s:>8}/{len(ps):<2}"
                  f"{p / max(len(ps), 1):>9.2f}{c / max(len(ps), 1):>9.0f}")
            tot[a] = [tot[a][0] + m, tot[a][1] + s, tot[a][2] + p, tot[a][3] + c]

        print("\n  —— 11 道现状漏题的英文改写与命中变化（看改写质量）")
        base_qids = json.loads((ROOT / "retrieval/tmp/_scan_quota.json")
                               .read_text(encoding="utf-8"))["default_miss"][group]
        for q, en, sels in rows:
            if q["qid"] not in base_qids or not sels:
                continue
            marks = " ".join(f"{a.split()[0]}={'✅' if stat(sels[a], q['must'], 0)[0] == 0 else '❌'}"
                             for a in arms)
            print(f"    [{q['qid']}] {marks}")
            print(f"       中文: {q['question'][:70]}")
            print(f"       英文: {en['translation'][:110]}")

    print("\n\n########## 汇总（各组之和；篇数/字符见各组表）")
    print(f"{'臂':<12}{'未命中':>8}{'有原文支撑':>11}")
    for a in arms:
        print(f"{a:<12}{tot[a][0]:>8}{tot[a][1]:>11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
