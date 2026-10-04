r"""**生产链路端到端答案质量评估**（多答案开放域：正确性 + 完整性）

## 被测对象 = **生产代码原样**
`paperpilot.components.set_judge.run()`（`PAPERPILOT_QA_READER=set` 那条线）。
检索走**新接入的** `ChunkIndex.search_multiroute`（多路子查询 + 加权 RRF）；
判定走**新接入的** 证据窗口 + 双判官 + 第三轮。→ 测的就是线上会跑的东西。

## 索引适配（**不改生产代码**）
`set_judge` 只依赖鸭子类型（`tests/test_set_judge.py` 已证明）：
`pdfs` / `_doc_chunks()` / `search_hybrid()`。这里做一个 `Corpus50Index`，
把 `corpus50` 的块与 bge-m3 向量包成该契约，并**直接借用生产的 `search_multiroute` 方法**。

## 指标（用户口径）
| 指标 | 定义 | 依赖 gold？ |
|---|---|---|
| **正确性（接地）** | 对**每个交付项**，用独立判官判"其证据是否支持该答案" → 通过率 | ❌ **不依赖**（开放域必需） |
| 正确性（对 gold） | \|交付 ∩ gold\| / \|交付\| | ✅ |
| **完整性（覆盖）** | \|交付 ∩ gold\| / \|gold\| | ✅ |
| 集合 F1 | 2PR/(P+R) | ✅ |
| 判官分歧 | A≠B 的比例（分歧走第三轮） | ❌ |

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_prod_eval.py --per-cluster 1 --grounding
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_prod_eval.py            # 全部 28 facet
"""
from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
DEV = HERE / "data" / "r2dev"

from paperpilot.agents import embedder as E  # noqa: E402
from paperpilot.components import set_judge as SJ  # noqa: E402
from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))


class Corpus50Index:
    """把 `corpus50` 的块包成 `set_judge` 要的鸭子类型；**检索用生产实现**。"""

    search_multiroute = E.ChunkIndex.search_multiroute      # ★ 生产方法，原样借
    search_hybrid = E.ChunkIndex.search_hybrid

    def __init__(self, pdfs: list[str], docs: list[str], texts: list[str], vecs: np.ndarray):
        self.pdfs = list(pdfs)
        self._owner = list(docs)
        self._chunks = [SimpleNamespace(text=t, chunk_id=f"{d}#{i}", title_path=[""],
                                        page_span=(1, 1), embed_text=t)
                        for i, (t, d) in enumerate(zip(texts, docs))]
        self._vecs = vecs.astype("float32")

    def _doc_chunks(self):
        return self._chunks

    def vectors(self):
        return self._vecs

    def _select(self, chunks, order, top_k):
        return list(order)[: min(top_k, len(order))]

    def _hit_extra(self, i: int):
        return {"pdf": self._owner[i], "section": ""}


K_LIST = [20, 30, 40, 50, 0]        # 0 = 不限（判定全部候选）；改 k 只需切前缀

# ★ 独立接地判官（2026-10-01 改）：① **不给**主判官的 `why`（那会把它带偏）；② **双判官**，
#   **双方都 YES 才算通过**（保守）。
# ⚠️ **问句必须与主判官 `SYS_SET` 完全同口径**（"该论文是否做了这件事"），**不能**问成
#    "证据是否**足以支持**" —— 后者是更高的门槛（充分性 vs 存在性），会让接地率被人为压低
#    （实测第一版写成"足以支持"→ 通过率掉到 `0/12`、`2/12`，与主判官口径不可比）。
#    独立性来自：不同提示词措辞 + **看不到主判官的判果/理由** + **双判官都须同意**。
GROUND_SYS = """你要**独立**判断「**该论文是否自己做了**」题干所述这件事。你会看到该论文自己的若干片段。

只输出三档之一：
- `YES`：该论文**自己做了**这件事（片段里有本文自己的做法/实验/设置）。
- `NO`：没有做；或只是**引用的他人工作**；或只是任务清单/评价指标/相关工作/未来工作里的一个词。
- `PARTIAL`：片段不足以判断（不要因为"没看到"就给 NO，也不要因为"看起来相关"就给 YES）。

判定纪律：
1. 只看给出的片段，**不要用先验知识补充**；
2. 区分「本文做的」与「引用他人做的」（`X et al. proposed…` → `NO`）；
3. 列出/提及 ≠ 做了；
4. 中文论断 ↔ 英文片段注意同义改写。

只输出 JSON：{"support":"YES|NO|PARTIAL","why":"≤30 字中文理由"}"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cluster", type=int, default=0, help="每簇取几个 facet（0=全部 28）")
    ap.add_argument("--b", type=int, default=12, help="每篇给判官的候选块数（窗口从中选）")
    ap.add_argument("--win-chars", type=int, default=4200)
    ap.add_argument("--retr", default="multiroute", choices=["multiroute", "hybrid"])
    ap.add_argument("--grounding", action="store_true", help="额外跑独立判官评「正确性（接地）」")
    ap.add_argument("--topk", type=int, default=0,
                    help="候选篇上限（0=不限 → 判全部候选，**k 可离线扫**；"
                         "定稿口径 = 30）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="R2_PROD_EVAL.csv")
    args = ap.parse_args()
    os.environ["PAPERPILOT_SET_RETR"] = args.retr
    os.environ["PAPERPILOT_SET_WIN_CHARS"] = str(args.win_chars)
    os.environ["PAPERPILOT_SET_TOPK"] = str(args.topk)

    pmap = json.loads((DEV / "corpus50" / "pdf_map_all.json").read_text(encoding="utf-8"))
    pm_ok = {d for d, v in pmap.items() if v.get("ok")}
    gg = pd.read_csv(DEV / "gold_final3.csv")
    gg["gold"] = gg["gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    NG = {(int(c), str(f)): set(x[x["gold"]]["docid"]) for (c, f), x in gg.groupby(["cluster", "facet"])}
    combos = {(int(k.split("|")[0]), k.split("|")[1])
              for k in json.loads((DEV / "facets_v2_selected.json")
                                  .read_text(encoding="utf-8"))["per_combo"]}
    sys.path.insert(0, str(HERE / "tmp"))
    from _r2_facets_v2 import F2 as FAC  # noqa: E402

    sel = []
    for ci in (1, 2, 3):
        fs = sorted(f for (c, f) in combos if c == ci)
        if args.per_cluster:
            lst = sorted(fs, key=lambda f: len(NG.get((ci, f), set())))
            fs = [lst[len(lst) // 2]]
        sel += [(ci, f) for f in fs]
    print("=" * 122)
    print(f"【生产链路端到端评估】{len(sel)} 个 facet ｜ 检索 `{args.retr}` ｜ b={args.b}"
          f" ｜ 窗口 ≤{args.win_chars} 字符 ｜ 双判官 + 第三轮")
    print(f"  被测 = `set_judge.run`（`src/paperpilot/components/set_judge.py`）"
          f" ｜ 判官配置: A={llm.is_configured()} B={llm.judge_configured()}")

    # ── 建索引（每簇一次，编码一簇一次）──
    IDX: dict[int, Corpus50Index] = {}
    t0 = time.time()
    for ci in (1, 2, 3):
        ch = pd.read_parquet(DEV / "prodchunk50" / "mineru" / f"c{ci - 1}.parquet")
        ls = pd.read_parquet(DEV / "corpus50" / f"c{ci - 1}.parquet")
        docs = [d for d in ls["docid"].tolist() if d in set(ch["docid"]) and d in pm_ok]
        texts, owner = [], []
        for d in docs:
            for t in ch[ch.docid == d]["text"].astype(str).tolist():
                texts.append(t)
                owner.append(d)
        vecs = np.asarray(E.encode_texts(texts), dtype="float32")
        IDX[ci] = Corpus50Index(docs, owner, texts, vecs)
        print(f"  簇{ci} 索引就绪：{len(docs)} 篇 / {len(texts)} 块 ｜ {time.time() - t0:.0f}s",
              flush=True)

    rows, deliv_rows = [], []
    for ci, facet in sel:
        claim = str(FAC[facet][1])
        gold = (NG.get((ci, facet)) or set()) & set(IDX[ci].pdfs)
        try:
            res = SJ.run(claim, IDX[ci], claim=claim, b=args.b, workers=args.workers)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ {ci} {facet} 失败：{type(e).__name__}: {str(e)[:90]}")
            continue
        # ── ★ 用**全部候选**的判果离线扫 k（改 k 只需切前缀，不必重跑判官）──
        labels = {str(k2): str(v2) for k2, v2 in (res.get("labels") or {}).items()}
        cand = [str(c["pdf"]) for c in (res.get("candidates") or [])]
        cscore = {str(c["pdf"]): c["score"] for c in (res.get("candidates") or [])}
        for p in cand:
            lab = labels.get(p, "unclear")
            deliv_rows.append(dict(cluster=ci, facet=facet, docid=p, label=lab,
                                   rank=cand.index(p) + 1, score=cscore.get(p),
                                   in_gold=int(p in gold),
                                   A=(res.get("verdicts", {}).get(p, {}) or {}).get("A", ""),
                                   B=(res.get("verdicts", {}).get(p, {}) or {}).get("B", ""),
                                   third=(res.get("verdicts", {}).get(p, {}) or {}).get("third", "")))
        # ── 接地复核（只对"判 yes 的交付项"，与 k 无关；逐篇 hits 只取一次）──
        sup: dict[str, str] = {}
        if args.grounding and res["papers"]:
            per = SJ.per_paper_hits(IDX[ci], claim, b=args.b, claim=claim)

            anch = list(res.get("anchors") or [])

            def gcheck(p):
                """★ 独立判官 + **同一份证据窗口**。

                ⚠️ 第一版错在"给接地判官的是**原始 top-12 块**、而主判官看的是**窗口**
                （锚点优先、≤4,200 字符）" → 接地判官看不到证据 → 假阴性爆表
                （实测 `2/12`、`1/8`）。**"独立"指判官独立，不是证据不同**：
                正确性 = "**交付的证据**能否支持该答案" → 必须喂**同一份窗口**。
                同时**不给**主判官的 `why`（那才叫不独立），且**双判官都 YES 才算通过**。
                """
                pdf = str(p["pdf"])
                user = SJ.build_user(claim, per.get(pdf, []),
                                     max_chars=args.win_chars, anchors=anch)
                try:
                    a = str((llm.chat_json(GROUND_SYS, user, temperature=0.0,
                                           max_tokens=200) or {}).get("support") or "")
                    b = str((llm.judge_json(GROUND_SYS, user, temperature=0.0,
                                            max_tokens=200) or {}).get("support") or "")
                    ok = a.upper().startswith("Y") and b.upper().startswith("Y")
                    return pdf, ("YES" if ok else f"A={a or '-'}/B={b or '-'}")
                except Exception:  # noqa: BLE001
                    return pdf, ""

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                sup = dict(ex.map(gcheck, res["papers"]))
        for k in K_LIST:
            top = cand if k <= 0 else cand[:k]
            pred = {p for p in top if labels.get(p) == "yes"}
            inter = len(pred & gold)
            P = inter / max(len(pred), 1)
            R = inter / max(len(gold), 1)
            gk = [p for p in pred if p in sup]
            okk = sum(1 for p in gk if sup[p].upper().startswith("Y"))
            rows.append(dict(cluster=ci, facet=facet, k=k, n_gold=len(gold), n_pred=len(pred),
                             n_cand=len(cand), inter=inter, setP=P, setR=R,
                             setF1=2 * P * R / (P + R) if (P + R) else 0.0,
                             n_dissent=res.get("n_dissent", 0),
                             n_unclear=len(res.get("unclear") or []),
                             n_ground=len(gk), n_ground_ok=okk,
                             ground_rate=(okk / len(gk)) if gk else None))
        line = "  ".join(
            f"k={k}:{'∞' if k <= 0 else k} P{rec.setP.mean():.2f}/R{rec.setR.mean():.2f}/"
            f"F1{rec.setF1.mean():.2f}" for k in K_LIST
            for rec in [pd.DataFrame([r for r in rows
                                      if r['cluster'] == ci and r['facet'] == facet and r['k'] == k])]
            if len(rec))
        print(f"  {ci} {facet:<20} gold {len(gold):>2} 候选 {len(cand):>2} ｜ {line}"
              f" ｜ 接地 {sum(1 for v in sup.values() if v.upper().startswith('Y'))}/{len(sup)}"
              f" ｜{time.time() - t0:.0f}s", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(HERE / "results" / args.out, index=False, encoding="utf-8-sig")
    if deliv_rows:
        pd.DataFrame(deliv_rows).to_csv(HERE / "results" / args.out.replace(".csv", "_DELIV.csv"),
                                        index=False, encoding="utf-8-sig")
    print("\n" + "=" * 122)
    print("【★★ k 对齐扫描】同一次运行的全部判果切前缀（**零新调用**）｜ 生产链路 `set_judge.run`")
    print(f"  {'k':>5}{'交付/题':>9}{'P(对gold)':>11}{'R完整性':>9}{'F1':>8}"
          f"{'接地通过率':>11}{'接地样本':>9}")
    for k in K_LIST:
        t = R[R.k == k]
        if not len(t):
            continue
        g = t[t.n_ground > 0]
        gr = (g.n_ground_ok.sum() / g.n_ground.sum()) if g.n_ground.sum() else float("nan")
        print(f"  {'∞' if k <= 0 else k:>5}{t.n_pred.mean():>9.1f}{t.setP.mean():>11.3f}"
              f"{t.setR.mean():>9.3f}{t.setF1.mean():>8.3f}{gr:>11.3f}{int(g.n_ground.sum()):>9}")
    r0 = R[R.k == K_LIST[0]]
    print(f"\n  gold 均 {r0.n_gold.mean():.1f} 篇 ｜ 候选均 {r0.n_cand.mean():.1f} 篇"
          f" ｜ A/B 分歧均 {r0.n_dissent.mean():.1f} ｜ unclear 均 {r0.n_unclear.mean():.1f}")
    print("\n  分簇 F1：")
    for k in K_LIST:
        t = R[R.k == k]
        if len(t):
            print(f"    k={'∞' if k <= 0 else k:>2}：" + " ｜ ".join(
                f"簇{int(c)} {x.setF1.mean():.3f}" for c, x in t.groupby("cluster")))
    print(f"\n  → 已写 results/{args.out} + {args.out.replace('.csv', '_DELIV.csv')}"
          f"（{time.time() - t0:.0f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
