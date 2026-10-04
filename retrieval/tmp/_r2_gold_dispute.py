"""**A-2｜争议对复判（更富证据）**：新真值是不是因为"证据不够"而偏严？

## 为什么
`_r2_gold_recalib.py` 只给判官 **4 块 ≤4200 字符** → 论文真正做了 X 但不在那 4 块里时，
判官会说 "no" → **假阴性** → 新真值偏严（8.1 → 5.6 正例/题，可能掉过头）。

## 本脚本
对**争议对**用**富证据**（12 块 / ≤12000 字符，锚点块 ∪ 干净查询 top-12）重判：
  · `dispute`  ：锚点=正例 但新真值=否（**锚点可能太松** 的候选）
  · `gain`     ：锚点=否 但新真值=正例（**锚点漏掉** 的候选）
  · `sample`   ：两者都=否 的随机抽样（**假阴性** 的对照，看富证据能不能翻出正例）

判读：
  · 若 dispute 大量**翻回 yes** → 新真值偏严是"证据不够"造成的，锚点真值更可信
  · 若 dispute 仍是 no → **锚点真值确实太松**（只是"提及"不是"做了"）
"""
from __future__ import annotations

import importlib.util as _iu
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "tmp"))
from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))

spec = _iu.spec_from_file_location("rc", HERE / "tmp" / "_r2_gold_recalib.py")
rc = _iu.module_from_spec(spec)
spec.loader.exec_module(rc)          # 复用 SYS / SYS_STRICT / call / rx_toks

_m = rc._m
F = rc.F
CACHE = HERE / "data" / "r2dev" / "clusters"
SUBQ = HERE / "data" / "r2dev" / "subqueries.json"
RECAL = HERE / "data" / "r2dev" / "gold_recalib.csv"
OUT = HERE / "data" / "r2dev" / "gold_recalib_rich.csv"
RICH_TOPK, RICH_MAX_CHUNKS, RICH_MAX_CHARS = 12, 12, 12000


def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample", type=int, default=40, help="两者皆否的随机抽样数")
    args = ap.parse_args()

    df = pd.read_csv(RECAL)
    df["anchor_gold"] = df["anchor_gold"].astype(bool)
    df["new_gold"] = df["new_gold"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
    dis = df[df["anchor_gold"] & ~df["new_gold"]]
    gain = df[~df["anchor_gold"] & df["new_gold"]]
    both_no = df[~df["anchor_gold"] & ~df["new_gold"]]
    rng = np.random.default_rng(0)
    smp = both_no.iloc[rng.choice(len(both_no), min(args.sample, len(both_no)), replace=False)] \
        if len(both_no) else both_no
    print(f"争议对统计：锚点正&新否 **{len(dis)}** ｜ 锚点否&新正 {len(gain)}"
          f" ｜ 两者皆否 {len(both_no)}（抽 {len(smp)}）")

    subq_all = json.loads(SUBQ.read_text(encoding="utf-8"))
    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()

    want = pd.concat([dis, gain, smp])[["cluster", "facet", "docid"]].drop_duplicates()
    keys = {(int(r.cluster), str(r.facet), str(r.docid)) for r in want.itertuples()}
    print(f"需重判 {len(keys)} 对（富证据 {RICH_MAX_CHUNKS} 块 / ≤{RICH_MAX_CHARS} 字符）\n")

    jobs: list[dict] = []
    for ci in range(len(meta)):
        sub = pd.read_parquet(CACHE / f"c{ci}.parquet")
        docs = sub["docid"].tolist()
        titles = sub["title"].tolist() if "title" in sub.columns else ["" for _ in docs]
        chunks, owner = [], []
        for i, t in enumerate(sub["full_paper"].tolist()):
            cs = _m.chunks_of(t)
            chunks += cs
            owner += [docs[i]] * len(cs)
        owner = np.array(owner)
        C = enc.encode(chunks, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        low = [c.lower() for c in chunks]
        idx = {d: np.where(owner == d)[0] for d in docs}
        for facet in meta[ci]["usable"]:
            pat, zh, anchor, para = F[facet]
            rx = rc.rx_toks(pat)
            desens = [" ".join(w for w in s.split() if w.lower() not in rx) or s
                      for s in subq_all.get(facet, [])]
            qvec = enc.encode([zh] + desens, normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
            sim = qvec @ C.T
            hit = np.array([_m.judge(c, facet) for c in low])
            for d in docs:
                if (ci + 1, facet, d) not in keys:
                    continue
                ids = idx[d]
                sem = ids[np.argsort(-sim[:, ids].max(axis=0))[:RICH_TOPK]]
                orderl, seen = [], set()
                for j in list(ids[hit[ids]]) + list(sem):
                    if int(j) not in seen:
                        seen.add(int(j))
                        orderl.append(int(j))
                ev, tot = [], 0
                for j in orderl:
                    if len(ev) >= RICH_MAX_CHUNKS:
                        break
                    t = chunks[j]
                    if tot + len(t) > RICH_MAX_CHARS:
                        t = t[: max(0, RICH_MAX_CHARS - tot)]
                    if not t:
                        break
                    ev.append(t)
                    tot += len(t)
                jobs.append(dict(cluster=ci + 1, facet=facet, docid=d,
                                 claim=f"该论文{F[facet][1]}",
                                 title=str(titles[docs.index(d)] or "")[:120],
                                 evidence="\n---\n".join(ev)))
        print(f"  簇{ci + 1} 证据就绪", flush=True)

    lock = threading.Lock()
    n = {"i": 0}

    def work(j: dict) -> dict:
        user = f"论文标题：{j['title']}\n\n论断：**{j['claim']}**\n\n论文原文片段：\n{j['evidence']}"
        a = rc.call("PAPERPILOT_LLM", rc.SYS, user)
        b = rc.call("PAPERPILOT_JUDGE", rc.SYS, user)
        la, lb = str(a.get("label") or "ERR").lower(), str(b.get("label") or "ERR").lower()
        third = {}
        if la == "entail" and lb == "entail":
            final = "yes"
        elif "entail" not in (la, lb) and "ERR" not in (la, lb):
            final = "no"
        else:
            third = rc.call("PAPERPILOT_LLM", rc.SYS_STRICT,
                            f"论文标题：{j['title']}\n\n论断：**{j['claim']}**\n\n"
                            f"论文原文片段：\n{j['evidence']}")
            final = "yes" if str(third.get("answer", "")).upper().startswith("Y") else "no"
        with lock:
            n["i"] += 1
            if n["i"] % 40 == 0:
                print(f"  重判 {n['i']}/{len(jobs)} …", flush=True)
        return dict(cluster=j["cluster"], facet=j["facet"], docid=j["docid"],
                    A_label=la, B_label=lb, agree=int(la == lb), need_third=int(bool(third)),
                    third=str(third.get("answer") or "")[:12], new_gold_rich=final,
                    ev_chars=len(j["evidence"]), A_quote=str(a.get("quote") or "")[:150])
    t0 = time.time()
    rows = []
    if OUT.exists():
        cached = pd.read_csv(OUT)
        have = {(int(r.cluster), str(r.facet), str(r.docid)) for r in cached.itertuples()}
        rows = [r._asdict() for r in cached.itertuples()] if keys <= have else []
        if rows:
            print(f"复用已有 {len(rows)} 条富证据判定")
    if not rows:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(work, jobs):
                rows.append(r)
        out = pd.DataFrame(rows)
        out.to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"\n完成 {len(out)} 对（{time.time() - t0:.0f}s）→ {OUT}")
    out = pd.DataFrame(rows)

    m = df.merge(out, on=["cluster", "facet", "docid"], how="inner", suffixes=("_old", "_rich"))
    if "new_gold_rich" not in m.columns:
        m = m.rename(columns={c: c.replace("_rich", "") for c in m.columns
                              if c.endswith("_rich")})
    print("\n" + "=" * 112)
    print("【争议对：4 块证据 → 12 块证据，判官改口了吗】")
    grp = {"dispute（锚点正 → 新否）": m[(m["anchor_gold"]) & (~m["new_gold"])],
           "gain（锚点否 → 新正）": m[(~m["anchor_gold"]) & (m["new_gold"])],
           "sample（两者皆否）": m[(~m["anchor_gold"]) & (~m["new_gold"])]}
    for nm, g in grp.items():
        if not len(g):
            continue
        fl = float((g["new_gold_rich"] == "yes").mean())
        ag = g["agree"].mean() if "agree" in g.columns else float("nan")
        th = g["need_third"].mean() if "need_third" in g.columns else float("nan")
        print(f"  {nm:<26} n={len(g):<4} 富证据后判 yes 的比例 **{fl:.3f}**"
              f" ｜ 两判官一致 {ag:.3f} ｜ 走三轮 {th:.3f}")
    d = grp["dispute（锚点正 → 新否）"]
    if len(d):
        flip = float((d["new_gold_rich"] == "yes").mean())
        print(f"\n  → 判读：{'⚠️ 多数翻回 yes → **新真值偏严**（证据不够）；锚点真值更可信' if flip > 0.5 else '✅ 多数仍是 no → **锚点真值确实太松**（只是提及，不是做了）'}")
        for _, r in d[d["new_gold_rich"] == "yes"].head(5).iterrows():
            print(f"    · {r['facet']:<16}{r['docid']:<7} A={r.get('A_label')} B={r.get('B_label')}"
                  f" → yes")
    m.to_csv(HERE / "results" / "R2_GOLD_DISPUTE.csv", index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
