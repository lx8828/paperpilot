"""探测：语料里有多少篇**能通过 arXiv 拿到 PDF**（fetch 工具的前提评估）。

两段：
  ① **本地**：有没有 arxiv id、id 是什么格式（现代 YYMM.NNNNN / 老式 cs/9811009）
     —— 没有 id 的（旧语料里那些非 arXiv 论文）就是 arXiv-only 策略下的**硬缺口**。
  ② **联网抽样**：拿到 id ≠ PDF 一定在（可能撤稿、删除、id 有误）。
     各抽样 N 条做 HEAD 请求，看真实状态码。

⚠️ 礼貌抓取：每请求间隔 1.2 秒 + 标明 UA（arXiv 的 robots 要求）。

用法：
    python retrieval/scripts/probe_pdf_availability.py --sample 60
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
UA = "paperpilot-research/0.1 (academic retrieval index; contact: local)"
SLEEP = 1.2
MODERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
OLDSTYLE = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$")


def norm_id(s: str) -> str:
    """规范化 arxiv id：去空白、去版本号（PDF 端点不带版本也能取到最新版）。"""
    t = str(s).strip()
    t = re.sub(r"^arxiv:", "", t, flags=re.I)
    return re.sub(r"v\d+$", "", t)


def probe(aid: str) -> tuple[int, str, str]:
    """HEAD https://arxiv.org/pdf/<id> → (状态码, content-type, 备注)。"""
    url = f"https://arxiv.org/pdf/{aid}"
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            ct = r.headers.get("Content-Type", "")
            cl = r.headers.get("Content-Length", "?")
            return r.status, ct, f"{cl}B"
    except urllib.error.HTTPError as e:
        return e.code, "", e.reason or ""
    except Exception as e:  # noqa: BLE001
        return -1, "", type(e).__name__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60, help="每侧抽多少条做联网实测")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    print("=" * 90)
    print("① 本地：arxiv id 覆盖率与格式")
    print("=" * 90)
    a = pd.read_parquet(HERE / "data/arxiv/meta/corpus_text.parquet",
                        columns=["docid", "arxiv_id"])
    l = pd.read_parquet(HERE / "data/litsearch/derived/corpus_text.parquet",
                        columns=["corpusid", "arxiv", "text"])
    l = l[l["text"] != ""].reset_index(drop=True)

    sides = []
    for tag, s in (("arXiv", a["arxiv_id"]), ("LitSearch", l["arxiv"])):
        v = s.astype(str).str.strip().replace({"None": "", "nan": "", "nan.0": ""})
        has = v.str.len() > 0
        m = v[has].map(lambda x: bool(MODERN.match(norm_id(x))))
        o = v[has].map(lambda x: bool(OLDSTYLE.match(norm_id(x))))
        sides.append({"语料侧": tag, "总篇数": len(v), "有arxiv id": int(has.sum()),
                      "占比": f"{100 * has.mean():.1f}%",
                      "现代格式": int(m.sum()), "老式格式": int(o.sum()),
                      "其它格式": int((has.sum() - m.sum() - o.sum()))})
        print(f"\n  [{tag}] {len(v):,} 篇")
        print(f"    有 id : {int(has.sum()):,}（{100 * has.mean():.1f}%）")
        print(f"    格式  : 现代 {int(m.sum()):,} | 老式 {int(o.sum()):,} | 其它 "
              f"{int(has.sum() - m.sum() - o.sum()):,}")
        if (has.sum() - m.sum() - o.sum()) > 0:
            bad = v[has][~(m | o)].head(5).tolist()
            print(f"    其它样例: {bad}")

    tot = sum(x["总篇数"] for x in sides)
    noid = sum(x["总篇数"] - x["有arxiv id"] for x in sides)
    print("\n" + "=" * 90)
    print("★ arXiv-only 策略下的**硬缺口**（没有 arxiv id → 拿不到 PDF）")
    print("=" * 90)
    print(f"  合计 {tot:,} 篇 | 无 id {noid:,} 篇 = **{100 * noid / tot:.2f}%**")
    print(f"    └ 全部来自 LitSearch 侧（那 4.76w 非 arXiv 论文）")
    print(f"  有 id {tot - noid:,} 篇 —— 理论上可获取（实际能否取到见下）")

    if args.skip_probe:
        return 0

    print("\n" + "=" * 90)
    print(f"② 联网实测（每侧抽样 {args.sample} 条 HEAD 请求，间隔 {SLEEP}s）")
    print("=" * 90)
    rng = np.random.default_rng(args.seed)
    for tag, v in (("arXiv", a["arxiv_id"]),
                   ("LitSearch(有id)", l.loc[l["arxiv"].astype(str).str.strip().ne("") &
                                             l["arxiv"].astype(str).str.lower().ne("nan"),
                                             "arxiv"])):
        v = v.astype(str).str.strip()
        v = v[v.str.len() > 0]
        ids = np.array([norm_id(x) for x in v])
        pick = ids[rng.choice(len(ids), min(args.sample, len(ids)), replace=False)]
        res = {"200": 0, "404": 0, "other": 0}
        detail: list[str] = []
        t0 = time.time()
        for i, aid in enumerate(pick, 1):
            st, ct, note = probe(aid)
            ispdf = "pdf" in ct.lower()
            if st == 200 and ispdf:
                res["200"] += 1
            elif st == 404:
                res["404"] += 1
            else:
                res["other"] += 1
                detail.append(f"{aid} → HTTP {st} ct={ct[:30]} {note}")
            if i % 15 == 0:
                print(f"    {tag} {i}/{len(pick)}  ({time.time() - t0:.0f}s)"
                      f"  200:{res['200']} 404:{res['404']} other:{res['other']}")
            time.sleep(SLEEP)
        n = len(pick)
        print(f"\n  [{tag}] 抽样 {n} 条：**可获取 {res['200']} ({100 * res['200'] / n:.1f}%)**"
              f" | 404 {res['404']} | 其它 {res['other']}")
        for d in detail[:6]:
            print(f"      {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
