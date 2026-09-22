"""arXiv 元数据抓取（OAI-PMH，官方推荐的批量方式）。

为什么用 OAI-PMH 而不是爬 API：
  arXiv 官方 bulk data 文档明确说 OAI-PMH 是"批量下载或保持元数据最新副本的**首选方式**"，
  并要求**不要**以程序方式抓全站（会返回数 TB）。本脚本按其规范：专用 UA + 请求间隔 3s。

抓什么：
  `set=cs:cs`（CS 全档）× `metadataPrefix=arXiv`（**含 abstract**）× 按月切片。
  再按 `<created>` 过滤 —— 因为 OAI 的 datestamp 是**更新日**，早期月份里混着大量老论文的更新。

断点续抓：
  **按"月"切片**，每月落一组原始 XML（`raw/<YYYY-MM>.pageNNN.xml.gz`）。
  重跑时已存在的分片直接读，不从网络重取 → 天然可中断、可复跑、可审计。

用法：
    python retrieval/scripts/fetch_arxiv.py --target 50000            # 抓满 5 万篇即停
    python retrieval/scripts/fetch_arxiv.py --target 50000 --months 8 # 最多往前覆盖 8 个月
"""
from __future__ import annotations

import argparse
import gzip
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ARXIV_DIR = HERE / "data" / "arxiv"
RAW = ARXIV_DIR / "raw"
META = ARXIV_DIR / "meta"

OAI = "https://oaipmh.arxiv.org/oai"
NS_OAI = "{http://www.openarchives.org/OAI/2.0/}"
NS_ARX = "{http://arxiv.org/OAI/arXiv/}"
UA = "paperpilot-research/0.1 (academic retrieval index; contact: local)"
SLEEP = 3.0            # arXiv 官方建议"突发 4 req/s 后休眠 1s"；这里更保守
MAX_PAGES_PER_MONTH = 60


def fetch(params: dict, tries: int = 4) -> str:
    url = OAI + "?" + urllib.parse.urlencode(params)
    last: Exception | None = None
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"      [重试 {t + 1}/{tries}] {type(e).__name__}: {e}")
            time.sleep(8 * (t + 1))
    raise RuntimeError(f"抓取失败: {url}") from last


def _txt(el, tag: str) -> str:
    x = el.find(NS_ARX + tag)
    return " ".join((x.text or "").split()) if x is not None and x.text else ""


def parse_records(xml: str) -> tuple[list[dict], str | None]:
    """解析一页 → (记录列表, 下一页 resumptionToken)。"""
    root = ET.fromstring(xml)
    out = []
    for rec in root.iter(NS_OAI + "record"):
        md = rec.find(".//" + NS_ARX + "arXiv")
        if md is None:
            continue
        authors = []
        for a in md.findall(f"{NS_ARX}authors/{NS_ARX}author"):
            fn = (a.findtext(NS_ARX + "forenames") or "").strip()
            kn = (a.findtext(NS_ARX + "keyname") or "").strip()
            authors.append(" ".join(x for x in (fn, kn) if x))
        out.append({
            "arxiv_id": _txt(md, "id"),
            "created": _txt(md, "created"),
            "updated": _txt(md, "updated"),
            "title": _txt(md, "title"),
            "abstract": _txt(md, "abstract"),
            "categories": _txt(md, "categories"),
            "comments": _txt(md, "comments"),
            "license": _txt(md, "license"),
            "authors": "; ".join(authors[:8]),
        })
    m = re.search(r"<resumptionToken[^>]*>([^<]*)</resumptionToken>", xml)
    tok = m.group(1).strip() if m and m.group(1).strip() else None
    return out, tok


def harvest_month(ym: str, set_spec: str) -> list[dict]:
    """抓一个自然月（按 OAI datestamp）。已有分片则直接读，不走网络。"""
    y, mo = (int(x) for x in ym.split("-"))
    lo = f"{y:04d}-{mo:02d}-01"
    monthend = (f"{y:04d}-{mo:02d}-28" if mo == 2 else
                f"{y:04d}-{mo:02d}-30" if mo in (4, 6, 9, 11) else f"{y:04d}-{mo:02d}-31")
    # ⚠️ OAI 的 datestamp 是**更新日**，且服务端索引**有明显滞后**（clamp 到"今天"
    # 仍然返回 badArgument: "until date too late"，整个月直接 0 条）。
    # → 当月**不指定 until**（"从本月 1 号到现在"语义相同，且不受滞后影响）；
    #   过去的月份才用月末（它在过去，不会越界）。
    hi = "" if ym == date.today().strftime("%Y-%m") else monthend
    if lo > date.today().isoformat():
        print(f"    {ym} 起始日晚于今天，跳过")
        return []
    recs: list[dict] = []
    token: str | None = None
    for page in range(MAX_PAGES_PER_MONTH):
        f = RAW / f"{ym}.page{page:03d}.xml.gz"
        if f.exists():
            xml = gzip.decompress(f.read_bytes()).decode("utf-8")
            cached = True
        else:
            if token:
                params = {"verb": "ListRecords", "resumptionToken": token}
            else:
                params = {"verb": "ListRecords", "metadataPrefix": "arXiv",
                          "set": set_spec, "from": lo}
                if hi:
                    params["until"] = hi
            xml = fetch(params)
            # 自动降级：若服务端仍嫌 until 太晚，去掉 until 重试一次
            if "date too late" in xml and "until" in params:
                print(f"    {ym} until={params['until']} 被拒（too late）→ 去掉 until 重试")
                params.pop("until")
                xml = fetch(params)
            f.write_bytes(gzip.compress(xml.encode("utf-8")))
            cached = False
            time.sleep(SLEEP)
        page_recs, token = parse_records(xml)
        recs += page_recs
        tag = "缓存" if cached else "网络"
        print(f"    {ym} 第 {page:02d} 页  {len(page_recs):>5} 条（{tag}）"
              f"  累计 {len(recs):>6}  下一页={'有' if token else '无'}")
        if not token:
            break
    return recs


def months_back(start: str, n: int) -> list[str]:
    """从 start(YYYY-MM) 起，往前 n 个月，返回倒序列表（最近的在前）。"""
    y, m = (int(x) for x in start.split("-"))
    out = []
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50000, help="目标篇数（created 过滤后计）")
    ap.add_argument("--months", type=int, default=8, help="最多回溯几个月")
    ap.add_argument("--start-month", default="", help="起始月 YYYY-MM（默认=当前月）")
    ap.add_argument("--min-created", default="2024-01-01",
                    help="只保留 created >= 此日期的记录（OAI datestamp 是更新日，会混入老论文）")
    ap.add_argument("--set", default="cs:cs", help="OAI set，cs:cs = CS 全档")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    start = args.start_month or date.today().strftime("%Y-%m")
    months = months_back(start, args.months)
    print(f"目标 {args.target} 篇（created >= {args.min_created}）| set={args.set}")
    print(f"回溯月份（近→远）：{months}\n")

    seen: set[str] = set()
    kept: list[dict] = []
    for ym in months:
        raw_n = len(kept)
        recs = harvest_month(ym, args.set)
        new = 0
        for r in recs:
            aid = r["arxiv_id"]
            if (not aid) or aid in seen or not r["abstract"]:
                continue
            if r["created"] and r["created"] < args.min_created:
                continue
            seen.add(aid)
            kept.append(r)
            new += 1
        print(f"  → {ym}: 收 {len(recs)} 条，**新增合格 {new}** 条，累计 {len(kept)} 条\n")
        if len(kept) >= args.target:
            print(f"已达目标 {args.target}，停止")
            break
        if new == 0 and len(kept) == raw_n:
            print("   （本月无新增，继续往前）")

    df = pd.DataFrame(kept)
    if df.empty:
        print("\n未抓到任何记录 —— 检查原始响应：retrieval/data/arxiv/raw/*.xml.gz")
        return 1
    out = META / "arxiv_meta.parquet"
    df.to_parquet(out, index=False)
    print(f"\n已写出：{out}  （{len(df)} 篇）")
    print(f"  列：{list(df.columns)}")
    yr = df["created"].str[:4].value_counts().sort_index()
    print(f"  年份分布：{yr.to_dict()}")
    cat = df["categories"].str.split().explode().value_counts().head(10)
    print(f"  分类 top10：{cat.to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
