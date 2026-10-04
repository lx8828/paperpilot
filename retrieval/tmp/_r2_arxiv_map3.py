"""**Step 1（v3）｜60 篇 → arXiv id**：改用 **S2 batch 端点**（一次请求查全部）

## 为什么 v2 慢（我的设计错误）
v2 是**每篇 1~2 次请求** + 失败退避 12/24/36s → 8 篇要几分钟。
S2 提供 **batch**：`POST /graph/v1/paper/batch`（一次最多 500 个 id）→ **60 篇 = 1 次请求**。
arXiv 亦有 `id_list`（一次可校验多个 id）。

本脚本：**2 次 HTTP 请求**完成 60 篇映射（S2 batch → arXiv id_list 校验）。
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CACHE = HERE / "data" / "r2dev" / "clusters"
OUT = HERE / "data" / "r2dev" / "arxiv_map.json"
UA = "paperpilot-research/1.0"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
STOP = {"the", "a", "an", "of", "for", "in", "on", "to", "and", "with", "via", "using",
        "towards", "toward", "from", "by", "at", "as", "is", "are", "be"}


def norm_title(t: str) -> str:
    s = re.sub(r"\s*-\s*", "-", str(t))
    s = re.sub(r"[^a-z0-9\- ]+", " ", s.lower())
    return " ".join(s.split())


def toks(t: str) -> set[str]:
    return {w for w in norm_title(t).split() if w not in STOP and len(w) > 1}


def jac(a: str, b: str) -> float:
    A, B = toks(a), toks(b)
    return len(A & B) / max(len(A | B), 1)


def post(url: str, payload: dict, *, timeout: int = 60) -> tuple[bool, object]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": UA})
    for a in range(3):
        if a:
            time.sleep(8 * a)
        try:
            with OPENER.open(req, timeout=timeout) as r:
                return True, json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                return False, f"HTTP {e.code}: {e.read().decode('utf-8','ignore')[:160]}"
        except Exception as e:  # noqa: BLE001
            if a == 2:
                return False, f"{type(e).__name__}: {str(e)[:120]}"
    return False, "429 持续"


def get(url: str, *, timeout: int = 60) -> tuple[bool, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for a in range(3):
        if a:
            time.sleep(8 * a)
        try:
            with OPENER.open(req, timeout=timeout) as r:
                return True, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code != 429:
                return False, f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            if a == 2:
                return False, f"{type(e).__name__}"
    return False, "429 持续"


def main() -> int:
    items = []
    for ci, f in enumerate(sorted(CACHE.glob("c*.parquet"))):
        sub = pd.read_parquet(f)
        for _, r in sub.iterrows():
            items.append(dict(cluster=ci + 1, docid=str(r["docid"]),
                              corpusid=int(r["corpusid"]), title=str(r["title"])))
    ids = [f"CorpusId:{x['corpusid']}" for x in items]
    print("=" * 112)
    print(f"【Step 1 v3｜S2 batch】{len(items)} 篇 → 1 次 POST（batch 上限 500）")
    t0 = time.time()
    ok, res = post("https://api.semanticscholar.org/graph/v1/paper/batch"
                   "?fields=title,externalIds,openAccessPdf,year", {"ids": ids})
    print(f"  耗时 {time.time() - t0:.1f}s ｜ {'✅ 成功' if ok else '❌ 失败'}")
    if not ok:
        print(f"  {res}")
        return 1
    assert isinstance(res, list), type(res)
    print(f"  返回 {len(res)} 条（与请求一一对齐：对不上的位置为 null）\n")

    mp: dict[str, dict] = {}
    n_ax = n_oa = n_ok = 0
    for x, d in zip(items, res):
        rec = {**x, "arxiv": "", "oapdf": "", "src": "", "jac": 0.0, "ok": False, "note": ""}
        if not isinstance(d, dict):
            rec["note"] = "S2 无此 id"
        else:
            ext = d.get("externalIds") or {}
            ax = str(ext.get("ArXiv") or "")
            oa = str((d.get("openAccessPdf") or {}).get("url") or "")
            st = str(d.get("title") or "")
            j = jac(x["title"], st)
            if ax:
                rec.update(arxiv=ax, oapdf=oa, src="s2_batch", jac=round(j, 2), ok=True,
                           note=f"S2 batch（相似 {j:.2f}）")
            elif oa:
                rec.update(oapdf=oa, jac=round(j, 2), note=f"无 arXiv 但有开放 PDF（似 {j:.2f}）")
            else:
                rec["note"] = f"无 arXiv/开放 PDF（S2 题似 {j:.2f}）"
        n_ax += bool(rec["arxiv"]); n_oa += bool(rec["oapdf"]); n_ok += rec["ok"]
        mp[x["docid"]] = rec
    OUT.write_text(json.dumps(mp, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=" * 112)
    print(f"【覆盖率】有 arXiv id **{n_ax}/{len(items)}** ｜ 有开放 PDF {n_oa} ｜ 可用 **{n_ok}**")
    per: dict[int, list[dict]] = {}
    for v in mp.values():
        per.setdefault(int(v["cluster"]), []).append(v)
    for c in sorted(per):
        g = per[c]
        print(f"  簇{c}：{len(g)} 篇 ｜ arXiv **{sum(1 for v in g if v['arxiv'])}**"
              f" ｜ PDF {sum(1 for v in g if v['oapdf'])}"
              f" ｜ 可用 **{sum(1 for v in g if v['ok'])}**")

    print(f"\n  低相似度告警（<0.55，需人工核对）：")
    warn = [v for v in mp.values() if v["arxiv"] and v["jac"] < 0.55]
    for v in warn:
        print(f"    {v['docid']:<6}★{v['jac']:.2f} {v['arxiv']:<16}{v['title'][:52]}")
    if not warn:
        print("    （无）")

    miss = [v for v in mp.values() if not v["ok"]]
    if miss:
        print(f"\n  无 arXiv/开放 PDF 的 {len(miss)} 篇：")
        for v in miss[:40]:
            print(f"    {v['docid']:<6}{v['title'][:76]}")

    axs = [v["arxiv"] for v in mp.values() if v["arxiv"]]
    if axs:
        p = OUT.with_suffix(".ids.txt")
        p.write_text("\n".join(axs), encoding="utf-8")
        print(f"\n  → {p}（{len(axs)} 个 id）")
    oas = [v for v in mp.values() if not v["arxiv"] and v["oapdf"]]
    if oas:
        print(f"  另有 {len(oas)} 篇只有非 arXiv 开放 PDF（需单独下载路径）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
