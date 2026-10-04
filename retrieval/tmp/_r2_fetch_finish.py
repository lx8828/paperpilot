"""**收尾下载**：只下 `aclanthology.org` / `aclweb.org`（快且可靠），**硬跳过 acm/doi**。

实测（2026-09-29）：`dl.acm.org` / `doi.org` 的 socket 会**挂住数分钟**，
`urllib` 的 `timeout` 只管单次 socket 操作、**管不住"服务器极慢地滴流数据"** → 必须按主机白名单跳过。

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_fetch_finish.py
"""
from __future__ import annotations

import json
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
PAPERS = ROOT / "assets" / "papers"
M = HERE / "data" / "r2dev" / "arxiv_map.json"
OUT = HERE / "data" / "r2dev" / "pdf_map.json"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
OK_HOSTS = ("aclanthology.org", "aclweb.org")      # ✅ 白名单：直链、快
LIMIT_SEC = 25                                     # 硬墙钟上限
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

m = json.loads(M.read_text(encoding="utf-8"))
PAPERS.mkdir(parents=True, exist_ok=True)


def dl_fast(url: str, dest: Path) -> tuple[bool, str]:
    """带硬墙钟的下载：超时即返回，不等 socket。"""
    box: dict = {}

    def work() -> None:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept": "application/pdf,*/*"})
            op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with op.open(req, timeout=15) as r:
                box["data"] = r.read()
        except Exception as e:  # noqa: BLE001
            box["err"] = f"{type(e).__name__}"

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(LIMIT_SEC)
    if t.is_alive():
        return False, f"超 {LIMIT_SEC}s（主机滴流，硬放弃）"
    if "err" in box:
        return False, box["err"]
    data = box.get("data") or b""
    if not data[:5].startswith(b"%PDF"):
        return False, f"非 PDF（{len(data)}B）"
    dest.write_bytes(data)
    return True, f"{len(data) / 1024:.0f}KB"


mp: dict[str, dict] = {}
n_ax = 0
for v in m.values():
    if not v["arxiv"] or v["docid"] == "C2P19":
        continue
    pdf = PAPERS / f"{v['arxiv'].split('v')[0]}.pdf"
    if pdf.exists():
        mp[v["docid"]] = {"pdf": pdf.name, "arxiv": v["arxiv"], "src": "arxiv", "ok": True}
        n_ax += 1
print(f"① arXiv 就绪 **{n_ax}** 篇")

todo = [v for v in m.values() if not v["arxiv"] and v["oapdf"]]
have, skip, go = [], [], []
for v in todo:
    dest = PAPERS / f"{v['docid']}.pdf"
    host = urlparse(v["oapdf"]).netloc
    if dest.exists() and dest.stat().st_size > 20000:
        mp[v["docid"]] = {"pdf": dest.name, "src": "oapdf", "ok": True, "url": v["oapdf"]}
        have.append(v["docid"])
    elif any(h in host for h in OK_HOSTS):
        go.append(v)
    else:
        skip.append((v["docid"], host))
print(f"② 非 arXiv：已有 **{len(have)}** ｜ 本轮下 **{len(go)}**（ACL 白名单）｜ 跳过 {len(skip)}（acm/doi）")
for d, h in skip:
    print(f"    ⏭️ {d:<6}{h}（付费墙/滴流，不投入）")


def one(v: dict) -> tuple[str, dict]:
    dest = PAPERS / f"{v['docid']}.pdf"
    ok, why = dl_fast(v["oapdf"], dest)
    return v["docid"], {"pdf": dest.name if ok else "", "src": "oapdf", "ok": ok,
                        "url": v["oapdf"], "note": why}


if go:
    with ThreadPoolExecutor(max_workers=5) as ex:
        for docid, rec in ex.map(one, go):
            mp[docid] = rec
            print(f"  {'✅' if rec['ok'] else '❌'} {docid:<6}{rec['note']:<26}"
                  f"{m[docid]['title'][:44]}", flush=True)

for v in m.values():
    if v["docid"] not in mp:
        host = urlparse(v["oapdf"]).netloc if v["oapdf"] else ""
        mp[v["docid"]] = {"pdf": "", "src": "none", "ok": False,
                          "note": "无 arXiv/开放 PDF" if not v["oapdf"] else f"跳过 {host}"}

OUT.write_text(json.dumps(mp, ensure_ascii=False, indent=1), encoding="utf-8")
n_ok = sum(1 for v in mp.values() if v["ok"])
print("\n" + "=" * 108)
print(f"【PDF 就绪】**{n_ok}/{len(m)}** 篇（{n_ok / len(m):.0%}）｜ 映射 {OUT.name}")
per: dict[int, list] = {}
for docid, v in mp.items():
    per.setdefault(int(m[docid]["cluster"]), []).append((docid, v))
for c in sorted(per):
    g = per[c]
    print(f"  簇{c}: **{sum(1 for _, v in g if v['ok'])}/{len(g)}** 篇")
miss = [(d, v) for d, v in mp.items() if not v["ok"]]
if miss:
    print(f"\n  缺 {len(miss)} 篇：")
    for d, v in miss:
        print(f"    {d:<6}{v.get('note', ''):<24}{m[d]['title'][:56]}")
