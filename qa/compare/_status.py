"""查看三列进度：各 run JSON 已落盘记录数 + 日志最后进度。"""
from __future__ import annotations
import glob
import json
import os

for c in ("B0", "B1", "B2"):
    files = sorted(glob.glob(f"qa/compare/run_*_{c}.json"))
    if not files:
        print(f"{c}: (未开始)")
        continue
    f = files[-1]
    try:
        recs = json.loads(open(f, encoding="utf-8").read())
    except Exception:
        recs = []
    meta = f.replace(".json", ".meta.json")
    n_papers = "?"
    if os.path.exists(meta):
        try:
            n_papers = len(json.load(open(meta, encoding="utf-8"))["papers"])
        except Exception:
            pass
    print(f"{c}: {len(recs)} 条记录 / 目标 {n_papers} 篇 ｜ {os.path.basename(f)}")

logs = sorted(glob.glob("qa/compare/compare_*.log"))
if logs:
    raw = open(logs[-1], "rb").read()
    text = None
    for enc in ("utf-16", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text:
        tail = [l for l in text.splitlines() if l.strip()][-4:]
        print("--- log tail ---")
        for l in tail:
            print(l.encode("utf-8", "replace").decode("utf-8", "replace")[:120])
