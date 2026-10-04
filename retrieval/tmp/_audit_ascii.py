"""把导出的 preds/queries 转成**纯 ASCII**（`ensure_ascii=True`），消除官方 runner 的编码歧义。

背景：官方 `run_evaluation.py` 用 `open(path)`（无 encoding）→ Windows 上是 GBK，
非 ASCII 文件会解码成乱码或直接报错 → 指标不可信。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parents[1]
SRC = HERE / "results" / "official_audit_k40"
DST = HERE / "results" / "official_audit_k40_ascii"
DST.mkdir(parents=True, exist_ok=True)

for name in ("preds.jsonl", "queries.jsonl"):
    src = SRC / name
    if not src.exists():
        print(f"缺 {src}")
        continue
    n = 0
    with open(DST / name, "w", encoding="ascii") as out:
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            out.write(json.dumps(json.loads(line), ensure_ascii=True) + "\n")
            n += 1
    print(f"{name}: {n} 行 → {DST / name}（纯 ASCII，{ (DST / name).stat().st_size } B）")

# 自检：按 GBK 与按 UTF-8 读结果一致
p = DST / "preds.jsonl"
gbk = [json.loads(l) for l in open(p, encoding="gbk")]
utf = [json.loads(l) for l in open(p, encoding="utf-8")]
print(f"自检：GBK 与 UTF-8 读出的行数 {len(gbk)} vs {len(utf)}；第 1 行完全一致={gbk[0] == utf[0]}")
