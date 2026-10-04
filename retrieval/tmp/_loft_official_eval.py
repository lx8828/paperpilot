"""**评估审计**：把官方 LoFT 评测包完整拉下来，用**官方代码**跑我们的预测，与我的复现版对比。

为什么必须做：我们之前的 `subspan_em` 是**照抄逻辑自己实现**的（不是 import 官方模块），
存在实现偏差风险（归一化、匹配口径、gold/pred 顺序等）。

官方入口：`run_evaluation.py --answer_file_path <queries.jsonl> --pred_file_path <preds.jsonl> --task_type multi_value_rag`
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RAW = "https://raw.githubusercontent.com/google-deepmind/loft/main/"
PKG = Path(__file__).resolve().parents[1] / "data" / "loft" / "official"
PKG.mkdir(parents=True, exist_ok=True)

FILES = [
    "run_evaluation.py",
    "evaluation/__init__.py",
    "evaluation/loft_evaluation.py",
    "evaluation/rag.py",
    "evaluation/retrieval.py",
    "evaluation/sql.py",
    "evaluation/icl.py",
    "evaluation/utils.py",
    "evaluation/metrics.py",
]
for f in FILES:
    dst = PKG / f
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        txt = urllib.request.urlopen(RAW + f, timeout=60).read().decode("utf-8", "replace")
        dst.write_text(txt, encoding="utf-8")
        print(f"OK   {f}  ({len(txt)} 字)")
    except Exception as e:  # noqa: BLE001
        print(f"SKIP {f}  {type(e).__name__} {str(e)[:70]}")

print("\n" + "=" * 100)
p = PKG / "run_evaluation.py"
if p.exists():
    txt = p.read_text(encoding="utf-8")
    print("【run_evaluation.py 全部内容】")
    print(txt[:6000])
