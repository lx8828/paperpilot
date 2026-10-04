"""**回归检验**：`extract` 补丁是否会**损害集合判定**？

R2 的题是**纯集合题**（"哪些篇做了 X"，无内容要求）→ `extract` 在此**无用**。
但多要一个字段有可能让判官分心 → 必须确认 **篇级集合 F1 不退化**。

做法：复用 `_r2_reader.py` 的**全部流程与真值**（保证可比），只把判定提示换成
`SYS_SET_EXTRACT`，再算 F1 与基线 `R2_r2reader_b6.csv` 对照。
"""
from __future__ import annotations

import importlib.util as _iu
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

spec = _iu.spec_from_file_location("r2reader_reg", ROOT / "retrieval" / "tmp" / "_r2_reader.py")
rd = _iu.module_from_spec(spec)
sys.modules["r2reader_reg"] = rd
spec.loader.exec_module(rd)

from paperpilot.components import set_judge  # noqa: E402

rd.SYS = set_judge.SYS_SET_EXTRACT          # ← 唯一改动：提示词换成带 extract 的版本
sys.argv = ["_r2_reader.py", "--workers", "8", "--tag", "r2reader_extract"]
print("=" * 100)
print("【回归】R2 纯集合题：`SYS_SET` vs `SYS_SET_EXTRACT`（同流程、同真值，只换提示）")
print("=" * 100)
raise SystemExit(rd.main())
