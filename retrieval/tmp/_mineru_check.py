"""核查 MinerU 是否可用 + 现有产物覆盖情况。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from paperpilot import ingest  # noqa: E402

print("=" * 100)
print("【MinerU 可用性】")
print(f"  MINERU_VENV   = {ingest.MINERU_VENV}")
print(f"  mineru_cmd()  = {ingest.mineru_cmd()}")
ok, why = ingest.mineru_available()
print(f"  available     = {ok}  {why}")
print(f"  version       = {ingest.mineru_version()!r}")
print(f"  backend       = {ingest.MINERU_BACKEND}  ｜ timeout {ingest.MINERU_TIMEOUT}s/篇")
print(f"  MINERU_OUT    = {ingest.MINERU_OUT}")

venv_py = ingest.MINERU_VENV / "Scripts" / "python.exe"
print(f"  venv python   = {venv_py.exists()}  {venv_py}")

try:
    import torch
    print(f"  torch cuda    = {torch.cuda.is_available()}"
          + (f"  {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
except Exception as e:  # noqa: BLE001
    print(f"  torch         = 不可用（{type(e).__name__}）")

print()
print("【现有 MinerU 产物覆盖】")
out = ingest.MINERU_OUT
dirs = sorted(p.name for p in out.iterdir() if p.is_dir()) if out.is_dir() else []
print(f"  out_mineru 下共 **{len(dirs)}** 个产物目录")

HERE = ROOT / "retrieval"
pmap = json.loads((HERE / "data" / "r2dev" / "pdf_map.json").read_text(encoding="utf-8"))
has_pdf = [(d, v["pdf"]) for d, v in pmap.items() if v["ok"]]
n_have = sum(1 for _, p in has_pdf if (out / Path(p).stem).is_dir())
print(f"  47 篇新下载的 PDF 中已有产物：**{n_have}/{len(has_pdf)}**")

# 老 25 篇（我们自己那 5 组）
sys.path.insert(0, str(HERE / "scripts"))
from _qa_groups import papers  # noqa: E402

old = [f"{s}.pdf" for g in [f"group{i}" for i in range(1, 6)] for s in papers(g)]
n_old = sum(1 for p in old if (out / Path(p).stem).is_dir())
print(f"  老 25 篇（5 组）中已有产物：**{n_old}/{len(old)}**")
print(f"\n  合计待跑：{len(has_pdf) - n_have + (len(old) - n_old)} 篇")
print(f"  按 60~180s/篇估：{(len(has_pdf) - n_have + (len(old) - n_old)) * 60 / 60:.0f}"
      f"~{(len(has_pdf) - n_have + (len(old) - n_old)) * 180 / 60:.0f} 分钟")
