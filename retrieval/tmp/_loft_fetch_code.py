"""抓 LoFT 官方评测代码与 QAMPARI 官方 prompt（保证 subspan_em 口径一致、prompt 可比）。"""
from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parents[1] / "data" / "loft"
OUT.mkdir(parents=True, exist_ok=True)


def get(url: str, binary: bool = False):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read() if binary else r.read().decode("utf-8", "replace")


print("【1】LoFT 仓库文件树（筛 eval / prompt / metric 相关）")
try:
    tree = json.loads(get("https://api.github.com/repos/google-deepmind/loft/git/trees/main?recursive=1"))
    paths = [t["path"] for t in tree.get("tree", [])]
    print(f"  共 {len(paths)} 个文件")
    for p in paths:
        low = p.lower()
        if any(k in low for k in ("eval", "metric", "prompt", "subspan", "rag")):
            print("   ", p)
except Exception as e:  # noqa: BLE001
    print("  ERR", type(e).__name__, str(e)[:120])

print("\n【2】官方 QAMPARI prompt 包")
try:
    z = get("https://storage.googleapis.com/loft-bench/prompts/qampari.zip", binary=True)
    dst = OUT / "prompts_qampari.zip"
    dst.write_bytes(z)
    print(f"  下载 {len(z) / 1e6:.2f} MB → {dst}")
    with zipfile.ZipFile(dst) as zf:
        for n in zf.namelist()[:20]:
            print(f"    {n}  ({zf.getinfo(n).file_size / 1024:.1f} KB)")
except Exception as e:  # noqa: BLE001
    print("  ERR", type(e).__name__, str(e)[:120])
