"""临时：盘点可用论文（报告缓存 / MinerU 产物 / 标题）。"""
import json
from pathlib import Path

paps = sorted(Path("src/paperpilot/assets/papers").glob("*.pdf"))
print(f"论文总数 {len(paps)}")
print(f"{'stem':18} {'report':7} {'mineru':7} title")
for p in paps:
    stem = p.stem
    rep = Path(f"assets/artifacts/out_views/{stem}.report.json")
    minu = Path(f"assets/artifacts/out_mineru/{stem}")
    title = ""
    if rep.exists():
        try:
            title = str(json.loads(rep.read_text(encoding="utf-8")).get("title") or "")[:56]
        except Exception:
            title = "(report解析失败)"
    print(f"{stem:18} {'Y' if rep.exists() else '-':7} "
          f"{'Y' if minu.exists() else '-':7} {title}")
