"""group2 选篇体检（临时）：PDF / MinerU 产物 / 已有旧题 / 标题。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CAND = ["2609.08696v1", "2609.02094v1", "2609.02264v1", "2609.02786v1",
        "2609.03035v1", "2609.03335v1", "2609.03416v1", "2609.03718v1",
        "2609.04048v1", "2609.04170v1"]
V = Path("assets/artifacts/out_views")
P = Path("assets/papers")
M = Path("assets/artifacts/out_mineru")

print(f"{'stem':<16}{'PDF':<5}{'MinerU':<8}{'旧题':<5}title")
for s in CAND:
    t = ""
    rf = V / f"{s}.report.json"
    if rf.exists():
        try:
            t = str(json.loads(rf.read_text(encoding="utf-8")).get("title") or "")[:44]
        except Exception:  # noqa: BLE001
            pass
    qf = Path("qa/questions") / f"{s}.json"
    n = "-"
    if qf.exists():
        d = json.loads(qf.read_text(encoding="utf-8"))
        n = str(len(d if isinstance(d, list) else d.get("questions", [])))
    has_pdf = "有" if (P / f"{s}.pdf").exists() else "无"
    has_m = "有" if (M / s).is_dir() else "-"
    print(f"{s:<16}{has_pdf:<5}{has_m:<8}{n:<5}{t}")
