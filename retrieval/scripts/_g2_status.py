"""group2 五篇的**解析路与块数**自检（临时）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from paperpilot.agents import document_cache as dc  # noqa: E402

for s in ["2604.20087", "2605.09341", "2606.18837", "2606.25389", "2609.02094v1"]:
    pdf = f"{s}.pdf"
    dc._ordered_chunks_cached.cache_clear()
    dc._current_source_cached.cache_clear()
    dc._mineru_status_cached.cache_clear()
    try:
        cs = dc.ordered_chunks(pdf)
        st, why = dc.mineru_status(pdf)
        print(f"{s:<16} 路={dc.current_source(pdf):<8} status={st:<9} 块={len(cs):<4} {why[:60]}")
    except Exception as e:  # noqa: BLE001
        print(f"{s:<16} 失败：{type(e).__name__}: {e}")
