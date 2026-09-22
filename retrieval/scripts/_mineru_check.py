"""MinerU 环境自检（临时）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from paperpilot import ingest  # noqa: E402

ok, why = ingest.mineru_available()
print("可用 :", ok)
print("原因 :", why)
print("命令 :", ingest.mineru_cmd())
print("版本 :", ingest.mineru_version())
