"""为 recall_set 中缺 cvec 的论文预建向量缓存（一次性；完成后后续尺子秒级）。"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from paperpilot.agents.embedder import ChunkIndex

data = json.load(open("qa/recall/recall_set_v1.json", encoding="utf-8"))
pids = sorted({it["pid"] for it in data["items"]})
todo = [p for p in pids if not Path(f"assets/artifacts/out_views/qasper_{p}.cvec.npy").exists()]
prog = Path("qa/recall/_prewarm_progress.txt")
t0 = time.time()
prog.write_text(f"todo {len(todo)}\n", encoding="utf-8")
for i, pid in enumerate(todo, 1):
    ts = time.time()
    ChunkIndex(f"qasper_{pid}.qpdf").vectors()
    with prog.open("a", encoding="utf-8") as f:
        f.write(f"{i}/{len(todo)} {pid} {time.time()-ts:.0f}s\n")
    if i % 5 == 0 or i == len(todo):
        print(f"[{i}/{len(todo)}] t={time.time()-t0:.0f}s", flush=True)
print("prewarm done")
