"""临时：cap 上限对"接收块被推出甜点区"的影响扫描。

问题：甜点区 1000-3000；cap=4500 是否会把健康接收块(≤3000)并后推出甜点区(>3000)？
对比 cap ∈ {3000, 3500, 4000, 4500}：
    1) 真正被"推出甜点区"的接收块数（并前 ≤3000 且并后 >3000）
    2) 因 cap 放弃并入而残留的碎块数（<900 未消灭）
    3) 接收后 >3000 总块数（含原本就 >3000 的接收块）
零 encode 静态。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.qasper_source import load_papers  # noqa: E402


def merge_cap_stats(chunks, th: int, cap: int):
    n = len(chunks)
    lens = [len(c.text) for c in chunks]
    merged: set[int] = set()
    recv_src: dict[int, list[int]] = {}
    i = 0
    while i < n:
        if i in merged or lens[i] >= th:
            i += 1
            continue
        j = i + 1
        while j < n and j in merged:
            j += 1
        forward = j < n
        if not forward:
            j = i - 1
            while j >= 0 and j in merged:
                j -= 1
        if j < 0:
            i += 1
            continue
        if lens[j] + lens[i] > cap:
            i += 1
            continue
        recv_src.setdefault(j, []).append(i)
        lens[j] += lens[i]
        merged.add(i)
        i += 1
    # 统计
    pushed = 0      # 接收块并前≤3000 并后>3000
    over3k = 0      # 接收后 >3000 的总接收块
    recv_total = 0
    for j, srcs in recv_src.items():
        added = sum(lens[s] for s in srcs)
        before = lens[j] - added
        after = lens[j]
        recv_total += 1
        if after > 3000:
            over3k += 1
            if before <= 3000:
                pushed += 1
    tiny_left = sum(1 for idx in range(n) if idx not in merged and lens[idx] < th)
    return {"recv": recv_total, "pushed": pushed, "over3k": over3k,
            "tiny_left": tiny_left, "merged": len(merged)}


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    pids = sorted({it["pid"] for it in data["items"]})
    TH = 900
    all_chunks = []
    for pid in pids:
        all_chunks += ordered_chunks(f"qasper_{pid}.qpdf")
    print(f"> 全部 {len(all_chunks)} 块 | TH={TH} | 甜点区 1000-3000")
    print(f"| cap | 并入块 | 接收块数 | 推出甜点区(并前≤3k→后>3k) | 接收后>3k | 残留碎块<{TH} |")
    print("|---|---|---|---|---|---|")
    for cap in (3000, 3500, 4000, 4500):
        s = merge_cap_stats(all_chunks, TH, cap)
        print(f"| {cap} | {s['merged']} | {s['recv']} | {s['pushed']} | {s['over3k']} | "
              f"{s['tiny_left']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
