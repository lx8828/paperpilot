"""临时：静态模拟不同切块 target 会切掉多少 base 健康块（不 encode）。

问题：B2@1600 把 base 健康块切碎。用户提议 target=2500。
判断口径：
  base 块 = 某节标题下连续段落（flow 中 [lo,hi] 区间，段落连续）。
  新打包 = 对同一 flow 按 target 累计切分 → 一串段落区间 [s,e)。
  若某个新块边界 s 落在 (lo,hi) 内 → base 块被拦腰切开（答案与上下文分家风险）。
统计每个策略切了几块 base 块，按 base 块大小分桶（<1k / 1-2.5k / 2.5-4k / >4k）。

策略：
  S1600 / S2500 / S4000 : 自由流打包 target=T
  S_anchor1600/2500     : 标题为软锚（切点不跨标题边界：先按标题内打包，段超 T 才在标题内切）
对比"切成的新块数量与块大小分布"谁更贴近 base 且更均匀。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.document_cache import ordered_chunks  # noqa: E402
from paperpilot.qasper_source import (_clean_para, _split_title,  # noqa: E402
                                      load_papers)


def flow_of(paper) -> list[tuple[list[str], str]]:
    flow: list[tuple[list[str], str]] = []
    for sec in paper.get("full_text") or []:
        parts = _split_title(sec.get("section_name") or "")
        path = parts or ["(PREAMBLE)"]
        if path == ["(PREAMBLE)"]:
            continue
        for p in (sec.get("paragraphs") or []):
            cl = _clean_para(p)
            if cl:
                flow.append((list(path), cl))
    return flow


def pack_ranges(flow, target: int, anchor: bool):
    """→ [(s, e)] 段落区间（[s,e) 左闭右开）。anchor=True：不跨标题边界累计。"""
    if not anchor:
        # 自由流
        cur_s = 0
        cur_len = 0
        out = []
        i = 0
        while i < len(flow):
            plen = len(flow[i][1]) + 1
            if cur_len and cur_len + plen > target:
                out.append((cur_s, i))
                cur_s = i
                cur_len = 0
            cur_len += plen
            i += 1
        if cur_s < len(flow):
            out.append((cur_s, len(flow)))
        return out
    # anchor：按标题分段内打包
    out = []
    i = 0
    n = len(flow)
    while i < n:
        path = flow[i][0]
        cur_s = i
        cur_len = 0
        while i < n and flow[i][0] == path:
            plen = len(flow[i][1]) + 1
            if cur_len and cur_len + plen > target:
                out.append((cur_s, i))
                cur_s = i
                cur_len = 0
            cur_len += plen
            i += 1
        if cur_s < i:
            out.append((cur_s, i))
    return out


def main() -> int:
    data = json.loads(Path("qa/recall/recall_set_v1.json").read_text(encoding="utf-8"))
    papers = load_papers()
    pids = sorted({it["pid"] for it in data["items"]})

    def fresh():
        return {"total": 0, "cut": 0, "cut_buckets": {0: 0, 1: 0, 2: 0, 3: 0},
                "buckets": {0: 0, 1: 0, 2: 0, 3: 0}}
    strategies = {
        "S_base": fresh(),
        "S1600_free": fresh(), "S2500_free": fresh(),
        "S1600_anchor": fresh(), "S2500_anchor": fresh(),
        "S4000_anchor": fresh(),
    }
    # 记录每策略产出的块大小分布（新块，字符）
    sizes = {k: [] for k in strategies}
    for pid in pids:
        paper = papers[pid]
        flow = flow_of(paper)
        bchunks = ordered_chunks(f"qasper_{pid}.qpdf")
        # base 块 → flow 区间：按 text 段落指纹定位
        base_ranges = []
        for c in bchunks:
            segs = [s for s in c.text.split("\n") if s.strip()]
            if not segs:
                continue
            # 找到首末段在 flow 中的位置（同段可能重复，取与 text 顺序一致）
            pos = []
            fi = 0
            for s in segs:
                while fi < len(flow) and flow[fi][1] != s:
                    fi += 1
                if fi >= len(flow):
                    break
                pos.append(fi)
                fi += 1
            if len(pos) < 2:  # 单段块
                if len(segs) == 1:
                    try:
                        g = next(i for i, (_p, t) in enumerate(flow) if t == segs[0])
                        base_ranges.append((g, g + 1, len(c.text)))
                    except StopIteration:
                        pass
                continue
            # pos 单调递增才可信
            if pos == sorted(pos) and pos[-1] - pos[0] == len(pos) - 1:
                base_ranges.append((pos[0], pos[-1] + 1, len(c.text)))
        for name, t, anchor in [("S_base", None, None),
                                ("S1600_free", 1600, False),
                                ("S2500_free", 2500, False),
                                ("S1600_anchor", 1600, True),
                                ("S2500_anchor", 2500, True),
                                ("S4000_anchor", 4000, True)]:
            st = strategies[name]
            if name == "S_base":
                packs = [(lo, hi) for lo, hi, _ in base_ranges]
                for lo, hi, n in base_ranges:
                    st["total"] += 1
                    b = 0 if n <= 1000 else (1 if n <= 2500 else (2 if n <= 4000 else 3))
                    st["buckets"][b] += 1
            else:
                packs = pack_ranges(flow, t, anchor)
            if name != "S_base":
                # 新块大小分布
                for s, e in packs:
                    ln = sum(len(flow[i][1]) + 1 for i in range(s, e))
                    sizes[name].append(ln)
                # 每个 base 块被几个新块切开
                for lo, hi, n in base_ranges:
                    st["total"] += 1
                    b = 0 if n <= 1000 else (1 if n <= 2500 else (2 if n <= 4000 else 3))
                    st["buckets"][b] += 1
                    ncut = sum(1 for s, _e in packs if lo < s < hi)
                    if ncut:
                        st["cut"] += 1
                        st["cut_buckets"][b] += 1
    def pct(x, p):
        if not x:
            return 0
        return int(sorted(x)[min(len(x) - 1, int(p / 100 * len(x)))])
    L = ["# 静态模拟：自由流 vs 标题锚点打包会切掉多少 base 块", ""]
    L.append(f"> 论文 {len(pids)} 篇 | base 块区间 {strategies['S_base']['total']} | "
             f"分桶 <1k/1-2.5k/2.5-4k/>4k: "
             f"{strategies['S_base']['buckets']}")
    L.append("")
    L.append("| 策略 | base块被切开 | <1k切开 | 1-2.5k切开 | 2.5-4k切开 | >4k切开 | 新块中位 | 新块p90 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name in strategies:
        st = strategies[name]
        if name == "S_base":
            L.append(f"| {name} | — | — | — | — | — | {pct(sizes.get(name, []), 50)} | "
                     f"{pct(sizes.get(name, []), 90)} |")
            continue
        L.append(f"| {name} | {st['cut']}/{st['total']} | {st['cut_buckets'][0]} | "
                 f"{st['cut_buckets'][1]} | {st['cut_buckets'][2]} | {st['cut_buckets'][3]} | "
                 f"{pct(sizes[name], 50)} | {pct(sizes[name], 90)} |")
    L.append("")
    L.append("> 读法：'切开'=该策略产出的块边界落在 base 块内部（上下文分家风险）。")
    L.append("> anchor=标题为硬边界(同节内打包)；free=跨节自由打包。")
    txt = "\n".join(L)
    print(txt)
    Path("qa/recall/SPLIT_SIM_20260910.md").write_text(txt + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
