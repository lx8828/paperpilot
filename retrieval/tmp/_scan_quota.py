"""零 LLM 成本扫描：5 篇语料下 `top_k` × 篇级配额（floor / per-paper cap）× 融合模式。

为什么这么扫（2026-09-24）：
    · `top_k=12` 是**单篇时代**定的；
    · 生产多篇路径 `search_layered(mode="quota")` 的 `_search_quota` **只用 `floor` 和
      `top_k`**（`per_paper_k` 在 quota 下是**死参数**，读代码即知）；
    · `floor` 是**从尾部替换**：5 篇 × floor → 最多吃掉尾部 `5*floor` 个槽位
      （floor=2 就用掉 12 里的 10 个 ≈ rrf 的"过度均衡"）→ **cap 必须按篇数重定**。

方法（关键：零重复编码）：
    每题的**全局全量排名**与**每篇全量排名**各算一次，转成"该块命中了哪几个 must_all 锚点"
    的**位掩码**；之后 70+ 个配置全是在掩码上做**纯 Python 选块**（等价复刻
    `_search_quota` / `_fuse` 的选择逻辑），不再碰向量/BM25。
    因此两组合计只跑 120 题 × 6 次检索，配置再多也不加成本。

指标：
    miss   必现锚点(must_all)未全部落进候选的题数 ← 主指标，与 `_check_reachable --search` 同口径
    qhit   至少一条 gold 引文（前 40 字符）落进候选的题数 ← 次指标（"有原文支撑"）
    papers 平均命中篇数 ／ chars 平均候选字符数（= 上下文成本）
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "retrieval" / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from _qa_groups import group_gold_file, material_dir, papers  # noqa: E402
from paperpilot.agents.embedder import ChunkIndex, MultiChunkIndex  # noqa: E402

QCHARS = 40                       # 引文探针长度（前 N 字符，空白折叠后匹配）


# ── 探针（题）加载 ────────────────────────────────────────────────────────────

def norm(s: str) -> str:
    return " ".join(str(s or "").split())


def load_probes(group: str) -> list[dict]:
    """收集**需要下钻检索**的题（与 `_check_reachable --search` 同口径）。"""
    out: list[dict] = []

    def add(q: dict) -> None:
        must = [str(x) for x in (q.get("must_all") or [])]
        if not must:
            return
        out.append({
            "qid": str(q.get("qid")),
            "question": str(q.get("question") or ""),
            "must": must,
            "quotes": [norm(e.get("quote"))[:QCHARS]
                       for e in (q.get("evidence") or []) if e.get("quote")],
        })

    for stem in papers(group):
        gf = material_dir(group) / f"{stem}.questions.json"
        if not gf.exists():
            continue
        for q in json.loads(gf.read_text(encoding="utf-8"))["questions"]:
            if str(q.get("category") or "") in ("L0", "negative"):
                continue
            add(q)
    gfile = group_gold_file(group)
    if gfile.exists():
        for q in json.loads(gfile.read_text(encoding="utf-8"))["questions"]:
            if str(q.get("kind") or "") in ("M0", "X5"):
                continue
            add(q)
    return out


# ── 池子构建：全量排名 → 掩码 ────────────────────────────────────────────────

def build_pool(idx: MultiChunkIndex, q: dict, cache: Path) -> dict:
    """返回 `{"glob": [entry...], "per": {pdf: [entry...]}}`，entry = 掩码元组。

    `entry = (pdf, chunk_id, score, amask, qmask, tlen)`
        amask: 该块命中 `must_all` 的位掩码（全中 = (1<<n)-1）
        qmask: 该块命中 gold 引文的位掩码（≥1 即"有原文支撑"）
    """
    must, quotes = q["must"], q["quotes"]

    def mask_of(text: str) -> tuple[int, int]:
        am = qm = 0
        for i, a in enumerate(must):
            if a in text:
                am |= 1 << i
        for i, qt in enumerate(quotes):
            if qt and qt in norm(text):
                qm |= 1 << i
        return am, qm

    def ent(pdf: str, h: dict) -> tuple:
        am, qm = mask_of(str(h.get("text") or ""))
        # `sec` 复刻 `ChunkIndex._section_of`：取 title_path[0]（形如 "1 · Introduction"）的节名。
        # ⚠️ 该节名**不含篇名** → 跨篇同名会撞成一个节（正是节级配额在多篇下的病灶）。
        tp = list(h.get("title_path") or [])
        head = str(tp[0]) if tp else ""
        sec = head.split(" · ", 1)[1].strip() if " · " in head else head.strip()
        return (pdf, str(h.get("chunk_id")), float(h.get("score") or 0.0),
                am, qm, len(str(h.get("text") or "")), sec)

    n_all = len(idx._doc_chunks())
    glob = [ent(str(h.get("pdf") or ""), h) for h in idx.search_hybrid(q["question"], top_k=n_all)]
    per: dict[str, list[tuple]] = {}
    for p in idx.pdfs:
        ci = ChunkIndex(p)
        per[p] = [ent(p, h) for h in ci.search_hybrid(q["question"], top_k=len(ci._doc_chunks()))]
    return {"glob": glob, "per": per}


# ── 选块：复刻生产逻辑（在掩码上）────────────────────────────────────────────

def sel_hybrid(pool: dict, n: int, **_kw) -> list[tuple]:
    return pool["glob"][:n]


def sel_quota(pool: dict, n: int, floor: int = 1, **_kw) -> list[tuple]:
    """**逐行复刻** `MultiChunkIndex._search_quota`（尾部替换式保底）。"""
    out = list(pool["glob"][:n])
    have = {(e[0], e[1]) for e in out}
    slot = len(out) - 1
    for p, lst in pool["per"].items():
        if slot < 0:
            break
        for e in lst[:floor]:
            if (e[0], e[1]) in have:
                continue
            if slot < 0:
                break
            out[slot] = e
            have.add((e[0], e[1]))
            slot -= 1
    return out


def sel_capfloor(pool: dict, n: int, cap: int = 3, floor: int = 1, **_kw) -> list[tuple]:
    """**候选新方案**：先按全局序取，但每篇**最多 cap 块**；再补每篇保底 floor。

    为什么：`quota` 只保底不设上限 → 排序好的那篇可以吃掉全部 12 个名额的前 7 个
    （切得细的篇尤其如此）。多篇语料需要**同时**有上限（防霸占）与保底（防漏篇）。
    cap 过紧（cap*篇数 < n）时自动放宽到 `ceil(n/篇数)`，保证选得满。
    """
    npap = max(len(pool["per"]), 1)
    eff = max(cap, math.ceil(n / npap))
    out: list[tuple] = []
    cnt: Counter = Counter()
    for e in pool["glob"]:
        if cnt[e[0]] < eff:
            out.append(e)
            cnt[e[0]] += 1
        if len(out) >= n:
            break
    if len(out) < n:                                    # cap 收紧后仍不满 → 用全局补满
        seen = {(e[0], e[1]) for e in out}
        for e in pool["glob"]:
            if (e[0], e[1]) not in seen:
                out.append(e)
                seen.add((e[0], e[1]))
            if len(out) >= n:
                break
    have = {(e[0], e[1]) for e in out}
    slot = len(out) - 1                                 # 再补保底（同 quota：从尾部替换）
    for p, lst in pool["per"].items():
        if slot < 0:
            break
        for e in lst[:floor]:
            if (e[0], e[1]) in have:
                continue
            if slot < 0:
                break
            out[slot] = e
            have.add((e[0], e[1]))
            slot -= 1
    return out


def sel_seccap_flat(pool: dict, n: int, cap: int = 1, **_kw) -> list[tuple]:
    """**现语义**：节级配额 key = 节名（**不含篇名**）→ 跨篇同名撞成一个节。

    复刻 `ChunkIndex._cap_select`：保序过滤，每节最多 cap 块，**不足 n 也不补**
    （这是生产行为：cap>0 时 `_cap_select` 直接 `return out`）。
    """
    out: list[tuple] = []
    used: Counter = Counter()
    for e in pool["glob"]:
        if used[e[6]] >= cap:
            continue
        out.append(e)
        used[e[6]] += 1
        if len(out) >= n:
            break
    return out


def sel_seccap_paper(pool: dict, n: int, cap: int = 2, **_kw) -> list[tuple]:
    """**修正语义**：节级配额 key = (篇, 节名) —— 每篇内各自限流，互不干扰。"""
    out: list[tuple] = []
    used: Counter = Counter()
    for e in pool["glob"]:
        k = (e[0], e[6])
        if used[k] >= cap:
            continue
        out.append(e)
        used[k] += 1
        if len(out) >= n:
            break
    return out


def sel_rrf(pool: dict, n: int, pk: int = 4, **_kw) -> list[tuple]:
    """复刻 `_fuse(mode="rrf")`：篇内前 `pk` 块按 1/(60+rank) 融合。"""
    rows: list[tuple[float, float, tuple]] = []
    for _p, lst in pool["per"].items():
        for rank, e in enumerate(lst[:pk], 1):
            rows.append((1.0 / (60 + rank), e[2], e))
    rows.sort(key=lambda t: (-t[0], -t[1]))
    return [e for _fs, _s, e in rows[:n]]


def _zsel(pool: dict, n: int, kind: str) -> list[tuple]:
    rows: list[tuple[float, float, tuple]] = []
    for _p, lst in pool["per"].items():
        ss = [e[2] for e in lst]
        mu = sum(ss) / len(ss) if ss else 0.0
        sd = (sum((x - mu) ** 2 for x in ss) / len(ss)) ** 0.5 if len(ss) > 1 else 0.0
        for e in lst:
            fs = ((e[2] - mu) / sd) if (kind == "znorm" and sd > 1e-9) else (e[2] - mu)
            rows.append((fs, e[2], e))
    rows.sort(key=lambda t: (-t[0], -t[1]))
    return [e for _fs, _s, e in rows[:n]]


def sel_zmean(pool: dict, n: int, **_kw) -> list[tuple]:
    return _zsel(pool, n, "zmean")


def sel_znorm(pool: dict, n: int, **_kw) -> list[tuple]:
    return _zsel(pool, n, "znorm")


ALL_PAPER_CAP = 5      # 5 篇语料固定


def configs() -> list[tuple[str, callable, dict]]:
    cfgs: list[tuple[str, callable, dict]] = []
    for n in (12, 16, 20, 24, 32, 48):
        cfgs.append((f"hybrid  N={n:<3}", sel_hybrid, {"n": n}))
    for n in (12, 16, 20, 24, 32, 48):
        for fl in (1, 2, 3):
            cfgs.append((f"quota   N={n:<3} floor={fl}", sel_quota, {"n": n, "floor": fl}))
    for n in (12, 20, 24, 32):
        for cap in (2, 3, 4, 5):
            for fl in (1, 2):
                cfgs.append((f"capfloor N={n:<3} cap={cap} floor={fl}", sel_capfloor,
                             {"n": n, "cap": cap, "floor": fl}))
    for n in (12, 20, 24):
        for pk in (4, 6, 8):
            cfgs.append((f"rrf     N={n:<3} pk={pk}", sel_rrf, {"n": n, "pk": pk}))
    for n in (12, 24):
        for cap in (1, 2, 3):
            cfgs.append((f"seccap-flat N={n:<3} cap={cap}", sel_seccap_flat,
                         {"n": n, "cap": cap}))
            cfgs.append((f"seccap-篇内 N={n:<3} cap={cap}", sel_seccap_paper,
                         {"n": n, "cap": cap}))
    for n in (12, 20, 24):
        cfgs.append((f"zmean   N={n:<3}", sel_zmean, {"n": n}))
    for n in (12, 20, 24):
        cfgs.append((f"znorm   N={n:<3}", sel_znorm, {"n": n}))
    return cfgs


def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--groups", default="group1,group2")
    ap.add_argument("--top", type=int, default=0, help="只打印前 N 行（0=全部）")
    ap.add_argument("--dump", default="", help="把全部配置的结果落盘成 JSON（便于二次分析）")
    args = ap.parse_args()

    all_cfg = configs()
    grand: dict[str, list[int]] = {name: [] for name, _f, _p in all_cfg}
    grand_q: dict[str, list[int]] = {name: [] for name, _f, _p in all_cfg}
    grand_p: dict[str, list[int]] = {name: [] for name, _f, _p in all_cfg}
    grand_c: dict[str, list[int]] = {name: [] for name, _f, _p in all_cfg}
    default_miss: dict[str, list[str]] = {}

    for group in [g.strip() for g in args.groups.split(",") if g.strip()]:
        ps = load_probes(group)
        corpus = [f"{s}.pdf" for s in papers(group)]
        idx = MultiChunkIndex(corpus)
        print(f"\n########## {group} ｜ {len(corpus)} 篇 / {len(idx._doc_chunks())} 单元 "
              f"｜ 需检索题 {len(ps)}", flush=True)

        pools: list[dict] = []
        t0 = time.time()
        for i, q in enumerate(ps, 1):
            pools.append(build_pool(idx, q, ROOT / "retrieval" / "tmp"))
            if i % 10 == 0 or i == len(ps):
                print(f"    池子 {i}/{len(ps)}（{time.time() - t0:.0f}s）", flush=True)

        rows: list[tuple[int, int, float, float, str]] = []
        for name, fn, kw in all_cfg:
            miss = qhit = 0
            npap = nchar = 0
            for q, pool in zip(ps, pools):
                sel = fn(pool, **kw)
                if not sel:
                    miss += 1
                    continue
                full = (1 << len(q["must"])) - 1
                am = qm = 0
                for e in sel:
                    am |= e[3]
                    qm |= e[4]
                if am != full:
                    miss += 1
                if qm:
                    qhit += 1
                npap += len({e[0] for e in sel})
                nchar += sum(e[5] for e in sel)
            nq = max(len(ps), 1)
            rows.append((miss, -qhit, npap / nq, nchar / nq, name))
            grand[name].append(miss)
            grand_q[name].append(qhit)
            grand_p[name].append(npap / nq)
            grand_c[name].append(nchar / nq)

        rows.sort(key=lambda t: (t[0], t[1], -t[2], t[3]))
        print(f"\n{'配置':<26}{'未命中':>7}{'有原文支撑':>11}{'平均篇数':>10}{'平均字符':>10}")
        show = rows[: args.top] if args.top else rows
        for miss, nqhit, paps, chars, name in show:
            print(f"{name:<26}{miss:>4}/{len(ps):<3}{-nqhit:>8}/{len(ps):<3}"
                  f"{paps:>10.2f}{chars:>10.0f}")

        dq = sel_quota(pools[0], n=12, floor=1)
        assert dq is not None
        bad = []
        for q, pool in zip(ps, pools):
            full = (1 << len(q["must"])) - 1
            am = 0
            for e in sel_quota(pool, n=12, floor=1):
                am |= e[3]
            if am != full:
                bad.append(q["qid"])
        default_miss[group] = bad
        print(f"    ↑ 当前生产默认 quota N=12 floor=1 未命中：{len(bad)} 道 {bad}")

    ng = max(len(grand[nm]) for nm, _f, _p in all_cfg)
    print(f"\n\n########## 汇总（{ng} 组 × 60 题 = {ng * 60} 题；miss 为各组之和）")
    agg = sorted(
        ((sum(grand[nm]), -sum(grand_q[nm]),
          sum(grand_p[nm]) / len(grand_p[nm]), sum(grand_c[nm]) / len(grand_c[nm]), nm)
         for nm, _f, _p in all_cfg), key=lambda t: (t[0], t[1], -t[2], t[3]))
    print(f"{'配置':<26}{'未命中':>9}{'有原文支撑':>11}{'平均篇数':>10}{'平均字符':>10}")
    for miss, nqhit, paps, chars, name in (agg[: args.top] if args.top else agg):
        print(f"{name:<26}{miss:>6}/{ng * 60}{-nqhit:>9}/{ng * 60}"
              f"{paps:>10.2f}{chars:>10.0f}")
    if args.dump:
        Path(args.dump).write_text(json.dumps(
            {"rows": [{"cfg": nm, "miss": sum(grand[nm]), "qhit": sum(grand_q[nm]),
                       "papers": round(sum(grand_p[nm]) / len(grand_p[nm]), 3),
                       "chars": round(sum(grand_c[nm]) / len(grand_c[nm]))}
                      for nm, _f, _p in all_cfg],
             "default_miss": default_miss}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  → 明细已落盘 {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
