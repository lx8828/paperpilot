"""切块质量审计（MinerU 49 篇；对照同篇 pymupdf）。

三组指标（前两组零 encode，第三组用本地 bge-m3）：
  A 长度     分位数 / 超长(>4000) / 近空(<100) / 跨页
  B 自包含性 首句悬空指代·连接词 / 块内含标题行 / 表格块 caption / 孤立公式块
  C 主题一致性（抽样）块内前后半 cos / 块↔节标题 cos / 块↔前邻 cos(边界锐度)
"""
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.agents.document_cache import (ordered_chunks,  # noqa: E402
                                              retrieval_chunks)
from paperpilot.tools import analyzer                              # noqa: E402
from paperpilot.tools.mineru_bridge import (_find_content_list,  # noqa: E402
                                            chunks_from_mineru, element_text)
from paperpilot.tools.pdf_parser import parse_pdf                 # noqa: E402
from paperpilot.tools.chunker import chunk_document               # noqa: E402

MINERU_OUT = ROOT / "assets/artifacts/out_mineru"
PAPERS = ROOT / "assets/papers"
SAMPLE_N = 12                       # C 组（要 encode）抽多少篇

# ── B 组：悬空指代 / 连接词开头（依赖上文才能读懂）──────────────────────────
_LEAD_REF = re.compile(
    r"^(this|these|those|it|they|such|its|their|the above|the former|the latter|"
    r"the proposed|our method|the method|the model|as shown|as described)\b", re.I)
_LEAD_CONJ = re.compile(
    r"^(however|moreover|furthermore|therefore|thus|hence|also|additionally|"
    r"in addition|meanwhile|consequently|similarly|on the other hand)\b", re.I)
_LEAD_CN = re.compile(r"^(这|该|此|其|上述|前者|后者|因此|所以|然而|但是|此外|另外|"
                      r"同时|并且|而且|不过|于是|总之|综上)")


def _sec_title(path: list[str]) -> str:
    """`L1 3.1 · 3.1 Datasets` → `3.1 Datasets`（去掉 label 前缀）。"""
    tail = (path[-1] if path else "").split("·")
    return tail[-1].strip() if len(tail) > 1 else (path[-1] if path else "")


def _first_sentence(text: str) -> str:
    t = " ".join(text.split())
    m = re.split(r"(?<=[.!?。！？])\s*", t, maxsplit=1)
    return (m[0] if m else t)[:160]


def analyze(chunks, tag: str, *, embed=False, cl=None) -> dict:
    n = len(chunks)
    out: dict = {"tag": tag, "n": n}
    if not n:
        return out
    lens = sorted(len(c.text) for c in chunks)
    q = lambda p: lens[min(int(n * p), n - 1)]
    out["len"] = dict(min=lens[0], p25=q(.25), med=q(.5), p75=q(.75), max=lens[-1])
    out["short100"] = sum(1 for x in lens if x < 100)
    out["short200"] = sum(1 for x in lens if x < 200)
    out["long4000"] = sum(1 for x in lens if x > 4000)
    out["multipage"] = sum(1 for c in chunks if c.page_span[1] - c.page_span[0] > 1)
    out["with_part"] = sum(1 for c in chunks if c.part)

    # B 组
    ref = conj = cn = has_title = 0
    for c in chunks:
        fs = _first_sentence(c.text)
        ref += bool(_LEAD_REF.match(fs))
        conj += bool(_LEAD_CONJ.match(fs))
        cn += bool(_LEAD_CN.match(fs))
        sec = _sec_title(list(c.title_path))
        has_title += bool(sec) and sec[:24].lower() in " ".join(c.text.split()).lower()[:400]
    out["lead_ref"] = ref
    out["lead_conj"] = conj
    out["lead_cn"] = cn
    out["has_title"] = has_title
    # 表格块：**按真实 `type=="table"` 元素统计**（2026-09-22 修：原用
    # `text.count("|")>=4` 的启发式，会把含 `|` 的正文块、公式块都算成表格 →
    # 严重低估 caption 覆盖率，得出"64.7% 无 caption"的错误结论。
    # 真实数字见 `_fix_check3` 口径：342 个表元素、仅 12% 无 caption。）
    if cl is not None:
        tabs = [e for e in cl if e.get("type") == "table"]
        out["table_blocks"] = len(tabs)
        out["table_with_cap"] = sum(
            1 for e in tabs
            if [x for x in (e.get("table_caption") or []) if str(x).strip()])
    # 孤立公式块：块内以 $$ 为主、正文很少
    out["eq_only"] = sum(
        1 for c in chunks
        if c.text.count("$$") >= 2 and len(re.sub(r"\$+", "", c.text)) < 120)

    if not embed or n < 3:
        return out
    # C 组（要 encode）
    from paperpilot.agents.embedder import encode_texts
    big = [c for c in chunks if len(c.text) >= 400]
    if len(big) >= 3:
        halves = []
        for c in big:
            h = len(c.text) // 2
            halves += [c.text[:h], c.text[h:]]
        v = encode_texts(halves)
        cs = [float(v[2 * i] @ v[2 * i + 1]) for i in range(len(big))]
        out["intra_cos"] = sum(cs) / len(cs)
    titles = [_sec_title(list(c.title_path)) for c in big] or ["x"]
    vt = encode_texts(titles)
    vc = encode_texts([c.text[:2000] for c in big])
    if len(big) >= 3:
        out["chunk_title_cos"] = sum(float(vc[i] @ vt[i]) for i in range(len(big))) / len(big)
        adj = []
        for a, b in zip(big, big[1:]):
            if list(a.title_path) == list(b.title_path):
                adj.append(float(encode_texts([a.text[:2000]])[0]
                                 @ encode_texts([b.text[:2000]])[0]))
        if adj:
            out["adj_cos"] = sum(adj) / len(adj)
    return out


def main() -> int:
    stems = sorted(p.name for p in MINERU_OUT.iterdir() if p.is_dir())
    print(f"MinerU 产物 {len(stems)} 篇 ｜ 抽样算 embedding 的前 {SAMPLE_N} 篇\n")

    rows_m: list[dict] = []
    rows_p: list[dict] = []
    for i, stem in enumerate(stems):
        cl = _find_content_list(MINERU_OUT / stem)
        if cl is None:
            continue
        emb = i < SAMPLE_N
        # ⚠️ 必须套 `analyzer.extractable` —— 生产两条路径都套了（去前言/参考文献/空块），
        # 不套的话 pymupdf 侧的超长块（Preamble/References 不参与二级切分）会污染长度统计。
        rows_m.append(analyze(analyzer.extractable(chunks_from_mineru(cl)),
                              "mineru", embed=emb, cl=cl))
        pdf = PAPERS / f"{stem}.pdf"
        if pdf.exists():
            try:
                rows_p.append(analyze(
                    analyzer.extractable(chunk_document(parse_pdf(str(pdf))["blocks"])),
                    "pymupdf", embed=emb))
            except Exception:  # noqa: BLE001
                pass
        if emb:
            print(f"  [embed {i + 1}/{SAMPLE_N}] {stem}", flush=True)

    def agg(rows, key):
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else 0

    def agg_len(rows, key):
        vals = [r["len"][key] for r in rows if "len" in r]
        return sum(vals) / len(vals) if vals else 0

    tot_m = sum(r["n"] for r in rows_m)
    tot_p = sum(r["n"] for r in rows_p)
    print(f"\n{'=' * 92}\n【A 长度】（每篇块均值）；分母：MinerU {tot_m} 块 / pymupdf {tot_p} 块")
    print(f"  {'指标':<26}{'MinerU':>12}{'pymupdf':>12}")
    for k, lab in (("min", "最短块（均值）"), ("p25", "p25"), ("med", "中位"),
                   ("p75", "p75"), ("max", "最长块（均值）")):
        print(f"  {lab:<26}{agg_len(rows_m, k):>12.0f}{agg_len(rows_p, k):>12.0f}")
    for k, lab in (("short100", "<100 字（近空）"), ("short200", "<200 字"),
                   ("long4000", ">4000 字（超限）"), ("multipage", "跨页块"),
                   ("with_part", "带 part（子块）")):
        print(f"  {lab:<26}{agg(rows_m, k):>12.1f}{agg(rows_p, k):>12.1f}")

    print(f"\n【B 自包含性】（占全部块的比例 %）")
    print(f"  {'指标':<28}{'MinerU':>12}{'pymupdf':>12}   说明")
    notes = {"lead_ref": "首句悬空指代（this/it/该/其）→ 离开上文读不懂",
             "lead_conj": "首句连接词（however/therefore）→ 依赖上文",
             "lead_cn": "首句中文指代/连接",
             "has_title": "块内含所属标题行 → 自带上下文",
             "eq_only": "孤立公式块（几乎无正文）"}
    for k in ("lead_ref", "lead_conj", "lead_cn", "has_title", "eq_only"):
        m = sum(r.get(k, 0) for r in rows_m)
        p = sum(r.get(k, 0) for r in rows_p)
        print(f"  {k:<28}{m / max(tot_m,1) * 100:>11.1f}%{p / max(tot_p,1) * 100:>11.1f}%"
              f"   {notes[k]}")
    tm = sum(r.get("table_blocks", 0) for r in rows_m)
    tp = sum(r.get("table_blocks", 0) for r in rows_p)
    cm = sum(r.get("table_with_cap", 0) for r in rows_m)
    cp = sum(r.get("table_with_cap", 0) for r in rows_p)
    print(f"  {'表格块带 caption':<26}{cm / max(tm,1) * 100:>11.1f}%"
          f"{cp / max(tp,1) * 100:>11.1f}%   （共 {tm} / {tp} 个表格块）")

    have = [r for r in rows_m if "intra_cos" in r]
    havep = [r for r in rows_p if "intra_cos" in r]
    if have:
        print(f"\n【C 主题一致性】（抽样 {len(have)} 篇，cosine 越高越好；边界项越低越好）")
        print(f"  {'指标':<26}{'MinerU':>12}{'pymupdf':>12}")
        for k, lab in (("intra_cos", "块内前后半 cos（内部一致）"),
                       ("chunk_title_cos", "块↔节标题 cos（归属正确）"),
                       ("adj_cos", "块↔前邻 cos（低=边界干净）")):
            print(f"  {lab:<26}{agg(have, k):>12.3f}{agg(havep, k):>12.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
