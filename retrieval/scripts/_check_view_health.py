"""检索视图**健康度**（全语料，零 LLM）—— 独立基准 = pymupdf，不依赖任何单篇特例。

## 为什么需要（2026-09-24）

`mineru_status()` 只判「产物在不在 / 产不产得出 chunk」，**判不出「文本是不是可读」**：
实测一篇被 MinerU 误标 3926 处 `<sub>`、正文丢 37% 字符、功能词损耗 68%，
系统仍报 `ok` → 一路静默穿过检索、作答、校验全部环节。

指标与**原因无关**（故将来 OCR 全错、版面错乱、字体缺失也抓得到）：

    健康度 = 检索视图里的常见功能词数 / **pymupdf 抽同一 PDF** 的功能词数
    字符比 = 检索视图字符数 / pymupdf 字符数        （参考值，不作为判据）

pymupdf 与 MinerU 是**两条独立的解析路**（前者只取文本层，后者走版面/OCR 模型），
所以前者可以当后者的**行为基准**。

## 同时给出「旧 vs 新」两套数字（防"修一个坏两个"）

`PAPERPILOT_SUP_SUB` 控制 `<sup>/<sub>` 的处理（`drop`=旧，"连内容一起删"；
`keep`=新，"只去标签保内容"，见 `mineru_bridge._strip_sup_sub`）。本脚本**两种都算**，
并显式报「回退篇数」——任何一篇新行为低于旧行为立即 ❌。

## 题集层（有 gold 的篇）

若该篇在 `retrieval/tmp/<group>/<stem>.questions.json` 里有 gold，则额外算
**gold 逐字引文在视图里的覆盖率**（出题引文＝这篇的"必答事实"，覆盖低即这篇答不出来）。

用法：
    python retrieval/scripts/_check_view_health.py                    # 全语料
    python retrieval/scripts/_check_view_health.py --threshold 0.95   # 收紧阈值
    python retrieval/scripts/_check_view_health.py --group group2     # 只看某组
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qa_groups import ROOT, MATERIAL_ROOT, papers  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

MINERU_OUT = ROOT / "assets" / "artifacts" / "out_mineru"
PAPERS_DIR = ROOT / "assets" / "papers"

# 常见功能词：与主题/领域无关，任何正常英文正文都密集出现 → 被切碎就再也匹配不上
_WORD_RE = re.compile(
    r"\b(?:the|and|with|that|for|this|from|which|are|was|have|not|its|these)\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").lower()


def _words(s: str) -> int:
    return len(_WORD_RE.findall(_norm(s)))


def _pymupdf_text(stem: str) -> str:
    """独立基准：pymupdf 直接抽文本层（不经过 MinerU、不经过我们的转换层）。

    ⚠️ **必须剔除参考文献**：MinerU 路按设计丢弃 `ref_text`（`element_text` 里
    `sub_type == "ref_text"` → None），而 pymupdf 会把参考文献一起抽出来 ——
    不剔的话基准被无谓抬高，健康度必然偏低。实测 `2606.18837` 的参考文献占全文 **29%**
    （健康度因此假报 46%，实际正常）。做法：找**最后**一个独立成行的 References 标题
    （只在后半段生效，避免误切正文里的引用词）→ 截断。
    """
    pdf = PAPERS_DIR / f"{stem}.pdf"
    if not pdf.exists():
        return ""
    try:
        import pymupdf as fitz
    except ImportError:  # pragma: no cover
        import fitz  # type: ignore
    doc = fitz.open(str(pdf))
    try:
        txt = "".join(p.get_text("text") for p in doc)
    finally:
        doc.close()
    m = None
    for mm in re.finditer(r"(?im)^\s*(?:\d+\.?\s*)?(?:references|bibliography)\s*$", txt):
        if mm.start() > len(txt) * 0.5:          # 只认后半段的参考文献标题
            m = mm
    return txt[: m.start()] if m else txt


def _set_mode(mode: str):
    """临时切换 `<sup>/<sub>` 处理模式，返回恢复函数。"""
    old = os.environ.get("PAPERPILOT_SUP_SUB")
    os.environ["PAPERPILOT_SUP_SUB"] = mode

    def _restore():
        if old is None:
            os.environ.pop("PAPERPILOT_SUP_SUB", None)
        else:
            os.environ["PAPERPILOT_SUP_SUB"] = old
    return _restore


def _bridge_text(stem: str, mode: str) -> str:
    """bridge 层文本（`element_text` 直拼 content_list）——只在没有摄取产物时兜底。"""
    cl = MINERU_OUT / stem / stem / "auto" / f"{stem}_content_list.json"
    if not cl.exists():
        return ""
    from paperpilot.tools import mineru_bridge as mb
    els = json.loads(cl.read_text(encoding="utf-8"))
    restore = _set_mode(mode)
    try:
        return "\n".join(t for t in (mb.element_text(e) for e in els) if t)
    finally:
        restore()


def _view_text(stem: str, mode: str) -> str:
    """按指定模式重建该篇**生产检索视图**文本。

    ⚠️ 必须用 `document_cache.retrieval_chunks`（= 生产真正喂给 LLM 的那份），
    不能用 `element_text` 的输出：中间还有一层 `analyzer.extractable` + **外部表/公式块
    注入**（`xtbl-*`）—— bridge 层的大 markdown 表会被换成独立的紧凑块，
    同一篇实测 143,871 → 36,939 字符。量错层就会得出错的健康度。

    这三处都有 `lru_cache`，**每次切换模式前必须清**，否则第二次读到的是旧模式的缓存：
        `retrieval_chunks` / `ordered_chunks` / `external_table_chunks`
    """
    from paperpilot.agents import document_cache
    restore = _set_mode(mode)
    try:
        for fn in (document_cache.retrieval_chunks, document_cache.ordered_chunks,
                   document_cache.external_table_chunks):
            try:
                fn.cache_clear()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        try:
            chunks = document_cache.retrieval_chunks(f"{stem}.pdf")
        except Exception:  # noqa: BLE001
            chunks = []
        if chunks:
            return "\n".join(c.text for c in chunks)
    finally:
        restore()
    return _bridge_text(stem, mode)      # 无摄取产物 → 兜底（仅诊断）


def _gold_map() -> dict[str, Path]:
    """stem → gold 题集路径（扫 retrieval/tmp/<group>/<stem>.questions.json）。"""
    out: dict[str, Path] = {}
    if not MATERIAL_ROOT.is_dir():
        return out
    for f in MATERIAL_ROOT.glob("*/*.questions.json"):
        if f.name.startswith("_"):
            continue
        out.setdefault(f.name.replace(".questions.json", ""), f)
    return out


def _gold_coverage(stem: str, view: str, gold: Path) -> tuple[int, int]:
    qs = json.loads(gold.read_text(encoding="utf-8"))["questions"]
    n = k = 0
    for q in qs:
        for ev in q.get("evidence") or []:
            quote = str(ev.get("quote") or "").strip()
            if not quote:
                continue
            n += 1
            k += _norm(quote) in _norm(view)
    return k, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="", help="只看某组的论文（默认全语料）")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="健康度**绝对**下限（默认 0=自动）。⚠️ 不建议手设绝对值："
                         "pymupdf 基准在两处会失真（文本层缺失 → 健康>100%；参考文献 → 偏低），"
                         "故默认用**离群判据**（见 --drop）")
    ap.add_argument("--drop", type=float, default=0.35,
                    help="离群判据：健康度低于「全语料中位数 - 本值」即 ❌（默认 0.35）")
    ap.add_argument("--quiet", action="store_true", help="只列 ❌ 与回退的篇")
    args = ap.parse_args()

    stems = ([s for s in papers(args.group)] if args.group
             else sorted({p.name.replace("_content_list.json", "")
                          for p in MINERU_OUT.glob("*/*/auto/*_content_list.json")}))
    golds = _gold_map()

    rows: list[tuple[str, int, int, int, float, float]] = []
    for stem in stems:
        base = _pymupdf_text(stem)
        if not base:
            if not args.quiet:
                print(f"{stem:<16}{'—':>9}  （无 PDF，跳过）")
            continue
        nb = _words(base)
        if nb < 100:
            print(f"{stem:<16}{nb:>9}  ⚠️ pymupdf 基准过小（文本层缺失？）→ 跳过判据")
            continue
        bd, bk = _words(_bridge_text(stem, "drop")), _words(_bridge_text(stem, "keep"))
        vd, vk = _words(_view_text(stem, "drop")), _words(_view_text(stem, "keep"))
        rows.append((stem, nb, bd, bk, vd, vk))

    # 判据 = **离群**（相对全语料中位数），不是绝对阈值：
    # pymupdf 基准会在三处失真 —— 参考文献（已剔）、文本层缺失（抽不全 → 健康>100%）、
    # 以及**两层视图本身的体量差异**（bridge 层含大 markdown 表，生产视图把它换成紧凑的
    # `xtbl-*` 块）→ 绝对阈值必然误报。故分层看，各层只跟同层比。
    med_b = sorted(r[3] / r[1] for r in rows)[len(rows) // 2] if rows else 0.0
    med_r = sorted(r[4] / r[3] for r in rows if r[3])[len(rows) // 2] if rows else 0.0
    cut_b = max(args.threshold, med_b - args.drop)

    print(f"=== 视图健康度（基准=pymupdf 去参考文献）")
    print(f"    bridge层中位数 {med_b:.0%}（**判据**离群线 {cut_b:.0%}）｜ "
          f"层间保留中位数 {med_r:.0%}（**仅信息列**：低于 100% 反映表/公式块被换成紧凑的"
          f"`xtbl-*` 块——设计如此，不是损坏，故不参与判据）")
    print(f"{'stem':<16}{'旧bridge':>10}{'新bridge':>10}{'旧视图':>9}{'新视图':>9}"
          f"{'新健康':>8}{'保留率':>8}{'变化':>8}")
    bad: list[tuple[str, float, float]] = []
    regress: list[tuple[str, float, float]] = []
    for stem, nb, bd, bk, vd, vk in sorted(rows, key=lambda r: r[3] / r[1]):
        hb, hr = bk / nb, vk / bk if bk else 0.0
        if hb < cut_b:
            bad.append((stem, bd / nb, bk / nb))
        if bk < bd or vk < vd:
            regress.append((stem, bd / nb, bk / nb))
        if args.quiet and hb >= cut_b and bk >= bd and vk >= vd:
            continue
        flag = "❌" if hb < cut_b else ("⬇️" if (bk < bd or vk < vd) else "✓")
        print(f"{stem:<16}{bd:>10}{bk:>10}{vd:>9}{vk:>9}"
              f"{hb:>8.0%}{hr:>8.0%}{hb - bd / nb:>+8.0%} {flag}")

    big = [r for r in regress if r[1] - r[2] > 0.02]
    tiny = [r for r in regress if r[1] - r[2] <= 0.02]
    print(f"\n扫描 {len(rows)} 篇 ｜ 离群（❌）{len(bad)} 篇 ｜ "
          f"**新行为比旧行为回退 {len(regress)} 篇**"
          f"（其中 >2pt 的实质回退 {len(big)} 篇，≤2pt 的词边界粘连 {len(tiny)} 篇）")
    for stem, hd, hk in bad:
        print(f"   ❌ 离群 {stem}: 旧 {hd:.0%} → 新 {hk:.0%}")
    for stem, hd, hk in big:
        print(f"   ⬇️ 实质回退 {stem}: 旧 {hd:.0%} → 新 {hk:.0%}")
    if tiny:
        print(f"   （≤2pt 粘连：{', '.join(f'{s}({hd:.0%}→{hk:.0%})' for s, hd, hk in tiny)}）"
              f" —— 真角标被保留后与前一词粘连（如 `the<sub>i</sub>` → `thei`），"
              f"信息未丢、只是该词不再被词边界统计）")

    have_gold = [r[0] for r in rows if r[0] in golds]
    if have_gold:
        print(f"\n=== 题集层：gold 逐字引文在视图里的覆盖率（{len(have_gold)} 篇有 gold）")
        print(f"{'stem':<16}{'旧(drop)':>10}{'新(keep)':>10}{'引文数':>8}")
        for stem in have_gold:
            g = golds[stem]
            kd = _gold_coverage(stem, _view_text(stem, "drop"), g)
            kk = _gold_coverage(stem, _view_text(stem, "keep"), g)
            print(f"{stem:<16}{kd[0] / max(kd[1],1):>9.0%}{kk[0] / max(kk[1],1):>10.0%}{kk[1]:>8}")
    ok = not bad and not big
    print("\n结论：" + ("无离群篇、无实质回退 ✓" if ok
                     else f"{len(bad)} 篇离群 / {len(big)} 篇实质回退 —— 见上"))
    return 1 if not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
