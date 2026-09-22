"""第 1 步：拼装 LitSearch 语料的「检索文本表」+ 校验 qrels 覆盖。

**数据源选择（2026-09-18 实测后确定）**

LitSearch 提供同一批 64,183 篇的两个 config：

| config | 内容 | 问题 |
|---|---|---|
| `corpus_clean` | `title / abstract / citations / full_paper`（**extracted**） | 有 6,760 篇 title 为空、6,197 篇双空 —— 字段提取失败存成**空字符串**（不是 null） |
| **`corpus_s2orc`** | `content.text` + **`content.annotations`（字符级偏移）** + `year` + `externalids` | **无此问题** |

`corpus_s2orc` 的 annotations 给出 `title` / `abstract` 等实体的 **(start,end) 偏移**，
直接从 `content.text` 精确切片即可 —— 这是**标注**而非重新抽取，实测精度 100%
（对比：按版式启发式猜标题，人工核对精度仅约 53%）。

⚠️ 一个坑：`title` 标注可能是**多个 span**（如 `[{1,80},{1232,1311}]`，
第二条是页眉/重复标题）。取 min~max 会**把作者块也切进来** → 因此**只取最早的那个 span**。

**顺带拿到的两个能力**（之前判定为缺失）：
- `year`      → 元数据过滤（"2024 年以后的进展"）可做；
- `externalids.arxiv` → 可挂 arXiv 链接 / 将来取 PDF。

产物（`retrieval/data/litsearch/derived/`，在 .gitignore 的 `data/` 下）：
    corpus_text.parquet   corpusid / title / abstract / year / arxiv / text / n_chars / src
    corpus_build.json     构建报告（覆盖率、来源统计、qrels 校验）

用法：
    python retrieval/scripts/build_corpus.py
    python retrieval/scripts/build_corpus.py --limit-shards 1   # 快测
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
LS = HERE / "data" / "litsearch"
S2_DIR = LS / "corpus_s2orc"
CLEAN_DIR = LS / "corpus_clean"
QFILE = LS / "query" / "full-00000-of-00001.parquet"
OUT_DIR = LS / "derived"

COLS = ["corpusid", "title", "abstract"]
_WS = re.compile(r"\s+")


def norm(s) -> str:
    return _WS.sub(" ", str(s if s is not None else "")).strip()


# ── s2orc annotations 解析 ──────────────────────────────────────
def ann_spans(v) -> list[tuple[int, int]]:
    """把 annotations 的某个字段归一成按 start 升序的 (start,end) 列表。

    实测该字段是 **JSON 字符串**（如 `'[{"end":80,"start":1}]'`），
    但也兼容"已是 list/dict"的情况（不同 datasets 版本）。
    """
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:  # noqa: BLE001
            return []
    if isinstance(v, dict):
        v = [v]
    if not isinstance(v, (list, tuple)):
        return []
    out = []
    for s in v:
        if isinstance(s, dict) and "start" in s and "end" in s:
            try:
                out.append((int(s["start"]), int(s["end"])))
            except (TypeError, ValueError):
                continue
    return sorted(out)


def first_span(text: str, spans: list[tuple[int, int]]) -> str:
    """取最早那个 span 的文本（title 专用：避免把作者块带进来）。"""
    if not spans or not text:
        return ""
    lo, hi = spans[0]
    return norm(text[max(lo, 0):max(hi, 0)])


def join_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """拼接全部 span（abstract 专用：通常 1 个，多个时按序拼）。"""
    if not spans or not text:
        return ""
    parts, seen = [], set()
    for lo, hi in spans:
        s = norm(text[max(lo, 0):max(hi, 0)])
        if s and s not in seen:
            seen.add(s)
            parts.append(s)
    return " ".join(parts)


def read_s2orc(files: list[Path]) -> pd.DataFrame:
    """从 corpus_s2orc 用偏移切出 title/abstract，并取出 year / arxiv。"""
    rows = []
    for f in files:
        df = pd.read_parquet(f, columns=["corpusid", "content", "year", "externalids"])
        for cid, content, year, ext in zip(df["corpusid"], df["content"], df["year"],
                                           df["externalids"]):
            c = content if isinstance(content, dict) else {}
            text = c.get("text") or ""
            ann = c.get("annotations") or {}
            if not isinstance(ann, dict):
                ann = {}
            title = first_span(text, ann_spans(ann.get("title")))
            abstract = join_spans(text, ann_spans(ann.get("abstract")))
            arxiv = ""
            if isinstance(ext, dict):
                arxiv = norm(ext.get("arxiv"))
            rows.append({
                "corpusid": int(cid),
                "title_s2": title,
                "abstract_s2": abstract,
                "year": int(year) if isinstance(year, (int, float)) and year == year else 0,
                "arxiv": arxiv,
            })
        print(f"  [s2orc] {f.name:<38} {len(df):6d} 行")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-shards", type=int, default=0, help="只处理前 N 个分片（快测）")
    args = ap.parse_args()

    s2_files = sorted(S2_DIR.glob("*.parquet"))
    cl_files = sorted(CLEAN_DIR.glob("*.parquet"))
    if not s2_files:
        print(f"缺 corpus_s2orc：{S2_DIR}\n请先运行："
              f"python retrieval/scripts/fetch_litsearch.py corpus_s2orc")
        return 2
    if not cl_files:
        print(f"缺 corpus_clean：{CLEAN_DIR}")
        return 2
    if args.limit_shards:
        s2_files = s2_files[:args.limit_shards]
        cl_files = cl_files[:args.limit_shards]

    print(f"① 读 corpus_s2orc（{len(s2_files)} 分片，用偏移切 title/abstract）")
    s2 = read_s2orc(s2_files)

    print(f"\n② 读 corpus_clean（{len(cl_files)} 分片，作兜底与对照）")
    parts = []
    for f in cl_files:
        d = pd.read_parquet(f, columns=COLS)
        parts.append(d)
        print(f"  [clean] {f.name:<38} {len(d):6d} 行")
    cl = pd.concat(parts, ignore_index=True)
    del parts

    print(f"\n③ 合并（s2orc 为主，clean 兜底）")
    df = cl.merge(s2, on="corpusid", how="outer")
    cl_t = df["title"].map(norm)
    cl_a = df["abstract"].map(norm)
    s2_t = df["title_s2"].map(norm)
    s2_a = df["abstract_s2"].map(norm)
    patched_title = int(((cl_t == "") & (s2_t != "")).sum())
    patched_abstr = int(((cl_a == "") & (s2_a != "")).sum())
    df["title"] = s2_t.where(s2_t != "", cl_t)
    df["abstract"] = s2_a.where(s2_a != "", cl_a)
    df["src"] = ["s2orc" if (t or a) else "clean" for t, a in zip(s2_t != "", s2_a != "")]
    df["text"] = [f"{t}\n{a}" if a else t for t, a in zip(df["title"], df["abstract"])]
    df["n_chars"] = df["text"].str.len()
    df["year"] = df["year"].fillna(0).astype(int)
    df["arxiv"] = df["arxiv"].fillna("")
    order = ["corpusid", "title", "abstract", "year", "arxiv", "text", "n_chars", "src"]
    df = df[order].sort_values("corpusid").reset_index(drop=True)
    if df["corpusid"].duplicated().any():
        n = int(df["corpusid"].duplicated().sum())
        print(f"  ⚠️ 重复 corpusid {n} 个 → 去重")
        df = df.drop_duplicates("corpusid", keep="first").reset_index(drop=True)

    # ── 覆盖率对照 ─────────────────────────────────────────────
    print(f"\n[ 覆盖率：clean 单独 vs 合并后 ]")
    n = len(df)
    for col, patched in (("title", patched_title), ("abstract", patched_abstr)):
        empty_now = int((df[col] == "").sum())
        print(f"  {col:<9} 空值: 合并后 {empty_now:5d} ({100 * empty_now / n:.1f}%)"
              f"  ← 其中 {patched} 篇由 s2orc 的偏移标注补上")
    print(f"  text      空值: {int((df['text'] == '').sum()):5d}"
          f" ({100 * (df['text'] == '').mean():.1f}%)")
    print(f"  year 覆盖: {int((df['year'] > 0).sum())} 篇（{100 * (df['year'] > 0).mean():.1f}%）"
          f" | arxiv 覆盖: {int((df['arxiv'] != '').sum())} 篇"
          f"（{100 * (df['arxiv'] != '').mean():.1f}%）")
    ln = df["n_chars"]
    print(f"  text 长度: 中位 {ln.median():.0f} / 均值 {ln.mean():.0f} / 最大 {ln.max()}"
          f" | token 合计 ≈ {ln.sum() / 3.2 / 1e6:.1f} M")

    # ── qrels 覆盖校验 ─────────────────────────────────────────
    print("\n[ qrels 覆盖校验 ]")
    q = pd.read_parquet(QFILE)
    have = set(int(x) for x in df["corpusid"])
    has_text = set(int(x) for x in df.loc[df["text"] != "", "corpusid"])
    total_gold = missing = notext = 0
    unusable = []
    for i, row in q.iterrows():
        gids = [int(x) for x in row["corpusids"]]
        total_gold += len(gids)
        missing += sum(1 for g in gids if g not in have)
        notext += sum(1 for g in gids if g in have and g not in has_text)
        if all(g not in has_text for g in gids):
            unusable.append(int(i))
    print(f"  语料 {len(df)} 篇（有文本 {len(has_text)}，无文本 {len(df) - len(has_text)}）")
    print(f"  gold 判断 {total_gold} 个 | ① 不在语料 {missing} | ② 在语料但无文本 {notext}")
    print(f"  可用查询 {len(q) - len(unusable)} / {len(q)}（剔除 {len(unusable)} 条）")
    if not missing and not notext:
        print("  → 全部 gold 都有可检索文本：597 条全部可用 ✓")

    # ── 落盘 ───────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "corpus_text.parquet"
    df.to_parquet(out, index=False)
    report = {
        "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "source": "corpus_s2orc (primary, annotation offsets) + corpus_clean (fallback)",
        "n_docs": int(len(df)),
        "docs_with_text": len(has_text),
        "patched_title_from_s2orc": patched_title,
        "patched_abstract_from_s2orc": patched_abstr,
        "empty": {c: int((df[c] == "").sum()) for c in ("title", "abstract", "text")},
        "year_coverage": int((df["year"] > 0).sum()),
        "arxiv_coverage": int((df["arxiv"] != "").sum()),
        "text_chars": {"median": float(ln.median()), "mean": float(ln.mean()),
                       "max": int(ln.max()), "sum": int(ln.sum())},
        "qrels": {"n_queries": int(len(q)), "n_gold": total_gold,
                  "n_gold_missing": missing, "n_gold_no_text": notext,
                  "n_queries_unusable": len(unusable), "unusable_rows": unusable},
    }
    (OUT_DIR / "corpus_build.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出：\n  {out.relative_to(HERE)}  ({out.stat().st_size / 1e6:.1f} MB)"
          f"\n  {(OUT_DIR / 'corpus_build.json').relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
