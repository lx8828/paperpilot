"""**探针**：LitSearch `full_paper` 纯文本里到底有没有章节结构？

这决定"能否复用生产的 `chunk_document`（标题树 + 段落二级切分）"：
· `chunk_document` → `find_headings(blocks)` → 打分依赖**版面行高**（`doc_text_h_median`）
· LitSearch 只有纯文本 → 必须造**伪 blocks**（无版面）
· 文档说「**编号标题（L1/L2/L3）rest 校验通过后强制判定**」→ 若文本保留编号标题，无需行高也能切
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CACHE = HERE / "data" / "r2dev" / "clusters"
sub = pd.read_parquet(CACHE / "c0.parquet")
t = str(sub["full_paper"].iloc[0])

print("=" * 108)
print(f"【LitSearch full_paper 结构探针】docid={sub['docid'].iloc[0]} ｜ {len(t):,} 字符")
print("=" * 108)

# ① 段落分隔符是什么？
print("\n① 分隔符统计：")
for pat, nm in (("\n\n", "空行(\\n\\n)"), ("\n", "单换行"), ("\\n\\n\\n+", "多空行")):
    print(f"   {nm:<16}{len(re.findall(pat, t)):>7}")

# ② 疑似章节标题（编号开头 + 短行）
paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
print(f"\n② 按空行切出 **{len(paras)}** 段（中位 {int(pd.Series([len(p) for p in paras]).median())} 字符/段）")

RE_NUM = re.compile(r"^(\d+(?:\.\d+){0,2})\.?\s+(\S.{0,80})$")
RE_L1 = re.compile(r"^(\d+)\.?\s+([A-Z].{2,80})$")
cands = []
for i, p in enumerate(paras):
    one = p.replace("\n", " ")
    if len(one) > 110:
        continue
    m = RE_NUM.match(one)
    if m and m.group(2)[:1].isupper():
        cands.append((i, m.group(1), m.group(2)[:56]))
    elif one.rstrip(".") in ("Abstract", "ABSTRACT", "References", "REFERENCES",
                             "Introduction", "Conclusion", "Acknowledgements"):
        cands.append((i, "-", one[:56]))
print(f"   疑似编号/特殊标题 **{len(cands)}** 个：")
for i, no, txt in cands[:32]:
    lvl = no.count(".") + 1 if no != "-" else 0
    print(f"     [para {i:>4}] {'L' + str(lvl) if lvl else 'SPECIAL':<8} {no:<8}{txt}")

# ③ 前 1200 字符原样
print("\n③ 开头 1200 字符（看是否有版面痕迹/标题格式）：")
print("-" * 108)
print(t[:1200])
print("-" * 108)
print("\n④ 全文里独占行的短行（≤90 字符，=疑似标题）：")
short = [ln.strip() for ln in t.split("\n") if 0 < len(ln.strip()) <= 90]
print(f"   短行（≤90 字符）共 {len(short)} 条 ｜ 例：")
for s in short[:12]:
    print(f"     {s[:80]}")
