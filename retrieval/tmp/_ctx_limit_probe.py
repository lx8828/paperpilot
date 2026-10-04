"""**上下文上限探针**：在 1m 档上，"整档直读"到底可不可行？

做法：拿同一道题，把语料按不同字符预算截断成 prompt，看模型能不能正常作答。
→ 找出"直读能覆盖语料的百分之几"，与"检索只需 ~7k token"对比。
"""
from __future__ import annotations

import importlib.util as _iu
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "tmp"))
spec = _iu.spec_from_file_location("qr", HERE / "tmp" / "_qampari_run.py")
qr = _iu.module_from_spec(spec)
sys.modules["qr"] = qr
spec.loader.exec_module(qr)

DATA = HERE / "data" / "loft" / "qampari" / "1m"
corpus = [json.loads(l) for l in (DATA / "corpus.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]
queries = [json.loads(l) for l in (DATA / "test_queries.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]
full = [str(c.get("title_text") or "") + " \n " + str(c.get("passage_text") or "")
        for c in corpus]
pid = [str(c.get("pid")) for c in corpus]
total_chars = sum(len(x) for x in full)

blocks, acc = [], 0
for i, t in enumerate(full):
    blocks.append(f"ID: {pid[i]} | TITLE: {corpus[i].get('title_text')} | CONTENT: {t}")
    acc += len(blocks[-1]) + 1
    if acc > 1_200_000:
        break
ALLB = "\n".join(blocks)
print(f"1m 档语料合计 {total_chars:,} 字符 ≈ {total_chars / 4 / 1000:.0f}k token"
      f" ｜ 已构建前 {acc:,} 字符的块串")

reader, RUSE = qr.make_reader(True, "deepseek-chat", 600)
q = queries[0]
print(f"\n探针问题：{q['query_text']}\n")
print(f"  {'预算(字符)':>12}{'≈token':>9}{'占语料':>8}  结果")
for budget in (30_000, 100_000, 200_000, 400_000, 600_000, 900_000):
    ctx = ALLB[:budget]
    p = (f"{qr.CORPUS_INSTRUCTION}\n\n{qr.FORMATTING}\n\n{ctx}\n\n"
         + qr.QUERY_COT.format(query=q["query_text"]))
    a = reader(p)
    n = len(qr.parse_answers(a))
    ok = "✅ 有输出" if a.strip() else "❌ 空输出（超上下文/被截断）"
    print(f"  {budget:>12,}{budget / 4 / 1000:>9.0f}k{budget / total_chars:>8.1%}  "
          f"{ok} ｜ 解析出 {n} 个答案")
    if not a.strip() and budget >= 400_000:
        break
print(f"\n  检索臂对照：prompt ≈ **27,000 字符（7k token，占语料 0.8%）** → 全部有输出")
