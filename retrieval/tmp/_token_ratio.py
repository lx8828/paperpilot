"""实测 **字符→token 换算率**（我此前混用了 /4 与 /4.94，导致 S/C 阶梯数字不一致）。

做法：把真实语料塞给 API，读回 `usage.prompt_tokens` → 得到该文本类型的**真实**比率。
· 论文正文（我们 PDF / LitSearch 全文）
· 维基 passage（QAMPARI）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))


def measure(name: str, text: str) -> None:
    llm.reset_usage()
    n = len(text)
    body = text + "\n\n只回复两个字：可以"
    try:
        out = llm.chat_text("你是测试助手。", body, temperature=0.0, max_tokens=8)
    except Exception as e:  # noqa: BLE001
        print(f"  {name:<30} ❌ {type(e).__name__}: {str(e)[:80]}")
        return
    u = llm.usage_stats()
    pt = int(u["prompt_tokens"])
    print(f"  {name:<30}{n:>11,} 字符{pt:>10,} tok  → **{n / max(pt, 1):.2f} 字符/token**"
          f"  （回显 {out.strip()[:6]!r}）")


print("=" * 100)
print("【实测 字符→token 换算率】deepseek-chat（同一模型、同一 API）")

# ① 论文正文（我们 PDF 的检索视图）
from run_fullctx_qa import group_corpus  # noqa: E402

corpus = group_corpus("group1")
txt = []
for p in corpus:
    from paperpilot.agents.document_cache import retrieval_chunks
    txt += [c.text for c in retrieval_chunks(p)]
measure(f"论文正文·PDF×{len(corpus)}（检索视图）", "\n\n".join(txt))

# ② LitSearch 论文全文（R2 开发线语料）
import pandas as pd  # noqa: E402

sub = pd.read_parquet(HERE / "data" / "r2dev" / "clusters" / "c0.parquet")
measure("LitSearch 论文全文×20 篇", "\n\n".join(sub["full_paper"].astype(str).tolist())[:1_000_000])

# ③ 维基 passage（QAMPARI 128k）
rows = [json.loads(x) for x in (HERE / "data" / "loft" / "qampari" / "128k" /
                                "corpus.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
measure("QAMPARI 维基 passage×755",
        "\n".join(f"ID: {r['pid']} | TITLE: {r['title_text']} | CONTENT: {r['passage_text']}"
                  for r in rows))

# ④ 英文填充（此前我用的 4.94 口径，作对照）
measure("英文填充句（对照）", ("This is a filler sentence used only to occupy context space. "
                            * 20_000)[:1_000_000])
