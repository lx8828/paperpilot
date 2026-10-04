"""**人工裁定样本导出**（不依赖任何自动参考标注）+ 打印当前配置的模型名。

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_adjudge.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))
print("【当前配置的模型】")
for pre in ("PAPERPILOT_LLM", "PAPERPILOT_JUDGE"):
    base, key, model = llm.config(pre)
    print(f"  {pre:<18} model={model or '(未配置)'}  base={base or '-'}  key={'set' if key else '-'}")

df = pd.read_csv(HERE / "data" / "r2dev" / "calib2.csv")
ok = df[df["LLM判定"] != "ERROR"].copy()

# 取"分歧"与"双方一致但可疑"的样本，覆盖四种组合
groups = {
    "A NLI高·LLM非entail（NLI疑似过度宽松）": ok[(ok["NLI_entail"] >= 0.7) & (ok["LLM判定"] != "entail")],
    "B NLI低·LLM=entail（NLI疑似漏判）": ok[(ok["NLI_entail"] < 0.3) & (ok["LLM判定"] == "entail")],
    "C 双方一致 entail": ok[(ok["NLI_entail"] >= 0.7) & (ok["LLM判定"] == "entail")],
    "D 双方一致非 entail": ok[(ok["NLI_entail"] < 0.3) & (ok["LLM判定"] != "entail")],
    "E NLI 报矛盾": ok[ok["NLI_contra"] >= 0.5],
}
print("\n【分歧格局】")
for k, g in groups.items():
    print(f"  {k:<34} {len(g)} 条")

N = 3
sel = pd.concat([g.head(N).assign(组=k)
                 for k, g in groups.items() if len(g)]).drop_duplicates(["facet", "docid"])

print(f"\n{'=' * 100}\n【人工裁定样本 {len(sel)} 条】每条的『正确答案』请只依据证据原文判断\n")
for i, (_, r) in enumerate(sel.iterrows(), 1):
    print("=" * 100)
    print(f"[{i}] 组={r['组']}")
    print(f"  论断：{r['论断']}")
    print(f"  构造标签={r['构造标签']}（块内含锚点={r['词面命中']}）｜ 在真值={r['在真值']}")
    print(f"  NLI：entail={r['NLI_entail']} contra={r['NLI_contra']} ｜ "
          f"LLM：{r['LLM判定']}（{str(r['LLM理由'])[:60]}）")
    ev = re.sub(r"\s+", " ", str(r["证据"]))
    print(f"  证据（{len(ev)} 字）：{ev[:900]}")
print("\n" + "=" * 100)
print("裁定口径：证据**直接说明**论文做了这件事 → entail；只是提到相关词但未说明论文做了 → neutral；")
print("          明确说没做/相反 → contradict。")
