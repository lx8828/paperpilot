"""**三方判定对比**：NLI（279M） vs glm-4-flash（弱裁判） vs **deepseek-chat（主链路，更强）**。

## 为什么要做
上一轮我用 `glm-4-flash` 当"参考标注"否定了本地 NLI —— 但**裁判本身也是弱模型**，
用户质疑得对。主链路其实配着 **`deepseek-chat`**（明显强于 glm-4-flash）。
所以：**用 deepseek-chat 重标同一批 60 条**，然后看
- 两个 LLM 之间是否一致（若一致 → "NLI 错"的结论变稳）
- 若两个 LLM 也互相打架 → 说明需要真正的人工金标，而不是换模型

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_judge2.py
"""
from __future__ import annotations

import importlib.util as _iu
import json
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

spec = _iu.spec_from_file_location("cl", HERE / "tmp" / "_r2_calib_llm.py")
_cl = _iu.module_from_spec(spec)
sys.modules["cl"] = _cl
spec.loader.exec_module(_cl)

SRC = HERE / "data" / "r2dev" / "calib2.csv"
OUT = HERE / "data" / "r2dev" / "calib2_3judge.csv"


def main() -> int:
    df = pd.read_csv(SRC)
    print(f"样本 {len(df)} 条。用 deepseek-chat（主链路）重标…", flush=True)
    rows = []
    for i, r in df.iterrows():
        rows.append(_cl.label_one(str(r["论断"]), str(r["证据"]), "PAPERPILOT_LLM"))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(df)} …", flush=True)
    llm2 = pd.DataFrame(rows).rename(columns={"LLM判定": "DS判定", "LLM置信度": "DS置信度",
                                              "LLM理由": "DS理由"})
    df = pd.concat([df, llm2], axis=1)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    u = llm.usage_stats()
    print(f"完成（token: prompt {u['prompt_tokens']} / completion {u['completion_tokens']}）→ {OUT}")

    ok = df[(df["LLM判定"] != "ERROR") & (df["DS判定"] != "ERROR")].copy()
    print(f"\n有效 {len(ok)}/{len(df)} 条")
    print("\n【三方判定分布】")
    print("  NLI        ：" + " ｜ ".join(f"{k} {v}" for k, v in ok["NLI_判定"].value_counts().items()))
    print("  glm-4-flash：" + " ｜ ".join(f"{k} {v}" for k, v in ok["LLM判定"].value_counts().items()))
    print("  deepseek   ：" + " ｜ ".join(f"{k} {v}" for k, v in ok["DS判定"].value_counts().items()))

    a_gd = float((ok["LLM判定"] == ok["DS判定"]).mean())
    print(f"\n【两个 LLM 之间一致率】{a_gd:.3f}")
    nli2 = ok["NLI_判定"].map(lambda s: "entail" if "entail" in s else
                              ("contradict" if "contra" in s else "neutral"))
    print(f"【NLI vs glm-4-flash】{(nli2 == ok['LLM判定']).mean():.3f}")
    print(f"【NLI vs deepseek】  {(nli2 == ok['DS判定']).mean():.3f}")
    print(f"【NLI vs 两者一致时才比】" +
          (f"{float((nli2 == ok['DS判定'])[ok['LLM判定'] == ok['DS判定']].mean()):.3f}"
           if (ok["LLM判定"] == ok["DS判定"]).any() else " 无"))

    # 三方都用 entail/非 entail 二值化（更适合我们的用途）
    for name, col in (("glm-4-flash", "LLM判定"), ("deepseek", "DS判定")):
        y = (ok[col] == "entail").astype(int).values
        print(f"\n【NLI 阈值 vs {name}】正类={int(y.sum())}/{len(y)}")
        print(f"  {'阈值':>6}{'P':>9}{'R':>9}{'F1':>9}")
        for thr in (0.3, 0.5, 0.7, 0.9):
            p = (ok["NLI_entail"] >= thr).astype(int).values
            tp = int(((p == 1) & (y == 1)).sum())
            fp = int(((p == 1) & (y == 0)).sum())
            fn = int(((p == 0) & (y == 1)).sum())
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            print(f"  {thr:>6.1f}{pr:>9.3f}{rc:>9.3f}{(2 * pr * rc / (pr + rc) if pr + rc else 0):>9.3f}")

    print("\n【两个 LLM 打架的样本（需要人工金标）】")
    fight = ok[ok["LLM判定"] != ok["DS判定"]]
    print(f"  {len(fight)} 条")
    for _, r in fight.head(10).iterrows():
        print("-" * 92)
        print(f"[{r['构造标签']}] {r['论断']} ｜ NLI ent={r['NLI_entail']} con={r['NLI_contra']}")
        print(f"  glm-4-flash={r['LLM判定']}（{str(r['LLM理由'])[:40]}）")
        print(f"  deepseek   ={r['DS判定']}（{str(r['DS理由'])[:40]}）")
        print(f"  证据：{re.sub(chr(10), ' ', str(r['证据']))[:260]}…")

    print("\n【三方都判非 entail，但『构造=支持』的样本】")
    t = ok[(ok["构造标签"] == "支持") & (nli2 != "entailment") & (ok["LLM判定"] != "entail")
           & (ok["DS判定"] != "entail")]
    print(f"  {len(t)} 条 → 说明词面锚点误判的比例")
    print("\n【三方都判 entail，但『构造=不支持』的样本（锚点漏判）】")
    t2 = ok[(ok["构造标签"] == "不支持") & (nli2 == "entailment") & (ok["LLM判定"] == "entail")
            & (ok["DS判定"] == "entail")]
    print(f"  {len(t2)} 条")
    for _, r in t2.head(5).iterrows():
        print(f"    · {r['论断']} ｜ 证据：{re.sub(chr(10), ' ', str(r['证据']))[:200]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
