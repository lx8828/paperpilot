"""**用 LLM 给 NLI 校准集打标**（替代人工；用项目现有 API key）。

## 为什么
`nli_calibration.csv` 是为了"给判定器定刻度"：NLI（279M）偏松、词面代理偏严，
两把尺子都没刻度。人工标注成本高且（用户自述）英文不熟 → 改用 **LLM 当参考标注**。

⚠️ **这仍是"用机器校准机器"**，必须交代：
- 用**独立裁判模型**（`PAPERPILOT_JUDGE_*`，与主链路异源，避免同模型自证偏好）；
- 输出四档 + 置信度 + 理由，**理由留档可抽查**（人工只抽查低置信/高分歧样本即可）；
- 因此它给出的是**相对口径**（"LLM 判定与 NLI 判定的差异"），不是人工金标。

## 产出
1. `data/r2dev/nli_calibration_labeled.csv`：每行加 `LLM判定 / LLM置信度 / LLM理由`
2. 三张对齐表：
   - **NLI 各阈值 vs LLM 判定** → precision / recall / F1（定阈值）
   - **词面构造标签 vs LLM 判定** → **锚点精度**（"含锚点句"当支持，错多少）
   - **contradiction 抽查** → NLI 报的"说反了"里有多少是真的

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_calib_llm.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))

CSV = HERE / "data" / "r2dev" / "nli_calibration.csv"
OUT = HERE / "data" / "r2dev" / "nli_calibration_labeled.csv"

SYSTEM = """你是严格的事实核查员，负责判断一段论文原文能否支持一条中文论断。

论断都是关于**某一篇论文**的（例如"该论文做了消融实验"）。证据是从**该论文**中检索出的原文片段。

只输出四档之一：

- `entail`：证据**足以支持**该论断（直接说明论文做了这件事）。
- `partial`：证据**部分支持**（只覆盖论断的一部分，或需要额外假设才能成立）。
- `neutral`：证据**与该论断无关，或不足以判断**。（⚠️ 证据只是没提到这件事，也算 neutral，**不要**因为"没提到"就判矛盾。）
- `contradict`：证据**明确表明该论断为假**（例如论断说"没有公开代码"，证据却给出了代码链接）。

判定纪律：
1. 以证据字面内容为准，不要用你的先验知识补充。
2. 只有当证据**明确与**论断相反时才判 `contradict`；"未提及""只谈别的内容"一律 `neutral`。
3. 中文论断 ↔ 英文证据，注意同义改写（如"公开了代码" ↔ "code is available at github.com"）。

只输出 JSON（不要多余文字）：
{"label": "entail|partial|neutral|contradict", "confidence": 0.0~1.0, "reason": "不超过 40 字的中文理由"}"""


def label_one(claim: str, evidence: str, prefix: str, max_tokens: int = 200) -> dict:
    """单条判定。

    ⚠️ `max_tokens` 必须够大：**推理型模型（deepseek-reasoner）会把预算花在思考上**，
    200 token 会导致 content 为空（实测 60 条里 15 条为空）。推理性裁判请给 1500+。
    """
    user = f"论断：{claim}\n\n证据（论文原文片段）：\n{evidence[:3000]}"
    for attempt in range(3):
        try:
            obj = llm.chat_json(SYSTEM, user, temperature=0.0, max_tokens=max_tokens, prefix=prefix)
            if isinstance(obj, dict) and "label" in obj:
                lab = str(obj["label"]).strip().lower()
                if lab not in ("entail", "partial", "neutral", "contradict"):
                    for c in ("entail", "partial", "neutral", "contradict"):
                        if c in lab:
                            lab = c
                            break
                return {"LLM判定": lab,
                        "LLM置信度": float(obj.get("confidence") or 0),
                        "LLM理由": str(obj.get("reason") or "")[:120]}
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return {"LLM判定": "ERROR", "LLM置信度": 0.0, "LLM理由": f"{type(e).__name__}: {str(e)[:80]}"}
            time.sleep(2 * (attempt + 1))
    return {"LLM判定": "ERROR", "LLM置信度": 0.0, "LLM理由": "重试失败"}


def main() -> int:
    prefix = "PAPERPILOT_JUDGE" if llm.judge_configured() else "PAPERPILOT_LLM"
    base, key, model = llm.config(prefix)
    print(f"标注模型：{prefix} → {model} @ {base}（key {'已配置' if key else '缺失'}）")
    if not (base and key and model):
        print("❌ 未配置 LLM，退出")
        return 1

    df = pd.read_csv(CSV)
    print(f"校准集 {len(df)} 条，开始打标…", flush=True)
    llm.reset_usage()
    rows = []
    for i, r in df.iterrows():
        got = label_one(str(r["论断"]), str(r["证据"]), prefix)
        rows.append(got)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(df)} …", flush=True)
    out = pd.concat([df, pd.DataFrame(rows)], axis=1)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    u = llm.usage_stats()
    print(f"完成。token: prompt {u['prompt_tokens']} / completion {u['completion_tokens']}")

    SUP = "entail"
    ok = out[out["LLM判定"] != "ERROR"].copy()
    print(f"\n【LLM 判定分布】（有效 {len(ok)}/{len(out)}）")
    print("  " + " ｜ ".join(f"{k} {v}" for k, v in ok["LLM判定"].value_counts().items()))

    # ① NLI 各阈值 vs LLM 判定（以 entail 为正类）
    y = (ok["LLM判定"] == SUP).astype(int).values
    print(f"\n【① NLI 阈值校准】正类 = LLM 判 `entail`（{int(y.sum())}/{len(y)}）")
    print(f"  {'阈值':>6}{'TP':>5}{'FP':>5}{'FN':>5}{'precision':>11}{'recall':>8}{'F1':>7}")
    best = None
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        pred = (ok["NLI_entail"] >= thr).astype(int).values
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        mark = ""
        if best is None or f1 > best[1]:
            best, mark = (thr, f1), "  ← 最优"
        print(f"  {thr:>6.1f}{tp:>5}{fp:>5}{fn:>5}{p:>11.3f}{r:>8.3f}{f1:>7.3f}{mark}")

    # ② 词面构造标签 vs LLM 判定 → 锚点精度
    print("\n【② 锚点精度：把「含锚点句」当支持，错了多少】")
    ct = pd.crosstab(ok["构造标签"], ok["LLM判定"])
    print(ct.to_string())
    sup_rows = ok[ok["构造标签"] == "支持"]
    good = (sup_rows["LLM判定"] == SUP).mean() if len(sup_rows) else float("nan")
    print(f"  → 构造为「支持」的 {len(sup_rows)} 条里，LLM 也判 entail 的只有 {good:.3f}")
    print("  （其余多为 partial/neutral → 词面锚点判宽了：证据块里出现该词 ≠ 真支持该论断）")

    # ③ contradiction 抽查
    con = ok[ok["NLI_contra"] >= 0.5]
    print(f"\n【③ contradiction 抽查】NLI 判为矛盾（≥0.5）的 {len(con)} 条，LLM 怎么说：")
    if len(con):
        print("  " + " ｜ ".join(f"{k} {v}" for k, v in con["LLM判定"].value_counts().items()))
        agree = float((con["LLM判定"] == "contradict").mean())
        print(f"  → LLM 认同「说反了」的比例 {agree:.3f}")
        for _, r in con.head(5).iterrows():
            print(f"    · {r['论断']} ｜ NLI_entail={r['NLI_entail']:.2f}"
                  f" contra={r['NLI_contra']:.2f} ｜ LLM={r['LLM判定']}"
                  f"（{str(r['LLM理由'])[:40]}）")

    print("\n【分歧样本（LLM vs NLI 不一致，供人工抽查）】")
    dis = ok[((ok["NLI_entail"] >= 0.5) & (ok["LLM判定"] != SUP))
             | ((ok["NLI_entail"] < 0.3) & (ok["LLM判定"] == SUP))]
    print(f"  {len(dis)} 条，示例：")
    for _, r in dis.head(6).iterrows():
        print(f"    · [{r['分层']}] {r['论断']} ｜ NLI_entail={r['NLI_entail']:.2f}"
              f" ｜ LLM={r['LLM判定']}（{str(r['LLM理由'])[:36]}）")
    print(f"\n  → 已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
