"""**第三方裁判**：用更强的模型（优先 `deepseek-reasoner`，即 R1 推理版）复标，估计 LLM 判定的绝对可靠性。

## 为什么
前面只有"glm-4-flash vs deepseek-chat"两个 LLM（虽异源，但都属于"小/中"量级）。
用户问"LLM 效果够不够好" → 需要一个**更强的参照**才能回答。

## 做法
1. 先探测 `deepseek-reasoner` 是否可用（1 次调用）
2. 可用则对同一批 60 条复标 → 三方一致率（chat / reasoner / NLI）
3. 输出"两个 LLM 一致 vs 不一致"的分档，并给出**判定噪声对下游指标的影响估计**

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_judge3.py
"""
from __future__ import annotations

import importlib.util as _iu
import os
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

SRC = HERE / "data" / "r2dev" / "calib2_3judge.csv"
OUT = HERE / "data" / "r2dev" / "calib2_4judge.csv"


def probe(model: str) -> bool:
    saved = os.environ.get("PAPERPILOT_LLM_MODEL")
    os.environ["PAPERPILOT_LLM_MODEL"] = model
    try:
        r = _cl.label_one("该论文做了消融实验", "We perform an ablation study of each component.",
                          "PAPERPILOT_LLM")
        ok = r["LLM判定"] != "ERROR"
        print(f"  探测 {model}: {'可用' if ok else '不可用'} → {r['LLM判定']} {str(r['LLM理由'])[:40]}")
        return ok
    finally:
        if saved is None:
            os.environ.pop("PAPERPILOT_LLM_MODEL", None)
        else:
            os.environ["PAPERPILOT_LLM_MODEL"] = saved


def main() -> int:
    df = pd.read_csv(SRC)
    print("【探测更强的裁判】")
    strong = None
    for m in ("deepseek-reasoner", "deepseek-chat"):
        if probe(m):
            strong = m
            break
    if strong is None:
        print("无可用的更强模型")
        return 1
    print(f"→ 使用 {strong} 作为第三方裁判（{len(df)} 条）")

    saved = os.environ.get("PAPERPILOT_LLM_MODEL")
    os.environ["PAPERPILOT_LLM_MODEL"] = strong
    rows = []
    try:
        for i, r in df.iterrows():
            # 推理模型需要更大的 token 预算（否则思考吃光预算 → content 为空）
            got = _cl.label_one(str(r["论断"]), str(r["证据"]), "PAPERPILOT_LLM", max_tokens=1600)
            if got["LLM判定"] == "ERROR":
                got = _cl.label_one(str(r["论断"]), str(r["证据"]), "PAPERPILOT_LLM", max_tokens=2600)
            rows.append(got)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(df)} …", flush=True)
    finally:
        if saved is None:
            os.environ.pop("PAPERPILOT_LLM_MODEL", None)
        else:
            os.environ["PAPERPILOT_LLM_MODEL"] = saved
    df = pd.concat([df, pd.DataFrame(rows).rename(columns={
        "LLM判定": "ST判定", "LLM置信度": "ST置信度", "LLM理由": "ST理由"})], axis=1)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    u = llm.usage_stats()
    print(f"完成（token prompt {u['prompt_tokens']} / completion {u['completion_tokens']}）→ {OUT}")

    nli2 = df["NLI_判定"].map(lambda s: "entail" if "entail" in s else
                              ("contradict" if "contra" in s else "neutral"))
    print("\n【三方/四方判定分布】")
    print("  NLI        ：" + " ｜ ".join(f"{k} {v}" for k, v in df["NLI_判定"].value_counts().items()))
    print("  glm-4-flash：" + " ｜ ".join(f"{k} {v}" for k, v in df["LLM判定"].value_counts().items()))
    print("  ds-chat    ：" + " ｜ ".join(f"{k} {v}" for k, v in df["DS判定"].value_counts().items()))
    print(f"  {strong:<10}：" + " ｜ ".join(f"{k} {v}" for k, v in df["ST判定"].value_counts().items()))

    valid = df[(df["LLM判定"] != "ERROR") & (df["DS判定"] != "ERROR") & (df["ST判定"] != "ERROR")]
    print(f"\n【三裁判均有效】{len(valid)}/{len(df)} 条（ERROR {len(df) - len(valid)}）")
    print("【两两一致率（仅在均有效的样本上）】")
    pairs = [("glm-4-flash", "LLM判定", "ds-chat", "DS判定"),
             (f"{strong}", "ST判定", "ds-chat", "DS判定"),
             (f"{strong}", "ST判定", "glm-4-flash", "LLM判定")]
    for n1, c1, n2, c2 in pairs:
        sub = valid[(valid[c1] != "ERROR") & (valid[c2] != "ERROR")]
        print(f"  {n1:<20} vs {n2:<12} {(sub[c1] == sub[c2]).mean():.3f}（n={len(sub)}）")
    nliv = valid[valid["NLI_判定"].notna()]
    for n1, c1 in ((f"{strong}", "ST判定"), ("ds-chat", "DS判定"), ("glm-4-flash", "LLM判定")):
        bn = nliv["NLI_判定"].map(lambda s: "entail" if "entail" in s else
                                  ("contradict" if "contra" in s else "neutral"))
        print(f"  NLI                  vs {n1:<12} {(bn == nliv[c1]).mean():.3f}（n={len(nliv)}）")

    # 判定噪声对下游的影响：以"两个较强 LLM 一致"为金标
    gold = valid[valid["DS判定"] == valid["ST判定"]]
    print(f"\n【以「两个较强 LLM 一致」为金标】{len(gold)}/{len(valid)} 条有效样本"
          f"（覆盖 {len(gold) / max(len(valid), 1):.0%}）")
    y = (gold["DS判定"] == "entail").astype(int).values
    for nm, col in (("glm-4-flash", "LLM判定"),):
        p = (gold[col] == "entail").astype(int).values
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        print(f"  {nm:<12} precision {pr:.3f} ｜ recall {rc:.3f} ｜ F1 {2 * pr * rc / max(pr + rc, 1e-9):.3f}")
    for thr in (0.3, 0.5, 0.7, 0.9):
        p = (gold["NLI_entail"] >= thr).astype(int).values
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        print(f"  NLI@{thr:<5.1f}    precision {pr:.3f} ｜ recall {rc:.3f}"
              f" ｜ F1 {2 * pr * rc / max(pr + rc, 1e-9):.3f}")

    print(f"\n【两个较强 LLM 仍打架的样本（{int((df['DS判定'] != df['ST判定']).sum())} 条）】")
    for _, r in df[df["DS判定"] != df["ST判定"]].head(8).iterrows():
        print("-" * 92)
        print(f"[{r['构造标签']}] {r['论断']} ｜ NLI={r['NLI_判定']}(ent {r['NLI_entail']})")
        print(f"  ds-chat ={r['DS判定']}（{str(r['DS理由'])[:52]}）")
        print(f"  {strong} ={r['ST判定']}（{str(r['ST理由'])[:52]}）")
        print(f"  证据：{re.sub(chr(10), ' ', str(r['证据']))[:220]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
