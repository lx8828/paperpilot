"""**NLI 审计**：我们到底用了什么模型？"NLI 出错"是我的结论，还是我的用法有 bug？

## 三个疑点（按怀疑强度排序）
1. **DeBERTa-v3 的 fast tokenizer 有已知问题** → 若用了 `DebertaV2TokenizerFast`，
   可能整批输入被错误切分 → 表现为"离谱的预测"（如含 "Ablation Study" 的文本给 entail=0.135）。
2. **输入顺序 / 标签顺序**（premise、hypothesis）容易搞反。
3. **参考标注本身就是小模型**（glm-4-flash）→ 用弱模型否定另一个弱模型不可靠。

## 本脚本做
- 读**本地缓存的模型卡**（离线、权威）→ 参数规模 / 底座 / 训练数据 / 用法注意事项
- **fast vs slow tokenizer** 对比：切分结果 + 预测差异
- 在 calib2 的**分歧样本**上跑 4 个 NLI 候选，看是"个别模型错"还是"全错"
- 打印完整证据，供**人工裁定**（不再依赖任何自动参考标注）

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_r2_nli_audit.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()

HERE = Path(__file__).resolve().parents[1]
HF = "F:/hf_cache/hub"
CAL = HERE / "data" / "r2dev" / "calib2.csv"

CANDIDATES = [
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "cross-encoder/nli-deberta-v3-base",
]


def model_card(name: str) -> None:
    key = "models--" + name.replace("/", "--")
    snap = sorted(glob.glob(f"{HF}/{key}/snapshots/*"))
    print("=" * 100)
    print(f"【{name}】")
    if not snap:
        print("  （无本地快照）")
        return
    rm = Path(snap[-1]) / "README.md"
    if rm.exists():
        txt = rm.read_text(encoding="utf-8", errors="replace")
        # 只摘关键信息行
        keep = []
        for line in txt.splitlines():
            low = line.lower()
            if any(k in low for k in ("parameter", "base model", "based on", "trained on",
                                      "train data", "tokenizer", "use_fast", "performance",
                                      "xnli", "mnli", "fever", "anli", "label",
                                      "deberta", "language", "slow", "bug")):
                s = line.strip()
                if s and len(s) < 300 and s not in keep:
                    keep.append(s)
        print("\n".join(keep[:28]))
    else:
        print("  （无 README）")


def params_of(name: str) -> str:
    key = "models--" + name.replace("/", "--")
    for f in glob.glob(f"{HF}/{key}/snapshots/*/*.safetensors"):
        import struct
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            head = json.loads(fh.read(n))
        tot = 0
        for v in head.values():
            if isinstance(v, dict) and "shape" in v:
                p = 1
                for d in v["shape"]:
                    p *= d
                tot += p
        return f"{tot / 1e6:.0f}M 参数（safetensors 统计）"
    for f in glob.glob(f"{HF}/{key}/snapshots/*/pytorch_model.bin"):
        return f"{Path(f).stat().st_size / 1e6:.0f}MB (.bin)"
    return "?"


def main() -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for c in CANDIDATES:
        model_card(c)
        print(f"  → 规模：{params_of(c)}")

    df = pd.read_csv(CAL)
    # 挑出"分歧最大"的 8 条（NLI 高但 LLM 非 entail，或反之，或 NLI 报矛盾）
    dis = df[((df["NLI_entail"] >= 0.7) & (df["LLM判定"] != "entail"))
             | ((df["NLI_entail"] < 0.3) & (df["LLM判定"] == "entail"))
             | (df["NLI_contra"] >= 0.5)].head(8)

    print("\n" + "=" * 100)
    print("【fast vs slow tokenizer 对比】以 mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 为准")
    name = CANDIDATES[0]
    tk_fast = AutoTokenizer.from_pretrained(name, use_fast=True)
    tk_slow = AutoTokenizer.from_pretrained(name, use_fast=False)
    print(f"  fast class = {type(tk_fast).__name__}")
    print(f"  slow class = {type(tk_slow).__name__}")
    r0 = dis.iloc[0]
    prem, hyp = str(r0["证据"])[:600], str(r0["论断"])
    for tag, tk in (("fast", tk_fast), ("slow", tk_slow)):
        enc = tk(prem, hyp, truncation=True, max_length=512)
        ids = enc["input_ids"]
        back = tk.decode(ids, skip_special_tokens=True)
        print(f"  [{tag}] tokens={len(ids)} ｜ 回译是否含论断：{'论断' in back or hyp[:6] in back}")
        print(f"       回译前 160 字：{back[:160]!r}")

    md = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float16).eval().to(dev)
    lab = [md.config.id2label[i].lower() for i in range(md.config.num_labels)]
    i_ent = next(i for i, l in enumerate(lab) if "entail" in l)
    print(f"  id2label = {md.config.id2label}")

    def infer(tk, prem: str, hyp: str) -> np.ndarray:
        b = tk(prem, hyp, return_tensors="pt", truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            return torch.softmax(md(**b).logits.float(), -1)[0].cpu().numpy()

    print("\n【同一条样本：fast vs slow 预测】")
    for _, r in dis.head(5).iterrows():
        pf = infer(tk_fast, str(r["证据"]), str(r["论断"]))
        ps = infer(tk_slow, str(r["证据"]), str(r["论断"]))
        print(f"  {r['论断']:<18} fast[ent {pf[i_ent]:.3f}]  slow[ent {ps[i_ent]:.3f}]"
              f"  ｜ calib2 记录 {r['NLI_entail']:.3f}")
    del md
    if dev == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("【4 个 NLI 候选在分歧样本上的表现】（entail 概率；`—` = 该模型无 entail 标签）")
    print(f"  {'论断':<20}{'LLM':<10}" + "".join(f"{c.split('/')[-1][:16]:>18}" for c in CANDIDATES))
    res = {}
    for c in CANDIDATES:
        tk = AutoTokenizer.from_pretrained(c, use_fast=False)
        md = AutoModelForSequenceClassification.from_pretrained(c, torch_dtype=torch.float16).eval().to(dev)
        lb = [md.config.id2label[i].lower() for i in range(md.config.num_labels)]
        ie = next((i for i, l in enumerate(lb) if "entail" in l), None)
        ic = next((i for i, l in enumerate(lb) if "contra" in l), None)
        vals = []
        for _, r in dis.iterrows():
            b = tk(str(r["证据"]), str(r["论断"]), return_tensors="pt",
                   truncation=True, max_length=512).to(dev)
            with torch.no_grad():
                p = torch.softmax(md(**b).logits.float(), -1)[0].cpu().numpy()
            vals.append((float(p[ie]) if ie is not None else float("nan"),
                         float(p[ic]) if ic is not None else float("nan")))
        res[c] = vals
        del md
        if dev == "cuda":
            torch.cuda.empty_cache()
    for k, (_, r) in enumerate(dis.iterrows()):
        row = "".join(f"{res[c][k][0]:>10.3f}/{res[c][k][1]:>6.3f}" for c in CANDIDATES)
        print(f"  {str(r['论断'])[:18]:<20}{str(r['LLM判定']):<10}{row}")

    print("\n【分歧样本的完整证据（供人工裁定）】")
    for i, (_, r) in enumerate(dis.iterrows()):
        print("=" * 100)
        print(f"[{i}] 论断：{r['论断']} ｜ 构造标签={r['构造标签']} ｜ LLM={r['LLM判定']}"
              f"（{str(r['LLM理由'])[:60]}）")
        print(f"    记录 NLI_entail={r['NLI_entail']} NLI_contra={r['NLI_contra']}")
        print(f"    证据全文（{len(str(r['证据']))} 字）：")
        print("    " + str(r["证据"]).replace("\n", "\n    ")[:1100])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
