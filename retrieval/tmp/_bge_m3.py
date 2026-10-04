"""**S1｜BGE-M3 三路输出自实现**（dense / sparse / colbert），**不装 FlagEmbedding**。

## 为什么自实现
· 本地 `bge-m3` 是 **SentenceTransformer 纯 dense 打包**（`modules.json` = Transformer→Pooling→Normalize，
  权重 391 个张量中 **0 个**含 `sparse`/`colbert`）；
· 官方 `BGEM3FlagModel` 需要 `FlagEmbedding` 包（未安装，装了有动环境风险）；
· 但两个头只是**小线性层**，已单独下载：
    `sparse_linear.pt`  = `Linear(1024→1)`      ← 每 token 一个**标量**权重
    `colbert_linear.pt` = `Linear(1024→1024)`   ← 每 token 一个 1024 维向量（存的是 fp16）
  → 用现有 `transformers` + `torch` 直接加载即可（零新依赖）。

## 三路的定义（与官方一致）
| 路 | 计算 |
|---|---|
| **dense** | `normalize(last_hidden_state[:, 0])`（CLS） |
| **sparse** | `relu(sparse_linear(h))` 得每 token 标量 → **按 token id 做 scatter-max** → `{token_id: w}` |
| **colbert** | `normalize(colbert_linear(h))` → 每 token 一个向量（配 attention_mask） |

⚠️ **sparse 是实现要点**：`sparse_linear` 是 `(1,1024)`，**匹配仍按 token id 重合**
（XLM-R sentencepiece 词表 ~250k）→ 它**不是**"语义同义映射"，
跨语言只在**共享 piece**（数字 / 拉丁缩写 / 共有子词）上生效。**这一点必须实测，不能假设。**

## 用法
  ./.venv/Scripts/python.exe -u retrieval/tmp/_bge_m3.py --smoke     # S1 验收
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parents[1]
DEV = HERE / "data" / "r2dev"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HF = Path(os.environ.get("HF_HOME", r"F:\hf_cache")) / "hub" / "models--BAAI--bge-m3"
VOCAB_SIZE = 250002                       # XLM-R sentencepiece


def _find_snapshot() -> Path:
    """挑含 `config.json` + 权重 + `tokenizer.json` 的快照目录。"""
    snaps = sorted((HF / "snapshots").glob("*"))
    best = None
    for s in snaps:
        has_tok = (s / "tokenizer.json").exists()
        has_cfg = (s / "config.json").exists()
        has_w = (s / "pytorch_model.bin").exists() or (s / "model.safetensors").exists()
        if has_tok and has_cfg and has_w:
            return s
        if has_tok and has_w:
            best = best or s
    if best:
        return best
    raise FileNotFoundError(f"找不到 bge-m3 快照（{HF/'snapshots'}）")


def _head_path(name: str, snap: Path) -> Path:
    for p in (snap / name, *(d / name for d in (HF / "snapshots").glob("*"))):
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到 {name}（需先 hf_hub_download('BAAI/bge-m3', '{name}')）")


class BGEM3:
    """BGE-M3 三路编码器（dense / sparse / colbert）。"""

    def __init__(self, device: str | None = None, fp16: bool = True):
        from transformers import AutoModel, AutoTokenizer

        self.snap = _find_snapshot()
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(str(self.snap))
        try:
            self.enc = AutoModel.from_pretrained(str(self.snap))
        except Exception:                                   # 兜底：ST 打包格式
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(str(self.snap))
            self.enc = AutoModel.from_config(cfg)
            sd = torch.load(str(self.snap / "pytorch_model.bin"), map_location="cpu")
            miss = self.enc.load_state_dict(sd, strict=False)
            print(f"  [兜底加载] missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")
        self.enc.to(self.dev).eval()
        self.sparse_lin = nn.Linear(1024, 1).to(self.dev).eval()
        self.colbert_lin = nn.Linear(1024, 1024).to(self.dev).eval()
        self.sparse_lin.load_state_dict(torch.load(str(_head_path("sparse_linear.pt", self.snap)),
                                                   map_location="cpu"))
        self.colbert_lin.load_state_dict(torch.load(str(_head_path("colbert_linear.pt", self.snap)),
                                                    map_location="cpu"))
        self.fp16 = bool(fp16 and self.dev == "cuda")
        if self.fp16:
            self.enc.half()
            self.sparse_lin.half()
            self.colbert_lin.half()
        # ⚠️ 特殊 token（`<s>` / `</s>` / `<pad>` / `<unk>` / `▁`）在英文块上**权重非零**，
        #    会让**任意** query（含纯中文）都拿到一个**近似常数**的分 → 对排序**只有稀释作用**。
        #    实测：纯中文 query 的 0.051 全部来自 `<s>`+`▁`（共享 token 只有这两个）。
        self.special_ids = set(self.tok.convert_tokens_to_ids(t) for t in
                               ("<s>", "</s>", "<pad>", "<unk>", "▁")
                               if self.tok.convert_tokens_to_ids(t) is not None)

    # ── 编码 ──
    @torch.no_grad()
    def encode(self, texts: list[str], *, max_len: int = 8192, batch: int = 4,
               drop_special: bool = True) -> dict:
        """返回 `{dense:(N,d), sparse:[{tid:w}], colbert:[(L,d)]}`。

        `drop_special`：从 sparse 词袋里剔除特殊 token（见 `special_ids` 的注释）。
        """
        D, S, C = [], [], []
        for i in range(0, len(texts), batch):
            chunk = [str(t) for t in texts[i:i + batch]]
            inp = self.tok(chunk, padding=True, truncation=True, max_length=max_len,
                           return_tensors="pt").to(self.dev)
            h = self.enc(**inp).last_hidden_state                    # (B,L,1024)
            # dense：CLS + 归一化
            d = F.normalize(h[:, 0].float(), dim=-1)
            # sparse：relu(linear) → 按 token id scatter-max
            w = F.relu(self.sparse_lin(h).squeeze(-1).float()) * inp["attention_mask"].float()
            ids = inp["input_ids"]
            for b in range(h.shape[0]):
                v = torch.zeros(VOCAB_SIZE, device=self.dev, dtype=torch.float32)
                v.scatter_reduce_(0, ids[b], w[b], reduce="amax", include_self=True)
                if drop_special:
                    v[torch.tensor(sorted(self.special_ids), device=self.dev)] = 0.0
                nz = v.nonzero(as_tuple=True)[0]
                S.append({int(t): float(v[t]) for t in nz})
            # colbert：linear + 逐 token 归一化
            cb = F.normalize(self.colbert_lin(h).float(), dim=-1)
            m = inp["attention_mask"].bool()
            for b in range(h.shape[0]):
                C.append(cb[b][m[b]].cpu().numpy().astype(np.float32))
            D.append(d.cpu().numpy().astype(np.float32))
        return {"dense": np.concatenate(D, 0), "sparse": S, "colbert": C}

    @staticmethod
    def sparse_score(q: dict[int, float], d: dict[int, float]) -> float:
        """稀疏点积：只走 query 的 key（`O(|q|)`）。"""
        return float(sum(w * d.get(t, 0.0) for t, w in q.items()))

    @staticmethod
    def colbert_score(q: np.ndarray, d: np.ndarray) -> float:
        """MaxSim：每个 query token 取与 doc token 的最大余弦，再求和（都已是单位向量）。"""
        return float((q @ d.T).max(axis=1).sum())


# ─────────────────────── S1 验收（冒烟 + 跨语言核验）───────────────────────
def _smoke() -> int:
    print("=" * 112)
    t0 = time.time()
    m = BGEM3()
    print(f"【S1 验收】快照 {m.snap.name} ｜ 设备 {m.dev} ｜ fp16 {m.fp16}"
          f" ｜ 加载 {time.time() - t0:.1f}s")
    print(f"  sparse_linear={tuple(m.sparse_lin.weight.shape)}"
          f"  colbert_linear={tuple(m.colbert_lin.weight.shape)}"
          f" ｜ 剔除的特殊 token id {sorted(m.special_ids)}")

    # ── 测试块：含代码链接的**最短**块（避免长块稀释信号）──
    import pandas as pd
    best = None
    for ci in range(3):
        p = DEV / "prodchunk" / "mineru" / f"c{ci}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        hit = df[df.text.astype(str).str.contains("github.com", case=False, regex=False)]
        for _, r in hit.iterrows():
            t = " ".join(str(r["text"]).split())
            if best is None or len(t) < len(best[1]):
                best = (str(r["docid"]), t)
    if best is None:
        raise SystemExit("语料里没找到含 github.com 的块")
    doc, ch = best
    pos = ch.lower().find("github.com")
    print(f"\n  测试块：{doc} ｜ {len(ch)} 字符 ｜ github.com 在第 {pos} 字符处")
    print(f"  上下文：…{ch[max(0, pos - 90):pos + 60]}…")

    inv = {v: k for k, v in m.tok.get_vocab().items()}
    out = m.encode([ch])
    sv = out["sparse"][0]
    ranked = sorted(sv.items(), key=lambda x: -x[1])
    rank = {t: i + 1 for i, (t, _) in enumerate(ranked)}
    print(f"\n【验收 1｜sparse 权重是否落在「代码/数字/实体」上】（块内 {len(sv)} 个非零 token）")
    print(f"  {'token':<18}{'权重':>9}{'块内排名':>9}")
    for t in ("▁github", "github", ".", "▁com", "com", "▁code", "▁available",
              "▁release", "▁http", "▁OpenQA", "▁Wikipedia", "25", "▁25"):
        tid = m.tok.convert_tokens_to_ids(t)
        w = sv.get(tid, 0.0)
        if w or t in ("▁github", "▁code"):
            print(f"  {t:<18}{w:>9.3f}{(rank.get(tid) or 0):>9}")
    n_special = len([1 for _, t in ranked if str(inv.get(t, "")).strip("▁") in
                     ("", "<s>", "</s>", "<pad>", "<unk>")])
    print(f"  → 归一化后已剔除特殊 token（残留 {n_special}）；"
          f"top-5：{[inv.get(t, '?') for t, _ in ranked[:5]]}")

    # ── 验收 2：跨语言核验（3 类 query）──
    qs = {"① 纯中文（无任何英文）": "本文有没有公开自己的代码和数据",
          "② L6 英文检索式（生产形态）": "the paper publicly releases its own code and data repository",
          "③ 题面英文短语（dev 形态）": "our code is available at github.com"}
    o = m.encode(list(qs.values()))
    nd = {v: (len(o["colbert"][i]), len(o["sparse"][i]), len(list(o["sparse"][i])))
          for i, v in enumerate(qs.values())}
    print(f"\n【验收 2｜跨语言核验】同一英文块 × 3 类 query")
    print(f"  {'query':<34}{'tok':>4}{'sparse':>9}{'dense':>8}{'colbert':>9}{'colb/√n':>9}"
          f"{'共享tok':>8}")
    for i, (nm, q) in enumerate(qs.items()):
        sp = m.sparse_score(o["sparse"][i], sv)
        dn = float(o["dense"][i] @ out["dense"][0])
        cq = o["colbert"][i]
        cb = m.colbert_score(cq, out["colbert"][0])
        shared = [inv.get(t, "?") for t in o["sparse"][i] if t in sv]
        print(f"  {nm:<34}{len(cq):>4}{sp:>9.3f}{dn:>8.3f}{cb:>9.1f}"
              f"{cb / max(len(cq), 1) ** 0.5:>9.2f}{len(shared):>8}")
        if i == 0:
            print(f"      ① 共享 token 明细：{shared or '（无 → sparse 无法跨语言）'}")
    print("\n  读法：")
    print("   · `dense` 是**语义**分 → ① 与 ③ 应当接近；")
    print("   · `sparse` 是**词法**分（按 token id）→ ① 若**为 0** 即证实「sparse 不能跨语言」；")
    print("   · `colbert` MaxSim 是**逐 token 求和** → 与 query 长度相关，故并列 `colb/√n` 做长度校正。")

    # ── 验收 3：吞吐 ──
    qs2 = [f"query number {i} about code release and ablation studies" for i in range(8)]
    t1 = time.time()
    m.encode(qs2, batch=4)
    print(f"\n【验收 3】8 条短文本编码 {time.time() - t1:.2f}s"
          f"（{m.dev}{'·fp16' if m.fp16 else ''}）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="跑 S1 验收")
    args = ap.parse_args()
    if args.smoke:
        return _smoke()
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
