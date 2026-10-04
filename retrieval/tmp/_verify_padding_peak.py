r"""**只用 tokenizer** 精确算"长度排序再批"对**峰值显存驱动量**的影响（不载模型权重）。

⚠️ 上一版把"峰值"写成了"批内最长单块 token" —— 两种排序下都是 7,608（因为最长的块总在
某个批里），**该指标没有意义**。真正驱动激活显存的是每步的

    batch_tokens = batch_size × max_token_len_in_batch

（`padding=True` 会补到批内最长）。所以要看 **max over batches of batch_tokens**，
以及它在两排序下的分布。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
DEV = HERE / "data" / "r2dev"
BATCH = 8      # = embedder.ENCODE_BATCH


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    d = pd.concat([pd.read_parquet(DEV / "prodchunk50" / "mineru" / f"c{i}.parquet")
                   for i in range(3)], ignore_index=True)
    texts = d["text"].astype(str).tolist()
    lens = np.array([len(x) for x in
                     tok(texts, truncation=False, add_special_tokens=True)["input_ids"]])
    print("=" * 108)
    print(f"块 {len(texts)} ｜ token 中位 {int(np.median(lens))} p90 {int(np.percentile(lens, 90))}"
          f" ｜ **max {int(lens.max())}** ｜ BATCH={BATCH}")

    def stats(order: np.ndarray, tag: str) -> dict:
        prods, pads = [], 0
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            mx = int(lens[b].max())
            prods.append(mx * len(b))
            pads += mx * len(b) - int(lens[b].sum())
        prods = np.array(prods)
        print(f"\n  【{tag}】批数 {len(prods)}")
        print(f"    总 token（含 padding）{prods.sum():>12,} ｜ padding 占 {pads / prods.sum():.1%}")
        print(f"    **单批 token 峰值 max {prods.max():,}** ｜ p99 {int(np.percentile(prods, 99)):,}"
              f" ｜ p90 {int(np.percentile(prods, 90)):,}")
        top = np.sort(prods)[-5:][::-1]
        print(f"    最大的 5 批：{', '.join(f'{int(x):,}' for x in top)}")
        return dict(tot=int(prods.sum()), peak=int(prods.max()),
                    p99=int(np.percentile(prods, 99)), pads=pads)

    a = stats(np.arange(len(lens)), "乱序（原）")
    s_order = np.array(sorted(range(len(lens)), key=lambda i: len(texts[i])))
    b = stats(s_order, "排序（现）")

    print("\n" + "=" * 108)
    print(f"  ★ 总计算量：{b['tot'] / a['tot']:.2f}×（省 {1 - b['tot'] / a['tot']:.1%}）")
    print(f"  ★ **峰值单批 token：{a['peak']:,} → {b['peak']:,}**"
          f"（降到 {b['peak'] / a['peak']:.2f}×）")
    print(f"  ★ p99 单批 token：{a['p99']:,} → {b['p99']:,}（{b['p99'] / a['p99']:.2f}×）")
    print(f"\n  → 峰值若仍是 {b['peak']:,} token，说明**最长的那几个块被排到一起**；")
    print(f"    治本要么限制单批 token 预算，要么对超长块单独处理（见下）。")

    # 预算式分批：按"批内 token 预算"切批（兼顾排序与上限）
    budget = int(np.percentile(lens, 90)) * BATCH       # 以 p90 块长 × batch 作为每批预算
    for bud, tag in ((budget, f"长度排序 + 每批 token 预算 {budget:,}"),):
        prods2, i, cur = [], 0, []
        for j in s_order:
            cand = cur + [j]
            if len(cand) > BATCH or (cur and int(lens[cand].max()) * len(cand) > bud):
                prods2.append(int(lens[cur].max()) * len(cur))
                cur = [j]
            else:
                cur = cand
        if cur:
            prods2.append(int(lens[cur].max()) * len(cur))
        p2 = np.array(prods2)
        print(f"\n  【{tag}】批数 {len(p2)} ｜ 总 token {p2.sum():,}"
              f" ｜ **峰值 {p2.max():,}** ｜ p99 {int(np.percentile(p2, 99)):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
