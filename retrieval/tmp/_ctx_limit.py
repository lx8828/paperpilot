"""**上下文上限探测**：deepseek-chat 到底能塞多少 token？

为什么必须先测：架构分流判据是 `S/C = 语料token / 上下文预算`，
预算 C 若不实测就是空谈。本脚本用**同一段 filler 递增**（前缀命中缓存 → 便宜）
二分出「能塞进的最大 token 数」，并记录**溢出时的原始报错**。

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_ctx_limit.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))

# 固定 filler（前缀共享 → 增长时命中前缀缓存，成本极低）
SENT = "This is a filler sentence used only to occupy context space for a limit probe. "
# ⚠️ 实测校准：deepseek 英文 ≈ 4.0 字符/token
CHARS_PER_TOK = 4.0


def fill(n_tok: int) -> str:
    return (SENT * ((int(n_tok * CHARS_PER_TOK) // len(SENT)) + 1))[: int(n_tok * CHARS_PER_TOK)]


def probe(n_tok: int) -> tuple[bool, str]:
    """塞 n_tok 个 token 的 filler，问一个只需 1 个字的问题。"""
    llm.reset_usage()
    t0 = time.time()
    try:
        out = llm.chat_text("你是测试助手。", fill(n_tok) + "\n\n只回复两个字：可以",
                            temperature=0.0, max_tokens=8)
        u = llm.usage_stats()
        return True, f"{time.time() - t0:.1f}s tok={u['prompt_tokens']:,} → {out.strip()[:12]!r}"
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        return False, msg[:170]


print("=" * 104)
print("【上下文上限探测 · deepseek-chat】同一 filler 前缀递增 → 命中缓存，便宜")
print(f"  base={llm.config()[0]}  model={llm.config()[2]}\n")

for n in (8_000, 32_000, 64_000):
    ok, info = probe(n)
    print(f"  {n:>7,} tok  {'✅' if ok else '❌'}  {info}", flush=True)
    if not ok:
        print("\n  → 连 8k 都塞不进，环境有问题，中止。")
        raise SystemExit(1)

# 二分：[lo 一定行, hi 待定]
lo, hi = 64_000, 1_000_000
ok, info = probe(hi)
print(f"  {hi:>7,} tok  {'✅' if ok else '❌'}  {info}", flush=True)
if ok:
    print(f"\n  → 1M 都能塞（上限 ≥ 1M）")
    raise SystemExit(0)

first_fail = info
while hi - lo > 4_000:
    mid = (lo + hi) // 2
    ok, info = probe(mid)
    print(f"  {mid:>7,} tok  {'✅' if ok else '❌'}  {info}", flush=True)
    if ok:
        lo = mid
    else:
        hi = mid
        first_fail = info

print(f"\n{'=' * 104}")
print(f"  **可行上限 ≈ {lo:,} token**（{hi:,} 起首次失败）")
print(f"  溢出报错原文：{first_fail[:300]}")
print()
print("  对 S/C 分流的意义：")
for nm, tk in (("生产 5 篇", 48_000), ("QAMPARI 128k", 105_322),
               ("R2 20 篇/簇", 266_423), ("QAMPARI 1m", 844_484)):
    print(f"    {nm:<16}{tk:>9,} tok  → S/C = {tk / lo:>5.2f}  "
          f"{'✅ 塞得下 → 可比直读' if tk <= lo else '❌ **塞不下 → 检索物理必需**'}")
