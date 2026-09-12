"""扩样本要多少题：按**观测到的翻转比**，精确反解达到 p<0.05 所需题量。

方法：McNemar 精确检验只依赖不一致对 (only_A, only_B)。
若观测翻转比为 r = only_A/(only_A+only_B)、不一致率为 d = (only_A+only_B)/N，
则"若真实效应与观测一致"，达到显著所需题量 N* ≈ n_d*/d，其中 n_d* 是最小的 n_d
使精确双尾 p < 0.05。

注意：这是"把已观测的效应做显著"所需样本，**不是**"扩样本会创造效应"。
若观测翻转比 ≈ 0.5，则无论样本多大都不会显著（此时应报告"无差异"而非"需扩样本"）。

用法：uv run python qa/compare/_power.py
"""
from __future__ import annotations

import io
import math
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def pval(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def need(ratio: float, rate: float) -> tuple[int, int] | tuple[None, None]:
    if ratio <= 0.5 or rate <= 0:
        return None, None
    for nd in range(2, 20000):
        k = round(ratio * nd)
        k = min(k, nd - k) if False else k
        if pval(k, nd - k) < 0.05:
            return nd, math.ceil(nd / rate)
    return None, None


# (标签, only_A, only_B, 该池题数)
OBS = [
    ("常规64 B2vsB0", 8, 5, 64),
    ("常规64 B2vsB1", 7, 2, 64),
    ("常规64 B0vsB1", 5, 3, 64),
    ("深水30 B2vsB0", 4, 4, 30),
    ("深水30 B2vsB1", 9, 2, 30),
    ("深水30 B0vsB1", 8, 1, 30),
]


def report(label: str, b: int, c: int, n: int) -> None:
    p = pval(b, c)
    if b + c == 0:
        print(f"{label}: 无翻转，无法比较")
        return
    ratio, rate = b / (b + c), (b + c) / n
    nd, N = need(ratio, rate)
    tail = (f"→ 达 p<0.05 需不一致 {nd} 对 ≈ **题量 {N}**"
            if nd else "→ 翻转比 ≤0.5，**样本再大也不会显著**（应结论为「无差异」）")
    print(f"{label:<14} 仅A过{b:>3} 仅B过{c:>3} 不一致{b + c:>3} (率{rate:.3f}) "
          f"翻转比{ratio:.3f} p={p:.3f}  {tail}")


print("== 单池 ==")
for label, b, c, n in OBS:
    report(label, b, c, n)

print("\n== 合并两池（仅作参考：深水池是按已知弱点挑的，非随机样本）==")
report("B2vsB0 合并", 8 + 4, 5 + 4, 64 + 30)
report("B2vsB1 合并", 7 + 9, 2 + 2, 64 + 30)
