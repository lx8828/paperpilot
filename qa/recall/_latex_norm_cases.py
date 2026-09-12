"""LaTeX 规范化 golden cases（可复算；有任一失败则退出码 1）。

用途：`mineru_bridge.latex_to_text` 是纯确定性变换，但**极易写出"看起来对、实际更糟"的实现**——
本仓库已踩过一次：首版只贴花括号、**没合并被 MinerU 拆开的单字母**，结果 `\\mathrm { R N N }` 仍为
`R N N`，token 数不升反降（5 → 2）。

用法：uv run python qa/recall/_latex_norm_cases.py
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(".").resolve() / "src"))

from paperpilot.tools.mineru_bridge import latex_to_text  # noqa: E402

# (名称, 输入, 必须包含, 必须不含)
CASES = [
    ("缩略语被逐字母拆开（★核心坑★）",
     r"\mathrm { R N N } ( \mathbf { x _ { i } } , h _ { i - 1 } ) ,\tag{1}",
     ["RNN"], ["R N N", "\\", "{"]),
    ("多字母命令名带空格",
     r"\operatorname { S o f t m a x } \bigl ( q \bigr )",
     ["Softmax"], ["\\", "{", "bigl"]),
    ("希腊字母脱壳",
     r"h _ { i } ^ { 2 } = ( 1 - \alpha _ { i } ) \odot h _ { i - 1 } ^ { 2 }",
     ["alpha", "odot"], ["\\", "{"]),
    ("array 环境被清除",
     r"\begin{array} { r l } & { z _ { i } = \sigma ( W _ { z } [ x ] ) } \end{array}",
     ["sigma"], ["begin", "array", "\\", "{"]),
    ("数学环境符 $$",
     r"$$ E = m c ^ { 2 } $$",
     ["E", "m"], ["$"]),
    ("全大写缩略语合并",
     r"\mathrm { L S T M } \to \mathrm { G R U }",
     ["LSTM", "GRU"], ["L S T M", "\\"]),
    ("★变量序列**不**合并（避免造假词 xit / Wqyi）",
     r"x _ { i } ^ { t } = W _ { q } y _ { i }",
     ["="], ["xit", "Wqyi", "\\", "{"]),
]

# 反向自检：全小写普通文本**不会**被合并（长度门槛救了它）——
# 残余风险只剩"全大写单字母串"（如 `U S A` → `USA`），实践中无害。
ANTI = ("a b c", "a b c")


def main() -> int:
    bad = 0
    for name, src, must, mustnot in CASES:
        got = latex_to_text(src)
        miss = [x for x in must if x not in got]
        leak = [x for x in mustnot if x in got]
        ok = not miss and not leak
        bad += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        入: {src[:88]}")
        print(f"        出: {got[:88]}")
        if not ok:
            print(f"        {'缺: ' + str(miss) if miss else ''} {'残留: ' + str(leak) if leak else ''}")
    # 反例自检：确认我们**明知**它有破坏性（所以只在 LaTeX 源上调用）
    anti_ok = latex_to_text(ANTI[0]) == ANTI[1]
    print(f"[{'PASS' if anti_ok else 'FAIL'}] 反例自检：普通文本 `a b c` **不被**合并（长度门槛已消除该风险）")
    bad += 0 if anti_ok else 1
    # 幂等性（避免二次规范化继续掉 token）
    src = CASES[1][1]
    idem = latex_to_text(latex_to_text(src)) == latex_to_text(src)
    print(f"[{'PASS' if idem else 'FAIL'}] 幂等性：规范化两次 == 一次")
    bad += 0 if idem else 1
    print(f"\n{'全部通过' if bad == 0 else f'{bad} 项失败'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
