"""临时验证：numbers.py 抽取后 (1) 旧归一行为不丢 (2) 新形态增量正确。"""
from __future__ import annotations
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

# ── oracle：抽取前的 validator._canonical_nums（完整复制） ──
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_NUM_TOK_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_EN_UNIT = {"k": 1e3, "m": 1e6, "b": 1e9,
            "thousand": 1e3, "million": 1e6, "billion": 1e9}
_CN_UNIT = {"百": 1e2, "千": 1e3, "万": 1e4, "百万": 1e6, "亿": 1e8}


def _parse_num_tok(tok: str) -> float:
    if "," in tok:
        parts = tok.split(",")
        if len(parts) == 2 and 0 < len(parts[1]) <= 2 and "." not in parts[1]:
            return float(parts[0] + "." + parts[1])
        return float(tok.replace(",", ""))
    return float(tok)


def old_canonical_nums(text: str) -> set[float]:
    body = _YEAR_RE.sub("", text)
    body = re.sub(r"\[\d+\]", "", body)
    low = body.lower()
    out: set[float] = set()
    unit_spans: list[tuple[int, int]] = []
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*(million|billion|thousand|[kKmMbB])", low):
        mult = _EN_UNIT.get(m.group(2).lower())
        if mult:
            out.add(round(_parse_num_tok(m.group(1)) * mult, 4))
            unit_spans.append((m.start(), m.end()))
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*(百万|亿|万|千|百)", body):
        mult = _CN_UNIT.get(m.group(2))
        if mult:
            out.add(round(_parse_num_tok(m.group(1)) * mult, 4))
            unit_spans.append((m.start(), m.end()))
    for m in _NUM_TOK_RE.finditer(body):
        if any(s <= m.start() < e for s, e in unit_spans):
            continue
        tok = m.group()
        has_pct = body[m.end():m.end() + 1] in ("%", "％")
        parts = tok.split(",") if "," in tok else []
        euro_dec = len(parts) == 2 and 0 < len(parts[1]) <= 2
        val = _parse_num_tok(tok)
        if has_pct or "." in tok or euro_dec or val >= 100:
            out.add(round(val, 4))
    return out


from paperpilot.components.numbers import parse_number, scan_numbers  # noqa: E402
from paperpilot.components.validator import _canonical_nums as new_canonical_nums  # noqa: E402


def main() -> int:
    fails = 0

    # ── 1) parse_number 单元断言 ──
    cases = [
        # 已有（不应回归）
        ("7.5", 7.5), ("7,5", 7.5), ("1,000,000", 1000000.0),
        ("1000", 1000.0), ("36 million", 36000000.0), ("32K", 32000.0),
        ("3600 万", 36000000.0), ("7200 万", 72000000.0), ("2亿", 200000000.0),
        ("0.5", 0.5),
        # 新形态：科学计数
        ("3.6×10^4", 36000.0), ("3.6*10^4", 36000.0), ("3.6e4", 36000.0),
        ("3.6×10⁻³", 0.0036), ("1e-3", 0.001),
        # 新形态：中文网络单位 w
        ("3.6w", 36000.0), ("2w", 20000.0),
        # 新形态：中文数词
        ("三万六千", 36000.0), ("一百二十", 120.0), ("一千零五十", 1050.0),
        ("七十八点六", 78.6), ("百分之七十八", 78.0), ("二十万", 200000.0),
        # 歧义 → None（不可误解析）
        ("三", None), ("百", None), ("m", None), ("abc", None),
        ("万分感谢", None), ("5", 5.0),  # 裸 5 能 parse 但显著判定由上层做
    ]
    for raw, exp in cases:
        got = parse_number(raw)
        if exp is None:
            ok = got is None
        else:
            ok = got is not None and abs(got - exp) < 1e-6
        print(f"  [{'OK ' if ok else 'FAIL'}] parse_number({raw!r}) = {got!r} 期望 {exp!r}")
        fails += 0 if ok else 1

    # ── 2) 旧⊆新（不丢旧行为）在真实误报文本上 ──
    print()
    real_texts = [
        # 545ff2 真实 answer 片段（历史数字误报）
        "该语料包含 3600 万条 En→Fr 句对，两者合计 7200 万句对 [4]。还生成了 32K 单位的共享子词词汇表 [4]。",
        # cf63a4 真实 answer 片段
        "人工纠错耗时约 3 天（纯人工需至少 7.5 天），准确率提升至约 80%，完成全部 25,000 个 token 的标注 [1][2]。",
        # 普通句子（不应产生误报数）
        "我们在 2026 年完成实验，用 5 个模型跑了 2 组对比，模型参数量 1.5b。",
        "accuracy improved by 3.2 points to reach 78.6% on ImageNet",
        "speedup is 7,5× over the baseline and 10× over cpu (36 million params).",
    ]
    for text in real_texts:
        old = old_canonical_nums(text)
        new = new_canonical_nums(text)
        lost = old - new
        gained = new - old
        print(f"  文本: {text[:70]}")
        print(f"    old={sorted(old)}")
        print(f"    new={sorted(new)}")
        if lost:
            print(f"    ❌ LOST(回归): {sorted(lost)}")
            fails += 1
        if gained:
            print(f"    ➕ GAINED: {sorted(gained)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
