"""数值归一化层（Normalizer，机器件，0 LLM 成本）。

职责：把"同一数值、不同写法"归一到同一 float 语义，供上层做存在性校验
（答案给的数须能在被引原文找到：36 million ↔ 3600 万 ↔ 3.6e7 视为同一数）。

不判断语义、不做业务过滤（显著判定 / 年份剔除 / 题干剔除等策略留在调用方
validator）。解析原则：**宁可不解析、不可误解析**——歧义即返回 None/跳过。

设计（2026-09-09 自 validator._canonical_nums 独立抽取，行为等价 + 增量形态）：
  parse_number(value)：单值串 → float | None。覆盖
    - 普通整数/小数           1000 / 7.5
    - 欧式小数逗号            7,5 → 7.5（末段 1-2 位判定）
    - 千分位                  1,000,000
    - 英文单位                36 million / 32K / 1.5b / 36 thousand
    - 中文/网络单位           3600 万 / 2亿 / 3.6w（w=万）
    - 科学计数                3.6×10^4 / 3.6e4 / 3.6*10^-2 / 10⁻³(上标)
    - 中文数词                三万六千 / 一百二十 / 一千零五十 / 二十万
    - 中文小数                七十八点六；百分之七十八（pct 标记，值=78）
    % 仅剥离不换算：5%→5、0.05→0.05、50→50 三者天然不等；5% 与 5 同值
    （只对齐"写法"，不做 ×100 之类跨量纲语义换算）。
  scan_numbers(text)：扫描所有数值候选 → [NumberToken]，value=None 表示未识别。

歧义规避：m/k/b/w 等单字母单位须**紧贴数字后**才生效；孤立中文数词
（"三""百"单独出现，如"百般""第三次"）不解析——漏解析代价 << 误解析
（漏只保持字面，误会错误放行编数）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_EN_UNIT = {"k": 1e3, "m": 1e6, "b": 1e9,
            "thousand": 1e3, "million": 1e6, "billion": 1e9}
_CN_UNIT = {"百": 1e2, "千": 1e3, "万": 1e4, "百万": 1e6, "亿": 1e8}

# 上标数字/负号（科学计数 10⁻³ 形态）→ 半角
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")

_CN_DIG = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
           "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_CHARS = "零〇一二两三四五六七八九十百千万亿点"
_CN_WORD = re.compile(f"[{_CN_CHARS}]+")
_CN_WITH_UNIT = re.compile(r"[十百千万亿点]")
# 归一舍入精度：8 位小数——足以抹掉浮点乘除噪声，又保留真实高精度数
# （旧实现 round4 会把 0.00001/0.00012 抹成 0.0/0.0001，丧失判别力，是精度 bug）
_RND = 8


def normalize_superscripts(text: str) -> str:
    """上标 → 半角（仅处理扫描所需，不落盘不改原文本）。"""
    return text.translate(_SUP)


# ── 数字串（阿拉伯） → 值 ──────────────────────────────────────────────────

def _parse_plain(tok: str) -> float:
    """纯数字串 → float。逗号判定：末段 1-2 位 → 欧式小数（7,5→7.5）；否则千分去掉。"""
    if "," in tok:
        parts = tok.split(",")
        if len(parts) == 2 and 0 < len(parts[1]) <= 2 and "." not in parts[1]:
            return float(parts[0] + "." + parts[1])
        return float(tok.replace(",", ""))
    return float(tok)


# ── 中文数词 → 值（保守：规范书面写法，口语省略如"两千三"不追） ────────────

def _cn_int(s: str) -> int | None:
    """规范中文整数 → int（支持 十/百/千/万/亿 分层与 零 占位）。"""
    if not s:
        return 0

    def small(t: str) -> int | None:  # 无 万/亿 的段
        val = 0
        cur = 0
        for ch in t:
            if ch in _CN_DIG:
                cur = _CN_DIG[ch]
            elif ch == "十":
                val += (cur or 1) * 10
                cur = 0
            elif ch == "百":
                val += (cur or 1) * 100
                cur = 0
            elif ch == "千":
                val += (cur or 1) * 1000
                cur = 0
            elif ch == "零":
                cur = 0
            else:
                return None
        return val + cur

    if "亿" in s:
        a, _, b = s.partition("亿")
        if "亿" in b:
            return None
        return (_cn_int(a or "一")) * 10**8 + (_cn_int(b) if b else 0)
    if "万" in s:
        a, _, b = s.partition("万")
        if "万" in b:
            return None
        return (_cn_int(a or "一")) * 10**4 + (_cn_int(b) if b else 0)
    return small(s)


def _cn_num(s: str) -> float | None:
    """中文数值词 → 值（支持小数"点"与 零开头）。孤立单字（三/百/十）不解析。"""
    s = s.strip()
    if len(s) < 2:
        return None  # 孤立数词/单位单字：歧义大（"三""百""十"多非量化义），不解析
    if "点" in s:
        left, _, right = s.partition("点")
        int_part = _cn_int(left) if left else 0
        frac = 0.0
        scale = 0.1
        for ch in right:
            if ch not in _CN_DIG:
                return None
            frac += _CN_DIG[ch] * scale
            scale /= 10
        return int_part + frac
    return _cn_int(s)


# ── 主入口：单值解析 ───────────────────────────────────────────────────────

def parse_number(value: str) -> float | None:
    """单值串 → 归一 float。识别不出/歧义 → None。已含"%"剥离（不换算）。"""
    s = normalize_superscripts(value.strip())
    # 剥离 %（全/半角）
    s = s.rstrip("%％")
    if not s:
        return None
    # 科学计数：A×10^B / A*10B / AeB
    m = re.match(r"^([+-]?(?:\d+(?:,\d{1,2})?)(?:\.\d+)?)\s*[xX×*]\s*10\s*\^?\s*([+-]?\d+)$", s)
    if not m:
        m = re.match(r"^([+-]?\d[\d,]*(?:\.\d+)?)\s*[eE]\s*([+-]?\d+)$", s)
    if m:
        return round(_parse_plain(m.group(1)) * 10 ** int(m.group(2)), _RND)
    # 英文单位：36 million / 32k / 1.5b（单位紧贴或单空格）
    m = re.match(r"^(\d[\d,]*(?:\.\d+)?)\s*(million|billion|thousand|[kmb])\b$", s, re.I)
    if m:
        mult = _EN_UNIT.get(m.group(2).lower())
        if mult:
            return round(_parse_plain(m.group(1)) * mult, _RND)
    # 中文/网络单位：3600 万 / 2亿 / 36百万 / 3.6w
    m = re.match(r"^(\d[\d,]*(?:\.\d+)?)\s*(百万|亿|万|千|百|[wW])$", s)
    if m:
        mult = _CN_UNIT.get(m.group(2)) or 1e4  # w → 万
        return round(_parse_plain(m.group(1)) * mult, _RND)
    # 纯数字（欧式/千分）
    if re.fullmatch(r"[+-]?\d[\d,]*(?:\.\d+)?", s):
        return round(_parse_plain(s), _RND)
    # 中文数词（含百分之前缀）
    pct = False
    body = s
    if s.startswith("百分之"):
        pct = True
        body = s[3:]
    if body and all(ch in _CN_CHARS for ch in body):
        v = _cn_num(body)
        if v is not None:
            return round(v, _RND)
    return None


@dataclass
class NumberToken:
    raw: str          # 原文片段
    start: int        # 归一化文本中的 [start,end)（上标已被半角化，与 raw 一致）
    end: int
    value: float | None   # 归一值；None = 未识别
    percent: bool = False  # 是否百分比写法（% / percent / 百分之）


# ── 文本扫描：全部数值候选（供上层按需取用） ───────────────────────────────

_SCI_RE = re.compile(r"(?:\\times|\\cdot|[xX×*])\s*10\s*\^?\s*\{?\s*([+-]?\d+)\s*\}?")
_E_RE = re.compile(r"[eE]([+-]?\d+)")
_EN_WORD_RE = re.compile(r"(million|billion|thousand|percent)\b", re.I)
# 百分号：允许 LaTeX / 数学写法包夹（`54$\%$`、`54\%`、`54 %`）
_PCT_RE = re.compile(r"[\\${}\s\^]{0,4}[%％]")
# 中文单位后紧跟「分」时不是量词（百分点 / 百分比 / 百分之 是固定词）
_CN_UNIT_NOT = ("分",)


def _left_boundary_ok(body: str, s: int) -> bool:
    """数字左侧须是词界 —— 决定它**能不能吸附单位**。

    `F1 百分点` 里 `F1` 的 1 不是独立数字，若允许它吸附「百」就会凭空造出 100
    （2026-09-11 实测根因⑤）。`v2.3`/`GPT4` 同理。
    注意：只限制"单位吸附"，不限制数字本身被提取（`v2.3` 仍给出 2.3）。
    """
    return not (s > 0 and body[s - 1].isascii() and body[s - 1].isalpha())


def _right_boundary_ok(body: str, end: int) -> bool:
    """单位右侧须是词界 —— 决定它**是不是一个完整的单位 token**。

    `15,000 most` 的 `m` 后面还有 `ost` → 不是 million；
    `2.7 kernels` 的 `k` 后面是 `ernels` → 不是千。
    这条替代"整词单位白名单"：**规定"长什么样"，不必枚举"有哪些成员"**。
    """
    nxt = body[end:end + 1]
    return not (nxt.isascii() and nxt.isalpha())


def scan_numbers(text: str) -> list[NumberToken]:
    r"""扫描文本中所有数值候选（阿拉伯单位式 / 中文数词），顺序保留。

    **单位吸附规则（2026-09-11 加，治闸门数字误杀，见 qa/recall/GATE_NUM_DIAG_20260911.md）**：
      ① **右词界**：单字母 k/m/b/w、中文单位后不得紧跟 ASCII 字母
         （`15,000 most` 不再 ×1e6；`2.7 kernels` 不再 ×1e3；`0.81 between` 不再 ×1e9）。
      ② **左词界**：数字紧跟在 ASCII 字母后（`F1`/`v2.3`/`GPT4`）时不吸附单位
         （`F1 百分点` 不再造出 100）。
      ③ **LaTeX 写法**（QASPER 文本源里公式是 LaTeX 残留，属**封闭语法**）：
         科学计数允许 `\times`/`\cdot`/`{…}` 包裹；百分号允许 `$\%$` 包裹。
      ④ **「百分」不是量词**：中文单位后紧跟「分」时不吸附。

    单位成员表（k/m/b/w + 百千万亿 + million/billion/thousand）刻意保持**极小且不再扩充**：
    **开放词表不该穷举**（`4W` 到底是瓦特还是 4 万，任何枚举都会错），
    漏识别只会少一个归一读数，最终由上层复核在真实上下文里裁决。
    """
    body = normalize_superscripts(text)
    out: list[NumberToken] = []
    cut_until = 0  # 复合 token（如 3.6×10^4 的指数"10"）已被吞，跳过

    # ── 阿拉伯数字候选（含单位/科学计数/percent 尾巴） ──
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", body):
        s, e = m.span()
        if s < cut_until:
            continue
        tok = m.group()
        j = e
        while j < len(body) and body[j] == " ":
            j += 1
        rest = body[j:j + 24]
        value: float | None = None
        percent = False
        end = e
        left_ok = _left_boundary_ok(body, s)  # ① 左词界：决定能否吸附单位
        # 1) 英文单词单位 / percent（多字词自带 \b，安全）
        wm = _EN_WORD_RE.match(rest)
        if wm:
            w = wm.group(1).lower()
            end = j + wm.end()
            if w == "percent":
                percent = True
                value = _parse_plain(tok)
            else:
                value = round(_parse_plain(tok) * _EN_UNIT[w], _RND)
        # 2) 科学计数 A×10^B / AeB（含 LaTeX \times \cdot 与 {…} 包裹）
        elif left_ok and (rest.startswith(("\\times", "\\cdot"))
                          or body[j:j + 1] in ("x", "X", "×", "*")):
            sm = _SCI_RE.match(rest)
            if sm:
                end = j + sm.end()
                value = round(_parse_plain(tok) * 10 ** int(sm.group(1)), _RND)
        elif body[j:j + 1] in ("e", "E"):
            em = _E_RE.match(rest)
            if em:
                end = j + em.end()
                value = round(_parse_plain(tok) * 10 ** int(em.group(1)), _RND)
        # 3) 中文单位（多字优先；④「百分」不算；① 右词界）
        if value is None and left_ok and body[j:j + 1] >= "\u4e00":
            mu = re.match(r"(百万|亿|千|百|万)", body[j:j + 3])
            if mu:
                ulen = len(mu.group(1))
                if (body[j + ulen:j + ulen + 1] not in _CN_UNIT_NOT
                        and _right_boundary_ok(body, j + ulen)):
                    end = j + ulen
                    value = round(_parse_plain(tok) * _CN_UNIT[mu.group(1)], _RND)
        # 4) 单字母单位 k/m/b/w（① 右词界 + ② 左词界）
        if value is None and left_ok and j < len(body):
            ch = body[j].lower()
            if ch in ("k", "m", "b", "w") and _right_boundary_ok(body, j + 1):
                mult = {"k": 1e3, "m": 1e6, "b": 1e9, "w": 1e4}[ch]
                value = round(_parse_plain(tok) * mult, _RND)
                end = j + 1
        # 5) % 后缀（含 `\%` / `$\%$`）
        if value is None:
            if _PCT_RE.match(rest):
                percent = True
            elif rest.startswith("percent") and re.match(r"percent\b", rest):
                percent = True
        if value is None:
            value = _parse_plain(tok)
        out.append(NumberToken(raw=body[s:end], start=s, end=end,
                               value=round(value, _RND), percent=percent))
        cut_until = max(cut_until, end)

    # ── 中文数词候选（独立字符集，不与上面重叠） ──
    for m in _CN_WORD.finditer(body):
        raw = m.group()
        if not _CN_WITH_UNIT.search(raw):
            continue
        if not (raw.startswith("百分之") or len(raw) >= 2):
            continue
        pct = raw.startswith("百分之")
        core = raw[3:] if pct else raw
        v = _cn_num(core)
        if v is not None:
            out.append(NumberToken(raw=raw, start=m.start(), end=m.end(),
                                   value=round(v, 4), percent=pct))

    out.sort(key=lambda t: (t.start, t.end))
    return out
