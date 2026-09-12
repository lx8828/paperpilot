"""原型：把 MinerU 的 LaTeX 公式文本规范化成**可检索文本**，并量化效果。

动机：MinerU 的 equation 元素是 LaTeX，且**字母被空格拆开**（`\\mathrm { R N N }`），
导致：向量侧语义弱、BM25 侧连 `RNN` 都匹配不上 → 公式块作检索单元质量极差。

难点其实只有一个：MinerU 在**花括号内外都插了空格**，所以先"贴住花括号"再拆结构即可。
（这是纯确定性字符串变换，可单测，无需模型。）

量化口径：规范化前后，公式块的 token（len>=3）与**该篇正文 QASPER full_text** 的重合情况
——重合提升 = 更可能匹配上用户提问所用的正文措辞。

用法：uv run python qa/recall/_latex_norm_probe.py
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

from paperpilot.qasper_source import load_papers  # noqa: E402
from paperpilot.tools.mineru_bridge import _find_content_list, element_text  # noqa: E402

TOK = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")
_NOISE_CMD = (r"left|right|bigl|bigr|big|Big|bigg|cdot|ldots|cdots|dots|quad|qquad|"
              r"mathrm|mathbf|mathit|text|textsc|operatorname|boldsymbol|mathcal|"
              r"begin|end|array|hline|nonumber|label|ref|hspace|vspace|displaystyle")


def normalize_latex(raw: str) -> str:
    """LaTeX（MinerU 形态）→ 可检索文本。见模块 docstring 的难点说明。"""
    t = raw or ""
    t = t.replace("$$", " ").replace("$", " ")
    # ① 贴住花括号：MinerU 会写成 `\mathrm { R N N }`，先把 `{`/`}` 两侧的空格去掉
    t = re.sub(r"\s*\{\s*", "{", t)
    t = re.sub(r"\s*\}\s*", "}", t)
    # ② 包内容的命令 → 只留内容（\mathrm{RNN} → RNN；\text{recall} → recall）
    t = re.sub(r"\\(?:mathrm|mathbf|mathit|textsc|text|operatorname|boldsymbol|mathcal)\s*", "", t)
    # ③ 纯结构/噪声命令直接删
    t = re.sub(r"\\(?:%s)\b" % _NOISE_CMD, " ", t)
    t = re.sub(r"\\tag\{[^}]*\}", " ", t)
    t = re.sub(r"\\begin\{[^}]*\}(?:\{[^}]*\})?", " ", t)
    t = re.sub(r"\\end\{[^}]*\}", " ", t)
    # ④ 希腊字母等命令 → 名字本身（\alpha → alpha，\Phi → Phi）
    t = re.sub(r"\\([A-Za-z]+)", r"\1", t)
    # ⑤ 下标/上标：_{x}/^{x} → 空格 + x（保留内容，避免粘连出假词）
    t = re.sub(r"[_^]\s*\{([^{}]*)\}", r" \1 ", t)
    # ⑥ 残余 LaTeX 符号
    t = re.sub(r"[{}&\\]", " ", t)
    t = re.sub(r"[_^]", " ", t)
    # ⑦ 规整空白
    t = " ".join(t.split())
    # ⑧ ★关键★ 合并被 MinerU 拆开的单字母序列：`R N N`→`RNN`、`S o f t m a x`→`Softmax`
    #    （MinerU 在花括号内逐字母插空格；不合并则 BM25/向量都匹配不上正文里的缩写）
    t = re.sub(r"\b([A-Za-z])(?: ([A-Za-z]))+\b",
               lambda m: m.group(0).replace(" ", ""), t)
    return t


def main() -> int:
    papers = load_papers()
    rows = []
    for dd in sorted(Path("assets/artifacts/out_mineru").iterdir()):
        if not dd.is_dir():
            continue
        pid = dd.name[:-2] if dd.name.endswith("v1") else dd.name
        paper = papers.get(pid)
        if not paper:
            continue
        full = " ".join(str(pp) for s in (paper.get("full_text") or [])
                        for pp in (s.get("paragraphs") or []))
        ftok = {w.lower() for w in TOK.findall(full)}
        for e in _find_content_list(dd) or []:
            if e.get("type") != "equation":
                continue
            raw = element_text(e) or ""
            if not raw.strip():
                continue
            norm = normalize_latex(raw)
            rb, ra = set(w.lower() for w in TOK.findall(raw)), set(w.lower() for w in TOK.findall(norm))
            rows.append((raw, norm, len(rb), len(ra),
                         sum(1 for w in rb if w in ftok), sum(1 for w in ra if w in ftok)))

    n = len(rows)
    tb = np.array([r[2] for r in rows])
    ta = np.array([r[3] for r in rows])
    hb = np.array([r[4] for r in rows])
    ha = np.array([r[5] for r in rows])
    print(f"公式块 {n} 个\n")
    print("| 指标 | 规范化前 | 规范化后 |")
    print("|---|---|---|")
    print(f"| token 数（中位） | {np.median(tb):.0f} | {np.median(ta):.0f} |")
    print(f"| **与正文重合的 token 数（中位）** | **{np.median(hb):.0f}** | **{np.median(ha):.0f}** |")
    print(f"| 至少 1 个 token 命中正文的块占比 | {(hb>0).mean():.0%} | {(ha>0).mean():.0%} |")
    print(f"| 命中率（命中/总 token，中位） | "
          f"{np.median([r[4]/max(r[2],1) for r in rows]):.0%} | "
          f"{np.median([r[5]/max(r[3],1) for r in rows]):.0%} |")
    print("\n== 规范化样例（前 6 个）==")
    for raw, norm, *_ in rows[:6]:
        print(f"  原: L{len(raw):4d} | {raw[:95]}")
        print(f"  新: L{len(norm):4d} | {norm[:95]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
