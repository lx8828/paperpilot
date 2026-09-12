"""表格能力线的**边界自测**（项目未装 ruff/mypy/basedpyright，故用可复跑的自测兜底）。

覆盖 2026-09-12 新增/改动且**易错**的纯函数（都在生产路径上）：

| 被测 | 为什么必须测 |
|---|---|
| `document_cache._clean_table_captions` | 上游 caption 会混进**图的标题**与**表头行文字**；正则若过宽会误删真 caption、过窄会漏清噪声 |
| `document_cache._table_header_cells` | 表头识别错 → 借错兄弟块的 caption |
| `document_cache._table_summary` | P2 双写的向量文本；空文本/无表格行/公式块都不能崩 |
| `_tgt.keys_in` / `gold_numbers` / `gold_keys` | 评测口径；粘连 caption、罗马数字、短数值过滤都要正确，否则**表格类指标失真** |
| `embedder._ext_quota` / `_ext_weights` / `rrf_order` | 并集默认值（=2）与按块类型加权；权重写错会静默改变线上排序 |

用法：`uv run python qa/recall/_selftest_tables_20260912.py`（不联网、不读语料、秒级）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "qa" / "recall"))

FAILS: list[str] = []


def ck(name: str, cond: bool) -> None:
    print(("  OK   " if cond else "  FAIL ") + name)
    if not cond:
        FAILS.append(name)


def main() -> int:
    os.environ["PAPERPILOT_QASPER_TABLES"] = "1"
    os.environ.pop("PAPERPILOT_EXT_QUOTA", None)      # 测代码默认值

    from paperpilot.agents.document_cache import (_clean_table_captions,
                                                  _table_header_cells, _table_summary)
    from paperpilot.agents.embedder import _ext_quota, _ext_weights, rrf_order
    import _tgt

    # 1) caption 清噪
    ck("caption 清噪: 去掉上方图标题",
       _clean_table_captions({"table_caption": ["Figure 6: Attention Distributions...",
                                                "Table 4: Average precision"]})
       == ["Table 4: Average precision"])
    ck("caption 清噪: 表头文字粘连时从 Table N 截断",
       _clean_table_captions({"table_caption": ["Age Race Gender Table 3: Annotator agreement"]})
       == ["Table 3: Annotator agreement"])
    ck("caption 清噪: 纯图标题 → 空（当作无 caption 处理）",
       _clean_table_captions({"table_caption": ["Figure 1: x"]}) == [])
    ck("caption 清噪: 罗马数字表号保留（曾因正则漏掉而误删）",
       _clean_table_captions({"table_caption": ["TABLE IV COMPARISON"]}) == ["TABLE IV COMPARISON"])
    ck("caption 清噪: 带后缀阿拉伯号保留",
       _clean_table_captions({"table_caption": ["Table 5a: ablations"]}) == ["Table 5a: ablations"])
    ck("caption 清噪: 含小数的表号保留",
       _clean_table_captions({"table_caption": ["Table 3.1: results"]}) == ["Table 3.1: results"])

    # 2) 表头与摘要
    ck("表头单元格", _table_header_cells("| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |") == ["A", "B", "C"])
    ck("摘要: 空串不崩", isinstance(_table_summary(""), str))
    ck("摘要: 无表格行（公式块）不崩", isinstance(_table_summary("公式: x = y"), str))
    s = _table_summary("Table 1: demo\n| Lang | Acc |\n|---|---|\n| en | 0.9 |")
    ck("摘要: 含列名与具体数值（P2 v2 的关键）", "Lang" in s and "Acc" in s and "0.9" in s)

    # 3) 评测口径
    ck("keys_in: 粘连 caption 能取到号", "table3" in _tgt.keys_in("Age Race Gender Table 3: x"))
    ck("keys_in: 罗马数字转阿拉伯", "table4" in _tgt.keys_in("TABLE IV COMPARISON"))
    ck("gold_numbers: 过滤短数（38h 54m 38s → 空）", _tgt.gold_numbers("38h 54m 38s") == set())
    ck("gold_numbers: 保留显著数（over 104k documents → 104）",
       _tgt.gold_numbers("over 104k documents") == {"104"})
    ck("gold_keys: 证据行解析", _tgt.gold_keys(["FLOAT SELECTED: Table 2: x"]) == {"table2"})

    # 4) 并集默认与按块类型加权
    class _C:
        def __init__(self, cid: str) -> None:
            self.chunk_id = cid

    ck("_ext_quota 默认 = 2（并集开启，2026-09-12 起）", _ext_quota() == 2)
    os.environ["PAPERPILOT_EXT_QUOTA"] = "0"
    ck("_ext_quota 可退回 0", _ext_quota() == 0)
    os.environ.pop("PAPERPILOT_EXT_QUOTA", None)

    wv, wb = _ext_weights([_C("p-1"), _C("xtbl-2")], 0.75)
    ck("α=0.75: 外部块权重 (1.5, 0.5)、文本块不动 (1, 1)",
       abs(wv[1] - 1.5) < 1e-9 and abs(wb[1] - 0.5) < 1e-9 and wv[0] == 1.0 and wb[0] == 1.0)
    ck("α=0.5: 全部 (1, 1) = 生产原样",
       all(abs(x - 1.0) < 1e-9 for x in (*_ext_weights([_C("p-1"), _C("xtbl-2")], 0.5)[0],
                                         *_ext_weights([_C("p-1"), _C("xtbl-2")], 0.5)[1])))

    order = rrf_order(np.array([1.0, 2.0]), np.array([2.0, 1.0]),
                      w_vec=np.array([1.0, 2.0]), w_bm=np.array([2.0, 1.0]))
    ck("rrf_order: 支持每块两路权重且返回全序", len(order) == 2)

    print()
    print(f"SELFTEST: {len(FAILS) == 0 and 'ALL PASS' or 'FAILED'} "
          f"(failed={len(FAILS)})")
    for f in FAILS:
        print("   FAIL:", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
