"""表块纯函数：caption 清噪 / 表头 / 摘要 / 并集配额 / 按块权重 / RRF 融合。

迁移自 `qa/recall/_selftest_tables_20260912.py`（改写为参数化用例），并补了边界。

为什么必须测：这些函数全都在**生产路径**上——
正则在 caption 上宽一点会**误删真 caption**（`TABLE IV` 就是这么踩过的），
配额/权重错一点会**静默**改变线上排序（不报错，只是结果变差）。
"""
from __future__ import annotations

import numpy as np
import pytest

from paperpilot.agents.document_cache import (
    _clean_table_captions, _md_rows, _table_header_cells, _table_summary)
from paperpilot.agents.embedder import _ext_quota, _ext_weights, rrf_order


class _C:
    """最小 chunk 替身（只需要 chunk_id）。"""

    def __init__(self, cid: str) -> None:
        self.chunk_id = cid


# ───────────────────────── caption 清噪 ─────────────────────────


@pytest.mark.parametrize("caps, expect", [
    (["Figure 6: Attention Distributions...", "Table 4: Average precision"],
     ["Table 4: Average precision"]),
    (["Age Race Gender Table 3: Annotator agreement"], ["Table 3: Annotator agreement"]),
    (["Figure 1: x"], []),
    (["TABLE IV COMPARISON"], ["TABLE IV COMPARISON"]),
    (["Table 5a: ablations"], ["Table 5a: ablations"]),
    (["Table 3.1: results"], ["Table 3.1: results"]),
    (["  Table 2:  spaced  "], ["Table 2:  spaced"]),
    (["无关文字", "Table 7: ok"], ["Table 7: ok"]),
    ([], []),
], ids=["去图标题", "表头粘连", "纯图标题丢弃", "罗马数字", "带后缀", "小数号",
        "去首尾空白", "多片段", "空列表"])
def test_clean_table_captions(caps, expect):
    assert _clean_table_captions({"table_caption": list(caps)}) == expect


def test_clean_table_captions_missing_and_none():
    """字段缺失 / 为 None（老产物）都不能崩。"""
    assert _clean_table_captions({}) == []
    assert _clean_table_captions({"table_caption": None}) == []


# ───────────────────────── 表头指纹 ─────────────────────────


@pytest.mark.parametrize("text, expect", [
    ("| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |", ["A", "B", "C"]),
    ("caption\n| Lang | Acc |\n|---|---|\n| en | 0.9 |", ["Lang", "Acc"]),
    ("| A | A | B |", ["A", "B"]),          # 同行去重
    ("公式: x = y", []),                     # 无表格行
    ("", []),
], ids=["基本", "跳过 caption 行", "同行去重", "公式块", "空"])
def test_table_header_cells(text, expect):
    assert _table_header_cells(text) == expect


def test_md_rows_skips_separator_only():
    rows = _md_rows("| a | b |\n|---|---|\n| 1 | 2 |")
    assert rows == [["a", "b"], ["1", "2"]]


# ───────────────────────── 表块摘要（P2 双写用的向量文本）─────────────────────────


def test_table_summary_normal():
    s = _table_summary("Table 1: demo\n| Lang | Acc |\n|---|---|\n| en | 0.9 |")
    assert "Lang" in s and "Acc" in s and "Table 1: demo" in s


@pytest.mark.parametrize("text", ["", "公式: x = y", "| |", "\n\n"])
def test_table_summary_never_crashes(text):
    assert isinstance(_table_summary(text), str)


def test_table_summary_respects_max_len():
    long_cap = "Table 1: " + "很长" * 400
    assert len(_table_summary(long_cap, max_len=50)) <= 50


# ───────────────────────── 并集配额（PAPERPILOT_EXT_QUOTA）─────────────────────────


def test_ext_quota_default_is_2():
    """默认 2 = 表池并集候选开启（线上默认行为，改动会静默改变候选集）。"""
    assert _ext_quota() == 2


@pytest.mark.parametrize("val, expect", [("0", 0), ("1", 1), ("3", 3)])
def test_ext_quota_env_override(monkeypatch, val, expect):
    monkeypatch.setenv("PAPERPILOT_EXT_QUOTA", val)
    assert _ext_quota() == expect


def test_ext_quota_bad_value_falls_back():
    """非法值不能崩（评测时手滑会传空串）。"""
    import os
    os.environ["PAPERPILOT_EXT_QUOTA"] = "abc"
    try:
        assert _ext_quota() == 2
    finally:
        os.environ.pop("PAPERPILOT_EXT_QUOTA", None)


# ───────────────────────── 按块类型加权（α）─────────────────────────


def test_ext_weights_alpha_half_is_identity():
    """α=0.5 → 全部 (1, 1) = 生产原样（**默认必须逐位不变**）。"""
    wv, wb = _ext_weights([_C("p-1"), _C("xtbl-2")], 0.5)
    assert list(wv) == [1.0, 1.0] and list(wb) == [1.0, 1.0]


def test_ext_weights_alpha_075_only_external():
    wv, wb = _ext_weights([_C("p-1"), _C("xtbl-2")], 0.75)
    assert (wv[0], wb[0]) == (1.0, 1.0)          # 文本块不动
    assert abs(wv[1] - 1.5) < 1e-9 and abs(wb[1] - 0.5) < 1e-9


# ───────────────────────── RRF 融合 ─────────────────────────


def test_rrf_order_returns_full_permutation():
    order = rrf_order(np.array([1.0, 2.0]), np.array([2.0, 1.0]),
                      w_vec=np.array([1.0, 1.0]), w_bm=np.array([1.0, 1.0]))
    assert sorted(int(i) for i in order) == [0, 1]


def test_rrf_order_without_bm():
    """BM25 缺失（纯向量）时也要返回全序。"""
    order = rrf_order(np.array([0.1, 0.9]), None)
    assert [int(i) for i in order] == [1, 0]
