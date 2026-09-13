"""目标表定位口径（`qa/recall/_tgt.py`：编号 ∪ 内容 联合判定）。

为什么必须测：这是**所有表格类指标**的尺子。口径错了，指标会同时出现
假阳性（量到别的表）与假阴性（把题丢出分母），而**报告数字看起来毫无异常**。
"""
from __future__ import annotations

import _tgt
import pytest


# ───────────────────────── 编号提取 ─────────────────────────


@pytest.mark.parametrize("text, expect_in", [
    ("Age Race Gender Table 3: x", "table3"),        # caption 被上游掺了表头文字
    ("TABLE IV COMPARISON", "table4"),               # 罗马数字
    ("Figure 5: curves", "figure5"),
    ("Table 13 Ensemble results", "table13"),
], ids=["粘连", "罗马", "figure", "普通"])
def test_keys_in(text, expect_in):
    assert expect_in in _tgt.keys_in(text)


def test_keys_in_ignores_body_beyond_head():
    """只看块头 300 字：表体深处的偶然 "Table N" 不该命中（否则假阳性）。"""
    text = "| a | b |\n" * 60 + "Table 9: deep in body"
    assert _tgt.keys_in(text) == set()


def test_keys_in_normalizes_suffix():
    assert "table5" in _tgt.keys_in("Table 5a: ablations")


# ───────────────────────── 证据编号 ─────────────────────────


def test_gold_keys_strips_float_selected():
    assert _tgt.gold_keys(["FLOAT SELECTED: Table 2: x"]) == {"table2"}


def test_gold_keys_multiple_and_none():
    assert _tgt.gold_keys(["Table 1: a", "Figure 2: b"]) == {"table1", "figure2"}
    assert _tgt.gold_keys(["没有编号"]) == set()
    assert _tgt.gold_keys([]) == set()


# ───────────────────────── 显著数值 ─────────────────────────


def test_gold_numbers_filters_short():
    """短数（38h 54m 38s）全部滤掉：否则表格里到处误命中。"""
    assert _tgt.gold_numbers("38h 54m 38s") == set()


def test_gold_numbers_keeps_significant():
    assert _tgt.gold_numbers("over 104k documents") == {"104"}
    assert _tgt.gold_numbers("accuracy 87.3%") == {"87.3"}
    assert _tgt.gold_numbers("1,024 examples") == {"1024"}
    assert _tgt.gold_numbers(None) == set()


# ───────────────────────── 联合定位 ─────────────────────────


class _C:
    def __init__(self, cid: str, text: str) -> None:
        self.chunk_id, self.text = cid, text


def test_locate_by_number():
    chunks = [_C("p-1", "正文"), _C("xtbl-7", "Table 3: results\n| a |\n| 1 |")]
    tgt, diag = _tgt.locate(chunks, ["Table 3: results"], "1")
    assert tgt == {"xtbl-7"} and diag["by_num"] == {"xtbl-7"}


def test_locate_by_content_only_when_mode_or():
    """编号取不到，但答案数值在表体里 → 新口径能救回；旧口径（num）必须救不回。"""
    chunks = [_C("xtbl-7", "| Model | Acc |\n| Ours | 87.3 |")]
    tgt_or, diag = _tgt.locate(chunks, [], "87.3", mode="or")
    tgt_num, _ = _tgt.locate(chunks, [], "87.3", mode="num")
    assert tgt_or == {"xtbl-7"} and diag["by_cnt"] == {"xtbl-7"}
    assert tgt_num == set()


def test_locate_ext_only_skips_body_chunks():
    """默认只看外部表块：正文块里出现同号也不该被当成目标表。"""
    chunks = [_C("p-1", "Table 3: 见正文"), _C("xtbl-7", "Table 3: results")]
    tgt, _ = _tgt.locate(chunks, ["Table 3: x"], "")
    assert tgt == {"xtbl-7"}


def test_locate_short_answer_requires_all_words():
    """短答案（无数字）：实词**全部**命中才算——宁可漏，不可误。"""
    chunks = [_C("xtbl-1", "the proposed encoder reaches high accuracy"),
              _C("xtbl-2", "completely unrelated content")]
    assert _tgt.locate(chunks, [], "proposed encoder", mode="or")[0] == {"xtbl-1"}


def test_locate_idx_with_ext_mask():
    texts = ["body text", "Table 1: x"]
    tgt, _ = _tgt.locate_idx(texts, ["Table 1: y"], "", ext_mask=[False, True])
    assert tgt == {1}
