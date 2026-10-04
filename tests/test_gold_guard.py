"""gold 隔离守卫的回归测试（2026-09-29）。

背景（`retrieval/results/R2_PROD_ANCHOR_AUDIT` / `R2_CLEAN_NUMBERS_20260929.md`）：
题集文件里 **`hint`（=出题者摘录的答案原文）与 `must_all`（判分锚点）同处**。
实测运行时 0 处读 `hint` —— 但那是**偶然安全**。本测试把它锁成**结构性安全**：
任何 runner 只要把 gold 字段混进 pipeline 输入，这里就会红。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "cli"
CLI_EVAL = CLI / "eval"              # 评测/跑批脚本 2026-10-01 移入 cli/eval/
for _p in (CLI, CLI_EVAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_multi_qa import (  # noqa: E402
    GOLD_KEYS,
    assert_no_gold,
    gold_of,
    pipeline_state,
)


def test_gold_keys_覆盖答案型字段():
    """守卫清单必须包含"答案原文"与"判分锚点"这两类。"""
    assert "hint" in GOLD_KEYS            # 答案原文
    assert "evidence" in GOLD_KEYS        # 逐字 gold 引文
    assert "must_all" in GOLD_KEYS        # 判分锚点
    assert "must_not" in GOLD_KEYS


def test_pipeline_state_是干净的():
    st = pipeline_state("问题？", ["a.pdf", "b.pdf"], route=["L3"], debug={})
    assert st["question"] == "问题？"
    assert st["pdfs"] == ["a.pdf", "b.pdf"]
    assert st["route"] == ["L3"]


@pytest.mark.parametrize("bad", [{"hint": "答案"}, {"evidence": [{"quote": "原文"}]},
                                {"must_all": ["0.1379"]}, {"must_not": ["没有"]}])
def test_污染输入会被响亮拦住(bad):
    """顶层混入 gold → 抛 AssertionError（不静默通过）。"""
    with pytest.raises(AssertionError, match="gold-guard"):
        assert_no_gold({**bad, "question": "q"}, "test")


def test_嵌套污染也会被拦住():
    """gold 藏在 l3_chunks / 嵌套结构里同样要炸（防止"只在顶层查一遍"）。"""
    with pytest.raises(AssertionError, match="gold-guard"):
        assert_no_gold({"l3_chunks": [{"text": "x", "must_have": ["A"]}]}, "nested")
    with pytest.raises(AssertionError, match="gold-guard"):
        pipeline_state("q", ["a.pdf"], title={"hint": "答案"})


def test_gold_of_只取gold字段():
    q = {"qid": "G1-M1-1", "question": "哪几篇？", "group": "B",
         "hint": "**两篇**：TAAL 与 MIDR", "must_all": ["TAAL", "MIDR"], "must_not": []}
    g = gold_of(q)
    assert set(g) == {"hint", "must_all", "must_not"}
    assert "question" not in g and "group" not in g


def test_真实题集确实同时含hint与must_all():
    """证明守卫在防**真东西**：若题集里根本没有 `hint`，这个守卫就是空转。"""
    qs = sorted((ROOT / "qa" / "multi").glob("group*.json"))
    if not qs:
        pytest.skip("qa/multi 题集不在本检出里")
    import json
    n_has_hint = n_both = 0
    for f in qs:
        for q in json.loads(f.read_text(encoding="utf-8")):
            if not isinstance(q, dict):
                continue
            if q.get("hint"):
                n_has_hint += 1
            if q.get("hint") and q.get("must_all"):
                n_both += 1
    assert n_has_hint > 0, "题集里没有 hint —— 守卫失效，请重新评估"
    assert n_both > 0, "hint 与 must_all 没有同处 —— 风险形态已变，请重新评估"


def test_cli_运行时不读gold_hint():
    """静态护栏：`cli/` 下**非注释行**不得读 `hint`。

    为什么静态查：`hint` 一旦被拼进 prompt，分数会**瞬间满分且看不出是泄露**；
    代码审查看不出"这次没拼"和"下次也没拼"。这条断言把现状钉死。
    """
    pat = re.compile(r"(get\([\"']hint[\"']\)|\[[\"']hint[\"']\])")
    offenders = []
    for f in sorted(list(CLI.glob("*.py")) + list(CLI_EVAL.glob("*.py"))):
        for i, ln in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            st = ln.strip()
            if st.startswith("#") or "gold-guard" in ln:
                continue
            if pat.search(ln):
                offenders.append(f"{f.name}:{i}: {st[:80]}")
    assert not offenders, ("cli/ 运行时代码里出现读 gold `hint` 的位置：\n  "
                          + "\n  ".join(offenders)
                          + "\n（hint 只允许用于判分/诊断；进 pipeline 会瞬间满分）")


def test_两个runner都调用了守卫():
    """`run_multi_qa`（直连节点）与 `run_group_qa`（完整图）都必须过 guard。"""
    assert "assert_no_gold" in (CLI_EVAL / "run_multi_qa.py").read_text(encoding="utf-8")
    assert "assert_no_gold" in (CLI_EVAL / "run_group_qa.py").read_text(encoding="utf-8")
