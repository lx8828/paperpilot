"""查询侧优化组件（`components/query_optimizer.py`）的契约测试。

这一组测试钉住的是**行为契约**，因为该组件的默认值是 A/B 实验的基线：
  · 未配置任何开关 → **只回原问题**（与抽取前的旧默认完全一致，基线不能被无意改掉）；
  · 旧开关 `PAPERPILOT_QUERY_REWRITE=1` 必须仍然等价于 `PAPERPILOT_QUERY_LEVELS=l3`；
  · 启用**未实现**的级别必须**响亮报错**（不做"开了却什么也没发生"的沉默失败）；
  · 融合原语 `quota_union` 的语义：主路优先、变体只做召回补充。
"""
from __future__ import annotations

import pytest

from paperpilot.components import query_optimizer as qo
from paperpilot.tools import llm

Q = "What is the accuracy of the model on the Dutch dataset?"


def _fake_chat(reply: str):
    """替换最底层 `_chat`：`chat_json` 等上层封装都会自动走它。"""
    return lambda *a, **k: reply


# ───────────────────────── 默认：全关 ─────────────────────────
def test_default_is_off_and_identical_to_original(monkeypatch):
    """未配置 → 只回原问题，`multi=False`（这是所有 A/B 的基线）。"""
    monkeypatch.delenv("PAPERPILOT_QUERY_LEVELS", raising=False)
    monkeypatch.delenv("PAPERPILOT_QUERY_REWRITE", raising=False)
    plan = qo.optimize(Q)
    assert plan.levels == ()
    assert plan.queries == [Q]
    assert plan.multi is False
    assert plan.primary == Q


def test_empty_levels_string_is_off(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_QUERY_LEVELS", "  ,  ")
    assert qo.enabled_levels() == ()
    assert qo.optimize(Q).queries == [Q]


def test_unknown_level_is_ignored(monkeypatch):
    """拼错的级别名静默忽略（不能因为一个错名就把基线打开）。"""
    monkeypatch.setenv("PAPERPILOT_QUERY_LEVELS", "l9,foo")
    assert qo.enabled_levels() == ()


# ───────────────────────── 向后兼容 ─────────────────────────
def test_legacy_env_maps_to_l3(monkeypatch):
    monkeypatch.delenv("PAPERPILOT_QUERY_LEVELS", raising=False)
    monkeypatch.setenv("PAPERPILOT_QUERY_REWRITE", "1")
    assert qo.enabled_levels() == ("l3",)


def test_new_env_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_QUERY_LEVELS", "l3")
    monkeypatch.setenv("PAPERPILOT_QUERY_REWRITE", "0")
    assert qo.enabled_levels() == ("l3",)


# ───────────────────────── L3：改写 ─────────────────────────
def test_l3_appends_variants_after_original(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_QUERY_LEVELS", "l3")
    monkeypatch.setattr(llm, "_chat", _fake_chat('```json\n{"queries": ["v1", "v2"]}\n```'))
    plan = qo.optimize(Q)
    assert plan.queries == [Q, "v1", "v2"]      # 原问题在首位（融合时主查询权重最高）
    assert plan.multi is True
    assert plan.summary()["n_queries"] == 3


def test_l3_caps_at_max_queries(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_QUERY_LEVELS", "l3")
    monkeypatch.setattr(llm, "_chat",
                        _fake_chat('{"queries": ["v1", "v2", "v3", "v4"]}'))
    assert qo.optimize(Q).queries == [Q, "v1", "v2"]     # max_queries 默认 3


def test_l3_dedupes_and_drops_echo(monkeypatch):
    """变体与原问题相同 / 变体之间重复 → 去掉（旧实现同）。"""
    monkeypatch.setenv("PAPERPILOT_QUERY_LEVELS", "l3")
    monkeypatch.setattr(llm, "_chat", _fake_chat(f'{{"queries": ["{Q}", "v1", "v1"]}}'))
    assert qo.optimize(Q).queries == [Q, "v1"]


def test_l3_drops_overlong_variant(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_QUERY_LEVELS", "l3")
    monkeypatch.setattr(llm, "_chat", _fake_chat('{"queries": ["' + "x" * 300 + '", "ok"]}'))
    assert qo.optimize(Q).queries == [Q, "ok"]


def test_l3_llm_failure_falls_back_to_original(monkeypatch):
    """LLM 抛错 → 空变体 → 只回原问题（改写失败绝不能影响主链路）。"""
    monkeypatch.setenv("PAPERPILOT_QUERY_LEVELS", "l3")

    def _boom(*a, **k):
        raise llm.LLMError("网络挂了")

    monkeypatch.setattr(llm, "_chat", _boom)
    plan = qo.optimize(Q)
    assert plan.queries == [Q]
    assert plan.multi is False


def test_l3_garbage_reply_falls_back(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_QUERY_LEVELS", "l3")
    monkeypatch.setattr(llm, "_chat", _fake_chat("抱歉，我不会"))
    assert qo.optimize(Q).queries == [Q]


# ───────────────────────── 未实现的级别：必须响亮报错 ─────────────────────────
@pytest.mark.parametrize("level", ["l1", "l2", "l4", "l5"])
def test_unimplemented_level_raises(monkeypatch, level):
    """刻意设计：启用未实现的级别 → 抛错，避免"开了却没发生"的沉默失败。"""
    monkeypatch.setenv("PAPERPILOT_QUERY_LEVELS", level)
    with pytest.raises(NotImplementedError):
        qo.optimize(Q)


# ───────────────────────── 融合原语：quota_union ─────────────────────────
def test_quota_union_puts_primary_first_and_never_drops_it():
    """主路结果优先；变体只补主路没覆盖到的位置（不稀释主查询排序）。"""
    got = qo.quota_union([[1, 2, 3], [9, 8, 7]], [2, 1])
    assert got[:2] == [1, 2]          # 主路前 2 个原样在最前
    assert 9 in got                   # 变体的第 1 个被补进来
    assert 7 not in got               # 变体配额只有 1 → 7 不进


def test_quota_union_dedupes_across_lists():
    got = qo.quota_union([[1, 2], [2, 3]], [2, 2])
    assert got.count(2) == 1
    assert set(got) == {1, 2, 3}


def test_quota_union_zero_quota_skips_path():
    assert qo.quota_union([[1], [9]], [1, 0]) == [1]


def test_quota_union_length_mismatch_raises():
    with pytest.raises(ValueError):
        qo.quota_union([[1]], [1, 1])


# ───────────────────────── 融合意图 ─────────────────────────
def test_fusion_default_is_equal_rrf_legacy(monkeypatch):
    """默认融合意图保持旧的 equal_rrf（新策略要靠显式选择，避免悄悄换基线）。"""
    monkeypatch.delenv("PAPERPILOT_QUERY_FUSION", raising=False)
    assert qo.optimize(Q).fusion == qo.FUSION_EQUAL_RRF


def test_fusion_env_and_explicit_override(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_QUERY_FUSION", "quota_union")
    assert qo.optimize(Q).fusion == qo.FUSION_QUOTA_UNION
    assert qo.optimize(Q, fusion=qo.FUSION_RERANK_ORIG).fusion == qo.FUSION_RERANK_ORIG
    # 非法值 → 回落 equal_rrf（不能因拼错而打开未经验证的策略）
    assert qo.optimize(Q, fusion="nonsense").fusion == qo.FUSION_EQUAL_RRF


def test_explicit_levels_bypass_env(monkeypatch):
    """显式传 levels 时不读 env（供离线实验直接指定）。"""
    monkeypatch.setenv("PAPERPILOT_QUERY_LEVELS", "l3")
    monkeypatch.setattr(llm, "_chat", _fake_chat('{"queries": ["v1"]}'))
    assert qo.optimize(Q, levels=()).queries == [Q]
    assert qo.optimize(Q, levels=("l3",)).queries == [Q, "v1"]
