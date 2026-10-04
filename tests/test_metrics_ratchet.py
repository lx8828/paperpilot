"""L0 不变量 · 指标棘轮（`evals/checks/metrics_ratchet.py`）。

守三件"比数字"时最容易做错的事：

  ① **容差**：M1 噪声底 ≈ ±16pt —— 没有容差会**天天误报**，闸门被"狼来了"淹没。
  ② **方向**：`qa.reader_offlabel`（应为 0）这类**越低越好**；
     一律按"降了就红"会把**修好了**判成回退。
  ③ **口径**：同样的 `(层,名,指标)` 换了口径就**不是同一个东西** —— 要提示，不许静默比。

★ 一律写临时目录，不碰真实 `evals/reports/` 与 `evals/baselines/metrics.json`。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RATCHET = ROOT / "evals" / "checks" / "metrics_ratchet.py"


def _load():
    sys.path.insert(0, str(ROOT / "evals"))
    spec = importlib.util.spec_from_file_location("metrics_ratchet", RATCHET)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["metrics_ratchet"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()


def _rec(metric: str, value: float, note: str = "口径相同") -> dict:
    return dict(layer="L3", name="g2", metric=metric, value=value, n=90,
                note=note, evidence_path="x.json")


def _base(metric: str, value: float, note: str = "口径相同") -> dict:
    return {f"L3|g2|{metric}": {"value": value, "metric": metric,
                               "tol": M.tol_of(metric),
                               "direction": M.direction_of(metric), "note": note}}


# ── ① 方向 ────────────────────────────────────────────────────────────────

def test_direction_defaults_up_but_health_metrics_are_down():
    assert M.direction_of("qa.ok_strict") == "up"
    for m in ("qa.reader_offlabel", "retrieval.gold_map_failed",
              "qa.parse_degraded"):
        assert M.direction_of(m) == "down", f"{m} 应为「越低越好」"


def test_unanswered_suffix_is_down_for_every_task():
    """★ `.unanswered` 用**后缀**规则判方向 —— 五个 LoFT 任务各一个前缀，
    逐个列名**必然漏**，而漏的后果是"空预测变多"被判成"变好"。"""
    for prefix in ("qampari", "loft_rag", "loft_retrieval", "loft_sql", "loft_icl"):
        m = f"{prefix}.unanswered"
        assert M.direction_of(m) == "down", f"{m} 应为越低越好"
        assert M.worse(m, 5, 0) == pytest.approx(5.0), "升了才是变差"


def test_worse_respects_direction():
    assert M.worse("qa.ok_strict", 0.40, 0.50) == pytest.approx(0.10)   # 降了 = 变差
    assert M.worse("qa.ok_strict", 0.60, 0.50) == pytest.approx(-0.10)
    # 健康度：**升了**才是变差（这正是"一律按降了红"会搞错的地方）
    assert M.worse("qa.reader_offlabel", 3, 0) == pytest.approx(3.0)
    assert M.worse("qa.reader_offlabel", 0, 3) == pytest.approx(-3.0)


# ── ② 容差 ────────────────────────────────────────────────────────────────

def test_within_tolerance_is_not_a_regression():
    reg, _, ok = M.compare(_base("qa.ok_strict", 0.50), [_rec("qa.ok_strict", 0.49)])
    assert not reg, "掉 1 个点 < 默认容差 2 点，不该报"
    assert ok == 1


def test_beyond_tolerance_is_a_regression():
    reg, _, _ = M.compare(_base("qa.ok_strict", 0.50), [_rec("qa.ok_strict", 0.40)])
    assert len(reg) == 1 and "基线 0.5000 → 本次 0.4000" in reg[0]


def test_noisy_metric_gets_wider_tolerance():
    """M1 噪声底 ≈ ±16pt —— 掉 12 个点**不该**报；掉 20 个点才报。"""
    assert M.tol_of("qa.kind_m1") == pytest.approx(0.16)
    reg, _, _ = M.compare(_base("qa.kind_m1", 0.50), [_rec("qa.kind_m1", 0.38)])
    assert not reg, "12pt < 16pt 容差，属噪声"
    reg, _, _ = M.compare(_base("qa.kind_m1", 0.50), [_rec("qa.kind_m1", 0.30)])
    assert len(reg) == 1, "20pt > 16pt，应报"


def test_health_metric_improving_is_not_a_regression():
    """`reader_offlabel` 从 3 → 0 是**修好了**，绝不能被判回退。"""
    reg, _, ok = M.compare(_base("qa.reader_offlabel", 3),
                           [_rec("qa.reader_offlabel", 0)])
    assert not reg and ok == 1
    reg, _, _ = M.compare(_base("qa.reader_offlabel", 0),
                          [_rec("qa.reader_offlabel", 5)])
    assert len(reg) == 1, "健康指标升高应报"


def test_relative_metrics_use_proportion():
    """token 数上万，"2 个点"没意义 → 按**相对幅度**判（容差 10%）。"""
    reg, _, _ = M.compare(_base("qa.avg_prompt_tokens", 50000),
                          [_rec("qa.avg_prompt_tokens", 48000)])
    assert not reg, "掉 4% < 10%，属噪声"
    reg, _, _ = M.compare(_base("qa.avg_prompt_tokens", 50000),
                          [_rec("qa.avg_prompt_tokens", 40000)])
    assert len(reg) == 1, "掉 20% > 10%，应报"


# ── ③ 口径 ────────────────────────────────────────────────────────────────

def test_changed_note_is_flagged_but_not_failed():
    """口径变了 → **提示**（人确认），但**不自动判回退**。

    为什么不自动拦：note 里含"分母/失败条数"这类**每次都会变**的细节，
    自动拦会永久误报。危险的是"判官协议变了还照比"，那要靠人看提示。
    """
    reg, notices, _ = M.compare(_base("qa.ok_strict", 0.50, "proto=3"),
                                [_rec("qa.ok_strict", 0.50, "proto=4")])
    assert not reg, "口径变化本身不是回退"
    assert any("口径可能变了" in n for n in notices), notices


def test_missing_data_is_a_notice_not_a_failure():
    reg, notices, _ = M.compare(_base("qa.ok_strict", 0.50), [])
    assert not reg
    assert any("无数据" in n for n in notices)


# ── 基线读写 ──────────────────────────────────────────────────────────────

def test_baseline_roundtrip_and_update_shape(tmp_path: Path):
    p = tmp_path / "metrics.json"
    b = _base("qa.ok_strict", 0.50)
    b["L3|g2|qa.ok_strict"]["note"] = "n"
    M.save_baseline(b, p)
    got = M.load_baseline(p)
    assert got["L3|g2|qa.ok_strict"]["value"] == pytest.approx(0.50)
    assert got["L3|g2|qa.ok_strict"]["direction"] == "up"

    # 每个条目都要带 tol/direction —— 否则"怎么比"就丢了
    for _, ent in got.items():
        assert {"value", "metric", "tol", "direction", "note"} <= set(ent)


def test_real_baseline_if_exists_is_wellformed():
    """真实基线（入库的那个）每个条目都要合规 —— 坏条目会让比较静默失真。"""
    if not M.BASELINE.exists():
        pytest.skip("还没有基线文件")
    b = json.loads(M.BASELINE.read_text(encoding="utf-8"))
    bad = [k for k, e in b.items()
           if not isinstance(e, dict) or "value" not in e or "tol" not in e]
    assert not bad, f"基线条目缺少必要字段：{bad}"
