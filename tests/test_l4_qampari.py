"""L4 · QAMPARI 官方口径 runner（`evals/runners/l4_qampari.py`）。

守两件"对外可比"最容易崩的事：

  ① **坏指标文件必须被拒收**：本仓就躺着一个 **0KB** 的 `preds_metrics.json`
     —— 长得像结果、其实没内容。若不拦，它会被当成"全 0 分"计入 →
     **看起来是回退，其实是文件坏了**。
  ② **口径与边界必须随数字一起给**：没有"128k 档 / gold 保证存在 / coverage 只有
     recall / 官方 f1 恒 0"这些限定，0.700 就是个会被误引用的裸数字。

★ 只读与临时目录，不碰真实 `evals/reports/`。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "evals" / "runners" / "l4_qampari.py"


def _load():
    sys.path.insert(0, str(ROOT / "evals"))
    spec = importlib.util.spec_from_file_location("l4_qampari", MOD)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["l4_qampari"] = mod
    spec.loader.exec_module(mod)
    return mod


L4 = _load()

_VALID = {"quality": {"em": 0.44, "coverage": 0.81, "subspan_em": 0.70, "f1": 0.0},
          "num_unanswered_queries": 0}


# ── ① 坏指标文件必须拒收 ──────────────────────────────────────────────────

def test_rejects_zero_byte_metrics_file(tmp_path: Path):
    """★ 最阴的一类：**0KB 文件**。当成 0 分会读出"指标崩了"，其实是文件坏了。

    ℹ️ 这条**分不出机制**（去掉 `size == 0` 判断也会绿，因为 `json.loads("")` 同样抛错）——
    真正有价值的是下面 `test_collect_skips_bad_runs_instead_of_emitting_zeros`
    那条**行为**断言。
    """
    f = tmp_path / "preds_metrics.json"
    f.write_text("", encoding="utf-8")
    assert f.stat().st_size == 0
    assert L4._load_quality(f) is None


def test_collect_skips_bad_runs_instead_of_emitting_zeros(tmp_path: Path,
                                                          monkeypatch):
    """★ **行为**断言：坏运行被**跳过**，绝不落成"全 0 分"记录。

    这是本层最要紧的一条 —— 若坏文件被当成 0 分计入，看表的人会以为
    "指标从 0.70 崩到 0"，而去查模型/数据，**其实是文件坏了**。
    """
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"
    good = runs / "good_run"
    bad = runs / "bad_run"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    good.joinpath("preds_metrics.json").write_text(json.dumps(_VALID), encoding="utf-8")
    bad.joinpath("preds_metrics.json").write_text("", encoding="utf-8")   # 0KB

    monkeypatch.setattr(L4, "RUNS", runs)
    monkeypatch.setattr(L4.R, "REPORT_DIR", reports)

    assert L4.collect() == 0
    recs = L4.R.load_all(reports)
    names = {r["name"] for r in recs}
    assert names == {"good_run"}, f"坏运行不该进记录，得到 {names}"
    assert all(r["value"] > 0 for r in recs), "不该出现 0 分记录"


def test_rejects_malformed_json(tmp_path: Path):
    f = tmp_path / "preds_metrics.json"
    f.write_text("{不是合法 JSON", encoding="utf-8")
    assert L4._load_quality(f) is None


@pytest.mark.parametrize("payload", [
    {},                                    # 空对象
    {"quality": {}},                       # 有 quality 但一个指标都没有
    {"quality": "str"},                    # quality 类型不对
    {"num_unanswered_queries": 0},         # 缺 quality
])
def test_rejects_missing_quality_fields(tmp_path: Path, payload: dict):
    f = tmp_path / "preds_metrics.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    assert L4._load_quality(f) is None


def test_rejects_missing_file(tmp_path: Path):
    assert L4._load_quality(tmp_path / "nope.json") is None


def test_accepts_valid_metrics(tmp_path: Path):
    f = tmp_path / "preds_metrics.json"
    f.write_text(json.dumps(_VALID), encoding="utf-8")
    got = L4._load_quality(f)
    assert got is not None and got["quality"]["subspan_em"] == pytest.approx(0.70)


# ── ② 口径与边界必须随数字一起给 ──────────────────────────────────────────

def test_note_states_official_metric_quirks():
    """官方口径的两个坑必须写在 note 里 —— 否则数字会被误用。"""
    n = L4.NOTE
    assert "coverage" in n and "只有 recall" in n
    assert "f1" in n and "恒为 0" in n, "官方 f1 恒 0（多值分支未赋值）必须写明"


def test_note_states_boundaries():
    """★ 边界：档位与「保证 gold 存在」—— 不写就会被人当成"从全量维基里找齐"。"""
    n = L4.NOTE
    assert "128k" in n
    assert "gold 存在" in n and "空集" in n
    assert "闭卷" in n, "约 0.150 来自模型闭卷知识 → 讲「检索贡献」要报差额"


def test_note_carries_published_reference_points():
    """数字必须**有坐标**：随行给出论文 Table 2 的公开数字作参照。"""
    for name in L4.PUBLISHED:
        assert name in L4.NOTE, f"note 里缺公开参照：{name}"
    assert L4.CLOSED_BOOK > 0


def test_metric_mapping_covers_official_metrics():
    got = {k for k, _ in L4.METRICS}
    assert {"subspan_em", "em", "coverage"} <= got
    # ★ 官方 f1 恒 0（无意义）→ 刻意**不**映射，别把它当指标入库
    assert "f1" not in got


def test_unanswered_metric_is_down_direction():
    """`qampari.unanswered` 是健康度（越低越好）—— 官方口径会把它剔出分母而虚高。"""
    sys.path.insert(0, str(ROOT / "evals"))
    spec = importlib.util.spec_from_file_location(
        "mr", ROOT / "evals" / "checks" / "metrics_ratchet.py")
    mr = importlib.util.module_from_spec(spec)
    sys.modules["mr"] = mr
    spec.loader.exec_module(mr)
    assert "qampari.unanswered" in mr.DOWN


def test_real_official_runs_if_present_are_readable():
    """若本机有 `official_runs/`，逐个校验 —— 坏文件应当**被识别为坏**而不是当 0 分。"""
    if not L4.RUNS.exists():
        pytest.skip("本机没有 official_runs/（未跑过 L4）")
    dirs = [d for d in L4.RUNS.glob("*") if d.is_dir()]
    if not dirs:
        pytest.skip("official_runs/ 是空的")
    ok = [d.name for d in dirs if L4._load_quality(d / "preds_metrics.json")]
    assert ok, "一个可读的官方结果都没有 —— 文件格式可能变了，请更新 _load_quality"
