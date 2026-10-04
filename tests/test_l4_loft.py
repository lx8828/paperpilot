"""L4 · LoFT runner（`evals/runners/l4_loft.py`）—— 五个任务的官方口径。

守三件"对外可比"最容易崩的事：

  ① **坏指标文件必须被跳过**，不能当 0 分：本仓就躺着一个 **0KB** 的
     `preds_metrics.json` —— 长得像结果、其实没内容。当 0 分会让人以为
     "指标从 0.70 崩到 0"，去查模型/数据，**其实是文件坏了**。
  ② **各任务的口径要点必须随数字一起给**：五个任务的指标**各不相同**
     （`retrieval` 的 recall 是 **Capped**、`sql` **不强制顺序**、`icl` **忽略多值**…），
     不写清楚就会被跨任务误比。
  ③ **指标要透传**：官方输出里有什么数值指标就落什么 —— 硬编码清单会
     "官方加指标而这里漏报"，且漏得**静默**。

★ 只写临时目录，不碰真实 `evals/reports/`。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "evals" / "runners" / "l4_loft.py"


def _load():
    sys.path.insert(0, str(ROOT / "evals"))
    spec = importlib.util.spec_from_file_location("l4_loft", MOD)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["l4_loft"] = mod
    spec.loader.exec_module(mod)
    return mod


L4 = _load()


def _q(**over) -> dict:
    base = {"em": 0.44, "coverage": 0.81, "subspan_em": 0.70, "f1": 0.0,
            "num_unanswered_queries": 0}
    base.update(over)
    return {"quality": base}


# ── ① 坏指标文件必须被跳过（不是当 0 分）──────────────────────────────────

@pytest.mark.parametrize("content", ["", "{不是 JSON", "{}", '{"quality": {}}',
                                     '{"quality": "str"}'])
def test_bad_metrics_are_rejected(tmp_path: Path, content: str):
    f = tmp_path / "preds_metrics.json"
    f.write_text(content, encoding="utf-8")
    assert L4._load_metrics(f) is None


def test_collect_skips_bad_runs_instead_of_emitting_zeros(tmp_path: Path, monkeypatch):
    """★ **行为**断言：坏运行被跳过，绝不落成"全 0 分"记录。

    这是本层最要紧的一条 —— 若坏文件被当成 0 分，看表的人会以为"指标崩了"。
    （这条测试在写出来时**先红**：`relative_to(ROOT)` 抛 ValueError 会中断整批，
     见 `_rel` 的注释。）
    """
    runs, reports = tmp_path / "runs", tmp_path / "reports"
    for name, body in (("good_run", json.dumps(_q())), ("bad_run", "")):
        d = runs / name
        d.mkdir(parents=True)
        d.joinpath("preds_metrics.json").write_text(body, encoding="utf-8")

    monkeypatch.setattr(L4, "TASKS",
                        {**L4.TASKS, "multi_value_rag": {**L4.TASKS["multi_value_rag"],
                                                         "runs": runs}})
    monkeypatch.setattr(L4.R, "REPORT_DIR", reports)

    assert L4.collect("multi_value_rag") == 0
    recs = L4.R.load_all(reports)
    assert {r["name"] for r in recs} == {"qampari_good_run"}, "坏运行不该进记录"
    # ⚠️ 只对**质量指标**要求 > 0：健康度 `*.unanswered` 本来就**应该**是 0
    #    （把它一起要求 >0 是"断言太糙"，实测踩过）
    quality = [r for r in recs if not r["metric"].endswith(".unanswered")]
    assert quality and all(r["value"] > 0 for r in quality), \
        f"不该出现 0 分的质量指标：{[r['metric'] for r in quality if r['value'] <= 0]}"
    assert any(r["metric"].endswith(".unanswered") for r in recs), \
        "健康度 `unanswered` 必须一起落（它决定官方分母是否被剔除）"


# ── ② 五个任务都与官方口径对齐 ────────────────────────────────────────────

def test_all_five_official_tasks_are_covered():
    """LoFT 官方五类任务：rag / multi_value_rag / retrieval / sql / icl。"""
    assert set(L4.TASKS) == {"rag", "multi_value_rag", "retrieval", "sql", "icl"}


@pytest.mark.parametrize("task,keyword", [
    ("multi_value_rag", "只有 recall"),   # coverage 无 precision
    ("retrieval", "Capped"),              # gold 数 > k 时除以 k
    ("sql", "不强制顺序"),
    ("icl", "多值被忽略"),
])
def test_each_task_note_states_its_protocol_caveat(task: str, keyword: str):
    """★ 每个任务的口径要点必须在 note 里 —— 这正是"能不能比"的关键。"""
    assert keyword in L4.TASKS[task]["extra"], (
        f"{task} 的口径要点里应含「{keyword}」；不写就会被跨口径误比")


def test_mv_rag_note_states_f1_is_meaningless_and_boundaries():
    n = L4.TASKS["multi_value_rag"]["extra"]
    assert "f1" in n and "恒为 0" in n and "别引用" in n
    assert "128k" in n and "gold 存在" in n and "空集" in n
    assert "闭卷" in n, "约 0.150 来自闭卷知识 → 讲「检索贡献」要报差额"
    for name in L4.PUBLISHED:
        assert name in n, f"note 缺公开参照：{name}"


def test_official_f1_is_not_emitted_for_multi_value(tmp_path: Path, monkeypatch):
    """多值分支不给 f1 赋值（官方恒 0）→ **刻意不进记录**，否则会被当结果引用。"""
    assert "f1" in L4.EXCLUDE["multi_value_rag"]
    reports = tmp_path / "reports"
    monkeypatch.setattr(L4.R, "REPORT_DIR", reports)
    L4._emit_run("multi_value_rag", "mv1", _q()["quality"], "x.json")
    got = {r["metric"] for r in L4.R.load_all(reports)}
    assert "qampari.f1" not in got, "官方恒 0 的 f1 不该被当成指标"
    assert {"qampari.em", "qampari.coverage", "qampari.subspan_em"} <= got


# ── ③ 指标透传（官方加指标不必改代码）─────────────────────────────────────

def test_metrics_are_passed_through_not_hardcoded(tmp_path: Path, monkeypatch):
    """★ 官方输出里**多出来的**数值指标也要落下来。

    硬编码清单的失败方式很阴：官方加个指标，这里**静默漏报** ——
    没人会发现少了什么。
    """
    reports = tmp_path / "reports"
    monkeypatch.setattr(L4.R, "REPORT_DIR", reports)
    q = {"em": 0.5, "some_new_official_metric@5": 0.61, "nested": {"a": 1},
         "text": "xx", "flag": True, "nan_like": float("nan")}
    n = L4._emit_run("rag", "r1", q, "x.json")
    got = {r["metric"]: r["value"] for r in L4.R.load_all(reports)}
    assert got["loft_rag.em"] == pytest.approx(0.5)
    assert got["loft_rag.some_new_official_metric@5"] == pytest.approx(0.61), \
        "官方新指标应被透传"
    # 非数值 / bool / NaN 一律不进记录（NaN 会让表里出现"NaN"、比大小永远为假）
    assert not {"loft_rag.nested", "loft_rag.text", "loft_rag.flag",
                "loft_rag.nan_like"} & set(got)
    assert n == 2


def test_unanswered_health_metric_is_emitted(tmp_path: Path, monkeypatch):
    """空预测题数**必须一起报** —— 官方口径把它剔出分母 → 指标虚高。"""
    reports = tmp_path / "reports"
    monkeypatch.setattr(L4.R, "REPORT_DIR", reports)
    L4._emit_run("sql", "s1", {"execution_accuracy": 0.8,
                               "num_unanswered_queries": 7}, "x.json")
    got = {r["metric"]: r["value"] for r in L4.R.load_all(reports)}
    assert got["loft_sql.unanswered"] == 7


@pytest.mark.parametrize("task,prefix", [
    ("multi_value_rag", "qampari"), ("rag", "loft_rag"),
    ("retrieval", "loft_retrieval"), ("sql", "loft_sql"), ("icl", "loft_icl"),
])
def test_metric_prefix_is_dataset_scoped(task: str, prefix: str):
    """指标名带**数据集前缀** —— 五类任务的数放同一张表里也不会混。"""
    assert L4.TASKS[task]["prefix"] == prefix


def test_real_runs_if_present_are_readable():
    if not L4.QAMPARI_RUNS.exists():
        pytest.skip("本机没有 official_runs/（未跑过 L4）")
    dirs = [d for d in L4.QAMPARI_RUNS.glob("*") if d.is_dir()]
    if not dirs:
        pytest.skip("official_runs/ 是空的")
    ok = [d.name for d in dirs if L4._load_metrics(d / "preds_metrics.json")]
    assert ok, "一个可读的官方结果都没有 —— 格式可能变了，请更新 _load_metrics"
