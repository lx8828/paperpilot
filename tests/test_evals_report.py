"""L0 不变量 · 统一评测记录格式（`evals/report.py`）。

守两件事：

  ① **不合规的记录写不进去**（缺 `n` / 缺 `note` / 指标名乱写 / 值不是数）
     —— 因为 `n` 与 `note` 是"能不能比"的前提：没有样本量的率不可比，
     没写口径的指标**看起来能比、实际会误导**。
  ② **写进去的能读回来、能对齐成表**（round-trip），且坏行不会毁掉整表。

★ 测试一律写 `tmp_path`，**不碰真实的 `evals/reports/`**。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "evals" / "report.py"


def _load():
    spec = importlib.util.spec_from_file_location("evals_report", MOD)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evals_report"] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load()


def _ok(**over) -> dict:
    rec = dict(layer="L3", name="group2_graph", metric="qa.ok_strict", value=0.51,
               n=90, note="严格锚点全命中／全部题",
               evidence_path="qa/multi/_runs/group2_graph.json")
    rec.update(over)
    return rec


# ── ① 不合规的记录写不进去 ────────────────────────────────────────────────

@pytest.mark.parametrize("drop,why", [
    ("n", "样本量"),
    ("note", "口径"),
    ("metric", "指标名"),
    ("value", "数值"),
    ("layer", "分层"),
    ("evidence_path", "证据"),
])
def test_missing_field_is_rejected_with_reason(drop: str, why: str):
    """缺任何必填字段都要**明确报错**，且说明"为什么这个字段不能缺"。"""
    rec = _ok()
    rec.pop(drop)
    with pytest.raises(R.ReportError) as e:
        R.validate(rec)
    assert drop in str(e.value), f"报错里应指出缺的是 {drop}"
    assert why in str(e.value), f"报错里应说明理由（含「{why}」）"


@pytest.mark.parametrize("over", [
    dict(value="0.51"),          # 字符串
    dict(value=True),            # bool 不是数值（Python 里 bool 是 int 子类，容易漏）
    dict(n=0),                   # 没有样本量的率不可比
    dict(n=-3),
    dict(metric="QA.OK"),        # 大写
    dict(metric="okstrict"),     # 没有点分隔
    dict(metric="qa ok"),        # 空格
    dict(layer="L9"),            # 不存在的层
])
def test_bad_values_are_rejected(over: dict):
    with pytest.raises(R.ReportError):
        R.validate(_ok(**over))


# ── ② round-trip：写得进、读得回、能对齐 ─────────────────────────────────

def test_emit_and_load_roundtrip(tmp_path: Path):
    p = R.emit(_ok(), _ok(metric="qa.ok", value=0.6, n=164), path=tmp_path / "L3.jsonl")
    assert p.exists()
    recs = R.load_all(tmp_path)
    assert len(recs) == 2
    assert {r["metric"] for r in recs} == {"qa.ok", "qa.ok_strict"}
    assert all(r.get("at") for r in recs), "emit 应自动补时间戳"


def test_emit_rejects_invalid_before_writing(tmp_path: Path):
    """校验发生在**写盘之前** —— 半个非法记录都不该落盘。"""
    p = tmp_path / "L3.jsonl"
    with pytest.raises(R.ReportError):
        R.emit(_ok(), _ok(n=0), path=p)
    assert not p.exists(), "有非法记录时不该写出任何内容"


def test_load_all_skips_bad_lines(tmp_path: Path, capsys):
    """一行脏数据不能废掉整张表（但要告警）。"""
    f = tmp_path / "L3.jsonl"
    f.write_text(json.dumps(_ok(), ensure_ascii=False) + "\n{坏行}\n"
                 + json.dumps(_ok(metric="qa.ok"), ensure_ascii=False) + "\n",
                 encoding="utf-8")
    recs = R.load_all(tmp_path)
    assert len(recs) == 2
    assert "跳过坏行" in capsys.readouterr().err


def test_table_keeps_latest_and_shows_note(tmp_path: Path):
    """同一 (层,项,指标) 只显示**最新**；且表格必须带**口径**（否则还是会比错）。"""
    R.emit(_ok(value=0.10), path=tmp_path / "L3.jsonl")
    R.emit(_ok(value=0.51), path=tmp_path / "L3.jsonl")
    out = R.table(R.load_all(tmp_path))
    assert "0.5100" in out and "0.1000" not in out, "应只显示最新值"
    assert "口径" in out and "严格锚点全命中" in out, "表格必须带口径"
    assert "qa/multi/_runs/group2_graph.json" in out, "表格必须带证据路径"


def test_table_handles_empty(tmp_path: Path):
    assert "还没有任何记录" in R.table(R.load_all(tmp_path))


def test_markdown_export_has_all_columns(tmp_path: Path):
    R.emit(_ok(), path=tmp_path / "L3.jsonl")
    md = R.to_markdown(R.load_all(tmp_path))
    for col in ("层", "指标", "运行/项", "值", "n", "口径", "证据"):
        assert col in md, f"Markdown 缺列：{col}"


# ── ③ 真实落盘的记录也必须合规（防止 runner 写坏东西）────────────────────

def test_real_reports_if_any_are_valid():
    """若 `evals/reports/` 已有真实记录（本地跑过批），逐条校验。

    跑批环境没跑过就是空的 → 跳过。这条的价值在于：**runner 一旦 emit 了坏记录，
    这里会直接红**，而不是"表格里少一行"这种没人发现的静默失效。
    """
    if not R.REPORT_DIR.exists():
        pytest.skip("还没有落盘记录（未跑过批）")
    bad: list[str] = []
    for f in sorted(R.REPORT_DIR.glob("*.jsonl")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                R.validate(json.loads(line))
            except (ValueError, R.ReportError) as e:
                bad.append(f"{f.name}:{i}  {e}")
    assert not bad, "已落盘的记录里有不合规的：\n  " + "\n  ".join(bad)
