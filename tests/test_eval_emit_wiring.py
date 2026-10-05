"""L0 不变量 · **跑批入口的 emit 接线**：汇总记录必须真能落盘。

## 为什么单独一个文件

`test_evals_report.py` 守「记录**合不合规**」，`test_metrics_ratchet.py` 守
「掉点**有没有人管**」—— 但**没有一条守「入口到底有没有把记录写出来」**。

于是 2026-10-05 实测撞上：README 写着「L1/L2/L3 已全部接上 `report.emit`」，
而 `_r2_retr_eval.py` 的 `_emit_report` 引用了 `main()` 的**局部** `args` 却没传进去：

    NameError: name 'args' is not defined

它被函数内的 `except Exception` **静默吞掉**（设计如此：报告失败不该弄挂评测）：
评测照跑、数字都对，只是**一条记录都没落** —— 整批跑完只在 stderr 留一行 `⚠️`。
README 写"已接上"的那一天到真跑的那一天之间，这个 bug **不可能被发现**。

★ 本测试断言的是**副作用**（记录真的落了、指标齐全、口径写进去了），
  **不是**"没抛异常" —— 对上面那个 bug，后者完全无效（异常被吞掉，调用方看不到）。
★ 「测试也要被测」：修复前那次真实跑批的 stderr 就是
  `⚠️ 汇总记录落盘失败（…）：NameError: name 'args' is not defined`
  —— 本测试断言 `"落盘失败" not in stderr`，故当时**必然红**（非推测）。

## 为什么不放进 `test_eval_assets.py`

那个文件整体 `pytestmark = skipif(无 git 工作区)` —— 本测试与 git 无关，
放进去会在"无 git 环境"下被静默跳过（又一个"看起来在守、其实没跑"）。
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "retrieval" / "tmp" / "_r2_retr_eval.py"

# `_emit_report` 应落的六个指标（与它内部的 `COLS` 一一对应）
WANT = {"r2.mrecall@10", "r2.strecall@10", "r2.setf1@10",
        "r2.andcg@10", "r2.ev_recall@12", "r2.ev_recall_c12k"}

_MOD = None


def _load():
    """按**路径**加载 L2 检索侧入口（延迟到首次调用，避免收集期副作用）。

    ⚠️ 不能 `import retrieval.tmp._r2_retr_eval`：该脚本靠 `__file__` 推
    `HERE` / `ROOT`，并按相对 `retrieval/` 算数据路径（133/135 个脚本都是这个模式）。
    """
    global _MOD
    if _MOD is None:
        spec = importlib.util.spec_from_file_location("r2_retr_eval_wiring", SRC)
        assert spec and spec.loader, SRC
        mod = importlib.util.module_from_spec(spec)
        sys.modules["r2_retr_eval_wiring"] = mod
        spec.loader.exec_module(mod)
        _MOD = mod
    return _MOD


def _rows(**over) -> pd.DataFrame:
    """一份最小的「主表」：一行 = 一个 (arm, cluster, facet)。"""
    data = {
        "arm": ["A_x", "A_x"],
        "MRecall@10": [0.40, 0.60],
        "StRecall@10": [0.80, 0.90],
        "setF1@10": [0.40, 0.50],
        "aNDCG@10": [0.30, 0.40],
        "ev_recall@12": [0.50, 0.60],
        "ev_recall_c12k": [0.45, 0.55],
    }
    data.update(over)
    return pd.DataFrame(data)


def _call_emit(mod, m: pd.DataFrame, tmp_path: Path) -> None:
    """按**真实签名**调 `_emit_report`。

    `args` 以关键字显式传入 —— 这正是当初漏掉的那一步。
    `KDIR` / `GOLDF` 只用到 `.name`（写进口径），不读文件，故无需真实存在。
    """
    mod._emit_report(
        ["A_x"], m, [0, 0, 0],
        KDIR=mod.DEV / "prodchunk", GOLDF=mod.GOLD2,
        outp=tmp_path / "fake.csv",
        args=SimpleNamespace(corpus="20", combos="new", gold="new"))


def _read(tmp_path: Path) -> list[dict]:
    f = tmp_path / "L2.jsonl"
    if not f.exists():
        return []
    return [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ─────────────── ① 结构：辅助函数必须**拿得到** `args` ───────────────

def test_emit_report_signature_receives_args():
    """`_emit_report` 必须显式接收 `args`。

    它要把口径（`args.corpus` / `args.combos`）写进 `note`，而 `note` 是"能不能比"
    的前提；但它是**模块级函数**，看不到 `main()` 的局部 `args`。
    曾发生：漏传 → `NameError` → 被 `except` 吞掉 → **一天没记录也没人发现**。
    """
    mod = _load()
    params = inspect.signature(mod._emit_report).parameters
    assert "args" in params, (
        "`_emit_report` 没接收 `args` —— 它写口径时会 `NameError`，"
        "而异常会被函数内的 `except` 静默吞掉（评测照跑、**无记录**）。\n"
        f"当前签名：_emit_report{inspect.signature(mod._emit_report)}")


# ─────────────── ② 行为：记录真的落盘，且**不走静默失败分支** ───────────────

def test_l2_retr_emit_wiring_writes_records(tmp_path, monkeypatch, capsys):
    """L2 检索侧的 `_emit_report` 必须真把记录落盘。"""
    mod = _load()
    import evals.report as R            # 落盘目标由 `R.REPORT_DIR` 决定
    monkeypatch.setattr(R, "REPORT_DIR", tmp_path)

    _call_emit(mod, _rows(), tmp_path)

    err = capsys.readouterr().err
    assert "落盘失败" not in err, (
        "emit 走进了**静默失败分支** —— 评测数字仍然对，但一条记录都没落：\n  "
        + err.strip()
        + "\n★ 2026-10-05 的真实案例：`NameError: name 'args' is not defined`。")

    recs = _read(tmp_path)
    assert recs, "`_emit_report` 完全没落盘 —— 接线坏了（它自己会吞异常，不会报错）"
    assert {r["metric"] for r in recs} == WANT, (
        f"指标不全，得到 {sorted(r['metric'] for r in recs)}")
    # ★ `n` 与 `note` 是"能不能比"的前提（`evals/report.py` 会拦，这里确认真的写进去了）
    assert all(r["n"] == 2 for r in recs), "样本量必须是主表的真实行数"
    assert all(r["note"] for r in recs), "口径（note）必须落进去，否则数字不可比"
    assert all(r["layer"] == "L2" for r in recs)


# ─────────────── ③ 降级：算不出 / 缺列 → 跳过该项，不写 NaN、不崩 ───────────────

def test_emit_skips_all_nan_metric_without_silent_failure(tmp_path, monkeypatch, capsys):
    """全 NaN 的指标应被**跳过**（`report.py` 会拒收 NaN），而不是写进去或崩掉。

    NaN 是 float，**能骗过类型检查**，然后在表里变成"NaN"、比大小永远为假。
    """
    mod = _load()
    import evals.report as R
    monkeypatch.setattr(R, "REPORT_DIR", tmp_path)

    m = _rows()
    m.loc[0, "ev_recall_c12k"] = float("nan")
    m.loc[1, "ev_recall_c12k"] = float("nan")
    _call_emit(mod, m, tmp_path)

    err = capsys.readouterr().err
    assert "落盘失败" not in err, f"跳过 NaN 不该走失败分支：\n  {err.strip()}"
    got = {r["metric"] for r in _read(tmp_path)}
    assert "r2.ev_recall_c12k" not in got, "全 NaN 的指标必须被跳过，不能写成 NaN"
    assert "r2.mrecall@10" in got, "其它指标不该被连累"


def test_emit_skips_missing_column(tmp_path, monkeypatch, capsys):
    """列不存在（口径变了 / 脚本降级）→ 跳过该指标，不崩、也不静默失败。"""
    mod = _load()
    import evals.report as R
    monkeypatch.setattr(R, "REPORT_DIR", tmp_path)

    m = _rows().drop(columns=["aNDCG@10"])
    _call_emit(mod, m, tmp_path)

    err = capsys.readouterr().err
    assert "落盘失败" not in err, f"缺列不该走失败分支：\n  {err.strip()}"
    got = {r["metric"] for r in _read(tmp_path)}
    assert "r2.andcg@10" not in got, "缺的列应被跳过"
    assert len(got) == len(WANT) - 1, f"其余指标应完整，得到 {sorted(got)}"
