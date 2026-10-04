"""**评测结果的统一记录格式** —— 让 L1/L2/L3 的数字能放在一张表里比。

## 问题（2026-10-05 普查）

三层各写各的，格式和位置都不同：

    L1 篇级    `run_retrieval_eval.py`  → **纯文本**（`"\\n".join(...)`）
    L2 判官    `_r2_retr_eval.py`       → **CSV**（`to_csv`）
    L3 端到端  `run_group_qa.py`        → **JSON**（每题明细）

而且**汇总指标基本只 `print` 到 stdout，不落盘** —— 跑完就没了。
后果：**无法横向对比，也无法与历史比**（要翻 stdout 日志考古）。

## 格式（**刻意做得小**）

一条记录 = 一个"可比的数"：

    layer          L0..L4               分层
    name           本次运行/评测项名      如 "group2_graph"
    metric         指标名（统一命名）     如 "qa.ok_strict"
    value          数值
    n             **样本量**             ★ 没有 n 的率不可比
    note          **口径**               ★ 不写口径就会比错（见下）
    evidence_path  原始产物路径（可回溯）
    at            时间戳
    baseline/delta 可选：与基线比

## 为什么 `n` 与 `note` 是**必填**

· **`n`**：90 题 51% 与 11 题 51% 是两回事，光看率会得出相反结论。
· **`note`**：`ok` / `ok_strict` / `MRecall@5` 名字相近但含义不同 ——
  这正是这套评测最容易"用自己的口径跟别人的数字比"的地方。
  **没写口径的指标，比没有更危险**（它看起来能比）。

## 落盘位置

`evals/reports/<layer>.jsonl`（**gitignore**：可再生成）。要入库的是
**汇总摘要**（`.md`），不是每次跑的原始记录 —— 与"报告入库、产物不入库"一致。

## 用法

    from evals.report import emit          # runner 里落盘
    emit(layer="L3", name="group2_graph", metric="qa.ok_strict",
         value=0.51, n=90, note="严格锚点全命中／全部题", evidence_path="...")

    python evals/report.py --table         # 所有记录汇总成一张表
    python evals/report.py --md            # 导出 Markdown 摘要（可入库）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "evals" / "reports"

LAYERS = ("L0", "L1", "L2", "L3", "L4")

# 必填字段与其理由（放进错误信息里，让"缺字段"的人一眼知道为什么不能缺）
REQUIRED: dict[str, str] = {
    "layer": "分层（L0..L4）",
    "name": "本次运行/评测项名，如 group2_graph",
    "metric": "指标名，如 qa.ok_strict（小写、点分隔、无空格）",
    "value": "数值（int/float；算不出来就别 emit，不要写 None）",
    "n": "样本量 —— 没有 n 的率不可比",
    "note": "口径说明 —— 不写口径的指标比没有更危险（看起来能比）",
    "evidence_path": "证据路径：原始产物在哪（可回溯，别只给数字）",
}

# 指标名约定：小写、点分隔（`qa.ok_strict` / `retrieval.mrecall@5`）。
METRIC_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_@]+)+$")


class ReportError(ValueError):
    """记录不合规（缺字段 / 类型错 / 指标名不合约定）。"""


def validate(rec: dict[str, Any]) -> None:
    """校验一条记录；不合规则抛 `ReportError`，附**缺失理由**。"""
    missing = [k for k in REQUIRED if rec.get(k) in (None, "")]
    if missing:
        why = "\n".join(f"    {k}：{REQUIRED[k]}" for k in missing)
        raise ReportError(f"记录缺以下字段：\n{why}\n  收到：{rec}")

    if rec["layer"] not in LAYERS:
        raise ReportError(f"layer 必须是 {LAYERS} 之一，收到 {rec['layer']!r}")
    for k in ("value", "n"):
        if isinstance(rec[k], bool) or not isinstance(rec[k], (int, float)):
            raise ReportError(f"{k} 必须是数值，收到 {rec[k]!r}")
    if rec["n"] <= 0:
        raise ReportError(f"n 必须 > 0（没有样本量的率不可比），收到 {rec['n']!r}")
    if not METRIC_RE.match(str(rec["metric"])):
        raise ReportError(
            f"metric 需形如 `qa.ok_strict` / `retrieval.mrecall@5`"
            f"（小写、点分隔、无空格），收到 {rec['metric']!r}")


def emit(*records: dict[str, Any], path: Path | None = None) -> Path:
    """校验并**追加**写入 `evals/reports/<layer>.jsonl`（同一层一个文件）。

    记录按**层**分文件（而不是一次运行一个文件）：同一层的历次结果天然排在一起，
    正好是"跟上次比"要的形状。
    """
    if not records:
        raise ReportError("emit() 至少要一条记录")
    for r in records:
        validate(r)
    p = path or (REPORT_DIR / f"{records[0]['layer']}.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for r in records:
            out = dict(r)
            out.setdefault("at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    return p


def load_all(root: Path | None = None) -> list[dict[str, Any]]:
    """读回所有记录（按层分文件）。坏行**跳过并告警**，不让一行脏数据废掉整表。"""
    d = root or REPORT_DIR
    recs: list[dict[str, Any]] = []
    for f in sorted(d.glob("*.jsonl")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except ValueError:
                print(f"⚠️ 跳过坏行 {f.name}:{i}", file=sys.stderr)
    return recs


def _latest(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """每个 `(layer, name, metric)` 只留**最新**一条 —— 表格才不会被历史淹没。"""
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in recs:
        k = (str(r.get("layer")), str(r.get("name")), str(r.get("metric")))
        if k not in best or str(r.get("at", "")) >= str(best[k].get("at", "")):
            best[k] = r
    return list(best.values())


def table(recs: list[dict[str, Any]], *, latest: bool = True) -> str:
    """对齐成一张表。默认只显示每个指标的**最新**值。"""
    rows = _latest(recs) if latest else recs
    if not rows:
        return "（还没有任何记录 —— 先让 runner 调用 `evals.report.emit`）"
    rows.sort(key=lambda r: (str(r.get("layer")), str(r.get("metric")), str(r.get("name"))))
    w = (10, 22, 20, 10, 6)
    head = f"{'层':<{w[0]}}{'指标':<{w[2]}}{'运行/项':<{w[1]}}{'值':>{w[3]}}{'n':>{w[4]}}"
    lines = [head, "-" * len(head)]
    for r in rows:
        v = r.get("value")
        vs = f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"{str(r.get('layer')):<{w[0]}}{str(r.get('metric')):<{w[2]}}"
                     f"{str(r.get('name')):<{w[1]}}{vs:>{w[3]}}{str(r.get('n')):>{w[4]}}")
    lines.append("")
    lines.append("口径（note）—— ★ 比数字前先读这里：")
    for r in rows:
        lines.append(f"  [{r.get('layer')}] {r.get('metric')} ｜ {r.get('name')}")
        lines.append(f"        {r.get('note')}")
        if r.get("evidence_path"):
            lines.append(f"        证据：{r.get('evidence_path')}")
    return "\n".join(lines)


def to_markdown(recs: list[dict[str, Any]]) -> str:
    """导出 Markdown 摘要（**这个可以入库**）。"""
    rows = _latest(recs)
    rows.sort(key=lambda r: (str(r.get("layer")), str(r.get("metric")), str(r.get("name"))))
    out = ["# 评测结果汇总", "",
           "| 层 | 指标 | 运行/项 | 值 | n | 口径 | 证据 |",
           "|---|---|---|---:|---:|---|---|"]
    for r in rows:
        v = r.get("value")
        vs = f"{v:.4f}" if isinstance(v, float) else str(v)
        out.append(f"| {r.get('layer')} | `{r.get('metric')}` | {r.get('name')} | {vs} "
                   f"| {r.get('n')} | {r.get('note')} | `{r.get('evidence_path')}` |")
    out += ["", f"（共 {len(rows)} 条；生成于 {datetime.now(timezone.utc).isoformat(timespec='seconds')}）"]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="评测结果统一记录（汇总/导出）")
    ap.add_argument("--table", action="store_true", help="打印汇总表（默认）")
    ap.add_argument("--md", action="store_true", help="导出 Markdown 摘要到 stdout")
    ap.add_argument("--all", action="store_true", help="显示全部历史（默认只显示最新）")
    args = ap.parse_args()

    recs = load_all()
    if not recs:
        print(f"（{REPORT_DIR.relative_to(ROOT)} 下还没有记录）")
        return 0
    print(to_markdown(recs) if args.md else table(recs, latest=not args.all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
