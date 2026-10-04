"""**用官方 LoFT CLI 逐个跑所有运行的预测**（闭合审计链）

对每个 `R2_QAMPARI_*.csv`：
  1. 把预测写成官方要求的 `preds.jsonl`（`{"qid","model_outputs":[[...]],"num_turns":1}`）
  2. 以 **subprocess 调官方 CLI** `run_evaluation.py --task_type multi_value_rag`
  3. 收官方 metrics.json
  4. 另算 **ALCE/HELMET 官方** `compute_qampari_f1` 的 precision / recall
产出：`results/official_runs/<run>/{queries,preds,preds_metrics*}`（全部可回溯）
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = Path(__file__).resolve().parents[1]
OFF = R / "data" / "loft" / "official"
PY = str(R.parent / ".venv" / "Scripts" / "python.exe")
GOLD_SRC = OFF / "audit" / "queries.jsonl"          # 官方格式 gold（100 题）
OUT = R / "results" / "official_runs"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(OFF))
from evaluation import utils as ou  # noqa: E402

gold_all = {json.loads(l)["qid"]: json.loads(l)["answers"]
            for l in GOLD_SRC.read_text(encoding="utf-8").splitlines() if l.strip()}


def parse_pred(s) -> list[str]:
    s = str(s)
    try:
        v = json.loads(s)
    except Exception:  # noqa: BLE001
        try:
            v = json.loads(s + '"]')
        except Exception:  # noqa: BLE001
            return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v] if isinstance(v, list) else []


def alce(gold, pred):
    preds = [p for p in pred if p]
    flat = list(gold)
    prec = (sum(1 for p in preds if p in flat) / len(preds)) if preds else 0.0
    hit = sum(1 for g in gold if g in preds)
    rec = hit / len(gold) if gold else 0.0
    return len(preds), prec, rec, (0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))


rows = []
for f in sorted((R / "results").glob("R2_QAMPARI_*.csv")):
    if "PRECISION" in f.name:
        continue
    run = f.stem.replace("R2_QAMPARI_", "")
    d = pd.read_csv(f)
    rd = OUT / run
    rd.mkdir(parents=True, exist_ok=True)
    shutil.copy(GOLD_SRC, rd / "queries.jsonl")
    # ⚠️ 官方 CLI 用 `open(path)` 平台默认编码（Windows=GBK）读文件 →
    #    预测必须 **ASCII 转义**（ensure_ascii=True），否则 UnicodeDecodeError。
    with (rd / "preds.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for _, r in d.iterrows():
            fh.write(json.dumps({"qid": r["qid"], "model_outputs": [parse_pred(r["pred_raw"])],
                                 "num_turns": 1}, ensure_ascii=True) + "\n")
    cp = subprocess.run([PY, str(OFF / "run_evaluation.py"),
                         "--answer_file_path", str(rd / "queries.jsonl"),
                         "--pred_file_path", str(rd / "preds.jsonl"),
                         "--task_type", "multi_value_rag"],
                        capture_output=True, text=True, encoding="utf-8", cwd=str(OFF))
    mpath = rd / "preds_metrics.json"
    if not mpath.exists():
        print(f"  ✗ {run} 官方 CLI 失败：{cp.stderr[-200:]}")
        continue
    m = json.loads(mpath.read_text(encoding="utf-8"))["quality"]
    np_, pr_, rc_, f1_ = [], [], [], []
    for _, r in d.iterrows():
        g = [ou.normalize_answer(x) for x in gold_all.get(r["qid"], [])]
        p = [ou.normalize_answer(x) for x in parse_pred(r["pred_raw"])]
        a, b, c, e = alce(g, p)
        np_.append(a); pr_.append(b); rc_.append(c); f1_.append(e)
    rows.append(dict(运行=run, n=len(d), gold均=np.mean([len(gold_all.get(q, [])) for q in d["qid"]]),
                     预测均=float(np.mean(np_)),
                     比值=float(np.mean(np_)) / np.mean([len(gold_all.get(q, [])) for q in d["qid"]]),
                     官方em=m["em"], 官方coverage=m["coverage"], 官方subspan_em=m["subspan_em"],
                     ALCE_P=float(np.mean(pr_)), ALCE_R=float(np.mean(rc_)), ALCE_F1=float(np.mean(f1_))))

t = pd.DataFrame(rows).sort_values("官方subspan_em", ascending=False)
print("=" * 118)
print("【官方 CLI 逐个运行（multi_value_rag）】全部产物已存 results/official_runs/<run>/")
print(f"  {'运行':<20}{'n':>4}{'gold均':>8}{'预测均':>8}{'比值':>7}"
      f"{'官方em':>8}{'官方cov':>9}{'官方subspan':>11}{'ALCE_P':>8}{'ALCE_R':>8}{'ALCE_F1':>9}")
for _, r in t.iterrows():
    print(f"  {r['运行']:<20}{int(r['n']):>4}{r['gold均']:>8.2f}{r['预测均']:>8.2f}{r['比值']:>7.3f}"
          f"{r['官方em']:>8.3f}{r['官方coverage']:>9.4f}{r['官方subspan_em']:>11.3f}"
          f"{r['ALCE_P']:>8.3f}{r['ALCE_R']:>8.3f}{r['ALCE_F1']:>9.3f}")
t.to_csv(R / "results" / "R2_QAMPARI_OFFICIAL_ALLRUNS.csv", index=False, encoding="utf-8")
print(f"\n已写 {R / 'results' / 'R2_QAMPARI_OFFICIAL_ALLRUNS.csv'}")
print(f"官方产物根目录：{OUT}")
