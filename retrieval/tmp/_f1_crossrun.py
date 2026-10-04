"""跨运行对照：每次跑的 subspan_em 与 F1 是否稳定满足「F1 > subspan_em」。"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = Path(__file__).resolve().parents[1] / "results"

rows = []
for f in sorted(glob.glob(str(R / "R2_QAMPARI_*.csv"))):
    d = pd.read_csv(f)
    name = Path(f).stem.replace("R2_QAMPARI_", "")
    if "subspan_em" not in d.columns:
        continue
    rows.append(dict(
        运行=name, n=len(d),
        subspan_em=d["subspan_em"].mean(),
        coverage=d["coverage"].mean() if "coverage" in d else float("nan"),
        setP=d["setP"].mean() if "setP" in d else float("nan"),
        setF1=d["setF1"].mean() if "setF1" in d else float("nan"),
        em=d["em"].mean() if "em" in d else float("nan"),
        均预测数=d["n_pred"].mean() if "n_pred" in d else float("nan"),
    ))
t = pd.DataFrame(rows)
t["F1-subspan"] = t["setF1"] - t["subspan_em"]
print(f"{'运行':<26}{'n':>4}{'subspan_em':>11}{'集合F1':>9}{'Δ(F1-sub)':>11}"
      f"{'集合P':>8}{'coverage':>9}{'em':>7}{'均预测':>7}")
for _, r in t.iterrows():
    print(f"{r['运行']:<26}{int(r['n']):>4}{r['subspan_em']:>11.4f}{r['setF1']:>9.4f}"
          f"{r['F1-subspan']:>+11.4f}{r['setP']:>8.3f}{r['coverage']:>9.3f}"
          f"{r['em']:>7.3f}{r['均预测数']:>7.2f}")
print(f"\n所有运行都满足 F1 > subspan_em？"
      f"{'✅ 是' if (t['F1-subspan'] > 0).all() else '❌ 否'}（{int((t['F1-subspan'] > 0).sum())}/{len(t)}）")
print(f"Δ 范围：{t['F1-subspan'].min():+.4f} ~ {t['F1-subspan'].max():+.4f}")
