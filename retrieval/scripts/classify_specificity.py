"""L1 意图识别的第一块：Broad / Specific 查询分类评测。

**真值**：LitSearch 的 `specificity` 字段（论文 Table 1 标注准则、作者团队人工标注）：

    specificity = 1（Specific/具体）: 语料中能满足该查询的论文 **≤ 5 篇**
    specificity = 0（Broad/宽泛）   : 能满足的论文 **> 5 且 ≤ 20 篇**

    分布：Broad 155 (26.0%) / Specific 442 (74.0%)，与论文 Table 2 完全一致。

**为什么这个任务不平凡**：判定标准**不是"查询文字看起来抽象还是具体"**，而是
"**这个领域里有多少篇论文能满足它**" —— 这是**语料依赖 + 知识密集**的属性。
实测两类的表面统计差异很小（字符数中位 134 vs 137；词数中位都是 19），
所以它考验的是**世界知识估计话题拥挤度**的能力，容易被当成普通文本分类而低估难度。

**四路对照**（而不是只测 LLM，因为"能打到多少"必须有个参照系）：

    A. majority   多数类（全猜 Specific）                → 及格线 74.0%
    B. heuristic  文本表面特征（长度/约束词/专名/数字）+ 逻辑回归（交叉验证）
    C. llm        零样本 LLM（按论文原始准则判断）
    D. corpus     检索结果分布特征 + 逻辑回归（交叉验证）  ← 直接测"语料里够不够宽"

**必须单看 Broad 的 recall**：Broad 是少数类且是"该扩展"的那一类；
若 Broad recall 低，路由会漏掉大部分该扩展的查询 —— 看总 accuracy 会掩盖这一点。

用法：
    python retrieval/scripts/classify_specificity.py --limit 60      # 冒烟（会真调 LLM）
    python retrieval/scripts/classify_specificity.py                 # 全量 597
    python retrieval/scripts/classify_specificity.py --skip-llm      # 不调 LLM（只跑 A/B/D）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DERIVED = HERE / "data" / "litsearch" / "derived"
EMB = DERIVED / "emb"
QFILE = HERE / "data" / "litsearch" / "query" / "full-00000-of-00001.parquet"
RESULTS = HERE / "results"
CACHE = DERIVED / "specificity_llm.json"      # LLM 判定缓存（重跑不重复花钱）

BROAD, SPECIFIC = 0, 1


# ───────────────────────── 极简 LLM 客户端（纯 stdlib，不依赖 paperpilot）─────────
def load_env() -> None:
    """把项目根的 .env 读进 os.environ（不覆盖已存在的环境变量）。"""
    env = HERE.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class LLM:
    def __init__(self) -> None:
        self.base = os.environ.get("PAPERPILOT_LLM_BASE_URL", "").rstrip("/")
        self.key = os.environ.get("PAPERPILOT_LLM_API_KEY", "")
        self.model = os.environ.get("PAPERPILOT_LLM_MODEL", "")
        self.calls = 0
        self.ptok = 0
        self.ctok = 0

    def ok(self) -> bool:
        return bool(self.base and self.key and self.model)

    def chat(self, system: str, user: str, temperature: float = 0.0,
             timeout: int = 90) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        u = d.get("usage") or {}
        self.calls += 1
        self.ptok += int(u.get("prompt_tokens") or 0)
        self.ctok += int(u.get("completion_tokens") or 0)
        return d["choices"][0]["message"]["content"]


_SYS_LLM = (
    "你在给学术文献检索查询打「具体性(specificity)」标签。\n"
    "**判定标准不是查询文字看起来抽象还是具体，而是：在这个研究领域里，"
    "能满足该查询的论文大约有多少篇。**\n"
    "规则（出自 LitSearch 标注准则）：\n"
    "  · specificity=1（具体 Specific）：能满足该查询的论文 **≤ 5 篇**\n"
    "  · specificity=0（宽泛 Broad）   ：能满足的论文 **多于 5 篇但不超过 20 篇**\n"
    "判断方法：用你对这个研究方向的知识估计**话题的拥挤程度**——\n"
    "  热门且做法众多（如参数高效微调、提示学习）→ 偏 broad；\n"
    "  只有少数特定工作能满足（如某种特定架构+特定任务的组合）→ 偏 specific。\n"
    "注意：查询里出现少量术语不代表一定 specific，要看**整体约束的组合有多窄**。\n"
    '只输出 JSON：{"specificity": 0 或 1, "why": "一句话理由"}')


def _parse_spec(text: str) -> int | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    v = obj.get("specificity")
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return v if v in (0, 1) else None


# ───────────────────────── 特征 ─────────────────────────
_CONSTRAINT = re.compile(r"\b(and|both|with|using|via|such as|specifically|"
                         r"that (?:contain|involve|use|combine)|as well as)\b", re.I)
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
_CAMEL = re.compile(r"\b(?:[A-Z][a-z]+[A-Z][A-Za-z]*|[a-z]+[A-Z][A-Za-z]*)\b")
_NUMBER = re.compile(r"\d")


def text_features(q: str) -> dict[str, float]:
    """B 路：查询的表面特征（零成本）。"""
    words = q.split()
    return {
        "n_chars": len(q),
        "n_words": len(words),
        "n_constraint": len(_CONSTRAINT.findall(q)),
        "n_acronym": len(_ACRONYM.findall(q)),
        "n_camel": len(_CAMEL.findall(q)),
        "has_number": float(bool(_NUMBER.search(q))),
        "n_comma": q.count(","),
        "n_proper": float(sum(1 for w in words[1:] if w[:1].isupper())),
        "has_quote": float('"' in q or "'" in q),
    }


def corpus_features(sims: np.ndarray) -> dict[str, float]:
    """D 路：检索结果分布特征（零成本 —— 反正要检索）。

    直觉：Broad 查询该有**很多**中高分候选且分布平缓；
          Specific 查询该**少数高分后快速掉落**。
    直接对应真值定义（"语料里有多少篇满足"）。
    """
    s = np.sort(sims)[::-1]
    top1 = float(s[0])
    top5 = float(s[:5].mean())
    top20 = float(s[:20].mean())
    top100 = float(s[:100].mean())
    return {
        "sim_top1": top1,
        "sim_top5": top5,
        "sim_top20": top20,
        "sim_top100": top100,
        "drop_top1_5": top1 - top5,
        "drop_top5_20": top5 - top20,
        "drop_top20_100": top20 - top100,
        "ratio_20_1": top20 / max(top1, 1e-6),
        "ratio_100_20": top100 / max(top20, 1e-6),
        "n_gt_055": float((sims > 0.55).sum()),
        "n_gt_060": float((sims > 0.60).sum()),
        "n_gt_065": float((sims > 0.65).sum()),
        "std_top100": float(s[:100].std()),
    }


# ───────────────────────── 评测 ─────────────────────────
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """返回 accuracy + 每类 P/R/F1 + 混淆矩阵（Broad 是关注重点）。"""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[BROAD, SPECIFIC], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[BROAD, SPECIFIC])
    return {
        "classifier": name,
        "accuracy": float((y_true == y_pred).mean()),
        "broad_precision": float(p[0]), "broad_recall": float(r[0]), "broad_f1": float(f[0]),
        "specific_precision": float(p[1]), "specific_recall": float(r[1]),
        "specific_f1": float(f[1]),
        "n_broad": int((y_true == BROAD).sum()), "n_specific": int(sup[1]),
        "cm": cm.tolist(),   # [[TN_broad_correct, broad→specific], [specific→broad, correct]]
    }


def cv_accuracy(X: np.ndarray, y: np.ndarray, seed: int = 20260918) -> tuple[np.ndarray, float]:
    """5 折分层交叉验证 + 返回袋外预测（用于算每类指标）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), -1, dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
    return oof, float((oof == y).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟）")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    load_env()
    q = pd.read_parquet(QFILE)
    if args.limit:
        # 冒烟时**保持类别比例**（每类取一半），否则小样本里 Broad 可能一条都没有
        per = max(2, args.limit // 2)
        q = q.groupby("specificity", group_keys=False).head(per).reset_index(drop=True)
    y = q["specificity"].to_numpy()
    n = len(q)
    print(f"查询 {n} 条 | Broad {(y == BROAD).sum()} / Specific {(y == SPECIFIC).sum()}")

    rows: list[dict] = []

    # ── A. 多数类基线 ──────────────────────────────────────
    maj = np.full(n, SPECIFIC)
    rows.append(evaluate(y, maj, "A. 多数类（全猜 Specific）"))
    print(f"  A. 多数类 accuracy = {(y == maj).mean():.4f}")

    # ── B. 表面特征 + 逻辑回归（交叉验证）──────────────────
    ftxt = pd.DataFrame([text_features(str(s)) for s in q["query"]])
    Xb = ftxt.to_numpy(dtype=float)
    pred_b, acc_b = cv_accuracy(Xb, y)
    rows.append(evaluate(y, pred_b, "B. 表面特征 + LR (5折CV)"))
    print(f"  B. 表面特征 accuracy = {acc_b:.4f}  （{Xb.shape[1]} 维）")

    # ── D. 检索分布特征 + 逻辑回归（交叉验证）──────────────
    print("  计算检索相似度分布（编码 597 条查询）…")
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    model.max_seq_length = 512
    if dev == "cuda":
        model.half()
    qv = model.encode(q["query"].tolist(), batch_size=32, normalize_embeddings=True,
                      show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    mat = np.concatenate([np.load(p) for p in sorted(EMB.glob("part_*.npy"))], axis=0)
    sims = qv @ mat.T                       # (n, 63269) cosine（都已归一化）
    print(f"    相似度矩阵 {sims.shape}")
    fcor = pd.DataFrame([corpus_features(sims[i]) for i in range(n)])
    Xd = fcor.to_numpy(dtype=float)
    pred_d, acc_d = cv_accuracy(Xd, y)
    rows.append(evaluate(y, pred_d, "D. 检索分布 + LR (5折CV)"))
    print(f"  D. 检索分布 accuracy = {acc_d:.4f}  （{Xd.shape[1]} 维）")

    # ── C. LLM 零样本 ──────────────────────────────────────
    if not args.skip_llm:
        llm = LLM()
        if not llm.ok():
            print("  C. LLM 跳过（未配置 PAPERPILOT_LLM_*）")
        else:
            cache: dict[str, int] = {}
            if CACHE.exists():
                try:
                    cache = json.loads(CACHE.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    cache = {}
            need = [i for i in range(n) if str(q.loc[i, "query"]) not in cache]
            print(f"  C. LLM 零样本：缓存命中 {n - len(need)}，待判定 {len(need)}（workers={args.workers}）")
            t0 = time.time()
            if need:
                def work(i: int):
                    txt = str(q.loc[i, "query"])
                    try:
                        return txt, _parse_spec(llm.chat(_SYS_LLM, f"查询：{txt}"))
                    except (urllib.error.URLError, TimeoutError, OSError) as e:  # noqa: BLE001
                        return txt, None
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    for k, (txt, v) in enumerate(ex.map(work, need), 1):
                        if v is not None:
                            cache[txt] = v
                        if k % 50 == 0:
                            print(f"      {k}/{len(need)}  {time.time() - t0:.0f}s")
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            pred_c = np.array([cache.get(str(s), -1) for s in q["query"]])
            n_fail = int((pred_c < 0).sum())
            ok = pred_c >= 0
            if ok.sum() > 0:
                rows.append(evaluate(y[ok], pred_c[ok], f"C. LLM 零样本（有效 {ok.sum()}）"))
                print(f"  C. LLM accuracy = {(y[ok] == pred_c[ok]).mean():.4f}"
                      f"  | 失败 {n_fail} 条 | 用时 {time.time() - t0:.0f}s"
                      f" | tokens {llm.ptok}+{llm.ctok}")

    # ── 报告 ───────────────────────────────────────────────
    tbl = pd.DataFrame(rows)
    cols = ["classifier", "accuracy", "broad_precision", "broad_recall", "broad_f1",
            "specific_precision", "specific_recall", "specific_f1"]
    print("\n" + "=" * 104)
    print("Broad/Specific 分类评测（Broad = specificity 0 = 语料中 ≤20 篇可满足）")
    print("=" * 104)
    print(tbl[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n混淆矩阵 [行=真值, 列=预测]，顺序 [Broad, Specific]：")
    for r in rows:
        print(f"  {r['classifier']:<32} {r['cm']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    def md_table(df: pd.DataFrame) -> str:
        c = [str(x) for x in df.columns]
        out = ["| " + " | ".join(c) + " |", "|" + "|".join(["---"] * len(c)) + "|"]
        for _, rr in df.iterrows():
            out.append("| " + " | ".join(
                f"{rr[x]:.4f}" if isinstance(rr[x], float) else str(rr[x]) for x in df.columns) + " |")
        return "\n".join(out)

    out = RESULTS / "LITSEARCH_SPECIFICITY.md"
    out.write_text(
        f"# Broad / Specific 查询分类评测（{time.strftime('%Y%m%d_%H%M%S')}）\n\n"
        f"- 真值：LitSearch `specificity`（论文 Table 1：Broad = 语料中可满足 **>5 且 ≤20 篇**；"
        f"Specific = **≤5 篇**；作者团队逐条人工标注）\n"
        f"- 分布：Broad {(y == BROAD).sum()} / Specific {(y == SPECIFIC).sum()}（共 {n}）\n"
        f"- **多数类及格线 = {(y == SPECIFIC).mean():.4f}**（全猜 Specific）\n"
        f"- B/D 为 5 折分层交叉验证的袋外结果；C 为零样本（未训练）\n\n"
        + md_table(tbl[cols]) + "\n\n## 混淆矩阵（行=真值，列=预测，顺序 [Broad, Specific]）\n\n"
        + "\n".join(f"- `{r['classifier']}` → {r['cm']}" for r in rows) + "\n",
        encoding="utf-8")
    (RESULTS / "specificity_report.json").write_text(
        json.dumps({"n": n, "rows": rows,
                    "n_constraint_feats": int(Xb.shape[1]), "n_corpus_feats": int(Xd.shape[1])},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写出：{out.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
