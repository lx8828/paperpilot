"""**A｜R2 真值重标**：用双 LLM（异源）重标全部 (论文, facet) 对，修正"词面锚点漏 ~43%"

## 问题（`R2_CALIB_20260928.md §A.5`）
锚点真值 precision 0.895 ／ recall **0.567** → **漏 ~43% 的真阳性**
（只写 "we remove each component" 的篇不算"做了消融"）。绝对值因此不可报。

## 做法
对每个 (cluster, facet, paper) 对：
1. **证据池** = （锚点命中的块）∪（用**干净查询** `zh` + 净化子查询做语义检索的 top-k 块）
   → 前者是现行真值依据；后者专门用来**捞出被锚点漏掉的正例**
2. **两个异源 LLM 各判一次**（`deepseek-chat` / `glm-4-flash`），四档
   （`entail/partial/neutral/contradict`）+ 置信度 + **逐字引用 span**
3. **采信规则**：
   - 两方都 `entail` → **正例**
   - 两方都非 `entail` → **负例**
   - 分歧 → **第三轮**：用 `deepseek` 跑**严格对抗提示**（"能否找到理由说明这**不是本文自己做的**？找不到才回答 YES"），
     并标 `low_conf`
4. 提示里显式加纪律：**证据描述的动作若属"被引用的他人工作"（`X et al. proposed…`）→ 判 neutral**
   （这正是 `glm-4-flash` 已知的错法）

## 产出
`data/r2dev/gold_recalib.csv` —— 逐 (cluster, facet, docid) 的 `anchor_gold` vs `new_gold`
+ 两方判定 + 引用 span + 是否低置信。**不改任何既有文件**。

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_recalib.py --limit 12   # 冒烟
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_gold_recalib.py --workers 8  # 全量
"""
from __future__ import annotations

import importlib.util as _iu
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _hfcache import ensure_hf_home  # noqa: E402

ensure_hf_home()
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))

spec = _iu.spec_from_file_location("stdm", HERE / "tmp" / "_r2_std_metrics.py")
_m = _iu.module_from_spec(spec)
sys.modules["stdm"] = _m
spec.loader.exec_module(_m)
F = _m.F

CACHE = HERE / "data" / "r2dev" / "clusters"
SUBQ = HERE / "data" / "r2dev" / "subqueries.json"
OUT = HERE / "data" / "r2dev" / "gold_recalib.csv"
CHUNK, OVERLAP = 1000, 100
EV_TOPK_SEM, EV_MAX_CHUNKS, EV_MAX_CHARS = 3, 4, 4200

SYS = """你是严格的事实核查员，判断**某篇论文**是否做了某件事。你会看到该论文的若干原文片段。

只输出四档之一：
- `entail`：证据**足以支持**"该论文做了这件事"（本文自己做的）。
- `partial`：只覆盖一部分，或需要额外假设才成立。
- `neutral`：证据与该论断**无关或不足**；或者**这件事是本文引用的他人工作**（见纪律 2）。
- `contradict`：证据**明确表明**该论断为假。

判定纪律（务必逐条遵守）：
1. 以证据字面为准，**不要用先验知识补充**；"没提到"一律 `neutral`，不要判 `contradict`。
2. ⚠️ **区分"本文做的"与"引用他人做的"**：若证据里动作的主语是**被引用的他文**
   （如 "X et al. proposed…"、"Prior work introduces…"、"Several studies have…"），
   且**没有**说明本文自己也做了 → 判 `neutral`。这是最常见的错法，务必小心。
3. **列出所有答案不等于做了这件事**：证据只是"任务清单/评价指标列表/相关工作列表"里出现该词 → `neutral`。
4. 中文论断 ↔ 英文证据，注意同义改写（"做了消融"↔"we ablate each component"、
   "公开了代码"↔"code is available at github.com"）。

只输出 JSON：
{"label":"entail|partial|neutral|contradict","confidence":0.0~1.0,
 "quote":"证据中支持该判定的**逐字原文片段**（≤120 字符；判 neutral/contradict 时给最能说明问题的片段，找不到给空串）",
 "reason":"不超过 30 字的中文理由"}"""

SYS_STRICT = """你是**审稿人式的严格核查员**。任务：判断「某篇论文**自己**做了某件事」是否成立。

你会看到该论文的若干原文片段。请执行一次**对抗式复核**：
- 主动寻找"这**不算**本文做了这件事"的理由：是不是引用的他人工作？是不是只是任务清单里的一个词？
  是不是评价指标/相关工作/未来工作里的提及？是不是只覆盖了一部分？
- **只有找不到任何这类理由时**，才回答 YES。

只输出 JSON：{"answer":"YES|NO","why":"≤30 字中文理由","quote":"最能支持你判定的逐字原文（≤120 字符，找不到给空串）"}"""


def call(prefix: str, system: str, user: str, max_tokens: int = 400) -> dict:
    for attempt in range(3):
        try:
            obj = llm.chat_json(system, user, temperature=0.0, max_tokens=max_tokens, prefix=prefix)
            if isinstance(obj, dict):
                return obj
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return {"_err": f"{type(e).__name__}: {str(e)[:90]}"}
            time.sleep(1.5 * (attempt + 1))
    return {"_err": "retry failed"}


def toks(s: str) -> set[str]:
    import re
    return {t for t in re.findall(r"[a-z]{3,}", str(s).lower())}


def rx_toks(pat: str) -> set[str]:
    import re
    return {t for t in re.findall(r"[a-z]{3,}", re.sub(r"\\[a-zA-Z]", " ", str(pat)).lower())}


def main() -> int:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 对（冒烟）")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    base_l, _, model_l = llm.config("PAPERPILOT_LLM")
    base_j, _, model_j = llm.config("PAPERPILOT_JUDGE")
    print(f"判官 A：{model_l}（PAPERPILOT_LLM）｜判官 B：{model_j}（PAPERPILOT_JUDGE）\n")

    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    subq_all = json.loads(SUBQ.read_text(encoding="utf-8"))

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("BAAI/bge-m3", device=dev, trust_remote_code=True)
    enc.max_seq_length = 512
    if dev == "cuda":
        enc.half()

    jobs: list[dict] = []
    for ci in range(len(meta)):
        sub = pd.read_parquet(CACHE / f"c{ci}.parquet")
        docs = sub["docid"].tolist()
        titles = (sub["title"].tolist() if "title" in sub.columns
                  else ["" for _ in docs])
        chunks, owner = [], []
        for i, t in enumerate(sub["full_paper"].tolist()):
            cs = _m.chunks_of(t)
            chunks += cs
            owner += [docs[i]] * len(cs)
        owner = np.array(owner)
        C = enc.encode(chunks, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        low = [c.lower() for c in chunks]
        idx = {d: np.where(owner == d)[0] for d in docs}
        print(f"【簇{ci + 1}】{len(docs)} 篇 / {len(chunks)} 块", flush=True)

        for facet in meta[ci]["usable"]:
            pat, zh, anchor, para = F[facet]
            hit = np.array([_m.judge(c, facet) for c in low])
            anchor_gold = {d for d in docs if hit[idx[d]].any()}
            if not anchor_gold or len(anchor_gold) == len(docs):
                continue
            rx = rx_toks(pat)
            desens = [" ".join(w for w in s.split() if w.lower() not in rx) or s
                      for s in subq_all.get(facet, [])]
            qvec = enc.encode([zh] + desens, normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
            sim = qvec @ C.T
            for d in docs:
                ids = idx[d]
                sem_top = ids[np.argsort(-sim[:, ids].max(axis=0))[:EV_TOPK_SEM]]
                anch = ids[hit[ids]][:EV_MAX_CHUNKS]
                orderl, seen = [], set()
                for j in list(anch) + list(sem_top):
                    if int(j) not in seen:
                        seen.add(int(j))
                        orderl.append(int(j))
                orderl = orderl[:EV_MAX_CHUNKS]
                ev, tot = [], 0
                for j in orderl:
                    t = chunks[j]
                    if tot + len(t) > EV_MAX_CHARS:
                        t = t[: max(0, EV_MAX_CHARS - tot)]
                    if not t:
                        break
                    ev.append(t)
                    tot += len(t)
                jobs.append(dict(cluster=ci + 1, facet=facet, docid=d,
                                 anchor_gold=bool(d in anchor_gold),
                                 title=str(titles[docs.index(d)] or "")[:120],
                                 evidence="\n---\n".join(ev),
                                 claim=f"该论文{F[facet][1]}"))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"\n待判 (cluster, facet, doc) 对：**{len(jobs)}** ｜ 每对 2 次调用 + 分歧第三轮\n")

    done: dict[tuple, dict] = {}
    if OUT.exists() and not args.fresh:
        for r in pd.read_csv(OUT).to_dict("records"):
            done[(int(r["cluster"]), str(r["facet"]), str(r["docid"]))] = r
        print(f"已有缓存 {len(done)} 条")
    todo = [j for j in jobs if (j["cluster"], j["facet"], j["docid"]) not in done]

    lock = threading.Lock()
    cnt = {"n": 0}

    def work(j: dict) -> dict:
        user = (f"论文标题：{j['title']}\n\n论断：**{j['claim']}**\n\n"
                f"论文原文片段：\n{j['evidence']}")
        a = call("PAPERPILOT_LLM", SYS, user)
        b = call("PAPERPILOT_JUDGE", SYS, user)
        la = str(a.get("label") or "ERR").lower()
        lb = str(b.get("label") or "ERR").lower()
        final, third = "", {}
        if la == "entail" and lb == "entail":
            final = "yes"
        elif "entail" not in (la, lb) and "ERR" not in (la, lb):
            final = "no"
        else:
            third = call("PAPERPILOT_LLM", SYS_STRICT,
                         f"论文标题：{j['title']}\n\n论断：**{j['claim']}**\n\n"
                         f"论文原文片段：\n{j['evidence']}")
            final = "yes" if str(third.get("answer", "")).upper().startswith("Y") else "no"
        with lock:
            cnt["n"] += 1
            if cnt["n"] % 25 == 0:
                print(f"  判了 {cnt['n']}/{len(todo)} …", flush=True)
        return dict(cluster=j["cluster"], facet=j["facet"], docid=j["docid"],
                    anchor_gold=j["anchor_gold"], agent_A=model_l, A_label=la,
                    A_conf=a.get("confidence"), A_quote=str(a.get("quote") or "")[:160],
                    agent_B=model_j, B_label=lb, B_conf=b.get("confidence"),
                    B_quote=str(b.get("quote") or "")[:160],
                    agree=int(la == lb), need_third=int(bool(third)),
                    third=str(third.get("answer") or "")[:12],
                    new_gold=final, A_err=str(a.get("_err") or "")[:60],
                    B_err=str(b.get("_err") or "")[:60], evidence_chars=len(j["evidence"]))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(work, todo):
            done[(r["cluster"], r["facet"], r["docid"])] = r
    df = pd.DataFrame(list(done.values())).sort_values(["cluster", "facet", "docid"])
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n完成 {len(todo)} 对（{time.time() - t0:.0f}s）→ {OUT}")

    # ── 汇总：锚点真值 vs 新真值 ──
    print("\n" + "=" * 112)
    print("【锚点真值 vs 双 LLM 重标真值】（按 (cluster, facet) 聚合）")
    df["anchor_gold"] = df["anchor_gold"].astype(bool)
    df["new_gold"] = df["new_gold"].astype(str).str.strip().str.lower().isin(["yes", "y", "true"])
    rows = []
    for (c, f), g in df.groupby(["cluster", "facet"]):
        A = set(g[g["anchor_gold"]]["docid"])
        N = set(g[g["new_gold"]]["docid"])
        tp = len(A & N)
        rows.append(dict(cluster=c, facet=f, n_paper=len(g), n_anchor=len(A), n_new=len(N),
                         tp=tp,
                         anchor_P=tp / len(A) if A else float("nan"),
                         anchor_R=tp / len(N) if N else float("nan"),
                         added=len(N - A), removed=len(A - N)))
    s = pd.DataFrame(rows)
    print(f"  {'簇':>3}{'facet':<18}{'锚点正例':>9}{'新真值':>8}{'新增':>6}{'删去':>6}"
          f"{'锚点P':>8}{'锚点R':>8}")
    for _, r in s.iterrows():
        print(f"  {int(r['cluster']):>3}{r['facet']:<18}{int(r['n_anchor']):>9}{int(r['n_new']):>8}"
              f"{int(r['added']):>6}{int(r['removed']):>6}{r['anchor_P']:>8.3f}{r['anchor_R']:>8.3f}")
    print(f"\n  **合计**：锚点正例 {int(s['n_anchor'].sum())} ｜ 新真值 {int(s['n_new'].sum())}"
          f" ｜ 锚点 P {df[df['anchor_gold']]['new_gold'].mean():.3f}"
          f" ｜ 锚点 R {df[df['new_gold']]['anchor_gold'].mean():.3f}"
          f"（= 1 − 漏报率）")
    print(f"  （旧标定：P 0.895 / R 0.567，即漏 ~43%）")
    print(f"  两判官一致率 {df['agree'].mean():.3f} ｜ 走第三轮的比例 "
          f"{df['need_third'].mean():.3f} ｜ 调用错误 A {int((df['A_err'] != '').sum())} / "
          f"B {int((df['B_err'] != '').sum())}")
    s.to_csv(HERE / "results" / "R2_GOLD_RECALIB_SUMMARY.csv", index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
