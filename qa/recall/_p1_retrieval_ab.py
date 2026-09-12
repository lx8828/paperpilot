"""P1 检索层配对 A/B：**旧表文本 vs 新表文本**（同环境、确定性、0 LLM）。

- 正文向量从 cvec 缓存**切片取**（P1 不改正文文本，故缓存有效）；
- 外部块文本用两种 table_to_md 现场构造并现场编码；
- 指标：目标表进混池 top-12 / 表池内位次（RRF）。
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT / "qa" / "recall"))
os.environ["PAPERPILOT_QASPER_TABLES"] = "1"

from _table_policy_ab import text_vectors  # noqa: E402
from paperpilot.agents.document_cache import MINERU_OUT, ordered_chunks  # noqa: E402
from paperpilot.agents.embedder import BM25Index, encode_query, encode_texts  # noqa: E402
from paperpilot.qasper_source import gold_answer_full, load_papers  # noqa: E402
from paperpilot.tools.mineru_bridge import (_TableParser, _cell_clean,  # noqa: E402
                                            _find_content_list, _strip_html, table_to_md)


def legacy_md(html: str) -> str:
    """旧实现：首行当表头、忽略 colspan/rowspan。"""
    p = _TableParser()
    p.feed(html or "")
    p.close()
    lines = []
    for i, row in enumerate(p.rows):
        cells = [_cell_clean(t) for (t, _cs, _rs, _th) in row]
        if not any(cells):
            continue
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("|" + "---|" * len(row))
    return "\n".join(lines)


def ext_text(el: dict, md_fn) -> str:
    """复现 element_text 对 table 的拼法：cap + md(body) + footnote。"""
    cap = _strip_html(" ".join(el.get("table_caption") or []))
    body = md_fn(el.get("table_body") or "")
    fn = _strip_html(" ".join(el.get("table_footnote") or []))
    return "\n".join(x for x in (cap, body, fn) if x)


def cap_key(t: str) -> str:
    m = re.match(r"\s*(table|figure)\s*([A-Za-z]?\d+)", str(t), re.I)
    return f"{m.group(1).lower()}{m.group(2).lower()}" if m else ""


def rrf_order(vs, bs, n):
    rv = np.empty(n)
    rv[np.argsort(-vs, kind="stable")] = np.arange(n)
    rb = np.empty(n)
    rb[np.argsort(-bs, kind="stable")] = np.arange(n)
    return np.argsort(-(1.0 / (60 + rv + 1) + 1.0 / (60 + rb + 1)), kind="stable")


def main() -> int:
    papers = load_papers()
    exp = json.loads(Path("qa/recall/qasper_tbl_exp_20260911.json").read_text(encoding="utf-8"))
    out = {"old": [], "new": []}          # 混池目标表位次
    pool = {"old": [], "new": []}         # 表池内位次
    lens = {"old": [], "new": []}
    for x in exp:
        pid, qid = x["pid"], x["qid"]
        q = next((qq for qq in papers[pid]["qas"]
                  if str(qq.get("question_id") or "") == qid), None)
        if q is None:
            continue
        _g, evs = gold_answer_full(q)
        want = set()
        for e in evs:
            m = re.match(r"\s*(?:FLOAT SELECTED:)?\s*(table|figure)\s*([A-Za-z]?\d+)",
                         str(e), re.I)
            if m:
                want.add(f"{m.group(1).lower()}{m.group(2).lower()}")
        if not want:
            continue
        texts = ordered_chunks(f"qasper_{pid}.qpdf")
        tv = text_vectors(pid, len(texts))
        if tv is None:
            continue
        cl = _find_content_list(MINERU_OUT / f"{pid}v1") or []
        qv = encode_query(q["question"])
        for name, md_fn in (("old", legacy_md), ("new", table_to_md)):
            exts = [ext_text(el, md_fn) for el in cl if el.get("type") == "table"]
            exts = [t for t in exts if t]
            if not exts:
                continue
            lens[name] += [len(t) for t in exts]
            vecs = np.vstack([tv, encode_texts(exts)])
            alltext = [c.text for c in texts] + exts
            vs = (qv @ vecs.T).ravel()
            bs = BM25Index(alltext).score(q["question"])
            n_t = len(texts)
            is_ext = np.zeros(len(alltext), dtype=bool)
            is_ext[n_t:] = True
            tgt = {n_t + i for i, t in enumerate(exts) if cap_key(t) in want}
            if not tgt:
                continue
            order = rrf_order(vs, bs, len(alltext))
            out[name].append(hash(qid) if False else next(
                (k for k, c in enumerate(order) if c in tgt), None))
            # 表池内位次
            sub_v, sub_b = vs[n_t:], bs[n_t:]
            sub = rrf_order(sub_v, sub_b, len(exts))
            pool[name].append(next((k for k, c in enumerate(sub)
                                    if (n_t + c) in tgt), None))
    for name in ("old", "new"):
        v = [x if x is not None else 10 ** 6 for x in out[name]]
        p = [x if x is not None else 10 ** 6 for x in pool[name]]
        L = np.array(lens[name])
        print(f"== {name} ==  n={len(v)} | 混池进 top-12: {sum(1 for x in v if x < 12)} | "
              f"混池位次中位 {np.median(v):.0f} | 表池 top-1 {sum(1 for x in p if x < 1)} "
              f"| 表池 top-3 {sum(1 for x in p if x < 3)} | 表池位次中位 {np.median(p):.0f} "
              f"| 表块长度中位 {np.median(L):.0f}")
    n = min(len(out["old"]), len(out["new"]))
    d_mix = sum(1 for i in range(n) if (out["new"][i] if out["new"][i] is not None else 999) <
                (out["old"][i] if out["old"][i] is not None else 999))
    d_pool = sum(1 for i in range(n) if (pool["new"][i] if pool["new"][i] is not None else 999) <
                 (pool["old"][i] if pool["old"][i] is not None else 999))
    print(f"\n配对（n={n}）：混池位次改善 {d_mix} 题 | 表池位次改善 {d_pool} 题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
