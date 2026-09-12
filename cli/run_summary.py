"""全量汇总 + ev✗ 审计：读 assets/artifacts/out_claims/*.claims.json，重做本地 parse+chunk 核对。

不调用 LLM，可反复运行。
用法：
    uv run python run_summary.py                     # 汇总总表
    uv run python run_summary.py --audit-miss        # 额外对未命中的 ev✗ 做原因分类
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.tools import analyzer
from paperpilot.tools.chunker import chunk_document
from paperpilot.tools.evidence import (
    alnum_norm,
    dict_to_claim,
    norm_text,
    verify_evidence,
)
from paperpilot.tools.pdf_parser import parse_pdf

ROOT = Path(__file__).resolve().parents[1]  # cli/ → 项目根
PAPERS_DIR = ROOT / "assets" / "papers"

# 复用同目录 run_claims.py 的表打印函数（不触发其 main）
sys.path.insert(0, str(Path(__file__).resolve().parent))  # cli/
from run_claims import print_table  # noqa: E402  # 隐式相对导入：CLI 脚本互引，见 pyproject basedpyright 豁免


def _bigrams(tokens: list[str]) -> set[str]:
    return {" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)}


def classify_miss(claim: Any, text_map: dict[str, str],
                  gram_cache: dict[str, set[str]]) -> str:
    """对一条 ev✗ 归因：验证器过严 / 漂移 / 释义 / 疑似无匹配。"""
    nq = norm_text(claim.evidence_quote)
    src = norm_text(text_map.get(claim.chunk_id, ""))
    if not nq or not src:
        return "4-无匹配(空证据或缺原文)"
    # ① 强归一（剥掉所有非字母数字）后能命中 → 只是断行连字符/标点等字符噪声
    if alnum_norm(nq) and alnum_norm(nq) in alnum_norm(src):
        return "1-字符级噪声(验证器过严)"
    # ② 证据在其它 chunk 逐字存在 → LLM 把来源 chunk 标错了
    for cid, t in text_map.items():
        if cid != claim.chunk_id and nq and nq in norm_text(t):
            return "2-跨chunk漂移(来源标错)"
    # ③ 大词组（bigram）覆盖率 ≥40% → 大部分词汇在原文，属于释义/改写
    qb = _bigrams(nq.split())
    if qb:
        if claim.chunk_id not in gram_cache:
            gram_cache[claim.chunk_id] = _bigrams(src.split())
        cov = len(qb & gram_cache[claim.chunk_id]) / len(qb)
        if cov >= 0.4:
            return f"3-释义改写(bigram覆盖{cov:.0%})"
    return "4-无匹配(疑似拼接/幻觉)"


def build_row(data: dict[str, Any], target_chunks: list[Any]) -> dict[str, Any]:
    claims = [dict_to_claim(d) for d in data["claims"]]
    text_map = {c.chunk_id: c.text for c in target_chunks}
    hit, loose, miss = verify_evidence(claims, text_map)
    return {
        "pdf": data["pdf"],
        "n_chunks": len(target_chunks),
        "n_claims": len(claims),
        "by_type": data.get("by_type", {}),
        "n_errors": len(data.get("errors", [])),
        "hit": len(hit),
        "loose": len(loose),
        "miss": len(miss),
    }


def audit_miss(pairs: list[tuple[str, dict[str, Any], list[Any]]]) -> None:
    cat_total = Counter()
    samples: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    per_pdf: dict[str, Counter[str]] = {}
    for pdf, data, target in pairs:
        claims = [dict_to_claim(d) for d in data["claims"]]
        text_map = {c.chunk_id: c.text for c in target}
        _, _, miss = verify_evidence(claims, text_map)
        if not miss:
            continue
        gram_cache: dict[str, set[str]] = {}
        pc = Counter()
        for c in miss:
            cat = classify_miss(c, text_map, gram_cache)
            cat_total[cat] += 1
            pc[cat] += 1
            if len(samples[cat]) < 4:
                samples[cat].append((pdf, c))
        per_pdf[pdf] = pc

    print("\n" + "=" * 100)
    print(f"ev✗ 审计：全量 {sum(cat_total.values())} 条未命中，按原因分类")
    print("=" * 100)
    for cat in sorted(cat_total):
        print(f"  {cat:<46s} {cat_total[cat]:>5d}")
    # 每类样例
    for cat in sorted(samples):
        print(f"\n◆ {cat} —— 样例：")
        for pdf, c in samples[cat]:
            ev = c.evidence_quote.replace("\n", " ")
            print(f"  [{pdf} | {c.type}] {c.claim_id} <- {c.chunk_id}")
            print(f"    证据: {ev[:220]}")
    # 各篇 miss 高亮
    print("\n各篇 ev✗ 构成（数字为空表示 0）：")
    for pdf, pc in sorted(per_pdf.items()):
        parts = " ".join(f"{k[:2]}={v}" for k, v in sorted(pc.items()))
        print(f"  {pdf:<22s} {parts}")
    print("=" * 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "assets/artifacts/out_claims"),
                    help="claims 输出目录（默认 assets/artifacts/out_claims）")
    ap.add_argument("--audit-miss", action="store_true",
                    help="对未命中(ev✗)做原因分类审计")
    args = ap.parse_args()

    out_dir = Path(args.out)
    files = sorted(p for p in out_dir.glob("*.claims.json"))
    if not files:
        print(f"[错误] {out_dir} 下没有 .claims.json 文件")
        return 1

    pairs: list[tuple[str, dict[str, Any], list[Any]]] = []
    rows: list[dict[str, Any]] = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        pdf_path = PAPERS_DIR / data["pdf"]
        if not pdf_path.exists():
            print(f"  [跳过] 找不到来源 PDF: {data['pdf']}")
            continue
        chunks = chunk_document(parse_pdf(str(pdf_path))["blocks"], max_len=4000)
        target = analyzer.extractable(chunks)
        pairs.append((data["pdf"], data, target))
        rows.append(build_row(data, target))

    if rows:
        print(f"\n汇总：{len(rows)} 篇 | 目录 {out_dir}")
        print_table(rows)

    if args.audit_miss:
        audit_miss(pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
