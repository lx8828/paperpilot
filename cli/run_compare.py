"""对比测试 harness：B0 直接 LLM / B1 朴素 RAG / B2 PaperPilot 三列同裁判评测。

设计：qa/COMPARE_DESIGN.md（公平性约束在文档 §1，务必先读）。

用法：
    1. 锁定样本（从旧 run 文件取论文，确保 pipeline 产物已存在）：
        uv run python cli/run_compare.py sample --from-run qa/qasper_overnight_full_20260907_142814.json \
            --n 20 --out qa/compare/papers_compare.json
    2. 分别跑三列（可分开/后台跑，结果互不依赖）：
        uv run python cli/run_compare.py run --column B1 --papers qa/compare/papers_compare.json \
            --out qa/compare/run_<ts>_B1.json
        uv run python cli/run_compare.py run --column B0 --papers qa/compare/papers_compare.json \
            --out qa/compare/run_<ts>_B0.json
        uv run python cli/run_compare.py run --column B2 --papers qa/compare/papers_compare.json \
            --out qa/compare/run_<ts>_B2.json

约定（与 run_qasper_eval.py 同源）：
    - 裁判：llm.judge_json（.env PAPERPILOT_JUDGE_*），score 1~5，pass>=4；三列同 prompt 盲判。
    - 用量：llm.reset_usage()/usage_stats() 读每题的生成与裁判 token/调用数。
    - B0 超长论文（全文估算 token 超 --b0-max-tokens）不硬塞：记为 context_overflow（这正是
      "直接 LLM 放不下、我们能放下"的边界证据），不消耗调用。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.qasper_source import _clean_para, gold_answer, load_papers
from paperpilot.tools import llm

ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / "qa"
OUT_VIEWS = ROOT / "out_views"
QASPER_PREFIX = "qasper_"
QASPER_SUFFIX = ".qpdf"
PASS_SCORE = 4

# B1 朴素 RAG 检索预算 = 与 L3 同款 top-k（只差：纯向量、无 hybrid/无 claims/无漏斗）
B1_TOP_K = 12
# B0 直接 LLM 单次输入的全文上限（token 估算 = 字符数/4；deepseek-chat 上下文 64K，留余量）
B0_MAX_TOKENS = 56000

_SYS_NEUTRAL = (
    "你是严谨的论文问答助手。只依据下方提供的论文材料作答，不要用外部知识补充材料"
    "中没有的细节。若材料不足以回答，请明确说明缺什么。直接给出答案，不要客套。"
)

_JUDGE_SYS = (
    "你是严谨的论文问答裁判。你会看到：① 论文中的一道问题；② 人工标注的标准答案"
    "（ground truth，可能包含论文原文摘录）；③ 待评测系统的回答。请独立判断：系统的回答"
    "是否**准确**地回答了问题（与标准答案一致、不编造、不遗漏关键信息）。只输出 JSON："
    '{"score": 1-5, "reason": "一句话理由"}。评分基准：5=完全正确且细节齐全；'
    "4=正确，仅缺次要细节或措辞差异；3=部分正确，有重要遗漏或轻微错误；"
    "2=明显错误或遗漏核心；1=答非所问或编造。"
)
_JUDGE_USER = """【问题】
{question}

【标准答案（人工 gold + 原文证据）】
gold: {gold}
evidence: {evidence}

【系统回答】
{answer}

【系统引用的论文原文】
{cites}

请打分（1-5）。注意：系统回答若声称有引用，引用原文必须真实支撑其说法，否则视为编造降分。"""


# ───────────────────────── 工具 ─────────────────────────


def _qpdf(pid: str) -> str:
    return f"{QASPER_PREFIX}{pid}{QASPER_SUFFIX}"


def qtype_of(q: dict) -> str:
    """QASPER 无 question_type 字段，从 answers 判定题型（与 gold_answer 同优先级）。"""
    for a in q.get("answers") or []:
        inner = a.get("answer") or {}
        if inner.get("unanswerable"):
            continue
        if inner.get("free_form_answer"):
            return "free_form"
        if inner.get("yes_no") is not None:
            return "yes_no"
        if inner.get("extractive_spans"):
            return "extractive"
    return "unanswerable"


def full_text_of(paper: dict) -> str:
    """B0 全文（直接 LLM）：标题 + 摘要 + 全部 section（清洗 S2ORC 引用占位）。"""
    parts: list[str] = []
    title = str(paper.get("title") or "").strip()
    abstract = str(paper.get("abstract") or "").strip()
    if title:
        parts.append(f"TITLE: {title}")
    if abstract:
        parts.append(f"ABSTRACT: {abstract}")
    for sec in paper.get("full_text") or []:
        sec_name = str(sec.get("section_name") or "").strip()
        paras = [_clean_para(p) for p in (sec.get("paragraphs") or []) if _clean_para(p)]
        if not paras:
            continue
        parts.append(f"=== {sec_name or '(PREAMBLE)'} ===")
        parts.append("\n".join(paras))
    return "\n\n".join(parts)


def est_tokens(text: str) -> int:
    """粗略 token 估算（英文 ≈ 4 字符/token）。"""
    return max(1, round(len(text) / 4))


def judge(q: dict, answer: str, cites_text: str) -> tuple[int, str]:
    """异源裁判打分（.env PAPERPILOT_JUDGE_*，现 glm-4-flash）。返回 (score, reason)。"""
    gold, ev = gold_answer(q)
    user = _JUDGE_USER.format(
        question=q.get("question", ""),
        gold=gold or "（该题在论文中无答案，系统应诚实说明）",
        evidence=(ev or ""),
        answer=(answer or ""),
        cites=cites_text or "（无引用）",
    )
    try:
        raw = llm.judge_json(_JUDGE_SYS, user, temperature=0.0)
        score = int(float(raw.get("score", 0))) if isinstance(raw, dict) else 0
        reason = str(raw.get("reason", ""))[:200] if isinstance(raw, dict) else ""
    except (llm.LLMError, ValueError, TypeError) as e:
        return 0, f"裁判调用失败: {e}"
    return score, reason


def _usage_delta(after: dict, before: dict) -> dict:
    return {k: after.get(k, 0) - before.get(k, 0) for k in ("calls", "prompt_tokens", "completion_tokens")}


# ───────────────────────── B0 / B1 作答 ─────────────────────────


def answer_b0(q: dict, paper: dict, max_tokens: int) -> dict:
    """直接 LLM：全文单次喂入。超限如实记 context_overflow，不硬塞。"""
    question = q.get("question", "")
    full = full_text_of(paper)
    est = est_tokens(full + "\n" + question)
    base = {"question": question, "est_tokens": est, "full_chars": len(full)}
    if est > max_tokens:
        return {**base, "status": "overflow", "answer": "",
                "reason": f"全文估算 {est//1000}K token > {max_tokens//1000}K 上限（context_overflow）"}
    user = f"【论文全文】\n{full}\n\n【问题】\n{question}"
    answer = llm.chat_text(_SYS_NEUTRAL, user, temperature=0.0)
    return {**base, "status": "ok", "answer": answer, "reason": ""}


def _fmt_chunk(e: dict, i: int) -> str:
    sec = ""
    for p in reversed(list(e.get("title_path") or [])):
        if " · " in p:
            sec = p.split(" · ", 1)[1]
            break
    if not sec and (e.get("title_path") or []):
        sec = str((e.get("title_path") or [])[-1])
    return f"[{i}] (sec: {sec or '?'})\n{e.get('text') or ''}"


def answer_b1(q: dict, pdf: str) -> dict:
    """朴素 RAG：ChunkIndex 纯向量 top-k → 单次作答。无 claims/无漏斗/无引用/无拒答约束外。"""
    from paperpilot.agents.embedder import ChunkIndex

    question = q.get("question", "")
    hits = ChunkIndex(pdf).search(question, top_k=B1_TOP_K)
    if not hits:
        ctx = "（未检索到相关段落）"
    else:
        ctx = "\n\n".join(_fmt_chunk(h, i) for i, h in enumerate(hits, 1))
    user = f"【检索到的论文片段】\n{ctx}\n\n【问题】\n{question}"
    answer = llm.chat_text(_SYS_NEUTRAL, user, temperature=0.0)
    return {"status": "ok", "answer": answer, "n_retrieved": len(hits), "reason": ""}


# ───────────────────────── 主流程 ─────────────────────────


def load_sample_papers(run_file: str, n: int, seed: int) -> list[dict]:
    prev = json.loads(Path(run_file).read_text(encoding="utf-8"))
    papers = load_papers()
    seen: list[str] = []
    for r in prev:
        pid = (r.get("paper") or "").replace(QASPER_PREFIX, "").replace(QASPER_SUFFIX, "")
        if pid in papers and pid not in seen:
            seen.append(pid)
    rng_seed = seed
    # 保持 run 顺序（确定性），按给定 n 取前 n 篇且产物必须存在
    out: list[dict] = []
    for pid in seen:
        report = OUT_VIEWS / f"qasper_{pid}.report.json"
        if not report.exists():
            continue
        p = papers[pid]
        qas = p.get("qas") or []
        n_unans = sum(1 for qq in qas if gold_answer(qq)[0] is None)
        out.append({
            "pid": pid, "title": str(p.get("title") or "")[:100],
            "n_qas": len(qas), "n_unans": n_unans,
            "full_chars": len(full_text_of(p)),
        })
        if len(out) >= n:
            break
    _ = rng_seed
    return out


def run_column(column: str, papers_meta: list[dict], out_file: Path,
               max_tokens: int, q_limit: int | None) -> int:
    papers = load_papers()
    recs: list[dict] = []
    err = 0
    # 进程级预热（不计入任何题耗时）：LLM 连接/模型首载一次到位，避免第一题被冷启动污染
    try:
        llm.chat_text("你是连通性测试助手。", "回复 OK 即可。", temperature=0.0, max_tokens=8)
    except Exception:  # noqa: BLE001（预热失败不阻断，只是该进程第一题可能慢一点）
        pass
    llm.reset_usage()
    for pi, pm in enumerate(papers_meta, 1):
        pid = pm["pid"]
        pdf = _qpdf(pid)
        paper = papers[pid]
        print(f"\n{'='*70}\n[{pi}/{len(papers_meta)}] {column} {pid} "
              f"{paper.get('title','')[:50]} ｜ {len(paper.get('qas') or [])} 题", flush=True)
        # 预热（不计入题耗时）：B1/B2 先把 chunk 向量索引与 embedder 模型加载好
        # （search 一次强制 encode_query 权重加载，避免计时被权重加载污染）
        idx_t = 0.0
        if column in ("B1", "B2"):
            from paperpilot.agents.embedder import ChunkIndex
            t0 = time.time()
            try:
                ChunkIndex(pdf).search("warmup", top_k=1)
            except Exception:  # noqa: BLE001（论文无向量时无碍）
                pass
            idx_t = time.time() - t0
        qas = (paper.get("qas") or [])[:q_limit] if q_limit else (paper.get("qas") or [])
        for qi, q in enumerate(qas, 1):
            llm.reset_usage()
            t0 = time.time()
            qtype = qtype_of(q)
            gold, ev = gold_answer(q)
            is_unans = not gold
            try:
                if column == "B2":
                    from paperpilot.graph import ask as graph_ask
                    r = graph_ask(q.get("question", ""), pdf)
                    answer = r.get("answer") or ""
                    cites = list(r.get("cites") or [])
                    dbg = r.get("debug") or {}
                    ans_dbg = dbg.get("answer") or {}
                    level = ans_dbg.get("level", "")
                    route = list(r.get("route") or [])
                    st = "ok"
                else:
                    if column == "B0":
                        res = answer_b0(q, paper, max_tokens)
                    else:
                        res = answer_b1(q, pdf)
                    answer = res.get("answer") or ""
                    cites: list = []
                    level = ""
                    route = []
                    st = res.get("status", "ok")
                    reason0 = res.get("reason", "")
            except Exception as e:  # noqa: BLE001
                u_ans = llm.usage_stats()
                recs.append({
                    "column": column, "pid": pid, "qid": q.get("question_id", ""),
                    "qtype": qtype, "unanswerable": is_unans,
                    "question": q.get("question", "")[:300],
                    "answer": "", "level": "error", "route": [],
                    "score": 0, "reason": f"问答异常: {type(e).__name__}: {e}"[:200],
                    "status": "error",
                    "time_s": round(time.time() - t0, 2),
                    "gen": _usage_delta(u_ans, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}),
                    "judge": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
                    "cites_n": 0, "index_warm_s": round(idx_t, 2),
                })
                err += 1
                print(f"  [err] {q.get('question_id','')[:12]}: {e}", flush=True)
                continue
            u_ans = llm.usage_stats()

            t_ans = time.time() - t0
            if st == "overflow":
                # 不调用裁判（没作答，无需打分）；overflow 本身就是结论
                score, jreason, j_usage = 0, reason0, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            else:
                cite_text = "\n".join(
                    f"[{i+1}] (p{c.get('page','?')}) {str(c.get('evidence',''))}"
                    for i, c in enumerate(cites[:5])) if column == "B2" else ""
                score, jreason = judge(q, answer, cite_text)
                j_usage = _usage_delta(llm.usage_stats(), u_ans)
            status = ("pass" if score >= PASS_SCORE else
                      ("overflow" if st == "overflow" else "fail"))
            rec = {
                "column": column, "pid": pid, "qid": q.get("question_id", ""),
                "qtype": qtype, "unanswerable": is_unans,
                "question": q.get("question", "")[:300],
                "answer": answer[:600], "level": level, "route": route,
                "score": score, "reason": jreason[:200],
                "status": status, "time_s": round(time.time() - t0, 2),
                "time_s_ans": round(t_ans, 2),
                "gen": u_ans,
                "judge": j_usage,
                "cites_n": len(cites), "index_warm_s": round(idx_t, 2),
            }
            if column == "B0":
                rec["est_tokens"] = est_tokens(full_text_of(paper) + q.get("question", ""))
                rec["full_chars"] = len(full_text_of(paper))
            recs.append(rec)
            print(f"  [{status}] {q.get('question_id','')[:12]} {qtype[:9]:9} "
                  f"score={score} t={rec['time_s']:.1f}s calls={u_ans['calls']} "
                  f"pt={u_ans['prompt_tokens']} ct={u_ans['completion_tokens']}",
                  flush=True)
        # 每篇结束落盘一次（断点可恢复进度认知）
        out_file.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    out_file.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[完成] {column} 落盘 {out_file}（{len(recs)} 条，error={err}）", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    sp = sub.add_parser("sample", help="从旧 run 文件锁 N 篇论文样本")
    sp.add_argument("--from-run", required=True)
    sp.add_argument("--n", type=int, default=20)
    sp.add_argument("--out", default=str(QA_DIR / "compare" / "papers_compare.json"))

    sp2 = sub.add_parser("run", help="跑某一列")
    sp2.add_argument("--column", required=True, choices=["B0", "B1", "B2"])
    sp2.add_argument("--papers", required=True)
    sp2.add_argument("--out", required=True)
    sp2.add_argument("--b0-max-tokens", type=int, default=B0_MAX_TOKENS)
    sp2.add_argument("--q-limit", type=int, default=None, help="每篇最多跑 N 题（调试用）")
    args = ap.parse_args()

    llm._load_dotenv(str(ROOT))
    if not llm.is_configured():
        print("[错误] LLM 未配置")
        return 1

    if args.mode == "sample":
        papers = load_sample_papers(args.from_run, args.n, seed=42)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(papers, ensure_ascii=False, indent=1), encoding="utf-8")
        tot_q = sum(p["n_qas"] for p in papers)
        tot_u = sum(p["n_unans"] for p in papers)
        print(f"锁定 {len(papers)} 篇 / {tot_q} 题（无答案 {tot_u}）→ {out}")
        for p in papers:
            print(f"  {p['pid']}  {p['n_qas']} 题 ({p['n_unans']} unans)  {p['full_chars']} 字符")
        return 0

    papers_meta = json.loads(Path(args.papers).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "column": args.column, "generator": "deepseek-chat(v4-flash)",
        "judge": "glm-4-flash", "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_papers": len(papers_meta),
        "n_questions": sum(p["n_qas"] for p in papers_meta),
        "papers": [p["pid"] for p in papers_meta],
    }
    (out.with_suffix(".meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return run_column(args.column, papers_meta, out,
                      args.b0_max_tokens, args.q_limit)


if __name__ == "__main__":
    raise SystemExit(main())
