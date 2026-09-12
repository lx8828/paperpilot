"""QASPER 基础通过测试：采样论文 → 处理成 report → 跑 QASPER 题 → GLM 裁判打分。

评测协议（2026-09-05 定）：
    1. 数据：QASPER train（888 篇，5,049 题中 train 2,593）。数据自带 gold answer
       （free_form / yes_no / extractive）与 evidence 段。
    2. 采样：--papers N 随机采 N 篇（seed 固定可复现）；每篇跑它全部题目
       （QASPER 篇均 ~2.9 题，故以"篇"为单位而非硬凑每篇 6~10 题）。
    3. 处理：每篇先 process_pdf(qasper_<id>.qpdf) 生成 report（有产物缓存则复用）。
    4. 问答：graph.ask 跑漏斗 → answer + cites + route。
    5. 裁判：GLM-5.3（与主链路异源）对比【QASPER gold + evidence】与【系统答案 +
       系统 cites 证据】，输出 score 1~5 + reason。
    6. 指标：通过率(≥4)、各题类分桶、unknown 率、citation 有效率、平均耗时/调用。
    7. unanswerable 题单独统计"诚实拒答率"（系统答 unknown/说明无答案 = 好）。

用法：
    uv run python cli/run_qasper_eval.py --papers 20            # 采 20 篇
    uv run python cli/run_qasper_eval.py --papers 20 --seed 7
    uv run python cli/run_qasper_eval.py --paper 1909.00694     # 定向单篇（调试）
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

from paperpilot.graph import ask as graph_ask
from paperpilot.pipeline import process_pdf
from paperpilot.qasper_source import load_papers, gold_answer
from paperpilot.tools import llm

ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / "qa"

_QASPER_PREFIX = "qasper_"
_QASPER_SUFFIX = ".qpdf"

# 裁判打分参考线
PASS_SCORE = 4

_JUDGE_SYS = (
    "你是严谨的论文问答裁判。你会看到：① 论文中的一道问题；② 人工标注的标准答案"
    "（ground truth，可能包含论文原文摘录）；③ 待评测系统（PaperPilot）的回答与它引用的"
    "论文原文片段。请独立判断：系统的回答是否**准确**地回答了问题（与标准答案一致、不编造、"
    "不遗漏关键信息）。只输出 JSON："
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


def _qpdf(pid: str) -> str:
    return f"{_QASPER_PREFIX}{pid}{_QASPER_SUFFIX}"


# ── 评测侧判据：无答案题的"片段合格式"（2026-09-11 新增）─────────────────────
# 背景：QASPER 的 gold 空题（论文本身没答案）里，若系统**不给结论、但明确说明"论文
# 未提供该信息"、并给出检索到的相关原文片段（带 cites）**——对用户是**有效交付**
# （能自行判断），产品口径上优于空拒答；但裁判按"回答完整性"打分会给 3（不到 pass 线）。
# → 单列 excerpt_ok 观察桶，不计 fail。
# 仅对 **gold 空的无答案题** 生效：有答案题出现同款"缺失口吻" = 误断言缺失，仍判 fail。
ABSENCE_RE = re.compile(
    r"(未给出|未提供|未提及|未列出|未报告|并未说明|并未给出|并未具体说明|"
    r"没有给出|没有提供|没有出现|未出现|无法给出|找不到|未找到|没有找到|无法找到|"
    r"无法判断|无法确认|信息不足|"
    r"未(?:明确|直接|具体|详细)?(?:说明|给出|提供|提及|指出|解释|报告|列出)|"
    r"没有(?:明确|直接|具体)?(?:说明|给出|提供|提及|解释|报告|列出)|"
    r"(?:论文|文中|文献|原文|文章)[^，。；]{0,16}(?:没有|未|不曾|并未)[^，。；]{0,12}"
    r"(?:给出|提供|报告|说明|列出))", re.I)


def _absence_tone(ans: str) -> bool:
    """答案是否明确说明"论文未提供该信息"。"""
    return bool(ABSENCE_RE.search(ans or ""))


def _ensure_report(pid: str, *, force: bool = False) -> str:
    """确保该 QASPER 论文已有 report 产物；返回虚拟 pdf 名。"""
    name = _qpdf(pid)
    report_file = ROOT / "assets/artifacts/out_views" / f"{Path(name).stem}.report.json"
    if report_file.exists() and not force:
        return name
    print(f"  [pipeline] 处理 {pid}…（LLM，可能 1~3 分钟）")
    process_pdf(name, force=force, workers=4, verbose=False)
    return name


def _judge(q: dict, answer: str, cites: list[dict]) -> tuple[int, str, bool]:
    """GLM 裁判打分。返回 (score, reason, is_pass)。LLMError 时返回 (0, err, False)。"""
    gold, ev = gold_answer(q)
    cite_text = "\n".join(
        f"[{i+1}] (p{c.get('page','?')}) {str(c.get('evidence',''))}"
        for i, c in enumerate(cites[:5])) or "（无引用）"
    user = _JUDGE_USER.format(
        question=q.get("question", ""),
        gold=gold or "（该题在论文中无答案，系统应诚实说明）",
        evidence=(ev or ""),
        answer=(answer or ""),
        cites=cite_text,
    )
    try:
        raw = llm.judge_json(_JUDGE_SYS, user, temperature=0.0)
        score = int(raw.get("score", 0)) if isinstance(raw, dict) else 0
        reason = str(raw.get("reason", "")) if isinstance(raw, dict) else ""
    except (llm.LLMError, ValueError) as e:
        return 0, f"裁判调用失败: {e}", False
    return score, reason, score >= PASS_SCORE


def run_one(q: dict, name: str) -> dict:
    t0 = time.time()
    gold, ev = gold_answer(q)
    is_unans = not gold
    try:
        r = graph_ask(q["question"], name)
    except Exception as e:  # noqa: BLE001
        return {"qid": q.get("question_id", ""), "paper": q.get("pdf", name),
                "question": q["question"], "gold": gold,
                "answer": "", "route": [], "level": "error",
                "score": 0, "reason": f"问答异常: {type(e).__name__}: {e}",
                "unanswerable": is_unans, "time_s": round(time.time() - t0, 2),
                "llm_calls": 0, "status": "error",
                # 与正常记录的字段保持一致（下游聚合用 .get，但缺键会让"零引用率"统计口径漂移）
                "validator_action": "", "issues": [],
                "no_citation": 0, "no_citation_substantive": 0}
    elapsed = time.time() - t0
    answer = r.get("answer") or ""
    cites = list(r.get("cites") or [])
    dbg = r.get("debug") or {}
    ans_dbg = dbg.get("answer") or {}
    score, reason, is_pass = _judge(q, answer, cites)
    route = list(r.get("route") or [])
    level = ans_dbg.get("level", "")

    # 状态判定（2026-09-06 修正：无答案题被诚实拒答且裁判认可 = pass）：
    #   pass          裁判认可（score≥4）。含两种情况：
    #                 · 有答案题答对了
    #                 · 无答案题系统诚实拒答，裁判给了≥4（这就是"做对了"）
    #   honest_refuse 无答案题系统拒答但裁判未认可（score<4）——行为诚实但没达到满分线，单独观察
    #   excerpt_ok    无答案题：未给结论，但明确说明"论文未提供"并给出相关原文片段（带 cites）
    #                 —— 产品口径为**合格**（有效交付，用户可自行判断），不计 fail（2026-09-11 新增）
    #   unknown_ok    有答案题系统拒答（没答出来，非 pass）
    #   fail          有答案题系统作答但裁判未认可
    status = "pass" if is_pass else ("unknown_ok" if level == "unknown" else "fail")
    if is_unans:
        if is_pass:
            status = "pass"
        elif level == "unknown":
            status = "honest_refuse"
        elif _absence_tone(answer) and cites:
            # 片段合格式：说明论文未提供 + 给出相关片段（有用于判断的材料）
            status = "excerpt_ok"
        else:
            status = "fail"
    # 输出闸门自检结果（2026-09-13 起落进记录，用于长期观测"无来源断言/零引用"频次）：
    #   validator_action: pass / repaired / fallback
    #   issues: 精简为 [sev/type]，避免记录被长文本撑大（完整 issues 在 out['validator']）
    #   no_citation: 零引用标记（1/0）+ 是否含实质断言（substantive）——A 档新增的观测项
    vres = r.get("validator") or {}
    v_issues = list(vres.get("issues") or [])
    no_cite = next((i for i in v_issues if i.get("type") == "no_citation"), None)
    rec = {
        "qid": q.get("question_id", ""),
        "paper": q.get("pdf", name),
        "question": q["question"],
        "gold": (gold or "")[:300],
        "unanswerable": is_unans,
        "answer": answer,
        "cites": cites,
        "cites_n": len(cites),
        "route": route,
        "level": level,
        "score": score,
        "reason": reason,
        "time_s": round(elapsed, 2),
        "status": status,
        "validator_action": vres.get("action", ""),
        "issues": [{"sev": i.get("sev"), "type": i.get("type")} for i in v_issues],
        "no_citation": (1 if no_cite else 0),
        "no_citation_substantive": (1 if (no_cite and no_cite.get("substantive")) else 0),
    }
    return rec


def load_sample(n: int, seed: int, only_pid: str | None) -> list[tuple[str, dict]]:
    papers = load_papers()
    if only_pid:
        return [(only_pid, papers[only_pid])] if only_pid in papers else []
    rng = random.Random(seed)
    ids = list(papers.keys())
    rng.shuffle(ids)
    return [(pid, papers[pid]) for pid in ids[:n]]


def load_from_run(run_file: str) -> list[tuple[str, dict]]:
    """复用旧 run json 中的论文列表（保持顺序去重），保证同样本可比。"""
    path = Path(run_file)
    if not path.exists():
        raise FileNotFoundError(f"缺 run 记录: {run_file}")
    prev = json.loads(path.read_text(encoding="utf-8"))
    papers = load_papers()
    out: list[tuple[str, dict]] = []
    for r in prev:
        pid = (r.get("paper") or "").replace(_QASPER_PREFIX, "").replace(_QASPER_SUFFIX, "")
        if pid in papers and all(p != pid for p, _ in out):
            out.append((pid, papers[pid]))
    return out


def build_summary(recs: list[dict]) -> str:
    total = len(recs)
    n_pass = sum(1 for r in recs if r.get("status") == "pass")
    n_fail = sum(1 for r in recs if r.get("status") == "fail")
    n_unk = sum(1 for r in recs if r.get("status") == "unknown_ok")
    n_honest = sum(1 for r in recs if r.get("status") == "honest_refuse")
    n_exc = sum(1 for r in recs if r.get("status") == "excerpt_ok")
    n_err = sum(1 for r in recs if r.get("status") == "error")
    scored = [r for r in recs if r.get("score", 0) > 0]
    avg_score = round(sum(r["score"] for r in scored) / len(scored), 2) if scored else 0
    avg_time = round(sum(r.get("time_s", 0) for r in recs) / total, 2) if total else 0

    # 分桶：有答案题（pass/fail）按 level 分布
    lines = [f"# QASPER 基础通过测试（{time.strftime('%Y-%m-%d %H:%M')}）\n",
             f"- 总题数: {total} ｜ 通过(pass≥4): {n_pass} ｜ 失败: {n_fail} ｜ "
             f"unknown_ok: {n_unk} ｜ 无答案拒答未认可: {n_honest} ｜ "
             f"无答案片段合格(excerpt_ok): {n_exc} ｜ error: {n_err}",
             f"- 有分题均分: {avg_score}/5 ｜ 平均耗时: {avg_time}s\n",
             "| paper | qid | 问题 | gold | 系统 | route | score | 状态 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        lines.append(
            f"| {r['paper'].replace(_QASPER_SUFFIX,'')} | {r['qid'][:12]} | "
            f"{r['question'][:38]} | {(r.get('gold') or '—')[:30]} | "
            f"{r['answer'][:38]} | {'→'.join(r['route']) if r.get('route') else '-'} | "
            f"{r.get('score','?')} | {r['status']} |")
    # 通过率主指标
    pass_rate = n_pass / total if total else 0
    lines.append(f"\n## 主通过率（仅计有答案题 pass / 全部题）：{pass_rate:.1%}")
    # 无答案题：pass 与 excerpt_ok 都是"合格行为"（后者是有效交付，只是不给结论）
    un = [r for r in recs if r.get("unanswerable")]
    if un:
        un_pass = sum(1 for r in un if r.get("status") == "pass")
        un_exc = sum(1 for r in un if r.get("status") == "excerpt_ok")
        lines.append(f"## 无答案题合格率（pass {un_pass} + excerpt_ok {un_exc}）："
                     f"{un_pass + un_exc}/{len(un)} = "
                     f"{(un_pass + un_exc) / len(un):.1%}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=20, help="采样论文篇数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--paper", default=None, help="定向单篇 pid（调试）")
    ap.add_argument("--from-run", default=None,
                    help="复用旧 run json 中的论文列表重跑（同样本可比）")
    ap.add_argument("--force", action="store_true", help="强制重跑 report（不读缓存）")
    ap.add_argument("--no-judge", action="store_true", help="跳过 GLM 裁判（只跑问答）")
    args = ap.parse_args()

    llm._load_dotenv(str(ROOT))
    if not llm.is_configured():
        print("[错误] LLM 未配置")
        return 1

    sample = (load_from_run(args.from_run) if args.from_run
              else load_sample(args.papers, args.seed, args.paper))
    if not sample:
        print(f"[错误] 没有可评测论文（--papers={args.papers}, "
              f"--paper={args.paper}, --from-run={args.from_run}）")
        return 1
    print(f"采样 {len(sample)} 篇，开始处理…", flush=True)

    all_recs: list[dict] = []
    skipped: list[str] = []
    for pid, paper in sample:
        try:
            name = _ensure_report(pid, force=args.force)
        except Exception as e:  # noqa: BLE001
            print(f"  [跳过] {pid} pipeline 失败: {type(e).__name__}: {e}", flush=True)
            skipped.append(pid)
            continue
        print(f"\n{'='*70}\n[{pid}] {paper['title'][:60]}｜{len(paper['qas'])} 题",
              flush=True)
        for q in paper["qas"]:
            try:
                rec = run_one(q, name)
                if args.no_judge:
                    rec["score"] = 0
                    rec["status"] = "skipped-judge"
                all_recs.append(rec)
            except Exception as e:  # noqa: BLE001
                rec = {"qid": q.get("question_id", ""), "paper": name,
                       "question": q.get("question", ""), "status": "error",
                       "reason": f"{type(e).__name__}: {e}"}
                all_recs.append(rec)
            print(f"  [{rec['status']}] {rec['qid'][:14]} score={rec.get('score','?')} "
                  f"→ {rec.get('level','?')} {rec.get('time_s')}s | "
                  f"{rec['question'][:50]}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_file = QA_DIR / f"qasper_run_{ts}.json"
    run_file.write_text(json.dumps(all_recs, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    sum_file = QA_DIR / f"qasper_summary_{ts}.md"
    sum_file.write_text(build_summary(all_recs), encoding="utf-8")
    print(f"\n记录: {run_file}\n汇总: {sum_file}")
    if skipped:
        print(f"跳过（pipeline 失败）: {len(skipped)} 篇")
        for pid in skipped:
            print(f"  - {pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
