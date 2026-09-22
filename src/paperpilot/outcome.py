"""成果层：每篇论文的「**跨篇可比、可算**」的规范化产物。

## 为什么要有这一层

`report.json` 是**阅读层**（给人读 + **单篇**溯源）。它**能解析，但不可比、不可算**：

| | report（阅读层）| 成果层（本模块）|
|---|---|---|
| 值 | `rep_text` 自由文本（"本文在 XSum 上评测"）| **规范字段** |
| 词表 | 无 | **受控**（数据集/指标/方法族归一）|
| 关键数值 | **藏在句子里** | **独立字段 `metrics[].value`** ← 能算"谁在这指标上最好" |
| 粒度 | 215 组，不齐 | 每篇固定几项 |
| 溯源 | ✅ claim→chunk→page | ✅ 同样保留（**不许丢**）|

## 三条硬约束（沿用项目既有纪律）

1. **严格**：每个 `quote` 必须 verbatim 在来源块里找到，`metrics[].value` 还必须在
   **它自己的 quote** 里 —— 否则 `verified=False`（**保留但标记**，不丢弃）。
   实测代价约 **3%**（42 个数值里 41 个通过，2026-09-21）。
2. **后端绑锚，不信 LLM 给的块号**（同 `skeleton.py` 的做法）：LLM 只给「值 + 原文片段」，
   块锚点由后端在检索视图里搜出来绑定 → `anchor_chunk` / `anchor_page`。
3. **源 = claims + 检索视图**（`retrieval_chunks`，含 MinerU 表格/公式注入）。
   只读 claims（`ordered_chunks`）会丢 **40.1%** 的数值，**包括数据集表**（实测同上）。

## 缓存（这是"按需生成"能成立的关键）

成果层是**论文自身的属性**，不随查询变 → 落盘 `out_outcome/<stem>.outcome.json`，
指纹（模型 + PDF + VERSION）匹配即复用。所以长期成本 ≈ **被真正看过的论文数**。

## 表里的数值没有 claim 背书

表值只有 `chunk_id`（没有 `group_id`）→ 溯源链少一环：
    `value → chunk_id → page → 原文片段`
正文里的数值有 claim 背书时，`Finding.group_id` 会一并带上。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from pydantic import BaseModel, Field

from paperpilot.prompts.outcome import OUTCOME_SYSTEM, build_user
from paperpilot.tools import llm

ROOT = Path(__file__).resolve().parents[2]      # src/paperpilot/outcome.py → 项目根
OUTCOME_DIR = ROOT / "assets/artifacts/out_outcome"

# schema / prompt 变更时 +1 → 已缓存的成果层自动失效重建
VERSION = 6
MIN_QUOTE = 20          # quote 短于此不值当去搜（易误命中）
MAX_METRICS = 8
# 每类候选主张组的条数上限（控制 prompt 体量；核心类不设限）
GROUP_CAP = {"method_core": 12, "result_primary": 12, "ablation": 0, "detail": 0}
FALLBACK_CAP = 12       # type=limitation 等其他组

# 公认指标词表 —— **"能不能跨篇比"的判据由后端判定，不交给 LLM**。
# 命中 → 规范写名 + `comparable=True`；未命中（论文自创，如 `LaSE`）→ 原名 + False。
# 加新指标就往这里加一条（这是**唯一**需要维护的地方）。
KNOWN_METRICS = {
    "bleu": "BLEU", "sacrebleu": "sacreBLEU", "chrf": "chrF", "comet": "COMET",
    "rouge": "ROUGE", "rouge-1": "ROUGE-1", "rouge-2": "ROUGE-2", "rouge-l": "ROUGE-L",
    "mrouge": "mROUGE", "rouge-lsum": "ROUGE-Lsum",
    "meteor": "METEOR", "bertscore": "BERTScore", "cider": "CIDEr",
    "accuracy": "Accuracy", "acc": "Accuracy", "f1": "F1",
    "precision": "Precision", "recall": "Recall",
    "exact match": "EM", "em": "EM", "mrr": "MRR", "ndcg": "NDCG", "map": "MAP",
    "wer": "WER", "cer": "CER", "perplexity": "PPL", "ppl": "PPL",
}

# LLM 把模型名写进 dataset 时的兜底拆分：`CNN/DailyMail (mT5)` → dataset + model
_MODEL_IN_DATASET_RE = re.compile(r"^(.*?)\s*[（(]\s*([^()（）]{1,24})\s*[)）]\s*$")

# 笼统词（是"数据划分"不是"数据集"）—— 后端**直接清空**。
# `dataset` 是跨篇对齐的主键，宁可空着，也不让 "Test set" 冒充数据集进来。
_GENERIC_DATASETS = {
    "test set", "test", "testset", "dev", "dev set", "valid", "validation",
    "validation set", "train", "training set", "evaluation set", "overall",
    "测试集", "验证集", "训练集", "全集",
}


def _canon_metric(name: str) -> tuple[str, bool]:
    """指标名 → `(规范名, 是否公认可比)`。**后端判定**，不让 LLM 猜。"""
    raw = str(name or "").strip()
    key = re.sub(r"[\s_]+", " ", raw.lower()).strip()
    for cand in (key, key.removesuffix(" score").strip()):
        if cand in KNOWN_METRICS:
            return KNOWN_METRICS[cand], True
    return raw, False


# ── 模型 ─────────────────────────────────────────────────────────────────────

class Sourced(BaseModel):
    """带溯源锚的条目基类。`verified=False` = **有这条，但回溯不到原文**（保留不丢）。

    `verify` 是**校验档位**（比 `verified` 更细）：

    | 值 | 含义 |
    |---|---|
    | `exact`  | `quote` **逐字**在原文里（最强）|
    | `values` | `quote` 不是逐字，但**它报的数值同块共现**（表格列式散开时的兜底）|
    | `""`     | 未通过 |
    """
    quote: str = ""
    anchor_chunk: str = ""      # 后端搜出来绑定的块
    anchor_page: int = 0
    verified: bool = False
    verify: str = ""


class DatasetRef(Sourced):
    name: str = ""


class MetricResult(Sourced):
    name: str = ""              # 规范化后的指标名（后端按 `KNOWN_METRICS` 归一）
    value: str = ""             # **本文方法**的数值（严格：必须出现在本条的 quote 里）
    dataset: str = ""           # 数据集 / 语言对 ← **跨篇对齐的主键**（笼统词会被后端清空）
    model: str = ""             # 评测用的模型/配置（mT5 / GPT-4o / Flan-T5 …）
    baseline: str = ""          # 对照数值（须 verbatim 在 quote 里，否则后端**清空**）
    delta: str = ""             # 后端算的 `value − baseline`（不靠 LLM，防幻觉）
    comparable: bool = False     # 指标是否**公认** → 自创指标（LaSE）为 False，**不可跨篇比**
    group_id: str = ""          # 有 claim 背书时才有（正文里的数值）


class Finding(BaseModel):
    text: str = ""
    group_id: str = ""          # → report.groups[].group_id（可再回 claim/chunk/page）
    verified: bool = False


class PaperOutcome(BaseModel):
    """一篇论文的成果卡（跨篇可比、可算、每条可回原文）。"""
    pdf: str = ""
    title: str = ""
    problem: str = ""
    method_family: list[str] = Field(default_factory=list)
    datasets: list[DatasetRef] = Field(default_factory=list)
    metrics: list[MetricResult] = Field(default_factory=list)
    key_findings: list[Finding] = Field(default_factory=list)
    limitations: list[Finding] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)      # 生成信息 + 严格校验统计

    # ── 便利视图 ───────────────────────────────────────
    @property
    def good_metrics(self) -> list[MetricResult]:
        """**只有严格通过的**指标 —— 跨篇比较只该用这些。"""
        return [m for m in self.metrics if m.verified]

    @property
    def good_datasets(self) -> list[DatasetRef]:
        return [d for d in self.datasets if d.verified]

    def summary_line(self) -> str:
        n_m, n_mok = len(self.metrics), len(self.good_metrics)
        n_d, n_dok = len(self.datasets), len(self.good_datasets)
        return (f"{self.pdf} | 指标 {n_mok}/{n_m} 已核 | 数据集 {n_dok}/{n_d} 已核 "
                f"| 结论 {len(self.key_findings)} | 局限 {len(self.limitations)}")


# ── 缓存 ─────────────────────────────────────────────────────────────────────

def outcome_path(pdf: str) -> Path:
    return OUTCOME_DIR / f"{Path(pdf).stem}.outcome.json"


def _fingerprint(pdf: str) -> str:
    """指纹：VERSION + 模型 + PDF 内容版本。任一项变 → 重建。"""
    p = ROOT / "assets" / "papers" / Path(pdf).name
    try:
        st = p.stat()
        stamp = f"{st.st_size}-{st.st_mtime_ns}"
    except OSError:
        stamp = "missing"
    _, _, model = llm.config()
    return f"v{VERSION}|{model}|{stamp}"


def load_outcome(pdf: str) -> PaperOutcome | None:
    """读缓存（指纹不匹配 → None，**不报错**）。"""
    p = outcome_path(pdf)
    if not p.exists():
        return None
    try:
        o = PaperOutcome.model_validate_json(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 缓存损坏 → 当作没有
        return None
    return o if o.meta.get("fingerprint") == _fingerprint(pdf) else None


# ── 输入装配 ─────────────────────────────────────────────────────────────────

def _select_groups(pdf: str) -> tuple[list[dict], str]:
    """挑出进 prompt 的主张组 → (组列表, 渲染好的文本块)。

    只喂**核心**几类：claims 已概括过正文，全喂 140 条纯属浪费。
    """
    from paperpilot.pipeline import load_claim_models, load_group_models

    claims = {c.claim_id: c for c in load_claim_models(pdf)}
    groups = load_group_models(pdf)
    picked: list[dict] = []
    used: dict[str, int] = {}
    for g in sorted(groups, key=lambda x: -float(x.score.get("total", 0))):
        cap = GROUP_CAP.get(g.label, FALLBACK_CAP if g.type == "limitation" else 0)
        if cap <= 0 or used.get(g.label, 0) >= cap:
            continue
        used[g.label] = used.get(g.label, 0) + 1
        c = claims.get(g.rep_claim_id)
        picked.append({
            "group_id": g.group_id,
            "label": g.label,
            "type": g.type,
            "text": g.rep_text,
            "evidence": (c.evidence_quote if c else "")[:300],
            "imp": g.importance,
        })
    lines = []
    for g in picked:
        lines.append(f"- [{g['group_id']}] ({g['label']}/{g['type']}, i={g['imp']}) {g['text']}")
        if g["evidence"]:
            lines.append(f"    证据: {g['evidence']}")
    return picked, "\n".join(lines)


def _table_block(pdf: str) -> str:
    """表格/公式原文块 = `retrieval_chunks` 相对 `ordered_chunks` 的**注入增量**。

    只喂增量（不重复喂正文）—— 正文已被 claims 概括，表格是 claims 看不到的部分。
    """
    from paperpilot.agents.document_cache import ordered_chunks, retrieval_chunks

    base = {c.chunk_id: c.text for c in ordered_chunks(pdf)}
    out = []
    for c in retrieval_chunks(pdf):
        extra = c.text[len(base.get(c.chunk_id, "")):].strip()
        if extra:
            out.append(f"[{c.chunk_id}] p{c.page_span[0]}\n{extra}")
    return "\n\n".join(out)


# ── 严格校验（纯函数，无 LLM）─────────────────────────────────────────────────

def _norm_num(s: str) -> str:
    return str(s or "").replace(",", "").replace(" ", "")


def _value_in_quote(value: str, quote: str) -> bool:
    """数值是否**逐字**出现在 quote 里（容忍千分位逗号差异）。"""
    v = _norm_num(value)
    if not v:
        return False
    return _norm_num(quote).find(v) >= 0


def _locate(quote: str, index: list[tuple]) -> tuple[str, int] | None:
    """在检索视图里搜出 quote 所在块 → (chunk_id, page)。**后端绑锚，不信 LLM**。

    `index` = `[(chunk_id, page, norm_text, alnum_norm)]` —— 由调用方**预计算一次**：
    52 块 × 每条 quote 都重算归一化会白烧几十秒。
    """
    from paperpilot.tools.evidence import alnum_norm, norm_text

    nq = norm_text(quote or "")
    if len(nq) < MIN_QUOTE:
        return None
    for cid, page, t, _a in index:                   # 严格：逐字
        if nq in t:
            return cid, page
    an = alnum_norm(nq)                              # 兜底：字符级噪声（断行/符号字形）
    if an:
        for cid, page, _t, a in index:
            if an in a:
                return cid, page
    return None


def _locate_values(values: list[str], index: list[tuple]) -> tuple[str, int] | None:
    """退一步的判据：这些数值是否在**同一块里共现** → `(chunk_id, page)`。

    为什么需要（2026-09-21 实测）：表格在文本层常是**列式散开**的（每个数一行、
    中间夹着别的数），LLM 报的"那一行"是**语义重建**，**不可能逐字命中** ——
    实测 μPLAN 同一篇重跑，强档从 8/8 掉到 **0/8**，全卡在这里。

    这一档能挡住**编造的数值**（编的数不会出现在论文里），但**证明不了**
    "这个数属于这一行 / 这个数据集" —— 所以只是第二档，标 `verify="values"`。
    """
    vals = [v for v in values if str(v or "").strip()]
    if not vals:
        return None
    for cid, page, t, _a in index:
        nt = _norm_num(t)
        if all(_norm_num(v) in nt for v in vals):
            return cid, page
    return None


def _normalize(o: PaperOutcome) -> None:
    """后端规范化（**不交给 LLM**）：指标名归一 + 可比性判定 + `split`/`model` 分离。

    两件事 LLM 做不可靠，所以放后端：
      · **可比性**：自创指标（`LaSE`）不能与公认指标（`BLEU`）摆同一列比较；
      · **模型名**：LLM 习惯把模型塞进 split（`CNN/DailyMail (mT5)`）→ 拆出来，
        否则"同一数据集不同模型"筛不出来。
    """
    for m in o.metrics:
        m.name, m.comparable = _canon_metric(m.name)
        mm = _MODEL_IN_DATASET_RE.match(m.dataset or "")
        if mm and not m.model:
            m.dataset, m.model = mm.group(1).strip(), mm.group(2).strip()
        if re.sub(r"[\s_]+", " ", (m.dataset or "").lower()).strip() in _GENERIC_DATASETS:
            m.dataset = ""                       # "Test set" 不是数据集
    for d in o.datasets:
        d.name = re.sub(r"\s+", " ", str(d.name or "").strip())


def _strict_verify(o: PaperOutcome, pdf: str) -> dict:
    """逐条回溯原文并绑锚。返回统计。**不丢弃任何条**，只标 `verified`。"""
    from paperpilot.tools.evidence import alnum_norm, norm_text

    from paperpilot.agents.document_cache import retrieval_chunks

    chunks = retrieval_chunks(pdf)
    index = [(c.chunk_id, int(c.page_span[0]), norm_text(c.text), alnum_norm(c.text))
             for c in chunks]

    for d in o.datasets:
        loc = _locate(d.quote, index)
        if loc:
            d.anchor_chunk, d.anchor_page, d.verified = loc[0], loc[1], True

    for m in o.metrics:
        loc = _locate(m.quote, index)
        if loc and _value_in_quote(m.value, m.quote):
            # 强档：quote 逐字命中，且数值就在这个 quote 里面
            m.anchor_chunk, m.anchor_page = loc[0], loc[1]
            m.verified, m.verify = True, "exact"
        elif loc:                      # quote 找到了但数值不在里面 → 数值存疑
            m.anchor_chunk, m.anchor_page = loc[0], loc[1]
        else:
            # 弱档（兜底）：表格列式散开 → quote 是语义重建，逐字命中不了。
            # 退一步要求**报出来的数值同块共现**（含 baseline）。
            loc2 = _locate_values([m.value, m.baseline], index)
            if loc2:
                m.anchor_chunk, m.anchor_page = loc2[0], loc2[1]
                m.verified, m.verify = True, "values"
        # baseline 也必须逐字在该 quote 里 —— 否则**清空**。
        # ⚠️ 只对强档做：弱档的 baseline 已由"同块共现"保证过，再按逐字要求会误清。
        # ⚠️ 校验只能证明"这个数在原文里"，**证明不了"它是基线"**（那是 LLM 的语义判断，
        #    prompt 里已要求判不了就留空）。宁可没有 baseline，也不留一个错的。
        if m.verify == "exact" and m.baseline and not _value_in_quote(m.baseline, m.quote):
            m.baseline = ""
        # delta 由后端算（LLM 不算数，防幻觉）；拿不到 baseline 就空着
        if m.baseline:
            try:
                d = float(_norm_num(m.value)) - float(_norm_num(m.baseline))
                m.delta = f"{d:+.4g}"
            except ValueError:
                m.delta = ""

    labels = {g["group_id"] for g in o.meta.get("_groups", [])}
    for f in o.key_findings + o.limitations:
        f.verified = bool(f.group_id and f.group_id in labels)

    return {
        "datasets": [len(o.datasets), len(o.good_datasets)],
        "metrics": [len(o.metrics), len(o.good_metrics)],
        "metrics_exact": sum(1 for m in o.metrics if m.verify == "exact"),
        "metrics_values": sum(1 for m in o.metrics if m.verify == "values"),
        "findings": len(o.key_findings) + len(o.limitations),
    }


# ── 主入口 ───────────────────────────────────────────────────────────────────

def build_outcome(pdf: str, *, force: bool = False, verbose: bool = True
                  ) -> PaperOutcome:
    """生成（或复用缓存的）成果层。

    Args:
        pdf: `assets/papers/` 下的文件名（须已有 `report.json`）。
        force: True 时忽略缓存重建。
        verbose: 打印进度。

    Returns:
        `PaperOutcome`。**失败抛异常**（`llm.LLMError` / `FileNotFoundError`），
        由调用方（workflow / CLI）决定是否降级 —— 与 `process_pdf` 一致。
    """
    log = print if verbose else (lambda *a: None)
    if not force:
        cached = load_outcome(pdf)
        if cached is not None:
            log(f"  [outcome] 缓存复用 {outcome_path(pdf).name} | {cached.summary_line()}")
            return cached

    from paperpilot.pipeline import _display_title

    t0 = time.time()
    groups, groups_block = _select_groups(pdf)
    table_block = _table_block(pdf)
    title = _display_title(pdf) or Path(pdf).stem
    log(f"  [outcome] 抽取：候选组 {len(groups)} 条 | 表格/公式 {len(table_block):,} 字符")

    from paperpilot.tools import mock_llm
    if mock_llm.llm_enabled():
        o = _mock_outcome(pdf, title, groups)
    else:
        raw = llm.chat_json(OUTCOME_SYSTEM, build_user(title, groups_block, table_block),
                            temperature=0.0, max_tokens=2048)
        o = _coerce(pdf, title, raw)

    _normalize(o)                                 # 指标名/可比性/split-model（后端定）
    o.meta["_groups"] = groups                    # 校验 group_id 合法性用，落盘前删掉
    o.meta["stats"] = _strict_verify(o, pdf)
    o.meta["seconds"] = round(time.time() - t0, 1)
    o.meta["usage"] = llm.usage_stats()
    o.meta.pop("_groups", None)
    o.meta["fingerprint"] = _fingerprint(pdf)

    OUTCOME_DIR.mkdir(parents=True, exist_ok=True)
    outcome_path(pdf).write_text(
        json.dumps(o.model_dump(), ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"  [outcome] 完成 {o.meta['seconds']}s | {o.summary_line()}")
    return o


def _coerce(pdf: str, title: str, raw: object) -> PaperOutcome:
    """把 LLM 返回的任意形状**防御性**转成 `PaperOutcome`（形状不对就给空，不崩）。"""
    d = raw if isinstance(raw, dict) else {}

    def sl(v: object) -> list:
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []

    def ss(v: object) -> list[str]:
        return [str(x) for x in v if isinstance(x, (str, int, float))] if isinstance(v, list) else []

    def items(key: str, cls: type) -> list:
        out = []
        for x in sl(d.get(key))[:MAX_METRICS if key == "metrics" else 64]:
            try:
                out.append(cls.model_validate(x))
            except Exception:  # noqa: BLE001 单条坏 → 跳过，不影响整张卡
                continue
        return out

    return PaperOutcome(
        pdf=pdf,
        title=title,
        problem=str(d.get("problem") or "").strip()[:200],
        method_family=[s.lower().strip() for s in ss(d.get("method_family"))][:6],
        datasets=items("datasets", DatasetRef),
        metrics=items("metrics", MetricResult),
        key_findings=[Finding(text=str(x.get("text") or "").strip(),
                              group_id=str(x.get("group_id") or ""))
                      for x in sl(d.get("key_findings")) if x.get("text")],
        limitations=[Finding(text=str(x.get("text") or "").strip(),
                             group_id=str(x.get("group_id") or ""))
                     for x in sl(d.get("limitations")) if x.get("text")],
    )


def _mock_outcome(pdf: str, title: str, groups: list[dict]) -> PaperOutcome:
    """演示模式（`PAPERPILOT_MOCK_LLM=1`）：不联网，直接从主张组拼一张卡。"""
    top = [g for g in groups if g["label"] == "core_claim"][:1]
    return PaperOutcome(
        pdf=pdf, title=title,
        problem=(top[0]["text"] if top else "（演示）本卡由主张组直接拼装，未调用 LLM"),
        method_family=["(demo)"],
        key_findings=[Finding(text=g["text"], group_id=g["group_id"]) for g in groups[:6]],
        limitations=[Finding(text=g["text"], group_id=g["group_id"])
                     for g in groups if g["type"] == "limitation"][:5],
    )


__all__ = ["OUTCOME_DIR", "DatasetRef", "Finding", "MetricResult", "PaperOutcome",
           "VERSION", "build_outcome", "load_outcome", "outcome_path"]
