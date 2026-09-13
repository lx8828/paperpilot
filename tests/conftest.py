"""CI 测试基础设施：**一切外部依赖都被替掉**（无 key / 无 GPU / 无模型 / 无网络）。

三条隔离原则：

1. **路径隔离** —— 生产把 `assets/papers`、`assets/artifacts` 写死在模块常量里；
   测试全部 monkeypatch 到 `tmp_path`，**绝不碰真实论文与产物**
   （这也是"别人 clone 下来就能跑"的前提：CI 上这些目录是空的）。
2. **外部依赖替身** —— LLM（`llm._chat`）、embedding（`embedder.encode_texts/encode_query`）、
   MinerU（假 `content_list.json`）三样都替掉。**这是"不依赖真实 API Key"的落地方式。**
3. **网络硬护栏** —— `urllib.request.urlopen` 被换成"一用就炸"：
   万一哪天漏了替身，测试会**明确失败**，而不是偶尔联网成功变成不确定的测试。

测试分两层（见 `pyproject.toml` 的 `-m 'not local'`）：

| 标记 | 谁跑 | 说明 |
|---|---|---|
| （无标记） | **CI**：`uv run pytest -q` | 离线、确定、秒级 |
| `@pytest.mark.local` | 只有本地：`uv run pytest -m local` | 需要真实模型 / LLM，默认**跳过**（保证任何人的"一条命令"都是绿的） |
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
# 生产代码（src）、仓库根（web/app.py 等）、评测口径模块（qa/recall/_tgt.py）
for _p in (ROOT / "src", ROOT, ROOT / "qa" / "recall"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 这些 env 会改变**默认行为**；测试里一律清掉，测的才是真正的代码默认值
_ENV_KEYS = (
    "PAPERPILOT_EXT_QUOTA", "PAPERPILOT_EXT_RRF_ALPHA", "PAPERPILOT_USE_MINERU",
    "PAPERPILOT_MINERU_INJECT", "PAPERPILOT_QASPER_TABLES", "PAPERPILOT_TABLE_V1",
    "PAPERPILOT_TABLE_EMBED_SUMMARY", "PAPERPILOT_PDF_ID_STRICT",
    "PAPERPILOT_QUERY_REWRITE", "PAPERPILOT_VALIDATOR_GATE", "PAPERPILOT_VALIDATOR_LLM",
    "PAPERPILOT_VALIDATOR_MISSING", "PAPERPILOT_VALIDATOR_REPAIR_MID",
    "PAPERPILOT_CHUNK_VIEW_DIR",
    # LLM/裁判的 key 也必须清掉：否则"本机 .env 里有 key"会让
    # `is_configured()` 变 True，测试结果**取决于开发者的机器**（CI 上则相反）。
    "PAPERPILOT_LLM_BASE_URL", "PAPERPILOT_LLM_API_KEY", "PAPERPILOT_LLM_MODEL",
    "PAPERPILOT_JUDGE_BASE_URL", "PAPERPILOT_JUDGE_API_KEY", "PAPERPILOT_JUDGE_MODEL",
)

_CACHE_MODULES = (
    "paperpilot.agents.document_cache", "paperpilot.agents.embedder",
    "paperpilot import paper_identity", "paperpilot.ingest", "paperpilot.jobs",
)


def _clear_all_caches() -> None:
    """清掉所有 lru_cache（路径被重定向后，旧缓存必须作废）。

    泛化处理：模块里任何带 `cache_clear` 的对象都清一遍 —— 新增缓存函数时
    **不用改这里**（避免"忘了清缓存"导致测试间互相污染）。
    """
    import importlib
    for dotted in ("paperpilot.agents.document_cache", "paperpilot.agents.embedder",
                   "paperpilot.paper_identity", "paperpilot.ingest", "paperpilot.jobs"):
        try:
            mod = importlib.import_module(dotted)
        except Exception:  # noqa: BLE001
            continue
        for name in dir(mod):
            fn = getattr(mod, name, None)
            cc = getattr(fn, "cache_clear", None)
            if callable(cc):
                try:
                    cc()
                except Exception:  # noqa: BLE001
                    pass


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    """每个测试都在"干净 + 断网"的环境里跑（autouse）。"""
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    import urllib.request

    def _blocked(*_a, **_k):
        raise AssertionError("测试禁止真实网络请求：请使用 fake_llm / fake_embed 替身")

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    _clear_all_caches()
    yield
    _clear_all_caches()


@pytest.fixture
def tmp_assets(tmp_path, monkeypatch):
    """把生产写死的 assets 路径**全部**重定向到 tmp_path。

    目录形状与线上一致（`<root>/assets/...`），这样 `web/app.py` 那种
    "自己拼 `ROOT / "assets/..."`" 的代码只要把 `ROOT` 指过来就行。
    """
    papers = tmp_path / "assets" / "papers"
    claims = tmp_path / "assets/artifacts/out_claims"
    views = tmp_path / "assets/artifacts/out_views"
    mineru = tmp_path / "assets/artifacts/out_mineru"
    jobs = tmp_path / "assets/artifacts/out_jobs"
    for p in (papers, claims, views, mineru, jobs):
        p.mkdir(parents=True, exist_ok=True)

    from paperpilot import ingest, jobs as jobs_mod, paper_identity, pipeline
    from paperpilot.agents import document_cache, embedder
    from paperpilot.agents.nodes import report as node_report

    for mod, attr, val in (
        (paper_identity, "PAPERS_DIR", papers),
        (document_cache, "PAPERS_DIR", papers),
        (document_cache, "MINERU_OUT", mineru),
        (pipeline, "PAPERS_DIR", papers),
        (pipeline, "CLAIMS_DIR", claims),
        (pipeline, "VIEW_DIR", views),
        (pipeline, "MINERU_OUT", mineru),
        (ingest, "PAPERS_DIR", papers),
        (ingest, "ASSETS", tmp_path / "assets"),
        (ingest, "ARTIFACT_ROOT", tmp_path / "assets" / "artifacts"),
        (ingest, "MINERU_OUT", mineru),
        (embedder, "CHUNK_VIEW_DIR", views),
        (jobs_mod, "JOB_DIR", jobs),
        (node_report, "VIEW_DIR", views),
    ):
        monkeypatch.setattr(mod, attr, val)
    _clear_all_caches()
    return SimpleNamespace(papers=papers, claims=claims, views=views, mineru=mineru,
                           jobs=jobs, root=tmp_path)


# ───────────────────────── 假 LLM ─────────────────────────

# 小结 PDF 里**逐字存在**的句子（claims 的 evidence_quote 用它 → 证据回核判 hit）
PDF_QUOTE = "We propose a dual-tower retrieval architecture"
PDF_QUOTE2 = "The method reaches 87.3% accuracy on the test set."


def _canned(system: str) -> str | None:
    """按 system prompt 返回预设响应（None = 不认识 → 由调用方兜底）。"""
    from paperpilot.agents.nodes import answer as A, judge as J
    from paperpilot.components import validator as V
    from paperpilot.prompts import analyzer as Pa, figures as Pf, report as Pr
    from paperpilot.prompts import skeleton as Ps, viewer as Pv

    table: dict[str, object] = {
        # ① claims 提取：给 1 条 method 主张，证据**逐字来自 PDF** → 回核 hit
        Pa.SYSTEM_PROMPT: [{"type": "method",
                            "text": "提出双塔检索结构（dup-tower retrieval）",
                            "evidence_quote": PDF_QUOTE}],
        # ② 去重：单组
        Pv.DEDUPE_SYSTEM: [{"best_id": "c0001", "claim_ids": ["c0001"]}],
        # ③ 打标：核心命题（→ importance 顶格 → 进 core_points）
        Pv.LABEL_SYSTEM: [{"group_id": "g1", "label": "core_claim", "why": "主结论"}],
        # ④ 骨架 / 图表指南：空（不测这两条线）
        Ps.SKELETON_SYSTEM: [],
        Pf.GUIDE_SYSTEM: [],
        # ⑤ 概述 / 导读
        Pr.OVERVIEW_SYSTEM: {"overview": "本文提出双塔检索结构，在测试集上达到 87.3%。"},
        Pr.GUIDE_SYSTEM: {"guide": "一句话：用两个塔分别编码图文，再对齐。"},
        # ⑥ 判够（L0/L3 都给"够" → 走直答/作答路径）
        J._SYS_L0: {"enough": True, "target_sections": [], "gap": ""},
        J._SYS_L3: {"enough": True, "target_sections": [], "gap": ""},
        # ⑦ 闸门 LLM 体检：无问题
        V._SYS_UNCITED: {"uncited": []},
        V._SYS_VALIDATE: {"unsupported": [], "off_topic": False},
    }
    # 问答生成（chat_text）：答案带 [1] 引用 → cites 非空
    table[A.SYSTEM] = "本文提出双塔检索结构，用对比学习对齐图文表示 [1]。"
    table[A.SYSTEM_EXTRACT] = "（抽取式回答）87.3%"
    table[A.SYSTEM_EXTRACT_TABLE] = "（表格抽取）"
    table[A.SYSTEM_UNKNOWN] = "论文中没有提到这一点。"

    for key, val in table.items():
        if key == system:
            return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
    return None


@pytest.fixture
def fake_llm(monkeypatch):
    """把 LLM 换成本地路由器（**零网络、零 key**）；同时让 `llm.is_configured()` 为真。

    替换点在**最底层 `_chat`**：于是 `chat_json / chat_text / judge_json` 以及
    validator 的裁判模型调用全都自动走假实现，不用逐个打补丁。
    """
    from paperpilot.tools import llm

    monkeypatch.setenv("PAPERPILOT_LLM_BASE_URL", "https://fake.local/v1")
    monkeypatch.setenv("PAPERPILOT_LLM_API_KEY", "test-key")
    monkeypatch.setenv("PAPERPILOT_LLM_MODEL", "fake-model")
    monkeypatch.setenv("PAPERPILOT_JUDGE_BASE_URL", "https://fake.local/v1")
    monkeypatch.setenv("PAPERPILOT_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("PAPERPILOT_JUDGE_MODEL", "fake-judge")

    calls: list[tuple[str, str]] = []

    def _fake_chat(system: str, user: str, *, temperature: float,
                   max_tokens: int | None, prefix: str = "PAPERPILOT_LLM") -> str:
        calls.append((system, user))
        got = _canned(system)
        if got is None:
            return "[]"          # 未映射 → 空数组（多数阶段都能安全处理）
        return got

    monkeypatch.setattr(llm, "_chat", _fake_chat)
    monkeypatch.setattr(llm, "_USAGE", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    fake_llm.calls = calls        # type: ignore[attr-defined]
    return _fake_chat


# ───────────────────────── 假 embedding ─────────────────────────


@pytest.fixture
def fake_embed(monkeypatch):
    """确定性的"词袋哈希"向量器：词重叠越多 → 余弦越高。

    为什么不用随机向量：那样检索链路测不出"能不能召回正确的块"（只测不崩）。
    这里用 MD5 桶 + 归一化，跨平台/跨进程稳定（**不依赖 Python 的 hash 随机化**）。
    """
    import numpy as np
    from paperpilot.agents import embedder

    def _vec(text: str) -> "np.ndarray":
        v = np.zeros(embedder.EMBED_DIM, dtype="float32")
        for tok in str(text).lower().replace("|", " ").split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
            v[h % embedder.EMBED_DIM] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def _texts(texts: list[str]):
        if not texts:
            return np.zeros((0, embedder.EMBED_DIM), dtype="float32")
        return np.stack([_vec(t) for t in texts]).astype("float32")

    monkeypatch.setattr(embedder, "encode_texts", _texts)
    monkeypatch.setattr(embedder, "encode_query", _vec)
    return _vec


# ───────────────────────── 假论文（真 PDF + 假 MinerU 产物）─────────────────────────

TINY_PDF = "tiny.pdf"


@pytest.fixture
def tiny_pdf(tmp_assets):
    """现场生成一份小 PDF（**真** pymupdf 解析：普通库、无模型 → CI 可跑）。"""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "PaperPilot Test Paper", fontsize=18)
    page.insert_text((72, 110), "1 Introduction", fontsize=15)
    page.insert_text((72, 140), PDF_QUOTE + " for composed image retrieval.", fontsize=11)
    page.insert_text((72, 156), PDF_QUOTE2, fontsize=11)
    page.insert_text((72, 190), "2 Method", fontsize=15)
    page.insert_text((72, 220), "The encoder is trained with contrastive learning.", fontsize=11)
    path = tmp_assets.papers / TINY_PDF
    doc.save(str(path))
    doc.close()
    return TINY_PDF


@pytest.fixture
def fake_mineru(tmp_assets):
    """造一份极小的 MinerU `content_list.json`（替掉需要 GPU 的版面解析）。

    内容：一块正文 + 一张表（含 caption 与 HTML 表体）。
    默认模式（`PAPERPILOT_USE_MINERU` 未设）下它只影响**检索视图**与评测，
    报告链仍读 pymupdf —— 与线上一致。
    """
    def _write(pdf_name: str = TINY_PDF, *, with_table: bool = True,
               fig_caption_noise: bool = False) -> Path:
        stem = Path(pdf_name).stem
        d = tmp_assets.mineru / stem
        d.mkdir(parents=True, exist_ok=True)
        caps = ["Figure 6: Attention Distributions"] if fig_caption_noise else []
        items: list[dict[str, object]] = [{
            "type": "text", "text_level": 2, "page_idx": 0,
            "text": "1 Introduction " + PDF_QUOTE,
        }]
        if with_table:
            items.append({
                "type": "table", "page_idx": 0,
                "table_caption": [*caps, "Table 1: Accuracy of the proposed method"],
                "table_body": "<table><tr><td>Model</td><td>Acc</td></tr>"
                              "<tr><td>Ours</td><td>87.3</td></tr></table>",
            })
        # 文件名必须匹配 `<...>_content_list.json`（`mineru_bridge._find_content_list`
        # 按 `*_content_list.json` 递归找，与真实 MinerU 产物布局一致）
        (d / f"{stem}_content_list.json").write_text(
            json.dumps(items, ensure_ascii=False), encoding="utf-8")
        return d

    return _write


# ───────────────────────── 假 MinerU 环境 ─────────────────────────


@pytest.fixture
def fake_mineru_env(monkeypatch):
    """让 `run_mineru` 认为"MinerU 可用 + 产物已存在" → 走**复用**分支。

    这样 CI 上不需要 `.venv-mineru`、不跑子进程、不碰 GPU，但走的是**真实**的
    摄取代码路径（版本比对 + 产物可解析性校验 + meta 落盘）。
    """
    from paperpilot import ingest

    monkeypatch.setattr(ingest, "mineru_available", lambda: (True, ""))
    monkeypatch.setattr(ingest, "mineru_cmd", lambda: "mineru-fake")
    monkeypatch.setattr(ingest, "mineru_version", lambda: "0.0.0-test")


# ───────────────────────── web app（FastAPI）─────────────────────────


@pytest.fixture(scope="session")
def webapp():
    """加载 `web/app.py`（它不是包的一部分，用 spec 加载）。

    ⚠️ 必须先注册进 `sys.modules`：`app.py` 开了 `from __future__ import annotations`，
    pydantic 解析 `AskBody` 的注解时要能按模块名取到命名空间，否则会报
    `PydanticUserError: AskBody is not fully defined`。
    """
    import importlib.util
    import sys as _sys

    from paperpilot.tools import llm

    # `web/app.py` 导入时会 `llm._load_dotenv(ROOT)` → 把**本机 .env**（可能含真 key）
    # 灌进 `os.environ`，测试结果就会取决于开发者机器。这里禁掉它，保证结果只由测试决定。
    llm._load_dotenv = lambda *_a, **_k: None      # type: ignore[assignment]

    spec = importlib.util.spec_from_file_location("pp_webapp", ROOT / "web" / "app.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    rebuild = getattr(getattr(mod, "AskBody", None), "model_rebuild", None)
    if callable(rebuild):
        rebuild()
    return mod


@pytest.fixture
def webapp_tmp(webapp, tmp_assets, monkeypatch):
    """把 web 层也指到 tmp 论文库（`app.py` 里自己拼的 ROOT/PAPERS_DIR 一起改）。"""
    monkeypatch.setattr(webapp, "ROOT", tmp_assets.root)
    monkeypatch.setattr(webapp, "PAPERS_DIR", tmp_assets.papers)
    return webapp
