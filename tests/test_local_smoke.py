"""本地真模型冒烟（`uv run pytest -m local`）：**需要真 key / 真模型 / 真 MinerU**，CI 不跑。

为什么要有这一档：`-m local` 此前只有 1 条（validator 的 uncited 检查），于是"逻辑对错"归 CI、
"效果高低"靠 `qa/` 脚本**手工读报告** —— 中间缺一层"**跑得动、判得死**"的真模型冒烟。
这三条就是那层，失败时能直接指出是哪一环坏了：

| 用例 | 需要 | 判什么 | 首次典型耗时 |
|---|---|---|---|
| `test_local_embedding_ranking` | bge-m3 | 中文提问能否命中真·跨语言检索（假向量测不出这条） | 30~60s（含载模型） |
| `test_local_llm_answer_has_citation` | 上面的 + LLM key | 真 LLM 走完问答：答案非空、引用能解析回真实页码 | +10~40s |
| `test_local_mineru_single_page` | `.venv-mineru`（有 GPU 更快） | 真 MinerU 能出 content_list 且可解析成块 | 30~120s |

三条都**自带 skip**（缺模型 / 缺 key / 缺 MinerU 环境就跳过，不红）：
"一条命令在任何人机器上都该是绿的"对 local 档同样成立 —— 只是它默认不跑
（`addopts` 里是 `-m 'not local and not ui'`）。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.local

ROOT = Path(__file__).resolve().parents[1]
DEMO_PDF = ROOT / "demo" / "demo_paper.pdf"

# 演示论文里**逐字存在**的两处事实（`demo/make_demo_pdf.py` 生成，见文件正文）
FACT_ARCH = "dual-tower"
FACT_NUM = "87.3"


def _require_real_embedder() -> None:
    """真向量可用吗？不可用就 skip（而不是红）。"""
    from paperpilot.agents import embedder

    if os.environ.get("PAPERPILOT_MOCK_EMBED") == "1":
        pytest.skip("PAPERPILOT_MOCK_EMBED=1：这是真模型用例，别在演示模式下跑")
    try:
        v = embedder.encode_texts(["探针 probe sentence"])
    except Exception as e:  # noqa: BLE001  模型没下载 / 没网 / 依赖缺失
        pytest.skip(f"真向量模型不可用（{type(e).__name__}: {e}）")
    assert v.shape == (1, embedder.EMBED_DIM)


@pytest.fixture
def demo_in_tmp(tmp_assets) -> str:
    """把演示论文拷进**隔离**的论文库：不碰真实 `assets/`，也不依赖本机别的论文。"""
    assert DEMO_PDF.exists(), "演示论文应随仓库提供"
    (tmp_assets.papers / "demo_paper.pdf").write_bytes(DEMO_PDF.read_bytes())
    return "demo_paper.pdf"


@pytest.fixture
def real_env():
    """把本机 `.env` 灌进 `os.environ`。

    为什么必须显式做：conftest 的 autouse `isolate` 会**清掉** LLM/裁判的 key
    （那是有意设计：CI 与"别人 clone"不该有 key）——于是 local 用例得自己按
    `web/app.py` 的方式加载 `.env`。这些 key 会在**下一个测试**的 `isolate` 里被清掉，
    不会污染别的用例。
    """
    from paperpilot.tools import llm

    llm._load_dotenv(str(ROOT))
    return True


# ───────────────────────── ① 真向量检索位次 ─────────────────────────


def test_local_embedding_ranking(demo_in_tmp):
    """真 bge-m3：**中文提问**要能命中英文论文里的正确块（我们的真实使用场景）。

    CI 里用的是"词袋哈希"假向量（只测管道通不通），**测不出**：
      · 模型是否真的加载成了；
      · 中文 query 与英文 chunk 是否在同一个语义空间（跨语言检索）；
      · 向量维数/归一化有没有被改坏。
    """
    _require_real_embedder()
    from paperpilot.agents.embedder import ChunkIndex

    idx = ChunkIndex(demo_in_tmp)

    # ① 语义型：中文问"提出了什么检索结构" → 含 dual-tower 的块必须进 top-3
    hits = idx.search_hybrid("这篇论文提出了什么检索结构？", top_k=3)
    assert hits, "检索一条都没返回"
    joined = " ".join(str(h.get("text", "")) for h in hits).lower()
    assert FACT_ARCH in joined, (
        "top-3 没命中主题块（跨语言检索坏了？）—— "
        f"命中块：{[str(h.get('text', ''))[:50] for h in hits]}")

    # ② 事实型：另一条独立查询（避免"碰巧命中"）——数字必须能被检索到
    hits2 = idx.search_hybrid("accuracy on the public benchmark", top_k=3)
    assert any(FACT_NUM in str(h.get("text", "")) for h in hits2), (
        "top-3 没命中含 87.3 的块")

    # ③ 结构完整性：命中项必须带 chunk_id / 页码（[n] 引用要能映射回去）
    for h in hits + hits2:
        assert h.get("chunk_id") and int(h.get("page") or 0) >= 1


# ───────────────────────── ② 真 LLM 作答 + 引用 ─────────────────────────


def test_local_llm_answer_has_citation(demo_in_tmp, real_env, monkeypatch):
    """真 LLM + 真向量走完一次问答：答案非空、引用能解析回**真实页码**。

    报告用**演示模式**先产出（快、确定、不烧 token）；**提问那一步换成真 LLM**。
    这样测的是"真模型这一环通不通"，不是"报告内容好不好"（后者是 `qa/` 的评测活）。
    """
    from paperpilot.tools import llm

    if not llm.is_configured():
        pytest.skip("未配置 LLM（PAPERPILOT_LLM_*）—— `-m local` 需要真 key")
    _require_real_embedder()

    from paperpilot.pipeline import process_pdf

    monkeypatch.setenv("PAPERPILOT_MOCK_LLM", "1")      # 报告链走固定示例（快）
    try:
        process_pdf(demo_in_tmp, force=True, verbose=False)
    finally:
        monkeypatch.delenv("PAPERPILOT_MOCK_LLM")       # 提问换回真 LLM

    from paperpilot.graph import ask as graph_ask

    r = graph_ask("这篇论文提出的检索结构叫什么？在公开基准上准确率是多少？", demo_in_tmp)
    assert r.get("answer"), f"真 LLM 没给出答案：{r}"
    # 护栏：**这条不能悄悄退化成 mock 测试**（答案带着演示模式的标记就说明开关没清掉）
    from paperpilot.tools import mock_llm

    assert mock_llm._TAG not in r["answer"], "答案带演示模式标记 → 提问没有真走 LLM"
    assert "ingest_blocked" not in (r.get("route") or []), r
    cites = r.get("cites") or []
    assert cites, "答案没有引用（cites 为空）—— 引用链路或作答提示词坏了"
    for c in cites:
        assert int(c.get("page") or 0) >= 1, f"引用没有真实页码：{c}"
    assert (r.get("validator") or {}).get("action") in {"pass", "repaired", "fallback"}


# ───────────────────────── ③ 真 MinerU 解析 ─────────────────────────


def test_local_mineru_single_page(demo_in_tmp):
    """真 MinerU：单页 PDF 能跑出产物，且 bridge 能解析成块。

    最慢、也最"环境相关"的一条（模型加载 + 可能起 GPU 子进程）；没 `.venv-mineru` 就 skip。
    """
    from paperpilot import ingest
    from paperpilot.tools.mineru_bridge import chunks_from_mineru_dir

    ok, why = ingest.mineru_available()
    if not ok:
        pytest.skip(f"MinerU 环境不可用：{why}")

    m = ingest.run_mineru(demo_in_tmp, force=True, timeout=600)
    assert m.get("status") == "ok", f"MinerU 跑失败：{m.get('reason')}"

    ch = chunks_from_mineru_dir(ingest.MINERU_OUT / Path(demo_in_tmp).stem)
    assert ch, "MinerU 产物解析不出块（content_list 结构变了？见 mineru_bridge）"
