"""引用对齐的**串篇 / 误匹配**回归（2026-10-02 建；2026-10-03 按评审扩写）。

核心事实：**`chunk_id` 只在篇内唯一** —— 每篇论文的块都从 `c1` 开始。
所以这条链上任何**只按 cid** 的匹配或去重都会**串篇**，而且是**静默**的：
用户点引用会跳到**另一篇论文**上，看起来还像"跳错了页"。

评审指出的三处（下面都有对应测试，均已实测坐实）：
  1. `fullctx._extract_cites` 用 `cid in answer`（宽子串）→ `c1` 会被 `c12` 命中；
  2. 同函数 `seen` 只按 cid → A、B 都有 `c1` 时，答案引 `[P2…c1]` 返回 **A 篇的 c1**；
  3. `_cites_from` 的锚点分支只取 cid、**丢了 `P`** → 同样串篇。
另发现（同样致命）：`_entries_from_cites` 把整条 `ref` 当 `chunk_id`、且**不产出 `pdf`**
→ 那条锚点通道在**真实修复路径上根本命中不了**（"修好了"其实没修上），
且所有 cite 会被标成第 1 篇。见 `test_entries_from_cites_*`。
"""
from __future__ import annotations

import pytest

from paperpilot.agents.nodes.answer import _cites_from
from paperpilot.components import fullctx
from paperpilot.components.repairer import _entries_from_cites


# ── 桩：让 evidence 自带"我来自哪篇"，取错了篇一眼可见 ──────────────────────
class _Chunk:
    def __init__(self, cid: str, text: str) -> None:
        self.chunk_id, self.text = cid, text


class _Idx:
    def __init__(self, pdf: str) -> None:
        self.pdf = str(pdf)

    def _doc_chunks(self):
        return [_Chunk("c1", f"[{self.pdf} 的 c1]"), _Chunk("c12", f"[{self.pdf} 的 c12]")]


@pytest.fixture
def index_two_papers(monkeypatch):
    """A、B 两篇**都有 `c1`**（现实里必然发生），A 另有 c12。"""
    monkeypatch.setattr(fullctx, "ChunkIndex", _Idx)
    return [
        {"p": 1, "key": "P1·§1·¶c1", "pdf": "A.pdf", "chunk_id": "c1",
         "section": "1", "page": 1},
        {"p": 2, "key": "P2·§1·¶c1", "pdf": "B.pdf", "chunk_id": "c1",
         "section": "1", "page": 1},
        {"p": 1, "key": "P1·§1·¶c12", "pdf": "A.pdf", "chunk_id": "c12",
         "section": "1", "page": 1},
    ]


# ═══════════ `fullctx._extract_cites`（评审点 1、2：此前**根本没被覆盖**）═══════════


def test_cross_paper_same_cid(index_two_papers):
    """★ 评审点 2：A、B 都有 `c1`，答案引 `[P2·§1·¶c1]` → 必须是 **B** 篇。

    旧实现返回 `[('A.pdf','c1')]`（`seen` 只按 cid，先遍历到 A 就锁死了）。
    """
    got = fullctx._extract_cites("结论见 [P2·§1·¶c1]。", index_two_papers)
    assert [(c["p"], c["pdf"], c["chunk_id"]) for c in got] == [(2, "B.pdf", "c1")]
    assert "B.pdf" in got[0]["evidence"]          # 连原文也必须是 B 篇的


def test_both_papers_same_cid_both_kept(index_two_papers):
    """同一次答案同时引 A 的 c1 与 B 的 c1 → **两条都要在**（旧去重会吃掉一条）。"""
    got = fullctx._extract_cites("A 见 [P1·§1·¶c1]，B 见 [P2·§1·¶c1]。", index_two_papers)
    assert sorted((c["p"], c["pdf"]) for c in got) == [(1, "A.pdf"), (2, "B.pdf")]


def test_c12_anchor_does_not_pull_in_c1(monkeypatch):
    """★ 评审点 1：答案引 `[P1·§1·¶c12]`，而语料里**只有 c1** → 必须返回空。

    旧判据 `"c1" in answer` 会被 `"c12"` 命中 → 凭空多出一条**错**引用。
    """
    monkeypatch.setattr(fullctx, "ChunkIndex", _Idx)
    index = [{"p": 1, "key": "P1·§1·¶c1", "pdf": "A.pdf", "chunk_id": "c1",
              "section": "1", "page": 1}]
    assert fullctx._extract_cites("见 [P1·§1·¶c12]。", index) == []


def test_bare_cid_is_not_a_citation(index_two_papers):
    """**没有锚点形态**的裸 `c1`（正文里的普通词）一律不算引用。"""
    assert fullctx._extract_cites("这里的 c1 只是个词，不是锚点。", index_two_papers) == []


def test_extract_cites_without_p_field_still_disambiguates(monkeypatch):
    """★ index **没有** `p` 字段时（老格式）也要正确消歧 —— 从 `key` 里刨篇号。

    只认显式 `p` 的话，这类 index 会**静默返回零条引用**（比给错的更隐蔽：
    用户会以为"这篇没有引用"）。`key` 从第一天起就带篇号（`P2·§…·¶c1`）。
    """
    monkeypatch.setattr(fullctx, "ChunkIndex", _Idx)
    index = [{"key": "P1·§1·¶c1", "pdf": "A.pdf", "chunk_id": "c1", "section": "1", "page": 1},
             {"key": "P2·§1·¶c1", "pdf": "B.pdf", "chunk_id": "c1", "section": "1", "page": 1}]
    got = fullctx._extract_cites("见 [P2·§1·¶c1]。", index)
    assert [(c["pdf"], c["chunk_id"]) for c in got] == [("B.pdf", "c1")]


def test_anchor_tolerates_messy_section(index_two_papers):
    """章节名里带 `·`、或带空格时也要能切对（`§` 后非贪婪到第一个 `·¶`）。"""
    got = fullctx._extract_cites("见 [P2·§3.1 · 引言·¶c1]。", index_two_papers)
    assert [(c["p"], c["pdf"]) for c in got] == [(2, "B.pdf")]


def test_unknown_anchor_is_ignored(index_two_papers):
    """锚点指向语料里没有的块 → 忽略（不抛异常、不产生空引用）。"""
    assert fullctx._extract_cites("见 [P1·§1·¶c404]。", index_two_papers) == []


# ═══════════ `_entries_from_cites`：`ref` 必须被拆开（否则下游全废）═══════════


def test_entries_from_cites_splits_ref():
    """`ref = "<stem>#<cid>"` 要拆成**裸 cid + 单独 pdf**，并保留 `p`。

    旧实现把整条 `ref` 塞进 `chunk_id` 且不产出 `pdf` → 后果有两个：
      · 锚点分支永远匹配不上（`"2408.09273#c9"` ≠ 锚点里的 `c9`）→ 修复后引用被清空；
      · `e.get("pdf")` 恒空 → 回退 `pdfs[0]` → 多篇下**所有 cite 标成第 1 篇**。
    """
    e = _entries_from_cites([{"n": None, "pdf": "2408.09273.pdf", "p": 2,
                              "chunk_id": "c9", "ref": "2408.09273#c9",
                              "evidence": "原文", "page": 3}])
    assert e[0]["chunk_id"] == "c9"
    assert e[0]["pdf"] == "2408.09273.pdf"
    assert e[0]["p"] == 2


def test_entries_from_cites_falls_back_to_ref():
    """老数据只有 `ref`（无 pdf/chunk_id）时也要能拆出来。"""
    e = _entries_from_cites([{"ref": "A#c7", "evidence": "原文", "page": 1}])
    assert e[0]["chunk_id"] == "c7"
    assert e[0]["pdf"] == "A"


# ═══════════ `_cites_from` 锚点通道（评审点 3）═══════════

# 形如 `_entries_from_cites` 的真实产物（带 p / pdf / 裸 cid）
_E_P = [{"kind": "chunk", "p": 1, "pdf": "A.pdf", "chunk_id": "c1", "text": "A 的 c1", "page": 1},
        {"kind": "chunk", "p": 2, "pdf": "B.pdf", "chunk_id": "c1", "text": "B 的 c1", "page": 1}]


def test_cites_from_anchor_matches_by_paper():
    """★ 评审点 3：锚点 `[P2…c1]` 必须命中 **B** 篇（旧实现丢 `P` → 取到 A）。"""
    got = _cites_from("见 [P2·§1·¶c1]", _E_P, "A.pdf")
    assert [(c["pdf"], c["chunk_id"]) for c in got] == [("B.pdf", "c1")]
    assert got[0]["p"] == 2


def test_cites_from_anchor_ambiguous_without_p():
    """拿不到「第几篇」**且** cid 有歧义 → **宁缺勿错**：返回空。

    给一条错的引用比不给更糟（用户会顺着跳到别的论文上）。
    """
    no_p = [{k: v for k, v in e.items() if k != "p"} for e in _E_P]
    assert _cites_from("见 [P2·§1·¶c1]", no_p, "A.pdf") == []


def test_cites_from_anchor_unique_cid_needs_no_p():
    """cid 在条目里唯一时，不需要「第几篇」也能安全命中。"""
    entries = [{"kind": "chunk", "pdf": "A.pdf", "chunk_id": "c9", "text": "A 的 c9", "page": 3}]
    got = _cites_from("见 [P1·§3·¶c9]", entries, "A.pdf")
    assert [(c["pdf"], c["chunk_id"]) for c in got] == [("A.pdf", "c9")]


def test_cites_from_anchor_is_recovered_and_carries_evidence():
    """锚点通道要带回 evidence / page / pdf（前端高亮与"出自哪一篇"都靠它）。"""
    got = _cites_from("结论 [P2·§3.3·¶c1]", _E_P, "A.pdf")
    assert got[0]["evidence"] == "B 的 c1" and got[0]["page"] == 1


# ═══════════ `_cites_from` 的 `[n]` 通道：也不能串篇 ═══════════


def test_bracket_n_channel_still_works():
    """`[n]` 通道（检索路径）不能被破坏。"""
    entries = [{"kind": "chunk", "pdf": "A.pdf", "chunk_id": "c1", "text": "A", "page": 1},
               {"kind": "chunk", "pdf": "B.pdf", "chunk_id": "c9", "text": "B", "page": 3}]
    got = _cites_from("见 [2]", entries, "A.pdf")
    assert [(c["n"], c["pdf"], c["chunk_id"]) for c in got] == [(2, "B.pdf", "c9")]


def test_n_channel_dedup_by_paper():
    """`[n]` 去重也要带篇：A、B 各有一条 `c1`，引 `[1][2]` 必须给**两条**。"""
    entries = [{"kind": "chunk", "pdf": "A.pdf", "chunk_id": "c1", "text": "A", "page": 1},
               {"kind": "chunk", "pdf": "B.pdf", "chunk_id": "c1", "text": "B", "page": 1}]
    got = _cites_from("见 [1][2]", entries, "A.pdf")
    assert [(c["n"], c["pdf"]) for c in got] == [(1, "A.pdf"), (2, "B.pdf")]


def test_claim_channel_dedup_by_paper():
    """claim 通道同理：同一个 `gid` 出现在两篇里 → 两条都要在。"""
    entries = [{"kind": "claim", "pdf": "A.pdf", "gid": "G1", "evidence": "a", "page": 1},
               {"kind": "claim", "pdf": "B.pdf", "gid": "G1", "evidence": "b", "page": 2}]
    got = _cites_from("见 [1][2]", entries, "A.pdf")
    assert [(c["n"], c["pdf"]) for c in got] == [(1, "A.pdf"), (2, "B.pdf")]


def test_two_channels_do_not_duplicate():
    """两条通道共用去重 → 同一个块不会计两次。"""
    entries = [{"kind": "chunk", "p": 1, "pdf": "A.pdf", "chunk_id": "c1",
                "text": "A 的 c1", "page": 1}]
    got = _cites_from("先 [1] 再 [P1·§1·¶c1]", entries, "A.pdf")
    assert len(got) == 1


def test_cites_from_needs_evidence():
    """没有 evidence 的条目不该变成引用（`_entries_from_cites` 的守卫）。"""
    assert _entries_from_cites([{"pdf": "A.pdf", "ref": "A#c1", "evidence": ""}]) == []
