"""论文数据源：**chunk 来源 + 能力声明**（生产代码只依赖本接口）。

## 为什么有这一层（2026-09-22 重构）

以前生产代码用**文件名前缀**猜能力 —— 一个 `is_qasper(pdf)` 被用来回答四个不同问题：

    有没有磁盘文件？  → 做不做内容指纹 / 跑不跑 MinerU / 会不会被摄取闸门拦
    chunk 从哪来？    → 读数据集构造，还是解析 PDF
    标题从哪来？      → 数据集自带 title，还是 PDF metadata
    产物怎么打标？    → `source` 字段写什么

结果是测试数据源（无 PDF 的文本集）**穿透了 7 个生产文件**（`pipeline` / `document_cache`
/ `worker` / `ingest` / `paper_identity` / `splitter` / 自带适配模块），每处都得写分支；
新增第二个测试源还要再改这 7 处 —— 而**产品目标是 arXiv PDF**，生产不该知道评测集存在。

现在：生产只问**接口能力**（`has_file` / `name` / `title()` / `chunks()`），具体源由
**使用方注册**：

    生产包 → 默认只有 PDF 路径（下面的 "默认路径"）
    测试侧 → `register(match, factory)` 注入自己的源，**生产包不 import 它**

## 默认路径（关键：零行为变化）

`resolve(pdf)` 返回 `None` 时，调用方走**原生 PDF 路径**（pymupdf / MinerU）。
也就是说：**没有注册任何源时，全链行为与引入本层之前逐字节一致**。
本模块**不实现** PDF 解析——那是 `document_cache` 的既有逻辑，此处不搬。

## 接口契约

    name         产物打标值（写进 claims.json / report 的 `source` 字段，整链同源校验用）
    has_file     有无磁盘文件身份 → 决定①内容指纹 ②MinerU ③摄取闸门
    title()      整篇标题（报告标题源）
    chunks()     报告视图 chunk（claims / report / cites 用它；**必须确定性可复现**）
    extra_chunks() 仅**检索视图**追加的块（如从别处补的表格/公式），默认 `[]`
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from paperpilot.models.schema import Chunk


@runtime_checkable
class PaperSource(Protocol):
    """一个"论文"的数据来源。生产代码只依赖本协议。"""

    name: str
    """产物打标值：写进 `claims.json` 的 `source`，用于"整链同源"校验。"""

    has_file: bool
    """有无磁盘文件身份。

    False → 不做内容指纹、不跑 MinerU、摄取闸门直接放行
    （这些机制的前提是"磁盘上有份文件"，无文件的源天然不适用）。
    """

    def title(self) -> str:
        """整篇标题（空串 = 调用方回退成文件名）。"""
        ...

    def chunks(self) -> list[Chunk]:
        """**报告视图** chunk（按正文顺序，已去前言/参考文献/空块）。

        必须**确定性**：同一篇每次调用 result 的 `chunk_id` 序列一致，
        否则 claims 锚点、向量缓存、cites 全部失效。
        """
        ...

    def extra_chunks(self) -> list[Chunk]:
        """仅**检索视图**追加的块（报告视图看不到它们）。

        典型用途：从另一份产物补表格/公式文本（表值只在图里时结构上不可达）。
        默认实现返回 `[]`。
        """
        ...


# ── 注册表 ───────────────────────────────────────────────────────────────────
#
# 后注册的优先（`insert(0, ...)`）→ 使用方可覆盖已有源的判定。
# 生产包**不注册任何源** —— 注册动作全部发生在测试脚本 / 评测入口。

_MATCHERS: list[tuple[Callable[[str], bool], Callable[[str], PaperSource]]] = []


def register(match: Callable[[str], bool],
             factory: Callable[[str], PaperSource]) -> None:
    """注册数据源：`match(pdf)` 为真时，用 `factory(pdf)` 造源。

    只在**使用方**（测试脚本 / 评测入口）调用；生产包不调用。
    """
    _MATCHERS.insert(0, (match, factory))


def unregister_all() -> None:
    """清空注册表（测试用：避免用例间互相污染）。"""
    _MATCHERS.clear()


def registered_names() -> list[str]:
    """已注册源的类名（诊断/测试用，不触发实例化）。"""
    return [getattr(f, "__name__", repr(f)) for _, f in _MATCHERS]


def resolve(pdf: str) -> PaperSource | None:
    """解析该论文的数据源；**无匹配 → None（调用方走原生 PDF 路径）**。"""
    for match, factory in _MATCHERS:
        try:
            if match(pdf):
                return factory(pdf)
        except Exception:  # noqa: BLE001      坏 matcher 不应阻断整链
            continue
    return None


def has_file(pdf: str) -> bool:
    """该源有无磁盘文件身份（决定：内容指纹 / MinerU / 摄取闸门）。

    无匹配 → True（默认 PDF 路径必然有文件）。
    """
    src = resolve(pdf)
    return True if src is None else bool(src.has_file)


def source_name(pdf: str) -> str | None:
    """该源自称的名字；无匹配 → None（调用方自行判定 mineru/pymupdf）。"""
    src = resolve(pdf)
    return None if src is None else str(src.name)
