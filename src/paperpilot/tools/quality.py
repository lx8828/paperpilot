"""Claim 质量筛查：找出表格/数字碎片等噪声句（LLM 判定，分批调用）。

防误杀策略（单次 LLM 二值判定不稳定，曾出现整篇 60% 误伤）：
- 每批独立判定 ROUNDS 轮，只保留**每轮都被判定为噪声**的 claim（交集），宁漏勿杀。
- 若某篇噪声比例超过阈值，视为"疑似误判"，调用方应降级为人工待查而非自动剔除。
"""
from __future__ import annotations

from typing import Any

from paperpilot.prompts.quality import QUALITY_SYSTEM, build_user
from paperpilot.tools import llm

CHUNK = 100    # 每批最大 claim 数
ROUNDS = 2     # 判定轮数（取交集）
ABNORMAL = 0.15  # 噪声占比超过此值判定为异常（疑似误杀）


def _call_batch(batch: list[dict[str, Any]]) -> set[str]:
    rows = None
    for attempt in range(3):
        try:
            rows = llm.chat_json(QUALITY_SYSTEM, build_user(batch),
                                 temperature=0.0)
            break
        except (llm.LLMError, ValueError) as e:
            if attempt < 2:
                print(f"  [quality] 调用失败，重试: {e}")
            else:
                print(f"  [quality] 重试仍失败: {e}")
    if not isinstance(rows, list):
        return set()
    return {str(r.get("claim_id", "")).strip() for r in rows if isinstance(r, dict)}


def scan_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """claims: [{claim_id, type, text, title_path}]。返回(保守的) noisy 列表。"""
    claim_map = {c["claim_id"]: c for c in claims}
    noisy_ids: set[str] = set()
    for i in range(0, len(claims), CHUNK):
        batch = claims[i:i + CHUNK]
        votes = [_call_batch(batch) for _ in range(ROUNDS)]
        common = votes[0].intersection(*votes[1:]) if len(votes) > 1 else votes[0]
        noisy_ids |= common
    return [{"claim_id": cid,
             "reason": "多轮判定一致：疑似表格/数字碎片等读不通句"}
            for cid in sorted(noisy_ids) if cid in claim_map]


def is_abnormal(n_noisy: int, n_total: int) -> bool:
    """噪声占比异常（疑似误杀）时返回 True，应人工复核而非自动剔除。"""
    return n_total > 0 and n_noisy / n_total > ABNORMAL
