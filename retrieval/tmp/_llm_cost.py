"""**LLM 成本核算**：模型可用性 + 按实测 token 数推算各方案成本。

## 实测锚点（来自本项目真实运行）
- 产品链路（148 题重跑，2026-09-27）：**¥0.026/题**，均 prompt 22,479 / completion 1,956，均 4.8 次调用
- 判定调用（本机校准实测）：**≈ 567~620 prompt token/次**（system ~350 + 论断 ~20 + 证据 ~250）
- 前缀缓存：语料/系统提示放前面 → 命中价约为未命中的 **1/5**

用法：./.venv/Scripts/python.exe -u retrieval/tmp/_llm_cost.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from paperpilot.tools import llm  # noqa: E402

llm._load_dotenv(str(ROOT))

# **官方定价**（api-docs.deepseek.com/quick_start/pricing，2026-09-29 抓取）
# 单位：¥/百万 token；美元按 7.1 汇率折算；低峰 = 高峰半价
USD = 7.1
PRICE = {
    # deepseek-flash = DeepSeek-V4.1-Flash（官方推荐名）
    "deepseek-flash": dict(hit=0.003 * USD, miss=0.15 * USD, out=0.60 * USD),
    # 高峰：miss 0.30 / out 1.20 / hit 0.006
    "deepseek-v4-pro": dict(hit=0.022 * USD, miss=0.66 * USD, out=1.98 * USD),
    "GLM-4-Flash(裁判·免费)": dict(hit=0.0, miss=0.0, out=0.0),
    "GLM-4.7-Flash(免费·200K)": dict(hit=0.0, miss=0.0, out=0.0),
}

CANDIDATE_MODELS = ["deepseek-chat", "deepseek-reasoner",
                    "deepseek-flash", "deepseek-v4-flash",
                    "deepseek-v4-pro", "glm-4-flash"]


def cost_of(p_in_miss: int, p_in_hit: int, p_out: int, price: dict) -> float:
    return (p_in_miss * price["miss"] + p_in_hit * price["hit"] + p_out * price["out"]) / 1e6


def main() -> int:
    print("【1】模型名有效性（各 1 次最小调用）")
    for m in CANDIDATE_MODELS:
        import os
        saved = os.environ.get("PAPERPILOT_LLM_MODEL")
        os.environ["PAPERPILOT_LLM_MODEL"] = m
        try:
            r = llm.chat_json("只输出 JSON。", "返回 {\"ok\": 1}", temperature=0, max_tokens=30)
            print(f"   {m:<22} ✅ 可用  {r}")
        except Exception as e:  # noqa: BLE001
            print(f"   {m:<22} ❌ {type(e).__name__}: {str(e)[:70]}")
        finally:
            if saved is None:
                os.environ.pop("PAPERPILOT_LLM_MODEL", None)
            else:
                os.environ["PAPERPILOT_LLM_MODEL"] = saved
        time.sleep(0.5)

    # 各方案 token 预算（按本项目实测的 token/字符比：英文约 4 字符/token）
    PASSAGE = 130            # 维基段落 ~560 字符
    CHUNK = 250              # 论文块 1000 字符
    SYS = 350                # 判定 system prompt
    FEWSHOT = 2000           # LoFT 风格 few-shot（可缓存）

    plans = {
        "① 直读·128k 档（105k 语料全塞）": dict(miss=1200, hit=104_000, out=60),
        "② 直读·1m 档（844k 语料全塞）": dict(miss=1200, hit=843_000, out=60),
        "②' 直读·1m 档（**无缓存**）": dict(miss=843_000, hit=0, out=60),
        "③ 检索 top-20 段落 + 1 次 reader": dict(miss=20 * PASSAGE, hit=FEWSHOT, out=60),
        "④ 逐篇判定 20 篇 × b=3 块 + 聚合": dict(miss=20 * (SYS + CHUNK * 3) + 1500, hit=20 * SYS, out=1_100),
        "⑤ 两阶段（GLM 免费初筛 + DS 终判 10 篇）": dict(miss=10 * (SYS + CHUNK * 2) + 1200, hit=10 * SYS, out=600),
        "⑥ 现有产品链路（实测）": dict(miss=22_479, hit=0, out=1_956),
    }

    print("\n【2】各方案成本（官方价·低峰；命中价 = 未命中的 1/50）")
    print(f"  {'方案':<34}{'未命中in':>9}{'命中in':>9}{'out':>7}"
          f"{'flash ¥/题':>11}{'pro ¥/题':>10}{'免费模型':>9}")
    for nm, p in plans.items():
        c_f = cost_of(p["miss"], p["hit"], p["out"], PRICE["deepseek-flash"])
        c_p = cost_of(p["miss"], p["hit"], p["out"], PRICE["deepseek-v4-pro"])
        c_g = cost_of(p["miss"], p["hit"], p["out"], PRICE["GLM-4-Flash(裁判·免费)"])
        print(f"  {nm:<34}{p['miss']:>9,}{p['hit']:>9,}{p['out']:>7,}"
              f"{c_f:>11.4f}{c_p:>10.3f}{c_g:>9.4f}")

    print("\n【3】批量成本（flash·低峰）")
    for nm, p in plans.items():
        c = cost_of(p["miss"], p["hit"], p["out"], PRICE["deepseek-flash"])
        print(f"  {nm:<34} 100 题 ¥{c * 100:>7.2f} ｜ 1,000 题 ¥{c * 1000:>8.2f}"
              f" ｜ 10,000 题 ¥{c * 10000:>9.1f}")

    print("\n【3b】QAMPARI 128k 档一次完整评测（100 题 test）")
    p3 = plans["③ 检索 top-20 段落 + 1 次 reader"]
    p1 = plans["① 直读·128k 档（105k 语料全塞）"]
    for nm, p in (("检索路线", p3), ("直读路线", p1)):
        c = cost_of(p["miss"], p["hit"], p["out"], PRICE["deepseek-flash"])
        # 直读：语料前缀只在第一题未命中，其余命中
        if nm == "直读路线":
            c_total = (104_000 * PRICE["deepseek-flash"]["miss"] + 99 * 104_000 * PRICE["deepseek-flash"]["hit"]
                       + 100 * (1_200 * PRICE["deepseek-flash"]["miss"] + 60 * PRICE["deepseek-flash"]["out"])) / 1e6
        else:
            c_total = c * 100
        print(f"  {nm:<8} 100 题合计 **¥{c_total:.2f}**（均 ¥{c_total / 100:.4f}/题）")

    print("\n【4】真实调用过的用量（本机本次会话累计）")
    u = llm.usage_stats()
    print(f"  {u}")
    print(f"  → 折算 v4-flash：¥{(u['cache_miss_tokens'] * 1.0 + u['cache_hit_tokens'] * 0.2 + u['completion_tokens'] * 2.0) / 1e6:.3f}")

    print("""
【读法】
· 最贵的不是"判定"，是"把语料塞进输入"（128k 档全塞 vs 检索 top-20 差 20~70 倍输入量）
· 前缀缓存是最大杠杆：语料/系统提示放 prompt 前面 → 命中价 ≈ 未命中的 1/5
· GLM-4-Flash 官方免费 → 可当**逐篇初筛**，只在终判/疑难用 DeepSeek
· 结论：按 v4-flash 价，"逐篇 LLM 判定"≈ ¥0.02/题，与现有产品链路（¥0.026/题）同量级
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
