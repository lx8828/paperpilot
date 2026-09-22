"""一次性编排：**抓取 → 追加校验 → 建索引（向量编码 + BM25）**。

背景：arXiv 侧要从「cs 近 3 个月（61,426 篇）」扩到「created ≥ 2024-01（目标 ~540,000 篇）」。
三步的资源完全不同，串行跑最省事：
    ① 抓取     吃网络（3s 限速），不吃 GPU/内存
    ② 向量编码 吃 GPU（bge-m3 fp16，实测 60 篇/s），不吃网络
    ③ 建 BM25  吃内存（CountVectorizer 峰值 ~4 GB），不吃 GPU/网络

──────────────────────── 关键设计：**追加语义** ────────────────────────
`fetch_arxiv.py` 按「月」切片、**近→远**遍历，且已有分片直接读缓存。
所以把 `--months` 从 3 扩到 34：前 3 个月会**逐条重放**（顺序完全一致），
新抓的月份**追加在后面** → 已编码的 docid `0..61425` **保持不变**。

→ 因此向量编码可以**增量**：`build_arxiv_index.py` 会跳过行数已正确的 part 文件
  （part 0..11 是满 5000 行 → 跳过；part 12 只有 1426 行 → 重算）。
  省掉 6 万篇的重复编码（约 17 分钟）。

⚠️ 这个增量正确性依赖「重放一致」。所以本脚本在抓取后**先校验**：
   把备份元数据与原元数据的 `arxiv_id` 序列逐条比对。
   - 一致 → 追加安全，走增量编码
   - 不一致 → **转全量重建**（删掉 emb/ 与 bm25/，从头编码），避免行列错位
     这种静默 bug 会污染之后所有实验。

用法：
    python retrieval/scripts/expand_arxiv_corpus.py --target 540000 --months 34
    python retrieval/scripts/expand_arxiv_corpus.py --skip-fetch          # 只重建索引
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]          # retrieval/
ROOT = HERE.parent                                  # paperpilot/
SCRIPTS = HERE / "scripts"
ARXIV = HERE / "data" / "arxiv"
SRC = ARXIV / "meta" / "arxiv_meta.parquet"
BACKUP = ARXIV / "meta" / "arxiv_meta.prev.parquet"
EMB, BM25DIR = ARXIV / "emb", ARXIV / "bm25"


def now() -> str:
    return time.strftime("%H:%M:%S")


def run(tag: str, argv: list[str]) -> int:
    print(f"\n{'=' * 92}\n[{now()}] ▶ {tag}\n  $ {' '.join(argv)}\n{'=' * 92}", flush=True)
    t0 = time.time()
    p = subprocess.run(argv, cwd=str(ROOT),
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    print(f"\n[{now()}] ✔ {tag} 结束 | exit={p.returncode} | "
          f"用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    return p.returncode


def verify_raw_shards() -> int:
    """校验 raw/*.xml.gz 能解压。

    ⚠️ 必要性：抓取是「一页写一个 gz」，**进程被中断可能留下截断的文件**
    （`write_bytes` 写一半被杀）。续跑时该分片会被当作"已缓存"直接读 →
    `gzip.decompress` 抛异常 → 整个续跑崩掉，而且崩在最前面。
    所以续跑前先体检，坏的直接删掉，让 fetch 重新下载那一页。
    """
    raw = ARXIV / "raw"
    if not raw.exists():
        return 0
    files = sorted(raw.glob("*.xml.gz"))
    bad: list[str] = []
    for f in files:
        try:
            gzip.decompress(f.read_bytes())
        except Exception as e:  # noqa: BLE001
            print(f"[{now()}] ⚠ 分片损坏 → 删除待重抓：{f.name}（{type(e).__name__}）",
                  flush=True)
            f.unlink(missing_ok=True)
            bad.append(f.name)
    print(f"[{now()}] raw 分片体检：{len(files) - len(bad)} 个完好"
          + (f"，**删除 {len(bad)} 个损坏**" if bad else "，无损坏"), flush=True)
    return len(bad)


def verify_append_safe() -> bool:
    """比对备份与新元数据的 arxiv_id 序列，判断既有 docid 是否保持不变。"""
    if not BACKUP.exists():
        print(f"[{now()}] ⚠ 无备份 {BACKUP.name} → 无法校验，按**全量重建**处理")
        return False
    prev = pd.read_parquet(BACKUP, columns=["arxiv_id"])
    new = pd.read_parquet(SRC, columns=["arxiv_id"])
    n = min(len(prev), len(new))
    if n == 0:
        print(f"[{now()}] ⚠ 元数据为空 → 按全量重建处理")
        return False
    a = prev["arxiv_id"].to_numpy()[:n]
    b = new["arxiv_id"].to_numpy()[:n]
    diff = int((a != b).sum())
    print(f"\n[{now()}] 【追加校验】旧 {len(prev):,} 篇 vs 新 {len(new):,} 篇；"
          f"比对前 {n:,} 条 → 不一致 {diff} 条")
    if diff == 0 and len(new) > len(prev):
        print(f"[{now()}] ✔ 前缀逐条一致 → **追加安全**，走增量编码"
              f"（省掉 {len(prev):,} 篇的重复编码）")
        return True
    print(f"[{now()}] ✘ 前缀不一致或未增长 → **转全量重建**")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=540_000)
    ap.add_argument("--months", type=int, default=34,
                    help="向前覆盖多少个月。created>=2024-01 → 需要 33 个月（2024-01..2026-09）")
    ap.add_argument("--min-created", default="2024-01-01")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    py = sys.executable
    print(f"[{now()}] 编排开始 | 目标 {args.target:,} 篇 | 覆盖 {args.months} 个月 | "
          f"created >= {args.min_created}", flush=True)

    # ── 备份（只在第一次跑时做，保护“追加校验”的参照物）────────────
    if SRC.exists() and not BACKUP.exists():
        shutil.copy2(SRC, BACKUP)
        n0 = len(pd.read_parquet(BACKUP, columns=["arxiv_id"]))
        print(f"[{now()}] 已备份现有元数据 → {BACKUP.name}（{n0:,} 篇），用作校验基准",
              flush=True)
    elif BACKUP.exists():
        n0 = len(pd.read_parquet(BACKUP, columns=["arxiv_id"]))
        print(f"[{now()}] 备份已存在（{n0:,} 篇），沿用", flush=True)

    # ── ⓪ 续跑前体检：中断可能留下截断的分片 ────────────────────────
    verify_raw_shards()

    # ── ① 抓取 ────────────────────────────────────────────────────
    if not args.skip_fetch:
        rc = run("① 抓取 arXiv 元数据（OAI-PMH）",
                 [py, str(SCRIPTS / "fetch_arxiv.py"),
                  "--target", str(args.target), "--months", str(args.months),
                  "--min-created", args.min_created])
        if rc != 0:
            print(f"[{now()}] ✘ 抓取失败（exit={rc}）→ 中止。已抓到的分片都在 raw/ 里，"
                  f"重跑会自动续抓。", flush=True)
            return rc

    if not SRC.exists():
        print(f"[{now()}] ✘ 缺 {SRC}", flush=True)
        return 2

    # ── ② 追加校验 → 决定增量还是全量 ─────────────────────────────
    safe = (not args.force_rebuild) and verify_append_safe()
    if not safe:
        for d in (EMB, BM25DIR):
            if d.exists():
                shutil.rmtree(d)
                print(f"[{now()}] 已删除 {d.relative_to(HERE)}（强制全量重建）", flush=True)
    else:
        print(f"[{now()}] 保留已有 emb/ → 编码将增量跳过已完成的 part", flush=True)

    # ── ③ 向量编码 + BM25（同一脚本，顺序执行）────────────────────
    rc = run("②+③ 建索引（bge-m3 向量编码 → BM25）",
             [py, str(SCRIPTS / "build_arxiv_index.py")])
    if rc != 0:
        print(f"[{now()}] ✘ 建索引失败（exit={rc}）。向量编码支持断点续跑，"
              f"重跑会跳过已完成的 part。", flush=True)
        return rc

    # ── 汇总 ─────────────────────────────────────────────────────
    d = pd.read_parquet(ARXIV / "meta" / "corpus_text.parquet",
                        columns=["docid", "created", "n_chars"])
    print(f"\n{'=' * 92}\n[{now()}] ✅ 全部完成 | 总用时 {(time.time() - t_all) / 60:.1f} 分钟\n"
          f"  语料：{len(d):,} 篇\n"
          f"  年份：{d['created'].str[:4].value_counts().sort_index().to_dict()}\n"
          f"  文本长度：p50={int(d['n_chars'].median())} p90={int(d['n_chars'].quantile(0.9))}\n"
          f"{'=' * 92}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
