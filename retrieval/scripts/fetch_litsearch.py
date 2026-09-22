"""下载 LitSearch（princeton-nlp/LitSearch）到 retrieval/data/litsearch/。

LitSearch（EMNLP 2024, Princeton NLP）：597 条真实文献检索查询 + 6 万+ 篇近期
ML/NLP 论文语料。本脚本只负责**下载**，不改数据；解析与评测在别的脚本里。

保持 HuggingFace 仓库的原始目录结构，便于核对与复用：

    retrieval/data/litsearch/
      README.md
      query/full-00000-of-00001.parquet                 0.1 MB
      corpus_clean/full-0000{0..5}-of-00006.parquet     ~1.26 GB
      corpus_s2orc/full-0000{0..7}-of-00008.parquet     ~1.60 GB

用法：
    python retrieval/scripts/fetch_litsearch.py query          # 只下查询（0.1MB，先跑这个）
    python retrieval/scripts/fetch_litsearch.py corpus_clean   # 语料（1.26GB）
    python retrieval/scripts/fetch_litsearch.py corpus_s2orc   # 带元数据的语料（1.60GB）
    python retrieval/scripts/fetch_litsearch.py all            # 全部（~2.9GB）

特性：
  · 幂等：文件已存在且体积一致 → 跳过（可中断后重跑）
  · 断点安全：先写 .part 再改名，避免半成品冒充完整文件
  · 代理：尊重 HTTPS_PROXY / HTTP_PROXY 环境变量（本机需开梯子）
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = "princeton-nlp/LitSearch"
API = f"https://huggingface.co/api/datasets/{REPO}"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
HERE = Path(__file__).resolve().parents[1]          # retrieval/
DEST = HERE / "data" / "litsearch"

CONFIGS = ("query", "corpus_clean", "corpus_s2orc")


def list_files() -> list[str]:
    """从 HF API 取仓库文件清单（比硬编码更耐版本变化）。"""
    with urllib.request.urlopen(API, timeout=30) as r:
        d = json.loads(r.read().decode("utf-8"))
    return [s["rfilename"] for s in (d.get("siblings") or [])]


def head_size(name: str) -> int:
    """远端文件体积（HEAD，不下载）。失败返回 -1。"""
    try:
        req = urllib.request.Request(RAW + name, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:  # noqa: BLE001  体积探测失败不该阻断下载
        return -1


def download(name: str, expect: int) -> str:
    """下载单个文件；返回 'skip' / 'ok' / 'fail'。"""
    out = DEST / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and (expect <= 0 or out.stat().st_size == expect):
        print(f"  [skip] {name}  ({out.stat().st_size / 1e6:.1f} MB 已存在)")
        return "skip"

    tmp = out.with_suffix(out.suffix + ".part")
    got = 0
    try:
        req = urllib.request.Request(RAW + name)
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0) or expect
            while True:
                buf = r.read(1 << 20)                   # 1MB
                if not buf:
                    break
                f.write(buf)
                got += len(buf)
                if total > 0:
                    sys.stdout.write(
                        f"\r  [get ] {name}  {got / 1e6:7.1f} / {total / 1e6:.1f} MB "
                        f"({100 * got / total:5.1f}%)")
                    sys.stdout.flush()
    except Exception as e:  # noqa: BLE001
        print(f"\r  [fail] {name}  {type(e).__name__}: {str(e)[:140]}")
        tmp.unlink(missing_ok=True)
        return "fail"

    print()
    if total and got != total:
        print(f"  [fail] {name}  体积不符（{got} != {total}）→ 已丢弃")
        tmp.unlink(missing_ok=True)
        return "fail"
    tmp.replace(out)
    print(f"  [ok  ] {name}  {got / 1e6:.1f} MB")
    return "ok"


def main() -> int:
    args = [a for a in sys.argv[1:] if a]
    if not args:
        print(__doc__)
        return 2
    want = list(CONFIGS) if args[0] == "all" else args
    bad = [w for w in want if w not in CONFIGS]
    if bad:
        print(f"未知配置 {bad}；可选：{list(CONFIGS)} 或 all")
        return 2

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""
    print(f"目标目录: {DEST}")
    print(f"代理    : {proxy or '(未设置，若失败请开梯子并设置 HTTPS_PROXY)'}")

    files = list_files()
    picked = [f for f in files if f.split("/")[0] in want or f in ("README.md",)]
    picked = [f for f in picked if not f.startswith(".")]

    # 预取体积（用于跳过判断与总量提示）
    plan: list[tuple[str, int, str]] = []
    total = 0
    for f in picked:
        n = head_size(f)
        plan.append((f, n, f.split("/")[0]))
        total += max(n, 0)
    print(f"\n待处理 {len(plan)} 个文件，合计约 {total / 1e6:.1f} MB\n")

    ok = skip = fail = 0
    for f, n, _cfg in plan:
        st = download(f, n)
        ok += st == "ok"
        skip += st == "skip"
        fail += st == "fail"

    print(f"\n完成：新下 {ok}，已存在 {skip}，失败 {fail}")
    if fail:
        print("（失败的多为网络波动，直接重跑本脚本即可续传）")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
