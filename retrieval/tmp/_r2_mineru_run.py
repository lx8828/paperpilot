"""**Step 2a｜MinerU 解析（生产口径）· 可续跑 + 进度落盘**

## 为什么必须用 MinerU
生产默认 `PAPERPILOT_MINERU=1` / `PAPERPILOT_USE_MINERU=1`：
  · `ordered_chunks` = MinerU 骨架（表格结构化、公式 LaTeX、图内文字不污染正文）
  · `retrieval_chunks` = 再按页注入 MinerU 的 **table/equation** → 表值可被召回/作答
pymupdf 单独用会**丢表格数值**（仓库记录：pymupdf 把表撕碎）→ 那不是我想要的对照口径。

## 幂等 / 续跑
· `ingest.run_mineru()` 自带复用判定：产物目录存在 + MinerU 版本一致 + PDF 指纹一致 → `reused`
· 本脚本另写 `mineru_progress.json`（每篇一条），**中断后重跑自动跳过已完成的**
· 日志：`retrieval/tmp/_mineru_run.log`

用法：
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_mineru_run.py --limit 3     # 冒烟
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_mineru_run.py              # 全 47（约 1~2.5h）
  ./.venv/Scripts/python.exe -u retrieval/tmp/_r2_mineru_run.py --status     # 只看进度
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "retrieval"
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["PAPERPILOT_MINERU"] = "1"          # 真的跑 MinerU
os.environ["PAPERPILOT_USE_MINERU"] = "1"

from paperpilot import ingest  # noqa: E402

PROG = HERE / "data" / "r2dev" / "mineru_progress.json"     # 可被 `--progress` 覆盖


def load_prog() -> dict:
    return json.loads(PROG.read_text(encoding="utf-8")) if PROG.exists() else {}


def set_prog(p: Path) -> None:
    """⚠️ `PROG` 是模块级全局（`one()` 里要写它）→ 必须 `global` 覆盖，改局部无效。"""
    global PROG
    PROG = p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=2, help="并发进程数（6GB 显存建议 2）")
    ap.add_argument("--map", default="",
                    help="PDF 映射文件名（默认 `pdf_map.json`；扩语料用 `corpus50/pdf_map_all.json`）")
    ap.add_argument("--progress", default="",
                    help="进度文件名（默认 `mineru_progress.json`；扩语料用 `mineru_progress50.json`）")
    args = ap.parse_args()

    dev_dir = HERE / "data" / "r2dev"
    mpath = dev_dir / (args.map or "pdf_map.json")
    if args.progress:
        set_prog(dev_dir / args.progress)
    pmap = json.loads(mpath.read_text(encoding="utf-8"))
    if "cluster" in next(iter(pmap.values())):
        # 扩语料的合并映射自带 cluster（150 篇）
        items = [(d, v["pdf"], int(v["cluster"])) for d, v in pmap.items() if v.get("ok")]
    else:
        arx = json.loads((dev_dir / "arxiv_map.json").read_text(encoding="utf-8"))
        items = [(d, v["pdf"], int(arx[d]["cluster"])) for d, v in pmap.items() if v["ok"]]
    items.sort(key=lambda x: (x[2], x[0]))
    prog = load_prog()
    print(f"  [映射] {mpath.name} → {len(items)} 篇 ｜ 进度文件 {PROG.name}")

    done = [p for _, p, _ in items if (prog.get(p) or {}).get("status") in ("ok", "reused")]
    print("=" * 106)
    print(f"【MinerU 解析】待跑 {len(items)} 篇 ｜ 已完成 {len(done)} ｜ "
          f"MinerU {ingest.mineru_version()} ｜ backend {ingest.MINERU_BACKEND}")
    print(f"  进度文件：{PROG.name} ｜ MINERU_OUT：{ingest.MINERU_OUT}")

    if args.status:
        print(f"\n  {'pdf':<18}{'簇':>3}{'状态':<10}{'秒':>7}  原因")
        for d, pdf, cl in items:
            r = prog.get(pdf) or {}
            if r:
                print(f"  {pdf:<18}{cl:>3}{r.get('status', ''):<10}"
                      f"{r.get('seconds', 0):>7.0f}  {str(r.get('reason', ''))[:44]}")
        print(f"\n  合计：完成 {len(done)}/{len(items)}")
        return 0

    todo = [(d, pdf, cl) for d, pdf, cl in items
            if args.force or (prog.get(pdf) or {}).get("status") not in ("ok", "reused")]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("\n  ✅ 全部已完成（无需重跑）")
        return 0
    print(f"  本轮跑 **{len(todo)}** 篇\n")

    t00 = time.time()
    lock = threading.Lock()
    n_done = {"n": 0}

    def one(job: tuple[str, str, int]) -> None:
        docid, pdf, cl = job
        t0 = time.time()
        try:
            r = ingest.run_mineru(pdf, force=args.force)
        except Exception as e:  # noqa: BLE001
            r = {"status": "failed", "reason": f"{type(e).__name__}: {str(e)[:90]}", "seconds": 0.0}
        sec = time.time() - t0
        with lock:
            prog[pdf] = {**r, "docid": docid, "cluster": cl, "seconds": round(sec, 1),
                         "ts": datetime.now().isoformat(timespec="seconds")}
            PROG.write_text(json.dumps(prog, ensure_ascii=False, indent=1), encoding="utf-8")
            n_done["n"] += 1
            i, el = n_done["n"], time.time() - t00
            eta = el / i * (len(todo) - i)
            flag = {"ok": "✅", "reused": "↩︎", "skipped": "⏭️"}.get(r["status"], "❌")
            print(f"  [{i}/{len(todo)}] {flag} {pdf:<18}簇{cl} {sec:>6.0f}s "
                  f"｜ 已用 {el / 60:.1f}min ｜ ETA {eta / 60:.0f}min ｜ "
                  f"{str(r.get('reason', ''))[:36]}", flush=True)

    w = max(1, args.workers)
    print(f"  并发 {w} 进程（每进程显存 ≈1.9GB）\n")
    if w == 1:
        for job in todo:
            one(job)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=w) as ex:
            list(ex.map(one, todo))

    n_ok = sum(1 for _, p, _ in items if (prog.get(p) or {}).get("status") in ("ok", "reused"))
    print(f"\n{'=' * 106}\n【完成】**{n_ok}/{len(items)}** 篇 ｜ 耗时 {(time.time() - t00) / 60:.1f} 分钟")
    for d, pdf, cl in items:
        r = prog.get(pdf) or {}
        if r.get("status") not in ("ok", "reused"):
            print(f"  ❌ {pdf:<18}簇{cl} {str(r.get('reason', '未跑'))[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
