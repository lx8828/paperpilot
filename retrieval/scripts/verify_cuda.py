"""验证 CUDA 真可用，并粗略量一下编码算力。

用途：换驱动 / 换 venv / 重装 torch 之后，一条命令确认 GPU 还能用。
（"能 import torch" 不等于 "能算" —— 所以这里真的跑一次矩阵乘并同步。）

用法：
    python retrieval/scripts/verify_cuda.py
"""
from __future__ import annotations

import time

import torch


def main() -> int:
    print(f"torch          : {torch.__version__}")
    print(f"cuda_available : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("→ CUDA 不可用：检查驱动；或按 retrieval/README.md 重装 CUDA 版 torch")
        return 1

    print(f"device         : {torch.cuda.get_device_name(0)}")
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 1024 ** 3
    print(f"capability     : sm_{p.major}{p.minor}   显存 {vram:.2f} GB")

    # 真实算一次（能 import 不等于能算）
    t0 = time.time()
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    _ = (a @ b).sum().item()
    torch.cuda.synchronize()
    print(f"\nfp32 4096x4096 矩阵乘 : {time.time() - t0:.2f}s")
    print(f"峰值显存              : {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    # 编码耗时外推（用朴素 FLOPs 估算，比合成线性层可信）
    print("\n[量级参考] 编码 64,183 篇 title+abstract（约 350 token/篇）")
    for name, params in (("BERT-base 量级 (~110M)", 110e6), ("bge-m3 量级 (~568M)", 568e6)):
        flops = 2 * params * 350 * 64183                    # 粗略：2·N·T
        for tflops in (25, 60):                             # 4050 laptop 的实际有效区间
            sec = flops / (tflops * 1e12)
            print(f"   {name:<24} @{tflops:>2} TFLOPS 有效 → 约 {sec / 60:5.1f} 分钟")
    print("   （加 tokenization / 数据加载开销，实际再乘 1.5~2）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
