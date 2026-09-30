#!/usr/bin/env python3
"""
诊断分布式通信后端及 NVLink 使用情况
要求：在 Ray 环境中运行（或使用 torchrun）
"""

import torch
import torch.distributed as dist
import os
import time
import subprocess
import sys

def print_nvlink_status():
    try:
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "-c"],
            capture_output=True,
            text=True,
            check=True
        )
        print("=== nvidia-smi nvlink -c ===")
        print(result.stdout)
    except Exception as e:
        print(f"无法运行 nvidia-smi: {e}")

def check_p2p():
    print("=== P2P 检查 ===")
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        for j in range(device_count):
            if i != j:
                can = torch.cuda.can_device_access_peer(i, j)
                print(f"GPU {i} -> GPU {j}: {'支持' if can else '不支持'} P2P")

def diagnose():
    # 初始化进程组（如果尚未初始化）
    if not dist.is_initialized():
        # 尝试从环境变量获取后端（Ray 可能注入）
        backend = os.environ.get("RAY_DISTRIBUTED_BACKEND", "nccl")
        print(f"初始化进程组，后端: {backend}")
        dist.init_process_group(backend=backend, init_method='env://')
    else:
        print("进程组已初始化")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size}, backend: {dist.get_backend()}")

    if dist.get_backend() != "nccl":
        print("警告：当前后端不是 NCCL，可能走 PCIe 或 Gloo，带宽受限！")
    else:
        # 检查 NCCL 环境变量
        print("NCCL 环境变量:")
        for k in ["NCCL_DEBUG", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME"]:
            v = os.environ.get(k, "(未设置)")
            print(f"  {k}={v}")

    # 检查 P2P
    check_p2p()

    # 测试带宽
    size_gb = 1.0  # 用小数据快速测试
    elem_count = int(size_gb * 1024**3 / 4)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    tensor = torch.randn(elem_count, device=device)
    gathered = [torch.zeros(elem_count, device=device) for _ in range(world_size)]

    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    dist.all_gather(gathered, tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if rank == 0:
        total_data = size_gb * world_size  # 所有 rank 数据总量
        bandwidth = total_data / elapsed
        print(f"\n测试带宽: {bandwidth:.2f} GB/s (使用 {size_gb}GB 数据, world_size={world_size})")
        print("如果带宽低于 100 GB/s，很可能未使用 NVLink 或使用 PCIe。")
        print("若使用 NCCL 且带宽低，检查 NVLink 是否开启，或环境变量 NCCL_P2P_DISABLE=0。")

    # 打印 NVLink 状态
    if rank == 0:
        print_nvlink_status()

if __name__ == "__main__":
    diagnose()
