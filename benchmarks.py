import argparse
import ctypes
import os
import re
import sys
import tempfile

import torch
from flash_attn.flash_attn_interface import flash_attn_func

# e.g. python3 benchmarks.py --seq_lens 512 1024 2048 4096 8192 


def run_flash_attn_and_capture_log(q, k, v, softmax_scale):
    libc = ctypes.CDLL(None)
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    with tempfile.TemporaryFile(mode="w+b") as tmp:
        try:
            libc.fflush(None)
            os.dup2(tmp.fileno(), stdout_fd)
            os.dup2(tmp.fileno(), stderr_fd)

            out = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                # 其余参数使用默认值：dropout_p=0.0, causal=False, window_size=(-1, -1),
                # softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False
            )
            torch.cuda.synchronize()
            libc.fflush(None)
        finally:
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

        tmp.seek(0)
        kernel_log = tmp.read().decode("utf-8", errors="ignore")

    return out, kernel_log


def parse_args():
    parser = argparse.ArgumentParser(description="FlashAttention benchmark script")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[2048], help="seq len list")
    parser.add_argument("--num_heads", type=int, default=32, help="head num")
    parser.add_argument("--head_dim", type=int, default=128, help="head dim")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16", help="dtype")
    return parser.parse_args()


def demo_flash_attn(batch_size, seq_lens, num_heads, head_dim, dtype):
    for seq_len in seq_lens:
        print(f"Testing FlashAttention with [{batch_size}, {seq_len}, {num_heads}, {head_dim}]:")
        # 创建随机 Q/K/V，形状 (batch, seqlen, nheads, headdim)
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype, requires_grad=False)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype, requires_grad=False)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype, requires_grad=False)

        # 显式启用 softmax_scale，通常设置为 1 / sqrt(D)
        softmax_scale = 1.0 / (head_dim ** 0.5)

        # 直接使用 flash_attn_func 内部 printf 打印的 runtime（单位：ms），取 10 次平均
        num_runs = 10
        runtimes = []
        for _ in range(num_runs):
            out, kernel_log = run_flash_attn_and_capture_log(q, k, v, softmax_scale)
            kernel_log = kernel_log.strip()
            matches = re.findall(r"runtime\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", kernel_log)
            if not matches:
                print("Warning: failed to parse runtime from kernel log, skip TFLOPS calc.")
                break
            runtimes.append(float(matches[-1]))

        if len(runtimes) != num_runs:
            continue
        runtime_ms = sum(runtimes) / num_runs

        # flops = 2 * B * H * L * L * D
        flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        tflops = flops / (runtime_ms * 1e-3) / 1e12

        print(f"[avg over {num_runs} runs: [{runtime_ms:.2f}, {tflops:.2f}]")

        # 防止编译器/运行时优化掉 out
        _ = out



if __name__ == "__main__":
    args = parse_args()
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"Running FlashAttention benchmark with batch_size={args.batch_size}, seq_lens={args.seq_lens}, "
          f"num_heads={args.num_heads}, head_dim={args.head_dim}, dtype={args.dtype}")
    demo_flash_attn(
        batch_size=args.batch_size,
        seq_lens=args.seq_lens,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dtype=torch_dtype,
    )