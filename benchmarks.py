import argparse

import torch
from flash_attn.flash_attn_interface import flash_attn_func


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
        print(f"Testing FlashAttention with seq_len={seq_len}...")
        # 创建随机 Q/K/V，形状 (batch, seqlen, nheads, headdim)
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype, requires_grad=False)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype, requires_grad=False)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=dtype, requires_grad=False)

        # 显式启用 softmax_scale，通常设置为 1 / sqrt(D)
        softmax_scale = 1.0 / (head_dim ** 0.5)

        out = flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            # 其余参数使用默认值：dropout_p=0.0, causal=False, window_size=(-1, -1),
            # softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False
        )



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