# -*- coding: utf-8 -*-
# 与 tools/test_notebook_answers.py 同样的执行方式：运行 11 节题目区的账本单元。
import torch


def kv_cache_bytes(seq_len, num_layers, num_kv_heads, head_dim, batch_size=1, dtype_bytes=2):
    """估算 K 和 V 的理论存储量；不包含 allocator、workspace 和碎片。"""
    values = (seq_len, num_layers, num_kv_heads, head_dim, batch_size, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError('序列长度、层数、头数、维度、batch 和 dtype 字节数必须为正数')
    return 2 * seq_len * num_layers * num_kv_heads * head_dim * batch_size * dtype_bytes


examples = [(1024, 32, 32, 128), (2048, 32, 32, 128), (4096, 32, 32, 128)]
for seq_len, layers, kv_heads, head_dim in examples:
    size_gb = kv_cache_bytes(seq_len, layers, kv_heads, head_dim) / 1e9
    print(f"seq_len={seq_len:4d} -> KV cache ≈ {size_gb:5.2f} GB")

seq_len, num_layers, head_dim = 4096, 32, 128
for name, kv_heads in [("MHA", 32), ("GQA", 8), ("MQA", 1)]:
    gb = kv_cache_bytes(seq_len, num_layers, kv_heads, head_dim) / 1e9
    print(f"{name:>3s}: kv_heads={kv_heads:2d}, KV cache ≈ {gb:5.2f} GB")
