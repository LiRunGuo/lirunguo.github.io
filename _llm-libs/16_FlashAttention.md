---
title: "FlashAttention 高效注意力"
excerpt: "IO-awareness原理、在线Softmax、O(N²)→O(N)内存、分块计算推导"
collection: llm-libs
permalink: /llm-libs/16-flash-attention
category: inference
---


## 1. 库的简介和在LLM开发中的作用

FlashAttention 是由 Tri Dao 等人提出的一种IO感知（IO-aware）的精确注意力算法，其核心目标是在不牺牲计算精度的前提下，大幅减少注意力计算对高带宽内存（HBM）的读写次数。传统注意力计算需要将完整的 N×N 注意力矩阵写入HBM，这在长序列场景下会导致严重的内存瓶颈。FlashAttention 通过分块计算（tiling）和在线softmax（online softmax）技术，将中间结果保留在SRAM中，从而将HBM读写复杂度从 O(N²d) 降低到 O(N²d²/M)，其中 M 为SRAM大小。

在LLM开发中，FlashAttention 的作用主要体现在：

- **训练加速**：在训练大语言模型时，注意力机制是最耗时的模块之一。FlashAttention 可带来 2-4 倍的训练加速。
- **长序列支持**：传统注意力因 O(N²) 内存占用难以处理长序列，FlashAttention 的 O(N) 内存占用使得训练和推理更长的序列成为可能。
- **推理优化**：通过 KV Cache 版本（flash_attn_with_kvcache），在自回归推理中高效利用缓存的键值对。
- **变长序列处理**：flash_attn_varlen_func 支持同一批次中不同长度的序列，避免padding带来的计算浪费。

## 2. 安装方式

### 基础安装

```bash
pip install flash-attn
```

该命令会从预编译轮子安装，需确保 CUDA 版本兼容（要求 CUDA 11.6+）。

### 从源码编译

```bash
pip install flash-attn --no-build-isolation
```

### 环境要求

- Python >= 3.7
- PyTorch >= 1.12.0
- CUDA >= 11.6
- GPU 架构：Ampere (A100)、Ada Lovelace (H100) 或更新的架构获得最佳性能，也支持 Turing (T4) 架构

### 验证安装

```python
import flash_attn
print(flash_attn.__version__)

# 快速验证功能
import torch
from flash_attn import flash_attn_func

q = torch.randn(2, 8, 16, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 8, 16, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 16, 64, device='cuda', dtype=torch.float16)
out = flash_attn_func(q, k, v)
print(f"Output shape: {out.shape}")  # torch.Size([2, 8, 16, 64])
```

## 3. 核心类/函数/工具的详细说明

### 3.1 flash_attn_func

核心注意力函数，执行精确的注意力计算，无近似。

```python
from flash_attn import flash_attn_func

out = flash_attn_func(
    q,                # 查询张量，形状 [batch_size, seqlen_q, num_heads, head_dim]
    k,                # 键张量，形状 [batch_size, seqlen_k, num_heads_k, head_dim]
    v,                # 值张量，形状 [batch_size, seqlen_k, num_heads_k, head_dim]
    dropout=0.0,      # dropout 概率，训练时使用
    causal=False,     # 是否应用因果掩码
    softmax_scale=None,  # softmax 缩放因子，默认为 1/sqrt(head_dim)
)
```

**参数详解：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `q` | Tensor | 查询张量，支持 bf16、fp16 数据类型 |
| `k` | Tensor | 键张量，head_dim 需与 q 一致。支持 GQA：num_heads_k 可小于 num_heads |
| `v` | Tensor | 值张量，形状与 k 相同 |
| `dropout` | float | Dropout 概率，0.0 表示不丢弃。仅在训练时生效 |
| `causal` | bool | True 时应用因果掩码，位置 i 只能关注位置 ≤ i |
| `softmax_scale` | float | 缩放因子，默认 1/√d。可自定义以实现不同缩放策略 |

**返回值：** Tensor，形状 `[batch_size, seqlen_q, num_heads, head_dim]`

**代码示例：**

```python
import torch
from flash_attn import flash_attn_func

# 基本使用
batch_size, seqlen, num_heads, head_dim = 4, 512, 32, 64
q = torch.randn(batch_size, seqlen, num_heads, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch_size, seqlen, num_heads, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch_size, seqlen, num_heads, head_dim, device='cuda', dtype=torch.float16)

# 标准注意力
out = flash_attn_func(q, k, v)
print(f"Standard attention output: {out.shape}")

# 因果注意力（用于自回归生成）
out_causal = flash_attn_func(q, k, v, causal=True)
print(f"Causal attention output: {out_causal.shape}")

# 带 dropout 的注意力（训练时）
out_dropout = flash_attn_func(q, k, v, dropout=0.1, causal=True)
print(f"Attention with dropout: {out_dropout.shape}")

# 自定义 softmax 缩放
out_scaled = flash_attn_func(q, k, v, softmax_scale=0.1)
print(f"Custom scaled attention: {out_scaled.shape}")
```

### 3.2 flash_attn_varlen_func

支持变长序列的注意力计算，在同一批次中处理不同长度的序列，无需padding。

```python
from flash_attn import flash_attn_varlen_func

out = flash_attn_varlen_func(
    q,                # 查询张量，形状 [total_q, num_heads, head_dim]（已拼接所有序列）
    k,                # 键张量，形状 [total_k, num_heads_k, head_dim]
    v,                # 值张量，形状 [total_k, num_heads_k, head_dim]
    cu_seqlens_q,     # Q的累积序列长度，形状 [batch_size + 1]
    cu_seqlens_k,     # K的累积序列长度，形状 [batch_size + 1]
    max_seqlen_q,     # Q中最大序列长度
    max_seqlen_k,     # K中最大序列长度
    dropout=0.0,
    causal=False,
    softmax_scale=None,
)
```

**参数详解：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `q` | Tensor | 拼接后的查询张量，形状为 [total_q, num_heads, head_dim] |
| `k` | Tensor | 拼接后的键张量 |
| `v` | Tensor | 拼接后的值张量 |
| `cu_seqlens_q` | Tensor | Q 的累积序列长度，如 [0, len1, len1+len2, ...] |
| `cu_seqlens_k` | Tensor | K 的累积序列长度 |
| `max_seqlen_q` | int | 批次中 Q 的最大序列长度 |
| `max_seqlen_k` | int | 批次中 K 的最大序列长度 |

**代码示例：**

```python
import torch
from flash_attn import flash_attn_varlen_func

# 3个不同长度的序列: 长度分别为 128, 256, 64
lengths = [128, 256, 64]
total_len = sum(lengths)
num_heads, head_dim = 32, 64

# 拼接所有序列（无需padding）
q = torch.randn(total_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(total_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(total_len, num_heads, head_dim, device='cuda', dtype=torch.float16)

# 构建累积序列长度
cu_seqlens = torch.tensor([0, 128, 384, 448], dtype=torch.int32, device='cuda')

out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=256,  # 最大序列长度
    max_seqlen_k=256,
    causal=True,
)
print(f"Varlen output shape: {out.shape}")  # [448, 32, 64]
```

### 3.3 flash_attn_with_kvcache

专为推理设计的KV Cache注意力函数，避免在自回归生成中重复计算已处理的键值对。

```python
from flash_attn import flash_attn_with_kvcache

out = flash_attn_with_kvcache(
    q,                # 新的查询，形状 [batch_size, seqlen_q, num_heads, head_dim]
    k_cache,          # K缓存，形状 [batch_size, seqlen_k, num_heads_k, head_dim]
    v_cache,          # V缓存，形状 [batch_size, seqlen_k, num_heads_k, head_dim]
    k=None,           # 新的K（可选，追加到cache）
    v=None,           # 新的V（可选，追加到cache）
    cache_seqlens=None,  # 每个序列当前缓存长度
    causal=True,
    softmax_scale=None,
)
```

**参数详解：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `q` | Tensor | 当前步的查询，通常 seqlen_q=1（单步生成） |
| `k_cache` | Tensor | 预分配的K缓存，包含已计算的键 |
| `v_cache` | Tensor | 预分配的V缓存，包含已计算的值 |
| `k` | Tensor | 新产生的键，会被自动追加到 k_cache |
| `v` | Tensor | 新产生的值，会被自动追加到 v_cache |
| `cache_seqlens` | Tensor | 各序列已缓存的长度，形状 [batch_size] |

**代码示例：**

```python
import torch
from flash_attn import flash_attn_with_kvcache

batch_size, num_heads, head_dim = 1, 32, 64
max_seqlen = 2048  # 最大序列长度

# 预分配KV缓存
k_cache = torch.zeros(batch_size, max_seqlen, num_heads, head_dim,
                       device='cuda', dtype=torch.float16)
v_cache = torch.zeros(batch_size, max_seqlen, num_heads, head_dim,
                       device='cuda', dtype=torch.float16)

# 模拟自回归生成
prompt_len = 128
# 初始prefill：将prompt的KV填入cache
q_prefill = torch.randn(batch_size, prompt_len, num_heads, head_dim,
                          device='cuda', dtype=torch.float16)
k_prefill = torch.randn(batch_size, prompt_len, num_heads, head_dim,
                          device='cuda', dtype=torch.float16)
v_prefill = torch.randn(batch_size, prompt_len, num_heads, head_dim,
                          device='cuda', dtype=torch.float16)

# 写入缓存
k_cache[:, :prompt_len] = k_prefill
v_cache[:, :prompt_len] = v_prefill

cache_seqlens = torch.tensor([prompt_len], dtype=torch.int32, device='cuda')

# 生成阶段：每步生成一个token
for step in range(10):
    # 当前步的query（长度为1）
    q_step = torch.randn(batch_size, 1, num_heads, head_dim,
                          device='cuda', dtype=torch.float16)
    k_step = torch.randn(batch_size, 1, num_heads, head_dim,
                          device='cuda', dtype=torch.float16)
    v_step = torch.randn(batch_size, 1, num_heads, head_dim,
                          device='cuda', dtype=torch.float16)

    out = flash_attn_with_kvcache(
        q_step, k_cache, v_cache,
        k=k_step, v=v_step,  # 新的KV会自动追加到cache
        cache_seqlens=cache_seqlens,
        causal=True,
    )
    cache_seqlens += 1  # 更新缓存长度
    print(f"Step {step}: output shape {out.shape}")
```

### 3.4 其他辅助函数

```python
from flash_attn import (
    flash_attn_func,           # 标准注意力
    flash_attn_varlen_func,    # 变长序列注意力
    flash_attn_with_kvcache,   # KV Cache推理注意力
    flash_attn_qkvpacked_func, # 打包QKV的注意力（单次调用）
    flash_attn_kvpacked_func,  # 打包KV的注意力
)
```

**flash_attn_qkvpacked_func：** 当 Q、K、V 来源于同一输入且具有相同长度时，可以将它们打包为一个张量以提高效率。

```python
from flash_attn import flash_attn_qkvpacked_func

# QKV打包形式: [batch, seqlen, 3, num_heads, head_dim]
qkv = torch.randn(4, 512, 3, 32, 64, device='cuda', dtype=torch.float16)
out = flash_attn_qkvpacked_func(qkv, causal=True)
```

## 4. 在LLM开发中的典型使用场景和代码示例

### 4.1 集成到Transformer模型

```python
import torch
import torch.nn as nn
from flash_attn import flash_attn_func

class FlashAttentionLayer(nn.Module):
    """使用FlashAttention的注意力层"""
    def __init__(self, d_model, num_heads, causal=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, D = x.shape

        # 投影并重塑为 [B, S, H, D]
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim)

        # 使用FlashAttention
        out = flash_attn_func(q, k, v, causal=self.causal)

        # 合并头并输出投影
        out = out.reshape(B, S, D)
        return self.out_proj(out)


class FlashTransformerBlock(nn.Module):
    """使用FlashAttention的Transformer块"""
    def __init__(self, d_model, num_heads, d_ff, causal=True):
        super().__init__()
        self.attn = FlashAttentionLayer(d_model, num_heads, causal=causal)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
```

### 4.2 支持GQA（Grouped Query Attention）

```python
import torch
from flash_attn import flash_attn_func

def grouped_query_attention(q, k, v, num_kv_groups):
    """
    GQA: 多个查询头共享同一组键值头
    q: [B, S, num_q_heads, head_dim]
    k: [B, S, num_kv_heads, head_dim]
    v: [B, S, num_kv_heads, head_dim]
    """
    # FlashAttention原生支持GQA，只需传入不同head数的K/V
    # num_q_heads 必须是 num_kv_heads 的整数倍
    out = flash_attn_func(q, k, v, causal=True)
    return out

# 示例: 32个查询头，8个KV头（4个查询头共享1个KV头）
B, S, num_q_heads, num_kv_heads, head_dim = 2, 512, 32, 8, 64
q = torch.randn(B, S, num_q_heads, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(B, S, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(B, S, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)

out = grouped_query_attention(q, k, v, num_kv_groups=4)
print(f"GQA output shape: {out.shape}")  # [2, 512, 32, 64]
```

### 4.3 长序列训练

```python
import torch
from flash_attn import flash_attn_func

def train_long_sequence():
    """展示FlashAttention在长序列上的优势"""
    # 16K序列长度，标准注意力会因O(N²)内存而失败
    # 但FlashAttention只需O(N)内存
    batch_size, seqlen, num_heads, head_dim = 1, 16384, 32, 64

    q = torch.randn(batch_size, seqlen, num_heads, head_dim,
                     device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, seqlen, num_heads, head_dim,
                     device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, seqlen, num_heads, head_dim,
                     device='cuda', dtype=torch.float16)

    out = flash_attn_func(q, k, v, causal=True)
    print(f"Long sequence attention output: {out.shape}")
    # 对比：标准注意力在16K序列上需要 ~16K * 16K * 4 bytes ≈ 1GB 的S矩阵
    # FlashAttention 只需要 ~16K * 64 * 2 bytes ≈ 2MB 的中间存储

train_long_sequence()
```

### 4.4 与HuggingFace Transformers集成

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 方式1: 使用flash_attention_2参数加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2",  # 自动使用FlashAttention
)

# 方式2: 手动替换注意力实现
# 在modeling代码中将标准attention替换为flash_attn_func
```

## 5. 数学原理

### 5.1 标准注意力计算

给定查询矩阵 Q、键矩阵 K、值矩阵 V，标准注意力的计算如下：

$$S = \frac{QK^T}{\sqrt{d}}$$

$$P = \text{softmax}(S)$$

$$O = PV$$

其中：
- Q ∈ ℝ^{N×d}，K ∈ ℝ^{N×d}，V ∈ ℝ^{N×d}
- S ∈ ℝ^{N×N} 为注意力分数矩阵
- P ∈ ℝ^{N×N} 为注意力权重矩阵
- O ∈ ℝ^{N×d} 为输出矩阵
- d 为头维度（head dimension）

**内存复杂度：** O(N²)，因为需要存储完整的 S 和 P 矩阵

**HBM读写复杂度：** O(N²d)，每次计算 S 的一行需要从 HBM 读取 K 的一列（d 个元素），共 N² 次 d 维读写

### 5.2 FlashAttention 分块计算

FlashAttention 的核心思想是将 Q、K、V 分成小块，在 GPU 的 SRAM（片上高速缓存）中完成注意力的全部计算，避免将中间结果写回 HBM。

**分块策略：**

将 Q 分为 T_r 个块 Q_1, Q_2, ..., Q_{T_r}，将 K、V 分为 T_c 个块 K_1, K_2, ..., K_{T_c} 和 V_1, V_2, ..., V_{T_c}。

对每个 Q_i 块：

1. **计算注意力分数：** S_{ij} = Q_i K_j^T / √d（在 SRAM 中计算）
2. **在线 Softmax：** P_{ij} = softmax(S_{ij})（使用在线算法，不需要完整 S 矩阵）
3. **累加输出：** O_i += P_{ij} V_j

### 5.3 在线 Softmax 算法（Online Softmax）

在线 Softmax 是 FlashAttention 的关键创新。传统 Softmax 需要先计算所有分数找到最大值，再统一做指数和归一化。在线 Softmax 允许逐块更新结果，无需存储完整的注意力矩阵。

**维护两个运行统计量：**

- m_i：运行最大值（running maximum）
- l_i：运行求和（running sum）

**更新规则：**

当处理第 j 个 K、V 块时：

1. 计算当前块的行最大值：
   m_{ij} = max(S_{ij})  (对 S_{ij} 的每一行取最大值)

2. 更新运行最大值：
   m_i^{new} = max(m_i^{old}, m_{ij})

3. 更新运行求和（修正之前的指数）：
   l_i^{new} = e^{m_i^{old} - m_i^{new}} · l_i^{old} + Σ exp(S_{ij} - m_i^{new})

4. 更新输出（修正之前的输出）：
   O_i^{new} = e^{m_i^{old} - m_i^{new}} · O_i^{old} · l_i^{old} / l_i^{new} + (exp(S_{ij} - m_i^{new}) · V_j) / l_i^{new}

**数值说明：**

当最大值从 m_i^{old} 更新为 m_i^{new} 时，之前计算的指数 exp(S_{ik} - m_i^{old}) 需要乘以修正因子 exp(m_i^{old} - m_i^{new})，这就是在线算法的核心——在发现更大的值时，回溯修正之前的结果。

### 5.4 复杂度分析

**内存复杂度：** O(N)

FlashAttention 不需要存储完整的 N×N 注意力矩阵，只需维护：
- 每行的运行最大值 m_i：N 个标量
- 每行的运行求和 l_i：N 个标量
- 输出矩阵 O：N×d

总内存：O(Nd)，其中 d << N 时近似为 O(N)。

**HBM 读写复杂度：** O(N²d²/M)

其中 M 为 SRAM 的大小。分块大小 B_r × B_c 由 SRAM 容量决定：
- B_r · d ≤ M（一个 Q 块加上输出能放入 SRAM）
- B_c · d ≤ M（一个 K、V 块能放入 SRAM）

需要从 HBM 读取 Q、K、V 各 T_r · T_c 次：
- T_r = ⌈N / B_r⌉ ≈ N / (M/d) = Nd/M
- T_c = ⌈N / B_c⌉ ≈ Nd/M
- 总读写次数 ≈ (Nd/M)² · M = N²d²/M

相比标准注意力的 O(N²d)，当 d²/M < d 即 d < M 时（通常成立，因为 SRAM 大小约 192KB，而 d 通常为 64-128），FlashAttention 的 HBM 读写次数更少。

### 5.5 因果注意力的隐式实现

在因果注意力中，位置 i 不能关注位置 j > i。标准实现需要构造一个 N×N 的因果掩码矩阵，并将其与 S 相乘。

FlashAttention 在分块计算中隐式应用因果掩码：

- 当 Q_i 的起始位置大于 K_j 的结束位置时（即所有查询位置都在所有键位置之后），正常计算
- 当 Q_i 的结束位置小于 K_j 的起始位置时（即所有查询位置都在所有键位置之前），跳过该块（S_{ij} 全为 -∞）
- 部分重叠时，只需在 SRAM 中对 S_{ij} 应用因果掩码，然后正常做在线 Softmax

这种方式避免了在 HBM 中存储 N×N 的掩码矩阵。

## 6. 代码原理/架构原理

### 6.1 IO感知计算模型

FlashAttention 的设计基于 GPU 内存层次结构的深刻理解：

```
┌─────────────────────────────────────┐
│           HBM (高带宽内存)           │  ~40-80 GB, ~2 TB/s 带宽
│   存储模型参数、激活值、KV Cache     │
└──────────────┬──────────────────────┘
               │ 读写延迟高
┌──────────────▼──────────────────────┐
│           SRAM (片上缓存)            │  ~192 KB/SM, ~19 TB/s 带宽
│   计算时的快速暂存区                 │
└──────────────┬──────────────────────┘
               │ 读写延迟极低
┌──────────────▼──────────────────────┐
│           CUDA Core (计算单元)       │  执行矩阵乘法、softmax等
└─────────────────────────────────────┘
```

关键洞察：现代 GPU 的计算能力远超内存带宽。标准注意力是"内存受限"（memory-bound）的——计算时间主要花在等待数据从 HBM 搬运到 SRAM，而非实际计算。FlashAttention 通过减少 HBM 读写次数，将注意力从内存受限变为计算受限，从而充分利用 GPU 算力。

### 6.2 CUDA Kernel 设计

FlashAttention 的 CUDA 实现包含以下关键设计：

**1. 线程块映射：**

每个线程块（thread block）负责计算一个 Q 块与所有 K、V 块的注意力。线程块的数量等于 T_r（Q 的块数）。

**2. 共享内存使用：**

- 加载 Q_i 块到共享内存
- 逐块加载 K_j、V_j 到共享内存
- 在共享内存中计算 S_{ij}、在线 Softmax、累加 O_i
- 最终将 O_i 写回 HBM

**3. Warp 级优化：**

- 使用 Warp 级矩阵乘法（wmma/mma）加速 QK^T 和 PV 计算
- 通过 Warp shuffle 指令在寄存器间快速交换数据

**4. 反向传播：**

FlashAttention 的反向传播同样采用分块策略：
- 不存储前向的注意力矩阵 P，而是重新计算 S_{ij}
- 利用前向传播保存的 m_i 和 l_i 来高效重算 softmax
- 这是一种"重计算而非存储"（recomputation vs. materialization）的权衡，用少量额外计算换取大量内存节省

### 6.3 与标准注意力的对比

```python
import torch
import torch.nn.functional as F
import time
from flash_attn import flash_attn_func

def standard_attention(q, k, v, causal=True):
    """标准PyTorch注意力实现"""
    # q, k, v: [B, S, H, D] -> [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        S = scores.shape[-1]
        mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))

    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2)  # [B, S, H, D]


# 性能对比
B, S, H, D = 2, 4096, 32, 64
q = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16)
k = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16)
v = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16)

# 标准注意力
torch.cuda.synchronize()
t0 = time.time()
out_std = standard_attention(q, k, v, causal=True)
torch.cuda.synchronize()
t_std = time.time() - t0

# FlashAttention
torch.cuda.synchronize()
t0 = time.time()
out_flash = flash_attn_func(q, k, v, causal=True)
torch.cuda.synchronize()
t_flash = time.time() - t0

print(f"Standard attention: {t_std*1000:.1f} ms")
print(f"FlashAttention:     {t_flash*1000:.1f} ms")
print(f"Speedup: {t_std/t_flash:.2f}x")
# 典型输出：FlashAttention 比 标准实现快 2-4 倍

# 数值精度验证
diff = (out_std - out_flash).abs().max().item()
print(f"Max difference: {diff:.6f}")  # 差异极小，在fp16精度范围内
```

## 7. 序列并行：Ring Attention

Ring Attention 是 FlashAttention 的扩展，用于在多个 GPU 间分布长序列的计算。

### 7.1 基本原理

在 Ring Attention 中，序列被切分为多个块，分布在不同的 GPU 上。每个 GPU 持有一块 Q 和对应的 K、V，通过环形通信传递 K、V 块：

1. GPU_i 持有 Q_i, K_i, V_i
2. 第 1 轮：GPU_i 用 Q_i 和本地 K_i, V_i 计算注意力
3. 第 2 轮：K、V 沿环形向右移动，GPU_i 收到 K_{i-1}, V_{i-1}，用 Q_i 继续计算
4. 重复直到所有 K、V 块都被处理

### 7.2 代码示例

```python
# Ring Attention 通常通过框架集成使用
# 以下展示原理性代码

import torch
import torch.distributed as dist

def ring_attention_step(q_local, k_block, v_block, causal=True):
    """Ring Attention中的单步计算"""
    from flash_attn import flash_attn_func
    # 在本地Q和接收到的K/V块之间计算注意力
    out = flash_attn_func(q_local, k_block, v_block, causal=causal)
    return out

# 实际使用中，推荐使用 flash_attn 的 ring_attention 实现
# from flash_attn.bert_padding import ring_attention
```

### 7.3 Ring Attention 的通信与计算重叠

Ring Attention 的关键优化是将 K、V 的环形通信与注意力计算重叠：
- 当 GPU_i 在用 K_j, V_j 计算注意力时，同时异步接收 K_{j-1}, V_{j-1}
- 计算和通信完全重叠，实现了通信开销的"隐藏"

## 8. 常见注意事项和最佳实践

### 8.1 数据类型限制

FlashAttention 仅支持 `float16` 和 `bfloat16` 数据类型：

```python
# 正确
q = torch.randn(..., dtype=torch.float16, device='cuda')
k = torch.randn(..., dtype=torch.float16, device='cuda')
v = torch.randn(..., dtype=torch.float16, device='cuda')

# 正确（bf16在Ampere及以上架构）
q = torch.randn(..., dtype=torch.bfloat16, device='cuda')

# 错误 - float32 不支持
# q = torch.randn(..., dtype=torch.float32, device='cuda')
```

### 8.2 维度对齐要求

- `head_dim` 必须能被 8 整除（fp16）或 16 整除（某些bf16场景）
- 推荐 `head_dim` 为 64、128 等值
- Q 的 `num_heads` 必须是 K/V 的 `num_heads` 的整数倍（GQA）

### 8.3 内存布局注意

FlashAttention 使用 `[B, S, H, D]` 的内存布局，而非 PyTorch 标准的 `[B, H, S, D]`。在集成时需要注意布局转换：

```python
# PyTorch标准布局: [B, H, S, D]
q_bhsd = torch.randn(2, 32, 512, 64, device='cuda', dtype=torch.float16)

# FlashAttention布局: [B, S, H, D]
q_bshd = q_bhsd.transpose(1, 2)  # 需要转置

out = flash_attn_func(q_bshd, k_bshd, v_bshd, causal=True)

# 转回标准布局
out_bhsd = out.transpose(1, 2)
```

### 8.4 因果注意力的效率提示

当使用因果注意力时，FlashAttention 会自动跳过不需要计算的块，性能提升显著。对于自回归模型，始终设置 `causal=True`。

### 8.5 Dropout 的确定性

训练时使用 Dropout 可能导致结果不确定。如需确定性结果：

```python
# 设置全局随机种子
torch.manual_seed(42)
# 或确保使用确定性算法
torch.use_deterministic_algorithms(True)
```

### 8.6 梯度检查点

FlashAttention 的反向传播会重计算前向的注意力分数，而非存储它们。这在梯度检查点（gradient checkpointing）场景下尤为有利：

```python
from torch.utils.checkpoint import checkpoint

# 与梯度检查点配合使用
def attn_fn(q, k, v):
    return flash_attn_func(q, k, v, causal=True)

# 自动处理重计算，FlashAttention的重计算开销很小
out = checkpoint(attn_fn, q, k, v)
```

### 8.7 性能调优建议

1. **序列长度对齐**：将序列长度对齐到 128 或 256 的倍数，可以避免尾部块的低效计算
2. **使用 packed QKV**：当 Q、K、V 来源于同一输入时，使用 `flash_attn_qkvpacked_func` 减少内存分配
3. **批量大小与头数的平衡**：较大的批量大小和头数通常能更好地利用 GPU 并行度
4. **选择合适的 head_dim**：head_dim=64 或 128 通常是性能最优的选择
5. **bf16 vs fp16**：在 Ampere 及以上架构上，bf16 通常训练更稳定（更大的数值范围），但 fp16 在某些场景下更快

### 8.8 常见错误排查

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `CUDA error: misaligned address` | head_dim 未对齐 | 确保 head_dim 是 8 的倍数 |
| `NotImplementedError: Only fp16 and bf16` | 使用了 float32 | 转换为 fp16 或 bf16 |
| `RuntimeError: CUDA out of memory` | 序列过长或批量过大 | 减小批量大小或使用梯度检查点 |
| `head_dim must be divisible by 8` | head_dim 不是 8 的倍数 | 调整模型配置 |

### 8.9 FlashAttention-2 和 FlashAttention-3

- **FlashAttention-2**：优化了 CUDA kernel 的并行度，在 A100 上接近理论峰值。改进了工作分配（work partitioning），减少了 warp 间的同步开销。
- **FlashAttention-3**：针对 Hopper 架构（H100）优化，利用 TMA（Tensor Memory Accelerator）和 FP8 支持，进一步提升了吞吐量。

安装时默认获取最新版本，通常已包含 FlashAttention-2 的优化。FlashAttention-3 需要特定硬件支持。
