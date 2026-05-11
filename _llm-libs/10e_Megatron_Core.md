---
title: "Megatron-Core 大模型训练框架"
excerpt: "张量并行(Column/Row)、流水线并行(1F1B)、序列并行、上下文并行(Ring Attention)"
collection: llm-libs
permalink: /llm-libs/10e-megatron-core
category: training
toc: true
---


## 1. 简介

Megatron-Core 是 NVIDIA 开发的大语言模型（LLM）分布式训练核心库，源自 Megatron-LM 项目。它提供了一套模块化、可组合的并行训练原语，旨在支持千亿甚至万亿参数规模的模型高效训练。Megatron-Core 将 Megatron-LM 中经过大规模验证的训练技术抽取为独立库，使其可以被更广泛地集成到各类训练框架中。

### 在 LLM 开发中的作用

- **多维并行训练**：提供张量并行（TP）、流水线并行（PP）、序列并行（SP）、上下文并行（CP）和数据并行（DP）等多种并行策略，支持灵活组合。
- **高性能 Transformer 实现**：针对 NVIDIA GPU 深度优化的 Transformer 层实现，支持 fp8 混合精度训练。
- **模块化设计**：核心组件（并行层、通信原语、配置系统）可独立使用或组合，便于集成到自定义训练流程中。
- **大规模验证**：已在 NVIDIA 的 GPT、LLaMA、Mixtral 等千亿级模型训练中得到验证，是工业界大模型训练的事实标准之一。

---

## 2. 安装方式

### 从 PyPI 安装

```bash
pip install megatron-core
```

### 从源码安装

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .
```

### 依赖说明

- Python >= 3.8
- PyTorch >= 2.0（建议使用 NVIDIA PyTorch 容器以获得最佳兼容性）
- NVIDIA GPU（计算能力 >= 8.0，即 Ampere 架构及以上，fp8 需要 Hopper 架构）
- CUDA >= 11.8
- NCCL（NVIDIA 集合通信库）
- Apex（可选，用于融合内核优化）

### 使用 NVIDIA 容器（推荐）

```bash
# 拉取 NGC 容器
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# 容器内已预装 PyTorch、NCCL、Apex 等依赖
pip install megatron-core
```

---

## 3. 核心类与函数详细说明

### 3.1 TransformerConfig — 模型配置

`TransformerConfig` 是 Megatron-Core 的核心配置类，定义了 Transformer 模型的所有超参数和并行策略。

```python
from megatron.core import transformer

config = transformer.TransformerConfig(
    num_layers=32,                  # Transformer 层数
    hidden_size=4096,               # 隐藏层维度
    num_attention_heads=32,         # 注意力头数
    num_query_groups=32,            # GQA 中的 KV 组数（等于 num_attention_heads 为标准 MHA）
    kv_channels=128,                # 每个注意力头的维度 (= hidden_size / num_attention_heads)
    ffn_hidden_size=11008,          # FFN 中间层维度（通常为 hidden_size 的 2.7~4 倍）
    normalization="LayerNorm",      # 归一化方式："LayerNorm" 或 "RMSNorm"
    norm_epsilon=1e-5,              # 归一化的 epsilon
    # --- 并行策略 ---
    tensor_model_parallel_size=1,   # 张量并行度（TP）
    pipeline_model_parallel_size=1, # 流水线并行度（PP）
    context_parallel_size=1,        # 上下文并行度（CP）
    # --- 混合精度 ---
    bf16=True,                      # 是否使用 BF16
    fp8=False,                      # 是否启用 FP8 训练（需要 Hopper GPU）
    fp8_format="hybrid",            # FP8 格式："hybrid" 或 "e4m3"
    # --- 序列并行 ---
    sequence_parallel=True,         # 是否启用序列并行
    # --- 其他 ---
    apply_residual_connection_post_norm=False,  # 残差连接是否在归一化之后
    hidden_dropout=0.1,             # 隐藏层 dropout
    attention_dropout=0.1,          # 注意力 dropout
    kv_channels=128,                # KV 通道数
    rotary_percent=1.0,             # RoPE 应用的比例
    add_bias_linear=False,          # 线性层是否添加偏置
    gated_linear_unit=True,         # 是否使用门控线性单元（SwiGLU）
    activation_func="silu",         # 激活函数
)
```

#### 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_layers` | int | — | Transformer 总层数 |
| `hidden_size` | int | — | 模型隐藏维度 |
| `num_attention_heads` | int | — | 查询头数 |
| `num_query_groups` | int | — | KV 组数，用于 GQA/MQA |
| `tensor_model_parallel_size` | int | 1 | TP 并行度，即沿张量维度切分的 GPU 数 |
| `pipeline_model_parallel_size` | int | 1 | PP 并行度，即沿层维度切分的 GPU 数 |
| `sequence_parallel` | bool | True | 是否启用序列并行（需要 TP > 1） |
| `bf16` | bool | False | 是否使用 BF16 混合精度 |

### 3.2 initialize_model_parallel — 初始化并行组

在创建模型之前，必须先初始化模型并行组。该函数创建 TP、PP、DP、CP 各并行维度的进程组。

```python
from megatron.core import mpu

# 初始化并行组
mpu.initialize_model_parallel(
    tensor_model_parallel_size=4,    # TP 大小
    pipeline_model_parallel_size=2,  # PP 大小
    virtual_pipeline_model_parallel_size=None,  # 虚拟流水线并行（ interleaved PP）
    context_parallel_size=1,         # CP 大小
)
```

#### 并行组关系

- 总 GPU 数 = TP × PP × CP × DP
- DP = 总 GPU 数 / (TP × PP × CP)
- 例如：16 GPU, TP=4, PP=2, CP=1 → DP=2

#### 获取并行组信息

```python
# 获取各并行维度的 world size 和 rank
tp_rank = mpu.get_tensor_model_parallel_rank()       # 当前 TP rank
tp_world = mpu.get_tensor_model_parallel_world_size() # TP world size

pp_rank = mpu.get_pipeline_model_parallel_rank()     # 当前 PP rank
pp_world = mpu.get_pipeline_model_parallel_world_size()

dp_rank = mpu.get_data_parallel_rank()               # 当前 DP rank
dp_world = mpu.get_data_parallel_world_size()

cp_rank = mpu.get_context_parallel_rank()            # 当前 CP rank
cp_world = mpu.get_context_parallel_world_size()
```

### 3.3 并行线性层

#### ColumnParallelLinear — 列并行线性层

权重矩阵按列切分，每个 GPU 持有部分列。输入完整复制到每个 GPU，输出是局部的。

```python
from megatron.core.tensor_parallel import ColumnParallelLinear

col_linear = ColumnParallelLinear(
    input_size=4096,        # 输入维度
    output_size=11008,      # 输出维度（完整值，会自动按 tp 切分）
    config=config,          # TransformerConfig 实例
    init_method=None,       # 权重初始化方法
    bias=False,             # 是否有偏置
    gather_output=False,    # 是否在输出后 all-gather 汇聚（False=保持切分状态）
    skip_bias_add=False,    # 是否跳过偏置加法（返回偏置供融合操作使用）
)
```

**数据流**：输入 X（每个 GPU 持有完整副本）→ XW_i（每个 GPU 计算自己持有的列切片）→ 输出 Y_i（局部结果，无需通信）

#### RowParallelLinear — 行并行线性层

权重矩阵按行切分，每个 GPU 持有部分行。输入是局部的，输出需要 all-reduce 汇聚。

```python
from megatron.core.tensor_parallel import RowParallelLinear

row_linear = RowParallelLinear(
    input_size=11008,          # 输入维度（完整值，会自动按 tp 切分）
    output_size=4096,          # 输出维度
    config=config,             # TransformerConfig 实例
    init_method=None,          # 权重初始化方法
    bias=False,                # 是否有偏置
    input_is_parallel=True,    # 输入是否已经是按 tp 切分的状态
    skip_bias_add=False,       # 是否跳过偏置加法
)
```

**数据流**：输入 Y_i（局部）→ Y_i × W_i^T（每个 GPU 计算部分结果）→ All-Reduce Σ → 输出 Z（完整）

### 3.4 ParallelMLP — 并行 MLP 模块

MLP 由两层线性变换组成：第一层用 Column Parallel，第二层用 Row Parallel。

```python
from megatron.core.tensor_parallel.mappings import TensorParallelMode
from megatron.core.transformer.mlp import ParallelMLP

mlp = ParallelMLP(
    config=config,  # TransformerConfig 实例
)
```

#### 内部结构

```
输入 X
  │
  ▼
ColumnParallelLinear (hidden_size → ffn_hidden_size)  # 按列切分
  │
  ▼
激活函数 (GeLU / SiLU)
  │
  ▼
RowParallelLinear (ffn_hidden_size → hidden_size)     # 按行切分
  │
  ▼
All-Reduce (汇聚结果)
  │
  ▼
输出 Z
```

### 3.5 ParallelSelfAttention — 并行自注意力模块

自注意力中 QKV 投影使用 Column Parallel，输出投影使用 Row Parallel。

```python
from megatron.core.transformer.attention import ParallelSelfAttention

attn = ParallelSelfAttention(
    config=config,  # TransformerConfig 实例
)
```

#### 内部结构

```
输入 X
  │
  ▼
ColumnParallelLinear (hidden_size → 3 * hidden_size)  # QKV 联合投影，按列切分
  │
  ▼
Split → Q, K, V (每个 GPU 持有部分头)
  │
  ▼
Scaled Dot-Product Attention (局部计算，无需通信)
  │
  ▼
RowParallelLinear (hidden_size → hidden_size)         # 输出投影，按行切分
  │
  ▼
All-Reduce (汇聚结果)
  │
  ▼
输出
```

### 3.6 TransformerLayer — 完整的 Transformer 层

将自注意力、MLP、归一化和残差连接组合成完整的 Transformer 层。

```python
from megatron.core.transformer.transformer_layer import TransformerLayer

layer = TransformerLayer(
    config=config,             # TransformerConfig 实例
    layer_number=1,            # 层编号（用于 RoPE 等位置相关计算）
    self_attn=None,            # 可选：自定义注意力模块
    mlp=None,                  # 可选：自定义 MLP 模块
)
```

### 3.7 TransformerBlock — Transformer 模型主体

包含多层 TransformerLayer，并处理流水线并行的层划分。

```python
from megatron.core.transformer.transformer_block import TransformerBlock

block = TransformerBlock(
    config=config,               # TransformerConfig 实例
    transformer_layer_list=None, # 可选：自定义层列表
    post_layer_norm=True,        # 是否在最后添加 LayerNorm
)
```

### 3.8 TensorParallelMode 与 parallel_mode 上下文管理器

`TensorParallelMode` 和 `parallel_mode` 用于控制张量并行的行为模式。

```python
from megatron.core.tensor_parallel import (
    TensorParallelMode,
    get_tensor_model_parallel_group,
)

# TensorParallelMode 枚举值：
# - Column: 列并行模式（用于 QKV 投影、MLP 第一个线性层）
# - Row: 行并行模式（用于输出投影、MLP 第二个线性层）
```

### 3.9 通信原语

Megatron-Core 提供了高效的集合通信封装：

```python
from megatron.core.tensor_parallel import (
    copy_to_tensor_model_parallel_region,   # 恒等操作（前向），all-reduce（反向）
    gather_from_tensor_model_parallel_region, # all-gather（前向），恒等（反向）
    reduce_from_tensor_model_parallel_region, # reduce-scatter（前向），恒等（反向）
    scatter_to_tensor_model_parallel_region,  # 恒等（前向），all-reduce（反向）
)
```

#### 通信原语与前向/反向行为

| 原语 | 前向传播 | 反向传播 | 典型用途 |
|------|----------|----------|----------|
| `copy_to_tp_region` | Identity | All-Reduce | Column Parallel 输入 |
| `gather_from_tp_region` | All-Gather | Identity | Column Parallel 输出汇聚 |
| `reduce_from_tp_region` | Reduce-Scatter | Identity | Row Parallel 输出 |
| `scatter_to_tp_region` | Identity | All-Reduce | Row Parallel 输入分发 |

---

## 4. 典型使用场景和代码示例

### 4.1 基础模型构建

```python
import torch
from megatron.core import mpu
from megatron.core.transformer import TransformerConfig, TransformerBlock

# Step 1: 初始化分布式环境
torch.distributed.init_process_group(backend="nccl")

# Step 2: 初始化模型并行组
mpu.initialize_model_parallel(
    tensor_model_parallel_size=4,     # 4路张量并行
    pipeline_model_parallel_size=1,   # 不使用流水线并行
    context_parallel_size=1,          # 不使用上下文并行
)

# Step 3: 创建模型配置
config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_query_groups=8,               # GQA: 8组KV头
    ffn_hidden_size=11008,
    seq_length=4096,
    normalization="RMSNorm",
    bf16=True,
    tensor_model_parallel_size=4,
    sequence_parallel=True,           # 启用序列并行
    gated_linear_unit=True,           # SwiGLU FFN
    add_bias_linear=False,
)

# Step 4: 构建模型
model = TransformerBlock(config=config)

# Step 5: 前向传播
# 输入: [seq_len, batch_size, hidden_size]
input_ids = torch.randint(0, 32000, (4096, 2)).cuda()
hidden_states = torch.randn(4096, 2, 4096, dtype=torch.bfloat16).cuda()
output = model(hidden_states)
```

### 4.2 张量并行训练 (TP)

```python
import torch
import torch.nn as nn
from megatron.core import mpu
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import TransformerConfig

torch.distributed.init_process_group(backend="nccl")
mpu.initialize_model_parallel(tensor_model_parallel_size=4)

config = TransformerConfig(
    num_layers=1,
    hidden_size=4096,
    num_attention_heads=32,
    tensor_model_parallel_size=4,
    bf16=True,
)

# 手动构建一个张量并行的两层 MLP
class TensorParallelMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(
            input_size=config.hidden_size,          # 4096
            output_size=config.ffn_hidden_size,     # 11008，自动按4切分为2752
            config=config,
            gather_output=False,   # 不汇聚，保持切分状态传给下一层
            bias=False,
        )
        self.dense_4h_to_h = RowParallelLinear(
            input_size=config.ffn_hidden_size,      # 11008，输入已按4切分
            output_size=config.hidden_size,         # 4096
            config=config,
            input_is_parallel=True,  # 输入已经是切分状态
            bias=False,
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        # x: [seq, batch, hidden] - 每个 GPU 持有完整副本
        intermediate = self.dense_h_to_4h(x)   # 每GPU: [seq, batch, ffn/tp]
        intermediate = self.activation(intermediate)
        output = self.dense_4h_to_h(intermediate)  # All-Reduce → [seq, batch, hidden]
        return output

model = TensorParallelMLP(config).cuda()
x = torch.randn(512, 2, 4096, dtype=torch.bfloat16).cuda()
y = model(x)  # 输出形状: [512, 2, 4096]，所有GPU结果一致
```

### 4.3 流水线并行训练 (PP)

```python
import torch
from megatron.core import mpu
from megatron.core.transformer import TransformerConfig, TransformerBlock

torch.distributed.init_process_group(backend="nccl")
mpu.initialize_model_parallel(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,  # 4 级流水线
)

config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    bf16=True,
)

# TransformerBlock 自动根据 PP rank 获取对应的层
# PP rank 0: layer 0-7, PP rank 1: layer 8-15, ...
model = TransformerBlock(config=config).cuda()

# 在实际训练中，使用 Megatron 训练框架处理流水线调度
# 此处展示概念性前向传播
pp_rank = mpu.get_pipeline_model_parallel_rank()
num_layers_per_stage = config.num_layers // mpu.get_pipeline_model_parallel_world_size()

print(f"Pipeline stage {pp_rank}: handles layers "
      f"{pp_rank * num_layers_per_stage} - {(pp_rank + 1) * num_layers_per_stage - 1}")
```

### 4.4 序列并行 (SP)

```python
import torch
from megatron.core import mpu
from megatron.core.transformer import TransformerConfig, TransformerLayer

torch.distributed.init_process_group(backend="nccl")
mpu.initialize_model_parallel(tensor_model_parallel_size=4)

config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    tensor_model_parallel_size=4,
    sequence_parallel=True,   # 关键：启用序列并行
    bf16=True,
)

# 当 sequence_parallel=True 时:
# - LayerNorm/Dropout 的输入/输出沿序列维度切分
# - 每个 GPU 只持有 1/4 的序列长度
# - 注意力计算前 all-gather 恢复完整序列
# - 注意力计算后 reduce-scatter 重新切分

layer = TransformerLayer(config=config, layer_number=1).cuda()

# SP 模式下的输入: [seq_len/tp, batch, hidden]
# 例如完整序列 4096, tp=4 → 每个 GPU 输入 seq_len=1024
sp_seq_len = 4096 // 4  # = 1024
hidden_states = torch.randn(sp_seq_len, 2, 4096, dtype=torch.bfloat16).cuda()
output = layer(hidden_states)  # 输出同样为 [seq_len/tp, batch, hidden]
```

### 4.5 上下文并行 (CP) — 超长序列训练

```python
import torch
from megatron.core import mpu
from megatron.core.transformer import TransformerConfig, TransformerBlock

torch.distributed.init_process_group(backend="ncll")
mpu.initialize_model_parallel(
    tensor_model_parallel_size=2,
    context_parallel_size=4,  # 4路上下文并行
)

config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    tensor_model_parallel_size=2,
    context_parallel_size=4,
    seq_length=131072,         # 128K 超长序列
    bf16=True,
)

model = TransformerBlock(config=config).cuda()

# CP 模式下：每个 GPU 处理 seq_len/cp 的序列片段
# 通过 Ring Attention 机制在 CP 组内通信
# 实现超长序列的高效训练
cp_seq_len = 131072 // 4  # = 32768
hidden_states = torch.randn(cp_seq_len, 1, 4096, dtype=torch.bfloat16).cuda()
output = model(hidden_states)
```

### 4.6 完整训练循环示例

```python
import torch
from megatron.core import mpu
from megatron.core.transformer import TransformerConfig, TransformerBlock
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig

# --- 初始化 ---
torch.distributed.init_process_group(backend="nccl")
mpu.initialize_model_parallel(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2,
)

# --- 配置 ---
model_config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    ffn_hidden_size=11008,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2,
    sequence_parallel=True,
    bf16=True,
)

optim_config = OptimizerConfig(
    optimizer="adam",           # 优化器类型
    lr=1e-4,                    # 学习率
    weight_decay=0.01,          # 权重衰减
    adam_beta1=0.9,             # Adam beta1
    adam_beta2=0.95,            # Adam beta2
    adam_eps=1e-8,              # Adam epsilon
    clip_grad=1.0,              # 梯度裁剪阈值
)

# --- 构建模型 ---
model = TransformerBlock(config=model_config).cuda()

# --- 构建优化器 ---
optimizer = get_megatron_optimizer(optim_config, [model])

# --- 训练循环 ---
for step in range(1000):
    # 生成模拟数据
    batch_size = 4
    seq_len = 2048
    input_ids = torch.randint(0, 32000, (seq_len, batch_size)).cuda()
    labels = torch.randint(0, 32000, (seq_len, batch_size)).cuda()

    # 前向传播
    optimizer.zero_grad()
    hidden_states = model(input_ids)  # 简化示例，实际需要embedding层

    # 计算损失（简化）
    loss = hidden_states.float().mean()

    # 反向传播
    loss.backward()

    # 梯度裁剪与参数更新
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

---

## 5. 数学原理

### 5.1 张量并行 (Tensor Parallelism)

#### 列并行 (Column Parallel)

对于线性变换 Y = XW + b，将权重 W ∈ R^{d×k} 按列切分为 n 份：

$$W = [W_1, W_2, \ldots, W_n]$$

每个 GPU_i 持有 $W_i \in R^{d \times (k/n)}$，独立计算：

$$Y_i = XW_i + b_i$$

- **前向**：输入 X 完整复制到每个 GPU，各 GPU 独立计算局部结果，无需通信
- **反向**：梯度 $\frac{\partial L}{\partial X} = \sum_{i=1}^{n} \frac{\partial L}{\partial Y_i} W_i^T$，需要 All-Reduce 聚合

#### 行并行 (Row Parallel)

将权重 W ∈ R^{k×d} 按行切分为 n 份：

$$W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_n \end{bmatrix}$$

每个 GPU_i 持有 $W_i \in R^{(k/n) \times d}$，输入 Y_i 是第 i 个分片：

$$Z_i = Y_i W_i$$

最终结果通过 All-Reduce 聚合：

$$Z = \sum_{i=1}^{n} Z_i$$

- **前向**：输入 Y_i 已按 tp 维度切分，各 GPU 独立计算后 All-Reduce
- **反向**：梯度 $\frac{\partial L}{\partial Y_i} = \frac{\partial L}{\partial Z} W_i^T$，各 GPU 独立计算，无需通信

#### MLP 的张量并行

```
X (完整)
  │
  ├──→ GPU_0: Y_0 = X·W_0         (Column Parallel, W_0是W的前1/n列)
  ├──→ GPU_1: Y_1 = X·W_1
  ├──→ ...
  └──→ GPU_n: Y_n = X·W_n
        │
        ▼ GeLU(Y_i)                (各GPU独立计算激活函数)
        │
  GPU_0: Z_0 = GeLU(Y_0)·W_0'     (Row Parallel, W_0'是W'的前1/n行)
  GPU_1: Z_1 = GeLU(Y_1)·W_1'
  ...
  GPU_n: Z_n = GeLU(Y_n)·W_n'
        │
        ▼ All-Reduce
  Z = Z_0 + Z_1 + ... + Z_n        (汇聚完整结果)
```

整个 MLP 只需要 **1 次 All-Reduce**（在 Row Parallel 输出时），Column Parallel 的输入复制通过前向恒等+反向 All-Reduce 实现，零额外通信。

#### Self-Attention 的张量并行

- QKV 投影：Column Parallel，每个 GPU 持有部分注意力头
- Attention 计算：各 GPU 独立计算自己持有的头的注意力
- Output 投影：Row Parallel，All-Reduce 聚合结果
- 同样只需 **1 次 All-Reduce**

### 5.2 流水线并行 (Pipeline Parallelism)

#### 基本概念

将模型的 L 层平均分配到 P 个阶段（stage），每个 stage 在不同的 GPU 集上执行：

```
Stage 0 (GPU 0-3): Layer 0 ~ L/P-1
Stage 1 (GPU 4-7): Layer L/P ~ 2L/P-1
...
Stage P-1:         Layer (P-1)L/P ~ L-1
```

#### 1F1B 调度 (One Forward One Backward)

1F1B 是流水线并行的核心调度策略，分为三个阶段：

**预热阶段（Warmup）**：逐个执行前向传播，填充流水线

```
Stage 0: F0, F1, F2, ...
Stage 1:    F0, F1, F2, ...
Stage 2:       F0, F1, F2, ...
```

**稳态阶段（Steady State）**：交替执行 1 个前向 + 1 个反向

```
Stage 0: ... F3, B0, F4, B1, F5, B2, ...
Stage 1: ... F3, B0, F4, B1, F5, B2, ...
Stage 2: ... F3, B0, F4, B1, F5, B2, ...
```

**冷却阶段（Cooldown）**：逐个执行剩余的反向传播

```
Stage 0: ... B3, B4, B5
Stage 1:    ... B3, B4, B5
Stage 2:       ... B3, B4, B5
```

**关键指标**：
- 微批次数 M 需要大于等于 stage 数 P（否则流水线无法填满）
- 总步数 = M + P - 1（M 个微批次，P-1 步流水线填充）
- 稳态显存占用：每个 stage 只需保存 1 份前向激活（而不是 M 份）
- Bubble 比例 = (P-1) / M（M >> P 时接近 0）

### 5.3 序列并行 (Sequence Parallelism)

#### 动机

在标准张量并行中，非张量并行区域（如 LayerNorm、Dropout）的输入和输出在每个 GPU 上都是完整复制的，造成显存浪费。

#### 原理

在 TP 的非张量并行区域（LayerNorm、Dropout 等），沿序列维度将激活切分为 tp 份：

```
标准 TP（无 SP）:
  GPU_0: LayerNorm(X_full)    → 完整序列激活，显存 = seq_len × batch × hidden
  GPU_1: LayerNorm(X_full)    → 完整序列激活（重复）
  GPU_2: LayerNorm(X_full)    → 完整序列激活（重复）
  GPU_3: LayerNorm(X_full)    → 完整序列激活（重复）

序列并行（SP）:
  GPU_0: LayerNorm(X_0)       → 1/4 序列激活，显存 = (seq_len/4) × batch × hidden
  GPU_1: LayerNorm(X_1)       → 1/4 序列激活
  GPU_2: LayerNorm(X_2)       → 1/4 序列激活
  GPU_3: LayerNorm(X_3)       → 1/4 序列激活
```

#### 通信模式变化

SP 改变了 TP 区域边界的通信方式：

- **无 SP**：Column Parallel 前向恒等 / 反向 All-Reduce
- **有 SP**：Column Parallel 前向 All-Gather / 反向 Reduce-Scatter
  - All-Gather：将切分的序列恢复完整，供注意力计算使用
  - Reduce-Scatter：注意力输出后重新沿序列维度切分

总体通信量不变（All-Gather + Reduce-Scatter = All-Reduce），但显存节省了 (tp-1)/tp。

### 5.4 上下文并行 (Context Parallelism)

#### 动机

对于超长序列（如 128K tokens），即使使用 SP，单个 GPU 的显存仍可能不够。CP 将序列沿序列维度进一步切分到多个 GPU。

#### Ring Attention 实现

CP 基于 Ring Attention，将 Q, K, V 沿序列维度切分，通过环形通信实现因果注意力：

```
Step 1: 每个 CP rank 持有 Q_i, K_i, V_i（序列的第 i 段）
Step 2: 计算本地注意力 Q_i × K_i^T
Step 3: 通过环形通信传递 K, V 到下一个 CP rank
Step 4: 计算 Q_i × K_{i+1}^T（注意因果 mask）
Step 5: 重复直到所有 K, V 块都被处理
Step 6: 累积得到完整的注意力输出
```

**关键**：Ring Attention 在计算注意力的同时进行通信，实现了计算与通信的重叠，CP 扩展几乎是零额外开销。

### 5.5 混合精度训练 (FP8)

#### FP8 数据格式

- **E4M3**：4 位指数 + 3 位尾数，范围较小但精度较高，用于前向传播
- **E5M2**：5 位指数 + 2 位尾数，范围较大但精度较低，用于反向传播

#### FP8 训练流程

```
权重 (FP32) → Cast to FP8 (E4M3) → 前向计算 (FP8)
                                       ↓
                                   损失计算 (BF16/FP32)
                                       ↓
梯度 (FP8 E5M2) ← Cast from BF16 ← 反向计算 (FP8)
     ↓
权重更新 (FP32)
```

FP8 可将训练吞吐量提升约 2 倍（相比 BF16），同时减少显存占用约 50%。

---

## 6. 代码原理与架构原理

### 6.1 整体架构

```
megatron.core/
├── tensor_parallel/
│   ├── mappings.py          # 通信原语（all-reduce, all-gather, reduce-scatter）
│   ├── layers.py            # ColumnParallelLinear, RowParallelLinear
│   └── __init__.py          # TP 初始化、并行组管理
├── pipeline_parallel/
│   ├── schedules.py         # 流水线调度（1F1B 等）
│   └── __init__.py          # PP 初始化
├── transformer/
│   ├── transformer_config.py    # TransformerConfig
│   ├── transformer_layer.py     # TransformerLayer
│   ├── transformer_block.py     # TransformerBlock
│   ├── attention.py             # ParallelSelfAttention
│   ├── mlp.py                   # ParallelMLP
│   └── custom_layers/          # 自定义层（TE、FlashAttention 等）
├── optimizer/
│   └── optimizer.py         # 优化器封装
├── models/
│   ├── gpt/                 # GPT 模型实现
│   ├── llama/               # LLaMA 模型实现
│   └── mixtral/             # Mixtral MoE 模型实现
└── datasets/
    └── ...                  # 数据加载工具
```

### 6.2 通信与计算重叠

Megatron-Core 的核心设计理念之一是最大化通信与计算的重叠：

1. **张量并行**：Column Parallel 的反向 All-Reduce 与上一层计算重叠
2. **流水线并行**：1F1B 调度使不同 stage 的计算与通信天然重叠
3. **上下文并行**：Ring Attention 在计算当前块注意力的同时传递下一个 K/V 块

### 6.3 模型并行初始化流程

```
1. torch.distributed.init_process_group()
   └── 初始化全局进程组（所有 GPU）

2. mpu.initialize_model_parallel(tp, pp, cp)
   ├── 创建 TP 进程组（相邻 GPU）
   ├── 创建 PP 进程组（跨 TP 组）
   ├── 创建 CP 进程组（跨 TP+PP 组）
   └── DP 组自动确定 = 全局 / (TP × PP × CP)

3. 构建 TransformerBlock
   ├── 根据 PP rank 确定本 stage 负责的层
   ├── 每层创建 TransformerLayer
   │   ├── ParallelSelfAttention（内含 Column+Row Parallel）
   │   └── ParallelMLP（内含 Column+Row Parallel）
   └── 参数按 TP 切分，每个 GPU 只持有 1/tp 的参数
```

### 6.4 前向传播中的数据流

以一个完整的 Transformer 层为例（TP=2, SP=True）：

```
输入 hidden_states: [seq_len/2, batch, hidden]  (SP 切分)

1. Self-Attention 子层:
   a. All-Gather → [seq_len, batch, hidden]     (恢复完整序列，供注意力计算)
   b. LayerNorm → [seq_len, batch, hidden]
   c. QKV Column Parallel → 每个 GPU: [seq_len, batch, hidden*3/2]
   d. Split QKV → Q,K,V 各 [seq_len, batch, hidden/2]
   e. Scaled Dot-Product Attention → [seq_len, batch, hidden/2]
   f. Output Row Parallel + All-Reduce → [seq_len, batch, hidden]
   g. Reduce-Scatter → [seq_len/2, batch, hidden]  (重新 SP 切分)
   h. 残差连接

2. MLP 子层:
   a. All-Gather → [seq_len, batch, hidden]
   b. LayerNorm → [seq_len, batch, hidden]
   c. Column Parallel (SwiGLU) → [seq_len, batch, ffn/2]
   d. Row Parallel + All-Reduce → [seq_len, batch, hidden]
   e. Reduce-Scatter → [seq_len/2, batch, hidden]
   f. 残差连接

输出: [seq_len/2, batch, hidden]
```

---

## 7. 常见注意事项和最佳实践

### 7.1 并行策略选择

| 模型规模 | GPU 数 | 推荐配置 | 说明 |
|---------|--------|----------|------|
| 7B | 8 | TP=8 或 TP=4+DP=2 | 单节点内 TP 即可 |
| 13B | 16 | TP=8+DP=2 | 单节点 TP + 跨节点 DP |
| 70B | 64 | TP=8+PP=4+DP=2 | 需要 PP 突破单节点限制 |
| 175B+ | 256+ | TP=8+PP=8+DP=4 | 多维并行组合 |

**原则**：
- TP 优先在节点内使用（NVLink 带宽高），典型 TP=8 对应 8-GPU 节点
- PP 用于跨节点扩展（通信量小，只需发送激活值）
- DP 用于进一步扩大批次大小

### 7.2 显存优化

- **启用序列并行**：`sequence_parallel=True` 可在 TP>1 时节省 (tp-1)/tp 的激活显存
- **选择性激活重计算**：只重计算注意力部分，MLP 部分保留，在显存和计算间取得平衡
- **FP8 训练**：在 H100/H200 上启用 fp8，显存减半，吞吐翻倍
- **微批次大小**：适当减小 micro_batch_size 以降低单步显存峰值

```python
# 选择性激活重计算配置
config = TransformerConfig(
    # ...
    recompute_granularity="selective",  # "full" 或 "selective"
    recompute_method="block",           # "uniform" 或 "block"
    recompute_num_layers=8,             # 重计算的层数
)
```

### 7.3 1F1B 调度注意事项

- 微批次数 M 应 ≥ stage 数 P，建议 M ≥ 2P 以减少 bubble
- 虚拟流水线并行（Interleaved PP）可将 bubble 减半：每个 stage 交错持有不相邻的层
- PP 会导致首尾 stage 负载不均衡（首尾有 embedding/loss 计算），注意调节

### 7.4 通信优化

- **NCCL 环境变量**：设置 `NCCL_IB_DISABLE=0` 启用 InfiniBand，`NCCL_NET_GDR_LEVEL=5` 启用 GPUDirect RDMA
- **TP 限制在节点内**：TP 通信量大（每层 2 次 All-Reduce），跨节点 TP 性能急剧下降
- **CP 用于超长序列**：Ring Attention 的计算通信重叠特性使 CP 几乎零开销扩展

### 7.5 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| TP 通信超时 | 跨节点 TP 带宽不足 | 将 TP 限制在单节点内 |
| PP 死锁 | M < P 导致流水线无法填充 | 增大微批次数或减小 PP 度 |
| OOM in Activation | 激活显存过大 | 启用 SP、激活重计算、减小 micro_batch |
| 精度下降 | FP8 精度不足 | 使用 hybrid FP8 格式，增大 fp8_amax_history_len |
| 初始化失败 | 并行组配置冲突 | 确保 TP × PP × CP × DP = 总 GPU 数 |

### 7.6 与其他框架集成

- **Megatron-LM 训练框架**：Megatron-Core 的原生训练框架，提供完整训练脚本
- **NeMo**：NVIDIA 的端到端训练框架，基于 Megatron-Core
- **Megatron-DeepSpeed**：结合 DeepSpeed ZeRO 优化的分支
- **自定义训练框架**：可单独导入 Megatron-Core 的并行层和通信原语

```python
# 在自定义框架中使用 Megatron-Core 的并行层
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import TransformerConfig

# 只使用并行层，不依赖完整训练框架
config = TransformerConfig(
    num_layers=1, hidden_size=4096, num_attention_heads=32,
    tensor_model_parallel_size=4, bf16=True,
)
linear = ColumnParallelLinear(4096, 11008, config=config, gather_output=False)
```

---

## 8. 参考资料

- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Megatron-Core 文档](https://docs.nvidia.com/megatron-core/)
- [Megatron-LM 论文: Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)
- [Tensor Parallelism 论文: Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [Sequence Parallelism 论文: Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- [Context Parallelism: Ring Attention](https://arxiv.org/abs/2311.09441)
