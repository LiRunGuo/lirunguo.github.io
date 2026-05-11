---
title: "PyTorch 深度学习框架"
excerpt: "张量操作、autograd、nn.Module、优化器、分布式训练(DDP/FSDP)、混合精度(AMP)、CUDA"
collection: llm-libs
permalink: /llm-libs/01-pytorch
category: core
---


## 1. 简介

PyTorch 是由 Meta AI（原 Facebook AI Research）开发的开源深度学习框架，以其动态计算图、Pythonic 的 API 设计和强大的 GPU 加速能力著称。在 LLM（大语言模型）开发领域，PyTorch 是事实上的标准框架——几乎所有主流 LLM（如 GPT 系列、LLaMA、Mistral、Qwen 等）均基于 PyTorch 构建。

### PyTorch 在 LLM 开发中的核心角色

- **模型构建**：通过 `nn.Module` 定义 Transformer 架构（注意力层、MLP层、嵌入层等）
- **自动求导**：`autograd` 引擎自动计算反向传播梯度，无需手动推导
- **训练优化**：丰富的优化器和学习率调度器支持高效训练
- **分布式训练**：DDP/FSDP 支持多卡、多节点的大模型训练
- **混合精度**：AMP 自动混合精度训练，在保持精度的同时显著提升训练速度
- **GPU 加速**：CUDA 后端提供极致的矩阵运算性能

## 2. 安装

```bash
# CPU 版本
pip install torch

# GPU 版本（CUDA 11.8）
pip install torch --index-url https://download.pytorch.org/whl/cu118

# GPU 版本（CUDA 12.1）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 验证安装
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 3. 核心模块详解

### 3.1 张量操作 (torch.Tensor)

张量（Tensor）是 PyTorch 的核心数据结构，类似于 NumPy 的 ndarray，但支持 GPU 加速和自动求导。

#### 3.1.1 张量创建

```python
import torch

# 从列表创建
a = torch.tensor([1.0, 2.0, 3.0])  # 1D 张量
b = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # 2D 张量，指定类型

# 常用工厂函数
zeros = torch.zeros(3, 4)        # 全零张量，形状 (3, 4)
ones = torch.ones(2, 3, 5)       # 全一张量
empty = torch.empty(2, 3)        # 未初始化张量（值不确定）
rand = torch.rand(2, 3)          # 均匀分布 [0, 1) 随机张量
randn = torch.randn(2, 3)        # 标准正态分布 N(0,1) 随机张量
arange = torch.arange(0, 10, 2)  # 等差序列：[0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # 等间隔序列：[0, 0.25, 0.5, 0.75, 1.0]

# LLM 中常用的初始化方法
# Xavier/Glorot 初始化（适用于 Sigmoid/Tanh）
xavier = torch.empty(512, 512)
torch.nn.init.xavier_uniform_(xavier)

# Kaiming/He 初始化（适用于 ReLU，Transformer 中常用）
kaiming = torch.empty(512, 512)
torch.nn.init.kaiming_normal_(kaiming, mode='fan_in', nonlinearity='relu')

# 从 NumPy 转换
import numpy as np
np_array = np.array([1, 2, 3])
tensor_from_np = torch.from_numpy(np_array)  # 共享内存
np_from_tensor = a.numpy()  # CPU 张量转 NumPy，共享内存
```

#### 3.1.2 张量属性与变换

```python
x = torch.randn(2, 3, 4)

# 基本属性
x.shape       # torch.Size([2, 3, 4]) — 形状
x.dtype       # torch.float32 — 数据类型
x.device      # cpu 或 cuda:0 — 所在设备
x.ndim        # 3 — 维度数
x.numel()     # 24 — 元素总数

# 形状变换
x.view(2, 12)       # 返回新视图（共享内存），要求张量连续
x.reshape(2, 12)    # 优先返回视图，必要时拷贝
x.permute(2, 0, 1)  # 维度重排：(2,3,4) → (4,2,3)
x.transpose(0, 1)   # 交换两个维度
x.unsqueeze(0)      # 在维度0插入：(2,3,4) → (1,2,3,4)
x.squeeze()         # 移除所有大小为1的维度

# LLM 中的典型用法：注意力掩码维度扩展
# attention_mask: (batch_size, seq_len) → (batch_size, 1, 1, seq_len)
attention_mask = torch.ones(4, 128)  # batch=4, seq_len=128
expanded_mask = attention_mask[:, None, None, :]  # (4, 1, 1, 128)
```

#### 3.1.3 索引与切片

```python
x = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

x[0]          # 第一行：[0, 1, 2, 3]
x[:, 1]       # 第二列：[1, 5, 9]
x[0:2, 1:3]   # 子矩阵：[[1,2],[5,6]]

# 高级索引
indices = torch.tensor([0, 2])
x[:, indices]     # 选取第0和第2列
x[x > 5]          # 布尔索引：[6, 7, 8, 9, 10, 11]

# LLM 常用：选取特定 token 的嵌入
# hidden_states: (batch, seq_len, hidden_dim)
hidden_states = torch.randn(2, 10, 768)
# 获取每个序列最后一个 token 的表示
last_token = hidden_states[:, -1, :]  # (2, 768)
```

#### 3.1.4 广播机制

PyTorch 广播规则与 NumPy 一致：

1. 从最右边的维度开始对齐
2. 两个维度大小相同，或其中一个为1，即可广播
3. 缺失的维度视为大小为1

```python
# (3, 1) + (1, 4) → (3, 4)
a = torch.ones(3, 1)
b = torch.ones(1, 4)
c = a + b  # 形状 (3, 4)

# LLM 典型场景：RoPE 位置编码
# freqs: (seq_len, head_dim/2)  →  扩展为 (1, seq_len, 1, head_dim/2)
# query: (batch, seq_len, num_heads, head_dim)
freqs = torch.randn(128, 64)
freqs = freqs[None, :, None, :]  # (1, 128, 1, 64) — 自动广播到 batch 和 num_heads 维度
```

### 3.2 自动求导 (autograd)

`autograd` 是 PyTorch 自动微分引擎的核心，它通过动态构建计算图来实现自动反向传播。

#### 3.2.1 基本用法

```python
# requires_grad=True 标记需要计算梯度的张量
x = torch.tensor([2.0, 3.0], requires_grad=True)

# 前向计算（自动构建计算图）
y = x ** 2        # y = [4, 9]
z = y.sum()       # z = 13

# 反向传播
z.backward()

# 查看梯度
print(x.grad)     # dz/dx = 2x = [4, 6]
```

#### 3.2.2 计算图与梯度累积

PyTorch 采用动态计算图（Define-by-Run），每次前向传播时构建新的计算图。

```python
# 梯度累积：默认情况下梯度会累加
x = torch.tensor([2.0], requires_grad=True)

for _ in range(3):
    y = x ** 2
    y.backward()
    print(x.grad)  # 4.0 → 8.0 → 12.0（梯度累积！）

# 正确做法：每次反向传播前清零梯度
x = torch.tensor([2.0], requires_grad=True)
for _ in range(3):
    y = x ** 2
    if x.grad is not None:
        x.grad.zero_()  # 清零梯度
    y.backward()
    print(x.grad)  # 4.0 → 4.0 → 4.0（正确）
```

#### 3.2.3 梯度控制

```python
x = torch.tensor([2.0], requires_grad=True)

# no_grad：推理时禁用梯度计算，节省内存
with torch.no_grad():
    y = x ** 2  # y.requires_grad = False

# detach：从计算图中分离张量
y = x ** 2
z = y.detach()  # z 不再跟踪梯度，但与 y 共享数据

# torch.enable_grad：在 no_grad 内部临时启用梯度
with torch.no_grad():
    with torch.enable_grad():
        y = x ** 2  # 此处可以计算梯度
```

#### 3.2.4 LLM 中的 autograd 应用

```python
# 典型训练循环中的梯度管理
model = ...  # nn.Module
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    # 1. 前向传播
    outputs = model(batch['input_ids'])
    loss = loss_fn(outputs, batch['labels'])

    # 2. 反向传播
    optimizer.zero_grad()    # 清零梯度（等价于 for p in model.parameters(): p.grad=None）
    loss.backward()          # 计算梯度

    # 3. 梯度裁剪（防止梯度爆炸，LLM 训练中常用）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 4. 更新参数
    optimizer.step()
```

### 3.3 nn.Module

`nn.Module` 是所有 PyTorch 模型的基类，提供了参数管理、子模块注册、设备迁移等核心功能。

#### 3.3.1 自定义模型

```python
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()  # 必须调用父类初始化
        # 在 __init__ 中定义所有层和参数
        self.fc1 = nn.Linear(input_dim, hidden_dim)   # 全连接层
        self.activation = nn.GELU()                     # 激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.1)               # Dropout 正则化

    def forward(self, x):
        # 在 forward 中定义前向计算逻辑
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 实例化并使用
model = SimpleMLP(input_dim=768, hidden_dim=3072, output_dim=768)
output = model(torch.randn(2, 10, 768))  # (batch, seq_len, output_dim)
```

#### 3.3.2 参数管理

```python
# 访问所有参数
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")

# 访问特定层的参数
print(model.fc1.weight.shape)  # (hidden_dim, input_dim) = (3072, 768)
print(model.fc1.bias.shape)    # (hidden_dim,) = (3072,)

# 冻结参数（LLM 微调中常用：冻结预训练层，只训练新层）
for name, param in model.named_parameters():
    if 'fc1' in name:  # 只冻结 fc1 层
        param.requires_grad = False

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}")
```

#### 3.3.3 Transformer 核心组件示例

```python
class MultiHeadSelfAttention(nn.Module):
    """简化版多头自注意力，展示 nn.Module 的典型用法"""
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape

        # 线性投影并分头
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scale = self.head_dim ** 0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)
```

### 3.4 优化器

优化器负责根据梯度更新模型参数。在 LLM 训练中，选择合适的优化器对收敛速度和最终性能至关重要。

#### 3.4.1 SGD

```python
# 随机梯度下降（带动量和权重衰减）
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,           # 学习率
    momentum=0.9,      # 动量系数，加速收敛
    weight_decay=1e-4  # L2 正则化（权重衰减）
)
```

#### 3.4.2 Adam

```python
# Adam 优化器（LLM 中最常用的优化器之一）
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,           # 学习率
    betas=(0.9, 0.999),  # (β1, β2) 一阶和二阶动量衰减系数
    eps=1e-8,          # 防止除零的小常数
    weight_decay=0.0   # 权重衰减
)
```

#### 3.4.3 AdamW（推荐用于 LLM）

```python
# AdamW：解耦权重衰减的 Adam（LLM 训练的标准选择）
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),   # LLM 中 β2 常用 0.95 而非 0.999
    eps=1e-8,
    weight_decay=0.01,   # 权重衰减系数
    amsgrad=False        # 是否使用 AMSGrad 变体
)
```

#### 3.4.4 学习率调度器

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# 方式1：余弦退火（LLM 训练中最常用）
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100000,    # 周期长度（通常等于总训练步数）
    eta_min=1e-5     # 最小学习率
)

# 方式2：线性预热 + 余弦退火（LLM 标准配置）
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=2000)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=98000, eta_min=1e-5)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[2000]  # 在第 2000 步切换调度器
)

# 训练循环中使用
for step, batch in enumerate(dataloader):
    loss = model(batch).loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()  # 每个 step 后更新学习率
```

### 3.5 分布式训练 (torch.distributed)

#### 3.5.1 DistributedDataParallel (DDP)

DDP 是最常用的单机多卡/多机多卡训练方式，每个 GPU 持有完整的模型副本。

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp():
    dist.init_process_group(backend="nccl")  # NCCL 后端用于 GPU 通信
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def main():
    setup_ddp()
    local_rank = int(os.environ["LOCAL_RANK"])

    model = SimpleMLP(768, 3072, 768).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for batch in dataloader:
        batch = {k: v.cuda(local_rank) for k, v in batch.items()}
        output = model(batch['input_ids'])
        loss = output.loss
        optimizer.zero_grad()
        loss.backward()        # DDP 自动同步梯度
        optimizer.step()

    dist.destroy_process_group()

# 启动命令：torchrun --nproc_per_node=4 train.py
```

#### 3.5.2 FullyShardedDataParallel (FSDP)

FSDP 将模型参数、梯度和优化器状态分片到各 GPU 上，支持训练远大于单卡显存的模型。

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = SimpleMLP(768, 3072, 768).cuda(local_rank)

# FULL_SHARD：完全分片（最省显存，通信量最大）
# SHARD_GRAD_OP：仅分片梯度和优化器状态（折中方案）
# NO_SHARD：等同于 DDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=local_rank
)
```

### 3.6 混合精度训练 (torch.cuda.amp)

混合精度训练使用 FP16/BF16 进行前向和反向传播，用 FP32 维护主权重，在几乎不损失精度的前提下大幅提升训练速度。

#### 3.6.1 自动混合精度

```python
from torch.cuda.amp import autocast, GradScaler

model = SimpleMLP(768, 3072, 768).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()  # 梯度缩放器，防止 FP16 梯度下溢

for batch in dataloader:
    batch = {k: v.cuda() for k, v in batch.items()}
    optimizer.zero_grad()

    # autocast 上下文管理器：自动选择 FP16/BF16 进行计算
    with autocast(dtype=torch.bfloat16):  # 或 torch.float16
        output = model(batch['input_ids'])
        loss = output.loss

    # 缩放损失 → 反向传播 → 反缩放梯度 → 更新参数
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # 反缩放，以便正确裁剪梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)      # 如果梯度中含 inf/nan 则跳过此步
    scaler.update()             # 更新缩放因子
```

#### 3.6.2 BF16 vs FP16

```python
# BF16 (bfloat16)：指数位8位，尾数位7位 — 与 FP32 相同的数值范围
# FP16 (float16)：指数位5位，尾数位10位 — 更高精度但数值范围小

# BF16 不需要 GradScaler（不易溢出）
with autocast(dtype=torch.bfloat16):
    output = model(inputs)
    loss = output.loss
loss.backward()  # 直接反向传播，无需 scaler
optimizer.step()

# FP16 需要 GradScaler（容易溢出/下溢）
# 如上 3.6.1 示例所示
```

### 3.7 CUDA 操作

#### 3.7.1 设备管理

```python
# 查看可用 GPU
print(torch.cuda.device_count())    # GPU 数量
print(torch.cuda.current_device())  # 当前设备 ID

# 张量设备迁移
x = torch.randn(2, 3)
x_gpu = x.to('cuda')             # CPU → GPU
x_gpu = x.cuda()                 # 等价写法
x_cpu = x_gpu.to('cpu')          # GPU → CPU
x_gpu1 = x_gpu.to('cuda:1')     # GPU 0 → GPU 1

# 模型设备迁移
model = SimpleMLP(768, 3072, 768)
model = model.to('cuda')
```

#### 3.7.2 显存管理

```python
# 查看显存使用情况
print(torch.cuda.memory_allocated() / 1e9, "GB")      # 当前张量占用的显存
print(torch.cuda.memory_reserved() / 1e9, "GB")       # CUDA 缓存池占用的显存
print(torch.cuda.max_memory_allocated() / 1e9, "GB")  # 峰值显存

# 清理显存缓存
torch.cuda.empty_cache()  # 释放未使用的缓存（不影响仍在使用的张量）

# 手动垃圾回收
import gc
del large_tensor
gc.collect()
torch.cuda.empty_cache()
```

## 4. 数学原理

### 4.1 反向传播算法

反向传播是训练神经网络的核心算法，基于链式法则（Chain Rule）高效计算损失函数对每个参数的梯度。

**前向传播**：
```
z₁ = W₁x + b₁          →  a₁ = σ(z₁)
z₂ = W₂a₁ + b₂         →  a₂ = σ(z₂)
L = ℓ(a₂, y)            （损失函数）
```

**反向传播**（链式法则）：
```
∂L/∂a₂ = ∂ℓ/∂a₂
∂L/∂z₂ = ∂L/∂a₂ · σ'(z₂)
∂L/∂W₂ = ∂L/∂z₂ · a₁ᵀ
∂L/∂b₂ = ∂L/∂z₂
∂L/∂a₁ = W₂ᵀ · ∂L/∂z₂
∂L/∂z₁ = ∂L/∂a₁ · σ'(z₁)
∂L/∂W₁ = ∂L/∂z₁ · xᵀ
∂L/∂b₁ = ∂L/∂z₁
```

PyTorch 的 `autograd` 引擎自动执行以上计算，用户只需调用 `loss.backward()`。

### 4.2 Adam 优化器的动量更新公式

Adam 结合了一阶动量（均值）和二阶动量（方差）的自适应学习率：

```
初始化：m₀ = 0, v₀ = 0, t = 0

每步更新：
  t = t + 1
  gₜ = ∇θ L(θₜ₋₁)                          （当前梯度）
  mₜ = β₁ · mₜ₋₁ + (1 - β₁) · gₜ           （一阶动量：梯度的指数移动平均）
  vₜ = β₂ · vₜ₋₁ + (1 - β₂) · gₜ²          （二阶动量：梯度平方的指数移动平均）
  m̂ₜ = mₜ / (1 - β₁ᵗ)                       （偏差修正）
  v̂ₜ = vₜ / (1 - β₂ᵗ)                       （偏差修正）
  θₜ = θₜ₋₁ - η · m̂ₜ / (√v̂ₜ + ε)           （参数更新）
```

**AdamW 的关键区别**：将权重衰减从梯度计算中解耦：
```
θₜ = θₜ₋₁ - η · (m̂ₜ / (√v̂ₜ + ε) + λ · θₜ₋₁)
```
其中 λ 是权重衰减系数。这种解耦方式使得权重衰减的效果更加稳定，不受梯度自适应缩放的影响。

### 4.3 混合精度的 FP16/BF16 数值范围

| 格式 | 指数位 | 尾数位 | 最大值 | 最小正规数 | 精度（epsilon） |
|------|--------|--------|--------|-----------|-----------------|
| FP32 | 8 | 23 | 3.4×10³⁸ | 1.2×10⁻³⁸ | 1.19×10⁻⁷ |
| FP16 | 5 | 10 | 65504 | 6.1×10⁻⁵ | 9.77×10⁻⁴ |
| BF16 | 8 | 7 | 3.4×10³⁸ | 1.2×10⁻³⁸ | 3.91×10⁻³ |

- **FP16**：尾数精度高但范围小，容易溢出（>65504）或下溢（<6.1e-5），需要 GradScaler
- **BF16**：范围与 FP32 相同，不易溢出，但尾数精度较低；现代 GPU（A100/H100）原生支持
- **混合精度原理**：用低精度（FP16/BF16）做前向和反向传播以加速，用 FP32 维护主权重和优化器状态以保持精度

## 5. 在 LLM 开发中的典型使用场景

### 5.1 从零构建 GPT-2 风格模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPT2Block(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attention_mask=None):
        # Pre-LN Transformer
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, attn_mask=attention_mask, need_weights=False)
        x = residual + x

        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x
        return x

class GPT2Model(nn.Module):
    def __init__(self, vocab_size=50257, hidden_dim=768, num_layers=12,
                 num_heads=12, ffn_dim=3072, max_seq_len=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.blocks = nn.ModuleList([
            GPT2Block(hidden_dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # 权重共享
        self.lm_head.weight = self.token_embedding.weight

        # 因果掩码
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        for block in self.blocks:
            x = block(x, attention_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# 使用示例
model = GPT2Model()
input_ids = torch.randint(0, 50257, (2, 128))  # batch=2, seq_len=128
logits = model(input_ids)
print(logits.shape)  # (2, 128, 50257)
```

### 5.2 参数高效微调（LoRA）

```python
class LoRALinear(nn.Module):
    """LoRA：低秩适配器，冻结原始权重，仅训练低秩矩阵"""
    def __init__(self, original_linear, r=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad = False  # 冻结原始权重
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features

        self.lora_A = nn.Parameter(torch.randn(in_dim, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_dim))
        self.scaling = alpha / r

    def forward(self, x):
        # W'x = Wx + (B @ A) * scaling * x
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

# 应用 LoRA 到模型
model = GPT2Model()
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and 'q_proj' in name:
        # 替换 Q 投影为 LoRA 版本
        parent_name, child_name = name.rsplit('.', 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(module, r=8, alpha=16))
```

## 6. 代码原理 / 架构原理

### 6.1 PyTorch 执行模型

PyTorch 采用 **Eager Mode（急切模式）** 执行：

1. **前向传播**：每个操作立即执行，同时构建计算图
2. **反向传播**：沿计算图反向遍历，应用链式法则计算梯度
3. **计算图是动态的**：每次前向传播都构建新图，支持条件分支、循环等动态控制流

### 6.2 autograd 引擎架构

```
前向传播 → 构建 autograd 计算图（DAG）
                ↓
每个节点是一个 Function 对象，记录：
  - forward 的输入/输出
  - backward 的计算方式
                ↓
loss.backward() → 反向遍历 DAG：
  1. 从 loss 节点开始
  2. 按拓扑排序反向遍历
  3. 每个节点调用 backward 计算局部梯度
  4. 通过链式法则累积梯度
```

### 6.3 DDP 通信原理

```
GPU 0: Model Replica → Forward → Backward → AllReduce (梯度同步)
GPU 1: Model Replica → Forward → Backward → AllReduce (梯度同步)
GPU 2: Model Replica → Forward → Backward → AllReduce (梯度同步)
GPU 3: Model Replica → Forward → Backward → AllReduce (梯度同步)

AllReduce 操作：
  1. Reduce-Scatter：各 GPU 的梯度求和后分片
  2. All-Gather：收集所有分片，每 GPU 获得完整平均梯度
```

## 7. 常见注意事项和最佳实践

### 7.1 显存优化

```python
# 1. 使用梯度检查点（用计算换显存）
from torch.utils.checkpoint import checkpoint

class GPT2BlockWithCheckpoint(nn.Module):
    def forward(self, x):
        # 不保存中间激活，反向传播时重新计算
        return checkpoint(self._forward, x)

    def _forward(self, x):
        # 原始前向逻辑
        ...

# 2. 使用 mixed precision 减少显存占用
with autocast(dtype=torch.bfloat16):
    output = model(inputs)  # 激活值和梯度用 BF16 存储，节省约 50% 显存

# 3. 及时释放不需要的张量
del output, loss
torch.cuda.empty_cache()
```

### 7.2 训练稳定性

```python
# 1. 梯度裁剪（LLM 训练必备）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. 学习率预热（防止训练初期梯度爆炸）
scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=2000)

# 3. 使用 BF16 代替 FP16（避免精度问题）
with autocast(dtype=torch.bfloat16):  # 推荐，A100/H100 原生支持
    ...

# 4. 检测 NaN/Inf
if torch.isnan(loss) or torch.isinf(loss):
    print("警告：检测到异常损失值，跳过此步")
    continue
```

### 7.3 数据加载最佳实践

```python
from torch.utils.data import DataLoader, DistributedSampler

# 分布式训练中必须使用 DistributedSampler
sampler = DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(
    dataset,
    batch_size=8,            # 每卡 batch size
    sampler=sampler,         # 分布式采样器
    num_workers=4,           # 数据加载线程数
    pin_memory=True,         # 锁页内存，加速 CPU→GPU 传输
    drop_last=True,          # 丢弃最后不完整的 batch
)

# 每个 epoch 需要设置 sampler 的 epoch
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in dataloader:
        ...
```

### 7.4 模型保存与加载

```python
# 保存完整模型（不推荐，不灵活）
torch.save(model, 'model.pth')

# 推荐方式：只保存 state_dict
torch.save(model.state_dict(), 'model_state_dict.pth')

# 加载
model = GPT2Model()
model.load_state_dict(torch.load('model_state_dict.pth'))
model.eval()  # 切换到评估模式

# 分布式训练中保存（只由 rank 0 保存）
if dist.get_rank() == 0:
    torch.save(model.module.state_dict(), 'model.pth')
```

### 7.5 常见陷阱

1. **忘记 `optimizer.zero_grad()`**：导致梯度累积，训练不稳定
2. **在 `no_grad` 中创建的叶子张量设 `requires_grad=True`**：梯度不会被计算
3. **就地修改张量**：可能破坏计算图，导致 `backward()` 报错
4. **不同设备上的张量运算**：CPU 和 GPU 张量不能直接运算，需先统一设备
5. **DDP 中只保存 rank 0 的模型**：避免各 rank 同时写文件造成冲突
6. **忽略 `model.eval()` 和 `model.train()`**：Dropout 和 BatchNorm 在训练/推理时行为不同
