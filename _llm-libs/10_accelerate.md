---
title: "Accelerate 分布式训练库"
excerpt: "Accelerator、DDP/FSDP/DeepSpeed集成、device_map自动分片、混合精度"
collection: llm-libs
permalink: /llm-libs/10-accelerate
category: training
---


## 1. 简介与在LLM开发中的作用

### 1.1 什么是 Accelerate

Accelerate 是 HuggingFace 开发的一个轻量级库，旨在简化混合精度训练和分布式训练的代码编写。它提供了一套统一的 API，屏蔽了底层分布式框架（PyTorch DDP、FSDP、DeepSpeed）的差异，让同一份代码无需修改即可在不同硬件配置上运行——从单GPU到多GPU多节点集群。

### 1.2 在LLM开发中的核心作用

- **消除分布式训练样板代码**：传统分布式训练需要处理 `DistributedDataParallel`、`distributed sampler`、`gradient scaling` 等繁琐逻辑，Accelerate 将其统一封装
- **大模型加载与设备映射**：通过 `device_map="auto"` 实现大模型自动分片到多GPU/CPU，解决单卡显存不足问题
- **训练策略快速切换**：一行配置即可在 DDP/FSDP/DeepSpeed 之间切换，方便对比不同并行策略的效果
- **混合精度训练**：自动管理 fp16/bf16 的 loss scaling 和类型转换
- **与 Transformers 生态深度集成**：Trainer 类原生支持 Accelerate 配置

---

## 2. 安装方式

```bash
# 基础安装
pip install accelerate

# 安装 DeepSpeed 集成支持
pip install accelerate[deepspeed]

# 安装 TPU 支持
pip install accelerate[tpu]

# 完整安装
pip install accelerate[deepspeed,tpu]

# 从源码安装
pip install git+https://github.com/huggingface/accelerate
```

验证安装：

```bash
accelerate env  # 查看当前环境信息
```

---

## 3. 核心类与函数详细说明

### 3.1 Accelerator 类

`Accelerator` 是 Accelerate 的核心类，负责管理训练过程中的设备分配、混合精度、梯度同步等。

#### `__init__` 参数详解

```python
from accelerate import Accelerator

accelerator = Accelerator(
    device_placement=True,          # 是否自动将对象放置到正确设备
    split_batches=False,            # 是否在进程间拆分批次（而非复制）
    mixed_precision="no",           # 混合精度模式："no"/"fp16"/"bf16"
    gradient_accumulation_steps=1,  # 梯度累积步数
    step_scheduler_with_optimizer=True,  # 是否与优化器同步调度器
    cpu=False,                      # 强制在CPU上运行
    deepspeed_plugin=None,          # DeepSpeed 插件配置
    fsdp_plugin=None,               # FSDP 插件配置
    rng_types=None,                 # 要同步的随机数生成器类型
    log_with=None,                  # 日志记录器："tensorboard"/"wandb"/"comet_ml"
    project_config=None,            # 项目配置（如目录路径）
    dispatch_batches=False,         # 是否在一个进程上准备批次后分发
    even_batches=True,              # 确保所有进程的批次大小一致
)
```

**关键参数说明**：

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `mixed_precision` | 混合精度训练模式，fp16适用于大多数GPU，bf16适用于Ampere+架构 | `"fp16"` / `"bf16"` |
| `gradient_accumulation_steps` | 模拟更大batch_size，每N步才执行一次优化器更新 | `4` / `8` / `16` |
| `split_batches` | True时数据加载器的batch_size是全局大小，自动拆分到各进程 | `False` |
| `device_placement` | False时需要手动 `.to(device)` | `True` |

#### `prepare()` 方法

将模型、优化器、数据加载器和调度器包装为分布式版本：

```python
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)
```

**参数**：接受任意数量的 PyTorch 对象（`nn.Module`、`Optimizer`、`DataLoader`、`LRScheduler`）

**返回值**：与输入相同数量和顺序的包装后对象。模型被包装为 `DistributedDataParallel`（或 FSDP/DeepSpeed 等效物），数据加载器自动使用 `DistributedSampler`

**注意**：调用 `prepare()` 后，模型参数和优化器状态会被移动到正确的设备上，无需手动 `.to(device)`

#### `backward()` 方法

替代 `loss.backward()`，自动处理混合精度的 loss scaling：

```python
loss = model(inputs)
accelerator.backward(loss)  # 替代 loss.backward()
```

**参数**：
- `loss`：需要反向传播的损失张量
- `retain_graph`：是否保留计算图（同 `torch.autograd.backward`）

#### `save_model()` 方法

安全地保存模型，只在主进程上保存，避免多进程写入冲突：

```python
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
accelerator.save_model(unwrapped_model, save_directory, safe_serialization=True)
```

**参数**：
- `model`：要保存的模型（建议先 `unwrap_model`）
- `save_directory`：保存目录
- `safe_serialization`：是否使用 safetensors 格式（推荐True，避免pickle安全风险）

#### 其他常用方法

```python
# 等待所有进程同步
accelerator.wait_for_everyone()

# 解包DDP模型获取原始模型
raw_model = accelerator.unwrap_model(model)

# 检查是否为主进程（用于日志/保存）
if accelerator.is_main_process:
    print("Only main process logs")

# 获取当前进程编号和总进程数
accelerator.process_index   # 当前进程编号
accelerator.num_processes   # 总进程数

# 检查是否为本地主进程
accelerator.is_local_main_process

# 获取当前设备
accelerator.device  # torch.device("cuda:0") 等

# 梯度累积相关
accelerator.gradient_state  # 梯度累积状态
accelerator.sync_gradients  # 是否同步梯度（在累积中间步为False）

# 聚合跨进程的值
accelerator.gather(tensor)          # 聚合所有进程的张量
accelerator.reduce(tensor, reduction="mean")  # 跨进程规约

# 打印（只在主进程）
accelerator.print("Hello")  # 等价于 if is_main_process: print()

# 剪辑梯度（替代torch.nn.utils.clip_grad_norm_）
accelerator.clip_grad_norm_(model, max_norm=1.0)
accelerator.clip_grad_value_(model, clip_value=1.0)

# 获取状态用于断点续训
accelerator.get_state_dict(model)
accelerator.save_state(save_directory)      # 保存完整训练状态
accelerator.load_state(load_directory)      # 恢复训练状态
```

### 3.2 完整训练循环示例

```python
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

# 1. 初始化 Accelerator
accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=4,
)

# 2. 准备模型和数据
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer,
    num_warmup_steps=100, num_training_steps=num_training_steps
)

# 3. prepare() 包装所有对象
model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

# 4. 训练循环
model.train()
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss

        # 梯度累积：每 gradient_accumulation_steps 步才更新一次
        loss = loss / accelerator.gradient_accumulation_steps
        accelerator.backward(loss)

        if step % accelerator.gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    # 评估
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        # 聚合所有进程的预测结果
        all_predictions = accelerator.gather(predictions)
        all_labels = accelerator.gather(batch["labels"])

    # 只在主进程保存模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(f"./model_epoch_{epoch}")
```

### 3.3 混合精度训练

#### fp16 混合精度

fp16（半精度浮点数）将部分计算从 fp32 降至 fp16，减少显存占用并加速计算：

```python
accelerator = Accelerator(mixed_precision="fp16")
```

**数学原理**：
- fp16 范围：±65504，精度约 3位十进制
- fp32 范围：±3.4×10³⁸，精度约 7位十进制
- 动态损失缩放（Dynamic Loss Scaling）：为避免 fp16 梯度下溢，将 loss 乘以缩放因子 `S`，反向传播后梯度为 `S × ∂L/∂W`，更新前除以 `S` 恢复。若出现 inf/nan，跳过该步并减小 `S`

```
L_scaled = L × S
∂L_scaled/∂W = S × ∂L/∂W
W_update = W - lr × (∂L_scaled/∂W) / S
```

#### bf16 混合精度

```python
accelerator = Accelerator(mixed_precision="bf16")
```

**bf16 vs fp16**：
- bf16 动态范围与 fp32 相同（8位指数），但精度较低（7位尾数 vs fp16的10位尾数）
- bf16 不需要 loss scaling，因为其动态范围足够大
- bf16 需要 Ampere 架构及以上（A100、RTX 30/40系列等）

### 3.4 梯度累积

梯度累积是一种在显存受限时模拟大 batch_size 的技术：

```python
accelerator = Accelerator(gradient_accumulation_steps=8)
```

**原理**：不每步都更新参数，而是累积多步的梯度后再执行一次优化器更新。等效 batch_size = `per_device_batch_size × gradient_accumulation_steps × num_processes`

```python
for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss / accelerator.gradient_accumulation_steps  # 除以累积步数以归一化
    accelerator.backward(loss)

    # 检查是否到了累积完成步
    if (step + 1) % accelerator.gradient_accumulation_steps == 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

更推荐使用 Accelerate 提供的上下文管理器：

```python
for step, batch in enumerate(dataloader):
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

`accelerator.accumulate(model)` 会自动处理梯度累积逻辑，在累积未完成时跳过 `optimizer.step()` 和 `optimizer.zero_grad()`。

### 3.5 DeepSpeed 集成

DeepSpeed 提供了 ZeRO（Zero Redundancy Optimizer）优化，将模型状态分片到多个设备上：

```python
from accelerate import Accelerator, DeepSpeedPlugin

# 方式1：通过 DeepSpeedPlugin 配置
deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=2,              # ZeRO 阶段：0/1/2/3
    offload_optimizer_device="cpu",  # 将优化器状态卸载到CPU
    offload_param_device="none",     # 将参数卸载到设备
    gradient_accumulation_steps=4,
)

accelerator = Accelerator(
    mixed_precision="fp16",
    deepspeed_plugin=deepspeed_plugin,
)
```

**ZeRO 阶段说明**：

| 阶段 | 分片内容 | 显存节省 | 通信开销 |
|------|----------|----------|----------|
| ZeRO-0 | 无分片（等同DDP） | 基线 | 最低 |
| ZeRO-1 | 优化器状态分片 | ~4x | 低 |
| ZeRO-2 | 优化器状态+梯度分片 | ~8x | 中等 |
| ZeRO-3 | 优化器状态+梯度+参数分片 | 与GPU数线性相关 | 高 |

**方式2：通过配置文件**

```json
// ds_config.json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "fp16": {
        "enabled": true
    }
}
```

```bash
accelerate launch --use_deepspeed --deepspeed_config_file ds_config.json train.py
```

### 3.6 FSDP 集成

FSDP（Fully Sharded Data Parallel）是 PyTorch 原生的全分片数据并行方案：

```python
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import FullStateDictConfig, ShardingStrategy

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片
    mixed_precision_policy=None,
    auto_wrap_policy=None,          # 自动包装策略
    limit_all_gathers=True,        # 限制all-gather并发
    state_dict_type=FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    ),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```

**FSDP 与 DeepSpeed ZeRO-3 对比**：
- FSDP 是 PyTorch 原生实现，与 PyTorch 生态集成更好
- DeepSpeed ZeRO-3 提供更多优化（如 NVMe offload），但引入额外依赖
- 两者在功能上类似，都是全分片方案

### 3.7 device_map 与大模型加载

#### device_map 参数

在 Transformers 中通过 `device_map` 控制模型层到设备的映射：

```python
from transformers import AutoModelForCausalLM

# 自动映射：根据各设备可用显存自动分配
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    device_map="auto",              # 自动分配
    torch_dtype=torch.float16,
)

# 均衡映射：尽可能均匀分布
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    device_map="balanced",          # 均衡分布
)

# 顺序映射：依次填满设备
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    device_map="sequential",        # 顺序填充
)

# 自定义映射：手动指定每层所在设备
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.h.0": 0,
    "transformer.h.1": 0,
    "transformer.h.2": 1,
    "transformer.h.3": 1,
    "transformer.ln_f": 1,
    "lm_head": 1,
}
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    device_map=device_map,
)
```

**查看设备映射结果**：

```python
print(model.hf_device_map)
# 输出示例：{'transformer.word_embeddings': 0, 'transformer.h.0': 0, ...}
```

#### max_memory 参数

控制每个设备最多使用多少显存：

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    max_memory={
        0: "20GiB",       # GPU 0 最多使用 20GB
        1: "20GiB",       # GPU 1 最多使用 20GB
        "cpu": "60GiB",   # CPU 内存最多使用 60GB
    },
    torch_dtype=torch.float16,
)
```

#### offload 到 CPU/Disk

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8bit量化 + 自动映射，适合单卡加载大模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    device_map="auto",
    load_in_8bit=True,    # 8bit量化加载
)

# 4bit量化（QLoRA）
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    device_map="auto",
    quantization_config=quantization_config,
)
```

### 3.8 notebook_launcher

在 Jupyter Notebook 中启动分布式训练：

```python
from accelerate import notebook_launcher

def training_function():
    from accelerate import Accelerator
    accelerator = Accelerator()
    # ... 训练代码 ...

# 在notebook中启动2个进程的分布式训练
notebook_launcher(training_function, args=(), num_processes=2)
```

**参数**：
- `function`：训练函数（无参或接受 `args` 参数）
- `args`：传递给训练函数的参数元组
- `num_processes`：启动的进程数
- `mixed_precision`：混合精度模式
- `fp8_backend`：FP8 后端

### 3.9 accelerate launch 命令

`accelerate launch` 是启动分布式训练的命令行工具，替代 `torchrun`/`torch.distributed.launch`：

```bash
# 使用默认配置启动
accelerate launch train.py

# 指定GPU数量
accelerate launch --num_processes 4 train.py

# 使用混合精度
accelerate launch --mixed_precision fp16 train.py

# 使用DeepSpeed
accelerate launch --use_deepspeed --zero_stage 2 train.py

# 使用FSDP
accelerate launch --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED train.py

# 多节点启动
# 主节点
accelerate launch --main_process_ip 192.168.1.1 --main_process_port 29500 \
    --num_processes 4 --num_machines 2 --machine_rank 0 train.py
# 从节点
accelerate launch --main_process_ip 192.168.1.1 --main_process_port 29500 \
    --num_processes 4 --num_machines 2 --machine_rank 1 train.py
```

### 3.10 accelerate config 命令

交互式配置训练环境：

```bash
accelerate config
```

会依次询问：
1. 计算平台（No distributed / multi-GPU / TPU）
2. GPU 数量
3. 是否使用混合精度及类型
4. 是否使用 DeepSpeed 及配置
5. 是否使用 FSDP 及配置

配置完成后，`accelerate launch` 会自动使用该配置。

```bash
# 查看当前配置
accelerate config show

# 查看环境信息
accelerate env
```

也可以通过 Python 代码查看配置：

```python
from accelerate import Accelerator
accelerator = Accelerator()
print(accelerator.state)  # 打印当前分布式状态
```

---

## 4. 在LLM开发中的典型使用场景

### 4.1 场景一：单机多卡微调LLM

```python
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader

accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=8,
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=50, num_training_steps=1000)

model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

model.train()
for step, batch in enumerate(train_dataloader):
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    if step % 100 == 0 and accelerator.is_main_process:
        accelerator.print(f"Step {step}, Loss: {loss.item():.4f}")
```

启动命令：
```bash
accelerate launch --num_processes 4 --mixed_precision bf16 train.py
```

### 4.2 场景二：DeepSpeed ZeRO-3 训练超大模型

```python
from accelerate import Accelerator, DeepSpeedPlugin

deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=3,
    offload_optimizer_device="cpu",
    offload_param_device="cpu",   # 参数也卸载到CPU，进一步节省GPU显存
    gradient_accumulation_steps=16,
)

accelerator = Accelerator(
    mixed_precision="fp16",
    deepspeed_plugin=deepspeed_plugin,
)

# ZeRO-3下可以加载超过单卡显存的模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")
# ... 训练代码 ...
```

### 4.3 场景三：device_map="auto" 推理大模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    device_map="auto",           # 自动分片到可用设备
    torch_dtype=torch.float16,
)

inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 4.4 场景四：断点续训

```python
from accelerate import Accelerator

accelerator = Accelerator()

# ... 准备模型和数据 ...

# 保存训练状态
accelerator.save_state("checkpoint_epoch_1")

# 从断点恢复
accelerator.load_state("checkpoint_epoch_1")

# 继续训练
# ...
```

---

## 5. 数学原理

### 5.1 混合精度训练的损失缩放

在 fp16 混合精度训练中，梯度值可能小于 fp16 的最小表示范围（≈6×10⁻⁸），导致梯度下溢（underflow）变为零。动态损失缩放通过乘以缩放因子解决此问题：

```
前向传播：L_scaled = L × S （S为缩放因子，初始如 65536）
反向传播：∂L_scaled/∂W = S × ∂L/∂W
参数更新：W = W - lr × (∂L_scaled/∂W / S)

若 ∂L_scaled/∂W 中出现 inf/nan：
  - 跳过本次参数更新
  - S = S / 2（缩小缩放因子）
否则：
  - 正常更新
  - 可选：S = S × 2（逐步增大缩放因子）
```

### 5.2 ZeRO 优化的显存分析

设模型参数量为 Ψ，使用 k 个 GPU：

| 状态 | DDP显存 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------|---------|--------|--------|--------|
| 参数 | Ψ | Ψ | Ψ | Ψ/k |
| 梯度 | Ψ | Ψ | Ψ/k | Ψ/k |
| 优化器状态 | 2Ψ(Adam) | 2Ψ/k | 2Ψ/k | 2Ψ/k |
| 每GPU总计 | 4Ψ | 2Ψ + 2Ψ/k | Ψ + 3Ψ/k | 4Ψ/k |

ZeRO-3 下每GPU显存与GPU数量成反比，使得训练超大模型成为可能。

### 5.3 FSDP 的 all-gather 通信

FSDP 在前向传播时通过 all-gather 收集完整的参数分片，反向传播时再收集梯度分片：

```
前向：各GPU持有参数分片 → all-gather → 完整参数 → 计算 → 丢弃
反向：各GPU持有梯度分片 → all-gather → 完整梯度 → 计算 → reduce-scatter → 累积到分片
```

---

## 6. 架构原理

### 6.1 统一的分布式训练抽象层

Accelerate 的核心设计理念是**抽象层统一**：

```
用户代码
   │
   ▼
Accelerator（统一接口层）
   │
   ├─── PyTorch DDP    （默认分布式后端）
   ├─── FSDP           （全分片数据并行）
   └─── DeepSpeed      （ZeRO优化）
```

用户代码只与 `Accelerator` 交互，底层是 DDP/FSDP/DeepSpeed 由配置决定。切换分布式策略只需修改配置，代码无需改动。

### 6.2 prepare() 的工作流程

```
输入对象 → 类型判断 → 相应包装器
   │
   ├── nn.Module → DistributedDataParallel / FSDP / DeepSpeedEngine
   ├── DataLoader → 添加 DistributedSampler + 设置 batch_sampler
   ├── Optimizer → 添加到混合精度管理器
   └── LRScheduler → 关联到优化器步数管理
```

### 6.3 梯度累积的同步控制

在多进程梯度累积中，中间步不需要跨进程同步梯度，只在累积完成步才同步：

```
Step 1: backward → 累积（不同步）
Step 2: backward → 累积（不同步）
...
Step N: backward → 同步梯度 → optimizer.step() → zero_grad()
```

Accelerate 通过 `sync_gradients` 标志自动控制这一行为，避免不必要的通信开销。

### 6.4 device_map 的计算逻辑

`device_map="auto"` 的分配算法：

1. 获取各设备的可用显存（通过 `torch.cuda.mem_get_info()`）
2. 按 `max_memory` 参数约束上限
3. 将模型各层按参数量排序
4. 使用贪心算法将层分配到当前显存最充足的设备
5. 对于无法放入GPU的层，回退到CPU或磁盘

---

## 7. 常见注意事项与最佳实践

### 7.1 注意事项

1. **prepare() 调用时机**：必须在创建数据加载器之后、训练循环开始之前调用。`prepare()` 会修改 DataLoader 的 sampler

2. **不要手动 .to(device)**：使用 `device_placement=True`（默认）时，`prepare()` 会自动处理设备迁移，手动 `.to(device)` 可能导致冲突

3. **模型保存前 unwrap**：`accelerator.save_model()` 内部会自动 unwrap，但使用 `model.save_pretrained()` 前需要先 `accelerator.unwrap_model(model)`

4. **数据加载器的 shuffle**：`prepare()` 会将 DataLoader 的 sampler 替换为 `DistributedSampler`，因此 shuffle 参数会被忽略。如需控制 shuffle，请设置 `DistributedSampler` 的 `set_epoch()`：

```python
for epoch in range(num_epochs):
    train_dataloader.sampler.set_epoch(epoch)  # 确保每轮不同的shuffle顺序
    for batch in train_dataloader:
        ...
```

5. **混合精度下手动创建的 tensor**：如果在前向传播中手动创建 tensor，需要确保其 dtype 与模型一致，否则可能触发类型不匹配错误

6. **DeepSpeed 与 device_map 不兼容**：使用 DeepSpeed 时不支持 `device_map="auto"`，因为 DeepSpeed 有自己的设备管理机制

7. **梯度累积与日志**：累积中间步的 loss 值是未归一化的，需要除以 `gradient_accumulation_steps` 才得到真实 loss

### 7.2 最佳实践

1. **使用 `accelerate.accumulate()` 上下文管理器**替代手动判断累积步数，代码更简洁

2. **使用 `accelerator.print()`** 替代 `print()`，避免所有进程同时打印

3. **评估时使用 `accelerator.gather()`** 聚合所有进程的预测结果

4. **大模型推理用 `device_map="auto"`**，训练用 FSDP/DeepSpeed ZeRO-3

5. **优先使用 bf16 而非 fp16**（如果硬件支持），bf16 不需要 loss scaling，训练更稳定

6. **使用 `accelerate config` 配置环境**，避免在命令行传递大量参数

7. **配置日志记录**：使用 `log_with="wandb"` 等参数自动记录训练指标

```python
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(
    project_name="my-llm-project",
    config={"lr": 2e-5, "epochs": 3}
)

# 记录指标
accelerator.log({"loss": loss.item()}, step=step)

# 训练结束
accelerator.end_training()
```

8. **使用 `save_state()`/`load_state()` 实现断点续训**，确保优化器状态、调度器状态和随机数状态都能正确恢复
