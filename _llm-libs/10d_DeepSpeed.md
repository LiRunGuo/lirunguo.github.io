---
title: "DeepSpeed 分布式训练框架"
excerpt: "ZeRO-1/2/3显存分析、ZeRO-Offload、激活检查点、HF Trainer集成"
collection: llm-libs
permalink: /llm-libs/10d-deepspeed
category: training
---


## 1. 库简介与在 LLM 开发中的作用

### 1.1 什么是 DeepSpeed

DeepSpeed 是微软开发的分布式深度学习训练优化库，旨在降低大模型训练和推理的硬件门槛。其核心创新是 ZeRO（Zero Redundancy Optimizer）技术，通过消除数据并行中的内存冗余，使训练千亿参数模型成为可能。

### 1.2 在 LLM 开发中的核心价值

- **ZeRO 优化**：将模型状态分片到多个 GPU，使每张 GPU 只需存储 1/N 的状态，突破单卡显存瓶颈
- **ZeRO-Offload**：将优化器状态和计算卸载到 CPU/NVMe，进一步扩展可用内存
- **混合精度训练**：原生支持 fp16/bf16 训练，结合动态损失缩放防止下溢
- **极简 API**：仅需几行代码即可将现有训练脚本升级为分布式训练
- **推理优化**：内核级优化、动态 batching 和模型并行推理
- **HuggingFace 集成**：与 Transformers Trainer 深度集成，配置即用

---

## 2. 安装方式

### 2.1 基础安装

```bash
# PyPI 安装
pip install deepspeed

# 验证安装
ds_report  # 查看DeepSpeed环境信息
```

### 2.2 完整安装（含扩展）

```bash
# 安装所有可选扩展（如 NVMe Offload）
pip install deepspeed[all]

# 或仅安装特定扩展
pip install deepspeed[nvme]     # NVMe Offload 支持
pip install deepspeed[1bit]     # 1-bit Adam 优化器
pip install deepspeed[sparse]   # 稀疏注意力
```

### 2.3 从源码编译

```bash
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
pip install .
# 或带 CUDA 扩展编译
DS_BUILD_OPS=1 pip install . --global-option="--build_ext"
```

### 2.4 依赖要求

```bash
pip install torch transformers accelerate
```

---

## 3. 核心类/函数/工具详细说明

### 3.1 ZeRO 优化阶段

DeepSpeed 的 ZeRO（Zero Redundancy Optimizer）分为三个阶段，逐步消除数据并行中的内存冗余：

| 阶段 | 分片内容 | 内存节省 | 通信开销 |
|------|---------|---------|---------|
| ZeRO-1 | 优化器状态 | ~4x | 与 DDP 相同 |
| ZeRO-2 | 优化器状态 + 梯度 | ~8x | 与 DDP 相同 |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | 线性于 GPU 数 | 额外 all-gather |

**ZeRO-1（优化器状态分片）**：

每个 GPU 只保存 1/N 的优化器状态（如 Adam 的动量 m 和方差 v），但仍然保存完整的模型参数和梯度。在优化器 step 前，通过 reduce-scatter 同步梯度。

```json
{
    "zero_optimization": {
        "stage": 1
    }
}
```

**ZeRO-2（梯度分片）**：

在 ZeRO-1 基础上，每个 GPU 只保存 1/N 的梯度。梯度通过 reduce-scatter 直接分配到对应的 GPU，其余梯度立即释放。

```json
{
    "zero_optimization": {
        "stage": 2
    }
}
```

**ZeRO-3（参数分片）**：

在 ZeRO-2 基础上，每个 GPU 只保存 1/N 的模型参数。前向和反向传播时，通过 all-gather 按需收集所需参数，计算后立即释放。

```json
{
    "zero_optimization": {
        "stage": 3
    }
}
```

### 3.2 DeepSpeedConfig — 配置文件

DeepSpeed 使用 JSON 配置文件 `ds_config.json` 来管理所有训练参数。

```json
{
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": true,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": false
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "none"
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "wall_clock_breakdown": false
}
```

**核心配置项说明**：

| 配置项 | 说明 |
|--------|------|
| `train_batch_size` | 全局 batch size = `train_micro_batch_size_per_gpu` × `gradient_accumulation_steps` × `num_gpus` |
| `train_micro_batch_size_per_gpu` | 每个 GPU 的 micro batch size |
| `gradient_accumulation_steps` | 梯度累积步数 |
| `fp16.enabled` | 启用 fp16 混合精度 |
| `bf16.enabled` | 启用 bf16 混合精度（与 fp16 互斥） |
| `zero_optimization.stage` | ZeRO 阶段：1、2 或 3 |
| `zero_optimization.offload_optimizer.device` | 优化器卸载设备："cpu"、"nvme" 或 "none" |

### 3.3 DeepSpeedEngine — 训练引擎

`deepspeed.initialize()` 是 DeepSpeed 的核心入口，返回配置好的训练引擎。

```python
import deepspeed

model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
    model=model,                           # PyTorch 模型
    optimizer=optimizer,                   # 可选，DeepSpeed 也可从 config 创建
    args=None,                             # 命令行参数
    config_params=ds_config,               # DeepSpeed 配置（dict 或文件路径）
    model_parameters=model.parameters(),   # 模型参数（用于创建优化器）
    dist_init_backend="nccl",              # 分布式后端
)
```

**返回值说明**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `model_engine` | DeepSpeedEngine | 封装了模型、优化器、调度器的训练引擎 |
| `optimizer` | DeepSpeedOptimizer | 配置好的优化器 |
| `train_dataloader` | DataLoader | 配置好的数据加载器 |
| `lr_scheduler` | LRScheduler | 学习率调度器 |

**DeepSpeedEngine 关键方法**：

```python
# 前向传播
outputs = model_engine(inputs)

# 反向传播
model_engine.backward(loss)

# 优化器步进（包含梯度累积、梯度裁剪、ZeRO 同步等）
model_engine.step()

# 保存检查点
model_engine.save_checkpoint(save_dir, tag=None)

# 加载检查点
model_engine.load_checkpoint(load_dir, tag=None)

# 获取当前损失缩放
scale = model_engine.get_global_grad_norm()

# 销毁引擎（清理资源）
model_engine.destroy()
```

### 3.4 训练 API 详解

#### 3.4.1 backward() — 反向传播

```python
model_engine.backward(loss)
```

与标准 PyTorch 的 `loss.backward()` 不同，DeepSpeed 的 `backward()` 内部处理了：
- 混合精度损失缩放
- ZeRO 梯度分片和 reduce-scatter
- 梯度累积计数
- 梯度裁剪

#### 3.4.2 step() — 优化器步进

```python
model_engine.step()
```

`step()` 内部处理了：
- 梯度累积达到指定步数后才执行优化器更新
- ZeRO 优化器状态更新
- 学习率调度
- 梯度清零

#### 3.4.3 完整训练循环

```python
import deepspeed
import torch

# 初始化
model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
    model=model,
    config_params=ds_config,
)

# 训练循环
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 将数据移到 GPU
        inputs = batch["input_ids"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)

        # 前向传播
        outputs = model_engine(inputs, labels=labels)
        loss = outputs.loss

        # 反向传播（DeepSpeed 处理损失缩放和梯度分片）
        model_engine.backward(loss)

        # 优化器步进（DeepSpeed 处理梯度累积和裁剪）
        model_engine.step()
```

### 3.5 ZeRO-Offload — 卸载到 CPU/NVMe

ZeRO-Offload 将计算和内存压力卸载到 CPU 或 NVMe，使得在有限 GPU 资源下训练更大模型成为可能。

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",        // 卸载优化器状态到 CPU
            "pin_memory": true      // 使用固定内存加速 CPU-GPU 传输
        },
        "offload_param": {
            "device": "cpu",        // 卸载参数到 CPU
            "pin_memory": true
        }
    }
}
```

**ZeRO-Offload 配置选项**：

| 选项 | 说明 |
|------|------|
| `device` | `"cpu"`、`"nvme"` 或 `"none"` |
| `pin_memory` | 使用 CUDA pinned memory，加速 CPU↔GPU 数据传输 |
| `nvme_path` | NVMe 设备路径（仅 `device="nvme"` 时需要） |
| `buffer_count` | CPU 缓冲区数量 |
| `fast_init` | 快速初始化模式 |

**ZeRO-Infinity（NVMe Offload）**：

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true
        }
    },
    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "thread_count": 1,
        "single_submit": false,
        "overlap_events": true
    }
}
```

### 3.6 混合精度训练

#### 3.6.1 FP16 混合精度

```json
{
    "fp16": {
        "enabled": true,
        "initial_scale_power": 16,      // 初始损失缩放 = 2^16 = 65536
        "loss_scale_window": 1000,       // 连续无溢出步数达到此值时放大 scale
        "hysteresis": 2,                 // 溢出后缩小的延迟步数
        "min_loss_scale": 1              // 最小损失缩放
    }
}
```

**动态损失缩放机制**：
1. 初始损失缩放 $s = 2^{16}$
2. 若前向传播中出现溢出（inf/nan），则 $s = s / 2$，跳过本次更新
3. 若连续 `loss_scale_window` 步无溢出，则 $s = s \times 2$
4. $s$ 不低于 `min_loss_scale`

#### 3.6.2 BF16 混合精度

```json
{
    "bf16": {
        "enabled": true
    }
}
```

BF16 的优势：指数位与 fp32 相同（8 位），数值范围更大，不需要损失缩放，训练更稳定。推荐在支持 bf16 的硬件（Ampere+）上使用。

### 3.7 梯度累积

```json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8
}
```

等价关系：`train_batch_size = train_micro_batch_size_per_gpu × gradient_accumulation_steps × num_gpus`

DeepSpeed 内部自动处理梯度累积，无需手动 `loss = loss / accumulation_steps`。

### 3.8 激活检查点（Activation Checkpointing）

激活检查点通过在反向传播时重新计算中间激活值来减少显存占用，以计算换显存。

```python
import deepspeed

# 在模型中启用激活检查点
model = deepspeed.checkpointing.checkpoint(
    model,
    num_checkpoints=10,    # 检查点数量
)
```

或通过配置文件：

```json
{
    "activation_checkpointing": {
        "partition_activations": true,      // 分区激活值
        "cpu_checkpointing": false,         // 是否卸载到 CPU
        "contiguous_memory_optimization": true,  // 连续内存优化
        "number_checkpoints": 10,           // 检查点数量
        "synchronize_checkpoint_boundary": false, // 是否同步检查点边界
        "profile": false                    // 是否记录性能分析
    }
}
```

**显存节省效果**：激活检查点可减少 60-80% 的激活值显存，代价是约 33% 的额外计算开销。

### 3.9 DeepSpeed-Inference — 推理优化

DeepSpeed 提供了专门的推理引擎，支持模型并行和内核优化。

```python
import deepspeed

# 初始化推理引擎
model = deepspeed.init_inference(
    model=model,
    mp_size=1,                              # 模型并行大小
    dtype=torch.float16,                    # 推理精度
    replace_method="auto",                  # 自动替换内核
    replace_with_kernel_inject=True,        # 注入优化内核
)

# 推理
outputs = model(input_ids)
```

**推理配置**：

```json
{
    "tensor_parallel": {
        "tp_size": 2                         // 张量并行大小
    },
    "dtype": "fp16",
    "replace_with_kernel_inject": true
}
```

**DeepSpeed-Inference 的优化**：
- **内核融合**：将多个小算子融合为一个大内核，减少 GPU 核启动开销
- **量化推理**：支持 INT8/INT4 推理
- **动态 batching**：高效处理变长输入
- **模型并行**：张量并行自动切分大模型

### 3.10 与 HuggingFace Trainer 集成

DeepSpeed 与 HuggingFace Transformers 的 `Trainer` 深度集成，无需修改训练代码即可使用。

```python
from transformers import TrainingArguments, Trainer

# 方法一：通过 TrainingArguments 指定配置文件
training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",    # 指定 DeepSpeed 配置文件路径
    per_device_train_batch_size=2,
    num_train_epochs=3,
)

# 方法二：将 deepspeed 配置嵌入 TrainingArguments
training_args = TrainingArguments(
    output_dir="./output",
    deepspeed={
        "zero_optimization": {"stage": 2},
        "fp16": {"enabled": True},
        "train_batch_size": 16,
        "gradient_accumulation_steps": 4,
    },
)

# 使用 Trainer 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

**启动训练**：

```bash
# 使用 deepspeed 启动器
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json

# 或使用 torchrun
torchrun --nproc_per_node=4 train.py --deepspeed ds_config.json

# HuggingFace accelerate 启动
accelerate launch --use_deepspeed --num_processes=4 train.py --deepspeed ds_config.json
```

---

## 4. 在 LLM 开发中的典型使用场景和代码示例

### 4.1 场景一：ZeRO-2 + CPU Offload 微调 LLM

```python
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== 1. 创建 ds_config ==========
ds_config = {
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        }
    },
    "bf16": {
        "enabled": True,     # 使用 bf16 混合精度
    },
    "zero_optimization": {
        "stage": 2,                          # ZeRO-2
        "offload_optimizer": {
            "device": "cpu",                 # 优化器状态卸载到 CPU
            "pin_memory": True,
        },
        "allgather_partitions": True,
        "overlap_comm": True,
        "reduce_scatter": True,
        "contiguous_gradients": True,
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
}

# ========== 2. 加载模型 ==========
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ========== 3. 初始化 DeepSpeed ==========
model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
    model=model,
    config_params=ds_config,
    model_parameters=model.parameters(),
)

# ========== 4. 训练循环 ==========
model_engine.train()
for epoch in range(3):
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)

        outputs = model_engine(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        model_engine.backward(loss)
        model_engine.step()

    # 保存检查点
    model_engine.save_checkpoint(f"./checkpoints/epoch_{epoch}")
```

启动命令：

```bash
deepspeed --num_gpus=4 train.py
```

### 4.2 场景二：ZeRO-3 全参数微调大模型

```python
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ds_config = {
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 16,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
        }
    },
    "bf16": {
        "enabled": True,
    },
    "zero_optimization": {
        "stage": 3,                          # ZeRO-3: 参数分片
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
        "offload_param": {
            "device": "cpu",                 # 参数也卸载到 CPU
            "pin_memory": True,
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_clipping": 1.0,
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "number_checkpoints": 10,
    },
}

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")

model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
    model=model,
    config_params=ds_config,
)

# 训练循环同上
```

### 4.3 场景三：与 HuggingFace Trainer 集成微调

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# ds_config.json 已准备
training_args = TrainingArguments(
    output_dir="./deepspeed-output",
    deepspeed="ds_config.json",       # DeepSpeed 配置文件
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                         # 与 ds_config 中的 bf16 对应
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

def tokenize_fn(examples):
    return tokenizer(examples["instruction"], truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
```

启动命令：

```bash
deepspeed --num_gpus=4 train_hf.py
```

### 4.4 场景四：DeepSpeed 推理

```python
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 初始化推理引擎
ds_engine = deepspeed.init_inference(
    model=model,
    mp_size=1,                          # 模型并行大小
    dtype=torch.float16,
    replace_with_kernel_inject=True,    # 注入优化内核
)

# 推理
prompt = "Explain the concept of recursion:"
inputs = tokenizer(prompt, return_tensors="pt").to(ds_engine.module.device)

with torch.no_grad():
    outputs = ds_engine.module.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 5. 数学原理

### 5.1 ZeRO 内存分析

假设模型参数量为 $\Psi$，数值精度为 $\alpha$ 字节（fp32: $\alpha=4$，fp16: $\alpha=2$）。

#### 5.1.1 标准 DDP（Distributed Data Parallel）

在标准 DDP 中，每张 GPU 保存完整的模型副本：

| 状态 | 内存 |
|------|------|
| 模型参数 | $\Psi \alpha$ |
| 梯度 | $\Psi \alpha$ |
| 优化器状态（Adam） | $12\Psi$（主参数 fp32: $2\Psi \alpha$，动量 m: $\Psi \alpha$，方差 v: $\Psi \alpha$） |

总计（以 $\alpha=2$ 即 fp16 为例）：

$$\text{DDP 内存} = 2\Psi\alpha + 2\Psi\alpha + 12\Psi\alpha = 16\Psi\alpha$$

#### 5.1.2 ZeRO-1（优化器状态分片）

将优化器状态均匀分片到 $N_d$ 个 GPU：

| 状态 | 内存 |
|------|------|
| 模型参数 | $2\Psi\alpha$ |
| 梯度 | $2\Psi\alpha$ |
| 优化器状态 | $12\Psi\alpha / N_d$ |

$$\text{ZeRO-1 内存} = 4\Psi\alpha + \frac{12\Psi\alpha}{N_d}$$

当 $N_d = 64$ 时：$4\Psi\alpha + 0.1875\Psi\alpha \approx 4.19\Psi\alpha$，约为 DDP 的 1/4。

#### 5.1.3 ZeRO-2（梯度分片）

在 ZeRO-1 基础上，梯度也分片：

| 状态 | 内存 |
|------|------|
| 模型参数 | $2\Psi\alpha$ |
| 梯度 | $2\Psi\alpha / N_d$ |
| 优化器状态 | $12\Psi\alpha / N_d$ |

$$\text{ZeRO-2 内存} = 2\Psi\alpha + \frac{14\Psi\alpha}{N_d}$$

当 $N_d = 64$ 时：$2\Psi\alpha + 0.22\Psi\alpha \approx 2.22\Psi\alpha$，约为 DDP 的 1/7。

#### 5.1.4 ZeRO-3（参数分片）

所有状态均匀分片：

$$\text{ZeRO-3 内存} = \frac{16\Psi\alpha}{N_d}$$

当 $N_d = 64$ 时：$0.25\Psi\alpha$，约为 DDP 的 1/64。

#### 5.1.5 实际显存计算示例

以 7B 参数模型、fp16 精度（$\alpha = 2$）为例：

| 方案 | 每GPU显存 | GPU 数 | 总显存 |
|------|----------|--------|--------|
| DDP | 16 × 7B × 2 = 224 GB | 1 | 224 GB |
| ZeRO-1 (4 GPU) | 4×14 + 12×14/4 = 98 GB | 4 | 392 GB |
| ZeRO-2 (4 GPU) | 2×14 + 14×14/4 = 77 GB | 4 | 308 GB |
| ZeRO-3 (4 GPU) | 16×14/4 = 56 GB | 4 | 224 GB |
| ZeRO-3 (8 GPU) | 16×14/8 = 28 GB | 8 | 224 GB |
| ZeRO-3 + Offload (8 GPU) | ~14 GB (GPU) | 8 | ~112 GB (GPU) + CPU |

### 5.2 混合精度训练的数学原理

#### 5.2.1 FP16 前向传播

$$\hat{y} = \text{fp16}(f(\text{fp16}(x), \text{fp16}(W)))$$

其中 $\text{fp16}(\cdot)$ 表示截断为半精度浮点数。

#### 5.2.2 损失缩放

为防止小梯度在 fp16 下溢出为零，对损失乘以缩放因子 $s$：

$$\hat{L} = s \cdot L$$

梯度相应放大：

$$\hat{g} = s \cdot g$$

在更新前反缩放：

$$g = \frac{\hat{g}}{s}$$

#### 5.2.3 FP32 主权重

优化器维护 fp32 主权重 $W^{(32)}$，用于精确更新：

$$W^{(32)}_{t+1} = W^{(32)}_t - \eta \cdot g^{(32)}_t$$

$$W^{(16)}_{t+1} = \text{fp16}(W^{(32)}_{t+1})$$

这解释了为什么优化器状态占 $12\Psi\alpha$（fp32 主参数 $4\Psi$ + fp32 动量 $4\Psi$ + fp32 方差 $4\Psi$，以字节计）。

---

## 6. 代码原理/架构原理

### 6.1 整体架构

```
DeepSpeed
├── deepspeed.engine                    # 训练引擎
│   ├── DeepSpeedEngine                 # 主训练引擎
│   └── DeepSpeedTrainHybridEngine      # 混合引擎（训练+推理）
├── deepspeed.runtime                   # 运行时组件
│   ├── zero/                           # ZeRO 实现
│   │   ├── stage1.py                   # ZeRO-1
│   │   ├── stage2.py                   # ZeRO-2
│   │   ├── stage3.py                   # ZeRO-3
│   │   └── offload_config.py           # Offload 配置
│   ├── fp16/                           # 混合精度
│   │   └── unfused_optimizer.py        # fp16 优化器
│   ├── activation_checkpointing/       # 激活检查点
│   └── pipe/                           # 流水线并行
├── deepspeed.inference                 # 推理引擎
│   └── engine.py                       # 推理引擎实现
├── deepspeed.ops                       # 算子
│   ├── adam/                           # FusedAdam 等
│   ├── transformer/                    # 融合 Transformer 内核
│   └── sparse_attention/              # 稀疏注意力
├── deepspeed.checkpointing             # 检查点管理
└── deepspeed.utils                     # 工具函数
```

### 6.2 ZeRO-3 执行流程

```
前向传播:
    for each layer:
        all-gather 参数 W_i → 每个 GPU 获得完整 W_i
        Y_i = f(X_i, W_i)           # 前向计算
        释放非本地参数部分            # 仅保留 1/N
        X_{i+1} = Y_i

反向传播:
    for each layer (reverse):
        all-gather 参数 W_i
        计算梯度 ∂L/∂W_i, ∂L/∂X_i
        reduce-scatter ∂L/∂W_i      # 每个 GPU 只保留 1/N 的梯度
        释放非本地参数部分

优化器步进:
    each GPU:
        更新本地 1/N 的优化器状态
        更新本地 1/N 的参数
```

### 6.3 ZeRO-Offload 执行流程

```
GPU                                CPU
┌─────────────────┐               ┌──────────────────┐
│ 前向/反向计算    │               │ 优化器状态        │
│ 模型参数(部分)   │  ←→ 数据传输 → │ 梯度(部分)        │
│ 激活值           │               │ 主参数(fp32)      │
└─────────────────┘               └──────────────────┘

步骤:
1. GPU: 前向传播 + 反向传播（计算梯度）
2. 梯度 reduce-scatter 后传到 CPU
3. CPU: fp32 优化器更新
4. 更新后的参数传回 GPU
5. 重置损失缩放
```

### 6.4 通信优化策略

DeepSpeed 采用多种策略减少 ZeRO 带来的额外通信：

1. **overlap_comm**：将通信与计算重叠，在计算当前层时预取下一层参数
2. **reduce_bucket_size**：将多个小张量的 reduce 操作合并为一个大操作
3. **contiguous_gradients**：将梯度存储在连续内存中，减少内存碎片和通信次数
4. **allgather_partitions**：使用分区 all-gather 而非全局 all-gather

---

## 7. 常见注意事项和最佳实践

### 7.1 ZeRO 阶段选择指南

| 场景 | 推荐阶段 | 原因 |
|------|---------|------|
| GPU 显存充足，追求速度 | ZeRO-1 | 最小通信开销，仅分片优化器状态 |
| 中等显存压力 | ZeRO-2 | 梯度+优化器分片，通信开销仍较低 |
| 大模型（>10B），显存紧张 | ZeRO-3 | 参数分片，但需要额外 all-gather 通信 |
| 极大模型，GPU 显存不足 | ZeRO-3 + Offload | 将状态卸载到 CPU/NVMe |

### 7.2 配置最佳实践

1. **bf16 优先于 fp16**：在 Ampere+ GPU 上使用 bf16，避免损失缩放问题
   ```json
   {"bf16": {"enabled": true}}
   ```

2. **合理设置 gradient_accumulation_steps**：增大累积步数可减少通信频率
   ```json
   {"gradient_accumulation_steps": 4}
   ```

3. **ZeRO-3 启用 overlap_comm**：将通信与计算重叠
   ```json
   {"zero_optimization": {"stage": 3, "overlap_comm": true}}
   ```

4. **使用连续梯度**：减少内存碎片
   ```json
   {"zero_optimization": {"contiguous_gradients": true}}
   ```

5. **激活检查点与 ZeRO 配合**：进一步降低显存
   ```json
   {"activation_checkpointing": {"partition_activations": true}}
   ```

6. **ZeRO-3 保存模型时收集完整权重**：
   ```json
   {"zero_optimization": {"stage3_gather_16bit_weights_on_model_save": true}}
   ```

### 7.3 常见问题与解决方案

1. **问题：OOM（Out of Memory）**
   - ZeRO-2 OOM → 升级到 ZeRO-3
   - ZeRO-3 OOM → 启用 CPU Offload
   - ZeRO-3 + CPU Offload OOM → 启用 NVMe Offload
   - 减少 `train_micro_batch_size_per_gpu` 或增大 `gradient_accumulation_steps`

2. **问题：训练速度慢**
   - 确认 `overlap_comm: true` 已启用
   - 增大 `reduce_bucket_size` 和 `allgather_bucket_size`
   - 检查 CPU Offload 是否成为瓶颈（CPU→GPU 带宽不足）
   - 使用 `torch.compile()` 或 DeepSpeed 融合内核

3. **问题：检查点保存/加载失败**
   - ZeRO-3 检查点包含分片状态，必须使用 `model_engine.save_checkpoint()`
   - 加载时 GPU 数量可以不同，DeepSpeed 会自动重分片
   - 保存完整模型权重需设置 `stage3_gather_16bit_weights_on_model_save: true`

4. **问题：与 HuggingFace Trainer 集成时配置冲突**
   - `TrainingArguments` 中的参数不能与 `ds_config.json` 冲突
   - `bf16=True` 在 TrainingArguments 中设置时，ds_config 中也要 `"bf16": {"enabled": true}`
   - `per_device_train_batch_size` 与 `train_micro_batch_size_per_gpu` 对应

5. **问题：ZeRO-3 下 `model.generate()` 失败**
   - ZeRO-3 的参数分片与 `generate()` 不兼容
   - 解决：使用 `deepspeed.zero.GatheredParameters` 临时收集参数
   ```python
   with deepspeed.zero.GatheredParameters(model.parameters()):
       outputs = model.generate(**inputs)
   ```

### 7.4 ZeRO-3 特有注意事项

1. **参数初始化**：ZeRO-3 下每个 GPU 只持有 1/N 的参数，`model.parameters()` 返回的是分片后的参数
2. **模型打印**：打印模型结构可能触发 all-gather，建议在初始化前打印
3. **学习率调度器**：DeepSpeed 从配置文件创建优化器时，需要通过 `deepspeed.initialize()` 返回的调度器管理学习率
4. **梯度裁剪**：必须使用 `model_engine.step()` 内置的梯度裁剪，而非手动 `torch.nn.utils.clip_grad_norm_`

### 7.5 调试技巧

```bash
# 查看 DeepSpeed 环境信息
ds_report

# 启用详细日志
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json 2>&1 | tee train.log

# 内存分析
DS_DEBUG=1 deepspeed --num_gpus=4 train.py

# 性能分析
# 在 ds_config.json 中添加：
# "wall_clock_breakdown": true,
# "flops_profiler": {"enabled": true, "profile_step": 10}
```

### 7.6 版本兼容性

- DeepSpeed >= 0.8.0：推荐版本，包含 ZeRO-3 改进和稳定性修复
- DeepSpeed >= 0.10.0：改进的 HuggingFace 集成
- DeepSpeed >= 0.12.0：ZeRO-Infinity 稳定版
- transformers >= 4.21.0：`TrainingArguments(deepspeed=...)` 支持
- transformers >= 4.34.0：改进的 DeepSpeed ZeRO-3 集成

建议使用最新版本以获得最佳性能和兼容性。
