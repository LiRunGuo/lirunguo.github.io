---
title: "TRL 对齐训练库（结合源码）"
excerpt: "SFTTrainer/DPOTrainer/GRPOTrainer、DPO/GRPO/KTO数学原理、对齐训练流程"
collection: llm-libs
permalink: /llm-libs/09-trl
category: training
---


## 1. 库的简介

TRL（Transformer Reinforcement Learning）是 HuggingFace 生态中的核心 LLM 对齐训练库，为大型语言模型的全流程对齐提供了从监督微调（SFT）到偏好优化（DPO）、组相对策略优化（GRPO）等一整套训练方案。TRL 深度集成了 `transformers`、`accelerate`、`peft` 和 `datasets` 等 HuggingFace 核心库，使得从预训练模型出发进行对齐训练变得极其简洁。

在 LLM 对齐领域，TRL 的核心地位体现在：

- **全链路覆盖**：从 SFT 监督微调 → 奖励模型训练 → 偏好对齐（DPO/KTO/GRPO/PPO），覆盖了对齐训练的完整流程
- **算法前沿**：及时跟进最新对齐算法（如 GRPO、DAPO、DiscoPOP 等），提供开箱即用的实现
- **工程优化**：支持 DeepSpeed ZeRO、FSDP、vLLM 加速生成、Liger Kernel 优化、梯度检查点、激活卸载等工程优化
- **PEFT 兼容**：原生支持 LoRA/QLoRA 等参数高效微调方法，大幅降低训练显存需求

安装方式：

```bash
pip install trl
# 或安装完整依赖
pip install trl[peft]       # PEFT 支持
pip install trl[vllm]       # vLLM 加速生成
pip install trl[deepspeed]  # DeepSpeed 支持
```

---

## 2. 源码架构分析

### 2.1 整体架构图

TRL 的源码目录结构如下：

```
trl/
├── __init__.py                    # 顶层导出，统一暴露 API
├── trainer/                       # 训练器核心目录
│   ├── base_trainer.py            # _BaseTrainer 基类（继承 transformers.Trainer）
│   ├── base_config.py             # _BaseConfig 基类（继承 transformers.TrainingArguments）
│   ├── sft_trainer.py             # SFTTrainer - 监督微调训练器
│   ├── sft_config.py              # SFTConfig - SFT 训练配置
│   ├── dpo_trainer.py             # DPOTrainer - 直接偏好优化训练器
│   ├── dpo_config.py              # DPOConfig - DPO 训练配置
│   ├── grpo_trainer.py            # GRPOTrainer - 组相对策略优化训练器
│   ├── grpo_config.py             # GRPOConfig - GRPO 训练配置
│   ├── kto_trainer.py             # KTOTrainer - Kahneman-Tversky 优化训练器
│   ├── kto_config.py              # KTOConfig - KTO 训练配置
│   ├── rloo_trainer.py            # RLHFTrainer (RLOO) - RLHF 训练器
│   ├── rloo_config.py             # RLOOConfig - RLOO 训练配置
│   ├── reward_trainer.py          # RewardTrainer - 奖励模型训练器
│   ├── reward_config.py           # RewardConfig - 奖励模型训练配置
│   ├── model_config.py            # ModelConfig - 模型配置
│   ├── callbacks.py               # 训练回调（如 SyncRefModelCallback）
│   └── utils.py                   # 训练工具函数
├── models/                        # 模型相关
│   ├── utils.py                   # 模型工具（梯度检查点禁用、FSDP/DeepSpeed 准备等）
│   ├── causal_model.py            # 因果语言模型封装
│   └── reward_model.py            # 奖励模型封装
├── rewards/                       # 内置奖励函数
├── data_utils.py                  # 数据处理（chat template 应用、数据集打包等）
├── chat_template_utils.py         # Chat 模板工具
├── generation/                    # 生成相关（vLLM 生成封装）
│   └── vllm_generation.py         # VLLMGeneration 类
├── experimental/                  # 实验性功能（如 KTO 新实现）
├── skills/                        # 技能相关
└── import_utils.py                # 导入工具和依赖检测
```

**模块间关系**：

- `_BaseTrainer` 继承自 `transformers.Trainer`，所有专用 Trainer 均继承自 `_BaseTrainer`
- `_BaseConfig` 继承自 `transformers.TrainingArguments`，所有专用 Config 均继承自 `_BaseConfig`
- 每个 Trainer 对应一个 Config，Config 中定义该训练方法的特有参数
- `data_utils.py` 为各 Trainer 提供统一的数据处理工具（如 `apply_chat_template`、`is_conversational`、`pack_dataset`）
- `models/` 提供模型加载、DeepSpeed/FSDP 准备等基础设施
- `generation/vllm_generation.py` 为 GRPO 等 RL 训练器提供 vLLM 加速生成能力

### 2.2 Trainer 继承体系

```
transformers.Trainer
    └── _BaseTrainer (trl/trainer/base_trainer.py)
        ├── SFTTrainer    (trl/trainer/sft_trainer.py)
        ├── DPOTrainer    (trl/trainer/dpo_trainer.py)
        ├── GRPOTrainer   (trl/trainer/grpo_trainer.py)
        ├── KTOTrainer    (trl/trainer/kto_trainer.py)
        ├── RLHFTrainer   (trl/trainer/rloo_trainer.py)
        └── RewardTrainer (trl/trainer/reward_trainer.py)
```

`_BaseTrainer` 的核心作用是：
- 统一模型卡片生成（`create_model_card`）
- 定义 `_tag_names`、`_name`、`_paper` 等元数据属性
- 继承 `transformers.Trainer` 的全部训练循环、分布式训练、日志记录等能力

对应的配置类继承体系：

```
transformers.TrainingArguments
    └── _BaseConfig (trl/trainer/base_config.py)
        ├── SFTConfig    - 默认 learning_rate=2e-5
        ├── DPOConfig    - 默认 learning_rate=1e-6
        ├── GRPOConfig   - 默认 learning_rate=1e-6
        ├── KTOConfig
        ├── RLOOConfig
        └── RewardConfig - 默认 learning_rate=1e-4
```

`_BaseConfig` 重写了以下关键默认值：
- `logging_steps`：默认 10（transformers 默认 500）
- `gradient_checkpointing`：默认 True（transformers 默认 False）
- `bf16`：当 fp16 未设置时默认 True

### 2.3 SFTTrainer 源码分析

SFTTrainer 的核心代码位于 `trl/trainer/sft_trainer.py`。

#### 数据处理流程

SFTTrainer 支持三种数据格式：
1. **语言建模格式**：数据集包含 `text` 列，直接 tokenize
2. **Prompt-Completion 格式**：数据集包含 `prompt` 和 `completion` 列
3. **对话格式**：数据集包含 `messages` 列（符合 ChatML 格式）

数据处理流程（源码核心路径）：

```
原始数据 → maybe_convert_to_chatml() → apply_chat_template() → tokenize → pack_dataset() → DataCollatorForLanguageModeling
```

关键数据处理函数：

- `maybe_convert_to_chatml()`：将标准格式数据转换为对话格式
- `apply_chat_template()`：应用模型的 chat template 进行 tokenize
- `pack_dataset()`：当 `packing=True` 时，将多个序列打包成固定长度的块，减少 padding 浪费
  - 支持三种打包策略：`bfd`（最佳适应递减）、`bfd_split`（分割溢出）、`wrapped`（激进裁剪）

`DataCollatorForLanguageModeling`（源码 `sft_trainer.py:350`）负责批处理数据整理：
- 动态 padding 到批次内最大长度
- 根据 `completion_only_loss` 和 `assistant_only_loss` 设置 labels 中的 `-100` 掩码
- 支持 `padding_free` 模式（配合 FlashAttention 消除 padding）

#### 训练循环

SFTTrainer 复用 `transformers.Trainer` 的标准训练循环，核心区别在于：

1. **损失计算**：支持三种损失类型（`SFTConfig.loss_type`）：
   - `nll`（默认）：标准负对数似然损失
   - `chunked_nll`：分块计算交叉熵，减少峰值显存（仅对非 `-100` token 计算 lm_head 投影）
   - `dft`：Dynamic Fine-Tuning

2. **分块交叉熵**（`_chunked_cross_entropy_loss`，源码 `sft_trainer.py:104`）：
   - 仅对有效 token（`labels != -100`）计算 lm_head 投影
   - 按 `chunk_size` 分块处理，峰值激活显存为 `chunk_size * vocab_size`
   - 使用 `torch.utils.checkpoint.checkpoint` 实现梯度检查点

3. **激活卸载**：`activation_offloading=True` 时将激活卸载到 CPU，节省 GPU 显存

### 2.4 DPOTrainer 源码分析

DPOTrainer 的核心代码位于 `trl/trainer/dpo_trainer.py`。

#### 偏好对的处理

DPOTrainer 使用 `DataCollatorForPreference`（源码 `dpo_trainer.py:92`）处理偏好数据：

- 输入格式：每个样本包含 `prompt_ids`、`chosen_ids`、`rejected_ids`
- 拼接策略：将 chosen 和 rejected 拼接为一个大批次，前半为 chosen，后半为 rejected
- 输出：`input_ids`（prompt + completion 拼接）、`attention_mask`、`completion_mask`（标记哪些 token 属于 completion）

源码核心逻辑（`dpo_trainer.py:152-200`）：

```python
# 拼接 prompt + chosen 和 prompt + rejected
prompt_chosen_ids = [example["prompt_ids"] + example["chosen_ids"] for example in examples]
prompt_rejected_ids = [example["prompt_ids"] + example["rejected_ids"] for example in examples]
# completion_mask: 0 = prompt tokens, 1 = completion tokens
chosen_mask = [[0] * len(example["prompt_ids"]) + [1] * len(example["chosen_ids"]) for example in examples]
rejected_mask = [[0] * len(example["prompt_ids"]) + [1] * len(example["rejected_ids"]) for example in examples]
# 合并为一个批次
input_ids = prompt_chosen_ids + prompt_rejected_ids  # 前半 chosen，后半 rejected
```

#### 损失计算

DPOTrainer 的 `_compute_loss` 方法（源码 `dpo_trainer.py`）执行以下步骤：

1. **前向传播**：将拼接的 `[chosen, rejected]` 批次送入模型，获取 logits
2. **计算 per-token log probabilities**：
   ```python
   shift_logits = outputs.logits[..., :-1, :]
   shift_labels = input_ids[..., 1:]
   per_token_logps = selective_log_softmax(shift_logits, shift_labels)
   per_token_logps[shift_completion_mask == 0] = 0.0  # 屏蔽非 completion token
   logps = per_token_logps.sum(dim=1)  # 序列级 log probability
   chosen_logps, rejected_logps = logps.chunk(2, dim=0)  # 分离 chosen 和 rejected
   ```

3. **参考模型计算**：
   - 若 `precompute_ref_log_probs=True`，使用预计算的参考 log probabilities
   - 否则，在 `torch.no_grad()` 下运行参考模型获取 `ref_chosen_logps` 和 `ref_rejected_logps`
   - PEFT 模型通过禁用 adapter 或使用 "ref" adapter 获取参考输出

4. **计算 log ratios 和损失**：
   ```python
   chosen_logratios = chosen_logps - ref_chosen_logps
   rejected_logratios = rejected_logps - ref_rejected_logps
   delta_score = chosen_scores - rejected_scores
   # sigmoid DPO 损失
   per_sequence_loss = -F.logsigmoid(self.beta * delta_score)
   ```

DPO 支持 15+ 种损失类型，包括 `sigmoid`、`hinge`、`ipo`、`robust`、`discopop` 等，可通过 `loss_type` 参数指定，还支持多损失组合（通过 `loss_weights` 加权）。

### 2.5 GRPOTrainer 源码分析

GRPOTrainer 的核心代码位于 `trl/trainer/grpo_trainer.py`，是最复杂的训练器之一。

#### 组采样与生成

GRPOTrainer 对每个 prompt 采样 G 个响应（由 `num_generations` 控制，默认 8），核心步骤：

1. **生成阶段**：使用模型对每个 prompt 生成 `num_generations` 个 completion
   - 支持标准 `model.generate()` 和 vLLM 加速生成
   - 生成参数：`temperature`、`top_p`、`top_k`、`min_p`、`max_completion_length` 等

2. **奖励计算**：对每个生成的 completion 计算奖励
   - 支持自定义奖励函数（Callable）、预训练奖励模型（str/PreTrainedModel）
   - 多奖励函数时按 `reward_weights` 加权求和
   - 奖励缩放策略：`group`（组内标准化）、`batch`（批次标准化）、`none`

3. **优势计算**：计算组内标准化优势
   ```python
   # 对同一 prompt 的 G 个响应计算相对优势
   advantages = (rewards - rewards.mean()) / rewards.std()
   ```

#### 损失计算

GRPOTrainer 的 `_compute_loss` 方法（源码 `grpo_trainer.py`）执行以下步骤：

1. **计算当前策略的 per-token log probabilities**：
   ```python
   input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
   per_token_logps, entropies = self._get_per_token_logps_and_entropies(model, input_ids, attention_mask, ...)
   ```

2. **计算重要性采样比率**：
   ```python
   log_ratio = per_token_logps - old_per_token_logps
   # 支持 token 级和 sequence 级
   if self.importance_sampling_level == "token":
       log_importance_weights = log_ratio
   elif self.importance_sampling_level == "sequence":
       log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1)
   coef_1 = torch.exp(log_importance_weights)  # ρ_t = π_θ / π_old
   ```

3. **计算 KL 散度**（当 `beta != 0` 时）：
   ```python
   per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
   ```

4. **PPO-style 裁剪损失**：
   ```python
   # 裁剪目标
   coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)
   per_token_loss1 = coef_1 * advantages  # 原始比率 × 优势
   per_token_loss2 = coef_2 * advantages  # 裁剪比率 × 优势
   per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
   ```

5. **总损失**：策略损失 + KL 惩罚
   ```python
   loss = (per_token_loss + beta * per_token_kl) * mask
   ```

GRPO 支持多种损失类型：`grpo`、`dapo`（默认）、`dr_grpo`、`bnpo`、`cispo`、`sapo`、`luspo`、`vespo` 等，主要区别在于 token 级损失的归一化方式和裁剪策略。

---

## 3. 核心类/函数详细说明

### 3.1 SFTTrainer

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",  # 模型名称或路径或 PreTrainedModel 实例
    args=SFTConfig(...),                   # SFT 训练配置
    train_dataset=dataset,                 # 训练数据集
    eval_dataset=eval_dataset,             # 评估数据集（可选）
    processing_class=tokenizer,            # tokenizer 或 processor
    data_collator=None,                    # 数据整理器（默认自动选择）
    formatting_func=None,                  # 数据格式化函数（可选）
    callbacks=None,                        # 训练回调
    peft_config=None,                      # PEFT 配置（如 LoRA）
)
```

**关键参数说明**：
- `model`：支持字符串路径、`PreTrainedModel` 实例或 `PeftModel` 实例
- `formatting_func`：自定义数据格式化函数，接收样本返回格式化文本

### 3.2 SFTConfig

```python
from trl import SFTConfig

config = SFTConfig(
    output_dir="./results",
    # === 训练参数 ===
    learning_rate=2e-5,              # 学习率（默认 2e-5，比 transformers 默认的 5e-5 更小）
    per_device_train_batch_size=4,   # 每设备训练批次大小
    gradient_accumulation_steps=8,   # 梯度累积步数
    num_train_epochs=3,              # 训练轮数

    # === 序列长度与打包 ===
    max_length=1024,                 # 最大序列长度（默认 1024）
    packing=False,                   # 是否启用序列打包（默认 False）
    packing_strategy="bfd",          # 打包策略："bfd"、"bfd_split"、"wrapped"
    padding_free=False,              # 无 padding 模式（需 FlashAttention）

    # === 损失控制 ===
    completion_only_loss=None,       # 仅在 completion 部分计算损失
    assistant_only_loss=False,       # 仅在 assistant 部分计算损失
    loss_type="nll",                 # 损失类型："nll"、"chunked_nll"、"dft"

    # === 显存优化 ===
    gradient_checkpointing=True,     # 梯度检查点（默认 True）
    activation_offloading=False,     # 激活卸载到 CPU
    bf16=True,                       # BF16 混合精度（默认 True）

    # === 数据处理 ===
    dataset_text_field="text",       # 文本列名
    dataset_num_proc=None,           # 数据处理进程数
    truncation_mode="keep_start",    # 截断模式
)
```

### 3.3 DPOTrainer

```python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",      # 待训练模型
    ref_model=None,                            # 参考模型（None 时使用初始策略）
    args=DPOConfig(...),                       # DPO 训练配置
    train_dataset=dataset,                     # 偏好数据集
    processing_class=tokenizer,                # tokenizer
    data_collator=None,                        # 数据整理器（默认 DataCollatorForPreference）
    peft_config=None,                          # PEFT 配置
    loss_type=["sigmoid"],                     # 损失类型列表
)
```

**关键参数说明**：
- `model`：待训练的策略模型
- `ref_model`：参考模型。若为 None，DPOTrainer 自动使用训练前的初始策略作为参考。PEFT 模型通过 adapter 切换实现参考策略

### 3.4 DPOConfig

```python
from trl import DPOConfig

config = DPOConfig(
    output_dir="./results",
    # === 核心训练参数 ===
    learning_rate=1e-6,               # 学习率（默认 1e-6，比 SFT 更小）
    beta=0.1,                         # KL 约束系数（β 越大，偏离参考模型越少）

    # === 损失类型 ===
    loss_type=["sigmoid"],            # 损失类型列表，支持 15+ 种
    # "sigmoid": 标准 DPO 损失
    # "hinge": 铰链损失
    # "ipo": IPO 损失（β 为正则化参数 τ）
    # "robust": 鲁棒 DPO（配合 label_smoothing）
    # "discopop": DiscoPOP 损失
    # "sft": 纯 SFT 损失
    # "sigmoid_norm": 标准化 sigmoid DPO
    loss_weights=None,                # 多损失组合的权重

    # === 参考模型控制 ===
    precompute_ref_log_probs=False,   # 预计算参考模型 log probs（节省训练时显存）
    sync_ref_model=False,             # 同步参考模型（TR-DPO）
    ref_model_mixup_alpha=0.6,        # TR-DPO 混合系数 α
    ref_model_sync_steps=512,         # TR-DPO 同步频率

    # === f-散度控制 ===
    f_divergence_type="reverse_kl",   # f-散度类型："reverse_kl"、"forward_kl"、"js_divergence"、"alpha_divergence"

    # === 序列长度 ===
    max_length=1024,                  # 最大序列长度
    padding_free=False,               # 无 padding 模式

    # === 其他 ===
    disable_dropout=True,             # 禁用 dropout（默认 True）
    label_smoothing=0.0,              # 标签平滑（Robust DPO/EXO 使用）
)
```

### 3.5 GRPOTrainer

```python
from trl import GRPOTrainer, GRPOConfig

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",       # 待训练模型
    reward_funcs=accuracy_reward,               # 奖励函数（支持多种类型）
    args=GRPOConfig(...),                       # GRPO 训练配置
    train_dataset=dataset,                      # 训练数据集（需含 "prompt" 列）
    processing_class=tokenizer,                 # tokenizer
    reward_processing_classes=None,             # 奖励模型对应的 tokenizer
    peft_config=None,                           # PEFT 配置
    tools=None,                                 # 工具函数列表（Agent 训练）
    rollout_func=None,                          # 自定义生成函数
    environment_factory=None,                   # 环境工厂（Agent 训练）
)
```

**reward_funcs 支持类型**：
- 字符串：HuggingFace Hub 上的奖励模型 ID
- `PreTrainedModel`：预训练的序列分类模型
- `Callable`：自定义奖励函数，接收 prompts 和 completions，返回奖励列表

### 3.6 GRPOConfig

```python
from trl import GRPOConfig

config = GRPOConfig(
    output_dir="./results",
    # === 核心训练参数 ===
    learning_rate=1e-6,                # 学习率（默认 1e-6）
    beta=0.0,                          # KL 系数（0.0 时不加载参考模型；DeepSeek-R1 用 0.001）
    num_generations=8,                 # 每个 prompt 的生成数 G（默认 8，最小 2）
    epsilon=0.2,                       # PPO 裁剪参数 ε（默认 0.2）
    num_iterations=1,                  # 每批次的迭代次数 μ

    # === 生成参数 ===
    max_completion_length=256,         # 生成最大长度（默认 256）
    temperature=1.0,                   # 采样温度
    top_p=1.0,                         # Top-p 采样
    generation_batch_size=None,        # 生成批次大小

    # === 损失类型 ===
    loss_type="dapo",                  # 损失类型（默认 dapo）
    # "grpo": 序列长度归一化（有长度偏差）
    # "dapo": 全局活跃 token 归一化（消除长度偏差，默认）
    # "dr_grpo": 全局常数归一化
    # "bnpo": 本地批次归一化
    # "cispo": 裁剪重要性采样权重

    # === 奖励处理 ===
    scale_rewards="group",             # 奖励缩放："group"、"batch"、"none"
    reward_weights=None,               # 各奖励函数权重
    mask_truncated_completions=False,  # 屏蔽被截断的 completion

    # === vLLM 加速 ===
    use_vllm=False,                    # 是否使用 vLLM 生成
    vllm_mode="colocate",              # vLLM 模式："colocate"、"server"

    # === 参考模型同步 ===
    sync_ref_model=False,              # TR-DPO 式参考模型同步
    ref_model_mixup_alpha=0.6,
    ref_model_sync_steps=512,
)
```

### 3.7 KTOTrainer

KTO（Kahneman-Tversky Optimization）不需要成对偏好数据，而是分别处理期望（desirable）和不期望（undesirable）输出。

```python
from trl import KTOTrainer, KTOConfig

# 注意：KTOTrainer 当前位于 trl.experimental
trainer = KTOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    ref_model=None,                    # 参考模型
    args=KTOConfig(...),
    train_dataset=dataset,             # 需含 prompt + completion + label（desirable/undesirable）
    processing_class=tokenizer,
)
```

KTO 的优势在于：不需要成对偏好数据，只需要标注每个输出是"好"还是"坏"，数据收集成本更低。

### 3.8 RewardTrainer 与 RewardConfig

RewardTrainer 用于训练奖励模型（Reward Model），该模型随后可用于 PPO/DPO 等 RLHF 训练。

```python
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", num_labels=1
)

trainer = RewardTrainer(
    model=model,
    args=RewardConfig(
        output_dir="./results",
        learning_rate=1e-4,                  # 默认 1e-4
        max_length=1024,
        center_rewards_coefficient=0.01,     # 奖励中心化系数（推荐 0.01）
        disable_dropout=True,
    ),
    train_dataset=dataset,              # 偏好数据集（chosen + rejected）
    processing_class=tokenizer,
)
```

RewardConfig 关键参数：
- `center_rewards_coefficient`：鼓励奖励模型输出均值为零的奖励值，推荐 0.01
- `disable_dropout`：默认 True，禁用 dropout 保证训练稳定性

---

## 4. 数学原理

### 4.1 SFT（监督微调）

SFT 的目标是最大化训练数据上的对数似然：

$$L_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \sum_{t} \log P_\theta(y_t | x, y_{<t}) \right]$$

其中 $x$ 是输入（prompt），$y$ 是目标输出（completion），$y_t$ 是第 $t$ 个 token，$y_{<t}$ 是第 $t$ 个 token 之前的所有 token。

在 TRL 的实现中：
- 当 `completion_only_loss=True` 时，损失仅在 completion 部分计算（prompt 部分的 labels 设为 `-100`）
- 当 `assistant_only_loss=True` 时，损失仅在 assistant 回复部分计算
- 当 `packing=True` 时，多个序列被打包到固定长度块中，有效利用计算资源

### 4.2 DPO（直接偏好优化）

DPO 的核心思想是绕过显式的奖励模型训练，直接从偏好数据优化策略。

**Bradley-Terry 偏好模型**：

$$P(y_w > y_l | x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)$$

其中 $y_w$ 是被偏好的响应（chosen），$y_l$ 是不被偏好的响应（rejected），$\sigma$ 是 sigmoid 函数。

**DPO 损失函数**：

$$L_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma\left(\beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)\right) \right]$$

**隐式奖励**：

$$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

DPO 的关键洞察是：最优策略与参考策略的对数比率等价于奖励函数，因此无需显式训练奖励模型。

在 TRL 的源码实现中（`dpo_trainer.py`）：
```python
# 计算 log ratios
chosen_logratios = chosen_logps - ref_chosen_logps      # log(π_θ/π_ref) for chosen
rejected_logratios = rejected_logps - ref_rejected_logps  # log(π_θ/π_ref) for rejected
delta_score = chosen_scores - rejected_scores
# sigmoid DPO 损失
per_sequence_loss = -F.logsigmoid(self.beta * delta_score)
```

**其他 DPO 损失变体**：
- **Hinge**：$L = \max(0, 1 - \beta \cdot \text{delta\_score})$
- **IPO**：$L = (\text{delta\_score} - 1/2\beta)^2$，将 DPO 的正则化从 KL 散度替换为平方损失
- **Robust DPO**：引入 `label_smoothing` 处理噪声偏好标签
- **DiscoPOP**：使用 log-ratio 调制的柔性损失函数

### 4.3 GRPO（组相对策略优化）

GRPO 最初在 DeepSeekMath 论文中提出，核心思想是对每个 prompt 采样一组响应，利用组内相对优势进行策略优化。

**组采样**：对每个 prompt $x$，采样 $G$ 个响应 $\{y_1, y_2, ..., y_G\}$

**优势计算**：

$$\tilde{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$

其中 $r_i$ 是第 $i$ 个响应的奖励，$\text{mean}(r)$ 和 $\text{std}(r)$ 是组内均值和标准差。

**策略损失**（PPO-style 裁剪目标）：

$$L_{\text{GRPO}} = -\mathbb{E}\left[\min\left(\rho_t \tilde{A}, \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon) \tilde{A}\right)\right]$$

其中 $\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 是重要性采样比率。

**总损失**：

$$L = L_{\text{GRPO}} + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

KL 散度的实现（近似估计）：

$$D_{\text{KL}} = \frac{\pi_{\text{ref}}(a|s)}{\pi_\theta(a|s)} - \log \frac{\pi_{\text{ref}}(a|s)}{\pi_\theta(a|s)} - 1$$

在 TRL 源码中（`grpo_trainer.py`）：
```python
# 重要性采样比率
log_ratio = per_token_logps - old_per_token_logps
coef_1 = torch.exp(log_ratio)  # ρ_t

# 裁剪目标
coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)
per_token_loss1 = coef_1 * advantages
per_token_loss2 = coef_2 * advantages
per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

# KL 散度
per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
```

**损失归一化**：不同 `loss_type` 的区别主要在于归一化方式：
- `grpo`：按序列长度归一化（有长度偏差）
- `dapo`（默认）：按全局活跃 token 数归一化（消除长度偏差）
- `dr_grpo`：按 `max_completion_length` 常数归一化

### 4.4 KTO（Kahneman-Tversky 优化）

KTO 基于前景理论（Prospect Theory），对期望和不期望输出使用不同的价值函数。

**KTO 损失**：

对于期望输出（desirable，$y \in \mathcal{D}$）：

$$L_{\text{desirable}} = \lambda_w \cdot v\left(\beta \left(\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} - z_{\text{ref}}\right)\right)$$

对于不期望输出（undesirable，$y \in \mathcal{U}$）：

$$L_{\text{undesirable}} = \lambda_l \cdot v\left(\beta \left(z_{\text{ref}} - \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right)\right)$$

其中：
- $v(\cdot)$ 是前景理论的价值函数（损失厌恶：对损失的敏感度高于收益）
- $z_{\text{ref}}$ 是参考 KL 散度项
- $\lambda_w$ 和 $\lambda_l$ 是收益和损失的权重

KTO 的优势是不需要成对偏好数据，只需要标注单个输出是"好"还是"坏"。

### 4.5 PPO（近端策略优化）

PPO 是最经典的 RLHF 方法，TRL 通过 `RLHFTrainer`（RLOO 实现）提供支持。

**PPO 裁剪损失**：

$$L_{\text{CLIP}} = -\mathbb{E}\left[\min\left(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon) \hat{A}_t\right)\right]$$

**总损失**：

$$L = L_{\text{CLIP}} + c_1 L_{\text{VF}} - c_2 H(\pi)$$

其中：
- $L_{\text{VF}}$ 是价值函数损失
- $H(\pi)$ 是策略熵（鼓励探索）
- $c_1$ 和 $c_2$ 是系数

PPO 的完整训练需要四个模型：策略模型、参考模型、奖励模型和价值模型，显存开销较大。DPO 和 GRPO 在一定程度上解决了这个问题。

---

## 5. 代码原理

### 5.1 SFTTrainer 训练流程

```
1. 数据格式化
   ├── 语言建模格式: text 列 → 直接 tokenize
   ├── Prompt-Completion 格式: prompt + completion → tokenize → 设置 completion_mask
   └── 对话格式: messages → apply_chat_template() → tokenize → 设置 assistant_mask

2. 数据打包 (packing, 可选)
   ├── pack_dataset() 将多个短序列打包到固定长度块
   ├── 支持 bfd/bfd_split/wrapped 三种打包策略
   └── 减少 padding 浪费，提高 GPU 利用率

3. 批处理整理
   └── DataCollatorForLanguageModeling
       ├── 动态 padding
       ├── 根据 completion_only_loss / assistant_only_loss 设置 labels 掩码
       └── 支持 padding_free 模式（返回 position_ids 替代 attention_mask）

4. 标准训练循环（继承自 transformers.Trainer）
   ├── 前向传播
   ├── 损失计算（nll / chunked_nll / dft）
   │   └── chunked_nll: 仅对有效 token 计算 lm_head 投影，分块处理
   ├── 反向传播
   └── 优化器步骤
```

### 5.2 DPOTrainer 训练流程

```
1. 数据预处理
   └── 将偏好数据拆分为 prompt_ids + chosen_ids / rejected_ids

2. 批处理整理
   └── DataCollatorForPreference
       ├── 拼接 prompt+chosen 和 prompt+rejected
       ├── 前半批次 = chosen, 后半批次 = rejected
       └── 生成 completion_mask 标记哪些 token 属于 completion

3. 前向传播
   ├── 策略模型前向: 同时计算 chosen 和 rejected 的 logits
   ├── selective_log_softmax: 高效计算 per-token log probabilities
   ├── 屏蔽非 completion token: per_token_logps[mask == 0] = 0.0
   └── 序列级 logps: 对 completion token 的 logps 求和

4. 参考模型前向（no_grad）
   ├── 方式1: precompute_ref_log_probs=True → 使用预计算值
   ├── 方式2: ref_model → 独立参考模型前向
   └── 方式3: PEFT adapter → 切换到 "ref" adapter 或禁用 adapter

5. 损失计算
   ├── chosen_logratios = chosen_logps - ref_chosen_logps
   ├── rejected_logratios = rejected_logps - ref_rejected_logps
   ├── delta_score = chosen_scores - rejected_scores
   ├── 支持多种 f-散度类型转换 scores
   └── per_sequence_loss = -logsigmoid(beta * delta_score)  # sigmoid DPO

6. 标准训练循环
```

### 5.3 GRPOTrainer 训练流程

```
1. 生成阶段
   ├── 对每个 prompt 生成 num_generations 个 completion
   ├── 支持标准 generate() 或 vLLM 加速
   └── 记录 old_per_token_logps（旧策略的 token 级 log probs）

2. 奖励计算
   ├── 对每个 (prompt, completion) 对计算奖励
   ├── 支持多个奖励函数，按 reward_weights 加权
   ├── 奖励缩放: group/batch/none
   └── 多目标聚合: sum_then_normalize / normalize_then_sum

3. 优势计算
   ├── 组内标准化: A_i = (r_i - mean(r)) / std(r)
   └── 同一 prompt 的 G 个响应共享一组优势值

4. 策略更新（_compute_loss）
   ├── 前向传播获取当前 per_token_logps
   ├── 计算重要性采样比率: ρ = exp(per_token_logps - old_per_token_logps)
   ├── PPO 裁剪损失:
   │   ├── per_token_loss1 = ρ * A
   │   ├── per_token_loss2 = clip(ρ, 1-ε, 1+ε) * A
   │   └── per_token_loss = -min(loss1, loss2)
   ├── KL 散度惩罚: per_token_kl = exp(ref_logps - logps) - (ref_logps - logps) - 1
   ├── 总损失: (per_token_loss + β * per_token_kl) * mask
   └── 损失归一化（按 loss_type 选择归一化方式）

5. 优化器步骤
```

---

## 6. 在 LLM 开发中的典型使用场景和代码示例

### 6.1 SFT 微调对话模型

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import AutoTokenizer

# 加载模型和 tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载对话数据集（messages 格式）
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")

# 配置训练参数
sft_config = SFTConfig(
    output_dir="./sft-output",
    learning_rate=2e-5,                      # SFT 学习率
    per_device_train_batch_size=4,            # 每设备批次大小
    gradient_accumulation_steps=8,            # 梯度累积，等效批次大小=4*8=32
    num_train_epochs=1,                       # 训练轮数
    max_length=2048,                          # 最大序列长度
    packing=True,                             # 启用序列打包
    assistant_only_loss=True,                 # 仅在 assistant 回复上计算损失
    gradient_checkpointing=True,              # 梯度检查点
    bf16=True,                                # BF16 混合精度
    logging_steps=10,
    save_steps=500,
)

# 创建 Trainer 并训练
trainer = SFTTrainer(
    model=model_name,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("./sft-output/final")
```

### 6.2 DPO 对齐偏好

```python
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 加载偏好数据集
# 数据集需包含 "prompt"、"chosen"、"rejected" 列
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:10000]")

# 配置 DPO 训练参数
dpo_config = DPOConfig(
    output_dir="./dpo-output",
    learning_rate=1e-6,                       # DPO 学习率（比 SFT 更小）
    per_device_train_batch_size=2,             # 每设备批次大小
    gradient_accumulation_steps=16,            # 梯度累积
    num_train_epochs=1,
    beta=0.1,                                 # KL 约束系数
    loss_type=["sigmoid"],                    # 标准 DPO 损失
    max_length=1024,                          # 最大序列长度
    gradient_checkpointing=True,
    bf16=True,
)

# 创建 Trainer 并训练
# ref_model=None 时自动使用初始策略作为参考
trainer = DPOTrainer(
    model="./sft-output/final",               # 使用 SFT 后的模型
    ref_model=None,                           # 参考模型 = 初始策略
    args=dpo_config,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./dpo-output/final")
```

### 6.3 GRPO 训练

```python
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward
from datasets import load_dataset

# 加载数学推理数据集
dataset = load_dataset("trl-lib/DeepMath-103K", split="train[:5000]")

# 自定义奖励函数
def format_reward(completions, **kwargs):
    """检查回答是否包含正确的 LaTeX 格式"""
    rewards = []
    for completion in completions:
        if "\\boxed{" in completion:
            rewards.append(1.0)
        else:
            rewards.append(-0.5)
    return rewards

# 配置 GRPO 训练参数
grpo_config = GRPOConfig(
    output_dir="./grpo-output",
    learning_rate=1e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    beta=0.001,                               # KL 系数（DeepSeek-R1 推荐 0.001）
    num_generations=8,                        # 每个 prompt 生成 8 个响应
    max_completion_length=512,                # 最大生成长度
    epsilon=0.2,                              # PPO 裁剪参数
    loss_type="dapo",                         # DAPO 损失（消除长度偏差）
    temperature=1.0,                          # 采样温度
    scale_rewards="group",                    # 组内奖励标准化
    mask_truncated_completions=True,          # 屏蔽被截断的 completion
    gradient_checkpointing=True,
    bf16=True,
    log_completions=True,                     # 日志记录生成样本
)

# 创建 Trainer 并训练
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[accuracy_reward, format_reward],  # 多奖励函数
    reward_weights=[0.8, 0.2],                     # 奖励权重
    args=grpo_config,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./grpo-output/final")
```

### 6.4 奖励模型训练

```python
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# 加载偏好数据集
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:10000]")

# 创建序列分类模型（num_labels=1 输出标量奖励）
model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# 配置奖励模型训练
reward_config = RewardConfig(
    output_dir="./reward-output",
    learning_rate=1e-4,                       # 奖励模型学习率
    per_device_train_batch_size=4,
    num_train_epochs=1,
    max_length=1024,
    center_rewards_coefficient=0.01,          # 奖励中心化系数
    disable_dropout=True,
    gradient_checkpointing=True,
    bf16=True,
)

# 创建 Trainer 并训练
trainer = RewardTrainer(
    model=model,
    args=reward_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
```

### 6.5 在线 DPO 训练

在线 DPO 不使用预收集的偏好数据，而是在训练过程中实时生成偏好对。

```python
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 使用 prompt 数据集（无需预收集偏好对）
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:5000]")

# 在线 DPO 配置
dpo_config = DPOConfig(
    output_dir="./online-dpo-output",
    learning_rate=5e-7,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    beta=0.1,
    loss_type=["sigmoid"],
    max_length=1024,
    gradient_checkpointing=True,
    bf16=True,
)

# 在线 DPO：使用同一模型同时作为策略模型和生成模型
# 每个训练步骤中：
# 1. 对 prompt 生成两个响应
# 2. 使用奖励模型对两个响应打分
# 3. 将高分响应作为 chosen，低分响应作为 rejected
# 4. 计算 DPO 损失并更新策略
trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=dpo_config,
    train_dataset=dataset,
)

trainer.train()
```

### 6.6 使用 LoRA 进行高效训练

```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

# 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                                     # LoRA 秩
    lora_alpha=32,                            # LoRA alpha
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标层
    bias="none",
)

# 在 SFTTrainer 中使用 LoRA
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-7B-Instruct",        # 大模型也可用 LoRA 训练
    args=SFTConfig(
        output_dir="./lora-sft-output",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        max_length=2048,
        bf16=True,
    ),
    train_dataset=dataset,
    peft_config=lora_config,                  # 传入 LoRA 配置
)

trainer.train()

# 保存 LoRA adapter（仅保存增量参数）
trainer.save_model("./lora-sft-output/adapter")
```

---

## 7. 常见注意事项和最佳实践

### 7.1 学习率选择

| 训练方法 | 推荐学习率 | 说明 |
|---------|-----------|------|
| SFT | 2e-5 | TRL 默认值，适用于全参数微调 |
| DPO | 1e-6 | TRL 默认值，偏好对齐需要更小学习率 |
| GRPO | 1e-6 | TRL 默认值，RL 训练需要保守学习率 |
| Reward Model | 1e-4 | TRL 默认值，奖励模型收敛较快 |
| LoRA/QLoRA | 可适当提高 | 如 SFT+LoRA 可用 1e-4 |

### 7.2 批次大小与梯度累积

- DPO 的有效批次需为偶数（chosen + rejected 拼接为一个批次）
- GRPO 的有效批次必须能被 `num_generations` 整除
- 梯度累积步数 = 等效批次大小 / (per_device_batch_size × GPU 数量)

### 7.3 显存优化策略

按推荐优先级排列：

1. **LoRA/QLoRA**：仅训练少量参数，显存降低 60-80%
2. **gradient_checkpointing=True**：TRL 默认开启，用计算换显存（约 20% 慢）
3. **activation_offloading=True**：将激活卸载到 CPU，进一步降低 GPU 显存
4. **chunked_nll 损失**（SFT）：分块计算交叉熵，降低峰值显存
5. **precompute_ref_log_probs=True**（DPO）：预计算参考 log probs，训练时不需要参考模型
6. **BF16 混合精度**：TRL 默认开启，显存减半
7. **packing=True**（SFT）：减少 padding 浪费，等效提升有效批次大小

### 7.4 DPO 特有注意事项

- **beta 选择**：`beta=0.1` 是常用默认值。beta 越大，策略偏离参考模型越少；beta 越小，训练更激进但可能不稳定
- **参考模型管理**：
  - 全参数训练时，`ref_model=None` 会自动创建参考模型副本（显存翻倍）
  - PEFT 训练时，通过 adapter 切换实现参考策略，不额外占显存
  - `precompute_ref_log_probs=True` 可在训练前预计算参考 log probs，释放参考模型显存
- **f-散度选择**：默认 `reverse_kl` 即可；`forward_kl` 更鼓励探索；`js_divergence` 是两者折中

### 7.5 GRPO 特有注意事项

- **num_generations 选择**：默认 8，至少需要 2。更多生成 → 更准确的优势估计 → 但计算成本线性增长
- **beta 设置**：
  - `beta=0.0`（默认）：不加载参考模型，节省显存，纯 RL 信号驱动
  - `beta=0.001`：DeepSeek-R1 推荐值，轻微 KL 约束防止策略崩溃
- **loss_type 选择**：
  - `dapo`（默认）：推荐，消除长度偏差
  - `grpo`：有长度偏差，不推荐
  - `dr_grpo`：常数归一化，简单但有效
- **mask_truncated_completions=True**：DAPO 论文推荐，屏蔽被截断的 completion 防止噪声
- **vLLM 加速**：GRPO 的生成阶段耗时占比高，启用 `use_vllm=True` 可显著加速
- **温度设置**：`temperature=1.0`（默认）保证生成多样性；训练后期可适当降低

### 7.6 数据格式注意事项

- **SFT 数据**：推荐使用对话格式（`messages` 列），利用 `assistant_only_loss=True` 只训练 assistant 回复
- **DPO 数据**：确保 chosen 和 rejected 是对同一 prompt 的不同回复，质量差异应明确
- **GRPO 数据**：只需 prompt 列，不需要预收集回复
- **对话格式**：TRL 使用 `apply_chat_template()` 自动处理 chat template，确保 tokenizer 设置了正确的 chat template

### 7.7 分布式训练注意事项

- 使用 `accelerate` 进行多 GPU 训练：`accelerate launch train.py`
- DeepSpeed ZeRO-3 与 PEFT 结合时，注意 `model.forward` 的特殊处理
- GRPO 的 `ds3_gather_for_generation` 控制 ZeRO-3 下生成时是否聚合权重
- FSDP 配合 `padding_free=True` 和 FlashAttention 效果最佳

### 7.8 监控与调试

- 使用 `wandb` 或 `tensorboard` 监控训练指标
- DPO 关注 `rewards/chosen`、`rewards/rejected`、`rewards/margins` 和 `rewards/accuracies`
- GRPO 关注 `loss/policy`、`loss/kl`、`rewards/mean`、`rewards/std`
- 开启 `log_completions=True`（GRPO）可在日志中查看生成样本质量
- SFT 关注 `loss`、`num_correct_tokens`、`entropy`

### 7.9 常见问题排查

- **NaN 损失**：降低学习率，检查数据中是否存在异常值，确保 BF16 精度
- **OOM（显存不足）**：减小批次大小，启用 gradient_checkpointing + activation_offloading，使用 LoRA
- **DPO 奖励无差异**：增大 beta 值，检查 chosen/rejected 质量差异是否足够
- **GRPO 训练不稳定**：降低学习率，增大 beta（加入 KL 约束），启用 mask_truncated_completions
- **生成质量下降**：检查 KL 约束是否过松（beta 太小），减少训练步数，使用更保守的 epsilon
