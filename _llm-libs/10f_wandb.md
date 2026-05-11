---
title: "Weights & Biases 实验追踪"
excerpt: "wandb.init/log、Sweep超参搜索、Artifact版本管理、HF集成"
collection: llm-libs
permalink: /llm-libs/10f-wandb
category: training
toc: true
---


## 1. 库简介与在 LLM 开发中的作用

[W&B (Weights & Biases)](https://wandb.ai/) 是一个基于云端的机器学习实验追踪与可视化平台，提供实验管理、超参数搜索、模型版本控制、数据集版本管理和团队协作等核心功能。在 LLM 开发中，W&B 扮演着"实验中枢"的角色：

- **实验追踪**：记录每次训练的超参数、指标曲线、系统资源消耗，确保实验可复现
- **超参数搜索**：通过 Sweep 功能自动化搜索最优超参数组合（学习率、batch size、权重衰减等）
- **模型与数据版本管理**：Artifact 机制支持模型检查点、数据集的版本化存储与追溯
- **可视化**：内置丰富的图表类型，支持训练曲线、注意力热力图、生成样本展示等
- **团队协作**：云端仪表盘共享实验结果，支持 Alert 实时通知训练异常
- **生态集成**：与 HuggingFace Transformers、PyTorch Lightning 等深度学习框架无缝集成

## 2. 安装方式

```bash
# 基础安装
pip install wandb

# 安装后需登录（会打开浏览器进行认证）
wandb login

# 也可通过环境变量设置 API Key
export WANDB_API_KEY="your-api-key-here"

# 升级到最新版本
pip install --upgrade wandb
```

## 3. 核心类/函数/工具详细说明

### 3.1 wandb.init() — 初始化实验运行

```python
import wandb

run = wandb.init(
    project="llm-finetuning",       # 项目名称，用于组织实验
    entity="my-team",               # 团队/用户名，默认为个人账号
    config={                        # 超参数配置字典
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 16,
        "model_name": "meta-llama/Llama-2-7b-hf",
        "lora_rank": 8,
        "lora_alpha": 16,
        "weight_decay": 0.01,
        "warmup_steps": 100,
    },
    name="llama2-lora-exp01",       # 运行名称，在仪表盘中显示
    tags=["lora", "llama2", "sft"], # 标签，方便筛选
    notes="LoRA微调Llama2，使用Alpaca数据集",  # 备注说明
    dir="./wandb_logs",             # 本地日志存储目录
    save_code=True,                 # 保存运行脚本的代码快照
    reinit=True,                    # 允许在同一脚本中多次初始化
    resume="allow",                 # 允许恢复中断的运行（"must"必须恢复，"allow"允许恢复）
    id="unique-run-id-123",         # 自定义运行ID，用于恢复时匹配
    group="lora-ablation",          # 分组名称，用于组织相关实验
    job_type="train",               # 作业类型（train/eval/test）
)
```

**关键参数说明**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `project` | str | 项目名，相同项目的运行聚合在一起 |
| `entity` | str | W&B 用户名或团队名 |
| `config` | dict | 超参数配置，运行后可通过 `wandb.config` 访问 |
| `name` | str | 运行的显示名称 |
| `tags` | list | 标签列表，用于筛选 |
| `resume` | str | 恢复策略：`"allow"`、`"must"`、`"never"` |
| `id` | str | 唯一运行标识，用于断点恢复 |
| `group` | str | 实验分组，适合超参数扫描场景 |
| `save_code` | bool | 是否保存代码快照 |

**返回值**：`wandb.sdk.wandb_run.Run` 对象，包含 `id`、`name`、`url`、`config` 等属性。

```python
# 访问运行信息
print(f"运行ID: {run.id}")
print(f"运行URL: {run.url}")
print(f"配置: {run.config}")
```

### 3.2 wandb.log() — 记录指标

```python
# 基础用法：记录标量指标
wandb.log({"train/loss": 2.345, "train/accuracy": 0.67})

# 指定步数（step）
wandb.log({"train/loss": 1.876}, step=100)

# 记录多个指标（同一step下）
wandb.log({
    "train/loss": 1.523,
    "train/learning_rate": 1.8e-5,
    "train/perplexity": 4.58,
    "train/tokens_per_sec": 5200,
}, step=200)

# 使用 commit=False 延迟提交（攒一批再提交）
wandb.log({"train/loss": 1.2}, commit=False)
wandb.log({"train/accuracy": 0.85}, commit=True)  # 此时一起提交

# 记录自定义图表
wandb.log({
    "attention_heatmap": wandb.plots.HeatMap(
        x_labels=range(seq_len),
        y_labels=range(seq_len),
        matrix_values=attention_weights,
    )
})

# 训练循环中的典型用法
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "train/epoch": epoch + step / len(train_dataloader),
            }, step=global_step)
            global_step += 1
```

**关键参数说明**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `data` | dict | 要记录的指标字典，键为指标名，值为数值或W&B对象 |
| `step` | int | 可选，自定义步数；不指定则自动递增 |
| `commit` | bool | 是否立即提交，默认True；设False可合并多条log |

### 3.3 wandb.config — 配置管理

```python
# 方式1：通过 init 传入
run = wandb.init(config={"lr": 2e-5, "epochs": 3})

# 方式2：通过 argparse + config
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=2e-5)
args = parser.parse_args()
wandb.init(config=vars(args))

# 方式3：运行时动态设置
wandb.config.update({"additional_key": "value"})

# 方式4：从配置文件加载
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)
wandb.init(config=config)

# 访问配置
learning_rate = wandb.config.learning_rate
epochs = wandb.config.epochs

# Sweep 中，config 会被 Sweep Agent 自动更新为超参数组合
# 此时 wandb.config 包含 Sweep 定义的参数值
```

### 3.4 Artifact — 模型与数据版本管理

```python
# ===== 创建并记录模型 Artifact =====
# 方式1：直接记录目录
artifact = wandb.Artifact(
    name="llama2-lora-model",       # Artifact名称
    type="model",                    # 类型：model, dataset, code 等
    description="LoRA微调后的Llama2模型",
    metadata={                       # 元数据
        "base_model": "meta-llama/Llama-2-7b-hf",
        "lora_rank": 8,
        "train_steps": 1000,
    }
)
artifact.add_dir("./model_output")   # 添加整个目录
artifact.add_file("./model_output/config.json")  # 添加单个文件
wandb.log_artifact(artifact)

# 方式2：记录单个文件
artifact = wandb.Artifact("tokenizer", type="model")
artifact.add_file("./tokenizer.model")
wandb.log_artifact(artifact)

# ===== 使用 Artifact =====
# 下载最新版本
artifact = wandb.use_artifact("llama2-lora-model:latest")
artifact_dir = artifact.download()  # 下载到本地并返回路径

# 下载特定版本
artifact = wandb.use_artifact("llama2-lora-model:v3")

# 下载特定别名的版本
artifact = wandb.use_artifact("llama2-lora-model:best-model")

# ===== 链接 Artifact 到运行 =====
# 在训练运行中记录Artifact时，W&B自动关联
# 也可以手动将输入/输出Artifact关联到运行
run = wandb.init()
input_artifact = run.use_artifact("dataset:v2")  # 标记为输入
output_artifact = wandb.Artifact("model", type="model")
output_artifact.add_dir("./checkpoints")
run.log_artifact(output_artifact)  # 标记为输出

# ===== 数据集版本管理 =====
dataset_artifact = wandb.Artifact(
    "alpaca-cleaned",
    type="dataset",
    metadata={"source": "tatsu-lab/alpaca", "split": "train", "size": 52002}
)
dataset_artifact.add_file("alpaca_train.jsonl")
wandb.log_artifact(dataset_artifact)
```

**Artifact 版本管理机制**：
- 每次调用 `log_artifact` 都会创建新版本（v0, v1, v2...）
- 可以为版本设置别名（alias），如 `latest`、`best`、`production`
- Artifact 支持增量上传，相同内容的文件不会重复上传
- 元数据（metadata）用于存储与 Artifact 相关的结构化信息

### 3.5 Sweep — 超参数搜索

```python
# ===== 步骤1：定义 Sweep 配置 =====
sweep_config = {
    "method": "bayes",  # 搜索策略: "grid", "random", "bayes"
    "metric": {
        "name": "eval/loss",
        "goal": "minimize"
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3
        },
        "lora_rank": {
            "values": [4, 8, 16, 32]
        },
        "lora_alpha": {
            "distribution": "int_uniform",
            "min": 8,
            "max": 64
        },
        "batch_size": {
            "values": [8, 16, 32]
        },
        "weight_decay": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.1
        },
        "warmup_steps": {
            "distribution": "int_uniform",
            "min": 50,
            "max": 500
        }
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3,
        "s": 2
    },
    "run_cap": 20,  # 最大运行次数
}

# ===== 步骤2：创建 Sweep =====
sweep_id = wandb.sweep(
    sweep=sweep_config,
    project="llm-finetuning"
)
print(f"Sweep ID: {sweep_id}")

# ===== 步骤3：定义训练函数 =====
def train():
    # wandb.init() 会自动接收 Sweep 分配的超参数
    wandb.init()

    # 从 wandb.config 获取当前超参数组合
    config = wandb.config
    learning_rate = config.learning_rate
    lora_rank = config.lora_rank
    batch_size = config.batch_size

    # ... 训练逻辑 ...
    for step in range(100):
        loss = simulate_training(learning_rate, lora_rank, step)
        wandb.log({"train/loss": loss, "eval/loss": loss * 1.1})

    # 记录模型
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file("./model.bin")
    wandb.log_artifact(artifact)

# ===== 步骤4：启动 Sweep Agent =====
wandb.agent(sweep_id, function=train, count=5)  # 运行5次

# ===== 也可以用命令行启动 =====
# wandb agent <entity>/<project>/<sweep_id>
```

**搜索策略对比**：
| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `grid` | 遍历所有参数组合 | 参数空间小、需要穷举 |
| `random` | 随机采样参数组合 | 参数空间大、快速探索 |
| `bayes` | 贝叶斯优化，利用历史结果指导搜索 | 需要高效找到最优解 |

**参数分布类型**：
| 分布类型 | 说明 | 示例 |
|----------|------|------|
| `int_uniform` | 整数均匀分布 | `{"min": 1, "max": 100}` |
| `uniform` | 浮点均匀分布 | `{"min": 0.0, "max": 1.0}` |
| `log_uniform` | 对数均匀分布 | `{"min": 1e-6, "max": 1e-2}` |
| `log_uniform_values` | 对数均匀分布（值域） | `{"min": 1e-6, "max": 1e-3}` |
| `categorical` | 分类变量 | `{"values": ["adam", "sgd"]}` |
| `values` | 离散值列表 | `{"values": [4, 8, 16]}` |

### 3.6 与 HuggingFace Trainer 集成

```python
from transformers import TrainingArguments, Trainer

# 方式1：通过 report_to 参数启用
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    report_to="wandb",  # 关键：将日志输出到W&B
    run_name="llama2-sft-run01",  # W&B中的运行名称
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# 方式2：同时报告到多个平台
training_args = TrainingArguments(
    ...
    report_to=["wandb", "tensorboard"],  # 同时输出到W&B和TensorBoard
)

# 完整训练示例
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# 初始化W&B
wandb.init(
    project="llm-finetuning",
    name="llama2-sft-alpaca",
    config={
        "model_name": "meta-llama/Llama-2-7b-hf",
        "dataset": "tatsu-lab/alpaca",
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 16,
    }
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
dataset = load_dataset("tatsu-lab/alpaca")

training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",
    run_name=wandb.run.name,
    num_train_epochs=wandb.config.epochs,
    per_device_train_batch_size=wandb.config.batch_size,
    learning_rate=wandb.config.learning_rate,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

trainer.train()

# 记录最终模型为Artifact
artifact = wandb.Artifact("llama2-sft-model", type="model")
artifact.add_dir("./results")
wandb.log_artifact(artifact)

wandb.finish()
```

### 3.7 可视化工具

```python
# ===== wandb.Table — 记录结构化数据 =====
# 记录模型生成结果
columns = ["input_prompt", "generated_text", "reference", "score"]
data = [
    ["什么是机器学习？", "机器学习是AI的子领域...", "机器学习是一种...", 0.85],
    ["解释深度学习", "深度学习使用神经网络...", "深度学习是机器学习...", 0.78],
]
table = wandb.Table(columns=columns, data=data)
wandb.log({"generation_examples": table})

# 动态添加行
table = wandb.Table(columns=["step", "input", "output", "loss"])
for step, sample in enumerate(eval_samples):
    output = model.generate(sample["input"])
    table.add_data(step, sample["input"], output, sample["loss"])
wandb.log({"eval_results": table})

# ===== wandb.Image — 记录图像 =====
import matplotlib.pyplot as plt
import numpy as np

# 记录matplotlib图表
fig, ax = plt.subplots()
attention = np.random.rand(12, 12)
ax.imshow(attention, cmap="viridis")
ax.set_title("Attention Heatmap - Layer 6")
wandb.log({"attention_map": wandb.Image(fig)})
plt.close()

# 记录numpy数组为图像
image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
wandb.log({"sample_image": wandb.Image(image_array, caption="Generated sample")})

# ===== wandb.plot — 内置图表类型 =====
# 折线图
data = [[x, x**2] for x in range(100)]
table = wandb.Table(data=data, columns=["x", "y"])
wandb.log({"parabola": wandb.plot.line(table, "x", "y", title="Quadratic")})

# 柱状图
table = wandb.Table(
    data=[["train", 0.95], ["eval", 0.88], ["test", 0.87]],
    columns=["split", "accuracy"]
)
wandb.log({"accuracy_chart": wandb.plot.bar(table, "split", "accuracy")})

# 散点图
table = wandb.Table(data=[[i, i*0.5+np.random.randn()] for i in range(50)],
                    columns=["perplexity", "loss"])
wandb.log({"perplexity_vs_loss": wandb.plot.scatter(table, "perplexity", "loss")})
```

### 3.8 Alert — 训练通知

```python
# 在训练异常时发送通知
import wandb

wandb.init(project="llm-finetuning")

for step in range(10000):
    loss = train_step()

    if loss > 10.0:
        wandb.alert(
            title="训练Loss爆炸！",
            text=f"Step {step}: loss = {loss:.4f}，超过阈值10.0，请检查学习率设置。",
            level=wandb.AlertLevel.WARN,  # INFO, WARN, ERROR
            wait_duration=300,  # 5分钟内不重复发送相同alert
        )

    if loss < 0.01:
        wandb.alert(
            title="训练收敛完成",
            text=f"Step {step}: loss = {loss:.6f}，训练已收敛。",
            level=wandb.AlertLevel.INFO,
        )

# Alert 会在 W&B 网页端显示，并可通过 Slack/Webhook 集成发送到其他渠道
```

### 3.9 离线模式

```python
# 方式1：通过环境变量设置
import os
os.environ["WANDB_MODE"] = "offline"  # 离线模式，日志保存在本地

# 方式2：通过 wandb.init 参数
wandb.init(mode="offline")

# 方式3：通过 wandb settings
wandb.init(settings=wandb.Settings(mode="offline"))

# 离线模式下，所有日志保存在本地 wandb/ 目录
# 训练完成后，使用命令行同步到云端：
# wandb sync ./wandb/offline-run-20231001_123456-abc123

# 批量同步所有离线运行
# wandb sync --sync-all

# 禁用W&B（完全不记录）
os.environ["WANDB_MODE"] = "disabled"
# 或
wandb.init(mode="disabled")
```

### 3.10 wandb.finish() — 结束运行

```python
# 标记运行结束，上传所有未同步的数据
wandb.finish(exit_code=0)  # 0=成功，非0=失败

# 不调用finish()时，运行会在脚本退出时自动结束
# 但建议显式调用，特别是在Jupyter Notebook中
```

## 4. 在 LLM 开发中的典型使用场景和代码示例

### 4.1 大语言模型微调实验追踪

```python
import wandb
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def main():
    # 1. 初始化实验
    wandb.init(
        project="llm-instruction-tuning",
        name="mistral-7b-lora-alpaca",
        tags=["mistral", "lora", "sft"],
        config={
            "model_name": "mistralai/Mistral-7B-v0.1",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 3e-4,
            "epochs": 3,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 512,
            "dataset": "tatsu-lab/alpaca",
        }
    )
    config = wandb.config

    # 2. 加载模型与LoRA配置
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 记录数据集Artifact
    dataset_artifact = wandb.Artifact("alpaca-dataset", type="dataset")
    dataset_artifact.add_reference("https://huggingface.co/datasets/tatsu-lab/alpaca")
    wandb.log_artifact(dataset_artifact)

    # 4. 训练
    training_args = TrainingArguments(
        output_dir="./results",
        report_to="wandb",
        run_name=wandb.run.name,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )
    trainer.train()

    # 5. 记录模型Artifact
    model_artifact = wandb.Artifact("mistral-lora-model", type="model")
    model_artifact.add_dir("./results/checkpoint-best")
    wandb.log_artifact(model_artifact)

    # 6. 记录生成样本
    table = wandb.Table(columns=["prompt", "generated", "reference"])
    for sample in eval_samples[:20]:
        generated = generate_text(model, sample["prompt"])
        table.add_data(sample["prompt"], generated, sample["output"])
    wandb.log({"generation_samples": table})

    wandb.finish()

if __name__ == "__main__":
    main()
```

### 4.2 超参数搜索（Sweep）完整示例

```python
import wandb

sweep_config = {
    "method": "bayes",
    "metric": {"name": "eval/loss", "goal": "minimize"},
    "early_terminate": {"type": "hyperband", "min_iter": 5, "s": 2},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5, "max": 5e-4
        },
        "lora_rank": {"values": [4, 8, 16, 32]},
        "lora_alpha": {
            "distribution": "int_uniform",
            "min": 8, "max": 64
        },
        "weight_decay": {
            "distribution": "uniform",
            "min": 0.0, "max": 0.1
        },
    },
}

sweep_id = wandb.sweep(sweep_config, project="llm-hp-search")

def sweep_train():
    wandb.init()
    config = wandb.config

    # 使用Sweep分配的超参数进行训练
    model = setup_model(lora_rank=config.lora_rank, lora_alpha=config.lora_alpha)
    train_result = train_model(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    wandb.log({
        "eval/loss": train_result.eval_loss,
        "eval/perplexity": train_result.perplexity,
    })

wandb.agent(sweep_id, function=sweep_train, count=10)
```

### 4.3 RLHF 训练追踪

```python
import wandb

wandb.init(
    project="rlhf-training",
    name="llama2-7b-rlhf",
    config={
        "ppo_epochs": 4,
        "kl_coeff": 0.2,
        "reward_model": "reward-model-v2",
        "actor_lr": 1e-6,
        "critic_lr": 5e-6,
    }
)

# PPO训练循环
for ppo_epoch in range(num_ppo_epochs):
    for batch in ppo_dataloader:
        # 生成回复
        query_tensors, response_tensors = generate_responses(actor_model, batch)

        # 计算奖励
        rewards = compute_rewards(reward_model, query_tensors, response_tensors)

        # PPO更新
        stats = ppo_step(
            actor_model, critic_model,
            query_tensors, response_tensors, rewards
        )

        # 记录RLHF特有指标
        wandb.log({
            "rlhf/mean_reward": stats["mean_reward"],
            "rlhf/kl_divergence": stats["kl_divergence"],
            "rlhf/actor_loss": stats["actor_loss"],
            "rlhf/critic_loss": stats["critic_loss"],
            "rlhf/entropy": stats["entropy"],
            "rlhf/approx_kl": stats["approx_kl"],
            "rlhf/clip_fraction": stats["clip_fraction"],
            "rlhf/explained_variance": stats["explained_variance"],
        }, step=global_step)

        # 定期记录生成样本
        if global_step % 100 == 0:
            table = wandb.Table(columns=["prompt", "response", "reward"])
            for i in range(min(5, len(query_tensors))):
                prompt = tokenizer.decode(query_tensors[i])
                response = tokenizer.decode(response_tensors[i])
                table.add_data(prompt, response, rewards[i].item())
            wandb.log({"rlhf/samples": table}, step=global_step)
```

## 5. 数学原理

### 5.1 贝叶斯超参数优化（Bayesian Sweep）

W&B 的 Bayesian Sweep 基于高斯过程（Gaussian Process）代理模型：

1. **代理模型**：用高斯过程拟合超参数 $\mathbf{x}$ 与目标指标 $y$ 之间的映射：
$$y \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$
   其中 $m(\mathbf{x})$ 为均值函数，$k(\mathbf{x}, \mathbf{x}')$ 为核函数（通常使用 Matérn 核）。

2. **采集函数**：利用代理模型的预测分布，通过采集函数选择下一组超参数：
   - **Expected Improvement (EI)**：
   $$\text{EI}(\mathbf{x}) = \mathbb{E}[\max(y - y^*, 0)]$$
   其中 $y^*$ 为当前最优观测值。

3. **迭代过程**：每次运行后更新代理模型，利用新观测改进预测，逐步聚焦到最优超参数区域。

### 5.2 Hyperband 早停

Hyperband 基于 Successive Halving 策略，数学上通过分配资源 $R$ 和缩减因子 $\eta$：
$$s_{\max} = \lfloor \log_\eta(R) \rfloor$$
每轮保留 $\frac{1}{\eta}$ 的候选，将资源集中到有潜力的超参数组合上。

## 6. 代码原理/架构原理

### 6.1 客户端-服务器架构

```
┌──────────────────────────────────────────────┐
│              W&B Cloud Server                │
│  ┌─────────────┐  ┌──────────────────────┐   │
│  │  API Gateway │  │   Storage Backend    │   │
│  │  (GraphQL)   │  │  ┌───────┐ ┌──────┐ │   │
│  └──────┬───────┘  │  │Metadata│ │Files │ │   │
│         │          │  │  DB    │ │Store │ │   │
│         │          │  └───────┘ └──────┘ │   │
│         │          └──────────────────────┘   │
└─────────┼─────────────────────────────────────┘
          │ HTTPS
          │
┌─────────┴─────────────────────────────────────┐
│              W&B Client (SDK)                  │
│  ┌──────────────┐  ┌────────────────────┐     │
│  │ wandb.init() │  │ Internal File      │     │
│  │ wandb.log()  │──│ Writer & Buffer    │     │
│  │ wandb.config │  │ (异步上传)          │     │
│  └──────────────┘  └────────────────────┘     │
│  ┌──────────────┐  ┌────────────────────┐     │
│  │ Artifact API  │  │ Sweep Agent       │     │
│  │ (版本管理)    │  │ (超参搜索协调)     │     │
│  └──────────────┘  └────────────────────┘     │
└────────────────────────────────────────────────┘
```

### 6.2 数据流

1. **初始化阶段**：`wandb.init()` 向服务器创建新的 Run 记录，获取唯一 Run ID
2. **记录阶段**：`wandb.log()` 将指标数据写入本地缓冲区，异步批量上传到服务器
3. **Artifact阶段**：`log_artifact()` 计算文件哈希，增量上传唯一内容块
4. **同步阶段**：`wandb.finish()` 确保所有缓冲数据上传完毕

### 6.3 关键设计决策

- **异步上传**：日志记录不阻塞训练循环，后台线程处理网络I/O
- **增量存储**：Artifact 使用内容寻址存储（类似Git），相同文件只存一份
- **断点恢复**：通过 `resume` 参数和 Run ID 支持中断后继续记录
- **Sweep协调**：Sweep Controller 在服务器端运行，Agent 通过轮询获取下一次超参数组合

## 7. 常见注意事项和最佳实践

### 7.1 性能相关

```python
# 1. 控制log频率，避免过多小批次上传
# 不好：每步都log
for step in range(100000):
    wandb.log({"loss": loss}, step=step)  # 太频繁

# 好：每N步log一次
LOG_INTERVAL = 50
for step in range(100000):
    if step % LOG_INTERVAL == 0:
        wandb.log({"loss": loss}, step=step)

# 2. 使用 commit=False 合并同一步的多次log
wandb.log({"train/loss": loss}, commit=False)
wandb.log({"train/lr": lr}, commit=False)
wandb.log({"train/grad_norm": grad_norm}, commit=True)  # 最后一次commit

# 3. 大文件使用Artifact，不要用log
# 不好
wandb.log({"model_file": wandb.File("model.pt")]})  # 不适合大文件

# 好
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pt")
wandb.log_artifact(artifact)
```

### 7.2 实验组织最佳实践

```python
# 1. 使用命名约定
wandb.init(
    name=f"{model_name}-{method}-{dataset}-{timestamp}",
    # 如: "llama2-7b-lora-alpaca-20240101"
)

# 2. 善用tags进行筛选
wandb.init(tags=["lora", "v2", "production-candidate"])

# 3. 使用group组织Sweep实验
wandb.init(group="lora-rank-ablation")

# 4. 配置层次化
wandb.init(config={
    "model": {"name": "llama2-7b", "dtype": "bfloat16"},
    "training": {"lr": 2e-5, "epochs": 3, "batch_size": 16},
    "lora": {"rank": 8, "alpha": 16, "dropout": 0.05},
    "data": {"name": "alpaca", "split": "train", "size": 52002},
})
```

### 7.3 常见陷阱

1. **忘记调用 `wandb.finish()`**：在Jupyter Notebook中尤其重要，否则运行状态会卡在"running"
2. **离线模式忘记同步**：离线训练完成后需手动 `wandb sync` 上传数据
3. **Sweep中忘记在train函数内调用 `wandb.init()`**：Sweep Agent 会自动注入超参数到 `wandb.config`，前提是调用了 `init`
4. **Artifact命名冲突**：同一项目下Artifact名称必须唯一，不同类型也不能重名
5. **step参数不一致**：混用自动step和手动step会导致图表错乱，建议始终显式指定step
6. **环境变量优先级**：`WANDB_MODE` 环境变量优先于代码中的设置，排查问题时需注意

### 7.4 团队协作

```python
# 1. 使用entity指定团队
wandb.init(entity="my-research-team", project="llm-projects")

# 2. 使用报告(Reports)功能组织实验对比
# 在W&B UI中创建Report，嵌入多个运行的图表

# 3. 使用Alert通知关键节点
wandb.alert(title="训练完成", text="模型已收敛", level=wandb.AlertLevel.INFO)

# 4. 使用Artifact共享模型和数据
# 团队成员可通过 use_artifact 下载他人上传的Artifact
artifact = wandb.use_artifact("my-research-team/llm-projects/best-model:v5")
```

### 7.5 安全与隐私

```python
# 1. 敏感数据不要log
# 不好：记录用户原始输入
wandb.log({"user_input": sensitive_text})

# 好：记录脱敏后的统计信息
wandb.log({"input_length": len(text), "input_type": "question"})

# 2. 使用私有项目
# W&B默认项目为私有，团队内可见
# 需要公开时显式设置

# 3. API Key安全
# 不要将API Key硬编码在代码中
# 使用环境变量或 wandb login 命令
```
