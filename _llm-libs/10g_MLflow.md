---
title: "MLflow ML生命周期管理"
excerpt: "Tracking/Models/Registry/Projects、autolog、HuggingFace集成"
collection: llm-libs
permalink: /llm-libs/10g-mlflow
category: training
toc: true
---


## 1. 库简介与在 LLM 开发中的作用

[MLflow](https://mlflow.org/) 是一个开源的机器学习生命周期管理平台，由 Databricks 开发并维护，提供实验追踪（Tracking）、模型管理（Model Registry）、项目打包（Projects）和部署管道（Recipes）四大核心功能。在 LLM 开发中，MLflow 的角色包括：

- **实验追踪**：记录每次训练的参数、指标和产物（Artifact），支持本地和远程追踪服务器
- **模型注册与版本管理**：Model Registry 提供模型版本控制、阶段管理（Staging/Production/Archived）
- **自动日志**：`mlflow.autolog()` 自动捕获框架的训练指标和模型，零代码侵入
- **模型签名**：记录模型输入/输出的 Schema，确保部署时数据格式正确
- **与 HuggingFace 深度集成**：`mlflow.transformers` 模块支持 Transformer 模型的日志记录、保存和加载
- **可自托管**：完全开源，可部署在私有环境中，数据不离开企业网络

与 W&B 的关键区别：MLflow 是开源的、可完全自托管的方案，适合对数据隐私有严格要求的企业场景。

## 2. 安装方式

```bash
# 基础安装
pip install mlflow

# 安装含额外依赖的版本
pip install mlflow[extras]       # 包含所有可选依赖

# 仅安装特定集成
pip install mlflow[transformers] # HuggingFace Transformers 集成
pip install mlflow[pytorch]      # PyTorch 集成

# 安装 UI 依赖（通常已包含在基础安装中）
pip install mlflow

# 启动追踪服务器（用于远程实验追踪和模型注册）
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts

# 升级
pip install --upgrade mlflow
```

## 3. 核心类/函数/工具详细说明

### 3.1 Tracking — 实验追踪

#### mlflow.start_run() — 启动实验运行

```python
import mlflow

# 基础用法
with mlflow.start_run():
    mlflow.log_param("learning_rate", 2e-5)
    mlflow.log_metric("train_loss", 1.234)

# 指定实验名称
mlflow.set_experiment("LLM-FineTuning")
with mlflow.start_run():
    # ...

# 完整参数
with mlflow.start_run(
    run_name="llama2-lora-sft-01",         # 运行名称
    experiment_id="1234567890",             # 实验ID（与experiment_name二选一）
    run_id="existing-run-id",              # 恢复已有运行
    description="LoRA微调Llama2",          # 运行描述
    tags={"framework": "transformers",     # 标签
          "method": "lora",
          "dataset": "alpaca"},
) as run:
    print(f"Run ID: {run.info.run_id}")
    print(f"Artifact URI: {run.info.artifact_uri}")

# 嵌套运行（父-子关系）
with mlflow.start_run(run_name="parent") as parent_run:
    mlflow.log_param("model", "llama2-7b")

    with mlflow.start_run(run_name="child-lora", nested=True) as child_run:
        mlflow.log_param("lora_rank", 8)
        mlflow.log_metric("eval_loss", 1.5)

    with mlflow.start_run(run_name="child-full", nested=True) as child_run:
        mlflow.log_param("full_finetune", True)
        mlflow.log_metric("eval_loss", 1.2)
```

**关键参数说明**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `run_name` | str | 运行名称，显示在UI中 |
| `experiment_id` | str | 实验ID |
| `run_id` | str | 已有运行的ID，用于恢复 |
| `description` | str | 运行描述（Markdown支持） |
| `tags` | dict | 标签字典，用于筛选和分类 |
| `nested` | bool | 是否为嵌套（子）运行 |

#### mlflow.log_param / log_params — 记录参数

```python
# 记录单个参数
mlflow.log_param("learning_rate", 2e-5)
mlflow.log_param("model_name", "meta-llama/Llama-2-7b-hf")
mlflow.log_param("lora_rank", 8)

# 批量记录参数
mlflow.log_params({
    "learning_rate": 2e-5,
    "epochs": 3,
    "batch_size": 16,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "max_seq_length": 512,
    "optimizer": "adamw_torch",
    "scheduler": "cosine",
})

# 注意：参数值仅支持 str, int, float, bool, None
# 复杂结构需序列化为字符串
import json
mlflow.log_param("target_modules", json.dumps(["q_proj", "v_proj", "k_proj"]))
```

#### mlflow.log_metric / log_metrics — 记录指标

```python
# 记录单个指标（当前值）
mlflow.log_metric("train_loss", 1.234)

# 记录带时间步的指标
mlflow.log_metric("train_loss", 2.345, step=10)
mlflow.log_metric("train_loss", 1.876, step=20)
mlflow.log_metric("train_loss", 1.523, step=30)

# 批量记录指标
mlflow.log_metrics({
    "train_loss": 1.523,
    "train_accuracy": 0.67,
    "train_learning_rate": 1.8e-5,
    "train_perplexity": 4.58,
})

# 批量记录带时间步
mlflow.log_metrics({
    "eval_loss": 1.789,
    "eval_perplexity": 5.98,
    "eval_accuracy": 0.62,
}, step=100)

# 注意：metric值仅支持 float 或 int，不支持None
# 同一指标在同一step下多次log，后值覆盖前值
```

#### mlflow.log_artifact / log_artifacts — 记录产物

```python
import mlflow

# 记录单个文件
mlflow.log_artifact("model_config.json")
mlflow.log_artifact("training_log.txt", artifact_path="logs")  # 指定子目录

# 记录整个目录
mlflow.log_artifacts("./model_output", artifact_path="model")

# 记录文本内容（无需先写文件）
mlflow.log_text("Training completed successfully", "status.txt")
mlflow.log_text(json.dumps(config_dict, indent=2), "config.json")

# 记录字典
mlflow.log_dict({"metric": 0.95, "params": {"lr": 2e-5}}, "results.json")

# 记录图像
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
mlflow.log_figure(fig, "training_curve.png")
plt.close()

# 记录numpy图像
import numpy as np
image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
mlflow.log_image(image, "sample_image.png")
```

### 3.2 Experiments — 实验管理

```python
# 创建实验（返回实验ID）
experiment_id = mlflow.create_experiment(
    name="LLM-FineTuning",
    artifact_location="./artifacts",        # Artifact存储位置
    tags={"team": "nlp", "project": "llm"}, # 实验标签
)

# 设置当前实验（通过名称）
mlflow.set_experiment("LLM-FineTuning")

# 设置当前实验（通过ID）
mlflow.set_experiment(experiment_id="1234567890")

# 搜索实验
experiments = mlflow.search_experiments(
    filter_string="tags.team = 'nlp'",
    order_by=["creation_time DESC"],
    max_results=10,
)
for exp in experiments:
    print(f"实验: {exp.name}, ID: {exp.experiment_id}")

# 获取实验信息
experiment = mlflow.get_experiment_by_name("LLM-FineTuning")
print(f"实验ID: {experiment.experiment_id}")
print(f"Artifact位置: {experiment.artifact_location}")

# 搜索运行
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="metrics.eval_loss < 1.5 and params.model_name = 'llama2-7b'",
    order_by=["metrics.eval_loss ASC"],
    max_results=20,
)
# 返回pandas DataFrame
print(runs[["run_id", "metrics.eval_loss", "params.learning_rate"]])
```

### 3.3 Models — 模型日志与加载

#### mlflow.log_model() — 记录模型

```python
import mlflow
import mlflow.pyfunc
import mlflow.transformers

# ===== 方式1：记录 Transformers 模型 =====
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 使用 mlflow.transformers.log_model（推荐）
mlflow.transformers.log_model(
    transformers_model={
        "model": model,
        "tokenizer": tokenizer,
    },
    artifact_path="model",
    task="text-generation",                  # 任务类型
    model_config={                           # 推理配置
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    signature=signature,                     # 模型签名
    input_example=input_example,             # 输入示例
    pip_requirements=[                       # 依赖列表
        "transformers>=4.34.0",
        "torch>=2.0.0",
        "accelerate>=0.24.0",
    ],
)

# ===== 方式2：记录 PyTorch 模型 =====
import mlflow.pytorch

mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="pytorch-model",
    pip_requirements=["torch>=2.0.0"],
)

# ===== 方式3：使用 Python 函数自定义模型 =====
import mlflow.pyfunc

class LLMWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["model_dir"])
        self.model = AutoModelForCausalLM.from_pretrained(context.artifacts["model_dir"])

    def predict(self, context, model_input):
        prompts = model_input["prompt"].tolist()
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            results.append(self.tokenizer.decode(outputs[0]))
        return results

mlflow.pyfunc.log_model(
    artifact_path="llm-wrapper",
    python_model=LLMWrapper(),
    artifacts={"model_dir": "./model_output"},
    signature=signature,
)

# ===== 模型签名（Model Signature）=====
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, Schema

# 定义输入Schema
input_schema = Schema([
    ColSpec("string", "prompt"),
    ColSpec("long", "max_tokens"),
])
# 定义输出Schema
output_schema = Schema([
    ColSpec("string", "generated_text"),
])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# 从数据推断签名（更方便）
import pandas as pd
input_example = pd.DataFrame({
    "prompt": ["什么是机器学习？"],
    "max_tokens": [100],
})
signature = mlflow.models.infer_signature(
    input_example,
    output_example=pd.DataFrame({"generated_text": ["机器学习是..."]}),
)

mlflow.transformers.log_model(
    transformers_model={"model": model, "tokenizer": tokenizer},
    artifact_path="model",
    signature=signature,
    input_example=input_example,
)
```

#### mlflow.save_model() — 保存模型到本地

```python
# 保存到本地目录（不记录到MLflow Tracking）
mlflow.transformers.save_model(
    transformers_model={"model": model, "tokenizer": tokenizer},
    path="./saved_model",
    task="text-generation",
)
```

#### mlflow.pyfunc.load_model() — 加载模型

```python
# 从Tracking URI加载
model_uri = "runs:/<run-id>/model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# 从Model Registry加载
model_uri = "models:/llama2-lora/Production"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# 从本地路径加载
loaded_model = mlflow.pyfunc.load_model("./saved_model")

# 使用加载的模型进行推理
import pandas as pd
input_data = pd.DataFrame({"prompt": ["什么是深度学习？"], "max_tokens": [100]})
predictions = loaded_model.predict(input_data)

# 加载为原始Transformers模型
transformers_model = mlflow.transformers.load_model(model_uri)
# 返回 pipeline 对象，可直接调用
result = transformers_model("什么是深度学习？")
```

### 3.4 Model Registry — 模型注册与版本管理

```python
import mlflow

# ===== 注册模型 =====
# 方式1：通过log_model时指定registered_model_name
mlflow.transformers.log_model(
    transformers_model={"model": model, "tokenizer": tokenizer},
    artifact_path="model",
    registered_model_name="llama2-lora-sft",  # 自动注册
)

# 方式2：运行完成后手动注册
mlflow.register_model(
    model_uri="runs:/<run-id>/model",
    name="llama2-lora-sft",
    tags={"framework": "transformers", "method": "lora"},
)

# ===== 查看模型版本 =====
client = mlflow.tracking.MlflowClient()

# 获取所有注册模型
registered_models = client.search_registered_models()
for rm in registered_models:
    print(f"模型: {rm.name}, 最新版本: {rm.latest_versions}")

# 获取特定模型的所有版本
model_versions = client.search_model_versions("name='llama2-lora-sft'")
for mv in model_versions:
    print(f"版本: {mv.version}, 阶段: {mv.current_stage}, 运行ID: {mv.run_id}")

# 获取特定版本详情
model_version = client.get_model_version("llama2-lora-sft", "1")
print(f"描述: {model_version.description}")
print(f"标签: {model_version.tags}")

# ===== 阶段管理 =====
# 将模型版本推到 Staging
client.transition_model_version_stage(
    name="llama2-lora-sft",
    version=1,
    stage="Staging",       # None -> Staging -> Production -> Archived
)

# 从 Staging 推到 Production
client.transition_model_version_stage(
    name="llama2-lora-sft",
    version=1,
    stage="Production",
    archive_existing_versions=True,  # 归档当前Production版本
)

# ===== 加载特定阶段的模型 =====
# 加载Production版本
model_uri = "models:/llama2-lora-sft/Production"
model = mlflow.pyfunc.load_model(model_uri)

# 加载特定版本号
model_uri = "models:/llama2-lora-sft/3"
model = mlflow.pyfunc.load_model(model_uri)

# 加载最新版本
model_uri = "models:/llama2-lora-sft/latest"
model = mlflow.pyfunc.load_model(model_uri)

# ===== 添加描述和标签 =====
client.update_model_version(
    name="llama2-lora-sft",
    version=1,
    description="LoRA微调Llama2-7b，Alpaca数据集，eval_loss=1.234",
)

client.set_model_version_tag(
    name="llama2-lora-sft",
    version=1,
    key="approved_by",
    value="team-lead",
)

# ===== 删除模型版本 =====
client.delete_model_version(
    name="llama2-lora-sft",
    version=2,
)
```

**模型阶段流转**：
```
None → Staging → Production → Archived
                  ↑               ↓
                  └───────────────┘ (可回退)
```

| 阶段 | 说明 |
|------|------|
| `None` | 刚注册，未分配阶段 |
| `Staging` | 正在验证/测试中 |
| `Production` | 已上线部署 |
| `Archived` | 已归档，不再使用 |

### 3.5 Projects — 项目打包与运行

```python
# ===== MLproject 文件 =====
# 在项目根目录创建 MLproject 文件：
#
# name: llm-finetuning
#
# conda_env: conda.yaml          # 或 docker_env
# docker_env:
#   image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
#
# entry_points:
#   main:
#     parameters:
#       learning_rate: {type: float, default: 2e-5}
#       epochs: {type: int, default: 3}
#       lora_rank: {type: int, default: 8}
#       model_name: {type: string, default: "meta-llama/Llama-2-7b-hf"}
#     command: "python train.py --lr {learning_rate} --epochs {epochs} --lora_rank {lora_rank} --model {model_name}"
#
#   evaluate:
#     parameters:
#       model_uri: {type: string}
#       dataset: {type: string, default: "eval"}
#     command: "python evaluate.py --model_uri {model_uri} --dataset {dataset}"

# 运行项目
mlflow.run(
    uri="./",                          # 项目路径或Git URL
    entry_point="main",                # 入口点名称
    parameters={                       # 参数覆盖
        "learning_rate": 3e-4,
        "epochs": 5,
        "lora_rank": 16,
    },
    experiment_name="LLM-FineTuning",  # 实验名称
    env_manager="conda",              # 环境管理：conda, virtualenv, local
)

# 运行远程Git项目
mlflow.run(
    uri="https://github.com/my-org/llm-training.git",
    entry_point="main",
    version="v1.0.0",                 # Git分支/标签/commit
    parameters={"learning_rate": 1e-4},
)
```

### 3.6 Recipes — 预定义 ML 管道模板

```python
# MLflow Recipes（原 MLflow Pipelines）提供标准化的ML管道模板
# 适合常见的LLM开发流程

# ===== 使用 Recipe =====
# 1. 创建 recipe 配置文件 (recipe.yaml)
#
# recipe: classification/v1  # 或 regression/v1
# target: label
# primary_metric: accuracy
#
# data:
#   location: "./data/train.csv"
#   format: csv
#
# model:
#   algorithm: xgboost
#
# steps:
#   train:
#     estimator_params:
#       n_estimators: 100
#       max_depth: 6

# 2. 运行 recipe
from mlflow.recipes import Recipe

recipe = Recipe(profile="local")
recipe.run()
recipe.inspect()  # 查看各步骤结果

# 3. 预测
prediction = recipe.predict({"feature1": 1.0, "feature2": 2.0})
```

### 3.7 UI — 追踪服务器与可视化

```bash
# 启动本地追踪服务器
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --serve-artifacts

# 启动并指定Artifact存储到S3
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql://user:pass@db:5432/mlflow \
    --default-artifact-root s3://my-bucket/mlflow/artifacts

# 仅启动UI（使用已有数据库）
mlflow ui --port 5000
```

```python
# 在代码中设置追踪服务器地址
mlflow.set_tracking_uri("http://localhost:5000")

# 或通过环境变量
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# 带认证的远程服务器
os.environ["MLFLOW_TRACKING_USERNAME"] = "user"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"
```

### 3.8 自动日志 — mlflow.autolog()

```python
# ===== 通用自动日志 =====
import mlflow
mlflow.autolog()  # 自动检测并启用所有支持的框架

# ===== Transformers 自动日志 =====
mlflow.transformers.autolog(
    log_models=True,              # 是否自动记录模型
    log_input_examples=True,      # 是否记录输入示例
    log_model_signatures=True,    # 是否推断并记录模型签名
    log_traces=True,              # 是否记录推理追踪
)

# ===== PyTorch 自动日志 =====
mlflow.pytorch.autolog(
    log_models=True,
    log_datasets=True,
)

# 完整示例：自动日志 + Transformers
import mlflow

mlflow.transformers.autolog()

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# autolog 会自动记录：
# - 训练超参数（从TrainingArguments提取）
# - 训练指标（loss、learning_rate等）
# - 评估指标（eval_loss、eval_accuracy等）
# - 模型文件（最终模型）
# - Tokenizer
# - 训练状态
trainer.train()
```

### 3.9 与 HuggingFace 集成

```python
import mlflow
import mlflow.transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ===== 记录 Transformers Pipeline =====
# 创建pipeline
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-hf",
    tokenizer="meta-llama/Llama-2-7b-hf",
)

# 记录pipeline
with mlflow.start_run():
    mlflow.transformers.log_model(
        transformers_model=pipe,
        artifact_path="text-generator",
        task="text-generation",
    )

# ===== 记录模型+Tokenizer字典 =====
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

with mlflow.start_run():
    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer,
        },
        artifact_path="llama2",
        task="text-generation",
    )

# ===== 加载 Transformers 模型 =====
# 加载为pipeline
loaded_pipe = mlflow.transformers.load_model(
    model_uri="runs:/<run-id>/llama2",
    return_type="pipeline",  # 返回pipeline对象
)
result = loaded_pipe("Once upon a time")

# 加载为组件字典
components = mlflow.transformers.load_model(
    model_uri="runs:/<run-id>/llama2",
    return_type="components",  # 返回 {"model": ..., "tokenizer": ...}
)
model = components["model"]
tokenizer = components["tokenizer"]

# ===== 推理追踪（Tracing）=====
# MLflow Tracing 记录模型推理的详细步骤
mlflow.transformers.autolog(log_traces=True)

with mlflow.start_run():
    # 自动追踪pipeline的推理过程
    result = pipe("Explain quantum computing")
    # 可以在MLflow UI中查看推理追踪信息
```

## 4. 在 LLM 开发中的典型使用场景和代码示例

### 4.1 完整的 LLM 微调实验追踪

```python
import mlflow
import mlflow.transformers
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch

# 设置追踪服务器
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("LLM-Instruction-Tuning")

# 定义超参数
params = {
    "model_name": "mistralai/Mistral-7B-v0.1",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 3e-4,
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 512,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "optimizer": "adamw_torch",
    "scheduler": "cosine",
    "dataset": "tatsu-lab/alpaca",
    "bf16": True,
}

with mlflow.start_run(
    run_name="mistral-7b-lora-alpaca",
    description="Mistral-7B LoRA微调，使用Alpaca数据集",
    tags={"framework": "transformers", "method": "lora", "base_model": "mistral-7b"},
) as run:

    # 1. 记录所有参数
    mlflow.log_params(params)

    # 2. 加载和准备模型
    model = AutoModelForCausalLM.from_pretrained(
        params["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    # 3. 配置LoRA
    lora_config = LoraConfig(
        r=params["lora_rank"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable_params = model.print_trainable_parameters()

    # 4. 记录可训练参数信息
    mlflow.log_param("trainable_params_pct", trainable_params)

    # 5. 训练
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=params["epochs"],
        per_device_train_batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        weight_decay=params["weight_decay"],
        warmup_ratio=params["warmup_ratio"],
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=params["bf16"],
        report_to="none",  # 不使用其他追踪工具
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    train_result = trainer.train()

    # 6. 记录训练指标
    mlflow.log_metrics({
        "final_train_loss": train_result.training_loss,
        "total_train_steps": train_result.global_step,
    })

    # 7. 评估
    eval_result = trainer.evaluate()
    mlflow.log_metrics({
        "eval_loss": eval_result["eval_loss"],
        "eval_perplexity": 2.71828 ** eval_result["eval_loss"],
        "eval_runtime": eval_result["eval_runtime"],
    })

    # 8. 记录模型
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="model",
        task="text-generation",
        registered_model_name="mistral-7b-lora-alpaca",
    )

    # 9. 记录训练日志
    mlflow.log_artifact("./results/trainer_state.json", "training_logs")

    print(f"运行完成！Run ID: {run.info.run_id}")
    print(f"模型已注册为: mistral-7b-lora-alpaca")
```

### 4.2 模型注册与生产部署流程

```python
import mlflow

client = mlflow.tracking.MlflowClient()

# ===== Step 1: 注册新模型版本（已在训练时完成）=====
# mlflow.transformers.log_model(..., registered_model_name="llm-service")

# ===== Step 2: 验证模型（推到Staging）=====
client.transition_model_version_stage(
    name="mistral-7b-lora-alpaca",
    version=1,
    stage="Staging",
)

# ===== Step 3: 运行评估 =====
model_uri = "models:/mistral-7b-lora-alpaca/Staging"
model = mlflow.transformers.load_model(model_uri, return_type="pipeline")

# 在评估集上测试
eval_results = evaluate_model(model, eval_dataset)
print(f"评估结果: {eval_results}")

# ===== Step 4: 审核通过，推到Production =====
if eval_results["accuracy"] > 0.85:
    client.transition_model_version_stage(
        name="mistral-7b-lora-alpaca",
        version=1,
        stage="Production",
        archive_existing_versions=True,  # 归档旧Production版本
    )
    # 添加审核标签
    client.set_model_version_tag(
        name="mistral-7b-lora-alpaca",
        version=1,
        key="approved_by",
        value="tech-lead",
    )
    client.update_model_version(
        name="mistral-7b-lora-alpaca",
        version=1,
        description="通过评估，准确率>85%，已部署到生产环境",
    )

# ===== Step 5: 生产环境加载模型 =====
# 部署服务总是加载Production版本
production_model_uri = "models:/mistral-7b-lora-alpaca/Production"
production_model = mlflow.transformers.load_model(
    production_model_uri, return_type="pipeline"
)

# ===== Step 6: 回滚（如有问题）=====
# 归档当前Production版本
client.transition_model_version_stage(
    name="mistral-7b-lora-alpaca",
    version=1,
    stage="Archived",
)
# 恢复之前的版本
client.transition_model_version_stage(
    name="mistral-7b-lora-alpaca",
    version=0,  # 之前的版本号
    stage="Production",
)
```

### 4.3 多模型对比实验

```python
import mlflow
import pandas as pd

mlflow.set_experiment("LLM-Model-Comparison")

models_to_compare = [
    {"name": "meta-llama/Llama-2-7b-hf", "type": "base", "lr": 2e-5},
    {"name": "meta-llama/Llama-2-7b-chat-hf", "type": "instruct", "lr": 2e-5},
    {"name": "mistralai/Mistral-7B-v0.1", "type": "base", "lr": 3e-5},
    {"name": "mistralai/Mistral-7B-Instruct-v0.1", "type": "instruct", "lr": 3e-5},
]

results = []

for model_config in models_to_compare:
    with mlflow.start_run(
        run_name=f"compare-{model_config['name'].split('/')[-1]}",
        tags={"model_type": model_config["type"]},
    ):
        mlflow.log_params(model_config)

        # 训练和评估
        metrics = train_and_evaluate(model_config)
        mlflow.log_metrics(metrics)

        results.append({
            "model": model_config["name"],
            "type": model_config["type"],
            **metrics,
        })

# 使用MLflow的对比功能
df = mlflow.search_runs(
    experiment_names=["LLM-Model-Comparison"],
    order_by=["metrics.eval_loss ASC"],
)
print(df[["tags.mlflow.runName", "metrics.eval_loss", "metrics.eval_perplexity"]])
```

## 5. 数学原理

### 5.1 Perplexity 与 Loss 的关系

MLflow 中常记录的 perplexity 指标与交叉熵损失直接相关：

$$\text{Perplexity} = e^{H(p, q)} = e^{L_{CE}}$$

其中 $L_{CE}$ 是交叉熵损失（即训练中记录的 loss），$H(p, q)$ 是模型分布 $q$ 与真实分布 $p$ 之间的交叉熵。

Perplexity 可以理解为模型在每个位置上"平均不确定的词数"：perplexity 越低，模型预测越确定。

### 5.2 学习率调度

常见的 cosine scheduler 数学公式：

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t}{T}\pi))$$

其中 $\eta_{\max}$ 为初始学习率，$\eta_{\min}$ 为最低学习率，$t$ 为当前步数，$T$ 为总步数。

### 5.3 LoRA 参数量计算

LoRA 为权重矩阵 $W \in \mathbb{R}^{d \times k}$ 添加低秩分解：

$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

可训练参数量：$2 \times r \times (d + k)$（每个目标模块） vs 全量微调的 $d \times k$。

参数比例：$\frac{2r(d+k)}{dk} \approx \frac{2r}{\min(d,k)}$

## 6. 代码原理/架构原理

### 6.1 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                    MLflow 生态系统                            │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │   Tracking   │  │    Model     │  │     Projects       │ │
│  │   Server     │  │   Registry   │  │                    │ │
│  │              │  │              │  │  MLproject文件      │ │
│  │  - Runs      │  │  - 版本管理   │  │  - 环境管理        │ │
│  │  - Params    │  │  - 阶段管理   │  │  - 入口点定义      │ │
│  │  - Metrics   │  │  - 部署元数据 │  │  - 可复现执行      │ │
│  │  - Artifacts │  │              │  │                    │ │
│  └──────┬───────┘  └──────┬───────┘  └────────────────────┘ │
│         │                 │                                  │
│  ┌──────┴─────────────────┴──────────────────────────────┐  │
│  │              Backend Store (元数据存储)                 │  │
│  │    SQLite / PostgreSQL / SQLAlchemy兼容数据库           │  │
│  └──────────────────────────┬───────────────────────────┘  │
│                             │                               │
│  ┌──────────────────────────┴───────────────────────────┐  │
│  │           Artifact Store (产物存储)                    │  │
│  │    本地文件系统 / S3 / Azure Blob / GCS / HDFS        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  MLflow Client (SDK)                  │  │
│  │    REST API / 本地文件访问                             │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 数据存储分离

MLflow 的核心设计决策是将**元数据**（参数、指标、标签等）与**产物**（模型文件、日志、图片等）分离存储：

- **Backend Store**：存储 Run 元数据，支持 SQLite（本地单用户）和 PostgreSQL（远程多用户）
- **Artifact Store**：存储大文件产物，支持本地文件系统、S3、Azure Blob、GCS 等

这种分离允许：
1. 元数据查询走数据库（快速），产物走对象存储（大容量）
2. 可以独立扩展存储层
3. 支持混合部署（本地数据库 + 云端存储）

### 6.3 模型格式（MLmodel）

MLflow 保存模型时生成标准化的 `MLmodel` 文件：

```yaml
# MLmodel 文件示例
flavors:
  python_function:
    env: conda.yaml
    loader_module: mlflow.transformers
    python_version: 3.10.12
  transformers:
    code: null
    framework: pt
    pipeline_model_type: LlamaForCausalLM
    task: text-generation
    transformers_version: 4.36.0
model_uuid: abc123-def456
run_id: 789run-id
signature:
  inputs: '[{"name": "prompt", "type": "string"}]'
  outputs: '[{"name": "generated_text", "type": "string"}]'
```

`MLmodel` 文件是模型的"身份证"，记录了模型的所有风味（flavor）和元信息，使得 `mlflow.pyfunc.load_model()` 可以统一加载不同框架的模型。

### 6.4 自动日志机制

`mlflow.autolog()` 的工作原理：

1. **Patch 机制**：自动 patch 支持 framework 的训练函数（如 `Trainer.train()`）
2. **Hook 注册**：在 patch 点注册回调函数，捕获训练事件
3. **参数提取**：从 `TrainingArguments` 自动提取超参数
4. **指标劫持**：拦截 `Trainer` 的 logging callback，自动将指标转发到 MLflow
5. **模型捕获**：训练结束后自动调用 `log_model()`

## 7. 常见注意事项和最佳实践

### 7.1 追踪服务器配置

```python
# 1. 生产环境推荐配置
# Backend Store: PostgreSQL（高可用、并发安全）
# Artifact Store: S3/MinIO（大容量、低成本）
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql://user:pass@db:5432/mlflow \
    --default-artifact-root s3://mlflow/artifacts \
    --workers 4 \
    --gunicorn-opts "--timeout 120"

# 2. 开发环境快速启动
# 使用本地SQLite和文件系统
mlflow server --host 0.0.0.0 --port 5000

# 3. 代码中设置
mlflow.set_tracking_uri("http://tracking-server:5000")
```

### 7.2 性能优化

```python
# 1. 控制log_metric频率
# MLflow每个metric默认限制10000个数据点
# 过多的数据点会导致UI变慢
LOG_INTERVAL = 50
for step in range(100000):
    if step % LOG_INTERVAL == 0:
        mlflow.log_metric("train_loss", loss, step=step)

# 2. 批量记录
mlflow.log_metrics({"loss": loss, "lr": lr, "grad_norm": gn}, step=step)

# 3. 大文件使用Artifact而非metric
# 不要用metric记录大型数组，用Artifact
mlflow.log_dict(attention_weights, "attention_weights.json")

# 4. 异步日志（MLflow 2.x+）
# MLflow默认同步记录，可通过以下方式优化：
# - 减少log_metric调用频率
# - 使用后台线程批量flush
```

### 7.3 模型管理最佳实践

```python
# 1. 始终指定模型签名
from mlflow.models import infer_signature

signature = infer_signature(
    model_input=pd.DataFrame({"prompt": ["Hello"]}),
    model_output=pd.DataFrame({"generated_text": ["Hi there!"]}),
)
mlflow.transformers.log_model(..., signature=signature)

# 2. 提供输入示例
input_example = pd.DataFrame({"prompt": ["什么是机器学习？"]})
mlflow.transformers.log_model(..., input_example=input_example)

# 3. 显式指定pip_requirements
mlflow.transformers.log_model(
    ...,
    pip_requirements=[
        "transformers==4.36.0",
        "torch==2.1.0",
        "accelerate==0.24.0",
        "sentencepiece==0.1.99",
    ],
)

# 4. 使用Model Registry管理生产模型
# 不要直接用run_id加载生产模型，使用阶段管理
model_uri = "models:/my-model/Production"  # 好的做法
# 而不是
model_uri = "runs:/abc123/model"           # 避免这样做

# 5. 添加描述和标签
client.update_model_version(
    name="my-model", version=1,
    description="基于Mistral-7B的LoRA微调模型，Alpaca数据集训练"
)
client.set_model_version_tag(
    name="my-model", version=1,
    key="base_model", value="mistral-7b"
)
```

### 7.4 常见陷阱

1. **SQLite 并发问题**：SQLite 不支持并发写入，多进程训练时使用 PostgreSQL
2. **Artifact 路径问题**：`default-artifact-root` 必须在服务器启动时指定，运行时无法更改
3. **模型加载依赖缺失**：`load_model` 需要所有训练时的依赖，建议使用 `conda.yaml` 或 Docker
4. **忘记设置 experiment**：不设置时所有运行进入 "Default" 实验，难以管理
5. **metric 覆盖**：同一 run + 同一 metric + 同一 step，后值覆盖前值
6. **autolog 与手动log冲突**：同时使用 autolog 和手动 log 可能产生重复指标
7. **模型版本不可删除**：一旦有下游引用（如 Staging/Production），无法直接删除版本

### 7.5 安全与权限

```python
# 1. 启用认证（服务器端）
mlflow server --app-name basic-auth

# 2. 使用环境变量管理凭证
import os
os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow.company.com"
os.environ["MLFLOW_TRACKING_USERNAME"] = "user"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

# 3. S3 Artifact存储权限
os.environ["AWS_ACCESS_KEY_ID"] = "key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "secret"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# 4. 敏感信息不要记录
# 不要记录API Key、用户隐私数据等
mlflow.log_param("dataset", "alpaca")  # 好
# mlflow.log_param("api_key", "sk-xxx")  # 坏！
```

### 7.6 MLflow vs W&B 选择指南

| 特性 | MLflow | W&B |
|------|--------|-----|
| 开源 | 完全开源 | 核心功能开源，高级功能收费 |
| 部署方式 | 自托管/本地 | 云端优先（支持私有部署） |
| 数据隐私 | 完全可控 | 数据存储在W&B服务器 |
| 超参数搜索 | 不内置（需配合Optuna等） | 内置Sweep功能 |
| 可视化 | 基础UI | 丰富的交互式仪表盘 |
| 模型管理 | Model Registry（内置） | Artifact（可选） |
| 团队协作 | 通过共享服务器 | 云端原生支持 |
| 学习曲线 | 较平缓 | 较平缓 |
| 生态集成 | Spark, Databricks, HuggingFace | HuggingFace, PyTorch, TensorFlow |

**选择建议**：
- 数据隐私要求高、需要自托管 → **MLflow**
- 需要丰富的可视化、团队协作 → **W&B**
- 已在 Databricks 生态中 → **MLflow**
- 需要内置超参数搜索 → **W&B**
