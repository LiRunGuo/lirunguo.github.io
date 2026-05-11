---
title: "Datasets 数据集库"
excerpt: "load_dataset、map批量处理、流式加载(streaming)、Arrow零拷贝架构"
collection: llm-libs
permalink: /llm-libs/10b-datasets
category: training
toc: true
---


## 1. 简介与在LLM开发中的作用

### 1.1 什么是 Datasets

Datasets 是 HuggingFace 开发的一个用于高效加载、处理和共享数据集的 Python 库。它基于 Apache Arrow 格式构建，提供零拷贝内存映射、惰性计算和流式处理能力，能够高效处理远超内存容量的超大数据集。

### 1.2 在LLM开发中的核心作用

- **统一的数据加载接口**：一行代码即可从 HuggingFace Hub、本地文件或内存加载数据，支持 20+ 种格式
- **高效的大规模数据处理**：Arrow 零拷贝 + 内存映射使得 TB 级数据集也能快速处理
- **与 Transformers 无缝集成**：数据集可直接设置 PyTorch/TensorFlow 格式，喂入模型
- **流式加载**：`streaming=True` 可逐条加载网络数据集，无需完整下载
- **多进程并行处理**：`map()` 支持 `num_proc` 参数，充分利用多核 CPU
- **丰富的数据操作**：`map`/`filter`/`shuffle`/`select`/`train_test_split` 等链式操作

---

## 2. 安装方式

```bash
# 基础安装
pip install datasets

# 安装音频支持
pip install datasets[audio]

# 安装图像支持
pip install datasets[vision]

# 完整安装
pip install datasets[audio,vision]

# 从源码安装
pip install git+https://github.com/huggingface/datasets
```

验证安装：

```python
import datasets
print(datasets.__version__)
```

---

## 3. 核心类与函数详细说明

### 3.1 load_dataset() — 加载数据集

`load_dataset()` 是最核心的函数，用于从各种来源加载数据集。

#### 从 HuggingFace Hub 加载

```python
from datasets import load_dataset

# 加载Hub上的数据集
dataset = load_dataset("imdb")                       # 返回 DatasetDict
dataset = load_dataset("imdb", split="train")        # 返回 Dataset
dataset = load_dataset("glue", "mrpc", split="train") # 加载子集

# 加载特定版本
dataset = load_dataset("imdb", revision="main")

# 加载特定文件
dataset = load_dataset("username/my_dataset", data_files="train.csv")
```

**主要参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | str | 数据集名称或路径 |
| `name` | str | 子集名称（如glue的mrpc/sst2等） |
| `data_dir` | str | 数据目录 |
| `data_files` | str/list/dict | 数据文件路径，支持通配符 |
| `split` | str | 指定分割（"train"/"test"/"validation"） |
| `revision` | str | 数据集版本/分支/commit |
| `streaming` | bool | 是否流式加载 |
| `num_proc` | int | 下载/处理时的进程数 |
| `trust_remote_code` | bool | 是否信任远程代码 |

#### 从本地文件加载

```python
# CSV 文件
dataset = load_dataset("csv", data_files="my_data.csv")
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

# JSON 文件
dataset = load_dataset("json", data_files="data.json")
dataset = load_dataset("json", data_files="data.jsonl", field="records")  # JSONL格式

# Parquet 文件
dataset = load_dataset("parquet", data_files="data.parquet")

# 文本文件（每行一条）
dataset = load_dataset("text", data_files="sentences.txt")

# 从Python对象直接创建
from datasets import Dataset
dataset = Dataset.from_dict({
    "text": ["Hello", "World"],
    "label": [0, 1]
})
dataset = Dataset.from_pandas(df)      # 从 pandas DataFrame
dataset = Dataset.from_list(records)   # 从字典列表
```

#### 从 Hub 加载时的缓存机制

HuggingFace Datasets 默认将下载的数据集缓存到 `~/.cache/huggingface/datasets/`。缓存使用哈希键（基于数据集名称、版本、配置等生成），相同配置不会重复下载。

```python
# 自定义缓存目录
dataset = load_dataset("imdb", cache_dir="/data/cache")

# 设置全局缓存目录
import os
os.environ["HF_DATASETS_CACHE"] = "/data/cache"
```

### 3.2 Dataset 与 DatasetDict

#### Dataset

`Dataset` 是单分割数据集的核心类，底层基于 Arrow 表格实现：

```python
from datasets import Dataset

# 基本属性
dataset.column_names     # 列名列表：['text', 'label']
dataset.features         # 特征类型：{'text': Value('string'), 'label': Value('int64')}
dataset.num_rows         # 行数
dataset.num_columns      # 列数
dataset.shape            # (num_rows, num_columns)

# 索引访问
dataset[0]               # 返回第一行的字典：{'text': '...', 'label': 0}
dataset["text"]          # 返回整个text列的列表

# 切片访问
dataset[:5]              # 前五行
dataset.select([0, 2, 4]) # 选择特定行

# 添加/删除列
dataset = dataset.add_column("new_col", [1, 2, 3, ...])
dataset = dataset.remove_columns(["unwanted_col"])
dataset = dataset.rename_column("old_name", "new_name")
```

#### DatasetDict

`DatasetDict` 是多个 `Dataset` 的字典容器，通常对应 train/validation/test 分割：

```python
from datasets import load_dataset

dataset_dict = load_dataset("imdb")
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test: Dataset({features: ['text', 'label'], num_rows: 25000})
# })

# 访问分割
train_ds = dataset_dict["train"]
test_ds = dataset_dict["test"]

# 对所有分割统一操作
dataset_dict = dataset_dict.map(tokenize_function, batched=True)
dataset_dict = dataset_dict.filter(lambda x: len(x["text"]) > 10)

# 常用方法
dataset_dict.keys()          # dict_keys(['train', 'test'])
dataset_dict.values()        # 所有Dataset
dataset_dict.items()         # 键值对
```

### 3.3 map() — 批量处理函数

`map()` 是数据处理的核心方法，对数据集的每条（或每批）记录应用一个函数。

```python
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

# 基本用法
dataset = dataset.map(tokenize_function)

# 批量处理（推荐，显著加速）
dataset = dataset.map(tokenize_function, batched=True)

# 多进程并行
dataset = dataset.map(tokenize_function, batched=True, num_proc=8)

# 移除原始列（节省内存）
dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 只保留指定列
dataset = dataset.map(tokenize_function, batched=True, keep_in_memory=True)
```

**关键参数详解**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `function` | Callable | 处理函数，接受一个字典（或字典批次）返回处理结果 |
| `batched` | bool | 是否批量处理。True时函数接收一批数据，每列是列表 |
| `batch_size` | int | 每批的样本数，默认1000 |
| `num_proc` | int | 并行进程数。建议设为CPU核心数，但注意内存消耗 |
| `remove_columns` | list | 处理后移除的列 |
| `fn_kwargs` | dict | 传递给function的额外参数 |
| `desc` | str | 进度条描述 |
| `load_from_cache_file` | bool | 是否使用缓存（默认True） |
| `cache_file_name` | str | 自定义缓存文件名 |

**batched=True 的行为**：

```python
# batched=False 时，函数每次接收一条记录
def process_single(example):
    # example = {"text": "hello", "label": 0}
    example["text_len"] = len(example["text"])
    return example

# batched=True 时，函数每次接收一批记录
def process_batch(examples):
    # examples = {"text": ["hello", "world"], "label": [0, 1]}
    examples["text_len"] = [len(t) for t in examples["text"]]
    return examples

dataset = dataset.map(process_batch, batched=True)
```

**传递额外参数**：

```python
def tokenize_with_max_length(example, max_len=512):
    return tokenizer(example["text"], truncation=True, max_length=max_len)

dataset = dataset.map(
    tokenize_with_max_length,
    fn_kwargs={"max_len": 256},
    batched=True,
)
```

### 3.4 filter() — 过滤数据

```python
# 过滤短文本
dataset = dataset.filter(lambda x: len(x["text"]) > 100)

# 过滤特定标签
dataset = dataset.filter(lambda x: x["label"] == 1)

# 使用多进程
dataset = dataset.filter(lambda x: len(x["text"]) > 100, num_proc=4)

# 过滤函数可接受批次
def filter_long_text(examples):
    return [len(t) > 100 for t in examples["text"]]

dataset = dataset.filter(filter_long_text, batched=True)
```

### 3.5 select() / shuffle() / train_test_split()

#### select() — 选择特定行

```python
# 选择指定索引的行
subset = dataset.select([0, 10, 20, 30])

# 选择前1000行
subset = dataset.select(range(1000))

# 随机选择（先shuffle再select）
subset = dataset.shuffle(seed=42).select(range(1000))
```

#### shuffle() — 随机打乱

```python
# 随机打乱
dataset = dataset.shuffle(seed=42)

# 指定缓冲区大小（大数据集建议增大buffer_size）
dataset = dataset.shuffle(seed=42, buffer_size=10_000)
```

**注意**：`shuffle()` 默认使用 1000 大小的缓冲区进行近似随机。对于完美随机，需要 `buffer_size >= len(dataset)`，但这会消耗大量内存。

#### train_test_split() — 划分训练/测试集

```python
# 按比例划分
split = dataset.train_test_split(test_size=0.2, seed=42)
# DatasetDict({
#     train: Dataset({...}, num_rows: 800)
#     test: Dataset({...}, num_rows: 200)
# })

# 按绝对数量划分
split = dataset.train_test_split(test_size=1000, seed=42)

# 创建验证集（两次划分）
split1 = dataset.train_test_split(test_size=0.2, seed=42)
split2 = split1["train"].train_test_split(test_size=0.125, seed=42)  # 0.125 * 0.8 = 0.1
# split2["train"] → 72% 训练集
# split2["test"] → 8% 验证集
# split1["test"] → 20% 测试集
```

**参数**：
- `test_size`：测试集大小（比例或绝对数量）
- `train_size`：训练集大小（与test_size二选一）
- `shuffle`：是否在划分前打乱（默认True）
- `seed`：随机种子
- `stratify_by_column`：按某列的类别比例分层划分

### 3.6 流式加载（streaming=True）

对于大数据集，`streaming=True` 可以逐条加载，无需完整下载到磁盘：

```python
# 流式加载（返回 IterableDataset）
dataset = load_dataset("imdb", streaming=True)

# 逐条迭代
for example in dataset["train"]:
    print(example["text"])
    break  # 只看第一条

# 流式数据集的操作
dataset = dataset.shuffle(buffer_size=1000, seed=42)
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.filter(lambda x: len(x["text"]) > 100)

# 只取前N条
dataset = dataset.take(100)  # 取前100条
dataset = dataset.skip(100)  # 跳过前100条
```

**IterableDataset vs Dataset 的区别**：

| 特性 | Dataset | IterableDataset |
|------|---------|-----------------|
| 索引访问 | 支持 `dataset[0]` | 不支持 |
| 随机访问 | 支持 | 不支持 |
| len() | 支持 | 不支持 |
| 多次迭代 | 支持 | 可重复迭代 |
| 内存占用 | 需下载完整数据 | 极低 |
| shuffle | 完美随机 | 近似随机（buffer） |
| 适用场景 | 中小数据集 | 超大数据集/在线流 |

**将IterableDataset转为Dataset**（仅小数据适用）：

```python
# 获取少量数据用于调试
small_dataset = list(dataset["train"].take(100))
from datasets import Dataset
debug_dataset = Dataset.from_list(small_dataset)
```

### 3.7 保存与导出

#### save_to_disk()

```python
# 保存为 Arrow 格式
dataset.save_to_disk("my_dataset_path")
dataset_dict.save_to_disk("my_dataset_dict_path")

# 加载
from datasets import load_from_disk
dataset = load_from_disk("my_dataset_path")
dataset_dict = load_from_disk("my_dataset_dict_path")
```

#### to_csv() / to_parquet() / to_json()

```python
# 导出为 CSV
dataset.to_csv("output.csv")

# 导出为 Parquet
dataset.to_parquet("output.parquet")

# 导出为 JSON
dataset.to_json("output.jsonl")          # JSONL 格式（每行一条）
dataset.to_json("output.json", orient="records")  # JSON 数组格式
```

#### 推送到 HuggingFace Hub

```python
# 推送到Hub
dataset.push_to_hub("username/my_dataset")

# 私有仓库
dataset.push_to_hub("username/my_dataset", private=True)

# 推送 DatasetDict
dataset_dict.push_to_hub("username/my_dataset")
```

### 3.8 与 Transformers 集成

#### set_format() — 设置输出格式

```python
# 设置为 PyTorch 张量格式
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 现在索引访问返回张量
dataset[0]["input_ids"]  # torch.Tensor

# 也可以设置为 numpy
dataset.set_format(type="numpy", columns=["input_ids", "attention_mask", "labels"])

# 也可以设置为 pandas
dataset.set_format(type="pandas")

# 重置格式
dataset.reset_format()
```

#### 与 DataLoader 结合

```python
from torch.utils.data import DataLoader

# 方式1：set_format + DataLoader
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    input_ids = batch["input_ids"]       # shape: (32, seq_len)
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    # ... 模型前向传播 ...

# 方式2：使用 with_format（不修改原数据集）
torch_dataset = dataset.with_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
```

#### 完整的 LLM 微调数据处理流程

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 1. 加载数据集
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# 2. 定义tokenize函数
def tokenize_function(examples):
    # 将 instruction + input + output 拼接
    prompts = [
        f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        if inp else f"### Instruction:\n{inst}\n\n### Response:\n{out}"
        for inst, inp, out in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        )
    ]
    return tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length",
    )

# 3. 批量tokenize
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=8,
    remove_columns=dataset.column_names,
)

# 4. 添加labels（对于语言模型，labels = input_ids）
tokenized_dataset = tokenized_dataset.map(
    lambda x: {"labels": x["input_ids"]},
    batched=True,
)

# 5. 划分训练/验证集
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

# 6. 设置格式
split.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 7. 创建DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(split["train"], batch_size=8, shuffle=True)
val_loader = DataLoader(split["test"], batch_size=8)
```

### 3.9 并发下载与缓存机制

#### 多线程下载

```python
# 设置下载并发数
dataset = load_dataset("big_dataset", num_proc=8)
```

#### 缓存机制详解

Datasets 的缓存系统基于哈希键实现：

```
~/.cache/huggingface/datasets/
├── imdb/
│   └── <hash>/
│       ├── imdb-train.arrow      # Arrow 格式数据
│       ├── imdb-test.arrow
│       └── dataset_info.json
└── gluedata/
    └── ...
```

缓存键由以下因素决定：
- 数据集名称和版本
- 加载配置（name、data_files等）
- `map()` 的函数哈希和参数

**禁用缓存**（调试时有用）：

```python
dataset = dataset.map(tokenize_function, load_from_cache_file=False)
```

**清理缓存**：

```python
from datasets import disable_caching, enable_caching

# 全局禁用缓存
disable_caching()

# 重新启用
enable_caching()
```

---

## 4. 在LLM开发中的典型使用场景

### 4.1 场景一：预处理指令微调数据集

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("tatsu-lab/alpaca", split="train")

def format_and_tokenize(examples):
    # 格式化为对话模板
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        if inp:
            text = f"Below is an instruction that describes a task, paired with an input.\n\n### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        else:
            text = f"Below is an instruction that describes a task.\n\n### Instruction:\n{inst}\n\n### Response:\n{out}"
        texts.append(text)

    tokenized = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = [
        [-100 if t == tokenizer.pad_token_id else t for t in ids]
        for ids in tokenized["input_ids"]
    ]
    return tokenized

processed = dataset.map(
    format_and_tokenize,
    batched=True,
    num_proc=8,
    remove_columns=dataset.column_names,
)
processed.save_to_disk("./alpaca_tokenized")
```

### 4.2 场景二：流式加载 TB 级预训练数据

```python
from datasets import load_dataset

# 流式加载 C4 数据集（约300GB）
c4 = load_dataset("allenai/c4", "en", streaming=True, split="train")

# 定义预处理函数
def preprocess(examples):
    # 只保留长度适中的文本
    return {"text": [t for t in examples["text"] if 100 < len(t) < 10000]}

c4 = c4.filter(lambda x: 100 < len(x["text"]) < 10000)
c4 = c4.shuffle(buffer_size=10_000, seed=42)

# 逐批迭代，不需要将整个数据集加载到内存
for i, example in enumerate(c4):
    if i >= 100000:  # 只处理前10万条
        break
    text = example["text"]
    # ... 处理文本 ...
```

### 4.3 场景三：构建自定义数据集并推送Hub

```python
from datasets import Dataset, DatasetDict
import json

# 从本地文件加载
with open("train.jsonl") as f:
    train_data = [json.loads(line) for line in f]
with open("test.jsonl") as f:
    test_data = [json.loads(line) for line in f]

train_ds = Dataset.from_list(train_data)
test_ds = Dataset.from_list(test_data)

# 创建 DatasetDict
ds_dict = DatasetDict({"train": train_ds, "test": test_ds})

# 推送到 Hub
ds_dict.push_to_hub("my-username/my-llm-dataset", private=True)
```

### 4.4 场景四：数据集质量过滤与清洗

```python
from datasets import load_dataset
import re

dataset = load_dataset("openwebtext", split="train", streaming=True)

def is_quality_text(text):
    # 长度过滤
    if len(text) < 200 or len(text) > 100000:
        return False
    # 去重检查（简单hash）
    # 重复行检测
    lines = text.split("\n")
    unique_ratio = len(set(lines)) / max(len(lines), 1)
    if unique_ratio < 0.5:
        return False
    # 语言检测（简单启发式）
    if len(re.findall(r'[\u4e00-\u9fff]', text)) / max(len(text), 1) > 0.3:
        return True  # 中文内容
    return True

dataset = dataset.filter(lambda x: is_quality_text(x["text"]))
```

---

## 5. 数学原理

### 5.1 Arrow 列式存储与零拷贝

Apache Arrow 是一种列式内存格式，Datasets 基于它实现高效的数据访问。

**行式 vs 列式存储**：

```
行式存储（如CSV）:
Row1: [text="hello", label=0, length=5]
Row2: [text="world", label=1, length=5]
Row3: [text="foo",   label=0, length=3]

列式存储（Arrow）:
text列:   ["hello", "world", "foo"]
label列:  [0, 1, 0]
length列: [5, 5, 3]
```

列式存储的优势：
- **向量化操作**：对单列操作时数据在内存中连续，CPU缓存命中率高
- **类型统一**：同一列数据类型相同，可使用 SIMD 指令加速
- **压缩率更高**：同类型数据连续存储，压缩效果更好
- **零拷贝读取**：Arrow 的内存布局是标准化的，不同进程/语言可以直接共享内存而不需要序列化/反序列化

**零拷贝内存映射**：

Datasets 使用内存映射（mmap）技术将 Arrow 文件映射到虚拟内存空间：

```
磁盘文件 (.arrow) ←mmap→ 虚拟内存 ←按需加载→ 物理内存
```

操作系统按需将文件内容加载到物理内存，不需要的页面可被换出。这意味着：
- 数据集大小不受物理内存限制
- 多进程共享同一份内存映射，不重复加载
- 只有实际访问的数据才会加载到内存

### 5.2 惰性计算与缓存

`map()` 操作采用惰性计算策略：

1. 首次调用 `map()` 时，计算结果会被缓存到磁盘（Arrow 文件）
2. 相同的 `map()` 调用（相同函数+参数）会直接从缓存读取
3. 缓存键由函数源代码的哈希 + 参数的哈希生成

这使得数据处理管道可以安全地重复运行而无需重复计算。

### 5.3 流式处理的缓冲区随机

`IterableDataset.shuffle()` 使用有限缓冲区实现近似随机洗牌：

```
数据流 → 缓冲区(buffer_size=N) → 随机采样 → 输出
```

算法步骤：
1. 从数据流中填充缓冲区至 `buffer_size`
2. 从缓冲区中随机选择一个元素输出
3. 从数据流中读取下一个元素放入缓冲区
4. 重复2-3直到数据流耗尽后，随机输出缓冲区剩余元素

当 `buffer_size` 等于数据集大小时，等价于完美随机洗牌。实际使用中，`buffer_size` 远小于数据集大小，因此是近似随机。

---

## 6. 架构原理

### 6.1 整体架构

```
用户接口层
    │
    ├── load_dataset() / Dataset / DatasetDict / IterableDataset
    │
数据处理层
    │
    ├── map() / filter() / select() / shuffle() / sort()
    │
Arrow 存储层
    │
    ├── Memory-mapped Arrow Tables
    ├── 缓存管理
    └── 流式 Arrow IPC Reader
    │
下载层
    │
    ├── HuggingFace Hub API
    ├── 并发下载
    └── 本地文件读取
```

### 6.2 Dataset 的内部结构

```python
# Dataset 的核心属性
dataset._data         # pyarrow.Table，底层Arrow表格
dataset._indices      # 可选的索引映射（用于select/sort等操作的惰性实现）
dataset._format_type  # 输出格式：None/"torch"/"numpy"/"pandas"
dataset._format_columns  # 格式化的列
```

**索引映射的惰性实现**：`select()`、`sort()` 等操作不修改底层的 Arrow 表，而是创建一个索引映射。只有当实际访问数据时，才通过索引映射从底层 Arrow 表中提取对应行。这种惰性设计避免了数据的复制。

### 6.3 map() 的执行流程

```
map(function, batched=True, num_proc=4)
    │
    ├── 计算缓存键（函数哈希 + 参数哈希）
    ├── 检查缓存是否存在
    │   └── 存在 → 直接加载缓存结果
    │   └── 不存在 ↓
    ├── 将数据集分块（batch_size块）
    ├── 多进程并行处理（num_proc个worker）
    │   ├── Worker 1: 处理 chunk[0:batch_size]
    │   ├── Worker 2: 处理 chunk[batch_size:2*batch_size]
    │   ├── Worker 3: ...
    │   └── Worker 4: ...
    ├── 收集结果并写入Arrow文件
    └── 返回新的Dataset（指向新的Arrow文件）
```

### 6.4 流式处理的架构

```
IterableDataset
    │
    ├── 自定义迭代器（来自Hub的HTTP流）
    │   ├── HTTP Range请求分块下载
    │   └── Arrow IPC Stream 解码器
    │
    ├── map()/filter() → 包装为 LazyIterable
    │   └── 在迭代时逐条/逐批应用函数
    │
    ├── shuffle() → ShuffledIterable
    │   └── 缓冲区随机采样
    │
    └── take(n)/skip(n) → 截断/跳过迭代器
```

流式加载的核心是 Arrow IPC Stream 格式，它支持逐块解码而不需要读取整个文件。Datasets 通过 HTTP Range 请求按需下载 Arrow 数据块。

---

## 7. 常见注意事项与最佳实践

### 7.1 注意事项

1. **map() 的缓存陷阱**：如果处理函数依赖于外部可变状态（如全局变量、模型权重），缓存可能导致使用过时的结果。使用 `load_from_cache_file=False` 强制重新计算

2. **num_proc 与内存消耗**：每个 worker 进程都会复制一份数据集索引，`num_proc=8` 意味着 8 倍的索引内存。对于超大数据集，需要权衡并行度和内存

3. **shuffle() 的缓冲区限制**：`Dataset.shuffle()` 默认 `buffer_size=1000`，对小数据集不够随机；`IterableDataset.shuffle()` 更是近似随机

4. **set_format() 是原地修改**：`dataset.set_format()` 会修改数据集本身。如果需要保留原格式，使用 `dataset.with_format()`

5. **streaming=True 不支持索引**：`IterableDataset` 不支持 `dataset[i]`、`len(dataset)` 或 `select()`。调试时可用 `take(n)` 获取少量数据

6. **并发写入问题**：多进程环境下不要对同一个 Dataset 写入，Arrow 文件是不可变的，写入会产生新文件

7. **大列的内存问题**：`dataset["text"]` 会将整列加载到内存。对于大数据集，应该使用 `map()` 逐批处理

### 7.2 最佳实践

1. **优先使用 batched=True**：批量处理比逐条处理快 10-100 倍，特别是配合 tokenizer 的批量模式

2. **使用 num_proc 加速 map()**：
```python
# 推荐设为CPU核心数
import multiprocessing
dataset = dataset.map(process_fn, batched=True, num_proc=multiprocessing.cpu_count())
```

3. **使用 with_format 而非 set_format**：避免意外修改原数据集
```python
torch_ds = dataset.with_format("torch", columns=["input_ids", "labels"])
# 原 dataset 不受影响
```

4. **大数据集使用 streaming=True**：避免磁盘空间不足和长时间等待
```python
dataset = load_dataset("allenai/c4", "en", streaming=True)
```

5. **及时 remove_columns**：tokenize 后移除原始文本列，显著减少内存占用
```python
dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
```

6. **使用 desc 参数标注进度条**：多个 map 操作时便于区分
```python
dataset.map(tokenize_fn, batched=True, desc="Tokenizing")
dataset.map(filter_fn, desc="Filtering")
```

7. **调试时禁用缓存**：
```python
from datasets import disable_caching
disable_caching()  # 确保每次重新计算
```

8. **数据集版本管理**：使用 `revision` 参数指定数据集版本，确保可复现性
```python
dataset = load_dataset("imdb", revision="1.0.0")
```

9. **使用 concatenate_datasets 合并数据集**：
```python
from datasets import concatenate_datasets
merged = concatenate_datasets([dataset1, dataset2, dataset3])
```

10. **利用 interleave_datasets 交替混合多源数据**：
```python
from datasets import interleave_datasets
# 交替混合不同来源的数据，适合多任务训练
mixed = interleave_datasets([dataset_a, dataset_b], probabilities=[0.7, 0.3])
```
