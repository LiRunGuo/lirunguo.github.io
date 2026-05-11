---
title: "llama-cpp-python Python绑定"
excerpt: "Llama类、create_completion/chat、GBNF语法约束、嵌入提取、OpenAI服务器"
collection: llm-libs
permalink: /llm-libs/15-llama-cpp-python
category: inference
toc: true
---


## 1. 库的简介和在LLM开发中的作用

### 1.1 什么是 llama-cpp-python

llama-cpp-python 是 llama.cpp 的官方 Python 绑定库，提供了简洁的 Pythonic API 来调用 llama.cpp 的全部功能。它在底层通过 C 接口调用 llama.cpp 的共享库（`.so`/`.dylib`/`.dll`），在 Python 层面封装了高级推理接口，使开发者无需编写 C 代码即可享受 llama.cpp 的高性能推理能力。

### 1.2 在LLM开发中的核心作用

| 作用 | 说明 |
|------|------|
| **快速原型开发** | 用 Python 几行代码即可加载模型并生成文本，适合研究和实验 |
| **应用集成** | 将 LLM 推理能力嵌入到 Python 应用中（Web服务、数据处理流水线等） |
| **OpenAI 替代** | 内置 OpenAI 兼容服务器，可直接替换 OpenAI API |
| **语法约束生成** | 通过 GBNF 语法约束输出格式，实现结构化生成 |
| **嵌入提取** | 从模型中提取文本嵌入向量，用于 RAG、语义搜索等 |

---

## 2. 安装方式

### 2.1 基础安装（仅 CPU）

```bash
pip install llama-cpp-python
```

### 2.2 带 CUDA 支持安装

```bash
# 方法1：预编译 wheel（推荐）
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 方法2：从源码编译
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-binary :all:

# 方法3：使用 conda
conda install -c conda-forge llama-cpp-python
```

### 2.3 带 Metal 支持安装（macOS）

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-binary :all:
```

### 2.4 带 ROCm 支持安装（AMD GPU）

```bash
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --no-binary :all:
```

### 2.5 带 VULKAN 支持安装

```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-binary :all:
```

### 2.6 验证安装

```python
from llama_cpp import Llama

# 检查是否支持 GPU
llama = Llama(model_path="model.gguf", n_gpu_layers=1)
print(f"GPU layers loaded: {llama.n_gpu_layers}")
```

---

## 3. 核心类/函数/工具的详细说明

### 3.1 Llama 类

`Llama` 是 llama-cpp-python 的核心类，封装了模型加载和推理的全部功能。

#### 3.1.1 构造函数参数

```python
from llama_cpp import Llama

llm = Llama(
    model_path="model.gguf",       # 模型文件路径（必填）
    n_ctx=512,                     # 上下文窗口大小
    n_batch=512,                   # 批次大小（prompt处理）
    n_ubatch=128,                  # 物理批次大小
    n_gpu_layers=0,                # 卸载到GPU的层数，-1为全部
    split_mode=LLAMA_SPLIT_MODE_LAYER,  # 多GPU分割模式
    main_gpu=0,                    # 主GPU编号
    tensor_split=None,             # 多GPU张量分割比例
    rope_freq_base=0.0,            # RoPE频率基数（0=使用模型默认）
    rope_freq_scale=1.0,           # RoPE频率缩放因子
    seed=LLAMA_DEFAULT_SEED,       # 随机种子
    f16_kv=True,                   # KV Cache使用FP16
    logits_all=False,              # 是否返回所有token的logits
    vocab_only=False,              # 仅加载词表（不加载权重）
    use_mmap=True,                 # 使用内存映射
    use_mlock=False,               # 锁定内存防止换页
    embedding=False,               # 是否启用嵌入模式
    numa=False,                    # NUMA优化
    chat_format=None,              # 聊天格式（chatml、llama-2等）
    chat_handler=None,             # 自定义聊天处理器
    lora_base=None,                # LoRA基础模型路径
    lora_path=None,                # LoRA适配器路径
    lora_scale=1.0,               # LoRA缩放因子
    verbose=True,                  # 打印详细日志
)
```

**关键参数详解**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | str | 必填 | GGUF 模型文件路径 |
| `n_ctx` | int | 512 | 上下文窗口大小。增大此值会线性增加 KV Cache 内存占用 |
| `n_batch` | int | 512 | 提示词处理时的批次大小，增大可加速长 prompt 处理 |
| `n_gpu_layers` | int | 0 | GPU 卸载层数。设为 -1 卸载所有层。Apple Silicon 设为 1 即可 |
| `use_mmap` | bool | True | 使用 mmap 加载模型，加快启动速度 |
| `use_mlock` | bool | False | 锁定模型内存，防止被操作系统换出到磁盘 |
| `embedding` | bool | False | 设为 True 时，模型专注于嵌入提取 |
| `rope_freq_scale` | float | 1.0 | RoPE 频率缩放，大于1可扩展有效上下文长度（如设为2.0可近似翻倍上下文） |

#### 3.1.2 RoPE 上下文扩展

```python
# 方法1：RoPE 频率缩放（简单但有质量损失）
llm = Llama(
    model_path="model.gguf",
    n_ctx=8192,              # 目标上下文长度
    rope_freq_scale=0.5,     # 缩放因子 = 原始长度 / 目标长度
)

# 方法2：YaRN 扩展（更高质量）
llm = Llama(
    model_path="model.gguf",
    n_ctx=8192,
    rope_freq_base=10000,    # 原始 RoPE 基数
    rope_freq_scale=0.5,
)
```

### 3.2 create_completion() — 文本补全

```python
result = llm.create_completion(
    prompt="人工智能的未来是",          # 输入提示词
    max_tokens=128,                    # 最大生成 token 数
    temperature=0.7,                   # 采样温度（0=贪心）
    top_p=0.9,                         # Top-p 核采样
    top_k=40,                          # Top-k 采样
    min_p=0.05,                        # Min-p 采样
    typical_p=1.0,                     # 典型采样概率
    stop=["。", "\n"],                 # 停止词列表
    stream=False,                      # 是否流式输出
    seed=None,                         # 随机种子
    repeat_penalty=1.1,               # 重复惩罚
    frequency_penalty=0.0,            # 频率惩罚
    presence_penalty=0.0,             # 存在惩罚
    tfs_z=1.0,                        # 尾部自由采样参数
    mirostat_mode=0,                  # Mirostat 采样模式（0=禁用，1=Mirostat，2=Mirostat 2.0）
    mirostat_tau=5.0,                 # Mirostat 目标熵
    mirostat_eta=0.1,                 # Mirostat 学习率
    grammar=None,                      # LlamaGrammar 语法约束
    logprobs=None,                     # 返回 top-N logprobs
)
```

**返回值结构**：
```python
{
    "id": "cmpl-xxx",
    "object": "text_completion",
    "created": 1699999999,
    "model": "model.gguf",
    "choices": [{
        "text": "充满无限可能的。...",
        "index": 0,
        "logprobs": None,
        "finish_reason": "stop"  # "stop" | "length"
    }],
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 20,
        "total_tokens": 28
    }
}
```

**基础用法示例**：

```python
from llama_cpp import Llama

llm = Llama(model_path="llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048)

# 简单补全
result = llm.create_completion(
    prompt="量子计算的基本原理是",
    max_tokens=200,
    temperature=0.7,
    stop=["。"]
)
print(result["choices"][0]["text"])
```

### 3.3 create_chat_completion() — 对话补全

```python
result = llm.create_chat_completion(
    messages=[                         # 消息列表
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "解释什么是机器学习"},
        {"role": "assistant", "content": "机器学习是..."},
        {"role": "user", "content": "它和深度学习有什么区别？"},
    ],
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    stream=False,
    stop=None,
    response_format=None,              # JSON模式: {"type": "json_object"}
    tools=None,                        # 函数调用工具定义
    tool_choice=None,                  # "auto" | "none" | {"type": "function", "function": {"name": "..."}}
    grammar=None,
    logprobs=None,
)
```

**消息格式**：

| 角色 | 说明 |
|------|------|
| `system` | 系统提示词，定义模型行为和角色 |
| `user` | 用户输入 |
| `assistant` | 模型回复（可提供历史对话） |
| `tool` | 工具调用结果（配合 function calling 使用） |

**返回值结构**：
```python
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "created": 1699999999,
    "model": "model.gguf",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "机器学习是一种...",
            "tool_calls": None
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 30,
        "completion_tokens": 50,
        "total_tokens": 80
    }
}
```

**对话示例**：

```python
from llama_cpp import Llama

llm = Llama(
    model_path="llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,
    chat_format="llama-2"  # 指定聊天格式
)

# 单轮对话
result = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "你是一个专业的Python编程助手。"},
        {"role": "user", "content": "如何读取CSV文件？"}
    ],
    max_tokens=300,
    temperature=0.7
)
print(result["choices"][0]["message"]["content"])

# 多轮对话
messages = [
    {"role": "system", "content": "你是一个有用的AI助手。"}
]

# 第一轮
messages.append({"role": "user", "content": "什么是RAG？"})
result1 = llm.create_chat_completion(messages=messages, max_tokens=200)
assistant_msg1 = result1["choices"][0]["message"]["content"]
messages.append({"role": "assistant", "content": assistant_msg1})

# 第二轮（模型会参考前文）
messages.append({"role": "user", "content": "它有哪些优势？"})
result2 = llm.create_chat_completion(messages=messages, max_tokens=200)
print(result2["choices"][0]["message"]["content"])
```

### 3.4 语法约束（Grammar）— LlamaGrammar 和 GBNF

语法约束是 llama-cpp-python 的特色功能，允许用 GBNF（GGML BNF）语法定义输出格式，确保模型输出严格符合指定结构。

#### 3.4.1 LlamaGrammar 类

```python
from llama_cpp import LlamaGrammar

# 从 GBNF 字符串创建
grammar = LlamaGrammar.from_string("""
    root ::= sentence
    sentence ::= noun verb object
    noun ::= "cat" | "dog" | "bird"
    verb ::= "chases" | "watches" | "ignores"
    object ::= "the mouse" | "the fish" | "the worm"
""")

# 从文件创建
grammar = LlamaGrammar.from_file("grammar.gbnf")

# 在推理中使用
result = llm.create_completion(
    prompt="Write a sentence: ",
    grammar=grammar,
    max_tokens=50,
    temperature=0.0
)
```

#### 3.4.2 GBNF 语法参考

GBNF 是 BNF（巴科斯范式）的扩展，用于定义上下文无关文法：

```gbnf
# 基本规则定义
root ::= rule1 | rule2          # 选择（或）
rule1 ::= "literal" rule2      # 字面量和规则串联

# 字符类
digit ::= [0-9]
alpha ::= [a-zA-Z]
alnum ::= [a-zA-Z0-9]

# 重复
word ::= alpha+                 # 一个或多个
spaces ::= space*               # 零个或多个
optional ::= (rule)?            # 零个或一个

# JSON 示例
root ::= object
object ::= "{" ws "}" | "{" key-value ("," key-value)* "}"
key-value ::= string ":" value
value ::= object | array | string | number | "true" | "false" | "null"
array ::= "[" ws "]" | "[" value ("," value)* "]"
string ::= "\"" [^"]* "\""
number ::= [0-9]+ ("." [0-9]+)? (("e" | "E") ("+" | "-")? [0-9]+)?
ws ::= [ \t\n]*
```

#### 3.4.3 实用 JSON 输出约束

```python
from llama_cpp import Llama, LlamaGrammar

llm = Llama(model_path="model.gguf", n_ctx=2048)

# 强制输出 JSON 数组
json_grammar = LlamaGrammar.from_string("""
    root ::= array
    array ::= "[" ws item ("," ws item)* ws "]"
    item ::= "{" ws "\\"name\\"" ":" ws string "," ws "\\"age\\"" ":" ws number ws "}"
    string ::= "\\"" [^"]* "\\""
    number ::= [0-9]+
    ws ::= [ \\t\\n]*
""")

result = llm.create_completion(
    prompt="Generate 3 person records: ",
    grammar=json_grammar,
    max_tokens=200,
    temperature=0.7
)
print(result["choices"][0]["text"])
# 输出类似: [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Carol", "age": 35}]
```

#### 3.4.4 response_format JSON 模式（更简单的方式）

```python
# 使用 OpenAI 兼容的 JSON 模式（更简洁）
result = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "以JSON格式输出，包含name和age字段。"},
        {"role": "user", "content": "描述三个人"}
    ],
    response_format={"type": "json_object"},
    max_tokens=200
)
```

### 3.5 嵌入提取

#### 3.5.1 create_embedding()

```python
llm = Llama(model_path="model.gguf", embedding=True, n_ctx=512)

# 单文本嵌入
result = llm.create_embedding(
    input="这是一段测试文本",     # 输入文本
)

# 返回结构
# {
#     "object": "list",
#     "data": [{
#         "object": "embedding",
#         "embedding": [0.123, -0.456, 0.789, ...],  # 嵌入向量
#         "index": 0
#     }],
#     "model": "model.gguf",
#     "usage": {"prompt_tokens": 8, "total_tokens": 8}
# }

# 多文本嵌入
result = llm.create_embedding(
    input=["第一段文本", "第二段文本", "第三段文本"]
)
```

#### 3.5.2 embed() — 简化嵌入接口

```python
# 直接返回 numpy 数组，更方便后续处理
embeddings = llm.embed("这是一段测试文本")
# 返回: numpy.ndarray, shape=(n_embd,)

# 批量嵌入
embeddings = llm.embed(["文本1", "文本2", "文本3"])
# 返回: numpy.ndarray, shape=(n_texts, n_embd)
```

#### 3.5.3 嵌入相似度计算

```python
import numpy as np
from llama_cpp import Llama

llm = Llama(model_path="embed-model.gguf", embedding=True, n_ctx=512)

texts = [
    "机器学习是人工智能的一个分支",
    "深度学习使用神经网络进行学习",
    "今天天气真好，适合出去玩",
    "猫是一种常见的宠物动物"
]

embeddings = llm.embed(texts)

# 余弦相似度
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 计算文本间的相似度矩阵
n = len(texts)
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])

# 语义搜索
query = "什么是AI？"
query_emb = llm.embed(query)
scores = [cosine_similarity(query_emb, emb) for emb in embeddings]
best_idx = np.argmax(scores)
print(f"最相关的文本: {texts[best_idx]} (score: {scores[best_idx]:.4f})")
```

### 3.6 流式输出

#### 3.6.1 Completion 流式输出

```python
from llama_cpp import Llama

llm = Llama(model_path="model.gguf", n_ctx=2048)

# 流式文本补全
stream = llm.create_completion(
    prompt="请用中文介绍Python编程语言：",
    max_tokens=300,
    temperature=0.7,
    stream=True
)

for chunk in stream:
    text = chunk["choices"][0]["text"]
    print(text, end="", flush=True)
print()  # 换行
```

#### 3.6.2 Chat Completion 流式输出

```python
stream = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "写一首关于编程的诗"}
    ],
    max_tokens=200,
    temperature=0.8,
    stream=True
)

for chunk in stream:
    delta = chunk["choices"][0].get("delta", {})
    if "content" in delta:
        print(delta["content"], end="", flush=True)
print()
```

### 3.7 OpenAI 兼容服务器

llama-cpp-python 内置了与 OpenAI API 兼容的 HTTP 服务器。

#### 3.7.1 启动服务器

```bash
# 基础启动
python -m llama_cpp.server \
  --model model.gguf \
  --host 0.0.0.0 \
  --port 8000

# 带GPU加速
python -m llama_cpp.server \
  --model model.gguf \
  --n_gpu_layers -1 \
  --host 0.0.0.0 \
  --port 8000

# 完整配置
python -m llama_cpp.server \
  --model model.gguf \
  --n_ctx 4096 \
  --n_gpu_layers -1 \
  --n_batch 512 \
  --host 0.0.0.0 \
  --port 8000 \
  --chat_format chatml \
  --embedding
```

**服务器关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必填 | GGUF 模型路径 |
| `--n_ctx` | 512 | 上下文窗口大小 |
| `--n_gpu_layers` | 0 | GPU 层数 |
| `--n_batch` | 512 | 批次大小 |
| `--host` | localhost | 监听地址 |
| `--port` | 8000 | 监听端口 |
| `--chat_format` | 自动检测 | 聊天格式 |
| `--embedding` | False | 启用嵌入端点 |
| `--api_key` | 无 | API 密钥认证 |

#### 3.7.2 使用 OpenAI SDK 调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx"  # 如服务端设置了 --api_key 则需匹配
)

# Chat Completion
response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "什么是量子计算？"}
    ],
    max_tokens=300,
    temperature=0.7
)
print(response.choices[0].message.content)

# 流式 Chat
stream = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "写一个Python快速排序"}],
    max_tokens=300,
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# Embeddings
embedding = client.embeddings.create(
    model="local-model",
    input="这是一段测试文本"
)
print(len(embedding.data[0].embedding))  # 嵌入维度
```

### 3.8 模型下载与管理

#### 3.8.1 从 HuggingFace 下载模型

```python
from llama_cpp import Llama

# 直接指定 HuggingFace 仓库和文件名（自动下载）
llm = Llama.from_pretrained(
    repo_id="TheBloke/Llama-2-7B-Chat-GGUF",      # HF 仓库ID
    filename="llama-2-7b-chat.Q4_K_M.gguf",        # 文件名
    n_ctx=2048,
    n_gpu_layers=0,
)
```

#### 3.8.2 手动下载和管理

```python
import os
from huggingface_hub import hf_hub_download

# 下载模型
model_path = hf_hub_download(
    repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_K_M.gguf",
    local_dir="./models"  # 本地存储目录
)

# 使用本地路径加载
llm = Llama(model_path=model_path, n_ctx=2048)
```

#### 3.8.3 多模型管理

```python
# 不同任务使用不同模型
chat_model = Llama(model_path="./models/chat.Q4_K_M.gguf", n_ctx=4096)
embed_model = Llama(model_path="./models/embed.gguf", embedding=True, n_ctx=512)
code_model = Llama(model_path="./models/coder.Q4_K_M.gguf", n_ctx=4096)

# 注意：同时加载多个模型会占用大量内存，建议按需加载
```

### 3.9 LoRA 适配器

```python
# 加载基础模型并应用 LoRA
llm = Llama(
    model_path="./models/base.Q4_K_M.gguf",
    lora_base="./models/base-f16.gguf",   # LoRA 基础模型（FP16）
    lora_path="./loras/chinese_lora.bin",  # LoRA 适配器路径
    lora_scale=1.0,                         # 缩放因子
    n_ctx=2048,
)
```

---

## 4. 在LLM开发中的典型使用场景和代码示例

### 4.1 场景一：本地 RAG 系统

```python
import numpy as np
from llama_cpp import Llama

# 加载嵌入模型和生成模型
embed_llm = Llama(model_path="./models/embed.gguf", embedding=True, n_ctx=512)
gen_llm = Llama(model_path="./models/chat.Q4_K_M.gguf", n_ctx=4096)

# 文档库
documents = [
    "Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年创建。",
    "机器学习是人工智能的一个子领域，它使计算机能够从数据中学习而无需显式编程。",
    "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的层次表示。",
    "自然语言处理（NLP）是人工智能的一个领域，专注于计算机与人类语言之间的交互。",
    "计算机视觉是人工智能的一个领域，专注于让计算机理解和处理视觉信息。",
]

# 构建向量索引
doc_embeddings = embed_llm.embed(documents)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def rag_query(query: str, top_k: int = 2) -> str:
    # 1. 检索相关文档
    query_emb = embed_llm.embed(query)
    scores = [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embeddings]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    context = "\n".join([documents[i] for i in top_indices])

    # 2. 生成回答
    result = gen_llm.create_chat_completion(
        messages=[
            {"role": "system", "content": f"根据以下参考信息回答问题。如果参考信息中没有答案，请说明。\n\n参考信息：\n{context}"},
            {"role": "user", "content": query}
        ],
        max_tokens=300,
        temperature=0.5
    )
    return result["choices"][0]["message"]["content"]

# 测试
answer = rag_query("什么是深度学习？它和机器学习有什么关系？")
print(answer)
```

### 4.2 场景二：结构化信息提取

```python
from llama_cpp import Llama, LlamaGrammar

llm = Llama(model_path="./models/chat.Q4_K_M.gguf", n_ctx=2048)

# 定义输出语法：提取人名、地点和事件
extract_grammar = LlamaGrammar.from_string("""
    root ::= "{" ws "\\"people\\"" ":" ws array "," ws "\\"locations\\"" ":" ws array "," ws "\\"events\\"" ":" ws array ws "}"
    array ::= "[" ws "]" | "[" ws string ("," ws string)* ws "]"
    string ::= "\\"" [^"\\\\]* "\\""
    ws ::= [ \\t\\n]*
""")

text = """
2023年5月，张伟和李明在北京参加了全球AI峰会。会议期间，王芳在上海举办了
一场关于大模型技术的研讨会。随后，张伟前往深圳参观了AI实验室。
"""

result = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "从文本中提取人名、地点和事件，以JSON格式输出。"},
        {"role": "user", "content": text}
    ],
    grammar=extract_grammar,
    max_tokens=200,
    temperature=0.0
)

import json
extracted = json.loads(result["choices"][0]["message"]["content"])
print(extracted)
# {"people": ["张伟", "李明", "王芳"], "locations": ["北京", "上海", "深圳"], "events": [...]}
```

### 4.3 场景三：流式对话机器人

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/chat.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,  # 全部卸载到 GPU
    chat_format="chatml"
)

def chat():
    messages = [
        {"role": "system", "content": "你是一个友好、专业的AI助手，用中文回答问题。"}
    ]

    print("聊天机器人已启动（输入 'quit' 退出）\n")

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # 流式生成回复
        print("AI: ", end="", flush=True)
        full_response = ""

        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            stream=True
        )

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                text = delta["content"]
                print(text, end="", flush=True)
                full_response += text

        print("\n")

        # 保存到对话历史
        messages.append({"role": "assistant", "content": full_response})

        # 简单的上下文管理：超过上下文长度时保留系统提示+最近对话
        if len(str(messages)) > 3500:
            messages = [messages[0]] + messages[-4:]  # 系统提示+最近2轮

if __name__ == "__main__":
    chat()
```

### 4.4 场景四：批量文本处理流水线

```python
from llama_cpp import Llama
import json

llm = Llama(model_path="./models/chat.Q4_K_M.gguf", n_ctx=2048)

def classify_text(text: str, categories: list[str]) -> str:
    """文本分类"""
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": f"将以下文本分类到这些类别之一：{', '.join(categories)}。只输出类别名称。"},
            {"role": "user", "content": text}
        ],
        max_tokens=10,
        temperature=0.0
    )
    return result["choices"][0]["message"]["content"].strip()

def summarize_text(text: str, max_length: int = 100) -> str:
    """文本摘要"""
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": f"用不超过{max_length}字总结以下文本。"},
            {"role": "user", "content": text}
        ],
        max_tokens=max_length,
        temperature=0.3
    )
    return result["choices"][0]["message"]["content"].strip()

def translate_text(text: str, target_lang: str = "English") -> str:
    """文本翻译"""
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": f"将以下文本翻译为{target_lang}。只输出翻译结果。"},
            {"role": "user", "content": text}
        ],
        max_tokens=300,
        temperature=0.3
    )
    return result["choices"][0]["message"]["content"].strip()

# 批量处理
articles = [
    "苹果公司发布了最新一代iPhone，搭载了更强大的AI芯片...",
    "研究团队在《自然》杂志发表论文，发现了一种新型超导材料...",
    "中国国家足球队在世界杯预选赛中以2:1战胜对手...",
]

categories = ["科技", "科学", "体育", "政治", "经济"]

for article in articles:
    category = classify_text(article, categories)
    summary = summarize_text(article)
    translation = translate_text(article, "English")
    print(f"分类: {category}")
    print(f"摘要: {summary}")
    print(f"英文: {translation}")
    print("---")
```

---

## 5. 数学原理

### 5.1 采样方法数学原理

#### 5.1.1 Temperature 采样

Temperature 通过缩放 logits（未归一化概率）来控制输出的随机性：

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

其中 $z_i$ 是第 $i$ 个 token 的 logit，$T$ 是温度参数：

- $T \to 0$：分布趋向 one-hot（贪心解码）
- $T = 1$：原始分布
- $T > 1$：分布更均匀（更随机）

#### 5.1.2 Top-p（核采样）

Top-p 采样选择概率累积和达到 $p$ 的最小 token 集合：

1. 按概率降序排列 token：$x_1, x_2, ..., x_V$
2. 计算累积概率：$C = \{x_i : \sum_{j=1}^{i} P(x_j) \leq p\}$
3. 从 $C$ 中按归一化概率采样

```
示例（p=0.9）:
token概率: A=0.4, B=0.3, C=0.2, D=0.05, E=0.03, F=0.02
累积: A=0.4, A+B=0.7, A+B+C=0.9 ✓ → 候选集 {A, B, C}
归一化: A=0.4/0.9, B=0.3/0.9, C=0.2/0.9
```

#### 5.1.3 Top-k 采样

Top-k 采样只保留概率最高的 $k$ 个 token：

1. 按概率降序排列 token
2. 只保留前 $k$ 个
3. 归一化后采样

#### 5.1.4 Min-p 采样

Min-p 设置概率的相对最低阈值，只保留概率 ≥ 最大概率 × min_p 的 token：

$$C = \{x_i : P(x_i) \geq \min\_p \times \max_j P(x_j)\}$$

这解决了 top-p 在分布平坦时保留过多 token 的问题。

#### 5.1.5 Mirostat 采样

Mirostat 是一种自适应采样方法，目标是在生成过程中维持恒定的"惊奇度"（perplexity）：

$$\text{目标：保持 } H(P) \approx \tau$$

其中 $\tau$ 是目标熵（`mirostat_tau`），$\eta$ 是学习率（`mirostat_eta`）。

Mirostat 动态调整 top-k 的值来维持目标熵：
- 如果输出太可预测（$H < \tau$），增加 $k$（引入更多随机性）
- 如果输出太随机（$H > \tau$），减少 $k$（增加确定性）

### 5.2 嵌入向量数学原理

嵌入向量是 Transformer 最后一层隐藏状态的平均池化：

$$\mathbf{e} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{h}_i$$

其中 $\mathbf{h}_i$ 是第 $i$ 个 token 在最后一层的隐藏状态，$n$ 是序列长度。

余弦相似度衡量两个向量的方向相似性：

$$\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|} = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2} \cdot \sqrt{\sum_i b_i^2}}$$

值域为 $[-1, 1]$，1 表示完全相同方向，0 表示正交，-1 表示完全相反。

### 5.3 语法约束的数学原理

GBNF 语法约束通过**logit 掩码**实现：

1. 根据 GBNF 语法，计算当前位置允许的 token 集合 $A$
2. 将不在 $A$ 中的 token 的 logits 设为 $-\infty$：

$$z_i' = \begin{cases} z_i & \text{if } i \in A \\ -\infty & \text{if } i \notin A \end{cases}$$

3. 对修改后的 logits 应用 softmax 和采样

这确保了生成的 token 序列始终符合定义的语法规则。语法解析器维护一个有限状态机（FSM），每个生成步骤根据当前状态确定允许的 token 集合。

---

## 6. 代码原理/架构原理

### 6.1 整体架构

```
┌──────────────────────────────────────┐
│         Python Application           │
├──────────────────────────────────────┤
│         llama-cpp-python             │
│  ┌──────────────────────────────┐   │
│  │    High-Level API            │   │
│  │  create_completion()         │   │
│  │  create_chat_completion()    │   │
│  │  create_embedding()          │   │
│  │  embed()                     │   │
│  ├──────────────────────────────┤   │
│  │    Llama Class               │   │
│  │  __init__() → load model     │   │
│  │  tokenize / detokenize       │   │
│  │  eval (llama_decode)         │   │
│  │  sample (sampler)            │   │
│  ├──────────────────────────────┤   │
│  │    LlamaGrammar              │   │
│  │  GBNF parser → FSM          │   │
│  │  logit masking              │   │
│  ├──────────────────────────────┤   │
│  │    Chat Format Handlers      │   │
│  │  ChatML / LLaMA-2 / ...     │   │
│  │  messages → prompt string    │   │
│  ├──────────────────────────────┤   │
│  │    Server (OpenAI compat.)   │   │
│  │  FastAPI / Starlette         │   │
│  │  /v1/chat/completions        │   │
│  │  /v1/completions             │   │
│  │  /v1/embeddings              │   │
│  └──────────────┬───────────────┘   │
├─────────────────┼────────────────────┤
│   CFFI / ctypes │ Python ↔ C 桥梁    │
├─────────────────┼────────────────────┤
│         llama.cpp C Library          │
│  libllama.so / libllama.dylib       │
└──────────────────────────────────────┘
```

### 6.2 Python-C 桥接原理

llama-cpp-python 使用 CFFI（C Foreign Function Interface）调用 llama.cpp 的 C 接口：

1. **加载共享库**：运行时加载 `libllama.so`（Linux）/ `libllama.dylib`（macOS）/ `llama.dll`（Windows）
2. **函数声明**：声明 C 函数签名，如 `llama_model_load_from_file`、`llama_decode` 等
3. **内存管理**：Python 侧的 `Llama` 对象持有 C 侧的模型和上下文指针，通过 `__del__` 析构函数释放
4. **数据转换**：
   - Python `str` → C `char*`（编码为 UTF-8）
   - Python `list[int]` → C `llama_token*` 数组
   - C `float*` → Python `list[float]` / `numpy.ndarray`

### 6.3 Chat Format 处理

`create_chat_completion()` 内部将 messages 列表转换为模型特定的 prompt 格式：

```python
# ChatML 格式（如 Qwen、Mistral 等）
# messages → prompt 转换
"""
<|im_start|>system
你是一个有用的AI助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
"""

# LLaMA-2 格式
"""
[INST] <<SYS>>
你是一个有用的AI助手。
<</SYS>>

你好 [/INST]
"""

# 自动检测格式：根据 GGUF 元数据中的 tokenizer.chat_template
```

### 6.4 采样器架构

llama-cpp-python 使用 llama.cpp 的链式采样器架构：

```
Logits → Repetition Penalty → Frequency/Presence Penalty → Temperature
    → Top-K → Top-P → Min-P → Typical-P → Tail-Free Sampling
    → Mirostat (or) → Grammar Mask → Random Sample → Token
```

每个采样器修改 logits 或候选集，最终由随机采样器选择 token。

---

## 7. 常见注意事项和最佳实践

### 7.1 性能优化

- **GPU 加速是关键**：`n_gpu_layers=-1` 将所有层卸载到 GPU，推理速度可提升 5-20 倍
- **Batch Size 调优**：`n_batch` 设为 512 或 1024 可加速 prompt 处理
- **复用模型实例**：避免重复加载模型，使用全局单例或连接池
- **异步处理**：生产环境使用服务器模式 + 连续批处理，而非直接调用 Python API

### 7.2 内存管理

- **模型只加载一次**：`Llama()` 构造函数开销大，应复用实例
- **上下文长度设为实际需要**：`n_ctx` 直接影响 KV Cache 大小
- **使用 mmap**：默认开启，启动速度快且多进程可共享物理页
- **注意 Python GC**：显式 `del llm` 可释放模型内存，不要依赖垃圾回收时机

```python
# 推荐模式：全局单例
_model = None

def get_model():
    global _model
    if _model is None:
        _model = Llama(model_path="model.gguf", n_ctx=4096, n_gpu_layers=-1)
    return _model
```

### 7.3 采样参数建议

| 场景 | temperature | top_p | repeat_penalty | 说明 |
|------|------------|-------|----------------|------|
| 事实问答 | 0.0-0.3 | 0.9 | 1.1 | 低温度，确定性输出 |
| 创意写作 | 0.7-1.0 | 0.95 | 1.1 | 高温度，更多样性 |
| 代码生成 | 0.0-0.2 | 0.9 | 1.1 | 低温度，确保正确性 |
| 数据提取 | 0.0 | 1.0 | 1.0 | 贪心解码，确定性 |
| 对话聊天 | 0.5-0.7 | 0.9 | 1.15 | 适中温度，自然回复 |

### 7.4 常见问题与解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `ValueError: Model path does not exist` | 模型路径错误 | 使用绝对路径或检查文件是否存在 |
| `RuntimeError: CUDA out of memory` | GPU 显存不足 | 减少 `n_gpu_layers` 或使用更低精度量化 |
| 生成速度极慢 | 未启用 GPU 加速 | 设置 `n_gpu_layers=-1` |
| 生成内容重复 | 采样参数不当 | 增大 `repeat_penalty`（1.1-1.3），检查 `temperature` |
| 输出乱码 | 聊天格式不匹配 | 检查 `chat_format` 是否与模型匹配 |
| `SIGSEGV` 崩溃 | C 库版本不匹配 | 重新编译安装，确保版本一致 |
| 嵌入全为零 | 未启用嵌入模式 | 创建时设置 `embedding=True` |

### 7.5 服务器部署建议

```bash
# 生产环境推荐配置
python -m llama_cpp.server \
  --model model.Q4_K_M.gguf \
  --n_ctx 4096 \
  --n_gpu_layers -1 \
  --n_batch 1024 \
  --host 0.0.0.0 \
  --port 8000 \
  --api_key your-secret-key \    # 启用认证
  --embedding                     # 同时启用嵌入
```

```python
# 使用多 worker 处理并发（需要 gunicorn 等部署工具）
# 注意：多 worker 模式下每个 worker 会加载一份模型，内存消耗翻倍
# 推荐使用单 worker + 连续批处理模式
```

### 7.6 版本兼容性

- llama-cpp-python 的版本必须与底层 llama.cpp 版本兼容
- 从 PyPI 安装时会自动编译对应版本的 C 库
- 如果遇到兼容性问题，尝试从源码重新编译：
  ```bash
  pip install llama-cpp-python --no-binary :all: --force-reinstall
  ```
- GGUF 格式版本需与库版本匹配（当前推荐 GGUF v3）
