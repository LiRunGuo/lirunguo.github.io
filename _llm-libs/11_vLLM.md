---
title: "vLLM 高性能推理引擎"
excerpt: "PagedAttention原理、连续批处理、LLM类/SamplingParams、OpenAI兼容服务器"
collection: llm-libs
permalink: /llm-libs/11-vllm
category: inference
toc: true
---


## 1. 简介

vLLM 是一个高性能的大语言模型（LLM）推理与服务引擎，由加州大学伯克利分校 Sky Computing Lab 开发。其核心目标是解决 LLM 推理中的显存瓶颈问题，通过创新的 PagedAttention 机制实现接近零浪费的 KV cache 显存管理，显著提升推理吞吐量。

### 在 LLM 开发中的作用

- **推理加速**：通过 PagedAttention 和连续批处理（Continuous Batching）技术，vLLM 的推理吞吐量比传统 HuggingFace Transformers 高出 10-24 倍。
- **服务部署**：提供与 OpenAI API 兼容的服务器，可快速将模型部署为 API 服务。
- **多 GPU 支持**：内置张量并行（Tensor Parallelism），支持多 GPU 分布式推理。
- **离线与在线推理**：同时支持离线批量推理和在线实时推理两种模式。

---

## 2. 安装方式

### 基本安装

```bash
# 使用 pip 安装（需要 CUDA 12.1+）
pip install vllm

# 从源码安装
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### 指定 CUDA 版本

```bash
# CUDA 12.1
pip install vllm

# CUDA 11.8
pip install vllm==0.4.0  # 旧版本支持
```

### 依赖说明

- Python >= 3.9
- PyTorch >= 2.1
- NVIDIA GPU（计算能力 >= 7.0，即 Volta 架构及以上）
- CUDA >= 11.8

---

## 3. 核心类与函数详细说明

### 3.1 LLM 类

`LLM` 类是 vLLM 离线推理的核心入口，封装了模型加载、推理调度和 KV cache 管理等功能。

#### 初始化参数

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",       # 模型名称或本地路径（必填）
    tokenizer=None,                          # 分词器路径，默认与model相同
    tokenizer_mode="auto",                   # 分词器模式："auto"或"slow"
    tokenizer_revision=None,                 # 分词器的HuggingFace revision
    trust_remote_code=False,                 # 是否信任远程代码（自定义模型需要）
    tensor_parallel_size=1,                  # 张量并行数（GPU数量）
    dtype="auto",                            # 数据类型："auto","float16","bfloat16","float32"
    quantization=None,                       # 量化方式："awq","gptq","squeezellm","fp8"等
    gpu_memory_utilization=0.9,              # GPU显存利用率（0~1），预留部分给其他操作
    swap_space=4,                            # CPU交换空间大小（GB），用于KV cache换出
    enforce_eager=False,                     # 是否强制使用eager模式（禁用CUDA Graph）
    max_seq_len_to_capture=8192,            # CUDA Graph捕获的最大序列长度
    max_model_len=None,                      # 模型最大序列长度，默认从模型配置读取
    speculative_model=None,                  # 推测解码的草稿模型
    num_speculative_tokens=None,             # 推测解码的草稿token数
    enable_prefix_caching=False,             # 是否启用前缀缓存
    enable_lora=False,                       # 是否启用LoRA
    max_lora_rank=0,                         # LoRA最大秩
    max_num_seqs=256,                        # 最大并发序列数
    max_num_batched_tokens=None,             # 每次迭代最大批处理token数
    revision=None,                           # 模型的HuggingFace revision
    code_revision=None,                      # 代码的HuggingFace revision
    seed=0,                                  # 随机种子
)
```

#### 关键参数详解

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `model` | HuggingFace 模型 ID 或本地路径 | - |
| `tensor_parallel_size` | 张量并行的 GPU 数量 | 根据模型大小选择 |
| `gpu_memory_utilization` | 预留的 GPU 显存比例 | 0.85~0.95 |
| `swap_space` | CPU 交换空间，用于 KV cache 溢出 | 4~16 GB |
| `enable_prefix_caching` | 启用前缀缓存，复用公共前缀的 KV cache | 长系统提示时建议开启 |
| `quantization` | 量化方式，降低显存占用 | "awq"或"fp8" |

#### generate() 方法

```python
outputs = llm.generate(
    prompts,                    # 输入提示，可以是字符串列表或TokensPrompt列表
    sampling_params=None,       # SamplingParams对象，控制生成行为
    use_tqdm=True,             # 是否显示进度条
    lora_request=None,         # LoRA请求（需启用enable_lora）
)
```

**返回值**：`List[RequestOutput]`，每个 `RequestOutput` 包含：

```python
for output in outputs:
    prompt = output.prompt           # 原始输入提示
    generated_text = output.outputs[0].text        # 生成的文本
    token_ids = output.outputs[0].token_ids        # 生成的token ID列表
    finish_reason = output.outputs[0].finish_reason  # 结束原因："stop"或"length"
    cumulative_logprob = output.outputs[0].cumulative_logprob  # 累积对数概率
```

#### 完整离线推理示例

```python
from vllm import LLM, SamplingParams

# 1. 初始化模型
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,          # 使用2块GPU进行张量并行
    gpu_memory_utilization=0.9,      # 使用90%的GPU显存
    enable_prefix_caching=True,      # 启用前缀缓存
)

# 2. 设置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512,
)

# 3. 批量推理
prompts = [
    "请解释什么是机器学习？",
    "Python中的装饰器是什么？",
    "量子计算的基本原理是什么？",
]
outputs = llm.generate(prompts, sampling_params)

# 4. 输出结果
for output in outputs:
    print(f"提示: {output.prompt!r}")
    print(f"生成: {output.outputs[0].text!r}")
    print("---")
```

### 3.2 SamplingParams 类

`SamplingParams` 控制文本生成的采样策略，是 vLLM 中最常用的配置类之一。

```python
from vllm import SamplingParams

params = SamplingParams(
    n=1,                          # 每个提示生成的补全数量
    best_of=None,                 # 从n个候选中选择最佳的个数（需n<best_of）
    presence_penalty=0.0,         # 存在惩罚：正值降低重复token的概率
    frequency_penalty=0.0,        # 频率惩罚：基于token出现频率的惩罚
    repetition_penalty=1.0,       # 重复惩罚：>1时惩罚重复token
    temperature=1.0,              # 温度：控制随机性，0为贪心解码
    top_p=1.0,                    # Top-p（核采样）：0.1表示只考虑概率前10%的token
    top_k=-1,                     # Top-k：只从概率最高的k个token中采样，-1表示不限制
    min_p=0.0,                    # Min-p：过滤掉概率低于最高概率token的min_p倍的token
    seed=None,                    # 采样随机种子
    use_beam_search=False,        # 是否使用束搜索
    length_penalty=1.0,           # 长度惩罚：束搜索时对长度的惩罚
    early_terminating=None,       # 早停策略："high"或"low"
    stop=None,                    # 停止词列表，遇到这些字符串时停止生成
    stop_token_ids=None,          # 停止token ID列表
    include_stop_str_in_output=False,  # 是否在输出中包含停止字符串
    ignore_eos=False,             # 是否忽略EOS token
    max_tokens=16,                # 最大生成token数
    min_tokens=0,                 # 最小生成token数
    logprobs=None,                # 返回top-N个token的对数概率
    prompt_logprobs=None,         # 返回提示token的对数概率
    skip_special_tokens=True,     # 是否跳过特殊token
    spaces_between_special_tokens=True,  # 特殊token之间是否加空格
    logits_processors=None,       # 自定义logits处理器列表
    truncate_prompt_tokens=None,  # 截断提示到指定token数
)
```

#### 核心参数详解

**temperature（温度）**
- 控制输出的随机性
- `temperature=0`：贪心解码，始终选择概率最高的token
- `temperature<1`：更确定性，输出更集中
- `temperature>1`：更随机，输出更多样化
- 公式：`P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)`

**top_p（核采样）**
- 只从累积概率达到 `top_p` 的最小token集合中采样
- `top_p=0.1`：只考虑概率最高的、累积概率达到10%的token
- `top_p=1.0`：考虑所有token（不做过滤）

**top_k**
- 只从概率最高的 `top_k` 个token中采样
- `top_k=50`：只从概率最高的50个token中采样
- `top_k=-1`：不做限制

**frequency_penalty / presence_penalty**
- `frequency_penalty`：根据token已出现的次数进行惩罚，次数越多惩罚越大
- `presence_penalty`：只要token出现过就施加固定惩罚，不考虑次数
- 取值范围：[-2.0, 2.0]，正值惩罚重复，负值鼓励重复

#### 采样参数组合示例

```python
from vllm import SamplingParams

# 贪心解码（确定性输出）
greedy_params = SamplingParams(temperature=0, max_tokens=256)

# 创意写作（高随机性）
creative_params = SamplingParams(
    temperature=1.2,
    top_p=0.95,
    top_k=100,
    max_tokens=1024,
    repetition_penalty=1.1,
)

# 代码生成（低随机性）
code_params = SamplingParams(
    temperature=0.2,
    top_p=0.9,
    max_tokens=512,
    stop=["\n\n", "def ", "class "],  # 遇到这些模式时停止
)

# 多候选生成
multi_params = SamplingParams(
    n=3,                  # 生成3个候选
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
)
```

### 3.3 OpenAI 兼容服务器

vLLM 提供了与 OpenAI API 兼容的 HTTP 服务器，可以直接作为 OpenAI API 的替代后端。

#### 启动服务器

```bash
# 基本启动
vllm serve meta-llama/Llama-2-7b-hf

# 完整参数启动
vllm serve meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --enable-prefix-caching \
    --quantization awq \
    --served-model-name my-llama2
```

#### 常用服务器参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 监听地址 | "localhost" |
| `--port` | 监听端口 | 8000 |
| `--tensor-parallel-size` | 张量并行数 | 1 |
| `--gpu-memory-utilization` | GPU显存利用率 | 0.9 |
| `--max-model-len` | 最大序列长度 | 模型默认 |
| `--enable-prefix-caching` | 启用前缀缓存 | False |
| `--quantization` | 量化方式 | None |
| `--served-model-name` | 服务中的模型名称 | 模型原始名称 |
| `--chat-template` | 聊天模板路径 | 自动检测 |
| `--enable-lora` | 启用LoRA | False |

#### API 调用示例

**Completions API（/v1/completions）**

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM默认不需要API key
)

# 文本补全
response = client.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    prompt="请解释什么是深度学习：",
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
)
print(response.choices[0].text)
```

**Chat Completions API（/v1/chat/completions）**

```python
# 多轮对话
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "请解释什么是Transformer架构？"},
    ],
    max_tokens=512,
    temperature=0.7,
)
print(response.choices[0].message.content)

# 流式输出
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "user", "content": "写一首关于春天的诗"},
    ],
    max_tokens=256,
    stream=True,  # 启用流式输出
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**使用 curl 调用**

```bash
# Completions API
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "prompt": "什么是机器学习？",
        "max_tokens": 256,
        "temperature": 0.7
    }'

# Chat Completions API
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-hf",
        "messages": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "解释量子计算"}
        ],
        "max_tokens": 512
    }'
```

### 3.4 离线推理 vs 在线推理

| 特性 | 离线推理 | 在线推理 |
|------|----------|----------|
| 入口 | `LLM` 类 | `vllm serve` 命令 |
| 适用场景 | 批量数据处理、评估 | 实时API服务 |
| 延迟要求 | 不敏感 | 敏感 |
| 吞吐优化 | 全局调度 | 连续批处理 |
| 代码复杂度 | 低 | 需要HTTP服务器 |
| 典型用途 | 数据集推理、基准测试 | 生产环境部署 |

**离线推理示例（批量处理数据集）**

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
params = SamplingParams(temperature=0, max_tokens=128)

# 读取数据集并批量推理
import json
with open("dataset.jsonl") as f:
    prompts = [json.loads(line)["prompt"] for line in f]

outputs = llm.generate(prompts, params)

# 保存结果
results = []
for output in outputs:
    results.append({
        "prompt": output.prompt,
        "completion": output.outputs[0].text,
    })

with open("results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

**在线推理示例（实时API服务）**

```python
# 终端1：启动服务器
# vllm serve meta-llama/Llama-2-7b-hf --port 8000

# 终端2：客户端调用
import openai
import time

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 实时请求
start = time.time()
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=64,
)
latency = time.time() - start
print(f"响应: {response.choices[0].message.content}")
print(f"延迟: {latency:.2f}s")
```

---

## 4. 典型使用场景与代码示例

### 4.1 RAG（检索增强生成）

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", enable_prefix_caching=True)
params = SamplingParams(temperature=0.3, max_tokens=512)

# 共享的系统提示会通过前缀缓存复用KV cache
system_prompt = "你是一个专业的问答助手。根据以下上下文回答问题，如果上下文中没有答案，请说明。"

def rag_query(question: str, context: str) -> str:
    prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n上下文：{context}\n\n问题：{question} [/INST]"
    outputs = llm.generate([prompt], params)
    return outputs[0].outputs[0].text

# 多个查询共享system_prompt的KV cache
answer1 = rag_query("什么是BERT？", "BERT是Google提出的预训练语言模型...")
answer2 = rag_query("什么是GPT？", "GPT是OpenAI提出的生成式预训练模型...")
```

### 4.2 多模型批量推理

```python
from vllm import LLM, SamplingParams

# 使用张量并行加载大模型
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,          # 4卡并行
    gpu_memory_utilization=0.92,
    quantization="awq",              # 使用AWQ量化降低显存
)

# 不同任务的采样参数
factual_params = SamplingParams(temperature=0, max_tokens=256)
creative_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1024)

# 事实性问答
factual_prompts = ["法国的首都是什么？", "水的化学式是什么？"]
factual_outputs = llm.generate(factual_prompts, factual_params)

# 创意写作
creative_prompts = ["写一篇关于人工智能未来的科幻短文"]
creative_outputs = llm.generate(creative_prompts, creative_params)
```

### 4.3 流式推理服务

```python
# 服务端启动：
# vllm serve meta-llama/Llama-2-7b-chat-hf --enable-prefix-caching

# 客户端流式调用
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

def chat_stream(message: str):
    """流式聊天函数，实时输出模型响应"""
    stream = client.chat.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        messages=[
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": message},
        ],
        max_tokens=512,
        temperature=0.7,
        stream=True,
    )
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end="", flush=True)
    print()
    return full_response

chat_stream("请详细解释Transformer中的自注意力机制")
```

### 4.4 LoRA 多适配器服务

```python
# 启动支持LoRA的服务器
# vllm serve meta-llama/Llama-2-7b-hf \
#     --enable-lora \
#     --max-lora-rank 16

# 客户端使用LoRA
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 使用基础模型
response_base = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "写一首诗"}],
    max_tokens=128,
)

# 使用LoRA适配器（假设已加载）
response_lora = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "写一首诗"}],
    max_tokens=128,
    extra_body={"lora_name": "poetry-lora"},
)
```

---

## 5. 数学原理

### 5.1 PagedAttention：分页注意力机制

PagedAttention 是 vLLM 的核心创新，其设计灵感来源于操作系统的虚拟内存分页管理。

#### 问题背景

在标准 Transformer 推理中，KV cache（键值缓存）需要为每个序列预分配最大长度的连续显存空间。这导致严重的显存浪费：

- **内部碎片**：序列实际长度远小于最大长度时，预分配的空间大量闲置
- **外部碎片**：不同长度的序列频繁分配/释放，导致显存碎片化

实验表明，传统方法的 KV cache 显存利用率仅为 20%~40%。

#### PagedAttention 机制

PagedAttention 将 KV cache 分成固定大小的 **blocks**（类似操作系统的内存页），每个 block 存储 KV cache 的一部分，通过 **BlockTable**（类似页表）建立逻辑到物理的映射。

**核心思想**：

1. **KV cache 分块**：将每个序列的 KV cache 分割为固定大小的 blocks（默认 block 大小为 16 个 token）
2. **BlockTable 映射**：每个序列维护一个 BlockTable，记录逻辑 block 到物理 block 的映射关系
3. **动态分配**：仅在需要时分配新的物理 block，无需预分配最大长度

**数学公式**：

传统方法的显存利用率：

$$
\text{显存利用率}_{\text{传统}} = \frac{\sum_{i=1}^{N} L_i \cdot d}{\sum_{i=1}^{N} L_{\max} \cdot d} = \frac{\sum_{i=1}^{N} L_i}{N \cdot L_{\max}}
$$

其中 $N$ 是序列数，$L_i$ 是第 $i$ 个序列的实际长度，$L_{\max}$ 是最大序列长度，$d$ 是每个 token 的 KV cache 大小。

PagedAttention 的显存利用率：

$$
\text{显存利用率}_{\text{PagedAttn}} = \frac{\sum_{i=1}^{N} L_i \cdot d}{\sum_{i=1}^{N} \lceil L_i / B \rceil \cdot B \cdot d}
$$

其中 $B$ 是 block 大小。当 $B \ll L_i$ 时，利用率接近 1：

$$
\text{显存利用率}_{\text{PagedAttn}} \approx \frac{\sum L_i}{\sum L_i + N \cdot (B/2)} \approx 1 - \frac{N \cdot B}{2 \sum L_i}
$$

**PagedAttention 计算过程**：

对于注意力计算 $\text{Attention}(Q, K, V)$，PagedAttention 按块进行计算：

$$
O_i = \sum_{j} \frac{\exp(q_i^T k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i^T k_{j'} / \sqrt{d_k})} v_j
$$

其中 $k_j$ 和 $v_j$ 通过 BlockTable 从物理 blocks 中读取，逻辑上连续但物理上可以不连续。

#### BlockTable 结构示例

```
序列A（逻辑block: [0, 1, 2]）→ BlockTable: [5, 2, 8]  → 物理block: 5, 2, 8
序列B（逻辑block: [0, 1]）   → BlockTable: [3, 7]    → 物理block: 3, 7

物理block池:
  [0] [1] [2:K/V of A] [3:K/V of B] [4] [5:K/V of A]
  [6] [7:K/V of B] [8:K/V of A] [9] ...
```

### 5.2 连续批处理（Continuous Batching）

传统批处理（Static Batching）在所有序列完成生成后才处理下一批，造成严重的"等待最慢序列"问题。

**连续批处理**（又称 iteration-level scheduling）在每次迭代级别进行调度：

1. 每个解码步骤（iteration），检查哪些序列已经完成生成
2. 完成的序列立即从 batch 中移除
3. 等待队列中的新序列立即插入 batch
4. 保持 GPU 始终满载运行

**吞吐量公式**：

$$
\text{吞吐量} = \frac{\text{完成序列数}}{\text{总时间}} \propto \frac{B_{\text{avg}}}{\bar{L}}
$$

其中 $B_{\text{avg}}$ 是平均批次大小，$\bar{L}$ 是平均输出长度。连续批处理通过保持 $B_{\text{avg}}$ 始终接近最大值，最大化吞吐量。

### 5.3 张量并行（Tensor Parallelism）

张量并行将模型权重沿特定维度切分到多个 GPU 上，每个 GPU 只存储部分权重并计算部分结果。

**线性层的切分**：

- **列并行**（Column Parallel）：将权重矩阵按列切分，每个 GPU 计算部分输出，然后 All-Gather 合并
- **行并行**（Row Parallel）：将权重矩阵按行切分，每个 GPU 计算部分结果，然后 All-Reduce 求和

**MLP 层的张量并行**：

$$
Y = \text{GeLU}(X \cdot A) \cdot B
$$

- 第一层 $A$ 使用列并行：$A = [A_1, A_2, ..., A_n]$，每个 GPU 计算 $X \cdot A_i$
- 第二层 $B$ 使用行并行：$B = [B_1; B_2; ...; B_n]^T$，每个 GPU 计算 $\text{GeLU}(X \cdot A_i) \cdot B_i$
- 最后 All-Reduce 合并结果

**通信开销**：每次前向传播需要 2 次 All-Reduce（Attention 层 + MLP 层），通信量与隐藏维度成正比。

### 5.4 前缀缓存（Prefix Caching）

当多个请求共享相同的前缀（如系统提示）时，前缀的 KV cache 只需计算一次并缓存复用。

**哈希策略**：前缀的 KV cache 以 block 为单位缓存，通过前缀 token 的哈希值作为 key 进行查找。

$$
\text{cache\_key} = \text{hash}(\text{prefix\_tokens}[:k \cdot B])
$$

**节省的计算量**：

$$
\text{节省比例} = \frac{L_{\text{prefix}}}{L_{\text{prefix}} + L_{\text{unique}}}
$$

当系统提示很长而用户输入较短时（如 RAG 场景），节省比例可达 80% 以上。

---

## 6. 代码原理与架构原理

### 6.1 整体架构

```
┌─────────────────────────────────────────────────┐
│                  API Server                      │
│         (OpenAI-compatible HTTP Server)          │
└───────────────────────┬─────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────┐
│              LLMEngine                          │
│  ┌─────────────┐  ┌─────────────┐              │
│  │ Scheduler   │  │ Tokenizer   │              │
│  │ (调度器)     │  │ (分词器)    │              │
│  └──────┬──────┘  └─────────────┘              │
│         │                                       │
│  ┌──────▼──────────────────────────────────┐   │
│  │           ModelRunner                    │   │
│  │  ┌────────────────────────────────────┐ │   │
│  │  │  PagedAttention + CUDA Kernels     │ │   │
│  │  │  ┌──────────┐  ┌──────────────┐   │ │   │
│  │  │  │BlockTable│  │KV Cache Pool │   │ │   │
│  │  │  │Manager   │  │(物理block池) │   │ │   │
│  │  │  └──────────┘  └──────────────┘   │ │   │
│  │  └────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 6.2 调度器（Scheduler）

调度器是 vLLM 的核心组件，负责管理请求的生命周期和 GPU 资源分配。

**三个关键队列**：

1. **waiting 队列**：等待处理的请求
2. **running 队列**：正在解码的请求
3. **swapped 队列**：KV cache 被换出到 CPU 的请求

**调度策略**：

```python
# 伪代码：每次迭代的调度逻辑
def schedule():
    # 1. 优先处理running队列（继续解码）
    for seq in running:
        if gpu_has_room(seq):
            allocate_new_block(seq)
        else:
            swap_to_cpu(seq)  # KV cache换出到CPU

    # 2. 尝试从swapped队列换回
    for seq in swapped:
        if gpu_has_room(seq):
            swap_from_cpu(seq)

    # 3. 从waiting队列调度新请求
    for seq in waiting:
        if gpu_has_room(seq):
            preempt_if_needed()  # 抢占低优先级序列
            add_to_running(seq)
```

### 6.3 KV Cache 内存管理

```python
# BlockManager 核心逻辑（简化版）
class BlockTableManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size     # 每个block的token数（默认16）
        self.free_blocks = list(range(num_blocks))  # 空闲block列表
        self.block_tables = {}           # seq_id -> BlockTable

    def allocate(self, seq_id, num_tokens):
        """为序列分配block"""
        num_blocks_needed = ceil(num_tokens / self.block_size)
        if len(self.free_blocks) < num_blocks_needed:
            return False  # 显存不足
        blocks = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        self.block_tables[seq_id].extend(blocks)
        return True

    def free(self, seq_id):
        """释放序列的所有block"""
        for block in self.block_tables[seq_id]:
            self.free_blocks.append(block)
        del self.block_tables[seq_id]
```

### 6.4 CUDA Graph 优化

vLLM 使用 CUDA Graph 捕获解码步骤的 GPU 操作，减少 CPU-GPU 同步开销：

1. **预热阶段**：用模拟输入执行解码步骤，捕获 CUDA Graph
2. **解码阶段**：重放 CUDA Graph，只需更新输入数据，无需重新提交 kernel

CUDA Graph 适用于批量大小固定的解码步骤，可减少约 30%~50% 的 CPU 开销。

### 6.5 推测解码（Speculative Decoding）

vLLM 支持推测解码，使用一个小型草稿模型快速生成候选 token，然后由大模型并行验证：

1. 草稿模型自回归生成 $k$ 个候选 token
2. 大模型一次前向传播验证所有候选 token
3. 接受正确的 token，拒绝错误的 token 并重新采样

期望加速比：$\text{speedup} \approx \frac{1}{1 - \alpha \cdot k}$，其中 $\alpha$ 是草稿模型的准确率。

---

## 7. 常见注意事项与最佳实践

### 7.1 显存管理

```python
# 推荐设置
llm = LLM(
    model="model-name",
    gpu_memory_utilization=0.9,    # 留10%给CUDA和其他操作
    swap_space=4,                   # 设置适当的CPU交换空间
)

# 避免设置过高
# gpu_memory_utilization=0.99  # 危险！可能导致OOM
```

### 7.2 张量并行配置

```python
# 根据模型大小选择GPU数量
# 7B模型: 1张GPU (80GB) 或 2张GPU (24GB each)
# 13B模型: 2张GPU
# 70B模型: 4-8张GPU

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    quantization="awq",  # 量化可减少GPU数量需求
)
```

### 7.3 前缀缓存使用建议

```python
# 适用场景：多个请求共享相同系统提示
llm = LLM(
    model="model-name",
    enable_prefix_caching=True,  # RAG、Agent等场景强烈推荐
)

# 不适用场景：每个请求的提示完全不同时，前缀缓存无效果
# 此时启用反而会带来额外的哈希计算开销
```

### 7.4 采样参数最佳实践

```python
# 1. 需要确定性输出时
params = SamplingParams(temperature=0)  # 贪心解码，结果可复现

# 2. 需要多样性时，避免temperature和top_p同时极端
params = SamplingParams(temperature=0.7, top_p=0.9)  # 推荐组合

# 3. 避免重复输出
params = SamplingParams(
    temperature=0.7,
    frequency_penalty=0.3,    # 轻度频率惩罚
    repetition_penalty=1.1,   # 轻度重复惩罚
)

# 4. 控制输出长度
params = SamplingParams(
    max_tokens=256,
    min_tokens=50,            # 避免过短的无意义输出
    stop=["\n\n"],            # 合理设置停止条件
)
```

### 7.5 量化模型使用

```python
# AWQ量化：4-bit量化，精度损失小
llm = LLM(model="TheBloke/Llama-2-7B-AWQ", quantization="awq")

# GPTQ量化：另一种4-bit量化方式
llm = LLM(model="TheBloke/Llama-2-7B-GPTQ", quantization="gptq")

# FP8量化：需要H100/AMD MI300等支持FP8的GPU
llm = LLM(model="model-name", quantization="fp8")
```

### 7.6 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| OOM (Out of Memory) | 显存不足 | 降低 `gpu_memory_utilization`、使用量化、减少 `max_model_len` |
| 推理速度慢 | 批次太小 | 增加 `max_num_seqs`、启用连续批处理 |
| 首次推理慢 | CUDA Graph 捕获 | 正常现象，后续推理会加速 |
| 精度下降 | 量化损失 | 使用更高精度量化或全精度模型 |
| 多GPU通信慢 | 节点内带宽不足 | 确保 GPU 之间使用 NVLink 连接 |

### 7.7 性能优化清单

1. **启用前缀缓存**：当多个请求共享前缀时
2. **使用量化**：AWQ/GPTQ 可减少 3-4 倍显存占用
3. **调整 `gpu_memory_utilization`**：在不 OOM 的前提下尽量提高
4. **增大 `max_num_seqs`**：提高并发度，增加吞吐量
5. **使用 CUDA Graph**：默认启用，减少 CPU 开销
6. **调整 block 大小**：默认 16，大多数情况下不需要修改
7. **合理设置 `max_model_len`**：避免不必要的显存预分配
