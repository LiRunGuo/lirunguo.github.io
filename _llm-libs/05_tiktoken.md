---
title: "tiktoken 分词库"
excerpt: "BPE分词、encoding_for_model、token计数与成本估算、cl100k_base/p50k_base编码"
collection: llm-libs
permalink: /llm-libs/05-tiktoken
category: core
---


## 1. 库的简介和在LLM开发中的作用

tiktoken 是由 OpenAI 开发的一个快速、开源的分词器（tokenizer）库，主要用于将文本转换为整数序列（token），以及将整数序列还原为文本。它是 GPT 系列模型（GPT-3、GPT-3.5、GPT-4 等）所使用的 BPE（Byte Pair Encoding）分词算法的官方 Python 实现。

tiktoken 的核心价值在于：

- **快速高效**：相比同类的开源分词器，tiktoken 的速度通常快 3-6 倍，因为它使用 Rust 编写核心逻辑，并通过 PyO3 提供 Python 绑定。
- **官方标准**：作为 OpenAI 官方出品，tiktoken 提供的分词结果与 OpenAI API 实际使用的分词完全一致，确保了 token 计数的准确性。
- **成本估算**：在调用 OpenAI API 时，费用按 token 数量计费。使用 tiktoken 可以在调用 API 之前精确预估 token 数量和成本。
- **上下文窗口管理**：LLM 有最大上下文窗口限制（如 GPT-4 的 128K tokens），tiktoken 可帮助截断或分段输入文本，确保不超出限制。

## 2. 安装方式

```bash
# 基础安装
pip install tiktoken

# 指定版本安装
pip install tiktoken==0.7.0

# 使用 conda 安装
conda install -c conda-forge tiktoken
```

安装后验证：

```python
import tiktoken
print(tiktoken.__version__)  # 输出版本号
```

> **注意**：tiktoken 首次使用某个编码时，会从网络下载编码数据并缓存在本地。如果环境无法联网，需要提前下载好编码数据。

## 3. 核心类/函数/工具的详细说明

### 3.1 encoding_for_model()

根据模型名称获取对应的 Encoding 对象。这是最常用的函数，推荐优先使用。

```python
def encoding_for_model(model_name: str) -> Encoding:
    """
    参数:
        model_name: 模型名称字符串，如 "gpt-4", "gpt-3.5-turbo", "text-davinci-003"

    返回:
        Encoding: 该模型对应的编码器对象

    异常:
        KeyError: 如果模型名称不被支持
    """
```

示例：

```python
import tiktoken

# 获取 GPT-4 的编码器
enc = tiktoken.encoding_for_model("gpt-4")

# 获取 GPT-3.5-turbo 的编码器
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

# 获取 GPT-3 的编码器
enc = tiktoken.encoding_for_model("text-davinci-003")
```

### 3.2 get_encoding()

根据编码名称直接获取 Encoding 对象，适用于已知编码名称的场景。

```python
def get_encoding(encoding_name: str) -> Encoding:
    """
    参数:
        encoding_name: 编码名称字符串，如 "cl100k_base", "p50k_base", "r50k_base"

    返回:
        Encoding: 对应的编码器对象

    异常:
        ValueError: 如果编码名称不存在
    """
```

示例：

```python
import tiktoken

# cl100k_base: GPT-4 / GPT-3.5-turbo 使用
enc = tiktoken.get_encoding("cl100k_base")

# p50k_base: Codex 模型使用
enc = tiktoken.get_encoding("p50k_base")

# r50k_base (即 GPT-3 的编码)
enc = tiktoken.get_encoding("r50k_base")
```

### 3.3 Encoding 类

Encoding 是 tiktoken 的核心类，封装了具体的编码和解码逻辑。

#### encode()

将文本字符串编码为 token ID 列表。

```python
def encode(
    self,
    text: str,
    *,
    allowed_special: Union[Literal["all"], Set[str]] = set(),
    disallowed_special: Union[Literal["all"], Set[str]] = "all"
) -> list[int]:
    """
    参数:
        text: 要编码的文本字符串
        allowed_special: 允许编码的特殊 token 集合，默认为空集
            - 设为 "all" 表示允许所有特殊 token
            - 传入一个 set 如 {"<|endoftext|>"} 表示只允许这些特殊 token
        disallowed_special: 禁止编码的特殊 token 集合，默认为 "all"（禁止所有）
            - 设为空集 set() 表示不禁止任何特殊 token
            - 设为 "all" 表示禁止所有特殊 token

    返回:
        list[int]: token ID 列表

    异常:
        ValueError: 如果文本中包含被 disallowed_special 禁止的特殊 token
    """
```

示例：

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

# 基本编码
tokens = enc.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]

# 处理中文
tokens = enc.encode("你好，世界！")
print(tokens)  # [57668, 53901, 3922, 244, 9254, 244, 89163, 105]

# 编码包含特殊 token 的文本
# 默认情况下，encode 会拒绝包含特殊 token 的文本
text = "Hello<|endoftext|>World"
# enc.encode(text)  # 会抛出 ValueError

# 允许特定特殊 token
tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
print(tokens)

# 允许所有特殊 token
tokens = enc.encode(text, allowed_special="all")
print(tokens)
```

#### decode()

将 token ID 列表解码回文本字符串。

```python
def decode(self, tokens: list[int], errors: str = "replace") -> str:
    """
    参数:
        tokens: token ID 列表
        errors: 解码错误处理策略，默认 "replace"
            - "replace": 用替换字符替代无法解码的 token
            - "ignore": 忽略无法解码的 token
            - "strict": 遇到错误抛出异常

    返回:
        str: 解码后的文本字符串
    """
```

示例：

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

tokens = enc.encode("Hello, world!")
text = enc.decode(tokens)
print(text)  # Hello, world!

# 部分解码：解码 token 列表的一个子集
partial = enc.decode(tokens[:2])
print(partial)  # Hello,
```

#### encode_ordinary()

将文本编码为 token 列表，但将所有特殊 token 视为普通文本处理，不会抛出异常。

```python
def encode_ordinary(self, text: str) -> list[int]:
    """
    参数:
        text: 要编码的文本字符串

    返回:
        list[int]: token ID 列表

    说明:
        与 encode() 不同，此方法将特殊 token 字符串（如 <|endoftext|>）
        视为普通文本进行编码，而不是作为特殊 token 处理。
    """
```

示例：

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text = "Hello<|endoftext|>World"

# encode_ordinary 将 <|endoftext|> 当作普通文本
tokens = enc.encode_ordinary(text)
print(tokens)  # 特殊 token 会被拆分为多个普通 token

# 对比 encode，需要显式允许特殊 token
tokens_special = enc.encode(text, allowed_special={"<|endoftext|>"})
print(tokens_special)  # <|endoftext|> 会被编码为单个特殊 token ID
```

#### encode_with_special_tokens()

将文本编码为 token 列表，自动允许所有特殊 token。

```python
def encode_with_special_tokens(self, text: str) -> list[int]:
    """
    参数:
        text: 要编码的文本字符串

    返回:
        list[int]: token ID 列表

    说明:
        等价于 encode(text, allowed_special="all", disallowed_special=set())
        所有特殊 token 都会被识别并编码为对应的特殊 token ID。
    """
```

示例：

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text = "Hello<|endoftext|>World"

# 等价于 encode(text, allowed_special="all")
tokens = enc.encode_with_special_tokens(text)
print(tokens)  # <|endoftext|> 被编码为特殊 token ID
```

### 3.4 token 计数与成本估算

tiktoken 最常见的用途之一是统计 token 数量，用于 API 调用前的成本估算。

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """统计文本在指定模型下的 token 数量"""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)

def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gpt-4"
) -> float:
    """估算 API 调用成本（美元），价格可能随时间变化，请以官网为准"""
    pricing = {
        "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
        "gpt-4-turbo": {"prompt": 0.01 / 1000, "completion": 0.03 / 1000},
        "gpt-3.5-turbo": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000},
    }
    rates = pricing.get(model, pricing["gpt-4"])
    cost = (prompt_tokens * rates["prompt"] +
            completion_tokens * rates["completion"])
    return cost

# 使用示例
text = "请帮我写一篇关于人工智能发展的文章，不少于500字。"
token_count = count_tokens(text, model="gpt-4")
print(f"Token 数量: {token_count}")

cost = estimate_cost(token_count, 500, model="gpt-4")
print(f"预估成本: ${cost:.4f}")
```

### 3.5 上下文窗口管理

```python
import tiktoken

def truncate_text_to_token_limit(
    text: str,
    model: str = "gpt-4",
    max_tokens: int = 4096,
    truncation_strategy: str = "end"
) -> str:
    """
    将文本截断到指定的 token 限制内

    参数:
        text: 输入文本
        model: 模型名称
        max_tokens: 最大 token 数量
        truncation_strategy: 截断策略
            - "end": 保留开头，截断末尾
            - "start": 保留末尾，截断开头
            - "middle": 保留开头和末尾，截断中间

    返回:
        str: 截断后的文本
    """
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)

    if len(tokens) <= max_tokens:
        return text

    if truncation_strategy == "end":
        truncated = tokens[:max_tokens]
    elif truncation_strategy == "start":
        truncated = tokens[-max_tokens:]
    elif truncation_strategy == "middle":
        half = max_tokens // 2
        truncated = tokens[:half] + tokens[-(max_tokens - half):]
    else:
        raise ValueError(f"未知截断策略: {truncation_strategy}")

    return enc.decode(truncated)

# 使用示例
long_text = "这是一段很长的文本..." * 1000
result = truncate_text_to_token_limit(long_text, model="gpt-4", max_tokens=4096)
print(f"截断后 token 数: {len(tiktoken.encoding_for_model('gpt-4').encode(result))}")
```

### 3.6 不同编码对照

| 编码名称 | 适用模型 | 词表大小 | 说明 |
|---------|---------|---------|------|
| `cl100k_base` | GPT-4, GPT-3.5-turbo, text-embedding-ada-002 | ~100k | 目前最常用的编码 |
| `p50k_base` | Codex (code-davinci-002, code-cushman-001) | ~50k | 主要用于代码模型 |
| `r50k_base` | GPT-3 (davinci, curie, babbage, ada) | ~50k | 旧版 GPT-3 编码 |
| `o200k_base` | GPT-4o | ~200k | GPT-4o 使用的新编码 |

查看编码支持的模型：

```python
import tiktoken

# 列出所有可用编码
print(tiktoken.list_encoding_names())
# 输出: ['r50k_base', 'p50k_base', 'cl100k_base', 'o200k_base']

# 查看某编码的词表大小
enc = tiktoken.get_encoding("cl100k_base")
print(f"词表大小: {enc.n_vocab}")  # 100277
```

## 4. 在LLM开发中的典型使用场景和代码示例

### 4.1 API 调用前的 token 预估

```python
import tiktoken

def prepare_api_call(system_prompt: str, user_message: str, model: str = "gpt-4"):
    """在调用 API 前预估 token 使用量"""
    enc = tiktoken.encoding_for_model(model)

    system_tokens = len(enc.encode(system_prompt))
    user_tokens = len(enc.encode(user_message))
    total_input_tokens = system_tokens + user_tokens

    # 预留输出空间
    max_context = {"gpt-4": 8192, "gpt-4-32k": 32768, "gpt-3.5-turbo": 4096}
    context_limit = max_context.get(model, 4096)
    max_output_tokens = context_limit - total_input_tokens

    print(f"系统提示 tokens: {system_tokens}")
    print(f"用户消息 tokens: {user_tokens}")
    print(f"总输入 tokens: {total_input_tokens}")
    print(f"最大可用输出 tokens: {max_output_tokens}")

    if total_input_tokens > context_limit:
        print(f"警告: 输入超过上下文窗口限制 ({context_limit})!")

    return {
        "total_input_tokens": total_input_tokens,
        "max_output_tokens": max_output_tokens
    }

# 使用示例
prepare_api_call(
    system_prompt="你是一个专业的Python编程助手。",
    user_message="请解释Python中的装饰器模式，并给出3个实际使用例子。",
    model="gpt-4"
)
```

### 4.2 长文本分块处理

在处理超长文档时，需要将文本分成多个 chunk，每个 chunk 不超过模型的上下文窗口。

```python
import tiktoken
from typing import List

def split_text_into_chunks(
    text: str,
    model: str = "gpt-4",
    chunk_size: int = 3000,
    overlap: int = 200
) -> List[str]:
    """
    将长文本按 token 数量分块

    参数:
        text: 要分块的文本
        model: 模型名称
        chunk_size: 每个块的最大 token 数量
        overlap: 相邻块之间的重叠 token 数量，保证上下文连续性

    返回:
        List[str]: 分块后的文本列表
    """
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    total_tokens = len(tokens)

    chunks = []
    start = 0
    while start < total_tokens:
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap  # 向前滑动，保留重叠部分

    return chunks

# 使用示例
with open("long_document.txt", "r", encoding="utf-8") as f:
    long_text = f.read()

chunks = split_text_into_chunks(long_text, chunk_size=3000, overlap=200)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {len(tiktoken.encoding_for_model('gpt-4').encode(chunk))} tokens")
```

### 4.3 流式输出的 token 追踪

```python
import tiktoken

class TokenCounter:
    """流式输出中追踪 token 数量"""

    def __init__(self, model: str = "gpt-4"):
        self.enc = tiktoken.encoding_for_model(model)
        self.total_tokens = 0
        self.buffer = ""

    def add_chunk(self, text_chunk: str):
        """添加一个流式文本片段"""
        self.buffer += text_chunk

    def get_token_count(self) -> int:
        """获取当前总 token 数"""
        return len(self.enc.encode(self.buffer))

    def finalize(self) -> int:
        """最终确认 token 数量"""
        self.total_tokens = self.get_token_count()
        return self.total_tokens

# 使用示例（模拟流式输出）
counter = TokenCounter(model="gpt-4")
for chunk in ["你好", "，我", "是一个", "AI助手"]:
    counter.add_chunk(chunk)

total = counter.finalize()
print(f"总输出 tokens: {total}")
```

### 4.4 对话历史管理

```python
import tiktoken
from typing import List, Dict

def manage_conversation_history(
    messages: List[Dict[str, str]],
    model: str = "gpt-4",
    max_tokens: int = 4096,
    reserved_for_output: int = 1000
) -> List[Dict[str, str]]:
    """
    管理对话历史，确保总 token 数不超过限制

    参数:
        messages: OpenAI API 格式的消息列表
        model: 模型名称
        max_tokens: 模型上下文窗口大小
        reserved_for_output: 为输出预留的 token 数量

    返回:
        List[Dict[str, str]]: 截断后的消息列表
    """
    enc = tiktoken.encoding_for_model(model)
    input_limit = max_tokens - reserved_for_output

    # 计算每条消息的 token 数（包括消息格式的额外 token）
    def message_tokens(msg: Dict[str, str]) -> int:
        # 每条消息约有 4-5 个格式 token
        return len(enc.encode(msg["content"])) + 5

    # 从最早的消息开始删除，直到满足限制
    total = sum(message_tokens(m) for m in messages)
    while total > input_limit and len(messages) > 1:
        # 保留系统消息，删除最早的对话消息
        removed = messages.pop(1 if messages[0]["role"] == "system" else 0)
        total -= message_tokens(removed)

    return messages

# 使用示例
messages = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
    {"role": "user", "content": "请解释量子计算"},
]

managed = manage_conversation_history(messages, max_tokens=4096, reserved_for_output=1000)
print(f"保留 {len(managed)} 条消息")
```

## 5. 数学原理：BPE（Byte Pair Encoding）算法详解

BPE 是 tiktoken 使用的核心分词算法，最初由 Sennrich 等人在 2016 年引入 NLP 领域。其核心思想是通过迭代合并最高频的相邻 token 对来构建词表。

### 5.1 算法步骤

**步骤一：初始化**

将输入文本中的每个字节（byte）作为一个初始 token。对于 UTF-8 编码的文本，每个字节对应 0-255 中的一个值。

```
初始词表: V₀ = {b0, b1, b2, ..., b255}  (共 256 个基础 token)
```

**步骤二：迭代合并**

统计当前词表中所有相邻 token 对的频率，选择频率最高的对合并为新 token。

```
第 i 次迭代:
  1. 统计所有相邻 token 对 (a, b) 的频率 f(a, b)
  2. 找到频率最高的对: (a*, b*) = argmax f(a, b)
  3. 创建新 token c = merge(a*, b*)，加入词表: Vᵢ = Vᵢ₋₁ ∪ {c}
  4. 在所有序列中将 (a*, b*) 替换为 c
```

**步骤三：停止条件**

当词表大小达到预设值时停止迭代。

```
停止条件: |Vₙ| = 预设词表大小
```

### 5.2 合并公式

```
merge(a, b) → c

其中:
  a, b 是当前词表中的 token
  c 是新创建的 token
  选择标准: c = merge(a*, b*)，其中 (a*, b*) = argmax f(a, b)
```

### 5.3 具体示例

假设训练语料为: `low lower lowest`（简化展示，实际以字节为单位）

**初始化**（以字符为单位）:
```
词表: {l, o, w, e, r, s, t, _}
序列: [l, o, w] [l, o, w, e, r] [l, o, w, e, s, t]
```

**第 1 次迭代**:
- 相邻对频率: (l,o):3, (o,w):3, (w,e):2, (e,r):1, (e,s):1, (s,t):1
- 最高频对: (l,o) 或 (o,w)，频率均为 3
- 假设选择 (l,o) → 合并为 `lo`
```
词表: {l, o, w, e, r, s, t, _, lo}
序列: [lo, w] [lo, w, e, r] [lo, w, e, s, t]
```

**第 2 次迭代**:
- 相邻对频率: (lo,w):3, (w,e):2, (e,r):1, (e,s):1, (s,t):1
- 最高频对: (lo,w)，频率 3 → 合并为 `low`
```
词表: {l, o, w, e, r, s, t, _, lo, low}
序列: [low] [low, e, r] [low, e, s, t]
```

**继续迭代**，直到词表达到预设大小。

### 5.4 编码过程

在分词时（推理阶段），BPE 按照训练时学到的合并规则顺序，依次尝试合并相邻 token：

```python
def bpe_encode(text, merge_rules):
    """
    BPE 编码的简化实现

    参数:
        text: 输入文本
        merge_rules: 训练得到的有序合并规则列表

    返回:
        list: token 序列
    """
    # 步骤1: 将文本转为字节序列
    tokens = list(text.encode('utf-8'))

    # 步骤2: 按优先级顺序应用合并规则
    for (a, b), c in merge_rules:
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                new_tokens.append(c)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens
```

### 5.5 BPE 的数学性质

1. **确定性**：给定合并规则序列，BPE 的编码结果是确定的（贪婪合并）。
2. **可逆性**：解码是编码的逆过程，因为每个合并操作都是可逆的。
3. **频率偏好**：高频出现的子串会被优先合并为独立 token，使得常用词/子词的编码更短。
4. **OOV 免疫**：由于基础词表覆盖所有字节值，任何输入文本都可以被编码，不会出现未知 token。

## 6. 代码原理/架构原理

### 6.1 tiktoken 的整体架构

```
tiktoken
├── 核心层 (Rust 实现)
│   ├── BPE 编码引擎
│   ├── 合并规则加载与缓存
│   └── 高性能字节处理
├── Python 绑定层 (PyO3)
│   ├── Encoding 类
│   └── 编码名称映射
└── Python API 层
    ├── encoding_for_model()
    ├── get_encoding()
    └── 辅助函数
```

### 6.2 核心设计决策

**1. 基于 Rust 的高性能实现**

tiktoken 的 BPE 编码核心用 Rust 实现，通过 PyO3 框架提供 Python 接口。这使得 tiktoken 在处理大规模文本时具有显著的性能优势。

```
Python 调用 → PyO3 绑定 → Rust BPE 核心 → 返回结果
```

**2. 惰性加载与缓存**

编码数据（合并规则、词表映射）在首次使用时从远程下载，然后缓存在本地磁盘。后续使用直接从缓存加载，避免重复下载。

```python
# tiktoken 的缓存机制（简化）
import tiktoken_ext

# 首次调用时下载并缓存
enc = tiktoken.get_encoding("cl100k_base")  # 可能需要网络

# 后续调用从缓存加载
enc = tiktoken.get_encoding("cl100k_base")  # 从本地缓存
```

**3. 特殊 Token 处理策略**

tiktoken 对特殊 token 有严格的处理策略：
- `encode()` 默认禁止特殊 token，防止用户输入注入特殊控制 token
- `encode_ordinary()` 将特殊 token 视为普通文本
- `encode_with_special_tokens()` 显式允许所有特殊 token

这种设计是安全考虑：防止 prompt injection 攻击中用户输入被误认为特殊控制 token。

### 6.3 编码数据格式

tiktoken 的编码数据包含两个核心部分：

```python
# 简化的编码数据结构
{
    "pat_str": r"""'(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,5}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
    # 预分词正则表达式，先将文本拆分为粗粒度片段

    "special_tokens": {
        "<|endoftext|>": 100257,
        "<|fim_prefix|>": 100258,
        ...
    },
    # 特殊 token 映射

    "bpe_ranks": {
        (b'\x61', b'\x20'): 0,   # a + 空格 → 优先级 0
        ...
    }
    # BPE 合并规则及其优先级
}
```

**预分词正则** 是 tiktoken 性能优化的关键之一：先通过正则将文本拆分为词/子词级别的片段（如英文单词、数字、标点等），然后在每个片段内独立执行 BPE。这避免了跨词边界的无效合并。

### 6.4 与 OpenAI API 的关系

```
用户文本 → tiktoken.encode() → token ID 列表 → OpenAI API
                                                      ↓
用户文本 ← tiktoken.decode() ← token ID 列表 ← OpenAI API 响应
```

tiktoken 确保客户端和 API 服务端使用完全相同的分词规则，从而保证：
- token 计数完全准确
- 成本估算完全可靠
- 上下文窗口检查完全可信

## 7. 常见注意事项和最佳实践

### 7.1 特殊 Token 安全处理

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

# 错误：默认 encode 会拒绝包含特殊 token 的文本
try:
    enc.encode("Hello<|endoftext|>World")
except ValueError as e:
    print(f"错误: {e}")  # 文本中包含不允许的特殊 token

# 正确方式1：如果需要处理特殊 token，显式声明
tokens = enc.encode("Hello<|endoftext|>World", allowed_special={"<|endoftext|>"})

# 正确方式2：将特殊 token 视为普通文本
tokens = enc.encode_ordinary("Hello<|endoftext|>World")

# 安全建议：处理用户输入时，始终使用默认的 encode() 或 encode_ordinary()
# 不要对用户输入使用 allowed_special="all"，防止注入攻击
```

### 7.2 编码一致性

```python
import tiktoken

# 注意：不同模型使用不同编码，token 数量可能不同
text = "Hello, 世界！"

enc_gpt4 = tiktoken.encoding_for_model("gpt-4")
enc_gpt3 = tiktoken.encoding_for_model("text-davinci-003")

print(f"GPT-4 tokens: {len(enc_gpt4.encode(text))}")
print(f"GPT-3 tokens: {len(enc_gpt3.encode(text))}")
# 结果可能不同，因为使用了不同的 BPE 编码
```

### 7.3 中文文本的 token 效率

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

# 中文字符通常消耗更多 token
english = "Hello world"
chinese = "你好世界"

print(f"英文 '{english}': {len(enc.encode(english))} tokens")
print(f"中文 '{chinese}': {len(enc.encode(chinese))} tokens")
# 中文通常 1 个字 ≈ 1-2 个 token，英文 1 个词 ≈ 1-1.5 个 token
# 因此中文内容的 token 成本通常是同等含义英文的 2-3 倍
```

### 7.4 离线环境使用

```python
import tiktoken

# tiktoken 首次使用时会从网络下载编码数据
# 在离线环境中，可以提前下载并设置缓存路径

import os
os.environ["TIKTOKEN_CACHE_DIR"] = "/path/to/cache"

# 或者手动下载编码数据到缓存目录
# 缓存位置默认为: ~/.cache/tiktoken/
# 也可以通过环境变量 TIKTOKEN_CACHE_DIR 指定

# 使用本地文件直接创建 Encoding（高级用法）
from tiktoken import Encoding
enc = Encoding(
    name="cl100k_base",
    pat_str=r"""'(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,5}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
    mergeable_ranks={...},  # 合并规则字典
    special_tokens={...},   # 特殊 token 字典
)
```

### 7.5 性能优化最佳实践

```python
import tiktoken

# 最佳实践1：复用 Encoding 对象，避免重复创建
enc = tiktoken.encoding_for_model("gpt-4")

texts = ["文本1", "文本2", "文本3"]
for text in texts:
    tokens = enc.encode(text)  # 复用同一个 enc 对象
    print(len(tokens))

# 最佳实践2：批量计数时使用列表推导
token_counts = [len(enc.encode(t)) for t in texts]

# 最佳实践3：对于只需要计数的场景，encode 后取长度即可
# 不需要 decode 时避免不必要的解码操作
```

### 7.6 常见陷阱

1. **模型名称拼写错误**：使用 `encoding_for_model()` 时确保模型名称与 OpenAI API 使用的名称完全一致。
2. **忽略消息格式的额外 token**：OpenAI API 的每条消息都会额外消耗约 4-5 个格式 token，计算总 token 时需要考虑。
3. **截断导致的文本不完整**：使用 `decode()` 解码截断后的 token 列表时，可能在多字节字符的中间截断，导致乱码。tiktoken 的 `decode()` 使用 `errors="replace"` 来处理这种情况。
4. **不同编码模型混用**：确保为正确的模型选择正确的编码，不要混用。

```python
# 陷阱示例：截断可能导致不完整的 UTF-8 字符
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("你好世界")
# 如果在 "你好" 和 "世界" 之间截断
partial_text = enc.decode(tokens[:3])  # 可能产生不完整的字符
print(partial_text)  # 输出可能包含替换字符 �
```

### 7.7 与其他库配合使用

```python
import tiktoken
from openai import OpenAI

client = OpenAI()

def smart_chat_completion(messages, model="gpt-4", max_tokens=4096):
    """带 token 检查的 API 调用封装"""
    enc = tiktoken.encoding_for_model(model)

    # 预估输入 token 数
    input_tokens = sum(
        len(enc.encode(m["content"])) + 5  # +5 为消息格式额外 token
        for m in messages
    )

    if input_tokens > max_tokens * 0.8:
        print(f"警告: 输入 token 数 ({input_tokens}) 已超过上下文窗口的 80%")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return response

# 使用
response = smart_chat_completion([
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "解释量子计算的基本原理。"},
])
```
