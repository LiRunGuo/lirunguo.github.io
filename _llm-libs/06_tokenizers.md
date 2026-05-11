---
title: "tokenizers 分词库"
excerpt: "BPE/WordPiece/Unigram算法、训练自定义分词器、与transformers集成"
collection: llm-libs
permalink: /llm-libs/06-tokenizers
category: core
toc: true
---


## 1. 库的简介和在LLM开发中的作用

tokenizers 是由 HuggingFace 开发的高性能分词器库，使用 Rust 编写核心逻辑，为 Python 提供快速绑定。它是 HuggingFace transformers 生态系统的分词基础设施，几乎所有 transformers 中的预训练模型都通过 tokenizers 库完成文本的编码与解码。

tokenizers 的核心价值在于：

- **极致性能**：Rust 核心使其比纯 Python 实现快 10-80 倍，适合大规模语料处理。
- **多种算法**：支持 BPE、WordPiece、Unigram 三种主流子词分词算法，覆盖了当前几乎所有 LLM 的分词方案。
- **完整流水线**：提供从预分词（PreTokenizer）、分词（Model）、后处理（PostProcessor）的完整分词流水线，高度可定制。
- **训练自定义分词器**：可以从零开始在自有语料上训练分词器，为领域特定 LLM 构建专属词表。
- **HuggingFace 生态集成**：作为 transformers 的底层分词引擎，与模型仓库（Hub）无缝集成，支持 `from_pretrained()` 一键加载。

### tokenizers 与 tiktoken 的关系

| 特性 | tokenizers | tiktoken |
|------|-----------|---------|
| 开发者 | HuggingFace | OpenAI |
| 支持算法 | BPE, WordPiece, Unigram | BPE |
| 自定义训练 | 支持 | 不支持 |
| 与 transformers 集成 | 深度集成 | 无直接集成 |
| 适用场景 | 训练/微调自定义模型 | OpenAI API 调用 |
| 语言 | Rust + Python | Rust + Python |

## 2. 安装方式

```bash
# 基础安装
pip install tokenizers

# 通常与 transformers 一起安装
pip install transformers  # transformers 会自动安装 tokenizers

# 指定版本
pip install tokenizers==0.19.1

# 使用 conda 安装
conda install -c conda-forge tokenizers
```

安装后验证：

```python
import tokenizers
print(tokenizers.__version__)  # 输出版本号
```

## 3. 核心类/函数/工具的详细说明

### 3.1 Tokenizer 类

Tokenizer 是 tokenizers 库的核心入口类，封装了完整的分词流水线。

#### from_pretrained()

从 HuggingFace Hub 加载预训练模型的分词器。

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
# 参数:
#   identifier: str - HuggingFace Hub 上的模型标识符，如 "bert-base-uncased"
#   revision: str = "main" - 模型的 Git 分支/标签/commit
# 返回:
#   Tokenizer - 加载的分词器对象
```

示例：

```python
from tokenizers import Tokenizer

# 从 Hub 加载不同模型的分词器
bert_tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
gpt2_tokenizer = Tokenizer.from_pretrained("gpt2")
llama_tokenizer = Tokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

#### from_file()

从本地 JSON 文件加载分词器。

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("path/to/tokenizer.json")
# 参数:
#   path: str - tokenizer.json 文件的路径
# 返回:
#   Tokenizer - 加载的分词器对象
```

#### save()

将分词器保存为 JSON 文件。

```python
tokenizer.save("path/to/tokenizer.json")
# 参数:
#   path: str - 保存路径
#   pretty: bool = True - 是否格式化 JSON 输出
```

### 3.2 编码与解码

#### encode()

将文本编码为 Encoding 对象。

```python
encoding = tokenizer.encode("Hello, world!")
# 参数:
#   sequence: str - 要编码的文本
#   is_pretokenized: bool = False - 是否已预分词
#   add_special_tokens: bool = True - 是否添加特殊 token
# 返回:
#   Encoding - 包含 token IDs、token 字符串、注意力掩码等信息的对象
```

Encoding 对象的关键属性：

```python
encoding = tokenizer.encode("Hello, world!")

print(encoding.ids)          # [101, 7592, 1010, 2088, 999, 102]  token ID 列表
print(encoding.tokens)       # ['[CLS]', 'hello', ',', 'world', '!', '[SEP]']  token 字符串列表
print(encoding.attention_mask)  # [1, 1, 1, 1, 1, 1]  注意力掩码
print(encoding.special_tokens_mask)  # [1, 0, 0, 0, 0, 1]  特殊 token 掩码
print(encoding.type_ids)     # [0, 0, 0, 0, 0, 0]  token 类型 ID（BERT 等模型使用）
print(encoding.offsets)      # [(0,0), (0,5), (5,6), (7,12), (12,13), (0,0)]
                              # 每个 token 在原文中的字符偏移量
```

#### decode()

将 token ID 列表解码回文本。

```python
text = tokenizer.decode([101, 7592, 1010, 2088, 999, 102])
# 参数:
#   ids: List[int] - token ID 列表
#   skip_special_tokens: bool = True - 是否跳过特殊 token
# 返回:
#   str - 解码后的文本
```

#### encode_batch() / decode_batch()

批量编码/解码，提升处理效率。

```python
# 批量编码
encodings = tokenizer.encode_batch(["Hello!", "World!"])

# 批量解码
texts = tokenizer.decode_batch([[101, 7592, 999, 102], [101, 2088, 999, 102]])

# 批量编码时可以设置并行数
tokenizer.enable_padding()
tokenizer.enable_truncation(max_length=512)
encodings = tokenizer.encode_batch(
    ["文本1", "文本2", "文本3"],
)
```

### 3.3 Tokenizer 的组件流水线

Tokenizer 的完整分词流水线由以下组件组成：

```
输入文本 → Normalizer → PreTokenizer → Model → PostProcessor → Encoding
```

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# 创建自定义分词器
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)
```

### 3.4 Normalizer（文本规范化）

Normalizer 在分词前对文本进行规范化处理。

```python
from tokenizers import normalizers

# 常用规范化器

# NFD Unicode 规范化 + 小写 + 去除变音符号
norm = normalizers.NFD() + normalizers.Lowercase() + normalizers.StripAccents()

# BERT 风格规范化（NFD + 小写 + 去除变音符号 + 去除控制字符）
norm = normalizers.BertNormalizer(
    clean_text=True,           # 去除控制字符
    handle_chinese_chars=True, # 在中文字符周围添加空格
    strip_accents=True,        # 去除变音符号
    lowercase=True             # 转小写
)

# 设置规范化器
tokenizer.normalizer = norm

# 直接使用规范化
result = norm.normalize_str("Héllo WÖRLD")
print(result)  # "hello world"
```

### 3.5 三种算法模型

#### BPE（Byte Pair Encoding）

```python
from tokenizers.models import BPE

# 创建空的 BPE 模型
bpe = BPE(unk_token="[UNK]")

# 从文件加载 BPE 模型
bpe = BPE.from_file(
    vocab="path/to/vocab.json",        # 词表文件
    merges="path/to/merges.txt",       # 合并规则文件
    unk_token="[UNK]",                 # 未知 token
)

# 带初始词表创建
bpe = BPE.from_file(
    vocab="vocab.json",
    merges="merges.txt",
    unk_token="[UNK]",
    dropout=0.1  # BPE dropout，训练时随机跳过某些合并，增强鲁棒性
)
```

#### WordPiece

```python
from tokenizers.models import WordPiece

# 创建空的 WordPiece 模型
wp = WordPiece(unk_token="[UNK]")

# 从文件加载
wp = WordPiece.from_file(
    vocab="path/to/vocab.txt",
    unk_token="[UNK]"
)

# 指定子词前缀
wp = WordPiece(unk_token="[UNK]", continuing_subword_prefix="##")
# BERT 使用 "##" 作为子词前缀，如 "play" → ["play", "##ing"]
```

#### Unigram

```python
from tokenizers.models import Unigram

# 从文件加载 Unigram 模型
unigram = Unigram.from_file("path/to/unigram.json")

# 直接创建（需要提供 token 和对数概率列表）
# Unigram 模型需要预训练的 token 及其概率
```

### 3.6 PreTokenizer（预分词器）

预分词器在 BPE/WordPiece/Unigram 之前将文本切分为更小的片段。

```python
from tokenizers.pre_tokenizers import (
    Whitespace,
    WhitespaceSplit,
    Punctuation,
    Metaspace,
    Digits,
    Sequence,
)

# 1. Whitespace: 基于空白字符分割，同时保留空白作为 token 前缀
pre = Whitespace()
result = pre.pre_tokenize_str("Hello  World!")
# [('Hello', (0, 5)), (' World', (5, 11)), ('!', (11, 12))]

# 2. WhitespaceSplit: 简单的空白分割
pre = WhitespaceSplit()
result = pre.pre_tokenize_str("Hello World!")
# [('Hello', (0, 5)), ('World!', (6, 12))]

# 3. Punctuation: 将标点符号分离
pre = Punctuation()
result = pre.pre_tokenize_str("Hello, World!")
# [('Hello', (0, 5)), (',', (5, 6)), (' World', (6, 12)), ('!', (12, 13))]

# 4. Metaspace: 用特殊字符 ▁ 替换空格（SentencePiece 风格）
pre = Metaspace(replacement="▁", add_prefix_space=True)
result = pre.pre_tokenize_str("Hello World")
# [('▁Hello', (0, 5)), ('▁World', (5, 11))]

# 5. Digits: 将数字分离
pre = Digits(individual_digits=False)
result = pre.pre_tokenize_str("price123")
# [('price', (0, 5)), ('123', (5, 8))]

# 6. 组合多个预分词器
pre = Sequence([Whitespace(), Digits(individual_digits=True)])
```

### 3.7 PostProcessor（后处理器）

后处理器在分词完成后对结果进行调整，如添加特殊 token。

#### BertProcessing

```python
from tokenizers.processors import BertProcessing

# 为 BERT 风格模型添加 [CLS] 和 [SEP]
processor = BertProcessing(
    ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ("[CLS]", tokenizer.token_to_id("[CLS]")),
)
tokenizer.post_processor = processor
```

#### TemplateProcessing

```python
from tokenizers.processors import TemplateProcessing

# 模板定义格式，支持单句和句对输入
processor = TemplateProcessing(
    single="[CLS] $A [SEP]",                        # 单句模板
    pair="[CLS] $A [SEP] $B [SEP]",                 # 句对模板
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),  # 特殊 token 映射
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)
tokenizer.post_processor = processor
```

模板语法说明：
- `$A`：第一个输入序列
- `$B`：第二个输入序列（仅句对）
- `[CLS]`, `[SEP]`：特殊 token
- `:1` 后缀：设置 segment ID，如 `$B:1` 表示 B 序列的 type_id 为 1

### 3.8 Padding 和 Truncation

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

# 启用 Padding
tokenizer.enable_padding(
    direction="right",       # 填充方向："right" 或 "left"（GPT 系列通常用 left）
    pad_id=0,                # [PAD] 的 token ID
    pad_type_id=0,           # [PAD] 的 type ID
    pad_token="[PAD]",       # [PAD] 的字符串表示
    length=None,             # 固定长度，None 表示自动按批次中最长序列填充
    pad_to_multiple_of=8,    # 填充到 8 的倍数（GPU 对齐优化）
)

# 启用 Truncation
tokenizer.enable_truncation(
    max_length=512,          # 最大长度
    stride=0,                # 截断时的滑动步长
    strategy="longest_first",# 截断策略："longest_first", "only_first", "only_second"
)

# 禁用 padding/truncation
tokenizer.no_padding()
tokenizer.no_truncation()
```

### 3.9 训练自定义分词器

#### BpeTrainer

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. 创建空 BPE 模型的分词器
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 2. 配置训练器
trainer = BpeTrainer(
    vocab_size=30000,              # 目标词表大小
    min_frequency=2,               # 最低出现频率
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    show_progress=True,            # 显示训练进度
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 初始字母表
    end_of_word_suffix="</w>",     # 词尾后缀标记
)

# 3. 训练
files = ["data/train_1.txt", "data/train_2.txt"]
tokenizer.train(files, trainer)

# 4. 保存
tokenizer.save("custom_bpe_tokenizer.json")
```

#### WordPieceTrainer

```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = WordPieceTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    continuing_subword_prefix="##",  # 子词前缀（WordPiece 特有）
    # WordPiece 使用似然比作为合并标准，而非 BPE 的频率
)

files = ["data/train.txt"]
tokenizer.train(files, trainer)
tokenizer.save("custom_wp_tokenizer.json")
```

#### UnigramTrainer

```python
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Metaspace

tokenizer = Tokenizer(Unigram())
tokenizer.pre_tokenizer = Metaspace()

trainer = UnigramTrainer(
    vocab_size=30000,
    special_tokens=["[PAD]", "[UNK]"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
    # Unigram 从大词表逐步剪枝，而非逐步合并
)

files = ["data/train.txt"]
tokenizer.train(files, trainer)
tokenizer.save("custom_unigram_tokenizer.json")
```

#### 从迭代器训练

```python
# 不需要写文件，直接从 Python 迭代器训练
def batch_iterator(batch_size=1000):
    with open("large_corpus.txt", "r") as f:
        batch = []
        for line in f:
            batch.append(line.strip())
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
    length=None,  # 总行数，用于显示进度条
)
```

## 4. 在LLM开发中的典型使用场景和代码示例

### 4.1 从 Hub 加载分词器用于推理

```python
from tokenizers import Tokenizer

# 加载预训练模型的分词器
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

# 编码文本
encoding = tokenizer.encode("Hello, how are you?")

print(f"Tokens: {encoding.tokens}")
print(f"IDs: {encoding.ids}")
print(f"Attention Mask: {encoding.attention_mask}")

# 解码
text = tokenizer.decode(encoding.ids, skip_special_tokens=True)
print(f"Decoded: {text}")
```

### 4.2 为领域特定 LLM 训练分词器

```python
from tokenizers import (
    Tokenizer,
    normalizers,
    pre_tokenizers,
    processors,
)
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# 1. 创建分词器
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. 设置规范化器（中文场景建议不过度规范化）
tokenizer.normalizer = normalizers.NFD()

# 3. 设置预分词器
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Whitespace(),
    pre_tokenizers.Punctuation(),
])

# 4. 配置训练器
trainer = BpeTrainer(
    vocab_size=32000,
    min_frequency=3,
    special_tokens=[
        "<pad>", "<s>", "</s>", "<unk>", "<mask>",
    ],
)

# 5. 在领域语料上训练
tokenizer.train(["medical_corpus.txt", "legal_corpus.txt"], trainer)

# 6. 设置后处理器（LLaMA 风格）
tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> </s> $B </s>",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>")),
    ],
)

# 7. 保存
tokenizer.save("domain_tokenizer.json")
print(f"词表大小: {tokenizer.get_vocab_size()}")
```

### 4.3 与 transformers 库配合使用

```python
from transformers import PreTrainedTokenizerFast

# 方式1：从 tokenizers 加载到 transformers
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("custom_tokenizer.json")

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<cls>",
    sep_token="<sep>",
    mask_token="<mask>",
)

# 方式2：直接从文件创建
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="custom_tokenizer.json",
)

# 现在可以像普通 transformers 分词器一样使用
encoded = fast_tokenizer("Hello, world!", return_tensors="pt")
print(encoded)
```

### 4.4 多语言分词器训练

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# ByteLevel 预分词器适合多语言场景
# 它将文本先转为字节级表示，天然支持所有语言
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

trainer = BpeTrainer(
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
    initial_alphabet=ByteLevel.alphabet(),  # 256 个基础字节
)

# 使用多语言语料训练
files = [
    "corpus/en.txt",
    "corpus/zh.txt",
    "corpus/ja.txt",
    "corpus/ko.txt",
]
tokenizer.train(files, trainer)

# 测试多语言编码
for text in ["Hello world", "你好世界", "こんにちは"]:
    encoding = tokenizer.encode(text)
    print(f"{text}: {len(encoding.ids)} tokens")
```

### 4.5 批量处理大规模数据集

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("gpt2")
tokenizer.enable_truncation(max_length=512)
tokenizer.enable_padding(length=512)

# 批量编码
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world!",
    "Machine learning is fascinating.",
]

encodings = tokenizer.encode_batch(texts)

# 提取为模型输入格式
import numpy as np

input_ids = np.array([e.ids for e in encodings])
attention_mask = np.array([e.attention_mask for e in encodings])

print(f"Input IDs shape: {input_ids.shape}")        # (3, 512)
print(f"Attention Mask shape: {attention_mask.shape}")  # (3, 512)
```

### 4.6 对齐分析（Token 与原文的映射）

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
encoding = tokenizer.encode("Hello, beautiful world!")

# 获取每个 token 在原文中的字符偏移
for token, (start, end) in zip(encoding.tokens, encoding.offsets):
    original_text = "Hello, beautiful world!"[start:end]
    print(f"Token: {token:15s} | 原文片段: '{original_text}' | 偏移: [{start}:{end}]")

# 输出示例:
# Token: [CLS]           | 原文片段: '' | 偏移: [0:0]
# Token: hello           | 原文片段: 'Hello' | 偏移: [0:5]
# Token: ,               | 原文片段: ',' | 偏移: [5:6]
# Token: beautiful       | 原文片段: 'beautiful' | 偏移: [7:16]
# Token: world           | 原文片段: 'world' | 偏移: [17:22]
# Token: !               | 原文片段: '!' | 偏移: [22:23]
# Token: [SEP]           | 原文片段: '' | 偏移: [0:0]
```

### 4.7 对比不同分词器的效率

```python
from tokenizers import Tokenizer
import time

# 加载不同分词器
tokenizers_config = {
    "BERT": "bert-base-uncased",
    "GPT-2": "gpt2",
    "RoBERTa": "roberta-base",
}

text = "Natural language processing is a subfield of linguistics and artificial intelligence."

for name, model_id in tokenizers_config.items():
    tok = Tokenizer.from_pretrained(model_id)
    encoding = tok.encode(text)
    print(f"{name:10s}: {len(encoding.ids):3d} tokens → {encoding.tokens}")
```

## 5. 数学原理

### 5.1 BPE（Byte Pair Encoding）

BPE 的原理已在 tiktoken 文档中详述，此处补充 tokenizers 中的实现细节。

tokenizers 中的 BPE 实现与 tiktoken 的主要区别：

1. **预分词**：tokenizers 通过 PreTokenizer 先将文本切分为词/子词片段，然后在每个片段内独立执行 BPE 合并。
2. **BPE Dropout**：训练时以概率 p 随机跳过某些合并操作，增加分词的多样性，提升模型鲁棒性。

```python
# BPE Dropout 示例
from tokenizers.models import BPE

bpe = BPE.from_file(
    vocab="vocab.json",
    merges="merges.txt",
    unk_token="[UNK]",
    dropout=0.1  # 训练时 10% 概率跳过合并
)
# 推理时 dropout=0.0，确保确定性
```

BPE 的合并公式（回顾）：

```
第 i 次迭代:
  (a*, b*) = argmax_{(a,b)} count(a, b)
  merge(a*, b*) → c
  V_new = V_old ∪ {c}
```

### 5.2 WordPiece：基于似然的合并策略

WordPiece 与 BPE 的关键区别在于合并策略：BPE 选择频率最高的对合并，而 WordPiece 选择使语言模型似然增益最大的对合并。

**合并评分公式**：

```
score(merge(A, B)) = P(AB) / (P(A) × P(B))

其中:
  P(A) = count(A) / total_count  (token A 的概率)
  P(B) = count(B) / total_count  (token B 的概率)
  P(AB) = count(AB) / total_count (合并后 token AB 的概率)
```

**直观理解**：
- 如果 A 和 B 经常一起出现（即 P(AB) 远大于 P(A)×P(B)），说明合并它们能更好地压缩语料。
- 相比 BPE 的纯频率统计，WordPiece 的似然比考虑了各 token 的独立概率，避免了偏向高频 token 的偏差。

**算法步骤**：

1. 初始化词表为所有字符。
2. 对于每对相邻 token (A, B)，计算 score = P(AB) / (P(A) × P(B))。
3. 选择 score 最大的对合并。
4. 重复直到词表达到预设大小。

**WordPiece 的解码方式**：

WordPiece 使用 `##` 前缀标记子词。解码时：
- 如果 token 以 `##` 开头，将其直接拼接到前一个 token 后面。
- 否则，在前面添加一个空格。

```python
# WordPiece 子词示例
# "unaffable" → ["un", "##aff", "##able"]
# 解码: "un" + "##aff" → "unaff" + "##able" → "unaffable"
```

### 5.3 Unigram：基于损失增量的剪枝策略

Unigram 模型与 BPE/WordPiece 的思路完全相反：不是逐步合并构建词表，而是从一个足够大的初始词表逐步剪枝。

**核心思想**：

1. 从一个大词表 V（如数百万个 token）出发。
2. 为每个 token 计算一个对数概率：`log P(token)`。
3. 对于输入文本，使用 Viterbi 算法找到概率最大的分词路径。
4. 逐步删除对总体损失影响最小的 token（即删除后损失增量最小的 token）。
5. 重复直到词表缩减到目标大小。

**损失函数**：

```
L(V) = -Σ log P(x_i | V)

其中:
  x_i 是训练语料中的第 i 个样本
  P(x_i | V) 是使用词表 V 对 x_i 的最优分词路径的概率
```

**剪枝公式**：

```
ΔL(token_k) = L(V \ {token_k}) - L(V)

选择删除的 token: token* = argmin_{token_k} ΔL(token_k)

即：删除对损失影响最小的 token
```

**Viterbi 分词**：

给定词表 V 和输入文本 x，Unigram 使用 Viterbi 算法找到最优分词：

```
最优分词 = argmax_{segmentation} Π P(token_i)

等价于: argmin_{segmentation} Σ -log P(token_i)
```

这是一个最短路径问题，可以用动态规划高效求解。

```python
# Unigram 分词的简化 Viterbi 实现
def unigram_tokenize(text, token_probs):
    """
    参数:
        text: 输入文本
        token_probs: dict, token → log 概率
    返回:
        list: 最优分词结果
    """
    n = len(text)
    # dp[i] = 到达位置 i 的最小负对数概率
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    backpointer = [None] * (n + 1)

    for i in range(n):
        if dp[i] == float('inf'):
            continue
        for j in range(i + 1, n + 1):
            substr = text[i:j]
            if substr in token_probs:
                cost = dp[i] - token_probs[substr]
                if cost < dp[j]:
                    dp[j] = cost
                    backpointer[j] = i

    # 回溯获取分词
    tokens = []
    pos = n
    while pos > 0:
        start = backpointer[pos]
        tokens.append(text[start:pos])
        pos = start

    return list(reversed(tokens))
```

**三种算法对比总结**：

| 特性 | BPE | WordPiece | Unigram |
|------|-----|-----------|---------|
| 构建方向 | 自底向上（合并） | 自底向上（合并） | 自顶向下（剪枝） |
| 合并/删除标准 | 频率最高 | 似然比最大 | 损失增量最小 |
| 分词方式 | 贪心合并 | 贪心合并 | Viterbi 最优 |
| 典型模型 | GPT-2, LLaMA | BERT, DistilBERT | T5, ALBERT, SentencePiece |
| 多分词支持 | 否（确定性） | 否（确定性） | 是（可采样多种分词） |

## 6. 代码原理/架构原理

### 6.1 tokenizers 的整体架构

```
tokenizers
├── Rust 核心层 (tokenizers-lib)
│   ├── tokenizer/
│   │   ├── tokenizer.rs      # 核心分词器逻辑
│   │   ├── normalizer.rs     # 文本规范化
│   │   ├── pre_tokenizer.rs  # 预分词
│   │   ├── model.rs          # 分词模型（BPE/WordPiece/Unigram）
│   │   └── post_processor.rs # 后处理
│   └── parallelism/
│       └── 多线程批量处理
├── Python 绑定层 (PyO3)
│   └── 绑定 Rust 核心到 Python
└── Python API 层
    ├── Tokenizer 类
    ├── models/ 模块
    ├── pre_tokenizers/ 模块
    ├── post_processors/ 模块
    ├── normalizers/ 模块
    └── trainers/ 模块
```

### 6.2 分词流水线详解

```
输入文本: "Hello, World!"
    │
    ▼
┌─────────────────┐
│   Normalizer    │  文本规范化（小写、NFD、去变音符号等）
│  "hello, world!"│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PreTokenizer   │  预分词（空白分割、标点分割等）
│ ["hello", ",",  │
│  " world", "!"] │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Model       │  核心分词模型（BPE/WordPiece/Unigram）
│ ["hel", "lo",   │  将每个预分词片段进一步切分为子词
│  ",", " world", │
│  "!"]           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PostProcessor  │  后处理（添加 [CLS]、[SEP] 等特殊 token）
│ ["[CLS]", "hel",│
│  "lo", ",", ... │
│  "[SEP]"]       │
└────────┬────────┘
         │
         ▼
    Encoding 对象
    (ids, tokens, offsets, ...)
```

### 6.3 Rust 性能优化的关键

1. **零拷贝字符串处理**：Rust 的字符串操作避免了不必要的内存分配和拷贝。
2. **高效的数据结构**：使用 Rust 的 HashMap 和 Vec，内存布局紧凑，缓存友好。
3. **并行化**：`encode_batch()` 使用 Rayon 库实现自动并行，充分利用多核 CPU。
4. **预分配内存**：编码时预先估计 token 数量，减少动态内存分配。

### 6.4 与 transformers 的集成架构

```
用户代码
   │
   ▼
transformers.AutoTokenizer
   │
   ▼
transformers.PreTrainedTokenizerFast
   │
   ▼ (内部持有)
tokenizers.Tokenizer (Rust 核心)
   │
   ▼
HuggingFace Hub (下载 tokenizer.json)
```

`PreTrainedTokenizerFast` 是 transformers 中快速分词器的基类，它内部持有一个 `tokenizers.Tokenizer` 实例，将所有分词操作委托给 Rust 核心执行。这种架构使得 transformers 的用户无需直接使用 tokenizers 库，但底层受益于其高性能。

### 6.5 tokenizer.json 文件格式

```json
{
  "version": "1.0",
  "truncation": {
    "direction": "Right",
    "max_length": 512,
    "strategy": "LongestFirst",
    "stride": 0
  },
  "padding": {
    "direction": "Right",
    "pad_id": 0,
    "pad_type_id": 0,
    "pad_token": "[PAD]",
    "strategy": "BatchLongest"
  },
  "normalizer": {
    "type": "BertNormalizer",
    "clean_text": true,
    "handle_chinese_chars": true,
    "strip_accents": true,
    "lowercase": true
  },
  "pre_tokenizer": {
    "type": "BertPreTokenizer"
  },
  "post_processor": {
    "type": "BertProcessing",
    "sep": ["[SEP]", 102],
    "cls": ["[CLS]", 101]
  },
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "vocab": {
      "[PAD]": 0,
      "[UNK]": 1,
      "[CLS]": 2,
      "[SEP]": 3,
      ...
    }
  }
}
```

这个 JSON 格式是 tokenizers 的标准序列化格式，包含了分词器的完整配置，使得分词器可以跨语言、跨平台复用。

## 7. 常见注意事项和最佳实践

### 7.1 选择合适的分词算法

```python
# 根据模型架构选择分词算法

# BERT 系列 → WordPiece
from tokenizers.models import WordPiece
# 优点: 处理未登录词好，子词前缀 ## 清晰
# 缺点: 合并策略偏向高频 token

# GPT 系列 → BPE
from tokenizers.models import BPE
# 优点: 实现简单，训练快速
# 缺点: 对低频子串可能过度切分

# T5 / ALBERT → Unigram
from tokenizers.models import Unigram
# 优点: 概率化分词，支持多种分词方式
# 缺点: 训练较慢，需要 Viterbi 解码
```

### 7.2 词表大小选择

```python
# 词表大小的权衡
# 太小: 编码效率低，常见词被过度切分，序列过长
# 太大: 嵌入矩阵内存占用大，低频 token 学不好

# 经验值:
# 英文模型: 30k-50k
# 多语言模型: 50k-100k（甚至 250k）
# 代码模型: 50k-100k

from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(
    vocab_size=32000,  # 根据语料和语言选择
    min_frequency=3,   # 过滤低频 token，减少噪声
)
```

### 7.3 中文分词的特殊考虑

```python
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# 中文场景的推荐配置
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 方案1：ByteLevel（推荐，天然支持中文）
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 方案2：CharacterLevel 预分词（每个字符独立）
# 适合纯中文场景

# 规范化器：中文场景不要过度规范化
# BertNormalizer 的 handle_chinese_chars=True 会在中文字符周围加空格
tokenizer.normalizer = normalizers.BertNormalizer(
    clean_text=True,
    handle_chinese_chars=True,  # 对中文很重要
    strip_accents=False,         # 保留变音符号
    lowercase=False,             # 中文不需要小写
)

trainer = BpeTrainer(
    vocab_size=50000,     # 中文通常需要更大的词表
    min_frequency=2,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
)
```

### 7.4 Padding 策略选择

```python
# 对于不同任务选择不同的 padding 策略

# 分类任务：通常右填充
tokenizer.enable_padding(direction="right", pad_to_multiple_of=8)

# 生成任务（GPT 等）：通常左填充，确保生成从最后位置开始
tokenizer.enable_padding(direction="left", pad_to_multiple_of=8)

# GPU 优化：填充到 8 的倍数，提升 Tensor Core 利用率
tokenizer.enable_padding(pad_to_multiple_of=8)
```

### 7.5 避免常见陷阱

```python
# 陷阱1：训练和推理使用不同的预分词器
# 确保训练时和推理时的预分词配置完全一致
tokenizer = Tokenizer.from_file("my_tokenizer.json")
# 不要在使用时重新设置 pre_tokenizer

# 陷阱2：特殊 token 的 ID 不连续
# 训练时指定的 special_tokens 列表顺序决定了它们的 ID
trainer = BpeTrainer(
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    # ID 分别为:     0        1        2        3        4
)

# 陷阱3：忘记设置后处理器
# 加载预训练模型时后处理器已配置好，但从零训练时容易忘记
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# 陷阱4：offsets 中特殊 token 的偏移为 (0, 0)
# 特殊 token 不对应原文中的任何字符
encoding = tokenizer.encode("Hello, world!")
for token, offset in zip(encoding.tokens, encoding.offsets):
    if offset == (0, 0):
        print(f"特殊 token: {token}")
```

### 7.6 性能优化建议

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

# 1. 批量处理优于逐条处理
# 慢：
results = [tokenizer.encode(text) for text in texts]
# 快：
results = tokenizer.encode_batch(texts)

# 2. 预先启用 padding/truncation，避免每条单独处理
tokenizer.enable_padding()
tokenizer.enable_truncation(max_length=512)

# 3. 使用 num_threads 控制并行度
tokenizer.encode_batch(texts, num_threads=4)

# 4. 对于极大量数据，使用迭代器训练而非一次性加载
def text_iterator():
    for file in large_files:
        with open(file) as f:
            for line in f:
                yield line.strip()

tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
```

### 7.7 与 SentencePiece 的关系

```python
# SentencePiece 是另一个流行的分词库（Google 开发）
# tokenizers 的 Unigram 模型兼容 SentencePiece 模型

# 从 SentencePiece 模型转换
from transformers import AutoTokenizer

# 直接加载 SentencePiece 模型
sp_tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

# tokenizers 可以处理 SentencePiece 的 Metaspace 预分词
from tokenizers.pre_tokenizers import Metaspace
pre_tokenizer = Metaspace(replacement="▁", add_prefix_space=True)
```

### 7.8 完整示例：从零训练到推理

```python
from tokenizers import (
    Tokenizer,
    normalizers,
    pre_tokenizers,
    processors,
)
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# ===== 第一步：创建和训练 =====
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.Lowercase(),
    normalizers.StripAccents(),
])
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
)

tokenizer.train(["my_corpus.txt"], trainer)

# ===== 第二步：配置后处理 =====
tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> </s> $B </s>",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>")),
    ],
)

# ===== 第三步：启用 padding/truncation =====
tokenizer.enable_padding(pad_id=0, pad_token="<pad>", pad_to_multiple_of=8)
tokenizer.enable_truncation(max_length=2048)

# ===== 第四步：保存 =====
tokenizer.save("my_tokenizer.json")

# ===== 第五步：加载和推理 =====
tokenizer = Tokenizer.from_file("my_tokenizer.json")

# 单条推理
encoding = tokenizer.encode("Hello, this is a test!")
print(f"Tokens: {encoding.tokens}")
print(f"IDs: {encoding.ids}")

# 批量推理
encodings = tokenizer.encode_batch(["Hello!", "World!"])
for enc in encodings:
    print(f"Tokens: {enc.tokens}, Length: {len(enc.ids)}")

# ===== 第六步：集成到 transformers =====
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="my_tokenizer.json",
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
)

# 可以直接用于模型训练
encoded = fast_tokenizer("Hello, world!", return_tensors="pt")
print(encoded["input_ids"].shape)
```
