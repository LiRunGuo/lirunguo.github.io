---
title: "LlamaIndex RAG框架"
excerpt: "Index/Retriever/QueryEngine、RAG管道抽象、Document/Node、Agent工具调用"
collection: llm-libs
permalink: /llm-libs/22-llamaindex
category: agent
toc: true
---


## 1. 库的简介和在LLM开发中的作用

LlamaIndex（原名 GPT Index）是一个用于构建基于大语言模型（LLM）应用的"数据框架"。它的核心使命是**将私有数据与LLM连接起来**，使得LLM能够基于外部数据生成回答，而不仅仅依赖训练时的知识。

在LLM开发中，LlamaIndex 主要用于构建 **RAG（Retrieval-Augmented Generation，检索增强生成）** 系统。它提供了从数据摄取、索引构建、检索到响应合成的完整管道抽象，开发者无需从零实现各个组件，只需组合配置即可搭建生产级RAG应用。

**核心价值：**
- 统一的数据连接层：支持 160+ 种数据源的加载器
- 灵活的索引抽象：向量索引、关键词索引、摘要索引等多种检索策略
- 可组合的查询管道：检索器、响应合成器、查询引擎可自由组合
- Agent 支持：内置 ReAct、Function Calling 等 Agent 模式

```bash
pip install llama-index
```

## 2. 安装方式

```bash
# 基础安装
pip install llama-index

# 安装特定集成（示例）
pip install llama-index-llms-openai        # OpenAI LLM
pip install llama-index-embeddings-huggingface  # HuggingFace 嵌入
pip install llama-index-vector-stores-chroma    # Chroma 向量存储
pip install llama-index-readers-file           # 文件读取器

# 从源码安装
pip install git+https://github.com/run-llama/llama_index.git
```

> **注意**：LlamaIndex v0.10+ 采用了模块化架构，各集成包独立安装，不再将所有依赖打包在一起。

## 3. 核心类/函数/工具的详细说明

### 3.1 数据加载：SimpleDirectoryReader 与各种 Reader

#### SimpleDirectoryReader

`SimpleDirectoryReader` 是最常用的数据加载工具，可以从目录中批量读取文件。

```python
from llama_index.core import SimpleDirectoryReader

# 基本用法：从目录加载所有支持文件
reader = SimpleDirectoryReader(input_dir="./data")
documents = reader.load_data()
print(f"加载了 {len(documents)} 个文档")

# 指定文件列表
reader = SimpleDirectoryReader(input_files=["./data/report.pdf", "./data/notes.txt"])
documents = reader.load_data()

# 递归读取子目录
reader = SimpleDirectoryReader(input_dir="./data", recursive=True, filename_as_id=True)
documents = reader.load_data()

# 排除特定文件
reader = SimpleDirectoryReader(
    input_dir="./data",
    exclude=["*.tmp", "*.log"],  # 排除模式
    required_exts=[".pdf", ".txt", ".md"],  # 只包含指定扩展名
)
documents = reader.load_data()
```

**关键参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `input_dir` | str | 输入目录路径 |
| `input_files` | List[str] | 指定文件列表 |
| `recursive` | bool | 是否递归读取子目录，默认 False |
| `filename_as_id` | bool | 使用文件名作为文档ID，默认 False |
| `exclude` | List[str] | 排除的文件模式 |
| `required_exts` | List[str] | 要求的文件扩展名 |
| `num_files_limit` | int | 最多加载文件数 |

#### 其他 Reader

LlamaIndex 提供了大量专门的 Reader，位于 `llama-index-readers-*` 包中：

```python
# PDF 读取器
from llama_index.readers.file import PDFReader

# Web 页面读取器
from llama_index.readers.web import SimpleWebPageReader
reader = SimpleWebPageReader()
docs = reader.load_data(["https://example.com/article"])

# 数据库读取器（需安装 llama-index-readers-database）
# from llama_index.readers.database import DatabaseReader
# reader = DatabaseReader(uri="sqlite:///mydb.sqlite")
# docs = reader.load_data(query="SELECT content FROM articles")
```

### 3.2 Document 和 Node

#### Document

`Document` 是 LlamaIndex 中最基础的数据容器，代表一个完整的源文档。

```python
from llama_index.core import Document

# 创建文档
doc = Document(
    text="这是一段示例文本，用于演示Document的创建。",
    metadata={
        "filename": "example.txt",
        "author": "张三",
        "category": "技术文档",
    },
    metadata_template="{key}: {value}",  # 元数据格式化模板
    excluded_llm_metadata_keys=["author"],  # LLM不看到的元数据字段
    excluded_embed_metadata_keys=["filename"],  # 嵌入不包含的元数据字段
)

# 文档属性
print(doc.text)         # 文本内容
print(doc.metadata)     # 元数据字典
print(doc.doc_id)       # 文档唯一ID
print(doc.hash)         # 内容哈希
```

**关键参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `text` | str | 文档文本内容 |
| `metadata` | dict | 元数据键值对 |
| `doc_id` | str | 自定义文档ID |
| `excluded_llm_metadata_keys` | List[str] | 发送给LLM时排除的元数据 |
| `excluded_embed_metadata_keys` | List[str] | 嵌入时排除的元数据 |

#### Node

`Node` 是 Document 的子单元，是 LlamaIndex 索引和检索的最小单位。一个 Document 被切分成多个 Node。

```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

# 使用文本切分器将 Document 转为 Node
doc = Document(text="这是第一句话。这是第二句话。这是第三句话。", metadata={"source": "demo"})

parser = SentenceSplitter(
    chunk_size=1024,      # 每个Node最大字符数
    chunk_overlap=200,    # 相邻Node重叠字符数
)
nodes = parser.get_nodes_from_documents([doc])

for node in nodes:
    print(f"Node ID: {node.node_id}")
    print(f"文本: {node.text}")
    print(f"元数据: {node.metadata}")
    print(f"关系: {node.relationships}")  # 与其他Node的关系
```

**Node 的 Relationships：**
- `NodeRelationship.SOURCE`：指向源 Document
- `NodeRelationship.PREVIOUS`：前一个 Node
- `NodeRelationship.NEXT`：后一个 Node
- `NodeRelationship.PARENT`：父 Node（层级切分时）
- `NodeRelationship.CHILD`：子 Node 列表

**常用切分器：**
```python
from llama_index.core.node_parser import (
    SentenceSplitter,       # 按句子切分
    TokenTextSplitter,      # 按Token切分
    HierarchicalNodeParser, # 层级切分（父子关系）
    SentenceWindowNodeParser,  # 句子窗口切分
    SemanticSplitterNodeParser, # 语义切分
)

# 语义切分器（需嵌入模型）
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
semantic_parser = SemanticSplitterNodeParser(
    buffer_size=1,           # 缓冲句子数
    breakpoint_percentile_threshold=95,  # 语义断点阈值
    embed_model=embed_model,
)
```

### 3.3 索引（Index）

#### VectorStoreIndex

最常用的索引类型，基于向量相似度检索。

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# 方式1：从文档直接构建
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 方式2：从Node构建
nodes = parser.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)

# 方式3：使用持久化向量存储
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# 方式4：增量插入
index.insert(doc=Document(text="新文档内容"))
```

**构建参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `nodes` | List[Node] | 节点列表 |
| `storage_context` | StorageContext | 存储上下文 |
| `show_progress` | bool | 显示构建进度 |
| `insert_batch_size` | int | 插入批次大小，默认 2048 |

#### SummaryIndex

存储所有文档的摘要，检索时返回全部节点（适用于需要全局信息的场景）。

```python
from llama_index.core import SummaryIndex

index = SummaryIndex.from_documents(documents)
```

#### KeywordTableIndex

基于关键词表（倒排索引）的检索，从每个Node中提取关键词并建立映射。

```python
from llama_index.core import KeywordTableIndex

index = KeywordTableIndex.from_documents(documents)

# 使用检索
retriever = index.as_retriever(retriever_mode="simple")
# retriever_mode: "simple"（精确匹配）, "rake"（RAKE算法提取关键词）
```

**索引对比：**
| 索引类型 | 检索方式 | 适用场景 |
|---------|---------|---------|
| VectorStoreIndex | 向量相似度 | 语义搜索、相似文档查找 |
| SummaryIndex | 遍历所有节点 | 摘要生成、全局信息提取 |
| KeywordTableIndex | 关键词匹配 | 精确关键词检索 |

### 3.4 检索器（Retriever）

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

# 基本检索器
retriever = index.as_retriever(
    similarity_top_k=5,  # 返回最相似的top-k个节点
)

# 执行检索
nodes = retriever.retrieve("什么是RAG？")
for node_with_score in nodes:
    print(f"分数: {node_with_score.score:.4f}")
    print(f"文本: {node_with_score.node.text[:100]}")
    print(f"元数据: {node_with_score.node.metadata}")
```

**混合检索（Hybrid Search）：**

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever

# 向量检索器
vector_retriever = index.as_retriever(similarity_top_k=5)

# 关键词检索器（需支持BM25的向量存储）
# 以 Qdrant 为例：
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# hybrid_retriever = index.as_retriever(
#     vector_store_query_mode="hybrid",
#     similarity_top_k=5,
#     sparse_top_k=5,
#     alpha=0.5,  # 向量与稀疏检索的权重，0=纯稀疏，1=纯向量
# )

# QueryFusion：融合多个检索器结果
fusion_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever],  # 可添加多个检索器
    num_queries=1,        # 生成额外查询的数量（Query Expansion）
    similarity_top_k=5,
    mode="reciprocal_rerank",  # 融合模式：reciprocal_rerank 或 dist_based
)
```

### 3.5 查询引擎（QueryEngine）

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine(
    similarity_top_k=3,              # 检索top-k节点
    response_mode="tree_summarize",   # 响应合成模式
    streaming=False,                  # 是否流式输出
)

# 执行查询
response = query_engine.query("什么是RAG？")
print(response)                    # 响应文本
print(response.source_nodes)       # 来源节点
print(response.metadata)           # 元数据
```

**response_mode 选项：**
| 模式 | 说明 |
|------|------|
| `refine` | 逐个节点迭代精炼答案（默认） |
| `compact` | 尽量合并节点后精炼，减少LLM调用 |
| `tree_summarize` | 层级式总结，将多个节点层层合成 |
| `simple_summarize` | 一次性截断合并所有节点 |
| `no_text` | 只返回检索节点，不调用LLM |
| `accumulate` | 分别对每个节点生成答案，拼接返回 |
| `compact_accumulate` | compact + accumulate |

**流式响应：**

```python
query_engine = index.as_query_engine(streaming=True, similarity_top_k=3)
response = query_engine.query("解释LlamaIndex的架构")

# 方式1：逐token打印
response.print_response_stream()

# 方式2：迭代处理
for text in response.response_gen:
    print(text, end="", flush=True)
```

### 3.6 响应合成：ResponseSynthesizer

```python
from llama_index.core import get_response_synthesizer
from llama_index.core import VectorStoreIndex, PromptTemplate

index = VectorStoreIndex.from_documents(documents)

# 创建响应合成器
synthesizer = get_response_synthesizer(
    response_mode="compact",       # 合成模式
    streaming=True,                # 流式输出
    # 自定义QA提示模板
    text_qa_template=PromptTemplate(
        "上下文信息如下：\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "根据上下文信息（不依赖先验知识），回答以下问题。\n"
        "问题: {query_str}\n"
        "回答: "
    ),
)

# 与查询引擎配合
query_engine = index.as_query_engine(response_synthesizer=synthesizer)
```

**Refine 模式原理：**
1. 用第一个节点 + 问题生成初始答案
2. 用后续节点 + 前一轮答案 + 问题进行精炼
3. 重复直到所有节点处理完毕

**Compact 模式原理：**
1. 先尽量将多个节点拼接到一个上下文窗口中
2. 对拼接后的上下文执行 refine 流程
3. 减少LLM调用次数，更高效

### 3.7 Agent

#### ReActAgent

基于 ReAct（Reasoning + Acting）范式的智能体。

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# 定义工具函数
def multiply(a: int, b: int) -> int:
    """将两个整数相乘并返回结果。"""
    return a * b

def add(a: int, b: int) -> int:
    """将两个整数相加并返回结果。"""
    return a + b

# 包装为 LlamaIndex 工具
multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

# 创建 Agent
llm = OpenAI(model="gpt-4o")
agent = ReActAgent.from_tools(
    [multiply_tool, add_tool],
    llm=llm,
    verbose=True,  # 打印推理过程
)

# 执行任务
response = agent.chat("3加5等于多少？然后把结果乘以2")
print(response)
```

#### FunctionCallingAgent

利用 LLM 原生 Function Calling 能力的 Agent（仅支持兼容 OpenAI Function Calling 的模型）。

```python
from llama_index.core.agent import FunctionCallingAgent

agent = FunctionCallingAgent.from_tools(
    [multiply_tool, add_tool],
    llm=llm,
    verbose=True,
)

response = agent.chat("计算 (10 + 20) * 3")
```

#### 查询引擎作为工具

```python
from llama_index.core.tools import QueryEngineTool

# 将查询引擎包装为工具
query_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="knowledge_base",
    description="包含公司内部技术文档的知识库，可用于查询技术相关问题和公司政策。",
)

agent = ReActAgent.from_tools([query_tool, add_tool], llm=llm, verbose=True)
response = agent.chat("查阅公司文档中关于API限流的规定，然后计算限流阈值的2倍是多少")
```

### 3.8 嵌入模型

#### HuggingFaceEmbedding

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# 使用本地 HuggingFace 模型
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5",  # 中文嵌入模型
    max_length=512,                        # 最大序列长度
    cache_folder="./model_cache",          # 模型缓存目录
    embed_batch_size=10,                   # 批量嵌入大小
)

# 设置为全局默认
Settings.embed_model = embed_model

# 直接使用
embeddings = embed_model.get_text_embedding("Hello, world!")
print(f"嵌入维度: {len(embeddings)}")  # 通常为 384 或 768

# 批量嵌入
embeddings = embed_model.get_text_embedding_batch(
    ["第一句话", "第二句话", "第三句话"]
)
```

#### OpenAIEmbedding

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",  # 嵌入模型名称
    dimensions=1536,                 # 输出维度（text-embedding-3 支持自定义）
    api_key="sk-...",               # API Key，也可通过环境变量设置
)

Settings.embed_model = embed_model
```

### 3.9 LLM 配置：Settings

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 全局配置 LLM
Settings.llm = OpenAI(
    model="gpt-4o",
    temperature=0.1,      # 低温度使输出更确定性
    max_tokens=1024,      # 最大生成token数
    system_prompt="你是一个专业的技术助手，用中文回答问题。",  # 系统提示
)

# 全局配置嵌入模型
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 全局配置切分参数
Settings.chunk_size = 1024       # Node 最大大小
Settings.chunk_overlap = 200     # Node 重叠大小

# 全局配置回调
from llama_index.core.callbacks import TokenCountingHandler
token_counter = TokenCountingHandler()
Settings.callback_manager.add_handler(token_counter)

# 查看token使用量
# print(token_counter.total_llm_token_count)
```

**Settings 可配置项：**
| 配置项 | 说明 |
|-------|------|
| `Settings.llm` | 默认LLM |
| `Settings.embed_model` | 默认嵌入模型 |
| `Settings.chunk_size` | 默认切分大小 |
| `Settings.chunk_overlap` | 默认切分重叠 |
| `Settings.num_output` | 默认输出token数 |
| `Settings.context_window` | 上下文窗口大小 |
| `Settings.callback_manager` | 回调管理器 |

## 4. 在LLM开发中的典型使用场景和代码示例

### 场景1：构建基础RAG问答系统

```python
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 配置模型
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 2. 加载文档
documents = SimpleDirectoryReader("./knowledge_base").load_data()
print(f"加载了 {len(documents)} 个文档")

# 3. 构建索引
index = VectorStoreIndex.from_documents(documents)

# 4. 创建查询引擎（自定义提示词）
qa_template = PromptTemplate(
    "上下文信息如下：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "请仅根据上下文信息回答问题。如果上下文中没有相关信息，请回答"我不知道"。\n"
    "问题: {query_str}\n"
    "回答: "
)

query_engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=qa_template,
    response_mode="compact",
)

# 5. 查询
response = query_engine.query("公司的API限流策略是什么？")
print(response)

# 6. 查看来源
for node in response.source_nodes:
    print(f"[分数: {node.score:.4f}] 来源: {node.node.metadata.get('filename', 'unknown')}")
```

### 场景2：多文档对比分析

```python
from llama_index.core import SummaryIndex, Document
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# 为每个文档构建独立的索引和查询引擎
doc1 = Document(text="产品A的特性：轻量级、易部署...", metadata={"product": "A"})
doc2 = Document(text="产品B的特性：高性能、可扩展...", metadata={"product": "B"})

index_a = VectorStoreIndex.from_documents([doc1])
index_b = VectorStoreIndex.from_documents([doc2])

engine_a = index_a.as_query_engine(similarity_top_k=3)
engine_b = index_b.as_query_engine(similarity_top_k=3)

# 将查询引擎包装为工具
tool_a = QueryEngineTool.from_defaults(
    query_engine=engine_a,
    metadata=ToolMetadata(
        name="product_a",
        description="包含产品A的详细信息，包括特性、价格、部署方式等",
    ),
)
tool_b = QueryEngineTool.from_defaults(
    query_engine=engine_b,
    metadata=ToolMetadata(
        name="product_b",
        description="包含产品B的详细信息，包括特性、价格、部署方式等",
    ),
)

# 创建子问题查询引擎（自动分解问题）
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[tool_a, tool_b]
)

# 提出对比问题
response = sub_question_engine.query("产品A和产品B在部署方式上有什么区别？")
print(response)
```

### 场景3：带对话记忆的聊天机器人

```python
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)  # 对话记忆token上限

index = VectorStoreIndex.from_documents(documents)

chat_engine = index.as_chat_engine(
    chat_mode="condense_question",  # 对话模式
    memory=memory,
    similarity_top_k=3,
    verbose=True,
)

# 多轮对话
response1 = chat_engine.chat("公司的年假政策是什么？")
print(response1)

response2 = chat_engine.chat("那病假呢？")  # 会自动结合上下文理解"那病假"
print(response2)

# 重置对话
chat_engine.reset()
```

**chat_mode 选项：**
| 模式 | 说明 |
|------|------|
| `condense_question` | 将对话历史+新消息浓缩为一个独立问题再检索 |
| `condense_plus_context` | 浓缩问题 + 注入上下文 |
| `react` | 使用 ReAct Agent 模式 |
| `best` | 自动选择最佳模式 |
| `simple` | 不检索，直接LLM对话 |

## 5. 数学原理

### 5.1 向量相似度检索

VectorStoreIndex 的核心是向量相似度计算，最常用的是**余弦相似度**：

$$\text{cosine\_sim}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \cdot \sqrt{\sum_{i=1}^{n} B_i^2}}$$

其中 $A$ 和 $B$ 分别是查询和文档的嵌入向量。余弦相似度取值范围 $[-1, 1]$，值越大表示越相似。

部分向量存储也支持**欧氏距离**（L2 Distance）：

$$d(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}$$

以及**内积（点积）**：

$$\text{ip}(A, B) = A \cdot B = \sum_{i=1}^{n} A_i B_i$$

### 5.2 Reciprocal Rerank Fusion

混合检索中的结果融合算法。对于查询 $q$，第 $i$ 个检索器返回的节点排序为 $r_i$，该节点在排序中的位置为 $\text{rank}_i(d)$，则融合分数为：

$$\text{score}(d) = \sum_{i=1}^{k} \frac{1}{\text{rank}_i(d) + c}$$

其中 $c$ 为平滑常数（默认60），用于降低高位排名的过度影响。

### 5.3 TF-IDF（KeywordTableIndex 的基础）

关键词索引中的权重计算：

$$\text{tf-idf}(t, d) = \text{tf}(t, d) \times \text{idf}(t) = f_{t,d} \times \log\frac{N}{n_t}$$

其中 $f_{t,d}$ 为词 $t$ 在文档 $d$ 中的出现频率，$N$ 为文档总数，$n_t$ 为包含词 $t$ 的文档数。

## 6. 代码原理/架构原理

### RAG 管道的抽象层

LlamaIndex 将 RAG 管道抽象为四个核心阶段：

```
数据摄取 (Ingestion)
    │
    ├── Document: 原始文档容器
    ├── Node: 文档分片，最小的索引/检索单位
    └── NodeParser: 将 Document 切分为 Node
         │
         ▼
索引构建 (Indexing)
    │
    ├── VectorStoreIndex: 将 Node 嵌入为向量并存入向量数据库
    ├── SummaryIndex: 存储 Node 文本列表
    └── KeywordTableIndex: 提取关键词建立倒排索引
         │
         ▼
检索 (Retrieval)
    │
    ├── 向量相似度检索: 查询嵌入 → KNN 搜索
    ├── 关键词检索: 查询关键词 → 倒排索引查找
    └── 混合检索: 多路召回 + 融合排序
         │
         ▼
响应合成 (Response Synthesis)
    │
    ├── Refine: 逐节点迭代精炼
    ├── Compact: 合并后精炼
    └── Tree Summarize: 层级式总结
```

### 核心设计模式

1. **分层抽象**：每个阶段都有独立的接口，上层组件（QueryEngine）组合下层组件（Retriever + ResponseSynthesizer），各层可独立替换。

2. **Settings 单例**：全局配置通过 `Settings` 类管理，避免在每个组件中重复传入模型配置。

3. **模块化集成**：v0.10+ 将向量存储、LLM、嵌入模型等拆分为独立包，通过统一的接口（`BasePydanticVectorStore`、`LLM`、`BaseEmbedding`）对接。

4. **可组合查询管道**：`QueryEngine` → `Retriever` + `ResponseSynthesizer`，也可以用 `QueryPipeline` 实现任意DAG管道：

```python
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate

# 定义管道步骤
prompt_str = "请根据以下上下文回答问题：\n上下文: {context}\n问题: {query}"
prompt_tmpl = PromptTemplate(prompt_str)

# 组装管道
p = QueryPipeline()
p.add_modules({
    "retriever": retriever,
    "prompt": prompt_tmpl,
    "llm": llm,
})
p.add_link("retriever", "prompt", dest_key="context")
p.add_link("prompt", "llm")

# 执行
response = p.run(query="什么是RAG？")
```

## 7. 常见注意事项和最佳实践

### 注意事项

1. **嵌入模型与LLM的一致性**：查询时的嵌入模型必须与构建索引时使用的一致，否则检索质量会严重下降。

2. **chunk_size 的选择**：
   - 过大：检索到的无关信息多，浪费上下文窗口
   - 过小：语义不完整，可能丢失上下文
   - 推荐：512-1024 为常用范围，根据文档类型调整

3. **元数据的正确使用**：
   - `excluded_llm_metadata_keys`：避免将无关元数据发送给LLM浪费token
   - `excluded_embed_metadata_keys`：避免嵌入时引入噪声
   - 善用元数据过滤：`MetadataFilters` 可在检索时按元数据筛选

```python
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

filters = MetadataFilters(
    filters=[MetadataFilter(key="category", value="技术文档")]
)
retriever = index.as_retriever(filters=filters, similarity_top_k=5)
```

4. **索引持久化**：默认索引存在内存中，重启后丢失。务必使用持久化存储：

```python
# 保存索引
index.storage_context.persist(persist_dir="./storage")

# 加载索引
from llama_index.core import load_index_from_storage, StorageContext
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

5. **API 调用成本**：大量文档构建索引会产生大量嵌入API调用，建议：
   - 使用本地嵌入模型（如 HuggingFace）节省费用
   - 增大批次大小 `insert_batch_size`
   - 只对新增文档增量插入

### 最佳实践

1. **选择合适的 response_mode**：
   - 简单事实查询 → `compact`（快速高效）
   - 需要综合多文档信息 → `tree_summarize`
   - 需要精炼答案 → `refine`（较慢但更准确）

2. **检索调优**：
   - 适当设置 `similarity_top_k`（通常3-10）
   - 使用混合检索提升召回率
   - 对检索结果使用 Reranker 重排

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2",
    top_n=3,  # 重排后保留的节点数
)

query_engine = index.as_query_engine(
    similarity_top_k=10,         # 先多检索
    node_postprocessors=[reranker],  # 再精排
)
```

3. **评估检索质量**：使用 LlamaIndex 的评估模块量化检索效果：

```python
from llama_index.core.evaluation import RetrieverEvaluator

evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate", "precision", "recall", "ap", "ndcg"],
)
# 评估结果可根据指标调整 similarity_top_k 和检索策略
```

4. **流式响应用于生产**：避免长时间阻塞用户等待，使用流式输出提升体验。

5. **监控 Token 使用**：通过 `TokenCountingHandler` 跟踪每次查询的 token 消耗，优化成本。

6. **中文场景建议**：
   - 嵌入模型选择中文优化模型（如 `BAAI/bge-small-zh-v1.5`）
   - 切分器优先使用 `SentenceSplitter`，注意中文标点处理
   - 自定义 Prompt 模板，确保中文指令清晰
