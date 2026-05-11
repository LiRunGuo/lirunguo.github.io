---
title: "Chroma 轻量向量数据库"
excerpt: "PersistentClient、Collection CRUD、嵌入函数、元数据过滤、RAG存储"
collection: llm-libs
permalink: /llm-libs/35-chroma
category: vector
toc: true
---


## 1. 库的简介和在LLM开发中的作用

Chroma（又称 ChromaDB）是一个开源的、专注于AI应用的嵌入式向量数据库。它的设计理念是**轻量、易用、开箱即用**，特别适合快速原型开发和小到中等规模的LLM应用场景。

### 核心特点

- **嵌入式架构**：无需独立部署数据库服务，可以直接嵌入Python应用中运行
- **零配置启动**：安装后即可使用，无需复杂的配置和初始化
- **内置嵌入支持**：集成了多种嵌入模型（Sentence Transformers、OpenAI等），自动处理向量化
- **持久化支持**：支持将数据保存到磁盘，重启后数据不丢失
- **丰富的过滤**：支持基于元数据的条件过滤查询

### 在LLM开发中的角色

在LLM应用开发中，Chroma主要扮演**RAG（Retrieval-Augmented Generation）系统的轻量级向量存储**角色：

1. **知识库存储**：将文档分块后生成向量嵌入，存入Chroma，构建可检索的知识库
2. **语义搜索**：根据用户查询的向量表示，从知识库中检索语义最相关的文档片段
3. **对话记忆**：存储对话历史向量，实现长期记忆和上下文检索
4. **Few-Shot样本检索**：存储示例的向量表示，在推理时动态检索相关示例

Chroma与LangChain、LlamaIndex等框架深度集成，是RAG应用开发中最常用的向量存储之一。

## 2. 安装方式

### 基础安装

```bash
# 基础安装（仅包含核心功能）
pip install chromadb

# 安装包含所有嵌入函数依赖的完整版
pip install chromadb[all]

# 仅安装特定嵌入函数的依赖
pip install chromadb  # 然后按需安装嵌入模型
pip install sentence-transformers  # SentenceTransformer嵌入
pip install openai  # OpenAI嵌入
pip install cohere  # Cohere嵌入
pip install google-generativeai  # Google嵌入
```

### 版本说明

```python
import chromadb
print(chromadb.__version__)  # 查看当前版本
```

### Docker部署（可选）

```bash
# 拉取Chroma服务端镜像
docker pull chromadb/chroma

# 启动Chroma服务端
docker run -p 8000:8000 chromadb/chroma

# 带持久化存储启动
docker run -p 8000:8000 -v ./chroma_data:/chroma/chroma chromadb/chroma
```

## 3. 核心类/函数/工具的详细说明

### 3.1 Client - 客户端连接

Chroma提供两种客户端模式：`PersistentClient`（持久化）和`HttpClient`（远程连接）。

#### PersistentClient - 持久化客户端

数据保存到磁盘，重启后数据不丢失，适合开发和生产环境。

```python
import chromadb

# 创建持久化客户端，数据保存到指定目录
client = chromadb.PersistentClient(path="./chroma_db")

# 如果目录已存在数据，会自动加载
# 同一路径只能有一个客户端实例
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | str | 数据持久化目录路径 |
| `settings` | Settings | 可选的配置项 |
| `tenant` | str | 可选的多租户标识 |
| `database` | str | 可选的数据库名称 |

#### HttpClient - 远程连接客户端

连接到独立运行的Chroma服务器，适合分布式部署。

```python
import chromadb

# 连接远程Chroma服务
client = chromadb.HttpClient(
    host="localhost",
    port=8000
)

# 带认证的连接
from chromadb.utils.auth_utils import get_credentials
client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    credentials=get_credentials()  # 基本认证
)

# 使用自定义headers（如API Key）
client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    headers={"X-Api-Key": "your-api-key"}
)
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `host` | str | 服务器地址，默认"localhost" |
| `port` | int | 服务器端口，默认8000 |
| `ssl` | bool | 是否使用HTTPS，默认False |
| `headers` | dict | 自定义HTTP头 |
| `credentials` | Credentials | 认证凭据 |

#### Client - 内存客户端

数据仅保存在内存中，程序退出后数据丢失，适合快速测试。

```python
import chromadb

# 创建纯内存客户端（临时测试用）
client = chromadb.Client()

# 也可以通过Settings创建
from chromadb.config import Settings
client = chromadb.Client(Settings(
    chroma_db_impl="chromadb.db.duckdb.DuckDB",
    persist_directory=None  # None表示纯内存
))
```

### 3.2 Collection - 集合操作

Collection是Chroma中存储向量数据的基本单元，类似于数据库中的"表"。

#### create_collection - 创建集合

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# 基本创建
collection = client.create_collection(
    name="my_documents",
    metadata={"description": "我的文档集合"}  # 可选的集合元数据
)

# 创建带嵌入函数的集合
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-api-key",
    model_name="text-embedding-ada-002"
)

collection = client.create_collection(
    name="openai_docs",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}  # 指定距离度量方式
)

# 指定不同的距离度量方式
# "l2" - L2距离（欧几里得距离），默认
# "cosine" - 余弦相似度
# "ip" - 内积
cosine_collection = client.create_collection(
    name="cosine_docs",
    metadata={"hnsw:space": "cosine"}
)

ip_collection = client.create_collection(
    name="ip_docs",
    metadata={"hnsw:space": "ip"}
)
```

**参数说明**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | str | 是 | 集合名称，必须唯一 |
| `metadata` | dict | 否 | 集合元数据，可包含`hnsw:space`等配置 |
| `embedding_function` | EmbeddingFunction | 否 | 嵌入函数，未提供时需手动提供向量 |
| `get_or_create` | bool | 否 | 若为True，存在则获取而非报错，默认False |

#### get_collection / get_or_create_collection

```python
# 获取已存在的集合（不存在则报错）
collection = client.get_collection(
    name="my_documents",
    embedding_function=openai_ef  # 如果集合使用自定义嵌入函数，获取时也要指定
)

# 获取或创建集合（推荐用法）
collection = client.get_or_create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}
)
```

#### list_collections / delete_collection

```python
# 列出所有集合
collections = client.list_collections()
for col in collections:
    print(f"集合名: {col.name}, ID: {col.id}")

# 删除集合
client.delete_collection(name="my_documents")

# 重置整个客户端数据（慎用！）
client.reset()  # 清除所有数据
```

### 3.3 数据操作

#### add() - 添加数据

```python
collection = client.get_or_create_collection(name="docs")

# 方式1：手动提供向量
collection.add(
    ids=["doc1", "doc2", "doc3"],             # 唯一标识符，必填
    embeddings=[                               # 向量列表，与ids一一对应
        [0.1, 0.2, 0.3, ...],
        [0.4, 0.5, 0.6, ...],
        [0.7, 0.8, 0.9, ...]
    ],
    documents=["文档1的内容", "文档2的内容", "文档3的内容"],  # 可选：原始文本
    metadatas=[                                # 可选：元数据
        {"source": "web", "page": 1},
        {"source": "pdf", "page": 5},
        {"source": "web", "page": 3}
    ]
)

# 方式2：使用嵌入函数自动生成向量（需在创建集合时指定embedding_function）
collection.add(
    ids=["doc4", "doc5"],
    documents=["这是新的文档内容", "另一个文档"],
    metadatas=[
        {"source": "api", "category": "tech"},
        {"source": "api", "category": "science"}
    ]
    # 无需提供embeddings，自动使用集合的嵌入函数生成
)
```

**参数说明**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `ids` | list[str] | 是 | 唯一标识符列表，重复id会报错 |
| `embeddings` | list[list[float]] | 条件 | 向量列表（有嵌入函数时可省略） |
| `documents` | list[str] | 否 | 原始文本文档列表 |
| `metadatas` | list[dict] | 否 | 元数据列表，用于过滤 |

#### query() - 查询数据

```python
# 基本查询：根据查询向量找最近的文档
results = collection.query(
    query_embeddings=[[0.1, 0.2, 0.3, ...]],  # 查询向量
    n_results=5                                 # 返回最相似的5个结果
)

# 使用文本查询（需集合有嵌入函数）
results = collection.query(
    query_texts=["什么是机器学习？"],  # 自动转换为向量
    n_results=5
)

# 带元数据过滤的查询
results = collection.query(
    query_texts=["技术相关内容"],
    n_results=5,
    where={"source": "web"}  # 只在source为web的文档中搜索
)

# 复杂过滤条件
results = collection.query(
    query_texts=["技术相关内容"],
    n_results=5,
    where={
        "$and": [
            {"source": "web"},
            {"page": {"$gt": 2}}
        ]
    }
)

# 指定返回内容
results = collection.query(
    query_texts=["查询文本"],
    n_results=5,
    include=["documents", "metadatas", "distances", "embeddings"]
)
```

**返回值说明**：

```python
# query返回一个字典，结构如下：
{
    'ids': [['doc1', 'doc3', 'doc5']],           # 匹配的文档ID
    'distances': [[0.12, 0.25, 0.38]],           # 距离值（越小越相似）
    'documents': [['文档1内容', '文档3内容', '文档5内容']],  # 原始文本
    'metadatas': [[{...}, {...}, {...}]],         # 元数据
    'embeddings': [[...], [...], [...]]           # 向量（仅include时返回）
}
```

**参数说明**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query_embeddings` | list[list[float]] | 条件 | 查询向量（与query_texts二选一） |
| `query_texts` | list[str] | 条件 | 查询文本（自动向量化） |
| `n_results` | int | 否 | 返回结果数量，默认10 |
| `where` | dict | 否 | 元数据过滤条件 |
| `where_document` | dict | 否 | 文档内容过滤条件 |
| `include` | list[str] | 否 | 返回哪些字段，默认全部 |

#### update() - 更新数据

```python
# 更新已有文档（id必须存在，否则报错）
collection.update(
    ids=["doc1", "doc2"],
    documents=["更新后的文档1内容", "更新后的文档2内容"],
    metadatas=[
        {"source": "updated", "page": 10},
        {"source": "updated", "page": 11}
    ],
    embeddings=[[0.2, 0.3, 0.4, ...], [0.5, 0.6, 0.7, ...]]
)
```

#### upsert() - 插入或更新

```python
# 如果id不存在则添加，存在则更新（最安全的写入方式）
collection.upsert(
    ids=["doc1", "doc6"],
    documents=["更新doc1", "新增doc6"],
    metadatas=[
        {"source": "upsert", "version": 2},
        {"source": "upsert", "version": 1}
    ]
)
```

#### delete() - 删除数据

```python
# 按ID删除
collection.delete(ids=["doc1", "doc2"])

# 按条件删除
collection.delete(
    where={"source": "web"}  # 删除所有source为web的文档
)

# 按文档内容删除
collection.delete(
    where_document={"$contains": "过期内容"}
)

# 删除集合中所有数据
collection.delete(
    where={}  # 空条件匹配所有
)
```

#### get() - 获取数据（非查询）

```python
# 按ID获取
results = collection.get(ids=["doc1", "doc3"])

# 按条件获取
results = collection.get(
    where={"source": "web"},
    include=["documents", "metadatas"]
)

# 获取所有数据
results = collection.get(
    include=["documents", "metadatas", "embeddings"]
)

# 限制返回数量
results = collection.get(
    limit=100,
    offset=0
)

# 按文档内容过滤
results = collection.get(
    where_document={"$contains": "机器学习"}
)
```

### 3.4 嵌入函数 - Embedding Functions

Chroma内置了多种嵌入函数，可自动将文本转换为向量。

#### SentenceTransformerEmbeddingFunction

```python
from chromadb.utils import embedding_functions

# 使用默认模型（all-MiniLM-L6-v2）
default_ef = embedding_functions.SentenceTransformerEmbeddingFunction()

# 指定模型
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",  # 多语言模型
    device="cpu",            # 运行设备：cpu或cuda
    normalize_embeddings=True  # 是否归一化向量
)

# 创建集合时使用
collection = client.create_collection(
    name="multilingual_docs",
    embedding_function=ef
)
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_name` | str | Sentence Transformers模型名称 |
| `device` | str | 运行设备，默认"cpu" |
| `normalize_embeddings` | bool | 是否L2归一化，默认False |
| `model_kwargs` | dict | 传递给模型的额外参数 |
| `encode_kwargs` | dict | 传递给encode()的额外参数 |

#### OpenAIEmbeddingFunction

```python
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="sk-your-api-key",
    model_name="text-embedding-ada-002",  # 或 text-embedding-3-small/large
    organization_id="org-xxx",  # 可选
    api_base="https://api.openai.com/v1",  # 可选，支持自定义端点
    api_type="open_ai"  # 可选：open_ai, azure, azure_ad
)

# Azure OpenAI
azure_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-azure-key",
    api_base="https://your-resource.openai.azure.com",
    api_type="azure",
    deployment_id="your-deployment-id"
)
```

#### HuggingFaceEmbeddingFunction

```python
from chromadb.utils import embedding_functions

hf_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="hf_your-token",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

#### GoogleGenerativeAIEmbeddingFunction

```python
from chromadb.utils import embedding_functions

google_ef = embedding_functions.GoogleGenerativeAIEmbeddingFunction(
    api_key="your-google-api-key",
    model_name="models/embedding-001"
)
```

#### CohereEmbeddingFunction

```python
from chromadb.utils import embedding_functions

cohere_ef = embedding_functions.CohereEmbeddingFunction(
    api_key="your-cohere-key",
    model_name="embed-english-v3.0",
    input_type="search_document"  # search_document, search_query, classification, clustering
)
```

#### ONNXMiniLM_L6_V2 - 内置嵌入函数

```python
# Chroma内置的ONNX格式MiniLM模型，无需额外依赖
from chromadb.utils import embedding_functions

# 这是Chroma的默认嵌入函数
default_ef = embedding_functions.ONNXMiniLM_L6_V2()
```

### 3.5 持久化 - 数据保存到磁盘

```python
import chromadb

# === 方式1：PersistentClient（推荐） ===
# 创建时指定路径，所有操作自动持久化
client = chromadb.PersistentClient(path="./chroma_db")

# 创建集合和添加数据
collection = client.get_or_create_collection(name="persisted_docs")
collection.add(
    ids=["doc1"],
    documents=["持久化存储的文档"],
    metadatas=[{"source": "test"}]
)

# 程序重启后，再次连接同一目录即可恢复数据
client2 = chromadb.PersistentClient(path="./chroma_db")
collection2 = client2.get_collection(name="persisted_docs")
print(collection2.count())  # 输出: 1

# === 方式2：备份与迁移 ===
# 查看集合信息
print(collection.name)
print(collection.count())
print(collection.metadata)

# 导出数据
all_data = collection.get(include=["documents", "metadatas", "embeddings"])
# 可以将all_data序列化保存到文件
import json
with open("backup.json", "w") as f:
    json.dump(all_data, f)
```

### 3.6 元数据过滤 - Where条件

Chroma支持丰富的元数据过滤语法，用于在查询时缩小搜索范围。

#### 基本比较操作符

```python
# 等于
results = collection.query(
    query_texts=["查询"],
    where={"category": "tech"}
)

# 不等于
results = collection.query(
    query_texts=["查询"],
    where={"category": {"$ne": "deprecated"}}
)

# 大于/大于等于
results = collection.query(
    query_texts=["查询"],
    where={"year": {"$gt": 2023}}
)
results = collection.query(
    query_texts=["查询"],
    where={"score": {"$gte": 0.8}}
)

# 小于/小于等于
results = collection.query(
    query_texts=["查询"],
    where={"price": {"$lt": 100}}
)

# 不支持数值类型的元数据需在添加时确保类型正确
# Chroma的元数据值支持：str, int, float, bool
```

#### 逻辑操作符

```python
# AND逻辑
results = collection.query(
    query_texts=["查询"],
    where={
        "$and": [
            {"category": "tech"},
            {"year": {"$gte": 2023}}
        ]
    }
)

# OR逻辑
results = collection.query(
    query_texts=["查询"],
    where={
        "$or": [
            {"category": "tech"},
            {"category": "science"}
        ]
    }
)

# 嵌套逻辑
results = collection.query(
    query_texts=["查询"],
    where={
        "$and": [
            {"status": "published"},
            {"$or": [
                {"category": "tech"},
                {"category": "science"}
            ]}
        ]
    }
)
```

#### 文档内容过滤

```python
# 包含指定文本
results = collection.query(
    query_texts=["查询"],
    where_document={"$contains": "机器学习"}
)

# 不包含指定文本
results = collection.query(
    query_texts=["查询"],
    where_document={"$not_contains": "deprecated"}
)

# 组合使用where和where_document
results = collection.query(
    query_texts=["查询"],
    where={"category": "tech"},
    where_document={"$contains": "深度学习"}
)
```

#### 过滤操作符完整列表

| 操作符 | 适用类型 | 说明 |
|--------|----------|------|
| `$eq` | str/int/float/bool | 等于（可简写为直接赋值） |
| `$ne` | str/int/float/bool | 不等于 |
| `$gt` | int/float | 大于 |
| `$gte` | int/float | 大于等于 |
| `$lt` | int/float | 小于 |
| `$lte` | int/float | 小于等于 |
| `$and` | list | 逻辑与 |
| `$or` | list | 逻辑或 |
| `$contains` | str | 文本包含（仅用于where_document） |
| `$not_contains` | str | 文本不包含（仅用于where_document） |

## 4. 在LLM开发中的典型使用场景和代码示例

### 4.1 RAG系统 - 知识库问答

这是Chroma最常见的使用场景，构建一个基于文档的问答系统。

```python
import chromadb
from chromadb.utils import embedding_functions

# === 1. 初始化 ===
client = chromadb.PersistentClient(path="./rag_db")

# 使用多语言嵌入模型
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

collection = client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"}
)

# === 2. 文档分块与索引 ===
def chunk_text(text, chunk_size=500, overlap=50):
    """简单的文本分块函数"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# 模拟文档数据
documents = [
    {"id": "doc1", "text": "Python是一种广泛使用的高级编程语言...", "source": "python_intro.pdf"},
    {"id": "doc2", "text": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习...", "source": "ml_basics.pdf"},
    {"id": "doc3", "text": "深度学习是机器学习的子集，使用多层神经网络来建模数据中的复杂模式...", "source": "dl_intro.pdf"},
]

# 分块并添加到集合
all_ids = []
all_docs = []
all_metadatas = []

for doc in documents:
    chunks = chunk_text(doc["text"])
    for i, chunk in enumerate(chunks):
        all_ids.append(f"{doc['id']}_chunk{i}")
        all_docs.append(chunk)
        all_metadatas.append({
            "source": doc["source"],
            "chunk_index": i,
            "total_chunks": len(chunks)
        })

collection.upsert(
    ids=all_ids,
    documents=all_docs,
    metadatas=all_metadatas
)

# === 3. 检索相关文档 ===
def retrieve_context(query, n_results=3):
    """检索与查询最相关的文档片段"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    contexts = []
    for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        contexts.append({
            "text": doc,
            "source": meta["source"],
            "distance": dist
        })
    return contexts

# === 4. 构建RAG问答 ===
def rag_query(query, llm_generate_func):
    """RAG查询流程"""
    # 步骤1：检索相关上下文
    contexts = retrieve_context(query, n_results=3)

    # 步骤2：构建提示词
    context_text = "\n\n".join([c["text"] for c in contexts])
    prompt = f"""基于以下参考资料回答问题。如果资料中没有相关信息，请说明。

参考资料：
{context_text}

问题：{query}
回答："""

    # 步骤3：调用LLM生成回答
    answer = llm_generate_func(prompt)
    return answer, contexts

# 使用示例
query = "什么是深度学习？"
# answer, sources = rag_query(query, your_llm_function)
```

### 4.2 对话记忆系统

使用Chroma存储和检索对话历史，实现长期记忆。

```python
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime

client = chromadb.PersistentClient(path="./memory_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction()

collection = client.get_or_create_collection(
    name="conversation_memory",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"}
)

def save_message(role, content, session_id, user_id="user1"):
    """保存对话消息到向量存储"""
    timestamp = datetime.now().isoformat()
    message_id = f"{user_id}_{session_id}_{timestamp}"

    collection.upsert(
        ids=[message_id],
        documents=[content],
        metadatas=[{
            "role": role,           # user 或 assistant
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": timestamp
        }]
    )

def recall_relevant(query, n_results=5, user_id="user1"):
    """检索与当前查询相关的历史对话"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"user_id": user_id},
        include=["documents", "metadatas", "distances"]
    )

    memories = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        memories.append({
            "content": doc,
            "role": meta["role"],
            "timestamp": meta["timestamp"]
        })
    return memories

# 使用示例
session_id = "session_001"
save_message("user", "我想学习Python编程", session_id)
save_message("assistant", "Python是一门非常适合初学者的语言...", session_id)
save_message("user", "推荐一些Python学习资源", session_id)

# 新对话中回忆相关上下文
relevant = recall_relevant("Python入门教程推荐")
```

### 4.3 Few-Shot样本检索

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./fewshot_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction()

collection = client.get_or_create_collection(
    name="few_shot_examples",
    embedding_function=ef
)

# 添加示例
examples = [
    {
        "id": "ex1",
        "input": "把这段话翻译成英文",
        "output": "Translate this passage into English.",
        "category": "translation"
    },
    {
        "id": "ex2",
        "input": "总结这篇文章的要点",
        "output": "Summarize the key points of this article.",
        "category": "translation"
    },
    {
        "id": "ex3",
        "input": "写一首关于春天的诗",
        "output": "春风拂面花正开\n桃红柳绿入梦来...",
        "category": "creative"
    }
]

collection.upsert(
    ids=[e["id"] for e in examples],
    documents=[e["input"] for e in examples],
    metadatas=[{"output": e["output"], "category": e["category"]} for e in examples]
)

# 根据新输入检索最相关的示例
def get_few_shot_examples(query, n_results=2, category=None):
    """检索与查询最相关的Few-Shot示例"""
    where_filter = {}
    if category:
        where_filter["category"] = category

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter if where_filter else None,
        include=["documents", "metadatas"]
    )

    examples = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        examples.append({
            "input": doc,
            "output": meta["output"]
        })
    return examples

# 使用示例
examples = get_few_shot_examples("把这句话翻成日语")
# 返回最相关的翻译示例
```

### 4.4 与LangChain集成

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 创建嵌入函数
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# 从文档创建向量存储
texts = text_splitter.split_text(your_document_text)
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    collection_name="langchain_docs",
    persist_directory="./langchain_chroma_db"
)

# 检索
results = vectorstore.similarity_search(
    query="什么是RAG？",
    k=3
)

# 创建检索器
retriever = vectorstore.as_retriever(
    search_type="mmr",         # 最大边际相关性
    search_kwargs={"k": 3, "fetch_k": 10}
)

# 与RAG链结合
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever
)

answer = qa_chain.run("什么是RAG？")
```

## 5. 数学原理

### 5.1 余弦相似度 (Cosine Similarity)

余弦相似度衡量两个向量之间的方向相似性，取值范围为[-1, 1]。

**公式**：

$$\cos(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|} = \frac{\sum_{i=1}^{n} a_i \cdot b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \times \sqrt{\sum_{i=1}^{n} b_i^2}}$$

**含义**：
- 值越接近1，表示方向越一致（越相似）
- 值为0表示正交（无相关性）
- 值接近-1表示方向相反

**Python实现**：

```python
import numpy as np

def cosine_similarity(a, b):
    """计算余弦相似度"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# 示例
a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 4.0, 6.0])  # 与a方向相同
c = np.array([1.0, 0.0, 0.0])  # 不同方向

print(cosine_similarity(a, b))  # 1.0（完全相同方向）
print(cosine_similarity(a, c))  # 约0.27（部分相似）
```

**在Chroma中使用**：设置 `metadata={"hnsw:space": "cosine"}`，此时距离值为 `1 - cosine_similarity`（越小越相似）。

### 5.2 L2距离 (欧几里得距离)

L2距离衡量两个向量在空间中的直线距离，值越小表示越相似。

**公式**：

$$d(A, B) = \|A - B\| = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

**含义**：
- 值为0表示两个向量完全相同
- 值越大表示差异越大
- 对向量的幅度敏感

**Python实现**：

```python
import numpy as np

def l2_distance(a, b):
    """计算L2距离"""
    return np.linalg.norm(a - b)

# 示例
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])
c = np.array([4.0, 5.0, 6.0])

print(l2_distance(a, b))  # 0.0（完全相同）
print(l2_distance(a, c))  # 约5.20（差异较大）
```

**在Chroma中使用**：这是默认的距离度量方式，设置 `metadata={"hnsw:space": "l2"}` 或不指定。

### 5.3 内积 (Inner Product / Dot Product)

内积直接计算两个向量的点积，值越大表示越相似。

**公式**：

$$ip(A, B) = A \cdot B = \sum_{i=1}^{n} a_i \cdot b_i$$

**含义**：
- 内积同时考虑了向量的方向和幅度
- 如果向量已归一化（单位向量），内积等于余弦相似度
- 适合最大内积搜索（MIPS）场景

**Python实现**：

```python
import numpy as np

def inner_product(a, b):
    """计算内积"""
    return np.dot(a, b)

# 归一化后的内积等于余弦相似度
a = np.array([1.0, 2.0, 3.0])
a_normalized = a / np.linalg.norm(a)
b = np.array([2.0, 4.0, 6.0])
b_normalized = b / np.linalg.norm(b)

print(inner_product(a_normalized, b_normalized))  # 1.0（归一化后方向相同）
```

**在Chroma中使用**：设置 `metadata={"hnsw:space": "ip"}`，此时距离值为 `-inner_product`（取负后越小越相似）。

### 5.4 三种度量方式的比较与选择

| 度量方式 | 值域 | 对幅度敏感 | 适用场景 |
|----------|------|------------|----------|
| L2距离 | [0, +∞) | 是 | 向量幅度有意义时 |
| 余弦相似度 | [-1, 1]→距离[0, 2] | 否 | 文本语义相似度（最常用） |
| 内积 | (-∞, +∞)→距离(-∞, +∞) | 是 | 归一化向量、MIPS搜索 |

**选择建议**：
- 文本语义搜索：**余弦相似度**（最常用，不受向量幅度影响）
- 需要考虑幅度差异：**L2距离**
- 向量已归一化或需要最大内积搜索：**内积**

## 6. 代码原理/架构原理

### 6.1 整体架构

Chroma的架构设计围绕"嵌入式优先"理念，主要组件如下：

```
┌─────────────────────────────────────┐
│           Client Layer              │
│  (PersistentClient / HttpClient)    │
├─────────────────────────────────────┤
│         Collection Layer            │
│  (CRUD操作 + 查询 + 过滤)           │
├─────────────────────────────────────┤
│       Embedding Function Layer      │
│  (文本→向量的自动转换)              │
├──────────────┬──────────────────────┤
│  Vector Store│   Metadata Store     │
│  (HNSW索引)  │   (SQLite/DuckDB)    │
├──────────────┴──────────────────────┤
│        Storage Backend              │
│  (内存 / DuckDB文件 / 远程HTTP)     │
└─────────────────────────────────────┘
```

### 6.2 向量索引 - HNSW

Chroma底层使用HNSW（Hierarchical Navigable Small World）算法进行近似最近邻搜索：

1. **构建索引**：插入向量时，HNSW自动构建多层导航图
2. **搜索过程**：从顶层开始贪心搜索，逐层下降到底层找到最近邻
3. **近似搜索**：以O(log N)的时间复杂度实现高质量的近似最近邻搜索

HNSW的关键配置参数：

```python
# 通过metadata设置HNSW参数
collection = client.create_collection(
    name="tuned_collection",
    metadata={
        "hnsw:space": "cosine",        # 距离度量
        "hnsw:M": 16,                   # 每个节点的最大连接数，默认16
        "hnsw:construction_ef": 100,    # 构建时的搜索宽度，默认100
        "hnsw:batch_size": 100          # 构建索引的批大小
    }
)
```

### 6.3 存储引擎

- **DuckDB**：Chroma使用DuckDB作为默认的元数据存储引擎，支持SQL查询和事务
- **向量存储**：HNSW索引直接管理向量数据
- **持久化**：数据通过DuckDB写入磁盘文件，保证持久性

### 6.4 查询执行流程

```
1. 接收查询文本或向量
   ↓
2. 如果是文本，调用嵌入函数转换为向量
   ↓
3. 如果有where条件，先在元数据存储中预过滤
   ↓
4. 在HNSW索引中执行近似最近邻搜索
   ↓
5. 对结果应用元数据过滤（后过滤）
   ↓
6. 返回排序后的结果
```

### 6.5 嵌入函数的架构

```python
# Chroma的嵌入函数遵循统一接口
class EmbeddingFunction(Protocol):
    def __call__(self, input: Documents) -> Embeddings:
        """将文档列表转换为向量列表"""
        ...

# 所有内置嵌入函数都实现此接口
# 在add/query时自动调用
# 也可以手动调用
ef = embedding_functions.SentenceTransformerEmbeddingFunction()
vectors = ef(["文本1", "文本2"])  # 返回二维数组
```

## 7. 常见注意事项和最佳实践

### 7.1 数据管理注意事项

```python
# ❌ 错误：重复ID会报错
collection.add(ids=["doc1"], documents=["内容1"])
collection.add(ids=["doc1"], documents=["内容2"])  # 报错！ID已存在

# ✅ 正确：使用upsert避免重复ID问题
collection.upsert(ids=["doc1"], documents=["内容1"])
collection.upsert(ids=["doc1"], documents=["内容2"])  # 更新而非报错

# ✅ 正确：批量操作提高效率
collection.upsert(
    ids=[f"doc{i}" for i in range(1000)],
    documents=[f"文档内容{i}" for i in range(1000)]
)
# 而非循环调用1000次
```

### 7.2 嵌入函数一致性

```python
# ⚠️ 关键：查询时使用的嵌入函数必须与存储时一致

# ✅ 正确：创建集合时指定嵌入函数，后续操作自动使用
collection = client.create_collection(
    name="consistent_ef",
    embedding_function=same_ef  # 存储和查询使用同一个
)

# ❌ 错误：存储用模型A，查询用模型B
# 向量空间不一致，搜索结果毫无意义
```

### 7.3 元数据设计最佳实践

```python
# ✅ 正确：元数据用于过滤，设计时考虑查询需求
collection.add(
    ids=["doc1"],
    documents=["文档内容"],
    metadatas=[{
        "source": "pdf",           # 字符串：用于精确匹配
        "page": 42,                # 整数：用于范围查询
        "timestamp": 1700000000,   # 数字时间戳：用于时间范围
        "is_processed": True,      # 布尔：用于状态过滤
        "tags": "tech,ai,ml"       # 用逗号分隔模拟多标签
    }]
)

# ⚠️ 注意：元数据值类型必须一致
# 如果某个字段有时是字符串有时是数字，查询时可能出问题
# ⚠️ Chroma不支持列表类型的元数据值
```

### 7.4 性能优化

```python
# 1. 批量操作
# ✅ 一次添加多条记录
collection.upsert(ids=ids, documents=docs, metadatas=metas)

# ❌ 避免逐条添加
# for id, doc, meta in zip(ids, docs, metas):
#     collection.upsert(ids=[id], documents=[doc], metadatas=[meta])

# 2. 选择合适的距离度量
# 文本语义搜索推荐cosine，默认是l2
collection = client.create_collection(
    name="cosine_collection",
    metadata={"hnsw:space": "cosine"}
)

# 3. 合理设置n_results
# 不要设置过大的n_results，会降低性能
results = collection.query(query_texts=["查询"], n_results=5)  # 足够

# 4. 利用元数据预过滤减少搜索空间
results = collection.query(
    query_texts=["查询"],
    where={"category": "tech"},  # 先过滤再搜索
    n_results=5
)

# 5. 调整HNSW参数
# 更大的M和ef_construct提高搜索精度但增加内存和构建时间
collection = client.create_collection(
    name="high_accuracy",
    metadata={
        "hnsw:M": 32,                 # 增大连接数
        "hnsw:construction_ef": 200,  # 增大构建搜索宽度
    }
)
```

### 7.5 持久化与并发

```python
# ⚠️ 注意：同一持久化路径只能有一个客户端实例
# ❌ 错误
client1 = chromadb.PersistentClient(path="./db")
client2 = chromadb.PersistentClient(path="./db")  # 可能导致数据损坏

# ✅ 正确：复用同一个客户端实例
client = chromadb.PersistentClient(path="./db")

# 如果需要并发访问，使用HttpClient连接独立服务
# 启动Chroma服务端后
client = chromadb.HttpClient(host="localhost", port=8000)
```

### 7.6 常见错误处理

```python
import chromadb
from chromadb.errors import InvalidCollectionException, InvalidArgumentError

# 处理集合不存在的错误
try:
    collection = client.get_collection(name="nonexistent")
except ValueError as e:
    print(f"集合不存在: {e}")
    # 改用get_or_create_collection
    collection = client.get_or_create_collection(name="nonexistent")

# 处理ID重复错误
try:
    collection.add(ids=["existing_id"], documents=["内容"])
except InvalidArgumentError:
    collection.upsert(ids=["existing_id"], documents=["内容"])

# 检查集合是否存在
collections = client.list_collections()
existing_names = [c.name for c in collections]
if "my_collection" in existing_names:
    collection = client.get_collection(name="my_collection")
else:
    collection = client.create_collection(name="my_collection")
```

### 7.7 集合统计与监控

```python
# 获取集合中记录数量
count = collection.count()
print(f"集合中有 {count} 条记录")

# 获取集合的元信息
print(collection.name)       # 集合名称
print(collection.id)         # 集合UUID
print(collection.metadata)   # 集合元数据

# 修改集合元数据
collection.modify(metadata={"description": "更新描述"})

# 修改集合名称
collection.modify(name="new_name")
```

### 7.8 常见陷阱总结

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| 嵌入函数不一致 | 存储和查询使用不同模型 | 始终使用同一个嵌入函数实例 |
| ID重复 | add()时使用已存在的ID | 使用upsert()替代add() |
| 元数据类型不一致 | 同一字段混用str和int | 统一元数据值的类型 |
| 同路径多客户端 | 可能导致数据损坏 | 确保同一路径只有一个实例 |
| 过大的n_results | 影响查询性能 | 根据需求设置合理值 |
| 未指定距离度量 | 默认L2，可能不适合文本 | 文本场景使用cosine |
| 大量小批量操作 | 性能低下 | 使用批量操作 |
