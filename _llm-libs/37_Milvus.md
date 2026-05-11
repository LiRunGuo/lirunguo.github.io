---
title: "Milvus 分布式向量数据库"
excerpt: "IVF/PQ/HNSW索引、hybrid_search、存算分离架构、MilvusClient"
collection: llm-libs
permalink: /llm-libs/37-milvus
category: vector
---


## 1. 简介

Milvus 是一款开源的高性能分布式向量数据库，专为海量向量数据的存储、索引和检索而设计。在 LLM（大语言模型）开发中，Milvus 扮演着关键的基础设施角色：它为 RAG（检索增强生成）系统提供高效的语义检索能力，使 LLM 能够基于外部知识库生成更准确、更可靠的回答。

### Milvus 在 LLM 开发中的核心作用

- **RAG 系统的向量存储引擎**：将文档经过 Embedding 模型编码后的向量存入 Milvus，在推理时快速检索最相关的文档片段
- **语义搜索引擎**：支持亿级向量数据的毫秒级检索，适用于企业级知识库问答系统
- **多模态检索**：支持图像、音频、文本等多种模态的向量检索，可用于多模态 LLM 应用
- **推荐系统**：基于用户和物品的向量表示进行相似度匹配，为 LLM 提供个性化上下文
- **去重与聚类**：在数据预处理阶段进行向量去重和聚类分析

### 核心特性

- 支持多种索引类型（IVF_FLAT, IVF_SQ8, IVF_PQ8, HNSW, ANNOY, SCANN 等）
- 支持混合搜索（向量搜索 + 标量过滤）
- 分布式架构，支持水平扩展
- 支持数据的持久化和热加载
- 提供 PyMilvus（Python SDK）和 RESTful API

## 2. 安装方式

### 安装 PyMilvus（Python SDK）

```bash
pip install pymilvus
```

### 安装 Milvus 服务器

**方式一：Docker（推荐用于开发）**

```bash
# 拉取 Milvus Standalone 镜像
docker pull milvusdb/milvus:latest

# 使用 docker-compose 启动
wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d
```

**方式二：Milvus Lite（轻量级，适合快速测试）**

```python
# Milvus Lite 无需额外安装服务器，随 PyMilvus 一起使用
from pymilvus import MilvusClient

# 使用本地文件作为存储
client = MilvusClient("./milvus_demo.db")
```

**方式三：Milvus Operator（Kubernetes 部署）**

```bash
# 安装 Milvus Operator
kubectl apply -f https://raw.githubusercontent.com/milvus-io/milvus-operator/main/deploy/manifests/deployment.yaml
```

### 安装验证

```python
from pymilvus import connections

# 连接到 Milvus 服务器
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)
print("Milvus 连接成功！")
```

## 3. 核心类/函数/工具详细说明

### 3.1 连接管理：connections

#### connections.connect()

建立与 Milvus 服务器的连接。

```python
from pymilvus import connections

connections.connect(
    alias="default",   # 连接别名，后续操作可引用
    host="localhost",  # Milvus 服务器地址
    port="19530",      # gRPC 端口号，默认19530
    user="",           # 用户名（如启用了认证）
    password="",       # 密码
    db_name="default", # 数据库名称
    token="",          # 认证令牌（替代user/password）
    server_name="",    # 服务器名称（TLS）
    server_pem_path="",# TLS 证书路径
)
```

**返回值**：无返回值，连接成功后即可通过别名引用。

#### connections.disconnect()

断开与 Milvus 的连接。

```python
connections.disconnect(alias="default")
```

#### connections.list_connections()

列出所有连接。

```python
conns = connections.list_connections()
# 返回格式: [("alias", <address>), ...]
```

### 3.2 Collection（集合）

Collection 是 Milvus 中存储数据的基本单元，类似于关系数据库中的表。

#### 定义 Schema

```python
from pymilvus import CollectionSchema, FieldSchema, DataType

# 主键字段
pk_field = FieldSchema(
    name="id",                # 字段名
    dtype=DataType.INT64,     # 数据类型
    is_primary=True,          # 是否为主键
    auto_id=False,            # 是否自动生成ID
    description="主键ID"
)

# 向量字段
vector_field = FieldSchema(
    name="embedding",
    dtype=DataType.FLOAT_VECTOR,  # 浮点向量类型
    dim=768,                      # 向量维度
    description="文本embedding向量"
)

# 标量字段
text_field = FieldSchema(
    name="text",
    dtype=DataType.VARCHAR,   # 变长字符串
    max_length=65535,         # 最大长度
    description="原始文本"
)

source_field = FieldSchema(
    name="source",
    dtype=DataType.VARCHAR,
    max_length=256,
    description="数据来源"
)

# 构建Schema
schema = CollectionSchema(
    fields=[pk_field, vector_field, text_field, source_field],
    description="RAG知识库文档集合",
    enable_dynamic_field=True  # 允许动态字段
)
```

**DataType 支持的类型**：

| 类型 | 说明 |
|------|------|
| `DataType.BOOL` | 布尔值 |
| `DataType.INT8/INT16/INT32/INT64` | 整数类型 |
| `DataType.FLOAT/DOUBLE` | 浮点类型 |
| `DataType.VARCHAR` | 变长字符串 |
| `DataType.JSON` | JSON 类型 |
| `DataType.ARRAY` | 数组类型 |
| `DataType.FLOAT_VECTOR` | 浮点向量 |
| `DataType.BINARY_VECTOR` | 二值向量 |
| `DataType.FLOAT16_VECTOR` | 半精度浮点向量 |
| `DataType.BFLOAT16_VECTOR` | BF16 向量 |
| `DataType.SPARSE_FLOAT_VECTOR` | 稀疏浮点向量 |

#### 创建 Collection

```python
from pymilvus import Collection

# 方式一：使用 Schema 创建
collection = Collection(
    name="rag_documents",  # 集合名称
    schema=schema,         # Schema对象
    using="default",       # 使用的连接别名
    shards_num=2,          # 分片数量（影响写入并行度）
)

# 方式二：使用 MilvusClient 快速创建（简化API）
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
client.create_collection(
    collection_name="rag_documents",
    dimension=768,         # 向量维度（自动生成schema）
    metric_type="COSINE",  # 距离度量类型
)
```

#### Collection 常用方法

```python
# 获取已存在的 Collection
collection = Collection("rag_documents")

# 获取集合信息
info = collection.describe()
# 返回: {'name': 'rag_documents', 'description': '...', 'fields': [...], ...}

# 获取行数
count = collection.num_entities

# 删除 Collection
collection.drop()

# 列出所有 Collection
from pymilvus import utility
collections = utility.list_collections()
```

### 3.3 索引类型

索引是加速向量搜索的核心机制。Milvus 支持多种索引类型，适用于不同场景。

#### 创建索引

```python
# 为向量字段创建索引
index_params = {
    "index_type": "IVF_FLAT",    # 索引类型
    "metric_type": "COSINE",     # 距离度量: L2, IP, COSINE
    "params": {                   # 索引参数
        "nlist": 128,             # 聚类中心数量
    }
}

collection.create_index(
    field_name="embedding",   # 向量字段名
    index_params=index_params,
    index_name="embedding_idx" # 索引名称
)
```

#### 各索引类型详解

**IVF_FLAT（倒排文件平坦索引）**

```python
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}  # 聚类中心数量，通常设为 4*sqrt(N)
}
```
- 原理：使用 K-means 将向量聚类为 nlist 个簇，搜索时只搜索最近的 nprobe 个簇
- 适用场景：中等规模数据集（百万级），需要较好召回率
- 参数：`nlist` 聚类数，搜索时可用 `nprobe` 控制精度

**IVF_SQ8（倒排文件标量量化索引）**

```python
index_params = {
    "index_type": "IVF_SQ8",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
```
- 原理：在 IVF 基础上，将向量从 float32 量化为 int8（每个维度8bit），内存占用减少约 75%
- 适用场景：内存受限但需要较高召回率的场景
- 注意：量化会带来一定的精度损失

**IVF_PQ8（倒排文件乘积量化索引）**

```python
index_params = {
    "index_type": "IVF_PQ8",
    "metric_type": "L2",
    "params": {
        "nlist": 128,    # 聚类数
        "m": 8,          # 乘积量化的子空间数（m必须能整除dim）
        "nbits": 8       # 每个子空间的量化位数
    }
}
```
- 原理：在 IVF 基础上使用乘积量化（Product Quantization），将向量分为 m 个子空间分别量化
- 适用场景：超大规模数据集（亿级），内存极度受限
- 参数：`m` 子空间数量，必须能整除向量维度；`nbits` 量化位数

**HNSW（分层可导航小世界图）**

```python
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 16,           # 每个节点的最大连接数
        "efConstruction": 256  # 构建时的搜索宽度
    }
}
```
- 原理：构建多层图结构，底层包含所有节点，上层为稀疏子集，搜索从顶层向下逐层逼近
- 适用场景：要求低延迟高召回率的场景
- 参数：`M` 影响内存和召回率（通常 12-64）；`efConstruction` 影响构建质量和速度（通常 100-500）
- 搜索参数：`ef` 搜索时的动态候选列表大小

**ANNOY（近似最近邻索引）**

```python
index_params = {
    "index_type": "ANNOY",
    "metric_type": "L2",
    "params": {"n_trees": 8}  # 树的数量
}
```
- 原理：构建多棵随机投影树，搜索时同时搜索多棵树取并集
- 适用场景：中小规模数据集，需要静态索引的场景
- 参数：`n_trees` 树的数量，越多越精确但构建越慢

**SCANN（可扩展最近邻索引）**

```python
index_params = {
    "index_type": "SCANN",
    "metric_type": "L2",
    "params": {
        "nlist": 128,
        "with_raw_data": True   # 是否保存原始数据
    }
}
```
- 原理：Google 提出的 ScaNN 算法，结合了各向异性量化和 IVF
- 适用场景：需要最先进召回率的场景

#### 索引构建与查看

```python
# 构建索引（显式触发）
collection.create_index(field_name="embedding", index_params=index_params)

# 查看索引信息
indexes = collection.indexes
for idx in indexes:
    print(f"索引名: {idx.index_name}")
    print(f"字段: {idx.field_name}")
    print(f"参数: {idx.params}")

# 删除索引
collection.drop_index(index_name="embedding_idx")
```

### 3.4 数据操作

#### insert() — 插入数据

```python
import numpy as np

# 准备数据
data = [
    {
        "id": i,
        "embedding": np.random.rand(768).tolist(),
        "text": f"这是第{i}条文档内容",
        "source": f"doc_{i % 10}"
    }
    for i in range(1000)
]

# 批量插入
result = collection.insert(data=data)

# 返回结果
print(result.insert_count)  # 插入的行数
print(result.primary_keys)  # 主键列表

# 也可以分字段插入
result = collection.insert(
    data=[
        [1, 2, 3],                              # id 列表
        [[0.1, 0.2, ...], [0.3, 0.4, ...], ...], # embedding 列表
        ["文本1", "文本2", "文本3"],               # text 列表
    ]
)
```

#### delete() — 删除数据

```python
# 按主键删除
result = collection.delete(expr="id in [1, 2, 3]")

# 按条件删除
result = collection.delete(expr='source == "doc_0"')

# 删除结果
print(result.delete_count)
```

#### upsert() — 插入或更新

```python
# 如果主键存在则更新，不存在则插入
data = [
    {
        "id": 1,
        "embedding": [0.1, 0.2, ...],
        "text": "更新后的文本",
        "source": "updated_doc"
    }
]

result = collection.upsert(data=data)
print(result.insert_count)  # 新插入的数量
print(result.upsert_count)  # 更新的数量
```

#### query() — 标量查询

```python
# 按条件查询标量数据
results = collection.query(
    expr='source == "doc_0" and id > 50',  # 过滤表达式
    output_fields=["id", "text", "source"],  # 返回字段
    limit=10,                                 # 返回数量限制
)

# 返回格式: [{"id": 51, "text": "...", "source": "doc_0"}, ...]
```

### 3.5 搜索

#### search() — 向量相似度搜索

```python
# 基本向量搜索
search_params = {
    "metric_type": "COSINE",  # 距离度量
    "params": {
        "nprobe": 16,         # IVF系列：搜索的聚类中心数
        # HNSW 使用 "ef": 64
    }
}

results = collection.search(
    data=[[0.1, 0.2, ...]],       # 查询向量列表（支持批量）
    anns_field="embedding",        # 向量字段名
    param=search_params,           # 搜索参数
    limit=10,                      # 返回Top-K结果数
    expr='source == "doc_0"',      # 过滤表达式（可选）
    output_fields=["text", "source"],  # 返回的标量字段
)

# 解析搜索结果
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, 距离: {hit.distance}, 文本: {hit.entity.get('text')}")
```

**搜索参数对照表**：

| 索引类型 | 搜索参数 | 说明 |
|----------|---------|------|
| IVF_FLAT/SQ8/PQ8 | `nprobe` | 搜索的聚类数，值越大召回越高但越慢 |
| HNSW | `ef` | 搜索时的候选列表大小，≥limit，值越大召回越高 |
| ANNOY | `search_k` | 搜索时检查的节点数，-1表示全量搜索 |
| SCANN | `nprobe`, `reorder_k` | 聚类搜索数和重排序数量 |

#### hybrid_search() — 混合搜索

混合搜索支持多向量字段的联合检索，适用于多路召回场景。

```python
from pymilvus import AnnSearchRequest

# 定义第一个搜索请求（稠密向量）
dense_search_param = {
    "data": [[0.1, 0.2, ...]],
    "anns_field": "dense_embedding",
    "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
    "limit": 10,
}
dense_request = AnnSearchRequest(**dense_search_param)

# 定义第二个搜索请求（稀疏向量/BM25）
sparse_search_param = {
    "data": [{"token_1": 0.5, "token_2": 0.3}],  # 稀疏向量
    "anns_field": "sparse_embedding",
    "param": {"metric_type": "IP"},
    "limit": 10,
}
sparse_request = AnnSearchRequest(**sparse_search_param)

# 执行混合搜索（加权融合）
from pymilvus import WeightedRanker

results = collection.hybrid_search(
    reqs=[dense_request, sparse_request],
    ranker=WeightedRanker(0.7, 0.3),  # 稠密向量权重0.7，稀疏向量权重0.3
    limit=10,
    output_fields=["text", "source"],
)

# 解析结果
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, 分数: {hit.distance}")
```

**Ranker 类型**：

- `WeightedRanker(w1, w2, ...)`：加权融合，各路搜索结果按权重加权求和
- `RRFRanker(k)`：倒数排名融合（Reciprocal Rank Fusion），`k` 为平滑常数（默认60）

```python
from pymilvus import RRFRanker

# 使用 RRF 融合
results = collection.hybrid_search(
    reqs=[dense_request, sparse_request],
    ranker=RRFRanker(k=60),
    limit=10,
)
```

### 3.6 分区

分区是 Collection 内部的数据组织方式，通过将数据按特定规则分组存储，可以加速搜索时的数据过滤。

```python
# 创建分区
collection.create_partition(
    name="partition_2024",  # 分区名称
    description="2024年的数据"
)

# 列出所有分区
partitions = collection.partitions
for p in partitions:
    print(f"分区名: {p.name}, 行数: {p.num_entities}")

# 插入数据到指定分区
data = [{"id": 1, "embedding": [...], "text": "..."}]
collection.insert(data=data, partition_name="partition_2024")

# 在指定分区中搜索
results = collection.search(
    data=[[0.1, 0.2, ...]],
    anns_field="embedding",
    param=search_params,
    limit=10,
    partition_names=["partition_2024"],  # 仅在指定分区搜索
)

# 删除分区
collection.drop_partition(name="partition_2024")
```

### 3.7 数据持久化：load() 与 release()

Milvus 采用存储计算分离架构，数据持久化在对象存储中，搜索时需要将数据加载到内存。

```python
# 加载 Collection 到内存（搜索前必须执行）
collection.load(
    replica_number=1,  # 副本数，增加副本可提高QPS
    timeout=None,      # 超时时间
)

# 查看加载状态
load_state = utility.load_state("rag_documents")
# 可能的值: Loaded, Loading, NotLoad

# 释放 Collection 的内存（不需要搜索时释放以节省资源）
collection.release()

# 刷新数据到持久化存储（确保插入的数据可被搜索）
collection.flush()
# 注意：flush 是一个耗时操作，Milvus 也会自动flush
```

**load 与 release 的使用原则**：
- 搜索前必须 `load()`，将索引和数据加载到内存
- 数据插入后需要 `flush()` 或等待自动 flush 才能被搜索到
- 长期不使用的 Collection 可以 `release()` 释放内存
- 增加 `replica_number` 可以提高搜索的 QPS

## 4. LLM 开发中的典型使用场景和代码示例

### 4.1 完整的 RAG 系统

```python
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from openai import OpenAI
import numpy as np

# ========== 初始化 ==========
connections.connect(host="localhost", port="19530")
openai_client = OpenAI(api_key="your-api-key")

# ========== 定义 Collection ==========
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
]
schema = CollectionSchema(fields=fields, description="RAG知识库")
collection = Collection(name="rag_kb", schema=schema)

# 创建 HNSW 索引
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 256}
}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()

# ========== 文档入库 ==========
def embed_texts(texts):
    """调用 OpenAI API 生成文本向量"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

def ingest_documents(documents):
    """将文档切片后入库"""
    # 文档切片（简单按字符数切片）
    chunks = []
    for doc in documents:
        text = doc["text"]
        chunk_size = 500
        for i in range(0, len(text), chunk_size):
            chunks.append({
                "text": text[i:i+chunk_size],
                "source": doc["source"]
            })

    # 批量生成向量
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # 插入 Milvus
    data = [
        {"text": c["text"], "source": c["source"], "embedding": e}
        for c, e in zip(chunks, embeddings)
    ]
    collection.insert(data=data)
    collection.flush()
    print(f"入库完成，共 {len(data)} 个切片")

# ========== 检索与生成 ==========
def rag_query(question, top_k=5):
    """RAG查询流程"""
    # 1. 问题向量化
    question_embedding = embed_texts([question])[0]

    # 2. 向量检索
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(
        data=[question_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text", "source"],
    )

    # 3. 组装上下文
    context = "\n".join([
        f"[来源: {hit.entity.get('source')}]\n{hit.entity.get('text')}"
        for hit in results[0]
    ])

    # 4. LLM 生成回答
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "基于以下参考资料回答用户问题。如果参考资料中没有相关信息，请说明。"},
            {"role": "user", "content": f"参考资料:\n{context}\n\n问题: {question}"}
        ]
    )
    return response.choices[0].message.content

# 使用示例
answer = rag_query("什么是向量数据库？")
print(answer)
```

### 4.2 多模态检索（图文混合）

```python
# 创建支持图文混合的 Collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
]
schema = CollectionSchema(fields=fields, description="多模态知识库")
collection = Collection(name="multimodal_kb", schema=schema)

# 创建索引并加载
collection.create_index(
    field_name="embedding",
    index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
)
collection.load()

# 搜索时可按内容类型过滤
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 16}},
    limit=10,
    expr='content_type == "image"',  # 仅搜索图片
    output_fields=["description", "image_url"],
)
```

### 4.3 使用 MilvusClient 简化 API

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 快速创建 Collection
client.create_collection(
    collection_name="quick_start",
    dimension=768,
    metric_type="COSINE",
    auto_id=True,
)

# 插入数据
data = [
    {"id": i, "vector": np.random.rand(768).tolist(), "text": f"doc_{i}"}
    for i in range(1000)
]
client.insert(collection_name="quick_start", data=data)

# 搜索
results = client.search(
    collection_name="quick_start",
    data=[np.random.rand(768).tolist()],
    limit=5,
    output_fields=["text"],
    search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
)

# 查询
results = client.query(
    collection_name="quick_start",
    filter="id > 500",
    output_fields=["id", "text"],
    limit=10,
)
```

## 5. 数学原理

### 5.1 IVF（Inverted File Index）

IVF 是最经典的近似最近邻搜索算法之一，核心思想是将搜索空间缩小到与查询最相关的子集。

**聚类阶段**：

1. 使用 K-means 算法将向量空间中的 N 个向量聚类为 `nlist` 个簇
2. 每个簇的中心（质心）即为 Voronoi 单元的中心
3. Voronoi 图将空间划分为 nlist 个区域，每个区域中的向量到其质心的距离最近

K-means 聚类目标函数：

$$\min_{C} \sum_{i=1}^{nlist} \sum_{x \in C_i} \|x - \mu_i\|^2$$

其中 $\mu_i$ 是第 $i$ 个聚类的质心，$C_i$ 是第 $i$ 个聚类中的向量集合。

**搜索阶段**：

1. 计算查询向量 $q$ 与所有 `nlist` 个聚类中心的距离
2. 选择距离最近的 `nprobe` 个聚类中心
3. 在这 `nprobe` 个聚类中进行精确搜索（暴力搜索）
4. 返回 Top-K 最近的向量

**复杂度分析**：

- 索引构建：O(N × nlist × iter)，iter 为 K-means 迭代次数
- 搜索：O(nlist + nprobe × N/nlist)，其中第一项为找聚类中心的开销，第二项为簇内搜索的开销
- 当 nprobe = nlist 时，退化为暴力搜索，召回率 100%

**nprobe 的影响**：

| nprobe | 召回率 | 搜索速度 |
|--------|--------|---------|
| 1 | 低 | 最快 |
| nlist/10 | 中等 | 较快 |
| nlist | 100% | 最慢 |

### 5.2 PQ（Product Quantization，乘积量化）

PQ 是一种有损压缩技术，通过将高维向量分解为低维子空间并分别量化来大幅减少内存占用。

**编码阶段**：

1. 将 D 维向量 $x$ 分割为 $m$ 个子向量：$x = [x_1, x_2, ..., x_m]$，每个子向量维度为 $D/m$
2. 对每个子空间独立进行 K-means 聚类，聚类中心数为 256（8bit 编码）
3. 每个子向量被编码为其最近聚类中心的索引（1 byte）
4. 原始向量被编码为 $m$ 个字节的码字：$code(x) = [c_1, c_2, ..., c_m]$

**距离计算（查表法）**：

计算查询向量 $q$ 与数据库向量 $x$ 的距离时：

1. 预计算：对每个子空间 $i$，计算 $q_i$ 与该子空间所有 256 个聚类中心的距离，得到查找表 $L_i$
2. 查表求和：$d(q, x) \approx \sum_{i=1}^{m} L_i[c_i]$

复杂度从暴力搜索的 O(N×D) 降低为 O(N×m)，且无需加载完整向量到内存。

**内存对比**（以 D=768 为例）：

| 编码方式 | 每向量内存 | 压缩比 |
|---------|-----------|--------|
| 原始 float32 | 3072 bytes | 1× |
| SQ8（标量量化） | 768 bytes | 4× |
| PQ8（m=8） | 8 bytes | 384× |
| PQ32（m=32） | 32 bytes | 96× |

### 5.3 HNSW（Hierarchical Navigable Small World）

HNSW 是目前最主流的图索引算法，具有出色的搜索性能和召回率。

**图结构**：

HNSW 构建一个多层图：
- 第 0 层（底层）：包含所有 N 个节点，每个节点最多有 `2M` 个邻居
- 第 l 层（l > 0）：包含第 l-1 层中的一部分节点，每个节点最多有 `M` 个邻居
- 层级分配：每个节点以概率 $P(l) = (1/M)^l$ 被分配到第 l 层及以上

**构建算法**：

1. 插入新节点 $q$ 时，从最高层的入口点开始
2. 贪心搜索找到当前层的最近节点
3. 以该最近节点为起点，在当前层和以下各层分别执行：
   - 在 `efConstruction` 范围内搜索最近邻候选
   - 选择最多 M 个邻居建立连接（启发式选择：优先选距离近且方向多样化的邻居）

**搜索算法**：

1. 从最高层的入口点开始
2. 在每一层贪心搜索：从当前最近的节点出发，检查其邻居中是否有更近的节点
3. 到达第 0 层后，在 `ef` 大小的候选列表中进行广度优先搜索
4. 从候选列表中返回 Top-K 最近的节点

**复杂度**：

- 构建：O(N × log(N) × M × efConstruction)
- 搜索：O(log(N)) 层间遍历 + O(ef × M) 第 0 层搜索
- 总搜索复杂度：O(log(N))

**参数影响**：

| 参数 | 影响 |
|------|------|
| M | 越大图越稠密，召回越高，但内存和构建时间增加 |
| efConstruction | 越大构建质量越好，但构建越慢 |
| ef（搜索时） | 越大召回越高，搜索越慢；应 ≥ limit |

## 6. 代码原理/架构原理

### 6.1 Milvus 整体架构

Milvus 采用云原生的存算分离架构，主要包含以下组件：

```
┌─────────────────────────────────────────────┐
│                  Client                      │
│          (PyMilvus / REST API)               │
└──────────────────┬──────────────────────────┘
                   │ gRPC / HTTP
┌──────────────────▼──────────────────────────┐
│              Access Layer                    │
│           (Proxy / Load Balancer)            │
│  - 请求路由、负载均衡                          │
│  - 结果聚合、SQL 解析                          │
└──────┬──────────┬──────────┬────────────────┘
       │          │          │
┌──────▼───┐ ┌────▼────┐ ┌───▼──────┐
│ QueryNode │ │ DataNode │ │ IndexNode │
│ (搜索执行) │ │ (数据写入) │ │ (索引构建) │
└──────┬───┘ └────┬────┘ └───┬──────┘
       │          │          │
┌──────▼──────────▼──────────▼──────────────┐
│              Storage Layer                  │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │ MetaStore │ │ LogStore │ │ ObjStore  │  │
│  │ (etcd)    │ │ (Kafka/  │ │ (MinIO/   │  │
│  │          │ │  Pulsar) │ │  S3)      │  │
│  └──────────┘ └──────────┘ └───────────┘  │
└────────────────────────────────────────────┘
```

**关键组件职责**：

- **Proxy**：无状态的接入层，负责请求解析、路由、负载均衡和结果聚合
- **QueryNode**：执行搜索和查询请求，维护内存中的段数据（Sealed Segment + Growing Segment）
- **DataNode**：处理数据写入，将 WAL（Write-Ahead Log）中的数据转换为段文件
- **IndexNode**：负责构建向量索引，是 CPU/内存密集型操作
- **etcd**：存储元数据（Collection Schema、索引信息、段信息等）
- **Kafka/Pulsar**：WAL 日志存储，保证数据写入的持久性和顺序性
- **MinIO/S3**：对象存储，存储段文件和索引文件

### 6.2 数据写入流程

```
Client → Proxy → WAL (Kafka) → DataNode → 段文件 → Object Storage
                ↓
              MetaStore (记录段元信息)
```

1. 客户端调用 `insert()`，Proxy 将数据写入 WAL
2. DataNode 消费 WAL 中的数据，积累到一定量后刷盘为 Sealed Segment
3. 刷盘后的 Segment 文件上传到 Object Storage
4. 元信息注册到 MetaStore

### 6.3 搜索执行流程

```
Client → Proxy → QueryNode(s)
                    │
                    ├── Sealed Segment（持久化段，已建索引）
                    └── Growing Segment（增量段，暴力搜索）
```

1. Proxy 接收搜索请求，确定目标 Collection 和分区
2. 将请求路由到包含相应段数据的 QueryNode
3. QueryNode 分别搜索 Sealed Segment（使用索引加速）和 Growing Segment（暴力搜索）
4. Proxy 聚合各 QueryNode 的结果，返回全局 Top-K

### 6.4 PyMilvus SDK 架构

```python
# PyMilvus 的两层 API 设计

# 底层 ORM API（细粒度控制）
from pymilvus import connections, Collection, FieldSchema, DataType
connections.connect(...)
collection = Collection(name="...", schema=schema)
collection.insert(data)
collection.search(...)

# 高层 MilvusClient API（简化操作）
from pymilvus import MilvusClient
client = MilvusClient(uri="...")
client.insert(collection_name="...", data=data)
client.search(collection_name="...", data=...)
```

## 7. 常见注意事项和最佳实践

### 7.1 索引选择指南

| 数据规模 | 延迟要求 | 内存预算 | 推荐索引 |
|---------|---------|---------|---------|
| < 100万 | 极低 | 充足 | HNSW |
| 100万-1亿 | 中等 | 充足 | IVF_FLAT |
| 100万-1亿 | 中等 | 受限 | IVF_SQ8 |
| > 1亿 | 中等 | 极受限 | IVF_PQ8 |
| < 50万 | 低 | 极少 | FLAT（暴力搜索） |

### 7.2 参数调优建议

```python
# HNSW 参数推荐
hnsw_params = {
    "M": 16,              # 通用场景推荐16，高召回场景推荐32-64
    "efConstruction": 256  # 推荐至少200，越高构建质量越好
}
# 搜索时
search_params = {"ef": 64}  # ef >= limit，推荐 2-4 倍 limit

# IVF 参数推荐
ivf_params = {
    "nlist": 128  # 推荐 4*sqrt(N)，N为向量数
}
# 搜索时
search_params = {"nprobe": 16}  # nprobe/nlist 在 10%-20% 较合适
```

### 7.3 常见问题与解决方案

**问题1：插入数据后搜索不到**

```python
# 原因：数据还在 Growing Segment 中或未 flush
# 解决方案：
collection.insert(data)
collection.flush()  # 显式刷盘（生产环境建议让 Milvus 自动flush）
# 重新 load（如果之前已 load，需要等待或重新 load）
```

**问题2：搜索召回率低**

```python
# IVF 系列：增大 nprobe
search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}  # 原来可能是 8 或 16

# HNSW：增大 ef
search_params = {"metric_type": "COSINE", "params": {"ef": 128}}  # 原来可能是 32 或 64
```

**问题3：内存不足**

```python
# 1. 使用压缩索引（IVF_SQ8 替代 IVF_FLAT，或使用 IVF_PQ8）
# 2. 释放不常用的 Collection
collection.release()
# 3. 减少副本数
collection.load(replica_number=1)
```

### 7.4 生产环境最佳实践

1. **批量插入**：每次插入建议 1000-10000 条，避免单条插入
2. **预建索引**：先插入足够量数据再建索引，避免频繁重建
3. **合理分区**：按时间或业务维度分区，搜索时指定分区以减少扫描范围
4. **监控指标**：关注 QPS、延迟 P99、召回率等核心指标
5. **数据备份**：定期备份 Collection Schema 和数据
6. **连接池**：在高并发场景下复用连接，避免频繁创建/销毁连接
7. **异步操作**：索引构建等耗时操作建议异步执行

```python
# 批量插入的最佳实践
batch_size = 5000
for i in range(0, len(all_data), batch_size):
    batch = all_data[i:i+batch_size]
    collection.insert(batch)
    print(f"已插入 {min(i+batch_size, len(all_data))}/{len(all_data)}")

# 最后一次 flush
collection.flush()
```

### 7.5 与 LangChain 集成

```python
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings

# 创建 Milvus 向量存储
vectorstore = Milvus.from_documents(
    documents=documents,       # LangChain Document 列表
    embedding=OpenAIEmbeddings(),
    collection_name="langchain_docs",
    connection_args={"host": "localhost", "port": "19530"},
)

# 相似度搜索
results = vectorstore.similarity_search(
    query="什么是向量数据库？",
    k=5,
)

# 作为检索器使用
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

### 7.6 与 LlamaIndex 集成

```python
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

# 创建 Milvus 向量存储
vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="llamaindex_docs",
    dim=1536,
)

# 构建索引
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

# 查询
query_engine = index.as_query_engine()
response = query_engine.query("什么是向量数据库？")
print(response)
```
