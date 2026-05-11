---
title: "Qdrant 高性能向量数据库"
excerpt: "HNSW索引、Filter过滤系统、Payload索引、量化、FastEmbed集成"
collection: llm-libs
permalink: /llm-libs/36-qdrant
category: vector
toc: true
---


## 1. 库的简介和在LLM开发中的作用

Qdrant（读作"quadrant"）是一个开源的高性能向量相似度搜索引擎和数据库，使用Rust语言开发。它专为大规模向量搜索场景设计，提供了丰富的过滤、排序和优化功能，是生产级LLM应用的理想选择。

### 核心特点

- **Rust实现**：高性能、内存安全、低延迟
- **多种部署模式**：支持内存模式、本地持久化、独立服务、分布式集群
- **高级过滤**：支持丰富的payload过滤条件（范围、匹配、嵌套等）
- **HNSW索引**：内置高性能HNSW近似最近邻算法
- **优化器**：自动优化存储和索引性能
- **多向量支持**：支持同一记录存储多个向量（命名向量）
- **量化支持**：支持标量量化、积量化等压缩技术，降低内存占用

### 在LLM开发中的角色

在LLM应用开发中，Qdrant主要扮演**高性能向量存储与检索引擎**的角色：

1. **大规模RAG系统**：支撑百万级甚至亿级文档的语义检索
2. **多模态搜索**：结合文本、图像等多模态嵌入向量进行跨模态检索
3. **推荐系统**：基于向量相似度的内容推荐
4. **去重与聚类**：利用向量相似度检测重复内容或进行语义聚类
5. **长期记忆**：为AI Agent提供持久化的长期记忆存储

与Chroma相比，Qdrant在性能、可扩展性和过滤能力方面更强，适合对延迟和吞吐量有要求的生产环境。

## 2. 安装方式

### Python客户端安装

```bash
# 安装Qdrant Python客户端
pip install qdrant-client

# 安装fastembed支持（内置嵌入模型）
pip install qdrant-client[fastembed]

# 安装完整依赖
pip install "qdrant-client[all]"
```

### Qdrant服务端安装

```bash
# Docker部署（推荐）
docker pull qdrant/qdrant

# 启动Qdrant服务
docker run -p 6333:6333 -p 6334:6334 \
    -v ./qdrant_storage:/qdrant/storage \
    qdrant/qdrant

# 带配置启动
docker run -p 6333:6333 \
    -v ./qdrant_config:/qdrant/production.yaml \
    qdrant/qdrant

# Docker Compose部署
# docker-compose.yml
```

```yaml
# docker-compose.yml 示例
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC API
    volumes:
      - ./qdrant_storage:/qdrant/storage
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334
```

### 版本检查

```python
from qdrant_client import QdrantClient
import qdrant_client

print(qdrant_client.__version__)  # 查看客户端版本
```

## 3. 核心类/函数/工具的详细说明

### 3.1 QdrantClient - 客户端连接

QdrantClient是所有操作的入口，支持多种连接方式。

#### 内存模式

```python
from qdrant_client import QdrantClient

# 纯内存模式，数据不持久化，适合测试
client = QdrantClient(":memory:")
```

#### 本地持久化模式

```python
from qdrant_client import QdrantClient

# 本地文件存储，数据持久化到磁盘
client = QdrantClient(path="./qdrant_data")
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | str | 本地存储路径 |

#### 远程连接模式

```python
from qdrant_client import QdrantClient

# 连接远程Qdrant服务（REST API）
client = QdrantClient(
    url="http://localhost:6333",  # REST API地址
    api_key="your-api-key",       # 可选的API密钥
    timeout=30                    # 请求超时（秒）
)

# 使用gRPC连接（更高性能）
client = QdrantClient(
    url="http://localhost:6334",  # gRPC地址
    grpc_port=6334,
    prefer_grpc=True,             # 优先使用gRPC
    api_key="your-api-key"
)

# Qdrant Cloud连接
client = QdrantClient(
    url="https://your-cluster-id.qdrant.tech",
    api_key="your-cloud-api-key"
)
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `url` | str | 服务器URL |
| `port` | int | REST API端口，默认6333 |
| `grpc_port` | int | gRPC端口，默认6334 |
| `prefer_grpc` | bool | 是否优先使用gRPC，默认False |
| `api_key` | str | API密钥 |
| `timeout` | int | 请求超时秒数 |
| `https` | bool | 是否使用HTTPS |

#### 客户端管理操作

```python
# 健康检查
health = client.get_cluster_status()
print(health)

# 获取集群信息
collections = client.get_collections()
for col in collections.collections:
    print(f"集合: {col.name}")

# 关闭连接
client.close()
```

### 3.2 Collection - 集合操作

#### create_collection - 创建集合

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(":memory:")

# 基本创建：指定向量维度和距离度量
client.create_collection(
    collection_name="my_documents",
    vectors_config=VectorParams(
        size=384,              # 向量维度，必须与嵌入模型输出一致
        distance=Distance.COSINE  # 距离度量方式
    )
)

# 距离度量选项：
# Distance.COSINE  - 余弦相似度（最常用）
# Distance.EUCLID  - 欧几里得距离（L2）
# Distance.DOT     - 内积（点积）
```

**VectorParams参数说明**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `size` | int | 是 | 向量维度 |
| `distance` | Distance | 是 | 距离度量方式 |
| `hnsw_config` | HnswConfigDiff | 否 | HNSW索引配置 |
| `quantization_config` | ScalarQuantization等 | 否 | 量化配置 |
| `on_disk` | bool | 否 | 向量是否存储在磁盘，默认False |

#### 多向量（命名向量）集合

```python
from qdrant_client.models import Distance, VectorParams

# 创建支持多向量的集合（如同时存储文本和图像向量）
client.create_collection(
    collection_name="multimodal_docs",
    vectors_config={
        "text": VectorParams(size=384, distance=Distance.COSINE),
        "image": VectorParams(size=512, distance=Distance.COSINE)
    }
)
```

#### HNSW参数配置

```python
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

client.create_collection(
    collection_name="optimized_collection",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE,
        hnsw_config=HnswConfigDiff(
            m=16,                # 每层最大连接数，默认16，越大精度越高但内存越大
            ef_construct=100,    # 构建时搜索宽度，默认100，越大索引质量越高
            full_scan_threshold=10000,  # 小于此数量时使用暴力搜索
            on_disk=False        # 索引是否存储在磁盘
        )
    )
)
```

#### 其他集合操作

```python
# 获取集合信息
info = client.get_collection(collection_name="my_documents")
print(f"向量数量: {info.points_count}")
print(f"向量维度: {info.config.params.vectors.size}")
print(f"距离度量: {info.config.params.vectors.distance}")
print(f"索引状态: {info.status}")

# 列出所有集合
collections = client.get_collections()

# 删除集合
client.delete_collection(collection_name="my_documents")

# 更新集合参数
from qdrant_client.models import OptimizersConfigDiff
client.update_collection(
    collection_name="my_documents",
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=20000  # 超过此数量才创建索引
    )
)
```

### 3.3 数据操作

#### 数据模型 - PointStruct

Qdrant中的基本数据单元是Point，由id、vector和payload组成：

```python
from qdrant_client.models import PointStruct

# 创建数据点
point = PointStruct(
    id=1,                                    # 唯一ID（整数或UUID字符串）
    vector=[0.1, 0.2, 0.3, ...],             # 向量
    payload={                                 # 载荷（元数据）
        "title": "文档标题",
        "source": "web",
        "page": 1,
        "tags": ["tech", "ai"]
    }
)
```

#### upsert() - 插入或更新

```python
from qdrant_client.models import PointStruct

# 插入单条数据
client.upsert(
    collection_name="my_documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={"title": "文档1", "source": "web"}
        )
    ]
)

# 批量插入
points = [
    PointStruct(
        id=i,
        vector=[float(j) * 0.1 for j in range(384)],  # 模拟384维向量
        payload={
            "title": f"文档{i}",
            "source": "web" if i % 2 == 0 else "pdf",
            "page": i % 100,
            "tags": ["tech"] if i % 3 == 0 else ["science"]
        }
    )
    for i in range(1, 101)
]

client.upsert(
    collection_name="my_documents",
    points=points
)

# 使用UUID作为ID
import uuid
from qdrant_client.models import PointStruct

client.upsert(
    collection_name="my_documents",
    points=[
        PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.1, 0.2, ...],
            payload={"title": "文档"}
        )
    ]
)

# 多向量upsert
client.upsert(
    collection_name="multimodal_docs",
    points=[
        PointStruct(
            id=1,
            vector={
                "text": [0.1, 0.2, ...],   # 文本向量
                "image": [0.3, 0.4, ...]    # 图像向量
            },
            payload={"title": "多模态文档"}
        )
    ]
)
```

#### 使用Batch批量插入

```python
from qdrant_client.models import Batch

# 使用Batch对象批量插入（更高效）
ids = [1, 2, 3]
vectors = [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
payloads = [
    {"title": "文档1", "source": "web"},
    {"title": "文档2", "source": "pdf"},
    {"title": "文档3", "source": "web"}
]

client.upsert(
    collection_name="my_documents",
    points=Batch(
        ids=ids,
        vectors=vectors,
        payloads=payloads
    )
)

# 大数据集分批插入
import numpy as np

def batch_upsert(client, collection_name, ids, vectors, payloads, batch_size=100):
    """分批插入大量数据"""
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_vectors = vectors[i:i+batch_size]
        batch_payloads = payloads[i:i+batch_size]

        client.upsert(
            collection_name=collection_name,
            points=Batch(
                ids=batch_ids,
                vectors=batch_vectors,
                payloads=batch_payloads
            )
        )
        print(f"已插入 {min(i+batch_size, len(ids))}/{len(ids)} 条记录")
```

#### search() - 向量搜索

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# 基本搜索
results = client.search(
    collection_name="my_documents",
    query_vector=[0.1, 0.2, 0.3, ...],  # 查询向量
    limit=5                                # 返回结果数量
)

# 解析搜索结果
for result in results:
    print(f"ID: {result.id}")
    print(f"得分: {result.score}")           # 相似度得分（越高越相似）
    print(f"载荷: {result.payload}")

# 带过滤条件的搜索
results = client.search(
    collection_name="my_documents",
    query_vector=[0.1, 0.2, 0.3, ...],
    query_filter=Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value="web")
            )
        ]
    ),
    limit=5
)

# 搜索并指定返回字段
results = client.search(
    collection_name="my_documents",
    query_vector=[0.1, 0.2, 0.3, ...],
    with_payload=True,           # 返回payload，默认True
    with_vectors=False,          # 不返回向量，默认False
    limit=5,
    score_threshold=0.7          # 最低相似度阈值
)

# 多向量搜索
results = client.search(
    collection_name="multimodal_docs",
    query_vector=("text", [0.1, 0.2, ...]),  # 指定使用text向量搜索
    limit=5
)
```

**search参数说明**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `collection_name` | str | 是 | 集合名称 |
| `query_vector` | list/tuple | 是 | 查询向量（命名向量用tuple） |
| `query_filter` | Filter | 否 | 过滤条件 |
| `limit` | int | 否 | 返回结果数，默认10 |
| `with_payload` | bool/list | 否 | 是否返回payload，默认True |
| `with_vectors` | bool/list | 否 | 是否返回向量，默认False |
| `score_threshold` | float | 否 | 最低相似度阈值 |
| `search_params` | SearchParams | 否 | 搜索参数（如ef_search） |

#### 搜索参数调优

```python
from qdrant_client.models import SearchParams

results = client.search(
    collection_name="my_documents",
    query_vector=[0.1, 0.2, 0.3, ...],
    search_params=SearchParams(
        hnsw_ef=128,          # 搜索时的ef参数，越大越精确但越慢
        exact=False,          # 是否使用精确搜索（暴力搜索），默认False
        quantization=None     # 量化配置
    ),
    limit=5
)
```

#### scroll() - 分页浏览

```python
# scroll用于浏览和过滤数据，不基于向量相似度
results, next_page_offset = client.scroll(
    collection_name="my_documents",
    limit=10,                          # 每页数量
    offset=None,                       # 偏移量，首次为None
    with_payload=True,
    with_vectors=False
)

for point in results:
    print(f"ID: {point.id}, Payload: {point.payload}")

# 翻页
results, next_page_offset = client.scroll(
    collection_name="my_documents",
    limit=10,
    offset=next_page_offset,           # 使用上一页返回的offset
    with_payload=True
)

# 带过滤条件的scroll
from qdrant_client.models import Filter, FieldCondition, MatchValue

results, _ = client.scroll(
    collection_name="my_documents",
    scroll_filter=Filter(
        must=[
            FieldCondition(key="source", match=MatchValue(value="web"))
        ]
    ),
    limit=100
)
```

#### delete() - 删除数据

```python
from qdrant_client.models import PointIdsList, Filter, FieldCondition, MatchValue

# 按ID删除
client.delete(
    collection_name="my_documents",
    points_selector=PointIdsList(
        points=[1, 2, 3]   # 要删除的ID列表
    )
)

# 按条件删除
client.delete(
    collection_name="my_documents",
    points_selector=Filter(
        must=[
            FieldCondition(key="source", match=MatchValue(value="deprecated"))
        ]
    )
)

# 删除所有数据（清空集合）
client.delete(
    collection_name="my_documents",
    points_selector=Filter(
        must=[]  # 空条件匹配所有
    )
)
```

#### retrieve() - 按ID获取

```python
# 按ID获取数据点
points = client.retrieve(
    collection_name="my_documents",
    ids=[1, 2, 3],
    with_payload=True,
    with_vectors=False
)

for point in points:
    print(f"ID: {point.id}, Payload: {point.payload}")
```

#### set_payload() / delete_payload() - 修改载荷

```python
from qdrant_client.models import PointIdsList

# 更新载荷（添加或更新字段）
client.set_payload(
    collection_name="my_documents",
    payload={
        "status": "processed",
        "processed_at": "2026-01-01"
    },
    points=[1, 2, 3]  # 要更新的ID列表
)

# 删除载荷字段
client.delete_payload(
    collection_name="my_documents",
    keys=["status", "processed_at"],  # 要删除的字段名
    points=[1, 2, 3]
)

# 清除所有载荷
client.clear_payload(
    collection_name="my_documents",
    points_selector=PointIdsList(points=[1, 2, 3])
)
```

#### update_vectors() - 更新向量

```python
from qdrant_client.models import PointVectors

# 更新已有数据点的向量（不改载荷）
client.update_vectors(
    collection_name="my_documents",
    points=[
        PointVectors(
            id=1,
            vector=[0.5, 0.6, 0.7, ...]  # 新向量
        )
    ]
)
```

### 3.4 过滤 - Filter系统

Qdrant的过滤系统非常强大，支持丰富的条件组合。

#### FieldCondition - 字段条件

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, MatchExcept, Range

# === 精确匹配 ===
Filter(must=[
    FieldCondition(key="source", match=MatchValue(value="web"))
])

# === 多值匹配（类似IN） ===
Filter(must=[
    FieldCondition(key="source", match=MatchAny(any=["web", "pdf"]))
])

# === 排除匹配（类似NOT IN） ===
Filter(must=[
    FieldCondition(key="source", match=MatchExcept(except_=["deprecated", "draft"]))
])
```

#### Range - 范围条件

```python
from qdrant_client.models import Filter, FieldCondition, Range

# 数值范围
Filter(must=[
    FieldCondition(
        key="page",
        range=Range(
            gte=1,      # 大于等于
            gt=None,    # 大于
            lte=100,    # 小于等于
            lt=None     # 小于
        )
    )
])

# 时间范围
Filter(must=[
    FieldCondition(
        key="timestamp",
        range=Range(gte=1700000000, lt=1700100000)
    )
])
```

#### 嵌套对象过滤

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, NestedCondition
from qdrant_client.models import NestedFieldCondition

# 过滤嵌套payload
# 假设payload结构: {"metadata": {"author": "张三", "year": 2024}}
Filter(must=[
    FieldCondition(
        key="metadata.author",
        match=MatchValue(value="张三")
    )
])

# 使用NestedCondition处理数组中的对象
# 假设payload结构: {"comments": [{"user": "A", "score": 5}, {"user": "B", "score": 3}]}
Filter(must=[
    NestedCondition(
        nested=NestedFieldCondition(
            path="comments",
            filter=Filter(
                must=[
                    FieldCondition(key="comments.user", match=MatchValue(value="A")),
                    FieldCondition(key="comments.score", range=Range(gte=4))
                ]
            )
        )
    )
])
```

#### 逻辑组合

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# must - 所有条件必须满足（AND）
Filter(must=[
    FieldCondition(key="source", match=MatchValue(value="web")),
    FieldCondition(key="page", range=Range(gte=1, lte=50))
])

# should - 至少满足一个条件（OR）
Filter(should=[
    FieldCondition(key="source", match=MatchValue(value="web")),
    FieldCondition(key="source", match=MatchValue(value="pdf"))
])

# must_not - 所有条件必须不满足（NOT）
Filter(must_not=[
    FieldCondition(key="status", match=MatchValue(value="deprecated"))
])

# 复杂组合
Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="tech")),
    ],
    should=[
        FieldCondition(key="source", match=MatchValue(value="web")),
        FieldCondition(key="source", match=MatchValue(value="pdf")),
    ],
    must_not=[
        FieldCondition(key="status", match=MatchValue(value="deprecated"))
    ],
    min_should=1  # should条件至少满足几个
)
```

#### Payload数组过滤

```python
# 检查数组是否包含某值
from qdrant_client.models import FieldCondition, MatchValue

# payload: {"tags": ["tech", "ai", "python"]}
FieldCondition(key="tags", match=MatchValue(value="ai"))  # 匹配包含"ai"的记录

# 数组大小过滤
from qdrant_client.models import PayloadField, FieldCondition, Range

FieldCondition(key="tags", range=Range(gte=2))  # tags数组至少有2个元素
```

#### 文本过滤

```python
from qdrant_client.models import FieldCondition, MatchText

# 文本匹配（分词后匹配任一词）
FieldCondition(key="title", match=MatchText(text="机器学习"))
```

### 3.5 Payload - 元数据存储和过滤

Payload是Qdrant中与向量关联的元数据，支持丰富的数据类型：

```python
from qdrant_client.models import PointStruct

# Payload支持的数据类型
point = PointStruct(
    id=1,
    vector=[0.1, 0.2, ...],
    payload={
        # 基本类型
        "title": "文档标题",               # 字符串
        "page": 42,                         # 整数
        "score": 0.95,                      # 浮点数
        "is_published": True,               # 布尔值

        # 数组
        "tags": ["tech", "ai", "python"],   # 字符串数组
        "scores": [0.9, 0.8, 0.7],          # 数值数组

        # 嵌套对象
        "metadata": {
            "author": "张三",
            "department": {
                "name": "AI Lab",
                "code": "AIL"
            }
        },

        # null值
        "optional_field": None
    }
)
```

#### Payload索引

为了提高过滤性能，可以为payload字段创建索引：

```python
# 创建字段索引
from qdrant_client.models import PayloadIndexType, PayloadSchemaType

# 为字段创建标量索引（适用于精确匹配）
client.create_payload_index(
    collection_name="my_documents",
    field_name="source",
    field_schema=PayloadSchemaType.KEYWORD  # 关键字类型
)

# 为数值字段创建索引
client.create_payload_index(
    collection_name="my_documents",
    field_name="page",
    field_schema=PayloadSchemaType.INTEGER
)

# 为浮点数字段创建索引
client.create_payload_index(
    collection_name="my_documents",
    field_name="score",
    field_schema=PayloadSchemaType.FLOAT
)

# 为文本字段创建全文索引
client.create_payload_index(
    collection_name="my_documents",
    field_name="title",
    field_schema=PayloadSchemaType.TEXT
)

# 为嵌套字段创建索引
client.create_payload_index(
    collection_name="my_documents",
    field_name="metadata.author",
    field_schema=PayloadSchemaType.KEYWORD
)

# 删除索引
client.delete_payload_index(
    collection_name="my_documents",
    field_name="source"
)
```

### 3.6 优化 - Optimize

```python
# 手动触发优化
client.update_collection(
    collection_name="my_documents",
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=20000,       # 超过此数量才创建HNSW索引
        memmap_threshold=50000,         # 超过此数量使用内存映射
        vacuum_threshold=10000          # 超过此数量执行vacuum
    )
)

# 查看优化状态
info = client.get_collection(collection_name="my_documents")
print(f"状态: {info.status}")  # green, yellow, red
print(f"优化器状态: {info.optimizer_status}")
```

## 4. 在LLM开发中的典型使用场景和代码示例

### 4.1 RAG系统 - 完整实现

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# === 1. 初始化 ===
client = QdrantClient(path="./rag_qdrant")
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
COLLECTION_NAME = "knowledge_base"
VECTOR_SIZE = 384

# 创建集合
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE
    )
)

# === 2. 文档处理 ===
def process_and_index_documents(documents):
    """处理文档并索引到Qdrant"""
    points = []
    for i, doc in enumerate(documents):
        # 文本分块
        chunks = chunk_text(doc["content"], chunk_size=300, overlap=50)

        for j, chunk in enumerate(chunks):
            # 生成向量
            vector = encoder.encode(chunk).tolist()

            points.append(PointStruct(
                id=i * 1000 + j,  # 生成唯一ID
                vector=vector,
                payload={
                    "doc_id": doc["id"],
                    "chunk_index": j,
                    "text": chunk,
                    "source": doc.get("source", "unknown"),
                    "category": doc.get("category", "general")
                }
            ))

    # 批量插入
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)

# 示例文档
documents = [
    {
        "id": "doc1",
        "content": "Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年发布。Python的设计哲学强调代码的可读性和简洁性...",
        "source": "python_intro.pdf",
        "category": "programming"
    },
    {
        "id": "doc2",
        "content": "机器学习是人工智能的一个分支，它使用算法和统计模型让计算机系统从数据中自动学习和改进，而无需显式编程...",
        "source": "ml_basics.pdf",
        "category": "ai"
    }
]

count = process_and_index_documents(documents)
print(f"已索引 {count} 个文本块")

# === 3. 语义检索 ===
def search_context(query, n_results=5, category=None):
    """语义检索相关上下文"""
    query_vector = encoder.encode(query).tolist()

    # 构建过滤条件
    query_filter = None
    if category:
        query_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=n_results,
        score_threshold=0.5  # 最低相似度阈值
    )

    return [
        {
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "score": hit.score
        }
        for hit in results
    ]

# === 4. RAG问答 ===
def rag_answer(query, llm_func, category=None):
    """RAG问答流程"""
    # 检索相关上下文
    contexts = search_context(query, n_results=3, category=category)

    if not contexts:
        return "未找到相关参考资料", []

    # 构建提示
    context_text = "\n\n".join([
        f"[来源: {c['source']}]\n{c['text']}" for c in contexts
    ])

    prompt = f"""基于以下参考资料回答问题。请标注信息来源。

参考资料：
{context_text}

问题：{query}
回答："""

    answer = llm_func(prompt)
    return answer, contexts

# 使用示例
# answer, sources = rag_answer("什么是机器学习？", your_llm_func)
```

### 4.2 使用FastEmbed内置嵌入

```python
from qdrant_client import QdrantClient

# 使用FastEmbed内置嵌入模型（无需手动安装sentence-transformers）
client = QdrantClient(
    ":memory:",
    fastembed_model="BAAI/bge-small-en-v1.5"  # 指定FastEmbed模型
)

# 创建集合时自动使用FastEmbed维度
client.create_collection(
    collection_name="fastembed_docs",
    vectors_config={
        "size": 384,
        "distance": "COSINE"
    }
)

# 直接使用文本插入（自动嵌入）
client.add(
    collection_name="fastembed_docs",
    documents=["Python编程语言简介", "机器学习基础教程"],
    metadata=[{"source": "web"}, {"source": "pdf"}],
    ids=[1, 2]
)

# 文本搜索（自动嵌入）
results = client.query(
    collection_name="fastembed_docs",
    query_text="编程学习",
    limit=5
)
```

### 4.3 多模态RAG

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(path="./multimodal_db")

# 创建多向量集合
client.create_collection(
    collection_name="multimodal_docs",
    vectors_config={
        "text": VectorParams(size=384, distance=Distance.COSINE),
        "image": VectorParams(size=512, distance=Distance.COSINE)
    }
)

# 插入多模态数据
from text_encoder import encode_text  # 假设的文本编码器
from image_encoder import encode_image  # 假设的图像编码器

client.upsert(
    collection_name="multimodal_docs",
    points=[
        PointStruct(
            id=1,
            vector={
                "text": encode_text("一只橘猫坐在窗台上"),
                "image": encode_image("cat.jpg")
            },
            payload={
                "description": "一只橘猫坐在窗台上",
                "image_path": "cat.jpg",
                "type": "photo"
            }
        )
    ]
)

# 文本到图像搜索
text_query = encode_text("可爱的猫咪")
results = client.search(
    collection_name="multimodal_docs",
    query_vector=("image", text_query),  # 用文本向量搜索图像向量
    limit=5
)

# 图像到文本搜索
image_query = encode_image("query_cat.jpg")
results = client.search(
    collection_name="multimodal_docs",
    query_vector=("text", image_query),  # 用图像向量搜索文本向量
    limit=5
)
```

### 4.4 AI Agent长期记忆

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range
from datetime import datetime
from sentence_transformers import SentenceTransformer

class AgentMemory:
    """AI Agent长期记忆系统"""

    def __init__(self, storage_path="./agent_memory"):
        self.client = QdrantClient(path=storage_path)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = "agent_memory"

        # 创建集合
        try:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        except Exception:
            pass  # 集合已存在

    def store(self, content, memory_type="observation", importance=5, agent_id="default"):
        """存储记忆"""
        vector = self.encoder.encode(content).tolist()
        timestamp = int(datetime.now().timestamp())

        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=timestamp,  # 用时间戳作为ID
                vector=vector,
                payload={
                    "content": content,
                    "type": memory_type,       # observation, action, reflection
                    "importance": importance,  # 1-10的重要度
                    "agent_id": agent_id,
                    "timestamp": timestamp
                }
            )]
        )

    def recall(self, query, n_results=5, memory_type=None, min_importance=3):
        """检索相关记忆"""
        query_vector = self.encoder.encode(query).tolist()

        conditions = [
            FieldCondition(key="agent_id", match=MatchValue(value="default")),
            FieldCondition(key="importance", range=Range(gte=min_importance))
        ]
        if memory_type:
            conditions.append(
                FieldCondition(key="type", match=MatchValue(value=memory_type))
            )

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            query_filter=Filter(must=conditions),
            limit=n_results
        )

        return [
            {
                "content": hit.payload["content"],
                "type": hit.payload["type"],
                "importance": hit.payload["importance"],
                "score": hit.score
            }
            for hit in results
        ]

    def reflect(self):
        """反思：检索重要记忆进行总结"""
        all_memories = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="importance", range=Range(gte=7))]
            ),
            limit=20
        )
        return all_memories

# 使用示例
memory = AgentMemory()
memory.store("用户偏好使用Python进行数据分析", importance=8)
memory.store("当前任务：构建RAG系统", memory_type="action", importance=9)
memory.store("API调用失败，需要重试", memory_type="observation", importance=4)

# 检索相关记忆
relevant = memory.recall("数据分析工具推荐", min_importance=5)
```

### 4.5 与LangChain集成

```python
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

# 创建嵌入
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 创建Qdrant客户端
client = QdrantClient(url="http://localhost:6333")

# 从文本创建向量存储
vectorstore = QdrantVectorStore.from_texts(
    texts=["文档1内容", "文档2内容", "文档3内容"],
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="langchain_docs"
)

# 相似度搜索
results = vectorstore.similarity_search(
    query="查询内容",
    k=3
)

# 创建检索器
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)

# RAG链
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=retriever
)
```

## 5. 数学原理

### 5.1 HNSW (Hierarchical Navigable Small World)

HNSW是Qdrant核心的近似最近邻搜索算法，是对传统NSW（Navigable Small World）算法的层次化扩展。

#### 5.1.1 基本思想

HNSW构建一个多层图结构，每一层都是一个NSW图：
- **底层（Layer 0）**：包含所有数据点，连接最密集
- **上层**：包含逐步减少的数据点，连接逐步稀疏
- **最顶层**：只包含少量数据点，作为搜索入口

#### 5.1.2 图的构建

**层次分配**：每个新插入的点被分配一个随机层级 `l`，层级由指数衰减分布决定：

$$P(l = L) = \frac{1}{M^L}$$

其中M是最大连接数参数。大多数点只在底层（l=0），少数点会出现在更高层。

**插入过程**：
1. 从最顶层的入口点开始
2. 在当前层贪心搜索找到最近的节点
3. 下降到下一层，以上一层找到的最近节点为起点继续贪心搜索
4. 到达目标层后，连接该层中最近的M个邻居
5. 如果某邻居的连接数超过M，移除最远的连接

```
Layer 2:    A───────────────B         (少量节点，远程连接)
            │               │
Layer 1:    A───C────D──────B───E     (中等数量节点)
            │   │    │      │   │
Layer 0:    A─C─F─D─G─H─B─I─E─J─K    (所有节点，密集连接)
```

#### 5.1.3 搜索过程

1. **从顶层开始**：以最顶层的入口点为起点
2. **贪心搜索**：在当前层中，不断移动到距离查询点更近的邻居节点
3. **层间切换**：当在当前层无法找到更近的节点时，下降到下一层
4. **底层精确搜索**：在底层（Layer 0）使用ef_search参数控制搜索宽度，返回最近邻结果

**搜索复杂度**：O(log N)，远优于暴力搜索的O(N)。

#### 5.1.4 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `M` | 16 | 每个节点的最大连接数（每层）。增大M提高搜索精度，但增加内存和构建时间 |
| `ef_construct` | 100 | 构建索引时的搜索宽度。增大ef_construct提高索引质量，但减慢构建速度 |
| `ef_search` | 自动 | 搜索时的搜索宽度。增大ef_search提高搜索精度，但减慢搜索速度 |

**参数选择建议**：

```python
# 高精度场景（如医疗、法律RAG）
HnswConfigDiff(m=32, ef_construct=200)
# 搜索时: SearchParams(hnsw_ef=256)

# 平衡场景（一般RAG应用）
HnswConfigDiff(m=16, ef_construct=100)  # 默认值
# 搜索时: SearchParams(hnsw_ef=128)

# 高吞吐场景（推荐系统）
HnswConfigDiff(m=8, ef_construct=50)
# 搜索时: SearchParams(hnsw_ef=64)
```

### 5.2 距离度量详解

#### 余弦相似度

$$\text{cos}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

Qdrant中余弦距离 = `1 - cos(A,B)`，值越小越相似。余弦距离范围为[0, 2]。

#### 欧几里得距离 (EUCLID)

$$d(A, B) = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}$$

值越小越相似，范围为[0, +∞)。

#### 内积 (DOT)

$$ip(A, B) = \sum_{i=1}^{n} a_i \cdot b_i$$

Qdrant中内积距离 = `-ip(A,B)`（取负后越小越相似）。

```python
# Python验证三种距离的关系
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 3.0, 4.0])

# L2距离
l2 = np.sqrt(np.sum((a - b) ** 2))  # 1.732

# 余弦相似度 → 余弦距离
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
cos_dist = 1 - cos_sim  # 0.025

# 内积 → 内积距离
ip = np.dot(a, b)  # 20.0
ip_dist = -ip  # -20.0
```

### 5.3 量化 (Quantization)

Qdrant支持多种量化方法来减少内存占用：

#### 标量量化 (Scalar Quantization)

将float32向量压缩为int8，内存减少4倍：

```python
from qdrant_client.models import ScalarQuantization, ScalarQuantizationType, ScalarQuantile

client.create_collection(
    collection_name="quantized_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        type=ScalarQuantizationType.INT8,
        quantile=ScalarQuantile(0.99),  # 量化范围覆盖99%的数据
        always_ram=True                  # 量化数据始终在内存中
    )
)
```

#### 二值量化 (Binary Quantization)

将向量压缩为1bit，内存减少32倍，速度大幅提升：

```python
from qdrant_client.models import BinaryQuantization

client.create_collection(
    collection_name="binary_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=BinaryQuantization(
        always_ram=True
    )
)
```

## 6. 代码原理/架构原理

### 6.1 Qdrant整体架构

```
┌───────────────────────────────────────┐
│          Client Layer                 │
│   (REST API / gRPC / Python SDK)      │
├───────────────────────────────────────┤
│        Collection Manager             │
│   (集合CRUD + 配置管理)               │
├───────────────────────────────────────┤
│     Segment Storage Engine            │
│  ┌─────────────┬─────────────────┐    │
│  │ Vector Index│ Payload Index   │    │
│  │ (HNSW)      │ (Field Indexes) │    │
│  ├─────────────┼─────────────────┤    │
│  │ Vector Store│ Payload Store   │    │
│  │ (mmap/RAM)  │ (mmap/RAM)     │    │
│  └─────────────┴─────────────────┘    │
├───────────────────────────────────────┤
│          Optimizer                    │
│  (段合并 + 索引构建 + 增量更新)       │
├───────────────────────────────────────┤
│        Storage Backend                │
│   (文件系统 / WAL / 快照)             │
└───────────────────────────────────────┘
```

### 6.2 Segment存储模型

Qdrant使用Segment作为数据存储的基本单元：

- **Writable Segment**：接收新数据的活跃段，数据未优化
- **Read-only Segment**：已优化的只读段，支持高效搜索
- **Optimizer**：自动将Writable Segment优化为Read-only Segment

```python
# 查看段信息
info = client.get_collection(collection_name="my_documents")
print(f"段数量: {info.segments_count}")
```

### 6.3 查询执行流程

```
1. 接收搜索请求（向量 + 过滤条件 + limit）
   ↓
2. 对每个Segment并行执行搜索
   │
   ├─→ 若有过滤条件，先在Payload Index中预过滤
   │    获取候选点ID集合
   │
   ├─→ 在HNSW索引中搜索最近邻
   │    - 从入口点开始贪心搜索
   │    - 遇到不在候选集中的点则跳过
   │    - 搜索ef_search个候选点
   │
   └─→ 合并各Segment的搜索结果
   ↓
3. 全局排序并返回Top-K结果
```

### 6.4 优化器工作原理

Qdrant的优化器自动管理段的合并和索引构建：

1. **增量优化**：当Writable Segment达到阈值时触发
2. **合并优化**：将多个小Segment合并为大Segment
3. **Vacuum优化**：清理已删除点的空间
4. **索引构建**：当数据量超过阈值时自动构建HNSW索引

```python
# 优化器配置
from qdrant_client.models import OptimizersConfigDiff

client.update_collection(
    collection_name="my_documents",
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=20000,       # 超过2万条才建HNSW索引
        memmap_threshold=50000,         # 超过5万条使用内存映射
        vacuum_threshold=10000,         # 超过1万条执行vacuum
        flush_interval_sec=5,           # 每5秒刷盘一次
        max_optimization_threads=1      # 优化线程数
    )
)
```

### 6.5 一致性与持久化

```python
from qdrant_client.models import WriteOrdering

# 写入一致性控制
client.upsert(
    collection_name="my_documents",
    points=points,
    wait=True  # 等待写入确认（强一致性）
)

# 不等待确认（最终一致性，更高吞吐）
client.upsert(
    collection_name="my_documents",
    points=points,
    wait=False
)

# 有序写入
client.upsert(
    collection_name="my_documents",
    points=points,
    ordering=WriteOrdering.STRONG  # 强有序
)
```

## 7. 常见注意事项和最佳实践

### 7.1 向量维度一致性

```python
# ❌ 错误：向量维度与集合定义不匹配
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# 使用768维向量插入 → 报错！
client.upsert(
    collection_name="docs",
    points=[PointStruct(id=1, vector=[0.1] * 768, payload={})]
)

# ✅ 正确：确保向量维度与集合定义一致
vector = encoder.encode("文本")  # 384维
client.upsert(
    collection_name="docs",
    points=[PointStruct(id=1, vector=vector.tolist(), payload={})]
)
```

### 7.2 ID管理

```python
import uuid

# ✅ 推荐使用UUID避免冲突
point_id = str(uuid.uuid4())

# ✅ 使用确定性ID（便于去重和更新）
def deterministic_id(text):
    """基于内容生成确定性ID"""
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()

doc_id = deterministic_id("文档内容")

# ⚠️ 整数ID需要自己管理唯一性
# 如果不确定是否重复，使用upsert而非insert
```

### 7.3 批量操作优化

```python
from qdrant_client.models import Batch
import uuid

# ✅ 大批量数据分批插入
def bulk_insert(client, collection_name, texts, encoder, batch_size=100):
    """高效批量插入"""
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        vectors = encoder.encode(batch_texts).tolist()

        client.upsert(
            collection_name=collection_name,
            points=Batch(
                ids=[str(uuid.uuid4()) for _ in batch_texts],
                vectors=vectors,
                payloads=[{"text": t, "index": i+j} for j, t in enumerate(batch_texts)]
            )
        )

# ⚠️ 控制批量大小
# 过大的batch可能导致超时
# 推荐100-500条一批
```

### 7.4 过滤性能优化

```python
# ✅ 为常用过滤字段创建索引
client.create_payload_index(
    collection_name="docs",
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name="docs",
    field_name="timestamp",
    field_schema=PayloadSchemaType.INTEGER
)

# ✅ 过滤条件顺序：先最严格的条件
# Qdrant会自动优化，但合理设计过滤条件有助于性能
Filter(must=[
    FieldCondition(key="category", match=MatchValue(value="tech")),  # 高选择性
    FieldCondition(key="year", range=Range(gte=2024))               # 低选择性
])
```

### 7.5 内存管理

```python
# 向量存储在磁盘（适合大数据集）
from qdrant_client.models import VectorParams, Distance

client.create_collection(
    collection_name="large_collection",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE,
        on_disk=True  # 向量存储在磁盘，减少内存占用
    )
)

# 使用量化减少内存
from qdrant_client.models import ScalarQuantization, ScalarQuantizationType

client.create_collection(
    collection_name="quantized_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        type=ScalarQuantizationType.INT8,
        always_ram=True
    )
)
```

### 7.6 连接管理

```python
# ✅ 复用客户端实例
client = QdrantClient(url="http://localhost:6333")

# 多个操作使用同一个client
client.create_collection(...)
client.upsert(...)
client.search(...)

# ✅ 使用完毕后关闭
client.close()

# ❌ 不要频繁创建和销毁客户端
# for i in range(100):
#     client = QdrantClient(url="http://localhost:6333")  # 浪费资源
#     client.search(...)
#     client.close()
```

### 7.7 错误处理

```python
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

client = QdrantClient(url="http://localhost:6333")

# 安全的集合创建
def ensure_collection(client, name, vectors_config):
    """确保集合存在"""
    try:
        client.get_collection(collection_name=name)
    except UnexpectedResponse as e:
        if e.status_code == 404:
            client.create_collection(
                collection_name=name,
                vectors_config=vectors_config
            )
        else:
            raise

# 安全的upsert
def safe_upsert(client, collection_name, points, max_retries=3):
    """带重试的upsert"""
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=collection_name, points=points)
            return True
        except UnexpectedResponse as e:
            if attempt < max_retries - 1:
                print(f"重试 {attempt + 1}/{max_retries}: {e}")
                import time
                time.sleep(2 ** attempt)  # 指数退避
            else:
                raise
    return False
```

### 7.8 常见陷阱总结

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| 向量维度不匹配 | 插入向量的维度与集合定义不一致 | 确保嵌入模型输出维度与集合配置匹配 |
| 未创建Payload索引 | 过滤查询缓慢 | 为常用过滤字段创建索引 |
| 批量过大 | 超时或内存溢出 | 控制batch_size在100-500 |
| 频繁创建客户端 | 浪费连接资源 | 复用客户端实例 |
| 搜索未设score_threshold | 返回大量低质量结果 | 设置合理的score_threshold |
| 忽略优化器配置 | 大数据集性能下降 | 根据数据量调整优化器参数 |
| 混用ID类型 | 整数和字符串ID混用 | 统一使用UUID字符串或自增整数 |
| wait=False丢失数据 | 写入未确认就重启 | 关键数据使用wait=True |

### 7.9 Qdrant vs Chroma 选择指南

| 特性 | Chroma | Qdrant |
|------|--------|--------|
| 部署复杂度 | 零配置 | 需启动服务（或使用内存模式） |
| 数据规模 | 万级 | 百万到亿级 |
| 过滤能力 | 基础元数据过滤 | 丰富嵌套过滤、全文搜索 |
| 多向量 | 不支持 | 支持命名向量 |
| 量化 | 不支持 | 标量/积/二值量化 |
| 分布式 | 不支持 | 支持集群 |
| 适用场景 | 原型开发、小规模应用 | 生产环境、大规模应用 |
| 嵌入函数 | 内置多种 | 需外部嵌入或FastEmbed |
