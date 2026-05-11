---
title: "pgvector PostgreSQL向量扩展"
excerpt: "vector/halfvec/sparsevec类型、HNSW/IVFFlat索引、SQL距离操作符"
collection: llm-libs
permalink: /llm-libs/38-pgvector
category: vector
---


## 1. 库的简介和在LLM开发中的作用

pgvector 是 PostgreSQL 的开源向量相似度搜索扩展，将高性能向量搜索能力无缝集成到成熟的关系数据库中。它支持精确和近似最近邻搜索、多种向量类型和距离函数，并完全兼容 PostgreSQL 的 ACID 事务、JOIN、备份恢复等特性。

在 LLM 开发中，pgvector 是构建 RAG 系统的独特选择——与专用向量数据库（Qdrant、Milvus）不同，pgvector 让你在已有的 PostgreSQL 基础设施上直接实现向量搜索，无需引入新的数据存储。这种"数据库+向量"的统一方案特别适合：
- 已有 PostgreSQL 基础设施，不想增加新组件的团队
- 需要在关系查询和向量搜索之间做复杂 JOIN 的场景
- 对数据一致性和事务性有严格要求的业务

**核心价值：**
- 零额外基础设施：复用现有 PostgreSQL，运维成本为零
- 事务一致性：向量操作和关系操作在同一事务中
- SQL 生态完整：JOIN、过滤、聚合、窗口函数等全部可用
- 多种向量类型：`vector`（单精度）、`halfvec`（半精度）、`sparsevec`（稀疏）、`bit`（二值）
- 多种索引：HNSW（高性能）和 IVFFlat（低内存）
- 30+ 语言驱动支持

## 2. 安装方式

### 安装扩展

```bash
# 从源码编译安装（Linux/Mac）
cd /tmp
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install

# Ubuntu/Debian APT 安装
sudo apt install postgresql-18-pgvector

# macOS Homebrew
brew install pgvector

# Docker（推荐快速开始）
docker pull pgvector/pgvector:pg18-trixie
docker run -p 5432:5432 pgvector/pgvector:pg18-trixie

# conda-forge
conda install -c conda-forge pgvector
```

### 安装 Python 客户端

```bash
pip install pgvector psycopg[binary]   # psycopg3（推荐）
pip install pgvector psycopg2-binary   # psycopg2（旧版）
pip install pgvector asyncpg           # 异步客户端
```

### 启用扩展

```sql
-- 在每个需要使用pgvector的数据库中执行
CREATE EXTENSION vector;

-- 检查版本
SELECT extversion FROM pg_extension WHERE extname = 'vector';

-- 升级扩展
ALTER EXTENSION vector UPDATE;
```

## 3. 核心类/函数/工具的详细说明

### 3.1 向量类型

#### vector — 单精度向量

```sql
-- 创建表时指定维度
CREATE TABLE items (
    id bigserial PRIMARY KEY,
    content TEXT,
    embedding vector(1536)    -- 1536维单精度向量（如OpenAI text-embedding-3-small）
);

-- 无维度约束（同一列可存储不同维度的向量）
CREATE TABLE embeddings (
    model_id bigint,
    item_id bigint,
    embedding vector,         -- 不指定维度
    PRIMARY KEY (model_id, item_id)
);
```

| 属性 | 值 |
|------|---|
| 存储 | `4 × 维度 + 8` 字节 |
| 元素类型 | 单精度浮点（float32） |
| 最大维度 | 16,000 |
| 最大可索引维度 | 2,000 |

#### halfvec — 半精度向量

```sql
CREATE TABLE items (
    id bigserial PRIMARY KEY,
    embedding halfvec(1536)   -- 半精度，存储减半
);
```

| 属性 | 值 |
|------|---|
| 存储 | `2 × 维度 + 8` 字节 |
| 元素类型 | 半精度浮点（float16） |
| 最大维度 | 16,000 |
| 最大可索引维度 | 4,000 |

#### sparsevec — 稀疏向量

```sql
CREATE TABLE items (
    id bigserial PRIMARY KEY,
    embedding sparsevec(30000)   -- 30000维稀疏向量（如SPLADE）
);

-- 插入格式：{index:value,...}/维度（索引从1开始）
INSERT INTO items (embedding) VALUES ('{1:0.5,100:0.3,500:0.8}/30000');
```

| 属性 | 值 |
|------|---|
| 存储 | `8 × 非零元素数 + 16` 字节 |
| 最大非零元素 | 16,000 |
| 最大可索引非零元素 | 1,000 |

#### bit — 二值向量

```sql
CREATE TABLE items (
    id bigserial PRIMARY KEY,
    embedding bit(768)        -- 二值量化向量
);

INSERT INTO items (embedding) VALUES ('10110010...');
```

| 属性 | 值 |
|------|---|
| 存储 | `维度 / 8 + 8` 字节 |
| 最大可索引维度 | 64,000 |

### 3.2 距离操作符

| 操作符 | 名称 | 公式 | 适用类型 |
|--------|------|------|---------|
| `<->` | 欧氏距离（L2） | $\sqrt{\sum(a_i - b_i)^2}$ | vector, halfvec, sparsevec |
| `<=>` | 余弦距离 | $1 - \cos(a, b)$ | vector, halfvec, sparsevec |
| `<#>` | 负内积 | $-(a \cdot b)$ | vector, halfvec, sparsevec |
| `<+>` | 曼哈顿距离（L1） | $\sum|a_i - b_i|$ | vector, halfvec, sparsevec |
| `<~>` | 汉明距离 | 不同位数 | bit |
| `<%>` | 杰卡德距离 | $1 - |A \cap B|/|A \cup B|$ | bit |

> **重要**：`<#>` 返回**负**内积，因为 PostgreSQL 索引只支持升序扫描。获取实际内积需乘以 -1。

### 3.3 CRUD 操作

#### 插入数据

```sql
-- 插入单条
INSERT INTO items (content, embedding) VALUES ('Hello world', '[1,2,3]');

-- 批量插入
INSERT INTO items (content, embedding) VALUES
    ('文档1', '[1,2,3]'),
    ('文档2', '[4,5,6]'),
    ('文档3', '[7,8,9]');

-- Upsert（冲突时更新）
INSERT INTO items (id, content, embedding) VALUES (1, '文档', '[1,2,3]')
ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding;
```

#### Python 插入（psycopg3）

```python
import psycopg
from pgvector.psycopg import register_vector

conn = psycopg.connect("dbname=postgres user=postgres")
register_vector(conn)

# 插入向量
embedding = [1.0, 2.0, 3.0]
conn.execute(
    "INSERT INTO items (content, embedding) VALUES (%s, %s)",
    ("Hello world", embedding),
)

# 批量插入
from psycopg.sql import SQL

with conn.cursor() as cur:
    cur.executemany(
        "INSERT INTO items (content, embedding) VALUES (%s, %s)",
        [("doc1", [1.0, 2.0, 3.0]), ("doc2", [4.0, 5.0, 6.0])],
    )
```

#### 搜索查询

```sql
-- L2 距离搜索（最近邻）
SELECT id, content, embedding <-> '[3,1,2]' AS distance
FROM items
ORDER BY embedding <-> '[3,1,2]'
LIMIT 5;

-- 余弦相似度搜索
SELECT id, content, 1 - (embedding <=> '[3,1,2]') AS similarity
FROM items
ORDER BY embedding <=> '[3,1,2]'
LIMIT 5;

-- 内积搜索（注意取负）
SELECT id, content, (embedding <#> '[3,1,2]') * -1 AS inner_product
FROM items
ORDER BY embedding <#> '[3,1,2]'
LIMIT 5;

-- 距离阈值过滤
SELECT * FROM items WHERE embedding <-> '[3,1,2]' < 5;

-- 查找与某条记录最相似的记录
SELECT * FROM items WHERE id != 1
ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1)
LIMIT 5;

-- 带条件过滤的向量搜索
SELECT id, content, embedding <=> '[3,1,2]' AS distance
FROM items
WHERE category = 'AI'
ORDER BY embedding <=> '[3,1,2]'
LIMIT 5;
```

#### 聚合函数

```sql
-- 平均向量
SELECT AVG(embedding) FROM items;

-- 分组平均
SELECT category, AVG(embedding) FROM items GROUP BY category;

-- 向量求和
SELECT SUM(embedding) FROM items;
```

### 3.4 向量函数

```sql
-- 向量维度
SELECT vector_dims('[1,2,3]');    -- 3

-- 欧氏范数
SELECT vector_norm('[3,4]');       -- 5

-- L2归一化
SELECT l2_normalize('[3,4]');      -- [0.6, 0.8]

-- 子向量提取
SELECT subvector('[1,2,3,4,5]'::vector, 2, 3);  -- [2,3,4]（从第2位开始取3个）

-- 二值量化
SELECT binary_quantize('[1,-2,3,-4]'::vector);   -- 10（正=1，负=0）

-- 向量运算
SELECT '[1,2,3]'::vector + '[4,5,6]'::vector;     -- [5,7,9]
SELECT '[4,5,6]'::vector - '[1,2,3]'::vector;     -- [3,3,3]
SELECT '[1,2,3]'::vector * '[4,5,6]'::vector;     -- [4,10,18]（逐元素乘法）
```

### 3.5 索引类型

#### HNSW 索引

HNSW（层级可导航小世界图）提供更好的查询性能，但构建较慢、内存占用更多。**无需训练数据，可在空表上创建。**

```sql
-- 创建HNSW索引（按距离度量选择操作符类）
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops);   -- 余弦距离
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);       -- L2距离
CREATE INDEX ON items USING hnsw (embedding vector_ip_ops);       -- 内积
CREATE INDEX ON items USING hnsw (embedding vector_l1_ops);       -- L1距离

-- 自定义参数
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
-- m: 每层最大连接数，默认16。增大→更准确但更多内存
-- ef_construction: 构建时候选列表大小，默认64。增大→构建更慢但图质量更高
```

**操作符类映射：**

| 类型 | L2 | 内积 | 余弦 | L1 | 汉明 | 杰卡德 |
|------|----|-----|------|----|----|------|
| `vector` | `vector_l2_ops` | `vector_ip_ops` | `vector_cosine_ops` | `vector_l1_ops` | — | — |
| `halfvec` | `halfvec_l2_ops` | `halfvec_ip_ops` | `halfvec_cosine_ops` | `halfvec_l1_ops` | — | — |
| `sparsevec` | `sparsevec_l2_ops` | `sparsevec_ip_ops` | `sparsevec_cosine_ops` | `sparsevec_l1_ops` | — | — |
| `bit` | — | — | — | — | `bit_hamming_ops` | `bit_jaccard_ops` |

**搜索参数调优：**

```sql
-- 设置搜索时的候选列表大小（默认40）
SET hnsw.ef_search = 100;   -- 增大→更准确但更慢

-- 单次查询设置（推荐，不影响全局）
BEGIN;
SET LOCAL hnsw.ef_search = 100;
SELECT * FROM items ORDER BY embedding <=> '[3,1,2]' LIMIT 5;
COMMIT;
```

**加速索引构建：**

```sql
-- 增加构建内存
SET maintenance_work_mem = '8GB';

-- 增加并行工作线程
SET max_parallel_maintenance_workers = 7;
SET max_parallel_workers = 8;

-- 并发创建索引（不锁表，生产环境推荐）
CREATE INDEX CONCURRENTLY ON items USING hnsw (embedding vector_cosine_ops);
```

#### IVFFlat 索引

IVFFlat（倒排文件+平坦压缩）构建更快、内存更少，但查询性能不如 HNSW。**需要训练数据，表中有数据后才能创建。**

```sql
-- 创建IVFFlat索引
CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
-- lists: 聚类数量。推荐值：行数/1000（≤100万行），sqrt(行数)（>100万行）

-- 搜索时设置探测列表数
SET ivfflat.probes = 10;
-- probes: 搜索的聚类数，默认1。推荐：sqrt(lists)
-- probes = lists 时退化为精确搜索
```

**索引对比：**

| 特性 | HNSW | IVFFlat |
|------|------|---------|
| 查询性能 | 更好 | 较差 |
| 构建速度 | 较慢 | 更快 |
| 内存占用 | 更多 | 更少 |
| 需要训练数据 | 否 | 是 |
| 增量更新 | 支持良好 | 新数据可能分布不均 |
| 适用场景 | 实时查询 | 批量导入+离线查询 |

### 3.6 Python 集成

#### psycopg3（推荐）

```python
import psycopg
from pgvector.psycopg import register_vector

# 连接并注册向量类型
conn = psycopg.connect("dbname=postgres user=postgres")
register_vector(conn)

# 创建表
conn.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id bigserial PRIMARY KEY,
        content TEXT,
        metadata JSONB,
        embedding vector(1536)
    )
""")

# 插入数据
import numpy as np

doc_embedding = np.array([0.1, 0.2, 0.3] + [0.0] * 1533)  # 1536维
conn.execute(
    "INSERT INTO documents (content, metadata, embedding) VALUES (%s, %s, %s)",
    ("深度学习入门教程", {"category": "AI", "year": 2024}, doc_embedding),
)

# 向量搜索
query_embedding = np.array([0.15, 0.22, 0.31] + [0.0] * 1533)
results = conn.execute(
    "SELECT id, content, 1 - (embedding <=> %s) AS similarity "
    "FROM documents ORDER BY embedding <=> %s LIMIT 5",
    (query_embedding, query_embedding),
).fetchall()

for row in results:
    print(f"ID: {row[0]}, 相似度: {row[2]:.4f}, 内容: {row[1]}")
```

#### asyncpg（异步）

```python
import asyncpg
from pgvector.asyncpg import register_vector

async def main():
    conn = await asyncpg.connect("postgresql://postgres@localhost/postgres")
    await register_vector(conn)

    # 插入
    await conn.execute(
        "INSERT INTO documents (content, embedding) VALUES ($1, $2)",
        "Hello world", [1.0, 2.0, 3.0],
    )

    # 搜索
    rows = await conn.fetch(
        "SELECT id, content, embedding <=> $1 AS distance "
        "FROM documents ORDER BY embedding <=> $1 LIMIT 5",
        [1.5, 2.5, 3.5],
    )
```

### 3.7 高级索引技巧

#### 半精度索引（节省空间）

```sql
-- 创建半精度索引
CREATE INDEX ON items USING hnsw ((embedding::halfvec(1536)) halfvec_cosine_ops);

-- 查询时也转为半精度
SELECT * FROM items
ORDER BY embedding::halfvec(1536) <=> '[1,2,3]'::halfvec(1536)
LIMIT 5;
```

#### 二值量化 + 重排序

```sql
-- 二值量化索引（极致压缩）
CREATE INDEX ON items USING hnsw ((binary_quantize(embedding)::bit(1536)) bit_hamming_ops);

-- 两阶段搜索：粗排+精排
SELECT * FROM (
    SELECT * FROM items
    ORDER BY binary_quantize(embedding)::bit(1536) <~> binary_quantize('[1,-2,3]'::vector)
    LIMIT 20   -- 先多取
) ORDER BY embedding <=> '[1,-2,3]' LIMIT 5;  -- 精排取前5
```

#### 部分索引（按过滤条件）

```sql
-- 为特定分类创建部分索引
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)
WHERE (category_id = 123);

-- 查询时自动使用
SELECT * FROM items WHERE category_id = 123
ORDER BY embedding <=> '[3,1,2]' LIMIT 5;
```

#### 迭代索引扫描（v0.8.0+）

当过滤条件导致结果不足时，自动扫描更多索引：

```sql
-- 严格排序（精确距离顺序）
SET hnsw.iterative_scan = strict_order;

-- 宽松排序（略失序但更高召回率）
SET hnsw.iterative_scan = relaxed_order;
```

## 4. 在LLM开发中的典型使用场景和代码示例

### 场景1：构建 RAG 知识库（完整示例）

```python
import psycopg
from pgvector.psycopg import register_vector
from openai import OpenAI
import numpy as np

# 初始化
conn = psycopg.connect("dbname=ragdb user=postgres")
register_vector(conn)
openai_client = OpenAI()

# 1. 创建表和索引
conn.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id bigserial PRIMARY KEY,
        content TEXT NOT NULL,
        source TEXT,
        category TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        embedding vector(1536)
    )
""")
conn.execute("""
    CREATE INDEX IF NOT EXISTS ON knowledge_base
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
""")
conn.commit()

# 2. 嵌入函数
def get_embedding(text: str) -> list:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small", input=text
    )
    return resp.data[0].embedding

# 3. 文档摄取
def index_documents(texts: list[str], sources: list[str], categories: list[str]):
    for text, source, category in zip(texts, sources, categories):
        embedding = get_embedding(text)
        conn.execute(
            "INSERT INTO knowledge_base (content, source, category, embedding) VALUES (%s, %s, %s, %s)",
            (text, source, category, embedding),
        )
    conn.commit()

# 索引文档
docs = [
    "公司年假政策：入职满1年享有10天年假，满3年15天，满5年20天。",
    "API限流策略：免费用户100次/分钟，付费用户1000次/分钟。",
    "退款政策：购买后7天内可无条件退款，超过7天需提供理由。",
]
index_documents(
    docs,
    ["员工手册", "API文档", "购买协议"],
    ["假期", "技术", "财务"],
)

# 4. RAG查询
def rag_query(question: str, category: str = None) -> str:
    query_embedding = get_embedding(question)

    if category:
        sql = """
            SELECT content, 1 - (embedding <=> %s) AS similarity
            FROM knowledge_base
            WHERE category = %s
            ORDER BY embedding <=> %s LIMIT 3
        """
        results = conn.execute(sql, (query_embedding, category, query_embedding)).fetchall()
    else:
        sql = """
            SELECT content, 1 - (embedding <=> %s) AS similarity
            FROM knowledge_base
            ORDER BY embedding <=> %s LIMIT 3
        """
        results = conn.execute(sql, (query_embedding, query_embedding)).fetchall()

    context = "\n".join([f"- {row[0]}" for row in results])

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "根据以下上下文回答问题。如果上下文中没有相关信息，回答'我不知道'。"},
            {"role": "user", "content": f"上下文：\n{context}\n\n问题：{question}"},
        ],
    )
    return response.choices[0].message.content

answer = rag_query("公司的年假政策是什么？")
print(answer)
```

### 场景2：混合检索（向量+全文搜索+RRF融合）

```sql
-- 创建支持混合检索的表
CREATE TABLE articles (
    id bigserial PRIMARY KEY,
    title TEXT,
    content TEXT,
    category TEXT,
    embedding vector(1536),
    textsearch tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))) STORED
);

-- 创建全文索引
CREATE INDEX ON articles USING gin(textsearch);

-- 创建向量索引
CREATE INDEX ON articles USING hnsw (embedding vector_cosine_ops);

-- 创建B-tree索引用于过滤
CREATE INDEX ON articles (category);
```

```python
def hybrid_search(query: str, limit: int = 5) -> list:
    """混合检索：向量搜索 + 全文搜索 + RRF融合"""
    query_embedding = get_embedding(query)

    # RRF融合搜索（在SQL中实现）
    sql = """
    WITH vector_results AS (
        SELECT id, title, content,
               ROW_NUMBER() OVER (ORDER BY embedding <=> %s) AS rank_v
        FROM articles
        LIMIT 50
    ),
    text_results AS (
        SELECT id, title, content,
               ROW_NUMBER() OVER (ORDER BY ts_rank_cd(textsearch, plainto_tsquery(%s)) DESC) AS rank_t
        FROM articles, plainto_tsquery(%s) query
        WHERE textsearch @@ query
        LIMIT 50
    ),
    rrf_scores AS (
        SELECT id, title, content,
               COALESCE(1.0 / (60 + rank_v), 0) +
               COALESCE(1.0 / (60 + rank_t), 0) AS rrf_score
        FROM vector_results
        FULL OUTER JOIN text_results USING (id, title, content)
    )
    SELECT id, title, content, rrf_score
    FROM rrf_scores
    ORDER BY rrf_score DESC
    LIMIT %s
    """
    return conn.execute(sql, (query_embedding, query, query, limit)).fetchall()
```

### 场景3：多租户向量搜索

```sql
-- 使用分区表实现多租户隔离
CREATE TABLE tenant_documents (
    id bigserial,
    tenant_id int NOT NULL,
    content TEXT,
    embedding vector(1536),
    PRIMARY KEY (tenant_id, id)
) PARTITION BY LIST(tenant_id);

-- 为每个租户创建分区
CREATE TABLE tenant_a PARTITION OF tenant_documents FOR VALUES IN (1);
CREATE TABLE tenant_b PARTITION OF tenant_documents FOR VALUES IN (2);

-- 每个分区自动拥有独立的向量索引
CREATE INDEX ON tenant_a USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON tenant_b USING hnsw (embedding vector_cosine_ops);

-- 查询时自动路由到对应分区
SELECT * FROM tenant_documents
WHERE tenant_id = 1
ORDER BY embedding <=> '[3,1,2]'
LIMIT 5;
```

## 5. 数学原理

### 5.1 HNSW 算法

pgvector 的 HNSW 实现基于 Malkov & Yashunin (2016) 的论文。详见 Qdrant 文档中的 HNSW 说明，算法原理相同。

pgvector 中的关键参数：
- `m`：每层每个节点的最大连接数，默认 16。增大 → 搜索更准确但内存增加（每个连接约 8-16 字节）
- `ef_construction`：构建时的动态候选列表大小，默认 64。增大 → 构建更慢但图质量更高
- `ef_search`：搜索时的候选列表大小，默认 40。增大 → 搜索更慢但召回率更高

### 5.2 IVFFlat 算法

IVFFlat 使用 K-Means 将向量空间划分为 `lists` 个聚类（倒排列表），搜索时只扫描 `probes` 个最近聚类的向量。

**构建过程：**
1. 对所有向量执行 K-Means 聚类，得到 `lists` 个聚类中心
2. 将每个向量分配到最近的聚类
3. 每个聚类内的向量以平坦格式存储

**搜索过程：**
1. 计算查询向量到所有聚类中心的距离
2. 选择最近的 `probes` 个聚类
3. 在这些聚类内做精确搜索
4. 返回全局最近邻

**lists 选择策略：**
- 行数 ≤ 1,000,000：`lists = rows / 1000`
- 行数 > 1,000,000：`lists = sqrt(rows)`

### 5.3 距离函数

**余弦距离：**
$$\text{cosine\_dist}(A, B) = 1 - \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

**欧氏距离（L2）：**
$$\text{L2}(A, B) = \sqrt{\sum_{i=1}^{d}(A_i - B_i)^2}$$

**负内积：**
$$\text{neg\_ip}(A, B) = -(A \cdot B) = -\sum_{i=1}^{d} A_i B_i$$

对于已归一化的向量（如 OpenAI 嵌入），余弦距离等价于负内积+1，因此使用 `<#>`（内积）搜索更快（避免了范数计算）。

### 5.4 Reciprocal Rank Fusion（RRF）

混合检索中的融合公式：

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

其中 $k = 60$（默认平滑常数），$R$ 为多个检索器的结果集。

## 6. 代码原理/架构原理

### 扩展架构

```
┌──────────────────────────────────────────────────┐
│              PostgreSQL 引擎                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │  查询优化器  │  │  事务管理器  │  │  WAL 日志  │ │
│  └──────┬─────┘  └────────────┘  └────────────┘ │
│         │                                        │
│  ┌──────▼─────────────────────────────────────┐  │
│  │            pgvector 扩展                     │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │ 自定义类型 │  │ 距离操作符│  │ 索引方法 │  │  │
│  │  │ vector   │  │  <->     │  │  HNSW    │  │  │
│  │  │ halfvec  │  │  <=>     │  │  IVFFlat │  │  │
│  │  │ sparsevec│  │  <#>     │  │          │  │  │
│  │  │ bit      │  │  <+>     │  │          │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │           存储引擎（共享缓冲区+mmap）        │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### 核心设计

1. **PostgreSQL 扩展机制**：pgvector 作为 PG 扩展运行，通过自定义类型（Custom Type）、操作符（Operator）和索引方法（Index Access Method）三个机制集成到 PostgreSQL 内核中，无需修改 PG 源码。

2. **索引与查询优化器集成**：pgvector 的距离操作符注册了成本估算函数，查询优化器可以自动决定是否使用索引（基于 `ef_search`/`probes` 和数据量）。当数据量很小时，优化器会选择顺序扫描。

3. **ACID 兼容**：向量插入、更新、删除和搜索操作都在标准 PostgreSQL 事务中执行，享有完整的 ACID 保证。这意味着你可以：
   - 在事务中同时插入关系数据和向量数据
   - 向量搜索看到的是事务一致性的快照
   - 支持时间点恢复（PITR）

4. **与关系查询的无缝结合**：向量搜索可以和 JOIN、过滤、聚合、窗口函数等组合使用——这是专用向量数据库难以实现的。

## 7. 常见注意事项和最佳实践

### 注意事项

1. **索引使用条件**：查询必须包含 `ORDER BY <距离操作符> ASC` 和 `LIMIT`，否则不会使用索引。不能在距离表达式外包裹函数（如 `1 - (embedding <=> ...)` 不会走索引）。

2. **IVFFlat 需要训练数据**：在空表上创建 IVFFlat 索引会导致聚类质量很差。应先插入数据再创建索引。

3. **向量维度上限**：HNSW 索引最大支持 2,000 维（`vector` 类型）。超过 2,000 维需使用 `halfvec`（4,000维）或降维技术。

4. **内存与性能**：
   - HNSW 索引存储在 PostgreSQL 的共享缓冲区中，需要足够大的 `shared_buffers`
   - 构建索引时需要 `maintenance_work_mem`（建议 1-8GB）
   - 大索引构建需要足够的共享内存（Docker 需设 `--shm-size`）

5. **HNSW 死元组**：频繁的 UPDATE/DELETE 会在 HNSW 索引中产生死元组，影响搜索质量。建议先 REINDEX 再 VACUUM：
   ```sql
   REINDEX INDEX CONCURRENTLY index_name;
   VACUUM table_name;
   ```

### 最佳实践

1. **选择合适的距离度量**：
   - OpenAI 等归一化嵌入 → 使用 `<#>`（内积，最快）
   - 自训练嵌入 → 使用 `<=>`（余弦，最通用）
   - 需要绝对距离 → 使用 `<->`（L2）

2. **索引参数调优**：
   - 实时性要求高、数据量中等（< 100万）→ HNSW，`m=16, ef_construction=64`
   - 精度要求极高 → HNSW，`m=32, ef_construction=128`
   - 数据量大（> 1000万）且可离线构建 → IVFFlat

3. **使用半精度减少存储**：
   ```sql
   -- 存储半精度向量（存储减半，精度损失极小）
   ALTER TABLE items ADD COLUMN embedding_half halfvec(1536);
   UPDATE items SET embedding_half = embedding::halfvec(1536);
   CREATE INDEX ON items USING hnsw (embedding_half halfvec_cosine_ops);
   ```

4. **过滤搜索优化**：
   - 为过滤字段创建 B-tree 索引
   - 使用部分索引为高频过滤值预建向量索引
   - 使用 `SET hnsw.iterative_scan = strict_order` 处理严格过滤

5. **生产部署建议**：
   - 使用 `CREATE INDEX CONCURRENTLY` 避免锁表
   - 设置 `hnsw.ef_search` 在会话级别而非全局
   - 定期 REINDEX 维护索引质量
   - 使用 `EXPLAIN ANALYZE` 验证索引使用

6. **中文场景建议**：
   - 嵌入模型选择中文优化模型
   - 全文搜索需使用 `zhparser` 或 `pg_jieba` 中文分词扩展
   - 混合检索中，中文全文搜索的查询词需合理分词
