---
title: "Arize Phoenix 开源可观测性"
excerpt: "OpenTelemetry追踪、嵌入UMAP可视化、评估器、本地优先架构"
collection: llm-libs
permalink: /llm-libs/34-arize-phoenix
category: eval
---


## 1. 库的简介和在LLM开发中的作用

Arize Phoenix是一个开源的AI可观测性平台，专为LLM应用的调试、评估和监控而设计。在LLM开发中，一次用户请求可能涉及多步推理、多次模型调用、文档检索、工具调用等复杂环节，传统日志系统无法有效追踪这些分布式调用链。Phoenix基于OpenTelemetry（OTEL）标准，提供trace级别的可观测性，让开发者清晰看到LLM应用中每一步的输入输出、耗时、token消耗等关键信息。

Phoenix的核心价值：
- **追踪可观测性**：基于OpenTelemetry标准追踪LLM应用运行时，自动记录每步调用细节
- **评估框架**：内置LLM评估器（幻觉检测、相关性评估等），支持自定义评估函数
- **嵌入可视化**：通过UMAP降维技术在向量空间中可视化嵌入聚类，发现数据漂移和异常
- **数据集管理**：创建版本化的评估数据集，支持从DataFrame、CSV等多种来源导入
- **实验追踪**：追踪和评估提示词、模型、检索策略等变更的效果
- **本地优先**：无需云服务，可完全在本地运行，保护数据隐私
- **广泛的框架集成**：支持20+框架自动追踪（LangChain、LlamaIndex、OpenAI、CrewAI等）

Phoenix与LangSmith等商业产品的关键区别在于：Phoenix是开源的、本地优先的、基于OpenTelemetry标准的，这意味着你为Phoenix编写的任何追踪代码都可以与任何兼容OTEL的后端互操作，不会被供应商锁定。

## 2. 安装方式

```bash
# 安装完整的Phoenix平台（包含服务器、追踪、评估等所有功能）
pip install arize-phoenix

# 安装后启动本地Phoenix服务器
phoenix serve
```

如果已部署Phoenix服务器，可使用轻量子包：

```bash
# 仅安装OpenTelemetry追踪包装器
pip install arize-phoenix-otel

# 仅安装Phoenix客户端（与Phoenix服务器REST API交互）
pip install arize-phoenix-client

# 仅安装评估工具
pip install 'arize-phoenix-evals>=2.0.0' openai
```

自动追踪框架集成包（按需安装）：

```bash
# LangChain自动追踪
pip install openinference-instrumentation-langchain

# OpenAI自动追踪
pip install openinference-instrumentation-openai

# LlamaIndex自动追踪
pip install openinference-instrumentation-llama-index

# 其他常用集成
pip install openinference-instrumentation-anthropic    # Anthropic
pip install openinference-instrumentation-crewai       # CrewAI
pip install openinference-instrumentation-dspy         # DSPy
pip install openinference-instrumentation-bedrock      # AWS Bedrock
```

配置环境变量：

```python
import os

# Phoenix服务器端点（默认本地: http://localhost:6006）
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

# 项目名称（用于组织追踪数据）
os.environ["PHOENIX_PROJECT_NAME"] = "my-llm-project"

# API密钥（使用Phoenix Cloud时需要）
os.environ["PHOENIX_API_KEY"] = "phoenix_api_key_..."

# gRPC端口覆盖（默认4317）
os.environ["PHOENIX_GRPC_PORT"] = "4317"

# 额外的请求头
os.environ["PHOENIX_CLIENT_HEADERS"] = "Authorization=Bearer token,custom-header=value"

# 禁用遥测
os.environ["PHOENIX_TELEMETRY_ENABLED"] = "false"
```

验证安装：

```python
import phoenix as px
print(px.__version__)
```

Docker部署：

```bash
# 使用Docker运行
docker run -p 6006:6006 arizephoenix/phoenix:latest

# 使用Docker Compose
# 参考项目中的 docker-compose.yml
```

## 3. 核心类/函数/工具的详细说明

### 3.1 启动Phoenix：px.launch_app()

`px.launch_app()`是Phoenix在Jupyter Notebook等交互式环境中启动本地UI的方式。它会启动一个本地Web服务器，并在输出中提供访问URL，方便开发者即时查看追踪数据和嵌入可视化。

```python
import phoenix as px

# 启动Phoenix本地UI（默认端口6006）
session = px.launch_app()

# 查看访问URL
print(f"Phoenix UI: {session.url}")

# 启动时指定端口
session = px.launch_app(port=8080)

# 使用完毕后关闭
session.close()
```

**参数说明**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `port` | int | 6006 | 服务器监听端口 |
| `host` | str | "0.0.0.0" | 服务器绑定地址 |

> **注意**：在生产环境中，推荐使用`phoenix serve`命令或Docker部署Phoenix服务器，而非`launch_app()`。`launch_app()`更适合开发调试阶段。

### 3.2 追踪（Tracing）：Span概念与追踪机制

#### 3.2.1 Span概念

Span是Phoenix追踪的基本单元，代表LLM应用中的一次操作。多个Span按父子关系组成一条Trace（追踪链）。Phoenix基于OpenInference语义规范定义了以下Span类型：

| Span类型 | `openinference_span_kind`值 | 说明 |
|----------|----------------------------|------|
| Chain | `"chain"` | 顺序执行的工作流/管道 |
| Agent | `"agent"` | 自主决策的AI代理 |
| Tool | `"tool"` | 工具/函数调用 |
| LLM | `"llm"` | 语言模型交互 |
| Retriever | `"retriever"` | 文档检索操作 |
| Embedding | `"embedding"` | 嵌入生成 |

每个Span包含以下核心属性：
- **name**：Span名称，标识操作类型
- **span_kind**：Span类型（chain/agent/tool/llm/retriever/embedding）
- **input**：操作的输入数据
- **output**：操作的输出数据
- **metadata**：附加元数据（如模型名称、token数量等）

#### 3.2.2 注册追踪器：register()

`register()`是设置OpenTelemetry追踪的最简方式，提供Phoenix感知的默认配置：

```python
from phoenix.otel import register

# 最简配置：自动追踪已安装的AI/ML库
tracer_provider = register(auto_instrument=True)

# 完整配置
tracer_provider = register(
    project_name="my-llm-app",        # 项目名称
    endpoint="http://localhost:6006/v1/traces",  # 追踪数据端点
    headers={"Authorization": "Bearer TOKEN"},   # 认证头
    batch=True,                        # 生产环境推荐：批量发送Span
    auto_instrument=True,              # 自动追踪已安装的AI/ML库
    protocol="grpc",                   # 传输协议：http/protobuf 或 grpc
)
```

**register()参数说明**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `project_name` | str | "default" | 项目名称，用于组织追踪数据 |
| `endpoint` | str | 自动推断 | 追踪数据收集端点URL |
| `headers` | dict | {} | 附加HTTP头（用于认证等） |
| `batch` | bool | False | 是否批量发送Span（生产推荐True） |
| `auto_instrument` | bool | False | 是否自动追踪已安装的AI/ML库 |
| `protocol` | str | "grpc" | OTLP传输协议 |

#### 3.2.3 自动追踪

开启`auto_instrument=True`后，Phoenix会自动检测并追踪已安装的OpenInference集成包支持的框架：

```python
from phoenix.otel import register

# 注册追踪器并启用自动追踪
tracer_provider = register(
    project_name="my-langchain-app",
    auto_instrument=True  # 自动追踪LangChain、OpenAI等
)

# 此后所有LangChain调用都会被自动追踪，无需额外代码
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
response = llm.invoke("你好，请介绍一下Python")  # 自动生成追踪Span
```

#### 3.2.4 手动追踪：@tracer.chain装饰器

对于自定义函数，可以使用追踪器提供的装饰器手动添加追踪：

```python
from phoenix.otel import register

tracer_provider = register(project_name="my-app")
tracer = tracer_provider.get_tracer(__name__)

# 使用@tracer.chain装饰器追踪工作流函数
@tracer.chain
def process_document(document: str) -> str:
    """顺序执行的文档处理管道"""
    cleaned = document.strip().lower()
    return cleaned

# 使用@tracer.tool装饰器追踪工具调用
@tracer.tool
def search_database(query: str) -> list:
    """数据库搜索工具"""
    # 模拟搜索
    return [{"id": 1, "text": f"搜索结果: {query}"}]

# 使用@tracer.agent装饰器追踪AI代理
@tracer.agent
def my_agent(task: str) -> str:
    """自主决策的AI代理"""
    return f"完成: {task}"

# 调用这些函数时会自动生成追踪Span
result = process_document("  Hello World  ")
```

**可用的装饰器**：
| 装饰器 | 对应Span类型 | 适用场景 |
|--------|-------------|---------|
| `@tracer.chain` | chain | 顺序执行的工作流/管道 |
| `@tracer.agent` | agent | 自主决策的AI代理 |
| `@tracer.tool` | tool | 函数/工具调用 |
| `@tracer.llm` | llm | 语言模型交互（需配合process_input/process_output） |
| `@tracer.retriever` | retriever | 文档检索操作 |
| `@tracer.embedding` | embedding | 嵌入生成 |

#### 3.2.5 手动追踪：tracer.start_as_current_span()

对于更细粒度的控制，可以使用上下文管理器手动创建Span：

```python
from phoenix.otel import register
from opentelemetry.trace import StatusCode, Status

tracer_provider = register(project_name="my-app")
tracer = tracer_provider.get_tracer(__name__)

def invoke_llm(messages: list, model: str = "gpt-4o") -> str:
    """手动追踪LLM调用"""
    with tracer.start_as_current_span(
        "llm_call",                       # Span名称
        openinference_span_kind="llm",     # Span类型
    ) as span:
        # 设置输入属性
        span.set_input(messages)
        span.set_attributes({
            "llm.input_messages": [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ],
            "llm.model_name": model,
        })

        # 执行LLM调用（示例）
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        output_text = response.choices[0].message.content

        # 设置输出属性
        span.set_output(output_text)
        span.set_attributes({
            "llm.output_messages": [
                {"role": "assistant", "content": output_text}
            ],
            "llm.token_count.prompt": response.usage.prompt_tokens if response.usage else 0,
            "llm.token_count.completion": response.usage.completion_tokens if response.usage else 0,
        })
        span.set_status(Status(StatusCode.OK))

        return output_text
```

**Span核心方法**：
| 方法 | 说明 |
|------|------|
| `span.set_input(value)` | 设置Span的输入值 |
| `span.set_output(value)` | 设置Span的输出值 |
| `span.set_attributes(dict)` | 批量设置属性（如模型名、token数等） |
| `span.set_status(status)` | 设置Span状态（OK/ERROR） |
| `span.record_exception(error)` | 记录异常信息 |

#### 3.2.6 上下文属性与追踪抑制

使用`using_attributes`可以为上下文中所有的Span添加通用属性：

```python
from openinference.instrumentation import using_attributes

# 上下文中所有Span将自动携带session_id和user_id
with using_attributes(session_id="session-123", user_id="user-456"):
    result = my_function("input")
```

使用`suppress_tracing`可以临时禁用追踪：

```python
from openinference.instrumentation import suppress_tracing

# 此代码块内的调用不会被追踪
with suppress_tracing():
    result = my_function("input")
```

### 3.3 与OpenTelemetry集成

Phoenix的核心架构基于OpenTelemetry标准，提供与标准OTEL组件的完全兼容。这意味着你可以使用标准的OpenTelemetry API来配置追踪：

```python
from opentelemetry import trace as trace_api
from phoenix.otel import HTTPSpanExporter, TracerProvider, SimpleSpanProcessor

# 方式1：使用Phoenix的TracerProvider（替代标准OTEL TracerProvider）
tracer_provider = TracerProvider()
span_exporter = HTTPSpanExporter(endpoint="http://localhost:6006/v1/traces")
span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
tracer_provider.add_span_processor(span_processor)
trace_api.set_tracer_provider(tracer_provider)

# 方式2：简化配置（自动推断Exporter类型）
from phoenix.otel import TracerProvider
tracer_provider = TracerProvider(endpoint="http://localhost:4317")
trace_api.set_tracer_provider(tracer_provider)

# 方式3：带项目名称和资源标签
from phoenix.otel import Resource, PROJECT_NAME, TracerProvider
tracer_provider = TracerProvider(
    resource=Resource({PROJECT_NAME: "my-project"})
)
trace_api.set_tracer_provider(tracer_provider)
```

**Phoenix OTEL组件与标准OTEL的对应关系**：
| Phoenix组件 | 标准OTEL组件 | 说明 |
|-------------|-------------|------|
| `phoenix.otel.TracerProvider` | `opentelemetry.sdk.trace.TracerProvider` | Phoenix感知的默认配置 |
| `phoenix.otel.HTTPSpanExporter` | `opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter` | HTTP协议导出器 |
| `phoenix.otel.GRPCSpanExporter` | `opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter` | gRPC协议导出器 |
| `phoenix.otel.SimpleSpanProcessor` | `opentelemetry.sdk.trace.SimpleSpanProcessor` | 简单Span处理器 |
| `phoenix.otel.BatchSpanProcessor` | `opentelemetry.sdk.trace.BatchSpanProcessor` | 批量Span处理器 |

### 3.4 嵌入可视化：UMAP降维

Phoenix提供嵌入（Embedding）可视化功能，通过UMAP（Uniform Manifold Approximation and Projection）降维技术将高维嵌入向量投影到2D/3D空间，帮助开发者直观地观察嵌入的聚类结构、发现数据漂移和异常值。

```python
import phoenix as px
import pandas as pd
import numpy as np

# 示例：创建带有嵌入的数据集
# 假设我们有文档嵌入和对应的类别标签
embeddings = np.random.randn(200, 1536)  # 200个文档的嵌入向量（如OpenAI text-embedding-ada-002维度）
categories = np.random.choice(["技术", "金融", "医疗", "教育"], size=200)

# 创建Phoenix数据集用于嵌入可视化
# 使用px.launch_app()启动UI后，可在嵌入探索器中查看
session = px.launch_app()

# 通过Phoenix客户端上传嵌入数据
from phoenix.client import Client

client = Client()

# 将嵌入数据组织为DataFrame
df = pd.DataFrame({
    "text": [f"文档内容_{i}" for i in range(200)],
    "category": categories,
    "embedding": list(embeddings),  # 嵌入向量列
})

# Phoenix UI中的嵌入探索器可以：
# 1. 自动对嵌入执行UMAP降维
# 2. 在2D/3D空间中可视化聚类
# 3. 按类别标签着色，发现聚类与语义的对应关系
# 4. 识别离群点和数据漂移
```

**UMAP降维原理**：
UMAP基于流形学习和拓扑数据分析，其核心思想是：
1. 在高维空间中构建局部连接图（使用k近邻）
2. 优化低维表示使局部连接关系尽可能保持
3. 相比t-SNE，UMAP更好地保留全局结构，且计算速度更快

在Phoenix UI的嵌入探索器中，你可以：
- 选择不同的嵌入列进行可视化
- 按元数据字段（如类别、时间）着色
- 缩放和平移查看局部细节
- 点击数据点查看原始文本内容
- 对比不同时间段的嵌入分布，检测数据漂移

### 3.5 评估（Evaluators）

Phoenix Evals提供轻量级、可组合的评估构建块，用于评估LLM应用的质量。

#### 3.5.1 内置LLM评估器

Phoenix提供多个预构建的LLM评估器：

```python
from phoenix.evals.llm import LLM
from phoenix.evals.metrics import (
    FaithfulnessEvaluator,           # 忠实度/幻觉检测
    ConcisenessEvaluator,            # 简洁性评估
    CorrectnessEvaluator,            # 正确性评估
    DocumentRelevanceEvaluator,      # 文档相关性评估
    RefusalEvaluator,                # 拒答检测
    ToolInvocationEvaluator,         # 工具调用评估
    ToolSelectionEvaluator,          # 工具选择评估
    ToolResponseHandlingEvaluator,   # 工具响应处理评估
)

# 初始化LLM（支持多提供商）
llm = LLM(provider="openai", model="gpt-4o")
# llm = LLM(provider="anthropic", model="claude-3-5-sonnet-20241022")
# llm = LLM(provider="google", model="gemini-1.5-pro")
# llm = LLM(provider="litellm", model="gpt-4o")  # 统一接口，支持100+提供商
```

**忠实度评估（幻觉检测）**：

```python
faithfulness = FaithfulnessEvaluator(llm=llm)

scores = faithfulness.evaluate({
    "input": "法国的首都是什么？",
    "context": "巴黎是法国的首都。",
    "output": "法国的首都是柏林。",  # 与上下文矛盾，属于幻觉
})
scores[0].pretty_print()
# Score(name='faithfulness', score=0.0, label='unfaithful', explanation='输出与上下文矛盾...')
```

**文档相关性评估**：

```python
doc_relevance = DocumentRelevanceEvaluator(llm=llm)

scores = doc_relevance.evaluate({
    "input": "如何重置密码？",
    "context": "公司成立于2010年，总部位于北京。",  # 与问题不相关
    "output": "请点击登录页面的'忘记密码'链接。",
})
scores[0].pretty_print()
```

**Score对象属性**：
| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | str | 评估器名称 |
| `score` | float | 数值评分 |
| `label` | str | 标签（如"unfaithful"、"helpful"） |
| `explanation` | str | LLM生成的评分解释 |

#### 3.5.2 自定义评估函数

使用`create_classifier`创建自定义的LLM评估器：

```python
from phoenix.evals import create_classifier
from phoenix.evals.llm import LLM

llm = LLM(provider="openai", model="gpt-4o")

# 创建自定义"有用性"评估器
helpfulness_evaluator = create_classifier(
    name="helpfulness",                       # 评估器名称
    prompt_template="""请评估以下回复是否有用。

用户问题: {input}
系统回复: {output}

请判断回复是否有用。""",                       # 提示词模板
    llm=llm,                                   # LLM实例
    choices={"helpful": 1.0, "not_helpful": 0.0},  # 分类标签与评分映射
)

# 运行评估
scores = helpfulness_evaluator.evaluate({
    "input": "如何重置密码？",
    "output": "请前往设置 > 账户 > 重置密码。"
})
scores[0].pretty_print()
```

**create_classifier参数说明**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | str | 评估器名称 |
| `prompt_template` | str | 提示词模板，支持`{input}`、`{output}`、`{context}`等占位符 |
| `llm` | LLM | LLM实例 |
| `choices` | dict | 标签到评分的映射，如`{"relevant": 1.0, "irrelevant": 0.0}` |

**支持嵌套数据的输入映射**：

```python
scores = helpfulness_evaluator.evaluate(
    {"data": {"query": "如何重置密码？", "response": "请前往设置。"}},
    input_mapping={"input": "data.query", "output": "data.response"}
)
```

#### 3.5.3 代码评估器（无需LLM）

Phoenix还提供不依赖LLM的代码评估器：

```python
from phoenix.evals.metrics import exact_match, MatchesRegex, PrecisionRecallFScore

# 精确匹配
match_result = exact_match({"output": "Paris", "expected": "Paris"})
# 返回 True

# 正则匹配
regex_evaluator = MatchesRegex(pattern=r"^\d{4}\-\d{2}\-\d{2}$")
regex_result = regex_evaluator.evaluate({"output": "2024-03-15"})
# 返回 True

# 精确率/召回率/F1
prf = PrecisionRecallFScore()
prf_result = prf.evaluate({"output": "A B C", "expected": "A B D"})
```

#### 3.5.4 批量评估：evaluate_dataframe

对整个DataFrame批量运行多个评估器：

```python
import pandas as pd
from phoenix.evals import create_classifier, evaluate_dataframe, async_evaluate_dataframe
from phoenix.evals.llm import LLM

llm = LLM(provider="openai", model="gpt-4o")

# 创建多个评估器
relevance_evaluator = create_classifier(
    name="relevance",
    prompt_template="回复是否与问题相关？\n\n问题: {input}\n回复: {output}",
    llm=llm,
    choices={"relevant": 1.0, "irrelevant": 0.0},
)

helpfulness_evaluator = create_classifier(
    name="helpfulness",
    prompt_template="回复是否有用？\n\n问题: {input}\n回复: {output}",
    llm=llm,
    choices={"helpful": 1.0, "not_helpful": 0.0},
)

# 准备评估数据
df = pd.DataFrame([
    {"input": "如何重置密码？", "output": "请前往设置 > 账户 > 重置密码。"},
    {"input": "今天天气如何？", "output": "我可以帮助您重置密码。"},  # 无关回复
])

# 同步批量评估
results_df = evaluate_dataframe(
    dataframe=df,
    evaluators=[relevance_evaluator, helpfulness_evaluator]
)

# 异步批量评估（大数据集推荐，最高20倍加速）
import asyncio
results_df = asyncio.run(async_evaluate_dataframe(
    dataframe=df,
    evaluators=[relevance_evaluator, helpfulness_evaluator],
))
```

### 3.6 数据集（Datasets）

Phoenix Client提供数据集管理功能，用于创建版本化的评估数据集。

#### 3.6.1 创建数据集

```python
from phoenix.client import Client

client = Client()

# 方式1：从字典列表创建
dataset = client.datasets.create_dataset(
    name="customer-support-qa",
    dataset_description="客服问答评估数据集",
    inputs=[
        {"question": "如何重置密码？"},
        {"question": "退货政策是什么？"},
        {"question": "如何追踪我的订单？"}
    ],
    outputs=[
        {"answer": "请点击登录页面的'忘记密码'链接重置密码。"},
        {"answer": "我们为原包装未使用的商品提供30天退货服务。"},
        {"answer": "您可以使用发送到邮箱的追踪号来追踪订单。"}
    ],
    metadata=[
        {"category": "account", "difficulty": "easy"},
        {"category": "policy", "difficulty": "medium"},
        {"category": "orders", "difficulty": "easy"}
    ]
)
```

**create_dataset参数说明**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | str | 数据集名称 |
| `dataset_description` | str | 数据集描述 |
| `inputs` | list[dict] | 输入数据列表 |
| `outputs` | list[dict] | 期望输出列表 |
| `metadata` | list[dict] | 元数据列表 |

```python
# 方式2：从Pandas DataFrame创建
import pandas as pd

df = pd.DataFrame({
    "prompt": ["你好", "你好啊", "早上好"],
    "response": ["你好！有什么可以帮您？", "您好！请问需要什么帮助？", "早上好！有什么可以为您效劳？"],
    "sentiment": ["neutral", "positive", "positive"],
    "length": [20, 25, 30]
})

dataset = client.datasets.create_dataset(
    name="greeting-responses",
    dataframe=df,
    input_keys=["prompt"],                # 映射到input的列
    output_keys=["response"],             # 映射到output的列
    metadata_keys=["sentiment", "length"] # 映射到metadata的列
)

# 方式3：从CSV文件创建
dataset = client.datasets.create_dataset(
    name="csv-dataset",
    csv_file_path="path/to/data.csv",
    input_keys=["question", "context"],
    output_keys=["answer"],
    metadata_keys=["source", "confidence"]
)
```

#### 3.6.2 数据集操作

```python
# 列出所有数据集
datasets = client.datasets.list()
for ds in datasets:
    print(f"数据集: {ds['name']} ({ds['example_count']} 个样本)")

# 获取特定数据集
dataset = client.datasets.get_dataset(dataset="customer-support-qa")

# 获取特定版本的数据集
versioned = client.datasets.get_dataset(
    dataset="customer-support-qa",
    version_id="version-123"
)

# 向已有数据集添加样本
client.datasets.add_examples_to_dataset(
    dataset="customer-support-qa",
    inputs=[{"question": "如何取消订阅？"}],
    outputs=[{"answer": "您可以在账户设置中取消订阅。"}],
    metadata=[{"category": "subscription", "difficulty": "medium"}]
)

# 从DataFrame追加样本
new_df = pd.DataFrame({
    "question": ["营业时间是什么？", "有在线客服吗？"],
    "answer": ["我们7x24小时营业", "有，可在官网使用在线客服"],
    "topic": ["hours", "support"]
})
client.datasets.add_examples_to_dataset(
    dataset="customer-support-qa",
    dataframe=new_df,
    input_keys=["question"],
    output_keys=["answer"],
    metadata_keys=["topic"]
)

# 查看数据集版本历史
versions = client.datasets.get_dataset_versions(dataset="customer-support-qa")
for version in versions:
    print(f"版本: {version['version_id']}, 创建时间: {version['created_at']}")

# 将数据集转为DataFrame
df = dataset.to_dataframe()
# 列: ['input', 'output', 'metadata']

# 遍历数据集
for example in dataset:
    print(f"输入: {example['input']}")
    print(f"输出: {example['output']}")
    print(f"元数据: {example['metadata']}")

# 序列化与反序列化
dataset_dict = dataset.to_dict()
restored_dataset = Dataset.from_dict(dataset_dict)
```

#### 3.6.3 实验（Experiments）

基于数据集运行实验和评估：

```python
from phoenix.client.experiments import run_experiment, evaluate_experiment

dataset = client.datasets.get_dataset(dataset="customer-support-qa")

# 定义任务函数
def my_task(input):
    return f"处理: {input['question']}"

# 运行实验
experiment = run_experiment(
    dataset=dataset,
    task=my_task,
    experiment_name="greeting-experiment"
)

# 带评估器的实验
def accuracy_evaluator(output, expected):
    """准确度评估器"""
    return 1.0 if output == expected.get("text") else 0.0

def detailed_evaluator(output, input, expected, metadata):
    """详细评估器（可访问所有字段）"""
    score = calculate_similarity(output, expected)
    return {"score": score, "label": "pass" if score > 0.8 else "fail"}

experiment = run_experiment(
    dataset=dataset,
    task=my_task,
    evaluators=[accuracy_evaluator, detailed_evaluator],
    experiment_name="evaluated-experiment"
)

# 对已有实验添加评估
evaluated = evaluate_experiment(
    experiment=experiment,
    evaluators=[accuracy_evaluator],
    print_summary=True
)

# 异步实验（大数据集推荐）
from phoenix.client.experiments import async_run_experiment
from phoenix.client import AsyncClient

async_client = AsyncClient()
dataset = await async_client.datasets.get_dataset(dataset="my-dataset")

async def async_task(input):
    return f"处理: {input['name']}"

experiment = await async_run_experiment(
    dataset=dataset,
    task=async_task,
    experiment_name="async-experiment",
    concurrency=5  # 并发执行5个任务
)
```

### 3.7 与LangChain集成

Phoenix通过OpenInference提供LangChain的自动追踪集成，无需修改LangChain代码即可获得完整的可观测性：

```python
# 第1步：安装集成包
# pip install arize-phoenix-otel openinference-instrumentation-langchain

# 第2步：注册追踪器并启用自动追踪
from phoenix.otel import register

tracer_provider = register(
    project_name="langchain-rag-app",
    auto_instrument=True  # 自动追踪LangChain
)

# 第3步：正常编写LangChain代码，所有调用将被自动追踪
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

# 创建LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 创建RAG链
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手，请根据以下上下文回答问题：\n{context}"),
    ("human", "{question}"),
])

# 所有LLM调用、检索操作、链执行都将自动生成追踪Span
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),  # 检索操作自动追踪为retriever Span
)

result = chain.invoke({"query": "什么是RAG？"})
# 在Phoenix UI中可以看到完整的追踪链：
# chain -> retriever -> llm 的层级关系
```

**LangChain自动追踪覆盖的组件**：
- LLM调用（ChatOpenAI、OpenAI等）
- 链执行（RetrievalQA、LLMChain等）
- 检索器（VectorStoreRetriever等）
- 工具调用（Agent工具等）
- Agent执行

## 4. 典型使用场景和代码示例

### 4.1 场景一：RAG应用全链路追踪

```python
"""
完整的RAG应用追踪示例：
展示如何追踪检索、重排序、生成等环节
"""
from phoenix.otel import register
from opentelemetry.trace import StatusCode, Status
import openai

# 注册追踪器
tracer_provider = register(
    project_name="rag-app",
    endpoint="http://localhost:6006/v1/traces",
)
tracer = tracer_provider.get_tracer(__name__)

@tracer.chain
def rag_pipeline(query: str) -> str:
    """RAG管道：检索 -> 重排序 -> 生成"""
    # 步骤1：检索相关文档
    documents = retrieve_documents(query)
    # 步骤2：重排序
    ranked_docs = rerank_documents(query, documents)
    # 步骤3：生成回答
    answer = generate_answer(query, ranked_docs)
    return answer

@tracer.retriever
def retrieve_documents(query: str, top_k: int = 5) -> list:
    """文档检索"""
    # 模拟向量检索
    docs = [
        {"text": "RAG是检索增强生成的缩写...", "score": 0.95},
        {"text": "向量数据库用于存储文档嵌入...", "score": 0.87},
    ][:top_k]
    return docs

@tracer.tool
def rerank_documents(query: str, documents: list) -> list:
    """文档重排序"""
    # 按相关性分数排序
    return sorted(documents, key=lambda x: x["score"], reverse=True)

@tracer.llm
def generate_answer(query: str, documents: list) -> str:
    """基于文档生成回答"""
    context = "\n".join([doc["text"] for doc in documents])
    messages = [
        {"role": "system", "content": f"请根据以下上下文回答问题：\n{context}"},
        {"role": "user", "content": query},
    ]

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    return response.choices[0].message.content

# 运行RAG管道
answer = rag_pipeline("什么是RAG？")
print(f"回答: {answer}")
# 在Phoenix UI中可以看到：rag_pipeline(chain) -> retrieve_documents(retriever) -> rerank_documents(tool) -> generate_answer(llm) 的完整追踪链
```

### 4.2 场景二：LLM输出质量评估

```python
"""
对LLM输出进行自动化质量评估
"""
import pandas as pd
from phoenix.evals import create_classifier, evaluate_dataframe
from phoenix.evals.llm import LLM
from phoenix.evals.metrics import FaithfulnessEvaluator

llm = LLM(provider="openai", model="gpt-4o")

# 创建评估数据
eval_data = pd.DataFrame([
    {
        "input": "Python是什么？",
        "context": "Python是一种高级编程语言，以简洁易读的语法著称。",
        "output": "Python是一种高级编程语言，语法简洁易读。",
    },
    {
        "input": "什么是机器学习？",
        "context": "机器学习是人工智能的一个分支，通过数据训练模型。",
        "output": "机器学习是一种烹饪技术，通过加热食物使其变色。",
    },
])

# 评估1：忠实度（幻觉检测）
faithfulness = FaithfulnessEvaluator(llm=llm)
faithfulness_scores = []
for _, row in eval_data.iterrows():
    scores = faithfulness.evaluate({
        "input": row["input"],
        "context": row["context"],
        "output": row["output"],
    })
    faithfulness_scores.append(scores[0])

for i, score in enumerate(faithfulness_scores):
    print(f"样本{i}: {score.label} (score={score.score})")
    print(f"  解释: {score.explanation}")

# 评估2：自定义相关性评估器
relevance = create_classifier(
    name="relevance",
    prompt_template="回复是否与问题相关？\n\n问题: {input}\n回复: {output}",
    llm=llm,
    choices={"relevant": 1.0, "irrelevant": 0.0},
)

# 批量评估
results_df = evaluate_dataframe(
    dataframe=eval_data,
    evaluators=[relevance],
)
print(results_df)
```

### 4.3 场景三：评估数据集与实验管理

```python
"""
创建评估数据集，运行实验并追踪结果
"""
from phoenix.client import Client
from phoenix.client.experiments import run_experiment

client = Client()

# 创建评估数据集
dataset = client.datasets.create_dataset(
    name="qa-eval-dataset",
    dataset_description="问答系统评估数据集",
    inputs=[
        {"question": "Python的创始人是谁？"},
        {"question": "什么是列表推导式？"},
        {"question": "如何处理Python中的异常？"},
    ],
    outputs=[
        {"answer": "Guido van Rossum"},
        {"answer": "列表推导式是一种简洁的创建列表的方式，如 [x**2 for x in range(10)]"},
        {"answer": "使用try-except语句块来捕获和处理异常"},
    ],
    metadata=[
        {"category": "basic", "difficulty": "easy"},
        {"category": "syntax", "difficulty": "medium"},
        {"category": "error_handling", "difficulty": "medium"},
    ]
)

# 定义任务函数
def qa_task(input):
    """调用你的QA系统"""
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "请简洁地回答以下问题。"},
            {"role": "user", "content": input["question"]},
        ],
    )
    return response.choices[0].message.content

# 定义评估器
def answer_quality_evaluator(output, expected):
    """评估答案质量"""
    expected_answer = expected.get("answer", "")
    # 简单的关键词匹配
    overlap = len(set(output.split()) & set(expected_answer.split()))
    total = len(set(expected_answer.split()))
    score = overlap / total if total > 0 else 0
    return {"score": score, "label": "pass" if score > 0.5 else "fail"}

# 运行实验
experiment = run_experiment(
    dataset=dataset,
    task=qa_task,
    evaluators=[answer_quality_evaluator],
    experiment_name="qa-quality-v1"
)

print(f"实验完成，共 {len(experiment.runs)} 次运行")
```

### 4.4 场景四：多Agent应用追踪

```python
"""
多Agent协作应用的追踪示例
"""
from phoenix.otel import register

tracer_provider = register(
    project_name="multi-agent-app",
    auto_instrument=True,
)
tracer = tracer_provider.get_tracer(__name__)

@tracer.agent
def research_agent(topic: str) -> str:
    """研究Agent：收集信息"""
    return f"关于'{topic}'的研究报告：..."

@tracer.agent
def writing_agent(research: str) -> str:
    """写作Agent：基于研究撰写文章"""
    return f"基于研究撰写的文章：{research[:50]}..."

@tracer.agent
def review_agent(article: str) -> str:
    """审稿Agent：审核文章质量"""
    return "审核通过，建议微调第二段。"

@tracer.chain
def content_creation_pipeline(topic: str) -> str:
    """内容创作管道：研究 -> 写作 -> 审核"""
    research = research_agent(topic)
    article = writing_agent(research)
    review = review_agent(article)
    return f"最终内容: {article}\n审核意见: {review}"

# 运行多Agent管道
result = content_creation_pipeline("人工智能在医疗领域的应用")
# Phoenix UI中可以看到层级追踪：
# content_creation_pipeline(chain)
#   ├── research_agent(agent)
#   ├── writing_agent(agent)
#   └── review_agent(agent)
```

## 5. 数学原理

### 5.1 UMAP降维原理

UMAP（Uniform Manifold Approximation and Projection）是Phoenix嵌入可视化的核心算法，用于将高维嵌入向量降维到2D/3D空间。

**核心数学步骤**：

1. **构建模糊单纯集合（Fuzzy Simplicial Set）**：
   - 对于每个数据点$x_i$，找到其k个最近邻
   - 计算局部连接性：对于近邻$x_j$，计算距离$d(x_i, x_j)$
   - 使用自适应核函数将距离转换为亲和度：
     $$a_{ij} = \exp\left(-\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}\right)$$
     其中$\rho_i$是$x_i$到其最近邻的距离，$\sigma_i$是归一化常数
   - 对称化：$\tilde{a}_{ij} = a_{ij} + a_{ji} - a_{ij} \cdot a_{ji}$

2. **低维嵌入优化**：
   - 在低维空间（2D）中初始化嵌入$y_i$
   - 使用交叉熵作为损失函数：
     $$C = -\sum_{i<j} \left[\tilde{a}_{ij} \ln\left(\frac{q_{ij}}{1-q_{ij}}\right) + (1-\tilde{a}_{ij}) \ln\left(\frac{1-q_{ij}}{q_{ij}}\right)\right]$$
     其中$q_{ij}$是低维空间中的亲和度
   - 通过随机梯度下降优化

3. **与t-SNE的区别**：
   - UMAP使用指数核而非高斯核，保留更多全局结构
   - UMAP计算复杂度更低（$O(n \log n)$ vs $O(n^2)$）
   - UMAP的损失函数同时优化近邻吸引和远距排斥

### 5.2 评估评分原理

Phoenix的LLM评估器本质上是通过LLM进行结构化分类：

1. 将评估任务构建为分类问题（如：relevant/irrelevant）
2. 通过提示词模板将输入数据格式化为LLM可理解的评估任务
3. LLM输出分类标签和解释
4. 将标签映射为数值评分（通过`choices`参数定义的映射）

## 6. 代码原理/架构原理

### 6.1 基于OpenTelemetry的可观测性架构

Phoenix的可观测性架构基于OpenTelemetry标准，其核心设计理念是"不发明新协议"：

```
┌─────────────────────────────────────────────────┐
│                 LLM应用代码                       │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐      │
│  │ LangChain │ │  OpenAI   │ │ LlamaIndex│      │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘      │
│        │             │             │             │
│  ┌─────▼─────────────▼─────────────▼─────┐      │
│  │     OpenInference Instrumentation      │      │
│  │  (自动插桩：拦截函数调用，生成Span)      │      │
│  └─────────────────┬─────────────────────┘      │
│                    │                             │
│  ┌─────────────────▼─────────────────────┐      │
│  │      OpenTelemetry SDK (Tracer)        │      │
│  │  (标准追踪API：TracerProvider, Span)    │      │
│  └─────────────────┬─────────────────────┘      │
│                    │                             │
│  ┌─────────────────▼─────────────────────┐      │
│  │     SpanProcessor (Simple/Batch)       │      │
│  │  (处理Span：采样、批处理、导出)          │      │
│  └─────────────────┬─────────────────────┘      │
│                    │                             │
│  ┌─────────────────▼─────────────────────┐      │
│  │     OTLP Exporter (HTTP/gRPC)          │      │
│  │  (导出Span：通过OTLP协议发送)           │      │
│  └─────────────────┬─────────────────────┘      │
└────────────────────│─────────────────────────────┘
                     │
              OTLP Protocol
                     │
┌────────────────────▼─────────────────────────────┐
│              Phoenix Server                       │
│  ┌──────────────────────────────────────────┐    │
│  │  Span接收 → 存储 → 索引 → 查询            │    │
│  └──────────────────────────────────────────┘    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │  Traces  │ │ Datasets │ │Experiments│         │
│  │  UI      │ │ Manager  │ │  Runner   │         │
│  └──────────┘ └──────────┘ └──────────┘         │
│  ┌──────────┐ ┌──────────┐                       │
│  │ Embedding│ │ Playground│                      │
│  │ Explorer │ │          │                       │
│  └──────────┘ └──────────┘                       │
└───────────────────────────────────────────────────┘
```

**关键架构决策**：

1. **OpenTelemetry标准**：Phoenix不发明新的追踪协议，而是完全基于OTEL标准。这意味着：
   - 追踪代码可以与任何OTEL兼容后端互操作
   - 可以使用标准的OTEL采样、处理、导出组件
   - 学习成本更低（OTEL是行业标准）

2. **OpenInference语义规范**：在OTEL之上，Phoenix使用OpenInference定义LLM特定的语义约定（如`openinference_span_kind`、`llm.model_name`等），使得不同框架的追踪数据具有统一的语义。

3. **本地优先**：Phoenix服务器默认在本地运行，所有数据存储在本地。这确保了数据隐私，并消除了对云服务的依赖。

4. **自动插桩**：通过OpenInference Instrumentation，Phoenix可以在不修改应用代码的情况下，自动拦截和追踪框架调用（如LangChain的链调用、OpenAI的API调用等）。

5. **轻量子包架构**：将完整平台（`arize-phoenix`）拆分为轻量子包（`arize-phoenix-otel`、`arize-phoenix-client`、`arize-phoenix-evals`），开发者只需安装所需部分。

## 7. 常见注意事项和最佳实践

### 7.1 生产环境部署

- **使用批量Span处理器**：生产环境中设置`batch=True`，减少网络开销
- **合理配置采样率**：高流量应用使用`TraceIdRatioBased`采样器降低数据量
- **使用容器化部署**：Docker/Kubernetes部署Phoenix服务器，确保稳定性和可扩展性
- **设置API密钥**：生产环境务必设置`PHOENIX_API_KEY`

```python
from phoenix.otel import register
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

tracer_provider = register(
    project_name="production-app",
    batch=True,                           # 批量发送
    sampler=TraceIdRatioBased(0.1),        # 10%采样率
    endpoint="http://phoenix-server:6006/v1/traces",
)
```

### 7.2 Span设计最佳实践

- **合理选择Span类型**：使用正确的`openinference_span_kind`，便于Phoenix UI正确分类和展示
- **记录关键属性**：为LLM Span设置`llm.model_name`、`llm.token_count`等属性
- **避免过深的嵌套**：Span层级不宜过深（建议不超过5层），否则会影响可读性
- **使用装饰器优先**：优先使用`@tracer.chain`等装饰器，代码更简洁

### 7.3 评估最佳实践

- **组合多个评估器**：使用多个评估维度（忠实度、相关性、简洁性）综合评估
- **先小规模验证**：在大规模评估前，先用小数据集验证评估器的有效性
- **使用异步评估**：大数据集使用`async_evaluate_dataframe`，可提升20倍速度
- **人工抽检**：定期人工审核评估结果，确保LLM评估器的准确性

### 7.4 常见注意事项

- **Python版本要求**：Phoenix要求Python >= 3.10, < 3.15
- **框架集成包必须安装**：`auto_instrument=True`只追踪已安装了对应OpenInference包的框架
- **数据隐私**：Phoenix默认不在本地收集追踪数据，但需注意不要将敏感信息（如API密钥）写入Span的input/output
- **使用suppress_tracing**：对于不需要追踪的内部操作（如缓存查询），使用`suppress_tracing()`避免不必要的追踪开销
- **嵌入可视化需要足够的样本量**：UMAP降维在样本量过少时效果不佳，建议至少100+样本
- **环境变量优先级**：`register()`的显式参数优先级高于环境变量

### 7.5 与其他工具的对比

| 特性 | Arize Phoenix | LangSmith | MLflow |
|------|--------------|-----------|--------|
| 开源 | 是 | 否（SaaS） | 是 |
| 本地部署 | 原生支持 | 不支持 | 支持 |
| 追踪标准 | OpenTelemetry | 自有协议 | 部分OTEL |
| 供应商锁定 | 无 | 高 | 低 |
| 嵌入可视化 | 内置UMAP | 无 | 无 |
| 自动追踪框架数 | 20+ | LangChain为主 | 有限 |
| 评估功能 | 内置LLM评估器 | 内置 | 需自行实现 |

### 7.6 故障排查

- **看不到追踪数据**：检查Phoenix服务器是否运行，端点配置是否正确，集成包是否安装
- **追踪数据不完整**：检查是否使用了`batch=True`且应用退出过快（批量数据可能未及时发送）
- **评估结果不准确**：检查评估用的LLM模型能力，尝试更换更强大的模型
- **嵌入可视化无聚类**：检查嵌入质量，确认样本量足够，尝试调整UMAP参数
