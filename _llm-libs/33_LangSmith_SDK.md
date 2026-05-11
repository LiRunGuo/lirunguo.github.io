---
title: "LangSmith SDK 可观测性"
excerpt: "@traceable追踪、evaluate()评估、数据集管理、trace→span层级"
collection: llm-libs
permalink: /llm-libs/33-langsmith
category: eval
---


## 1. 库的简介和在LLM开发中的作用

LangSmith是LangChain团队推出的LLM应用可观测性与评估平台，LangSmith SDK是其Python客户端库。在LLM开发中，调试和监控复杂的LLM应用链（Chain）是一个巨大挑战——一次用户请求可能涉及多步推理、多次API调用、文档检索等多个环节，传统日志无法有效追踪这些复杂的调用链。LangSmith通过提供trace级别的可观测性，让开发者能够清晰地看到LLM应用中每一步的输入输出、耗时、token消耗等信息。

LangSmith SDK的核心价值：
- **追踪可观测性**：自动记录LLM应用中每一步的输入输出、耗时、token消耗
- **评估框架**：内置评估功能，支持自定义评估器，对LLM输出进行自动化质量评估
- **数据集管理**：创建、上传、管理测试数据集，支持版本控制
- **与LangChain深度集成**：自动追踪LangChain组件，无需额外代码
- **协作支持**：团队共享追踪数据、评估结果，支持多人协作调试

## 2. 安装方式

```bash
# 基础安装
pip install langsmith

# 与LangChain一起安装（推荐）
pip install langchain langsmith

# 安装特定版本
pip install langsmith==0.1.50
```

配置环境变量：

```python
import os

# 必需：LangSmith API密钥
os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_..."

# 必需：项目名称（用于组织追踪数据）
os.environ["LANGSMITH_PROJECT"] = "my-llm-project"

# 可选：启用/禁用追踪
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # 启用追踪

# 可选：LangSmith端点（自托管时修改）
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
```

验证安装：

```python
import langsmith
print(langsmith.__version__)

# 验证API连接
from langsmith import Client
client = Client()
print(client.list_projects())
```

## 3. 核心类/函数/工具的详细说明

### 3.1 配置

LangSmith的配置主要通过环境变量完成：

```python
import os

# === 核心配置 ===
os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_..."          # API密钥
os.environ["LANGSMITH_PROJECT"] = "my-project"            # 项目名称

# === 追踪配置 ===
os.environ["LANGCHAIN_TRACING_V2"] = "true"               # 启用v2追踪
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"  # API端点

# === 高级配置 ===
os.environ["LANGSMITH_HIDE_INPUTS"] = "false"             # 是否隐藏输入（隐私保护）
os.environ["LANGSMITH_HIDE_OUTPUTS"] = "false"            # 是否隐藏输出
os.environ["LANGSMITH_HIDE_SECRETS"] = "true"             # 自动隐藏敏感信息
os.environ["LANGSMITH_SAMPLE_RATE"] = "1.0"               # 采样率（0.0-1.0）
os.environ["LANGSMITH_TRACING"] = "true"                  # 启用追踪的另一种方式
```

也可以通过Client对象在代码中配置：

```python
from langsmith import Client

client = Client(
    api_url="https://api.smith.langchain.com",
    api_key="lsv2_pt_...",
    timeout=30
)
```

### 3.2 @traceable装饰器

`@traceable`是LangSmith SDK最核心的功能之一，用于将任意Python函数标记为可追踪的。

```python
from langsmith import traceable

@traceable(
    name="my_function",           # 可选：追踪中显示的名称，默认为函数名
    run_type="chain",             # 可选：运行类型，如"chain"、"llm"、"retriever"、"tool"
    metadata={"version": "1.0"},  # 可选：元数据字典
    tags=["production", "v2"],    # 可选：标签列表
    reduce_fn=None,               # 可选：用于聚合子追踪结果的函数
)
def my_function(input_text: str) -> str:
    # 函数逻辑
    return f"处理结果: {input_text}"

# 调用时自动记录追踪
result = my_function("你好")
```

**run_type取值说明：**

| run_type | 说明 | 适用场景 |
|----------|------|----------|
| `chain` | 链式调用 | 多步骤组合逻辑 |
| `llm` | LLM调用 | 直接调用语言模型 |
| `retriever` | 检索器 | 文档检索 |
| `tool` | 工具调用 | 外部工具/API调用 |
| `parser` | 解析器 | 输出解析 |
| `embedding` | 嵌入 | 向量嵌入计算 |

**嵌套追踪：**

```python
from langsmith import traceable
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

@traceable(name="retriever", run_type="retriever")
def retrieve_documents(query: str) -> list:
    # 模拟文档检索
    return [f"文档1: {query}的相关信息", f"文档2: 更多关于{query}的内容"]

@traceable(name="generator", run_type="chain")
def generate_answer(query: str, documents: list) -> str:
    context = "\n".join(documents)
    response = llm.invoke(f"基于以下上下文回答问题：\n{context}\n\n问题：{query}")
    return response.content

@traceable(name="rag_pipeline", run_type="chain")
def rag_pipeline(query: str) -> str:
    # 嵌套追踪：rag_pipeline → retriever → generator
    docs = retrieve_documents(query)
    answer = generate_answer(query, docs)
    return answer

# 执行后，LangSmith中会看到完整的追踪树
result = rag_pipeline("什么是量子计算？")
```

**手动创建追踪：**

```python
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

@traceable(name="manual_trace")
def my_function(input_text: str) -> str:
    # 获取当前追踪上下文
    run = get_current_run_tree()

    # 添加元数据
    run.metadata["custom_field"] = "custom_value"

    # 手动添加子追踪
    with run.new_child(
        name="sub_step",
        run_type="tool",
        inputs={"param": "value"}
    ) as child_run:
        result = do_something()
        child_run.outputs = {"result": result}

    return result
```

### 3.3 Runnable追踪

LangChain的Runnable接口自动与LangSmith集成，无需额外代码即可追踪。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 只要配置了环境变量，以下链的所有步骤会自动追踪
llm = ChatOpenAI(model="gpt-4")

prompt = ChatPromptTemplate.from_template("用中文解释：{topic}")
chain = prompt | llm | StrOutputParser()

# 调用chain时，LangSmith自动记录：
# - prompt的输入和格式化后的提示词
# - LLM的调用参数、输入、输出、token消耗
# - 解析器的输入和输出
result = chain.invoke({"topic": "量子计算"})
```

**自动追踪的组件：**

| 组件类型 | 追踪内容 |
|----------|----------|
| ChatModel / LLM | 输入消息、模型参数、输出、token统计 |
| Chain | 链的输入输出、子组件调用 |
| Retriever | 查询、检索结果文档 |
| Tool | 工具输入输出 |
| Agent | 思考过程、工具调用决策 |
| Output Parser | 解析前后的数据 |

### 3.4 评估框架

LangSmith内置了评估框架，支持对LLM应用进行自动化评估。

#### evaluate() — 评估函数

```python
from langsmith.evaluation import evaluate

results = evaluate(
    target=target_function,           # 必需：被评估的目标函数
    data=dataset_name,                # 必需：数据集名称或数据集对象
    evaluators=[evaluator1, evaluator2], # 必需：评估器列表
    experiment_prefix="my-exp",       # 可选：实验名称前缀
    description="评估RAG系统质量",     # 可选：实验描述
    max_concurrency=5,                # 可选：最大并发数
    metadata={"version": "2.0"},      # 可选：元数据
)
```

**参数详解：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `target` | Callable | 被评估的目标函数，接收数据输入，返回输出 |
| `data` | str/Dataset | 数据集名称或对象 |
| `evaluators` | list | 评估器列表 |
| `experiment_prefix` | str | 实验名称前缀，便于区分不同实验 |
| `max_concurrency` | int | 最大并发评估数 |
| `metadata` | dict | 附加元数据 |

#### 自定义评估器

```python
from langsmith.evaluation import Evaluator, EvaluationResult

class RelevanceEvaluator(Evaluator):
    """自定义相关性评估器"""

    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)

    def evaluate(self, *, run, example) -> EvaluationResult:
        # run: 目标函数的运行结果
        # example: 数据集中的样本（包含input和reference_output）
        input_question = example.inputs["question"]
        actual_output = run.outputs["output"]
        reference = example.outputs.get("answer", "")

        # 使用LLM评估相关性
        eval_prompt = f"""评估以下回答与问题的相关性。

问题：{input_question}
回答：{actual_output}
参考答案：{reference}

请给出0到1之间的相关性分数。"""

        response = self.llm.invoke(eval_prompt)
        score = self._parse_score(response.content)

        return EvaluationResult(
            score=score,                    # 0.0-1.0之间的分数
            comment=f"相关性评估: {response.content}",  # 评估评论
            key="relevance"                 # 指标键名
        )

# 使用自定义评估器
results = evaluate(
    target=rag_pipeline,
    data="my-qa-dataset",
    evaluators=[RelevanceEvaluator()]
)
```

**简化的评估器定义（函数式）：**

```python
from langsmith.evaluation import evaluate

def relevance_evaluator(run, example) -> dict:
    """简化的函数式评估器"""
    question = example.inputs["question"]
    answer = run.outputs["output"]

    # 简单的关键词匹配评估
    keywords = example.outputs.get("keywords", [])
    covered = sum(1 for kw in keywords if kw in answer)
    score = covered / len(keywords) if keywords else 0

    return {"score": score, "comment": f"覆盖了{covered}/{len(keywords)}个关键词"}

results = evaluate(
    target=rag_pipeline,
    data="my-qa-dataset",
    evaluators=[relevance_evaluator]
)
```

#### 使用LLM评估器

```python
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI

# LangSmith提供了预构建的LLM评估器
from langsmith.evaluation.evaluators import (
    CriteriaEvaluator,
    LabeledCriteriaEvaluator,
    EmbeddingDistance,
    StringDistance
)

# 基于准则的评估（无需参考答案）
helpfulness_evaluator = CriteriaEvaluator(
    criteria="helpfulness",    # 内置准则：helpfulness, conciseness, correctness等
    llm=ChatOpenAI(model="gpt-4", temperature=0)
)

# 带标签的准则评估（需要参考答案）
correctness_evaluator = LabeledCriteriaEvaluator(
    criteria="correctness",
    llm=ChatOpenAI(model="gpt-4", temperature=0)
)

# 嵌入距离评估
embedding_evaluator = EmbeddingDistance(
    embeddings=OpenAIEmbeddings()
)

# 字符串距离评估
string_evaluator = StringDistance(
    distance="edit_distance"  # 编辑距离
)

results = evaluate(
    target=my_llm_function,
    data="test-dataset",
    evaluators=[
        helpfulness_evaluator,
        correctness_evaluator,
        embedding_evaluator
    ]
)
```

### 3.5 数据集管理

LangSmith提供了完整的数据集管理功能。

```python
from langsmith import Client

client = Client()

# === 创建数据集 ===

# 方式1：从字典列表创建
dataset = client.create_dataset(
    dataset_name="qa-test-dataset",
    description="问答系统测试数据集"
)

# 添加样本
client.create_examples(
    dataset_id=dataset.id,
    inputs=[
        {"question": "什么是机器学习？"},
        {"question": "深度学习和机器学习的区别？"},
    ],
    outputs=[
        {"answer": "机器学习是让计算机从数据中自动学习的技术。", "keywords": ["机器学习", "数据", "自动学习"]},
        {"answer": "深度学习是机器学习的子集，使用多层神经网络。", "keywords": ["深度学习", "机器学习", "神经网络"]},
    ]
)

# 方式2：从CSV上传
client.upload_csv(
    dataset_name="csv-dataset",
    csv_file="test_data.csv",
    input_keys=["question"],         # 输入列
    output_keys=["answer", "keywords"] # 输出列
)

# === 查询数据集 ===

# 列出所有数据集
datasets = list(client.list_datasets())
for ds in datasets:
    print(f"数据集: {ds.name}, 样本数: {client.list_examples(dataset_id=ds.id).total_count}")

# 获取特定数据集
dataset = client.read_dataset(dataset_name="qa-test-dataset")

# 获取数据集中的样本
examples = list(client.list_examples(dataset_id=dataset.id))
for ex in examples:
    print(f"输入: {ex.inputs}, 输出: {ex.outputs}")

# === 更新和删除 ===

# 更新数据集
client.update_dataset(
    dataset_id=dataset.id,
    description="更新后的描述"
)

# 删除数据集
client.delete_dataset(dataset_name="old-dataset")
```

### 3.6 Client — API客户端

`Client`是LangSmith SDK与LangSmith服务端交互的核心类。

```python
from langsmith import Client

client = Client(
    api_url="https://api.smith.langchain.com",  # API端点
    api_key="lsv2_pt_...",                       # API密钥（也可通过环境变量设置）
    timeout=30,                                  # 请求超时时间（秒）
    retry_max=2                                  # 最大重试次数
)

# === 追踪相关 ===

# 读取追踪数据
runs = list(client.list_runs(
    project_name="my-project",
    run_type="chain",
    error=True  # 仅查看出错的运行
))

# 获取特定运行的详情
run = client.read_run(run_id="abc-123")
print(f"运行名称: {run.name}")
print(f"输入: {run.inputs}")
print(f"输出: {run.outputs}")
print(f"耗时: {run.total_time}")
print(f"Token消耗: {run.total_tokens}")

# === 项目管理 ===

# 列出项目
projects = list(client.list_projects())

# 创建项目
project = client.create_project(
    project_name="new-project",
    description="新项目"
)

# === 评估结果 ===

# 读取评估结果
eval_results = list(client.list_evaluations(
    dataset_name="qa-test-dataset"
))
```

## 4. 在LLM开发中的典型使用场景和代码示例

### 4.1 RAG系统完整追踪与评估

```python
import os
from langsmith import traceable, Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 配置LangSmith
os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_..."
os.environ["LANGSMITH_PROJECT"] = "rag-evaluation"

# === 构建带追踪的RAG管道 ===

llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings()

# 假设已有向量数据库
vectorstore = Chroma.from_texts(
    ["机器学习是AI的子领域", "深度学习使用神经网络"],
    embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@traceable(name="retrieve", run_type="retriever")
def retrieve(query: str) -> list:
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]

@traceable(name="generate", run_type="chain")
def generate(query: str, contexts: list) -> str:
    context_text = "\n".join(contexts)
    prompt = f"基于以下上下文回答问题：\n{context_text}\n\n问题：{query}"
    response = llm.invoke(prompt)
    return response.content

@traceable(name="rag_pipeline", run_type="chain")
def rag_pipeline(question: str) -> dict:
    contexts = retrieve(question)
    answer = generate(question, contexts)
    return {"answer": answer, "contexts": contexts}

# === 创建评估数据集 ===

client = Client()
dataset = client.create_dataset(
    dataset_name="rag-eval-dataset",
    description="RAG系统评估数据集"
)

client.create_examples(
    dataset_id=dataset.id,
    inputs=[
        {"question": "什么是机器学习？"},
        {"question": "深度学习和机器学习有什么区别？"},
    ],
    outputs=[
        {"answer": "机器学习是让计算机从数据中自动学习的AI技术。"},
        {"answer": "深度学习是机器学习的子集，核心是多层神经网络。"},
    ]
)

# === 定义评估器 ===

def answer_quality_evaluator(run, example) -> dict:
    """评估答案质量"""
    actual = run.outputs["answer"]
    expected = example.outputs["answer"]

    eval_llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = f"""评估以下答案的质量。

问题：{example.inputs["question"]}
实际答案：{actual}
参考答案：{expected}

请从准确性和完整性两个维度评估，给出0-1之间的分数。"""

    response = eval_llm.invoke(prompt)
    score = 0.8  # 实际中应解析response
    return {"score": score}

def context_relevance_evaluator(run, example) -> dict:
    """评估检索上下文的相关性"""
    contexts = run.outputs.get("contexts", [])
    question = example.inputs["question"]

    if not contexts:
        return {"score": 0.0, "comment": "无检索上下文"}

    eval_llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = f"""评估检索到的上下文与问题的相关性。

问题：{question}
检索上下文：{contexts}

给出0-1之间的相关性分数。"""

    response = eval_llm.invoke(prompt)
    score = 0.7  # 实际中应解析response
    return {"score": score}

# === 执行评估 ===

def target_function(inputs: dict) -> dict:
    return rag_pipeline(inputs["question"])

results = evaluate(
    target=target_function,
    data="rag-eval-dataset",
    evaluators=[answer_quality_evaluator, context_relevance_evaluator],
    experiment_prefix="rag-v1"
)
```

### 4.2 Agent工作流追踪

```python
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import tool

# 定义带追踪的工具
@tool
@traceable(name="search_tool", run_type="tool")
def search_tool(query: str) -> str:
    """搜索相关信息"""
    return f"搜索结果：关于{query}的信息..."

@tool
@traceable(name="calculator", run_type="tool")
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"

# 创建Agent
# Agent的每一步思考、工具调用都会被自动追踪
llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [search_tool, calculator]
# agent = create_openai_functions_agent(llm, tools)
# agent_executor = AgentExecutor(agent=agent, tools=tools)

# 调用Agent时，LangSmith会记录：
# - Agent的每一步推理
# - 工具调用的输入输出
# - 最终答案
# result = agent_executor.invoke({"input": "计算2的10次方加上3的5次方"})
```

### 4.3 对比不同提示词模板

```python
from langsmith import traceable
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI

# 定义不同版本的提示词
prompt_v1 = """请回答以下问题：{question}"""

prompt_v2 = """你是一个专业的AI助手。请基于你的知识，详细、准确地回答以下问题：
{question}

要求：
1. 答案要准确无误
2. 尽量提供具体的细节和例子
3. 如果不确定，请明确说明"""

@traceable(name="llm_v1")
def answer_v1(inputs: dict) -> dict:
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    response = llm.invoke(prompt_v1.format(question=inputs["question"]))
    return {"answer": response.content}

@traceable(name="llm_v2")
def answer_v2(inputs: dict) -> dict:
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    response = llm.invoke(prompt_v2.format(question=inputs["question"]))
    return {"answer": response.content}

# 对比评估
def quality_evaluator(run, example) -> dict:
    answer = run.outputs["answer"]
    reference = example.outputs.get("answer", "")
    # 评估逻辑...
    return {"score": 0.8}

results_v1 = evaluate(
    target=answer_v1,
    data="qa-dataset",
    evaluators=[quality_evaluator],
    experiment_prefix="prompt-v1"
)

results_v2 = evaluate(
    target=answer_v2,
    data="qa-dataset",
    evaluators=[quality_evaluator],
    experiment_prefix="prompt-v2"
)
```

### 4.4 生产环境监控

```python
from langsmith import traceable, Client
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

@traceable(
    name="production_chat",
    run_type="chain",
    metadata={"environment": "production", "version": "2.1.0"},
    tags=["production", "chat-api"]
)
def chat(user_input: str, user_id: str) -> str:
    response = llm.invoke(user_input)
    return response.content

# 在生产API中使用
def handle_request(user_input: str, user_id: str):
    result = chat(user_input, user_id)
    return result

# 事后分析追踪数据
client = Client()

# 查看最近的错误
error_runs = list(client.list_runs(
    project_name="production-project",
    error=True,
    limit=10
))

for run in error_runs:
    print(f"运行: {run.name}, 错误: {run.error}, 时间: {run.start_time}")

# 分析延迟
recent_runs = list(client.list_runs(
    project_name="production-project",
    limit=100
))

latencies = [run.total_time for run in recent_runs if run.total_time]
avg_latency = sum(latencies) / len(latencies) if latencies else 0
print(f"平均延迟: {avg_latency:.2f}秒")

# 分析token消耗
total_tokens = sum(run.total_tokens or 0 for run in recent_runs)
print(f"总token消耗: {total_tokens}")
```

## 5. 数学原理

LangSmith SDK的核心不在于数学计算，而在于系统架构和可观测性设计。其数学相关的主要体现在评估框架中的相似度计算。

### 5.1 嵌入距离评估

嵌入距离评估使用向量空间中的距离度量来比较LLM输出与参考答案的相似度：

**余弦相似度：**

$$\cos(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

**欧几里得距离：**

$$d(A, B) = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}$$

LangSmith的`EmbeddingDistance`评估器支持多种距离度量方式。

### 5.2 字符串距离评估

**编辑距离（Levenshtein Distance）：**

$$d(s_1, s_2) = \min\{\text{insertions, deletions, substitutions}\}$$

即从一个字符串转换到另一个字符串所需的最少编辑操作次数。

### 5.3 LLM评估器的评分逻辑

基于准则的评估器（CriteriaEvaluator）的核心逻辑：

1. 将评估准则和待评估内容组合为提示词
2. LLM输出结构化评估结果（分数+理由）
3. 解析LLM输出提取数值分数

这种评估方式的数学本质是：

$$\text{Score} = \text{LLM}(\text{Criteria}, \text{Input}, \text{Output}, \text{Reference})$$

## 6. 代码原理/架构原理

### 6.1 追踪层级结构

LangSmith的追踪采用trace→span→child span的层级结构，类似于OpenTelemetry的追踪模型：

```
Trace (一次完整的用户请求)
├── Span: rag_pipeline (chain)
│   ├── Child Span: retrieve (retriever)
│   │   ├── 输入: {"query": "什么是量子计算？"}
│   │   ├── 输出: {"documents": ["文档1...", "文档2..."]}
│   │   ├── 耗时: 0.3s
│   │   └── 元数据: {"source": "chroma", "k": 3}
│   │
│   ├── Child Span: generate (chain)
│   │   ├── Child Span: llm_call (llm)
│   │   │   ├── 输入: [{"role": "user", "content": "..."}]
│   │   │   ├── 输出: {"content": "量子计算是..."}
│   │   │   ├── Token: {prompt: 150, completion: 80}
│   │   │   └── 耗时: 1.2s
│   │   │
│   │   └── 输出: {"answer": "量子计算是..."}
│   │
│   └── 输出: {"answer": "量子计算是...", "contexts": [...]}
```

### 6.2 追踪数据流

```
┌───────────────────────────────────────┐
│            应用代码                    │
│                                       │
│  @traceable ──→ RunTree 创建          │
│       │          │                    │
│       │          ├── 记录输入          │
│       │          ├── 执行函数体        │
│       │          ├── 记录输出          │
│       │          └── 计算耗时          │
│       │                               │
│       └──→ 子函数 @traceable          │
│              └──→ Child RunTree       │
│                     └──→ 递归嵌套      │
│                                       │
└───────────────┬───────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│        LangSmith SDK 内部              │
│                                       │
│  ┌─────────────────────────────────┐  │
│  │      RunTree Manager            │  │
│  │  - 管理追踪树结构                │  │
│  │  - 处理嵌套关系                  │  │
│  │  - 序列化为JSON                  │  │
│  └─────────────────────────────────┘  │
│                │                      │
│  ┌─────────────▼───────────────────┐  │
│  │      Batch Sender               │  │
│  │  - 批量发送追踪数据              │  │
│  │  - 异步非阻塞                    │  │
│  │  - 自动重试                      │  │
│  └─────────────────────────────────┘  │
│                │                      │
└────────────────┼──────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────┐
│         LangSmith 服务端               │
│                                       │
│  - 存储追踪数据                        │
│  - 构建索引                           │
│  - 提供查询API                        │
│  - 生成可视化界面                      │
└───────────────────────────────────────┘
```

### 6.3 @traceable实现原理

`@traceable`装饰器的核心实现逻辑：

1. **函数包装**：将原函数包装为一个可追踪的函数
2. **上下文传播**：通过Python的contextvars模块传播追踪上下文
3. **自动嵌套**：在已存在追踪上下文中调用时，自动创建子追踪
4. **异常处理**：捕获异常并记录到追踪中
5. **异步支持**：同时支持同步和异步函数

```python
# 简化的实现原理
import functools
from contextvars import ContextVar

current_run: ContextVar = ContextVar("current_run", default=None)

def traceable(name=None, run_type="chain", **kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs_inner):
            parent_run = current_run.get()

            # 创建新的RunTree
            run = RunTree(
                name=name or func.__name__,
                run_type=run_type,
                inputs=extract_inputs(args, kwargs_inner),
                parent=parent_run
            )

            # 设置为当前追踪上下文
            token = current_run.set(run)

            try:
                # 执行原函数
                result = func(*args, **kwargs_inner)
                run.outputs = result
                run.end_time = datetime.now()
                return result
            except Exception as e:
                run.error = str(e)
                raise
            finally:
                # 恢复父追踪上下文
                current_run.reset(token)
                # 异步发送追踪数据
                send_run_async(run)

        return wrapper
    return decorator
```

### 6.4 评估框架架构

```
┌──────────────────────────────────────────┐
│            evaluate()                     │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │  1. 加载数据集                      │  │
│  │     client.list_examples()         │  │
│  └───────────────┬────────────────────┘  │
│                  │                        │
│  ┌───────────────▼────────────────────┐  │
│  │  2. 执行目标函数                    │  │
│  │     target(example.inputs)         │  │
│  │     → 生成 Run                     │  │
│  └───────────────┬────────────────────┘  │
│                  │                        │
│  ┌───────────────▼────────────────────┐  │
│  │  3. 运行评估器                      │  │
│  │     evaluator.evaluate(            │  │
│  │       run=target_run,              │  │
│  │       example=example              │  │
│  │     )                              │  │
│  │     → 生成 EvaluationResult        │  │
│  └───────────────┬────────────────────┘  │
│                  │                        │
│  ┌───────────────▼────────────────────┐  │
│  │  4. 汇总结果                        │  │
│  │     上传到LangSmith服务端           │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

## 7. 常见注意事项和最佳实践

### 7.1 追踪性能优化

```python
# 大规模生产环境中，追踪可能影响性能
# 使用采样率控制追踪数据量
os.environ["LANGSMITH_SAMPLE_RATE"] = "0.1"  # 仅追踪10%的请求

# 对非关键路径禁用追踪
from langsmith import traceable

@traceable(name="non_critical_step", run_type="tool")
def non_critical_step():
    # 不影响核心业务逻辑的步骤
    pass

# 关闭追踪（调试完成后）
os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

### 7.2 敏感信息处理

```python
# 隐藏输入输出中的敏感信息
os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"

# 自动隐藏密钥等敏感信息
os.environ["LANGSMITH_HIDE_SECRETS"] = "true"

# 手动过滤敏感字段
@traceable(name="safe_pipeline")
def safe_pipeline(user_input: str) -> str:
    # 在追踪中只记录脱敏后的数据
    safe_input = user_input.replace(os.getenv("API_KEY", ""), "***")
    # ... 处理逻辑
    return result
```

### 7.3 项目组织

```python
# 按环境和功能组织项目
os.environ["LANGSMITH_PROJECT"] = "prod-customer-support"  # 生产环境
os.environ["LANGSMITH_PROJECT"] = "dev-experiment-v2"      # 开发实验

# 按版本追踪
@traceable(
    name="chat_api",
    metadata={"version": "2.1.0", "model": "gpt-4"},
    tags=["v2", "production"]
)
def chat_api(input_text):
    pass
```

### 7.4 常见问题

**问题1：追踪数据未出现在LangSmith中**

```python
# 检查配置
import os
print(os.environ.get("LANGCHAIN_TRACING_V2"))  # 应为"true"
print(os.environ.get("LANGSMITH_API_KEY"))     # 应非空
print(os.environ.get("LANGSMITH_PROJECT"))     # 应为有效项目名

# 确保LangChain版本兼容
# pip install --upgrade langchain langsmith
```

**问题2：评估数据集格式不匹配**

```python
# 确保target函数的输入格式与数据集的inputs字段匹配
# 数据集inputs: {"question": "..."}
# target函数应接收: def target(inputs: dict) -> dict:
#     question = inputs["question"]  # 从inputs中提取question字段
#     ...
#     return {"answer": result}  # 返回字典
```

**问题3：追踪嵌套过深导致性能问题**

```python
# 对内部实现细节不需要追踪的函数，不添加@traceable
# 仅对关键的入口函数和重要步骤添加追踪

@traceable(name="rag_pipeline")  # 追踪入口
def rag_pipeline(query):
    docs = retrieve_docs(query)      # 不追踪
    answer = generate_answer(query, docs)  # 不追踪
    return answer

# 而不是对每个内部函数都添加追踪
```

### 7.5 最佳实践总结

1. **入口函数必须追踪**：每个主要API端点或用户交互入口都应添加`@traceable`
2. **合理设置run_type**：正确标注每一步的类型，便于在LangSmith中过滤和分析
3. **添加有意义的元数据**：如版本号、模型名称、配置参数等，便于追溯和对比
4. **使用标签分类**：如`["production", "v2"]`，便于按标签筛选追踪
5. **评估与追踪结合**：在评估时自动记录追踪，便于分析低分样本的原因
6. **定期清理数据**：LangSmith追踪数据会累积，定期清理旧数据
7. **分离开发和生产项目**：避免开发实验数据污染生产监控数据
8. **建立评估基线**：首次评估的结果作为基线，后续变更对比基线检测退化
9. **利用比较功能**：LangSmith UI支持并排对比不同实验的结果
