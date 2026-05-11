---
title: "Ragas RAG评估框架"
excerpt: "evaluate()、Faithfulness/AnswerRelevancy/ContextPrecision等指标、TestsetGenerator"
collection: llm-libs
permalink: /llm-libs/31-ragas
category: eval
toc: true
---


## 1. 库的简介和在LLM开发中的作用

Ragas（RAG Assessment）是一个专门用于评估检索增强生成（Retrieval-Augmented Generation, RAG）系统的开源框架。在LLM开发中，RAG架构已成为让大语言模型利用外部知识生成答案的主流范式，但如何量化评估RAG系统的质量一直是一个挑战。Ragas提供了一套全面的指标体系，从答案忠实度、答案相关性、上下文精确度、上下文召回率等多个维度评估RAG管道的性能。

Ragas的核心价值：
- **多维度评估**：不局限于单一指标，而是从答案质量和检索质量两个维度全面评估
- **自动化评估**：利用LLM自动进行评估，减少人工标注成本
- **测试数据生成**：可通过TestsetGenerator自动生成评估用的测试数据集
- **与HuggingFace生态集成**：基于Dataset格式，便于与主流ML工具链配合
- **RAG管道集成**：可直接与LangChain、LlamaIndex等框架集成

## 2. 安装方式

```bash
# 基础安装
pip install ragas

# 从源码安装
pip install git+https://github.com/explodinggradients/ragas.git

# 安装特定版本
pip install ragas==0.1.9

# 安装包含所有依赖的完整版本
pip install "ragas[all]"
```

安装后验证：

```python
import ragas
print(ragas.__version__)
```

## 3. 核心类/函数/工具的详细说明

### 3.1 evaluate() — 核心评估函数

`evaluate()`是Ragas的核心评估入口，用于对RAG系统的输出进行多维度评估。

```python
from ragas import evaluate

result = evaluate(
    dataset=dataset,           # 必需：HuggingFace Dataset或DatasetDict，包含评估数据
    metrics=[metric1, metric2], # 必需：评估指标列表
    llm=llm,                   # 可选：用于评估的LLM，默认为ChatOpenAI(gpt-3.5-turbo)
    embeddings=embeddings,     # 可选：用于语义计算的嵌入模型
    column_map=None,           # 可选：列名映射字典，用于自定义数据集列名
    is_async=True,             # 可选：是否异步执行，默认True
    max_workers=16,            # 可选：最大并发工作数
    raise_exceptions=False,    # 可选：是否在遇到异常时抛出，默认False
    batch_size=None,           # 可选：批处理大小
)
```

**参数详解：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `dataset` | Dataset/DatasetDict | 评估数据集，必须包含question、answer、contexts、ground_truth列 |
| `metrics` | list[Metric] | 评估指标列表，如[Faithfulness(), AnswerRelevancy()] |
| `llm` | BaseLLM | 用于评估的LLM实例，支持LangChain ChatModel |
| `embeddings` | BaseEmbeddings | 嵌入模型，用于语义相似度计算 |
| `column_map` | dict | 列名映射，如{"question": "query", "answer": "response"} |
| `is_async` | bool | 异步执行可显著提升评估速度 |
| `max_workers` | int | 并发评估的最大工作线程数 |
| `batch_size` | int | 每批评估的样本数 |

**返回值：** 返回一个包含各指标得分的字典，如 `{'faithfulness': 0.85, 'answer_relevancy': 0.72}`。

**基本使用示例：**

```python
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from datasets import Dataset

# 准备评估数据
data = {
    "question": ["什么是量子计算？", "Python的GIL是什么？"],
    "answer": ["量子计算利用量子比特进行计算...", "GIL是全局解释器锁..."],
    "contexts": [["量子计算是利用量子力学原理..."], ["GIL是CPython中的互斥锁..."]],
    "ground_truth": ["量子计算是利用量子比特的信息处理方式", "GIL确保同一时刻只有一个线程执行Python字节码"]
}

dataset = Dataset.from_dict(data)

# 执行评估
result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()],
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    embeddings=OpenAIEmbeddings()
)

print(result)
# 输出示例: {'faithfulness': 0.85, 'answer_relevancy': 0.72, 'context_recall': 0.80, 'context_precision': 0.75}
```

### 3.2 指标详解

#### 3.2.1 Faithfulness — 答案忠实度

衡量生成的答案是否忠实于检索到的上下文，即答案中的陈述是否都能在上下文中找到依据。

```python
from ragas.metrics import Faithfulness

faithfulness = Faithfulness(
    # 无需额外参数，使用evaluate()中传入的llm
)

# 单独使用
score = faithfulness.score(
    question="什么是量子计算？",
    answer="量子计算利用量子比特进行计算，可以实现指数级加速。",
    contexts=["量子计算是利用量子力学原理（如叠加态和纠缠）进行信息处理的计算范式。"]
)
```

**评估流程：**
1. 将answer拆分为多个原子化声明（claims）
2. 对每个claim，判断是否可被context中的信息支持
3. 计算支持的claim占总claim的比例

#### 3.2.2 Answer Relevancy — 答案相关性

衡量生成的答案与问题的相关程度。

```python
from ragas.metrics import AnswerRelevancy

answer_relevancy = AnswerRelevancy(
    # 可自定义严格度
)

# 单独使用
score = answer_relevancy.score(
    question="什么是量子计算？",
    answer="量子计算利用量子比特进行计算。",
    contexts=["量子计算的相关背景信息..."]  # 可选，用于提升评估质量
)
```

**评估流程：**
1. 使用LLM根据answer反向生成若干个可能的问题
2. 计算每个生成问题与原始问题的语义相似度
3. 取平均值作为相关性得分

#### 3.2.3 Context Precision — 上下文精确度

衡量检索到的上下文中相关文档的排名质量，即相关文档是否排在前面。

```python
from ragas.metrics import ContextPrecision

context_precision = ContextPrecision()

# 单独使用
score = context_precision.score(
    question="什么是量子计算？",
    answer="量子计算利用量子比特进行计算。",
    contexts=["量子力学基础知识...", "量子计算利用量子比特进行信息处理..."],
    ground_truth="量子计算是利用量子比特的信息处理方式"
)
```

**评估流程：**
1. 判断contexts中每个文档是否与ground_truth相关
2. 计算每个位置的精确度precision@k
3. 加权平均得到最终得分

#### 3.2.4 Context Recall — 上下文召回率

衡量检索到的上下文是否包含了回答问题所需的所有信息。

```python
from ragas.metrics import ContextRecall

context_recall = ContextRecall()

# 单独使用
score = context_recall.score(
    question="什么是量子计算？",
    answer="量子计算利用量子比特进行计算。",
    contexts=["量子计算利用量子比特进行信息处理..."],
    ground_truth="量子计算是利用量子比特的信息处理方式"
)
```

**评估流程：**
1. 将ground_truth拆分为多个关键声明
2. 判断每个声明是否可从contexts中推导出来
3. 计算可推导的声明占总声明的比例

#### 3.2.5 Answer Similarity — 答案语义相似度

衡量生成答案与参考答案之间的语义相似度。

```python
from ragas.metrics import AnswerSimilarity

answer_similarity = AnswerSimilarity()

# 单独使用
score = answer_similarity.score(
    answer="量子计算利用量子比特进行计算",
    ground_truth="量子计算是利用量子比特的信息处理方式"
    # 无需question和contexts
)
```

**评估流程：**
1. 使用嵌入模型将answer和ground_truth分别编码为向量
2. 计算两个向量的余弦相似度

#### 3.2.6 Answer Correctness — 答案正确性

综合衡量答案的正确性，结合语义相似度和事实正确性。

```python
from ragas.metrics import AnswerCorrectness

answer_correctness = AnswerCorrectness(
    weights=[0.5, 0.5]  # [语义相似度权重, 事实正确性权重]
)

# 单独使用
score = answer_correctness.score(
    question="什么是量子计算？",
    answer="量子计算利用量子比特进行计算。",
    ground_truth="量子计算是利用量子比特的信息处理方式"
)
```

**评估流程：**
1. 计算语义相似度（Answer Similarity）
2. 计算事实正确性（类似Faithfulness但对照ground_truth）
3. 按权重加权平均

### 3.3 TestsetGenerator — 测试数据生成器

TestsetGenerator可以基于文档自动生成评估用的测试数据集。

```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 配置生成器
generator = TestsetGenerator.with_openai(
    generator_llm=ChatOpenAI(model="gpt-4"),
    critic_llm=ChatOpenAI(model="gpt-4"),      # 质量审查LLM
    embeddings=OpenAIEmbeddings()
)

# 或者使用通用方式
generator = TestsetGenerator(
    generator_llm=ChatOpenAI(model="gpt-4"),
    critic_llm=ChatOpenAI(model="gpt-4"),
    embeddings=OpenAIEmbeddings()
)

# 加载文档
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader("./docs", loader_cls=TextLoader)
documents = loader.load()

# 生成测试集
testset = generator.generate_with_llamaindex_docs(
    documents=documents,
    test_size=50,                # 生成的问题数量
    distributions={              # 问题类型分布
        simple: 0.5,             # 50%简单问题
        reasoning: 0.3,          # 30%推理问题
        multi_context: 0.2       # 20%多上下文问题
    }
)

# 也可以使用LangChain文档
testset = generator.generate_with_langchain_docs(
    documents=documents,
    test_size=50,
    distributions={simple: 0.5, reasoning: 0.3, multi_context: 0.2}
)

# 转换为pandas DataFrame
df = testset.to_pandas()
print(df.head())

# 转换为HuggingFace Dataset
testset_dataset = testset.to_dataset()
```

**问题演化类型说明：**

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| `simple` | 基于单个文档片段生成的简单问题 | 基础事实检索 |
| `reasoning` | 需要多步推理的问题 | 逻辑推理能力评估 |
| `multi_context` | 需要跨多个文档片段综合回答的问题 | 跨文档检索能力评估 |

### 3.4 数据格式

Ragas使用HuggingFace Dataset格式作为标准数据格式。

**标准列名：**

| 列名 | 类型 | 说明 | 必需指标 |
|------|------|------|----------|
| `question` | str | 用户问题 | 所有指标 |
| `answer` | str | RAG系统生成的答案 | Faithfulness, AnswerRelevancy, AnswerCorrectness |
| `contexts` | list[str] | 检索到的上下文文档列表 | Faithfulness, ContextPrecision, ContextRecall |
| `ground_truth` | str | 参考答案（金标准） | ContextRecall, AnswerCorrectness, AnswerSimilarity |

```python
from datasets import Dataset

# 标准格式构建数据集
data = {
    "question": [
        "什么是机器学习？",
        "深度学习和机器学习有什么区别？"
    ],
    "answer": [
        "机器学习是AI的子领域，通过数据训练模型。",
        "深度学习是机器学习的子集，使用神经网络。"
    ],
    "contexts": [
        ["机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。"],
        ["深度学习使用多层神经网络，是机器学习的一个子集。"]
    ],
    "ground_truth": [
        "机器学习是让计算机从数据中自动学习规律的技术。",
        "深度学习是机器学习的子领域，核心是多层神经网络。"
    ]
}

dataset = Dataset.from_dict(data)

# 自定义列名映射
result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness()],
    column_map={
        "question": "user_query",    # 数据集中的"question"列实际名为"user_query"
        "answer": "response",        # "answer"列实际名为"response"
    }
)
```

## 4. 在LLM开发中的典型使用场景和代码示例

### 4.1 评估完整RAG管道

```python
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    AnswerSimilarity
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset

# 1. 准备RAG系统的输出数据
rag_outputs = {
    "question": [
        "什么是向量数据库？",
        "RAG的工作原理是什么？",
        "如何减少LLM的幻觉？"
    ],
    "answer": [
        "向量数据库是专门存储和检索向量嵌入的数据库系统，支持高效的相似性搜索。",
        "RAG通过检索外部知识库中的相关文档，将检索结果作为上下文输入LLM生成答案。",
        "减少幻觉的方法包括：使用RAG引入外部知识、调整temperature参数、添加事实核查步骤。"
    ],
    "contexts": [
        ["向量数据库如Milvus、Pinecone等，专门用于高维向量的存储和相似性搜索。"],
        ["RAG（检索增强生成）先检索相关文档，再将其注入LLM的提示中生成答案。"],
        ["幻觉可通过RAG、链式思考提示、以及输出验证等方法缓解。"]
    ],
    "ground_truth": [
        "向量数据库是存储向量嵌入并支持相似性检索的专用数据库。",
        "RAG通过检索相关文档增强LLM的输入上下文，从而生成更准确的答案。",
        "减少幻觉可通过RAG引入事实知识、降低temperature、使用事实核查等方式。"
    ]
}

dataset = Dataset.from_dict(rag_outputs)

# 2. 配置评估
llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings()

# 3. 执行评估
result = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        AnswerCorrectness(),
        AnswerSimilarity()
    ],
    llm=llm,
    embeddings=embeddings
)

print("评估结果：")
for metric, score in result.items():
    print(f"  {metric}: {score:.4f}")
```

### 4.2 对比不同RAG配置

```python
import pandas as pd
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall

# 评估不同检索策略
configs = {
    "baseline_top3": baseline_dataset,
    "reranker_top5": reranker_dataset,
    "hybrid_search": hybrid_dataset
}

results = {}
for config_name, dataset in configs.items():
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall()],
        llm=ChatOpenAI(model="gpt-4", temperature=0)
    )
    results[config_name] = result

# 对比结果
comparison_df = pd.DataFrame(results).T
print(comparison_df)
```

### 4.3 与LangChain RAG管道集成

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from datasets import Dataset

# 构建RAG管道
loader = TextLoader("knowledge_base.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """基于以下上下文回答问题：
{context}

问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 运行RAG管道并收集结果
test_questions = [
    "什么是向量数据库？",
    "如何优化检索质量？"
]
test_ground_truths = [
    "向量数据库是存储和检索向量的专用数据库。",
    "可通过调整chunk大小、使用reranker、混合检索等方式优化。"
]

rag_responses = []
rag_contexts = []

for q in test_questions:
    # 获取检索的上下文
    retrieved_docs = retriever.invoke(q)
    contexts = [doc.page_content for doc in retrieved_docs]
    rag_contexts.append(contexts)

    # 获取RAG答案
    answer = rag_chain.invoke(q)
    rag_responses.append(answer)

# 构建评估数据集
eval_dataset = Dataset.from_dict({
    "question": test_questions,
    "answer": rag_responses,
    "contexts": rag_contexts,
    "ground_truth": test_ground_truths
})

# 评估
result = evaluate(
    dataset=eval_dataset,
    metrics=[Faithfulness(), AnswerRelevancy()],
    llm=ChatOpenAI(model="gpt-4", temperature=0)
)
print(result)
```

### 4.4 使用自定义LLM进行评估

```python
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_community.chat_models import ChatOllama

# 使用本地Ollama模型进行评估（降低成本）
local_llm = ChatOllama(model="llama3")

result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy()],
    llm=local_llm,
    embeddings=OpenAIEmbeddings()  # 嵌入模型仍可使用远程API
)
```

### 4.5 自动生成测试数据并评估

```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader

# 步骤1：生成测试数据
loader = DirectoryLoader("./knowledge_docs", glob="**/*.txt")
documents = loader.load()

generator = TestsetGenerator.with_openai(
    generator_llm=ChatOpenAI(model="gpt-4"),
    critic_llm=ChatOpenAI(model="gpt-4"),
    embeddings=OpenAIEmbeddings()
)

testset = generator.generate_with_langchain_docs(
    documents=documents,
    test_size=30,
    distributions={
        simple: 0.5,
        reasoning: 0.3,
        multi_context: 0.2
    }
)

# 步骤2：使用RAG管道回答测试问题
test_df = testset.to_pandas()
questions = test_df["question"].tolist()
ground_truths = test_df["ground_truth"].tolist()

answers = []
contexts = []
for q in questions:
    docs = retriever.invoke(q)
    ctx = [doc.page_content for doc in docs]
    contexts.append(ctx)
    answer = rag_chain.invoke(q)
    answers.append(answer)

# 步骤3：评估
eval_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

result = evaluate(
    dataset=eval_dataset,
    metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall()],
    llm=ChatOpenAI(model="gpt-4", temperature=0)
)
print(result)
```

## 5. 数学原理

### 5.1 Faithfulness 忠实度

忠实度评估的核心思想是验证答案中的每一个声明是否都能被上下文所支持。

**步骤1：声明抽取**

给定答案 $A$，使用LLM将其拆分为 $n$ 个原子化声明 $\{c_1, c_2, ..., c_n\}$。

**步骤2：声明验证**

对于每个声明 $c_i$，判断其是否可由上下文 $C$ 中的信息支持。设支持判断为 $s_i \in \{0, 1\}$，其中1表示支持，0表示不支持。

**步骤3：计算得分**

$$\text{Faithfulness} = \frac{\sum_{i=1}^{n} s_i}{n} = \frac{|\text{supported\_claims}|}{|\text{total\_claims}|}$$

**示例：**
- 答案拆分为3个声明：$c_1$: "量子计算使用量子比特"，$c_2$: "可实现指数级加速"，$c_3$: "已经完全商业化"
- 上下文支持 $c_1$ 和 $c_2$，但不支持 $c_3$
- Faithfulness = 2/3 ≈ 0.67

### 5.2 Answer Relevancy 答案相关性

答案相关性的核心思想是：如果答案与问题高度相关，那么从答案出发应该能推导出原始问题。

**步骤1：问题生成**

给定答案 $A$，使用LLM生成 $N$ 个可能的问题 $\{q_1, q_2, ..., q_N\}$。

**步骤2：语义相似度计算**

对每个生成的问题 $q_i$，计算其与原始问题 $q_{original}$ 的嵌入向量的余弦相似度：

$$\cos(E(q_i), E(q_{original})) = \frac{E(q_i) \cdot E(q_{original})}{\|E(q_i)\| \cdot \|E(q_{original})\|}$$

其中 $E(\cdot)$ 是嵌入函数。

**步骤3：计算得分**

$$\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos(E(q_i), E(q_{original}))$$

该分数的范围为 $[-1, 1]$，通常截断到 $[0, 1]$。值越接近1表示答案与问题越相关。

### 5.3 Context Precision 上下文精确度

上下文精确度衡量检索结果中相关文档的排名质量。好的检索系统应该将相关文档排在前面。

**步骤1：相关性判断**

对检索到的 $K$ 个上下文文档，逐一判断是否与ground_truth相关。设第 $k$ 个文档的相关性为 $rel_k \in \{0, 1\}$。

**步骤2：计算各位置的精确度**

$$\text{precision@k} = \frac{\sum_{i=1}^{k} rel_i}{k}$$

**步骤3：加权平均**

$$\text{Context Precision} = \frac{\sum_{k=1}^{K} \text{precision@k} \times rel_k}{|\text{相关文档数}|}$$

其中 $|\text{相关文档数}| = \sum_{k=1}^{K} rel_k$。

**示例：**
- 4个检索文档的相关性：[1, 0, 1, 0]
- precision@1 = 1/1 = 1.0
- precision@2 = 1/2 = 0.5（但rel_2=0，不计入加权）
- precision@3 = 2/3 ≈ 0.67
- precision@4 = 2/4 = 0.5（但rel_4=0，不计入加权）
- Context Precision = (1.0×1 + 0.67×1) / 2 = 0.835

### 5.4 Context Recall 上下文召回率

上下文召回率衡量检索到的上下文是否覆盖了回答ground_truth所需的所有信息。

$$\text{Context Recall} = \frac{|\text{可从上下文推导的ground\_truth声明}|}{|\text{ground\_truth声明总数}|}$$

将ground_truth拆分为多个声明后，逐一判断是否可从上下文推导。

### 5.5 Answer Similarity 答案语义相似度

使用嵌入模型的向量表示计算答案与参考答案的余弦相似度：

$$\text{Answer Similarity} = \cos(E(A), E(GT)) = \frac{E(A) \cdot E(GT)}{\|E(A)\| \cdot \|E(GT)\|}$$

### 5.6 Answer Correctness 答案正确性

答案正确性是语义相似度和事实正确性的加权组合：

$$\text{Answer Correctness} = w_1 \times \text{Answer Similarity} + w_2 \times \text{Factual Correctness}$$

其中事实正确性的计算方式类似于Faithfulness，但对照的是ground_truth而非检索上下文。默认权重 $w_1 = w_2 = 0.5$。

## 6. 代码原理/架构原理

### 6.1 整体架构

Ragas的架构遵循以下设计原则：

```
┌────────────────────────────────────────────────┐
│                   evaluate()                    │
│  ┌──────────────────────────────────────────┐  │
│  │           Metric Orchestrator             │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ │  │
│  │  │Faithful- │ │Answer    │ │Context   │ │  │
│  │  │ness      │ │Relevancy │ │Precision │ │  │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ │  │
│  │       │             │             │       │  │
│  └───────┼─────────────┼─────────────┼───────┘  │
│          │             │             │           │
│  ┌───────▼─────────────▼─────────────▼───────┐  │
│  │           LLM / Embeddings Layer          │  │
│  │  ┌─────────────┐  ┌─────────────────────┐ │  │
│  │  │  ChatOpenAI  │  │  OpenAIEmbeddings   │ │  │
│  │  └─────────────┘  └─────────────────────┘ │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  ┌───────────────────────────────────────────┐  │
│  │          Dataset Layer (HuggingFace)       │  │
│  └───────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
```

### 6.2 指标实现原理

每个Ragas指标都继承自`Metric`基类，实现以下核心方法：

- `_score(self, row)` — 对单条数据计算得分
- `_ascore(self, row)` — 异步版本的得分计算
- `adapt(self, language)` — 适配不同语言的提示词

指标的计算通常采用多步骤链式提示（Chain-of-Thought Prompting）：

1. **声明抽取步骤**：使用LLM从答案/ground_truth中抽取原子化声明
2. **验证步骤**：使用LLM判断每个声明是否可由上下文支持
3. **聚合步骤**：根据验证结果计算最终数值得分

### 6.3 异步执行机制

Ragas默认使用异步执行以提升评估效率：

```python
# 异步执行的核心逻辑
async def _evaluate_async(dataset, metrics, llm, ...):
    tasks = []
    for row in dataset:
        for metric in metrics:
            task = metric._ascore(row)
            tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return aggregate_results(results)
```

### 6.4 TestsetGenerator架构

```
┌─────────────────────────────────────┐
│         TestsetGenerator            │
│                                     │
│  ┌──────────┐    ┌──────────────┐  │
│  │ Generator │    │   Critic     │  │
│  │   LLM     │    │    LLM      │  │
│  └─────┬─────┘    └──────┬──────┘  │
│        │                  │         │
│  ┌─────▼──────────────────▼──────┐ │
│  │       Evolution Engine        │ │
│  │  ┌────────┐┌────────┐┌─────┐ │ │
│  │  │Simple  ││Reason- ││Multi│ │ │
│  │  │Evo     ││ing Evo ││Ctx  │ │ │
│  │  └────────┘└────────┘└─────┘ │ │
│  └───────────────────────────────┘ │
│        │                           │
│  ┌─────▼──────────────────────┐   │
│  │  Knowledge Graph (from     │   │
│  │  documents)                │   │
│  └────────────────────────────┘   │
└─────────────────────────────────────┘
```

TestsetGenerator首先将文档构建为知识图谱，然后基于不同的演化策略生成问题：
- **Simple Evolution**：直接从文档片段生成问题
- **Reasoning Evolution**：在简单问题基础上增加推理要求
- **Multi-Context Evolution**：跨多个文档片段综合生成问题

生成后，Critic LLM会审查问题质量，过滤低质量的问题。

## 7. 常见注意事项和最佳实践

### 7.1 评估LLM的选择

```python
# 推荐：使用更强的模型进行评估
# GPT-4评估质量高于GPT-3.5，但成本更高
result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness()],
    llm=ChatOpenAI(model="gpt-4", temperature=0)  # temperature=0确保评估稳定性
)

# 成本优化：仅对Faithfulness等需要复杂推理的指标使用强模型
from ragas.metrics import AnswerSimilarity

# AnswerSimilarity只需嵌入模型，无需LLM
similarity = AnswerSimilarity()
```

### 7.2 评估数据质量

- **ground_truth的质量至关重要**：不准确的ground_truth会导致所有依赖它的指标（ContextRecall、AnswerCorrectness）产生误导性结果
- **contexts应该反映真实检索结果**：不要手动构造理想的contexts，否则无法反映真实检索质量
- **样本量建议**：至少50-100条评估数据才能得到统计上有意义的结论

### 7.3 异步和并发设置

```python
# 大规模评估时的并发优化
result = evaluate(
    dataset=large_dataset,
    metrics=[Faithfulness(), AnswerRelevancy()],
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    is_async=True,          # 开启异步
    max_workers=8,          # 根据API限流调整
    batch_size=10           # 批量处理
)

# 如果遇到API限流，降低并发
result = evaluate(
    dataset=dataset,
    metrics=metrics,
    max_workers=4           # 降低并发数
)
```

### 7.4 常见问题排查

**问题1：评估结果异常低**

```python
# 检查数据格式是否正确
print(dataset.column_names)  # 应包含question, answer, contexts, ground_truth
print(dataset[0])           # 检查单条数据格式

# contexts必须是列表的列表
# 错误：contexts = ["文档1内容"]
# 正确：contexts = [["文档1内容"]]
```

**问题2：API调用超时**

```python
# 使用本地模型避免API限制
from langchain_community.chat_models import ChatOllama

local_llm = ChatOllama(model="llama3")
result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness()],
    llm=local_llm
)
```

**问题3：评估成本过高**

```python
# 策略1：分阶段评估，先用少量指标筛选
result_quick = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy()],  # 仅核心指标
    llm=ChatOpenAI(model="gpt-3.5-turbo")        # 使用更便宜的模型
)

# 策略2：对低分样本再用强模型重新评估
low_score_samples = filter_low_scores(result_quick, threshold=0.5)
result_detailed = evaluate(
    dataset=low_score_samples,
    metrics=all_metrics,
    llm=ChatOpenAI(model="gpt-4")
)
```

### 7.5 最佳实践总结

1. **评估LLM与生成LLM分离**：不要用同一个模型既生成答案又评估答案，避免自我偏好偏差
2. **temperature设为0**：评估时应使用确定性的输出（temperature=0）
3. **多轮评估取平均**：由于LLM的非确定性，建议多次评估取平均值
4. **关注指标组合**：单一指标不能全面反映RAG系统质量，应综合多个指标
5. **定期评估**：RAG系统更新后应重新评估，建立性能基线
6. **使用TestsetGenerator**：自动生成测试数据比手动标注更高效，但需人工审核质量
7. **自定义指标**：当内置指标不满足需求时，可继承Metric类创建自定义评估指标
