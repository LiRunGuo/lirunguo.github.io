---
title: "DSPy 声明式Prompt编程"
excerpt: "Signature、Module(Predict/ChainOfThought/ReAct)、Optimizer自动优化"
collection: llm-libs
permalink: /llm-libs/23-dspy
category: agent
---


## 1. 库的简介和在LLM开发中的作用

DSPy 是由 Stanford NLP 团队开发的一个**声明式 Prompt 编程框架**，其核心理念是将 LLM 的 prompt 工程**从手工编写转变为程序化自动优化**。传统开发中，开发者需要反复手动调整 prompt 模板和 few-shot 示例，而 DSPy 允许开发者以声明式的方式定义"做什么"（输入输出签名），框架自动优化"怎么做"（prompt 模板和示例选择）。

在 LLM 开发中，DSPy 解决了以下核心问题：

- **Prompt 脆弱性**：手动编写的 prompt 在模型切换或任务变化时往往失效，DSPy 可以自动重新优化
- **Few-shot 示例选择**：自动从训练数据中选择最优示例，而非依赖直觉
- **模块组合**：将复杂的 LLM 调用链拆分为可复用、可优化的模块
- **系统化评估**：提供标准化的评估框架，量化 prompt 和模块的性能
- **自动优化**：通过编译（compile）过程自动搜索最优的 prompt 和 few-shot 组合

DSPy 的范式转换：**从"手写 prompt"到"编程定义逻辑 + 自动优化 prompt"**。

## 2. 安装方式

```bash
# 基础安装
pip install dspy

# 安装特定版本
pip install dspy==2.6.0

# 安装额外依赖（如向量检索）
pip install dspy[weaviate]
pip install dspy[qdrant]
pip install dspy[milvus]
```

安装后配置 LLM：

```python
import dspy

# 方式一：使用 OpenAI
lm = dspy.LM('openai/gpt-4o', api_key='your-api-key')
dspy.configure(lm=lm)

# 方式二：使用本地模型（通过 Ollama）
lm = dspy.LM('ollama_chat/llama3', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

# 方式三：使用其他提供商
lm = dspy.LM('anthropic/claude-3-sonnet', api_key='your-api-key')
dspy.configure(lm=lm)
```

## 3. 核心类/函数/工具的详细说明

### 3.1 Signature — 输入输出定义

Signature（签名）是 DSPy 中最基础的抽象，以声明式语法定义模块的输入和输出。

```python
# 方式一：字符串语法定义（最常用）
class QASignature(dspy.Signature):
    """回答给定问题。"""  # 文档字符串作为任务描述
    question: str = dspy.InputField(desc="需要回答的问题")
    answer: str = dspy.OutputField(desc="问题的答案")

# 方式二：简写字符串语法（适用于简单场景）
# "输入字段1, 输入字段2 -> 输出字段1, 输出字段2"
signature = "question -> answer"
signature = "question, context -> answer, confidence"

# 方式三：完整类定义（推荐，可加描述）
class SummarizeSignature(dspy.Signature):
    """将长文本总结为简洁的摘要。"""
    text: str = dspy.InputField(desc="需要总结的原文")
    length: int = dspy.InputField(desc="摘要的目标字数")
    summary: str = dspy.OutputField(desc="生成的摘要")
    key_points: list = dspy.OutputField(desc="提取的关键要点")
```

**关键参数**：
- `desc`：字段描述，帮助 LLM 理解每个字段的含义和期望格式
- 类的文档字符串：作为整体任务描述，被嵌入到生成的 prompt 中
- 输入字段用 `dspy.InputField()`，输出字段用 `dspy.OutputField()`

**Signature 的工作原理**：DSPy 根据 Signature 自动生成 prompt 模板，包括任务描述、输入输出字段的格式化、few-shot 示例的插入位置等。开发者无需手写任何 prompt 模板。

### 3.2 Module — dspy.Predict、dspy.ChainOfThought、dspy.ReAct、dspy.ProgramOfThought

Module（模块）是 DSPy 中可执行的计算单元，封装了与 LLM 交互的策略。

#### dspy.Predict

最基本的模块，直接根据 Signature 调用 LLM。

```python
import dspy

# 使用 Signature 类
predictor = dspy.Predict(QASignature)

# 使用字符串 Signature
predictor = dspy.Predict("question -> answer")

# 调用
result = predictor(question="法国的首都是哪里？")
print(result.answer)  # "巴黎"

# 访问完整输出
print(result.toDict())  # {'question': '...', 'answer': '...'}
```

**参数说明**：
- `signature`：Signature 类或字符串
- 返回 `Prediction` 对象，可通过属性名访问输出字段

#### dspy.ChainOfThought

在调用 LLM 前自动添加"逐步思考"的推理链。

```python
cot = dspy.ChainOfThought("question -> answer")

result = cot(question="如果一辆火车每小时行驶60公里，3小时行驶多少公里？")
print(result.answer)      # "180公里"
print(result.rationale)   # 推理过程："我需要计算 60 × 3 = 180"
```

**参数说明**：
- `signature`：同 Predict
- 自动添加 `rationale` 输出字段，包含 LLM 的推理过程
- 适合需要多步推理的任务

#### dspy.ReAct

实现"推理+行动"循环，支持工具调用。

```python
# 定义工具函数
def search_wikipedia(query: str) -> str:
    """搜索维基百科"""
    # 实际实现中调用搜索 API
    return f"关于 {query} 的搜索结果..."

def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"

# 创建 ReAct 模块
react = dspy.ReAct(
    signature="question -> answer",
    tools=[search_wikipedia, calculate],
    max_iters=5  # 最大推理-行动迭代次数
)

result = react(question="爱因斯坦的出生年份的平方是多少？")
print(result.answer)
```

**参数说明**：
- `signature`：定义输入输出
- `tools`：工具函数列表，每个函数的文档字符串会作为工具描述
- `max_iters`：最大迭代次数，防止无限循环

#### dspy.ProgramOfThought

让 LLM 生成代码来解决问题，然后执行代码获取结果。

```python
pot = dspy.ProgramOfThought("question -> answer")

result = pot(question="计算1到100的和")
# LLM 会生成类似 sum(range(1, 101)) 的代码并执行
print(result.answer)  # "5050"
```

**参数说明**：
- `signature`：定义输入输出
- LLM 生成 Python 代码，框架在沙箱中执行并返回结果
- 适合需要精确计算或程序化处理的任务

#### 自定义 Module

```python
class RAGModule(dspy.Module):
    """检索增强生成模块"""

    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question: str):
        # 检索相关文档
        context = self.retrieve(question).passages
        # 生成答案
        prediction = self.generate_answer(context=context, question=question)
        return prediction

# 使用
rag = RAGModule(num_passages=5)
result = rag(question="什么是量子计算？")
print(result.answer)
```

**关键方法**：
- `__init__`：初始化子模块（子模块会被优化器追踪和优化）
- `forward`：定义前向逻辑，所有子模块必须在此方法中调用

### 3.3 Optimizer — dspy.BootstrapFewShot、dspy.MIPROv2、dspy.BootstrapFewShotWithRandomSearch

Optimizer（优化器）是 DSPy 的核心创新，自动搜索最优的 prompt 配置和 few-shot 示例。

#### dspy.BootstrapFewShot

通过引导采样自动选择 few-shot 示例。

```python
from dspy.teleprompt import BootstrapFewShot

# 定义评估指标
def answer_exact_match(example, prediction, trace=None):
    """检查预测答案是否与真实答案完全匹配"""
    return example.answer.lower().strip() == prediction.answer.lower().strip()

# 创建优化器
optimizer = BootstrapFewShot(
    metric=answer_exact_match,    # 评估指标函数
    max_bootstrapped_demos=4,     # 最大引导示例数
    max_labeled_demos=4,          # 最大标注示例数
    max_rounds=1,                 # 优化轮数
    max_errors=0                  # 允许的最大错误数
)

# 准备训练数据
trainset = [
    dspy.Example(question="1+1=?", answer="2").with_inputs("question"),
    dspy.Example(question="2+2=?", answer="4").with_inputs("question"),
    dspy.Example(question="3*3=?", answer="9").with_inputs("question"),
    # ... 更多示例
]

# 编译模块
compiled_module = optimizer.compile(
    student=RAGModule(),          # 待优化的模块
    trainset=trainset,            # 训练集
    # teacher=RAGModule(),        # 可选：教师模块（默认使用 student 自身）
)
```

**参数说明**：
- `metric`：评估函数，签名为 `(example, prediction, trace) -> bool/float`
- `max_bootstrapped_demos`：通过引导生成的最大示例数
- `max_labeled_demos`：直接从训练集使用的最大标注示例数
- `max_rounds`：优化迭代轮数

#### dspy.MIPROv2

高级优化器，同时优化 prompt 指令和 few-shot 示例。

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=answer_exact_match,
    num_threads=4,                # 并行线程数
    num_candidates=10,            # 候选 prompt 数量
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_trials=20,                # 优化试验次数
    minibatch_size=25,            # 小批量评估大小
    minibatch_full_eval_steps=5,  # 全量评估间隔
    autobase="auto",              # 自动选择基础配置
)

# 编译
compiled_module = optimizer.compile(
    student=RAGModule(),
    trainset=trainset,
    eval_kwargs={"display": True, "display_progress": True},
)
```

**参数说明**：
- `num_candidates`：每轮生成的候选 prompt 指令变体数
- `num_trials`：总优化试验次数（越多越好，但更耗时）
- `minibatch_size`：每次评估使用的小批量数据量
- MIPROv2 会同时搜索：prompt 指令的措辞 + few-shot 示例的选择

#### dspy.BootstrapFewShotWithRandomSearch

在 BootstrapFewShot 基础上增加随机搜索策略。

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

optimizer = BootstrapFewShotWithRandomSearch(
    metric=answer_exact_match,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=10,    # 候选程序数
    num_threads=4,                # 并行线程数
)

compiled_module = optimizer.compile(
    student=RAGModule(),
    trainset=trainset,
)
```

**参数说明**：
- `num_candidate_programs`：随机搜索的候选程序数量
- 每个候选程序使用不同的 few-shot 示例组合
- 最终选择评估分数最高的程序

### 3.4 数据集 — dspy.Dataset、Example

```python
# 创建 Example
example = dspy.Example(
    question="法国的首都是哪里？",
    answer="巴黎",
    context="法国是欧洲国家..."
).with_inputs("question")  # 指定哪些字段是输入

# 访问字段
print(example.question)  # "法国的首都是哪里？"
print(example.answer)    # "巴黎"

# with_inputs() 指定输入字段
# 未在 with_inputs 中的字段被视为输出/标签
example = dspy.Example(
    question="1+1=?",
    answer="2"
).with_inputs("question")

# 创建数据集
class MyDataset(dspy.Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train = [
            dspy.Example(question="Q1", answer="A1").with_inputs("question"),
            dspy.Example(question="Q2", answer="A2").with_inputs("question"),
        ]
        self._dev = [
            dspy.Example(question="Q3", answer="A3").with_inputs("question"),
        ]
        self._test = [
            dspy.Example(question="Q4", answer="A4").with_inputs("question"),
        ]

dataset = MyDataset()
trainset = dataset.train  # 训练集
devset = dataset.dev      # 验证集
testset = dataset.test    # 测试集

# 内置数据集
from dspy.datasets import HotPotQA
hotpot = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)
trainset = hotpot.train
devset = hotpot.dev
```

### 3.5 评估器 — dspy.Evaluate、metric函数

```python
import dspy

# 定义评估指标
def answer_match(example, prediction, trace=None):
    """答案匹配指标"""
    return example.answer.lower().strip() == prediction.answer.lower().strip()

def answer_f1(example, prediction, trace=None):
    """基于 F1 分数的指标"""
    pred_tokens = set(prediction.answer.lower().split())
    gold_tokens = set(example.answer.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    precision = len(pred_tokens & gold_tokens) / len(pred_tokens)
    recall = len(pred_tokens & gold_tokens) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# 使用 Evaluate 类
evaluator = dspy.Evaluate(
    devset=devset,               # 评估数据集
    metric=answer_match,          # 评估指标
    num_threads=4,                # 并行线程数
    display_progress=True,        # 显示进度条
    display_table=5,              # 显示前5个样本的详细结果
)

# 评估模块
score = evaluator(module)
print(f"准确率: {score}")

# 评估编译前后的对比
uncompiled_score = evaluator(rag_module)
compiled_score = evaluator(compiled_module)
print(f"优化前: {uncompiled_score}, 优化后: {compiled_score}")
```

**metric 函数签名**：
```python
def metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool | float:
    """
    参数:
        example: 标准答案示例
        prediction: 模块预测结果
        trace: 可选的执行追踪（用于高级评估）
    返回:
        bool 或 float 评分
    """
    pass
```

### 3.6 LM配置 — dspy.configure、dspy.LM

```python
import dspy

# 全局配置 LLM
lm = dspy.LM(
    model='openai/gpt-4o',       # 模型标识: 提供商/模型名
    api_key='sk-...',             # API 密钥
    api_base='https://api.openai.com/v1',  # 可选：自定义 API 端点
    temperature=0.0,              # 生成温度
    max_tokens=2048,              # 最大生成 token 数
    num_retries=3,                # API 调用重试次数
    cache=True,                   # 是否缓存响应
)
dspy.configure(lm=lm)

# 查看当前配置
print(dspy.settings.lm)  # 当前 LM 对象

# 配置检索器（用于 RAG）
from dspy.retrieve import ColBERTv2
rm = ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.configure(lm=lm, rm=rm)

# 多 LM 配置（不同模块使用不同模型）
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cheap_lm = dspy.LM('openai/gpt-4o-mini', temperature=0.0)
        self.expert_lm = dspy.LM('openai/gpt-4o', temperature=0.3)

    def forward(self, question):
        # 使用 dspy.context 临时切换 LM
        with dspy.context(lm=self.cheap_lm):
            initial = dspy.Predict("question -> draft")(question=question)

        with dspy.context(lm=self.expert_lm):
            final = dspy.Predict("draft, question -> answer")(
                draft=initial.draft, question=question
            )
        return final
```

### 3.7 断言 — dspy.Assert、dspy.Suggest

断言机制允许在模块执行过程中添加约束，确保 LLM 输出满足特定条件。

```python
import dspy

class ValidatedQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        prediction = self.generate(question=question)

        # dspy.Assert: 硬约束，条件不满足时抛出断言错误并重试
        dspy.Assert(
            len(prediction.answer) > 0,
            "答案不能为空",
        )

        # dspy.Suggest: 软约束，条件不满足时建议修改并重试
        dspy.Suggest(
            len(prediction.answer) < 500,
            "答案应该简洁，不超过500字",
        )

        return prediction

# 使用断言模块时需要包装
from dspy.primitives.assertions import assert_transform_module
validated_qa = assert_transform_module(ValidatedQA(), backtrack_handler)

# 或者直接在 forward 中使用
qa = ValidatedQA()
result = qa(question="什么是Python？")
```

**Assert vs Suggest 的区别**：
- `dspy.Assert`：硬约束，违反时立即触发重试（最多 `max_backtracks` 次后抛出异常）
- `dspy.Suggest`：软约束，违反时尝试重试但不会抛出异常
- 两者都会将失败信息反馈给 LLM，引导其修正输出

```python
# 配置最大回溯次数
from dspy.primitives.assertions import backtrack_handler

handler = backtrack_handler(max_backtracks=3)  # 最多重试3次
validated = assert_transform_module(ValidatedQA(), handler)
```

## 4. 在LLM开发中的典型使用场景和代码示例

### 场景一：构建并优化 RAG 系统

```python
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.retrieve import ColBERTv2

# 配置
lm = dspy.LM('openai/gpt-4o', api_key='your-key')
rm = ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.configure(lm=lm, rm=rm)

# 定义 Signature
class GenerateAnswer(dspy.Signature):
    """根据检索到的上下文回答问题。"""
    context: str = dspy.InputField(desc="检索到的相关文档")
    question: str = dspy.InputField(desc="用户的问题")
    answer: str = dspy.OutputField(desc="基于上下文的答案")

# 定义模块
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate(context=context, question=question)
        return prediction

# 评估指标
def gold_answer_match(example, prediction, trace=None):
    return example.answer.lower().strip() == prediction.answer.lower().strip()

# 准备数据
from dspy.datasets import HotPotQA
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

# 优化前评估
uncompiled_rag = RAG()
evaluator = dspy.Evaluate(devset=devset, metric=gold_answer_match, num_threads=4)
before_score = evaluator(uncompiled_rag)
print(f"优化前准确率: {before_score:.2f}")

# 编译优化
optimizer = BootstrapFewShot(
    metric=gold_answer_match,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)
compiled_rag = optimizer.compile(uncompiled_rag, trainset=trainset)

# 优化后评估
after_score = evaluator(compiled_rag)
print(f"优化后准确率: {after_score:.2f}")

# 使用优化后的模块
result = compiled_rag(question="阿尔伯特·爱因斯坦在哪出生？")
print(result.answer)
```

### 场景二：多步推理与程序化思维

```python
import dspy

# 配置
lm = dspy.LM('openai/gpt-4o', api_key='your-key')
dspy.configure(lm=lm)

# 数学推理：使用 ProgramOfThought
math_solver = dspy.ProgramOfThought("problem -> solution")
result = math_solver(problem="计算斐波那契数列第10项")
print(result.solution)

# 逻辑推理：使用 ChainOfThought
logic_solver = dspy.ChainOfThought("premise, hypothesis -> entailment, explanation")
result = logic_solver(
    premise="所有猫都是动物",
    hypothesis="小花是猫，所以小花是动物"
)
print(f"蕴含关系: {result.entailment}")
print(f"解释: {result.explanation}")

# 多模块组合
class MathReasoner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought("problem -> analysis, approach")
        self.solve = dspy.ProgramOfThought("analysis, approach -> solution")
        self.verify = dspy.ChainOfThought("problem, solution -> is_correct, correction")

    def forward(self, problem):
        # 步骤1：分析问题
        analysis = self.analyze(problem=problem)

        # 步骤2：求解
        solution = self.solve(
            analysis=analysis.analysis,
            approach=analysis.approach
        )

        # 步骤3：验证
        verification = self.verify(
            problem=problem,
            solution=solution.solution
        )

        # 如果不正确，返回修正
        if verification.is_correct.lower() == "no":
            return dspy.Prediction(
                solution=verification.correction,
                verified=False
            )

        return dspy.Prediction(
            solution=solution.solution,
            verified=True
        )

reasoner = MathReasoner()
result = reasoner(problem="一个圆的半径是5，求面积")
print(f"答案: {result.solution}, 已验证: {result.verified}")
```

### 场景三：使用断言确保输出质量

```python
import dspy
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

lm = dspy.LM('openai/gpt-4o', api_key='your-key')
dspy.configure(lm=lm)

class QualityControlledSummary(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought("document -> summary")
        self.extract = dspy.ChainOfThought("summary -> key_points")

    def forward(self, document):
        # 生成摘要
        pred = self.summarize(document=document)

        # 断言：摘要长度在合理范围内
        dspy.Assert(
            50 <= len(pred.summary) <= 500,
            f"摘要长度 {len(pred.summary)} 不在50-500范围内，请调整"
        )

        # 建议：摘要应包含原文关键词
        dspy.Suggest(
            any(kw in pred.summary for kw in document.split()[:5]),
            "摘要应包含原文的关键词"
        )

        # 提取要点
        points = self.extract(summary=pred.summary)

        # 断言：至少提取一个要点
        dspy.Assert(
            len(points.key_points) > 0,
            "必须提取至少一个关键要点"
        )

        return dspy.Prediction(
            summary=pred.summary,
            key_points=points.key_points
        )

# 包装断言模块
handler = backtrack_handler(max_backtracks=3)
module = assert_transform_module(QualityControlledSummary(), handler)

result = module(document="人工智能是计算机科学的一个分支...")
print(result.summary)
print(result.key_points)
```

## 5. 数学原理

### Prompt 优化的搜索空间

DSPy 的优化器在以下搜索空间中寻找最优配置：

1. **Few-shot 示例选择**：从训练集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$ 中选择子集 $S \subset \mathcal{D}$，使得目标指标最大化：

$$S^* = \arg\max_{S \subset \mathcal{D}, |S| \leq k} \mathbb{E}_{(x,y) \sim \mathcal{D}_{eval}}[\text{metric}(y, f_S(x))]$$

其中 $f_S$ 表示使用示例集 $S$ 的模块，$k$ 是最大示例数。

2. **Prompt 指令优化（MIPROv2）**：搜索最优的任务指令 $I$：

$$I^* = \arg\max_{I \in \mathcal{I}} \mathbb{E}[\text{metric}(y, f_{I,S}(x))]$$

其中 $\mathcal{I}$ 是指令的搜索空间，通过 LLM 生成候选指令变体。

### Bootstrap 采样

BootstrapFewShot 的核心是**引导采样**：

1. 对于训练集中的每个样本 $(x_i, y_i)$，尝试使用当前模块 $f$ 生成预测 $\hat{y}_i = f(x_i)$
2. 如果 $\text{metric}(y_i, \hat{y}_i) = \text{True}$，则将 $(x_i, \hat{y}_i)$ 作为有效的 few-shot 示例
3. 重复直到收集足够的有效示例

这保证选出的示例都是模块能正确处理的，避免在 prompt 中包含模型会失败的案例。

### 贝叶斯优化（MIPROv2）

MIPROv2 使用简化版的贝叶斯优化来搜索指令和示例的最优组合：

$$\theta_{t+1} = \arg\max_{\theta \in \Theta} \alpha(\theta | \mathcal{D}_{1:t})$$

其中 $\alpha$ 是采集函数（如 Expected Improvement），$\mathcal{D}_{1:t}$ 是前 $t$ 轮的评估结果。

## 6. 代码原理/架构原理

### 架构概览

```
┌──────────────────────────────────────────────────────┐
│                   应用层 (Application)                 │
│   编译后的模块 → invoke() 执行推理                      │
├──────────────────────────────────────────────────────┤
│                  优化层 (Optimizer)                    │
│   BootstrapFewShot / MIPROv2 / RandomSearch          │
│   搜索最优 prompt 模板 + few-shot 示例                 │
├──────────────────────────────────────────────────────┤
│                  模块层 (Module)                       │
│   Predict / ChainOfThought / ReAct / 自定义Module      │
│   定义计算逻辑和模块组合                                 │
├──────────────────────────────────────────────────────┤
│                 签名层 (Signature)                     │
│   声明式定义输入/输出字段和任务描述                       │
├──────────────────────────────────────────────────────┤
│                 基础设施层 (Infrastructure)             │
│   dspy.LM / dspy.Retrieve / dspy.configure           │
│   LLM 调用 / 检索 / 全局配置                           │
└──────────────────────────────────────────────────────┘
```

### 核心流程：从 Signature 到 Prompt

1. **Signature 解析**：将 Signature 类/字符串解析为结构化的输入输出字段定义
2. **Prompt 生成**：根据 Signature 自动生成 prompt 模板：
   ```
   {任务描述}
   
   ---
   
   Follow the following format.
   
   {输入字段名}: {字段描述}
   ...
   {输出字段名}: {字段描述}
   ...
   
   ---
   
   {few-shot 示例}
   
   ---
   
   {当前输入}
   ```
3. **LLM 调用**：将生成的 prompt 发送给 LLM，获取响应
4. **输出解析**：从 LLM 响应中提取各输出字段的值，构造 Prediction 对象

### 编译（Compile）流程

```
输入: Module + 训练集 + 评估指标

1. 初始化: 将 Module 中的子模块标记为可优化
2. 引导采样: 对训练集样本执行 Module，收集成功的示例
3. 搜索/优化:
   - BootstrapFewShot: 贪心选择最优示例组合
   - MIPROv2: 生成指令变体，贝叶斯搜索最优组合
   - RandomSearch: 随机采样多个配置，选最优
4. 输出: 编译后的 Module（包含最优 prompt 和 few-shot 示例）
```

### 模块组合与参数共享

DSPy 的 Module 仿 PyTorch 设计：
- `__init__` 中声明的子模块会被框架追踪（类似 `nn.Module` 的子模块注册）
- `forward` 定义执行逻辑
- 编译时，优化器会递归地优化所有子模块
- 每个子模块的 prompt 和示例独立优化

## 7. 常见注意事项和最佳实践

### 注意事项

1. **训练集质量**：优化效果严重依赖训练集质量，确保训练样本准确且具代表性
2. **评估指标设计**：指标函数应尽可能反映真实任务需求，过于宽松或严格都会影响优化效果
3. **API 成本**：MIPROv2 等高级优化器会产生大量 LLM 调用，注意控制成本
4. **缓存利用**：开启 `cache=True`（默认）避免重复调用，但修改 prompt 后需清除缓存
5. **Signature 稳定性**：一旦定义好 Signature 后不要频繁修改，否则之前优化的结果可能失效

### 最佳实践

```python
# 1. 使用 TypedDict 风格的 Signature 并添加描述
class GoodSignature(dspy.Signature):
    """清晰的任务描述。"""
    question: str = dspy.InputField(desc="需要回答的问题，用中文表述")
    answer: str = dspy.OutputField(desc="简洁准确的答案，不超过100字")

# 避免过于简单的 Signature
# BAD: "q -> a"  (字段名和描述都不清晰)

# 2. 从小规模优化开始，逐步增加复杂度
# 第一步：使用 BootstrapFewShot 快速验证
optimizer_small = BootstrapFewShot(metric=my_metric, max_bootstrapped_demos=2)
compiled_v1 = optimizer_small.compile(module, trainset=trainset[:10])

# 第二步：使用更多数据和更高级优化器
optimizer_full = MIPROv2(metric=my_metric, num_trials=20)
compiled_v2 = optimizer_full.compile(module, trainset=trainset)

# 3. 保存和加载编译后的模块
compiled_module.save("optimized_module.json")
loaded_module = RAGModule()
loaded_module.load("optimized_module.json")

# 4. 使用 dspy.inspect 检查生成的 prompt
import dspy
# 查看最近的 LLM 调用
for entry in dspy.settings.lm.history:
    print(f"Prompt: {entry['prompt']}")
    print(f"Response: {entry['response']}")
    print("---")

# 5. 断言中使用具体的错误信息
dspy.Assert(
    len(prediction.answer) > 0,
    "答案不能为空，请重新生成一个非空的答案"  # 具体的修正建议
)

# 6. 合理划分训练/验证/测试集
# 训练集：用于优化器搜索 few-shot 示例
# 验证集：用于优化器选择最优配置
# 测试集：最终评估，不参与优化过程

# 7. 使用 with_inputs 明确输入字段
example = dspy.Example(
    question="Q", context="C", answer="A"
).with_inputs("question", "context")  # 明确指定输入字段
# answer 不在 with_inputs 中，被视为标签/输出

# 8. 评估时使用多线程加速
evaluator = dspy.Evaluate(
    devset=testset,
    metric=my_metric,
    num_threads=8,           # 根据API限制调整
    display_progress=True,
    display_table=0,         # 生产环境设为0，不显示详情
)
```

### 调试技巧

```python
# 查看编译后的模块配置
print(compiled_module)  # 显示模块结构和优化后的配置

# 追踪单次执行
with dspy.context(trace=True):
    result = compiled_module(question="测试问题")
    for step in dspy.settings.trace:
        print(f"模块: {step[0].__class__.__name__}")
        print(f"输入: {step[1]}")
        print(f"输出: {step[2]}")

# 对比优化前后的 prompt
# 优化前
result_before = uncompiled_module(question="测试")
print("优化前最后一个 prompt:", dspy.settings.lm.history[-1])

# 优化后
result_after = compiled_module(question="测试")
print("优化后最后一个 prompt:", dspy.settings.lm.history[-1])
```
