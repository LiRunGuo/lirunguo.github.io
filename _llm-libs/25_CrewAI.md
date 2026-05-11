---
title: "CrewAI 角色驱动协作"
excerpt: "Agent/Task/Crew、Sequential/Hierarchical流程、Memory记忆系统"
collection: llm-libs
permalink: /llm-libs/25-crewai
category: agent
toc: true
---


## 1. 库的简介和在LLM开发中的作用

CrewAI 是一个基于 Python 的多智能体（Multi-Agent）协作编排框架，旨在将复杂的任务拆解为多个角色，各自负责一部分，通过流程协作完成。它的核心理念是**角色驱动的多 Agent 协作**——每个 Agent 拥有明确的角色（Role）、目标（Goal）和背景故事（Backstory），模拟真实团队中的分工协作。

在 LLM 开发中，CrewAI 的作用包括：

- **多 Agent 协同**：将单一 LLM 调用升级为多个专业化 Agent 的协作流程，每个 Agent 聚焦特定领域
- **流程编排**：支持顺序执行（Sequential）和层级管理（Hierarchical）两种核心流程模式
- **工具集成**：Agent 可调用搜索、代码执行、文件操作等外部工具，扩展 LLM 的能力边界
- **记忆系统**：内置短期、长期和实体记忆，使 Agent 能在多轮交互中保持上下文和积累经验
- **结构化输出**：通过 Pydantic 模型约束输出格式，确保结果的可预测性

CrewAI 完全独立于 LangChain 等其他框架，从零构建，轻量且快速。

## 2. 安装方式

```bash
# 基础安装
pip install crewai

# 安装包含工具的完整版
pip install 'crewai[tools]'

# 安装包含所有可选依赖
pip install 'crewai[agentops,tools]'
```

安装后可通过 CLI 验证：

```bash
crewai version
```

创建新项目（推荐方式）：

```bash
crewai create crew my_project
cd my_project
```

## 3. 核心类/函数/工具的详细说明

### 3.1 Agent 类

Agent 是 CrewAI 的核心执行单元，代表一个具有特定角色和能力的自主智能体。

**必需参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `role` | `str` | 定义 Agent 的职能和专业领域，如"高级数据研究员" |
| `goal` | `str` | 指导 Agent 决策的目标，如"发现最新的AI技术发展" |
| `backstory` | `str` | 提供上下文和个性，丰富交互体验 |

**可选参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm` | `Union[str, LLM, Any]` | `"gpt-4"` | Agent 使用的语言模型 |
| `tools` | `List[BaseTool]` | `[]` | Agent 可用的工具列表 |
| `function_calling_llm` | `Optional[Any]` | `None` | 用于工具调用的 LLM，覆盖 Crew 级别设置 |
| `memory` | `bool` | `True` | 是否维护交互记忆 |
| `verbose` | `bool` | `False` | 是否输出详细执行日志 |
| `allow_delegation` | `bool` | `False` | 是否允许委托任务给其他 Agent |
| `max_iter` | `int` | `20` | 最大迭代次数，超过后必须给出最佳答案 |
| `max_rpm` | `Optional[int]` | `None` | 每分钟最大请求数，防止速率限制 |
| `max_execution_time` | `Optional[int]` | `None` | 最大执行时间（秒） |
| `max_retry_limit` | `int` | `2` | 出错时最大重试次数 |
| `cache` | `bool` | `True` | 是否缓存工具执行结果 |
| `allow_code_execution` | `Optional[bool]` | `False` | 是否允许代码执行 |
| `code_execution_mode` | `Literal["safe", "unsafe"]` | `"safe"` | 代码执行模式（safe 使用 Docker） |
| `respect_context_window` | `bool` | `True` | 是否通过摘要保持上下文窗口 |
| `knowledge_sources` | `Optional[List[BaseKnowledgeSource]]` | `None` | 知识来源列表 |
| `step_callback` | `Optional[Any]` | `None` | 每步执行后的回调函数 |
| `use_system_prompt` | `Optional[bool]` | `True` | 是否使用系统提示（o1 模型需设为 False） |

**代码示例：**

```python
from crewai import Agent
from crewai_tools import SerperDevTool

# 基础研究 Agent
research_agent = Agent(
    role="AI技术研究员",
    goal="发现并总结AI领域的最新发展动态",
    backstory="你是一位经验丰富的技术研究员，擅长发现前沿技术趋势",
    tools=[SerperDevTool()],
    verbose=True
)

# 代码开发 Agent
dev_agent = Agent(
    role="高级Python开发者",
    goal="编写和调试Python代码",
    backstory="拥有10年经验的Python开发专家",
    allow_code_execution=True,
    code_execution_mode="safe",  # 使用Docker安全执行
    max_execution_time=300,      # 5分钟超时
    max_retry_limit=3            # 复杂代码任务多给重试机会
)

# 数据分析 Agent
analysis_agent = Agent(
    role="数据分析师",
    goal="对大型数据集进行深度分析",
    backstory="专精于大数据分析和模式识别",
    memory=True,
    respect_context_window=True,
    max_rpm=10,                        # 限制API调用频率
    function_calling_llm="gpt-4o-mini" # 工具调用用更便宜的模型
)
```

### 3.2 Task 类

Task 代表一个由 Agent 执行的具体任务，包含描述、预期输出、工具、回调等执行细节。

**核心参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `description` | `str` | 任务的清晰描述 |
| `expected_output` | `str` | 预期输出的详细描述 |
| `agent` | `Optional[BaseAgent]` | 负责执行此任务的 Agent |
| `tools` | `List[BaseTool]` | 任务级工具（覆盖 Agent 默认工具） |
| `context` | `Optional[List[Task]]` | 依赖的其他任务（其输出作为上下文） |
| `async_execution` | `Optional[bool]` | 是否异步执行，默认 `False` |
| `human_input` | `Optional[bool]` | 是否需要人工审核，默认 `False` |
| `output_pydantic` | `Optional[Type[BaseModel]]` | Pydantic 模型约束输出格式 |
| `output_json` | `Optional[Type[BaseModel]]` | 以 JSON 格式输出 |
| `output_file` | `Optional[str]` | 输出保存文件路径 |
| `callback` | `Optional[Any]` | 任务完成后的回调函数 |
| `guardrail` | — | 验证函数，在输出传递给下一任务前验证 |
| `max_retries` | `int` | guardrail 失败时最大重试次数 |

**代码示例：**

```python
from crewai import Task
from pydantic import BaseModel

class Blog(BaseModel):
    title: str
    content: str

# 基础任务
research_task = Task(
    description="研究AI Agent领域的最新发展，确保找到有趣且相关的信息",
    expected_output="包含10个要点的AI Agent最新发展列表",
    agent=research_agent
)

# 带结构化输出的任务
blog_task = Task(
    description="根据研究结果撰写一篇关于AI的博客文章，200字以内",
    expected_output="一篇引人入胜的博客，包含标题和正文",
    agent=blog_agent,
    output_pydantic=Blog,  # 强制输出符合Pydantic模型
)

# 异步任务 + 上下文依赖
research_ai_task = Task(
    description="研究AI最新发展",
    expected_output="AI最新发展列表",
    async_execution=True,  # 异步执行，不阻塞后续任务
    agent=research_agent,
)

research_ops_task = Task(
    description="研究AI Ops最新发展",
    expected_output="AI Ops最新发展列表",
    async_execution=True,  # 异步执行
    agent=research_agent,
)

write_blog_task = Task(
    description="撰写关于AI及其最新动态的博客文章",
    expected_output="一篇4段式的AI文章",
    agent=writer_agent,
    context=[research_ai_task, research_ops_task]  # 等待两个异步任务完成
)

# 带回调的任务
def callback_function(output):
    print(f"任务完成！输出: {output.raw}")

task_with_callback = Task(
    description='总结最新的AI新闻',
    expected_output='5条最重要的AI新闻摘要',
    agent=research_agent,
    callback=callback_function
)
```

**Task Guardrail（任务护栏）：**

Guardrail 用于在输出传递给下一任务前进行验证和转换：

```python
from typing import Tuple, Union, Any
from crewai import TaskOutput

def validate_blog_content(result: TaskOutput) -> Tuple[bool, Any]:
    """验证博客内容是否符合要求"""
    word_count = len(result.raw.split())
    if word_count > 200:
        return (False, "博客内容超过200字限制")
    return (True, result.raw.strip())

blog_task = Task(
    description="撰写关于AI的博客文章",
    expected_output="200字以内的博客文章",
    agent=blog_agent,
    guardrail=validate_blog_content,  # 验证不通过会自动重试
    max_retries=3
)
```

### 3.3 Crew 类

Crew 是一组 Agent 的协作团队，定义了任务执行策略和协作方式。

**核心参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `agents` | `List[BaseAgent]` | 团队中的 Agent 列表 |
| `tasks` | `List[Task]` | 团队需要执行的任务列表 |
| `process` | `Process` | 执行流程：`Process.sequential` 或 `Process.hierarchical` |
| `verbose` | `bool` | 是否输出详细日志，默认 `False` |
| `manager_llm` | `Optional[Any]` | 层级模式下管理 Agent 使用的 LLM（必需） |
| `manager_agent` | `Optional[Agent]` | 自定义管理 Agent（替代自动生成的） |
| `memory` | `bool` | 是否启用记忆系统 |
| `memory_config` | `Optional[Dict]` | 记忆系统配置 |
| `cache` | `bool` | 是否缓存工具执行结果，默认 `True` |
| `max_rpm` | `Optional[int]` | 覆盖所有 Agent 的 RPM 限制 |
| `step_callback` | `Optional[Any]` | 每个 Agent 步骤后的回调 |
| `task_callback` | `Optional[Any]` | 每个任务完成后的回调 |
| `planning` | `bool` | 是否启用规划能力 |
| `output_log_file` | `Optional[Union[str, bool]]` | 日志输出文件路径 |

**代码示例：**

```python
from crewai import Agent, Crew, Task, Process

# 顺序执行模式
crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[research_task, write_task],
    process=Process.sequential,  # 任务按顺序执行
    verbose=True
)

# 层级管理模式（需要Manager Agent）
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[task1, task2, task3],
    process=Process.hierarchical,  # Manager Agent 协调分配
    manager_llm="gpt-4o",          # 层级模式下必需
    verbose=True
)

# 自定义管理 Agent
manager = Agent(
    role="项目经理",
    goal="高效协调团队成员完成任务",
    backstory="经验丰富的项目管理者",
    allow_delegation=True
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.hierarchical,
    manager_agent=manager  # 使用自定义管理Agent
)

# 执行
result = crew.kickoff(inputs={"topic": "AI Agents"})
print(result.raw)

# 异步执行
result = await crew.kickoff_async(inputs={"topic": "AI Agents"})

# 批量执行
results = crew.kickoff_for_each(
    inputs=[{"topic": "AI医疗"}, {"topic": "AI金融"}]
)
```

### 3.4 Tool 工具系统

CrewAI 提供两种创建自定义工具的方式：

#### 方式一：`@tool` 装饰器（简洁方式）

```python
from crewai.tools import tool

@tool("搜索工具")
def search_tool(query: str) -> str:
    """根据查询字符串搜索互联网信息。"""
    # 实际搜索逻辑
    return f"搜索结果: {query}的相关信息"

@tool("计算器")
def calculator(expression: str) -> str:
    """计算数学表达式的结果。"""
    try:
        result = eval(expression)  # 注意：生产环境应使用安全的计算方式
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"
```

#### 方式二：继承 `BaseTool`（结构化方式）

```python
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    """工具输入模型"""
    argument: str = Field(..., description="输入参数描述")

class MyCustomTool(BaseTool):
    name: str = "自定义工具"
    description: str = "此工具的功能描述，对Agent理解何时使用至关重要"
    args_schema: Type[BaseModel] = MyToolInput

    def _run(self, argument: str) -> str:
        """工具核心逻辑"""
        return f"处理结果: {argument}"
```

#### 工具缓存

```python
@tool("可缓存的搜索工具")
def cached_search(query: str) -> str:
    """搜索并缓存结果。"""
    return f"搜索结果: {query}"

def cache_strategy(arguments: dict, result: str) -> bool:
    """自定义缓存策略：仅缓存成功结果"""
    return "错误" not in result

cached_search.cache_function = cache_strategy
```

### 3.5 Process 流程模式

#### Sequential（顺序执行）

任务按照定义顺序依次执行，前一个任务的输出作为后一个任务的上下文：

```python
from crewai import Crew, Process

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    process=Process.sequential
)
```

#### Hierarchical（层级管理）

由 Manager Agent 统一协调，负责分配任务、验证结果和决策。需要指定 `manager_llm` 或 `manager_agent`：

```python
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[task1, task2, task3],
    process=Process.hierarchical,
    manager_llm="gpt-4o"
)
```

Manager Agent 的职责：
- 分析任务需求，将任务分配给最合适的 Agent
- 验证每个任务的输出质量
- 在 Agent 之间协调信息流
- 做出最终决策

### 3.6 Memory 记忆系统

CrewAI 的记忆系统默认关闭，通过 `memory=True` 启用。

#### 记忆类型

| 类型 | 存储技术 | 说明 |
|------|----------|------|
| 短期记忆（Short-Term Memory） | RAG (Chroma + Embeddings) | 临时存储近期交互和结果，使 Agent 能在执行中回溯相关上下文 |
| 长期记忆（Long-Term Memory） | SQLite3 | 保留历史执行中的宝贵经验，使 Agent 能随时间积累知识 |
| 实体记忆（Entity Memory） | RAG (Chroma + Embeddings) | 捕获和组织任务中遇到的实体（人、地点、概念），构建关系映射 |
| 上下文记忆（Context Memory） | 组合以上三种 | 维护交互上下文，确保跨任务的连贯性和相关性 |
| 外部记忆（External Memory） | 外部提供者（如 Mem0） | 集成外部记忆系统，支持自定义存储实现 |

#### 代码示例

```python
from crewai import Crew, Agent, Task, Process
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSqliteStorage

# 基础记忆配置
crew = Crew(
    agents=[...],
    tasks=[...],
    process=Process.sequential,
    memory=True,  # 启用记忆
    verbose=True
)

# 自定义记忆配置
crew = Crew(
    agents=[...],
    tasks=[...],
    process=Process.sequential,
    memory=True,
    long_term_memory=LongTermMemory(
        storage=LTMSqliteStorage(
            db_path="/my_crew/long_term_memory_storage.db"
        )
    ),
    short_term_memory=ShortTermMemory(
        storage=RAGStorage(
            embedder_config={
                "provider": "openai",
                "config": {"model": "text-embedding-3-small"}
            },
            type="short_term",
            path="/my_crew/"
        )
    ),
    entity_memory=EntityMemory(
        storage=RAGStorage(
            embedder_config={
                "provider": "openai",
                "config": {"model": "text-embedding-3-small"}
            },
            type="short_term",
            path="/my_crew/"
        )
    ),
)

# 集成外部记忆（Mem0）
import os
os.environ["MEM0_API_KEY"] = "m0-xx"

crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,
    memory_config={
        "provider": "mem0",
        "config": {"user_id": "john"},
        "user_memory": {}
    },
)

# 重置记忆
crew.reset_memories(command_type='all')  # 重置所有
# 可选: 'long', 'short', 'entities', 'kickoff_outputs', 'knowledge'
```

### 3.7 流程控制

#### 回调机制

```python
# Agent 步骤回调
def step_callback(output):
    print(f"Agent步骤完成: {output}")

agent = Agent(
    role="研究员",
    goal="研究AI技术",
    backstory="资深研究员",
    step_callback=step_callback
)

# 任务完成回调
def task_callback(output):
    print(f"任务完成: {output.raw}")

task = Task(
    description="研究AI最新动态",
    expected_output="AI动态列表",
    agent=agent,
    callback=task_callback
)

# Crew 级别回调
crew = Crew(
    agents=[agent],
    tasks=[task],
    step_callback=step_callback,    # 所有Agent步骤后触发
    task_callback=task_callback,    # 所有任务完成后触发
)
```

#### before_kickoff / after_kickoff

```python
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff

@CrewBase
class MyCrew:
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @before_kickoff
    def prepare_inputs(self, inputs):
        inputs['additional_data'] = "额外信息"
        return inputs

    @after_kickoff
    def process_output(self, output):
        output.raw += "\n后处理完成"
        return output

    @agent
    def researcher(self) -> Agent:
        return Agent(config=self.agents_config['researcher'], verbose=True)

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential)
```

#### 规划（Planning）

```python
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    planning=True,          # 启用规划
    planning_llm="gpt-4o",  # 规划使用的模型
    process=Process.sequential
)
```

## 4. 在LLM开发中的典型使用场景和代码示例

### 场景一：内容研究与分析团队

```python
from crewai import Agent, Crew, Task, Process
from crewai_tools import SerperDevTool

# 定义 Agent
researcher = Agent(
    role="技术研究员",
    goal="发现{topic}领域的前沿技术和趋势",
    backstory="你是一位专注于技术趋势的资深研究员，擅长信息检索和整理",
    tools=[SerperDevTool()],
    verbose=True
)

analyst = Agent(
    role="技术分析师",
    goal="分析研究数据，提取关键洞察",
    backstory="你是一位敏锐的技术分析师，擅长从数据中发现模式和价值",
    verbose=True
)

writer = Agent(
    role="技术撰稿人",
    goal="将分析结果转化为清晰易懂的文章",
    backstory="你是一位经验丰富的技术作家，擅长将复杂概念简明表达",
    verbose=True
)

# 定义任务
research_task = Task(
    description="研究{topic}领域的最新发展，找出5个最重要的趋势",
    expected_output="包含5个关键趋势的详细列表，每个趋势附有简短说明",
    agent=researcher
)

analysis_task = Task(
    description="基于研究结果，分析各趋势的影响力和潜在价值",
    expected_output="趋势影响力分析报告，包含优先级排序和理由",
    agent=analyst,
    context=[research_task]
)

writing_task = Task(
    description="将分析报告转化为一篇结构清晰的技术文章",
    expected_output="一篇800字左右的技术洞察文章，包含标题、引言、主体和结论",
    agent=writer,
    context=[research_task, analysis_task],
    output_file="output/tech_insight.md"
)

# 组建团队
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential,
    verbose=True
)

# 执行
result = crew.kickoff(inputs={"topic": "大语言模型Agent"})
print(result.raw)
```

### 场景二：代码审查与优化团队

```python
from crewai import Agent, Crew, Task, Process

reviewer = Agent(
    role="代码审查专家",
    goal="识别代码中的问题和改进点",
    backstory="10年代码审查经验，精通代码质量标准和安全最佳实践",
    allow_code_execution=True,
    code_execution_mode="safe",
    verbose=True
)

optimizer = Agent(
    role="性能优化专家",
    goal="优化代码性能和可维护性",
    backstory="专精于性能调优和重构，深入理解算法复杂度",
    allow_code_execution=True,
    verbose=True
)

review_task = Task(
    description="审查提供的Python代码，找出潜在问题和安全漏洞",
    expected_output="代码审查报告，包含问题列表和严重程度评级",
    agent=reviewer
)

optimize_task = Task(
    description="基于审查结果，提供优化后的代码版本",
    expected_output="优化后的代码及改动说明",
    agent=optimizer,
    context=[review_task]
)

crew = Crew(
    agents=[reviewer, optimizer],
    tasks=[review_task, optimize_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff(inputs={"code": "def process(data): ..."})
```

### 场景三：层级管理的客服系统

```python
from crewai import Agent, Crew, Task, Process

triage_agent = Agent(
    role="客服分流专员",
    goal="快速分析客户问题并分配给合适的处理人员",
    backstory="经验丰富的客服分流专员，擅长快速判断问题类型和优先级",
    verbose=True
)

tech_support = Agent(
    role="技术支持工程师",
    goal="解决技术类客户问题",
    backstory="技术支持专家，精通产品技术细节和故障排除",
    verbose=True
)

billing_agent = Agent(
    role="账务处理专员",
    goal="处理账单和退款相关问题",
    backstory="账务处理专家，熟悉公司退费政策和账单系统",
    verbose=True
)

# 使用层级模式，Manager自动分配
crew = Crew(
    agents=[triage_agent, tech_support, billing_agent],
    tasks=[
        Task(description="处理客户工单", expected_output="工单处理结果", agent=triage_agent),
        Task(description="解决技术问题", expected_output="技术问题解决方案", agent=tech_support),
        Task(description="处理账务请求", expected_output="账务处理结果", agent=billing_agent),
    ],
    process=Process.hierarchical,
    manager_llm="gpt-4o",
    memory=True  # 记住客户历史交互
)
```

## 5. 数学原理

CrewAI 涉及的核心数学原理主要体现在其记忆系统的 RAG（Retrieval-Augmented Generation）机制中：

### 5.1 向量嵌入与相似度检索

短期记忆和实体记忆使用 RAG 进行存储和检索，其核心是向量嵌入：

1. **文本嵌入**：将文本通过 Embedding 模型映射为高维向量 $\mathbf{v} \in \mathbb{R}^d$
2. **余弦相似度**：衡量查询向量 $\mathbf{q}$ 与存储向量 $\mathbf{v}_i$ 的相关性：

$$\text{sim}(\mathbf{q}, \mathbf{v}_i) = \frac{\mathbf{q} \cdot \mathbf{v}_i}{\|\mathbf{q}\| \cdot \|\mathbf{v}_i\|}$$

3. **Top-K 检索**：返回相似度最高的 K 个记忆片段作为上下文注入 Agent 提示

### 5.2 上下文窗口管理

当 `respect_context_window=True` 时，CrewAI 通过摘要压缩管理上下文：

- 维护当前对话的 token 计数
- 当接近上下文窗口限制时，对较早的对话进行摘要压缩
- 摘要保留关键信息，减少 token 数量，保证最新交互的完整性

### 5.3 任务流程的图论视角

从图论角度看，CrewAI 的任务流程可建模为有向无环图（DAG）：

- **Sequential**：线性链式 DAG，每个节点只有一个前驱和一个后继
- **Hierarchical**：星型结构，Manager 为中心节点，任务节点与 Manager 双向连接
- **异步任务**：允许并行分支，形成更复杂的 DAG 拓扑

## 6. 代码原理/架构原理

### 6.1 角色驱动的多 Agent 协作架构

CrewAI 的核心架构围绕"角色"概念构建：

```
Crew（团队）
├── Process（流程策略）
│   ├── Sequential: 线性链式执行
│   └── Hierarchical: Manager 协调执行
├── Agents（智能体列表）
│   ├── Role + Goal + Backstory → 系统提示
│   ├── Tools → 工具调用能力
│   └── Memory → 上下文保持
├── Tasks（任务列表）
│   ├── Description + Expected Output → 任务定义
│   ├── Context → 任务间依赖
│   └── Guardrail → 输出验证
└── Memory（记忆系统）
    ├── Short-Term → RAG检索
    ├── Long-Term → SQLite持久化
    └── Entity → 实体关系图
```

### 6.2 Agent 执行循环

每个 Agent 的执行遵循 ReAct（Reasoning + Acting）模式：

```
1. 接收任务描述和上下文
2. 构建系统提示（Role + Goal + Backstory → System Prompt）
3. 循环（最多 max_iter 次）：
   a. 推理：分析当前状态，决定下一步行动
   b. 行动：调用工具或生成文本
   c. 观察：获取工具返回或环境反馈
   d. 判断：是否已达到目标？
4. 输出最终结果
```

### 6.3 Process 流程执行

**Sequential 模式**：
- 任务按列表顺序依次执行
- 每个任务的输出自动作为后续任务的上下文
- 适合线性工作流（研究→分析→撰写）

**Hierarchical 模式**：
- Manager Agent 接收所有任务和 Agent 信息
- Manager 动态分配任务给最合适的 Agent
- Manager 验证每个输出，决定是否重做或继续
- 适合复杂工作流，需要动态决策

### 6.4 工具调用机制

Agent 的工具调用基于 LLM 的 function calling 能力：

1. 将工具的 name、description 和参数 schema 转换为 function 定义
2. LLM 决定是否调用工具，以及传什么参数
3. 框架执行工具调用，将结果返回给 LLM
4. `function_calling_llm` 允许使用不同模型处理工具调用（降低成本）

## 7. 常见注意事项和最佳实践

### 7.1 安全注意事项

- **代码执行**：生产环境必须使用 `code_execution_mode="safe"`（Docker 隔离），仅在可信环境中使用 `"unsafe"`
- **API 密钥**：使用环境变量管理，切勿硬编码
- **Guardrail**：对关键任务添加护栏，防止有害输出传播到下游

### 7.2 性能优化

- **设置 max_rpm**：避免触发 API 速率限制
- **启用缓存**：对重复性任务设置 `cache=True`
- **异步执行**：无依赖关系的任务使用 `async_execution=True` 并行执行
- **合理选择模型**：主推理用强模型（gpt-4o），工具调用可用轻量模型（gpt-4o-mini）

### 7.3 Agent 设计建议

- **角色定义要具体**：避免模糊的角色描述，具体化 Agent 的专业领域
- **Goal 要可衡量**：目标应清晰可评估，帮助 Agent 自我判断是否完成
- **Backstory 要相关**：背景故事应与任务相关，帮助模型进入正确角色
- **工具匹配任务**：只为 Agent 提供其角色需要的工具，避免无关工具干扰

### 7.4 Task 设计建议

- **expected_output 要详细**：越具体的输出描述，LLM 生成的结果越符合预期
- **善用 context 传递**：明确指定任务间的依赖关系，避免信息丢失
- **使用 output_pydantic**：需要结构化输出时优先使用 Pydantic 模型约束
- **Guardrail 验证**：对关键输出添加验证逻辑，确保质量

### 7.5 记忆系统建议

- **记忆不是免费的**：启用记忆会增加 API 调用和存储开销，按需启用
- **定期重置**：长期运行的 Crew 应定期重置短期记忆，避免上下文膨胀
- **自定义 Embedder**：大规模部署时考虑使用本地嵌入模型降低成本
- **使用 CREWAI_STORAGE_DIR**：通过环境变量管理存储路径，便于运维

### 7.6 调试技巧

- **verbose=True**：开发阶段开启详细日志，帮助理解 Agent 的推理过程
- **step_callback**：通过回调监控每个 Agent 步骤，定位异常
- **output_log_file**：保存完整日志到文件，便于事后分析
- **max_iter**：Agent 陷入循环时降低 max_iter，快速暴露问题
