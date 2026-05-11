---
title: "smolagents 轻量Agent框架"
excerpt: "CodeAgent/ToolCallingAgent、@tool装饰器、HfApiModel、代码沙箱"
collection: llm-libs
permalink: /llm-libs/28-smolagents
category: agent
toc: true
---


## 1. 库的简介和在LLM开发中的作用

Smolagents 是 HuggingFace 推出的轻量级 Agent 框架，其核心理念是**用代码作为 Agent 的行动空间**（Code-as-Action）。与传统 Agent 框架（如 LangChain 的工具调用 JSON 格式）不同，Smolagents 让 Agent 直接生成并执行 Python 代码来完成操作，这使得 Agent 的行动能力更加灵活和强大。

在 LLM 开发中的作用：
- **快速构建 Agent**：极简 API，几行代码即可创建一个具备工具使用能力的 Agent
- **代码即行动**：Agent 生成可执行 Python 代码，而非结构化 JSON，表达能力更强
- **工具生态**：内置丰富的默认工具（搜索、代码执行、文件操作等），同时支持自定义工具
- **多 Agent 编排**：支持将多个 Agent 组合为协作系统
- **安全执行**：通过沙箱机制隔离 Agent 生成的代码执行环境

## 2. 安装方式

```bash
# 基础安装
pip install smolagents

# 安装可选依赖
pip install smolagents[openai]     # OpenAI 模型支持
pip install smolagents[transformers] # 本地 Transformers 模型支持
pip install smolagents[vision]     # 视觉能力支持
pip install smolagents[all]        # 完整安装

# 从源码安装
pip install git+https://github.com/huggingface/smolagents.git
```

## 3. 核心类/函数/工具的详细说明

### 3.1 Agent 类型

Smolagents 提供两种核心 Agent 类型，区别在于行动空间的表示形式。

#### `CodeAgent` — 代码生成型 Agent

CodeAgent 让 LLM 生成 Python 代码作为行动，通过代码执行器运行。这是 Smolagents 推荐的 Agent 类型。

```python
from smolagents import CodeAgent, HfApiModel

# 创建 CodeAgent
agent = CodeAgent(
    model=HfApiModel(),        # 使用的LLM模型
    tools=[],                  # 工具列表（可选）
    max_steps=10,              # 最大执行步数
    additional_authorized_imports=["numpy", "pandas"],  # 允许导入的模块
)

# 运行 Agent
result = agent.run("计算1到100的素数之和")
print(result)
```

**参数说明**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | Model | LLM 模型实例 |
| `tools` | list[Tool] | 可用工具列表 |
| `max_steps` | int | 最大执行步骤数，防止无限循环 |
| `additional_authorized_imports` | list[str] | 额外允许导入的 Python 模块 |
| `planning_interval` | int | 每隔多少步进行一次规划反思 |
| `verbosity_level` | int | 日志详细程度 (0=静默, 1=进度, 2=详细) |
| `grammar` | dict | 约束输出的语法（用于结构化输出） |

**CodeAgent 的工作方式**：

CodeAgent 在每一步会生成类似以下的 Python 代码：

```python
# Agent 内部生成的代码示例
primes = []
for num in range(2, 101):
    is_prime = True
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            is_prime = False
            break
    if is_prime:
        primes.append(num)

result = sum(primes)
print(result)  # 1060
```

#### `ToolCallingAgent` — 工具调用型 Agent

ToolCallingAgent 让 LLM 生成结构化的工具调用 JSON，类似于 OpenAI 的 function calling 模式。

```python
from smolagents import ToolCallingAgent, HfApiModel, tool

@tool
def search_web(query: str) -> str:
    """搜索网页获取信息。
    Args:
        query: 搜索关键词
    """
    # 实现搜索逻辑
    return f"搜索结果：{query}的相关信息..."

@tool
def calculate(expression: str) -> float:
    """计算数学表达式。
    Args:
        expression: 数学表达式字符串
    """
    return eval(expression)

agent = ToolCallingAgent(
    model=HfApiModel(),
    tools=[search_web, calculate],
    max_steps=5,
)

result = agent.run("搜索Python的最新版本号，然后计算3的10次方")
print(result)
```

**参数说明**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | Model | LLM 模型实例 |
| `tools` | list[Tool] | 可用工具列表（必须提供） |
| `max_steps` | int | 最大执行步骤数 |
| `planning_interval` | int | 规划反思间隔 |

### 3.2 Tool 系统

#### `@tool` 装饰器

最简单的创建自定义工具的方式，用装饰器将 Python 函数转为 Tool 对象。

```python
from smolagents import tool

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """获取指定城市的天气信息。

    Args:
        city: 城市名称，如"北京"、"上海"
        unit: 温度单位，"celsius"或"fahrenheit"，默认为celsius
    """
    # 模拟天气API调用
    weather_data = {
        "北京": {"temp_c": 25, "condition": "晴"},
        "上海": {"temp_c": 28, "condition": "多云"},
    }
    data = weather_data.get(city, {"temp_c": 20, "condition": "未知"})
    if unit == "fahrenheit":
        temp = data["temp_c"] * 9/5 + 32
        return f"{city}天气：{data['condition']}，温度{temp}°F"
    return f"{city}天气：{data['condition']}，温度{data['temp_c']}°C"

# 使用工具
result = get_weather(city="北京", unit="celsius")
print(result)  # "北京天气：晴，温度25°C"
```

**关键要求**：
- 函数必须有**文档字符串**（docstring），Agent 根据文档字符串理解工具用途
- 参数必须有**类型注解**
- 文档字符串中应使用 `Args:` 段落详细说明每个参数
- 函数应有**明确的返回值**

#### `Tool` 类

当工具需要更复杂的初始化逻辑或状态管理时，使用 Tool 类继承。

```python
from smolagents import Tool
import sqlite3

class DatabaseQueryTool(Tool):
    name = "database_query"
    description = "执行SQL查询并返回结果。用于从数据库获取信息。"

    # 定义输入参数的 JSON Schema
    inputs = {
        "query": {
            "type": "string",
            "description": "要执行的SQL查询语句"
        },
        "database": {
            "type": "string",
            "description": "数据库名称，默认为main"
        }
    }

    output_type = "string"

    def __init__(self, db_path: str = "data.db"):
        super().__init__()
        self.db_path = db_path

    def forward(self, query: str, database: str = "main") -> str:
        """执行SQL查询"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            return str(results)
        except Exception as e:
            return f"查询错误：{str(e)}"

# 使用
db_tool = DatabaseQueryTool(db_path="/path/to/data.db")
result = db_tool.forward("SELECT * FROM users LIMIT 5")
```

**Tool 类核心属性**：
| 属性/方法 | 类型 | 说明 |
|-----------|------|------|
| `name` | str | 工具名称，用于 Agent 调用 |
| `description` | str | 工具描述，Agent 根据此描述选择工具 |
| `inputs` | dict | 输入参数的 JSON Schema |
| `output_type` | str | 输出类型："string"、"number"、"object"等 |
| `forward()` | method | 工具的执行逻辑 |

#### 默认工具集

Smolagents 内置了多个常用工具：

```python
from smolagents import (
    DuckDuckGoSearchTool,    # DuckDuckGo 搜索
    VisitWebpageTool,        # 访问网页获取内容
    WikipediaSearchTool,     # Wikipedia 搜索
    PythonInterpreterTool,   # Python 代码执行
    FinalAnswerTool,         # 返回最终答案
)

# 使用内置工具
agent = CodeAgent(
    model=HfApiModel(),
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    max_steps=10,
)

result = agent.run("搜索HuggingFace的最新动态并总结")
```

### 3.3 Model 后端

#### `HfApiModel` — HuggingFace Inference API

```python
from smolagents import HfApiModel

# 使用 HF Inference API（默认模型）
model = HfApiModel()

# 指定模型
model = HfApiModel(
    model_id="Qwen/Qwen2.5-72B-Instruct",  # 模型ID
    token="hf_...",                         # HF token（可选，默认从环境变量读取）
    max_tokens=2048,                        # 最大生成token数
    temperature=0.7,                        # 温度参数
    top_p=0.95,                             # top_p 采样参数
)
```

**参数说明**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `model_id` | str | HuggingFace 模型ID |
| `token` | str | HF API token |
| `max_tokens` | int | 最大生成token数 |
| `temperature` | float | 采样温度 |
| `top_p` | float | nucleus采样参数 |
| `custom_role_conversions` | dict | 自定义角色映射 |

#### `OpenAIServerModel` — OpenAI 兼容 API

```python
from smolagents import OpenAIServerModel

# OpenAI 官方 API
model = OpenAIServerModel(
    model_id="gpt-4o",
    api_key="sk-...",
)

# 兼容 OpenAI 的服务（vLLM, Ollama 等）
model = OpenAIServerModel(
    model_id="local-model",
    api_base="http://localhost:8000/v1",   # API端点
    api_key="not-needed",                   # 本地服务无需key
    max_tokens=4096,
    temperature=0.5,
)
```

**参数说明**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `model_id` | str | 模型名称 |
| `api_base` | str | API基础URL |
| `api_key` | str | API密钥 |
| `max_tokens` | int | 最大生成token数 |
| `temperature` | float | 采样温度 |
| `organization` | str | OpenAI组织ID |

#### `TransformersModel` — 本地 Transformers 模型

```python
from smolagents import TransformersModel

# 加载本地模型
model = TransformersModel(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",              # 自动分配设备
    max_tokens=2048,
    temperature=0.7,
)

# 量化加载
model = TransformersModel(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    load_in_8bit=True,
)
```

**参数说明**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `model_id` | str | 模型ID或本地路径 |
| `device_map` | str | 设备映射策略 |
| `max_tokens` | int | 最大生成token数 |
| `temperature` | float | 采样温度 |
| `load_in_8bit` | bool | 是否8位量化 |
| `load_in_4bit` | bool | 是否4位量化 |

### 3.4 Agent 执行流程

Smolagents 的 Agent 执行遵循经典的 ReAct 循环：

```
┌─────────────┐
│   用户输入    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────────┐
│   Planning   │────→│  思考下一步行动    │
│   (规划)     │     │  (Thought)       │
└──────┬──────┘     └────────┬─────────┘
       │                     │
       ▼                     ▼
┌──────────────┐     ┌──────────────────┐
│ Code/ToolCall │────→│    执行行动       │
│  (代码/调用)  │     │   (Action)       │
└──────┬──────┘     └────────┬─────────┘
       │                     │
       ▼                     ▼
┌──────────────┐     ┌──────────────────┐
│  Execution   │────→│  观察执行结果     │
│  (执行)      │     │  (Observation)   │
└──────┬──────┘     └────────┬─────────┘
       │                     │
       │    ┌────────────────┘
       │    │
       ▼    ▼
┌──────────────────┐
│ 是否达到目标？     │──── 否 ──→ 回到 Planning
└──────┬───────────┘
       │ 是
       ▼
┌──────────────┐
│  返回最终答案  │
└──────────────┘
```

**执行流程详解**：

```python
from smolagents import CodeAgent, HfApiModel

agent = CodeAgent(
    model=HfApiModel(),
    tools=[],
    max_steps=6,               # 最多6步
    planning_interval=2,        # 每2步进行一次规划反思
    verbosity_level=2,          # 详细输出
)

# agent.run() 的内部流程：
# Step 1: Planning - 分析任务，制定计划
#   → "我需要先找出1到100中的素数，然后求和"
# Step 2: Action - 生成并执行代码
#   → 代码: primes = [n for n in range(2, 101) if all(n%i!=0 for i in range(2, int(n**0.5)+1))]
# Step 3: Observation - 观察执行结果
#   → "primes = [2, 3, 5, 7, 11, ..., 97]"
# Step 4: Action - 继续执行
#   → 代码: result = sum(primes)
# Step 5: Observation - 得到结果
#   → "result = 1060"
# Step 6: Final Answer - 返回最终答案
#   → "1到100的素数之和是1060"

result = agent.run("计算1到100的素数之和")
```

### 3.5 多 Agent 系统

#### `ManagedAgent` — 托管式子 Agent

将一个 Agent 作为另一个 Agent 的工具使用，实现层级式多 Agent 协作。

```python
from smolagents import CodeAgent, ToolCallingAgent, HfApiModel, ManagedAgent, tool

@tool
def search_web(query: str) -> str:
    """搜索网页信息。
    Args:
        query: 搜索关键词
    """
    return f"搜索结果：{query}的相关信息..."

# 创建专门的搜索 Agent
search_agent = ToolCallingAgent(
    model=HfApiModel(),
    tools=[search_web],
    name="search_agent",
    description="负责搜索网络信息，回答关于事实、新闻、数据的问题",
    max_steps=3,
)

# 创建代码计算 Agent
code_agent = CodeAgent(
    model=HfApiModel(),
    tools=[],
    name="code_agent",
    description="负责数学计算和数据处理，擅长编写Python代码解决计算问题",
    additional_authorized_imports=["math", "statistics"],
    max_steps=5,
)

# 将子 Agent 注册为 ManagedAgent
managed_search = ManagedAgent(search_agent, name="search_agent", description="搜索网络信息")
managed_code = ManagedAgent(code_agent, name="code_agent", description="数学计算和数据处理")

# 创建主管 Agent
manager_agent = CodeAgent(
    model=HfApiModel(),
    tools=[],                              # 主管自己的工具
    managed_agents=[managed_search, managed_code],  # 托管的子Agent
    max_steps=10,
)

result = manager_agent.run("搜索2024年全球GDP数据，然后计算前10大经济体的GDP总和")
print(result)
```

**ManagedAgent 参数说明**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `agent` | Agent | 被托管的 Agent 实例 |
| `name` | str | 托管名称，主管 Agent 通过此名称调用 |
| `description` | str | 描述信息，帮助主管 Agent 决定何时调用 |

#### Agent 编排模式

```python
# 模式1：层级式（Manager-Worker）
#   manager_agent → 调度 → search_agent / code_agent / writer_agent

# 模式2：顺序式（Pipeline）
#   agent1.run(task) → result1
#   agent2.run(result1) → result2
#   agent3.run(result2) → final_result

# 顺序式示例
from smolagents import CodeAgent, HfApiModel

researcher = CodeAgent(model=HfApiModel(), tools=[...], max_steps=5)
writer = CodeAgent(model=HfApiModel(), tools=[], max_steps=3)
reviewer = CodeAgent(model=HfApiModel(), tools=[], max_steps=3)

# 顺序执行
research = researcher.run("研究量子计算的最新进展")
article = writer.run(f"根据以下研究结果写一篇文章：{research}")
final = reviewer.run(f"审阅并改进这篇文章：{article}")
```

### 3.6 安全机制 — PythonExecutor 沙箱

CodeAgent 生成的代码通过 `PythonExecutor` 在受限环境中执行。

```python
from smolagents import CodeAgent, HfApiModel, LocalPythonExecutor

# 默认安全限制：
# - 禁止导入未经授权的模块
# - 禁止文件系统操作（open, os, subprocess等）
# - 禁止网络访问（requests, urllib等）
# - 限制执行时间

# 配置允许的导入
agent = CodeAgent(
    model=HfApiModel(),
    tools=[],
    additional_authorized_imports=[
        "numpy",          # 允许 numpy
        "pandas",         # 允许 pandas
        "math",           # 允许 math
        "datetime",       # 允许 datetime
        # 注意：os, subprocess, requests 等危险模块不会被授权
    ],
)

# 自定义 PythonExecutor
executor = LocalPythonExecutor(
    additional_authorized_imports=["numpy", "pandas"],
    max_print_outputs_length=1000,  # 限制打印输出的最大长度
)
agent = CodeAgent(
    model=HfApiModel(),
    tools=[],
    executor=executor,
)
```

**安全机制说明**：

| 机制 | 说明 |
|------|------|
| 模块导入白名单 | 只有 `additional_authorized_imports` 中列出的模块才能被导入 |
| 危险函数禁用 | `open()`, `exec()`, `eval()`, `__import__()` 等被禁止 |
| 执行步数限制 | `max_steps` 防止无限循环 |
| 输出长度限制 | 防止内存溢出 |
| 沙箱隔离 | 在独立命名空间中执行，不影响外部环境 |

## 4. 在LLM开发中的典型使用场景和代码示例

### 场景一：数据分析 Agent

```python
from smolagents import CodeAgent, HfApiModel
import pandas as pd

# 创建数据分析 Agent
agent = CodeAgent(
    model=HfApiModel(model_id="Qwen/Qwen2.5-72B-Instruct"),
    additional_authorized_imports=["pandas", "numpy", "math"],
    max_steps=10,
)

# 提供数据（通过变量注入）
data = pd.DataFrame({
    "product": ["A", "B", "C", "D", "E"],
    "sales": [1200, 850, 2100, 680, 1500],
    "region": ["华东", "华南", "华东", "华北", "华南"]
})

result = agent.run(
    f"分析以下销售数据，找出最佳销售区域和产品：\n{data.to_string()}",
    additional_args={"data": data}  # 注入数据变量
)
print(result)
```

### 场景二：研究助手 Agent

```python
from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool, VisitWebpageTool

agent = CodeAgent(
    model=HfApiModel(),
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    max_steps=15,
    planning_interval=3,  # 每3步重新规划
)

result = agent.run(
    "研究 Transformer 架构的核心创新点，"
    "找到原始论文 'Attention Is All You Need' 的关键结论，"
    "并总结 Transformer 相比 RNN 的优势"
)
print(result)
```

### 场景三：代码生成与调试 Agent

```python
from smolagents import CodeAgent, HfApiModel

agent = CodeAgent(
    model=HfApiModel(),
    additional_authorized_imports=["json", "re", "math", "collections"],
    max_steps=8,
)

result = agent.run(
    "编写一个Python函数，实现快速排序算法，"
    "要求支持自定义比较函数，并处理空列表和单元素列表的边界情况"
)
print(result)
```

### 场景四：多 Agent 协作 — 内容创作流水线

```python
from smolagents import (
    CodeAgent, ToolCallingAgent, ManagedAgent,
    HfApiModel, DuckDuckGoSearchTool, tool
)

@tool
def search_news(topic: str) -> str:
    """搜索最新新闻。
    Args:
        topic: 新闻主题
    """
    return f"关于'{topic}'的最新新闻：..."

# 研究员 Agent
researcher = ToolCallingAgent(
    model=HfApiModel(),
    tools=[search_news],
    name="researcher",
    description="负责信息搜集和事实核查",
    max_steps=5,
)

# 写手 Agent
writer = CodeAgent(
    model=HfApiModel(),
    tools=[],
    name="writer",
    description="负责根据研究结果撰写文章",
    max_steps=5,
)

# 编辑 Agent
editor = CodeAgent(
    model=HfApiModel(),
    tools=[],
    name="editor",
    description="负责审阅文章，检查逻辑、语法和风格",
    max_steps=3,
)

# 主编 Agent（协调者）
chief_editor = CodeAgent(
    model=HfApiModel(),
    managed_agents=[
        ManagedAgent(researcher, name="researcher", description="搜集信息"),
        ManagedAgent(writer, name="writer", description="撰写文章"),
        ManagedAgent(editor, name="editor", description="审阅文章"),
    ],
    max_steps=15,
)

result = chief_editor.run("写一篇关于AI在医疗领域应用趋势的文章")
print(result)
```

### 场景五：使用本地模型构建 Agent

```python
from smolagents import CodeAgent, TransformersModel
from smolagents import tool

@tool
def get_stock_price(symbol: str) -> str:
    """获取股票价格。
    Args:
        symbol: 股票代码
    """
    # 模拟API
    prices = {"AAPL": "178.50", "GOOGL": "141.20", "MSFT": "378.90"}
    return f"{symbol} 当前价格: ${prices.get(symbol, 'N/A')}"

# 使用本地 Transformers 模型
model = TransformersModel(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    max_tokens=2048,
)

agent = CodeAgent(
    model=model,
    tools=[get_stock_price],
    max_steps=5,
)

result = agent.run("查询苹果公司和谷歌的股价，并比较哪个更贵")
print(result)
```

### 场景六：使用 OpenAI 兼容服务

```python
from smolagents import CodeAgent, OpenAIServerModel

# 使用 vLLM 部署的本地模型
model = OpenAIServerModel(
    model_id="Qwen/Qwen2.5-72B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
    max_tokens=4096,
    temperature=0.7,
)

agent = CodeAgent(
    model=model,
    tools=[],
    additional_authorized_imports=["math", "random", "itertools"],
    max_steps=10,
)

result = agent.run("生成10个不重复的随机4位验证码")
print(result)
```

## 5. 数学原理

### 5.1 ReAct 循环

Smolagents 的执行流程基于 ReAct（Reasoning + Acting）框架，该框架将推理和行动交织进行：

**形式化定义**：

给定任务 $T$，Agent 在每一步 $t$ 执行：

1. **思考（Thought）**：基于当前上下文 $C_t$（包含原始任务、历史思考、历史行动、历史观察），LLM 生成推理步骤：
   $$\text{thought}_t = \text{LLM}(C_t)$$

2. **行动（Action）**：基于思考生成行动（代码或工具调用）：
   $$\text{action}_t = \text{LLM}(C_t \oplus \text{thought}_t)$$

3. **观察（Observation）**：执行行动并获取结果：
   $$\text{observation}_t = \text{Execute}(\text{action}_t)$$

4. **更新上下文**：
   $$C_{t+1} = C_t \oplus \text{thought}_t \oplus \text{action}_t \oplus \text{observation}_t$$

5. **终止条件**：当 LLM 判定已收集足够信息，生成最终答案；或达到 `max_steps` 限制。

### 5.2 Code-as-Action 的优势

传统工具调用 Agent 的行动空间是离散的、有限的工具集合：

$$\text{Action Space}_{\text{tool}} = \{f_1, f_2, \ldots, f_n\}$$

Code Agent 的行动空间是所有合法 Python 程序的集合：

$$\text{Action Space}_{\text{code}} = \{p \mid p \text{ is a valid Python program}\}$$

代码作为行动的优势：
- **组合性**：可以组合多个操作（循环、条件、变量），而非单一工具调用
- **表达力**：一个代码块可以表达复杂的逻辑，无需预定义所有工具
- **自省性**：代码中可以包含条件判断和错误处理
- **可调试性**：生成的代码可以被人类阅读和理解

### 5.3 规划反思机制

当设置 `planning_interval=k` 时，Agent 每 $k$ 步进行一次规划反思：

$$\text{Plan}_i = \text{LLM}(\text{Task} \oplus \text{History}_{1..ik} \oplus \text{"Re-evaluate the plan"})$$

这类似于蒙特卡洛树搜索中的回溯：定期审视当前进展，调整后续策略。

## 6. 代码原理/架构原理

### 6.1 整体架构

```
┌─────────────────────────────────────────────────────┐
│                   用户接口层                          │
│         agent.run(task) → result                    │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│                  Agent 核心层                        │
│  ┌─────────────────┐  ┌──────────────────────────┐  │
│  │   CodeAgent     │  │   ToolCallingAgent       │  │
│  │ (生成Python代码) │  │ (生成工具调用JSON)       │  │
│  └────────┬────────┘  └───────────┬──────────────┘  │
│           │                       │                  │
│  ┌────────▼───────────────────────▼───────────────┐  │
│  │          Prompt 构建器                          │  │
│  │  - 系统提示（Agent角色定义）                      │  │
│  │  - 工具描述（函数签名+文档）                      │  │
│  │  - 历史记录（Thought/Action/Observation）        │  │
│  │  - 管理Agent描述（如果有）                       │  │
│  └────────────────────────────────────────────────┘  │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│                  执行层                              │
│  ┌──────────────────┐  ┌─────────────────────────┐  │
│  │ LocalPythonExec  │  │    Tool.forward()       │  │
│  │ (沙箱代码执行)    │  │   (工具直接执行)        │  │
│  └──────────────────┘  └─────────────────────────┘  │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│                  模型层                              │
│  ┌──────────┐ ┌───────────────┐ ┌────────────────┐  │
│  │HfApiModel│ │OpenAIServer   │ │Transformers    │  │
│  │ (HF API) │ │(OpenAI兼容)   │ │ (本地模型)     │  │
│  └──────────┘ └───────────────┘ └────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 6.2 Prompt 构建机制

Smolagents 通过精心构建 prompt 来引导 LLM 生成正确格式的输出：

```python
# CodeAgent 的系统提示结构（简化版）
SYSTEM_PROMPT = """
You are an expert assistant that solves tasks using Python code.

You have access to the following tools:
{tool_descriptions}

You can also use the following Python modules:
{authorized_imports}

To solve the task, you must output Python code between ```python and ``` tags.

Follow this format:
Thought: Your reasoning about the next step
```python
# Your Python code here
```

Observation: The result of executing your code

Continue until you have the final answer, then call:
final_answer(your_answer)
"""
```

### 6.3 代码执行流程

```python
# LocalPythonExecutor 的核心逻辑（简化版）
class LocalPythonExecutor:
    def __init__(self, additional_authorized_imports=None):
        self.authorized_imports = set(additional_authorized_imports or [])
        # 预定义安全函数
        self.safe_builtins = {
            "print": print,
            "len": len,
            "range": range,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            # ... 更多安全内置函数
        }

    def execute(self, code: str, state: dict) -> tuple:
        """执行Agent生成的代码"""
        # 1. 解析代码，检查是否包含禁止操作
        self._validate_code(code)

        # 2. 构建受限的执行命名空间
        namespace = {
            "__builtins__": self.safe_builtins,
            **state,  # 之前的变量状态
        }

        # 3. 执行代码
        try:
            exec(code, namespace)
        except Exception as e:
            return None, str(e)

        # 4. 收集输出和状态变更
        return namespace.get("final_answer"), namespace

    def _validate_code(self, code: str):
        """验证代码安全性"""
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.authorized_imports:
                        raise ImportError(f"不允许导入模块: {alias.name}")
            # 检查其他危险操作...
```

### 6.4 多 Agent 编排原理

ManagedAgent 本质上将子 Agent 包装为一个特殊的 Tool：

```python
# ManagedAgent 的核心逻辑（简化版）
class ManagedAgent:
    def __init__(self, agent, name, description):
        self.agent = agent
        self.name = name
        self.description = description

    def __call__(self, task: str) -> str:
        """将子Agent作为工具调用"""
        return self.agent.run(task)

    # 当注册到主管Agent时，会被转换为Tool：
    # - name: self.name
    # - description: self.description
    # - forward(): 调用 self.agent.run()
```

当主管 Agent 决定调用子 Agent 时，它生成的代码类似：

```python
# 主管Agent内部生成的代码
result = search_agent("搜索量子计算的最新进展")
print(result)
```

## 7. 常见注意事项和最佳实践

### 7.1 模型选择建议

- **代码生成能力优先**：CodeAgent 依赖 LLM 生成正确的 Python 代码，建议使用代码能力强的模型（如 Qwen2.5-Coder、DeepSeek-Coder、GPT-4o）
- **工具调用能力优先**：ToolCallingAgent 需要 LLM 正确生成工具调用 JSON，建议使用支持 function calling 的模型
- **本地部署**：使用 `TransformersModel` 或 `OpenAIServerModel`（配合 vLLM），注意模型至少 7B 参数才具备较好的 Agent 能力
- **API 服务**：`HfApiModel` 适合快速原型，但可能有调用限制

### 7.2 工具设计最佳实践

```python
from smolagents import tool

# 好：清晰的单职责工具，文档完整
@tool
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """计算身体质量指数(BMI)。

    Args:
        weight_kg: 体重，单位千克
        height_m: 身高，单位米
    """
    return weight_kg / (height_m ** 2)

# 差：职责模糊，参数不明确
@tool
def health_calc(weight, height):
    """计算健康指标"""
    return weight / height ** 2
```

**设计原则**：
1. 每个工具只做一件事（单一职责）
2. 文档字符串清晰描述功能和参数
3. 参数使用有意义的名称和类型注解
4. 工具应返回字符串结果（Agent 易于理解）
5. 避免工具之间有重叠功能

### 7.3 安全注意事项

```python
# 1. 不要授权危险模块
# 错误做法：
agent = CodeAgent(
    model=HfApiModel(),
    additional_authorized_imports=["os", "subprocess", "requests"],  # 危险！
)

# 正确做法：只授权必要的模块
agent = CodeAgent(
    model=HfApiModel(),
    additional_authorized_imports=["math", "json", "datetime"],  # 安全
)

# 2. 设置合理的 max_steps
agent = CodeAgent(
    model=HfApiModel(),
    max_steps=10,  # 防止无限循环
)

# 3. 在生产环境中，考虑使用 Docker 容器进一步隔离
```

### 7.4 性能优化

```python
# 1. 减少 max_steps — 根据任务复杂度设置合理的步骤数
agent = CodeAgent(model=HfApiModel(), max_steps=5)   # 简单任务
agent = CodeAgent(model=HfApiModel(), max_steps=15)  # 复杂任务

# 2. 使用 planning_interval 减少无效步骤
agent = CodeAgent(
    model=HfApiModel(),
    planning_interval=3,  # 每3步反思一次，及时纠正方向
)

# 3. 提供清晰的初始提示 — 减少Agent探索时间
result = agent.run(
    "使用pandas计算data变量中'sales'列的平均值，"
    "data是一个已经加载的DataFrame，包含'sales'和'region'列"
)

# 4. 注入已有变量，避免Agent重复计算
agent.run(
    "分析data变量的统计特征",
    additional_args={"data": preloaded_dataframe}
)
```

### 7.5 调试技巧

```python
# 1. 使用 verbosity_level 查看详细执行过程
agent = CodeAgent(
    model=HfApiModel(),
    verbosity_level=2,  # 0=静默, 1=进度, 2=详细
)

# 2. 查看Agent的执行日志
result = agent.run("计算斐波那契数列的第20项")
# 日志会显示每一步的 Thought、Action、Observation

# 3. 查看 Agent 的消息历史
for msg in agent.write_memory_to_messages():
    print(f"[{msg['role']}] {msg['content'][:200]}...")
```

### 7.6 常见错误与解决

| 错误 | 原因 | 解决方法 |
|------|------|----------|
| `ImportError: 不允许导入模块` | 尝试导入未授权的模块 | 添加到 `additional_authorized_imports` |
| `MaxStepsExceeded` | 超过最大执行步数 | 增大 `max_steps` 或简化任务 |
| `Code execution error` | 生成的代码有语法或运行时错误 | 使用更好的模型，或在提示中给出更明确的指引 |
| `Tool not found` | Agent 调用了不存在的工具 | 检查工具是否正确注册 |
| `Token limit exceeded` | 对话历史过长 | 减少 `max_steps`，或使用更简洁的提示 |

### 7.7 与其他 Agent 框架的对比

| 特性 | Smolagents | LangChain Agent | AutoGen | CrewAI |
|------|-----------|----------------|---------|--------|
| 行动空间 | Python代码 | 工具调用JSON | 混合 | 工具调用 |
| 核心理念 | Code-as-Action | Chain/Tool | 多Agent对话 | 角色扮演 |
| 学习曲线 | 低 | 中高 | 中 | 低 |
| 多Agent | ManagedAgent | 少数支持 | 原生支持 | 原生支持 |
| 安全沙箱 | LocalPythonExecutor | 无内置 | Docker | 无内置 |
| HuggingFace集成 | 深度集成 | 有限 | 有限 | 无 |
| 代码可读性 | 高（生成的代码可读） | 低（JSON调用） | 中 | 低 |

### 7.8 适用场景总结

**适合使用 Smolagents 的场景**：
- 需要灵活代码执行能力的数据分析、科学计算
- 快速原型开发，用最少的代码构建 Agent
- 需要 HuggingFace 生态集成的场景
- 需要安全沙箱执行的代码生成任务
- 多 Agent 协作的研究型应用

**不适合使用 Smolagents 的场景**：
- 纯对话式应用（不需要工具/代码执行）
- 对延迟极其敏感的实时应用
- 需要极其复杂的多 Agent 编排（考虑 AutoGen）
- 生产级的高并发服务（Smolagents 更偏研究和原型）
