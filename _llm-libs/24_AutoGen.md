---
title: "AutoGen 多Agent对话框架"
excerpt: "ConversableAgent、GroupChat、代码执行(本地/Docker)、v0.4 AgentChat"
collection: llm-libs
permalink: /llm-libs/24-autogen
category: agent
---


## 1. 库的简介和在LLM开发中的作用

AutoGen 是由微软研究院开发的多Agent对话框架，其核心思想是**让多个AI Agent通过对话协作完成复杂任务**。与单Agent框架不同，AutoGen 强调 Agent 之间的消息传递与协作，支持人机混合交互和自动化代码执行。

在LLM开发中，AutoGen 的核心作用包括：
- **多Agent协作**：让多个专精不同领域的Agent协同完成复杂工作流
- **自动化代码执行**：Agent生成的代码可以自动运行并返回结果
- **人机协同**：支持在Agent对话流程中插入人类决策节点
- **工具调用编排**：将函数工具注册给Agent，实现自动化工具调用

AutoGen 目前存在两套API：
- **v0.2 API（经典API）**：基于 `ConversableAgent` 和 `GroupChat` 的对话式架构
- **v0.4+ API（AgentChat 新API）**：更现代的声明式API，基于 `AssistantAgent` 和 `UserProxyAgent`

```bash
pip install autogen-agentchat  # v0.4+ 新API
pip install autogen            # v0.2 经典API（仍可使用）
```

## 2. 安装方式

```bash
# 基础安装（v0.4+ AgentChat）
pip install autogen-agentchat

# 安装代码执行支持（Docker方式，推荐）
pip install autogen-agentchat[docker]

# 安装代码执行支持（本地方式）
pip install autogen-agentchat[local-exec]

# 安装所有可选依赖
pip install autogen-agentchat[all]

# v0.2 经典API安装（兼容性）
pip install pyautogen

# 从源码安装
pip install git+https://github.com/microsoft/autogen.git
```

> **注意**：Docker 代码执行需要本地安装 Docker 并启动 Docker 守护进程。本地代码执行方式安全性较低，仅建议在受控环境中使用。

## 3. 核心类/函数/工具的详细说明

### 3.1 ConversableAgent（v0.2 经典API）

`ConversableAgent` 是 AutoGen v0.2 的核心类，所有Agent都基于它构建。

```python
import autogen

# 创建一个通用Agent
assistant = autogen.ConversableAgent(
    name="assistant",
    system_message="你是一个有帮助的AI助手，用中文回答问题。",
    llm_config={
        "model": "gpt-4o",
        "api_key": "sk-...",      # 也可通过环境变量 OPENAI_API_KEY 设置
        "temperature": 0.7,
        "max_tokens": 2048,
    },
    human_input_mode="NEVER",   # 是否需要人类输入
    max_consecutive_auto_reply=10,  # 最大连续自动回复次数
    code_execution_config=False,    # 代码执行配置
)

# 创建一个用户代理Agent
user_proxy = autogen.ConversableAgent(
    name="user_proxy",
    system_message="你是用户的代理，负责执行代码并返回结果。",
    human_input_mode="TERMINATE",  # 对话终止时请求人类输入
    code_execution_config={
        "work_dir": "./coding",     # 代码执行工作目录
        "use_docker": False,        # 是否使用Docker
    },
    llm_config=False,               # 不使用LLM
)
```

**关键参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | str | Agent名称，用于对话中标识 |
| `system_message` | str | 系统提示词，定义Agent角色 |
| `llm_config` | dict/False | LLM配置字典，False表示不使用LLM |
| `human_input_mode` | str | 人类输入模式 |
| `max_consecutive_auto_reply` | int | 最大连续自动回复次数 |
| `code_execution_config` | dict/False | 代码执行配置 |
| `function_map` | dict | 函数名到函数对象的映射 |

**human_input_mode 选项：**
| 模式 | 说明 |
|------|------|
| `NEVER` | 从不请求人类输入，完全自动 |
| `ALWAYS` | 每次回复前都请求人类输入 |
| `TERMINATE` | 仅在对话终止条件满足时请求人类输入 |

**llm_config 常用配置：**
```python
llm_config = {
    "model": "gpt-4o",                    # 模型名称
    "api_key": "sk-...",                   # API密钥
    "temperature": 0.7,                    # 生成温度
    "max_tokens": 2048,                    # 最大token数
    "timeout": 120,                        # 请求超时（秒）
    "cache_seed": 42,                      # 缓存种子，None则不缓存
    "config_list": [                       # 多模型配置列表（支持fallback）
        {"model": "gpt-4o", "api_key": "sk-..."},
        {"model": "gpt-4o-mini", "api_key": "sk-..."},
    ],
    "functions": [...],                    # 可调用的函数定义（JSON Schema）
}
```

### 3.2 对话：initiate_chat() 与 send()

#### initiate_chat — 发起对话

```python
# user_proxy 发起与 assistant 的对话
user_proxy.initiate_chat(
    assistant,
    message="请帮我写一个Python函数，计算斐波那契数列的第n项。",
    # 可选参数
    max_turns=5,          # 最大对话轮数
    summary_method="last_msg",  # 摘要生成方式
    clear_history=True,   # 开始前是否清除历史
)

# 获取对话摘要
print(user_proxy.last_message(assistant))
```

**summary_method 选项：**
| 方法 | 说明 |
|------|------|
| `None` | 不生成摘要 |
| `"last_msg"` | 使用最后一条消息作为摘要 |
| `"reflection_with_llm"` | 使用LLM生成摘要 |

#### send — 发送单条消息

```python
# 在已建立的对话中发送消息
user_proxy.send(
    message="请把上面的函数改为使用记忆化递归实现。",
    recipient=assistant,
)
```

#### 自动回复机制

AutoGen 的对话循环基于 `auto_reply` 机制：

```
Agent A 发送消息 → Agent B 接收消息
                        │
                        ├── 检查 human_input_mode
                        │      ├── ALWAYS: 等待人类输入
                        │      ├── TERMINATE: 检查终止条件
                        │      └── NEVER: 自动处理
                        │
                        ├── 检查是否包含代码 → 执行代码并返回结果
                        │
                        └── 调用 LLM 生成回复 → 发回 Agent A
```

可以通过 `register_reply` 自定义回复逻辑：

```python
def custom_reply(recipient, messages, sender, config):
    """自定义回复函数"""
    last_message = messages[-1]["content"]
    if "统计" in last_message:
        return True, "这是一个统计类问题，建议使用pandas处理。"
    return False, None  # 返回 False 表示不处理，交给下一个回复逻辑

assistant.register_reply(
    trigger=autogen.ConversableAgent,  # 触发条件：来自ConversableAgent的消息
    reply_func=custom_reply,
    position=0,  # 优先级，0最高
)
```

### 3.3 GroupChat：多Agent协作

`GroupChat` 允许两个以上的Agent在同一对话中协作。

```python
import autogen

# 创建多个Agent
planner = autogen.ConversableAgent(
    name="planner",
    system_message="你是一个任务规划师。将复杂任务分解为清晰的步骤。只做规划，不写代码。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

coder = autogen.ConversableAgent(
    name="coder",
    system_message="你是一个Python程序员。根据规划师的步骤编写代码。只写代码，不做规划。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

reviewer = autogen.ConversableAgent(
    name="reviewer",
    system_message="你是一个代码审查员。检查代码的正确性、性能和安全性，给出修改建议。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 创建 GroupChat
groupchat = autogen.GroupChat(
    agents=[planner, coder, reviewer],
    messages=[],              # 对话历史
    max_round=10,             # 最大对话轮数
    speaker_selection_method="auto",  # 发言者选择方式
    allow_repeat_speaker=False,       # 是否允许同一Agent连续发言
)

# 创建 GroupChat 管理器
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,   # 管理器也需要LLM来决定发言顺序
)

# 发起群聊
user_proxy = autogen.ConversableAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
)
user_proxy.initiate_chat(manager, message="编写一个Web爬虫，抓取新闻网站的标题和链接。")
```

**speaker_selection_method 选项：**
| 方法 | 说明 |
|------|------|
| `"auto"` | 由LLM自动选择下一个发言者（默认） |
| `"round_robin"` | 按顺序轮流发言 |
| `"random"` | 随机选择发言者 |
| 自定义函数 | 传入 `(last_speaker, messages) -> next_speaker` 的函数 |

**自定义发言者选择：**
```python
def custom_speaker_selection(last_speaker, messages):
    """根据消息内容智能选择下一个发言者"""
    last_msg = messages[-1]["content"]
    if last_speaker == planner:
        return coder       # 规划师说完轮到编码者
    elif last_speaker == coder:
        return reviewer    # 编码者说完轮到审查者
    elif "需要修改" in last_msg:
        return coder       # 审查者要求修改则回到编码者
    else:
        return planner     # 否则回到规划师

groupchat = autogen.GroupChat(
    agents=[planner, coder, reviewer],
    messages=[],
    max_round=10,
    speaker_selection_method=custom_speaker_selection,
)
```

### 3.4 代码执行

#### LocalCommandLineCodeExecutor

在本地命令行执行Agent生成的代码。

```python
from autogen.coding import LocalCommandLineCodeExecutor

executor = LocalCommandLineCodeExecutor(
    timeout=60,           # 执行超时时间（秒）
    work_dir="./coding",  # 工作目录
)

# 与Agent集成
code_agent = autogen.ConversableAgent(
    name="code_agent",
    system_message="你是一个Python程序员，编写代码解决问题。",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config={
        "executor": executor,
    },
)
```

> **警告**：本地执行会运行Agent生成的任意代码，存在安全风险。仅建议在隔离的沙箱环境中使用。

#### DockerCommandLineCodeExecutor

在Docker容器中执行代码，提供安全隔离。

```python
from autogen.coding import DockerCommandLineCodeExecutor

docker_executor = DockerCommandLineCodeExecutor(
    image="python:3.11-slim",  # Docker镜像
    timeout=60,                 # 执行超时时间（秒）
    work_dir="./coding",        # 宿主机工作目录
    container_name="autogen_runner",  # 容器名称
)

# 使用方式与本地执行器相同
code_agent = autogen.ConversableAgent(
    name="code_agent",
    system_message="你是一个Python程序员，编写代码解决问题。代码将自动执行。",
    llm_config=llm_config,
    human_input_mode="NEVER",
    code_execution_config={
        "executor": docker_executor,
    },
)

# 记得在使用完毕后清理
# docker_executor.stop()
```

#### v0.2 简便配置方式

```python
# 简单的本地代码执行配置
code_execution_config = {
    "work_dir": "./coding",
    "use_docker": False,   # True则使用Docker
}

# 禁用代码执行
code_execution_config = False
```

### 3.5 人机交互

AutoGen 支持灵活的人机交互模式，通过 `human_input_mode` 和自定义输入函数实现。

```python
# 模式1：ALWAYS - 每轮都请求人类输入
human_agent = autogen.ConversableAgent(
    name="human",
    human_input_mode="ALWAYS",
)

# 模式2：TERMINATE - 仅在检测到终止词时请求输入
user_proxy = autogen.ConversableAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
)

# 模式3：自定义输入函数
def custom_input(prompt: str) -> str:
    """自定义人类输入函数"""
    print(f"\n[提示] {prompt}")
    user_input = input("请输入回复（输入 'skip' 跳过，'exit' 退出）: ")
    if user_input == "exit":
        return "TERMINATE"  # 终止对话
    if user_input == "skip":
        return ""           # 跳过本轮
    return user_input

human_agent = autogen.ConversableAgent(
    name="human",
    human_input_mode="ALWAYS",
    get_human_input=custom_input,
)
```

**对话终止检测**：AutoGen 默认在消息中检测 `TERMINATE` 关键词来触发终止。可以自定义：

```python
def is_termination_msg(msg):
    """自定义终止检测"""
    content = msg.get("content", "")
    return content.rstrip().endswith("TERMINATE") or "任务完成" in content

assistant = autogen.ConversableAgent(
    name="assistant",
    system_message="完成任务后在回复末尾加上 TERMINATE",
    llm_config=llm_config,
    is_termination_msg=is_termination_msg,
)
```

### 3.6 工具调用

#### register_for_execution / register_for_llm（v0.2）

AutoGen v0.2 使用装饰器模式注册工具函数，分别注册给执行端和LLM端。

```python
import autogen

# 定义工具函数
@user_proxy.register_for_execution()
@assistant.register_for_llm(name="get_weather", description="获取指定城市的天气信息")
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。
    
    Args:
        city: 城市名称，如"北京"、"上海"
    
    Returns:
        天气信息字符串
    """
    # 模拟天气API调用
    weather_data = {
        "北京": "晴天，气温15°C，空气质量良好",
        "上海": "多云，气温18°C，有轻微雾霾",
        "深圳": "小雨，气温22°C，湿度较高",
    }
    return weather_data.get(city, f"暂无{city}的天气数据")

@user_proxy.register_for_execution()
@assistant.register_for_llm(name="calculate", description="执行数学计算")
def calculate(expression: str) -> str:
    """执行数学计算表达式。
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
    
    Returns:
        计算结果字符串
    """
    try:
        result = eval(expression)  # 注意：生产环境应使用更安全的解析方式
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

# 发起对话
user_proxy.initiate_chat(
    assistant,
    message="今天北京的天气怎么样？另外帮我算一下 15 * 23 + 7",
)
```

**装饰器说明：**
| 装饰器 | 作用 |
|-------|------|
| `@agent.register_for_execution()` | 将函数注册到Agent的 `function_map`，使Agent能执行该函数 |
| `@agent.register_for_llm(name, description)` | 将函数的JSON Schema注册到Agent的 `llm_config.functions`，使LLM知道可调用 |

#### 手动注册方式

```python
# 手动注册到执行端
user_proxy.register_function(
    function_map={
        "get_weather": get_weather,
        "calculate": calculate,
    }
)

# 手动注册到LLM端（需要提供JSON Schema）
assistant.register_function(
    function_map={
        "get_weather": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                },
                "required": ["city"],
            },
        },
    }
)
```

### 3.7 AgentChat 新API（v0.4+）

AutoGen v0.4 引入了全新的 `autogen-agentchat` 包，提供更现代、更简洁的API。

#### AssistantAgent

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 创建模型客户端
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key="sk-...",
)

# 创建助手Agent
assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="你是一个有帮助的AI助手，用中文回答问题。",
    model_client_stream=True,  # 启用流式输出
    # 工具注册
    tools=[get_weather, calculate],  # 直接传入函数列表
    # 工具调用模式
    reflect_on_tool_use=True,  # 工具调用后让LLM反思结果
)

# 运行Agent
result = await assistant.run(task="今天北京天气如何？")
print(result.messages[-1].content)
```

#### UserProxyAgent

```python
from autogen_agentchat.agents import UserProxyAgent

user_proxy = UserProxyAgent(
    name="user",
    description="用户代理，在需要确认时请求人类输入。",
)
```

#### 多Agent团队协作（v0.4+）

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")

# 创建团队成员
planner = AssistantAgent(
    name="planner",
    model_client=model_client,
    system_message="你是任务规划师，将复杂任务分解为步骤。",
)

coder = AssistantAgent(
    name="coder",
    model_client=model_client,
    system_message="你是Python程序员，根据规划编写代码。",
)

reviewer = AssistantAgent(
    name="reviewer",
    model_client=model_client,
    system_message="你是代码审查员，检查代码质量并给出建议。",
)

# 创建团队（轮询模式）
team = RoundRobinGroupChat(
    participants=[planner, coder, reviewer],
    termination_condition=TextMentionTermination("APPROVE") | MaxMessageTermination(15),
)

# 运行团队任务
result = await team.run(task="编写一个CSV数据分析工具，支持统计、过滤和排序功能。")
print(result.messages[-1].content)
```

**终止条件（v0.4+）：**
| 条件 | 说明 |
|------|------|
| `TextMentionTermination(text)` | 检测到指定文本时终止 |
| `MaxMessageTermination(n)` | 达到最大消息数时终止 |
| `TokenUsageTermination(limit)` | 达到token使用上限时终止 |
| `TimeoutTermination(timeout)` | 达到时间上限时终止 |
| `HandoffTermination()` | Agent交接时终止 |
| 条件可用 `\|` (或) 和 `&` (与) 组合 |

#### 流式输出（v0.4+）

```python
from autogen_agentchat.base import TaskResult

# 流式获取Agent输出
async for message in assistant.run_stream(task="解释量子计算的基本原理"):
    if isinstance(message, TaskResult):
        print(f"\n最终结果: {message.messages[-1].content}")
    else:
        # 中间消息（Agent消息、工具调用等）
        print(f"[{message.source}] {message.models_dump()}")
```

## 4. 在LLM开发中的典型使用场景和代码示例

### 场景1：自动化代码编写与执行

```python
import autogen

llm_config = {
    "model": "gpt-4o",
    "api_key": "sk-...",
    "temperature": 0,
}

# 创建程序员Agent
coder = autogen.ConversableAgent(
    name="coder",
    system_message="""你是一个Python专家。
    当你编写代码时：
    1. 将代码放在 ```python 和 ``` 之间
    2. 确保代码完整可运行
    3. 包含必要的import语句
    4. 添加注释说明关键逻辑
    完成任务后回复 TERMINATE""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 创建用户代理（可执行代码）
user_proxy = autogen.ConversableAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=5,
    code_execution_config={
        "work_dir": "./output",
        "use_docker": False,
    },
    llm_config=False,
)

# 发起任务
user_proxy.initiate_chat(
    coder,
    message="编写一个Python脚本，读取当前目录下所有.csv文件，合并成一个DataFrame并输出统计摘要。",
)
```

### 场景2：多Agent协作完成研究任务

```python
import autogen

llm_config = {"model": "gpt-4o", "api_key": "sk-..."}

# 研究助手 - 负责搜索和整理信息
researcher = autogen.ConversableAgent(
    name="researcher",
    system_message="你是一个研究助手。根据用户的问题，整理关键信息并提供分析。回答要基于事实，注明来源。",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 写作助手 - 负责将研究结果整理成报告
writer = autogen.ConversableAgent(
    name="writer",
    system_message="""你是一个技术写作专家。根据研究助手提供的信息，撰写结构清晰的技术报告。
    报告格式：1.摘要 2.背景 3.分析 4.结论
    完成后回复 TERMINATE""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 审稿人 - 负责审核报告质量
reviewer = autogen.ConversableAgent(
    name="reviewer",
    system_message="""你是一个审稿人。审查技术报告的：
    1.逻辑完整性 2.数据准确性 3.语言表达
    如果需要修改，明确指出问题和修改建议。如果质量合格，回复 APPROVE""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 创建群聊
groupchat = autogen.GroupChat(
    agents=[researcher, writer, reviewer],
    messages=[],
    max_round=8,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# 发起任务
user_proxy = autogen.ConversableAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
)
user_proxy.initiate_chat(
    manager,
    message="研究当前主流的大语言模型微调技术（LoRA、QLoRA、Adapter等），并写一份对比分析报告。",
)
```

### 场景3：带工具调用的客服机器人

```python
import autogen
import json
from datetime import datetime

llm_config = {"model": "gpt-4o", "api_key": "sk-..."}

# 创建客服Agent
customer_service = autogen.ConversableAgent(
    name="customer_service",
    system_message="""你是一个智能客服。你可以查询订单状态、处理退换货、查看产品信息。
    使用提供的工具来查询信息，不要编造数据。
    回复用中文，语气友好专业。""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# 创建用户代理
user_proxy = autogen.ConversableAgent(
    name="user",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10,
    llm_config=False,
)

# 模拟订单数据库
ORDERS = {
    "ORD001": {"status": "已发货", "eta": "2024-01-15", "items": ["Python编程书", "机械键盘"]},
    "ORD002": {"status": "处理中", "eta": "2024-01-18", "items": ["显示器"]},
    "ORD003": {"status": "已签收", "eta": None, "items": ["耳机", "鼠标垫"]},
}

# 注册查询订单工具
@user_proxy.register_for_execution()
@customer_service.register_for_llm(name="query_order", description="根据订单号查询订单状态")
def query_order(order_id: str) -> str:
    """查询订单状态。

    Args:
        order_id: 订单编号，如 ORD001
    """
    order = ORDERS.get(order_id)
    if order:
        return json.dumps(order, ensure_ascii=False)
    return f"未找到订单 {order_id}，请检查订单号是否正确。"

# 注册退换货工具
@user_proxy.register_for_execution()
@customer_service.register_for_llm(name="request_return", description="申请退换货")
def request_return(order_id: str, reason: str, return_type: str = "退货") -> str:
    """申请退换货。

    Args:
        order_id: 订单编号
        reason: 退换货原因
        return_type: 退换货类型，"退货"或"换货"
    """
    if order_id not in ORDERS:
        return f"未找到订单 {order_id}"
    return f"已为您提交{return_type}申请，订单号：{order_id}，原因：{reason}。预计1-2个工作日处理。"

# 开始对话
user_proxy.initiate_chat(
    customer_service,
    message="你好，我想查询订单ORD001的状态，另外想申请ORD003的退货。",
)
```

## 5. 数学原理

### 5.1 基于LLM的Agent选择策略

在 `GroupChat` 的 `speaker_selection_method="auto"` 模式下，Manager Agent 使用LLM选择下一个发言者。本质上是一个**条件生成问题**：

给定对话历史 $H = \{m_1, m_2, ..., m_t\}$ 和Agent集合 $A = \{a_1, a_2, ..., a_n\}$，LLM被提示生成下一个发言者的名称：

$$\text{next\_speaker} = \arg\max_{a_i \in A} P(a_i \mid H, \text{prompt})$$

这里 LLM 充当一个分类器，将对话历史映射到 Agent 名称空间。

### 5.2 ReAct 推理循环

AutoGen 中 AssistantAgent 支持的 ReAct 循环可形式化为：

1. **观察**（Observation）$o_t$：接收上一轮的执行结果或用户输入
2. **思考**（Thought）$\tau_t$：LLM基于观察生成推理步骤
3. **行动**（Action）$a_t$：选择并执行一个工具/函数
4. 循环直到生成最终答案 $y$

$$\tau_t, a_t = \text{LLM}(o_t, \text{prompt})$$
$$o_{t+1} = \text{Execute}(a_t)$$

### 5.3 消息传递的形式化

AutoGen 的对话本质上是消息的传递和转换。每条消息 $m$ 可以表示为：

$$m = (\text{sender}, \text{content}, \text{role}, \text{metadata})$$

一轮对话（turn）中，Agent $A$ 向 Agent $B$ 发送消息的过程：

$$m_B = \text{Agent}_B.\text{auto\_reply}(m_A)$$

其中 `auto_reply` 是Agent的回复生成函数，其优先级链为：
1. 人类输入（如果 `human_input_mode` 允许）
2. 代码执行结果（如果消息中包含代码块）
3. 自定义 `register_reply` 函数
4. LLM 生成

## 6. 代码原理/架构原理

### 多Agent对话框架架构

```
┌─────────────────────────────────────────────────┐
│                   对话层                          │
│  ┌──────────────┐    ┌───────────────────────┐  │
│  │ initiate_chat│    │     GroupChat         │  │
│  │    send()    │    │  ┌─────────────────┐  │  │
│  │              │    │  │ GroupChatManager│  │  │
│  └──────┬───────┘    │  │ (选择发言者)    │  │  │
│         │            │  └────────┬────────┘  │  │
│         ▼            │           │            │  │
│  ┌──────────────┐    │           ▼            │  │
│  │ConversableAgent│  │  ┌─────────────────┐  │  │
│  │              │◄──┼──│  Speaker        │  │  │
│  │ auto_reply() │    │  │  Selection     │  │  │
│  │ register_reply│   │  └─────────────────┘  │  │
│  └──────┬───────┘    └───────────────────────┘  │
│         │                                       │
└─────────┼───────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│                   能力层                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │   LLM    │  │   Code   │  │   Function   │  │
│  │ Client   │  │ Executor │  │   Tools      │  │
│  │          │  │          │  │              │  │
│  │ OpenAI   │  │ Local /  │  │ register_    │  │
│  │ Azure    │  │ Docker   │  │ for_llm      │  │
│  │ Local    │  │          │  │ register_    │  │
│  │          │  │          │  │ for_execution│  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────┘
```

### 核心设计原则

1. **基于消息传递的协作**：Agent之间不直接调用方法，而是通过消息传递通信。这种松耦合设计使得Agent可以独立开发和测试。

2. **自动回复链（auto_reply chain）**：每个Agent维护一个回复函数链，按优先级依次尝试：
   ```
   human_input → code_execution → registered_replies → llm_generate
   ```
   任何一层返回有效回复即终止链。

3. **统一Agent抽象**：`ConversableAgent` 同时作为LLM Agent和用户代理，通过配置参数（`llm_config`、`human_input_mode`、`code_execution_config`）决定行为模式，而非通过继承不同的子类。

4. **GroupChat 管理器模式**：多Agent对话由一个中央管理器协调，管理器负责选择发言者、维护对话历史、检测终止条件。这是一种集中式控制模式。

5. **v0.4+ 的声明式API**：新API从命令式（`initiate_chat`）转向声明式（`team.run(task=...)`），更注重描述"要做什么"而非"怎么做"：
   - 终止条件可组合：`TextMentionTermination("APPROVE") | MaxMessageTermination(10)`
   - Agent作为独立组件，通过 `Team` 编排
   - 原生支持异步和流式

### 消息流转示例

以一个简单的两Agent对话为例：

```
UserProxy                          Assistant
   │                                  │
   │──── "写一个排序函数" ────────────►│  (1) initiate_chat
   │                                  │
   │◄── "好的，这是代码..." ──────────│  (2) LLM生成回复（含代码）
   │                                  │
   │   [执行代码，得到输出]            │
   │                                  │
   │──── "执行结果：[1,2,3]" ────────►│  (3) 自动发送执行结果
   │                                  │
   │◄── "排序函数正常工作。TERMINATE" │  (4) LLM确认，触发终止
   │                                  │
   │   [检测到TERMINATE，对话结束]     │
```

## 7. 常见注意事项和最佳实践

### 注意事项

1. **API 调用成本控制**：
   - 多Agent对话会产生大量LLM调用，尤其是 GroupChat
   - 设置 `max_round` 限制对话轮数
   - 使用 `cache_seed` 启用缓存减少重复调用
   - 考虑使用更便宜的模型处理简单Agent（如 gpt-4o-mini）

2. **代码执行安全**：
   - **永远不要在生产环境使用 `use_docker=False`**
   - Docker 执行器提供沙箱隔离，但仍需注意资源限制
   - 对Agent的 `system_message` 中明确约束代码行为

3. **无限循环防护**：
   - 始终设置 `max_consecutive_auto_reply` 和 `max_round`
   - 使用终止条件（`TERMINATE` 关键词或自定义检测函数）
   - 监控对话轮数，设置超时

4. **GroupChat 发言者选择**：
   - `"auto"` 模式依赖LLM判断，可能选择不当
   - 对于固定工作流，使用自定义 `speaker_selection_method` 函数更可靠
   - `allow_repeat_speaker=False` 避免同一Agent独占对话

5. **LLM 配置一致性**：
   - 参与同一对话的Agent应使用相同或兼容的 `llm_config`
   - `function_map` 和 `functions` 列表必须对应
   - 工具函数的 `description` 要清晰准确，LLM依赖它决定是否调用

### 最佳实践

1. **Agent 角色设计原则**：
   - 每个 Agent 有明确单一的职责（单一职责原则）
   - `system_message` 具体化角色、约束和输出格式
   - 避免角色重叠，否则GroupChat发言者选择会混乱

2. **工具函数设计**：
   - 函数签名要有完整的类型注解
   - docstring 详细描述参数和返回值（LLM依赖此信息）
   - 返回字符串而非复杂对象，方便LLM理解
   - 包含错误处理，返回友好的错误信息

3. **调试技巧**：
   ```python
   # 启用详细日志
   autogen.set_logging(level=autogen.logging.DEBUG)

   # 查看对话历史
   for msg in user_proxy.chat_messages[assistant]:
       print(f"[{msg['role']}] {msg['content'][:100]}...")
   ```

4. **性能优化**：
   - 对不涉及推理的Agent使用 `llm_config=False`
   - 批量任务使用异步API（v0.4+）
   - 复用Agent实例，避免重复初始化

5. **v0.2 → v0.4 迁移建议**：
   - 新项目优先使用 v0.4+ API
   - `ConversableAgent` → `AssistantAgent`
   - `GroupChat` → `RoundRobinGroupChat` 或 `SelectorGroupChat`
   - `initiate_chat()` → `team.run(task=...)`
   - 终止逻辑从 `TERMINATE` 字符串 → 组合式 `TerminationCondition`

6. **中文场景建议**：
   - `system_message` 使用中文，确保LLM用中文回复
   - 工具函数的 `description` 和 docstring 使用中文
   - GroupChat 的发言者选择提示词需包含中文Agent名称的说明
